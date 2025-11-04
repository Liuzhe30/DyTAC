import os, json, argparse, numpy as np, pickle, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.GPT_fused import GPTFusedDynamics as GPTDynamics

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def make_seq(Z, L):
    X, Y = [], []
    for i in range(len(Z) - L):
        X.append(Z[i:i+L]); Y.append(Z[i+1:i+L+1])
    return np.stack(X), np.stack(Y)

def load_modality(path):
    X = np.load(path).astype(np.float32)
    X = np.log1p(np.maximum(X, 0.0))
    return X

def prep_modality(X):
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    return sc, Xs

def pca_fit(Xs, max_comp):
    n_comp = int(min(max_comp, Xs.shape[0], Xs.shape[1], 64))
    pca = PCA(n_components=n_comp, random_state=0).fit(Xs)
    Z = pca.transform(Xs)
    return pca, Z

def tensors_for(Z, L, device):
    Xseq, Yseq = make_seq(Z, L=L)
    return (torch.tensor(Xseq, dtype=torch.float32, device=device),
            torch.tensor(Yseq, dtype=torch.float32, device=device))

def train_loop(model, Xt, Yt, decoders, cfg, ckpt_path, device):
    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["train"].get("lr", 5e-4),
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg["train"].get("sched_T0", 500),
        T_mult=cfg["train"].get("sched_Tmult", 2)
    )
    lam_latent = float(cfg["train"].get("lambda_latent", 0.1))
    lam_atac   = float(cfg["train"].get("lambda_atac",   0.9))
    lam_hic    = float(cfg["train"].get("lambda_hic",    0.9))
    clip_norm  = float(cfg["train"].get("grad_clip", 1.0))
    patience   = int(cfg["train"].get("early_stop_patience", 200))
    min_delta  = float(cfg["train"].get("early_stop_min_delta", 1e-3))
    beta       = float(cfg["train"].get("ema_beta", 0.98))

    best = float("inf"); bad = 0; ema = None
    epochs = int(cfg["train"]["epochs"])
    model.train()

    for ep in range(epochs):
        opt.zero_grad()
        pred_z = model(Xt)     # (N,L,D_total)
        tgt_z  = Yt

        loss_lat = ((pred_z - tgt_z) ** 2).mean()
        loss = lam_latent * loss_lat

        # decode per modality
        offset = 0
        for name, dec in decoders.items():
            D = dec["D"]
            C, Mu, feat_std = dec["C"], dec["Mu"], dec["feat_std"]
            pred_block = pred_z[..., offset:offset+D]
            tgt_block  =  tgt_z[..., offset:offset+D]
            offset += D

            pf = pred_block.reshape(-1, D)
            tf =  tgt_block.reshape(-1, D)
            pred_xs = pf @ C + Mu
            tgt_xs  = tf @ C + Mu
            w = (1.0 / (feat_std ** 2)).unsqueeze(0)
            rec = (((pred_xs - tgt_xs) ** 2) * w).mean()

            if name == "atac": loss = loss + lam_atac * rec
            if name == "hic":  loss = loss + lam_hic  * rec

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        sched.step(ep + 1)

        cur = float(loss.detach().cpu().item())
        ema = cur if ema is None else beta * ema + (1.0 - beta) * cur
        score = ema

        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{epochs} loss {cur:.6f} ema {ema:.6f}")

        if score < best - min_delta:
            best = score; bad = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break

def main(cfg):
    os.makedirs(cfg["output"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["fig_dir"], exist_ok=True)
    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    times = np.load(cfg["data"]["time_path"]).astype(np.float32)
    modality = cfg.get("modality", "atac")  # 'atac'|'hic'|'fused'

    # --- load modalities ---
    use_atac = modality in ("atac", "fused")
    use_hic  = modality in ("hic", "fused")

    if use_atac:
        X_atac = load_modality(cfg["data"]["atac_path"])
        sc_a, Xs_a = prep_modality(X_atac)
        pca_a, Z_a = pca_fit(Xs_a, cfg["model"]["latent_dim"])
    else:
        sc_a = pca_a = Z_a = None

    if use_hic:
        X_hic = load_modality(cfg["data"]["hic_path"])
        sc_h, Xs_h = prep_modality(X_hic)
        pca_h, Z_h = pca_fit(Xs_h, cfg["model"]["latent_dim_hic"])
    else:
        sc_h = pca_h = Z_h = None

    # --- build latent sequence Z ---
    if modality == "atac":
        Z = Z_a
    elif modality == "hic":
        Z = Z_h
    else:
        Z = np.concatenate([Z_a, Z_h], axis=1)

    L = max(2, min(3, len(Z) - 1))
    Xt, Yt = tensors_for(Z, L, device)

    # --- decoders dict (per modality) ---
    decoders = {}
    if use_atac:
        C_a  = torch.from_numpy(pca_a.components_.astype(np.float32)).to(device)  # (D_a,P_a)
        Mu_a = torch.from_numpy(pca_a.mean_.astype(np.float32)).to(device)        # (P_a,)
        fs_a = torch.tensor(Xs_a.std(axis=0)+1e-6, dtype=torch.float32, device=device)
        decoders["atac"] = dict(D=Z_a.shape[1], C=C_a, Mu=Mu_a, feat_std=fs_a)
    if use_hic:
        C_h  = torch.from_numpy(pca_h.components_.astype(np.float32)).to(device)
        Mu_h = torch.from_numpy(pca_h.mean_.astype(np.float32)).to(device)
        fs_h = torch.tensor(Xs_h.std(axis=0)+1e-6, dtype=torch.float32, device=device)
        decoders["hic"] = dict(D=Z_h.shape[1], C=C_h, Mu=Mu_h, feat_std=fs_h)

    # --- model ---
    model = GPTDynamics(
        d_model=Z.shape[1],
        n_layer=cfg["model"].get("tfm_layers", 4),
        n_head=cfg["model"].get("tfm_heads", 4),
        d_ff=cfg["model"].get("tfm_ff", 256),
        dropout=cfg["model"]["dropout"],
        gated=cfg["model"].get("gated", True)
    ).to(device)

    ckpt = os.path.join(cfg["output"]["ckpt_dir"], "gpt.pt")
    train_loop(model, Xt, Yt, decoders, cfg, ckpt, device)

    # --- save all ---
    meta = {"modality": modality}
    with open(os.path.join(cfg["output"]["ckpt_dir"], "meta.json"), "w") as f: json.dump(meta, f)
    np.save(os.path.join(cfg["output"]["ckpt_dir"], "Z.npy"), Z.astype(np.float32))
    np.save(os.path.join(cfg["output"]["ckpt_dir"], "times.npy"), times.astype(np.float32))
    if use_atac:
        with open(os.path.join(cfg["output"]["ckpt_dir"], "scaler_atac.pkl"), "wb") as f: pickle.dump(sc_a, f)
        with open(os.path.join(cfg["output"]["ckpt_dir"], "pca_atac.pkl"), "wb") as f: pickle.dump(pca_a, f)
    if use_hic:
        with open(os.path.join(cfg["output"]["ckpt_dir"], "scaler_hic.pkl"), "wb") as f: pickle.dump(sc_h, f)
        with open(os.path.join(cfg["output"]["ckpt_dir"], "pca_hic.pkl"), "wb") as f: pickle.dump(pca_h, f)
    print("saved.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    main(cfg)
