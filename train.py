import os, json, argparse, numpy as np, pickle, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from models.GPT_model import GPTDynamics

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def make_seq(Z, L):
    X, Y = [], []
    for i in range(len(Z) - L):
        X.append(Z[i:i+L]); Y.append(Z[i+1:i+L+1])
    return np.stack(X), np.stack(Y)

def main(cfg):
    os.makedirs(cfg["output"]["ckpt_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["fig_dir"], exist_ok=True)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = np.load(cfg["data"]["atac_path"]).astype(np.float32)
    t = np.load(cfg["data"]["time_path"]).astype(np.float32)
    X = np.log1p(np.maximum(X, 0.0))

    T, P = X.shape
    n_comp = min(cfg["model"]["latent_dim"], T, P, 16)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=n_comp, random_state=0).fit(Xs)
    Z = pca.transform(Xs)

    L = max(2, min(3, len(Z) - 1))
    Xseq, Yseq = make_seq(Z, L=L)
    Xt = torch.tensor(Xseq, dtype=torch.float32, device=device)   # (N,L,D)
    Yt = torch.tensor(Yseq, dtype=torch.float32, device=device)   # (N,L,D)

    C  = torch.from_numpy(pca.components_.astype(np.float32)).to(device)  # (D,P)
    Mu = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)        # (P,)
    feat_std = torch.tensor(Xs.std(axis=0) + 1e-6, dtype=torch.float32, device=device)  # (P,)

    model = GPTDynamics(
        d_model=Z.shape[1],
        n_layer=cfg["model"].get("tfm_layers", 4),
        n_head=cfg["model"].get("tfm_heads", 4),
        d_ff=cfg["model"].get("tfm_ff", 256),
        dropout=cfg["model"]["dropout"]
    ).to(device)

    lr = cfg["train"].get("lr", 5e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # warm restarts help traverse loss plateaus smoothly
    T0 = cfg["train"].get("sched_T0", 500)
    Tmult = cfg["train"].get("sched_Tmult", 2)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=T0, T_mult=Tmult)

    lam_latent = float(cfg["train"].get("lambda_latent", 0.1))
    lam_recon  = float(cfg["train"].get("lambda_recon", 0.9))
    clip_norm  = float(cfg["train"].get("grad_clip", 1.0))

    patience = int(cfg["train"].get("early_stop_patience", 200))
    min_delta = float(cfg["train"].get("early_stop_min_delta", 1e-3))
    beta = float(cfg["train"].get("ema_beta", 0.98))
    best = float("inf"); bad = 0; ema = None

    epochs = int(cfg["train"]["epochs"])

    model.train()
    for ep in range(epochs):
        opt.zero_grad()

        pred_z = model(Xt)     # (N,L,D)
        tgt_z  = Yt            # (N,L,D)

        loss_latent = ((pred_z - tgt_z) ** 2).mean()

        pred_flat = pred_z.reshape(-1, pred_z.shape[-1])   # (N*L,D)
        tgt_flat  = tgt_z.reshape(-1, tgt_z.shape[-1])

        pred_xs = pred_flat @ C + Mu                       # (N*L,P)
        tgt_xs  = tgt_flat  @ C + Mu

        w = (1.0 / (feat_std ** 2)).unsqueeze(0)           # (1,P)
        loss_recon = (((pred_xs - tgt_xs) ** 2) * w).mean()

        loss = lam_latent * loss_latent + lam_recon * loss_recon
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()
        sched.step(ep + 1)

        cur = float(loss.detach().cpu().item())
        ema = cur if ema is None else beta * ema + (1.0 - beta) * cur
        score = ema

        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}/{epochs} loss {cur:.6f} ema {ema:.6f} "
                  f"(lat {float(loss_latent):.6f} recon {float(loss_recon):.6f})")

        improved = (best - score) > min_delta
        if improved:
            best = score
            bad = 0
            torch.save(model.state_dict(), os.path.join(cfg["output"]["ckpt_dir"], "gpt.pt"))
        else:
            bad += 1
            if bad >= patience:
                print("early stop (no EMA improvement)"); break

    with open(os.path.join(cfg["output"]["ckpt_dir"], "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
    with open(os.path.join(cfg["output"]["ckpt_dir"], "pca.pkl"), "wb") as f: pickle.dump(pca, f)
    np.save(os.path.join(cfg["output"]["ckpt_dir"], "Z.npy"), Z)
    np.save(os.path.join(cfg["output"]["ckpt_dir"], "times.npy"), t)
    print("saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    main(cfg)
