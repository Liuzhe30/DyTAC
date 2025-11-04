import os, json, argparse, numpy as np, torch, pickle
import matplotlib.pyplot as plt
from models.GPT_fused import GPTFusedDynamics as GPTDynamics

def roll_latent(model, z0, steps):
    cur = torch.tensor(z0.reshape(1,1,-1), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            y = model(cur)
            nxt = y[:, -1:, :]
            out.append(nxt.squeeze(0).squeeze(0).numpy())
            cur = nxt
    return np.vstack(out)

def assimilate_dual(z, obs_dict, pca_dict, alpha_base):
    z_new = z.copy(); offset = 0
    for name, pdata in pca_dict.items():
        D = pdata["D"]; pca = pdata["pca"]; obs = obs_dict[name]
        z_block = z_new[offset:offset+D]
        x_hat_std = pca.inverse_transform(z_block.reshape(1,-1)).reshape(-1)
        resid = obs - x_hat_std
        J = pca.components_
        sigma_pred2 = float((x_hat_std**2).mean() + 1e-8)
        sigma_obs2  = float((obs**2).mean() + 1e-8)
        alpha = alpha_base * (sigma_pred2 / (sigma_pred2 + sigma_obs2))
        dz = resid @ J.T
        z_new[offset:offset+D] = z_block + alpha * dz
        offset += D
    return z_new

def rmse(a, b): return float(np.sqrt(((a-b)**2).mean()))
def r2(a, b):
    num = ((a-b)**2).sum()
    den = ((a - a.mean())**2).sum() + 1e-12
    return float(1.0 - num/den)

def main(cfg):
    ck = cfg["output"]["ckpt_dir"]
    meta = json.load(open(os.path.join(ck, "meta.json")))
    modality = meta.get("modality", cfg.get("modality","atac"))
    times = np.load(os.path.join(ck, "times.npy"))
    Z = np.load(os.path.join(ck, "Z.npy"))
    D_total = Z.shape[1]

    use_atac = modality in ("atac", "fused")
    use_hic  = modality in ("hic", "fused")

    pca_dict = {}
    if use_atac:
        with open(os.path.join(ck, "pca_atac.pkl"), "rb") as f: pca_a = pickle.load(f)
        with open(os.path.join(ck, "scaler_atac.pkl"), "rb") as f: sc_a = pickle.load(f)
        X_true_a = np.load(cfg["data"]["atac_path"]).astype(np.float32)
        X_true_a = np.log1p(np.maximum(X_true_a, 0.0))
        X_true_a_std = sc_a.transform(X_true_a)
        pca_dict["atac"] = {"pca": pca_a, "D": pca_a.components_.shape[0]}
    if use_hic:
        with open(os.path.join(ck, "pca_hic.pkl"), "rb") as f: pca_h = pickle.load(f)
        with open(os.path.join(ck, "scaler_hic.pkl"), "rb") as f: sc_h = pickle.load(f)
        X_true_h = np.load(cfg["data"]["hic_path"]).astype(np.float32)
        X_true_h = np.log1p(np.maximum(X_true_h, 0.0))
        X_true_h_std = sc_h.transform(X_true_h)
        pca_dict["hic"] = {"pca": pca_h, "D": pca_h.components_.shape[0]}

    model = GPTDynamics(
        d_model=D_total,
        n_layer=cfg["model"].get("tfm_layers", 4),
        n_head=cfg["model"].get("tfm_heads", 4),
        d_ff=cfg["model"].get("tfm_ff", 256),
        dropout=cfg["model"]["dropout"],
        gated=cfg["model"].get("gated", True)
    )
    model.load_state_dict(torch.load(os.path.join(ck, "gpt.pt"), map_location="cpu"))
    model.eval()

    steps = len(times) - 1
    z0 = Z[0].copy()

    # ----- no-assim -----
    z_no = np.vstack([z0, roll_latent(model, z0, steps)])
    # decode
    offset = 0
    preds_no = {}
    if use_atac:
        D = pca_dict["atac"]["D"]
        preds_no["atac"] = pca_dict["atac"]["pca"].inverse_transform(z_no[:, offset:offset+D])
        offset += D
    if use_hic:
        D = pca_dict["hic"]["D"]
        preds_no["hic"] = pca_dict["hic"]["pca"].inverse_transform(z_no[:, offset:offset+D])

    # ----- with-assim -----
    alpha = cfg["train"].get("assim_alpha", 0.6)
    idx_assim = {np.argmin(np.abs(times-0.33)), np.argmin(np.abs(times-1.0)), np.argmin(np.abs(times-4.0))}
    z = z0.copy()
    z_list = [z.copy()]
    for i in range(1, len(times)):
        y = roll_latent(model, z, 1)[-1]
        z = y
        if i in idx_assim:
            obs = {}
            if use_atac: obs["atac"] = X_true_a_std[i]
            if use_hic:  obs["hic"]  = X_true_h_std[i]
            z = assimilate_dual(z, obs, pca_dict, alpha)
        z_list.append(z.copy())
    z_as = np.vstack(z_list)
    # decode
    offset = 0
    preds_as = {}
    if use_atac:
        D = pca_dict["atac"]["D"]
        preds_as["atac"] = pca_dict["atac"]["pca"].inverse_transform(z_as[:, offset:offset+D])
        offset += D
    if use_hic:
        D = pca_dict["hic"]["D"]
        preds_as["hic"] = pca_dict["hic"]["pca"].inverse_transform(z_as[:, offset:offset+D])

    # ----- fused metrics: concatenate modalities along feature axis -----
    # build fused true/pred arrays in std-space
    has_both = use_atac and use_hic
    if has_both:
        X_true_fused = np.concatenate([X_true_a_std, X_true_h_std], axis=1)
        X_no_fused   = np.concatenate([preds_no["atac"], preds_no["hic"]], axis=1)
        X_as_fused   = np.concatenate([preds_as["atac"], preds_as["hic"]], axis=1)
    elif use_atac:
        X_true_fused = X_true_a_std
        X_no_fused   = preds_no["atac"]
        X_as_fused   = preds_as["atac"]
    else:
        X_true_fused = X_true_h_std
        X_no_fused   = preds_no["hic"]
        X_as_fused   = preds_as["hic"]

    # baseline = hold-last（用第0时刻平铺）
    bl_fused = np.tile(X_true_fused[0], (len(times), 1))

    # global metrics (fused)
    baseline_rmse = rmse(X_true_fused, bl_fused)
    baseline_r2   = r2(X_true_fused, bl_fused)
    no_rmse       = rmse(X_true_fused, X_no_fused)
    no_r2         = r2(X_true_fused, X_no_fused)
    as_rmse       = rmse(X_true_fused, X_as_fused)
    as_r2         = r2(X_true_fused, X_as_fused)

    # print to console in your desired format
    print(f"baseline_rmse_std {baseline_rmse:.6f} r2 {baseline_r2}")
    print(f"no_assim_rmse_std {no_rmse:.6f} r2 {no_r2}")
    print(f"with_assim_rmse_std {as_rmse:.6f} r2 {as_r2}")

    # per-time rmse
    err_no = np.sqrt(((X_true_fused - X_no_fused)**2).mean(axis=1))
    err_as = np.sqrt(((X_true_fused - X_as_fused)**2).mean(axis=1))
    for i, tt in enumerate(times):
        print(f"time {tt:.2f}h no-assim {err_no[i]:.4f} with-assim {err_as[i]:.4f}")

    # mean absolute difference between with-assim and no-assim
    diff_fused = float(np.abs(X_as_fused - X_no_fused).mean())
    print(f"mean |with-assim - no-assim| in std space: {diff_fused:.10f}")

    # write the SAME lines into metrics_fused.txt (under config-defined fig_dir)
    os.makedirs(cfg["output"]["fig_dir"], exist_ok=True)
    with open(os.path.join(cfg["output"]["fig_dir"], "metrics_fused.txt"), "w") as f:
        f.write(f"baseline_rmse_std {baseline_rmse:.6f} r2 {baseline_r2}\n")
        f.write(f"no_assim_rmse_std {no_rmse:.6f} r2 {no_r2}\n")
        f.write(f"with_assim_rmse_std {as_rmse:.6f} r2 {as_r2}\n")
        for i, tt in enumerate(times):
            f.write(f"time {tt:.2f}h no-assim {err_no[i]:.4f} with-assim {err_as[i]:.4f}\n")
        f.write(f"mean |with-assim - no-assim| in std space: {diff_fused:.10f}\n")

    # quick per-time plots (ATAC only if present) — keep original behavior
    if use_atac:
        var = X_true_a_std.var(axis=0)
        top_idx = np.argsort(var)[-6:][::-1]
        for k, p in enumerate(top_idx):
            plt.figure(figsize=(5,3))
            plt.plot(times, X_true_a_std[:,p], marker="o", label="true(std)")
            plt.plot(times, preds_no["atac"][:,p], marker="x", linestyle="--", label="no-assim")
            plt.plot(times, preds_as["atac"][:,p], marker="s", linestyle="-.", label="assim")
            plt.xlabel("time(h)"); plt.ylabel(f"ATAC peak {p} (std)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(cfg["output"]["fig_dir"], f"fused_atac_peak_std_{k}_{p}.png"), dpi=300)
            plt.close()

    # dump preds for notebook — keep original behavior
    if use_atac:
        np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_atac_std_noass.npy"),
                preds_no["atac"].astype(np.float32))
        np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_atac_std_assim.npy"),
                preds_as["atac"].astype(np.float32))
    if use_hic:
        np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_hic_std_noass.npy"),
                preds_no["hic"].astype(np.float32))
        np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_hic_std_assim.npy"),
                preds_as["hic"].astype(np.float32))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.json")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    main(cfg)
