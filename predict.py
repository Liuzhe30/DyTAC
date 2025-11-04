import os, json, argparse, numpy as np, torch, pickle
import matplotlib.pyplot as plt
from models.GPT_model import GPTDynamics

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

def assimilate(z, x_obs_std, pca, alpha_base):
    x_hat_std = pca.inverse_transform(z.reshape(1,-1)).reshape(-1)
    resid = x_obs_std - x_hat_std
    J = pca.components_             # (D, P), rows are orthonormal for PCA

    sigma_pred2 = float((x_hat_std**2).mean() + 1e-8)
    sigma_obs2  = float((x_obs_std**2).mean() + 1e-8)
    alpha = alpha_base * (sigma_pred2 / (sigma_pred2 + sigma_obs2))

    dz = resid @ J.T                # (P,) @ (P,D)^T -> (D,)
    return z + alpha * dz


def hold_last_baseline(X_std):
    Y = X_std.copy()
    for i in range(1, len(Y)):
        Y[i] = X_std[i-1]
    return Y

def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))

def r2(a, b):
    num = ((a - b) ** 2).sum()
    den = ((a - a.mean()) ** 2).sum() + 1e-12
    return float(1.0 - num / den)

def main(cfg):
    ck = cfg["output"]["ckpt_dir"]
    with open(os.path.join(ck, "pca.pkl"), "rb") as f: pca = pickle.load(f)
    with open(os.path.join(ck, "scaler.pkl"), "rb") as f: scaler = pickle.load(f)
    Z = np.load(os.path.join(ck, "Z.npy"))
    times = np.load(os.path.join(ck, "times.npy"))

    model = GPTDynamics(
        d_model=Z.shape[1],
        n_layer=cfg["model"].get("tfm_layers", 4),
        n_head=cfg["model"].get("tfm_heads", 4),
        d_ff=cfg["model"].get("tfm_ff", 256),
        dropout=cfg["model"]["dropout"]
    )
    model.load_state_dict(torch.load(os.path.join(ck, "gpt.pt"), map_location="cpu"))
    model.eval()

    X_true = np.load(cfg["data"]["atac_path"]).astype(np.float32)
    X_true = np.log1p(np.maximum(X_true, 0.0))
    X_true_std = scaler.transform(X_true)

    # baseline
    bl = hold_last_baseline(X_true_std)

    # no-assim rollout from z0
    steps = len(times) - 1
    z0 = Z[0].copy()
    z_no = np.vstack([z0, roll_latent(model, z0, steps)])
    X_pred_std_noass = pca.inverse_transform(z_no)

    # with-assim rollout
    alpha = cfg["train"].get("assim_alpha", 0.2)
    #idx_assim = {np.argmin(np.abs(times-1.0)), np.argmin(np.abs(times-4.0))}
    idx_assim = {
    np.argmin(np.abs(times-0.33)),  # 20min
    np.argmin(np.abs(times-1.0)),
    np.argmin(np.abs(times-4.0)),}
    z = z0.copy()
    z_list = [z.copy()]
    for i in range(1, len(times)):
        y = roll_latent(model, z, 1)[-1]
        z = y
        if i in idx_assim:
            z = assimilate(z, X_true_std[i], pca, alpha)
        z_list.append(z.copy())
    z_assim = np.vstack(z_list)
    X_pred_std_assim = pca.inverse_transform(z_assim)

    # metrics (global)
    os.makedirs(cfg["output"]["fig_dir"], exist_ok=True)
    m_bl = rmse(X_true_std, bl);    r_bl = r2(X_true_std, bl)
    m_na = rmse(X_true_std, X_pred_std_noass); r_na = r2(X_true_std, X_pred_std_noass)
    m_as = rmse(X_true_std, X_pred_std_assim); r_as = r2(X_true_std, X_pred_std_assim)

    print(f"baseline_rmse_std {m_bl:.6f} r2 {r_bl}")
    print(f"no_assim_rmse_std {m_na:.6f} r2 {r_na}")
    print(f"with_assim_rmse_std {m_as:.6f} r2 {r_as}")

    # per-time rmse
    err_no = np.sqrt(((X_true_std - X_pred_std_noass)**2).mean(axis=1))
    err_as = np.sqrt(((X_true_std - X_pred_std_assim)**2).mean(axis=1))
    for i, tt in enumerate(times):
        print(f"time {tt:.2f}h no-assim {err_no[i]:.4f} with-assim {err_as[i]:.4f}")

    metrics_path = os.path.join(cfg["output"]["fig_dir"], "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"baseline_rmse_std {m_bl:.6f} r2 {r_bl}\n")
        f.write(f"no_assim_rmse_std {m_na:.6f} r2 {r_na}\n")
        f.write(f"with_assim_rmse_std {m_as:.6f} r2 {r_as}\n")
        f.write("\n# Time-wise RMSE (std space)\n")
        f.write("time(h)\tno_assim_RMSE\twith_assim_RMSE\n")
        for i, tt in enumerate(times):
            f.write(f"{tt:.2f}\t{err_no[i]:.4f}\t{err_as[i]:.4f}\n")

    # plot rmse vs time
    plt.figure(figsize=(5,3))
    plt.plot(times, err_no, 'o-', label='No Assim')
    plt.plot(times, err_as, 's-', label='With Assim')
    plt.xlabel('Time (h)'); plt.ylabel('RMSE (std space)')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(cfg["output"]["fig_dir"], "rmse_vs_time.png"), dpi=300)
    plt.close()

    # plot some peaks (std space)
    var = X_true_std.var(axis=0)
    top_idx = np.argsort(var)[-6:][::-1]
    for k, p in enumerate(top_idx):
        plt.figure(figsize=(5,3))
        plt.plot(times, X_true_std[:,p], marker="o", label="true(std)")
        plt.plot(times, X_pred_std_noass[:,p], marker="x", linestyle="--", label="no-assim")
        plt.plot(times, X_pred_std_assim[:,p], marker="s", linestyle="-.", label="assim")
        plt.xlabel("time(h)"); plt.ylabel(f"peak {p} (std)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg["output"]["fig_dir"], f"peak_std_{k}_{p}.png"), dpi=300)
        plt.close()

    # original scale plots
    X_true_orig = np.expm1(scaler.inverse_transform(X_true_std))
    X_no_orig = np.expm1(scaler.inverse_transform(X_pred_std_noass))
    X_as_orig = np.expm1(scaler.inverse_transform(X_pred_std_assim))
    for k, p in enumerate(top_idx[:3]):
        plt.figure(figsize=(5,3))
        plt.plot(times, X_true_orig[:,p], marker="o", label="true(orig)")
        plt.plot(times, X_no_orig[:,p], marker="x", linestyle="--", label="no-assim")
        plt.plot(times, X_as_orig[:,p], marker="s", linestyle="-.", label="assim")
        plt.xlabel("time(h)"); plt.ylabel(f"peak {p} (orig)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(cfg["output"]["fig_dir"], f"peak_orig_{k}_{p}.png"), dpi=300)
        plt.close()

    # dump preds for the notebook
    np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_std_noass.npy"), X_pred_std_noass.astype(np.float32))
    np.save(os.path.join(cfg["output"]["fig_dir"], "X_pred_std_assim.npy"), X_pred_std_assim.astype(np.float32))

    diff = np.abs(X_pred_std_assim - X_pred_std_noass).mean()
    print("mean |with-assim - no-assim| in std space:", diff)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.json")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    
    main(cfg)
