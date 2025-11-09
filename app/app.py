import os, json, pickle, numpy as np, torch
import streamlit as st
import plotly.graph_objs as go
from sklearn.manifold import TSNE

# ---------- Styling ----------
PRIMARY = "#000080"
CORAL   = "#F08080"
SEA     = "#2E8B57"
INDIGO  = "#4B0082"
STEEL   = "#4682B4"
GRAYBG  = "#f7f8fb"

st.set_page_config(page_title="HyTAC – Digital Twin UI", page_icon="🧬", layout="wide")
st.markdown(f"""
<style>
body {{ background: {GRAYBG}; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 2rem; }}
h1, h2, h3 {{ letter-spacing: .2px; }}
div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
.sidebar .sidebar-content {{ background: white; }}
.card {{
  background: white; border-radius: 18px; padding: 18px 18px 10px;
  box-shadow: 0 6px 24px rgba(0,0,0,.06), 0 2px 8px rgba(0,0,0,.04);
  border: 1px solid #eceff5; margin-bottom: 16px;
}}
.small {{ font-size: 0.85rem; color: #6b7280; }}
.kpi {{ font-size: 1.05rem; color: #374151; }}
.figcap {{ font-size: 0.85rem; color: #4b5563; margin-top: -6px; }}
/* Animated banner */
.banner {{
  width: 100%; padding: 12px 18px; border-radius: 16px; color: white;
  font-weight: 600; letter-spacing: .3px; margin-bottom: 10px;
  background: linear-gradient(270deg, {PRIMARY}, {INDIGO}, {STEEL});
  background-size: 600% 600%;
  animation: gradientShift 8s ease infinite;
  box-shadow: 0 8px 30px rgba(0,0,0,.08);
  z-index: 999; 
  position: relative;  
  top: 0;
}}
@keyframes gradientShift {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}
</style>
""", unsafe_allow_html=True)

# ---------- Try import model classes; fallback ----------
GPTDynamics = None
GPTFusedDynamics = None
try:
    from models.GPT_model import GPTDynamics as _GD
    GPTDynamics = _GD
except Exception:
    pass

try:
    from models.GPT_fused import GPTFusedDynamics as _GFD
    GPTFusedDynamics = _GFD
except Exception:
    pass

class FallbackGPT(torch.nn.Module):
    def __init__(self, d_model, n_layer=4, n_head=4, d_ff=256, dropout=0.1, gated=True):
        super().__init__()
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head,
                                                     dim_feedforward=d_ff, dropout=dropout,
                                                     activation="gelu", batch_first=False)
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.proj = torch.nn.Linear(d_model, d_model)
    def forward(self, x):
        T = x.shape[1]
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        y = self.enc(x.transpose(0,1), mask=mask).transpose(0,1)
        return self.proj(y)

def get_model(d_model, fused, cfg):
    if fused:
        if GPTFusedDynamics is not None:
            return GPTFusedDynamics(d_model=d_model,
                                    n_layer=cfg["model"].get("tfm_layers",4),
                                    n_head=cfg["model"].get("tfm_heads",4),
                                    d_ff=cfg["model"].get("tfm_ff",256),
                                    dropout=cfg["model"].get("dropout",0.1),
                                    gated=cfg["model"].get("gated",True))
    else:
        if GPTDynamics is not None:
            return GPTDynamics(d_model=d_model,
                               n_layer=cfg["model"].get("tfm_layers",4),
                               n_head=cfg["model"].get("tfm_heads",4),
                               d_ff=cfg["model"].get("tfm_ff",256),
                               dropout=cfg["model"].get("dropout",0.1))
    return FallbackGPT(d_model=d_model,
                       n_layer=cfg["model"].get("tfm_layers",4),
                       n_head=cfg["model"].get("tfm_heads",4),
                       d_ff=cfg["model"].get("tfm_ff",256),
                       dropout=cfg["model"].get("dropout",0.1))

# ---------- Utility ----------
def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))
def r2(a,b):   return float(1 - ((a-b)**2).sum()/(((a-a.mean())**2).sum()+1e-12))

def roll(model, z0, steps):
    cur = torch.tensor(z0.reshape(1,1,-1), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            y = model(cur)
            nxt = y[:,-1:,:]
            out.append(nxt.squeeze(0).squeeze(0).cpu().numpy())
            cur = nxt
    return np.vstack(out)

def assimilate_block(z, obs_std, pca, alpha):
    xhat = pca.inverse_transform(z.reshape(1,-1)).reshape(-1)
    resid = obs_std - xhat
    J = pca.components_
    return z + alpha * (resid @ J.T)

def fused_assim(z, obs_dict, pca_dict, alpha):
    z_new = z.copy(); off=0
    for name, meta in pca_dict.items():
        D = meta["D"]; pca = meta["pca"]; obs = obs_dict.get(name)
        if obs is not None:
            z_new[off:off+D] = assimilate_block(z_new[off:off+D], obs, pca, alpha)
        off += D
    return z_new

def line(fig, x, y, name, color, dash=None):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                             name=name, line=dict(color=color, dash=dash), marker=dict(size=6)))

def nearest_index(times, target_h):
    return int(np.argmin(np.abs(times - target_h)))

# ---------- Header ----------
st.markdown(
    '<div class="banner">Primary CD4+ T cell — anti-CD3/CD28 stimulated · HyTAC Digital Twin</div>',
    unsafe_allow_html=True
)
st.title("HyTAC Viewer")
st.caption("ATAC + Hi-C multimodal digital twin with GPT-based dynamics and PCA-based assimilation")

# ---------- Sidebar ----------
st.sidebar.title("Controls")
default_cfg = "configs/default_modality.json" if os.path.exists("configs/default_modality.json") else "configs/default.json"
cfg_path = st.sidebar.text_input("Config path", value=default_cfg)
modality = st.sidebar.selectbox("Modality", ["fused"], index=0)

assim_on = st.sidebar.checkbox("Enable assimilation", value=True)
alpha = st.sidebar.slider("Assimilation alpha", 0.0, 1.0, 0.6, 0.05)
assim_marks = st.sidebar.multiselect("Assimilate at (h)", [0.0,0.33,1.0,4.0,24.0], default=[0.33,1.0,4.0])
peak_k = st.sidebar.slider("Top peaks to visualize (std/orig)", 3, 12, 6, 1)
sample_points = st.sidebar.slider("Scatter subsample", 2000, 30000, 20000, 1000)

load_btn = st.sidebar.button("Load & Run")

# ---------- Main Logic ----------
if load_btn and os.path.exists(cfg_path):
    cfg = json.load(open(cfg_path))
    fused = (modality == "fused")
    out_dir = cfg["output"]["ckpt_dir"]

    times = np.load(os.path.join(out_dir, "times.npy"))
    Z = np.load(os.path.join(out_dir, "Z.npy"))
    D_total = Z.shape[1]
    steps = len(times) - 1

    model = get_model(D_total, fused, cfg)
    model.load_state_dict(torch.load(os.path.join(out_dir, "gpt.pt"), map_location="cpu"), strict=False)
    model.eval()

    # load modality assets
    pca_dict = {}
    if fused:
        with open(os.path.join(out_dir, "pca_atac.pkl"), "rb") as f: pca_a = pickle.load(f)
        with open(os.path.join(out_dir, "scaler_atac.pkl"), "rb") as f: sc_a = pickle.load(f)
        with open(os.path.join(out_dir, "pca_hic.pkl"), "rb") as f: pca_h = pickle.load(f)
        with open(os.path.join(out_dir, "scaler_hic.pkl"), "rb") as f: sc_h = pickle.load(f)
        pca_dict["atac"] = {"pca": pca_a, "D": pca_a.components_.shape[0]}
        pca_dict["hic"]  = {"pca": pca_h, "D": pca_h.components_.shape[0]}
        X_a = np.log1p(np.maximum(np.load(cfg["data"]["atac_path"]).astype(np.float32), 0.0))
        X_h = np.log1p(np.maximum(np.load(cfg["data"]["hic_path"]).astype(np.float32), 0.0))
        X_a_std = sc_a.transform(X_a)
        X_h_std = sc_h.transform(X_h)
    else:
        with open(os.path.join(out_dir, "pca.pkl"), "rb") as f: pca_a = pickle.load(f)
        with open(os.path.join(out_dir, "scaler.pkl"), "rb") as f: sc_a = pickle.load(f)
        X_a = np.log1p(np.maximum(np.load(cfg["data"]["atac_path"]).astype(np.float32), 0.0))
        X_a_std = sc_a.transform(X_a)
        pca_dict["atac"] = {"pca": pca_a, "D": pca_a.components_.shape[0]}

    # ---- baseline (hold-last in std space of selected modality or concat) ----
    if fused:
        bl = np.tile(np.hstack([X_a_std[0], X_h_std[0]]), (len(times),1))
    else:
        bl = np.tile(X_a_std[0], (len(times),1))

    # ---- rollout no-assim ----
    z0 = Z[0].copy()
    z_no = np.vstack([z0, roll(model, z0, steps)])

    def decode(z_seq):
        if fused:
            off=0
            out={}
            D = pca_dict["atac"]["D"]
            out["atac_std"] = pca_dict["atac"]["pca"].inverse_transform(z_seq[:,off:off+D]); off+=D
            D = pca_dict["hic"]["D"]
            out["hic_std"]  = pca_dict["hic"]["pca"].inverse_transform(z_seq[:,off:off+D])
            out["atac_orig"]= np.expm1(sc_a.inverse_transform(out["atac_std"]))
            out["hic_orig"] = np.expm1(sc_h.inverse_transform(out["hic_std"]))
            return out
        else:
            atac_std = pca_dict["atac"]["pca"].inverse_transform(z_seq)
            atac_orig= np.expm1(sc_a.inverse_transform(atac_std))
            return {"atac_std": atac_std, "atac_orig": atac_orig}

    pred_no = decode(z_no)

    # ---- rollout with assimilation ----
    if assim_on:
        idx_assim = {int(np.argmin(np.abs(times - t))) for t in assim_marks}
        z=z0.copy(); seq=[z.copy()]
        for i in range(1,len(times)):
            y = roll(model, z, 1)[-1]; z=y
            if i in idx_assim:
                if fused:
                    obs = {"atac": X_a_std[i], "hic": X_h_std[i]}
                    z   = fused_assim(z, obs, pca_dict, alpha)
                else:
                    z   = assimilate_block(z, X_a_std[i], pca_dict["atac"]["pca"], alpha)
            seq.append(z.copy())
        z_as = np.vstack(seq)
        pred_as = decode(z_as)
    else:
        pred_as = pred_no

    # ---------- Header KPIs ----------
    ccols = st.columns(3 if fused else 2)
    if fused:
        ya, pa = X_a_std, pred_as["atac_std"]
        yh, ph = X_h_std, pred_as["hic_std"]
        with ccols[0]: st.metric("ATAC R² (with-assim)", f"{r2(ya,pa):.3f}")
        with ccols[1]: st.metric("Hi-C R² (with-assim)", f"{r2(yh,ph):.3f}")
        with ccols[2]: st.metric("Mean |assim−no| (std)", f"{(np.abs(pred_as['atac_std']-pred_no['atac_std']).mean()+np.abs(pred_as['hic_std']-pred_no['hic_std']).mean()):.4f}")
    else:
        ya, pa = X_a_std, pred_as["atac_std"]
        with ccols[0]: st.metric("ATAC R² (with-assim)", f"{r2(ya,pa):.3f}")
        with ccols[1]: st.metric("ATAC RMSE (with-assim)", f"{rmse(ya,pa):.3f}")

    # ---------- RMSE vs time ----------
    st.subheader("RMSE vs Time")
    if fused:
        err_no_a = np.sqrt(((X_a_std - pred_no["atac_std"])**2).mean(axis=1))
        err_as_a = np.sqrt(((X_a_std - pred_as["atac_std"])**2).mean(axis=1))
        err_no_h = np.sqrt(((X_h_std - pred_no["hic_std"])**2).mean(axis=1))
        err_as_h = np.sqrt(((X_h_std - pred_as["hic_std"])**2).mean(axis=1))
        fig = go.Figure()
        line(fig, times, err_no_a, "ATAC No-Assim", CORAL, "dash")
        line(fig, times, err_as_a, "ATAC With-Assim", PRIMARY)
        line(fig, times, err_no_h, "Hi-C No-Assim", SEA, "dash")
        line(fig, times, err_as_h, "Hi-C With-Assim", INDIGO)
        fig.update_layout(height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        err_no = np.sqrt(((X_a_std - pred_no["atac_std"])**2).mean(axis=1))
        err_as = np.sqrt(((X_a_std - pred_as["atac_std"])**2).mean(axis=1))
        fig = go.Figure()
        line(fig, times, err_no, "No-Assim", CORAL, "dash")
        line(fig, times, err_as, "With-Assim", PRIMARY)
        fig.update_layout(height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Top peaks (std) ----------
    st.subheader("Feature Trajectories (std space)")
    if fused:
        var_a = X_a_std.var(axis=0); idx_a = np.argsort(var_a)[-peak_k:][::-1]
        var_h = X_h_std.var(axis=0); idx_h = np.argsort(var_h)[-peak_k:][::-1]
        ca, ch = st.columns(2)
        with ca:
            fig = go.Figure()
            for p in idx_a:
                line(fig, times, X_a_std[:,p], f"True {p}", "#708090")
                line(fig, times, pred_no["atac_std"][:,p], f"No-Assim {p}", CORAL, "dash")
                line(fig, times, pred_as["atac_std"][:,p], f"With-Assim {p}", PRIMARY)
            fig.update_layout(height=320, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        with ch:
            fig = go.Figure()
            for p in idx_h:
                line(fig, times, X_h_std[:,p], f"True {p}", "#708090")
                line(fig, times, pred_no["hic_std"][:,p], f"No-Assim {p}", SEA, "dash")
                line(fig, times, pred_as["hic_std"][:,p], f"With-Assim {p}", INDIGO)
            fig.update_layout(height=320, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
    else:
        var = X_a_std.var(axis=0); idx = np.argsort(var)[-peak_k:][::-1]
        fig = go.Figure()
        for p in idx:
            line(fig, times, X_a_std[:,p], f"True {p}", "#708090")
            line(fig, times, pred_no["atac_std"][:,p], f"No-Assim {p}", CORAL, "dash")
            line(fig, times, pred_as["atac_std"][:,p], f"With-Assim {p}", PRIMARY)
        fig.update_layout(height=320, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- True vs Pred (std) ----------
    st.subheader("True vs Predicted (std space)")
    t_sel = int(np.argmin(np.abs(times - 4.0)))
    rng = np.random.default_rng(0)
    if fused:
        # ATAC
        idx = rng.choice(X_a_std.shape[1], size=min(sample_points, X_a_std.shape[1]), replace=False)
        y, xno, xas = X_a_std[t_sel,idx], pred_no["atac_std"][t_sel,idx], pred_as["atac_std"][t_sel,idx]
        lim = (min(y.min(),xno.min(),xas.min()), max(y.max(),xno.max(),xas.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xno, y=y, mode="markers", name=f"ATAC No-Assim (R² {r2(y,xno):.3f}, RMSE {rmse(y,xno):.3f})",
                                 marker=dict(size=5,color=CORAL,opacity=.35)))
        fig.add_trace(go.Scatter(x=xas, y=y, mode="markers", name=f"ATAC With-Assim (R² {r2(y,xas):.3f}, RMSE {rmse(y,xas):.3f})",
                                 marker=dict(size=5,color=PRIMARY,opacity=.35)))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(color="#222", dash="dot")))
        fig.update_layout(height=360); st.plotly_chart(fig, use_container_width=True)
        # Hi-C
        idx = rng.choice(X_h_std.shape[1], size=min(sample_points, X_h_std.shape[1]), replace=False)
        y, xno, xas = X_h_std[t_sel,idx], pred_no["hic_std"][t_sel,idx], pred_as["hic_std"][t_sel,idx]
        lim = (min(y.min(),xno.min(),xas.min()), max(y.max(),xno.max(),xas.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xno, y=y, mode="markers", name=f"Hi-C No-Assim (R² {r2(y,xno):.3f}, RMSE {rmse(y,xno):.3f})",
                                 marker=dict(size=5,color=SEA,opacity=.35)))
        fig.add_trace(go.Scatter(x=xas, y=y, mode="markers", name=f"Hi-C With-Assim (R² {r2(y,xas):.3f}, RMSE {rmse(y,xas):.3f})",
                                 marker=dict(size=5,color=INDIGO,opacity=.35)))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(color="#222", dash="dot")))
        fig.update_layout(height=360); st.plotly_chart(fig, use_container_width=True)
    else:
        idx = rng.choice(X_a_std.shape[1], size=min(sample_points, X_a_std.shape[1]), replace=False)
        y, xno, xas = X_a_std[t_sel,idx], pred_no["atac_std"][t_sel,idx], pred_as["atac_std"][t_sel,idx]
        lim = (min(y.min(),xno.min(),xas.min()), max(y.max(),xno.max(),xas.max()))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xno, y=y, mode="markers", name=f"No-Assim (R² {r2(y,xno):.3f}, RMSE {rmse(y,xno):.3f})",
                                 marker=dict(size=5,color=CORAL,opacity=.35)))
        fig.add_trace(go.Scatter(x=xas, y=y, mode="markers", name=f"With-Assim (R² {r2(y,xas):.3f}, RMSE {rmse(y,xas):.3f})",
                                 marker=dict(size=5,color=PRIMARY,opacity=.35)))
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Ideal", line=dict(color="#222", dash="dot")))
        fig.update_layout(height=360); st.plotly_chart(fig, use_container_width=True)

    # ---------- Residual vs Pred overlay ----------
    st.subheader("Residual vs Predicted (overlay)")
    def resid_plot(y, xno, xas, title):
        res_no = y - xno; res_as = y - xas
        ymin = min(res_no.min(), res_as.min()); ymax = max(res_no.max(), res_as.max())
        pad = 0.05*(ymax - ymin + 1e-9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xno, y=res_no, mode="markers", name="No-Assim", marker=dict(size=4,color=CORAL,opacity=.3)))
        fig.add_trace(go.Scatter(x=xas, y=res_as, mode="markers", name="With-Assim", marker=dict(size=4,color=PRIMARY,opacity=.3)))
        fig.add_hline(y=0, line=dict(color="#888", dash="dot"))
        fig.update_layout(title=title, height=330, yaxis=dict(range=[ymin-pad, ymax+pad]))
        st.plotly_chart(fig, use_container_width=True)

    if fused:
        idx = rng.choice(X_a_std.shape[1], size=min(sample_points, X_a_std.shape[1]), replace=False)
        resid_plot(X_a_std[t_sel,idx], pred_no["atac_std"][t_sel,idx], pred_as["atac_std"][t_sel,idx], "ATAC")
        idx = rng.choice(X_h_std.shape[1], size=min(sample_points, X_h_std.shape[1]), replace=False)
        resid_plot(X_h_std[t_sel,idx], pred_no["hic_std"][t_sel,idx],  pred_as["hic_std"][t_sel,idx],  "Hi-C")
    else:
        idx = rng.choice(X_a_std.shape[1], size=min(sample_points, X_a_std.shape[1]), replace=False)
        resid_plot(X_a_std[t_sel,idx], pred_no["atac_std"][t_sel,idx], pred_as["atac_std"][t_sel,idx], "ATAC")

else:
    st.info("Set the config path and click **Load & Run** in the sidebar to start.")
