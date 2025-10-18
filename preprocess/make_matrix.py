import os, re
import numpy as np
import pandas as pd

ATAC_DIR = "../data/raw/ATAC"
HIC_DIR = "../data/raw/HiC"
OUT_DIR = "../data"

TIME_MAP = {
    "T0": 0.0,
    "T20": 0.33,
    "T1H": 1.0,
    "T4H": 4.0,
    "T24H": 24.0
}

def get_time(name):
    for k in TIME_MAP:
        if k in name:
            return TIME_MAP[k]
    return None

def load_atac(path):
    cols = ["chrom","start","end","name","score","strand","signal","p","q","peak"]
    df = pd.read_csv(path, sep="\t", header=None, names=cols, usecols=["chrom","start","end","signal"])
    df["id"] = df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str)
    return df[["id","signal"]]

def build_atac_matrix():
    files = sorted([f for f in os.listdir(ATAC_DIR) if f.endswith(".narrowPeak")])
    all_ids = set()
    mats, times = [], []
    for f in files:
        t = get_time(f)
        if t is None: 
            continue
        df = load_atac(os.path.join(ATAC_DIR, f))
        all_ids.update(df["id"])
        mats.append((t, df))
    all_ids = sorted(list(all_ids))
    df_all = pd.DataFrame(index=all_ids)
    mats_sorted = sorted(mats, key=lambda x: x[0])
    for t, df in mats_sorted:
        df_all[f"time_{t}"] = df_all.index.map(df.set_index("id")["signal"])
    df_all = df_all.fillna(0.0)
    mat = df_all.T.to_numpy()
    times = [t for t, _ in mats_sorted]
    np.save(os.path.join(OUT_DIR, "atac_matrix.npy"), mat.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "timepoints.npy"), np.array(times, dtype=np.float32))
    print("ATAC:", mat.shape)
    return times

def load_hic_matrix(folder):
    chroms = sorted([f for f in os.listdir(folder) if re.search(r"chr\d+$", f)])
    vecs = []
    for c in chroms:
        p = os.path.join(folder, c)
        try:
            df = pd.read_csv(p, sep="\t", header=0, index_col=0, engine="python")
        except Exception:
            df = pd.read_csv(p, sep="\t", header=None, engine="python")
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        n = min(df.shape[0], df.shape[1])
        if n == 0:
            continue
        df = df.iloc[:n, :n]
        mat = df.to_numpy()
        tri = mat[np.triu_indices(n)]
        vecs.append(tri)
    if not vecs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vecs)

def build_hic_matrix():
    subs = sorted([d for d in os.listdir(HIC_DIR) if os.path.isdir(os.path.join(HIC_DIR, d))])
    mats, times = [], []
    for d in subs:
        t = get_time(d)
        if t is None:
            continue
        vec = load_hic_matrix(os.path.join(HIC_DIR, d))
        mats.append((t, vec))
        print("HiC:", d, vec.shape)
    mats_sorted = sorted(mats, key=lambda x: x[0])
    lengths = [len(v) for _, v in mats_sorted]
    if any(l == 0 for l in lengths):
        raise ValueError("Empty Hi-C vector detected; check parsing.")
    min_len = min(lengths)
    print(f"Aligning HiC vectors to length {min_len}")
    aligned = [v[:min_len] for _, v in mats_sorted]
    matrix = np.stack(aligned)
    np.save(os.path.join(OUT_DIR, "hic_matrix.npy"), matrix.astype(np.float32))
    print("HiC:", matrix.shape)
    return [t for t, _ in mats_sorted]


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    t1 = build_atac_matrix()
    t2 = build_hic_matrix()
    print("Done. Timepoints:", t1)

'''
ATAC: (5, 179964)
HiC: T0_HiC_pool1 (139504454,)
HiC: T1H_HiC_pool1 (139501904,)
HiC: T20_HiC_pool1 (139494693,)
HiC: T24H_HiC_pool1 (139537753,)
HiC: T4H_HiC_pool1 (139499648,)
Aligning HiC vectors to length 139494693
HiC: (5, 139494693)
Done. Timepoints: [0.0, 0.33, 1.0, 4.0, 24.0]
'''