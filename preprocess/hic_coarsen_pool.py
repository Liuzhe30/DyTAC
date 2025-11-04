import os, argparse, numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_path", default="../data/hic_matrix.npy")
    p.add_argument("--out_path", default="../data/hic_coarse.npy")
    p.add_argument("--pool", type=int, default=1000)  # pool size along feature axis
    args = p.parse_args()

    X = np.load(args.in_path, mmap_mode="r")  # (T, L)
    T, L = X.shape
    m = L // args.pool
    r = L % args.pool

    Ys = []
    for t in range(T):
        row = X[t]
        core = row[:m*args.pool].reshape(m, args.pool).mean(axis=1)
        if r > 0:
            tail = row[m*args.pool:].mean()
            core = np.concatenate([core, [tail]])
        Ys.append(core.astype(np.float32))
    Y = np.stack(Ys)  # (T, m+1)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, Y)
    print("coarse HIC:", Y.shape, "pool", args.pool)

if __name__ == "__main__":
    main()
