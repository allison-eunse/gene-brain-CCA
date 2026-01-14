#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-wide", required=True)
    ap.add_argument("--n-components", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    X = np.load(args.x_wide, mmap_mode="r")
    k = min(args.n_components, X.shape[1])
    pca = PCA(n_components=k, svd_solver="randomized", random_state=args.seed)
    Xp = pca.fit_transform(X)
    np.save(args.out, Xp.astype(np.float32))
    print(f"[save] {args.out}: {Xp.shape}")

if __name__ == "__main__":
    main()