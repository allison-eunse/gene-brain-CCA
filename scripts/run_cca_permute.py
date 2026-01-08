#!/usr/bin/env python3
"""
Run CCA between two matrices (already aligned by subject) with permutation testing.

Inputs:
  --x-gene: e.g. X_gene_pca{gene_pca_dim}.npy (N×d1)
  --x-fmri: e.g. X_fmri_pca512.npy (N×d2)
  --ids:    ids_common.npy (optional; saved to outputs for provenance)

Outputs:
  cca_results.json with:
    - canonical correlations (top K)
    - permutation p-values (shuffle rows of fmri)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cca_fit_transform(X: np.ndarray, Y: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple CCA via sklearn if available; fallback to a basic SVD-based approach is not included.
    """
    try:
        from sklearn.cross_decomposition import CCA  # type: ignore
    except Exception as e:
        raise SystemExit("[ERROR] scikit-learn is required for this script (sklearn.cross_decomposition.CCA).") from e

    cca = CCA(n_components=n_components, max_iter=5000)
    U, V = cca.fit_transform(X, Y)
    return U, V


def canon_corr(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    rs = []
    for i in range(min(U.shape[1], V.shape[1])):
        u = U[:, i]
        v = V[:, i]
        r = np.corrcoef(u, v)[0, 1]
        rs.append(float(r))
    return np.array(rs, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-gene", required=True)
    ap.add_argument("--x-fmri", required=True)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--n-components", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.x_gene, mmap_mode="r").astype(np.float64)
    Y = np.load(args.x_fmri, mmap_mode="r").astype(np.float64)
    if X.shape[0] != Y.shape[0]:
        raise SystemExit(f"[ERROR] row mismatch: X {X.shape} vs Y {Y.shape}")

    ncomp = min(args.n_components, X.shape[1], Y.shape[1])
    U, V = cca_fit_transform(X, Y, n_components=ncomp)
    r_obs = canon_corr(U, V)

    rng = np.random.default_rng(args.seed)
    perm_rs = np.zeros((args.n_perm, ncomp), dtype=np.float64)
    for b in range(args.n_perm):
        p = rng.permutation(X.shape[0])
        # Refit under permutation so the null reflects the best possible correlation
        # after destroying the sample-wise correspondence between X and Y.
        Up, Vp = cca_fit_transform(X, Y[p], n_components=ncomp)
        perm_rs[b] = canon_corr(Up, Vp)
        if (b + 1) % 100 == 0:
            print(f"[perm] {b+1}/{args.n_perm}", flush=True)

    # p-values: fraction of perm >= observed (one-sided)
    pvals = (np.sum(perm_rs >= r_obs[None, :], axis=0) + 1) / (args.n_perm + 1)

    res = {
        "n": int(X.shape[0]),
        "x_gene_dim": int(X.shape[1]),
        "x_fmri_dim": int(Y.shape[1]),
        "n_components": int(ncomp),
        "r_observed": [float(x) for x in r_obs.tolist()],
        "p_perm": [float(x) for x in pvals.tolist()],
        "n_perm": int(args.n_perm),
        "seed": int(args.seed),
    }
    if args.ids:
        ids = np.load(args.ids, allow_pickle=True).astype(str)
        np.save(out_dir / "ids_common.npy", ids)

    (out_dir / "cca_results.json").write_text(json.dumps(res, indent=2))
    np.save(out_dir / "perm_rs.npy", perm_rs.astype(np.float32))
    print(f"[save] {out_dir/'cca_results.json'}", flush=True)


if __name__ == "__main__":
    main()


