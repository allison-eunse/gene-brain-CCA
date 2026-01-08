#!/usr/bin/env python3
"""
Build X_gene for CCA from nesap-genomics DNABERT2 embeddings.

Input format (per gene, per chunk):
  <embed_root>/<GENE>/embeddings_{k}_layer_last.npy

This script produces one of:
  - subject × gene matrix (N × G) via per-gene feature reduction (mean over embedding dims)
  - subject × 512 matrix (N × 512) via PCA after reduction

Why this design?
  - Full N × (G*D) is huge; for CCA you typically want ~512 dims anyway.
  - We reduce each gene embedding (D dims) to 1 scalar per gene (mean/max/median),
    giving N × G (e.g., N~40k, G~112) which is cheap and stable.
  - Then we optionally PCA to 512 (or <=G) for a CCA-ready genetics representation.

If you prefer a different gene-side representation (e.g. concatenating D dims), adapt here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def read_gene_list(path: Path) -> List[str]:
    with path.open() as f:
        return [ln.strip() for ln in f if ln.strip()]


def _list_chunks(gene_dir: Path) -> List[Path]:
    # expects embeddings_{k}_layer_last.npy
    fps = sorted(gene_dir.glob("embeddings_*_layer_last.npy"))
    return [fp for fp in fps if fp.is_file()]


def _stack_gene_embeddings(gene_dir: Path, n_files: int) -> np.ndarray:
    parts = []
    for k in range(1, n_files + 1):
        fp = gene_dir / f"embeddings_{k}_layer_last.npy"
        if not fp.exists():
            raise FileNotFoundError(f"Missing chunk: {fp}")
        parts.append(np.load(fp, mmap_mode="r"))
    return np.concatenate(parts, axis=0)


def reduce_embedding(E: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.nanmean(E, axis=1)
    if method == "max":
        return np.nanmax(E, axis=1)
    if method == "median":
        return np.nanmedian(E, axis=1)
    raise ValueError(f"Unknown reduce method: {method}")


def pca_fit_transform(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    # local dependency-free PCA (SVD). For speed on large N, scikit-learn PCA is better,
    # but this keeps the script portable.
    rng = np.random.default_rng(seed)
    # center
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # randomized range finder
    # (N,D) -> (N,k)
    k = min(n_components, Xc.shape[1])
    P = rng.standard_normal((Xc.shape[1], k), dtype=np.float64)
    Z = Xc @ P
    # orthonormalize
    Q, _ = np.linalg.qr(Z, mode="reduced")
    # project and SVD in small space
    B = Q.T @ Xc
    Ub, Sb, Vtb = np.linalg.svd(B, full_matrices=False)
    # components in original space: Vtb[:k]
    # scores: Xc @ Vt.T
    W = Vtb[:k].T
    scores = Xc @ W
    return scores.astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", required=True, help="DNABERT2 embeddings root (contains gene subfolders)")
    ap.add_argument("--gene-list", required=True, help="Gene list txt, one gene per line")
    ap.add_argument("--iids", required=True, help="iids.npy (1D) aligned with embedding row order")
    ap.add_argument("--n-files", required=True, type=int, help="Number of chunks per gene (kmax), e.g. 49")
    ap.add_argument("--reduce", default="mean", choices=["mean", "max", "median"], help="Reduce D-dim embedding to 1 scalar per gene")
    ap.add_argument("--pca512", action="store_true", help="After reduction to N×G, run PCA to min(512,G)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    embed_root = Path(args.embed_root).expanduser().resolve()
    gene_list = Path(args.gene_list).expanduser().resolve()
    iids_path = Path(args.iids).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = read_gene_list(gene_list)
    iids = np.load(iids_path).astype(str)

    # Build X: N × G
    X_cols: List[np.ndarray] = []
    N: Optional[int] = None
    for g in genes:
        E = _stack_gene_embeddings(embed_root / g, args.n_files)
        if N is None:
            N = E.shape[0]
        elif E.shape[0] != N:
            raise ValueError(f"[{g}] row mismatch {E.shape[0]} != {N}")
        xg = reduce_embedding(E, args.reduce).astype(np.float32, copy=False)
        X_cols.append(xg)
        print(f"[gene] {g}: E={tuple(E.shape)} -> col={tuple(xg.shape)}", flush=True)

    X = np.stack(X_cols, axis=1)  # (N,G)
    if iids.shape[0] != X.shape[0]:
        raise ValueError(f"iids len {iids.shape[0]} != X rows {X.shape[0]}")

    np.save(out_dir / "ids_gene.npy", iids)
    np.save(out_dir / "X_gene_ng.npy", X)
    np.save(out_dir / "genes.npy", np.array(genes, dtype=object))

    if args.pca512:
        k = min(512, X.shape[1])
        Xp = pca_fit_transform(X.astype(np.float64, copy=False), n_components=k, seed=args.seed)
        np.save(out_dir / f"X_gene_pca{k}.npy", Xp)
        print(f"[save] X_gene_pca{k}.npy: {Xp.shape}", flush=True)

    print(f"[save] ids_gene.npy, X_gene_ng.npy, genes.npy in {out_dir}", flush=True)


if __name__ == "__main__":
    main()


