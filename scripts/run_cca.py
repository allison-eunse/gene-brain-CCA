#!/usr/bin/env python3
"""
Stage 1: Unsupervised CCA/SCCA for Gene-Brain Joint Embeddings

This script performs Canonical Correlation Analysis (CCA) to find linear combinations
of gene and brain features that maximize their correlation, producing joint embeddings
(canonical variates) for downstream clinical prediction.

Supports:
  - Conventional CCA: sklearn.cross_decomposition.CCA
  - Sparse CCA (SCCA): L1-penalized CCA for feature selection and denoising

Inputs:
  --x-gene: Gene matrix (N × D_gene), e.g., X_gene_pca{gene_pca_dim}.npy
  --x-fmri: fMRI matrix (N × D_fmri), e.g., X_fmri_pca{fmri_pca_dim}.npy
  --ids: (optional) ids_common.npy for provenance

Outputs:
  - U_gene.npy: Gene canonical variates (N × n_components)
  - V_fmri.npy: fMRI canonical variates (N × n_components)
  - W_gene.npy: Gene loadings/weights (D_gene × n_components)
  - W_fmri.npy: fMRI loadings/weights (D_fmri × n_components)
  - cca_results.json: Canonical correlations, method info, permutation p-values
  - perm_rs.npy: Permutation null distribution (if --n-perm > 0)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize to zero mean, unit variance."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu.squeeze(), sd.squeeze()


def conventional_cca(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    max_iter: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Conventional CCA using sklearn.

    Returns: (U, V, W_x, W_y, correlations)
      - U: X-side canonical variates (N × k)
      - V: Y-side canonical variates (N × k)
      - W_x: X-side loadings (D_x × k)
      - W_y: Y-side loadings (D_y × k)
      - correlations: canonical correlations (k,)
    """
    try:
        from sklearn.cross_decomposition import CCA
    except ImportError as e:
        raise SystemExit(
            "[ERROR] scikit-learn required. Install: pip install scikit-learn"
        ) from e

    cca = CCA(n_components=n_components, max_iter=max_iter)
    # IMPORTANT: use paired transform/fit_transform so we get BOTH U (X scores) and V (Y scores).
    # Using `cca.transform(Y)` is incorrect because it treats Y as X and returns X-side scores.
    U, V = cca.fit_transform(X, Y)

    # Compute correlations
    correlations = np.array(
        [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)]
    )

    # Canonical coefficients (most interpretable weights for the original features)
    W_x = cca.x_weights_
    W_y = cca.y_weights_

    return U, V, W_x, W_y, correlations


def sparse_cca(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    c1: float = 0.3,
    c2: float = 0.3,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sparse CCA using Penalized Matrix Decomposition (PMD) approach.

    This implements Witten et al. (2009) sparse CCA with L1 penalties on the
    canonical weight vectors, promoting sparsity for interpretability.

    Args:
        X, Y: Data matrices (N × D_x), (N × D_y)
        n_components: Number of canonical components
        c1, c2: Sparsity parameters for X and Y (0 < c <= 1).
                Lower values = more sparsity.
        max_iter: Maximum iterations for alternating optimization
        tol: Convergence tolerance

    Returns: (U, V, W_x, W_y, correlations)
    """
    try:
        from cca_zoo.linear import SCCA_PMD
    except ImportError:
        # Fallback to a simple iterative sparse CCA if cca-zoo not available
        print("[WARN] cca-zoo not found, using fallback sparse CCA implementation")
        return _sparse_cca_fallback(X, Y, n_components, c1, c2, max_iter, tol)

    scca = SCCA_PMD(
        latent_dimensions=n_components,
        c=[c1, c2],
        max_iter=max_iter,
        tol=tol,
    )
    scca.fit([X, Y])

    # Transform to get canonical variates
    U = scca.transform([X, Y])[0]
    V = scca.transform([X, Y])[1]

    # Get weights
    W_x = scca.weights[0]
    W_y = scca.weights[1]

    # Compute correlations
    correlations = np.array(
        [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)]
    )

    return U, V, W_x, W_y, correlations


def _sparse_cca_fallback(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int,
    c1: float = 0.3,
    c2: float = 0.3,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fallback sparse CCA using soft-thresholding (simplified PMD-like approach).

    This implements an iterative algorithm with L1 regularization on the canonical
    weight vectors using soft thresholding.
    """
    N, Dx = X.shape
    _, Dy = Y.shape
    k = n_components

    # Standardize
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

    # Compute cross-covariance
    Cxy = X.T @ Y / N

    W_x = np.zeros((Dx, k), dtype=np.float64)
    W_y = np.zeros((Dy, k), dtype=np.float64)

    def soft_threshold(v: np.ndarray, lam: float) -> np.ndarray:
        """L1 soft thresholding."""
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0)

    def l1_project(w: np.ndarray, c: float) -> np.ndarray:
        """Project onto L1 ball and normalize."""
        # Scale c to be proportion of max possible L1 norm
        max_l1 = np.sqrt(len(w))
        target_l1 = c * max_l1

        w_sign = np.sign(w)
        w_abs = np.abs(w)

        if w_abs.sum() <= target_l1:
            # Already within L1 ball
            norm = np.linalg.norm(w)
            return w / norm if norm > 0 else w

        # Binary search for threshold
        lo, hi = 0.0, w_abs.max()
        for _ in range(50):
            mid = (lo + hi) / 2
            w_shrunk = np.maximum(w_abs - mid, 0)
            if w_shrunk.sum() > target_l1:
                lo = mid
            else:
                hi = mid

        w_out = w_sign * np.maximum(w_abs - lo, 0)
        norm = np.linalg.norm(w_out)
        return w_out / norm if norm > 0 else w_out

    for comp in range(k):
        # Deflate cross-covariance for subsequent components
        if comp > 0:
            for c_prev in range(comp):
                wx_prev = W_x[:, c_prev : c_prev + 1]
                wy_prev = W_y[:, c_prev : c_prev + 1]
                Cxy = Cxy - (wx_prev @ wy_prev.T) * (wx_prev.T @ Cxy @ wy_prev)

        # Initialize with SVD
        U, S, Vt = np.linalg.svd(Cxy, full_matrices=False)
        wx = U[:, 0].copy()
        wy = Vt[0, :].copy()

        # Alternating optimization with L1 projection
        for it in range(max_iter):
            wx_old = wx.copy()
            wy_old = wy.copy()

            # Update wx: wx ∝ Cxy @ wy, then L1 project
            wx = Cxy @ wy
            wx = l1_project(wx, c1)

            # Update wy: wy ∝ Cxy.T @ wx, then L1 project
            wy = Cxy.T @ wx
            wy = l1_project(wy, c2)

            # Check convergence
            dx = np.linalg.norm(wx - wx_old)
            dy = np.linalg.norm(wy - wy_old)
            if dx < tol and dy < tol:
                break

        W_x[:, comp] = wx
        W_y[:, comp] = wy

    # Compute canonical variates and correlations
    U = X @ W_x
    V = Y @ W_y

    correlations = np.array(
        [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)]
    )

    return U, V, W_x, W_y, correlations


def run_permutation_test(
    X: np.ndarray,
    Y: np.ndarray,
    r_observed: np.ndarray,
    method: str,
    n_components: int,
    n_perm: int,
    seed: int,
    c1: float = 0.3,
    c2: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permutation test for CCA canonical correlations.

    Returns: (perm_rs, p_values)
    """
    rng = np.random.default_rng(seed)
    perm_rs = np.zeros((n_perm, n_components), dtype=np.float64)

    for b in range(n_perm):
        perm_idx = rng.permutation(X.shape[0])
        Y_perm = Y[perm_idx]

        if method == "conventional":
            _, _, _, _, r_perm = conventional_cca(X, Y_perm, n_components)
        else:
            _, _, _, _, r_perm = sparse_cca(X, Y_perm, n_components, c1, c2)

        perm_rs[b] = r_perm

        if (b + 1) % 100 == 0:
            print(f"[perm] {b + 1}/{n_perm}", flush=True)

    # One-sided p-values
    p_values = (np.sum(perm_rs >= r_observed[None, :], axis=0) + 1) / (n_perm + 1)

    return perm_rs, p_values


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1: CCA/SCCA for gene-brain joint embeddings"
    )

    # Input data
    ap.add_argument("--x-gene", required=True, help="Gene matrix (N × D_gene)")
    ap.add_argument("--x-fmri", required=True, help="fMRI matrix (N × D_fmri)")
    ap.add_argument("--ids", default=None, help="Subject IDs (optional, for provenance)")

    # CCA method
    ap.add_argument(
        "--method",
        choices=["conventional", "sparse"],
        default="conventional",
        help="CCA method: conventional or sparse (SCCA)",
    )
    ap.add_argument(
        "--n-components",
        type=int,
        default=10,
        help="Number of canonical components (default: 10)",
    )

    # Sparse CCA parameters
    ap.add_argument(
        "--c1",
        type=float,
        default=0.3,
        help="SCCA sparsity for gene (0 < c1 <= 1, lower = sparser, default: 0.3)",
    )
    ap.add_argument(
        "--c2",
        type=float,
        default=0.3,
        help="SCCA sparsity for fMRI (0 < c2 <= 1, lower = sparser, default: 0.3)",
    )

    # Permutation testing
    ap.add_argument(
        "--n-perm",
        type=int,
        default=0,
        help="Number of permutations (0 = skip permutation test)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument(
        "--prefix",
        default="",
        help="Output file prefix (e.g., 'cca_' or 'scca_')",
    )

    args = ap.parse_args()

    # Setup output
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (args.method + "_")

    # Load data
    print(f"[load] Gene: {args.x_gene}", flush=True)
    print(f"[load] fMRI: {args.x_fmri}", flush=True)

    X = np.load(args.x_gene, mmap_mode="r").astype(np.float64)
    Y = np.load(args.x_fmri, mmap_mode="r").astype(np.float64)

    if X.shape[0] != Y.shape[0]:
        raise SystemExit(f"[ERROR] Row mismatch: X {X.shape} vs Y {Y.shape}")

    N = X.shape[0]
    n_comp = min(args.n_components, X.shape[1], Y.shape[1])

    print(f"[info] N={N}, gene_dim={X.shape[1]}, fmri_dim={Y.shape[1]}", flush=True)
    print(f"[info] Method: {args.method}, n_components={n_comp}", flush=True)

    # Standardize
    X, _, _ = standardize(np.asarray(X))
    Y, _, _ = standardize(np.asarray(Y))

    # Run CCA
    if args.method == "conventional":
        print("[cca] Running conventional CCA...", flush=True)
        U, V, W_x, W_y, r_obs = conventional_cca(X, Y, n_comp)
    else:
        print(f"[scca] Running sparse CCA (c1={args.c1}, c2={args.c2})...", flush=True)
        U, V, W_x, W_y, r_obs = sparse_cca(X, Y, n_comp, args.c1, args.c2)

    print(f"[result] Canonical correlations: {r_obs.round(4).tolist()}", flush=True)

    # Compute sparsity (fraction of near-zero weights)
    sparsity_gene = float((np.abs(W_x) < 1e-6).sum() / W_x.size)
    sparsity_fmri = float((np.abs(W_y) < 1e-6).sum() / W_y.size)
    print(f"[result] Sparsity: gene={sparsity_gene:.2%}, fmri={sparsity_fmri:.2%}", flush=True)

    # Permutation test
    perm_rs = None
    p_perm = None
    if args.n_perm > 0:
        print(f"[perm] Running {args.n_perm} permutations...", flush=True)
        perm_rs, p_perm = run_permutation_test(
            X, Y, r_obs, args.method, n_comp, args.n_perm, args.seed, args.c1, args.c2
        )
        print(f"[result] Permutation p-values: {p_perm.round(4).tolist()}", flush=True)

    # Save canonical variates (joint embeddings for Stage 2)
    np.save(out_dir / f"{prefix}U_gene.npy", U.astype(np.float32))
    np.save(out_dir / f"{prefix}V_fmri.npy", V.astype(np.float32))

    # Save weights/loadings (for interpretation)
    np.save(out_dir / f"{prefix}W_gene.npy", W_x.astype(np.float32))
    np.save(out_dir / f"{prefix}W_fmri.npy", W_y.astype(np.float32))

    # Save permutation results
    if perm_rs is not None:
        np.save(out_dir / f"{prefix}perm_rs.npy", perm_rs.astype(np.float32))

    # Copy IDs for provenance
    if args.ids:
        ids = np.load(args.ids, allow_pickle=True).astype(str)
        np.save(out_dir / f"{prefix}ids.npy", ids)

    # Save results JSON
    results = {
        "method": args.method,
        "n_subjects": int(N),
        "gene_dim": int(X.shape[1]),
        "fmri_dim": int(Y.shape[1]),
        "n_components": int(n_comp),
        "canonical_correlations": [float(r) for r in r_obs.tolist()],
        "sparsity_gene": sparsity_gene,
        "sparsity_fmri": sparsity_fmri,
        "seed": args.seed,
    }
    if args.method == "sparse":
        results["c1"] = args.c1
        results["c2"] = args.c2
    if p_perm is not None:
        results["n_perm"] = args.n_perm
        results["p_perm"] = [float(p) for p in p_perm.tolist()]

    with open(out_dir / f"{prefix}results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"[save] Outputs saved to {out_dir} with prefix '{prefix}'", flush=True)


if __name__ == "__main__":
    main()


