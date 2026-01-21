#!/usr/bin/env python3
"""
Stratified CCA/SCCA Benchmark: Run MDD and Controls separately.

This script runs the same CCA/SCCA grid search as run_coupling_benchmark_v2.py,
but separately for MDD (label=1) and Control (label=0) subjects.

Key additions:
- Split by label before train/holdout split
- Run identical grid on both subsets
- Compare weights via cosine similarity
- Permutation testing for significance of r_mdd - r_ctrl difference

Lab server rules:
- Run via Slurm (sbatch/srun), not on login node
- No manual CUDA_VISIBLE_DEVICES
- Set OMP/MKL/OPENBLAS thread counts via SLURM_CPUS_PER_TASK
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ShuffleSplit

sys.path.insert(0, str(Path(__file__).parent.parent / "gene-brain-cca-2" / "scripts"))
from scca_pmd import SCCA_PMD


# -----------------------------------------------------------------------------
# Helper functions (copied from run_coupling_benchmark_v2.py)
# -----------------------------------------------------------------------------

def _as_2d(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"cov_extra must be 1D or 2D, got shape={x.shape}")


def _build_cov(age: np.ndarray, sex: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    cols = [
        np.ones(len(age), dtype=np.float64),
        age.astype(np.float64),
        sex.astype(np.float64),
    ]
    if extra is not None:
        extra = _as_2d(extra)
        cols.append(extra.astype(np.float64))
    return np.column_stack(cols)


def _residualize_train_val(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    age_tr: np.ndarray,
    sex_tr: np.ndarray,
    age_va: np.ndarray,
    sex_va: np.ndarray,
    extra_tr: np.ndarray | None = None,
    extra_va: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    C_tr = _build_cov(age_tr, sex_tr, extra_tr)
    CtC = C_tr.T @ C_tr
    CtX = C_tr.T @ X_tr.astype(np.float64, copy=False)
    B = np.linalg.solve(CtC, CtX)
    X_tr_r = X_tr.astype(np.float64, copy=False) - (C_tr @ B)
    C_va = _build_cov(age_va, sex_va, extra_va)
    X_va_r = X_va.astype(np.float64, copy=False) - (C_va @ B)
    return X_tr_r.astype(np.float32, copy=False), X_va_r.astype(np.float32, copy=False)


def _standardize_train_val_f32(
    X_tr: np.ndarray, X_va: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    X_tr_s = (X_tr - mu) / sd
    X_va_s = (X_va - mu) / sd
    return (
        X_tr_s.astype(np.float32, copy=False),
        X_va_s.astype(np.float32, copy=False),
        (mu.astype(np.float32, copy=False), sd.astype(np.float32, copy=False)),
    )


def _pca_train_val_maxdim(
    X_tr: np.ndarray, X_va: np.ndarray, max_components: int, seed: int
) -> tuple[np.ndarray, np.ndarray, PCA]:
    max_components = min(max_components, X_tr.shape[0], X_tr.shape[1])
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr).astype(np.float32, copy=False)
    X_va_pca = pca.transform(X_va).astype(np.float32, copy=False)
    return X_tr_pca, X_va_pca, pca


def compute_canonical_correlations(U: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    k = min(k, U.shape[1], V.shape[1])
    r = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)])
    return r


def _fit_and_score_with_weights(
    *,
    method: str,
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_te: np.ndarray,
    Xb_te: np.ndarray,
    k: int,
    c1: float,
    c2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit CCA/SCCA and return train/test correlations + weights."""
    if k <= 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0)), np.zeros((0, 0))

    if method == "cca":
        model = CCA(n_components=k, max_iter=500)
        model.fit(Xg_tr, Xb_tr)
        U_tr, V_tr = model.transform(Xg_tr, Xb_tr)
        U_te, V_te = model.transform(Xg_te, Xb_te)
        W_gene = model.x_weights_
        W_brain = model.y_weights_
    elif method == "scca":
        model = SCCA_PMD(latent_dimensions=k, c=[c1, c2], max_iter=500, tol=1e-6)
        model.fit([Xg_tr, Xb_tr])
        U_tr, V_tr = model.transform([Xg_tr, Xb_tr])
        U_te, V_te = model.transform([Xg_te, Xb_te])
        W_gene = model.weights_[0]
        W_brain = model.weights_[1]
    else:
        raise ValueError(f"Unknown method: {method}")

    r_train = compute_canonical_correlations(U_tr, V_tr, k)
    r_test = compute_canonical_correlations(U_te, V_te, k)
    return r_train, r_test, W_gene, W_brain


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))


# -----------------------------------------------------------------------------
# Main stratified benchmark
# -----------------------------------------------------------------------------

def run_group_benchmark(
    *,
    group_name: str,
    Xg: np.ndarray,
    Xb: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    extra: np.ndarray | None,
    holdout_frac: float,
    n_folds: int,
    seed: int,
    gene_pca_dims: list[int],
    methods: list[str],
    c_values: list[float],
    n_components: int,
) -> dict:
    """Run CCA/SCCA benchmark on a single group (MDD or Ctrl)."""
    n = Xg.shape[0]
    print(f"[{group_name}] N={n}, gene_dim={Xg.shape[1]}, brain_dim={Xb.shape[1]}")

    # Train/holdout split (random, not stratified since all same label)
    ss = ShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_idx, hold_idx = next(ss.split(np.zeros(n)))
    print(f"[{group_name}] Train: {len(train_idx)}, Holdout: {len(hold_idx)}")

    # CV folds on training set
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = list(kf.split(np.zeros(len(train_idx))))

    # Gene caching
    max_gene_dim = max(gene_pca_dims)
    max_gene_dim = min(max_gene_dim, len(train_idx) - 1, Xg.shape[1])

    # Precompute gene cache for folds
    gene_fold_cache = []
    for fold_idx, (tr_rel, va_rel) in enumerate(fold_splits):
        tr = train_idx[tr_rel]
        va = train_idx[va_rel]
        Xg_tr_r, Xg_va_r = _residualize_train_val(
            Xg[tr], Xg[va],
            age[tr], sex[tr], age[va], sex[va],
            extra_tr=None if extra is None else extra[tr],
            extra_va=None if extra is None else extra[va],
        )
        Xg_tr_s, Xg_va_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_va_r)
        Xg_tr_pca, Xg_va_pca, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_va_s, max_gene_dim, seed)
        gene_fold_cache.append({"tr": tr, "va": va, "Xg_tr_pca": Xg_tr_pca, "Xg_va_pca": Xg_va_pca})

    # Precompute gene cache for holdout
    Xg_tr_r, Xg_ho_r = _residualize_train_val(
        Xg[train_idx], Xg[hold_idx],
        age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
        extra_tr=None if extra is None else extra[train_idx],
        extra_va=None if extra is None else extra[hold_idx],
    )
    Xg_tr_s, Xg_ho_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_ho_r)
    Xg_tr_pca, Xg_ho_pca, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_ho_s, max_gene_dim, seed)

    # Precompute brain cache for folds
    brain_fold_cache = []
    for fold_idx, (tr_rel, va_rel) in enumerate(fold_splits):
        tr = train_idx[tr_rel]
        va = train_idx[va_rel]
        Xb_tr_r, Xb_va_r = _residualize_train_val(
            Xb[tr], Xb[va],
            age[tr], sex[tr], age[va], sex[va],
            extra_tr=None if extra is None else extra[tr],
            extra_va=None if extra is None else extra[va],
        )
        Xb_tr_s, Xb_va_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_va_r)
        brain_fold_cache.append({"Xb_tr_s": Xb_tr_s, "Xb_va_s": Xb_va_s})

    # Precompute brain cache for holdout
    Xb_tr_r, Xb_ho_r = _residualize_train_val(
        Xb[train_idx], Xb[hold_idx],
        age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
        extra_tr=None if extra is None else extra[train_idx],
        extra_va=None if extra is None else extra[hold_idx],
    )
    Xb_tr_s, Xb_ho_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_ho_r)

    # Grid search
    all_results = []
    effective_gene_dims = sorted({min(d, Xg_tr_pca.shape[1]) for d in gene_pca_dims})

    for method in methods:
        for gene_dim in effective_gene_dims:
            c_grid = [(1.0, 1.0)] if method == "cca" else list(product(c_values, c_values))
            for c1, c2 in c_grid:
                config_label = f"{method}_pca{gene_dim}" if method == "cca" else f"{method}_pca{gene_dim}_c{c1}_{c2}"

                # CV
                fold_r_val = []
                fold_r_train = []
                for fold_idx, gfc in enumerate(gene_fold_cache):
                    bfc = brain_fold_cache[fold_idx]
                    Xg_tr_f = gfc["Xg_tr_pca"][:, :gene_dim]
                    Xg_va_f = gfc["Xg_va_pca"][:, :gene_dim]
                    k = min(n_components, Xg_tr_f.shape[1], bfc["Xb_tr_s"].shape[1])
                    r_train, r_val, _, _ = _fit_and_score_with_weights(
                        method=method,
                        Xg_tr=Xg_tr_f, Xb_tr=bfc["Xb_tr_s"],
                        Xg_te=Xg_va_f, Xb_te=bfc["Xb_va_s"],
                        k=k, c1=c1, c2=c2,
                    )
                    fold_r_train.append(r_train[0] if len(r_train) > 0 else 0.0)
                    fold_r_val.append(r_val[0] if len(r_val) > 0 else 0.0)

                mean_r_val = float(np.mean(fold_r_val))
                std_r_val = float(np.std(fold_r_val))
                mean_r_train = float(np.mean(fold_r_train))

                # Holdout
                Xg_tr_ho = Xg_tr_pca[:, :gene_dim]
                Xg_ho_f = Xg_ho_pca[:, :gene_dim]
                k = min(n_components, Xg_tr_ho.shape[1], Xb_tr_s.shape[1])
                r_train_final, r_holdout, W_gene, W_brain = _fit_and_score_with_weights(
                    method=method,
                    Xg_tr=Xg_tr_ho, Xb_tr=Xb_tr_s,
                    Xg_te=Xg_ho_f, Xb_te=Xb_ho_s,
                    k=k, c1=c1, c2=c2,
                )

                result = {
                    "config": config_label,
                    "method": method,
                    "gene_pca_dim": gene_dim,
                    "c1": c1,
                    "c2": c2,
                    "mean_r_val_cc1": mean_r_val,
                    "std_r_val_cc1": std_r_val,
                    "mean_r_train_cc1": mean_r_train,
                    "overfitting_gap": mean_r_train - mean_r_val,
                    "r_holdout_cc1": float(r_holdout[0]) if len(r_holdout) > 0 else 0.0,
                    "r_train_final_cc1": float(r_train_final[0]) if len(r_train_final) > 0 else 0.0,
                    "generalization_gap": float(r_train_final[0] - r_holdout[0]) if len(r_train_final) > 0 else 0.0,
                }
                all_results.append(result)

    # Find best config by CV mean
    best = max(all_results, key=lambda x: x["mean_r_val_cc1"])

    # Refit best to get weights
    gene_dim = best["gene_pca_dim"]
    Xg_tr_ho = Xg_tr_pca[:, :gene_dim]
    Xg_ho_f = Xg_ho_pca[:, :gene_dim]
    k = min(n_components, Xg_tr_ho.shape[1], Xb_tr_s.shape[1])
    _, _, W_gene_best, W_brain_best = _fit_and_score_with_weights(
        method=best["method"],
        Xg_tr=Xg_tr_ho, Xb_tr=Xb_tr_s,
        Xg_te=Xg_ho_f, Xb_te=Xb_ho_s,
        k=k, c1=best["c1"], c2=best["c2"],
    )

    return {
        "group": group_name,
        "n_total": n,
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "all_results": all_results,
        "best_config": best,
        "W_gene": W_gene_best,
        "W_brain": W_brain_best,
        "train_idx": train_idx,
        "hold_idx": hold_idx,
    }


def run_permutation_test(
    *,
    Xg: np.ndarray,
    Xb: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    extra: np.ndarray | None,
    labels: np.ndarray,
    observed_diff: float,
    n_perm: int,
    holdout_frac: float,
    n_folds: int,
    seed: int,
    gene_pca_dim: int,
    method: str,
    c1: float,
    c2: float,
    n_components: int,
) -> dict:
    """
    Permutation test: shuffle labels, run stratified CCA, compute r_mdd - r_ctrl.
    Returns null distribution and p-value.
    """
    rng = np.random.RandomState(seed)
    null_diffs = []

    for perm_i in range(n_perm):
        if (perm_i + 1) % 100 == 0:
            print(f"  [perm] {perm_i + 1}/{n_perm}", flush=True)

        # Shuffle labels
        perm_labels = rng.permutation(labels)
        mdd_idx = np.where(perm_labels == 1)[0]
        ctrl_idx = np.where(perm_labels == 0)[0]

        # Run simplified benchmark on each group (just holdout r, no CV)
        r_mdd = _quick_holdout_r(
            Xg=Xg[mdd_idx], Xb=Xb[mdd_idx],
            age=age[mdd_idx], sex=sex[mdd_idx],
            extra=None if extra is None else extra[mdd_idx],
            holdout_frac=holdout_frac,
            seed=seed + perm_i,
            gene_pca_dim=gene_pca_dim,
            method=method, c1=c1, c2=c2, n_components=n_components,
        )
        r_ctrl = _quick_holdout_r(
            Xg=Xg[ctrl_idx], Xb=Xb[ctrl_idx],
            age=age[ctrl_idx], sex=sex[ctrl_idx],
            extra=None if extra is None else extra[ctrl_idx],
            holdout_frac=holdout_frac,
            seed=seed + perm_i,
            gene_pca_dim=gene_pca_dim,
            method=method, c1=c1, c2=c2, n_components=n_components,
        )
        null_diffs.append(r_mdd - r_ctrl)

    null_diffs = np.array(null_diffs)
    p_value = float(np.mean(np.abs(null_diffs) >= np.abs(observed_diff)))
    return {
        "observed_diff": observed_diff,
        "null_mean": float(np.mean(null_diffs)),
        "null_std": float(np.std(null_diffs)),
        "p_value": p_value,
        "n_perm": n_perm,
    }


def _quick_holdout_r(
    *,
    Xg: np.ndarray,
    Xb: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    extra: np.ndarray | None,
    holdout_frac: float,
    seed: int,
    gene_pca_dim: int,
    method: str,
    c1: float,
    c2: float,
    n_components: int,
) -> float:
    """Quick single holdout evaluation for permutation testing."""
    n = Xg.shape[0]
    if n < 10:
        return 0.0

    ss = ShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_idx, hold_idx = next(ss.split(np.zeros(n)))

    # Residualize + standardize
    Xg_tr_r, Xg_ho_r = _residualize_train_val(
        Xg[train_idx], Xg[hold_idx],
        age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
        extra_tr=None if extra is None else extra[train_idx],
        extra_va=None if extra is None else extra[hold_idx],
    )
    Xg_tr_s, Xg_ho_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_ho_r)

    # PCA
    max_dim = min(gene_pca_dim, len(train_idx) - 1, Xg_tr_s.shape[1])
    if max_dim < 1:
        return 0.0
    Xg_tr_pca, Xg_ho_pca, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_ho_s, max_dim, seed)

    Xb_tr_r, Xb_ho_r = _residualize_train_val(
        Xb[train_idx], Xb[hold_idx],
        age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
        extra_tr=None if extra is None else extra[train_idx],
        extra_va=None if extra is None else extra[hold_idx],
    )
    Xb_tr_s, Xb_ho_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_ho_r)

    k = min(n_components, Xg_tr_pca.shape[1], Xb_tr_s.shape[1])
    if k < 1:
        return 0.0

    _, r_holdout, _, _ = _fit_and_score_with_weights(
        method=method,
        Xg_tr=Xg_tr_pca, Xb_tr=Xb_tr_s,
        Xg_te=Xg_ho_pca, Xb_te=Xb_ho_s,
        k=k, c1=c1, c2=c2,
    )
    return float(r_holdout[0]) if len(r_holdout) > 0 else 0.0


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stratified CCA/SCCA benchmark")
    ap.add_argument("--x-gene-wide", required=True, help="Gene embedding matrix (N x D)")
    ap.add_argument("--x-brain", required=True, help="Brain feature matrix (N x D)")
    ap.add_argument("--labels", required=True, help="Labels (0=ctrl, 1=mdd)")
    ap.add_argument("--cov-age", required=True, help="Age covariate")
    ap.add_argument("--cov-sex", required=True, help="Sex covariate")
    ap.add_argument("--cov-extra", default=None, help="Extra covariate (e.g., eTIV)")
    ap.add_argument("--ids", required=True, help="Subject IDs")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--modality-tag", required=True, help="Modality tag for output naming")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gene-pca-dims", default="64,128,256,512", help="Comma-separated PCA dims")
    ap.add_argument("--c-values", default="0.1,0.3,0.5", help="Comma-separated SCCA c values")
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1000, help="Number of permutations")
    ap.add_argument("--skip-perm", action="store_true", help="Skip permutation testing")
    args = ap.parse_args()

    print("=" * 60)
    print("Stratified CCA/SCCA Benchmark")
    print(f"Modality: {args.modality_tag}")
    print(f"Start: {datetime.now()}")
    print("=" * 60)

    # Parse grid params
    gene_pca_dims = [int(x) for x in args.gene_pca_dims.split(",")]
    c_values = [float(x) for x in args.c_values.split(",")]
    methods = ["cca", "scca"]

    # Load data
    print("[1/6] Loading data...", flush=True)
    Xg = np.load(args.x_gene_wide, mmap_mode="r")
    Xb = np.load(args.x_brain)
    labels = np.load(args.labels)
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    ids = np.load(args.ids, allow_pickle=True).astype(str)
    extra = np.load(args.cov_extra) if args.cov_extra else None
    if extra is not None:
        extra = _as_2d(extra)

    print(f"  Gene: {Xg.shape}")
    print(f"  Brain: {Xb.shape}")
    print(f"  Labels: {len(labels)} (MDD={np.sum(labels == 1)}, Ctrl={np.sum(labels == 0)})")

    # Materialize gene
    print("[2/6] Materializing gene matrix...", flush=True)
    Xg = np.array(Xg, dtype=np.float32)

    # Split by label
    mdd_idx = np.where(labels == 1)[0]
    ctrl_idx = np.where(labels == 0)[0]
    print(f"  MDD: {len(mdd_idx)}, Ctrl: {len(ctrl_idx)}")

    # Run MDD benchmark
    print("[3/6] Running MDD benchmark...", flush=True)
    mdd_results = run_group_benchmark(
        group_name="mdd",
        Xg=Xg[mdd_idx], Xb=Xb[mdd_idx],
        age=age[mdd_idx], sex=sex[mdd_idx],
        extra=None if extra is None else extra[mdd_idx],
        holdout_frac=args.holdout_frac,
        n_folds=args.n_folds,
        seed=args.seed,
        gene_pca_dims=gene_pca_dims,
        methods=methods,
        c_values=c_values,
        n_components=args.n_components,
    )

    # Run Ctrl benchmark
    print("[4/6] Running Ctrl benchmark...", flush=True)
    ctrl_results = run_group_benchmark(
        group_name="ctrl",
        Xg=Xg[ctrl_idx], Xb=Xb[ctrl_idx],
        age=age[ctrl_idx], sex=sex[ctrl_idx],
        extra=None if extra is None else extra[ctrl_idx],
        holdout_frac=args.holdout_frac,
        n_folds=args.n_folds,
        seed=args.seed,
        gene_pca_dims=gene_pca_dims,
        methods=methods,
        c_values=c_values,
        n_components=args.n_components,
    )

    # Compare weights
    print("[5/6] Comparing weights...", flush=True)
    r_mdd = mdd_results["best_config"]["r_holdout_cc1"]
    r_ctrl = ctrl_results["best_config"]["r_holdout_cc1"]
    r_diff = r_mdd - r_ctrl

    # Cosine similarity of weights (CC1 only)
    W_gene_mdd = mdd_results["W_gene"][:, 0] if mdd_results["W_gene"].shape[1] > 0 else np.zeros(1)
    W_gene_ctrl = ctrl_results["W_gene"][:, 0] if ctrl_results["W_gene"].shape[1] > 0 else np.zeros(1)
    W_brain_mdd = mdd_results["W_brain"][:, 0] if mdd_results["W_brain"].shape[1] > 0 else np.zeros(1)
    W_brain_ctrl = ctrl_results["W_brain"][:, 0] if ctrl_results["W_brain"].shape[1] > 0 else np.zeros(1)

    # Pad to same length for cosine (since gene PCA dims may differ)
    gene_dim_mdd = len(W_gene_mdd)
    gene_dim_ctrl = len(W_gene_ctrl)
    if gene_dim_mdd != gene_dim_ctrl:
        max_dim = max(gene_dim_mdd, gene_dim_ctrl)
        W_gene_mdd_pad = np.zeros(max_dim)
        W_gene_ctrl_pad = np.zeros(max_dim)
        W_gene_mdd_pad[:gene_dim_mdd] = W_gene_mdd
        W_gene_ctrl_pad[:gene_dim_ctrl] = W_gene_ctrl
        gene_cosine = cosine_similarity(W_gene_mdd_pad, W_gene_ctrl_pad)
    else:
        gene_cosine = cosine_similarity(W_gene_mdd, W_gene_ctrl)

    brain_cosine = cosine_similarity(W_brain_mdd, W_brain_ctrl)

    comparison = {
        "r_mdd_holdout_cc1": r_mdd,
        "r_ctrl_holdout_cc1": r_ctrl,
        "r_diff": r_diff,
        "gene_weight_cosine_cc1": gene_cosine,
        "brain_weight_cosine_cc1": brain_cosine,
        "mdd_best_config": mdd_results["best_config"]["config"],
        "ctrl_best_config": ctrl_results["best_config"]["config"],
    }

    # Permutation test
    perm_results = None
    if not args.skip_perm:
        print(f"[6/6] Permutation test (n={args.n_perm})...", flush=True)
        # Use best config from whichever group had higher r
        if r_mdd >= r_ctrl:
            best = mdd_results["best_config"]
        else:
            best = ctrl_results["best_config"]

        perm_results = run_permutation_test(
            Xg=Xg, Xb=Xb, age=age, sex=sex, extra=extra, labels=labels,
            observed_diff=r_diff,
            n_perm=args.n_perm,
            holdout_frac=args.holdout_frac,
            n_folds=args.n_folds,
            seed=args.seed,
            gene_pca_dim=best["gene_pca_dim"],
            method=best["method"],
            c1=best["c1"],
            c2=best["c2"],
            n_components=args.n_components,
        )
        comparison["perm_p_value"] = perm_results["p_value"]
        comparison["perm_null_mean"] = perm_results["null_mean"]
        comparison["perm_null_std"] = perm_results["null_std"]
    else:
        print("[6/6] Skipping permutation test")

    # Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save MDD results
    mdd_out = out_dir / f"stratified_{args.modality_tag}_mdd"
    mdd_out.mkdir(parents=True, exist_ok=True)
    with open(mdd_out / "coupling_benchmark_full.json", "w") as f:
        json.dump(mdd_results["all_results"], f, indent=2)
    with open(mdd_out / "coupling_benchmark_summary.json", "w") as f:
        json.dump(mdd_results["best_config"], f, indent=2)
    np.save(mdd_out / "W_gene.npy", mdd_results["W_gene"])
    np.save(mdd_out / "W_brain.npy", mdd_results["W_brain"])

    # Save Ctrl results
    ctrl_out = out_dir / f"stratified_{args.modality_tag}_ctrl"
    ctrl_out.mkdir(parents=True, exist_ok=True)
    with open(ctrl_out / "coupling_benchmark_full.json", "w") as f:
        json.dump(ctrl_results["all_results"], f, indent=2)
    with open(ctrl_out / "coupling_benchmark_summary.json", "w") as f:
        json.dump(ctrl_results["best_config"], f, indent=2)
    np.save(ctrl_out / "W_gene.npy", ctrl_results["W_gene"])
    np.save(ctrl_out / "W_brain.npy", ctrl_results["W_brain"])

    # Save comparison
    with open(out_dir / f"stratified_comparison_{args.modality_tag}.json", "w") as f:
        json.dump(comparison, f, indent=2)

    if perm_results:
        with open(out_dir / f"stratified_perm_{args.modality_tag}.json", "w") as f:
            json.dump(perm_results, f, indent=2)

    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"  MDD holdout CC1:  {r_mdd:.4f}")
    print(f"  Ctrl holdout CC1: {r_ctrl:.4f}")
    print(f"  Difference:       {r_diff:.4f}")
    print(f"  Gene weight cos:  {gene_cosine:.4f}")
    print(f"  Brain weight cos: {brain_cosine:.4f}")
    if perm_results:
        print(f"  Perm p-value:     {perm_results['p_value']:.4f}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
