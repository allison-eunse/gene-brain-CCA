#!/usr/bin/env python3
"""
Coupling Benchmark: Systematic grid search over CCA/SCCA configurations.

For each brain feature set, systematically test:
- Gene PCA dimensions: [64, 128, 256, 512]
- CCA methods: [CCA, SCCA with c1/c2 grid]
- Primary metric: mean r_val across folds and r_holdout for CC1

Key principle: Optimize for r_val/r_holdout, NOT r_train.

Lab rules: run via Slurm; no login-node heavy compute; no manual CUDA_VISIBLE_DEVICES.
Uses /dev/shm for faster I/O on large matrices (per lab server guidelines).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from itertools import product

import numpy as np

# Force unbuffered output for real-time logging
os.environ['PYTHONUNBUFFERED'] = '1'
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Import local SCCA implementation
sys.path.insert(0, str(Path(__file__).parent.parent / "gene-brain-cca-2" / "scripts"))
from scca_pmd import SCCA_PMD


def _build_cov(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """(N,) age/sex -> (N,3) covariate matrix with intercept."""
    return np.column_stack(
        [np.ones(len(age), dtype=np.float64), age.astype(np.float64), sex.astype(np.float64)]
    )


def _residualize_train_val(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    age_tr: np.ndarray,
    sex_tr: np.ndarray,
    age_va: np.ndarray,
    sex_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit linear residualization on TRAIN only (intercept+age+sex),
    apply to both train and val.
    """
    C_tr = _build_cov(age_tr, sex_tr)
    CtC = C_tr.T @ C_tr
    CtX = C_tr.T @ X_tr.astype(np.float64, copy=False)
    B = np.linalg.solve(CtC, CtX)
    X_tr_r = X_tr.astype(np.float64, copy=False) - (C_tr @ B)

    C_va = _build_cov(age_va, sex_va)
    X_va_r = X_va.astype(np.float64, copy=False) - (C_va @ B)

    return X_tr_r.astype(np.float32, copy=False), X_va_r.astype(np.float32, copy=False)


def _standardize_train_val_f32(
    X_tr: np.ndarray, X_va: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Standardize using train-only statistics (float32 outputs).

    We avoid sklearn StandardScaler here to:
    - reduce repeated object overhead,
    - keep outputs in float32 to lower memory pressure,
    - make it easy to cache per-fold transforms.
    """
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
    """Fit PCA (train-only) to max_components once; downstream configs slice columns as needed."""
    max_components = min(max_components, X_tr.shape[0], X_tr.shape[1])
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr).astype(np.float32, copy=False)
    X_va_pca = pca.transform(X_va).astype(np.float32, copy=False)
    return X_tr_pca, X_va_pca, pca


def _fit_and_score(
    *,
    method: str,
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_te: np.ndarray,
    Xb_te: np.ndarray,
    k: int,
    c1: float,
    c2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit CCA/SCCA on train, return (r_train, r_test) canonical correlations for k components."""
    if k <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    if method == "cca":
        model = CCA(n_components=k, max_iter=500)
        model.fit(Xg_tr, Xb_tr)
        U_tr, V_tr = model.transform(Xg_tr, Xb_tr)
        U_te, V_te = model.transform(Xg_te, Xb_te)
    elif method == "scca":
        model = SCCA_PMD(latent_dimensions=k, c=[c1, c2], max_iter=500, tol=1e-6)
        model.fit([Xg_tr, Xb_tr])
        U_tr, V_tr = model.transform([Xg_tr, Xb_tr])
        U_te, V_te = model.transform([Xg_te, Xb_te])
    else:
        raise ValueError(f"Unknown method: {method}")

    r_train = compute_canonical_correlations(U_tr, V_tr, k)
    r_test = compute_canonical_correlations(U_te, V_te, k)
    return r_train, r_test


def _precompute_gene_cache(
    *,
    Xg: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    gene_pca_dims: list[int],
    seed: int,
) -> dict:
    """
    Cache leakage-safe preprocessing for the gene matrix:
    residualize + standardize + PCA(max_dim) for each fold and for the final train/holdout fit.

    This is mathematically equivalent to recomputing these steps per config, but dramatically faster.
    """
    max_dim = int(max(gene_pca_dims))
    folds = []
    for fold_idx, (tr, va) in enumerate(fold_splits):
        print(f"  [cache gene] fold {fold_idx + 1}/{len(fold_splits)}: residualize/scale/PCA(max={max_dim})", flush=True)
        Xg_tr_r, Xg_va_r = _residualize_train_val(Xg[tr], Xg[va], age[tr], sex[tr], age[va], sex[va])
        Xg_tr_s, Xg_va_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_va_r)
        Xg_tr_pca_max, Xg_va_pca_max, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_va_s, max_dim, seed)
        folds.append(
            {
                "tr": tr,
                "va": va,
                "Xg_tr_pca_max": Xg_tr_pca_max,
                "Xg_va_pca_max": Xg_va_pca_max,
            }
        )

    print(f"  [cache gene] holdout fit: residualize/scale/PCA(max={max_dim})", flush=True)
    Xg_tr_r, Xg_ho_r = _residualize_train_val(
        Xg[train_idx],
        Xg[hold_idx],
        age[train_idx],
        sex[train_idx],
        age[hold_idx],
        sex[hold_idx],
    )
    Xg_tr_s, Xg_ho_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_ho_r)
    Xg_tr_pca_max, Xg_ho_pca_max, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_ho_s, max_dim, seed)

    # Normalize requested dims to what PCA actually produced (handles edge cases automatically).
    max_dim_eff = int(Xg_tr_pca_max.shape[1])
    gene_pca_dims_eff = sorted({int(min(d, max_dim_eff)) for d in gene_pca_dims})
    if gene_pca_dims_eff != sorted({int(d) for d in gene_pca_dims}):
        print(
            f"  [cache gene] adjusted gene_pca_dims to feasible dims: {gene_pca_dims_eff}",
            flush=True,
        )

    return {
        "gene_pca_dims": gene_pca_dims_eff,
        "max_dim": max_dim_eff,
        "folds": folds,
        "holdout": {
            "Xg_tr_pca_max": Xg_tr_pca_max,
            "Xg_ho_pca_max": Xg_ho_pca_max,
        },
    }


def _precompute_brain_cache(
    *,
    Xb: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    train_idx: np.ndarray,
    hold_idx: np.ndarray,
    fold_splits: list[tuple[np.ndarray, np.ndarray]],
    brain_name: str,
) -> dict:
    """Cache leakage-safe preprocessing for a given brain feature matrix (residualize + standardize)."""
    folds = []
    for fold_idx, (tr, va) in enumerate(fold_splits):
        print(f"  [cache brain:{brain_name}] fold {fold_idx + 1}/{len(fold_splits)}: residualize/scale", flush=True)
        Xb_tr_r, Xb_va_r = _residualize_train_val(Xb[tr], Xb[va], age[tr], sex[tr], age[va], sex[va])
        Xb_tr_s, Xb_va_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_va_r)
        folds.append({"Xb_tr_s": Xb_tr_s, "Xb_va_s": Xb_va_s})

    print(f"  [cache brain:{brain_name}] holdout fit: residualize/scale", flush=True)
    Xb_tr_r, Xb_ho_r = _residualize_train_val(
        Xb[train_idx],
        Xb[hold_idx],
        age[train_idx],
        sex[train_idx],
        age[hold_idx],
        sex[hold_idx],
    )
    Xb_tr_s, Xb_ho_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_ho_r)
    return {"folds": folds, "holdout": {"Xb_tr_s": Xb_tr_s, "Xb_ho_s": Xb_ho_s}}


def _run_config_cached(
    *,
    gene_cache: dict,
    brain_cache: dict,
    gene_pca_dim: int,
    method: str,
    c1: float,
    c2: float,
    n_components: int,
) -> dict:
    """Run one (brain set, gene_pca_dim, method, c1, c2) config using cached preprocessing."""
    fold_results = []
    for fold_idx, gf in enumerate(gene_cache["folds"]):
        bf = brain_cache["folds"][fold_idx]
        Xg_tr = gf["Xg_tr_pca_max"][:, :gene_pca_dim]
        Xg_va = gf["Xg_va_pca_max"][:, :gene_pca_dim]
        Xb_tr = bf["Xb_tr_s"]
        Xb_va = bf["Xb_va_s"]
        k = min(n_components, Xg_tr.shape[1], Xb_tr.shape[1])
        r_train, r_val = _fit_and_score(method=method, Xg_tr=Xg_tr, Xb_tr=Xb_tr, Xg_te=Xg_va, Xb_te=Xb_va, k=k, c1=c1, c2=c2)
        fold_results.append(
            {
                "fold": fold_idx,
                "r_train": r_train.tolist(),
                "r_val": r_val.tolist(),
                "r_train_cc1": float(r_train[0]) if len(r_train) > 0 else 0.0,
                "r_val_cc1": float(r_val[0]) if len(r_val) > 0 else 0.0,
            }
        )

    r_val_cc1_list = [f["r_val_cc1"] for f in fold_results]
    r_train_cc1_list = [f["r_train_cc1"] for f in fold_results]
    cv_stats = {
        "mean_r_val_cc1": float(np.mean(r_val_cc1_list)),
        "std_r_val_cc1": float(np.std(r_val_cc1_list)),
        "mean_r_train_cc1": float(np.mean(r_train_cc1_list)),
        "std_r_train_cc1": float(np.std(r_train_cc1_list)),
        "overfitting_gap": float(np.mean(r_train_cc1_list) - np.mean(r_val_cc1_list)),
    }

    # Final fit on full train, evaluate on holdout
    Xg_tr = gene_cache["holdout"]["Xg_tr_pca_max"][:, :gene_pca_dim]
    Xg_ho = gene_cache["holdout"]["Xg_ho_pca_max"][:, :gene_pca_dim]
    Xb_tr = brain_cache["holdout"]["Xb_tr_s"]
    Xb_ho = brain_cache["holdout"]["Xb_ho_s"]
    k = min(n_components, Xg_tr.shape[1], Xb_tr.shape[1])
    r_train_final, r_holdout = _fit_and_score(method=method, Xg_tr=Xg_tr, Xb_tr=Xb_tr, Xg_te=Xg_ho, Xb_te=Xb_ho, k=k, c1=c1, c2=c2)
    holdout_stats = {
        "r_train": r_train_final.tolist(),
        "r_holdout": r_holdout.tolist(),
        "r_train_cc1": float(r_train_final[0]) if len(r_train_final) > 0 else 0.0,
        "r_holdout_cc1": float(r_holdout[0]) if len(r_holdout) > 0 else 0.0,
        "generalization_gap": float(r_train_final[0] - r_holdout[0]) if len(r_train_final) > 0 else 0.0,
    }
    return {"cv": cv_stats, "holdout": holdout_stats, "folds": fold_results}


def compute_canonical_correlations(U: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    """Compute canonical correlations for k components."""
    k = min(k, U.shape[1], V.shape[1])
    r = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)])
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-gene-wide", required=True, help="Wide gene embeddings (N x 768*num_genes)")
    ap.add_argument("--brain-features", nargs="+", required=True, help="List of brain feature .npy files")
    ap.add_argument("--brain-names", nargs="+", required=True, help="Names for brain feature sets")
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--gene-pca-dims", nargs="+", type=int, default=[64, 128, 256, 512])
    ap.add_argument("--methods", nargs="+", default=["cca", "scca"])
    ap.add_argument("--c1-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    ap.add_argument("--c2-grid", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    ap.add_argument("--n-components", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    
    # Validate inputs
    if len(args.brain_features) != len(args.brain_names):
        raise SystemExit("[ERROR] --brain-features and --brain-names must have same length")
    
    # Load data - use /dev/shm for faster I/O on large gene matrix (per lab server guidelines)
    print("[1/5] Loading data...", flush=True)
    
    # Copy gene matrix to /dev/shm for faster I/O
    gene_file = Path(args.x_gene_wide)
    shm_gene_path = Path("/dev/shm") / f"gene_wide_{os.getpid()}.npy"
    use_shm = False
    
    # Check if /dev/shm has enough space (need ~1.5GB for safety)
    shm_stat = os.statvfs("/dev/shm")
    shm_free_gb = (shm_stat.f_bavail * shm_stat.f_frsize) / (1024**3)
    gene_size_gb = gene_file.stat().st_size / (1024**3)
    
    if shm_free_gb > gene_size_gb * 1.2:  # 20% safety margin
        print(f"  Copying gene matrix to /dev/shm for faster I/O ({gene_size_gb:.1f}GB, {shm_free_gb:.1f}GB free)...", flush=True)
        shutil.copy(gene_file, shm_gene_path)
        Xg_raw = np.load(shm_gene_path, mmap_mode="r")
        use_shm = True
        print(f"  Gene matrix copied to RAM.", flush=True)
    else:
        print(f"  /dev/shm too small ({shm_free_gb:.1f}GB free, need {gene_size_gb:.1f}GB), using network storage.", flush=True)
        Xg_raw = np.load(args.x_gene_wide, mmap_mode="r")
    
    labels = np.load(args.labels)
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    ids = np.load(args.ids, allow_pickle=True).astype(str)
    
    print(f"  Gene embeddings: {Xg_raw.shape}", flush=True)
    print(f"  Subjects: {len(labels)}", flush=True)
    
    # Create stratified holdout split
    print("[2/5] Creating train/holdout split...", flush=True)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(labels)), labels))
    print(f"  Train: {len(train_idx)}, Holdout: {len(hold_idx)}", flush=True)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark for each brain feature set
    all_results = []

    # Materialize gene matrix once (from /dev/shm if available - much faster)
    print("[3/5] Precomputing leakage-safe caches (gene + folds)...", flush=True)
    print("  Materializing gene matrix into memory...", flush=True)
    Xg_array = np.array(Xg_raw)
    print(f"  Gene matrix loaded: {Xg_array.shape}", flush=True)

    # Precompute fold splits once
    y_train = labels[train_idx]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for tr_rel, va_rel in skf.split(np.zeros(len(train_idx)), y_train):
        fold_splits.append((train_idx[tr_rel], train_idx[va_rel]))

    # Cache gene preprocessing (heavy part) once per fold + once for train/holdout
    gene_cache = _precompute_gene_cache(
        Xg=Xg_array,
        age=age,
        sex=sex,
        labels=labels,
        train_idx=train_idx,
        hold_idx=hold_idx,
        fold_splits=fold_splits,
        gene_pca_dims=list(args.gene_pca_dims),
        seed=args.seed,
    )
    gene_pca_dims = gene_cache["gene_pca_dims"]

    print("[4/5] Running coupling benchmark (cached preprocessing)...", flush=True)
    for brain_path, brain_name in zip(args.brain_features, args.brain_names):
        if not Path(brain_path).exists():
            print(f"  [SKIP] {brain_name}: file not found ({brain_path})", flush=True)
            continue

        Xb_raw = np.load(brain_path, mmap_mode="r")
        print(f"  Brain feature: {brain_name} ({Xb_raw.shape[1]} features)", flush=True)

        if Xb_raw.shape[0] != Xg_array.shape[0]:
            print(f"  [SKIP] {brain_name}: row mismatch ({Xb_raw.shape[0]} vs {Xg_array.shape[0]})", flush=True)
            continue

        # Materialize brain features once per brain set
        Xb_array = np.array(Xb_raw)

        # Cache brain preprocessing (cheap) once per fold + holdout
        brain_cache = _precompute_brain_cache(
            Xb=Xb_array,
            age=age,
            sex=sex,
            train_idx=train_idx,
            hold_idx=hold_idx,
            fold_splits=fold_splits,
            brain_name=brain_name,
        )

        for gene_pca_dim in gene_pca_dims:
            for method in args.methods:
                if method == "cca":
                    configs = [(0.3, 0.3)]  # dummy sparsity params
                else:
                    configs = list(product(args.c1_grid, args.c2_grid))

                for c1, c2 in configs:
                    config_str = f"{brain_name}_{method}_pca{gene_pca_dim}"
                    if method == "scca":
                        config_str += f"_c{c1}_{c2}"
                    print(f"    Running: {config_str}", flush=True)

                    try:
                        result = _run_config_cached(
                            gene_cache=gene_cache,
                            brain_cache=brain_cache,
                            gene_pca_dim=int(gene_pca_dim),
                            method=method,
                            c1=float(c1),
                            c2=float(c2),
                            n_components=int(args.n_components),
                        )
                        result_entry = {
                            "brain_feature": brain_name,
                            "brain_dim": int(Xb_array.shape[1]),
                            "gene_pca_dim": int(gene_pca_dim),
                            "method": method,
                            "c1": float(c1) if method == "scca" else None,
                            "c2": float(c2) if method == "scca" else None,
                            **result,
                        }
                        all_results.append(result_entry)

                        print(
                            f"      CV mean r_val CC1: {result['cv']['mean_r_val_cc1']:.4f} ± {result['cv']['std_r_val_cc1']:.4f}",
                            flush=True,
                        )
                        print(f"      Holdout r CC1:     {result['holdout']['r_holdout_cc1']:.4f}", flush=True)

                    except Exception as e:
                        print(f"      [ERROR] {e}", flush=True)
                        all_results.append(
                            {
                                "brain_feature": brain_name,
                                "gene_pca_dim": int(gene_pca_dim),
                                "method": method,
                                "c1": float(c1) if method == "scca" else None,
                                "c2": float(c2) if method == "scca" else None,
                                "error": str(e),
                            }
                        )
    
    # Save results
    print("[5/5] Saving results...", flush=True)
    
    # Full results with fold details
    full_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_subjects": int(len(labels)),
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "holdout_frac": args.holdout_frac,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "gene_pca_dims": gene_pca_dims,
            "methods": args.methods,
            "c1_grid": args.c1_grid,
            "c2_grid": args.c2_grid,
            "used_shm": use_shm,
        },
        "results": all_results,
    }
    
    out_path = out_dir / "coupling_benchmark_full.json"
    out_path.write_text(json.dumps(full_results, indent=2))
    print(f"  Full results: {out_path}", flush=True)
    
    # Summary table (best configs per brain feature)
    summary = []
    for brain_name in args.brain_names:
        brain_results = [r for r in all_results if r.get("brain_feature") == brain_name and "cv" in r]
        if not brain_results:
            continue
        
        # Sort by mean_r_val_cc1 (our primary metric)
        brain_results.sort(key=lambda x: x["cv"]["mean_r_val_cc1"], reverse=True)
        best = brain_results[0]
        
        summary.append({
            "brain_feature": brain_name,
            "brain_dim": best.get("brain_dim"),
            "best_config": f"{best['method']}_pca{best['gene_pca_dim']}" + (f"_c{best['c1']}_{best['c2']}" if best["method"] == "scca" else ""),
            "mean_r_val_cc1": best["cv"]["mean_r_val_cc1"],
            "std_r_val_cc1": best["cv"]["std_r_val_cc1"],
            "r_holdout_cc1": best["holdout"]["r_holdout_cc1"],
            "overfitting_gap": best["cv"]["overfitting_gap"],
            "generalization_gap": best["holdout"]["generalization_gap"],
        })
    
    summary_path = out_dir / "coupling_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary: {summary_path}", flush=True)
    
    # Print summary table
    print("\n" + "=" * 80, flush=True)
    print("COUPLING BENCHMARK SUMMARY (sorted by mean_r_val_cc1)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Brain Feature':<25} {'Config':<25} {'r_val CC1':<15} {'r_holdout CC1':<15}", flush=True)
    print("-" * 80, flush=True)
    for s in sorted(summary, key=lambda x: x["mean_r_val_cc1"], reverse=True):
        print(f"{s['brain_feature']:<25} {s['best_config']:<25} {s['mean_r_val_cc1']:.4f} ± {s['std_r_val_cc1']:.4f}  {s['r_holdout_cc1']:.4f}", flush=True)
    print("=" * 80, flush=True)
    
    # Cleanup /dev/shm
    if use_shm and shm_gene_path.exists():
        print(f"  Cleaning up /dev/shm...", flush=True)
        shm_gene_path.unlink()
    
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
