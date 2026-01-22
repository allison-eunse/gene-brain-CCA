#!/usr/bin/env python3
"""
Phase 3 CCA Benchmark with Disk Caching and Checkpointing.

This script runs stratified CCA/SCCA analysis with:
- Disk-based preprocessing cache (for fast resume)
- Incremental checkpointing (survives crashes)
- Self-reporting results (per lab guidelines)
- Weight cosine similarity analysis

Lab Server Compliance:
- Run via Slurm (sbatch/srun), not on login node
- No manual CUDA_VISIBLE_DEVICES
- Uses /scratch for fast I/O when available
- Backs up results to /storage

Usage:
    python scripts/run_phase3_benchmark.py \
        --fm-model caduceus \
        --modality dmri \
        --cache-dir /scratch/connectome/$USER/phase3_cache \
        --checkpoint-path /scratch/.../checkpoint.json \
        --output-dir gene-brain-cca-2/derived/phase3_results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ShuffleSplit

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "gene-brain-cca-2" / "scripts"))
from scca_pmd import SCCA_PMD

from preprocessing_cache import (
    FoldCache,
    CheckpointManager,
    backup_to_storage,
    get_cache_dir,
)


# -----------------------------------------------------------------------------
# Preprocessing functions (same as run_stratified_coupling_benchmark.py)
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
    X_tr: np.ndarray, X_va: np.ndarray, max_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, PCA]:
    max_dim = min(max_dim, X_tr.shape[0] - 1, X_tr.shape[1])
    pca = PCA(n_components=max_dim, random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr)
    X_va_pca = pca.transform(X_va)
    return (
        X_tr_pca.astype(np.float32, copy=False),
        X_va_pca.astype(np.float32, copy=False),
        pca,
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit CCA/SCCA and return correlations + weights.
    
    Returns:
        r_train: Training correlations per component
        r_test: Test correlations per component
        W_gene: Gene weights (n_gene_features, k)
        W_brain: Brain weights (n_brain_features, k)
    """
    if method == "cca":
        model = CCA(n_components=k, max_iter=1000)
        model.fit(Xg_tr, Xb_tr)
        W_gene = model.x_weights_
        W_brain = model.y_weights_
    else:  # scca
        model = SCCA_PMD(n_components=k, c1=c1, c2=c2, max_iter=500, tol=1e-6)
        model.fit(Xg_tr, Xb_tr)
        W_gene = model.ws_[0]
        W_brain = model.ws_[1]

    # Compute correlations
    U_tr = Xg_tr @ W_gene
    V_tr = Xb_tr @ W_brain
    U_te = Xg_te @ W_gene
    V_te = Xb_te @ W_brain

    r_train = np.array([np.corrcoef(U_tr[:, i], V_tr[:, i])[0, 1] for i in range(k)])
    r_test = np.array([np.corrcoef(U_te[:, i], V_te[:, i])[0, 1] for i in range(k)])

    return r_train, r_test, W_gene, W_brain


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))


# -----------------------------------------------------------------------------
# Group benchmark with caching
# -----------------------------------------------------------------------------

def run_group_benchmark_cached(
    *,
    group_name: str,
    Xg: np.ndarray,
    Xb: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    extra: np.ndarray | None,
    fold_cache: FoldCache,
    checkpoint: CheckpointManager,
    holdout_frac: float,
    n_folds: int,
    seed: int,
    gene_pca_dims: list[int],
    methods: list[str],
    c_values: list[float],
    n_components: int,
) -> dict:
    """
    Run CCA/SCCA benchmark on a single group with disk caching.
    """
    n = Xg.shape[0]
    print(f"[{group_name}] N={n}, gene_dim={Xg.shape[1]}, brain_dim={Xb.shape[1]}")

    # Train/holdout split
    ss = ShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    train_idx, hold_idx = next(ss.split(np.zeros(n)))
    print(f"[{group_name}] Train: {len(train_idx)}, Holdout: {len(hold_idx)}")

    # CV folds on training set
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_splits = list(kf.split(np.zeros(len(train_idx))))

    # Compute max PCA dim
    max_gene_dim = max(gene_pca_dims)
    max_gene_dim = min(max_gene_dim, len(train_idx) - 1, Xg.shape[1])

    # Load or compute fold caches
    print(f"[{group_name}] Checking fold caches...")
    fold_data = []
    
    for fold_idx, (tr_rel, va_rel) in enumerate(fold_splits):
        if fold_cache.has_fold(fold_idx):
            # Load from cache
            cached = fold_cache.load_fold(fold_idx)
            fold_data.append(cached)
        else:
            # Compute and cache
            tr = train_idx[tr_rel]
            va = train_idx[va_rel]
            
            # Gene preprocessing
            Xg_tr_r, Xg_va_r = _residualize_train_val(
                Xg[tr], Xg[va],
                age[tr], sex[tr], age[va], sex[va],
                extra_tr=None if extra is None else extra[tr],
                extra_va=None if extra is None else extra[va],
            )
            Xg_tr_s, Xg_va_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_va_r)
            Xg_tr_pca, Xg_va_pca, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_va_s, max_gene_dim, seed)
            
            # Brain preprocessing
            Xb_tr_r, Xb_va_r = _residualize_train_val(
                Xb[tr], Xb[va],
                age[tr], sex[tr], age[va], sex[va],
                extra_tr=None if extra is None else extra[tr],
                extra_va=None if extra is None else extra[va],
            )
            Xb_tr_s, Xb_va_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_va_r)
            
            # Save to cache
            fold_cache.save_fold(
                fold_idx,
                Xg_tr_pca=Xg_tr_pca,
                Xg_va_pca=Xg_va_pca,
                Xb_tr_s=Xb_tr_s,
                Xb_va_s=Xb_va_s,
                tr_indices=tr,
                va_indices=va,
            )
            
            fold_data.append({
                "Xg_tr_pca": Xg_tr_pca,
                "Xg_va_pca": Xg_va_pca,
                "Xb_tr_s": Xb_tr_s,
                "Xb_va_s": Xb_va_s,
                "tr_indices": tr,
                "va_indices": va,
            })

    # Load or compute holdout cache
    if fold_cache.has_holdout():
        holdout_data = fold_cache.load_holdout()
        Xg_tr_pca = holdout_data["Xg_tr_pca"]
        Xg_ho_pca = holdout_data["Xg_ho_pca"]
        Xb_tr_s = holdout_data["Xb_tr_s"]
        Xb_ho_s = holdout_data["Xb_ho_s"]
    else:
        # Compute holdout preprocessing
        Xg_tr_r, Xg_ho_r = _residualize_train_val(
            Xg[train_idx], Xg[hold_idx],
            age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
            extra_tr=None if extra is None else extra[train_idx],
            extra_va=None if extra is None else extra[hold_idx],
        )
        Xg_tr_s, Xg_ho_s, _ = _standardize_train_val_f32(Xg_tr_r, Xg_ho_r)
        Xg_tr_pca, Xg_ho_pca, _ = _pca_train_val_maxdim(Xg_tr_s, Xg_ho_s, max_gene_dim, seed)
        
        Xb_tr_r, Xb_ho_r = _residualize_train_val(
            Xb[train_idx], Xb[hold_idx],
            age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx],
            extra_tr=None if extra is None else extra[train_idx],
            extra_va=None if extra is None else extra[hold_idx],
        )
        Xb_tr_s, Xb_ho_s, _ = _standardize_train_val_f32(Xb_tr_r, Xb_ho_r)
        
        fold_cache.save_holdout(
            Xg_tr_pca=Xg_tr_pca,
            Xg_ho_pca=Xg_ho_pca,
            Xb_tr_s=Xb_tr_s,
            Xb_ho_s=Xb_ho_s,
            train_indices=train_idx,
            holdout_indices=hold_idx,
        )

    # Effective PCA dims
    effective_gene_dims = sorted({min(d, Xg_tr_pca.shape[1]) for d in gene_pca_dims})

    # Grid search with checkpointing
    print(f"[{group_name}] Running grid search...")
    all_results = []
    best_result = None
    best_val_score = float("-inf")

    for method in methods:
        c_pairs = [(0.0, 0.0)] if method == "cca" else list(product(c_values, c_values))
        
        for gene_dim in effective_gene_dims:
            for c1, c2 in c_pairs:
                config_key = f"{group_name}_{method}_pca{gene_dim}_c{c1}_{c2}"
                
                # Check if already completed
                if checkpoint.is_completed(config_key):
                    print(f"  [skip] {config_key} (cached)")
                    continue
                
                # Cross-validation
                fold_results = []
                for fold_idx, fd in enumerate(fold_data):
                    Xg_tr = fd["Xg_tr_pca"][:, :gene_dim]
                    Xg_va = fd["Xg_va_pca"][:, :gene_dim]
                    Xb_tr = fd["Xb_tr_s"]
                    Xb_va = fd["Xb_va_s"]
                    
                    k = min(n_components, gene_dim, Xb_tr.shape[1])
                    r_train, r_val, _, _ = _fit_and_score(
                        method=method, Xg_tr=Xg_tr, Xb_tr=Xb_tr,
                        Xg_te=Xg_va, Xb_te=Xb_va, k=k, c1=c1, c2=c2
                    )
                    fold_results.append({
                        "r_train_cc1": float(r_train[0]) if len(r_train) > 0 else 0.0,
                        "r_val_cc1": float(r_val[0]) if len(r_val) > 0 else 0.0,
                    })
                
                mean_val = np.mean([f["r_val_cc1"] for f in fold_results])
                mean_train = np.mean([f["r_train_cc1"] for f in fold_results])
                
                result = {
                    "config": config_key,
                    "method": method,
                    "gene_pca_dim": gene_dim,
                    "c1": c1,
                    "c2": c2,
                    "mean_cv_val_cc1": float(mean_val),
                    "mean_cv_train_cc1": float(mean_train),
                    "overfitting_gap": float(mean_train - mean_val),
                }
                
                all_results.append(result)
                checkpoint.save_result(config_key, result, score=mean_val)
                
                if mean_val > best_val_score:
                    best_val_score = mean_val
                    best_result = result
                
                print(f"  [{config_key}] val={mean_val:.4f}")

    # Fit best config on full train, evaluate on holdout
    if best_result is not None:
        print(f"[{group_name}] Fitting best config: {best_result['config']}")
        
        gene_dim = best_result["gene_pca_dim"]
        method = best_result["method"]
        c1 = best_result["c1"]
        c2 = best_result["c2"]
        
        Xg_tr = Xg_tr_pca[:, :gene_dim]
        Xg_ho = Xg_ho_pca[:, :gene_dim]
        k = min(n_components, gene_dim, Xb_tr_s.shape[1])
        
        _, r_holdout, W_gene, W_brain = _fit_and_score(
            method=method, Xg_tr=Xg_tr, Xb_tr=Xb_tr_s,
            Xg_te=Xg_ho, Xb_te=Xb_ho_s, k=k, c1=c1, c2=c2
        )
        
        holdout_cc1 = float(r_holdout[0]) if len(r_holdout) > 0 else 0.0
        print(f"[{group_name}] Holdout CC1: {holdout_cc1:.4f}")
        
        return {
            "group": group_name,
            "n_samples": n,
            "best_config": best_result["config"],
            "best_cv_val_cc1": best_val_score,
            "holdout_cc1": holdout_cc1,
            "W_gene": W_gene,
            "W_brain": W_brain,
            "all_results": all_results,
        }
    
    return {
        "group": group_name,
        "n_samples": n,
        "best_config": None,
        "best_cv_val_cc1": 0.0,
        "holdout_cc1": 0.0,
        "W_gene": None,
        "W_brain": None,
        "all_results": all_results,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 3 CCA Benchmark with Caching")
    
    # Data paths
    ap.add_argument("--data-dir", default="derived_stratified_fm",
                    help="Base directory for aligned data")
    ap.add_argument("--fm-model", required=True,
                    help="Gene foundation model (dnabert2, evo2, hyenadna, caduceus)")
    ap.add_argument("--modality", required=True,
                    help="Brain modality (schaefer7, schaefer17, smri, dmri)")
    
    # Cache and checkpoint
    ap.add_argument("--cache-dir", default=None,
                    help="Directory for preprocessing cache (default: auto)")
    ap.add_argument("--checkpoint-path", default=None,
                    help="Path for checkpoint JSON (default: auto)")
    ap.add_argument("--output-dir", default="gene-brain-cca-2/derived/phase3_results",
                    help="Output directory for results")
    
    # Hyperparameters
    ap.add_argument("--gene-pca-dims", default="64,128,256,512",
                    help="Comma-separated PCA dimensions")
    ap.add_argument("--c-values", default="0.1,0.3,0.5",
                    help="Comma-separated SCCA sparsity values")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    
    args = ap.parse_args()
    
    # Parse hyperparameters
    gene_pca_dims = [int(x) for x in args.gene_pca_dims.split(",")]
    c_values = [float(x) for x in args.c_values.split(",")]
    methods = ["cca", "scca"]
    
    print("=" * 60)
    print("Phase 3 CCA Benchmark")
    print("=" * 60)
    print(f"FM Model:    {args.fm_model}")
    print(f"Modality:    {args.modality}")
    print(f"PCA dims:    {gene_pca_dims}")
    print(f"C values:    {c_values}")
    print(f"Job ID:      {os.environ.get('SLURM_JOB_ID', 'local')}")
    print(f"Start:       {datetime.now()}")
    print("=" * 60)
    
    # Setup directories
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = get_cache_dir("phase3_cache")
    
    if args.checkpoint_path:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = cache_dir / f"checkpoint_{args.fm_model}_{args.modality}.json"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Cache dir:   {cache_dir}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Output dir:  {output_dir}")
    
    # Load data
    data_dir = Path(args.data_dir) / args.fm_model / args.modality
    
    print(f"\n[1/5] Loading data from {data_dir}...")
    X_gene = np.load(data_dir / "X_gene_wide.npy")
    X_brain = np.load(data_dir / "X_brain.npy")
    labels = np.load(data_dir / "labels.npy", allow_pickle=True)
    cov_age = np.load(data_dir / "cov_age.npy")
    cov_sex = np.load(data_dir / "cov_sex.npy")
    
    cov_extra = None
    if (data_dir / "cov_extra.npy").exists():
        cov_extra = np.load(data_dir / "cov_extra.npy")
    
    print(f"  X_gene:  {X_gene.shape}")
    print(f"  X_brain: {X_brain.shape}")
    print(f"  labels:  {labels.shape}")
    
    # Split by label
    print("\n[2/5] Splitting by MDD/Control...")
    mdd_mask = labels == 1
    ctrl_mask = labels == 0
    
    print(f"  MDD:     {mdd_mask.sum()}")
    print(f"  Control: {ctrl_mask.sum()}")
    
    # Run benchmarks for each group
    print("\n[3/5] Running MDD benchmark...")
    mdd_cache = FoldCache(cache_dir, args.fm_model, f"{args.modality}_mdd", args.seed, args.n_folds)
    mdd_checkpoint = CheckpointManager(checkpoint_path.parent / f"checkpoint_mdd_{args.fm_model}_{args.modality}.json")
    
    mdd_result = run_group_benchmark_cached(
        group_name="MDD",
        Xg=X_gene[mdd_mask],
        Xb=X_brain[mdd_mask],
        age=cov_age[mdd_mask],
        sex=cov_sex[mdd_mask],
        extra=cov_extra[mdd_mask] if cov_extra is not None else None,
        fold_cache=mdd_cache,
        checkpoint=mdd_checkpoint,
        holdout_frac=args.holdout_frac,
        n_folds=args.n_folds,
        seed=args.seed,
        gene_pca_dims=gene_pca_dims,
        methods=methods,
        c_values=c_values,
        n_components=args.n_components,
    )
    
    print("\n[4/5] Running Control benchmark...")
    ctrl_cache = FoldCache(cache_dir, args.fm_model, f"{args.modality}_ctrl", args.seed, args.n_folds)
    ctrl_checkpoint = CheckpointManager(checkpoint_path.parent / f"checkpoint_ctrl_{args.fm_model}_{args.modality}.json")
    
    ctrl_result = run_group_benchmark_cached(
        group_name="Control",
        Xg=X_gene[ctrl_mask],
        Xb=X_brain[ctrl_mask],
        age=cov_age[ctrl_mask],
        sex=cov_sex[ctrl_mask],
        extra=cov_extra[ctrl_mask] if cov_extra is not None else None,
        fold_cache=ctrl_cache,
        checkpoint=ctrl_checkpoint,
        holdout_frac=args.holdout_frac,
        n_folds=args.n_folds,
        seed=args.seed,
        gene_pca_dims=gene_pca_dims,
        methods=methods,
        c_values=c_values,
        n_components=args.n_components,
    )
    
    # Compare weights
    print("\n[5/5] Computing weight comparison...")
    gene_cosine = 0.0
    brain_cosine = 0.0
    
    if mdd_result["W_gene"] is not None and ctrl_result["W_gene"] is not None:
        # Compare first component weights
        gene_cosine = cosine_similarity(
            mdd_result["W_gene"][:, 0],
            ctrl_result["W_gene"][:, 0]
        )
        brain_cosine = cosine_similarity(
            mdd_result["W_brain"][:, 0],
            ctrl_result["W_brain"][:, 0]
        )
    
    print(f"  Gene weight cosine:  {gene_cosine:.4f}")
    print(f"  Brain weight cosine: {brain_cosine:.4f}")
    
    # Self-reporting results (per lab guidelines)
    final_results = {
        "job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "fm_model": args.fm_model,
        "modality": args.modality,
        "seed": args.seed,
        "mdd_n": int(mdd_mask.sum()),
        "ctrl_n": int(ctrl_mask.sum()),
        "mdd_best_config": mdd_result["best_config"],
        "ctrl_best_config": ctrl_result["best_config"],
        "mdd_holdout_cc1": mdd_result["holdout_cc1"],
        "ctrl_holdout_cc1": ctrl_result["holdout_cc1"],
        "r_diff": mdd_result["holdout_cc1"] - ctrl_result["holdout_cc1"],
        "gene_cosine_mdd_ctrl": gene_cosine,
        "brain_cosine_mdd_ctrl": brain_cosine,
        "status": "FINISHED",
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save results
    result_file = output_dir / f"{args.fm_model}_{args.modality}_results.json"
    result_file.write_text(json.dumps(final_results, indent=2))
    print(f"\n[save] {result_file}")
    
    # Save weights
    if mdd_result["W_gene"] is not None:
        np.save(output_dir / f"{args.fm_model}_{args.modality}_W_gene_mdd.npy", mdd_result["W_gene"])
        np.save(output_dir / f"{args.fm_model}_{args.modality}_W_brain_mdd.npy", mdd_result["W_brain"])
    
    if ctrl_result["W_gene"] is not None:
        np.save(output_dir / f"{args.fm_model}_{args.modality}_W_gene_ctrl.npy", ctrl_result["W_gene"])
        np.save(output_dir / f"{args.fm_model}_{args.modality}_W_brain_ctrl.npy", ctrl_result["W_brain"])
    
    # Backup to storage if on scratch
    if "/scratch/" in str(result_file):
        backup_to_storage(result_file, "gene-brain-cca-2/derived/phase3_results")
    
    print("\n" + "=" * 60)
    print("Phase 3 Benchmark Complete")
    print("=" * 60)
    print(f"MDD Holdout CC1:     {mdd_result['holdout_cc1']:.4f}")
    print(f"Control Holdout CC1: {ctrl_result['holdout_cc1']:.4f}")
    print(f"Difference (MDD-Ctrl): {final_results['r_diff']:.4f}")
    print(f"Done: {datetime.now()}")


if __name__ == "__main__":
    main()
