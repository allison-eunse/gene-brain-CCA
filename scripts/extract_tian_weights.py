#!/usr/bin/env python3
"""
Extract and interpret CCA/SCCA weights for Tian subcortical ROIs.

This script:
1. Loads best coupling config from Phase 3 results
2. Refits CCA/SCCA on full training set (N=746)
3. Extracts brain-side weights (50 Tian ROIs)
4. Maps weights to ROI names (hippocampus, amygdala, etc.)
5. Saves weight rankings and summary

Purpose: Mechanistic validation that hippocampus/amygdala ROIs have high weights,
aligning with GDNF expression profiles (Jia et al., 2024).

Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

# Import local SCCA implementation
sys.path.insert(0, str(Path(__file__).parent.parent / "gene-brain-cca-2" / "scripts"))
from scca_pmd import SCCA_PMD


# Tian Scale III ROI labels (50 subcortical ROIs)
# Based on: https://github.com/yetianmed/subcortex
TIAN_S3_LABELS = [
    # Hippocampus (8 ROIs)
    "HIP-head-l", "HIP-head-r", "HIP-body-l", "HIP-body-r",
    "HIP-tail-l", "HIP-tail-r", "HIP-subiculum-l", "HIP-subiculum-r",
    # Amygdala (4 ROIs)
    "AMY-lateral-l", "AMY-lateral-r", "AMY-medial-l", "AMY-medial-r",
    # Thalamus (12 ROIs)
    "THA-VA-l", "THA-VA-r", "THA-VL-l", "THA-VL-r",
    "THA-VP-l", "THA-VP-r", "THA-IL-l", "THA-IL-r",
    "THA-MD-l", "THA-MD-r", "THA-PU-l", "THA-PU-r",
    # Striatum: NAc (4 ROIs)
    "NAc-core-l", "NAc-core-r", "NAc-shell-l", "NAc-shell-r",
    # Striatum: Caudate (6 ROIs)
    "CAU-head-l", "CAU-head-r", "CAU-body-l", "CAU-body-r",
    "CAU-tail-l", "CAU-tail-r",
    # Striatum: Putamen (6 ROIs)
    "PUT-anterior-l", "PUT-anterior-r", "PUT-posterior-l", "PUT-posterior-r",
    "PUT-ventral-l", "PUT-ventral-r",
    # Globus Pallidus (4 ROIs)
    "GP-internal-l", "GP-internal-r", "GP-external-l", "GP-external-r",
    # Hypothalamus (2 ROIs)
    "HTH-l", "HTH-r",
    # Brainstem nuclei (4 ROIs)
    "VTA-l", "VTA-r", "SN-l", "SN-r",
]

# Tian anatomical groups for summary features (7 systems)
TIAN_ANATOMICAL_GROUPS = {
    "Hippocampus": list(range(0, 8)),
    "Amygdala": list(range(8, 12)),
    "Thalamus": list(range(12, 24)),
    "NAc": list(range(24, 28)),
    "Caudate": list(range(28, 34)),
    "Putamen": list(range(34, 40)),
    "Other": list(range(40, 50)),  # GP + HTH + VTA/SN
}

# ROI categories for summary
ROI_CATEGORIES = {
    "Hippocampus": [0, 1, 2, 3, 4, 5, 6, 7],
    "Amygdala": [8, 9, 10, 11],
    "Thalamus": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "NAc": [24, 25, 26, 27],
    "Caudate": [28, 29, 30, 31, 32, 33],
    "Putamen": [34, 35, 36, 37, 38, 39],
    "Globus Pallidus": [40, 41, 42, 43],
    "Hypothalamus": [44, 45],
    "Brainstem (VTA/SN)": [46, 47, 48, 49],
}


def _build_cov(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """(N,) age/sex -> (N,3) covariate matrix with intercept."""
    return np.column_stack([
        np.ones(len(age), dtype=np.float64),
        age.astype(np.float64),
        sex.astype(np.float64)
    ])


def _residualize(X: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Residualize X for age and sex (fit on same data)."""
    C = _build_cov(age, sex)
    B = np.linalg.lstsq(C, X.astype(np.float64), rcond=None)[0]
    return (X.astype(np.float64) - C @ B).astype(np.float32)


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score standardization."""
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return ((X - mu) / sd).astype(np.float32)


def fc_edge_to_node_weights(fc_weights: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Convert FC edge weights to node weights.
    
    For each node, aggregate the absolute weights of all edges involving that node.
    """
    iu = np.triu_indices(n_nodes, k=1)
    
    node_weights = np.zeros(n_nodes)
    for edge_idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        w = abs(fc_weights[edge_idx])
        node_weights[i] += w
        node_weights[j] += w
    
    return node_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coupling-results", required=True,
                    help="coupling_benchmark_full.json from Phase 3")
    ap.add_argument("--x-gene-wide", required=True)
    ap.add_argument("--x-fmri-fc", required=True)
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    args = ap.parse_args()

    print("[1/5] Loading coupling results...", flush=True)
    results = json.loads(Path(args.coupling_results).read_text())
    
    # Find best config by r_holdout_cc1
    best_result = None
    best_r_holdout = -999
    for r in results.get("results", []):
        if "holdout" not in r:
            continue
        r_ho = r["holdout"].get("r_holdout_cc1", -999)
        if r_ho > best_r_holdout:
            best_r_holdout = r_ho
            best_result = r
    
    if best_result is None:
        raise SystemExit("[ERROR] No valid coupling results found!")
    
    print(f"  Best config: {best_result.get('method')} pca{best_result.get('gene_pca_dim')}", flush=True)
    print(f"  r_holdout_cc1: {best_r_holdout:.4f}", flush=True)

    print("[2/5] Loading data...", flush=True)
    Xg = np.load(args.x_gene_wide)
    Xb = np.load(args.x_fmri_fc)
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    labels = np.load(args.labels)
    
    print(f"  Gene: {Xg.shape}, Brain FC: {Xb.shape}", flush=True)
    
    # Train/holdout split (same as coupling benchmark)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, _ = next(sss.split(np.zeros(len(labels)), labels))
    
    print(f"  Training on {len(train_idx)} subjects", flush=True)

    print("[3/5] Preprocessing (residualize + standardize + PCA)...", flush=True)
    # Use train only
    Xg_tr = Xg[train_idx]
    Xb_tr = Xb[train_idx]
    age_tr = age[train_idx]
    sex_tr = sex[train_idx]
    
    # Residualize
    Xg_tr_r = _residualize(Xg_tr, age_tr, sex_tr)
    Xb_tr_r = _residualize(Xb_tr, age_tr, sex_tr)
    
    # Standardize
    Xg_tr_s = _standardize(Xg_tr_r)
    Xb_tr_s = _standardize(Xb_tr_r)
    
    # PCA on gene side
    gene_pca_dim = best_result.get("gene_pca_dim", 128)
    n_comp = min(gene_pca_dim, Xg_tr_s.shape[0], Xg_tr_s.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=args.seed)
    Xg_tr_pca = pca.fit_transform(Xg_tr_s)
    
    print(f"  Gene PCA: {Xg_tr_pca.shape}", flush=True)

    print("[4/5] Fitting CCA/SCCA and extracting weights...", flush=True)
    method = best_result.get("method", "cca")
    c1 = best_result.get("c1", 0.3)
    c2 = best_result.get("c2", 0.3)
    k = min(10, Xg_tr_pca.shape[1], Xb_tr_s.shape[1])
    
    if method == "cca":
        model = CCA(n_components=k, max_iter=500)
        model.fit(Xg_tr_pca, Xb_tr_s)
        # Brain-side weights (loadings)
        brain_weights = model.y_rotations_  # (brain_features, k)
    else:  # scca
        model = SCCA_PMD(latent_dimensions=k, c=[c1, c2], max_iter=500, tol=1e-6)
        model.fit([Xg_tr_pca, Xb_tr_s])
        brain_weights = model.weights_[1]  # (brain_features, k)
    
    print(f"  Brain weights shape: {brain_weights.shape}", flush=True)
    
    # Use CC1 weights (first component)
    fc_weights_cc1 = brain_weights[:, 0]
    
    n_fc_features = Xb_tr_s.shape[1]
    
    # Determine n_nodes from n_fc_features using n_edges = n*(n-1)/2
    # Solving: n^2 - n - 2*n_edges = 0 → n = (1 + sqrt(1 + 8*n_edges)) / 2
    n_edges = n_fc_features
    n_nodes = int((1 + np.sqrt(1 + 8 * n_edges)) / 2)
    summary_mode = n_nodes < 50  # Any grouping is summary mode
    
    if summary_mode:
        # Try to load meta file for actual group names
        meta_path = Path(args.x_fmri_fc).parent / "meta_tian_summary.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            node_labels = meta.get("groups", [f"Group_{i}" for i in range(n_nodes)])
        else:
            # Fallback: use first n_nodes from TIAN_ANATOMICAL_GROUPS
            node_labels = list(TIAN_ANATOMICAL_GROUPS.keys())[:n_nodes]
        node_weights = fc_edge_to_node_weights(fc_weights_cc1, n_nodes)
    else:
        n_nodes = 50
        node_labels = TIAN_S3_LABELS
        node_weights = fc_edge_to_node_weights(fc_weights_cc1, n_nodes)
    
    node_weights_norm = node_weights / (node_weights.sum() + 1e-10)

    print("[5/5] Saving weight analysis...", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ROI ranking
    ranking = sorted(
        [(i, node_labels[i] if i < len(node_labels) else f"NODE_{i}", node_weights[i], node_weights_norm[i])
         for i in range(n_nodes)],
        key=lambda x: x[2],
        reverse=True
    )
    
    # Save ranking
    ranking_data = [
        {"rank": r + 1, "roi_idx": idx, "roi_name": name, "weight": float(w), "weight_norm": float(wn)}
        for r, (idx, name, w, wn) in enumerate(ranking)
    ]
    ranking_filename = "tian_group_weight_ranking.json" if summary_mode else "tian_roi_weight_ranking.json"
    (out_dir / ranking_filename).write_text(json.dumps(ranking_data, indent=2))
    
    # Category summary
    category_summary = {}
    if summary_mode:
        for i, name in enumerate(node_labels):
            cat_ranks = [r + 1 for r, (idx, _, _, _) in enumerate(ranking) if idx == i]
            category_summary[name] = {
                "total_weight": float(node_weights[i]),
                "total_weight_norm": float(node_weights_norm[i]),
                "n_rois": 1,
                "ranks": cat_ranks,
                "mean_rank": float(np.mean(cat_ranks)) if cat_ranks else None,
            }
    else:
    for cat, indices in ROI_CATEGORIES.items():
            valid_indices = [i for i in indices if i < n_nodes]
            cat_weight = sum(node_weights[i] for i in valid_indices)
            cat_weight_norm = sum(node_weights_norm[i] for i in valid_indices)
        cat_ranks = [r + 1 for r, (idx, _, _, _) in enumerate(ranking) if idx in valid_indices]
        category_summary[cat] = {
            "total_weight": float(cat_weight),
            "total_weight_norm": float(cat_weight_norm),
            "n_rois": len(valid_indices),
            "ranks": cat_ranks,
            "mean_rank": float(np.mean(cat_ranks)) if cat_ranks else None,
        }
    
    (out_dir / "tian_category_summary.json").write_text(json.dumps(category_summary, indent=2))
    
    # Overall summary
    summary = {
        "best_config": {
            "method": method,
            "gene_pca_dim": gene_pca_dim,
            "c1": c1 if method == "scca" else None,
            "c2": c2 if method == "scca" else None,
            "r_holdout_cc1": best_r_holdout,
        },
        "n_train": len(train_idx),
        "unit_type": "group" if summary_mode else "roi",
        "n_units": n_nodes,
        "n_fc_features": int(len(fc_weights_cc1)),
        "top_10_units": [
            {"unit_name": name, "weight_norm": float(wn)}
            for _, name, _, wn in ranking[:10]
        ],
        "hippocampus_summary": category_summary.get("Hippocampus", {}),
        "amygdala_summary": category_summary.get("Amygdala", {}),
    }
    (out_dir / "tian_weight_summary.json").write_text(json.dumps(summary, indent=2))
    
    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("TIAN WEIGHT ANALYSIS", flush=True)
    print("=" * 60, flush=True)
    print(f"Best config: {method} pca{gene_pca_dim}, r_holdout={best_r_holdout:.4f}", flush=True)
    print("\nTop 10 units by weight:", flush=True)
    for r, (idx, name, w, wn) in enumerate(ranking[:10]):
        print(f"  {r+1:2d}. {name:<20s}  weight={wn:.4f}", flush=True)
    
    print("\nCategory summary (normalized weight, mean rank):", flush=True)
    if summary_mode:
        for cat in node_labels:
            cs = category_summary.get(cat, {})
            print(f"  {cat:<20s}: {cs.get('total_weight_norm', 0):.4f}, mean_rank={cs.get('mean_rank', 'N/A')}", flush=True)
    else:
    for cat in ["Hippocampus", "Amygdala", "Thalamus", "NAc", "Brainstem (VTA/SN)"]:
        cs = category_summary.get(cat, {})
        print(f"  {cat:<20s}: {cs.get('total_weight_norm', 0):.4f}, mean_rank={cs.get('mean_rank', 'N/A')}", flush=True)
    
    print("=" * 60, flush=True)
    print(f"\n[done] Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
