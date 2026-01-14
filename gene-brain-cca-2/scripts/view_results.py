#!/usr/bin/env python3
"""
Quick results viewer for gene-brain-cca-2 pipelines.

Usage:
    python view_results.py [--pipeline A|B|both]
"""

import argparse
import json
from pathlib import Path
import numpy as np


def print_header(text, char="="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f"{text:^60}")
    print(f"{char * 60}\n")


def view_pipeline_a(base_dir):
    """Display Pipeline A (Interpretable SCCA) results."""
    print_header("Pipeline A: Interpretable SCCA Results")
    
    json_path = base_dir / "interpretable" / "scca_interpretable_results.json"
    
    if not json_path.exists():
        print(f"❌ Results not found: {json_path}")
        print("   Run Pipeline A first: sbatch slurm/01_interpretable_scca.sbatch")
        return
    
    with open(json_path) as f:
        results = json.load(f)
    
    # Holdout-aware results
    split = results.get("split", {})
    print("📊 Dataset Split (one-touch holdout):")
    print("-" * 60)
    if split:
        print(
            f"  N_total={split.get('n_total')} | N_train={split.get('n_train')} | N_holdout={split.get('n_holdout')}"
        )
        print(
            f"  pos_ratio: total={split.get('pos_ratio_total', 0):.3f}, "
            f"train={split.get('pos_ratio_train', 0):.3f}, holdout={split.get('pos_ratio_holdout', 0):.3f}"
        )
        print(f"  holdout_frac={results.get('params', {}).get('holdout_frac', 'NA')}")

    train_fit = results.get("train_fit", {})
    r_train = train_fit.get("r_train", [])
    r_holdout = train_fit.get("r_holdout", [])
    
    if r_train:
        print("\n📈 Stage 1 correlation generalization:")
        print("-" * 60)
        print("  Train correlations (top 5):")
        for i, r in enumerate(r_train[:5], 1):
            print(f"    Component {i}: r = {r:.4f}  (r² = {r**2:.4f})")
        if r_holdout:
            print("\n  Holdout correlations (top 5):")
            for i, r in enumerate(r_holdout[:5], 1):
                print(f"    Component {i}: r = {r:.4f}  (r² = {r**2:.4f})")
        
        print(f"\n  Sparsity (train-fit weights):")
        print(f"    Gene features: {train_fit.get('sparsity_gene', 0):.2%} near-zero")
        print(f"    fMRI features: {train_fit.get('sparsity_fmri', 0):.2%} near-zero")
        
        # Calculate number of selected features
        n_genes_selected = int(111 * (1 - train_fit.get("sparsity_gene", 0)))
        n_rois_selected = int(180 * (1 - train_fit.get("sparsity_fmri", 0)))
        print(f"    → ~{n_genes_selected}/111 genes selected (approx)")
        print(f"    → ~{n_rois_selected}/180 ROIs selected (approx)")
    
    # Cross-validation stability
    folds = results.get("folds", [])
    if folds:
        print(f"\n📈 Cross-Validation Stability (train-only CV; {len(folds)} folds):")
        print("-" * 60)
        
        # Extract first component validation correlations across folds
        r1_val = [fold["r_val"][0] for fold in folds if "r_val" in fold and len(fold["r_val"]) > 0]
        if r1_val:
            r1_mean = np.mean(r1_val)
            r1_std = np.std(r1_val)
            r1_min = np.min(r1_val)
            r1_max = np.max(r1_val)
            
            print(f"  Component 1 validation correlation across folds:")
            print(f"    Mean: {r1_mean:.4f}")
            print(f"    Std:  {r1_std:.4f}")
            print(f"    Range: [{r1_min:.4f}, {r1_max:.4f}]")
            
            if r1_std < 0.05:
                print(f"    ✓ Low variance → stable estimates")
            elif r1_std < 0.10:
                print(f"    ⚠ Moderate variance")
            else:
                print(f"    ⚠ High variance → unstable estimates")
        
        # Sparsity stability
        sp_gene = [fold.get('sparsity_gene', 0) for fold in folds]
        sp_fmri = [fold.get('sparsity_fmri', 0) for fold in folds]
        
        print(f"\n  Sparsity stability:")
        print(f"    Gene: {np.mean(sp_gene):.2%} ± {np.std(sp_gene):.2%}")
        print(f"    fMRI: {np.mean(sp_fmri):.2%} ± {np.std(sp_fmri):.2%}")
    
    # Check if weight/score files exist
    print(f"\n📁 Additional Files:")
    print("-" * 60)
    artifacts = results.get("artifacts", {})
    if artifacts:
        for key, fname in artifacts.items():
            fpath = json_path.parent / fname
            if fpath.exists():
                try:
                    arr = np.load(fpath, allow_pickle=True)
                    shape = getattr(arr, "shape", None)
                    print(f"  ✓ {key}: {fname} ({shape})")
                except Exception:
                    print(f"  ✓ {key}: {fname}")
            else:
                print(f"  ✗ {key}: missing ({fname})")
    else:
        print("  (No artifacts listed in results JSON)")
    
    print()


def view_pipeline_b(base_dir):
    """Display Pipeline B (Predictive Suite) results."""
    print_header("Pipeline B: Predictive Suite Results")
    
    json_path = base_dir / "wide_gene" / "predictive_suite_results.json"
    
    if not json_path.exists():
        print(f"❌ Results not found: {json_path}")
        print("   Run Pipeline B first: sbatch slurm/02_predictive_wide_suite.sbatch")
        return
    
    with open(json_path) as f:
        results = json.load(f)
    
    # Organize results by category
    split = results.get("split", {})
    if split:
        print("📊 Dataset Split (one-touch holdout):")
        print("-" * 60)
        print(
            f"  N_total={split.get('n_total')} | N_train={split.get('n_train')} | N_holdout={split.get('n_holdout')}"
        )
        print(
            f"  pos_ratio: total={split.get('pos_ratio_total', 0):.3f}, "
            f"train={split.get('pos_ratio_train', 0):.3f}, holdout={split.get('pos_ratio_holdout', 0):.3f}"
        )
        print()

    cv = results.get("cv")
    hold = results.get("holdout")
    if cv is None and hold is None:
        # Back-compat: old flat dict
        cv = results
        hold = None

    print("📊 Classification Performance (AUC | AP):")
    print("-" * 60)
    
    # Baseline models
    if hold:
        print("  HOLDOUT (final): Baseline Models (LogReg):")
        for key in ['gene_only_logreg', 'fmri_only_logreg', 'early_fusion_logreg']:
            if key in hold:
                auc = hold[key].get('auc', 0)
                ap = hold[key].get('ap', 0)
                label = key.replace('_logreg', '').replace('_', ' ').title()
                print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")

        print("\n  HOLDOUT (final): CCA/SCCA Models (LogReg):")
        for key in ['cca_joint_logreg', 'scca_joint_logreg']:
            if key in hold:
                auc = hold[key].get('auc', 0)
                ap = hold[key].get('ap', 0)
                label = key.replace('_logreg', '').replace('_', ' ').title()
                print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")

        print("\n  HOLDOUT (final): Neural Network Models (MLP):")
        for key in ['gene_only_mlp', 'fmri_only_mlp', 'early_fusion_mlp',
                    'cca_joint_mlp', 'scca_joint_mlp']:
            if key in hold:
                auc = hold[key].get('auc', 0)
                ap = hold[key].get('ap', 0)
                label = key.replace('_mlp', '').replace('_', ' ').title()
                print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")

    print("\n  CV on TRAIN (for tuning): Baseline Models (LogReg):")
    for key in ['gene_only_logreg', 'fmri_only_logreg', 'early_fusion_logreg']:
        if key in cv:
            auc = cv[key].get('auc', 0)
            ap = cv[key].get('ap', 0)
            label = key.replace('_logreg', '').replace('_', ' ').title()
            print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")
    
    # CCA/SCCA models
    print("\n  CV on TRAIN (for tuning): CCA/SCCA Models (LogReg):")
    for key in ['cca_joint_logreg', 'scca_joint_logreg']:
        if key in cv:
            auc = cv[key].get('auc', 0)
            ap = cv[key].get('ap', 0)
            label = key.replace('_logreg', '').replace('_', ' ').title()
            print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")
    
    # MLP models
    print("\n  CV on TRAIN (for tuning): Neural Network Models (MLP):")
    for key in ['gene_only_mlp', 'fmri_only_mlp', 'early_fusion_mlp',
                'cca_joint_mlp', 'scca_joint_mlp']:
        if key in cv:
            auc = cv[key].get('auc', 0)
            ap = cv[key].get('ap', 0)
            label = key.replace('_mlp', '').replace('_', ' ').title()
            print(f"    {label:20s}: AUC={auc:.4f}, AP={ap:.4f}")
    
    # Find best model
    print("\n🏆 Best Model:")
    print("-" * 60)
    best_space = hold if hold else cv
    best_model = max(best_space.items(), key=lambda x: x[1].get('auc', 0))
    best_name = best_model[0].replace('_', ' ').title()
    best_auc = best_model[1].get('auc', 0)
    best_ap = best_model[1].get('ap', 0)
    where = "HOLDOUT" if hold else "CV"
    print(f"  ({where}) {best_name}")
    print(f"  AUC = {best_auc:.4f}")
    print(f"  AP  = {best_ap:.4f}")
    
    # Performance comparison
    print("\n📈 Key Comparisons:")
    print("-" * 60)
    
    # Modality comparison
    space = hold if hold else cv
    if 'gene_only_logreg' in space and 'fmri_only_logreg' in space:
        gene_auc = space['gene_only_logreg']['auc']
        fmri_auc = space['fmri_only_logreg']['auc']
        diff = fmri_auc - gene_auc
        better = "fMRI" if diff > 0 else "Gene"
        print(f"  Modality: {better} is more predictive")
        print(f"    Gene-only AUC: {gene_auc:.4f}")
        print(f"    fMRI-only AUC: {fmri_auc:.4f}")
        print(f"    Difference: {abs(diff):.4f} ({abs(diff)/max(gene_auc, fmri_auc)*100:.1f}%)")
    
    # Fusion benefit
    if 'early_fusion_logreg' in space and 'fmri_only_logreg' in space:
        best_single = max(space.get('gene_only_logreg', {}).get('auc', 0),
                         space.get('fmri_only_logreg', {}).get('auc', 0))
        fusion_auc = space['early_fusion_logreg']['auc']
        gain = fusion_auc - best_single
        print(f"\n  Fusion benefit:")
        print(f"    Best single modality: {best_single:.4f}")
        print(f"    Early fusion: {fusion_auc:.4f}")
        print(f"    Gain: {gain:+.4f} ({gain/best_single*100:+.1f}%)")
        if gain > 0.03:
            print(f"    → Substantial benefit from fusion ✓")
        elif gain > 0.01:
            print(f"    → Modest benefit from fusion")
        else:
            print(f"    → Minimal benefit from fusion")
    
    # CCA value
    if 'early_fusion_logreg' in space and 'cca_joint_logreg' in space:
        fusion_auc = space['early_fusion_logreg']['auc']
        cca_auc = space['cca_joint_logreg']['auc']
        diff = cca_auc - fusion_auc
        print(f"\n  CCA vs simple concatenation:")
        print(f"    Early fusion: {fusion_auc:.4f}")
        print(f"    CCA joint: {cca_auc:.4f}")
        print(f"    Difference: {diff:+.4f}")
        if abs(diff) < 0.01:
            print(f"    → CCA provides no advantage")
        elif diff > 0:
            print(f"    → CCA improves performance ✓")
        else:
            print(f"    → CCA hurts performance")
    
    # LogReg vs MLP
    if 'early_fusion_logreg' in space and 'early_fusion_mlp' in space:
        lr_auc = space['early_fusion_logreg']['auc']
        mlp_auc = space['early_fusion_mlp']['auc']
        diff = mlp_auc - lr_auc
        print(f"\n  Nonlinearity benefit (LogReg vs MLP):")
        print(f"    LogReg: {lr_auc:.4f}")
        print(f"    MLP: {mlp_auc:.4f}")
        print(f"    Difference: {diff:+.4f}")
        if diff > 0.02:
            print(f"    → Nonlinear patterns present ✓")
        elif abs(diff) < 0.01:
            print(f"    → Linear model sufficient")
        else:
            print(f"    → MLP may be overfitting")
    
    print()


def view_data_info(base_dir):
    """Display dataset information."""
    print_header("Dataset Information", char="-")
    
    # Check interpretable pipeline outputs
    ids_path = base_dir / "interpretable" / "ids_common.npy"
    xg_path = base_dir / "interpretable" / "X_gene_z.npy"
    xb_path = base_dir / "interpretable" / "X_fmri_z.npy"
    labels_path = base_dir / "interpretable" / "labels_common.npy"
    
    if ids_path.exists():
        ids = np.load(ids_path, allow_pickle=True)
        print(f"  Sample size: {len(ids)} subjects")
        
        if xg_path.exists():
            Xg = np.load(xg_path)
            print(f"  Gene features: {Xg.shape[1]} (after preprocessing)")
        
        if xb_path.exists():
            Xb = np.load(xb_path)
            print(f"  fMRI features: {Xb.shape[1]} ROIs")
        
        if labels_path.exists():
            y = np.load(labels_path)
            pos_rate = y.mean()
            print(f"  Labels: {len(y)} ({pos_rate:.1%} positive, {1-pos_rate:.1%} negative)")
            if pos_rate < 0.2 or pos_rate > 0.8:
                print(f"    ⚠ Imbalanced dataset → prioritize AP over AUC")
    else:
        print("  ❌ No data found. Run Pipeline A first.")
    
    # Check wide gene outputs
    xgw_path = base_dir / "wide_gene" / "X_gene_wide.npy"
    pred_json = base_dir / "wide_gene" / "predictive_suite_results.json"
    
    if xgw_path.exists():
        print(f"\n  Wide gene matrix: {xgw_path.stat().st_size / 1e9:.2f} GB")
        Xgw = np.load(xgw_path, mmap_mode='r')
        print(f"    Shape: {Xgw.shape} (111 genes × 768 dims)")

    if pred_json.exists():
        try:
            with open(pred_json) as f:
                pr = json.load(f)
            pca = pr.get("pca") or {}
            if pca:
                print(f"  Gene PCA (train-only): {pca.get('n_components')} components")
                ev = pca.get("explained_variance_ratio_sum")
                if ev is not None:
                    print(f"    Explained variance ratio sum: {ev:.3f}")
        except Exception:
            pass
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="View results from gene-brain-cca-2 pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_results.py              # View all results
  python view_results.py --pipeline A # View Pipeline A only
  python view_results.py --pipeline B # View Pipeline B only
        """
    )
    parser.add_argument(
        "--pipeline",
        choices=["A", "B", "both"],
        default="both",
        help="Which pipeline results to view (default: both)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("gene-brain-cca-2/derived"),
        help="Base directory for results (default: gene-brain-cca-2/derived)"
    )
    
    args = parser.parse_args()
    
    if not args.base_dir.exists():
        print(f"❌ Results directory not found: {args.base_dir}")
        print(f"   Make sure you're in the project root directory.")
        return 1
    
    # Display data info
    view_data_info(args.base_dir)
    
    # Display pipeline results
    if args.pipeline in ["A", "both"]:
        view_pipeline_a(args.base_dir)
    
    if args.pipeline in ["B", "both"]:
        view_pipeline_b(args.base_dir)
    
    print_header("For detailed interpretation, see RESULTS_GUIDE.md", char="-")
    
    return 0


if __name__ == "__main__":
    exit(main())
