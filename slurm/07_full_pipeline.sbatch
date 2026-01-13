#!/bin/bash
#SBATCH --job-name=cca_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

# =============================================================================
# Full End-to-End CCA Pipeline: Gene Build → Align → CCA/SCCA → Prediction
# =============================================================================
# This script runs the COMPLETE pipeline from scratch:
#   Step 0: Build gene features from DNABERT2 embeddings
#   Step 1: Align gene/fMRI subjects, residualize, PCA, create labels_common
#   Stage 1a: Conventional CCA → canonical variates
#   Stage 1b: Sparse CCA → sparse canonical variates
#   Stage 2:  Clinical prediction (LogReg + MLP) for both
#   Comparison and interpretation
#
# The goal is to determine whether the gene-brain relationship for your
# clinical trait (e.g., MDD) is driven by:
#   - GLOBAL patterns (CCA wins) - like averaging the crowd's roar
#   - LOCALIZED patterns (SCCA wins) - like a directional microphone
# =============================================================================

# =============================================================================
# PATHS (adjust if needed)
# =============================================================================
NESAP="/storage/bigdata/UKB/fMRI/nesap-genomics-allison"
FMRI_IDS="/storage/bigdata/UKB/fMRI/fmri_eids_180.npy"
FMRI_X="/storage/bigdata/UKB/fMRI/fmri_X_180.npy"

# Derived outputs
GENE_DIR="$PWD/derived/gene_x"
ALIGNED="$PWD/derived/aligned_pca"
LABELS="$ALIGNED/labels_common.npy"

# Stage 1/2 output directories
CCA_OUT="$PWD/derived/cca_stage1"
SCCA_OUT="$PWD/derived/scca_stage1"
PRED_CCA="$PWD/derived/stage2_cca"
PRED_SCCA="$PWD/derived/stage2_scca"
COMPARE_OUT="$PWD/derived/comparison"

# Parameters
N_COMPONENTS=10
N_PERM=1000
C1=0.3
C2=0.3
N_FOLDS=5
SEED=42
REDUCE_METHOD="max"  # mean, max, or median for gene embedding reduction

mkdir -p logs "$GENE_DIR" "$ALIGNED" "$CCA_OUT" "$SCCA_OUT" "$PRED_CCA" "$PRED_SCCA" "$COMPARE_OUT"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Assumes `cca_env` already exists and has dependencies installed:
#   conda create -n cca_env python=3.10 -y
#   conda activate /scratch/connectome/allie/envs/cca_env
#   pip install numpy scipy scikit-learn cca-zoo
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

echo "============================================================"
echo "Full End-to-End CCA Pipeline"
echo "============================================================"
echo "Started: $(date)"
echo ""

# =============================================================================
# STEP 0: BUILD GENE FEATURES (skip if already exists)
# =============================================================================
if [[ -f "$GENE_DIR/X_gene_ng.npy" && -f "$GENE_DIR/ids_gene.npy" ]]; then
    echo "[step0] Gene features already exist, skipping..."
else
    echo ">>> STEP 0: Building gene features from DNABERT2 embeddings <<<"
    python scripts/build_x_gene.py \
      --embed-root "$NESAP/DNABERT2_embedding_merged" \
      --gene-list "$NESAP/iids_labels_covariates/gene_list_filtered.txt" \
      --iids "$NESAP/iids_labels_covariates/iids.npy" \
      --n-files 49 \
      --reduce "$REDUCE_METHOD" \
      --out-dir "$GENE_DIR"
    echo "[step0] Gene features built: $GENE_DIR"
fi
echo ""

# =============================================================================
# STEP 1: ALIGN + RESIDUALIZE + PCA + LABELS (skip if already exists)
# =============================================================================
if [[ -f "$ALIGNED/pca_info.json" && -f "$ALIGNED/labels_common.npy" ]]; then
    echo "[step1] Aligned data already exists, skipping..."
else
    echo ">>> STEP 1: Aligning subjects, residualizing, PCA, creating labels_common <<<"
    python scripts/align_resid_pca.py \
      --ids-gene "$GENE_DIR/ids_gene.npy" \
      --x-gene "$GENE_DIR/X_gene_ng.npy" \
      --ids-fmri "$FMRI_IDS" \
      --x-fmri "$FMRI_X" \
      --cov-iids "$NESAP/iids_labels_covariates/iids.npy" \
      --cov-age "$NESAP/iids_labels_covariates/covariates_age.npy" \
      --cov-sex "$NESAP/iids_labels_covariates/covariates_sex.npy" \
      --cov-valid-mask "$NESAP/iids_labels_covariates/covariates_valid_mask.npy" \
      --labels "$NESAP/iids_labels_covariates/labels.npy" \
      --pca-dim 512 \
      --out-dir "$ALIGNED"
    echo "[step1] Aligned data created: $ALIGNED"
fi
echo ""

# =============================================================================
# READ PCA DIMENSIONS FROM pca_info.json
# =============================================================================
INFO_JSON="$ALIGNED/pca_info.json"
GENE_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['gene_pca_dim'])" "$INFO_JSON")"
FMRI_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fmri_pca_dim'])" "$INFO_JSON")"
X_GENE="$ALIGNED/X_gene_pca${GENE_PCA_DIM}.npy"
X_FMRI="$ALIGNED/X_fmri_pca${FMRI_PCA_DIM}.npy"

echo "[info] X_gene: $X_GENE (dim=$GENE_PCA_DIM)"
echo "[info] X_fmri: $X_FMRI (dim=$FMRI_PCA_DIM)"
echo "[info] Labels: $LABELS"
echo ""

# =============================================================================
# STAGE 1A: Conventional CCA
# =============================================================================
echo ">>> STAGE 1A: Conventional CCA <<<"

python scripts/run_cca.py \
  --x-gene "$X_GENE" \
  --x-fmri "$X_FMRI" \
  --ids "$ALIGNED/ids_common.npy" \
  --method conventional \
  --n-components "$N_COMPONENTS" \
  --n-perm "$N_PERM" \
  --seed "$SEED" \
  --out-dir "$CCA_OUT" \
  --prefix "conventional_"

echo ""

# =============================================================================
# STAGE 1B: Sparse CCA
# =============================================================================
echo ">>> STAGE 1B: Sparse CCA (c1=$C1, c2=$C2) <<<"

python scripts/run_cca.py \
  --x-gene "$X_GENE" \
  --x-fmri "$X_FMRI" \
  --ids "$ALIGNED/ids_common.npy" \
  --method sparse \
  --n-components "$N_COMPONENTS" \
  --c1 "$C1" \
  --c2 "$C2" \
  --n-perm "$N_PERM" \
  --seed "$SEED" \
  --out-dir "$SCCA_OUT" \
  --prefix "sparse_"

echo ""

# =============================================================================
# STAGE 2: Clinical Prediction
# =============================================================================
echo ">>> STAGE 2: Clinical Prediction <<<"

echo "[CCA] Running Stage 2 prediction..."
python scripts/stage2_predict.py \
  --u-gene "$CCA_OUT/conventional_U_gene.npy" \
  --v-fmri "$CCA_OUT/conventional_V_fmri.npy" \
  --labels "$LABELS" \
  --ids "$CCA_OUT/conventional_ids.npy" \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --out-dir "$PRED_CCA" \
  --prefix "cca_"

echo ""

echo "[SCCA] Running Stage 2 prediction..."
python scripts/stage2_predict.py \
  --u-gene "$SCCA_OUT/sparse_U_gene.npy" \
  --v-fmri "$SCCA_OUT/sparse_V_fmri.npy" \
  --labels "$LABELS" \
  --ids "$SCCA_OUT/sparse_ids.npy" \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --out-dir "$PRED_SCCA" \
  --prefix "scca_"

echo ""

# =============================================================================
# COMPARISON AND ANALYSIS
# =============================================================================
echo ">>> COMPARISON AND ANALYSIS <<<"

python3 << 'PYTHON'
import json
from pathlib import Path
import numpy as np

compare_out = Path("derived/comparison")
compare_out.mkdir(exist_ok=True)

# Load all results
cca_stage1 = json.loads(Path("derived/cca_stage1/conventional_results.json").read_text())
scca_stage1 = json.loads(Path("derived/scca_stage1/sparse_results.json").read_text())
cca_stage2 = json.loads(Path("derived/stage2_cca/cca_results.json").read_text())
scca_stage2 = json.loads(Path("derived/stage2_scca/scca_results.json").read_text())

print("\n" + "=" * 70)
print("FINAL COMPARISON REPORT")
print("=" * 70)

# Stage 1 comparison
print("\n--- STAGE 1: Canonical Correlations ---")
print(f"CCA  top-3 correlations: {cca_stage1['canonical_correlations'][:3]}")
print(f"SCCA top-3 correlations: {scca_stage1['canonical_correlations'][:3]}")
print(f"CCA  sparsity: gene={cca_stage1['sparsity_gene']:.1%}, fmri={cca_stage1['sparsity_fmri']:.1%}")
print(f"SCCA sparsity: gene={scca_stage1['sparsity_gene']:.1%}, fmri={scca_stage1['sparsity_fmri']:.1%}")

# Stage 2 comparison
print("\n--- STAGE 2: Clinical Prediction (AUC-ROC) ---")

def extract_best(results, prefix):
    """Extract best AUC for each feature set."""
    out = {}
    for r in results["results"]:
        fs = r["feature_set"]
        if fs not in out or r["auc_mean"] > out[fs]:
            out[fs] = r["auc_mean"]
    return out

cca_aucs = extract_best(cca_stage2, "CCA")
scca_aucs = extract_best(scca_stage2, "SCCA")

print(f"{'Feature Set':<15} {'CCA AUC':>12} {'SCCA AUC':>12} {'Δ (SCCA-CCA)':>15}")
print("-" * 55)
for fs in ["gene_only", "fmri_only", "joint"]:
    cca_auc = cca_aucs.get(fs, 0)
    scca_auc = scca_aucs.get(fs, 0)
    delta = scca_auc - cca_auc
    print(f"{fs:<15} {cca_auc:>12.4f} {scca_auc:>12.4f} {delta:>+15.4f}")

# Overall best
cca_best = cca_stage2["best_config"]["auc"]
scca_best = scca_stage2["best_config"]["auc"]
delta_best = scca_best - cca_best

print("\n--- INTERPRETATION ---")
print(f"Best CCA:  {cca_stage2['best_config']['feature_set']} + {cca_stage2['best_config']['model']} = {cca_best:.4f}")
print(f"Best SCCA: {scca_stage2['best_config']['feature_set']} + {scca_stage2['best_config']['model']} = {scca_best:.4f}")
print(f"Difference (SCCA - CCA): {delta_best:+.4f}")

print("\n" + "=" * 70)
if delta_best > 0.02:
    print("CONCLUSION: SCCA OUTPERFORMS CCA")
    print("-" * 70)
    print("The gene-brain relationship for your clinical trait is driven by")
    print("LOCALIZED, SPECIFIC PATTERNS rather than global interactions.")
    print("")
    print("Analogy: SCCA acts like a directional microphone that zeroes in on")
    print("the specific 'conversation' (biomarkers) in a noisy stadium.")
    print("")
    print("Next steps:")
    print("  1. Examine sparse_W_gene.npy and sparse_W_fmri.npy for feature importance")
    print("  2. Identify which genes/brain regions have non-zero weights")
    print("  3. These are your candidate biomarkers for the clinical phenotype")
elif delta_best < -0.02:
    print("CONCLUSION: CCA OUTPERFORMS SCCA")
    print("-" * 70)
    print("The gene-brain relationship is driven by GLOBAL, DISTRIBUTED patterns.")
    print("Many features contribute modestly to the correlation.")
    print("")
    print("Analogy: CCA captures the overall 'roar' of the crowd, which contains")
    print("more predictive signal than any specific conversation.")
else:
    print("CONCLUSION: CCA AND SCCA PERFORM SIMILARLY")
    print("-" * 70)
    print("Both global and localized patterns contribute to the gene-brain")
    print("relationship. Consider:")
    print("  1. Using ensemble methods combining both")
    print("  2. Examining SCCA weights anyway for interpretability")
    print("  3. Tuning sparsity parameters (c1, c2) for your specific use case")

print("=" * 70)

# Save comparison report
comparison = {
    "stage1": {
        "cca_correlations": cca_stage1["canonical_correlations"],
        "scca_correlations": scca_stage1["canonical_correlations"],
        "cca_sparsity": {"gene": cca_stage1["sparsity_gene"], "fmri": cca_stage1["sparsity_fmri"]},
        "scca_sparsity": {"gene": scca_stage1["sparsity_gene"], "fmri": scca_stage1["sparsity_fmri"]},
    },
    "stage2": {
        "cca_aucs": cca_aucs,
        "scca_aucs": scca_aucs,
        "cca_best": cca_stage2["best_config"],
        "scca_best": scca_stage2["best_config"],
        "delta_best": float(delta_best),
    },
    "conclusion": "scca_wins" if delta_best > 0.02 else ("cca_wins" if delta_best < -0.02 else "similar"),
}

Path("derived/comparison/comparison_report.json").write_text(json.dumps(comparison, indent=2))
print(f"\n[save] derived/comparison/comparison_report.json")
PYTHON

echo ""
echo "============================================================"
echo "Pipeline complete: $(date)"
echo "============================================================"


