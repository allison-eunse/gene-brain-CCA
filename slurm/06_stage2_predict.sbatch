#!/bin/bash
#SBATCH --job-name=stage2_predict
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

# =============================================================================
# Stage 2: Supervised Clinical Prediction from CCA Joint Embeddings
# =============================================================================
# This script evaluates the clinical predictive power of CCA joint embeddings.
# It runs cross-validated Logistic Regression and MLP on:
#   - gene_only: Gene canonical variates (U)
#   - fmri_only: fMRI canonical variates (V)
#   - joint: Concatenated [U, V]
#
# Comparison:
#   - If joint > max(gene_only, fmri_only): Gene-brain coupling is predictive
#   - If SCCA joint > CCA joint: Localized patterns drive the relationship
#
# Requires: scikit-learn
# =============================================================================

# Use aligned labels from align_resid_pca.py output
ALIGNED="$PWD/derived/aligned_pca"
LABELS="$ALIGNED/labels_common.npy"  # MDD labels (0/1) aligned to CCA subjects

# Run for both CCA and SCCA
CCA_DIR="$PWD/derived/cca_stage1"
SCCA_DIR="$PWD/derived/scca_stage1"

CCA_OUT="$PWD/derived/stage2_cca"
SCCA_OUT="$PWD/derived/stage2_scca"

# Prediction settings
N_FOLDS=5
SEED=42
MLP_HIDDEN="64,32"

mkdir -p logs "$CCA_OUT" "$SCCA_OUT"

# Auto-create conda environment if it doesn't exist
source /usr/anaconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^cca_env "; then
    echo "[setup] Creating cca_env..."
    conda create -n cca_env python=3.10 numpy scipy scikit-learn -y
    conda activate /scratch/connectome/allie/envs/cca_env
    pip install cca-zoo
else
    conda activate /scratch/connectome/allie/envs/cca_env
fi

echo "[$(date)] Starting Stage 2 Clinical Prediction"

# =============================================================================
# Stage 2 for Conventional CCA
# =============================================================================
echo ""
echo "=========================================="
echo "Stage 2: Conventional CCA Embeddings"
echo "=========================================="

python scripts/stage2_predict.py \
  --u-gene "$CCA_DIR/conventional_U_gene.npy" \
  --v-fmri "$CCA_DIR/conventional_V_fmri.npy" \
  --labels "$LABELS" \
  --ids "$CCA_DIR/conventional_ids.npy" \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --mlp-hidden "$MLP_HIDDEN" \
  --out-dir "$CCA_OUT" \
  --prefix "cca_"

# =============================================================================
# Stage 2 for Sparse CCA
# =============================================================================
echo ""
echo "=========================================="
echo "Stage 2: Sparse CCA Embeddings"
echo "=========================================="

python scripts/stage2_predict.py \
  --u-gene "$SCCA_DIR/sparse_U_gene.npy" \
  --v-fmri "$SCCA_DIR/sparse_V_fmri.npy" \
  --labels "$LABELS" \
  --ids "$SCCA_DIR/sparse_ids.npy" \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds "$N_FOLDS" \
  --seed "$SEED" \
  --mlp-hidden "$MLP_HIDDEN" \
  --out-dir "$SCCA_OUT" \
  --prefix "scca_"

# =============================================================================
# Summary Comparison
# =============================================================================
echo ""
echo "=========================================="
echo "COMPARISON: Conventional vs Sparse CCA"
echo "=========================================="

python3 - << 'PYTHON'
import json
from pathlib import Path

cca_results = Path("derived/stage2_cca/cca_results.json")
scca_results = Path("derived/stage2_scca/scca_results.json")

if cca_results.exists() and scca_results.exists():
    cca = json.loads(cca_results.read_text())
    scca = json.loads(scca_results.read_text())

    print(f"\nBest CCA config:  {cca['best_config']}")
    print(f"Best SCCA config: {scca['best_config']}")

    cca_auc = cca["best_config"]["auc"]
    scca_auc = scca["best_config"]["auc"]

    diff = scca_auc - cca_auc
    print(f"\nDifference (SCCA - CCA): {diff:+.4f}")

    if diff > 0.02:
        print("\n>>> SCCA outperforms CCA significantly <<<")
        print("Interpretation: The gene-brain relationship is driven by")
        print("LOCALIZED, SPECIFIC patterns (like the directional microphone")
        print("finding the three people actually talking in a stadium).")
    elif diff < -0.02:
        print("\n>>> CCA outperforms SCCA significantly <<<")
        print("Interpretation: The gene-brain relationship is driven by")
        print("GLOBAL patterns across the entire feature space.")
    else:
        print("\n>>> CCA and SCCA perform similarly <<<")
        print("Interpretation: Both global and localized patterns contribute.")
else:
    print("[WARN] Missing results files for comparison")
PYTHON

echo ""
echo "[$(date)] Stage 2 complete"



