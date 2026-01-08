#!/bin/bash
#SBATCH --job-name=cca_stage1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# =============================================================================
# Stage 1: Conventional CCA - Gene-Brain Joint Embeddings
# =============================================================================
# This runs conventional CCA to find linear combinations of gene and brain
# features that maximize their correlation. The canonical variates (U, V)
# are saved for Stage 2 clinical prediction.
#
# Requires: scikit-learn
# =============================================================================

# Adjust these paths on the lab server
ALIGNED="$PWD/derived/aligned_pca"
OUT_DIR="$PWD/derived/cca_stage1"

# CCA parameters
N_COMPONENTS=10   # Number of canonical components
N_PERM=1000       # Permutation tests (set to 0 to skip)
SEED=42

mkdir -p logs "$OUT_DIR"

source ~/.bashrc || true
conda activate <YOUR_ENV>

INFO_JSON="$ALIGNED/pca_info.json"
GENE_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['gene_pca_dim'])" "$INFO_JSON")"
FMRI_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fmri_pca_dim'])" "$INFO_JSON")"
X_GENE="$ALIGNED/X_gene_pca${GENE_PCA_DIM}.npy"
X_FMRI="$ALIGNED/X_fmri_pca${FMRI_PCA_DIM}.npy"

echo "[$(date)] Starting Conventional CCA (Stage 1)"
echo "[info] Input: $ALIGNED"
echo "[info] Output: $OUT_DIR"
echo "[info] X_gene: $X_GENE"
echo "[info] X_fmri: $X_FMRI"

python scripts/run_cca.py \
  --x-gene "$X_GENE" \
  --x-fmri "$X_FMRI" \
  --ids "$ALIGNED/ids_common.npy" \
  --method conventional \
  --n-components "$N_COMPONENTS" \
  --n-perm "$N_PERM" \
  --seed "$SEED" \
  --out-dir "$OUT_DIR" \
  --prefix "conventional_"

echo "[$(date)] Conventional CCA complete"
echo "[outputs]"
ls -lh "$OUT_DIR"/conventional_*


