#!/bin/bash
#SBATCH --job-name=scca_stage1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# =============================================================================
# Stage 1: Sparse CCA (SCCA) - Gene-Brain Joint Embeddings
# =============================================================================
# Sparse CCA adds L1 penalties to canonical coefficients, which:
#   - Automatically suppresses non-informative features (denoising)
#   - Produces interpretable results (sparse weights)
#   - Identifies specific biomarkers driving the correlation
#
# If SCCA outperforms conventional CCA in Stage 2, it indicates that the
# gene-brain relationship is driven by localized, specific patterns.
#
# Requires: cca-zoo (pip install cca-zoo) or uses fallback implementation
# =============================================================================

# Adjust these paths on the lab server
ALIGNED="$PWD/derived/aligned_pca"
OUT_DIR="$PWD/derived/scca_stage1"

# SCCA parameters
N_COMPONENTS=10   # Number of canonical components
C1=0.3            # Sparsity for gene (0 < c <= 1, lower = sparser)
C2=0.3            # Sparsity for fMRI (0 < c <= 1, lower = sparser)
N_PERM=1000       # Permutation tests (set to 0 to skip)
SEED=42

mkdir -p logs "$OUT_DIR"

# Auto-create conda environment if it doesn't exist
source ~/.bashrc || true
if ! conda env list | grep -q "^cca_env "; then
    echo "[setup] Creating cca_env..."
    conda create -n cca_env python=3.10 numpy scipy scikit-learn -y
    conda activate cca_env
    pip install cca-zoo
else
    conda activate cca_env
fi

INFO_JSON="$ALIGNED/pca_info.json"
GENE_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['gene_pca_dim'])" "$INFO_JSON")"
FMRI_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fmri_pca_dim'])" "$INFO_JSON")"
X_GENE="$ALIGNED/X_gene_pca${GENE_PCA_DIM}.npy"
X_FMRI="$ALIGNED/X_fmri_pca${FMRI_PCA_DIM}.npy"

echo "[$(date)] Starting Sparse CCA (Stage 1)"
echo "[info] Input: $ALIGNED"
echo "[info] Output: $OUT_DIR"
echo "[info] Sparsity: c1=$C1, c2=$C2"
echo "[info] X_gene: $X_GENE"
echo "[info] X_fmri: $X_FMRI"

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
  --out-dir "$OUT_DIR" \
  --prefix "sparse_"

echo "[$(date)] Sparse CCA complete"
echo "[outputs]"
ls -lh "$OUT_DIR"/sparse_*

# =============================================================================
# Grid search over sparsity (optional - uncomment to run)
# =============================================================================
# for C1 in 0.1 0.2 0.3 0.5 0.7; do
#   for C2 in 0.1 0.2 0.3 0.5 0.7; do
#     python scripts/run_cca.py \
#       --x-gene "$X_GENE" \
#       --x-fmri "$X_FMRI" \
#       --method sparse \
#       --n-components "$N_COMPONENTS" \
#       --c1 "$C1" --c2 "$C2" \
#       --n-perm 0 \
#       --out-dir "$OUT_DIR" \
#       --prefix "sparse_c1${C1}_c2${C2}_"
#   done
# done


