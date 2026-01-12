#!/bin/bash
#SBATCH --job-name=cca_perm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

ALIGNED="$PWD/derived/aligned_pca"
OUT_DIR="$PWD/derived/cca"

mkdir -p logs "$OUT_DIR"

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

INFO_JSON="$ALIGNED/pca_info.json"
GENE_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['gene_pca_dim'])" "$INFO_JSON")"
FMRI_PCA_DIM="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fmri_pca_dim'])" "$INFO_JSON")"
X_GENE="$ALIGNED/X_gene_pca${GENE_PCA_DIM}.npy"
X_FMRI="$ALIGNED/X_fmri_pca${FMRI_PCA_DIM}.npy"

python scripts/run_cca_permute.py \
  --x-gene "$X_GENE" \
  --x-fmri "$X_FMRI" \
  --ids "$ALIGNED/ids_common.npy" \
  --n-components 3 \
  --n-perm 1000 \
  --seed 42 \
  --out-dir "$OUT_DIR"


