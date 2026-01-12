#!/bin/bash
#SBATCH --job-name=fmri_fc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Adjust these on the lab server
ROOT_FMRI="/path/to/UKB_ROI"
GLOB="*_20227_2_0/hcp_mmp1_*.npy"
OUT_DIR="$PWD/derived/fmri_fc"

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

python scripts/build_x_fmri_fc.py \
  --root "$ROOT_FMRI" \
  --glob "$GLOB" \
  --out-dir "$OUT_DIR"


