#!/bin/bash
#SBATCH --job-name=fmri_fc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Adjust these on the lab server
ROOT_FMRI="/path/to/UKB_ROI"
GLOB="*_20227_2_0/hcp_mmp1_*.npy"
OUT_DIR="$PWD/derived/fmri_fc"

mkdir -p logs "$OUT_DIR"

source ~/.bashrc || true
conda activate <YOUR_ENV>

python scripts/build_x_fmri_fc.py \
  --root "$ROOT_FMRI" \
  --glob "$GLOB" \
  --out-dir "$OUT_DIR"


