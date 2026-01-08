#!/bin/bash
#SBATCH --job-name=align_pca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Adjust these on the lab server
NESAP="/path/to/nesap-genomics-allison"
COV_IIDS="$NESAP/iids_labels_covariates/iids.npy"
COV_AGE="$NESAP/iids_labels_covariates/covariates_age.npy"
COV_SEX="$NESAP/iids_labels_covariates/covariates_sex.npy"
COV_VALID="$NESAP/iids_labels_covariates/covariates_valid_mask.npy"

GENE_DIR="$PWD/derived/gene_x"
FMRI_DIR="$PWD/derived/fmri_fc"
OUT_DIR="$PWD/derived/aligned_pca"

mkdir -p logs "$OUT_DIR"

source ~/.bashrc || true
conda activate <YOUR_ENV>

python scripts/align_resid_pca.py \
  --ids-gene "$GENE_DIR/ids_gene.npy" \
  --x-gene "$GENE_DIR/X_gene_ng.npy" \
  --ids-fmri "$FMRI_DIR/ids_fmri.npy" \
  --x-fmri "$FMRI_DIR/X_fmri_fc.npy" \
  --cov-iids "$COV_IIDS" \
  --cov-age "$COV_AGE" \
  --cov-sex "$COV_SEX" \
  --cov-valid-mask "$COV_VALID" \
  --pca-dim 512 \
  --out-dir "$OUT_DIR"


