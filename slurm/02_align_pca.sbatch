#!/bin/bash
#SBATCH --job-name=align_pca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodelist=node2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -eo pipefail

# Adjust these on the lab server
NESAP="/storage/bigdata/UKB/fMRI/nesap-genomics-allison"
COV_IIDS="$NESAP/iids_labels_covariates/iids.npy"
COV_AGE="$NESAP/iids_labels_covariates/covariates_age.npy"
COV_SEX="$NESAP/iids_labels_covariates/covariates_sex.npy"
COV_VALID="$NESAP/iids_labels_covariates/covariates_valid_mask.npy"

GENE_DIR="$PWD/derived/gene_x"
FMRI_IDS="/storage/bigdata/UKB/fMRI/fmri_eids_180.npy"
FMRI_X="/storage/bigdata/UKB/fMRI/fmri_X_180.npy"
OUT_DIR="$PWD/derived/aligned_pca"

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

python scripts/align_resid_pca.py \
  --ids-gene "$GENE_DIR/ids_gene.npy" \
  --x-gene "$GENE_DIR/X_gene_ng.npy" \
  --ids-fmri "$FMRI_IDS" \
  --x-fmri "$FMRI_X" \
  --cov-iids "$COV_IIDS" \
  --cov-age "$COV_AGE" \
  --cov-sex "$COV_SEX" \
  --cov-valid-mask "$COV_VALID" \
  --labels "$NESAP/iids_labels_covariates/labels.npy" \
  --pca-dim 512 \
  --out-dir "$OUT_DIR"


