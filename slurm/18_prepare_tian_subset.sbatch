#!/bin/bash
#SBATCH --job-name=prep_tian_sub
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: proper environment setup
set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Prepare Tian subset: identify available subjects and subset all data
srun python scripts/prepare_tian_subset.py \
  --tian-ts-dir derived_schaefer_mdd/tian_timeseries \
  --ids-common gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --x-gene-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --cov-age gene-brain-cca-2/derived/interpretable/cov_age.npy \
  --cov-sex gene-brain-cca-2/derived/interpretable/cov_sex.npy \
  --out-dir derived_schaefer_mdd/tian_subset
