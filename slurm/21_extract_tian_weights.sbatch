#!/bin/bash
#SBATCH --job-name=tian_weights
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

# Extract and analyze CCA/SCCA weights for Tian subcortical ROIs
# Purpose: Mechanistic validation that hippocampus/amygdala have high weights
# (aligning with GDNF expression profiles)

srun python scripts/extract_tian_weights.py \
  --coupling-results derived_schaefer_mdd/tian_subset/coupling_benchmark/coupling_benchmark_full.json \
  --x-gene-wide derived_schaefer_mdd/tian_subset/X_gene_wide_tian_subset.npy \
  --x-fmri-fc derived_schaefer_mdd/tian_subset/X_fmri_tian_fc.npy \
  --cov-age derived_schaefer_mdd/tian_subset/cov_age_tian_subset.npy \
  --cov-sex derived_schaefer_mdd/tian_subset/cov_sex_tian_subset.npy \
  --labels derived_schaefer_mdd/tian_subset/labels_tian_subset.npy \
  --out-dir derived_schaefer_mdd/tian_subset/weight_analysis
