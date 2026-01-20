#!/bin/bash
#SBATCH --job-name=tian_weights_sum
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

srun python scripts/extract_tian_weights.py \
  --coupling-results derived_schaefer_mdd/tian_summary/coupling_benchmark/coupling_benchmark_full.json \
  --x-gene-wide derived_schaefer_mdd/tian_summary/X_gene_wide_tian_summary.npy \
  --x-fmri-fc derived_schaefer_mdd/tian_summary/X_fmri_tian_summary.npy \
  --cov-age derived_schaefer_mdd/tian_summary/cov_age_tian_summary.npy \
  --cov-sex derived_schaefer_mdd/tian_summary/cov_sex_tian_summary.npy \
  --labels derived_schaefer_mdd/tian_summary/labels_tian_summary.npy \
  --out-dir derived_schaefer_mdd/tian_summary/weight_analysis
