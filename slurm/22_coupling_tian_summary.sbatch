#!/bin/bash
#SBATCH --job-name=coup_tian_sum
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

# NOTE: DO NOT set CUDA_VISIBLE_DEVICES manually (Slurm handles GPU allocation)

srun python scripts/run_coupling_benchmark.py \
  --x-gene-wide derived_schaefer_mdd/tian_summary/X_gene_wide_tian_summary.npy \
  --brain-features derived_schaefer_mdd/tian_summary/X_fmri_tian_summary.npy \
  --brain-names tian_summary \
  --cov-age derived_schaefer_mdd/tian_summary/cov_age_tian_summary.npy \
  --cov-sex derived_schaefer_mdd/tian_summary/cov_sex_tian_summary.npy \
  --ids derived_schaefer_mdd/tian_summary/ids_tian_summary.npy \
  --labels derived_schaefer_mdd/tian_summary/labels_tian_summary.npy \
  --gene-pca-dims 128 \
  --methods scca \
  --c1-grid 0.1 \
  --c2-grid 0.1 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir derived_schaefer_mdd/tian_summary/coupling_benchmark
