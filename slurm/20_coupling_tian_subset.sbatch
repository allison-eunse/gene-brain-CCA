#!/bin/bash
#SBATCH --job-name=coup_tian
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
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

# Run coupling benchmark on Tian subset (N=746)
# REDUCED GRID: N=746 is too small for high-dim configs
# - No PCA 512 (risk of overfitting)
# - Smaller c1/c2 grid (less combinations)
# Focus: coupling strength (r_holdout), NOT prediction

srun python scripts/run_coupling_benchmark.py \
  --x-gene-wide derived_schaefer_mdd/tian_subset/X_gene_wide_tian_subset.npy \
  --brain-features derived_schaefer_mdd/tian_subset/X_fmri_tian_fc.npy \
  --brain-names tian_fc \
  --cov-age derived_schaefer_mdd/tian_subset/cov_age_tian_subset.npy \
  --cov-sex derived_schaefer_mdd/tian_subset/cov_sex_tian_subset.npy \
  --ids derived_schaefer_mdd/tian_subset/ids_tian_subset.npy \
  --labels derived_schaefer_mdd/tian_subset/labels_tian_subset.npy \
  --gene-pca-dims 64 128 256 \
  --methods cca scca \
  --c1-grid 0.1 0.3 \
  --c2-grid 0.1 0.3 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir derived_schaefer_mdd/tian_subset/coupling_benchmark
