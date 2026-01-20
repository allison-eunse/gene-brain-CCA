#!/bin/bash
#SBATCH --job-name=build_tian_fc
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
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

# Build Tian FC for the subset of subjects with available timeseries
# Uses ids_tian_subset.npy from Phase 1 (prepare_tian_subset.py)
srun python scripts/build_tian_fc_subset.py \
  --roi-root derived_schaefer_mdd/tian_timeseries \
  --ids-subset derived_schaefer_mdd/tian_subset/ids_tian_subset.npy \
  --out-dir derived_schaefer_mdd/tian_subset \
  --tag tian_fc
