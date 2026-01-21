#!/bin/bash
#SBATCH --job-name=coup_dmri_tab
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: run on compute node via Slurm, no manual CUDA_VISIBLE_DEVICES.
set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true

# Avoid CPU oversubscription from MKL/OpenBLAS in numpy/sklearn
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Conda activation (lab-standard)
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

# Real-time logging
export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Leakage-safe coupling benchmark (CCA + SCCA) for tabular dMRI features
srun -n 1 python scripts/run_coupling_benchmark.py \
  --x-gene-wide derived_tabular_overlap/dmri/gene_dnabert2/X_gene_wide.npy \
  --brain-features derived_tabular_overlap/dmri/X_dmri_raw.npy \
  --brain-names dmri_tabular \
  --cov-age derived_tabular_overlap/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap/dmri/cov_sex.npy \
  --ids derived_tabular_overlap/dmri/ids_common.npy \
  --labels derived_tabular_overlap/dmri/labels_common.npy \
  --gene-pca-dims 64 128 256 512 \
  --methods cca scca \
  --c1-grid 0.1 0.3 0.5 \
  --c2-grid 0.1 0.3 0.5 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"

