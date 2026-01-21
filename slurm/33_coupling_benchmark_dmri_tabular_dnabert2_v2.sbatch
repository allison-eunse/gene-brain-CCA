#!/bin/bash
#SBATCH --job-name=coup_dmri_tab_v2
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Leakage-safe coupling benchmark (v2) for cleaned dMRI features
srun -n 1 python scripts/run_coupling_benchmark_v2.py \
  --x-gene-wide derived_tabular_overlap_v2/dmri/gene_dnabert2/X_gene_wide.npy \
  --brain-features derived_tabular_overlap_v2/dmri/X_dmri_raw.npy \
  --brain-names dmri_tabular_v2 \
  --cov-age derived_tabular_overlap_v2/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/dmri/cov_sex.npy \
  --ids derived_tabular_overlap_v2/dmri/ids_common.npy \
  --labels derived_tabular_overlap_v2/dmri/labels_common.npy \
  --gene-pca-dims 64 128 256 512 \
  --methods cca scca \
  --c1-grid 0.1 0.3 0.5 \
  --c2-grid 0.1 0.3 0.5 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=coup_dmri_tab_v2
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

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

export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Leakage-safe coupling benchmark (v2) for cleaned dMRI (age/sex only)
srun -n 1 python scripts/run_coupling_benchmark_v2.py \
  --x-gene-wide derived_tabular_overlap_v2/dmri/gene_dnabert2/X_gene_wide.npy \
  --brain-features derived_tabular_overlap_v2/dmri/X_dmri_raw.npy \
  --brain-names dmri_tabular_v2 \
  --cov-age derived_tabular_overlap_v2/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/dmri/cov_sex.npy \
  --ids derived_tabular_overlap_v2/dmri/ids_common.npy \
  --labels derived_tabular_overlap_v2/dmri/labels_common.npy \
  --gene-pca-dims 64 128 256 512 \
  --methods cca scca \
  --c1-grid 0.1 0.3 0.5 \
  --c2-grid 0.1 0.3 0.5 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"

