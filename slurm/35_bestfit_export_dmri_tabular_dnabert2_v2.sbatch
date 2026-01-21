#!/bin/bash
#SBATCH --job-name=best_dmri_tab_v2
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

# Export best-fit artifacts for dMRI tabular coupling (v2 cleaned)
srun -n 1 python scripts/fit_best_coupling_and_export.py \
  --summary-json gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --x-gene-wide derived_tabular_overlap_v2/dmri/gene_dnabert2/X_gene_wide.npy \
  --x-brain derived_tabular_overlap_v2/dmri/X_dmri_raw.npy \
  --cov-age derived_tabular_overlap_v2/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/dmri/cov_sex.npy \
  --labels derived_tabular_overlap_v2/dmri/labels_common.npy \
  --ids derived_tabular_overlap_v2/dmri/ids_common.npy \
  --brain-names derived_tabular_overlap_v2/dmri/dmri_feature_names.txt \
  --gene-names derived_max_pooling/gene_x/genes.npy \
  --k 10 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/bestfit_dmri_tabular_dnabert2_v2

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=best_dmri_tab_v2
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

# Export best-fit artifacts for dMRI v2 (age/sex only)
srun -n 1 python scripts/fit_best_coupling_and_export.py \
  --summary-json gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --x-gene-wide derived_tabular_overlap_v2/dmri/gene_dnabert2/X_gene_wide.npy \
  --x-brain derived_tabular_overlap_v2/dmri/X_dmri_raw.npy \
  --cov-age derived_tabular_overlap_v2/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/dmri/cov_sex.npy \
  --labels derived_tabular_overlap_v2/dmri/labels_common.npy \
  --ids derived_tabular_overlap_v2/dmri/ids_common.npy \
  --brain-names derived_tabular_overlap_v2/dmri/dmri_feature_names.txt \
  --gene-names derived_max_pooling/gene_x/genes.npy \
  --k 10 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/bestfit_dmri_tabular_dnabert2_v2

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"

