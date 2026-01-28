#!/bin/bash
#SBATCH --job-name=pred_smri_dmri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --partition=debug
#SBATCH --chdir=/storage/bigdata/UKB/fMRI/gene-brain-CCA
#SBATCH --exclude=node1,node3

set -euo pipefail
mkdir -p logs gene-brain-cca-2/derived/wide_gene

# Avoid CPU oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Conda env (match other jobs)
export PS1="${PS1:-}"
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# sMRI with eTIV covariate
echo "[run] sMRI Prediction..."
srun python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene-wide derived_tabular_overlap_v2/smri/gene_dnabert2/X_gene_wide.npy \
  --x-fmri-raw derived_tabular_overlap_v2/smri/X_smri_raw.npy \
  --cov-age derived_tabular_overlap_v2/smri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/smri/cov_sex.npy \
  --cov-extra derived_tabular_overlap_v2/smri/cov_etiv.npy \
  --labels derived_tabular_overlap_v2/smri/labels_common.npy \
  --ids derived_tabular_overlap_v2/smri/ids_common.npy \
  --holdout-frac 0.2 --pca-components 512 \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/wide_gene/predictive_suite_results_smri_tabular.json

# dMRI (no extra covariates)
echo "[run] dMRI Prediction..."
srun python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene-wide derived_tabular_overlap_v2/dmri/gene_dnabert2/X_gene_wide.npy \
  --x-fmri-raw derived_tabular_overlap_v2/dmri/X_dmri_raw.npy \
  --cov-age derived_tabular_overlap_v2/dmri/cov_age.npy \
  --cov-sex derived_tabular_overlap_v2/dmri/cov_sex.npy \
  --labels derived_tabular_overlap_v2/dmri/labels_common.npy \
  --ids derived_tabular_overlap_v2/dmri/ids_common.npy \
  --holdout-frac 0.2 --pca-components 512 \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/wide_gene/predictive_suite_results_dmri_tabular.json

echo "[done] All predictions complete."
