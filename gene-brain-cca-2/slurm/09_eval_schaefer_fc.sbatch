#!/bin/bash
# Schaefer (DMN+Salience+Limbic) fMRI evaluation vs existing pipeline
# - Uses leakage-safe path (--x-gene-wide, --x-fmri-raw, covariates)
# - No CUDA env overrides; CPU-only
# - Respects lab rule: run via Slurm (no login-node heavy compute)

#SBATCH --job-name=eval_schaefer_fc
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Avoid PS1 issues under set -u
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2
mkdir -p logs

python scripts/run_predictive_suite.py \
  --x-gene-wide derived/wide_gene/X_gene_wide.npy \
  --x-fmri-raw /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_schaefer_mdd/X_fmri_schaefer_schaefer17_DMN_Sal_Limbic.npy \
  --cov-age derived/interpretable/cov_age.npy \
  --cov-sex derived/interpretable/cov_sex.npy \
  --labels derived/interpretable/labels_common.npy \
  --ids derived/interpretable/ids_common.npy \
  --k 10 \
  --pca-components 512 \
  --n-folds 5 \
  --holdout-frac 0.2 \
  --seed 42 \
  --out-json derived/wide_gene/predictive_suite_results_schaefer_fc.json

