#!/bin/bash
#SBATCH --job-name=coupling_bench
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Run coupling benchmark for all brain feature sets
# Grid: gene_pca_dim x (CCA, SCCA) x brain_feature_set
srun python scripts/run_coupling_benchmark.py \
  --x-gene-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
  --brain-features derived_schaefer_mdd/X_fmri_schaefer7_summary.npy \
                   derived_schaefer_mdd/X_fmri_schaefer17_summary.npy \
  --brain-names schaefer7_summary schaefer17_summary \
  --cov-age gene-brain-cca-2/derived/interpretable/cov_age.npy \
  --cov-sex gene-brain-cca-2/derived/interpretable/cov_sex.npy \
  --ids gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --gene-pca-dims 64 128 256 512 \
  --methods cca scca \
  --c1-grid 0.1 0.3 0.5 \
  --c2-grid 0.1 0.3 0.5 \
  --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --out-dir gene-brain-cca-2/derived/coupling_benchmark
