#!/bin/bash
#SBATCH --job-name=interp_scca
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=node1,node3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

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

# Set dummy PS1 to avoid "unbound variable" error from conda.sh with set -u
export PS1="${PS1:-}"
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Tip: For faster I/O, you can run with:
#   OUT_ROOT=/scratch/connectome/$USER/gene-brain-cca-2/derived sbatch ...
OUT_ROOT="${OUT_ROOT:-gene-brain-cca-2/derived}"
OUT="$OUT_ROOT/interpretable"
mkdir -p "$OUT"

srun -n 1 python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py \
  --ids-gene /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/ids_gene.npy \
  --x-gene   /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/X_gene_ng.npy \
  --ids-fmri /storage/bigdata/UKB/fMRI/fmri_eids_180.npy \
  --x-fmri   /storage/bigdata/UKB/fMRI/fmri_X_180.npy \
  --cov-iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --cov-age  /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_age.npy \
  --cov-sex  /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_sex.npy \
  --cov-valid-mask /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_valid_mask.npy \
  --labels   /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/labels.npy \
  --out-dir  "$OUT"

srun -n 1 python gene-brain-cca-2/scripts/run_scca_interpretable.py \
  --x-gene-raw "$OUT/X_gene_raw.npy" \
  --x-fmri-raw "$OUT/X_fmri_raw.npy" \
  --cov-age "$OUT/cov_age.npy" \
  --cov-sex "$OUT/cov_sex.npy" \
  --labels "$OUT/labels_common.npy" \
  --ids "$OUT/ids_common.npy" \
  --gene-names /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/genes.npy \
  --c1 0.3 --c2 0.3 --k 10 --n-folds 5 --seed 42 \
  --out-json "$OUT/scca_interpretable_results.json"

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"