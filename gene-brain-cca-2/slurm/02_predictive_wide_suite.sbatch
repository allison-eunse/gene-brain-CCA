#!/bin/bash
#SBATCH --job-name=wide_suite
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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
INTERP_DIR="$OUT_ROOT/interpretable"
OUT="$OUT_ROOT/wide_gene"
mkdir -p "$OUT"

# Assumes ids_common from Pipeline A for overlap
IDS="$INTERP_DIR/ids_common.npy"
if [[ ! -f "$IDS" ]]; then
  echo "[ERROR] Missing $IDS. Run Pipeline A (01_interpretable_scca.sbatch) first." >&2
  exit 1
fi

srun -n 1 python gene-brain-cca-2/scripts/build_x_gene_wide.py \
  --embed-root /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged \
  --gene-list /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt \
  --iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --ids-keep "$IDS" \
  --n-files 49 \
  --out-dir "$OUT"

srun -n 1 python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene-wide "$OUT/X_gene_wide.npy" \
  --x-fmri-raw "$INTERP_DIR/X_fmri_raw.npy" \
  --cov-age "$INTERP_DIR/cov_age.npy" \
  --cov-sex "$INTERP_DIR/cov_sex.npy" \
  --ids "$INTERP_DIR/ids_common.npy" \
  --labels "$INTERP_DIR/labels_common.npy" \
  --holdout-frac 0.2 \
  --pca-components 512 \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --seed 42 \
  --out-json "$OUT/predictive_suite_results.json"

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"