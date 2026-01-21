#!/bin/bash
#SBATCH --job-name=strat_fm
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: run on compute node via Slurm, no manual CUDA_VISIBLE_DEVICES.
# This script runs stratified CCA for a single FM model and modality combination.
# Required environment variables:
#   STRAT_FM_MODEL: dnabert2, evo2, hyenadna, or caduceus
#   STRAT_MODALITY: schaefer7, schaefer17, smri, or dmri

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "Stratified CCA/SCCA Benchmark"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true
env | grep STRAT || true

# Validate required environment variables
if [ -z "${STRAT_FM_MODEL:-}" ] || [ -z "${STRAT_MODALITY:-}" ]; then
    echo "[error] Required environment variables not set:"
    echo "  STRAT_FM_MODEL=${STRAT_FM_MODEL:-<not set>}"
    echo "  STRAT_MODALITY=${STRAT_MODALITY:-<not set>}"
    exit 1
fi

# Avoid CPU oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Conda activation
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Runtime overrides
FM_MODEL="${STRAT_FM_MODEL}"
MODALITY="${STRAT_MODALITY}"
N_PERM="${STRAT_N_PERM:-1000}"
N_FOLDS="${STRAT_N_FOLDS:-5}"
GENE_PCA_DIMS="${STRAT_GENE_PCA_DIMS:-64,128,256,512}"
C_VALUES="${STRAT_C_VALUES:-0.1,0.3,0.5}"
N_COMPONENTS="${STRAT_N_COMPONENTS:-10}"
HOLDOUT_FRAC="${STRAT_HOLDOUT_FRAC:-0.2}"

# Data paths
DATA_DIR="derived_stratified_fm/${FM_MODEL}/${MODALITY}"
OUT_DIR="gene-brain-cca-2/derived/stratified_fm"

# Check data exists
if [ ! -f "${DATA_DIR}/X_gene_wide.npy" ]; then
    echo "[error] Aligned data not found: ${DATA_DIR}"
    echo "Run slurm/50_prepare_all_fm_data.sbatch first."
    exit 1
fi

# Output file (includes FM model and modality)
OUT_COMP="${OUT_DIR}/stratified_comparison_${FM_MODEL}_${MODALITY}.json"
if [ -f "$OUT_COMP" ]; then
    echo "[skip] Results already exist: $OUT_COMP"
    exit 0
fi

mkdir -p "$OUT_DIR"

# Check for extra covariate
EXTRA_ARG=""
if [ -f "${DATA_DIR}/cov_extra.npy" ]; then
    EXTRA_ARG="--cov-extra ${DATA_DIR}/cov_extra.npy"
fi

echo "[run] Stratified CCA/SCCA: ${FM_MODEL} / ${MODALITY}"
echo "  n_perm=$N_PERM, n_folds=$N_FOLDS"
echo "  gene_pca_dims=$GENE_PCA_DIMS"
echo "  c_values=$C_VALUES"

srun -n 1 python scripts/run_stratified_coupling_benchmark.py \
  --x-gene-wide "${DATA_DIR}/X_gene_wide.npy" \
  --x-brain "${DATA_DIR}/X_brain.npy" \
  --labels "${DATA_DIR}/labels.npy" \
  --cov-age "${DATA_DIR}/cov_age.npy" \
  --cov-sex "${DATA_DIR}/cov_sex.npy" \
  --ids "${DATA_DIR}/ids_common.npy" \
  --out-dir "$OUT_DIR" \
  --modality-tag "${FM_MODEL}_${MODALITY}" \
  --holdout-frac "$HOLDOUT_FRAC" \
  --n-folds "$N_FOLDS" \
  --seed 42 \
  --gene-pca-dims "$GENE_PCA_DIMS" \
  --c-values "$C_VALUES" \
  --n-components "$N_COMPONENTS" \
  --n-perm "$N_PERM" \
  $EXTRA_ARG

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"
