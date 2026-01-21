#!/bin/bash
#SBATCH --job-name=strat_dmri
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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

# Conda activation (lab server compliant)
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Runtime overrides (for retry / debug)
STRAT_N_PERM="${STRAT_N_PERM:-1000}"
STRAT_N_FOLDS="${STRAT_N_FOLDS:-5}"
STRAT_GENE_PCA_DIMS="${STRAT_GENE_PCA_DIMS:-64,128,256,512}"
STRAT_C_VALUES="${STRAT_C_VALUES:-0.1,0.3,0.5}"
STRAT_N_COMPONENTS="${STRAT_N_COMPONENTS:-10}"
STRAT_HOLDOUT_FRAC="${STRAT_HOLDOUT_FRAC:-0.2}"

# Data paths for dMRI tabular v2 (already prepared with overlap)
DATA_DIR="derived_tabular_overlap_v2/dmri"

OUT_COMP="gene-brain-cca-2/derived/stratified_comparison_dmri_tabular.json"
if [ -f "$OUT_COMP" ]; then
    echo "[skip] Results already exist: $OUT_COMP"
    exit 0
fi

echo "[run] Stratified CCA/SCCA on dMRI tabular..."
srun -n 1 python scripts/run_stratified_coupling_benchmark.py \
  --x-gene-wide "${DATA_DIR}/gene_dnabert2/X_gene_wide.npy" \
  --x-brain "${DATA_DIR}/X_dmri_raw.npy" \
  --labels "${DATA_DIR}/labels_common.npy" \
  --cov-age "${DATA_DIR}/cov_age.npy" \
  --cov-sex "${DATA_DIR}/cov_sex.npy" \
  --ids "${DATA_DIR}/ids_common.npy" \
  --out-dir gene-brain-cca-2/derived \
  --modality-tag dmri_tabular \
  --holdout-frac "$STRAT_HOLDOUT_FRAC" \
  --n-folds "$STRAT_N_FOLDS" \
  --seed 42 \
  --gene-pca-dims "$STRAT_GENE_PCA_DIMS" \
  --c-values "$STRAT_C_VALUES" \
  --n-components "$STRAT_N_COMPONENTS" \
  --n-perm "$STRAT_N_PERM"

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"
