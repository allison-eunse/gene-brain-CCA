#!/bin/bash
#SBATCH --job-name=strat_sch7
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Lab server compliance: run on compute node via Slurm, no manual CUDA_VISIBLE_DEVICES.
set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true
env | grep STRAT || true

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

# Runtime overrides (from environment)
FM_MODEL="${STRAT_FM_MODEL:-dnabert2}"
N_PERM="${STRAT_N_PERM:-1000}"
N_FOLDS="${STRAT_N_FOLDS:-5}"
GENE_PCA_DIMS="${STRAT_GENE_PCA_DIMS:-64,128,256,512}"
C_VALUES="${STRAT_C_VALUES:-0.1,0.3,0.5}"
N_COMPONENTS="${STRAT_N_COMPONENTS:-10}"
HOLDOUT_FRAC="${STRAT_HOLDOUT_FRAC:-0.2}"

# Data paths
GENE_WIDE="derived_gene_wide/X_gene_wide_${FM_MODEL}.npy"
COV_DIR="/scratch/connectome/leee4321/UKB/iids_labels_covariates"
FMRI_DIR="derived_schaefer_mdd"
OUT_DIR="gene-brain-cca-2/derived"

# Check gene embedding exists
if [ ! -f "$GENE_WIDE" ]; then
    echo "[error] Gene embedding not found: $GENE_WIDE"
    echo "Run slurm/50_build_gene_wide.sbatch first."
    exit 1
fi

# Output file (includes FM model name)
OUT_COMP="${OUT_DIR}/stratified_comparison_schaefer7_${FM_MODEL}.json"
if [ -f "$OUT_COMP" ]; then
    echo "[skip] Results already exist: $OUT_COMP"
    exit 0
fi

echo "[run] Stratified CCA/SCCA on Schaefer7 with ${FM_MODEL}..."
srun -n 1 python scripts/run_stratified_coupling_benchmark.py \
  --x-gene-wide "$GENE_WIDE" \
  --x-brain "${FMRI_DIR}/X_fmri_schaefer7_summary.npy" \
  --labels "${COV_DIR}/labels.npy" \
  --cov-age "${COV_DIR}/covariates_age.npy" \
  --cov-sex "${COV_DIR}/covariates_sex.npy" \
  --ids "${COV_DIR}/iids.npy" \
  --out-dir "$OUT_DIR" \
  --modality-tag "schaefer7_${FM_MODEL}" \
  --holdout-frac "$HOLDOUT_FRAC" \
  --n-folds "$N_FOLDS" \
  --seed 42 \
  --gene-pca-dims "$GENE_PCA_DIMS" \
  --c-values "$C_VALUES" \
  --n-components "$N_COMPONENTS" \
  --n-perm "$N_PERM"

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"
