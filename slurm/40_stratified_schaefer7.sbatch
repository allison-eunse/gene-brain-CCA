#!/bin/bash
#SBATCH --job-name=strat_sch7
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

# Step 1: Prepare Schaefer7 overlap (if not already done)
OVERLAP_DIR="derived_schaefer_mdd/schaefer7_overlap"
if [ ! -f "${OVERLAP_DIR}/X_fmri.npy" ]; then
    echo "[prep] Creating Schaefer7 overlap..."
    srun -n 1 python scripts/prepare_schaefer_gene_overlap.py \
      --schaefer-fmri derived_schaefer_mdd/X_fmri_schaefer7_summary.npy \
      --schaefer-meta derived_schaefer_mdd/meta_schaefer7_summary.json \
      --gene-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
      --gene-ids gene-brain-cca-2/derived/wide_gene/ids_gene_overlap.npy \
      --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
      --cov-age gene-brain-cca-2/derived/interpretable/cov_age.npy \
      --cov-sex gene-brain-cca-2/derived/interpretable/cov_sex.npy \
      --ids-common gene-brain-cca-2/derived/interpretable/ids_common.npy \
      --out-dir "${OVERLAP_DIR}"
else
    echo "[skip] Schaefer7 overlap already exists"
fi

# Step 2: Run stratified benchmark
OUT_COMP="gene-brain-cca-2/derived/stratified_comparison_schaefer7.json"
if [ -f "$OUT_COMP" ]; then
    echo "[skip] Results already exist: $OUT_COMP"
    exit 0
fi

echo "[run] Stratified CCA/SCCA on Schaefer7..."
srun -n 1 python scripts/run_stratified_coupling_benchmark.py \
  --x-gene-wide "${OVERLAP_DIR}/X_gene_wide.npy" \
  --x-brain "${OVERLAP_DIR}/X_fmri.npy" \
  --labels "${OVERLAP_DIR}/labels.npy" \
  --cov-age "${OVERLAP_DIR}/cov_age.npy" \
  --cov-sex "${OVERLAP_DIR}/cov_sex.npy" \
  --ids "${OVERLAP_DIR}/ids.npy" \
  --out-dir gene-brain-cca-2/derived \
  --modality-tag schaefer7 \
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
