#!/bin/bash
#SBATCH --job-name=tian_extract
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/tian_extract_%A_%a.out
#SBATCH --error=logs/tian_extract_%A_%a.err
#SBATCH --array=1-1000
# NOTE: No --gres=gpu needed - this is CPU-only (NIfTI loading + mean computation)

# Lab server compliance: proper environment setup
set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI

# Get subject from list
OFFSET=${OFFSET:-0}
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET))
SUBJ=$(sed -n "${LINE}p" /storage/bigdata/UKB/fMRI/gene-brain-CCA/tian_subjects.txt)
if [ -z "$SUBJ" ]; then
  echo "[warn] empty subject at line ${SLURM_ARRAY_TASK_ID}" >&2
  exit 0
fi

# Use writable output directory (derived_schaefer_mdd/tian_timeseries) instead of UKB_ROI
OUTPUT_DIR=/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_schaefer_mdd/tian_timeseries

# Copy atlas to /dev/shm for faster repeated access (per lab server guidelines)
ATLAS_SRC=/storage/bigdata/UKB/fMRI/tian_atlas/Tian_Subcortex_S3_3T.nii
ATLAS_SHM=/dev/shm/tian_atlas_${SLURM_ARRAY_TASK_ID}.nii
cp "$ATLAS_SRC" "$ATLAS_SHM"

# Use srun for proper process tracking (per plan compliance)
srun python extract_tian_roi.py \
  --subject "$SUBJ" \
  --fmri-dir /storage/bigdata/UKB/fMRI/UKB_20227_1_recon \
  --atlas-file "$ATLAS_SHM" \
  --output-dir "$OUTPUT_DIR" \
  --quiet

# Cleanup /dev/shm (per lab server guidelines)
rm -f "$ATLAS_SHM"
