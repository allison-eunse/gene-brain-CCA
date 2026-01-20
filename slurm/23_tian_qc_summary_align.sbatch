#!/bin/bash
#SBATCH --job-name=tian_qc_sum
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: proper environment setup
set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Step 2: QC ROIs and subjects
srun python scripts/qc_tian_rois.py \
  --roi-root derived_schaefer_mdd/tian_timeseries \
  --ids-subset derived_schaefer_mdd/tian_subset/ids_tian_valid.npy \
  --out-dir derived_schaefer_mdd/tian_summary/qc_tian_rois \
  --std-threshold 1e-6 \
  --roi-validity-threshold 0.70 \
  --subject-dropout-threshold 0.40

# Step 3: Build Tian summary FC (7 groups -> 21 edges)
srun python scripts/build_tian_summary_fc.py \
  --roi-root derived_schaefer_mdd/tian_timeseries \
  --kept-rois derived_schaefer_mdd/tian_summary/qc_tian_rois/kept_roi_indices.npy \
  --kept-subjects derived_schaefer_mdd/tian_summary/qc_tian_rois/kept_subject_ids.npy \
  --out-dir derived_schaefer_mdd/tian_summary \
  --std-threshold 1e-6 \
  --group-valid-min-fraction 0.5

# Step 4: Align gene/covariates/labels to summary subjects
srun python scripts/align_tian_summary_data.py \
  --ids-keep derived_schaefer_mdd/tian_summary/ids_tian_summary.npy \
  --ids-full derived_schaefer_mdd/tian_subset/ids_tian_valid.npy \
  --x-gene-wide derived_schaefer_mdd/tian_subset/X_gene_wide_tian_subset.npy \
  --cov-age derived_schaefer_mdd/tian_subset/cov_age_tian_subset.npy \
  --cov-sex derived_schaefer_mdd/tian_subset/cov_sex_tian_subset.npy \
  --labels derived_schaefer_mdd/tian_subset/labels_tian_subset.npy \
  --out-dir derived_schaefer_mdd/tian_summary
