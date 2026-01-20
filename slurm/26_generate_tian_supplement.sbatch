#!/bin/bash
#SBATCH --job-name=tian_supp
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: proper environment setup
set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

srun python scripts/generate_tian_qc_supplement.py \
  --health-report derived_schaefer_mdd/tian_summary/tian_extraction_health_report.json \
  --overlap-csv derived_schaefer_mdd/tian_summary/tian_mean_fmri_mni_overlap.csv \
  --timepoint-checks derived_schaefer_mdd/tian_summary/tian_mask_timepoint_checks.json \
  --qc-report derived_schaefer_mdd/tian_summary/qc_tian_rois/qc_report.json \
  --coupling-summary derived_schaefer_mdd/tian_summary/coupling_benchmark/coupling_benchmark_summary.json \
  --out-dir derived_schaefer_mdd/tian_summary
