#!/bin/bash
#SBATCH --job-name=tian_extract
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs/tian_extract_%A_%a.out
#SBATCH -e logs/tian_extract_%A_%a.err
#SBATCH --array=1-1000

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI
OFFSET=${OFFSET:-0}
LINE=$((SLURM_ARRAY_TASK_ID + OFFSET))
SUBJ=$(sed -n "${LINE}p" /storage/bigdata/UKB/fMRI/gene-brain-CCA/tian_subjects.txt)
if [ -z "$SUBJ" ]; then
  echo "[warn] empty subject at line ${SLURM_ARRAY_TASK_ID}" >&2
  exit 0
fi
python extract_tian_roi.py --subject "$SUBJ" --fmri-dir /storage/bigdata/UKB/fMRI/UKB_20227_1_recon --atlas-file /storage/bigdata/UKB/fMRI/tian_atlas/Tian_Subcortex_S3_3T.nii --output-dir /storage/bigdata/UKB/fMRI/UKB_ROI --quiet
