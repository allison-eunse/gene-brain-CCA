#!/bin/bash
#SBATCH --job-name=sch7_summary
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o logs/sch7_summary_%j.out
#SBATCH -e logs/sch7_summary_%j.err

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

python scripts/build_schaefer7_summary.py \
  --fmri_ts_dir /storage/bigdata/UKB/fMRI/UKB_ROI \
  --glob "*/schaefer_400Parcels_17Networks_*.npy" \
  --labels_csv /storage/bigdata/UKB/fMRI/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv \
  --ids_keep gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --out_dir derived_schaefer_mdd \
  --tag schaefer7_summary

