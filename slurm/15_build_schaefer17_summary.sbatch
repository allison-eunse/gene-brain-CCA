#!/bin/bash
#SBATCH --job-name=build_s17
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

srun python scripts/build_schaefer17_summary.py \
  --fmri_ts_dir /storage/bigdata/UKB/fMRI/UKB_ROI \
  --glob "*/schaefer_400Parcels_17Networks_*.npy" \
  --labels_csv /storage/bigdata/UKB/fMRI/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv \
  --ids_keep gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --out_dir derived_schaefer_mdd \
  --tag schaefer17_summary
