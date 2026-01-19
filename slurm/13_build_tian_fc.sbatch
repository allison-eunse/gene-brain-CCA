#!/bin/bash
#SBATCH --job-name=build_tian_fc
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/build_tian_fc_%j.out
#SBATCH -e logs/build_tian_fc_%j.err

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

python scripts/build_tian_fc.py \
  --roi-root /storage/bigdata/UKB/fMRI/UKB_ROI \
  --ids-keep gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --out-dir derived_schaefer_mdd \
  --tag tian_fc

