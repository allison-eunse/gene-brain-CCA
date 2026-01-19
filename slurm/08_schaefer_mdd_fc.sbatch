#!/bin/bash
#SBATCH --job-name=schaefer_mdd_fc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# NOTE (lab server rule): Do NOT set CUDA_VISIBLE_DEVICES manually.
# This job is CPU-only.

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Some conda init scripts reference $PS1; under `set -u` it can error in non-interactive shells.
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

mkdir -p logs derived_schaefer_mdd

python scripts/build_x_fmri_schaefer_network_fc.py \
  --root /storage/bigdata/UKB/fMRI/UKB_ROI \
  --glob "*/schaefer_400Parcels_17Networks_*.npy" \
  --labels /storage/bigdata/UKB/fMRI/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv \
  --networks DefaultA,DefaultB,DefaultC,SalVentAttnA,SalVentAttnB,LimbicA,LimbicB \
  --ids-keep gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --out-dir derived_schaefer_mdd \
  --tag schaefer17_DMN_Sal_Limbic

# Basic output sanity checks
python - <<'PY'
import json, numpy as np
meta = json.load(open('derived_schaefer_mdd/meta_schaefer17_DMN_Sal_Limbic.json'))
X = np.load('derived_schaefer_mdd/X_fmri_schaefer_schaefer17_DMN_Sal_Limbic.npy', mmap_mode='r')
ids = np.load('derived_schaefer_mdd/ids_schaefer17_DMN_Sal_Limbic.npy', allow_pickle=True)
print('[check] meta:', meta)
print('[check] X:', X.shape, X.dtype)
print('[check] ids:', ids.shape, ids.dtype)
PY
