#!/bin/bash
#SBATCH --job-name=eval_sch17_dsl
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/eval_sch17_dsl_%j.out
#SBATCH -e logs/eval_sch17_dsl_%j.err

set -euo pipefail
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
  --x-fmri-raw derived_schaefer_mdd/X_fmri_schaefer_schaefer17_DMN_Sal_Limbic.npy \
  --cov-age gene-brain-cca-2/derived/interpretable/cov_age.npy \
  --cov-sex gene-brain-cca-2/derived/interpretable/cov_sex.npy \
  --ids gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --holdout-frac 0.2 --seed 42 \
  --pca-components 512 \
  --out-json gene-brain-cca-2/derived/wide_gene/predictive_suite_results_schaefer17_dmn_sal_limbic.json

