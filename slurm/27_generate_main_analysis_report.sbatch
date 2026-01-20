#!/bin/bash
#SBATCH --job-name=main_report
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

srun python scripts/generate_main_analysis_report.py \
  --coupling-summary gene-brain-cca-2/derived/coupling_benchmark/coupling_benchmark_summary.json \
  --interpretable-results gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json \
  --tian-coupling derived_schaefer_mdd/tian_summary/coupling_benchmark/coupling_benchmark_summary.json \
  --out-dir final_report
