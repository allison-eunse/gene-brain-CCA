#!/bin/bash
#SBATCH --job-name=plot_tab_v2
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

srun -n 1 python scripts/plot_tabular_v2_performance.py \
  --smri-full gene-brain-cca-2/derived/coupling_benchmark_smri_tabular_dnabert2_v2/coupling_benchmark_full.json \
  --smri-summary gene-brain-cca-2/derived/coupling_benchmark_smri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --smri-bestfit gene-brain-cca-2/derived/bestfit_smri_tabular_dnabert2_v2/bestfit_results.json \
  --dmri-full gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2/coupling_benchmark_full.json \
  --dmri-summary gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --dmri-bestfit gene-brain-cca-2/derived/bestfit_dmri_tabular_dnabert2_v2/bestfit_results.json \
  --out-dir gene-brain-cca-2/derived/figures_tabular_v2 \
  --top-n 10

srun -n 1 python scripts/write_tabular_v2_report.py \
  --smri-summary gene-brain-cca-2/derived/coupling_benchmark_smri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --smri-bestfit gene-brain-cca-2/derived/bestfit_smri_tabular_dnabert2_v2/bestfit_results.json \
  --dmri-summary gene-brain-cca-2/derived/coupling_benchmark_dmri_tabular_dnabert2_v2/coupling_benchmark_summary.json \
  --dmri-bestfit gene-brain-cca-2/derived/bestfit_dmri_tabular_dnabert2_v2/bestfit_results.json \
  --out-md gene-brain-cca-2/derived/reports/tabular_dnabert2_v2_results.md

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"
