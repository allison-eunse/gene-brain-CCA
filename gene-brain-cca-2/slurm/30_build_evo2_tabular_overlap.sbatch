#!/bin/bash
#SBATCH --job-name=tabular_evo2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=node1,node3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

mkdir -p logs
echo "============================================================"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true

# Avoid CPU oversubscription from MKL/OpenBLAS in numpy
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Set dummy PS1 to avoid "unbound variable" error from conda.sh with set -u
export PS1="${PS1:-}"
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

EMBED_ROOT=/storage/bigdata/NESAP/Evo2_embedding
GENE_LIST=/storage/bigdata/NESAP/gene_list_filtered.txt
IIDS=/storage/bigdata/NESAP/iids.npy

for modality in smri dmri; do
  OUT_DIR=/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_tabular_overlap/${modality}/gene_evo2
  IDS_KEEP=/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_tabular_overlap/${modality}/ids_common.npy
  srun -n 1 python gene-brain-cca-2/scripts/build_x_gene_wide_model.py \
    --embed-root "$EMBED_ROOT" \
    --gene-list "$GENE_LIST" \
    --iids "$IIDS" \
    --ids-keep "$IDS_KEEP" \
    --out-dir "$OUT_DIR" \
    --model evo2 \
    --use-glob
done

echo "============================================================"
echo "Done:  $(date)"
echo "============================================================"
