#!/bin/bash
#SBATCH --job-name=prep_fm_data
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Lab server compliance: run on compute node via Slurm, no manual CUDA_VISIBLE_DEVICES.
set -euo pipefail

mkdir -p logs

echo "============================================================"
echo "Prepare All FM Data for Stratified CCA"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "============================================================"
env | grep SLURM || true

# Avoid CPU oversubscription
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Conda activation
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Paths
NESAP_ROOT="/storage/bigdata/NESAP"
COV_DIR="/scratch/connectome/leee4321/UKB/iids_labels_covariates"
DERIVED_DIR="derived_stratified_fm"
FM_MODELS=("dnabert2" "evo2" "hyenadna" "caduceus")

# Brain modalities
declare -A BRAIN_DATA
BRAIN_DATA["schaefer7"]="derived_schaefer_mdd/X_fmri_schaefer7_summary.npy"
BRAIN_DATA["schaefer17"]="derived_schaefer_mdd/X_fmri_schaefer17_summary.npy"
BRAIN_DATA["smri"]="derived_tabular_overlap_v2/smri/X_smri_raw.npy"
BRAIN_DATA["dmri"]="derived_tabular_overlap_v2/dmri/X_dmri_raw.npy"

declare -A BRAIN_IDS
BRAIN_IDS["schaefer7"]="derived_schaefer_mdd/ids_schaefer7_summary.npy"
BRAIN_IDS["schaefer17"]="derived_schaefer_mdd/ids_schaefer17_summary.npy"
BRAIN_IDS["smri"]="derived_tabular_overlap_v2/smri/ids_common.npy"
BRAIN_IDS["dmri"]="derived_tabular_overlap_v2/dmri/ids_common.npy"

# Extra covariates (eTIV for sMRI only)
declare -A COV_EXTRA
COV_EXTRA["schaefer7"]=""
COV_EXTRA["schaefer17"]=""
COV_EXTRA["smri"]="derived_tabular_overlap_v2/smri/cov_etiv.npy"
COV_EXTRA["dmri"]=""

mkdir -p "$DERIVED_DIR"

# Step 1: Build wide gene matrices for all FM models
echo ""
echo "============================================================"
echo "Step 1: Building wide gene matrices"
echo "============================================================"

for fm in "${FM_MODELS[@]}"; do
    if [ -f "${DERIVED_DIR}/gene_wide/X_gene_wide_${fm}.npy" ]; then
        echo "[skip] ${fm} gene wide matrix already exists"
    else
        echo "[build] ${fm}..."
        python scripts/build_gene_wide_matrix.py \
            --embedding-root "$NESAP_ROOT" \
            --gene-list "${NESAP_ROOT}/gene_list_filtered.txt" \
            --out-dir "${DERIVED_DIR}/gene_wide" \
            --fm-models "$fm"
    fi
done

# Step 2: Align gene and brain data for each FM x modality combination
echo ""
echo "============================================================"
echo "Step 2: Aligning gene and brain data"
echo "============================================================"

for fm in "${FM_MODELS[@]}"; do
    for modality in schaefer7 schaefer17 smri dmri; do
        out_subdir="${DERIVED_DIR}/${fm}/${modality}"
        
        if [ -f "${out_subdir}/X_gene_wide.npy" ]; then
            echo "[skip] ${fm}/${modality} already aligned"
            continue
        fi
        
        echo "[align] ${fm}/${modality}..."
        
        # Build extra args
        extra_args=""
        if [ -n "${COV_EXTRA[$modality]}" ]; then
            extra_args="--cov-extra ${COV_EXTRA[$modality]}"
        fi
        
        python scripts/align_gene_brain_data.py \
            --gene-wide "${DERIVED_DIR}/gene_wide/X_gene_wide_${fm}.npy" \
            --gene-iids "${COV_DIR}/iids.npy" \
            --brain-data "${BRAIN_DATA[$modality]}" \
            --brain-ids "${BRAIN_IDS[$modality]}" \
            --labels "${COV_DIR}/labels.npy" \
            --label-iids "${COV_DIR}/iids.npy" \
            --cov-age "${COV_DIR}/covariates_age.npy" \
            --cov-sex "${COV_DIR}/covariates_sex.npy" \
            --out-dir "$out_subdir" \
            --modality-tag "${fm}_${modality}" \
            $extra_args
    done
done

echo ""
echo "============================================================"
echo "Data preparation complete!"
echo "Done:  $(date)"
echo "============================================================"

# List what was created
echo ""
echo "Created data directories:"
find "$DERIVED_DIR" -name "meta.json" -exec dirname {} \; | sort
