#!/bin/bash
# Submit all stratified CCA jobs for multiple gene FM models and brain modalities
# Run this inside tmux: tmux new -s stratified_cca
#
# This script:
# 1. First runs data preparation (builds gene matrices, aligns with brain data)
# 2. Then submits stratified CCA jobs for all FM x modality combinations
#
# Fixed bugs from v1:
# - Commas in --export values were being parsed incorrectly by Slurm
# - Now exports variables before sbatch call
# - Handles missing sacct by checking output files

set -e
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
mkdir -p logs

echo "============================================================"
echo "Stratified CCA Pipeline (v2.1)"
echo "Time: $(date)"
echo "============================================================"

# Default parameters
N_PERM="${STRAT_N_PERM:-1000}"
N_FOLDS="${STRAT_N_FOLDS:-5}"
GENE_PCA_DIMS="${STRAT_GENE_PCA_DIMS:-64,128,256,512}"
C_VALUES="${STRAT_C_VALUES:-0.1,0.3,0.5}"
N_COMPONENTS="${STRAT_N_COMPONENTS:-10}"
HOLDOUT_FRAC="${STRAT_HOLDOUT_FRAC:-0.2}"

# FM models and modalities to test
FM_MODELS=("dnabert2" "evo2" "hyenadna" "caduceus")
MODALITIES=("schaefer7" "schaefer17" "smri" "dmri")

echo ""
echo "Parameters:"
echo "  N_PERM=$N_PERM"
echo "  N_FOLDS=$N_FOLDS"
echo "  GENE_PCA_DIMS=$GENE_PCA_DIMS"
echo "  C_VALUES=$C_VALUES"
echo "  N_COMPONENTS=$N_COMPONENTS"
echo "  HOLDOUT_FRAC=$HOLDOUT_FRAC"
echo ""
echo "FM Models: ${FM_MODELS[*]}"
echo "Modalities: ${MODALITIES[*]}"

# ============================================================
# Step 1: Data Preparation
# ============================================================
echo ""
echo "============================================================"
echo "Step 1: Data Preparation"
echo "============================================================"

# Check if data already prepared
PREP_DONE=true
for fm in "${FM_MODELS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
        if [ ! -f "derived_stratified_fm/${fm}/${mod}/X_gene_wide.npy" ]; then
            PREP_DONE=false
            break 2
        fi
    done
done

if [ "$PREP_DONE" = true ]; then
    echo "[skip] All data already prepared"
else
    echo "[run] Submitting data preparation job..."
    PREP_JOB=$(sbatch --parsable --export=ALL slurm/50_prepare_all_fm_data.sbatch)
    echo "  Prep JobID: $PREP_JOB"
    
    echo "Waiting for data preparation to complete..."
    while true; do
        # Check if running
        IS_RUNNING=$(squeue -j "$PREP_JOB" 2>/dev/null | grep -v JOBID | wc -l || echo 0)
        
        if [ "$IS_RUNNING" -eq 0 ]; then
            # Job finished (success or fail). Check output files.
            ALL_FILES_EXIST=true
            for fm in "${FM_MODELS[@]}"; do
                for mod in "${MODALITIES[@]}"; do
                    if [ ! -f "derived_stratified_fm/${fm}/${mod}/X_gene_wide.npy" ]; then
                        ALL_FILES_EXIST=false
                        break 2
                    fi
                done
            done
            
            if [ "$ALL_FILES_EXIST" = true ]; then
                echo "Data preparation completed successfully!"
                break
            else
                echo "[error] Data preparation finished but output files are missing."
                echo "Check logs/prep_fm_data-${PREP_JOB}.out and .err"
                exit 1
            fi
        fi
        
        echo "[$(date +%H:%M:%S)] Prep job running..."
        sleep 30
    done
fi

# ============================================================
# Step 2: Submit Stratified CCA Jobs
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Submitting Stratified CCA Jobs"
echo "============================================================"

# Export common parameters (fix for comma parsing bug)
export STRAT_N_PERM="$N_PERM"
export STRAT_N_FOLDS="$N_FOLDS"
export STRAT_GENE_PCA_DIMS="$GENE_PCA_DIMS"
export STRAT_C_VALUES="$C_VALUES"
export STRAT_N_COMPONENTS="$N_COMPONENTS"
export STRAT_HOLDOUT_FRAC="$HOLDOUT_FRAC"

ALL_JOBS=""

for fm in "${FM_MODELS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
        # Set specific parameters for this job
        export STRAT_FM_MODEL="$fm"
        export STRAT_MODALITY="$mod"
        
        # Check if result already exists
        result_file="gene-brain-cca-2/derived/stratified_fm/stratified_comparison_${fm}_${mod}.json"
        if [ -f "$result_file" ]; then
            echo "[skip] ${fm}/${mod} - result exists"
            continue
        fi
        
        # Submit job
        JOB=$(sbatch --parsable --export=ALL slurm/51_stratified_fm.sbatch)
        echo "[submit] ${fm}/${mod} - JobID=$JOB"
        
        if [ -n "$ALL_JOBS" ]; then
            ALL_JOBS="$ALL_JOBS,$JOB"
        else
            ALL_JOBS="$JOB"
        fi
    done
done

if [ -z "$ALL_JOBS" ]; then
    echo ""
    echo "No jobs to submit - all results already exist."
    echo "To force re-run, delete files in gene-brain-cca-2/derived/stratified_fm/"
    exit 0
fi

echo ""
echo "All jobs submitted: $ALL_JOBS"
echo "Monitoring progress..."
echo "Press Ctrl+C to stop monitoring (jobs will continue running)"
echo ""

# ============================================================
# Step 3: Monitor Jobs
# ============================================================
while true; do
    RUNNING=$(squeue -j "$ALL_JOBS" 2>/dev/null | grep -v JOBID | wc -l || echo 0)
    
    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "All jobs completed! Time: $(date)"
        echo "============================================================"
        break
    fi
    
    echo "[$(date +%H:%M:%S)] $RUNNING job(s) still running..."
    squeue -j "$ALL_JOBS" --format="%.10i %.25j %.8T %.10M %.6D %R" 2>/dev/null | head -25 || true
    echo ""
    sleep 60
done

# ============================================================
# Step 4: Collect Errors and Summary
# ============================================================
echo ""
echo "Checking job states..."

# Collect errors
ERROR_SUMMARY="logs/stratified_fm_errors_$(date +%Y%m%d_%H%M%S).txt"
echo "============================================================" > "$ERROR_SUMMARY"
echo "Stratified CCA Error Summary" >> "$ERROR_SUMMARY"
echo "Time: $(date)" >> "$ERROR_SUMMARY"
echo "============================================================" >> "$ERROR_SUMMARY"
for f in logs/strat_fm-*.err; do
    if [ -s "$f" ] && grep -qi "error\|traceback\|exception\|killed\|cancelled" "$f" 2>/dev/null; then
        echo "" >> "$ERROR_SUMMARY"
        echo "----- $f -----" >> "$ERROR_SUMMARY"
        cat "$f" >> "$ERROR_SUMMARY"
    fi
done
echo "Error summary written to: $ERROR_SUMMARY"

# ============================================================
# Step 5: Generate Summary
# ============================================================
echo ""
echo "============================================================"
echo "Results Summary"
echo "============================================================"

for fm in "${FM_MODELS[@]}"; do
    echo ""
    echo "=== $fm ==="
    for mod in "${MODALITIES[@]}"; do
        result_file="gene-brain-cca-2/derived/stratified_fm/stratified_comparison_${fm}_${mod}.json"
        if [ -f "$result_file" ]; then
            # Extract key metrics
            r_mdd=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"r_mdd_holdout_cc1\", 0):.4f}')" 2>/dev/null || echo "N/A")
            r_ctrl=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"r_ctrl_holdout_cc1\", 0):.4f}')" 2>/dev/null || echo "N/A")
            p_val=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d.get(\"perm_p_value\", 1.0):.4f}')" 2>/dev/null || echo "N/A")
            echo "  ${mod}: MDD=$r_mdd, Ctrl=$r_ctrl, p=$p_val"
        else
            echo "  ${mod}: [not found]"
        fi
    done
done

echo ""
echo "============================================================"
echo "DONE! Check full results in:"
echo "  gene-brain-cca-2/derived/stratified_fm/"
echo "============================================================"
