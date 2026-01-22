#!/bin/bash
# =============================================================================
# Phase 3 CCA Benchmark - Master Submission Script
#
# Submits Phase 3 benchmarks focusing on best-performing combinations:
# - Caduceus (best FM model from stratified analysis)
# - dMRI and sMRI (structural > functional per stratified results)
#
# Lab Server Compliance:
# - Automates workflow with bash loops
# - Skips already-completed experiments
# - Prevents I/O storms with sleep between submissions
# - Uses /scratch for fast cache
#
# Usage:
#   ./submit_phase3.sh              # Submit all pending jobs
#   ./submit_phase3.sh --all-fm     # Include all 4 FM models
#   ./submit_phase3.sh --check      # Check status only, don't submit
# =============================================================================

set -euo pipefail

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Parse arguments
ALL_FM=false
CHECK_ONLY=false
for arg in "$@"; do
    case $arg in
        --all-fm)
            ALL_FM=true
            ;;
        --check)
            CHECK_ONLY=true
            ;;
    esac
done

# Configuration
if [ "$ALL_FM" = true ]; then
    FM_MODELS=("dnabert2" "evo2" "hyenadna" "caduceus")
else
    FM_MODELS=("caduceus")  # Focus on best performer from stratified analysis
fi

# Modalities: Structural > Functional (per stratified results)
MODALITIES=("dmri" "smri")

# Hyperparameter grid
GENE_PCA_DIMS="64,128,256,512"
C_VALUES="0.1,0.3,0.5"

# Output directory
OUTPUT_DIR="gene-brain-cca-2/derived/phase3_results"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

echo "============================================================"
echo "Phase 3 CCA Benchmark Submission"
echo "============================================================"
echo "FM Models:   ${FM_MODELS[*]}"
echo "Modalities:  ${MODALITIES[*]}"
echo "PCA dims:    $GENE_PCA_DIMS"
echo "C values:    $C_VALUES"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"
echo ""

# Count jobs
SUBMITTED=0
SKIPPED=0
PENDING=0

for fm in "${FM_MODELS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
        JOB_NAME="phase3_${fm}_${mod}"
        RESULT_FILE="${OUTPUT_DIR}/${fm}_${mod}_results.json"
        
        # Check if already completed
        if [ -f "$RESULT_FILE" ]; then
            # Verify it has FINISHED status
            if grep -q '"status": "FINISHED"' "$RESULT_FILE" 2>/dev/null; then
                echo "[SKIP] $JOB_NAME - already completed"
                ((SKIPPED++))
                continue
            fi
        fi
        
        # Check if data exists
        DATA_DIR="derived_stratified_fm/${fm}/${mod}"
        if [ ! -f "${DATA_DIR}/X_gene_wide.npy" ]; then
            echo "[SKIP] $JOB_NAME - data not found at $DATA_DIR"
            ((SKIPPED++))
            continue
        fi
        
        if [ "$CHECK_ONLY" = true ]; then
            echo "[PENDING] $JOB_NAME"
            ((PENDING++))
            continue
        fi
        
        echo "[SUBMIT] $JOB_NAME"
        
        # Export variables globally (lab guideline - avoid Slurm --export comma issues)
        export STRAT_FM_MODEL="$fm"
        export STRAT_MODALITY="$mod"
        export STRAT_GENE_PCA_DIMS="$GENE_PCA_DIMS"
        export STRAT_C_VALUES="$C_VALUES"
        
        # Submit job
        sbatch --job-name="$JOB_NAME" --export=ALL slurm/60_phase3_benchmark.sbatch
        
        ((SUBMITTED++))
        
        # Prevent I/O storm (lab guideline)
        sleep 2
    done
done

echo ""
echo "============================================================"
echo "Submission Summary"
echo "============================================================"

if [ "$CHECK_ONLY" = true ]; then
    echo "Mode:      Check only (no jobs submitted)"
    echo "Pending:   $PENDING"
    echo "Skipped:   $SKIPPED"
else
    echo "Submitted: $SUBMITTED"
    echo "Skipped:   $SKIPPED"
fi

echo ""
echo "Monitor jobs:      squeue --me"
echo "Check GPU util:    tail -f logs/gpu_stats_*.csv"
echo "View job output:   tail -f logs/phase3_*.out"
echo "Check results:     ls -la $OUTPUT_DIR"
