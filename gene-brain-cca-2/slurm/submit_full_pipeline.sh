#!/bin/bash
# =============================================================================
# MASTER SUBMISSION SCRIPT: gene-brain-cca-2 Full Pipeline
# =============================================================================
# This script:
#   1. Submits Pipeline A (Interpretable SCCA) using /scratch for fast I/O
#   2. Automatically chains Pipeline B after A completes (SLURM dependency)
#   3. Copies results back to persistent storage after EACH pipeline
#   4. Cleans up scratch directory after everything finishes
#
# Usage:
#   cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
#   bash gene-brain-cca-2/slurm/submit_full_pipeline.sh
#
# =============================================================================

set -euo pipefail

# --- Configuration ---
PROJ_ROOT="/storage/bigdata/UKB/fMRI/gene-brain-CCA"
SCRATCH_ROOT="/scratch/connectome/${USER}/gene-brain-cca-2/derived"
PERSISTENT_ROOT="${PROJ_ROOT}/gene-brain-cca-2/derived"
SLURM_DIR="${PROJ_ROOT}/gene-brain-cca-2/slurm"
LOG_DIR="${PROJ_ROOT}/logs"

# --- Setup ---
cd "$PROJ_ROOT"
mkdir -p "$LOG_DIR"
mkdir -p "$SCRATCH_ROOT"
mkdir -p "$PERSISTENT_ROOT/interpretable"
mkdir -p "$PERSISTENT_ROOT/wide_gene"

echo "============================================================"
echo "gene-brain-cca-2 Full Pipeline Submission"
echo "============================================================"
echo "Project root:     $PROJ_ROOT"
echo "Scratch output:   $SCRATCH_ROOT"
echo "Persistent dest:  $PERSISTENT_ROOT"
echo "Start time:       $(date)"
echo "============================================================"

# --- Submit Pipeline A ---
echo ""
echo "[1/4] Submitting Pipeline A (Interpretable SCCA)..."
export OUT_ROOT="$SCRATCH_ROOT"
JOB_A=$(sbatch \
    --parsable \
    --export=ALL,OUT_ROOT \
    "${SLURM_DIR}/01_interpretable_scca.sbatch")
echo "      Pipeline A submitted: Job ID = $JOB_A"

# --- Submit Copyback after Pipeline A ---
echo ""
echo "[2/4] Submitting copyback job after Pipeline A..."
JOB_COPY_A=$(sbatch --parsable \
    --dependency=afterok:${JOB_A} \
    --job-name=copyback_A \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --exclude=node1,node3 \
    --output="${LOG_DIR}/copyback_A_%j.out" \
    --error="${LOG_DIR}/copyback_A_%j.err" \
    --wrap="
set -eu
echo '=== Copyback Pipeline A Results ==='
echo 'Start: '\$(date)
mkdir -p '${PERSISTENT_ROOT}/interpretable'
rsync -av --progress '${SCRATCH_ROOT}/interpretable/' '${PERSISTENT_ROOT}/interpretable/' || \
    cp -av '${SCRATCH_ROOT}/interpretable/.' '${PERSISTENT_ROOT}/interpretable/'
echo 'Pipeline A results copied to: ${PERSISTENT_ROOT}/interpretable/'
echo 'Done: '\$(date)
")
echo "      Copyback A submitted: Job ID = $JOB_COPY_A (depends on $JOB_A)"

# --- Submit Pipeline B (depends on Pipeline A) ---
echo ""
echo "[3/4] Submitting Pipeline B (Predictive Suite) - waits for A..."
JOB_B=$(sbatch \
    --parsable \
    --dependency=afterok:${JOB_A} \
    --export=ALL,OUT_ROOT \
    "${SLURM_DIR}/02_predictive_wide_suite.sbatch")
echo "      Pipeline B submitted: Job ID = $JOB_B (depends on $JOB_A)"

# --- Submit Final Copyback + Cleanup (depends on Pipeline B) ---
echo ""
echo "[4/4] Submitting final copyback + cleanup job..."
JOB_FINAL=$(sbatch --parsable \
    --dependency=afterok:${JOB_B} \
    --job-name=copyback_cleanup \
    --ntasks=1 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=01:00:00 \
    --exclude=node1,node3 \
    --output="${LOG_DIR}/copyback_cleanup_%j.out" \
    --error="${LOG_DIR}/copyback_cleanup_%j.err" \
    --wrap="
set -eu
echo '=== Final Copyback + Cleanup ==='
echo 'Start: '\$(date)

# Copy Pipeline B results
mkdir -p '${PERSISTENT_ROOT}/wide_gene'
rsync -av --progress '${SCRATCH_ROOT}/wide_gene/' '${PERSISTENT_ROOT}/wide_gene/' || \
    cp -av '${SCRATCH_ROOT}/wide_gene/.' '${PERSISTENT_ROOT}/wide_gene/'
echo 'Pipeline B results copied to: ${PERSISTENT_ROOT}/wide_gene/'

# Verify key files exist
echo ''
echo '=== Verification ==='
for f in '${PERSISTENT_ROOT}/interpretable/scca_interpretable_results.json' \
         '${PERSISTENT_ROOT}/wide_gene/predictive_suite_results.json'; do
    if [ -f \"\$f\" ]; then
        echo \"✓ Found: \$f\"
    else
        echo \"✗ MISSING: \$f\"
    fi
done

# Cleanup scratch
echo ''
echo '=== Cleaning up scratch ==='
rm -rf '${SCRATCH_ROOT}'
echo 'Deleted: ${SCRATCH_ROOT}'

echo ''
echo '=== ALL DONE ==='
echo 'Persistent results in: ${PERSISTENT_ROOT}/'
echo 'Finished: '\$(date)
")
echo "      Final copyback+cleanup submitted: Job ID = $JOB_FINAL (depends on $JOB_B)"

# --- Summary ---
echo ""
echo "============================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================"
echo ""
echo "Job Chain:"
echo "  Pipeline A:        $JOB_A"
echo "    └─ Copyback A:   $JOB_COPY_A  (afterok:$JOB_A)"
echo "    └─ Pipeline B:   $JOB_B       (afterok:$JOB_A)"
echo "        └─ Final:    $JOB_FINAL   (afterok:$JOB_B)"
echo ""
echo "Expected timeline:"
echo "  Pipeline A:  ~4 hours"
echo "  Pipeline B:  ~8 hours (starts after A)"
echo "  Total:       ~12-13 hours"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/interp_scca_${JOB_A}.out"
echo "  tail -f logs/wide_suite_${JOB_B}.out"
echo ""
echo "After completion, view results with:"
echo "  python gene-brain-cca-2/scripts/view_results.py"
echo ""
echo "============================================================"

# Save job IDs for reference
echo "JOB_A=$JOB_A" > "${LOG_DIR}/latest_submission.env"
echo "JOB_COPY_A=$JOB_COPY_A" >> "${LOG_DIR}/latest_submission.env"
echo "JOB_B=$JOB_B" >> "${LOG_DIR}/latest_submission.env"
echo "JOB_FINAL=$JOB_FINAL" >> "${LOG_DIR}/latest_submission.env"
echo "SCRATCH_ROOT=$SCRATCH_ROOT" >> "${LOG_DIR}/latest_submission.env"
echo "PERSISTENT_ROOT=$PERSISTENT_ROOT" >> "${LOG_DIR}/latest_submission.env"
echo "SUBMITTED_AT=$(date -Iseconds)" >> "${LOG_DIR}/latest_submission.env"
echo ""
echo "Job IDs saved to: ${LOG_DIR}/latest_submission.env"
