#!/bin/bash
# =============================================================================
# Wait for Stratified Jobs to Finish, then Run Phase 3
#
# This script:
# 1. Monitors DNABERT2/Evo2 stratified jobs (82462-82469)
# 2. Once all are complete, runs the Phase 3 benchmark
# 3. Gathers and displays results
#
# Usage (in tmux):
#   ./run_phase3_after_stratified.sh
# =============================================================================

set -uo pipefail
# Note: Not using -e because grep returns non-zero when no match found

cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

echo "============================================================"
echo "Phase 3 Pipeline - Wait and Run"
echo "Started: $(date)"
echo "============================================================"

# Jobs to monitor (DNABERT2 + Evo2)
JOBS_TO_MONITOR=(82462 82463 82464 82465 82466 82467 82468 82469)

check_all_done() {
    for job in "${JOBS_TO_MONITOR[@]}"; do
        logfile="logs/strat_fm-${job}.out"
        if [ -f "$logfile" ]; then
            if ! grep -q "Done:" "$logfile" 2>/dev/null; then
                echo "false"
                return
            fi
        else
            echo "false"
            return
        fi
    done
    echo "true"
}

print_status() {
    echo ""
    echo "[$(date '+%H:%M:%S')] Checking job status..."
    local completed=0
    local running=0
    
    for job in "${JOBS_TO_MONITOR[@]}"; do
        logfile="logs/strat_fm-${job}.out"
        if [ -f "$logfile" ]; then
            if grep -q "Done:" "$logfile" 2>/dev/null; then
                completed=$((completed + 1))
            else
                running=$((running + 1))
                # Show progress
                last_line=$(tail -1 "$logfile" 2>/dev/null | head -c 60 || echo "...")
                echo "  Running $job: $last_line"
            fi
        else
            running=$((running + 1))
            echo "  Pending $job: no log file"
        fi
    done
    
    echo "  Completed: $completed/${#JOBS_TO_MONITOR[@]}"
    echo "  Running:   $running/${#JOBS_TO_MONITOR[@]}"
}

# =============================================================================
# Phase 1: Wait for stratified jobs to complete
# =============================================================================

echo ""
echo "============================================================"
echo "Phase 1: Waiting for stratified jobs to complete..."
echo "============================================================"

while [ "$(check_all_done)" = "false" ]; do
    print_status
    echo ""
    echo "  Sleeping 5 minutes... (Ctrl+C to abort)"
    sleep 300  # Check every 5 minutes
done

echo ""
echo "============================================================"
echo "All stratified jobs completed!"
echo "============================================================"

# Show final stratified results summary
echo ""
echo "Stratified Results Summary:"
echo "----------------------------"
ls -la gene-brain-cca-2/derived/stratified_fm/stratified_comparison_*.json 2>/dev/null || echo "No comparison files found"

# =============================================================================
# Phase 2: Run Phase 3 benchmark
# =============================================================================

echo ""
echo "============================================================"
echo "Phase 2: Running Phase 3 benchmark..."
echo "============================================================"
echo "Focus: Caduceus + dMRI/sMRI (best performers)"
echo ""

# Submit Phase 3 jobs
./submit_phase3.sh

echo ""
echo "Phase 3 jobs submitted. Waiting for completion..."
echo ""

# Wait for Phase 3 jobs to complete
sleep 60  # Give jobs time to start

# Monitor Phase 3 jobs
while true; do
    # Check if any Phase 3 results exist and are complete
    completed=0
    total=2  # Caduceus x (dmri, smri)
    
    for mod in dmri smri; do
        result_file="gene-brain-cca-2/derived/phase3_results/caduceus_${mod}_results.json"
        if [ -f "$result_file" ] && grep -q '"status": "FINISHED"' "$result_file" 2>/dev/null; then
            ((completed++))
        fi
    done
    
    echo "[$(date '+%H:%M:%S')] Phase 3 progress: $completed/$total complete"
    
    if [ "$completed" -eq "$total" ]; then
        break
    fi
    
    # Show job status from logs
    for mod in dmri smri; do
        log_pattern="logs/phase3_caduceus_${mod}-*.out"
        latest_log=$(ls -t $log_pattern 2>/dev/null | head -1 || echo "")
        if [ -n "$latest_log" ]; then
            last_line=$(tail -1 "$latest_log" 2>/dev/null | head -c 60 || echo "...")
            echo "  caduceus_${mod}: $last_line"
        fi
    done
    
    echo "  Sleeping 2 minutes..."
    sleep 120
done

# =============================================================================
# Phase 3: Gather and display results
# =============================================================================

echo ""
echo "============================================================"
echo "Phase 3: Gathering results..."
echo "============================================================"

python scripts/gather_phase3_results.py

echo ""
echo "============================================================"
echo "Pipeline Complete!"
echo "Finished: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - gene-brain-cca-2/derived/phase3_results/summary.csv"
echo "  - gene-brain-cca-2/derived/phase3_results/summary.md"
echo ""
echo "To view results:"
echo "  cat gene-brain-cca-2/derived/phase3_results/summary.md"
