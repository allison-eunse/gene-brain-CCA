#!/bin/bash
# =============================================================================
# Resubmit failed DNABERT2/Evo2 stratified CCA jobs with checkpointing
#
# Lab server compliance:
# - All compute via Slurm (sbatch/srun), never on login node
# - Uses /scratch for checkpoint files (fast I/O)
# - No manual CUDA_VISIBLE_DEVICES
# - Run monitoring in tmux for persistence
#
# Usage:
#   tmux new -s stratified_resume
#   ./resubmit_failed_stratified.sh
#
# Or one-liner:
#   tmux new -s stratified_resume "./resubmit_failed_stratified.sh; exec bash"
# =============================================================================

set -uo pipefail
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
mkdir -p logs

if [ -z "${TMUX:-}" ]; then
    echo "[error] Please run inside tmux: tmux new -s stratified_resume"
    exit 1
fi

echo "============================================================"
echo "Resubmit Failed Stratified CCA Jobs (with Checkpointing)"
echo "Time: $(date)"
echo "============================================================"

# Only DNABERT2 and Evo2 need resubmission (HyenaDNA/Caduceus completed)
FM_MODELS=("dnabert2" "evo2")
MODALITIES=("schaefer7" "schaefer17" "smri" "dmri")

# Parameters (same as original run)
export STRAT_N_PERM=1000
export STRAT_N_FOLDS=5
export STRAT_GENE_PCA_DIMS="64,128,256,512"
export STRAT_C_VALUES="0.1,0.3,0.5"
export STRAT_N_COMPONENTS=10
export STRAT_HOLDOUT_FRAC=0.2

ALL_JOBS=""

for fm in "${FM_MODELS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
        result_file="gene-brain-cca-2/derived/stratified_fm/stratified_comparison_${fm}_${mod}.json"

        if [ -f "$result_file" ]; then
            echo "[skip] ${fm}/${mod} - already complete"
            continue
        fi

        export STRAT_FM_MODEL="$fm"
        export STRAT_MODALITY="$mod"

        JOB=$(sbatch --parsable --export=ALL slurm/51_stratified_fm.sbatch)
        echo "[submit] ${fm}/${mod} - JobID=$JOB"

        if [ -n "$ALL_JOBS" ]; then
            ALL_JOBS="$ALL_JOBS,$JOB"
        else
            ALL_JOBS="$JOB"
        fi
        sleep 2
    done
done

if [ -z "$ALL_JOBS" ]; then
    echo "All jobs already complete!"
    exit 0
fi

echo ""
echo "Submitted jobs: $ALL_JOBS"
echo "Monitoring... (Ctrl+C safe - jobs continue in Slurm)"
echo "Attach later: tmux attach -t stratified_resume"
echo ""

while true; do
    completed=0
    total=0

    for fm in "${FM_MODELS[@]}"; do
        for mod in "${MODALITIES[@]}"; do
            total=$((total + 1))
            result_file="gene-brain-cca-2/derived/stratified_fm/stratified_comparison_${fm}_${mod}.json"

            if [ -f "$result_file" ]; then
                completed=$((completed + 1))
            else
                ckpt="/scratch/connectome/${USER}/stratified_checkpoints/perm_${fm}_${mod}.npy"
                if [ -f "$ckpt" ]; then
                    perm_count=$(python3 -c "import numpy as np; print(len(np.load('$ckpt')))" 2>/dev/null || echo "?")
                    echo "  [running] ${fm}/${mod}: ${perm_count}/1000 perms (checkpoint exists)"
                else
                    echo "  [pending] ${fm}/${mod}: waiting to start or no checkpoint yet"
                fi
            fi
        done
    done

    echo "[$(date '+%H:%M:%S')] Progress: $completed/$total complete"

    if [ "$completed" -eq "$total" ]; then
        echo ""
        echo "============================================================"
        echo "All DNABERT2/Evo2 jobs complete! Time: $(date)"
        echo "============================================================"
        break
    fi

    if grep -qi "TIME LIMIT|CANCELLED|ERROR|Traceback" logs/strat_fm-*.err 2>/dev/null; then
        echo "[warn] Detected errors in logs/strat_fm-*.err (check files)."
    fi

    sleep 300
done

echo ""
echo "Final results saved to:"
ls -la gene-brain-cca-2/derived/stratified_fm/stratified_comparison_*.json
