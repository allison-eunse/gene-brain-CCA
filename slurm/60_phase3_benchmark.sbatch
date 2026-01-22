#!/bin/bash
#SBATCH --job-name=phase3_cca
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclude=node1,node3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --requeue

# =============================================================================
# Phase 3 CCA Benchmark - Slurm Job Script
# 
# Lab Server Compliance:
# - No login node execution (via sbatch)
# - No manual CUDA_VISIBLE_DEVICES (let Slurm handle it)
# - Uses /scratch for fast I/O
# - GPU monitoring in background
# - Automatic backup to /storage
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "Phase 3 CCA Benchmark"
echo "JobID: ${SLURM_JOB_ID:-NA}"
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "FM:    ${STRAT_FM_MODEL:-not_set}"
echo "Mod:   ${STRAT_MODALITY:-not_set}"
echo "============================================================"

# Print Slurm environment for debugging
env | grep SLURM || true

# Thread control (lab guideline - avoid CPU oversubscription)
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"

# Conda activation (required for Python)
export PS1="${PS1-}"
set +u
source /usr/anaconda3/etc/profile.d/conda.sh
set -u
conda activate /scratch/connectome/allie/envs/cca_env

export PYTHONUNBUFFERED=1

# Change to project directory
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Create logs directory
mkdir -p logs

# GPU monitoring in background (lab guideline - save CSV for analysis)
# Only start if nvidia-smi is available (handles CPU-only nodes gracefully)
GPU_MON_PID=""
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
        --format=csv --loop-ms=5000 > "logs/gpu_stats_${SLURM_JOB_ID}.csv" 2>/dev/null &
    GPU_MON_PID=$!
    echo "[gpu-mon] Started monitoring (PID: $GPU_MON_PID)"
fi

# Cleanup on exit (handles both normal exit and signals)
cleanup() {
    if [ -n "$GPU_MON_PID" ]; then
        kill "$GPU_MON_PID" 2>/dev/null || true
        echo "[gpu-mon] Stopped"
    fi
}
trap cleanup EXIT SIGTERM SIGINT

# Use /scratch for cache (fast I/O per lab guidelines)
CACHE_DIR="/scratch/connectome/${USER}/phase3_cache"
if [ ! -d "/scratch/connectome/${USER}" ]; then
    # Fallback if /scratch not available
    CACHE_DIR="gene-brain-cca-2/derived/phase3_cache"
fi
mkdir -p "$CACHE_DIR"
echo "[cache] Using: $CACHE_DIR"

# Output directory (persistent storage)
OUTPUT_DIR="gene-brain-cca-2/derived/phase3_results"
mkdir -p "$OUTPUT_DIR"

# Validate required environment variables
if [ -z "${STRAT_FM_MODEL:-}" ] || [ -z "${STRAT_MODALITY:-}" ]; then
    echo "[ERROR] STRAT_FM_MODEL and STRAT_MODALITY must be set"
    echo "  Example: export STRAT_FM_MODEL=caduceus STRAT_MODALITY=dmri"
    exit 1
fi

# Run the benchmark
echo ""
echo "[run] Starting Phase 3 benchmark..."
srun python scripts/run_phase3_benchmark.py \
    --fm-model "$STRAT_FM_MODEL" \
    --modality "$STRAT_MODALITY" \
    --cache-dir "$CACHE_DIR" \
    --checkpoint-path "${CACHE_DIR}/checkpoint_${STRAT_FM_MODEL}_${STRAT_MODALITY}.json" \
    --output-dir "$OUTPUT_DIR" \
    --gene-pca-dims "${STRAT_GENE_PCA_DIMS:-64,128,256,512}" \
    --c-values "${STRAT_C_VALUES:-0.1,0.3,0.5}"

# Backup checkpoints to storage (lab guideline - prevent scratch deletion)
echo ""
echo "[backup] Copying checkpoints to storage..."
cp ${CACHE_DIR}/checkpoint_*.json "$OUTPUT_DIR/" 2>/dev/null || true

# Touch scratch files to prevent auto-deletion (lab guideline)
find "$CACHE_DIR" -type f -exec touch {} + 2>/dev/null || true

echo ""
echo "============================================================"
echo "Phase 3 Benchmark Complete"
echo "Done:  $(date)"
echo "============================================================"
