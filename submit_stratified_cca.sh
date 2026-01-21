#!/bin/bash
# Submit all stratified CCA jobs and monitor progress
# Run this inside tmux: tmux new -s stratified_cca

set -e
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
mkdir -p logs

EMAIL_NOTIFY="eunseyou@snu.ac.kr"

send_email() {
  local subject="$1"
  local body="$2"
  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$EMAIL_NOTIFY"
  elif command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "$body" | mailx -s "$subject" "$EMAIL_NOTIFY"
  elif command -v sendmail >/dev/null 2>&1; then
    {
      echo "To: ${EMAIL_NOTIFY}"
      echo "Subject: ${subject}"
      echo ""
      echo "$body"
    } | sendmail -t
  else
    echo "[warn] No mail command found; cannot send email." >&2
  fi
}

echo "============================================================"
echo "Submitting Stratified CCA Jobs"
echo "Time: $(date)"
echo "============================================================"

submit_jobs() {
  local tag="$1"
  local n_perm="${2:-1000}"
  local n_folds="${3:-5}"
  local gene_dims="${4:-64,128,256,512}"
  local c_values="${5:-0.1,0.3,0.5}"
  local n_components="${6:-10}"
  local holdout_frac="${7:-0.2}"

  echo ""
  echo "Submitting jobs ($tag): n_perm=$n_perm, n_folds=$n_folds, gene_dims=$gene_dims"
  JOB1=$(sbatch --parsable \
    --export=ALL,STRAT_N_PERM="$n_perm",STRAT_N_FOLDS="$n_folds",STRAT_GENE_PCA_DIMS="$gene_dims",STRAT_C_VALUES="$c_values",STRAT_N_COMPONENTS="$n_components",STRAT_HOLDOUT_FRAC="$holdout_frac" \
    slurm/40_stratified_schaefer7.sbatch)
  echo "Submitted Schaefer7:  JobID=$JOB1"

  JOB2=$(sbatch --parsable \
    --export=ALL,STRAT_N_PERM="$n_perm",STRAT_N_FOLDS="$n_folds",STRAT_GENE_PCA_DIMS="$gene_dims",STRAT_C_VALUES="$c_values",STRAT_N_COMPONENTS="$n_components",STRAT_HOLDOUT_FRAC="$holdout_frac" \
    slurm/41_stratified_schaefer17.sbatch)
  echo "Submitted Schaefer17: JobID=$JOB2"

  JOB3=$(sbatch --parsable \
    --export=ALL,STRAT_N_PERM="$n_perm",STRAT_N_FOLDS="$n_folds",STRAT_GENE_PCA_DIMS="$gene_dims",STRAT_C_VALUES="$c_values",STRAT_N_COMPONENTS="$n_components",STRAT_HOLDOUT_FRAC="$holdout_frac" \
    slurm/42_stratified_smri_tabular.sbatch)
  echo "Submitted sMRI:       JobID=$JOB3"

  JOB4=$(sbatch --parsable \
    --export=ALL,STRAT_N_PERM="$n_perm",STRAT_N_FOLDS="$n_folds",STRAT_GENE_PCA_DIMS="$gene_dims",STRAT_C_VALUES="$c_values",STRAT_N_COMPONENTS="$n_components",STRAT_HOLDOUT_FRAC="$holdout_frac" \
    slurm/43_stratified_dmri_tabular.sbatch)
  echo "Submitted dMRI:       JobID=$JOB4"

  export ALL_JOBS="$JOB1,$JOB2,$JOB3,$JOB4"
}

# Submit all jobs (full settings)
submit_jobs "full" 1000 5 "64,128,256,512" "0.1,0.3,0.5" 10 0.2

echo ""
echo "All jobs submitted. Monitoring progress..."
echo "Press Ctrl+C to stop monitoring (jobs will continue running)"
echo ""

# Wait for all jobs to complete

while true; do
    # Check job statuses
    RUNNING=$(squeue -j "$ALL_JOBS" 2>/dev/null | grep -v JOBID | wc -l || echo 0)
    
    if [ "$RUNNING" -eq 0 ]; then
        echo ""
        echo "============================================================"
        echo "All jobs completed! Time: $(date)"
        echo "============================================================"
        break
    fi
    
    echo "[$(date +%H:%M:%S)] $RUNNING job(s) still running..."
    squeue -j "$ALL_JOBS" --format="%.10i %.15j %.8T %.10M %.6D %R" 2>/dev/null || true
    echo ""
    sleep 60
done

collect_errors() {
  local out_file="$1"
  echo "============================================================" > "$out_file"
  echo "Stratified CCA Error Summary" >> "$out_file"
  echo "Time: $(date)" >> "$out_file"
  echo "============================================================" >> "$out_file"
  for f in logs/strat_*.err; do
    if [ -s "$f" ]; then
      echo "" >> "$out_file"
      echo "----- $f -----" >> "$out_file"
      cat "$f" >> "$out_file"
    fi
  done
}

echo ""
echo "Checking job states..."
FAILED=0
for jid in ${ALL_JOBS//,/ }; do
  state=$(sacct -j "$jid" --format=State --noheader | head -n 1 | awk '{print $1}')
  echo "  Job $jid state: ${state:-UNKNOWN}"
  if [[ "$state" != "COMPLETED" ]]; then
    FAILED=1
  fi
done

ERROR_SUMMARY="logs/stratified_errors_$(date +%Y%m%d_%H%M%S).txt"
collect_errors "$ERROR_SUMMARY"
echo "Error summary written to: $ERROR_SUMMARY"

if [ "$FAILED" -eq 1 ]; then
  echo ""
  echo "One or more jobs failed. Retrying with reduced settings..."
  send_email \
    "Stratified CCA failure detected" \
    "One or more jobs failed at $(date).\nError summary: ${ERROR_SUMMARY}\nRetrying with reduced settings."
  submit_jobs "retry" 200 3 "64,128,256" "0.1,0.3" 5 0.2

  echo ""
  echo "Monitoring retry jobs..."
  while true; do
    RUNNING=$(squeue -j "$ALL_JOBS" 2>/dev/null | grep -v JOBID | wc -l || echo 0)
    if [ "$RUNNING" -eq 0 ]; then
      echo ""
      echo "Retry jobs completed! Time: $(date)"
      break
    fi
    echo "[$(date +%H:%M:%S)] $RUNNING retry job(s) still running..."
    squeue -j "$ALL_JOBS" --format="%.10i %.15j %.8T %.10M %.6D %R" 2>/dev/null || true
    echo ""
    sleep 60
  done

  ERROR_SUMMARY="logs/stratified_errors_retry_$(date +%Y%m%d_%H%M%S).txt"
  collect_errors "$ERROR_SUMMARY"
  echo "Retry error summary written to: $ERROR_SUMMARY"
  send_email \
    "Stratified CCA retry completed" \
    "Retry jobs finished at $(date).\nError summary: ${ERROR_SUMMARY}\nCheck logs/: strat_*.out / strat_*.err"
fi

# Run analysis after all jobs complete
echo ""
echo "Running analysis script..."
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
python scripts/analyze_stratified_results.py --derived-dir gene-brain-cca-2/derived

echo ""
echo "============================================================"
echo "DONE! Check results in:"
echo "  gene-brain-cca-2/derived/stratified_comparison_report.md"
echo "============================================================"

send_email \
  "Stratified CCA completed" \
  "All stratified CCA jobs completed at $(date).\nReport: gene-brain-cca-2/derived/stratified_comparison_report.md\nError summary: ${ERROR_SUMMARY}"
