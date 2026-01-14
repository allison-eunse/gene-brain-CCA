# Quick Start Guide: gene-brain-cca-2

**⏱️ Total time to launch both pipelines: < 5 minutes**

---

## Prerequisites Check (30 seconds)

```bash
# 1. Navigate to project root
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# 2. Verify conda environment
conda env list | grep cca_env

# 3. Create logs directory
mkdir -p logs

# 4. Quick dependency check
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
python -c "from cca_zoo.linear import SCCA_PMD; print('✓ Dependencies OK')"
```

---

## Launch Pipeline A: Interpretable SCCA (2 minutes)

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
```

**Expected output:**
```
Submitted batch job 12345678
```

**Monitor progress:**
```bash
# Watch the log file (replace JOBID with actual job ID)
tail -f logs/interp_scca_12345678.out

# Or check job status
squeue -u $USER
```

**⏱️ Runtime:** ~4 hours

**When complete, verify:**
```bash
ls -lh gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json
# Should see a file ~5 KB
```

---

## Launch Pipeline B: Predictive Suite (2 minutes)

**⚠️ Wait for Pipeline A to complete first!**

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch
```

**Monitor:**
```bash
tail -f logs/wide_suite_<JOBID>.out
```

**⏱️ Runtime:** ~8 hours

**When complete, verify:**
```bash
ls -lh gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json
# Should see a file ~3 KB
```

---

## View Results (1 minute)

**Pipeline A (interpretability):**
```bash
cat gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json | python -m json.tool
```

Look for:
- `"r"`: Canonical correlations (higher is better, typically 0.1-0.5)
- `"sparsity_gene"` / `"sparsity_fmri"`: Fraction of features zeroed out (higher = more interpretable)

**Pipeline B (prediction):**
```bash
cat gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json | python -m json.tool
```

Compare AUC scores:
- `gene_only_logreg` vs `fmri_only_logreg` → Which modality is more predictive?
- `early_fusion_logreg` → Does combining help?
- `cca_joint_logreg` vs `early_fusion_logreg` → Does CCA add value?

---

## Common Quick Fixes

**Problem:** `sbatch: command not found`
```bash
# You're not on a SLURM system; run interactively instead (see README.md)
```

**Problem:** `conda: command not found`
```bash
source /usr/anaconda3/etc/profile.d/conda.sh
```

**Problem:** Pipeline B fails immediately
```bash
# Check Pipeline A completed:
ls gene-brain-cca-2/derived/interpretable/ids_common.npy
# If missing, run Pipeline A first
```

**Problem:** Job is pending for > 30 minutes
```bash
squeue -u $USER  # Check your position in queue
# Or run with lower resources (edit sbatch file)
```

---

## What's Next?

1. **Read full documentation:** `gene-brain-cca-2/README.md`
2. **Interpret results:** See "Understanding the Results" section
3. **Modify hyperparameters:** See "Advanced Usage" section
4. **Troubleshooting:** See "Troubleshooting" section or `TROUBLESHOOTING.md`

---

## Key File Paths Reference

| Component | Path |
|-----------|------|
| Project root | `/storage/bigdata/UKB/fMRI/gene-brain-CCA` |
| Pipeline scripts | `gene-brain-cca-2/scripts/` |
| SLURM launchers | `gene-brain-cca-2/slurm/` |
| Outputs | `gene-brain-cca-2/derived/` |
| Job logs | `logs/` (in project root) |
| Conda env | `/scratch/connectome/allie/envs/cca_env` |

---

**Need help?** See full `README.md` or contact the maintainer.
