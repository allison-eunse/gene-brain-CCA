# Quick Reference Card

One-page reference for the most common commands and file locations.

---

## 🚀 Running Pipelines

```bash
# Navigate to project root
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Verify setup
bash gene-brain-cca-2/scripts/verify_setup.sh

# Submit Pipeline A (run first)
sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch

# Submit Pipeline B (after A completes)
sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch

# View results
python gene-brain-cca-2/scripts/view_results.py
```

---

## 📊 Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f logs/<job_name>_<JOBID>.out

# Check for errors
cat logs/<job_name>_<JOBID>.err

# Job history
sacct -u $USER --format=JobID,JobName,State,ExitCode -S today
```

---

## 📁 Key File Paths

### Inputs (Pre-existing Data)
```
/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/
  ├── ids_gene.npy
  └── X_gene_ng.npy

/storage/bigdata/UKB/fMRI/
  ├── fmri_eids_180.npy
  └── fmri_X_180.npy

/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/
  ├── iids.npy
  ├── labels.npy
  ├── covariates_age.npy
  ├── covariates_sex.npy
  └── covariates_valid_mask.npy

/storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/
  └── <gene_name>/embeddings_*.npy
```

### Outputs (Created by Pipelines)
```
gene-brain-cca-2/derived/interpretable/
  ├── ids_common.npy                      # Overlap subject IDs
  ├── X_gene_z.npy                        # Gene data (4218×111)
  ├── X_fmri_z.npy                        # fMRI data (4218×180)
  ├── labels_common.npy                   # Labels
  ├── scca_interpretable_results.json     # Main results
  ├── scca_interpretable_results_U.npy    # Gene components
  └── scca_interpretable_results_V.npy    # Brain components

gene-brain-cca-2/derived/wide_gene/
  ├── X_gene_wide.npy                     # Full embeddings (4218×85248)
  ├── X_gene_pca512.npy                   # PCA reduced (4218×512)
  └── predictive_suite_results.json       # All AUCs/APs
```

---

## 🔧 Common Tasks

### Check if results exist
```bash
ls -lh gene-brain-cca-2/derived/interpretable/*.json
ls -lh gene-brain-cca-2/derived/wide_gene/*.json
```

### View results manually
```bash
cat gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json | python -m json.tool
cat gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json | python -m json.tool
```

### Check data integrity
```bash
python -c "import numpy as np; X = np.load('gene-brain-cca-2/derived/interpretable/X_gene_z.npy'); print(f'Shape: {X.shape}, Range: [{X.min():.2f}, {X.max():.2f}]')"
```

### Cancel a job
```bash
scancel <JOBID>
```

---

## ⚙️ Environment Setup

```bash
# Activate conda environment
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Verify packages
python -c "from cca_zoo.linear import SCCA_PMD; print('cca-zoo OK')"
python -c "from sklearn.decomposition import PCA; print('sklearn OK')"
```

---

## 🐛 Quick Debugging

**Problem:** Job fails immediately
```bash
# Check error log
cat logs/<job_name>_<JOBID>.err

# Run interactively to see full error
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
# Copy command from sbatch file and run
```

**Problem:** Pipeline B can't find ids_common.npy
```bash
# Verify Pipeline A completed
ls gene-brain-cca-2/derived/interpretable/ids_common.npy
# If missing, run Pipeline A first
```

**Problem:** Out of memory
```bash
# Edit sbatch file and increase memory:
#SBATCH --mem=64G   # or higher
```

---

## 📖 Documentation Quick Links

| Need | Document |
|------|----------|
| First time setup | [QUICKSTART.md](QUICKSTART.md) |
| Full reference | [README.md](README.md) |
| Error messages | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Interpret results | [RESULTS_GUIDE.md](RESULTS_GUIDE.md) |
| Copy-paste examples | [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) |
| Find anything | [INDEX.md](INDEX.md) |

---

## 📐 Key Metrics Reference

**Pipeline A (SCCA):**
- **r** (correlation): 0.1-0.5 typical, higher = stronger association
- **Sparsity**: 0.7-0.9 good (interpretable), <0.5 dense (hard to interpret)

**Pipeline B (Classification):**
- **AUC**: >0.8 excellent, 0.7-0.8 good, 0.6-0.7 moderate, <0.6 poor
- **AP**: Similar to AUC, use when labels imbalanced

---

## ⏱️ Expected Runtimes

| Pipeline | Runtime | Resources |
|----------|---------|-----------|
| Pipeline A | ~4 hours | 8 CPUs, 32GB RAM |
| Pipeline B | ~8 hours | 16 CPUs, 128GB RAM |

---

## 🆘 Getting Help

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for your specific error
2. Run `bash scripts/verify_setup.sh` to diagnose setup issues
3. Look at [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for similar scenarios
4. Contact maintainer with error log and job ID

---

**Print this card** or keep it open while running pipelines!
