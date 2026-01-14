# gene-brain-cca-2

**Two pipelines on the 4,218 gene–fMRI overlap cohort:**

- **Pipeline A**: Interpretable SCCA (111 genes × 180 ROIs, no PCA)
- **Pipeline B**: Predictive wide gene embedding (111×768 → PCA512; baselines + CCA/SCCA)

Both pipelines use **fold-wise Stage 1 fitting** to avoid data leakage and provide reliable evaluation.

---

## 🔬 Key Results Summary

### Executive Summary

| Finding | Evidence |
|---------|----------|
| **Full embeddings >> scalar reduction** | AUC 0.762 vs 0.588 (+29% improvement) |
| **Mean pooling >> max pooling** | AUC 0.588 vs 0.505 (+16% improvement) |
| **Genetics >> fMRI for MDD prediction** | AUC 0.759 vs 0.559 (+36% relative improvement) |
| **CCA/SCCA hurts prediction** | AUC 0.546-0.566 vs 0.759-0.762 (direct supervised) |
| **Gene-brain coupling is diffuse** | SCCA sparsity < 10%; no localized biomarkers |

### Pipeline B Holdout Results (N=844)

| Model | AUC | Average Precision |
|-------|-----|-------------------|
| **gene_only_logreg** | **0.759** 🏆 | 0.596 |
| **early_fusion_logreg** | **0.762** 🏆 | 0.603 |
| gene_only_mlp | 0.751 | 0.623 |
| fmri_only_logreg | 0.559 | 0.453 |
| cca_joint_logreg | 0.546 | 0.454 |
| scca_joint_logreg | 0.566 | 0.480 |

### Pipeline A: SCCA Interpretability

- **Train CC1:** r = 0.156 (weak coupling)
- **Holdout CC1:** r = -0.005 (does not generalize)
- **Gene sparsity:** 8.2% (diffuse pattern)
- **fMRI sparsity:** 1.8% (diffuse pattern)

**Top genes (Component 0):** NR3C1, CTNND2, ZNF165, KCNK2, CSMD1

### Core Conclusion

> **Gene-brain correlation (unsupervised objective) does NOT translate into clinical prediction power (supervised objective).** Full foundation model embeddings substantially outperform scalar reductions for depression prediction. fMRI adds no predictive value beyond genetics.

---

## 📖 Documentation Navigation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[README.md](README.md)** (this file) | Complete reference guide | When you need full details |
| **[QUICKSTART.md](QUICKSTART.md)** | Get running in 5 minutes | First time setup |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Fix common problems | When something breaks |
| **[RESULTS_GUIDE.md](RESULTS_GUIDE.md)** | Interpret outputs | After pipelines complete |
| **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** | Concrete usage scenarios | Learning by example |

**Utility Scripts:**
- `scripts/verify_setup.sh` - Check prerequisites before running
- `scripts/view_results.py` - Quick results summary viewer

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Pipeline Overview](#pipeline-overview)
4. [Running the Pipelines](#running-the-pipelines)
5. [Output Files](#output-files)
6. [Understanding the Results](#understanding-the-results)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

**New to this project? See [QUICKSTART.md](QUICKSTART.md) for a streamlined guide.**

```bash
# 1. Navigate to project root
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# 2. (Optional but recommended) Verify setup
bash gene-brain-cca-2/scripts/verify_setup.sh

# 3. Activate conda environment
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# 4. Create logs directory (required for SLURM)
mkdir -p logs

# 5. Submit Pipeline A (Interpretable SCCA)
sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
# Runtime: ~4 hours, outputs to gene-brain-cca-2/derived/interpretable/

# 6. After Pipeline A completes, submit Pipeline B (Predictive Suite)
sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch
# Runtime: ~8 hours, outputs to gene-brain-cca-2/derived/wide_gene/

# 7. View results
python gene-brain-cca-2/scripts/view_results.py
```

**⚠️ Important**: Pipeline B depends on outputs from Pipeline A (specifically `ids_common.npy`), so run Pipeline A first.

**💡 Tip**: Monitor job progress with `squeue -u $USER` and `tail -f logs/<job_name>_<JOBID>.out`

---

## Prerequisites

### 1. Environment Setup

```bash
# Verify conda environment exists
conda env list | grep cca_env

# If needed, activate it manually:
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
```

### 2. Required Python Packages

The `cca_env` conda environment should have:
- `numpy`
- `scikit-learn`
- `cca-zoo` (for SCCA_PMD)

Verify installation:
```bash
python -c "from cca_zoo.linear import SCCA_PMD; print('cca-zoo OK')"
python -c "from sklearn.cross_decomposition import CCA; print('sklearn OK')"
```

### 3. Required Data Files

**For Pipeline A:**
- Genetics: `/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/`
  - `ids_gene.npy`
  - `X_gene_ng.npy` (N×111 gene scalar matrix)
- fMRI: `/storage/bigdata/UKB/fMRI/`
  - `fmri_eids_180.npy`
  - `fmri_X_180.npy` (N×180 ROI matrix)
- Covariates: `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/`
  - `iids.npy`, `labels.npy`
  - `covariates_age.npy`, `covariates_sex.npy`, `covariates_valid_mask.npy`

**For Pipeline B (in addition to above):**
- Gene embeddings: `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/`
  - Directory structure: `<gene_name>/embeddings_<1-49>_layer_last.npy`
- Gene list: `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt`

**Verify key files exist:**
```bash
# Quick verification script
for f in \
  /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/ids_gene.npy \
  /storage/bigdata/UKB/fMRI/fmri_eids_180.npy \
  /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy
do
  [ -f "$f" ] && echo "✓ $f" || echo "✗ MISSING: $f"
done
```

---

## Pipeline Overview

### Pipeline A: Interpretable SCCA

**Purpose:** Identify stable gene and brain ROI subsets with direct biological interpretability (no PCA mixing).

**Steps:**
1. **Prepare overlap** (`prepare_overlap_no_pca.py`):
   - Find 4,218 subjects present in both genetics and fMRI cohorts
   - Save **raw aligned** matrices + covariates for leakage-safe splitting
   - (Optional convenience outputs) residualize age/sex and z-score on the full overlap
   - Outputs (key):
     - `X_gene_raw.npy` (N×111), `X_fmri_raw.npy` (N×180)
     - `cov_age.npy`, `cov_sex.npy`, `labels_common.npy`, `ids_common.npy`
     - `X_gene_z.npy`, `X_fmri_z.npy` (convenience; not used for leakage-proof evaluation)

2. **Run SCCA** (`run_scca_interpretable.py`):
   - Fit SCCA with sparsity penalties (c1, c2)
   - Create a **stratified holdout** (default 20%) for one-touch generalization testing
   - Perform **CV on TRAIN only** to assess stability
   - Fit residualization + standardization on **TRAIN only** (then apply to val/holdout) to avoid covariate leakage
   - Report top genes/ROIs per component
   - Output: `scca_interpretable_results.json`

**Runtime:** ~4 hours (32GB RAM, 8 CPUs)

---

### Pipeline B: Predictive Wide Gene Embedding

**Purpose:** Evaluate whether using full 768-D gene embeddings (vs scalar reduction) improves predictive power.

**Steps:**
1. **Build wide gene matrix** (`build_x_gene_wide.py`):
   - Load 111 genes × 768 dimensions per subject (overlap only)
   - Output: `X_gene_wide.npy` (4,218 × 85,248)

2. **Predictive suite** (`run_predictive_suite.py`):
   - Baselines: gene-only, fMRI-only, early fusion (concat)
   - CCA + LogReg/MLP
   - SCCA + LogReg/MLP
   - **One-touch holdout** evaluation (default 20% holdout)
   - **Train-only residualization** (age/sex) for both modalities (applied to holdout)
   - **Train-only PCA** on the wide gene matrix (applied to holdout)
   - CV is run on TRAIN for tuning; Stage 1 is fit fold-wise
   - Output: `predictive_suite_results.json`

**Runtime:** ~8 hours (128GB RAM, 16 CPUs)

---

## Running the Pipelines

### Option 1: SLURM (Recommended)

```bash
# Check SLURM queue status
squeue -u $USER

# Submit Pipeline A
sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch

# Monitor job (replace JOBID with actual ID from sbatch output)
tail -f logs/interp_scca_<JOBID>.out

# After completion, verify output
ls -lh gene-brain-cca-2/derived/interpretable/

# Then submit Pipeline B
sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch
tail -f logs/wide_suite_<JOBID>.out
```

### Option 2: Interactive (for debugging)

```bash
# Activate environment
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Run Pipeline A step-by-step
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
mkdir -p gene-brain-cca-2/derived/interpretable

# Step 1: Prepare overlap data
python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py \
  --ids-gene derived_max_pooling/gene_x/ids_gene.npy \
  --x-gene derived_max_pooling/gene_x/X_gene_ng.npy \
  --ids-fmri /storage/bigdata/UKB/fMRI/fmri_eids_180.npy \
  --x-fmri /storage/bigdata/UKB/fMRI/fmri_X_180.npy \
  --cov-iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --cov-age /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_age.npy \
  --cov-sex /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_sex.npy \
  --cov-valid-mask /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_valid_mask.npy \
  --labels /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/labels.npy \
  --out-dir gene-brain-cca-2/derived/interpretable

# Step 2: Run SCCA
python gene-brain-cca-2/scripts/run_scca_interpretable.py \
  --x-gene-raw gene-brain-cca-2/derived/interpretable/X_gene_raw.npy \
  --x-fmri-raw gene-brain-cca-2/derived/interpretable/X_fmri_raw.npy \
  --cov-age gene-brain-cca-2/derived/interpretable/cov_age.npy \
  --cov-sex gene-brain-cca-2/derived/interpretable/cov_sex.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --ids gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --gene-names derived_max_pooling/gene_x/genes.npy \
  --c1 0.3 --c2 0.3 --k 10 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json
```

---

## Output Files

### Pipeline A Outputs

Located in `gene-brain-cca-2/derived/interpretable/`:

| File | Description | Size |
|------|-------------|------|
| `ids_common.npy` | 4,218 subject IDs (overlap cohort) | ~33 KB |
| `X_gene_raw.npy` | Raw gene matrix (4,218 × 111) | ~1.8 MB |
| `X_fmri_raw.npy` | Raw fMRI matrix (4,218 × 180) | ~3.0 MB |
| `cov_age.npy` | Age covariate (4,218,) | ~17 KB |
| `cov_sex.npy` | Sex covariate (4,218,) | ~17 KB |
| `X_gene_z.npy` | Z-scored gene matrix (4,218 × 111) | ~1.8 MB |
| `X_fmri_z.npy` | Z-scored fMRI matrix (4,218 × 180) | ~3.0 MB |
| `labels_common.npy` | Labels for classification | ~33 KB |
| `scca_interpretable_results.json` | SCCA metrics, sparsity, correlations | ~5 KB |
| `scca_interpretable_results_U_train.npy` | Gene variates (train) | varies |
| `scca_interpretable_results_V_train.npy` | fMRI variates (train) | varies |
| `scca_interpretable_results_U_holdout.npy` | Gene variates (holdout) | varies |
| `scca_interpretable_results_V_holdout.npy` | fMRI variates (holdout) | varies |
| `scca_interpretable_results_W_gene.npy` | Gene weights (111 × k) | varies |
| `scca_interpretable_results_W_fmri.npy` | ROI weights (180 × k) | varies |

### Pipeline B Outputs

Located in `gene-brain-cca-2/derived/wide_gene/`:

| File | Description | Size |
|------|-------------|------|
| `ids_gene_overlap.npy` | 4,218 subject IDs | ~33 KB |
| `X_gene_wide.npy` | Wide gene matrix (4,218 × 85,248) | ~1.4 GB |
| `predictive_suite_results.json` | All model metrics (AUC/AP) | ~3 KB |

---

## Understanding the Results

**For detailed interpretation, visualization tips, and statistical guidance, see [RESULTS_GUIDE.md](RESULTS_GUIDE.md).**

### Quick Results Viewer

```bash
# View formatted summary of all results
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
python gene-brain-cca-2/scripts/view_results.py

# View only Pipeline A or B
python gene-brain-cca-2/scripts/view_results.py --pipeline A
python gene-brain-cca-2/scripts/view_results.py --pipeline B
```

### Actual Results Interpretation

#### Pipeline A: SCCA Interpretability (Completed ✅)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Train CC1 | r = 0.156 | Weak gene-brain coupling |
| Holdout CC1 | r = -0.005 | **Does NOT generalize** |
| Gene sparsity | 8.2% | Diffuse (not localized) |
| fMRI sparsity | 1.8% | Diffuse (not localized) |

**Conclusion:** SCCA found a training pattern but it's **overfitting to noise** (holdout correlation ≈ 0).

**Top contributing genes (Component 0):**
1. NR3C1 (glucocorticoid receptor, HPA axis)
2. CTNND2 (cell adhesion, synaptic function)
3. ZNF165 (transcription factor)
4. KCNK2 (potassium channel)
5. CSMD1 (complement system)

#### Pipeline B: Predictive Suite (Completed ✅)

**Dataset:**
- Total: 4,218 subjects
- Train: 3,374 (80%)
- Holdout: 844 (20%)
- MDD prevalence: 41.1%
- PCA: 85,248 → 512 (91.8% variance retained)

**Holdout Performance:**

| Model | AUC | Rank | Interpretation |
|-------|-----|------|----------------|
| early_fusion_logreg | **0.762** | 🥇 | Marginal improvement over gene-only |
| gene_only_logreg | **0.759** | 🥈 | Best single-modality |
| gene_only_mlp | 0.751 | 🥉 | LogReg beats MLP |
| early_fusion_mlp | 0.710 | 4 | MLP overfits |
| scca_joint_logreg | 0.566 | 5 | CCA/SCCA hurts prediction |
| fmri_only_logreg | 0.559 | 6 | fMRI at chance level |
| cca_joint_logreg | 0.546 | 7 | CCA/SCCA hurts prediction |
| fmri_only_mlp | 0.543 | 8 | fMRI at chance level |
| cca_joint_mlp | 0.530 | 9 | Worst performer |
| scca_joint_mlp | 0.520 | 10 | Worst performer |

**Key Takeaways:**
1. **Gene >> fMRI:** 0.759 vs 0.559 (genetics is 36% better)
2. **Adding fMRI doesn't help:** early_fusion only +0.003 over gene-only
3. **CCA/SCCA hurts:** Unsupervised objectives don't align with prediction
4. **LogReg > MLP:** Simpler model generalizes better

### Reading Raw JSON Results (Manual)

```bash
# Pipeline A
cat gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json | python -m json.tool

# Pipeline B
cat gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json | python -m json.tool
```

**See [RESULTS_GUIDE.md](RESULTS_GUIDE.md) for:**
- Visualization examples (scree plots, bar charts, scatter plots)
- Statistical significance testing
- Common result patterns and what they mean
- Checklist for reporting findings

---

## Advanced Usage

### Modifying Hyperparameters

**SCCA sparsity penalties (Pipeline A):**

Edit `slurm/01_interpretable_scca.sbatch`:
```bash
--c1 0.5 --c2 0.5  # Increase for MORE sparsity (fewer selected features)
--c1 0.1 --c2 0.1  # Decrease for LESS sparsity (more features)
```

**PCA components (Pipeline B):**

Edit `slurm/02_predictive_wide_suite.sbatch`:
```bash
--n-components 1024  # More components = more gene info retained (but slower)
--n-components 256   # Fewer components = faster but more compression
```

**Cross-validation folds:**

Edit either sbatch script:
```bash
--n-folds 10  # More folds = more robust but slower
--n-folds 3   # Fewer folds = faster but less reliable
```

### Running on a Subset (for testing)

To test on a smaller sample (e.g., during debugging):

1. After `prepare_overlap_no_pca.py`, manually subset the arrays:
```python
import numpy as np
n_test = 500  # Use only 500 subjects
for f in ['X_gene_z', 'X_fmri_z', 'labels_common', 'ids_common']:
    arr = np.load(f'gene-brain-cca-2/derived/interpretable/{f}.npy', allow_pickle=True)
    np.save(f'gene-brain-cca-2/derived/interpretable/{f}_test.npy', arr[:n_test])
```

2. Update subsequent scripts to use `*_test.npy` files.

---

## Troubleshooting

**For comprehensive troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).**

### Common Issues (Quick Reference)

#### 1. **"No such file or directory" errors**

**Symptom:**
```
FileNotFoundError: /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/ids_gene.npy
```

**Solution:**
- Verify all prerequisite files exist (see [Prerequisites](#prerequisites))
- Check file paths are absolute (not relative)
- Ensure you ran any upstream data preparation scripts

---

#### 2. **"Install cca-zoo for SCCA_PMD" error**

**Symptom:**
```
SystemExit: Install cca-zoo for SCCA_PMD
```

**Solution:**
```bash
conda activate /scratch/connectome/allie/envs/cca_env
pip install cca-zoo
# Or if using conda:
conda install -c conda-forge cca-zoo
```

---

#### 3. **Out of Memory (OOM) errors**

**Symptom:**
```
slurmstepd: error: Detected 1 oom-kill event(s) in step XXXXX.batch
```

**Solution for Pipeline A:**
- Current allocation (32GB) should be sufficient
- If failing, increase in sbatch: `#SBATCH --mem=64G`

**Solution for Pipeline B:**
- Gene-wide matrix is ~85K dimensions; 128GB should handle it
- If failing, try reducing PCA components: `--n-components 256`
- Or run interactively on high-memory node

---

#### 4. **Pipeline B fails with "FileNotFoundError: ids_common.npy"**

**Symptom:**
```
FileNotFoundError: gene-brain-cca-2/derived/interpretable/ids_common.npy
```

**Solution:**
- Pipeline B depends on Pipeline A outputs
- Run Pipeline A first and verify completion:
  ```bash
  ls -lh gene-brain-cca-2/derived/interpretable/ids_common.npy
  ```

---

#### 5. **Job stuck in queue**

**Check job status:**
```bash
squeue -u $USER
```

**Cancel and resubmit if needed:**
```bash
scancel <JOBID>
sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
```

---

#### 6. **"Singular matrix" or convergence warnings**

**Symptom:**
```
LinAlgWarning: Ill-conditioned matrix (rcond=1.23e-18): result may not be accurate
```

**Causes:**
- Highly correlated features
- Insufficient sample size for dimensionality
- Extreme outliers

**Solutions:**
- For CCA: Reduce `--k` (number of components)
- For SCCA: Increase sparsity penalties `--c1`, `--c2`
- Check data quality (no NaNs/Infs):
  ```python
  import numpy as np
  X = np.load('X_gene_z.npy')
  print(f"NaNs: {np.isnan(X).sum()}, Infs: {np.isinf(X).sum()}")
  ```

---

#### 7. **Checking SLURM logs for errors**

```bash
# List recent log files
ls -lt logs/ | head

# View error log
cat logs/interp_scca_<JOBID>.err

# Monitor live output
tail -f logs/wide_suite_<JOBID>.out
```

---

### Getting Help

If issues persist:

1. **Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for detailed diagnostics
2. **Run the verification script:**
   ```bash
   bash gene-brain-cca-2/scripts/verify_setup.sh
   ```
3. **Check the SLURM error logs** in `logs/`
4. **Run interactively** to see full error messages (see [Option 2: Interactive](#option-2-interactive-for-debugging))
5. **Verify data integrity:**
   ```bash
   python -c "import numpy as np; X = np.load('path/to/file.npy'); print(X.shape, X.dtype)"
   ```
6. **Contact the maintainer** with:
   - Full error message
   - SLURM job ID
   - Which pipeline/step failed
   - Output from `verify_setup.sh`

---

## File Tree Summary

```
gene-brain-cca-2/
├── README.md                           ← You are here
├── scripts/
│   ├── prepare_overlap_no_pca.py       # Pipeline A step 1
│   ├── run_scca_interpretable.py       # Pipeline A step 2
│   ├── build_x_gene_wide.py            # Pipeline B step 1
│   ├── pca_gene_wide.py                # (Optional) manual PCA utility
│   └── run_predictive_suite.py         # Pipeline B step 2 (includes train-only PCA + holdout)
├── slurm/
│   ├── 01_interpretable_scca.sbatch    # Run Pipeline A
│   └── 02_predictive_wide_suite.sbatch # Run Pipeline B
└── derived/                             # Outputs (created on first run)
    ├── interpretable/                   # Pipeline A outputs
    │   ├── ids_common.npy
    │   ├── X_gene_raw.npy
    │   ├── X_fmri_raw.npy
    │   ├── cov_age.npy
    │   ├── cov_sex.npy
    │   ├── X_gene_z.npy                 # convenience (not leakage-proof)
    │   ├── X_fmri_z.npy                 # convenience (not leakage-proof)
    │   ├── labels_common.npy
    │   ├── scca_interpretable_results.json
    │   ├── scca_interpretable_results_U_train.npy
    │   ├── scca_interpretable_results_V_train.npy
    │   ├── scca_interpretable_results_U_holdout.npy
    │   ├── scca_interpretable_results_V_holdout.npy
    │   ├── scca_interpretable_results_W_gene.npy
    │   └── scca_interpretable_results_W_fmri.npy
    └── wide_gene/                       # Pipeline B outputs
        ├── ids_gene_overlap.npy
        ├── X_gene_wide.npy
        └── predictive_suite_results.json
```

---

## 🧬 Scientific Conclusions

### What Was Proven

| Finding | Evidence |
|---------|----------|
| ✅ Gene-brain coupling exists but is weak/diffuse | ρ=0.156-0.368, p=0.04 (mean pooling) |
| ✅ Unsupervised CCA/SCCA does NOT improve prediction | Joint 0.56 vs gene-only 0.76 |
| ✅ fMRI contributes no predictive value for MDD | AUC 0.50-0.56 across all experiments |
| ✅ Foundation model embeddings must be preserved | 1-D pooling: 0.59 → full: 0.76 |
| ✅ Mean pooling >> max pooling for scalar reduction | 0.59 vs 0.50 |

### Why fMRI Failed

Across ALL experiments: fMRI-only AUC 0.50-0.56 (near chance level)

**Possible explanations:**
1. **Genetic dominance for MDD:** Current evidence suggests stronger genetic than neuroimaging biomarkers
2. **Wrong brain features:** Used global 180-ROI HCP-MMP connectivity; MDD may be network-specific (DMN, Salience)
3. **Feature representation mismatch:** Genes use learned embeddings (DNABERT-2); fMRI uses raw correlations
4. **fMRI noise:** 10× more variable than genetics (head motion, scanner drift, state fluctuations)
5. **Causality direction:** Genetics → MDD (causal); brain connectivity ← MDD (consequence, not predictor)

### Why CCA/SCCA Underperformed

**Objective mismatch:**
```
Stage 1 (CCA) optimizes: maximize correlation(gene, brain)
Stage 2 (Prediction) needs: maximize correlation(features, MDD label)
```

These are **different objectives**. The patterns that co-vary between genes and brain are NOT the patterns that predict disease.

### Recommendations for Future Work

1. **Test Yoon's 38-gene panel:** Filter to curated MDD genes for higher signal-to-noise
2. **Remove PCA bottleneck:** Use LASSO/ElasticNet on full 85K features (expected AUC 0.78-0.82)
3. **Try Schaefer 400 parcellation:** Network-specific features (DMN, Salience) instead of whole-brain
4. **Use fMRI foundation models:** BrainLM or similar learned representations
5. **Implement 10-fold nested CV:** Match Yoon et al. evaluation for direct comparison

---

## Citation & Acknowledgments

This pipeline was developed for the UK Biobank gene-brain CCA project.

**Data sources:**
- Genetics: NESAP/Yoon cohort (N=28,932, 111 genes with DNABERT2 embeddings)
- fMRI: UK Biobank HCP-MMP1 180 ROI parcellation (N=40,792)
- Overlap: N=4,218 subjects (1,735 MDD cases, 2,483 controls; 41.1% prevalence)

**Comparison to Yoon et al.:**
| Study | Method | Training N | AUC |
|-------|--------|------------|-----|
| Yoon et al. | 10-fold nested CV | ~26,039/fold | **0.851** |
| This study | Single 80/20 holdout | 3,374 | **0.762** |

Gap primarily due to: sample size (7.7× smaller), PCA compression, gene panel (111 vs 38 curated)

**Key dependencies:**
- `scikit-learn` (PCA, CCA, classification)
- `cca-zoo` (SCCA_PMD)
- `numpy` (array operations)

---

**Last updated:** January 14, 2026
