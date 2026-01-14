# gene-brain-cca-2: Complete Documentation Index

Quick reference to all documentation and resources in this folder.

---

## 📚 Documentation Files

### Getting Started
1. **[QUICKSTART.md](QUICKSTART.md)** - Fastest way to get running
   - 5-minute setup checklist
   - Submit commands
   - Basic result viewing

2. **[README.md](README.md)** - Complete reference manual
   - Detailed pipeline descriptions
   - Prerequisites and data requirements
   - Installation and configuration
   - Advanced usage and hyperparameter tuning
   - Troubleshooting quick reference

### Working with Results
3. **[RESULTS_GUIDE.md](RESULTS_GUIDE.md)** - Understanding your outputs
   - Metric definitions (AUC, AP, canonical correlation, sparsity)
   - Statistical significance testing
   - Visualization examples (plots and charts)
   - Common result patterns and interpretations
   - Reporting checklist for publications

4. **[USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)** - Concrete usage scenarios
   - First-time user walkthrough
   - Re-running with different hyperparameters
   - Debugging failed jobs
   - Creating publication figures
   - Exporting results for external analysis
   - Grid search examples

### Problem Solving
5. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive debugging guide
   - Pre-submission issues (conda, SLURM, dependencies)
   - Pipeline-specific errors (A and B)
   - Data quality diagnostics
   - Performance optimization
   - Debugging strategies and workflows

---

## 🛠️ Utility Scripts

Located in `scripts/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `verify_setup.sh` | Pre-flight check for all prerequisites | `bash scripts/verify_setup.sh` |
| `view_results.py` | Format and display pipeline results | `python scripts/view_results.py` |

---

## 🚀 Pipeline Scripts

Located in `scripts/`:

### Pipeline A (Interpretable SCCA)
- `prepare_overlap_no_pca.py` - Align, residualize, z-score gene/fMRI data
- `run_scca_interpretable.py` - Run SCCA with sparsity penalties and CV

### Pipeline B (Predictive Suite)
- `build_x_gene_wide.py` - Build 111×768 gene embedding matrix
- `pca_gene_wide.py` - Reduce to PCA512 components
- `run_predictive_suite.py` - Run all baselines + CCA/SCCA models

---

## 📋 SLURM Job Scripts

Located in `slurm/`:

| Script | Pipeline | Runtime | Resources |
|--------|----------|---------|-----------|
| `01_interpretable_scca.sbatch` | Pipeline A | ~4h | 8 CPUs, 32GB RAM |
| `02_predictive_wide_suite.sbatch` | Pipeline B | ~8h | 16 CPUs, 128GB RAM |

---

## 📊 Output Files

Located in `derived/`:

### Pipeline A Outputs (`derived/interpretable/`)
- `ids_common.npy` - 4,218 overlap subject IDs
- `X_gene_z.npy` - Z-scored gene matrix (4,218 × 111)
- `X_fmri_z.npy` - Z-scored fMRI matrix (4,218 × 180)
- `labels_common.npy` - Classification labels
- `scca_interpretable_results.json` - Main results (correlations, sparsity, CV metrics)
- `scca_interpretable_results_U.npy` - Gene canonical variates (4,218 × 10)
- `scca_interpretable_results_V.npy` - Brain canonical variates (4,218 × 10)

### Pipeline B Outputs (`derived/wide_gene/`)
- `ids_gene_overlap.npy` - Subject IDs
- `X_gene_wide.npy` - Full gene embeddings (4,218 × 85,248) [~1.4 GB]
- `X_gene_pca512.npy` - PCA-reduced gene matrix (4,218 × 512)
- `predictive_suite_results.json` - All model AUCs/APs

---

## 📖 Quick Navigation by Task

### "I want to..."

**...get started quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...understand what each pipeline does**
→ [README.md](README.md) § Pipeline Overview

**...check if I have everything installed**
→ `bash scripts/verify_setup.sh`

**...submit my first job**
→ [QUICKSTART.md](QUICKSTART.md) § Launch Pipeline A

**...see my results**
→ `python scripts/view_results.py` or [RESULTS_GUIDE.md](RESULTS_GUIDE.md)

**...figure out why my job failed**
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md) § Debugging Strategies

**...change hyperparameters**
→ [README.md](README.md) § Advanced Usage

**...run specific examples**
→ [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md)

**...make plots for a paper**
→ [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) § Scenario 7

**...export data to R/CSV**
→ [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) § Scenario 8

**...understand what AUC/AP/sparsity mean**
→ [RESULTS_GUIDE.md](RESULTS_GUIDE.md) § Key Metrics

**...know if my results are good**
→ [RESULTS_GUIDE.md](RESULTS_GUIDE.md) § Common Patterns

---

## 🔗 External Data Dependencies

Documented in [README.md](README.md) § Prerequisites:

- **Genetics**: `/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/`
- **fMRI**: `/storage/bigdata/UKB/fMRI/fmri_*_180.npy`
- **Covariates**: `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/`
- **Gene embeddings**: `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/`

---

## 📞 Getting Help

1. Check the documentation (above)
2. Run `bash scripts/verify_setup.sh` to diagnose setup issues
3. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for specific error messages
4. Contact maintainer with:
   - Error message
   - Job ID
   - Output of verification script

---

## 🏗️ Folder Structure

```
gene-brain-cca-2/
├── README.md                           # Complete reference
├── QUICKSTART.md                       # 5-minute guide
├── TROUBLESHOOTING.md                  # Debugging help
├── RESULTS_GUIDE.md                    # Results interpretation
├── USAGE_EXAMPLES.md                   # Concrete examples
├── INDEX.md                            # This file
├── scripts/
│   ├── verify_setup.sh                 # Pre-flight check
│   ├── view_results.py                 # Results viewer
│   ├── prepare_overlap_no_pca.py       # Pipeline A step 1
│   ├── run_scca_interpretable.py       # Pipeline A step 2
│   ├── build_x_gene_wide.py            # Pipeline B step 1
│   ├── pca_gene_wide.py                # Pipeline B step 2
│   └── run_predictive_suite.py         # Pipeline B step 3
├── slurm/
│   ├── 01_interpretable_scca.sbatch    # Run Pipeline A
│   └── 02_predictive_wide_suite.sbatch # Run Pipeline B
└── derived/                             # Outputs (created on run)
    ├── interpretable/                   # Pipeline A outputs
    └── wide_gene/                       # Pipeline B outputs
```

---

**Last updated:** January 2026
