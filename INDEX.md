# Gene-Brain CCA: Complete Documentation Index

Quick reference to all documentation, scripts, and resources in this project.

---

## 🔬 Quick Results Summary

| Experiment | Method | Best AUC | Key Finding |
|------------|--------|----------|-------------|
| **Exp 1 Mean Pool** | CCA on scalars | 0.588 | Mean > Max pooling |
| **Exp 1 Max Pool** | CCA on scalars | 0.505 | Near chance |
| **Exp 2 Pipeline B** | Direct supervised | **0.762** 🏆 | Full embeddings win |

**Core insight:** Full foundation model embeddings >> scalar reduction. See `gene-brain-cca-2/` for best results.

---

## 📚 Documentation Files

### This Directory

| File | Purpose | When to Use |
|------|---------|-------------|
| **[README.md](README.md)** | Complete project overview | Understand the project |
| **[QUICKSTART.md](QUICKSTART.md)** | Get running fast | First time users |
| **[INDEX.md](INDEX.md)** | Navigation guide | Find what you need |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history | Track changes |

### Experiment 2 (Recommended)

| File | Purpose |
|------|---------|
| **[gene-brain-cca-2/README.md](gene-brain-cca-2/README.md)** | Best results documentation |
| **[gene-brain-cca-2/QUICKSTART.md](gene-brain-cca-2/QUICKSTART.md)** | Quick start for Exp 2 |
| **[gene-brain-cca-2/RESULTS_GUIDE.md](gene-brain-cca-2/RESULTS_GUIDE.md)** | Interpret results |
| **[gene-brain-cca-2/TROUBLESHOOTING.md](gene-brain-cca-2/TROUBLESHOOTING.md)** | Fix issues |

### Reports

| File | Purpose |
|------|---------|
| **[final_report/comprehensive_report.md](final_report/comprehensive_report.md)** | Full scientific analysis |

---

## 🗂️ Project Structure

```
gene-brain-CCA/
│
├── 📄 DOCUMENTATION
├── ─────────────────
├── README.md                     # Main project overview
├── QUICKSTART.md                 # Quick start guide
├── INDEX.md                      # This file
├── CHANGELOG.md                  # Version history
│
├── 📊 EXPERIMENT 1: Scalar Gene Reduction
├── ─────────────────────────────────────
├── scripts/                      # Pipeline scripts
│   ├── build_x_gene.py           # DNABERT2 → scalar (mean/max pooling)
│   ├── build_x_fmri_fc.py        # ROI timeseries → FC
│   ├── align_resid_pca.py        # Align, residualize, PCA
│   ├── run_cca.py                # Stage 1: CCA/SCCA
│   └── stage2_predict.py         # Stage 2: Prediction
├── slurm/                        # SLURM job scripts
│   ├── 00_fmri_fc.sbatch
│   ├── 01_gene_x.sbatch
│   ├── 02_align_pca.sbatch
│   ├── 04_cca_stage1.sbatch
│   ├── 05_scca_stage1.sbatch
│   ├── 06_stage2_predict.sbatch
│   └── 07_full_pipeline.sbatch   # Run everything
├── derived_mean_pooling/         # Results: mean pooling
│   ├── gene_x/                   # Gene features
│   ├── aligned_pca/              # Aligned matrices
│   ├── cca_stage1/               # CCA results
│   ├── scca_stage1/              # SCCA results
│   ├── stage2_cca/               # Prediction results
│   └── comparison/               # CCA vs SCCA comparison
├── derived_max_pooling/          # Results: max pooling
│
├── 📊 EXPERIMENT 2: Full Embeddings (RECOMMENDED)
├── ─────────────────────────────────────────────
├── gene-brain-cca-2/             # ⭐ Best results here
│   ├── README.md                 # Full documentation
│   ├── scripts/                  # Leakage-safe pipelines
│   ├── slurm/                    # SLURM launchers
│   └── derived/                  # Results (AUC 0.762)
│
├── 📄 REPORTS
├── ─────────────────
├── final_report/                 # Scientific analysis
│   ├── comprehensive_report.md   # Full technical report
│   └── *.pdf                     # PDF exports
│
└── logs/                         # SLURM logs
```

---

## 🚀 Quick Navigation by Task

### "I want to..."

**...get started quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...understand the project**
→ [README.md](README.md)

**...get the best prediction results**
→ [gene-brain-cca-2/README.md](gene-brain-cca-2/README.md) (AUC 0.762)

**...run the original pipeline**
→ [QUICKSTART.md](QUICKSTART.md) § Option B

**...compare mean vs max pooling**
→ Submit `slurm/07_full_pipeline.sbatch`

**...see interpretation of results**
→ [gene-brain-cca-2/RESULTS_GUIDE.md](gene-brain-cca-2/RESULTS_GUIDE.md)

**...read the scientific conclusions**
→ [final_report/comprehensive_report.md](final_report/comprehensive_report.md)

**...understand why fMRI failed**
→ [README.md](README.md) § Scientific Conclusions

**...fix a problem**
→ [gene-brain-cca-2/TROUBLESHOOTING.md](gene-brain-cca-2/TROUBLESHOOTING.md)

---

## 📊 Key Results by Experiment

### Experiment 1: Scalar Gene Reduction

**Mean Pooling (`derived_mean_pooling/`):**
- Stage 1 CC1: r = 0.368, p = 0.040 ✅
- Stage 2 Gene-only: AUC = 0.588
- Stage 2 Joint: AUC = 0.581
- Sparsity: 0% (diffuse pattern)

**Max Pooling (`derived_max_pooling/`):**
- Stage 1 CC1: r = 0.347, p = 0.995 ❌
- Stage 2 Gene-only: AUC = 0.505 (near chance)
- **Conclusion:** Max pooling fails

### Experiment 2: Full Embeddings

**Pipeline B (`gene-brain-cca-2/derived/wide_gene/`):**
- Gene-only: AUC = **0.759** 🏆
- Early fusion: AUC = 0.762
- CCA joint: AUC = 0.546 (hurts!)
- fMRI-only: AUC = 0.559 (near chance)

**Key insight:** Direct supervised learning on full embeddings >> two-stage CCA approach.

---

## 🛠️ Scripts Reference

### Experiment 1 Scripts (`scripts/`)

| Script | Purpose | Input → Output |
|--------|---------|----------------|
| `build_x_gene.py` | Create gene matrix | Embeddings → N × 111 |
| `build_x_fmri_fc.py` | Create FC matrix | Timeseries → N × 16,110 |
| `align_resid_pca.py` | Align + PCA | Raw → Aligned PCA |
| `run_cca.py` | Stage 1 CCA/SCCA | Aligned → Variates |
| `stage2_predict.py` | Stage 2 Prediction | Variates → AUC |

### Experiment 2 Scripts (`gene-brain-cca-2/scripts/`)

| Script | Purpose | Input → Output |
|--------|---------|----------------|
| `prepare_overlap_no_pca.py` | Prepare data | Raw → Aligned |
| `run_scca_interpretable.py` | Pipeline A | Aligned → SCCA |
| `build_x_gene_wide.py` | Full embeddings | Embeddings → N × 85,248 |
| `run_predictive_suite.py` | Pipeline B | Wide → AUC 0.762 |

---

## 📋 SLURM Jobs Reference

### Experiment 1 (`slurm/`)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `00_fmri_fc.sbatch` | Build fMRI features | ~1h |
| `01_gene_x.sbatch` | Build gene features | ~30m |
| `02_align_pca.sbatch` | Align + PCA | ~10m |
| `04_cca_stage1.sbatch` | CCA Stage 1 | ~30m |
| `05_scca_stage1.sbatch` | SCCA Stage 1 | ~30m |
| `06_stage2_predict.sbatch` | Stage 2 Prediction | ~1h |
| `07_full_pipeline.sbatch` | Full comparison | ~6h |

### Experiment 2 (`gene-brain-cca-2/slurm/`)

| Script | Purpose | Runtime |
|--------|---------|---------|
| `01_interpretable_scca.sbatch` | Pipeline A | ~4h |
| `02_predictive_wide_suite.sbatch` | Pipeline B | ~8h |

---

## 🔗 Data Dependencies

| Data | Location | Size |
|------|----------|------|
| Gene embeddings | `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/` | ~50 GB |
| fMRI ROI data | `/storage/bigdata/UKB/fMRI/` | ~100 GB |
| Covariates | `/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/` | ~10 MB |
| Gene list | `.../gene_list_filtered.txt` | 111 genes |

---

## 📞 Getting Help

1. **Check documentation:** README.md, QUICKSTART.md, INDEX.md
2. **Run verification:** `bash gene-brain-cca-2/scripts/verify_setup.sh`
3. **See troubleshooting:** [gene-brain-cca-2/TROUBLESHOOTING.md](gene-brain-cca-2/TROUBLESHOOTING.md)
4. **Check logs:** `logs/` directory

---

**Project Location:** `/storage/bigdata/UKB/fMRI/gene-brain-CCA/`  
**Last updated:** January 14, 2026
