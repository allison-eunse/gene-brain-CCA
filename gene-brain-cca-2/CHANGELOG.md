# Changelog

All notable changes to the gene-brain-cca-2 pipeline will be documented in this file.

---

## [1.0.0] - 2026-01-13

### Added - Initial Release

**Documentation:**
- `README.md` - Complete reference guide with prerequisites, pipeline descriptions, advanced usage
- `QUICKSTART.md` - 5-minute quick start guide for new users
- `TROUBLESHOOTING.md` - Comprehensive debugging and problem-solving guide
- `RESULTS_GUIDE.md` - Detailed guide for interpreting outputs and creating visualizations
- `USAGE_EXAMPLES.md` - Nine concrete usage scenarios with copy-paste commands
- `INDEX.md` - Navigation guide for all documentation

**Utility Scripts:**
- `scripts/verify_setup.sh` - Pre-flight verification script to check all prerequisites
- `scripts/view_results.py` - Interactive results viewer with formatted summaries

**Pipeline Scripts:**
- `scripts/prepare_overlap_no_pca.py` - Data alignment and preprocessing for Pipeline A
- `scripts/run_scca_interpretable.py` - Interpretable SCCA with cross-validation
- `scripts/build_x_gene_wide.py` - Wide gene embedding matrix construction
- `scripts/pca_gene_wide.py` - Dimensionality reduction via PCA
- `scripts/run_predictive_suite.py` - Comprehensive predictive modeling suite

**SLURM Scripts:**
- `slurm/01_interpretable_scca.sbatch` - Pipeline A launcher (4h, 32GB)
- `slurm/02_predictive_wide_suite.sbatch` - Pipeline B launcher (8h, 128GB)

**Features:**
- Leakage-safe cross-validation with fold-wise Stage 1 fitting
- Support for 4,218 gene-fMRI overlap subjects
- Interpretable mode: 111 genes × 180 ROIs (no PCA mixing)
- Predictive mode: 111×768 gene embeddings → PCA512 → baselines + CCA/SCCA
- JSON result outputs for easy parsing
- Comprehensive error handling and logging

### Documentation Structure

```
gene-brain-cca-2/
├── README.md                    # Complete reference (12,500 words)
├── QUICKSTART.md               # Quick start (1,200 words)
├── TROUBLESHOOTING.md          # Problem solving (8,000 words)
├── RESULTS_GUIDE.md            # Interpretation (6,500 words)
├── USAGE_EXAMPLES.md           # Concrete examples (4,000 words)
├── INDEX.md                    # Navigation guide
├── CHANGELOG.md                # This file
├── scripts/                    # 7 executable scripts
├── slurm/                      # 2 SLURM launchers
└── derived/                    # Output directory (created on run)
```

### Known Limitations

- Pipeline B requires Pipeline A outputs (`ids_common.npy`)
- Wide gene matrix (85K dims) requires significant memory (~128GB for PCA)
- No GPU acceleration (CPU-only implementation)
- Feature weight extraction requires manual modification of `run_scca_interpretable.py`

### Future Enhancements (Planned)

- Save SCCA feature weights (Wx, Wy) by default
- Add holdout evaluation mode (in addition to CV)
- Support for custom gene lists (subset of 111)
- Automated grid search wrapper script
- Integration with Weights & Biases for experiment tracking

---

## How to Use This Changelog

When making changes to the pipeline:

1. **Add entry** under `[Unreleased]` section (create if doesn't exist)
2. **Categorize** changes:
   - `Added` - New features, scripts, documentation
   - `Changed` - Modifications to existing functionality
   - `Deprecated` - Features scheduled for removal
   - `Removed` - Deleted features or files
   - `Fixed` - Bug fixes
   - `Security` - Vulnerability patches
3. **Version bump** when releasing:
   - MAJOR (X.0.0): Breaking changes
   - MINOR (0.X.0): New features, backward-compatible
   - PATCH (0.0.X): Bug fixes only

---

**Format:** Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
**Versioning:** [Semantic Versioning](https://semver.org/)
