# Changelog

All notable changes to the Gene-Brain CCA project will be documented in this file.

---

## [2.0.0] - 2026-01-14

### Added - Comprehensive Documentation & Results

**New Documentation:**
- `QUICKSTART.md` - Quick start guide for new users
- `INDEX.md` - Complete navigation guide for all documentation
- `CHANGELOG.md` - This file

**README.md Updates:**
- Added key results summary at top
- Added project structure diagram
- Added cohort overview (N=4,218)
- Added data representations table
- Added comparison to Yoon et al.
- Added scientific conclusions section
- Added future directions

**gene-brain-cca-2 Updates (v1.1.0):**
- Updated README.md with actual pipeline results
- Fixed RESULTS_GUIDE.md examples (gene >> fMRI, not reverse)
- Added CHANGELOG.md entry

### Key Findings Documented

| Experiment | Best AUC | Status |
|------------|----------|--------|
| Exp 1 Mean Pool | 0.588 | ✅ Completed |
| Exp 1 Max Pool | 0.505 | ✅ Completed |
| Exp 2 Pipeline A | r=0.156 | ✅ Completed |
| Exp 2 Pipeline B | **0.762** | ✅ Completed |

### Scientific Conclusions

1. **Full embeddings >> scalar reduction** (+29% AUC improvement)
2. **Gene >> fMRI for MDD prediction** (+36% relative)
3. **CCA/SCCA hurts prediction** (objective mismatch)
4. **Gene-brain coupling is diffuse** (sparsity < 10%)
5. **fMRI adds minimal/no predictive value** (+0.003 in fusion)

---

## [1.0.0] - 2026-01-13

### Added - gene-brain-cca-2 Subproject

**New Subproject:**
- `gene-brain-cca-2/` - Redesigned leakage-safe pipelines
- Pipeline A: Interpretable SCCA (111 genes × 180 ROIs)
- Pipeline B: Predictive suite with full 768-D embeddings

**Documentation:**
- Complete README with pipeline descriptions
- QUICKSTART.md for rapid deployment
- TROUBLESHOOTING.md for debugging
- RESULTS_GUIDE.md for interpretation
- USAGE_EXAMPLES.md for concrete scenarios

**Key Features:**
- Leakage-safe cross-validation
- Fold-wise Stage 1 fitting
- Train-only preprocessing
- Holdout evaluation

---

## [0.2.0] - 2026-01-10

### Added - Max Pooling Experiment

**New Pipeline:**
- `derived_max_pooling/` directory for max pooling results
- Comparison with mean pooling

**Results:**
- Max pooling: CC1 = 0.347, p = 0.995 (not significant)
- Max pooling: Gene-only AUC = 0.505 (near chance)
- **Conclusion:** Max pooling fails; mean pooling preferred

---

## [0.1.0] - 2026-01-08

### Added - Initial Two-Stage Pipeline

**Core Scripts:**
- `scripts/build_x_gene.py` - Gene embedding construction
- `scripts/build_x_fmri_fc.py` - fMRI FC construction
- `scripts/align_resid_pca.py` - Data alignment and PCA
- `scripts/run_cca.py` - CCA/SCCA Stage 1
- `scripts/stage2_predict.py` - Prediction Stage 2

**SLURM Scripts:**
- Full set of sbatch files for pipeline stages

**Results (Mean Pooling):**
- `derived_mean_pooling/` directory
- Stage 1: CC1 = 0.368, p = 0.040 (significant)
- Stage 2: Gene-only AUC = 0.588

**Data:**
- 4,218 subjects with both genetics and fMRI
- 111 genes (DNABERT-2 embeddings)
- 180 ROIs (HCP-MMP1 parcellation)

---

## How to Use This Changelog

When making changes:

1. **Add entry** under `[Unreleased]` section
2. **Categorize** changes:
   - `Added` - New features
   - `Changed` - Modifications
   - `Fixed` - Bug fixes
   - `Removed` - Deleted features
3. **Version bump** when releasing:
   - MAJOR: Breaking changes
   - MINOR: New features
   - PATCH: Bug fixes

---

**Format:** Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
**Versioning:** [Semantic Versioning](https://semver.org/)
