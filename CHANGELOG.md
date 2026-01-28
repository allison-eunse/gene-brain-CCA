# Changelog

All notable changes to the Gene-Brain CCA project will be documented in this file.

---

## [3.0.0] - 2026-01-28

### Added - Phase 3: Multi-FM Stratified Analysis

**New Experiments:**
- Stratified CCA benchmark comparing MDD vs Control coupling across 4 genomic FM models:
  - HyenaDNA, Caduceus, DNABERT2, Evo2
- Brain modalities tested: Schaefer7 FC, Schaefer17 FC, sMRI tabular, dMRI tabular
- sMRI/dMRI → MDD direct prediction pipeline

**New Scripts:**
- `scripts/run_stratified_coupling_benchmark.py` - MDD vs Control CCA with permutation testing
- `scripts/plot_stratified_results.py` - Visualization (bar, forest, heatmap, cosine plots)
- `scripts/analyze_evo2_weights.py` - FM weight analysis
- `slurm/51_stratified_fm.sbatch` - Multi-FM stratified job launcher
- `slurm/37_predictive_smri_dmri_tabular.sbatch` - sMRI/dMRI prediction

**New Documentation:**
- `FINAL_ANALYSIS_REPORT.md` - Comprehensive stratified analysis and Evo2 evaluation
- `cross_model_comparison.md` - FM model comparison across modalities
- `evo2_analysis_summary.md` - Evo2 weight analysis results

**New Figures:**
- `figures/stratified/stratified_results_bar.png` - Bar plot of coupling by group
- `figures/stratified/stratified_results_forest.png` - Forest plot with CIs
- `figures/stratified/stratified_results_heatmap.png` - FM × modality heatmap
- `figures/stratified/stratified_results_cosine.png` - Weight similarity analysis

**Key Results:**
| Finding | Evidence |
|---------|----------|
| No MDD-specific coupling | All Δr (MDD-Ctrl) ≈ 0, p > 0.05 |
| sMRI fails to predict MDD | AUC = 0.561 (near chance) |
| dMRI fails to predict MDD | AUC = 0.553 (near chance) |
| FM models behave similarly | No FM shows unique coupling |

**Completion Status:**
| FM Model | Schaefer7 | Schaefer17 | sMRI | dMRI |
|----------|-----------|------------|------|------|
| HyenaDNA | ✅ | ✅ | ✅ | ✅ |
| Caduceus | ✅ | ✅ | ✅ | ✅ |
| DNABERT2 | ✅ | ✅ | ✅ | ✅ |
| Evo2 | ⏳ | ⏳ | ⏳ | ✅ |

**Conclusions:**
1. Gene-brain coupling does NOT differ between MDD and Controls
2. Structural MRI (sMRI/dMRI) provides no MDD prediction signal
3. All genomic FM models yield similar (null) coupling results
4. The Evo2 p=0.049 result is a likely false positive (multiple testing)

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
