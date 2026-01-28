# Gene-Brain CCA Pipeline

A two-stage pipeline for discovering gene-brain associations using **Canonical Correlation Analysis (CCA)** with UK Biobank data, and evaluating their utility for **Major Depressive Disorder (MDD)** prediction.

**Author:** Allie  
**Last Updated:** January 28, 2026  
**Dataset:** UK Biobank (N=3,374–7,116 with paired genetics + brain imaging)

---

## 🔬 Key Results

### Core Findings

| Finding | Evidence |
|---------|----------|
| **Full embeddings >> scalar reduction** | AUC 0.762 vs 0.588 (+29%) |
| **Genetics >> fMRI for MDD prediction** | AUC 0.759 vs 0.559 (+36%) |
| **CCA/SCCA hurts prediction** | AUC 0.55 vs 0.76 (direct supervised) |
| **Gene-brain coupling is diffuse** | SCCA sparsity < 10% |
| **fMRI adds minimal/no predictive value** | Early fusion +0.003 over gene-only |
| **sMRI/dMRI fail to predict MDD** | AUC 0.56/0.55 (near chance) |
| **No MDD vs Control coupling difference** | All stratified Δr ≈ 0, p > 0.05 |

### Multi-FM Stratified Analysis (NEW)

Tested 4 genomic foundation models × 4 brain modalities for MDD-specific coupling:

| FM Model | Schaefer7 | Schaefer17 | sMRI | dMRI |
|----------|-----------|------------|------|------|
| HyenaDNA | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 |
| Caduceus | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 |
| DNABERT2 | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 | ✅ r≈0 |
| Evo2 | ⏳ | ⏳ | ⏳ | ✅ r≈0 |

**Result**: No FM model shows significant MDD-specific gene-brain coupling.

> **Core Conclusion:** Gene-brain correlation (unsupervised) does NOT translate into clinical prediction power (supervised). Full foundation model embeddings substantially outperform scalar reductions. Neither functional nor structural brain imaging couples meaningfully with gene embeddings for MDD.

---

## 📖 Documentation Navigation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[README.md](README.md)** | Complete project overview | Understand the project |
| **[QUICKSTART.md](QUICKSTART.md)** | Get running in <10 minutes | First time users |
| **[INDEX.md](INDEX.md)** | Navigation guide | Find what you need |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history | Track changes |
| **[gene-brain-cca-2/](gene-brain-cca-2/README.md)** | Best results (AUC 0.762) | Run Experiment 2 |
| **[final_report/](final_report/comprehensive_report.md)** | Scientific analysis | Deep dive |

---

## Overview

This pipeline implements a rigorous framework for linking genetic embeddings (from foundation models like DNABERT2) to brain imaging features (fMRI functional connectivity), with downstream clinical prediction.

### Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Unsupervised Embedding (CCA / SCCA)                           │
│                                                                         │
│   Gene Features ──┐                                                     │
│   (N × p_gene)    ├──> CCA ──> Canonical Variates (U, V)               │
│   Brain Features ─┘            "Joint Embeddings"                       │
│   (N × p_brain)                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Supervised Prediction                                          │
│                                                                         │
│   Canonical Variates ──> Logistic Regression / MLP ──> Clinical Labels │
│   (U, V)                                                  (e.g., MDD)   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Conventional CCA vs Sparse CCA (SCCA)

| Feature | Conventional CCA | Sparse CCA (SCCA) |
|---------|------------------|-------------------|
| **Data Requirements** | Requires N ≫ p for stability | Can handle N ≪ p |
| **Noise Handling** | Sensitive to outliers | Denoises via L1 regularization |
| **Interpretability** | All features contribute | Identifies specific biomarkers |
| **UKB Utility** | Global patterns (40k N) | Localized clinical sub-networks |

**The Key Insight:** If SCCA outperforms CCA in Stage 2 prediction, it indicates the gene-brain relationship is driven by **localized, specific patterns** rather than a diffuse “everything contributes a little” effect.

**Important nuance (what “localized” means here):**

- **On the gene side (current setup)**: “localized” means **a subset of genes** carries most of the cross-modal coupling signal (because the genetics matrix is gene-level \(N\times G\), one scalar per gene).
- **On the brain side**: “localized” means **a subset of brain features** (e.g., ROIs or FC edges, depending on how you construct \(Y_{fmri}\)) drives the coupling.

So “localized vs global” is defined **in the feature space you feed into CCA/SCCA** (genes and ROIs/edges), not necessarily “one anatomical region vs whole brain” in a literal sense.

### Important note about genetics dimensionality (your current setup)

In this repo’s default genetics construction (`scripts/build_x_gene.py`), we **reduce each gene’s DNABERT2 embedding (768-D) to a single scalar per gene** (`--reduce mean|max|median`). That produces a **gene-level matrix**:

- `X_gene_ng.npy`: \(X_{gene} \in \mathbb{R}^{N \times G}\)
- where \(G\) is the number of genes in your gene list (**111** in `nesap-genomics/iids_labels_covariates/gene_list_filtered.txt`).

Because of this, PCA on the gene side is capped at \(G\): you cannot get 512 independent gene components from 111 gene features.

If you want sparsity/selection over **FM latent dimensions** instead (e.g., 768-D within-gene directions, or a 512-D genetic latent space), you would need to build a different genetics representation (e.g., keep per-gene embedding vectors and pool/concatenate before PCA).

## Project Structure

This project contains **three major experiment phases** organized as follows:

```
gene-brain-CCA/
│
├── 📊 PHASE 1: Original Two-Stage CCA (Scalar Gene Reduction)
├── ─────────────────────────────────────────────────────────
├── scripts/                      # Original pipeline scripts
│   ├── build_x_gene.py           # DNABERT2 → scalar gene matrix (111 features)
│   ├── build_x_fmri_fc.py        # ROI timeseries → FC vectors
│   ├── align_resid_pca.py        # Align subjects, residualize, PCA
│   ├── run_cca.py                # Stage 1: CCA / SCCA
│   └── stage2_predict.py         # Stage 2: Clinical prediction
├── slurm/                        # SLURM job scripts
├── derived_mean_pooling/         # Results: mean pooling (AUC 0.588)
├── derived_max_pooling/          # Results: max pooling (AUC 0.505)
│
├── 📊 PHASE 2: Leakage-Safe Pipelines with Full Embeddings
├── ─────────────────────────────────────────────────────────
├── gene-brain-cca-2/             # ⭐ RECOMMENDED: Redesigned pipelines
│   ├── README.md                 # Detailed documentation with results
│   ├── scripts/
│   │   ├── prepare_overlap_no_pca.py   # Pipeline A: data preparation
│   │   ├── run_scca_interpretable.py   # Pipeline A: interpretable SCCA
│   │   ├── build_x_gene_wide.py        # Pipeline B: full 768-D embeddings
│   │   └── run_predictive_suite.py     # Pipeline B: 10-model comparison
│   ├── slurm/
│   │   ├── 01_interpretable_scca.sbatch
│   │   └── 02_predictive_wide_suite.sbatch
│   └── derived/
│       ├── interpretable/        # Pipeline A outputs (SCCA weights)
│       ├── wide_gene/            # Pipeline B outputs (AUC 0.762 🏆)
│       └── stratified_fm/        # Phase 3 stratified results
│
├── 📊 PHASE 3: Multi-FM Stratified Analysis (NEW)
├── ─────────────────────────────────────────────────────────
├── scripts/
│   ├── run_stratified_coupling_benchmark.py  # MDD vs Ctrl CCA comparison
│   ├── plot_stratified_results.py            # Visualization
│   └── analyze_evo2_weights.py               # FM weight analysis
├── slurm/
│   ├── 51_stratified_fm.sbatch               # Multi-FM stratified jobs
│   └── 37_predictive_smri_dmri_tabular.sbatch # sMRI/dMRI prediction
├── figures/stratified/           # Result visualizations
│   ├── stratified_results_bar.png
│   ├── stratified_results_forest.png
│   ├── stratified_results_heatmap.png
│   └── stratified_results_cosine.png
│
├── 📄 REPORTS & DOCUMENTATION
├── ─────────────────────────────────────────────────────────
├── FINAL_ANALYSIS_REPORT.md      # Comprehensive stratified analysis
├── cross_model_comparison.md     # FM model comparison
├── evo2_analysis_summary.md      # Evo2 weight analysis
├── final_report/                 # Original analysis reports
│   ├── comprehensive_report.md
│   └── *.pdf
│
└── logs/                         # SLURM logs
```

### Experiment Overview

| Phase | Description | Best Metric | Key Finding |
|-------|-------------|-------------|-------------|
| **Phase 1 (Mean Pool)** | 768-D → 1 scalar/gene | AUC 0.588 | Mean > Max pooling |
| **Phase 1 (Max Pool)** | 768-D → 1 scalar/gene | AUC 0.505 | Near chance |
| **Phase 2 Pipeline A** | 111 scalars (SCCA) | r=0.16 | Coupling doesn't generalize |
| **Phase 2 Pipeline B** | 85,248 → PCA 512 | **AUC 0.762** 🏆 | Full embeddings win |
| **Phase 3 Stratified** | 4 FM × 4 modalities | Δr ≈ 0 | No MDD-specific coupling |
| **Phase 3 sMRI/dMRI** | Brain → MDD prediction | AUC 0.56 | Structural MRI uninformative |

### Recommended Workflow

1. **For new analyses:** Use `gene-brain-cca-2/` (Experiment 2)
2. **For reproduction:** Use `derived_mean_pooling/` or `derived_max_pooling/`
3. **For full documentation:** See `gene-brain-cca-2/README.md`

## Quick Start

### Prerequisites

```bash
pip install numpy scikit-learn
pip install cca-zoo  # Optional: for optimized Sparse CCA
```

### Full Pipeline (Single Job)

Submit the full comparison pipeline:

```bash
cd /path/to/gene-brain-CCA
mkdir -p derived logs

# Edit slurm/07_full_pipeline.sbatch to set paths
sbatch slurm/07_full_pipeline.sbatch
```

### Step-by-Step Execution

#### 1. Build Gene Features (DNABERT2 → N × G matrix)

```bash
python scripts/build_x_gene.py \
  --embed-root /path/to/DNABERT2_embedding_merged \
  --gene-list /path/to/gene_list_filtered.txt \
  --iids /path/to/iids.npy \
  --n-files 49 \
  --reduce mean \
  --out-dir derived/gene_x
```

**Outputs:** `ids_gene.npy`, `X_gene_ng.npy`, `genes.npy` (and optionally `X_gene_pca{k}.npy` if you pass `--pca512`, where \(k=\min(512, G)\); with the default gene list, \(k=111\)).

#### 2. Build fMRI Features (ROI → FC vectors)

```bash
python scripts/build_x_fmri_fc.py \
  --root /path/to/UKB_ROI \
  --glob "*_20227_2_0/hcp_mmp1_*.npy" \
  --out-dir derived/fmri_fc
```

**Outputs:** `ids_fmri.npy`, `X_fmri_fc.npy`

#### 3. Align Subjects + Residualize + PCA

```bash
python scripts/align_resid_pca.py \
  --ids-gene derived/gene_x/ids_gene.npy \
  --x-gene derived/gene_x/X_gene_ng.npy \
  --ids-fmri derived/fmri_fc/ids_fmri.npy \
  --x-fmri derived/fmri_fc/X_fmri_fc.npy \
  --cov-iids /path/to/iids.npy \
  --cov-age /path/to/covariates_age.npy \
  --cov-sex /path/to/covariates_sex.npy \
  --cov-valid-mask /path/to/covariates_valid_mask.npy \
  --pca-dim 512 \
  --out-dir derived/aligned_pca
```

**Outputs:** `ids_common.npy`, `X_gene_pca{gene_pca_dim}.npy`, `X_fmri_pca{fmri_pca_dim}.npy`, `pca_info.json`

To see the exact PCA dims written, inspect:

```bash
cat derived/aligned_pca/pca_info.json
```

#### 4. Stage 1: Run CCA (Conventional)

```bash
GENE_PCA_DIM=$(python3 -c "import json; print(json.load(open('derived/aligned_pca/pca_info.json'))['gene_pca_dim'])")
FMRI_PCA_DIM=$(python3 -c "import json; print(json.load(open('derived/aligned_pca/pca_info.json'))['fmri_pca_dim'])")

python scripts/run_cca.py \
  --x-gene derived/aligned_pca/X_gene_pca${GENE_PCA_DIM}.npy \
  --x-fmri derived/aligned_pca/X_fmri_pca${FMRI_PCA_DIM}.npy \
  --ids derived/aligned_pca/ids_common.npy \
  --method conventional \
  --n-components 10 \
  --n-perm 1000 \
  --out-dir derived/cca_stage1 \
  --prefix conventional_
```

**Outputs:**
- `conventional_U_gene.npy` - Gene canonical variates (N × k)
- `conventional_V_fmri.npy` - fMRI canonical variates (N × k)
- `conventional_W_gene.npy` - Gene loadings (for interpretation)
- `conventional_W_fmri.npy` - fMRI loadings
- `conventional_results.json` - Correlations and p-values

#### 5. Stage 1: Run SCCA (Sparse)

```bash
GENE_PCA_DIM=$(python3 -c "import json; print(json.load(open('derived/aligned_pca/pca_info.json'))['gene_pca_dim'])")
FMRI_PCA_DIM=$(python3 -c "import json; print(json.load(open('derived/aligned_pca/pca_info.json'))['fmri_pca_dim'])")

python scripts/run_cca.py \
  --x-gene derived/aligned_pca/X_gene_pca${GENE_PCA_DIM}.npy \
  --x-fmri derived/aligned_pca/X_fmri_pca${FMRI_PCA_DIM}.npy \
  --ids derived/aligned_pca/ids_common.npy \
  --method sparse \
  --n-components 10 \
  --c1 0.3 \
  --c2 0.3 \
  --n-perm 1000 \
  --out-dir derived/scca_stage1 \
  --prefix sparse_
```

**Sparsity parameters:**
- `--c1`: Gene sparsity (0 < c ≤ 1, lower = sparser)
- `--c2`: fMRI sparsity (0 < c ≤ 1, lower = sparser)

#### 6. Stage 2: Clinical Prediction

```bash
# For CCA embeddings
python scripts/stage2_predict.py \
  --u-gene derived/cca_stage1/conventional_U_gene.npy \
  --v-fmri derived/cca_stage1/conventional_V_fmri.npy \
  --labels /path/to/labels.npy \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds 5 \
  --out-dir derived/stage2_cca

# For SCCA embeddings
python scripts/stage2_predict.py \
  --u-gene derived/scca_stage1/sparse_U_gene.npy \
  --v-fmri derived/scca_stage1/sparse_V_fmri.npy \
  --labels /path/to/labels.npy \
  --models logreg mlp \
  --feature-sets gene_only fmri_only joint \
  --n-folds 5 \
  --out-dir derived/stage2_scca
```

**Outputs:**
- `stage2_results.json` - CV metrics (AUC, accuracy, F1)
- `oof_*.npy` - Out-of-fold predictions

## Interpreting Results

### Stage 1: Canonical Correlations

```python
import json
results = json.load(open("derived/cca_stage1/conventional_results.json"))
print(f"Canonical correlations: {results['canonical_correlations']}")
print(f"Permutation p-values: {results['p_perm']}")
```

### Stage 2: Prediction Performance

```python
results = json.load(open("derived/stage2_cca/cca_results.json"))
for r in results["results"]:
    print(f"{r['feature_set']:<15} {r['model']:<10} AUC={r['auc_mean']:.4f}")
```

### Key Comparisons

1. **Joint vs Unimodal:** If `joint > max(gene_only, fmri_only)`, the gene-brain coupling itself is predictive.

2. **SCCA vs CCA:** If SCCA outperforms CCA significantly (Δ > 0.02):
   - The relationship is driven by **localized patterns**
   - Examine `sparse_W_gene.npy` for important genes
   - Examine `sparse_W_fmri.npy` for important brain regions

### Identifying Biomarkers (SCCA)

```python
import numpy as np

# Load SCCA weights
W_gene = np.load("derived/scca_stage1/sparse_W_gene.npy")
genes = np.load("derived/gene_x/genes.npy", allow_pickle=True)

# Find genes with largest weights in component 1
comp = 0
top_k = 10
top_indices = np.argsort(np.abs(W_gene[:, comp]))[-top_k:][::-1]

print("Top genes contributing to canonical component 1:")
for idx in top_indices:
    print(f"  {genes[idx]}: weight={W_gene[idx, comp]:.4f}")
```

## Data Requirements

### Cohort Overview

```
Genetics cohort (NESAP/Yoon): 28,932 subjects
                                    ╲
                                     ╲ Overlap: 4,218 subjects
                                      ╲ (14.6% / 10.3%)
                                       ╳
                                      ╱
                                     ╱
fMRI cohort (UK Biobank):     40,792 subjects
```

**Final Analysis Cohort:**
- **N = 4,218** subjects with BOTH genetics AND fMRI
- **MDD Cases:** 1,735 (41.1%)
- **Controls:** 2,483 (58.9%)

### Input Data

| Data | Source | Format | Notes |
|------|--------|--------|-------|
| Gene embeddings | DNABERT2 | `(N × 768)` per gene | 111 genes from gene_list_filtered.txt |
| fMRI ROI | UK Biobank | `(T × 180)` per subject | HCP-MMP1 parcellation |
| Clinical labels | UK Biobank | `(N,)` binary | MDD diagnosis |
| Covariates | UK Biobank | `(N,)` each | Age, sex for residualization |

### Data Representations

| Representation | Shape | Used In | Description |
|---------------|-------|---------|-------------|
| **Scalar (Mean Pool)** | N × 111 | Exp 1 | Average 768-D → 1 per gene |
| **Scalar (Max Pool)** | N × 111 | Exp 1 | Maximum 768-D → 1 per gene |
| **Full Embeddings** | N × 85,248 | Exp 2 | 111 genes × 768 dimensions |
| **PCA Reduced** | N × 512 | Exp 2 | Full embeddings → PCA (91.8% variance) |
| **fMRI Connectivity** | N × 180 | Both | HCP-MMP1 ROI values |

### Alignment

The pipeline automatically handles subject alignment across modalities:
1. Gene embeddings aligned via `iids.npy`
2. fMRI subjects extracted from folder names
3. Intersection computed (N=4,218), covariates aligned
4. Stratified train/holdout split (80/20)

## Comparison to Yoon et al.

This study extends Yoon et al.'s gene-only MDD prediction by adding brain imaging:

| Study | Method | Training N | AUC | Notes |
|-------|--------|------------|-----|-------|
| **Yoon et al.** | 10-fold nested CV, 38 genes | ~26,039 | **0.851** | Reference |
| **This Study** | 80/20 holdout, 111 genes | 3,374 | **0.762** | +fMRI (no benefit) |

**Gap analysis:**
- Sample size (7.7× smaller): -0.06 to -0.08
- PCA compression: -0.02 to -0.03
- Gene panel (111 vs 38 curated): -0.02 to -0.04

**Key validation:** Our results confirm Yoon's approach—direct supervised learning on full embeddings outperforms unsupervised methods and multimodal fusion.

## References

- **Yoon et al.** - Gene foundation models for MDD classification
- **Witten et al. (2009)** - Penalized Matrix Decomposition for Sparse CCA
- **Hotelling (1936)** - Original CCA formulation
- **Glasser et al. (2016)** - HCP-MMP1 parcellation (180 ROIs)

## Lab Server Notes

- **CPU nodes:** node2, node4 (for CCA/SCCA)
- **GPU nodes:** node1, node3 (not needed for this pipeline)
- Shared filesystem: `/storage/bigdata/UKB/...`
- Always set `#SBATCH --chdir` to a shared path for compute node access

## Troubleshooting

### "cca-zoo not found"
The pipeline uses a fallback sparse CCA implementation. For best performance:
```bash
pip install cca-zoo
```

### Dimension mismatch errors
Ensure all inputs have the same row count:
```python
import numpy as np
Xg = sorted(__import__("pathlib").Path("derived/aligned_pca").glob("X_gene_pca*.npy"))[0]
Xb = sorted(__import__("pathlib").Path("derived/aligned_pca").glob("X_fmri_pca*.npy"))[0]
print("gene:", Xg, np.load(Xg, mmap_mode="r").shape)
print("fmri:", Xb, np.load(Xb, mmap_mode="r").shape)
```

### Low canonical correlations
1. Check that covariates are properly residualized
2. Increase `--pca-dim` to retain more variance
3. Try different gene reduction methods (`--reduce max` instead of `mean`)

### For Experiment 2 Issues

See detailed troubleshooting in:
- `gene-brain-cca-2/README.md` - Full documentation
- `gene-brain-cca-2/TROUBLESHOOTING.md` - Common issues and solutions

---

## Future Directions

Based on current results, recommended next steps:

1. **Test Yoon's 38-gene panel** - Filter to curated MDD genes for higher signal-to-noise
2. **Remove PCA bottleneck** - Use LASSO/ElasticNet on full 85K features
3. **Try Schaefer 400 parcellation** - Network-specific features (DMN, Salience)
4. **Explore fMRI foundation models** - BrainLM or similar learned representations
5. **Implement 10-fold nested CV** - Match Yoon et al. evaluation methodology

---

**Project Location:** `/storage/bigdata/UKB/fMRI/gene-brain-CCA/`