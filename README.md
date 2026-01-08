# Gene-Brain CCA Pipeline

A two-stage pipeline for discovering gene-brain associations using **Canonical Correlation Analysis (CCA)** with UK Biobank data.

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

## Directory Structure

```
gene-brain-CCA/
├── scripts/
│   ├── build_x_gene.py        # DNABERT2 embeddings → gene matrix
│   ├── build_x_fmri_fc.py     # ROI timeseries → FC vectors
│   ├── align_resid_pca.py     # Align subjects, residualize, PCA
│   ├── run_cca.py             # Stage 1: CCA / SCCA
│   ├── stage2_predict.py      # Stage 2: Clinical prediction
│   └── run_cca_permute.py     # (Legacy) CCA with permutation tests
├── slurm/
│   ├── 00_fmri_fc.sbatch      # Build fMRI features
│   ├── 01_gene_x.sbatch       # Build gene features
│   ├── 02_align_pca.sbatch    # Align and PCA
│   ├── 03_cca_perm.sbatch     # (Legacy) Basic CCA
│   ├── 04_cca_stage1.sbatch   # Stage 1: Conventional CCA
│   ├── 05_scca_stage1.sbatch  # Stage 1: Sparse CCA
│   ├── 06_stage2_predict.sbatch  # Stage 2: Prediction
│   └── 07_full_pipeline.sbatch   # Full comparison pipeline
├── derived/                   # Generated outputs (safe to rsync)
└── logs/                      # Slurm logs
```

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

### Input Data

| Data | Source | Format | Notes |
|------|--------|--------|-------|
| Gene embeddings | DNABERT2/Caduceus | `(N × D)` per gene | Chunked as `embeddings_k_layer_last.npy` |
| fMRI ROI | UK Biobank | `(T × R)` per subject | HCP-MMP1 (180 ROIs) or Schaefer |
| Clinical labels | UK Biobank | `(N,)` binary | MDD, PHQ-9 threshold, etc. |
| Covariates | UK Biobank | `(N,)` each | Age, sex for residualization |

### Alignment

The pipeline automatically handles subject alignment across modalities:
1. Gene embeddings aligned via `iids.npy`
2. fMRI subjects extracted from folder names
3. Intersection computed, covariates aligned

## References

- **Yoon et al.** - Gene foundation models for MDD classification (max pooling > mean pooling)
- **Witten et al. (2009)** - Penalized Matrix Decomposition for Sparse CCA
- **Hotelling (1936)** - Original CCA formulation

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
