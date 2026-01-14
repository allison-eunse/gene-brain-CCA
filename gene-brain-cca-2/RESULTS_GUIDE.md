# Results Interpretation Guide

This guide helps you understand and interpret the outputs from both pipelines.

---

## Table of Contents

1. [Pipeline A: Interpretable SCCA Results](#pipeline-a-interpretable-scca-results)
2. [Pipeline B: Predictive Suite Results](#pipeline-b-predictive-suite-results)
3. [Comparing Results](#comparing-results)
4. [Statistical Significance](#statistical-significance)
5. [Visualization Tips](#visualization-tips)
6. [Common Patterns & What They Mean](#common-patterns--what-they-mean)

---

## Pipeline A: Interpretable SCCA Results

### Output File

```bash
cat gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json | python -m json.tool
```

### Key Metrics

**1. Canonical Correlations (`r`)**

```json
"r": [0.342, 0.287, 0.213, ...]
```

- **What it is:** Correlation between gene and brain components
- **Range:** -1 to +1 (typically 0.1 to 0.5 for biological data)
- **Interpretation:**
  - `r > 0.3`: Strong gene-brain association
  - `r = 0.2-0.3`: Moderate association
  - `r < 0.2`: Weak association
  - `r < 0.1`: Negligible association

**Example:**
```json
"full": {
  "r": [0.342, 0.287, 0.213, 0.189, 0.156, ...]
}
```
→ First component captures 34% shared variance, second 29%, etc. (note: r² gives variance explained)

---

**2. Sparsity (`sparsity_gene`, `sparsity_fmri`)**

```json
"sparsity_gene": 0.73,
"sparsity_fmri": 0.82
```

- **What it is:** Fraction of features with near-zero weights (|w| < 1e-3)
- **Range:** 0 to 1
- **Interpretation:**
  - `> 0.9`: Very sparse (only ~10% features selected) → **highly interpretable**
  - `0.7-0.9`: Moderately sparse → **somewhat interpretable**
  - `< 0.5`: Dense → **hard to interpret**

**Example:**
```json
"sparsity_gene": 0.73,
"sparsity_fmri": 0.82
```
→ Only ~27% of genes (30/111) and ~18% of ROIs (32/180) are selected → **good interpretability**

---

**3. Cross-Validation Stability**

```json
"folds": [
  {"fold": 0, "r": [0.35, 0.29, ...], "sparsity_gene": 0.71, ...},
  {"fold": 1, "r": [0.33, 0.28, ...], "sparsity_gene": 0.75, ...},
  ...
]
```

- **What to check:** Consistency across folds
- **Good sign:** Similar `r` and sparsity values across folds
- **Bad sign:** High variance (e.g., fold 0: r=0.35, fold 1: r=0.15)

**How to assess:**
```python
import json, numpy as np
with open('scca_interpretable_results.json') as f:
    res = json.load(f)
r_values = [fold['r'][0] for fold in res['folds']]  # First component
print(f"Mean r: {np.mean(r_values):.3f} ± {np.std(r_values):.3f}")
# Good: ± < 0.05
# Unstable: ± > 0.1
```

---

### Extracting Top Features

**Gene Weights:**
```python
import numpy as np

# Load full-dataset canonical variates
U = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_U.npy')
V = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_V.npy')

# To get weights, you need to re-fit or save them separately
# For now, U and V give you the transformed data (scores)
print(f"Gene scores shape: {U.shape}")  # (4218, 10)
print(f"Brain scores shape: {V.shape}")  # (4218, 10)

# Correlation between first pair of components
from scipy.stats import pearsonr
r, p = pearsonr(U[:, 0], V[:, 0])
print(f"Component 1 correlation: r={r:.3f}, p={p:.3e}")
```

**Note:** To get actual feature weights (which genes/ROIs contribute), you'd need to save the `Wx` and `Wy` matrices from `run_scca_interpretable.py`. Consider adding:
```python
np.save(Path(args.out_json).with_suffix("_Wx.npy"), Wx.astype(np.float32))
np.save(Path(args.out_json).with_suffix("_Wy.npy"), Wy.astype(np.float32))
```

---

## Pipeline B: Predictive Suite Results

### Output File

```bash
cat gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json | python -m json.tool
```

### Key Metrics

**1. Area Under ROC Curve (AUC)**

```json
"gene_only_logreg": {"auc": 0.63, "ap": 0.58},
"fmri_only_logreg": {"auc": 0.71, "ap": 0.65},
...
```

- **What it is:** Probability that a randomly chosen positive case ranks higher than a negative case
- **Range:** 0.5 (random) to 1.0 (perfect)
- **Interpretation:**
  - `AUC > 0.8`: Excellent discrimination
  - `AUC = 0.7-0.8`: Good discrimination
  - `AUC = 0.6-0.7`: Moderate discrimination
  - `AUC < 0.6`: Poor discrimination
  - `AUC ≈ 0.5`: No better than random guessing

---

**2. Average Precision (AP)**

```json
"ap": 0.65
```

- **What it is:** Area under precision-recall curve (better for imbalanced data)
- **Range:** 0 to 1
- **Interpretation:** Similar to AUC, but more sensitive to performance on positive class
- **When to use:** Especially important if labels are imbalanced (e.g., 10% positive)

**Check label balance:**
```python
import numpy as np
y = np.load('gene-brain-cca-2/derived/interpretable/labels_common.npy')
pos_rate = y.mean()
print(f"Positive rate: {pos_rate:.2%}")
# If < 20% or > 80%, prioritize AP over AUC
```

---

### Model Comparisons

**Actual results from this study (holdout N=844):**
```json
{
  "gene_only_logreg": {"auc": 0.759, "ap": 0.596},
  "gene_only_mlp": {"auc": 0.751, "ap": 0.623},
  "fmri_only_logreg": {"auc": 0.559, "ap": 0.453},
  "fmri_only_mlp": {"auc": 0.543, "ap": 0.462},
  "early_fusion_logreg": {"auc": 0.762, "ap": 0.603},
  "early_fusion_mlp": {"auc": 0.710, "ap": 0.560},
  "cca_joint_logreg": {"auc": 0.546, "ap": 0.454},
  "cca_joint_mlp": {"auc": 0.530, "ap": 0.454},
  "scca_joint_logreg": {"auc": 0.566, "ap": 0.480},
  "scca_joint_mlp": {"auc": 0.520, "ap": 0.425}
}
```

**Key findings:**
- **Gene >> fMRI**: 0.759 vs 0.559 (genetics vastly outperforms brain imaging)
- **CCA/SCCA hurts**: 0.55 vs 0.76 (unsupervised reduces performance)
- **Fusion marginal**: 0.762 vs 0.759 (fMRI adds only +0.003)

---

### Questions to Answer

#### Q1: Which modality is more predictive?

**Compare:**
- `gene_only_logreg` vs `fmri_only_logreg`

**Actual results:**
```
Gene-only: AUC=0.759
fMRI-only: AUC=0.559
```
→ **Gene data is MUCH more predictive** (0.759 vs 0.559 = 36% relative improvement)

**Interpretation:** For MDD prediction, foundation model gene embeddings vastly outperform brain functional connectivity. This aligns with Yoon et al.'s findings that genetic features are strong MDD predictors.

---

#### Q2: Does fusion help?

**Compare:**
- Best single modality vs `early_fusion_logreg`

**Actual results:**
```
Gene-only: AUC=0.759
Early fusion: AUC=0.762
```
→ **Negligible improvement** (0.759 → 0.762 = +0.003 absolute)

**Interpretation:**
- Large gain (> 5%): Modalities are complementary
- Small gain (1-3%): Modalities overlap in information
- **No gain (< 1%):** Features are redundant; fMRI adds no value beyond genetics

**This study found:** fMRI provides essentially no additional predictive value for MDD when combined with genetic embeddings.

---

#### Q3: Does CCA/SCCA add value?

**Compare:**
- `gene_only_logreg` vs `cca_joint_logreg` vs `scca_joint_logreg`

**Actual results:**
```
Gene-only (direct): AUC=0.759
CCA joint: AUC=0.546
SCCA joint: AUC=0.566
```
→ **CCA/SCCA dramatically HURT performance** (-0.19 to -0.21 absolute)

**Why CCA/SCCA failed in this study:**
1. **Objective mismatch:** CCA optimizes gene-brain correlation, NOT MDD prediction
2. **Information loss:** Reducing to 10 canonical variates loses predictive signal
3. **fMRI is noise:** Brain features are near chance-level, so finding gene-brain correlation finds non-predictive patterns

**Key insight:** Unsupervised dimensionality reduction (CCA) does NOT align with supervised prediction objectives. Direct supervised learning vastly outperforms the two-stage approach.

---

#### Q4: LogReg vs MLP?

**Compare:**
- `*_logreg` vs `*_mlp` for each setting

**Example:**
```
Early fusion LogReg: AUC=0.75
Early fusion MLP: AUC=0.77
```

**Interpretation:**
- MLP > LogReg: Nonlinear patterns exist
- LogReg ≈ MLP: Linear model sufficient (prefer LogReg for interpretability)
- MLP < LogReg: Overfitting (MLP has more parameters)

---

## Comparing Results

### Pipeline A vs Pipeline B

| Aspect | Pipeline A | Pipeline B |
|--------|-----------|-----------|
| **Purpose** | Find interpretable gene-brain associations | Predict labels from multimodal data |
| **Metric** | Canonical correlation (r) | Classification AUC/AP |
| **Gene repr.** | 111 scalars (interpretable) | 512 PCs from 111×768 (richer) |
| **Key question** | Which genes/ROIs co-vary? | How well can we predict? |

**They answer different questions!**

- **High r, low AUC:** Strong gene-brain correlation, but not predictive of label
- **Low r, high AUC:** Weak correlation, but predictive features exist (possibly nonlinear)

---

### Making a Summary Table

**Template:**
```python
import json

# Load results
with open('gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json') as f:
    res_a = json.load(f)
with open('gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json') as f:
    res_b = json.load(f)

# Pipeline A summary
r_mean = sum(res_a['full']['r'][:3]) / 3  # Top 3 components
print(f"Pipeline A - Avg canonical correlation (top 3): {r_mean:.3f}")
print(f"  Gene sparsity: {res_a['full']['sparsity_gene']:.2%}")
print(f"  fMRI sparsity: {res_a['full']['sparsity_fmri']:.2%}")

# Pipeline B summary
print(f"\nPipeline B - Classification AUC:")
for key in ['gene_only_logreg', 'fmri_only_logreg', 'early_fusion_logreg', 'cca_joint_logreg']:
    auc = res_b[key]['auc']
    print(f"  {key}: {auc:.3f}")
```

---

## Statistical Significance

### Are Differences Meaningful?

**Rule of thumb for AUC differences:**
- Δ < 0.01: Negligible (likely noise)
- Δ = 0.01-0.03: Small but possibly meaningful
- Δ > 0.05: Substantial

**Formal test (DeLong test for AUC):**
```python
from scipy.stats import ttest_rel
import numpy as np

# If you saved fold-wise predictions, compare paired AUCs
# Example: gene_only vs fmri_only across 5 folds
gene_aucs = [0.57, 0.59, 0.58, 0.56, 0.60]
fmri_aucs = [0.71, 0.73, 0.72, 0.74, 0.70]

t, p = ttest_rel(gene_aucs, fmri_aucs)
print(f"Paired t-test: t={t:.2f}, p={p:.3f}")
# p < 0.05 → significant difference
```

**For canonical correlations:**
- Use permutation test (shuffle one modality, recompute r, repeat 1000×)
- Compare observed r to null distribution

---

## Visualization Tips

### 1. Canonical Correlation Scree Plot

**Pipeline A:**
```python
import matplotlib.pyplot as plt
import json

with open('scca_interpretable_results.json') as f:
    res = json.load(f)

r = res['full']['r']
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(r)+1), r)
plt.xlabel('Component')
plt.ylabel('Canonical Correlation')
plt.title('SCCA Component Strength')
plt.axhline(y=0.1, color='r', linestyle='--', label='r=0.1 threshold')
plt.legend()
plt.tight_layout()
plt.savefig('scca_scree.png', dpi=150)
```

---

### 2. AUC Comparison Bar Chart

**Pipeline B:**
```python
import matplotlib.pyplot as plt
import json

with open('predictive_suite_results.json') as f:
    res = json.load(f)

models = ['gene_only_logreg', 'fmri_only_logreg', 'early_fusion_logreg', 
          'cca_joint_logreg', 'scca_joint_logreg']
aucs = [res[m]['auc'] for m in models]
labels = ['Gene\nOnly', 'fMRI\nOnly', 'Early\nFusion', 'CCA\nJoint', 'SCCA\nJoint']

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, aucs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum'])
plt.ylabel('AUC')
plt.title('Classification Performance (LogReg)')
plt.ylim([0.5, max(aucs) + 0.1])
plt.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')

# Add value labels on bars
for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{auc:.3f}', ha='center', va='bottom')

plt.legend()
plt.tight_layout()
plt.savefig('auc_comparison.png', dpi=150)
```

---

### 3. Scatter Plot: Gene vs Brain Scores

**Pipeline A:**
```python
import numpy as np
import matplotlib.pyplot as plt

U = np.load('scca_interpretable_results_U.npy')
V = np.load('scca_interpretable_results_V.npy')
y = np.load('../labels_common.npy')

plt.figure(figsize=(8, 6))
plt.scatter(U[:, 0], V[:, 0], c=y, cmap='coolwarm', alpha=0.5, s=10)
plt.xlabel('Gene Component 1')
plt.ylabel('Brain Component 1')
plt.title('SCCA Component 1: Gene-Brain Projection')
plt.colorbar(label='Label')
plt.tight_layout()
plt.savefig('scca_component1.png', dpi=150)
```

---

## Common Patterns & What They Mean

### Pattern 1: High Sparsity but Low Correlation

**Observation:**
```json
"r": [0.15, 0.12, ...],
"sparsity_gene": 0.95,
"sparsity_fmri": 0.93
```

**Meaning:**
- SCCA selected very few features (good for interpretability)
- But weak correlation → those features don't co-vary strongly
- **Possible cause:** Over-regularization (c1/c2 too high)
- **Action:** Try lower sparsity penalties (e.g., c1=c2=0.1)

---

### Pattern 2: Low Sparsity with High Correlation

**Observation:**
```json
"r": [0.42, 0.38, ...],
"sparsity_gene": 0.20,
"sparsity_fmri": 0.15
```

**Meaning:**
- Strong gene-brain association (good!)
- But using most features (80-85%) → hard to interpret
- **Trade-off:** Correlation vs interpretability
- **Action:** Increase penalties for more focused patterns, accept lower r

---

### Pattern 3: Gene-Only Matches or Outperforms Early Fusion (THIS STUDY)

**Actual observation:**
```json
"gene_only_logreg": {"auc": 0.759},
"early_fusion_logreg": {"auc": 0.762}
```

**Meaning:**
- Brain features add almost no information (+0.003)
- fMRI may add noise or redundant patterns
- **This study's cause:** fMRI is near chance-level (0.559), so it contributes nothing

---

### Pattern 4: MLP Much Worse Than LogReg

**Observation:**
```json
"early_fusion_logreg": {"auc": 0.75},
"early_fusion_mlp": {"auc": 0.61}
```

**Meaning:**
- MLP overfitting (more parameters, harder to regularize)
- Early stopping may have kicked in too early
- **Action:** Tune MLP hyperparameters (hidden size, learning rate) or stick with LogReg

---

### Pattern 5: All Models Near Chance (AUC ≈ 0.50-0.55)

**Observation:**
```json
"gene_only_logreg": {"auc": 0.52},
"fmri_only_logreg": {"auc": 0.53},
"early_fusion_logreg": {"auc": 0.54}
```

**Meaning:**
- Labels not predictable from gene/brain features
- Possible issues:
  - Wrong label file
  - Features lack signal for this task
  - Need different preprocessing/feature engineering
- **Action:** Verify labels, check data quality, consider alternative features

---

## Quick Checklist for Reporting

When presenting results, include:

**Pipeline A (SCCA):**
- [ ] Top 3 canonical correlations
- [ ] Sparsity levels (gene and brain)
- [ ] Cross-validation stability (r range across folds)
- [ ] Top contributing genes/ROIs (if weights saved)

**Pipeline B (Prediction):**
- [ ] AUC for all models (table or plot)
- [ ] Best model identified
- [ ] Modality comparison (gene vs brain)
- [ ] Fusion benefit (if any)
- [ ] CCA/SCCA value-add (if any)

**Context:**
- [ ] Sample size (N=4,218)
- [ ] Label balance (% positive)
- [ ] Cross-validation scheme (5-fold stratified)
- [ ] Evaluation protocol (leakage-safe, fold-wise Stage 1)

---

## Example Summary Statement

> We analyzed 4,218 subjects with both genomic (111 genes × 768-D DNABERT-2 embeddings) and fMRI (180 ROIs, HCP-MMP1) data for Major Depressive Disorder (MDD) prediction.
> 
> **Interpretable SCCA (Pipeline A):** The top canonical component showed weak gene-brain correlation (r=0.156 train, r=-0.005 holdout), with low sparsity (8% genes, 2% ROIs zeroed), indicating the coupling pattern does NOT generalize and the relationship is diffuse rather than localized.
> 
> **Predictive modeling (Pipeline B):** Gene features (AUC=0.759) dramatically outperformed fMRI features (AUC=0.559) for MDD prediction. Early fusion yielded negligible improvement (AUC=0.762, +0.003). CCA/SCCA joint models performed poorly (AUC=0.55), demonstrating that unsupervised gene-brain alignment does NOT translate to clinical prediction power.
> 
> **Core conclusion:** Foundation model gene embeddings are highly predictive of MDD; brain functional connectivity adds no value. Direct supervised learning vastly outperforms two-stage unsupervised approaches.

---

**Last updated:** January 14, 2026
