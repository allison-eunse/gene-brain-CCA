# Gene-Brain CCA Analysis: Comprehensive Research Report

**Author:** Allie  
**Date:** January 2026  
**Project:** Gene-Brain Canonical Correlation Analysis  
**Location:** /storage/bigdata/UKB/fMRI/gene-brain-CCA/

---

## Executive Summary

This report documents two major experimental phases investigating the relationship between genetic embeddings and brain imaging for Major Depressive Disorder (MDD) prediction:

1. **Experiment 1 (Original Pipeline):** Unsupervised two-stage CCA/SCCA using scalar gene representations (mean-pooled and max-pooled) followed by supervised depression prediction
2. **Experiment 2 (gene-brain-cca-2):** Redesigned pipelines with (A) interpretable SCCA on scalars and (B) supervised prediction using full 768-dimensional foundation model embeddings

**Key Finding:** Gene-brain correlation (unsupervised objective) does not translate into clinical prediction power (supervised objective). Full foundation model embeddings substantially outperform scalar reductions for depression prediction.

---

## Table of Contents

1. Background and Connection to Yoon et al.
2. Experiment 1: Unsupervised Two-Stage CCA/SCCA
3. Experiment 2: gene-brain-cca-2 Pipelines
4. Complete Results Tables
5. Key Concepts Explained
6. Scientific Conclusions
7. Methodological Clarifications

---

# Part 1: Background and Connection to Yoon et al.

## 1.1 What Yoon's Paper Achieved

Yoon et al. used **DNABERT-2** (a DNA foundation model) to generate 768-dimensional embeddings from exon sequences of 111 genes associated with Major Depressive Disorder (MDD).

| Metric | Yoon et al. | Traditional PRS |
|--------|-------------|-----------------|
| **AUC for MDD** | **0.851** | 0.53-0.57 |
| **Improvement** | +49-60% over PRS | Baseline |
| **Sample Size** | ~29,000 subjects | Varies |
| **Method** | Direct supervised learning | Polygenic risk scoring |

**Why foundation models outperformed PRS:** Traditional Polygenic Risk Scores (PRS) use pre-defined genetic variants (SNPs) with fixed effect sizes. Foundation models like DNABERT-2 capture:
- **Regulatory motifs** (patterns that control gene expression)
- **Splicing signals** (how genes are processed into proteins)
- **Long-range sequence dependencies** (interactions between distant parts of the genome)

## 1.2 Research Question

**Central Question:** Can we combine Yoon's gene embeddings with brain imaging (fMRI) data to understand how genetic variation influences brain structure, and whether this gene-brain relationship predicts depression?

**The Challenge:** 
- Yoon's genetics cohort: ~29,000 subjects
- fMRI imaging cohort: ~41,000 subjects
- **Overlap (subjects with BOTH modalities): 4,218 subjects** (14.6% of genetics, 10.3% of fMRI)

This is NOT a bug - these are genuinely different UK Biobank subsets.

---

# Part 2: Experiment 1 - Unsupervised Two-Stage CCA/SCCA

## 2.1 Why CCA?

**Canonical Correlation Analysis (CCA)** finds linear transformations of two datasets (genes and brain) that are maximally correlated with each other.

**Rationale for starting unsupervised:**
1. **Discovery Phase:** Before asking "what predicts depression?", ask "do genes and brain even relate to each other?"
2. **Dimensionality Reduction:** CCA compresses 111 gene features and 180 brain features into a shared low-dimensional space (10 canonical components)
3. **Hypothesis-Free:** CCA doesn't require labels - it finds natural co-variation patterns

**Mathematical Objective:**
- CCA finds weight vectors w_gene and w_fmri such that:
  - U = X_gene x w_gene (gene canonical variate/score)
  - V = X_fmri x w_fmri (fMRI canonical variate/score)
  - Maximize: correlation(U, V)

## 2.2 Why Two Methods: Conventional CCA vs Sparse CCA (SCCA)?

| Aspect | Conventional CCA | Sparse CCA (SCCA) |
|--------|------------------|-------------------|
| **Uses** | All 111 genes + all 180 ROIs | Subset of genes + subset of ROIs |
| **Pattern** | "Global" (everything contributes) | "Localized" (specific biomarkers) |
| **Interpretability** | Low (all weights non-zero) | High (many weights = 0) |
| **Regularization** | None | L1-norm (LASSO) penalty |

**Scientific Question:**
- If SCCA >> CCA: The gene-brain relationship is driven by specific genes and brain regions (identifiable biomarkers)
- If SCCA = CCA: The relationship is diffuse (many features contribute small amounts)

## 2.3 Why Mean Pooling vs Max Pooling?

DNABERT-2 outputs 768 dimensions per gene. To use CCA, this was reduced to a manageable size:

| Method | Formula | Interpretation |
|--------|---------|----------------|
| **Mean Pooling** | Average of 768 values | "Typical" embedding value |
| **Max Pooling** | Maximum of 768 values | "Strongest" signal in embedding |

**Hypothesis:** If max pooling >> mean pooling for prediction, it suggests the foundation model's strongest activations contain the clinically relevant information.

## 2.4 Experiment 1 Results: Mean Pooling

### Stage 1: Unsupervised Gene-Brain Correlation

| Metric | CCA | SCCA |
|--------|-----|------|
| **1st Canonical Correlation** | 0.368 | 0.368 |
| **Permutation p-value** | 0.040 (significant) | 0.040 (significant) |
| **Gene Sparsity** | 0.0% | 0.0% |
| **fMRI Sparsity** | 0.0% | 0.0% |

**Interpretation:**
- **Statistically significant coupling (p < 0.05):** There IS a real mathematical relationship between gene embeddings and brain connectivity
- **No sparsity achieved:** Despite L1 penalties, SCCA could not induce sparsity - the relationship is diffuse and global
- **CCA = SCCA:** Identical correlations confirm the pattern is spread across all features

### Stage 2: Supervised Depression Prediction

| Feature Set | CCA AUC | SCCA AUC |
|-------------|---------|----------|
| **Gene Only** (U variates) | 0.588 | 0.588 |
| **fMRI Only** (V variates) | 0.517 | 0.514 |
| **Joint** (U + V) | 0.581 | 0.581 |

**Interpretation:**
- **Gene dominates:** Gene variates achieve 0.588 AUC; fMRI variates near chance (0.51)
- **Joint <= Gene:** Adding brain features does not help
- **fMRI-only performance is near chance level:** The canonical brain features show minimal predictive value for depression

## 2.5 Experiment 1 Results: Max Pooling

### Stage 1: Unsupervised Gene-Brain Correlation

| Metric | CCA | SCCA |
|--------|-----|------|
| **1st Canonical Correlation** | 0.347 | 0.347 |
| **Permutation p-value** | 0.995 (not significant) | 0.995 (not significant) |
| **Gene Sparsity** | 0.0% | 0.0% |
| **fMRI Sparsity** | 0.0% | 0.0% |

**Interpretation:**
- **Not statistically significant (p = 0.995):** The gene-brain correlation with max-pooled embeddings is indistinguishable from random chance
- **Lower correlation:** 0.347 vs 0.368 for mean pooling

### Stage 2: Supervised Depression Prediction

| Feature Set | CCA AUC | SCCA AUC |
|-------------|---------|----------|
| **Gene Only** | 0.505 | 0.494 |
| **fMRI Only** | 0.521 | 0.522 |
| **Joint** | 0.512 | 0.505 |

**Interpretation:**
- **All near chance (~0.50):** Max pooling produced embeddings with essentially no predictive power
- **Max pooling failed:** The single-strongest-signal approach lost critical information

## 2.6 Mean vs Max Pooling: Conclusion

| Metric | Mean Pooling | Max Pooling | Winner |
|--------|--------------|-------------|--------|
| **Stage 1 Correlation** | 0.368 (p=0.04) | 0.347 (p=0.995) | Mean |
| **Stage 2 Best AUC** | 0.588 | 0.522 | Mean |
| **Statistical Significance** | Yes | No | Mean |

**Conclusion:** Mean pooling preserves more predictive information than max pooling. However, 0.588 AUC is still far below Yoon's 0.851, leading to the hypothesis that scalar reduction (768 to 1) loses too much information.

---

# Part 3: Experiment 2 - gene-brain-cca-2 Pipelines

## 3.1 The Pivotal Realization

From Experiment 1:
1. Gene-brain correlation exists but does not predict depression
2. Scalar reduction (768 to 1 per gene) may discard predictive signal
3. SCCA could not achieve sparsity - the relationship is diffuse

**New Hypotheses:**
1. **Hypothesis A:** Even with diffuse patterns, can we identify which genes/ROIs contribute most?
2. **Hypothesis B:** Will using full 768-dimensional embeddings (85,248 total features) recover the lost predictive power?

## 3.2 Pipeline A: Interpretable SCCA

**Purpose:** Explore the gene-brain relationship at the scalar level (111 genes x 180 ROIs) with proper leakage-safe evaluation.

**Method:** Sparse CCA with L1 penalties (c1=0.3, c2=0.3)

**Key Improvements over Experiment 1:**
- **Holdout Split:** 20% of data (844 subjects) held out for final evaluation
- **Train-Only Preprocessing:** All standardization and residualization applied only to training data
- **Cross-Validation:** 5-fold CV on training set (3,374 subjects)

**Is Pipeline A just SCCA?** Yes - it uses only Sparse CCA.

### Pipeline A Results

| Metric | Value |
|--------|-------|
| **N Total** | 4,218 |
| **N Train / Holdout** | 3,374 / 844 |
| **Depression Prevalence** | 41.1% |
| **Training r (CC1)** | 0.17-0.22 |
| **Validation r (CC1)** | -0.02 to +0.08 |
| **Gene Sparsity** | 8-9% |
| **fMRI Sparsity** | 2% |

### What "Correlations don't generalize" Means

| Fold | Training r | Validation r |
|------|------------|--------------|
| 0 | 0.170 | -0.022 |
| 1 | 0.162 | +0.024 |
| 2 | 0.170 | +0.003 |
| 3 | 0.168 | +0.019 |
| 4 | 0.165 | -0.012 |

**Training correlation = 0.17:** SCCA found a pattern where genes and brain "move together" in training data.

**Validation correlation = 0.00:** When tested on held-out subjects, the correlation disappears.

**Why this happens:** SCCA is overfitting to noise. The gene-brain "coupling" found is likely statistical noise that happened to correlate in the specific training sample.

### What "Pattern is diffuse, not localized" Means

**Localized Pattern (what SCCA is designed to find):**
- 5 specific genes strongly correlate with 3 specific brain regions
- Easy to interpret: "Gene SLC6A4 affects the hippocampus"

**Diffuse Pattern (what was actually found):**
- Gene sparsity = 8-9% (91-92% of genes have non-zero weights)
- fMRI sparsity = 2% (98% of brain regions have non-zero weights)
- No "star player" - everything contributes equally

## 3.3 Pipeline B: Predictive Wide Gene

**Purpose:** Test whether using full 768-dimensional embeddings improves depression prediction.

**Technical Details:**
- **Gene Features:** 111 genes x 768 dimensions = 85,248 features, reduced via PCA to 512 dimensions (retains 91.8% variance)
- **fMRI Features:** 180 ROIs (unchanged)
- **Holdout:** Same 20% split as Pipeline A

**Models Tested:**
1. Gene-only (512-D PCA)
2. fMRI-only (180-D)
3. Early Fusion (512 + 180 = 692-D)
4. CCA Joint (10 canonical variates)
5. SCCA Joint (10 sparse canonical variates)

**Is Pipeline B using SCCA?** Partially - Pipeline B tests multiple methods including CCA and SCCA as comparison baselines, but its main contribution is testing whether supervised learning on full embeddings outperforms unsupervised dimensionality reduction.

### Pipeline B Results: Cross-Validation

| Model | AUC | Average Precision |
|-------|-----|-------------------|
| gene_only_logreg | 0.724 | 0.564 |
| gene_only_mlp | 0.686 | 0.541 |
| fmri_only_logreg | 0.509 | 0.414 |
| fmri_only_mlp | 0.501 | 0.412 |
| early_fusion_logreg | 0.725 | 0.566 |
| early_fusion_mlp | 0.672 | 0.541 |
| cca_joint_logreg | 0.534 | 0.435 |
| cca_joint_mlp | 0.526 | 0.434 |
| scca_joint_logreg | 0.542 | 0.442 |
| scca_joint_mlp | 0.543 | 0.445 |

### Pipeline B Results: Holdout

| Model | AUC | Average Precision |
|-------|-----|-------------------|
| gene_only_logreg | 0.759 | 0.596 |
| gene_only_mlp | 0.751 | 0.623 |
| fmri_only_logreg | 0.559 | 0.453 |
| fmri_only_mlp | 0.543 | 0.462 |
| early_fusion_logreg | 0.762 | 0.603 |
| early_fusion_mlp | 0.710 | 0.560 |
| cca_joint_logreg | 0.546 | 0.454 |
| cca_joint_mlp | 0.530 | 0.454 |
| scca_joint_logreg | 0.566 | 0.480 |
| scca_joint_mlp | 0.520 | 0.425 |

### Interpretation

**1. Full Gene Embeddings Dramatically Outperform Scalar Reduction**

| Method | Gene AUC | Improvement |
|--------|----------|-------------|
| Scalar (Exp1 Mean Pool) | 0.588 | Baseline |
| Full 768-D (Pipeline B) | 0.759 | +29% |

**2. Gene >> fMRI for Depression Prediction**

| Modality | Holdout AUC |
|----------|-------------|
| Gene Only | 0.759 |
| fMRI Only | 0.559 |

**3. CCA/SCCA Underperform Supervised Methods**

| Model | Holdout AUC |
|-------|-------------|
| Gene Only (direct) | 0.759 |
| CCA Joint | 0.546 |
| SCCA Joint | 0.566 |

CCA/SCCA find axes that maximize gene-brain correlation, but those axes do not align with depression prediction.

---

# Part 4: Complete Results Tables

## Master Comparison: All Experiments

| Experiment | Method | Best Model | AUC | Key Insight |
|------------|--------|------------|-----|-------------|
| Exp1 Mean Pool | Scalar to CCA to Supervised | gene_only LogReg | 0.588 | Mean pooling preserves some signal |
| Exp1 Max Pool | Scalar to CCA to Supervised | fmri_only MLP | 0.522 | Max pooling loses most signal |
| Exp2 Pipeline A | Scalar to SCCA (unsupervised) | - | r = 0.17 | Correlation does not generalize |
| Exp2 Pipeline B | Full 768D to PCA512 to LogReg | early_fusion LogReg | 0.762 | Full embeddings recover signal |
| Yoon et al. | Full 768D to Direct supervised | N/A | 0.851 | Reference with N=29k |

## Supervised vs Unsupervised Summary

| Part | What It Does | Uses Labels? | Method Type |
|------|--------------|--------------|-------------|
| Exp1 Stage 1 | Find gene-brain correlation | No | Unsupervised |
| Exp1 Stage 2 | Predict depression from variates | Yes | Supervised |
| Pipeline A | Find gene-brain correlation (with holdout) | No | Unsupervised |
| Pipeline B - PCA | Compress gene features | No | Unsupervised |
| Pipeline B - CCA/SCCA | Reduce gene+brain to variates | No | Unsupervised |
| Pipeline B - LogReg/MLP | Predict depression | Yes | Supervised |

---

# Part 5: Key Concepts Explained

## 5.1 What is "Scalar Reduction"?

Scalar reduction = Taking DNABERT-2's 768-dimensional embedding for each gene and collapsing it to 1 single number.

Example:
Gene SLC6A4 embedding: [0.23, -0.15, 0.87, 0.02, ..., 0.41]  <- 768 values
                              | Mean Pooling
                              v
                           0.34  <- 1 value (the average)

Why it discards information:
- Those 768 dimensions encode different aspects: regulatory patterns, splicing signals, structural motifs
- Averaging them into 1 number loses all that nuance
- Your results proved this: Scalar (1D per gene) = AUC 0.588, Full (768D to PCA 512) = AUC 0.759

## 5.2 Why Foundation Model Representations Matter

The richness of the 768-D representation is what makes foundation models valuable:
- DNABERT-2 was trained on millions of DNA sequences to learn meaningful patterns
- Each of those 768 dimensions captures something specific about genomic context
- If you reduce to 1 scalar, you are throwing away the main advantage of using a foundation model

## 5.3 What is "Supervised Feature Selection"?

Supervised feature selection = Using the clinical label (depression yes/no) to decide which features are important.

Examples:
- LASSO regression: Keeps genes whose weights predict depression, zeros out the rest
- Random Forest importance: Ranks features by how much they improve prediction
- SHAP values: Shows which features drove each prediction

This is different from SCCA:
- SCCA selects features based on gene-brain correlation (unsupervised - no labels used)
- Supervised selection picks features based on depression prediction (uses labels)

## 5.4 Where Do PCA-Reduced Features Come From?

They were computed in Pipeline B's run_predictive_suite.py script:

Step 1: Load raw gene embeddings
        X_gene_wide.npy = (4218 subjects x 85,248 features)
        |
        v
Step 2: Apply PCA (on training data only)
        PCA with n_components=512
        Keeps 91.8% of variance
        |
        v
Step 3: Transform all data
        X_gene_pca = (4218 subjects x 512 features)
        |
        v
Step 4: Train LogReg/MLP on X_gene_pca

PCA was fitted on the training set only, then applied to holdout - this prevents data leakage.

---

# Part 6: Scientific Conclusions

## 6.1 Key Findings

| Finding | Evidence |
|---------|----------|
| Full embeddings >> Scalar | 0.762 vs 0.588 (+29% improvement) |
| Mean pooling >> Max pooling | 0.588 vs 0.522 (+13% improvement) |
| Gene >> fMRI for MDD | 0.759 vs 0.559 (+36% relative improvement) |
| CCA/SCCA hurt performance | 0.546-0.566 vs 0.759-0.762 for direct supervised |
| Gene-brain correlation != clinical utility | r=0.37 but AUC=0.52 for joint prediction |
| Sparsity not achieved | SCCA sparsity < 10%, signal is diffuse |

## 6.2 Connection to Yoon et al.

| Aspect | Yoon | Your Analysis |
|--------|------|---------------|
| Foundation Model | DNABERT-2 | DNABERT-2 (same) |
| Representation | Full 768-D | Scalar (Exp1) vs Full (Exp2) |
| Sample Size | ~29,000 | 4,218 |
| Best AUC | 0.851 | 0.762 |
| Gap | - | -0.089 (mainly sample size) |

## 6.3 What This Means for Gene-Brain Research

1. The multimodal hypothesis is not supported for MDD: Adding brain imaging to genetics does not improve MDD prediction. The gene-brain coupling exists but is orthogonal to depression.

2. Foundation model representations matter: Scalar reduction discards critical information. Future multimodal studies should preserve full embeddings.

3. SCCA may not be the right tool: Works well when signal is localized (specific biomarkers). This signal is diffuse, so SCCA = CCA. Consider supervised feature selection instead.

---

# Part 7: Methodological Clarifications

## 7.1 Is Pipeline A "just SCCA"?

Yes. Pipeline A uses Sparse CCA exclusively to:
1. Find gene-brain canonical variates
2. Evaluate generalization via holdout and CV
3. Identify which genes/ROIs have non-zero weights

It does NOT include supervised prediction - it is purely an exploration of the unsupervised gene-brain relationship.

## 7.2 Is Pipeline B using SCCA?

Partially. Pipeline B includes SCCA as one of 10 tested models, but its main purpose is:
1. Primary Goal: Test whether full 768-D embeddings improve supervised prediction
2. CCA/SCCA Comparison: Included to show that unsupervised dimensionality reduction underperforms direct supervised learning

The "winning" models (gene_only, early_fusion) do NOT use CCA or SCCA - they use direct logistic regression on PCA-reduced features.

## 7.3 Two-Stage vs Direct Supervised

| Approach | Description | Performance |
|----------|-------------|-------------|
| Two-Stage (Exp1) | CCA/SCCA to variates to supervised | AUC 0.52-0.58 |
| Direct Supervised (Exp2 Pipeline B) | PCA to supervised (no CCA) | AUC 0.76 |

The two-stage approach is inferior because CCA's objective (maximize gene-brain correlation) does not align with the clinical objective (predict depression).

---

# Appendix: File Locations

## Experiment 1 Results

- Mean Pooling: /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_mean_pooling/
- Max Pooling: /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/

## Experiment 2 Results

- Pipeline A: /storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/interpretable/
- Pipeline B: /storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/wide_gene/

## Key Result Files

| File | Location | Contents |
|------|----------|----------|
| scca_interpretable_results.json | derived/interpretable/ | Pipeline A SCCA correlations and sparsity |
| predictive_suite_results.json | derived/wide_gene/ | Pipeline B all model AUCs |
| comparison_report.json | derived_*/comparison/ | Exp1 CCA vs SCCA comparison |
| conventional_results.json | derived_*/cca_stage1/ | Exp1 CCA Stage 1 |
| sparse_results.json | derived_*/scca_stage1/ | Exp1 SCCA Stage 1 |

---

End of Report
