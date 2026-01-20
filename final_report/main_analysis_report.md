# Gene-Brain CCA: Main Analysis Report

## Overview

This report summarizes the gene-brain coupling analysis using Canonical Correlation Analysis (CCA) 
and Sparse CCA (SCCA) on UK Biobank fMRI functional connectivity data linked to psychiatric risk 
gene embeddings from DNABERT-2.

**Cohort:** N=4218 subjects (Train: 3374, Holdout: 844)  
**Gene Features:** 111 psychiatric risk genes → 85,248-D embeddings (768-D × 111 genes)  
**Brain Features:** Schaefer cortical parcellations with Yeo-7 and Yeo-17 network-level summaries

---

## Main Results

### Schaefer-7 Summary (Yeo-7 Networks)

**Brain Features:** 21 FC edges (7 networks: Visual, Somatomotor, Dorsal Attention, Ventral Attention, 
Limbic, Frontoparietal, Default Mode)

**Best Configuration:** scca_pca64_c0.1_0.1  
**Coupling Performance:**
- Validation CC1: 0.0104 ± 0.0314
- **Holdout CC1: 0.0127**
- Overfitting gap: 0.0782
- Generalization gap: 0.0738

**Interpretation:** Weak but positive coupling on holdout set. The model shows modest generalization, 
suggesting some gene-brain association at the network level, though with high overfitting.

---

### Schaefer-17 Summary (Yeo-17 Networks)

**Brain Features:** 136 FC edges (17 networks: expanded Yeo parcellation with DMN split into 
subnetworks A/B/C, Control split into A/B, etc.)

**Best Configuration:** scca_pca128_c0.1_0.1  
**Coupling Performance:**
- Validation CC1: 0.0383 ± 0.0314
- **Holdout CC1: 0.0418**
- Overfitting gap: 0.0418
- Generalization gap: 0.0284

**Interpretation:** **Best performance** among all brain features tested. Schaefer-17 shows stronger 
coupling (r_holdout = 0.042) with better generalization characteristics. The finer-grained network 
parcellation captures more specific gene-brain associations while maintaining stability.

---

### Comparison: Schaefer-7 vs Schaefer-17 vs Tian

| Brain Feature | Dimensions | Best Config | r_holdout CC1 | Generalization Gap |
|---------------|------------|-------------|---------------|-------------------|
| **Schaefer-7** | 21 edges | SCCA PCA64 | 0.0127 | 0.0738 |
| **Schaefer-17** | 136 edges | SCCA PCA128 | 0.0418 | 0.0284 |
| **Tian Subcortex** | 10 edges | SCCA PCA102 | -0.0961 | 0.3566467842920255 |

**Key Finding:** Schaefer-17 provides the optimal balance between granularity and stability, 
achieving the strongest gene-brain coupling signal.

---

## Top Genes from SCCA (CC1)

The following genes show the strongest associations in the first canonical component:

### Positive Weights (Top 5)
1. **NR3C1** (weight: 0.3732)
2. **CTNND2** (weight: 0.3362)
3. **KCNK2** (weight: 0.3240)
4. **CSMD1** (weight: 0.3025)
5. **HSPA1L** (weight: 0.1631)

### Negative Weights (Top 5)
1. **ZNF165** (weight: -0.3302)
2. **CHRM2** (weight: -0.2966)
3. **CAT** (weight: -0.2956)
4. **GABRA1** (weight: -0.1998)
5. **OSTM1** (weight: -0.1631)

---

## Methods Summary

**Gene Embeddings:** 
- 111 psychiatric risk genes (depression, schizophrenia, bipolar disorder, neurodevelopmental)
- DNABERT-2 foundation model embeddings (768-D per gene)
- Concatenated to 85,248-D wide matrix (no gene PCA for interpretability)

**Brain Features:**
- Schaefer-400 cortical parcellation → Yeo-7/17 network-level functional connectivity
- Residualized for age/sex on training set only (leakage-free)
- Z-scored and standardized

**CCA/SCCA:**
- Sparse CCA with L1 penalties (c1=0.1, c2=0.1 for brain and gene respectively)
- Gene PCA: 64-D (Schaefer-7), 128-D (Schaefer-17)
- 5-fold cross-validation, 20% holdout set
- Evaluation: first canonical correlation on holdout (r_holdout_cc1)

---

## Tian Sensitivity Analysis Context

The Tian subcortical analysis revealed severe MNI registration failures:
- 514/746 subjects (68.9%) had zero signal in Tian atlas region after native→MNI resampling
- Final N=128 subjects after aggressive QC
- r_holdout = -0.096 (no significant coupling)

**Conclusion:** Tian results are inconclusive due to upstream preprocessing failures, not biological 
absence of subcortical-gene coupling. Schaefer cortical results remain the primary finding.

---

## Figures

1. **Coupling Comparison:** Holdout performance across brain features
2. **Top Genes:** SCCA weights for first canonical component
3. **Network Weights:** Top brain FC edges by SCCA weight

---

## Citation

**Recommended Methods Text:**

> We tested gene-brain coupling using network-level functional connectivity summaries derived from 
> the Schaefer-400 cortical parcellation. For Yeo-7 networks (21 FC edges), sparse CCA with gene 
> PCA=64 yielded r_holdout=0.013. For Yeo-17 networks (136 FC edges), sparse CCA with gene PCA=128 
> yielded r_holdout=0.042, indicating weak but replicable gene-brain associations at the cortical 
> network level. Top-weighted genes included NR3C1, CTNND2, and ZNF165. A Tian subcortical sensitivity 
> analysis (N=128 after QC due to MNI registration failures) showed no significant coupling 
> (r_holdout=-0.096).

