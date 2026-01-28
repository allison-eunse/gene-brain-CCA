# Final Analysis Report: Stratified CCA Results and Evo2 "Significant" Finding

**Date**: January 27, 2026  
**Analyst**: Critical evaluation of stratified CCA results and the p = 0.049 Evo2 finding

---

## Executive Summary

**Question**: Is the Evo2 p = 0.049 result for stratified dMRI analysis a true discovery, as suggested by the interpretation that "Evo2's evolutionary priors filter noise"?

**Answer**: **No.** The result is a likely false positive from multiple testing, characterized by instability, lack of biological coherence, and an unsupported claim about Evo2's training methodology.

---

## Part 1: Verification of the "Evolutionary Conservation" Claim

### The Claim
> "Evo2 is explicitly trained on evolutionary conservation across species. Unlike DNABERT2, which learns DNA 'grammar,' Evo2 prioritizes evolutionarily conserved features, filtering genetic noise."

### Web Search Results

I searched multiple sources to verify this claim:

**Sources Consulted:**
1. Arc Institute Evo2 Official Page: https://arcinstitute.org/tools/evo
2. Arc Institute Manuscripts: https://arcinstitute.org/manuscripts/Evo2
3. GitHub Repository: https://github.com/ArcInstitute/evo2
4. OpenGenome2 Dataset: https://huggingface.co/datasets/arcinstitute/opengenome2
5. BioRxiv Preprint: "Genome modeling and design across all domains of life with Evo 2" (Feb 2025)
6. Literature on DNA language models and evolutionary conservation
7. Comparison with species-aware models (Genome Biology, 2024)
8. PhyloGPN and conservation-explicit models (arXiv 2025)

### Finding: **The Claim is Scientifically Inaccurate**

**What Evo2 Actually Does:**
- **Training objective**: Standard autoregressive language modeling (next-nucleotide prediction)
- **Training data**: OpenGenome2 (8.8 trillion tokens from 128,000+ genomes across all domains of life)
- **Architecture**: StripedHyena 2 (hybrid state-space model + attention)
- **Conservation**: NOT an explicit training objective

**What "Evolutionary Conservation Training" Would Require:**
- Models like **PhyloGPN**: Integrates phylogenetic trees into the loss function
- Models like **GPN-MSA**: Uses multiple sequence alignments with conservation labels
- **Species-aware DNA LMs**: Train on aligned orthologous sequences

**Evo2 does NONE of these.**

**Key Distinction:**
- Training on multi-species genomes ≠ training on evolutionary conservation
- Evo2 may *implicitly* learn conservation patterns (conserved regions appear repeatedly), but this is an emergent property, not an explicit objective
- DNABERT2 also trains on multi-species data (and uses BPE tokenization)

**Conclusion**: The claim that "Evo2 has an evolutionary prior that filters noise" is **post-hoc rationalization**, not a property of the model's training.

---

## Part 2: Analysis of the Evo2 dMRI Result (p = 0.049)

### The Result

| Group   | r_holdout | Best Config              |
|---------|-----------|--------------------------|
| MDD     | -0.032    | SCCA PCA256 (c=0.3/0.1) |
| Control | +0.103    | SCCA PCA64 (c=0.5/0.3)  |

- **Difference**: r_diff = -0.135
- **Permutation p-value**: 0.049
- **Gene weight cosine similarity**: 0.0
- **Brain weight cosine similarity**: 0.0

### Critical Red Flags

#### 1. **Multiple Testing Not Accounted For**

You ran **16 tests**:
- 4 foundation models: DNABERT2, HyenaDNA, Caduceus, Evo2
- 4 modalities: fMRI Schaefer7, fMRI Schaefer17, sMRI, dMRI

With α = 0.05, expected false positives: **16 × 0.05 = 0.8**

**Observed**: 1 test with p < 0.05 (Evo2 dMRI)

**Conclusion**: This is exactly what chance predicts.

**Bonferroni correction**: 0.05/16 = 0.003125
- Evo2 p = 0.049 **FAILS** correction

#### 2. **Gene Weight Cosine = 0.0 (Instability)**

**What this means**: The gene patterns learned by MDD and Control are completely orthogonal (no similarity).

**Why this is a red flag**:
- If there was a true biological signal, both groups should show similar gene patterns (even with different magnitudes)
- Cosine = 0.0 indicates the models learned random, unstable patterns
- This is evidence of **overfitting to noise**, not "group-specific biology"

**Cross-model comparison**:
| Model      | Gene Weight Cosine |
|------------|--------------------|
| DNABERT2   | 0.0                |
| HyenaDNA   | -0.136             |
| Caduceus   | 0.0                |
| Evo2       | 0.0                |

All models show **zero or near-zero** gene pattern similarity between MDD and Control.

#### 3. **Incomparable Model Configurations**

MDD and Control selected **different PCA dimensionalities**:
- MDD: PCA256 (256-dimensional embedding)
- Control: PCA64 (64-dimensional embedding)

**Problem**: You cannot meaningfully compare gene-brain coupling patterns when the two groups operate in different feature spaces.

This is like comparing apples (256-dim) to oranges (64-dim).

#### 4. **Control Result Driven by Single Feature**

Analysis of gene weights reveals:

**Control group (PCA64)**:
- 62/64 weights ≈ 0 (96.9% sparse)
- **Dominant weight**: PC_PLOD1 = **0.999** (!)
- Second weight: PC_ARHGAP8 = 0.035

**The r = 0.103 coupling in controls is driven almost entirely by ONE PCA component.**

**Biological coherence check**:
- **PLOD1**: Procollagen-lysine 2-oxoglutarate 5-dioxygenase (collagen biosynthesis)
- **Function**: Extracellular matrix formation, connective tissue
- **Brain relevance**: None established
- **MDD relevance**: None established

**Conclusion**: The "significant" result is driven by a spurious correlation with a non-brain-related gene, not biology.

#### 5. **Direction Inconsistency Across Models**

Evo2 is the **ONLY** model where MDD coupling is negative:

| Model      | MDD r | Control r |
|------------|-------|-----------|
| DNABERT2   | +0.034 | +0.019   |
| HyenaDNA   | +0.058 | +0.006   |
| Caduceus   | +0.140 | +0.079   |
| **Evo2**   | **-0.032** | +0.103 |

**This sign flip suggests**: The Evo2 result reflects noise in the specific holdout split, not a stable biological pattern.

#### 6. **Overfitting Evidence**

**Control group** (SCCA PCA64):
```
CV mean:     r = 0.043
Holdout:     r = 0.103 (2.4× larger!)
```

**Interpretation**: The holdout correlation is **2.4 times larger** than cross-validation, suggesting:
- High variance across data splits
- Unstable estimates
- Possible lucky alignment of outlier subjects in the holdout

**MDD group** (SCCA PCA256):
```
CV mean:     r = 0.068
Holdout:     r = -0.032 (negative!)
```

**Generalization gap**: 0.118 (model performs 36% worse on holdout)

---

## Part 3: Cross-Model Comparison

### All dMRI Results

| Model      | MDD r | Control r | r_diff | p-value | Survives Bonferroni? |
|------------|-------|-----------|--------|---------|----------------------|
| DNABERT2   | +0.034 | +0.019   | +0.015 | 0.793   | ❌ No                |
| HyenaDNA   | +0.058 | +0.006   | +0.052 | 0.329   | ❌ No                |
| Caduceus   | +0.140 | +0.079   | +0.061 | 0.470   | ❌ No                |
| Evo2       | -0.032 | +0.103   | -0.135 | 0.049   | ❌ No (p > 0.003)   |

### Key Findings

1. **No model shows robust coupling**: All r < 0.15 (r² < 2.25%)
2. **No model survives multiple testing correction**
3. **Gene weight instability universal**: Cosine ≈ 0 for all models
4. **No consistent pattern**: Evo2 doesn't outperform others across modalities
5. **Caduceus shows largest effect size** (r_diff = +0.061), but p = 0.47

### Implication

If Evo2's "evolutionary prior" truly filtered noise, we would expect:
- Evo2 to consistently outperform other models across modalities ❌
- Gene weights to be stable (high cosine) ❌
- Significant results across multiple modalities ❌
- Effect sizes larger than chance ❌

**None of these are observed.**

---

## Part 4: What You Actually Have

### Summary of All Results

| Analysis                  | Gene  | fMRI  | sMRI      | dMRI      |
|---------------------------|-------|-------|-----------|-----------|
| **→ MDD Prediction (AUC)**| 0.759 | 0.559 | ❌ Not run | ❌ Not run |
| **Gene-Brain Coupling (r)**| —    | 0.01-0.04 | -0.045 | 0.009     |
| **Stratified CCA (p-value)**| —   | NS    | NS        | 0.049*    |

*Does not survive Bonferroni correction

### The Core Finding

**What works**:
- Gene embeddings → MDD: AUC = 0.76 (moderate prediction)

**What doesn't work**:
- Brain (fMRI) → MDD: AUC = 0.56 (chance level)
- Gene ↔ Brain coupling: r ≈ 0.01-0.04 (no coupling)
- Stratified CCA: No significant differences after correction

**Interpretation**:
1. Gene embeddings capture MDD-relevant variance
2. Brain connectivity does NOT capture MDD-relevant variance
3. Therefore, gene-brain CCA cannot find a coupling that relates to MDD

This is the "biological distance problem":
```
Gene → Molecular/cellular changes → Circuit dynamics → Macroscale structure/function
                                                            ↑
                                                          You are here
                                                          (too far from gene)
```

---

## Part 5: Recommended Scientific Conclusion

### What to Report

**Honest summary of findings**:

> "We tested whether DNA sequence embeddings from four genomic foundation models (DNABERT2, HyenaDNA, Caduceus, Evo2) exhibit linear coupling with macroscale brain connectivity (fMRI, sMRI, dMRI) in a UK Biobank sample (N=3,374; 50.7% MDD). Across all models and modalities, gene-brain coupling was weak (r_holdout < 0.15) and did not survive cross-validation or multiple testing correction. Stratifying by MDD diagnosis did not reveal robust group-specific patterns. Gene embeddings predicted MDD status with moderate accuracy (AUC=0.76), but brain connectivity features performed at chance (AUC=0.56), indicating that the brain phenotypes measured do not capture MDD-relevant variance at the population level. These results suggest that the biological pathway from genomic sequence to psychiatric phenotype operates through molecular and cellular mechanisms not linearly reflected in static, macroscale neuroimaging."

### What NOT to Report

❌ "Evo2's evolutionary priors revealed significant gene-brain coupling in controls"  
❌ "Evolutionary conservation filters noise to detect subtle biological signals"  
❌ "MDD and controls show different gene-brain coupling patterns"  
❌ "Foundation models differ in their ability to capture gene-brain relationships"

### Alternative Narratives (If Needed)

If you want to publish this work, frame it as:

1. **A rigorous null result**: "We systematically tested 4 state-of-the-art genomic foundation models across 4 brain imaging modalities and found no evidence of linear gene-brain coupling in MDD. This negative finding has important implications for imaging genetics research."

2. **Methodological contribution**: "We developed a comprehensive CCA pipeline for testing gene sequence embeddings against brain phenotypes, including stratified analysis and permutation testing. Our framework is publicly available for future studies."

3. **Biological insight**: "The lack of coupling suggests that MDD genetic risk operates primarily at molecular/cellular scales (neurotransmission, inflammation) not captured by macroscale MRI. Future work should target intermediate phenotypes (e.g., gene expression, dynamic connectivity, molecular imaging)."

---

## Part 6: Sources and Evidence

### Web Search Sources

1. **Evo2 Official Documentation**:
   - https://arcinstitute.org/tools/evo
   - https://github.com/ArcInstitute/evo2
   - https://arcinstitute.org/manuscripts/Evo2

2. **Training Data**:
   - https://huggingface.co/datasets/arcinstitute/opengenome2
   - OpenGenome2: 8.8T tokens, 128K+ genomes

3. **Scientific Literature**:
   - Original Evo paper (Science, 2024): autoregressive language modeling
   - Species-aware DNA models (Genome Biology, 2024): explicit conservation requires alignment-based training
   - PhyloGPN (arXiv, 2025): phylogenetic trees in loss function
   - DNABERT-2 (ICLR, 2024): BPE tokenization, multi-species data

### Your Results Sources

1. **Stratified CCA results**:
   - `/storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/stratified_fm/`
   - stratified_comparison_*.json files

2. **Gene weights**:
   - stratified_evo2_dmri_mdd/W_gene.npy
   - stratified_evo2_dmri_ctrl/W_gene.npy

3. **Gene list**:
   - /storage/bigdata/NESAP/gene_list_filtered.txt (111 MDD-related genes)

4. **Previous results**:
   - gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json
   - coupling_benchmark summaries

---

## Part 7: Final Recommendations

### For Your Thesis/Paper

1. **Do NOT build a narrative around the Evo2 p = 0.049 result**
   - It does not survive multiple testing correction
   - It is driven by unstable patterns (cosine = 0.0)
   - It conflicts with results from other models
   - The "evolutionary prior" explanation is scientifically unsupported

2. **Emphasize the consistent null finding**
   - This is the true result: no model shows robust coupling
   - Frame it as a rigorous negative result with implications
   - Discuss the "biological distance problem"

3. **Report the gene → MDD prediction success**
   - This validates that your gene embeddings are meaningful
   - Contrast with brain → MDD failure
   - Explain why CCA cannot bridge this gap

4. **Suggest alternative approaches**
   - Intermediate phenotypes (gene expression, cellular imaging)
   - Non-linear methods (kernel CCA, deep CCA)
   - Dynamic connectivity, task-based fMRI
   - Molecular imaging (PET)

### For Future Work

1. **Run sMRI/dMRI → MDD prediction** to complete the picture
2. **Test non-linear coupling** if you have time/interest
3. **Examine gene pathway enrichment** even without brain coupling
4. **Compare to polygenic risk scores** as a baseline

---

## Conclusion

The interpretation you received contains:
- ✅ **Valid points**: Biological distance problem, realistic effect sizes
- ❌ **Invalid claims**: Evo2 evolutionary conservation training
- ❌ **Risky framing**: Post-hoc rationalization of likely false positive

**My recommendation**: Report the null result honestly. It's a scientifically valuable finding that tells us where NOT to look for gene-brain connections in psychiatry. This is the foundation for better-targeted future research.

**The true story**: Static macroscale brain imaging is too far removed from genomic mechanisms to detect linear gene-brain coupling in MDD with current methods and sample sizes. This is a limitation of the phenotype, not a failure of your analysis.
