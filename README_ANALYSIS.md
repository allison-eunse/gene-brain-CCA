# Analysis Summary: Evo2 "Significant" Result Investigation

This folder contains a comprehensive analysis of the stratified CCA results, specifically addressing whether the Evo2 p = 0.049 finding represents a true discovery.

## Quick Answer

**No, the Evo2 p = 0.049 result is NOT a robust finding.** It is a likely false positive from multiple testing, characterized by:
- Failure to survive Bonferroni correction (threshold: p < 0.003)
- Gene weight instability (cosine similarity = 0.0 between MDD and Control)
- Inconsistent with other foundation models
- Driven by a single non-brain-related feature (PLOD1, a collagen gene)
- Based on an incorrect claim about Evo2's training methodology

## Files Created

### 1. **FINAL_ANALYSIS_REPORT.md** (Main Document)
Comprehensive analysis covering:
- Verification of "evolutionary conservation" claim via web research
- Detailed examination of Evo2 dMRI result (p = 0.049)
- Cross-model comparison (DNABERT2, HyenaDNA, Caduceus, Evo2)
- Statistical evidence for multiple testing issues
- Recommended scientific conclusions
- All sources and evidence

**Read this first** for the complete story.

### 2. **cross_model_comparison.md**
Side-by-side comparison of all 4 foundation models on dMRI:
- Summary table of results
- Statistical power analysis
- Bonferroni correction calculations
- Evidence that no model survives multiple testing correction

### 3. **evo2_analysis_summary.md**
Focused analysis of the Evo2 result specifically:
- Why gene weight cosine = 0.0 is a red flag
- Extreme sparsity patterns
- Single-feature driver (PLOD1)
- Lack of biological coherence

### 4. **analyze_evo2_weights.py**
Python script that analyzes the gene weights from the Evo2 stratified CCA.
Shows:
- Top 20 weighted genes for MDD and Control groups
- Sparsity patterns
- Lack of overlap between groups
- Known MDD gene rankings

## Key Findings

### Web Search Results

**Claim Evaluated**: "Evo2 is explicitly trained on evolutionary conservation"

**Finding**: **FALSE**

Evo2 uses:
- Standard autoregressive language modeling (next-nucleotide prediction)
- Multi-species training data (OpenGenome2: 128K+ genomes)
- NO explicit conservation objective

**Sources**:
- Arc Institute official documentation
- Evo2 GitHub repository
- BioRxiv preprint (Feb 2025)
- Comparison with true conservation-aware models (PhyloGPN, species-aware LMs)

### Statistical Analysis Results

**Multiple Testing**:
- Total tests: 16 (4 models × 4 modalities)
- Expected false positives at α=0.05: 0.8
- Observed: 1 (Evo2 dMRI p=0.049)
- **Bonferroni threshold**: 0.003
- **Evo2 result**: p=0.049 ❌ FAILS correction

**Gene Weight Stability**:
- All 4 models show gene weight cosine ≈ 0 between MDD and Control
- Evidence of instability/overfitting, not biology

**Effect Sizes**:
- All models: r_holdout < 0.15 (explains < 2.25% variance)
- Evo2 MDD: r = -0.032 (negative, no coupling)
- Evo2 Control: r = 0.103 (weak, driven by single feature)

### Gene Weight Analysis Results

**Control Group (drives the p=0.049)**:
- 96.9% sparse (62/64 weights ≈ 0)
- Dominated by PLOD1 (weight = 0.999)
- PLOD1 = collagen biosynthesis gene (NOT brain-related)

**MDD Group**:
- 97.7% sparse (250/256 weights ≈ 0)
- Top genes: EHD3, PCLO, FHIT (not canonical MDD genes)

**Overlap**: Only 1/10 shared genes in top 10

## What Your Complete Results Show

| Analysis                        | Gene  | fMRI  | sMRI      | dMRI      |
|---------------------------------|-------|-------|-----------|-----------|
| **→ MDD Prediction (AUC)**      | 0.759 | 0.559 | Not done  | Not done  |
| **Gene-Brain Coupling (r)**     | —     | 0.013-0.042 | -0.045 | 0.009     |
| **Stratified CCA (p-value)**    | —     | NS    | NS        | 0.049*    |

*Does not survive Bonferroni correction

**Key Insight**:
- Gene embeddings predict MDD (AUC=0.76) ✓
- Brain features don't predict MDD (AUC=0.56) ✗
- Therefore, gene-brain CCA finds no coupling ✗

This is the **"biological distance problem"**: genes affect brain at molecular/cellular level, not macroscale MRI.

## Recommended Conclusion for Your Thesis/Paper

> "We systematically tested whether DNA sequence embeddings from four genomic foundation models (DNABERT2, HyenaDNA, Caduceus, Evo2) exhibit linear coupling with macroscale brain connectivity across fMRI, sMRI, and dMRI modalities in a UK Biobank sample (N=3,374, 50.7% MDD). Gene-brain coupling was uniformly weak (r < 0.15) across all models and modalities, with no results surviving multiple testing correction. Gene embeddings predicted MDD status with moderate accuracy (AUC=0.76), but brain connectivity performed at chance (AUC=0.56). These findings suggest that MDD genetic risk operates primarily through molecular and cellular mechanisms not linearly reflected in static, macroscale neuroimaging phenotypes."

## What NOT to Conclude

❌ "Evo2's evolutionary priors filter noise to reveal gene-brain coupling"  
❌ "MDD and controls show different gene-brain coupling patterns"  
❌ "The p=0.049 result indicates a biological difference"  
❌ "Foundation models differ in detecting gene-brain relationships"

## Next Steps

If continuing this research:

1. **Complete the analysis**: Run sMRI/dMRI → MDD prediction to fill the gaps
2. **Try non-linear methods**: Kernel CCA or Deep CCA (if no time, mention as future work)
3. **Frame as rigorous null result**: Valuable for the field to know what doesn't work
4. **Suggest intermediate phenotypes**: Gene expression, dynamic connectivity, molecular imaging

## Questions?

If you need to cite specific sources or want to dive deeper into any aspect, all the evidence is documented in **FINAL_ANALYSIS_REPORT.md**.

The bottom line: **Report the consistent null finding honestly.** It's scientifically more valuable than claiming a false positive.
