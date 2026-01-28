# Evo2 dMRI "Significant" Result Analysis

## The p = 0.049 Finding

**Result Summary:**
- **MDD group**: r_holdout = -0.032 (no coupling, negative)
- **Control group**: r_holdout = +0.103 (weak positive coupling)
- **Difference**: -0.135 (p = 0.049 via permutation test)

## Critical Red Flags

### 1. **Different Model Configurations Selected**

The best models for each group were completely different:
- **MDD**: SCCA with PCA256, regularization (c1=0.3, c2=0.1)
- **Control**: SCCA with PCA64, regularization (c1=0.5, c2=0.3)

**Why this matters**: You cannot meaningfully compare gene-brain coupling patterns when the two groups used different:
- PCA dimensionality (256 vs 64 components)
- Regularization strength
- Feature representations

This is like comparing apples (256-dim embedding) to oranges (64-dim embedding).

### 2. **Gene Weight Cosine Similarity = 0.0**

From the JSON:
```json
"gene_weight_cosine_cc1": 0.0
```

**Interpretation**: The gene weight patterns learned by MDD and Control groups have ZERO similarity. They're completely orthogonal.

**What this means**:
- If there was a true biological signal, both groups should show similar gene patterns (even if with different strengths)
- Cosine = 0.0 suggests the models learned random, unstable patterns
- This is a hallmark of overfitting to noise

### 3. **Extreme Sparsity (>96% of weights = 0)**

**MDD group** (PCA256):
- 250/256 weights effectively zero (97.7% sparse)
- Only 6 non-zero components
- Top 3 PCA components dominate:
  - PC_EHD3: 0.772
  - PC_PCLO: 0.620
  - PC_FHIT: -0.102

**Control group** (PCA64):
- 62/64 weights effectively zero (96.9% sparse)
- Only 2 non-zero components
- Almost entirely driven by ONE component:
  - PC_PLOD1: 0.999 (!!!)
  - PC_ARHGAP8: 0.035

**Critical Issue**: The Control group's coupling is driven by a SINGLE PCA component (PLOD1) with weight ~1.0. This is unstable and likely noise.

### 4. **No Biological Coherence**

The top "genes" (PCA components named after genes) don't converge on known MDD biology:

**MDD top genes**:
- EHD3 (endocytic trafficking)
- PCLO (presynaptic protein)
- FHIT (tumor suppressor)

**Control top genes**:
- PLOD1 (collagen biosynthesis - not brain-related!)
- ARHGAP8 (Rho GTPase regulation)

**Issue**: None of these are canonical MDD genes (BDNF, SLC6A4, HTR2A, etc.). The coupling appears to be driven by random, high-variance features.

### 5. **Massive Overfitting**

From the full results:

**MDD best config (SCCA pca256)**:
```
mean_r_train_cc1: 0.130
mean_r_val_cc1: 0.068
r_holdout_cc1: -0.032
```
Generalization gap: 0.118 (model performs 36% worse on holdout)

**Control best config (SCCA pca64)**:
```
mean_r_train_cc1: 0.102
mean_r_val_cc1: 0.043
r_holdout_cc1: 0.103
```
Overfitting gap: 0.058

**Paradox**: Control group CV showed r=0.043, but holdout showed r=0.103. This suggests:
- High variance across splits
- Unstable estimates
- Possible lucky split in holdout

## Why This "Significant" Result is Likely a False Positive

### 1. **Multiple Comparisons Not Accounted For**

You ran 16 tests (4 FMs × 4 modalities). With α = 0.05, expect ~0.8 false positives.
Getting p = 0.049 in 1/16 tests is **exactly what chance predicts**.

### 2. **The "Significant" Direction is Suspicious**

The p-value is for the **difference** (r_diff = -0.135), which tests:
> "Is MDD coupling significantly WORSE than Control coupling?"

But:
- MDD: r = -0.03 (noise)
- Control: r = +0.10 (also noise, but positive)

You're testing whether -0.03 < 0.10, which is trivially true but meaningless - you're comparing two noise estimates.

### 3. **Gene Patterns Don't Match**

If Evo2 truly captured an evolutionary prior that filters noise:
- Both MDD and Control should show similar gene patterns (high cosine)
- Instead: cosine = 0.0 (completely different)
- This proves the result is unstable

### 4. **Control Group Result is Driven by One Feature**

The r = 0.103 in controls comes from a single PCA component (PLOD1 weight = 0.999).

PLOD1 is a **collagen biosynthesis gene**. It has NO established role in:
- Brain function
- fMRI connectivity
- MDD pathophysiology

This is a spurious correlation, not biology.

## Correct Interpretation

**The Evo2 p = 0.049 result does NOT indicate:**
- Evolutionary conservation filtering noise
- True gene-brain coupling in controls
- Biologically meaningful MDD vs Control differences

**The result DOES indicate:**
- High-dimensional noise mining
- Unstable model selection (different configs per group)
- Random false positive from multiple testing
- Overfitting to outlier features (PLOD1)

## Conclusion

The Evo2 dMRI result (p = 0.049) is **not scientifically interpretable** as evidence for gene-brain coupling because:

1. **Multiple testing**: 1/16 tests reached p < 0.05 (chance expectation)
2. **Instability**: Gene weight cosine = 0.0 (groups learned different patterns)
3. **Model mismatch**: MDD used PCA256, Control used PCA64 (not comparable)
4. **Single-feature driver**: Control result driven by PLOD1 (collagen gene, not brain-related)
5. **No biological coherence**: Top genes don't map to known MDD pathways

**Recommendation**: Do NOT build a narrative around this p = 0.049 result. It is more likely a false positive than a true discovery.

**Alternative interpretation**: The consistency of null results across all models (DNABERT2, HyenaDNA, Caduceus, Evo2) and modalities (fMRI, sMRI, dMRI) is the real finding - gene sequence embeddings do not linearly couple with macroscale brain connectivity.
