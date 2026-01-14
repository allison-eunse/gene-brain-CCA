# Gene–Brain CCA/SCCA Report (Experiments 1 & 2)

**Repo:** `gene-brain-CCA`  
**Cohort reality:** Overlap (genetics ∩ fMRI) = **N = 4,218** subjects  

This report summarizes:
- **Experiment 1** (original `gene-brain-CCA`): *two-stage* **unsupervised CCA/SCCA → supervised prediction**, comparing **mean pooling** vs **max pooling** gene reduction.
- **Experiment 2** (`gene-brain-cca-2`): **Pipeline A (interpretable SCCA)** and **Pipeline B (supervised wide-embedding prediction suite)** with leakage-safe protocol (holdout + train-only preprocessing).
- How these results relate to **Yoon et al.** (foundation-model sequence embeddings vs PRS baseline).

---

## 1) Key definitions (short, precise)

- **CCA (Canonical Correlation Analysis):** finds weight vectors \(w_x, w_y\) to maximize the correlation between \(U = Xw_x\) and \(V = Yw_y\).  
  - **Canonical variates / scores** \((U,V)\): the projected coordinates (“joint embedding coordinates”) for each subject.  
  - **Canonical weights** \((w_x, w_y)\): coefficients applied to original features to form \(U, V\).  
  - **Canonical correlation** \(r\): Pearson correlation between \(U\) and \(V\) for each component.
- **SCCA (Sparse CCA):** CCA with an **L1 constraint** (LASSO-like; “forces many weights to zero”) to promote **feature selection** and interpretability.
- **Permutation p-value (Stage 1):** shuffle subject pairing between X and Y, re-fit, and measure how often the null correlation ≥ observed correlation (tests whether cross-modal alignment is above chance).
- **AUC (ROC AUC):** probability a random case is ranked above a random control by the model (0.5 = chance).  
- **AP (Average Precision):** area under precision–recall curve (sensitive to prevalence; baseline AP roughly equals prevalence).
- **Holdout split:** a fixed test set **never used** for CV fitting/tuning (final unbiased evaluation).

---

## 2) Cohort + feature context (shared across experiments)

### Subjects
- **Overlap N:** 4,218 subjects (genetics + fMRI).
- **Labels in Experiment 1 Stage 2 files:**  
  - Cases: **1,735**, Controls: **2,483**  
  - Prevalence: **41.1%**

### Feature dimensions used in Experiment 1 alignment/PCA
From `derived_mean_pooling/aligned_pca/pca_info.json` and `derived_max_pooling/aligned_pca/pca_info.json`:
- Genetics input: **111** features (one scalar per gene, after pooling)
- fMRI input: **180** features
- Covariates removed (residualization; “regress out linear confounds”): **intercept + age + sex**
- PCA target was 512, but capped by input dimension → **gene_pca_dim = 111**, **fmri_pca_dim = 180**

---

## 3) Experiment 1 (original `gene-brain-CCA`): two-stage CCA/SCCA → prediction

### 3.1 Why this two-stage design?
- **Stage 1 (unsupervised CCA/SCCA):** tests whether there is a cross-modal axis where **genes and brain features covary** (association objective: maximize gene↔brain correlation).
- **Stage 2 (supervised):** tests whether the learned joint coordinates \((U,V)\) are **clinically useful** for predicting depression (prediction objective: maximize label discrimination).
- **CCA vs SCCA comparison:** if SCCA helps, it suggests the relationship is driven by a **localized subset of features** (few genes/ROIs matter) rather than a diffuse global pattern.

### 3.2 Mean pooling vs max pooling (gene representation choice)

You reduced DNABERT2 per-gene embeddings \(E \in \mathbb{R}^{N\times 768}\) into a single scalar per gene (\(N\times 111\)):
- **Mean pooling:** average over 768 embedding dimensions (smoother summary; less dominated by extreme activations).
- **Max pooling:** maximum over 768 embedding dimensions (emphasizes peaks; can amplify noise/outliers).

---

## 4) Experiment 1 results — Mean pooling (`derived_mean_pooling/`)

### 4.1 Stage 1 (unsupervised association)
From `derived_mean_pooling/cca_stage1/conventional_results.json`:
- **CCA CC1 correlation:** **0.36793897880580945**
- **Permutation p-value (CC1):** **0.03996003996003996**  
- Components beyond CC1 are not supported by permutation testing (p-values > 0.05).

From `derived_mean_pooling/scca_stage1/sparse_results.json`:
- **SCCA CC1 correlation:** **0.36794054095669865** (essentially identical to CCA)
- Reported sparsity (exact zeros): **gene = 0.0**, **fMRI = 0.0**

**Interpretation (Stage 1, mean pooling):**
- There is evidence for a real gene↔brain coupling, but it is effectively **one dominant axis** (CC1).
- With `c1=c2=0.3`, SCCA behaves like CCA here (no practical “exact-zero” feature selection in these outputs).

### 4.2 Stage 2 (supervised prediction from Stage 1 variates)
From `derived_mean_pooling/stage2_cca/cca_results.json` and `derived_mean_pooling/stage2_scca/scca_results.json`:

**CCA-derived variates (Stage2 CCA):**
| Feature set | Best model | AUC (mean ± std) |
|---|---:|---:|
| gene_only | logreg | **0.5884342917222588 ± 0.005766989395330789** |
| fmri_only | ~ | **~0.51–0.52** (near chance) |
| joint | logreg | **0.5810207718374951 ± 0.008115776020495933** |

**SCCA-derived variates (Stage2 SCCA):**
| Feature set | Best model | AUC (mean ± std) |
|---|---:|---:|
| gene_only | logreg | **0.5883608217119018 ± 0.006023726125015038** |
| fmri_only | ~ | **~0.51** (near chance) |
| joint | logreg | **0.5809891560948849 ± 0.008470681156971352** |

From `derived_mean_pooling/comparison/comparison_report.json`:
- **Conclusion:** `"similar"`  
- **Δ best AUC (SCCA − CCA):** **−7.347001035695744e−05** (negligible)

**Interpretation (Stage 2, mean pooling):**
- Depression prediction signal is **mostly on the gene side** (gene_only ~0.588 AUC).
- fMRI-only is near chance (~0.51–0.52 AUC).
- Joint (U+V) does not outperform gene-only → the gene↔brain aligned directions do not add predictive value beyond gene variates alone in this setup.

---

## 5) Experiment 1 results — Max pooling (`derived_max_pooling/`)

### 5.1 Stage 1 (unsupervised association)
From `derived_max_pooling/cca_stage1/conventional_results.json`:
- **CCA CC1 correlation:** **0.34709652490529913**
- **Permutation p-value (CC1):** **0.995004995004995**  
- No component achieves significance by permutation testing (p-values ~1.0 or otherwise > 0.05).

From `derived_max_pooling/scca_stage1/sparse_results.json`:
- **SCCA CC1 correlation:** **0.3470992316141201** (essentially identical)
- Reported sparsity (exact zeros): **gene = 0.0**, **fMRI = 0.0**

**Interpretation (Stage 1, max pooling):**
- Despite a numerically moderate CC1 correlation, permutation testing indicates it is **not above chance** (not reproducible cross-modal coupling under this representation).

### 5.2 Stage 2 (supervised prediction from Stage 1 variates)
From `derived_max_pooling/stage2_cca/cca_results.json` and `derived_max_pooling/stage2_scca/scca_results.json`:
- Best results are **near chance** overall.
- From `derived_max_pooling/comparison/comparison_report.json`:
  - **Conclusion:** `"similar"`
  - **Δ best AUC:** **0.00048446998809814623** (negligible)

**Interpretation (Stage 2, max pooling):**
- Canonical variates learned from max pooling are not clinically predictive (AUC ~0.5).

---

## 6) Experiment 1 bottom line (mean vs max)

- **Mean pooling** produced:
  - A statistically supported gene↔brain association axis (CC1 p≈0.04)
  - Modest gene-driven depression prediction from variates (AUC≈0.588)
  - No added predictive gain from including fMRI variates (joint ≤ gene_only)
- **Max pooling** produced:
  - No significant gene↔brain coupling by permutation testing (p≈1.0)
  - Near-chance supervised prediction from the variates
- In both pooling setups: **CCA and SCCA were effectively the same** in both Stage 1 correlation and Stage 2 AUC (comparison reports conclude `"similar"`).

---

## 7) Why Experiment 2 (`gene-brain-cca-2`) was necessary

Experiment 1 highlighted two core limitations that Experiment 2 explicitly addresses:

1. **Objective mismatch:** Stage 1 CCA/SCCA maximizes gene↔brain correlation, not depression prediction.  
2. **Information bottleneck:** rich embeddings → pooled scalars (111) → canonical variates (10) → prediction from 10 numbers.  
3. **Protocol rigor:** Experiment 2 uses **holdout + train-only preprocessing** (train-only PCA and train-only covariate regression) to prevent leakage.

---

## 8) Experiment 2 results (`gene-brain-cca-2`)

### 8.1 Pipeline A: Interpretable SCCA (leakage-safe association + feature weights)
From `gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json`:

**Data split (holdout):**
- n_total = 4218
- n_train = 3374
- n_holdout = 844
- pos_ratio_total = 0.4113323850165955 (stratified; train and holdout match)

**Generalization pattern (key observation):**
- Fold-wise training correlations are modest (~0.16–0.24 range across components), but
- Fold-wise validation correlations are near zero (often between ~−0.03 and ~0.07), and
- Final holdout correlations are near zero as well (some components slightly positive, many near 0 or negative).

**Sparsity (exact zeros):**
- gene sparsity ≈ **0.08198198198198198**
- fMRI sparsity ≈ **0.017777777777777778**

**Interpretability artifacts:**
- Pipeline A outputs gene and ROI weights and reports top genes/ROIs per component.
  - Example (Component 0, top genes): **NR3C1**, **CTNND2**, **ZNF165**, **KCNK2**, **CSMD1**, …
  - ROIs are reported as `roi_###` identifiers in this output.

**Interpretation (Pipeline A):**
- The gene↔brain association learned by SCCA is **not stable on validation/holdout** in this configuration (generalization is weak).
- Pipeline A remains useful as an **interpretability-focused** analysis (weights identify candidate genes/ROIs), but the generalization metrics caution against strong claims of a robust cross-modal axis without further validation.

---

### 8.2 Pipeline B: Supervised predictive suite with wide gene embeddings (plus CCA/SCCA baselines)
From `gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json`:

**Gene representation (wide):**
- \(X_{gene}\) shape: **(4218, 85248)** = 111 genes × 768 dims
- Train-only PCA: **512 components**
  - explained_variance_ratio_sum = **0.9176345467567444**

**Holdout performance highlights (AUC / AP):**
| Model | Holdout AUC | Holdout AP |
|---|---:|---:|
| gene_only_logreg | **0.7591949390869714** | 0.5960892784843602 |
| early_fusion_logreg | **0.7624362892049705** | 0.6028571727200946 |
| fmri_only_logreg | 0.5593387413820097 | 0.4531136378676749 |
| cca_joint_logreg | 0.5458398807832586 | 0.45435790447930624 |
| scca_joint_logreg | 0.5664476774189807 | 0.47982376270355925 |

**Interpretation (Pipeline B):**
- Using wide embeddings + supervised learning recovers strong predictive signal (gene-only AUC ~0.76 on holdout).
- Early fusion is slightly higher than gene-only (0.762 vs 0.759).
- Two-stage joint embeddings via CCA/SCCA underperform (AUC ~0.53–0.57), consistent with the **objective mismatch** and **bottleneck** conclusions from Experiment 1.

---

## 9) Is Experiment 2 “just SCCA”?

**No.**
- **Pipeline A**: is specifically **SCCA** used for **unsupervised association + interpretability** (weights on gene/ROI axes).
- **Pipeline B**: is primarily a **supervised** prediction benchmark suite on wide embeddings; it includes **CCA and SCCA only as baseline feature-extraction routes** (`cca_joint`, `scca_joint`) to test whether “unsupervised alignment → supervised prediction” helps.

---

## 10) Connecting back to Yoon et al. (foundation models vs PRS baseline)

**Given information about Yoon’s paper:**
- Yoon does **not** use PRS as the primary method; PRS is a **benchmark**.
- They use foundation models (Caduceus, DNABERT-2) to extract high-dimensional embeddings from raw DNA sequence (exons in 38 genes).
- Reported mean AUC ≈ **0.851**, improving over typical MDD PRS AUC ≈ **0.53–0.57**.

**How your results relate:**
- **Experiment 1** compressed embeddings heavily (768 → 1 scalar/gene → 10 variates), then predicted from 10 numbers; this is structurally unlike Yoon’s “use rich embeddings directly for supervised learning”.
- **Experiment 2 Pipeline B** moves closer to Yoon’s paradigm (retain wide embedding information via PCA512 + supervised learning) and achieves **holdout AUC ~0.76** even under the stricter overlap constraint (N=4,218).

**Primary reasons your setting differs from Yoon:**
- **Sample size:** Yoon ~28,932 vs your overlap N=4,218 (paired genetics+fMRI constraint).
- **Objective:** CCA/SCCA optimizes cross-modal correlation (gene↔brain), not depression prediction.
- **Representation:** pooling + CCA bottleneck can discard label-relevant genetic signal; wide-embedding supervised modeling preserves it.

---

## 11) Where the results live (for citation / reproducibility)

### Experiment 1
- Mean pooling: `derived_mean_pooling/`
  - Stage 1: `cca_stage1/conventional_results.json`, `scca_stage1/sparse_results.json`
  - Stage 2: `stage2_cca/cca_results.json`, `stage2_scca/scca_results.json`
  - Summary: `comparison/comparison_report.json`
- Max pooling: `derived_max_pooling/` (same structure)

### Experiment 2
- Pipeline A: `gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json`
- Pipeline B: `gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json`

---

## 12) Final takeaways (concise)

- **Experiment 1 mean pooling:** one significant gene↔brain axis (CC1 p≈0.04) + modest gene-driven prediction (AUC≈0.588), fMRI near chance, joint ≤ gene-only.  
- **Experiment 1 max pooling:** no significant coupling by permutation test; prediction ~chance.  
- **Experiment 2 Pipeline A:** leakage-safe SCCA shows weak validation/holdout gene↔brain generalization in this configuration.  
- **Experiment 2 Pipeline B:** wide embeddings + supervised learning yield strong holdout prediction (AUC ~0.76); CCA/SCCA joint-embedding baselines underperform.  
- **Link to Yoon:** your Pipeline B result is consistent with the claim that foundation-model embeddings carry strong predictive signal; the multimodal gene↔brain alignment (CCA/SCCA) does not improve MDD prediction in these runs under the overlap constraint.

