# Quick Start Guide: Gene-Brain CCA Pipeline

**⏱️ Get running in < 10 minutes**

---

## 🔬 Quick Results Summary

Before you start, here's what we found:

| Phase | Description | Best Metric | Outcome |
|-------|-------------|-------------|---------|
| **Phase 1 (Mean Pool)** | 768-D → 1 scalar/gene | AUC 0.588 | Modest prediction |
| **Phase 1 (Max Pool)** | 768-D → 1 scalar/gene | AUC 0.505 | Near chance |
| **Phase 2 (Full Embed)** | 85K → PCA 512 | **AUC 0.762** 🏆 | Best result |
| **Phase 3 (Stratified)** | 4 FM × 4 modalities | Δr ≈ 0 | No MDD-specific coupling |
| **Phase 3 (sMRI/dMRI)** | Brain → MDD | AUC 0.56 | Structural MRI fails |

**Key findings:**
1. Scalar pooling loses too much information - use full embeddings (AUC 0.762)
2. Gene-brain coupling does NOT differ between MDD and Controls
3. Structural MRI (sMRI/dMRI) provides no MDD prediction signal

---

## Prerequisites Check (1 minute)

```bash
# 1. Navigate to project root
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# 2. Check conda environment
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# 3. Verify dependencies
python -c "from sklearn.cross_decomposition import CCA; print('✓ sklearn OK')"
python -c "from cca_zoo.linear import SCCA_PMD; print('✓ cca-zoo OK')"

# 4. Create logs directory
mkdir -p logs
```

---

## Option A: Run Full Pipeline (Recommended)

The full pipeline compares mean vs max pooling, CCA vs SCCA:

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Submit the comparison pipeline
sbatch slurm/07_full_pipeline.sbatch
```

**Monitor progress:**
```bash
tail -f logs/full_pipeline_*.out
squeue -u $USER
```

**⏱️ Runtime:** ~6-8 hours total

---

## Option B: Run Step-by-Step

### Step 1: Build Gene Features (~30 min)

```bash
sbatch slurm/01_gene_x.sbatch
# Or run interactively:
python scripts/build_x_gene.py \
  --embed-root /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged \
  --gene-list /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt \
  --iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --n-files 49 \
  --reduce mean \
  --out-dir derived_mean_pooling/gene_x
```

**Output:** `derived_mean_pooling/gene_x/X_gene_ng.npy` (N × 111 genes)

### Step 2: Build fMRI Features (~1 hour)

```bash
sbatch slurm/00_fmri_fc.sbatch
# Or run interactively:
python scripts/build_x_fmri_fc.py \
  --root /path/to/UKB_ROI \
  --glob "*_20227_2_0/hcp_mmp1_*.npy" \
  --out-dir derived/fmri_fc
```

**Output:** `derived/fmri_fc/X_fmri_fc.npy` (N × 16,110 FC features)

### Step 3: Align + PCA (~10 min)

```bash
sbatch slurm/02_align_pca.sbatch
```

**Output:** `derived_mean_pooling/aligned_pca/` with aligned matrices

### Step 4: Stage 1 - CCA (~30 min)

```bash
sbatch slurm/04_cca_stage1.sbatch
```

**Output:** `derived_mean_pooling/cca_stage1/conventional_results.json`

### Step 5: Stage 2 - Prediction (~1 hour)

```bash
sbatch slurm/06_stage2_predict.sbatch
```

**Output:** `derived_mean_pooling/stage2_cca/cca_results.json`

---

## View Results (1 minute)

### Stage 1: Gene-Brain Correlation

```bash
cat derived_mean_pooling/cca_stage1/conventional_results.json | python -m json.tool
```

**Key metrics:**
- `canonical_correlations`: Gene-brain coupling strength
- `p_perm`: Permutation p-values (< 0.05 = significant)

### Stage 2: MDD Prediction

```bash
cat derived_mean_pooling/stage2_cca/cca_results.json | python -m json.tool
```

**Key metrics:**
- `auc_mean`: Cross-validated AUC (0.5 = chance, 1.0 = perfect)
- `auc_std`: Standard deviation across folds

### Compare CCA vs SCCA

```bash
cat derived_mean_pooling/comparison/comparison_report.json | python -m json.tool
```

---

## Expected Results

### Mean Pooling (Experiment 1)

| Metric | CCA | SCCA |
|--------|-----|------|
| **CC1 (Stage 1)** | 0.368 | 0.368 |
| **p-value** | 0.040 ✅ | 0.040 ✅ |
| **Gene-only AUC** | 0.588 | 0.588 |
| **Joint AUC** | 0.581 | 0.581 |

### Max Pooling (Experiment 1)

| Metric | CCA | SCCA |
|--------|-----|------|
| **CC1 (Stage 1)** | 0.347 | 0.347 |
| **p-value** | 0.995 ❌ | 0.995 ❌ |
| **Gene-only AUC** | 0.505 | 0.494 |

**Conclusion:** Mean pooling works; max pooling fails. Both are inferior to full embeddings (AUC 0.762 in gene-brain-cca-2).

---

## Next Steps

After running Experiment 1, you should:

1. **Try Experiment 2 (recommended):**
   ```bash
   cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
   sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
   sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch
   ```

2. **Read the comprehensive report:**
   ```bash
   cat final_report/comprehensive_report.md
   ```

3. **Explore results:**
   See `INDEX.md` for documentation navigation

---

## Common Issues

**Job stuck in queue:**
```bash
squeue -u $USER
scancel <JOBID>  # Cancel if needed
```

**Module not found:**
```bash
conda activate /scratch/connectome/allie/envs/cca_env
pip install cca-zoo scikit-learn numpy
```

**Dimension mismatch:**
```bash
python -c "import numpy as np; print(np.load('derived_mean_pooling/aligned_pca/X_gene_pca111.npy').shape)"
```

---

## File Locations

| Component | Path |
|-----------|------|
| Project root | `/storage/bigdata/UKB/fMRI/gene-brain-CCA` |
| Pipeline scripts | `scripts/` |
| SLURM launchers | `slurm/` |
| Mean pooling results | `derived_mean_pooling/` |
| Max pooling results | `derived_max_pooling/` |
| Experiment 2 | `gene-brain-cca-2/` |
| Final report | `final_report/` |

---

---

## Phase 3: Stratified Analysis (Optional)

If you want to test whether gene-brain coupling differs between MDD and Controls:

```bash
# Run stratified benchmark for a specific FM model and modality
sbatch slurm/51_stratified_fm.sbatch  # Edit to select FM_MODEL and MODALITY

# After completion, generate visualizations
python scripts/plot_stratified_results.py \
  --derived-dir gene-brain-cca-2/derived/stratified_fm \
  --out-dir figures/stratified
```

**Result**: No significant MDD-specific coupling was found for any FM model or modality.

---

**Need more help?** See `README.md` for full documentation or `INDEX.md` for navigation guide.

**Last updated:** January 28, 2026
