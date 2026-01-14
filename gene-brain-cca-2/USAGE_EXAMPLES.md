# Usage Examples

Concrete examples of how to use the `gene-brain-cca-2` pipelines for common scenarios.

---

## Scenario 1: First-Time User – Running Everything

**Goal:** Run both pipelines from scratch and view results.

```bash
# Step 1: Navigate to project
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Step 2: Verify everything is set up
bash gene-brain-cca-2/scripts/verify_setup.sh

# Step 3: Activate environment
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Step 4: Create logs directory
mkdir -p logs

# Step 5: Submit Pipeline A
sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
# Note the job ID (e.g., "Submitted batch job 12345678")

# Step 6: Monitor Pipeline A
tail -f logs/interp_scca_12345678.out
# Press Ctrl+C to exit when done

# Step 7: Wait for completion, then submit Pipeline B
# Check if Pipeline A finished:
ls gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json

# If file exists, submit Pipeline B:
sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch

# Step 8: Monitor Pipeline B
tail -f logs/wide_suite_<JOBID>.out

# Step 9: View results once both complete
python gene-brain-cca-2/scripts/view_results.py
```

**Time required:** ~12 hours total (4h + 8h), but you can submit and walk away.

---

## Scenario 2: Quick Test on Existing Data

**Goal:** Just re-run SCCA interpretation with different hyperparameters.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Assuming Pipeline A data preparation already done
# Re-run just the SCCA step with higher sparsity
python gene-brain-cca-2/scripts/run_scca_interpretable.py \
  --x-gene gene-brain-cca-2/derived/interpretable/X_gene_z.npy \
  --x-fmri gene-brain-cca-2/derived/interpretable/X_fmri_z.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --ids gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --c1 0.5 --c2 0.5 \
  --k 10 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/interpretable/scca_high_sparsity.json

# Compare to original results
echo "=== Original (c1=c2=0.3) ==="
cat gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json | \
  python -m json.tool | grep -A 3 '"full"'

echo "=== High sparsity (c1=c2=0.5) ==="
cat gene-brain-cca-2/derived/interpretable/scca_high_sparsity.json | \
  python -m json.tool | grep -A 3 '"full"'
```

---

## Scenario 3: Testing Different PCA Dimensionalities

**Goal:** See if PCA512 is optimal or if PCA256 works just as well (and is faster).

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Assuming X_gene_wide.npy already exists from Pipeline B step 1

# Create PCA256 version
python gene-brain-cca-2/scripts/pca_gene_wide.py \
  --x-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
  --n-components 256 \
  --seed 42 \
  --out gene-brain-cca-2/derived/wide_gene/X_gene_pca256.npy

# Run predictive suite with PCA256
python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene gene-brain-cca-2/derived/wide_gene/X_gene_pca256.npy \
  --x-fmri gene-brain-cca-2/derived/interpretable/X_fmri_z.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/wide_gene/predictive_pca256_results.json

# Compare AUCs
echo "=== PCA512 ==="
cat gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json | \
  python -m json.tool | grep -A 1 '"early_fusion_logreg"'

echo "=== PCA256 ==="
cat gene-brain-cca-2/derived/wide_gene/predictive_pca256_results.json | \
  python -m json.tool | grep -A 1 '"early_fusion_logreg"'
```

---

## Scenario 4: Checking Data Quality Before Running

**Goal:** Ensure data files are valid before submitting long jobs.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Run the verification script
bash gene-brain-cca-2/scripts/verify_setup.sh

# Manually inspect a few key files
python3 << 'EOF'
import numpy as np

def check_file(path, name):
    try:
        arr = np.load(path, allow_pickle=True)
        print(f"✓ {name}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        if np.issubdtype(arr.dtype, np.number):
            print(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]")
            print(f"  NaNs: {np.isnan(arr).sum()}")
        print()
    except Exception as e:
        print(f"✗ {name}: {e}\n")

# Check genetics data
check_file('derived_max_pooling/gene_x/ids_gene.npy', 'Gene IDs')
check_file('derived_max_pooling/gene_x/X_gene_ng.npy', 'Gene matrix')

# Check fMRI data
check_file('/storage/bigdata/UKB/fMRI/fmri_eids_180.npy', 'fMRI IDs')
check_file('/storage/bigdata/UKB/fMRI/fmri_X_180.npy', 'fMRI matrix')

# Check overlap
ids_g = set(np.load('derived_max_pooling/gene_x/ids_gene.npy', allow_pickle=True).astype(str))
ids_f = set(np.load('/storage/bigdata/UKB/fMRI/fmri_eids_180.npy', allow_pickle=True).astype(str))
overlap = ids_g & ids_f
print(f"Expected overlap: {len(overlap)} subjects")
print(f"(Should be ~4,218)")
EOF
```

---

## Scenario 5: Debugging a Failed Job

**Goal:** Figure out why a SLURM job failed.

```bash
# Step 1: Check job status
squeue -u $USER

# Step 2: Find the failed job ID (from email or sacct)
sacct -u $USER --format=JobID,JobName,State,ExitCode -S today

# Step 3: Check error log
cat logs/interp_scca_<JOBID>.err

# Step 4: Check output log for last messages
tail -50 logs/interp_scca_<JOBID>.out

# Step 5: Run the command interactively to see full traceback
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

# Copy the exact python command from the sbatch file and run it
# Example:
python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py \
  --ids-gene derived_max_pooling/gene_x/ids_gene.npy \
  --x-gene derived_max_pooling/gene_x/X_gene_ng.npy \
  --ids-fmri /storage/bigdata/UKB/fMRI/fmri_eids_180.npy \
  --x-fmri /storage/bigdata/UKB/fMRI/fmri_X_180.npy \
  --cov-iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --cov-age /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_age.npy \
  --cov-sex /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_sex.npy \
  --cov-valid-mask /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_valid_mask.npy \
  --labels /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/labels.npy \
  --out-dir gene-brain-cca-2/derived/interpretable

# Now you'll see the full Python traceback
```

---

## Scenario 6: Running Only Pipeline B (Gene-wide) on Existing Overlap

**Goal:** You already have `ids_common.npy` and want to skip Pipeline A entirely.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Verify ids_common.npy exists
ls gene-brain-cca-2/derived/interpretable/ids_common.npy

# Submit only the gene-wide pipeline
sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch

# Or run steps manually:
mkdir -p gene-brain-cca-2/derived/wide_gene

# Step 1: Build gene-wide matrix
python gene-brain-cca-2/scripts/build_x_gene_wide.py \
  --embed-root /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged \
  --gene-list /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt \
  --iids /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy \
  --ids-keep gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --n-files 49 \
  --out-dir gene-brain-cca-2/derived/wide_gene

# Step 2: PCA reduction
python gene-brain-cca-2/scripts/pca_gene_wide.py \
  --x-wide gene-brain-cca-2/derived/wide_gene/X_gene_wide.npy \
  --n-components 512 \
  --seed 42 \
  --out gene-brain-cca-2/derived/wide_gene/X_gene_pca512.npy

# Step 3: Predictive suite
python gene-brain-cca-2/scripts/run_predictive_suite.py \
  --x-gene gene-brain-cca-2/derived/wide_gene/X_gene_pca512.npy \
  --x-fmri gene-brain-cca-2/derived/interpretable/X_fmri_z.npy \
  --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
  --k 10 --c1 0.3 --c2 0.3 --n-folds 5 --seed 42 \
  --out-json gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json
```

---

## Scenario 7: Generating Publication-Quality Figures

**Goal:** Create plots for a paper or presentation.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Create a plotting script
cat > plot_results.py << 'EOF'
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

out_dir = Path('gene-brain-cca-2/figures')
out_dir.mkdir(exist_ok=True)

# =========================
# Figure 1: SCCA Scree Plot
# =========================
with open('gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json') as f:
    scca = json.load(f)

fig, ax = plt.subplots(figsize=(6, 4))
r = scca['full']['r'][:10]
ax.bar(range(1, len(r)+1), r, color='steelblue', edgecolor='black')
ax.set_xlabel('Component')
ax.set_ylabel('Canonical Correlation (r)')
ax.set_title('SCCA: Gene-Brain Correlations')
ax.axhline(0.1, color='red', linestyle='--', linewidth=1, label='r=0.1')
ax.legend()
ax.set_ylim([0, max(r)*1.1])
plt.tight_layout()
plt.savefig(out_dir / 'scca_scree.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'scca_scree.pdf', bbox_inches='tight')
print(f"✓ Saved: {out_dir}/scca_scree.[png|pdf]")

# ================================
# Figure 2: AUC Comparison Bar Plot
# ================================
with open('gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json') as f:
    pred = json.load(f)

models = ['gene_only_logreg', 'fmri_only_logreg', 'early_fusion_logreg', 
          'cca_joint_logreg', 'scca_joint_logreg']
labels = ['Gene\nOnly', 'fMRI\nOnly', 'Early\nFusion', 'CCA', 'SCCA']
aucs = [pred[m]['auc'] for m in models]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4ECDC4', '#FF6B6B', '#95E77D', '#FFE66D', '#C7CEEA']
bars = ax.bar(labels, aucs, color=colors, edgecolor='black', linewidth=1.5)
ax.set_ylabel('AUC')
ax.set_title('Classification Performance Comparison')
ax.set_ylim([0.5, 1.0])
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')

# Add value labels
for bar, auc in zip(bars, aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

ax.legend()
plt.tight_layout()
plt.savefig(out_dir / 'auc_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'auc_comparison.pdf', bbox_inches='tight')
print(f"✓ Saved: {out_dir}/auc_comparison.[png|pdf]")

# ===============================
# Figure 3: Component 1 Scatter
# ===============================
U = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_U.npy')
V = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_V.npy')
y = np.load('gene-brain-cca-2/derived/interpretable/labels_common.npy')

fig, ax = plt.subplots(figsize=(6, 6))
scatter = ax.scatter(U[:, 0], V[:, 0], c=y, cmap='RdBu_r', alpha=0.6, s=20, edgecolor='none')
ax.set_xlabel('Gene Component 1')
ax.set_ylabel('Brain Component 1')
ax.set_title(f'SCCA Component 1 (r={scca["full"]["r"][0]:.3f})')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Label')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / 'scca_component1_scatter.png', dpi=300, bbox_inches='tight')
plt.savefig(out_dir / 'scca_component1_scatter.pdf', bbox_inches='tight')
print(f"✓ Saved: {out_dir}/scca_component1_scatter.[png|pdf]")

print("\n✓ All figures saved to gene-brain-cca-2/figures/")
EOF

# Run the plotting script
python plot_results.py
```

---

## Scenario 8: Extracting Results for Statistical Analysis

**Goal:** Export results to CSV for analysis in R or other tools.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA

python3 << 'EOF'
import json
import numpy as np
import pandas as pd

# Load results
with open('gene-brain-cca-2/derived/interpretable/scca_interpretable_results.json') as f:
    scca = json.load(f)
with open('gene-brain-cca-2/derived/wide_gene/predictive_suite_results.json') as f:
    pred = json.load(f)

# Export SCCA fold-wise results
folds_data = []
for fold_res in scca['folds']:
    for i, r in enumerate(fold_res['r'][:5]):  # Top 5 components
        folds_data.append({
            'fold': fold_res['fold'],
            'component': i+1,
            'correlation': r,
            'gene_sparsity': fold_res['sparsity_gene'],
            'fmri_sparsity': fold_res['sparsity_fmri']
        })

df_scca = pd.DataFrame(folds_data)
df_scca.to_csv('gene-brain-cca-2/derived/scca_foldwise.csv', index=False)
print("✓ Saved: gene-brain-cca-2/derived/scca_foldwise.csv")

# Export predictive suite results
pred_data = []
for model, metrics in pred.items():
    pred_data.append({
        'model': model,
        'auc': metrics['auc'],
        'ap': metrics['ap']
    })

df_pred = pd.DataFrame(pred_data)
df_pred.to_csv('gene-brain-cca-2/derived/predictive_results.csv', index=False)
print("✓ Saved: gene-brain-cca-2/derived/predictive_results.csv")

# Export component scores for further analysis
U = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_U.npy')
V = np.load('gene-brain-cca-2/derived/interpretable/scca_interpretable_results_V.npy')
ids = np.load('gene-brain-cca-2/derived/interpretable/ids_common.npy', allow_pickle=True)
labels = np.load('gene-brain-cca-2/derived/interpretable/labels_common.npy')

# Create DataFrame with first 3 components
df_scores = pd.DataFrame({
    'subject_id': ids,
    'label': labels,
    'U1': U[:, 0], 'U2': U[:, 1], 'U3': U[:, 2],
    'V1': V[:, 0], 'V2': V[:, 1], 'V3': V[:, 2],
})
df_scores.to_csv('gene-brain-cca-2/derived/scca_scores.csv', index=False)
print("✓ Saved: gene-brain-cca-2/derived/scca_scores.csv")
EOF
```

---

## Scenario 9: Batch Testing Multiple Hyperparameter Combinations

**Goal:** Grid search over SCCA penalties to find optimal interpretability-correlation trade-off.

```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env

# Create a grid search script
cat > grid_search_scca.sh << 'EOF'
#!/bin/bash
set -e

mkdir -p gene-brain-cca-2/derived/grid_search

for c1 in 0.1 0.2 0.3 0.5 0.7; do
  for c2 in 0.1 0.2 0.3 0.5 0.7; do
    echo "Running c1=$c1, c2=$c2..."
    
    python gene-brain-cca-2/scripts/run_scca_interpretable.py \
      --x-gene gene-brain-cca-2/derived/interpretable/X_gene_z.npy \
      --x-fmri gene-brain-cca-2/derived/interpretable/X_fmri_z.npy \
      --labels gene-brain-cca-2/derived/interpretable/labels_common.npy \
      --ids gene-brain-cca-2/derived/interpretable/ids_common.npy \
      --c1 $c1 --c2 $c2 \
      --k 10 --n-folds 5 --seed 42 \
      --out-json gene-brain-cca-2/derived/grid_search/scca_c1${c1}_c2${c2}.json
  done
done

echo "✓ Grid search complete!"
EOF

chmod +x grid_search_scca.sh
./grid_search_scca.sh

# Summarize results
python3 << 'EOF'
import json
from pathlib import Path
import pandas as pd

results = []
for jf in Path('gene-brain-cca-2/derived/grid_search').glob('scca_c1*.json'):
    with open(jf) as f:
        data = json.load(f)
    
    # Parse hyperparameters from filename
    parts = jf.stem.replace('scca_', '').split('_')
    c1 = float(parts[0].replace('c1', ''))
    c2 = float(parts[1].replace('c2', ''))
    
    results.append({
        'c1': c1,
        'c2': c2,
        'r1': data['full']['r'][0],
        'gene_sparsity': data['full']['sparsity_gene'],
        'fmri_sparsity': data['full']['sparsity_fmri'],
    })

df = pd.DataFrame(results).sort_values('r1', ascending=False)
print(df.to_string(index=False))
df.to_csv('gene-brain-cca-2/derived/grid_search_summary.csv', index=False)
EOF
```

---

## Additional Resources

- **Full documentation:** [README.md](README.md)
- **Quick start:** [QUICKSTART.md](QUICKSTART.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Results interpretation:** [RESULTS_GUIDE.md](RESULTS_GUIDE.md)

---

**Last updated:** January 2026
