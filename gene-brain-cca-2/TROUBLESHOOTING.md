# Troubleshooting Guide: gene-brain-cca-2

This guide covers common issues and their solutions. Issues are organized by pipeline stage.

---

## Table of Contents

1. [Pre-Submission Issues](#pre-submission-issues)
2. [Pipeline A (Interpretable SCCA)](#pipeline-a-interpretable-scca)
3. [Pipeline B (Predictive Suite)](#pipeline-b-predictive-suite)
4. [General SLURM Issues](#general-slurm-issues)
5. [Data Quality Issues](#data-quality-issues)
6. [Performance Issues](#performance-issues)
7. [Debugging Strategies](#debugging-strategies)

---

## Pre-Submission Issues

### ❌ `sbatch: command not found`

**Cause:** Not on a SLURM-enabled system or SLURM not in PATH.

**Solutions:**
1. Check if on correct server:
   ```bash
   hostname  # Should be a compute cluster
   which sbatch  # Should return path like /usr/bin/sbatch
   ```

2. If no SLURM, run interactively (see README.md "Option 2: Interactive")

---

### ❌ `conda: command not found`

**Cause:** Conda not initialized in current shell.

**Solution:**
```bash
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
```

**Make permanent:**
Add to `~/.bashrc`:
```bash
# Add these lines to ~/.bashrc
source /usr/anaconda3/etc/profile.d/conda.sh
```

Then:
```bash
source ~/.bashrc
```

---

### ❌ `cca_env` conda environment doesn't exist

**Symptom:**
```
EnvironmentNameNotFound: Could not find conda environment: /scratch/connectome/allie/envs/cca_env
```

**Solution:**
1. List available environments:
   ```bash
   conda env list
   ```

2. If missing, create it:
   ```bash
   conda create -p /scratch/connectome/allie/envs/cca_env python=3.9
   conda activate /scratch/connectome/allie/envs/cca_env
   pip install numpy scikit-learn cca-zoo matplotlib
   ```

---

### ❌ Missing `logs/` directory

**Symptom:**
```
sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified
```
or
```
sbatch: error: Unable to open file logs/interp_scca_XXXXX.out
```

**Solution:**
```bash
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
mkdir -p logs
```

---

## Pipeline A (Interpretable SCCA)

### ❌ `FileNotFoundError: ids_gene.npy`

**Symptom:**
```
FileNotFoundError: /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/ids_gene.npy
```

**Diagnostic:**
```bash
ls -lh /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/
```

**Solution:**
1. Verify upstream gene data preparation completed
2. Check file permissions:
   ```bash
   ls -lh /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_max_pooling/gene_x/*.npy
   ```
3. If files missing, regenerate gene scalar matrix (contact data maintainer)

---

### ❌ `SystemExit: Install cca-zoo for SCCA_PMD`

**Cause:** `cca-zoo` package not installed.

**Solution:**
```bash
conda activate /scratch/connectome/allie/envs/cca_env
pip install cca-zoo

# Verify installation
python -c "from cca_zoo.linear import SCCA_PMD; print('Success')"
```

---

### ❌ No overlap subjects found (n=0)

**Symptom:**
```
[save] n=0, gene_dim=111, fmri_dim=180
```

**Cause:** Subject IDs don't match between gene and fMRI cohorts.

**Diagnostic:**
```python
import numpy as np
ids_gene = np.load('derived_max_pooling/gene_x/ids_gene.npy', allow_pickle=True).astype(str)
ids_fmri = np.load('/storage/bigdata/UKB/fMRI/fmri_eids_180.npy', allow_pickle=True).astype(str)
print(f"Gene IDs sample: {ids_gene[:5]}")
print(f"fMRI IDs sample: {ids_fmri[:5]}")
overlap = set(ids_gene) & set(ids_fmri)
print(f"Overlap: {len(overlap)} subjects")
```

**Solutions:**
1. Check ID format consistency (string vs int, leading zeros)
2. Verify correct fMRI file: should have ~40K subjects
3. Verify gene file: should have ~29K subjects
4. Expected overlap: ~4,218 subjects

---

### ❌ SCCA convergence warnings

**Symptom:**
```
ConvergenceWarning: Maximum number of iterations has been exceeded
```

**Impact:** Results may be unstable, but often acceptable.

**Solutions:**
1. Increase max iterations in `run_scca_interpretable.py`:
   ```python
   m = SCCA_PMD(..., max_iter=1000)  # default is 500
   ```

2. Relax tolerance:
   ```python
   m = SCCA_PMD(..., tol=1e-4)  # default is 1e-6
   ```

3. Reduce sparsity penalty (easier to converge):
   ```bash
   --c1 0.1 --c2 0.1  # instead of 0.3
   ```

---

### ❌ Out of memory during residualization

**Symptom:**
```
MemoryError: Unable to allocate X.XX GiB for an array
```

**Solution:**
Increase SLURM memory allocation in `01_interpretable_scca.sbatch`:
```bash
#SBATCH --mem=64G  # up from 32G
```

---

## Pipeline B (Predictive Suite)

### ❌ `FileNotFoundError: ids_common.npy`

**Cause:** Pipeline A didn't complete successfully.

**Solution:**
1. Check Pipeline A completion:
   ```bash
   ls -lh gene-brain-cca-2/derived/interpretable/ids_common.npy
   ```

2. If missing, run Pipeline A first:
   ```bash
   sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch
   ```

3. Wait for completion (check `squeue -u $USER`)

---

### ❌ Gene embedding files not found

**Symptom:**
```
FileNotFoundError: /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/<gene>/embeddings_1_layer_last.npy
```

**Diagnostic:**
```bash
ls /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/ | head
```

**Solutions:**
1. Verify embedding directory structure:
   ```bash
   # Should have subdirectories named after genes
   ls /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/
   ```

2. Check a specific gene:
   ```bash
   GENE=$(head -1 /storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt)
   ls /storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged/$GENE/
   ```

3. Verify file count (should have embeddings_1.npy through embeddings_49.npy)

---

### ❌ Out of memory during gene-wide build

**Symptom:**
```
slurmstepd: error: Detected 1 oom-kill event(s) in step XXXXX.batch
```

**Cause:** Building 4,218 × 85,248 matrix (~1.4 GB) requires significant temporary memory.

**Solutions:**
1. Increase memory allocation in `02_predictive_wide_suite.sbatch`:
   ```bash
   #SBATCH --mem=256G  # up from 128G
   ```

2. Process genes in chunks (modify `build_x_gene_wide.py` to save incrementally)

3. Use memory-mapped arrays (already implemented via `mmap_mode="r"`)

---

### ❌ PCA fails with "array is too big"

**Symptom:**
```
ValueError: array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.
```

**Cause:** 85,248 dimensions exceed single-array size limits on 32-bit systems.

**Solutions:**
1. Verify you're on 64-bit system:
   ```bash
   python -c "import sys; print(sys.maxsize > 2**32)"  # Should print True
   ```

2. Reduce PCA components:
   ```bash
   --n-components 256  # instead of 512
   ```

3. Use incremental PCA (modify `pca_gene_wide.py`):
   ```python
   from sklearn.decomposition import IncrementalPCA
   pca = IncrementalPCA(n_components=512, batch_size=500)
   ```

---

### ❌ Predictive suite shows poor performance (AUC < 0.55)

**Not necessarily an error!** But check:

1. **Label distribution:**
   ```python
   import numpy as np
   y = np.load('gene-brain-cca-2/derived/interpretable/labels_common.npy')
   print(f"Label distribution: {np.bincount(y)}")
   ```
   Severe imbalance may require different metrics or class weighting.

2. **Data leakage check:**
   Ensure SCCA/CCA fit inside CV folds (already implemented in code).

3. **Feature scaling:**
   Verify z-scoring completed (check `X.mean()` ≈ 0, `X.std()` ≈ 1).

4. **Random baseline:**
   AUC should be > 0.5 (random guessing). If ≈ 0.5, features may lack signal.

---

## General SLURM Issues

### ❌ Job pending for hours

**Check queue:**
```bash
squeue -u $USER
squeue  # See all jobs
```

**Reasons:**
- **`(Priority)`**: Other jobs ahead in queue
- **`(Resources)`**: Cluster busy, waiting for resources
- **`(AssocGrpCPULimit)`**: Account CPU limit reached

**Solutions:**
1. Reduce resource requirements:
   ```bash
   #SBATCH --mem=64G       # instead of 128G
   #SBATCH --cpus-per-task=8  # instead of 16
   ```

2. Use different partition (if available):
   ```bash
   #SBATCH --partition=short
   ```

3. Check cluster status:
   ```bash
   sinfo
   ```

---

### ❌ Job fails immediately (exit code 1)

**Check error log:**
```bash
cat logs/interp_scca_<JOBID>.err
```

**Common causes:**
- Python import errors → Check conda environment
- File not found → Verify paths in sbatch script
- Permission denied → Check file ownership

**Debug by running interactively:**
```bash
# Copy commands from sbatch file and run manually
source /usr/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/connectome/allie/envs/cca_env
cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py --help
```

---

### ❌ Job times out

**Symptom:**
```
CANCELLED AT 2026-01-13T08:00:00 DUE TO TIME LIMIT
```

**Solution:**
Increase time limit in sbatch script:
```bash
#SBATCH --time=12:00:00  # 12 hours instead of 8
```

**Optimize runtime:**
- Reduce `--n-folds` (fewer CV splits)
- Reduce `--k` (fewer CCA components)
- Use faster solvers (e.g., `svd_solver="randomized"` for PCA)

---

## Data Quality Issues

### ❌ NaN or Inf values in output

**Diagnostic:**
```python
import numpy as np
X = np.load('gene-brain-cca-2/derived/interpretable/X_gene_z.npy')
print(f"NaNs: {np.isnan(X).sum()}")
print(f"Infs: {np.isinf(X).sum()}")
print(f"Min: {X.min()}, Max: {X.max()}")
```

**Solutions:**
1. Check input data quality
2. Residualization may fail if covariates have missing values
3. Z-scoring fails if feature has zero variance:
   ```python
   # In prepare_overlap_no_pca.py, already handled:
   sd = np.where(sd == 0, 1, sd)
   ```

---

### ❌ Extremely low canonical correlations (r < 0.01)

**Possible causes:**
1. **Incorrect alignment:** Gene and fMRI subjects don't match
   ```python
   # Verify IDs match after prepare_overlap_no_pca.py
   ids = np.load('gene-brain-cca-2/derived/interpretable/ids_common.npy', allow_pickle=True)
   Xg = np.load('gene-brain-cca-2/derived/interpretable/X_gene_z.npy')
   Xb = np.load('gene-brain-cca-2/derived/interpretable/X_fmri_z.npy')
   assert len(ids) == len(Xg) == len(Xb)
   ```

2. **No true association:** Gene and brain may genuinely have weak correlation
3. **Over-regularization:** SCCA penalties too high
   ```bash
   --c1 0.05 --c2 0.05  # Try lower penalties
   ```

---

### ❌ Suspiciously perfect correlations (r > 0.95)

**Likely data leakage!**

**Check:**
1. Are train/test splits correct?
2. Was normalization fit on entire dataset instead of train only?
3. Are gene and brain features accidentally identical?

**Verify:**
```python
import numpy as np
from scipy.stats import pearsonr
Xg = np.load('gene-brain-cca-2/derived/interpretable/X_gene_z.npy')
Xb = np.load('gene-brain-cca-2/derived/interpretable/X_fmri_z.npy')
# They should NOT be highly correlated column-wise
for i in range(min(5, Xg.shape[1])):
    r, _ = pearsonr(Xg[:, i], Xb[:, 0])
    print(f"Gene {i} vs Brain 0: r={r:.3f}")
```

---

## Performance Issues

### ⏱️ Pipeline A takes > 6 hours

**Expected:** ~4 hours on 8 CPUs, 32GB RAM

**Slow if:**
- Fewer CPUs allocated
- Running on busy shared node
- Many CV folds (`--n-folds` > 10)

**Speed up:**
1. Reduce CV folds:
   ```bash
   --n-folds 3  # instead of 5
   ```

2. Reduce SCCA components:
   ```bash
   --k 5  # instead of 10
   ```

3. Request more CPUs:
   ```bash
   #SBATCH --cpus-per-task=16
   ```

---

### ⏱️ Pipeline B takes > 12 hours

**Expected:** ~8 hours on 16 CPUs, 128GB RAM

**Bottlenecks:**
1. **Gene-wide build** (~2 hours): I/O-bound, loading 111×49 files
2. **PCA** (~1 hour): CPU-bound, 85K dimensions
3. **Predictive suite** (~5 hours): Many model fits (3 baselines × 2 classifiers + CCA + SCCA × 2 classifiers)

**Speed up:**
1. Use randomized PCA (already default):
   ```python
   pca = PCA(n_components=512, svd_solver="randomized")
   ```

2. Reduce PCA components:
   ```bash
   --n-components 256
   ```

3. Reduce CV folds or models in `run_predictive_suite.py` (comment out MLP if only need LogReg)

---

## Debugging Strategies

### 🔍 General Debugging Workflow

1. **Read the error log:**
   ```bash
   cat logs/<job_name>_<JOBID>.err
   ```

2. **Run script interactively:**
   ```bash
   conda activate /scratch/connectome/allie/envs/cca_env
   cd /storage/bigdata/UKB/fMRI/gene-brain-CCA
   # Copy command from sbatch file and run with --help first
   python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py --help
   ```

3. **Test on small subset:**
   Load first 100 subjects in Python:
   ```python
   import numpy as np
   Xg = np.load('X_gene.npy')[:100]  # First 100 subjects only
   ```

4. **Check intermediate outputs:**
   ```bash
   # Verify each step's output
   ls -lh gene-brain-cca-2/derived/interpretable/
   python -c "import numpy as np; X = np.load('X_gene_z.npy'); print(X.shape)"
   ```

5. **Enable verbose output:**
   Add `print()` statements in Python scripts to track progress.

---

### 🔍 Verifying Data Integrity

**Template script:**
```python
import numpy as np

def check_array(path, expected_shape=None, expected_dtype=None):
    print(f"\n=== {path} ===")
    try:
        X = np.load(path, allow_pickle=True)
        print(f"Shape: {X.shape}")
        print(f"Dtype: {X.dtype}")
        print(f"NaNs: {np.isnan(X).sum() if np.issubdtype(X.dtype, np.number) else 'N/A'}")
        print(f"Infs: {np.isinf(X).sum() if np.issubdtype(X.dtype, np.number) else 'N/A'}")
        if np.issubdtype(X.dtype, np.number):
            print(f"Range: [{X.min():.3f}, {X.max():.3f}]")
        if expected_shape and X.shape != expected_shape:
            print(f"⚠️  WARNING: Expected {expected_shape}")
        print("✓ OK")
    except Exception as e:
        print(f"✗ ERROR: {e}")

# Check Pipeline A outputs
check_array('gene-brain-cca-2/derived/interpretable/X_gene_z.npy', expected_shape=(4218, 111))
check_array('gene-brain-cca-2/derived/interpretable/X_fmri_z.npy', expected_shape=(4218, 180))
check_array('gene-brain-cca-2/derived/interpretable/labels_common.npy', expected_shape=(4218,))
```

Save as `check_data.py` and run:
```bash
python check_data.py
```

---

### 🔍 Profiling Performance

**Find slow steps:**
```bash
# Add timestamps to sbatch scripts
time python gene-brain-cca-2/scripts/prepare_overlap_no_pca.py ...
time python gene-brain-cca-2/scripts/run_scca_interpretable.py ...
```

**Monitor resource usage:**
```bash
# While job is running
sstat -j <JOBID> --format=JobID,MaxRSS,MaxVMSize,AveCPU
```

---

## Still Stuck?

**Before asking for help, collect:**
1. Full error message (from `.err` log file)
2. SLURM job ID
3. Output of diagnostic commands above
4. Which pipeline/step failed
5. Whether this is first time running or was working before

**Contact:** Project maintainer with the above information.

---

**Last updated:** January 2026
