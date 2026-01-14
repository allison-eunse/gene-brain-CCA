#!/bin/bash
# Pre-flight check script for gene-brain-cca-2 pipelines
# Run this before submitting jobs to verify all dependencies and data files

set -u  # Exit on undefined variables

echo "=========================================="
echo "gene-brain-cca-2 Pre-flight Verification"
echo "=========================================="
echo ""

ERRORS=0
WARNINGS=0

# Project root (needed early for some checks)
PROJECT_ROOT="/storage/bigdata/UKB/fMRI/gene-brain-CCA"

# Colors for output (if terminal supports it)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'  # No Color
else
    GREEN=''
    YELLOW=''
    RED=''
    NC=''
fi

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

# === 1. Environment Checks ===
echo "--- Environment ---"

# Check conda
if command -v conda &> /dev/null; then
    check_pass "conda command available"
else
    check_fail "conda not found. Run: source /usr/anaconda3/etc/profile.d/conda.sh"
fi

# Check conda environment
if [ -d "/scratch/connectome/allie/envs/cca_env" ]; then
    check_pass "cca_env conda environment exists"
else
    check_fail "cca_env not found at /scratch/connectome/allie/envs/cca_env"
fi

# Check Python packages (requires environment activation)
if command -v conda &> /dev/null && [ -d "/scratch/connectome/allie/envs/cca_env" ]; then
    source /usr/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate /scratch/connectome/allie/envs/cca_env 2>/dev/null || true
    
    if python -c "import numpy" 2>/dev/null; then
        check_pass "numpy installed"
    else
        check_fail "numpy not installed in cca_env"
    fi
    
    if python -c "from sklearn.decomposition import PCA" 2>/dev/null; then
        check_pass "scikit-learn installed"
    else
        check_fail "scikit-learn not installed in cca_env"
    fi
    
    # Check local SCCA_PMD implementation
    SCRIPT_DIR="$PROJECT_ROOT/gene-brain-cca-2/scripts"
    if [ -f "$SCRIPT_DIR/scca_pmd.py" ]; then
        if python -c "import sys; sys.path.insert(0, '$SCRIPT_DIR'); from scca_pmd import SCCA_PMD" 2>/dev/null; then
            check_pass "Local SCCA_PMD implementation available"
        else
            check_fail "SCCA_PMD import failed. Check $SCRIPT_DIR/scca_pmd.py"
        fi
    else
        check_fail "scca_pmd.py not found at $SCRIPT_DIR/scca_pmd.py"
    fi
fi

# Check SLURM
if command -v sbatch &> /dev/null; then
    check_pass "SLURM (sbatch) available"
else
    check_warn "SLURM not available (you can run interactively)"
fi

echo ""

# === 2. Directory Structure ===
echo "--- Directory Structure ---"

PROJECT_ROOT="/storage/bigdata/UKB/fMRI/gene-brain-CCA"
cd "$PROJECT_ROOT" 2>/dev/null || {
    check_fail "Cannot access project root: $PROJECT_ROOT"
    echo ""
    echo "CRITICAL: Project root not accessible. Exiting."
    exit 1
}

if [ -d "gene-brain-cca-2/scripts" ]; then
    check_pass "gene-brain-cca-2/scripts/ exists"
else
    check_fail "gene-brain-cca-2/scripts/ not found"
fi

if [ -d "gene-brain-cca-2/slurm" ]; then
    check_pass "gene-brain-cca-2/slurm/ exists"
else
    check_fail "gene-brain-cca-2/slurm/ not found"
fi

if [ -d "logs" ]; then
    check_pass "logs/ directory exists"
else
    check_warn "logs/ directory missing. Creating it..."
    mkdir -p logs && check_pass "logs/ created" || check_fail "Failed to create logs/"
fi

echo ""

# === 3. Data Files for Pipeline A ===
echo "--- Pipeline A Data Files ---"

FILES_A=(
    "derived_max_pooling/gene_x/ids_gene.npy:Gene IDs"
    "derived_max_pooling/gene_x/X_gene_ng.npy:Gene scalar matrix"
    "/storage/bigdata/UKB/fMRI/fmri_eids_180.npy:fMRI subject IDs"
    "/storage/bigdata/UKB/fMRI/fmri_X_180.npy:fMRI data matrix"
    "/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/iids.npy:Covariate IDs"
    "/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_age.npy:Age covariates"
    "/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_sex.npy:Sex covariates"
    "/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/covariates_valid_mask.npy:Covariate mask"
    "/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/labels.npy:Labels"
)

for entry in "${FILES_A[@]}"; do
    IFS=":" read -r filepath desc <<< "$entry"
    if [ -f "$filepath" ]; then
        SIZE=$(du -h "$filepath" | cut -f1)
        check_pass "$desc ($SIZE)"
    else
        check_fail "$desc not found: $filepath"
    fi
done

echo ""

# === 4. Data Files for Pipeline B ===
echo "--- Pipeline B Data Files ---"

# Gene list file
GENE_LIST="/storage/bigdata/UKB/fMRI/nesap-genomics-allison/iids_labels_covariates/gene_list_filtered.txt"
if [ -f "$GENE_LIST" ]; then
    N_GENES=$(wc -l < "$GENE_LIST")
    check_pass "Gene list file ($N_GENES genes)"
else
    check_fail "Gene list not found: $GENE_LIST"
fi

# Check embedding root directory
EMBED_ROOT="/storage/bigdata/UKB/fMRI/nesap-genomics-allison/DNABERT2_embedding_merged"
if [ -d "$EMBED_ROOT" ]; then
    N_SUBDIRS=$(find "$EMBED_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
    check_pass "Gene embedding root directory ($N_SUBDIRS subdirectories)"
    
    # Check a sample gene (first in list)
    if [ -f "$GENE_LIST" ]; then
        SAMPLE_GENE=$(head -1 "$GENE_LIST" | tr -d '\r')
        if [ -d "$EMBED_ROOT/$SAMPLE_GENE" ]; then
            N_FILES=$(ls "$EMBED_ROOT/$SAMPLE_GENE"/embeddings_*_layer_last.npy 2>/dev/null | wc -l)
            if [ "$N_FILES" -ge 49 ]; then
                check_pass "Sample gene '$SAMPLE_GENE' has $N_FILES embedding files"
            else
                check_warn "Sample gene '$SAMPLE_GENE' has only $N_FILES embedding files (expected 49)"
            fi
        else
            check_warn "Sample gene directory not found: $EMBED_ROOT/$SAMPLE_GENE"
        fi
    fi
else
    check_fail "Gene embedding root not found: $EMBED_ROOT"
fi

echo ""

# === 5. Script Executability ===
echo "--- Script Files ---"

SCRIPTS=(
    "gene-brain-cca-2/scripts/prepare_overlap_no_pca.py"
    "gene-brain-cca-2/scripts/run_scca_interpretable.py"
    "gene-brain-cca-2/scripts/build_x_gene_wide.py"
    "gene-brain-cca-2/scripts/pca_gene_wide.py"
    "gene-brain-cca-2/scripts/run_predictive_suite.py"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ] || head -1 "$script" | grep -q "^#!"; then
            check_pass "$(basename "$script")"
        else
            check_warn "$(basename "$script") exists but may not be executable"
        fi
    else
        check_fail "$(basename "$script") not found"
    fi
done

echo ""

# === 6. SLURM Scripts ===
echo "--- SLURM Scripts ---"

SBATCH_SCRIPTS=(
    "gene-brain-cca-2/slurm/01_interpretable_scca.sbatch"
    "gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch"
)

for script in "${SBATCH_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        # Check for required SBATCH directives
        if grep -q "#SBATCH" "$script"; then
            check_pass "$(basename "$script")"
        else
            check_warn "$(basename "$script") exists but missing #SBATCH directives"
        fi
    else
        check_fail "$(basename "$script") not found"
    fi
done

echo ""

# === 7. Disk Space ===
echo "--- Disk Space ---"

REQUIRED_GB=50  # Rough estimate for all outputs
AVAIL_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')

if [ "$AVAIL_GB" -ge "$REQUIRED_GB" ]; then
    check_pass "Available disk space: ${AVAIL_GB}G (≥${REQUIRED_GB}G required)"
else
    check_warn "Available disk space: ${AVAIL_GB}G (may be tight; ${REQUIRED_GB}G recommended)"
fi

echo ""

# === Summary ===
echo "=========================================="
echo "Summary:"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "You're ready to run the pipelines:"
    echo "  sbatch gene-brain-cca-2/slurm/01_interpretable_scca.sbatch"
    echo "  sbatch gene-brain-cca-2/slurm/02_predictive_wide_suite.sbatch"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}✓ Checks passed with $WARNINGS warning(s).${NC}"
    echo "Review warnings above. You can likely proceed, but double-check."
    exit 0
else
    echo -e "${RED}✗ Found $ERRORS error(s) and $WARNINGS warning(s).${NC}"
    echo ""
    echo "Please fix errors before running pipelines."
    echo "See TROUBLESHOOTING.md for help."
    exit 1
fi
