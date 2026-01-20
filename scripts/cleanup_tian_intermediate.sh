#!/bin/bash
# Clean up intermediate files from Tian QC pipeline.
# Follow lab storage guidelines: back up before delete.

set -euo pipefail

ROOT="/storage/bigdata/UKB/fMRI/gene-brain-CCA"
BACKUP_DIR="${ROOT}/derived_schaefer_mdd/tian_subset"

echo "Cleaning up intermediate Tian files..."

tar -czf "${BACKUP_DIR}/tian_fc_backup_$(date +%Y%m%d).tar.gz" \
  "${BACKUP_DIR}/X_fmri_tian_fc.npy" \
  "${BACKUP_DIR}/coupling_benchmark" || true

rm -f "${BACKUP_DIR}/X_fmri_tian_fc.npy"

echo "=== Remaining Tian files ==="
du -sh "${ROOT}/derived_schaefer_mdd"/tian_* || true

echo "Cleanup complete. Backup saved to tian_fc_backup_*.tar.gz"
