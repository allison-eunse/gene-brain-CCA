## Tian Subcortical Sensitivity Analysis

### Status
- Final analyzed cohort: **N=128** (from 746 initial)
- Feature set: **5 anatomical groups → 10 FC edges**
- Method: **SCCA (PCA=102, c1=c2=0.1)**
- Holdout coupling: **r_holdout = -0.096** (no significant coupling)

### Data Quality Issue (Upstream)
**68.9% (514/746) of subjects** have zero signal in the Tian subcortical region after native→MNI registration.  
Native-space fMRI looks normal; failures occur in:
`/storage/bigdata/UKB/fMRI/UKB_20227_1_recon/<subj>/<subj>_MNI152_2mm.nii.gz`

This causes Tian ROI time series to be entirely zero for those subjects.

### Diagnostics
Generated reports:
- `derived_schaefer_mdd/tian_summary/tian_extraction_health_report.json`
- `derived_schaefer_mdd/tian_summary/tian_mean_fmri_mni_overlap.csv`
- `derived_schaefer_mdd/tian_summary/tian_mask_timepoint_checks.json`
- `derived_schaefer_mdd/tian_summary/qc_supplement_report.md`
- `derived_schaefer_mdd/tian_summary/qc_supplement_report.html`

### Current Pipeline (Compliant)
All stages use Slurm + `srun`, exclude node1/node3, and do **not** set `CUDA_VISIBLE_DEVICES` manually:
- `slurm/23_tian_qc_summary_align.sbatch`
- `slurm/22_coupling_tian_summary.sbatch`
- `slurm/24_extract_tian_summary_weights.sbatch`

### Optional Future Guardrail
Pre-flight QC to skip subjects with zero Tian atlas overlap:
`scripts/qc_mni_atlas_overlap.py`

Example usage (via Slurm):
```bash
srun python scripts/qc_mni_atlas_overlap.py \
  --ids-eid derived_schaefer_mdd/tian_subset/ids_tian_subset.npy \
  --roi-root derived_schaefer_mdd/tian_timeseries \
  --out-dir derived_schaefer_mdd/tian_summary
```
