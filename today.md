# January 21, 2026

## Tian Subcortical QC Analysis

### Objective
Diagnosed ROI-level validity issues in Tian S3 subcortical timeseries before building functional connectivity (FC) matrices.

### Work Completed
- Created `tian_qc_diagnosis.ipynb` notebook to analyze Tian S3 subcortical timeseries data
- Computed per-subject, per-ROI time-series standard deviations
- Flagged invalid ROIs (std < 1e-6)
- Analyzed validity rates across all 50 Tian S3 subcortical ROIs
- Updated SLURM batch scripts for Tian timeseries extraction and FC building
- Implemented comprehensive pipeline for Tian subcortical analysis

### Key Findings
- **ROI Coverage**: 50 subcortical regions (hippocampus, amygdala, thalamus, NAc, caudate, putamen, globus pallidus, hypothalamus, VTA, SN)
- **Quality Metrics**: 
  - Computed ROI valid rates (min/median/max)
  - Computed subject invalid rates
  - Identified lowest valid-rate ROIs
- **Visualizations**:
  - ROI validity heatmaps for subject subsets
  - ROI validity barplots with 90% threshold
  - FC matrix comparisons between "good" and "bad" subjects

### New Scripts Added
- `extract_tian_weights.py` - Extract Tian subcortical timeseries weights
- `prepare_tian_subset.py` - Prepare subset of subjects for Tian analysis
- `build_tian_fc_subset.py` - Build FC matrices for Tian subset
- `build_tian_summary_fc.py` - Build summary FC for Tian data
- `qc_tian_rois.py` - Quality control for Tian ROIs
- `qc_mni_atlas_overlap.py` - QC for MNI atlas overlap
- `align_tian_summary_data.py` - Align Tian summary data
- `generate_tian_qc_supplement.py` - Generate QC supplement
- `generate_main_analysis_report.py` - Generate main analysis report
- `build_schaefer17_summary.py` - Build Schaefer17 summary
- `run_coupling_benchmark.py` - Run coupling benchmark

### SLURM Jobs Updated
- Updated scripts 12-14 for Tian timeseries extraction and FC building
- Added new SLURM scripts 15-27 for comprehensive Tian pipeline

### Technical Details
- Dataset: UKB fMRI subcortical timeseries
- Location: `/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_schaefer_mdd/tian_timeseries`
- Parcellation: Tian S3 (50 subcortical ROIs)
- Threshold: EPS = 1e-6 for ROI validity

### Next Steps
- Review problematic ROIs and determine exclusion criteria
- Build final FC matrices for valid subjects/ROIs
- Proceed with CCA analysis using quality-controlled data
- Complete coupling benchmark analysis
