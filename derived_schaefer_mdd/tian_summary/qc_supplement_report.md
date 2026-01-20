# Tian QC Supplement Report

## Executive Summary
- Total subjects in Tian subset: **746**
- Subjects with all-zero Tian TS: **514** (68.9%)
- Subjects with ≥20 valid ROIs: **164**
- Subjects kept after QC: **144**
- Final subjects used in summary FC: **128**
- ROIs kept after QC: **18**

## Subject Flow
- Initial Tian subset: 746
- ≥20 valid ROIs: 164
- QC kept subjects: 144
- Summary FC subjects: 128

## MNI Registration Failure Evidence
- Confusion matrix (TS all-zero vs MNI atlas overlap):
  - TS all-zero & atlas overlap = 0: 514
  - TS nonzero & atlas overlap > 0: 232

## Coupling Benchmark Summary
- Best config: scca_pca102_c0.1_0.1
- Holdout CC1: -0.09608424057282026

## Notes
The MNI registration step produced fMRI volumes with zero signal in the subcortical (Tian) mask for ~69% of subjects. Native-space mean fMRI volumes appear normal, indicating a failure in the native→MNI resampling for those subjects.

## Figures
- ROI Validity Heatmap: tian_roi_validity_heatmap.png (subjects x ROIs)
- ROI Validity Rates: tian_roi_validity_rates.png
- Atlas Overlap Histogram: tian_atlas_overlap_hist.png
