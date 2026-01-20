#!/usr/bin/env python3
"""
Build Tian subcortical FC for the subset of subjects with available timeseries.

Unlike build_tian_fc.py, this script does NOT fail on missing subjects
because we already filtered to only available subjects in prepare_tian_subset.py.

Inputs:
- Tian ROI time series .npy per subject
- ids_tian_subset.npy (only subjects with available data)

Steps per subject:
1) load ROI-TS (T x R; expect R=50 for Tian S3)
2) z-score each ROI over time
3) correlation -> Fisher z -> upper triangle -> feature vector

Outputs:
- X_fmri_tian_fc.npy (N x 1225) where 1225 = 50*49/2
- meta_tian_fc.json

Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import numpy as np


def find_tian_file(eid: str, root: Path) -> str | None:
    """Find Tian timeseries file for a given EID."""
    pats = [
        root / eid / "tian_s3_*.npy",
        root / eid / f"tian_s3_{eid}_*.npy",
    ]
    for pat in pats:
        files = sorted(glob(str(pat)))
        if files:
            return files[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-root", required=True,
                    help="Directory with Tian timeseries (derived_schaefer_mdd/tian_timeseries)")
    ap.add_argument("--ids-subset", required=True,
                    help="ids_tian_subset.npy (pre-filtered to available subjects)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for FC matrix")
    ap.add_argument("--tag", default="tian_fc",
                    help="Tag for output files")
    args = ap.parse_args()

    print("[1/3] Loading subset IDs...", flush=True)
    ids_subset = np.load(args.ids_subset, allow_pickle=True).astype(str).tolist()
    print(f"  {len(ids_subset)} subjects in subset", flush=True)
    
    roi_root = Path(args.roi_root)
    
    print("[2/3] Building FC for each subject...", flush=True)
    feats_by_id = {}
    missing_files = []
    R_expected = None
    
    for i, eid in enumerate(ids_subset):
        fp = find_tian_file(eid, roi_root)
        if fp is None:
            missing_files.append(eid)
            print(f"  [WARN] Missing file for {eid}", flush=True)
            continue
        
        ts = np.load(fp)
        if ts.ndim != 2:
            missing_files.append(eid)
            print(f"  [WARN] Invalid shape for {eid}: {ts.shape}", flush=True)
            continue
        
        # Ensure T x R format (timepoints x ROIs)
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        
        T, R = ts.shape
        R_expected = R_expected or R
        
        if R != R_expected:
            missing_files.append(eid)
            print(f"  [WARN] ROI count mismatch for {eid}: {R} vs {R_expected}", flush=True)
            continue
        
        # Z-score over time
        mu = ts.mean(0, keepdims=True)
        sd = ts.std(0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        zt = (ts - mu) / sd
        
        # Correlation -> Fisher z -> upper triangle
        # Handle NaN from zero-variance ROIs (signal dropout in subcortical regions)
        with np.errstate(invalid='ignore'):
            corr = np.corrcoef(zt, rowvar=False)
        # Replace NaN with 0 (no correlation when ROI has no signal)
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -0.999999, 0.999999)
        z = np.arctanh(corr)
        iu = np.triu_indices_from(z, k=1)
        feats_by_id[eid] = z[iu].astype(np.float32, copy=False)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(ids_subset)} subjects", flush=True)
    
    print(f"  Completed: {len(feats_by_id)} subjects", flush=True)
    
    if missing_files:
        print(f"  [WARN] {len(missing_files)} subjects had issues", flush=True)
    
    if len(feats_by_id) == 0:
        raise SystemExit("[ERROR] No valid subjects found!")
    
    # Build matrix in order of ids_subset (skip missing)
    valid_ids = [eid for eid in ids_subset if eid in feats_by_id]
    X = np.vstack([feats_by_id[eid] for eid in valid_ids])
    
    print("[3/3] Saving outputs...", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / f"X_fmri_{args.tag}.npy", X)
    np.save(out_dir / f"ids_{args.tag}.npy", np.array(valid_ids, dtype=object))
    
    meta = {
        "tag": args.tag,
        "n_subjects": len(valid_ids),
        "n_rois": R_expected,
        "feature_length": int(X.shape[1]),  # Should be 50*49/2 = 1225
        "roi_root": str(roi_root),
        "n_missing": len(missing_files),
    }
    (out_dir / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\n[done] X_fmri_{args.tag}.npy: {X.shape}", flush=True)
    print(f"  {R_expected} ROIs -> {X.shape[1]} FC features", flush=True)


if __name__ == "__main__":
    main()
