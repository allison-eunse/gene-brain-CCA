#!/usr/bin/env python3
"""
Build Tian subcortical FC (Fisher-z, upper triangle) from per-subject Tian ROI time series.

Inputs:
- Tian ROI time series .npy per subject, located under /storage/bigdata/UKB/fMRI/UKB_ROI/<eid>/tian_s3_*.npy
- ids_keep (ids_common) to align ordering

Steps per subject:
1) load ROI-TS (T x R; expect R=50 for Tian S3)
2) z-score each ROI over time
3) correlation -> Fisher z -> upper triangle -> feature vector

Outputs:
- X_fmri_tian_fc.npy (N x D)
- ids_tian_fc.npy
- meta_tian_fc.json

Lab rules: run via Slurm; no login-node heavy compute; no manual CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
import numpy as np


def find_tian_file(eid: str, root: Path) -> str | None:
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
    ap.add_argument("--roi-root", default="/storage/bigdata/UKB/fMRI/UKB_ROI")
    ap.add_argument("--ids-keep", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="tian_fc")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str).tolist()
    keep_set = set(ids_keep)
    roi_root = Path(args.roi_root)

    feats_by_id = {}
    missing_files = []
    R_expected = None
    for eid in ids_keep:
        fp = find_tian_file(eid, roi_root)
        if fp is None:
            missing_files.append(eid)
            continue
        ts = np.load(fp)
        if ts.ndim != 2:
            missing_files.append(eid)
            continue
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T  # ensure T x R
        T, R = ts.shape
        R_expected = R_expected or R
        # z-score over time
        mu = ts.mean(0, keepdims=True)
        sd = ts.std(0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        zt = (ts - mu) / sd
        corr = np.corrcoef(zt, rowvar=False)
        corr = np.clip(corr, -0.999999, 0.999999)
        z = np.arctanh(corr)
        iu = np.triu_indices_from(z, k=1)
        feats_by_id[eid] = z[iu].astype(np.float32, copy=False)
        if args.limit and len(feats_by_id) >= args.limit:
            break

    missing = [eid for eid in ids_keep if eid not in feats_by_id]
    if missing:
        raise SystemExit(f"[ERROR] Missing {len(missing)} subjects. Example: {missing[:10]}")

    X = np.vstack([feats_by_id[eid] for eid in ids_keep])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_fmri_{args.tag}.npy", X)
    np.save(out_dir / f"ids_{args.tag}.npy", np.array(ids_keep, dtype=object))
    meta = {
        "tag": args.tag,
        "n_subjects": len(ids_keep),
        "feature_length": int(X.shape[1]),
        "roi_root": str(roi_root),
        "R": R_expected,
    }
    (out_dir / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print("[done]", X.shape)


if __name__ == "__main__":
    main()
