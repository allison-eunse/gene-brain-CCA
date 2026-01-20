#!/usr/bin/env python3
"""
Build Tian anatomical summary FC (7 groups -> 21 edges) after QC.

Uses a fixed ROI set (kept_rois) and QC-filtered subjects.
Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np


TIAN_ANATOMICAL_GROUPS = {
    "Hippocampus": list(range(0, 8)),
    "Amygdala": list(range(8, 12)),
    "Thalamus": list(range(12, 24)),
    "NAc": list(range(24, 28)),
    "Caudate": list(range(28, 34)),
    "Putamen": list(range(34, 40)),
    "Other": list(range(40, 50)),  # GP + HTH + VTA/SN
}


def find_tian_file(eid: str, root: Path) -> Optional[str]:
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
    ap.add_argument("--roi-root", required=True)
    ap.add_argument("--kept-rois", required=True)
    ap.add_argument("--kept-subjects", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--std-threshold", type=float, default=1e-6)
    ap.add_argument("--group-valid-min-fraction", type=float, default=0.5)
    args = ap.parse_args()

    roi_root = Path(args.roi_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_rois = np.load(args.kept_rois).astype(int).tolist()
    kept_subjects = np.load(args.kept_subjects, allow_pickle=True).astype(str).tolist()

    # Filter to only groups that have at least 1 kept ROI
    group_rois_all = {k: [r for r in v if r in kept_rois] for k, v in TIAN_ANATOMICAL_GROUPS.items()}
    group_rois = {k: v for k, v in group_rois_all.items() if len(v) > 0}
    group_names = list(group_rois.keys())
    
    print(f"[info] Using {len(group_names)} groups with kept ROIs: {group_names}", flush=True)
    for g in group_names:
        print(f"  {g}: {len(group_rois[g])} ROIs", flush=True)

    feats = []
    valid_ids = []
    dropped = []

    for i, eid in enumerate(kept_subjects):
        fp = find_tian_file(eid, roi_root)
        if fp is None:
            dropped.append(eid)
            continue
        ts = np.load(fp)
        if ts.ndim != 2:
            dropped.append(eid)
            continue
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        if ts.shape[1] != 50:
            dropped.append(eid)
            continue

        # Per-subject validity (std threshold)
        sd = ts.std(axis=0)
        valid_roi_mask = sd > float(args.std_threshold)

        group_ts = []
        group_ok = True
        for g in group_names:
            rois = group_rois[g]
            if not rois:
                group_ok = False
                break
            rois_valid = [r for r in rois if valid_roi_mask[r]]
            if len(rois_valid) / max(len(rois), 1) < float(args.group_valid_min_fraction):
                group_ok = False
                break
            group_ts.append(ts[:, rois_valid].mean(axis=1))

        if not group_ok:
            dropped.append(eid)
            continue

        G = np.stack(group_ts, axis=1)
        # Z-score per group
        mu = G.mean(axis=0, keepdims=True)
        sd = G.std(axis=0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        Gz = (G - mu) / sd

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(Gz, rowvar=False)
        if np.isnan(corr).any():
            dropped.append(eid)
            continue

        corr = np.clip(corr, -0.999999, 0.999999)
        z = np.arctanh(corr)
        iu = np.triu_indices_from(z, k=1)
        feats.append(z[iu].astype(np.float32, copy=False))
        valid_ids.append(eid)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(kept_subjects)}", flush=True)

    if not feats:
        raise SystemExit("[ERROR] No valid subjects produced summary FC.")

    X = np.vstack(feats)
    valid_ids = np.array(valid_ids, dtype=object)

    np.save(out_dir / "X_fmri_tian_summary.npy", X)
    np.save(out_dir / "ids_tian_summary.npy", valid_ids)

    meta = {
        "feature_type": "anatomical_summary_fc",
        "n_groups": len(group_names),
        "n_fc_features": int(X.shape[1]),
        "groups": group_names,
        "group_rois": {k: group_rois[k] for k in group_names},
        "n_subjects_input": int(len(kept_subjects)),
        "n_subjects_kept": int(len(valid_ids)),
        "n_subjects_dropped": int(len(dropped)),
        "std_threshold": float(args.std_threshold),
        "group_valid_min_fraction": float(args.group_valid_min_fraction),
    }
    (out_dir / "meta_tian_summary.json").write_text(json.dumps(meta, indent=2))

    print("[done] Summary FC saved:", out_dir / "X_fmri_tian_summary.npy", flush=True)
    print(f"  Subjects kept: {len(valid_ids)}", flush=True)
    n_edges = len(group_names) * (len(group_names) - 1) // 2
    print(f"  Features: {X.shape[1]} ({len(group_names)} groups -> {n_edges} edges)", flush=True)


if __name__ == "__main__":
    main()
