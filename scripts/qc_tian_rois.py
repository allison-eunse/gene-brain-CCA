#!/usr/bin/env python3
"""
QC Tian S3 ROI timeseries for validity and coverage.

Produces:
- kept_roi_indices.npy
- kept_subject_ids.npy
- qc_report.json

Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
import json
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np


TIAN_S3_LABELS = [
    "HIP-head-l", "HIP-head-r", "HIP-body-l", "HIP-body-r",
    "HIP-tail-l", "HIP-tail-r", "HIP-subiculum-l", "HIP-subiculum-r",
    "AMY-lateral-l", "AMY-lateral-r", "AMY-medial-l", "AMY-medial-r",
    "THA-VA-l", "THA-VA-r", "THA-VL-l", "THA-VL-r",
    "THA-VP-l", "THA-VP-r", "THA-IL-l", "THA-IL-r",
    "THA-MD-l", "THA-MD-r", "THA-PU-l", "THA-PU-r",
    "NAc-core-l", "NAc-core-r", "NAc-shell-l", "NAc-shell-r",
    "CAU-head-l", "CAU-head-r", "CAU-body-l", "CAU-body-r",
    "CAU-tail-l", "CAU-tail-r",
    "PUT-anterior-l", "PUT-anterior-r", "PUT-posterior-l", "PUT-posterior-r",
    "PUT-ventral-l", "PUT-ventral-r",
    "GP-internal-l", "GP-internal-r", "GP-external-l", "GP-external-r",
    "HTH-l", "HTH-r",
    "VTA-l", "VTA-r", "SN-l", "SN-r",
]


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
    ap.add_argument("--ids-subset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--std-threshold", type=float, default=1e-6)
    ap.add_argument("--roi-validity-threshold", type=float, default=0.90)
    ap.add_argument("--subject-dropout-threshold", type=float, default=0.20)
    args = ap.parse_args()

    roi_root = Path(args.roi_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = np.load(args.ids_subset, allow_pickle=True).astype(str)
    stds = []
    valid_ids = []
    missing = []

    for i, eid in enumerate(ids):
        fp = find_tian_file(eid, roi_root)
        if fp is None:
            missing.append(eid)
            continue
        ts = np.load(fp)
        if ts.ndim != 2:
            missing.append(eid)
            continue
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        _, r = ts.shape
        if r != 50:
            missing.append(eid)
            continue
        sd = ts.std(axis=0)
        stds.append(sd.astype(np.float32, copy=False))
        valid_ids.append(eid)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(ids)}", flush=True)

    if not stds:
        raise SystemExit("[ERROR] No valid subjects found!")

    stds = np.vstack(stds)
    valid_ids = np.array(valid_ids, dtype=object)
    valid_mask = stds > float(args.std_threshold)

    roi_valid_rate = valid_mask.mean(axis=0)

    # Always save diagnostic report first (before any threshold failures)
    diag_report = {
        "n_subjects_input": int(len(ids)),
        "n_subjects_valid_ts": int(len(valid_ids)),
        "n_missing_or_invalid": int(len(missing)),
        "n_rois_input": 50,
        "std_threshold": float(args.std_threshold),
        "roi_validity_threshold_requested": float(args.roi_validity_threshold),
        "subject_dropout_threshold_requested": float(args.subject_dropout_threshold),
        "roi_valid_rate": {TIAN_S3_LABELS[i]: float(roi_valid_rate[i]) for i in range(50)},
        "roi_valid_rate_min": float(roi_valid_rate.min()),
        "roi_valid_rate_median": float(np.median(roi_valid_rate)),
        "roi_valid_rate_max": float(roi_valid_rate.max()),
        "roi_valid_rate_sorted": sorted(
            [(TIAN_S3_LABELS[i], float(roi_valid_rate[i])) for i in range(50)],
            key=lambda x: x[1], reverse=True
        ),
    }
    (out_dir / "qc_diagnostic.json").write_text(json.dumps(diag_report, indent=2))
    print(f"[info] Diagnostic saved to {out_dir / 'qc_diagnostic.json'}", flush=True)
    print(f"[info] ROI validity: min={roi_valid_rate.min():.3f}, median={np.median(roi_valid_rate):.3f}, max={roi_valid_rate.max():.3f}", flush=True)

    kept_rois = np.where(roi_valid_rate >= float(args.roi_validity_threshold))[0]
    dropped_rois = np.where(roi_valid_rate < float(args.roi_validity_threshold))[0]

    if kept_rois.size == 0:
        print(f"[WARN] No ROIs passed {args.roi_validity_threshold:.0%} threshold. Top 10 ROIs by validity:", flush=True)
        for name, rate in diag_report["roi_valid_rate_sorted"][:10]:
            print(f"  {name}: {rate:.3f}", flush=True)
        raise SystemExit("[ERROR] No ROIs passed validity threshold.")

    subj_invalid_rate = 1.0 - valid_mask[:, kept_rois].mean(axis=1)
    kept_subject_mask = subj_invalid_rate <= float(args.subject_dropout_threshold)
    kept_subject_ids = valid_ids[kept_subject_mask]

    np.save(out_dir / "kept_roi_indices.npy", kept_rois)
    np.save(out_dir / "kept_subject_ids.npy", kept_subject_ids)

    report = {
        "n_subjects_input": int(len(ids)),
        "n_subjects_valid_ts": int(len(valid_ids)),
        "n_subjects_kept": int(len(kept_subject_ids)),
        "n_missing_or_invalid": int(len(missing)),
        "n_rois_input": 50,
        "n_rois_kept": int(len(kept_rois)),
        "std_threshold": float(args.std_threshold),
        "roi_validity_threshold": float(args.roi_validity_threshold),
        "subject_dropout_threshold": float(args.subject_dropout_threshold),
        "dropped_roi_indices": dropped_rois.tolist(),
        "dropped_roi_names": [TIAN_S3_LABELS[i] for i in dropped_rois.tolist()],
        "roi_valid_rate_min": float(roi_valid_rate.min()),
        "roi_valid_rate_median": float(np.median(roi_valid_rate)),
        "roi_valid_rate_max": float(roi_valid_rate.max()),
        "subject_invalid_rate_min": float(subj_invalid_rate.min()),
        "subject_invalid_rate_median": float(np.median(subj_invalid_rate)),
        "subject_invalid_rate_max": float(subj_invalid_rate.max()),
    }
    (out_dir / "qc_report.json").write_text(json.dumps(report, indent=2))

    print("[done] QC report saved:", out_dir / "qc_report.json", flush=True)
    print(f"  Subjects kept: {len(kept_subject_ids)}", flush=True)
    print(f"  ROIs kept: {len(kept_rois)}", flush=True)


if __name__ == "__main__":
    main()
