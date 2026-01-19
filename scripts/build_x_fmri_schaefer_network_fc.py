#!/usr/bin/env python3
"""Build network-specific fMRI functional connectivity (FC) features from Schaefer 400 (17-network) ROI time series.

Why:
- Full Schaefer-400 FC has 79,800 edges (400*399/2), which is high-dimensional.
- For MDD, we often care about networks like Default Mode (DefaultA/B/C), Salience/Ventral Attention (SalVentAttnA/B), and Limbic (LimbicA/B).

This script:
- loads each subject's Schaefer 400 ROI time series (.npy)
- selects ROIs belonging to specified Schaefer-17 networks
- z-scores each ROI across time
- computes ROI×ROI correlation, Fisher-z transform
- flattens the upper triangle (excluding diagonal) -> FC feature vector

**Lab server rules:**
- This is CPU work; do NOT run large jobs on the login node.
- Run via Slurm (sbatch/srun) after a small test run.

Inputs:
- Schaefer ROI time series files like:
  /storage/bigdata/UKB/fMRI/UKB_ROI/<sid>/schaefer_400Parcels_17Networks_<sid>.npy
- Schaefer 400 17-network labels CSV (ROI Name column):
  /storage/bigdata/UKB/fMRI/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv

Outputs:
- X_fmri_schaefer_<tag>.npy (float32, N×D)
- ids_common_order.npy (string/object, N,)  # saved in the same order as --ids-keep
- meta.json

Example (process ONLY overlap cohort, preserving ids_common.npy order):
python scripts/build_x_fmri_schaefer_network_fc.py \
  --root /storage/bigdata/UKB/fMRI/UKB_ROI \
  --glob "*/schaefer_400Parcels_17Networks_*.npy" \
  --labels /storage/bigdata/UKB/fMRI/atlases/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv \
  --networks DefaultA,DefaultB,DefaultC,SalVentAttnA,SalVentAttnB,LimbicA,LimbicB \
  --ids-keep /storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/interpretable/ids_common.npy \
  --out-dir /storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_schaefer_mdd
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


def load_schaefer_roi_names(labels_csv: Path) -> List[str]:
    """Return Schaefer 400 ROI Name list (len=400) from the official centroid CSV."""
    with labels_csv.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or "ROI Name" not in header:
            raise ValueError(
                f"Unexpected CSV header in {labels_csv}. Expected a 'ROI Name' column. Got: {header}"
            )
        roi_name_idx = header.index("ROI Name")
        roi_names: List[str] = []
        for row in reader:
            if not row:
                continue
            roi_names.append(str(row[roi_name_idx]))

    if len(roi_names) != 400:
        raise ValueError(f"Expected 400 ROI names, got {len(roi_names)} from {labels_csv}")
    return roi_names


def parse_schaefer17_network(roi_name: str) -> str:
    # Example: "17Networks_LH_DefaultA_PFCm_1" -> "DefaultA"
    parts = roi_name.split("_")
    if len(parts) < 3:
        return ""
    return parts[2]


def build_mask(roi_names: Sequence[str], networks: Sequence[str]) -> np.ndarray:
    netset = set(networks)
    mask = np.array([parse_schaefer17_network(n) in netset for n in roi_names], dtype=bool)
    if mask.sum() == 0:
        raise ValueError(f"No ROIs found for networks={list(networks)}")
    return mask


def ensure_tr(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D array, got {ts.ndim}D")
    T, R = ts.shape
    if T < R:
        ts = ts.T
    return ts


def fc_upper_fisher_z(ts_tr: np.ndarray) -> np.ndarray:
    """Compute Fisher-z FC upper triangle from (T×R) time series."""
    # z-score over time
    mu = ts_tr.mean(axis=0, keepdims=True)
    sd = ts_tr.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    zt = (ts_tr - mu) / sd

    # correlation via dot product (faster than np.corrcoef for many subjects)
    T = zt.shape[0]
    corr = (zt.T @ zt) / float(max(T - 1, 1))
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    iu = np.triu_indices_from(z, k=1)
    return z[iu].astype(np.float32, copy=False)


def load_ids_keep(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None

    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.tolist()]

    # text file fallback
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-subject Schaefer ROI-ts files")
    ap.add_argument(
        "--glob",
        required=True,
        help="Glob pattern relative to root (e.g. '*/schaefer_400Parcels_17Networks_*.npy')",
    )
    ap.add_argument("--labels", required=True, help="Schaefer 400 17-network centroid CSV (ROI Name column)")
    ap.add_argument(
        "--networks",
        default="DefaultA,DefaultB,DefaultC,SalVentAttnA,SalVentAttnB,LimbicA,LimbicB",
        help="Comma-separated Schaefer-17 network names to include",
    )
    ap.add_argument(
        "--ids-keep",
        default=None,
        help="Optional .npy or .txt list of subject IDs to process; output will be ordered to match this list",
    )
    ap.add_argument("--id-regex", default=r"(\d+)", help="Regex to extract subject ID from filename or parent folder")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--tag", default="mdd_networks", help="Tag for output filename")
    ap.add_argument("--limit", type=int, default=0, help="For testing: process at most N subjects (0=all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_csv = Path(args.labels).expanduser().resolve()
    roi_names = load_schaefer_roi_names(labels_csv)

    networks = [x.strip() for x in args.networks.split(",") if x.strip()]
    mask = build_mask(roi_names, networks)
    n_sel = int(mask.sum())
    feat_len = int(n_sel * (n_sel - 1) // 2)

    ids_keep = load_ids_keep(Path(args.ids_keep).expanduser().resolve()) if args.ids_keep else None
    keep_set: Optional[Set[str]] = set(ids_keep) if ids_keep else None

    print(f"[info] networks={networks}")
    print(f"[info] selected_rois={n_sel}/400 → feature_length={feat_len}")
    if ids_keep:
        print(f"[info] ids_keep provided: {len(ids_keep)} subjects; output will match this order")

    id_re = re.compile(args.id_regex)
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"[ERROR] No files matched: root={root} glob={args.glob}")

    feats_by_id: Dict[str, np.ndarray] = {}
    processed = 0

    for fp in files:
        # Prefer parent folder (UKB_ROI/<sid>/...) to avoid matching the \"400\" in
        # filenames like \"schaefer_400Parcels_17Networks_<sid>.npy\".
        m = id_re.search(fp.parent.name) or id_re.search(fp.name)
        if not m:
            continue
        sid = m.group(1)

        if keep_set is not None and sid not in keep_set:
            continue
        if sid in feats_by_id:
            continue

        ts = np.load(fp)
        ts = ensure_tr(ts)
        ts_sel = ts[:, mask]
        fc_vec = fc_upper_fisher_z(ts_sel)
        if fc_vec.shape[0] != feat_len:
            raise ValueError(f"Feature length mismatch at {fp}: {fc_vec.shape[0]} != {feat_len}")

        feats_by_id[sid] = fc_vec
        processed += 1

        if processed % 200 == 0:
            print(f"[info] processed {processed} subjects", flush=True)

        if args.limit and processed >= args.limit:
            break

        if ids_keep is not None and len(feats_by_id) == len(ids_keep):
            # found them all
            break

    if ids_keep is not None:
        missing = [sid for sid in ids_keep if sid not in feats_by_id]
        if missing:
            raise SystemExit(f"[ERROR] Missing {len(missing)} ids from Schaefer files. Example: {missing[:10]}")
        ids_out = ids_keep
        X = np.vstack([feats_by_id[sid] for sid in ids_out]).astype(np.float32, copy=False)
    else:
        # output in arbitrary processed order
        ids_out = sorted(feats_by_id.keys())
        X = np.vstack([feats_by_id[sid] for sid in ids_out]).astype(np.float32, copy=False)

    out_x = out_dir / f"X_fmri_schaefer_{args.tag}.npy"
    out_ids = out_dir / f"ids_{args.tag}.npy"
    out_meta = out_dir / f"meta_{args.tag}.json"

    np.save(out_x, X)
    np.save(out_ids, np.array(ids_out, dtype=object))

    meta = {
        "tag": args.tag,
        "networks": networks,
        "n_selected_rois": n_sel,
        "feature_length": feat_len,
        "n_subjects": int(X.shape[0]),
        "labels_csv": str(labels_csv),
        "glob": args.glob,
        "root": str(root),
        "ids_keep": str(Path(args.ids_keep).expanduser().resolve()) if args.ids_keep else None,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print(f"[done] saved {out_x} shape={X.shape}")
    print(f"[done] saved {out_ids} n={len(ids_out)}")
    print(f"[done] saved {out_meta}")


if __name__ == "__main__":
    main()
