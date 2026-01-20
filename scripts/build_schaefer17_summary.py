#!/usr/bin/env python3
"""
Build 17-network summary FC (136 features) from Schaefer 400 (17-network) ROI time series.

- Keep all 17 Schaefer network labels (VisCent, VisPeri, SomMotA, SomMotB, etc.)
- For each subject:
  * z-score ROI time series over time
  * average within each of the 17 network groups -> T x 17
  * compute correlation, Fisher z, take upper triangle (136 features)
- Output aligned to ids_keep order.

Lab rules: run via Slurm; no login-node heavy compute; no manual CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path
import numpy as np

# All 17 Schaefer network labels in standard order
ORDER = [
    "VisCent", "VisPeri",
    "SomMotA", "SomMotB",
    "DorsAttnA", "DorsAttnB",
    "SalVentAttnA", "SalVentAttnB",
    "LimbicA", "LimbicB",
    "ContA", "ContB", "ContC",
    "DefaultA", "DefaultB", "DefaultC",
    "TempPar",
]


def load_roi_names(csv_path: Path) -> list[str]:
    with csv_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        if "ROI Name" not in header:
            raise ValueError(f"'ROI Name' column missing in {csv_path}")
        idx = header.index("ROI Name")
        names = [row[idx] for row in reader if row]
    if len(names) != 400:
        raise ValueError(f"Expected 400 ROI names, got {len(names)}")
    return names


def roi_to_network(roi: str) -> str | None:
    """Extract the 17-network label from a Schaefer ROI name.
    
    ROI names follow pattern: "17Networks_LH_VisCent_ExStr_1"
    The network label is the 3rd component (index 2).
    """
    parts = roi.split("_")
    if len(parts) < 3:
        return None
    network = parts[2]
    if network in ORDER:
        return network
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri_ts_dir", required=True, help="Root dir containing per-subject Schaefer ROI time series")
    ap.add_argument("--glob", default="*/schaefer_400Parcels_17Networks_*.npy")
    ap.add_argument("--labels_csv", required=True, help="Schaefer 400 17-network centroid CSV")
    ap.add_argument("--ids_keep", required=True, help="ids_common.npy to align and order output")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="schaefer17_summary")
    ap.add_argument("--limit", type=int, default=0, help="Limit subjects for debugging")
    args = ap.parse_args()

    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str).tolist()
    keep_set = set(ids_keep)

    roi_names = load_roi_names(Path(args.labels_csv))
    networks = [roi_to_network(r) for r in roi_names]
    
    # Check for unmapped ROIs
    if any(g is None for g in networks):
        bad = [roi_names[i] for i, g in enumerate(networks) if g is None][:10]
        raise ValueError(f"Unmapped ROI(s): {bad}")
    networks = np.array(networks)

    files = sorted(glob(str(Path(args.fmri_ts_dir) / args.glob)))
    feats_by_id = {}
    seen = set()

    for fp in files:
        sid = Path(fp).stem.split("_")[-1]
        if sid in seen or sid not in keep_set:
            continue
        ts = np.load(fp)
        if ts.ndim != 2:
            continue
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T  # ensure T x R (timepoints x ROIs)
        
        # z-score over time
        mu = ts.mean(0, keepdims=True)
        sd = ts.std(0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        zt = (ts - mu) / sd
        
        # Network-average time series (T x 17)
        net_ts = []
        for net in ORDER:
            mask = networks == net
            if mask.sum() == 0:
                raise ValueError(f"No ROIs found for network {net}")
            net_ts.append(zt[:, mask].mean(axis=1))
        net_ts = np.stack(net_ts, axis=1)  # T x 17
        
        # Compute FC matrix (17 x 17)
        corr = np.corrcoef(net_ts, rowvar=False)
        corr = np.clip(corr, -0.999999, 0.999999)
        
        # Fisher z-transform
        z = np.arctanh(corr)
        
        # Upper triangle, excluding diagonal -> 17*16/2 = 136 features
        iu = np.triu_indices_from(z, k=1)
        feats_by_id[sid] = z[iu].astype(np.float32, copy=False)
        seen.add(sid)
        
        if args.limit and len(seen) >= args.limit:
            break
        if len(seen) == len(keep_set):
            break

    missing = [sid for sid in ids_keep if sid not in feats_by_id]
    if missing:
        raise SystemExit(f"[ERROR] Missing {len(missing)} subjects. Example: {missing[:10]}")

    X = np.vstack([feats_by_id[sid] for sid in ids_keep])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / f"X_fmri_{args.tag}.npy", X)
    np.save(out_dir / f"ids_{args.tag}.npy", np.array(ids_keep, dtype=object))
    
    # Feature count: 17 * 16 / 2 = 136
    feature_count = int(len(ORDER) * (len(ORDER) - 1) / 2)
    
    meta = {
        "tag": args.tag,
        "n_subjects": len(ids_keep),
        "feature_length": feature_count,
        "n_networks": len(ORDER),
        "order": ORDER,
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "glob": args.glob,
    }
    (out_dir / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"[done] {X.shape} (n_subjects={len(ids_keep)}, n_features={feature_count})")


if __name__ == "__main__":
    main()
