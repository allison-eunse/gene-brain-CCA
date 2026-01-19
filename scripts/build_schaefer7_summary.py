#!/usr/bin/env python3
"""
Build 7-network summary FC (21 features) from Schaefer 400 (17-network) ROI time series.

- Collapse Schaefer 17-net labels to 7-net groups:
  Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default (TempPar -> Default).
- For each subject:
  * z-score ROI time series over time
  * average within each 7-net group -> T x 7
  * compute correlation, Fisher z, take upper triangle (21 features)
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

MAP7 = {
    "VisCent": "Vis",
    "VisPeri": "Vis",
    "SomMotA": "SomMot",
    "SomMotB": "SomMot",
    "DorsAttnA": "DorsAttn",
    "DorsAttnB": "DorsAttn",
    "SalVentAttnA": "SalVentAttn",
    "SalVentAttnB": "SalVentAttn",
    "LimbicA": "Limbic",
    "LimbicB": "Limbic",
    "ContA": "Cont",
    "ContB": "Cont",
    "ContC": "Cont",
    "DefaultA": "Default",
    "DefaultB": "Default",
    "DefaultC": "Default",
    "TempPar": "Default",  # common folding convention
}

ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


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


def roi_to_group(roi: str) -> str | None:
    parts = roi.split("_")
    if len(parts) < 3:
        return None
    return MAP7.get(parts[2], None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri_ts_dir", required=True, help="Root dir containing per-subject Schaefer ROI time series")
    ap.add_argument("--glob", default="*/schaefer_400Parcels_17Networks_*.npy")
    ap.add_argument("--labels_csv", required=True, help="Schaefer 400 17-network centroid CSV")
    ap.add_argument("--ids_keep", required=True, help="ids_common.npy to align and order output")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="schaefer7_summary")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str).tolist()
    keep_set = set(ids_keep)

    roi_names = load_roi_names(Path(args.labels_csv))
    groups = [roi_to_group(r) for r in roi_names]
    if any(g is None for g in groups):
        bad = [roi_names[i] for i, g in enumerate(groups) if g is None][:10]
        raise ValueError(f"Unmapped ROI(s): {bad}")
    groups = np.array(groups)

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
            ts = ts.T  # ensure T x R
        # z-score over time
        mu = ts.mean(0, keepdims=True)
        sd = ts.std(0, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        zt = (ts - mu) / sd
        # group-average time series (T x 7)
        g_ts = []
        for g in ORDER:
            mask = groups == g
            g_ts.append(zt[:, mask].mean(axis=1))
        g_ts = np.stack(g_ts, axis=1)
        corr = np.corrcoef(g_ts, rowvar=False)
        corr = np.clip(corr, -0.999999, 0.999999)
        z = np.arctanh(corr)
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
    meta = {
        "tag": args.tag,
        "n_subjects": len(ids_keep),
        "feature_length": int(len(ORDER) * (len(ORDER) - 1) / 2),
        "order": ORDER,
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "glob": args.glob,
    }
    (out_dir / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print("[done]", X.shape)


if __name__ == "__main__":
    main()
