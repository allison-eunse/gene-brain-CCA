#!/usr/bin/env python3
"""
Build X_fmri for CCA from per-subject ROI time series (.npy).

Expected input patterns (examples):
  ROOT/<sid>_20227_2_0/hcp_mmp1_<sid>_20227_2_0.npy
  ROOT/<sid>/schaefer_400Parcels_17Networks_<sid>.npy

This script:
  - loads each subject's time series (T×R or R×T)
  - z-scores each ROI across time
  - computes ROI×ROI correlation
  - Fisher-z transform
  - flattens upper triangle (excluding diagonal) -> vector length R*(R-1)/2
  - stacks into X_fmri_fc (N×D)

Outputs:
  - ids_fmri.npy (string array)
  - X_fmri_fc.npy (float32)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _ensure_tr(ts: np.ndarray) -> np.ndarray:
    # want shape (T,R) with T >= R typically
    if ts.ndim != 2:
        raise ValueError(f"Expected 2D array, got {ts.ndim}D")
    T, R = ts.shape
    if T < R:
        ts = ts.T
    return ts


def _fc_upper(ts_tr: np.ndarray) -> np.ndarray:
    # z-score each ROI over time
    mu = ts_tr.mean(axis=0, keepdims=True)
    sd = ts_tr.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    zt = (ts_tr - mu) / sd
    corr = np.corrcoef(zt, rowvar=False)
    corr = np.clip(corr, -0.999999, 0.999999)
    z = np.arctanh(corr)
    iu = np.triu_indices_from(z, k=1)
    return z[iu].astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root directory containing per-subject ROI-ts files")
    ap.add_argument("--glob", required=True, help="Glob pattern relative to root (e.g. '*_20227_2_0/hcp_mmp1_*.npy')")
    ap.add_argument("--id-regex", default=r"^(\d+)", help="Regex to extract subject ID from parent folder name or filename")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=0, help="For debugging: max subjects to process (0=all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    id_re = re.compile(args.id_regex)
    files = sorted(root.glob(args.glob))
    if not files:
        raise SystemExit(f"[ERROR] No files matched: root={root} glob={args.glob}")

    ids: List[str] = []
    feats: List[np.ndarray] = []
    D = None

    for i, fp in enumerate(files):
        # derive subject id
        # prefer parent folder (e.g., 1000246_20227_2_0), else filename
        m = id_re.search(fp.parent.name) or id_re.search(fp.name)
        if not m:
            continue
        sid = m.group(1)

        ts = np.load(fp)
        ts = _ensure_tr(ts)
        feat = _fc_upper(ts)
        if D is None:
            D = feat.shape[0]
        elif feat.shape[0] != D:
            raise ValueError(f"Feature length mismatch at {fp}: {feat.shape[0]} != {D}")

        ids.append(sid)
        feats.append(feat)
        if (i + 1) % 200 == 0:
            print(f"[fmri] processed {i+1}/{len(files)}", flush=True)
        if args.limit and len(ids) >= args.limit:
            break

    X = np.vstack(feats) if feats else np.zeros((0, 0), dtype=np.float32)
    np.save(out_dir / "ids_fmri.npy", np.array(ids, dtype=object))
    np.save(out_dir / "X_fmri_fc.npy", X)
    print(f"[save] {out_dir/'ids_fmri.npy'}: {len(ids)}", flush=True)
    print(f"[save] {out_dir/'X_fmri_fc.npy'}: {X.shape}", flush=True)


if __name__ == "__main__":
    main()


