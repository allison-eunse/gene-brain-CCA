#!/usr/bin/env python3
"""
Prepare Tian subset data for sensitivity analysis.

Identifies subjects with available Tian timeseries and subsets all data to match.

Inputs:
- Tian timeseries directory (to identify available subjects)
- ids_common.npy, X_gene_wide.npy, labels_common.npy, cov_age.npy, cov_sex.npy

Outputs (to tian_subset/):
- ids_tian_subset.npy
- X_gene_wide_tian_subset.npy
- labels_tian_subset.npy
- cov_age_tian_subset.npy
- cov_sex_tian_subset.npy

Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from glob import glob

import numpy as np


def find_available_subjects(tian_ts_dir: Path) -> set[str]:
    """Find all subject EIDs that have Tian timeseries files."""
    available = set()
    
    # Each subject has a folder with their EID, containing tian_s3_*.npy
    for subdir in tian_ts_dir.iterdir():
        if not subdir.is_dir():
            continue
        eid = subdir.name
        # Check if there's a .npy file inside
        npy_files = list(subdir.glob("tian_s3_*.npy"))
        if npy_files:
            available.add(eid)
    
    return available


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tian-ts-dir", required=True,
                    help="Directory with Tian timeseries (derived_schaefer_mdd/tian_timeseries)")
    ap.add_argument("--ids-common", required=True,
                    help="ids_common.npy from gene-brain overlap")
    ap.add_argument("--x-gene-wide", required=True,
                    help="X_gene_wide.npy (N x 85248)")
    ap.add_argument("--labels", required=True,
                    help="labels_common.npy")
    ap.add_argument("--cov-age", required=True,
                    help="cov_age.npy")
    ap.add_argument("--cov-sex", required=True,
                    help="cov_sex.npy")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for subset data")
    args = ap.parse_args()

    print("[1/5] Finding available Tian subjects...", flush=True)
    tian_ts_dir = Path(args.tian_ts_dir)
    available = find_available_subjects(tian_ts_dir)
    print(f"  Found {len(available)} subjects with Tian timeseries", flush=True)

    print("[2/5] Loading common IDs and finding intersection...", flush=True)
    ids_common = np.load(args.ids_common, allow_pickle=True).astype(str)
    print(f"  ids_common: {len(ids_common)} subjects", flush=True)
    
    # Find intersection (subjects with both gene-brain data AND Tian timeseries)
    ids_common_set = set(ids_common)
    intersection = available & ids_common_set
    print(f"  Intersection (Tian ∩ gene-brain): {len(intersection)} subjects", flush=True)
    
    if len(intersection) == 0:
        raise SystemExit("[ERROR] No overlapping subjects found!")

    # Get indices of intersection subjects in ids_common order
    # Preserve order from ids_common for consistency
    keep_mask = np.array([eid in intersection for eid in ids_common])
    keep_indices = np.where(keep_mask)[0]
    ids_subset = ids_common[keep_mask]
    
    print(f"  Subset indices: {len(keep_indices)}", flush=True)

    print("[3/5] Loading and subsetting gene matrix...", flush=True)
    # Use mmap for memory efficiency on large gene matrix
    X_gene_wide = np.load(args.x_gene_wide, mmap_mode="r")
    print(f"  Gene matrix shape: {X_gene_wide.shape}", flush=True)
    
    # Subset rows
    X_gene_subset = np.array(X_gene_wide[keep_indices])
    print(f"  Subset gene matrix shape: {X_gene_subset.shape}", flush=True)

    print("[4/5] Loading and subsetting labels and covariates...", flush=True)
    labels = np.load(args.labels)
    cov_age = np.load(args.cov_age)
    cov_sex = np.load(args.cov_sex)
    
    labels_subset = labels[keep_indices]
    age_subset = cov_age[keep_indices]
    sex_subset = cov_sex[keep_indices]
    
    # Check class balance
    n_mdd = labels_subset.sum()
    n_ctrl = len(labels_subset) - n_mdd
    print(f"  Labels: {n_mdd} MDD, {n_ctrl} control", flush=True)

    print("[5/5] Saving subset data...", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(out_dir / "ids_tian_subset.npy", ids_subset)
    np.save(out_dir / "X_gene_wide_tian_subset.npy", X_gene_subset)
    np.save(out_dir / "labels_tian_subset.npy", labels_subset)
    np.save(out_dir / "cov_age_tian_subset.npy", age_subset)
    np.save(out_dir / "cov_sex_tian_subset.npy", sex_subset)
    
    # Save metadata
    meta = {
        "n_subjects": int(len(ids_subset)),
        "n_mdd": int(n_mdd),
        "n_control": int(n_ctrl),
        "gene_dim": int(X_gene_subset.shape[1]),
        "source_ids_common": str(args.ids_common),
        "source_tian_ts_dir": str(args.tian_ts_dir),
    }
    (out_dir / "meta_tian_subset.json").write_text(json.dumps(meta, indent=2))
    
    print(f"\n[done] Saved {len(ids_subset)} subjects to {out_dir}", flush=True)
    print(f"  ids_tian_subset.npy: {ids_subset.shape}", flush=True)
    print(f"  X_gene_wide_tian_subset.npy: {X_gene_subset.shape}", flush=True)
    print(f"  labels_tian_subset.npy: {labels_subset.shape}", flush=True)


if __name__ == "__main__":
    main()
