#!/usr/bin/env python3
"""
Align gene/covariate/label arrays to QC-filtered Tian summary subjects.

Lab rules: run via Slurm; no login-node heavy compute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def subset_by_ids(x: np.ndarray, ids_full: np.ndarray, ids_keep: np.ndarray) -> np.ndarray:
    idx_map = {sid: i for i, sid in enumerate(ids_full.astype(str))}
    keep_idx = [idx_map[sid] for sid in ids_keep.astype(str)]
    return x[np.array(keep_idx, dtype=np.int64)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-keep", required=True, help="ids_tian_summary.npy")
    ap.add_argument("--ids-full", required=True, help="ids_tian_subset.npy (full Tian subset)")
    ap.add_argument("--x-gene-wide", required=True)
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str)
    ids_full = np.load(args.ids_full, allow_pickle=True).astype(str)

    x_gene = np.load(args.x_gene_wide)
    cov_age = np.load(args.cov_age)
    cov_sex = np.load(args.cov_sex)
    labels = np.load(args.labels)

    x_gene_keep = subset_by_ids(x_gene, ids_full, ids_keep)
    cov_age_keep = subset_by_ids(cov_age, ids_full, ids_keep)
    cov_sex_keep = subset_by_ids(cov_sex, ids_full, ids_keep)
    labels_keep = subset_by_ids(labels, ids_full, ids_keep)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_gene_wide_tian_summary.npy", x_gene_keep)
    np.save(out_dir / "cov_age_tian_summary.npy", cov_age_keep)
    np.save(out_dir / "cov_sex_tian_summary.npy", cov_sex_keep)
    np.save(out_dir / "labels_tian_summary.npy", labels_keep)
    np.save(out_dir / "ids_tian_summary.npy", ids_keep.astype(object))

    print("[done] Aligned data saved to", out_dir, flush=True)
    print("  X_gene:", x_gene_keep.shape, flush=True)
    print("  cov_age:", cov_age_keep.shape, flush=True)
    print("  cov_sex:", cov_sex_keep.shape, flush=True)
    print("  labels:", labels_keep.shape, flush=True)


if __name__ == "__main__":
    main()
