#!/usr/bin/env python3
"""
Create cleaned tabular overlap matrices (v2) with NaN rows dropped.

Features
--------
- Drops any subject with any NaN in the brain matrix.
- Subsets all aligned arrays to the same rows.
- For sMRI: extracts eTIV (participant.p26521_i2) and optionally removes
  eTIV-related columns from the brain matrix.
- Optionally subsets feature name lists to match new brain columns.

Why
---
The original tabular matrices contain many NaNs (non-imaged subjects).
Leakage-safe CCA/SCCA code does not handle NaNs. This script produces
clean, aligned inputs for v2 benchmarks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np


def _read_lines(path: Path) -> List[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def _parse_drop_cols(drop_cols: Optional[str]) -> List[str]:
    if not drop_cols:
        return []
    return [c.strip() for c in drop_cols.split(",") if c.strip()]


def _load_feature_names(path: Optional[Path]) -> Optional[List[str]]:
    if path is None:
        return None
    return _read_lines(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["smri", "dmri"])
    ap.add_argument("--in-dir", required=True, help="derived_tabular_overlap/<modality>")
    ap.add_argument("--out-dir", required=True, help="derived_tabular_overlap_v2/<modality>")
    ap.add_argument("--columns-txt", required=True, help="*columns.txt with id col first")
    ap.add_argument("--feature-names-txt", default=None, help="Optional human-readable names aligned to columns")
    ap.add_argument("--drop-cols", default="", help="Comma-separated column names to drop from X (after cov extraction)")
    ap.add_argument("--etiv-col", default="participant.p26521_i2", help="eTIV column name (sMRI only)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load arrays
    X_brain = np.load(in_dir / f"X_{args.modality}_raw.npy", mmap_mode="r")
    X_gene = np.load(in_dir / "gene_dnabert2" / "X_gene_wide.npy", mmap_mode="r")
    ids = np.load(in_dir / "ids_common.npy", allow_pickle=True).astype(str)
    labels = np.load(in_dir / "labels_common.npy")
    cov_age = np.load(in_dir / "cov_age.npy")
    cov_sex = np.load(in_dir / "cov_sex.npy")

    # Load column names (id col + feature cols)
    cols_all = _read_lines(Path(args.columns_txt))
    if len(cols_all) < 2:
        raise SystemExit(f"[ERROR] columns-txt has too few lines: {args.columns_txt}")
    id_col = cols_all[0]
    feat_cols = cols_all[1:]
    if len(feat_cols) != X_brain.shape[1]:
        raise SystemExit(
            f"[ERROR] feature count mismatch: X={X_brain.shape[1]} vs columns={len(feat_cols)}"
        )

    feature_names = (
        _load_feature_names(Path(args.feature_names_txt).expanduser().resolve())
        if args.feature_names_txt
        else None
    )
    if feature_names is not None and len(feature_names) != len(feat_cols):
        raise SystemExit(
            f"[ERROR] feature-names length mismatch: {len(feature_names)} vs {len(feat_cols)}"
        )

    # Identify rows with any NaN in brain matrix
    nan_mask = np.isnan(X_brain).any(axis=1)
    keep_mask = ~nan_mask
    keep_idx = np.where(keep_mask)[0]

    # Subset all aligned arrays
    Xb = np.asarray(X_brain[keep_idx], dtype=np.float32)
    Xg = np.asarray(X_gene[keep_idx], dtype=np.float32)
    ids = ids[keep_idx]
    labels = labels[keep_idx]
    cov_age = cov_age[keep_idx]
    cov_sex = cov_sex[keep_idx]

    # Extract eTIV (sMRI only)
    cov_etiv = None
    if args.modality == "smri":
        if args.etiv_col not in feat_cols:
            raise SystemExit(f"[ERROR] eTIV column not found: {args.etiv_col}")
        etiv_idx = feat_cols.index(args.etiv_col)
        cov_etiv = Xb[:, etiv_idx].astype(np.float32, copy=False)

    # Drop specified columns from X (after extracting cov_etiv)
    drop_cols = set(_parse_drop_cols(args.drop_cols))
    if drop_cols:
        keep_feat_idx = [i for i, c in enumerate(feat_cols) if c not in drop_cols]
        Xb = Xb[:, keep_feat_idx]
        feat_cols = [feat_cols[i] for i in keep_feat_idx]
        if feature_names is not None:
            feature_names = [feature_names[i] for i in keep_feat_idx]

    # Save outputs
    np.save(out_dir / f"X_{args.modality}_raw.npy", Xb)
    gene_dir = out_dir / "gene_dnabert2"
    gene_dir.mkdir(parents=True, exist_ok=True)
    np.save(gene_dir / "X_gene_wide.npy", Xg)
    np.save(gene_dir / "ids_gene_overlap.npy", ids.astype(object))
    np.save(out_dir / "ids_common.npy", ids.astype(object))
    np.save(out_dir / "labels_common.npy", labels)
    np.save(out_dir / "cov_age.npy", cov_age)
    np.save(out_dir / "cov_sex.npy", cov_sex)
    if cov_etiv is not None:
        np.save(out_dir / "cov_etiv.npy", cov_etiv)

    # Save columns
    (out_dir / f"{args.modality}_columns.txt").write_text(
        "\n".join([id_col] + feat_cols) + "\n"
    )
    if feature_names is not None:
        (out_dir / f"{args.modality}_feature_names.txt").write_text(
            "\n".join(feature_names) + "\n"
        )

    print(
        f"[save] {args.modality}: n={Xb.shape[0]}, dim={Xb.shape[1]}, "
        f"kept_rows={len(keep_idx)}, dropped_rows={int(nan_mask.sum())}",
        flush=True,
    )


if __name__ == "__main__":
    main()

