#!/usr/bin/env python3
"""
Build human-readable feature name lists for UKB tabular MRI features (sMRI/dMRI).

Inputs
------
- A columns text file produced by our overlap prep, e.g.:
  derived_tabular_overlap/smri/smri_columns.txt
  derived_tabular_overlap/dmri/dmri_columns.txt
  (First line is the ID column; remaining lines are feature column names.)
- A field dictionary CSV from the UKB tabular bundle, e.g.:
  /scratch/connectome/3DCNN/data/2.UKB/4.MRI_tabular/smri_field_dictionary.csv
  /scratch/connectome/3DCNN/data/2.UKB/4.MRI_tabular/dmri_field_dictionary.csv

Outputs
-------
- A text file with one human-readable name per feature column (aligned to X_*_raw.npy columns).
- Optionally a .npy copy of the same names (object dtype) for convenience.

Lab notes
---------
This is tiny CPU work; safe to run on login node.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_columns_txt(path: Path) -> Tuple[str, List[str]]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) < 2:
        raise SystemExit(f"[ERROR] {path} has too few lines (need id_col + >=1 feature col).")
    id_col = lines[0]
    feat_cols = lines[1:]
    return id_col, feat_cols


def _load_dict(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Returns mapping: column_name -> {"description": ..., "category": ... (optional)}
    Works for both sMRI and dMRI dictionaries.
    """
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"[ERROR] Empty CSV header: {csv_path}")

        # Required
        if "Column Name" not in reader.fieldnames or "Description" not in reader.fieldnames:
            raise SystemExit(
                f"[ERROR] {csv_path} missing required columns. "
                f"Need at least 'Column Name' and 'Description'. Got: {reader.fieldnames}"
            )

        mapping: Dict[str, Dict[str, str]] = {}
        for row in reader:
            col = (row.get("Column Name") or "").strip()
            if not col:
                continue
            desc = (row.get("Description") or "").strip()
            cat = (row.get("Category") or "").strip() if "Category" in reader.fieldnames else ""
            mapping[col] = {"description": desc, "category": cat}
        return mapping


def _format_name(col: str, meta: Dict[str, str], include_col: bool) -> str:
    desc = (meta.get("description") or "").strip()
    cat = (meta.get("category") or "").strip()

    if desc and cat:
        base = f"{cat}: {desc}"
    elif desc:
        base = desc
    else:
        base = col

    return f"{base} [{col}]" if include_col else base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--columns-txt", required=True, help="smri_columns.txt or dmri_columns.txt")
    ap.add_argument("--field-dict-csv", required=True, help="smri_field_dictionary.csv or dmri_field_dictionary.csv")
    ap.add_argument("--out-txt", required=True, help="Output .txt with one name per feature (aligned to X columns)")
    ap.add_argument("--out-npy", default=None, help="Optional output .npy (object dtype) of names")
    ap.add_argument(
        "--include-col",
        action="store_true",
        help="Append the raw column name in brackets for uniqueness (recommended).",
    )
    args = ap.parse_args()

    columns_txt = Path(args.columns_txt).expanduser().resolve()
    dict_csv = Path(args.field_dict_csv).expanduser().resolve()
    out_txt = Path(args.out_txt).expanduser().resolve()
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    _, feat_cols = _read_columns_txt(columns_txt)
    mapping = _load_dict(dict_csv)

    names: List[str] = []
    missing: List[str] = []
    for c in feat_cols:
        meta = mapping.get(c, {})
        if not meta:
            missing.append(c)
        names.append(_format_name(c, meta, include_col=bool(args.include_col)))

    out_txt.write_text("\n".join(names) + "\n")
    print(f"[save] {out_txt} (n={len(names)})", flush=True)
    if missing:
        print(f"[warn] Missing {len(missing)}/{len(names)} columns in {dict_csv.name}. Example: {missing[:5]}", flush=True)

    if args.out_npy:
        out_npy = Path(args.out_npy).expanduser().resolve()
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, np.array(names, dtype=object))
        print(f"[save] {out_npy}", flush=True)


if __name__ == "__main__":
    main()

