#!/usr/bin/env python3
"""
QC gate: measure overlap between Tian atlas mask and mean_fmri_mni.

Outputs:
- ids_mni_atlas_ok.npy
- qc_mni_overlap_report.json
- qc_mni_overlap.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import nibabel as nib


def _load_ids(path: Path) -> List[str]:
    return np.load(path, allow_pickle=True).astype(str).tolist()


def _infer_subject_id(eid: str, roi_root: Path) -> str:
    files = sorted(glob(str(roi_root / eid / "tian_s3_*.npy")))
    if not files:
        return f"{eid}_20227_2_0"
    name = Path(files[0]).stem
    return name[len("tian_s3_") :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-eid", required=True, help="EID list (npy)")
    ap.add_argument("--roi-root", default="derived_schaefer_mdd/tian_timeseries")
    ap.add_argument("--fmri-dir", default="/storage/bigdata/UKB/fMRI/UKB_20227_1_recon")
    ap.add_argument("--atlas-file", default="/storage/bigdata/UKB/fMRI/tian_atlas/Tian_Subcortex_S3_3T.nii")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ids = _load_ids(Path(args.ids_eid))
    roi_root = Path(args.roi_root)
    fmri_dir = Path(args.fmri_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_img = nib.load(str(args.atlas_file))
    atlas_mask = atlas_img.get_fdata().astype(int) > 0

    rows = []
    ok_ids = []
    missing = 0

    for i, eid in enumerate(ids, 1):
        subj = _infer_subject_id(eid, roi_root)
        mean_fp = fmri_dir / subj / "mean_fmri_mni.nii.gz"
        if not mean_fp.exists():
            missing += 1
            rows.append({"eid": eid, "subject": subj, "mean_atlas_nonzero_frac": "", "status": "missing"})
            continue

        img = nib.load(str(mean_fp))
        vol = np.asanyarray(img.dataobj)
        nz = vol != 0
        atlas_nz_frac = float(np.mean(nz[atlas_mask]))

        status = "ok" if atlas_nz_frac > 0 else "zero_overlap"
        if status == "ok":
            ok_ids.append(eid)
        rows.append(
            {
                "eid": eid,
                "subject": subj,
                "mean_atlas_nonzero_frac": atlas_nz_frac,
                "status": status,
            }
        )

        if i % 200 == 0:
            print(f"Processed {i}/{len(ids)}", flush=True)

    ok_ids = np.array(ok_ids, dtype=object)
    np.save(out_dir / "ids_mni_atlas_ok.npy", ok_ids)

    report = {
        "n_total": len(ids),
        "n_ok": int(len(ok_ids)),
        "n_zero_overlap": int(sum(1 for r in rows if r["status"] == "zero_overlap")),
        "n_missing": int(missing),
        "fmri_dir": str(fmri_dir),
        "atlas_file": str(args.atlas_file),
        "roi_root": str(roi_root),
    }
    (out_dir / "qc_mni_overlap_report.json").write_text(json.dumps(report, indent=2))

    with (out_dir / "qc_mni_overlap.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eid", "subject", "mean_atlas_nonzero_frac", "status"])
        w.writeheader()
        w.writerows(rows)

    print("[done] ids_mni_atlas_ok.npy:", out_dir / "ids_mni_atlas_ok.npy")
    print("[done] qc_mni_overlap_report.json:", out_dir / "qc_mni_overlap_report.json")


if __name__ == "__main__":
    main()
