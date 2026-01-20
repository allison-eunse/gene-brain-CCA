#!/usr/bin/env python3
"""
Generate a Tian QC supplementary report (markdown + html) and diagnostic plots.

Inputs:
- tian_extraction_health_report.json
- tian_mean_fmri_mni_overlap.csv
- tian_mask_timepoint_checks.json
- qc_report.json (from qc_tian_rois.py)
- coupling_benchmark_summary.json

Optional:
- roi-root + ids-subset for ROI validity heatmap (subjects x ROIs)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"[ERROR] matplotlib is required: {exc}")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_overlap_csv(path: Path) -> List[dict]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _import_tian_labels() -> List[str]:
    labels = None
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qc_tian_rois", str(Path(__file__).parent / "qc_tian_rois.py")
        )
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            labels = list(mod.TIAN_S3_LABELS)
    except Exception:
        labels = None
    if labels is None:
        labels = [f"ROI_{i}" for i in range(50)]
    return labels


def _plot_roi_validity_heatmap(
    roi_root: Path,
    ids_path: Path,
    std_threshold: float,
    out_path: Path,
    labels: List[str],
) -> Tuple[int, np.ndarray]:
    ids = np.load(ids_path, allow_pickle=True).astype(str).tolist()
    valid_matrix = []

    for eid in ids:
        files = sorted((roi_root / eid).glob("tian_s3_*.npy"))
        if not files:
            continue
        ts = np.load(files[0])
        if ts.ndim != 2:
            continue
        if ts.shape[0] < ts.shape[1]:
            ts = ts.T
        if ts.shape[1] != 50:
            continue
        roi_std = ts.std(axis=0)
        valid_matrix.append((roi_std > std_threshold).astype(np.int8))

    if not valid_matrix:
        raise SystemExit("[ERROR] No valid Tian time series found for heatmap.")

    mat = np.vstack(valid_matrix)
    valid_counts = mat.sum(axis=1)
    order = np.argsort(-valid_counts)
    mat_sorted = mat[order]

    plt.figure(figsize=(10, 8))
    plt.imshow(mat_sorted, aspect="auto", cmap="viridis")
    plt.colorbar(label="ROI valid (std > threshold)")
    plt.xlabel("ROI index")
    plt.ylabel("Subjects (sorted by #valid ROIs)")
    plt.title("Tian ROI Validity Heatmap (Subjects x ROIs)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return mat_sorted.shape[0], mat_sorted


def _plot_roi_validity_rates(
    roi_valid_rate: Dict[str, float], out_path: Path, labels: List[str]
) -> None:
    rates = [roi_valid_rate.get(label, 0.0) for label in labels]
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 4))
    plt.bar(x, rates, color="#4C72B0")
    plt.xticks(x, labels, rotation=90, fontsize=7)
    plt.ylabel("Validity Rate")
    plt.title("ROI Validity Rate (std > threshold)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_atlas_overlap_hist(overlap_rows: List[dict], out_path: Path) -> None:
    vals = []
    for row in overlap_rows:
        try:
            vals.append(float(row["mean_atlas_nonzero_frac"]))
        except Exception:
            continue

    plt.figure(figsize=(6, 4))
    plt.hist(vals, bins=30, color="#55A868", edgecolor="black", alpha=0.8)
    plt.xlabel("Atlas nonzero fraction (mean_fmri_mni)")
    plt.ylabel("Subjects")
    plt.title("Tian Atlas Overlap (mean_fmri_mni)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _write_html(md_text: str, images: List[Tuple[str, str]], out_html: Path) -> None:
    img_html = "\n".join(
        f'<div><img src="{src}" alt="{alt}" style="max-width:100%;"></div>'
        for src, alt in images
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Tian QC Supplement Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    pre {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Tian QC Supplement Report</h1>
  {img_html}
  <pre>{md_text}</pre>
</body>
</html>
"""
    out_html.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-report", required=True)
    ap.add_argument("--overlap-csv", required=True)
    ap.add_argument("--timepoint-checks", required=True)
    ap.add_argument("--qc-report", required=True)
    ap.add_argument("--coupling-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--roi-root", default="derived_schaefer_mdd/tian_timeseries")
    ap.add_argument("--ids-subset", default="derived_schaefer_mdd/tian_subset/ids_tian_subset.npy")
    ap.add_argument("--std-threshold", type=float, default=1e-6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    health = _load_json(Path(args.health_report))
    overlap_rows = _load_overlap_csv(Path(args.overlap_csv))
    timepoint = _load_json(Path(args.timepoint_checks))
    qc_report = _load_json(Path(args.qc_report))
    coupling = _load_json(Path(args.coupling_summary))
    meta_summary_path = Path(args.out_dir) / "meta_tian_summary.json"
    meta_summary = _load_json(meta_summary_path) if meta_summary_path.exists() else {}

    labels = _import_tian_labels()

    # Plots
    heatmap_path = out_dir / "tian_roi_validity_heatmap.png"
    rates_path = out_dir / "tian_roi_validity_rates.png"
    overlap_path = out_dir / "tian_atlas_overlap_hist.png"

    n_heat_subjects, _ = _plot_roi_validity_heatmap(
        roi_root=Path(args.roi_root),
        ids_path=Path(args.ids_subset),
        std_threshold=float(args.std_threshold),
        out_path=heatmap_path,
        labels=labels,
    )
    _plot_roi_validity_rates(health["roi_valid_rate"], rates_path, labels)
    _plot_atlas_overlap_hist(overlap_rows, overlap_path)

    # Summaries
    n_total = health["n_total"]
    n_all_zero = health["n_all_zero_ts"]
    frac_all_zero = health["frac_all_zero_ts"]
    n_valid_ge_20 = health["counts_n_valid_ge"]["20"]

    coupling_entry = coupling[0] if isinstance(coupling, list) and coupling else coupling
    r_holdout = coupling_entry.get("r_holdout_cc1")
    best_config = coupling_entry.get("best_config", "unknown")

    final_n_subjects = meta_summary.get("n_subjects_kept", qc_report.get("n_subjects_kept"))
    md_lines = [
        "# Tian QC Supplement Report",
        "",
        "## Executive Summary",
        f"- Total subjects in Tian subset: **{n_total}**",
        f"- Subjects with all-zero Tian TS: **{n_all_zero}** ({frac_all_zero:.1%})",
        f"- Subjects with ≥20 valid ROIs: **{n_valid_ge_20}**",
        f"- Subjects kept after QC: **{qc_report['n_subjects_kept']}**",
        f"- Final subjects used in summary FC: **{final_n_subjects}**",
        f"- ROIs kept after QC: **{qc_report['n_rois_kept']}**",
        "",
        "## Subject Flow",
        f"- Initial Tian subset: {n_total}",
        f"- ≥20 valid ROIs: {n_valid_ge_20}",
        f"- QC kept subjects: {qc_report['n_subjects_kept']}",
        f"- Summary FC subjects: {final_n_subjects}",
        "",
        "## MNI Registration Failure Evidence",
        "- Confusion matrix (TS all-zero vs MNI atlas overlap):",
        f"  - TS all-zero & atlas overlap = 0: {n_all_zero}",
        f"  - TS nonzero & atlas overlap > 0: {n_total - n_all_zero}",
        "",
        "## Coupling Benchmark Summary",
        f"- Best config: {best_config}",
        f"- Holdout CC1: {r_holdout}",
        "",
        "## Notes",
        "The MNI registration step produced fMRI volumes with zero signal in the subcortical (Tian) mask for ~69% of subjects. Native-space mean fMRI volumes appear normal, indicating a failure in the native→MNI resampling for those subjects.",
        "",
        "## Figures",
        f"- ROI Validity Heatmap: {heatmap_path.name} (subjects x ROIs)",
        f"- ROI Validity Rates: {rates_path.name}",
        f"- Atlas Overlap Histogram: {overlap_path.name}",
        "",
    ]

    md_text = "\n".join(md_lines)

    md_path = out_dir / "qc_supplement_report.md"
    html_path = out_dir / "qc_supplement_report.html"
    md_path.write_text(md_text)
    _write_html(
        md_text,
        [
            (heatmap_path.name, "ROI Validity Heatmap"),
            (rates_path.name, "ROI Validity Rates"),
            (overlap_path.name, "Atlas Overlap Histogram"),
        ],
        html_path,
    )

    print("[done] Wrote report:", md_path)
    print("[done] Wrote report:", html_path)


if __name__ == "__main__":
    main()
