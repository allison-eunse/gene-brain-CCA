#!/usr/bin/env python3
"""
Analyze Stratified CCA Results: Cross-modality comparison and reporting.

This script collects results from all stratified CCA benchmarks and generates:
- A unified JSON report
- A Markdown summary table
- Comparison plots (optional)

Lab server rules:
- Can run on login node (lightweight analysis)
- All heavy compute should already be done via Slurm jobs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


MODALITIES = ["schaefer7", "schaefer17", "smri_tabular", "dmri_tabular"]


def load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON file, return None if not found."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def collect_results(derived_dir: Path) -> list[dict[str, Any]]:
    """Collect stratified comparison results from all modalities."""
    results = []
    for modality in MODALITIES:
        # Load comparison file
        comp_path = derived_dir / f"stratified_comparison_{modality}.json"
        comp = load_json(comp_path)
        if comp is None:
            print(f"[warn] Missing: {comp_path}")
            continue

        # Load permutation results if available
        perm_path = derived_dir / f"stratified_perm_{modality}.json"
        perm = load_json(perm_path)

        # Load MDD and Ctrl summaries
        mdd_summary_path = derived_dir / f"stratified_{modality}_mdd" / "coupling_benchmark_summary.json"
        ctrl_summary_path = derived_dir / f"stratified_{modality}_ctrl" / "coupling_benchmark_summary.json"
        mdd_summary = load_json(mdd_summary_path)
        ctrl_summary = load_json(ctrl_summary_path)

        result = {
            "modality": modality,
            "r_mdd_holdout_cc1": comp.get("r_mdd_holdout_cc1", 0.0),
            "r_ctrl_holdout_cc1": comp.get("r_ctrl_holdout_cc1", 0.0),
            "r_diff": comp.get("r_diff", 0.0),
            "gene_weight_cosine_cc1": comp.get("gene_weight_cosine_cc1", 0.0),
            "brain_weight_cosine_cc1": comp.get("brain_weight_cosine_cc1", 0.0),
            "mdd_best_config": comp.get("mdd_best_config", "N/A"),
            "ctrl_best_config": comp.get("ctrl_best_config", "N/A"),
        }

        # Add MDD details
        if mdd_summary:
            result["mdd_cv_mean"] = mdd_summary.get("mean_r_val_cc1", 0.0)
            result["mdd_cv_std"] = mdd_summary.get("std_r_val_cc1", 0.0)
            result["mdd_overfitting_gap"] = mdd_summary.get("overfitting_gap", 0.0)

        # Add Ctrl details
        if ctrl_summary:
            result["ctrl_cv_mean"] = ctrl_summary.get("mean_r_val_cc1", 0.0)
            result["ctrl_cv_std"] = ctrl_summary.get("std_r_val_cc1", 0.0)
            result["ctrl_overfitting_gap"] = ctrl_summary.get("overfitting_gap", 0.0)

        # Add permutation results
        if perm:
            result["perm_p_value"] = perm.get("p_value", None)
            result["perm_null_mean"] = perm.get("null_mean", 0.0)
            result["perm_null_std"] = perm.get("null_std", 0.0)
            result["n_perm"] = perm.get("n_perm", 0)

        results.append(result)

    return results


def generate_markdown_report(results: list[dict[str, Any]], out_path: Path) -> None:
    """Generate a Markdown report from stratified results."""
    lines = [
        "# Stratified CCA Results: MDD vs Controls",
        "",
        "This report compares gene-brain coupling between MDD and Control groups across modalities.",
        "",
        "## Summary Table",
        "",
        "| Modality | r_MDD | r_Ctrl | Δr | Gene cos | Brain cos | p-value |",
        "|----------|-------|--------|-----|----------|-----------|---------|",
    ]

    for r in results:
        p_str = f"{r.get('perm_p_value', 'N/A'):.3f}" if r.get("perm_p_value") is not None else "N/A"
        lines.append(
            f"| {r['modality']} | {r['r_mdd_holdout_cc1']:.4f} | "
            f"{r['r_ctrl_holdout_cc1']:.4f} | {r['r_diff']:.4f} | "
            f"{r['gene_weight_cosine_cc1']:.3f} | {r['brain_weight_cosine_cc1']:.3f} | {p_str} |"
        )

    lines.extend([
        "",
        "**Legend:**",
        "- r_MDD / r_Ctrl: Holdout CC1 correlation for each group",
        "- Δr: r_MDD - r_Ctrl (positive = MDD has stronger coupling)",
        "- Gene cos / Brain cos: Cosine similarity of CC1 weights between groups (1 = identical)",
        "- p-value: Permutation test significance for Δr",
        "",
        "---",
        "",
        "## Detailed Results",
        "",
    ])

    for r in results:
        lines.extend([
            f"### {r['modality'].replace('_', ' ').title()}",
            "",
            f"**MDD:**",
            f"- Best config: `{r.get('mdd_best_config', 'N/A')}`",
            f"- CV mean: {r.get('mdd_cv_mean', 0.0):.4f} ± {r.get('mdd_cv_std', 0.0):.4f}",
            f"- Holdout CC1: {r['r_mdd_holdout_cc1']:.4f}",
            f"- Overfitting gap: {r.get('mdd_overfitting_gap', 0.0):.4f}",
            "",
            f"**Controls:**",
            f"- Best config: `{r.get('ctrl_best_config', 'N/A')}`",
            f"- CV mean: {r.get('ctrl_cv_mean', 0.0):.4f} ± {r.get('ctrl_cv_std', 0.0):.4f}",
            f"- Holdout CC1: {r['r_ctrl_holdout_cc1']:.4f}",
            f"- Overfitting gap: {r.get('ctrl_overfitting_gap', 0.0):.4f}",
            "",
            f"**Comparison:**",
            f"- Δr (MDD - Ctrl): {r['r_diff']:.4f}",
            f"- Gene weight cosine: {r['gene_weight_cosine_cc1']:.4f}",
            f"- Brain weight cosine: {r['brain_weight_cosine_cc1']:.4f}",
        ])

        if r.get("perm_p_value") is not None:
            lines.extend([
                f"- Permutation p-value: {r['perm_p_value']:.4f} (n={r.get('n_perm', 0)})",
                f"- Null distribution: {r.get('perm_null_mean', 0.0):.4f} ± {r.get('perm_null_std', 0.0):.4f}",
            ])

        lines.extend(["", "---", ""])

    # Interpretation section
    lines.extend([
        "## Interpretation Guide",
        "",
        "| Outcome | Interpretation |",
        "|---------|----------------|",
        "| r_MDD >> r_Ctrl with low p-value | MDD has stronger gene-brain coupling |",
        "| r_MDD ≈ r_Ctrl | No group difference; coupling is stable across groups |",
        "| Weight cosine << 1 | Different genes/brain regions drive coupling in each group |",
        "| Both r near 0 | Signal is weak even within groups (not just cancellation) |",
        "",
        "## Notes",
        "",
        "- All analyses use leakage-safe preprocessing (residualization/PCA fit on train only)",
        "- 5-fold CV for model selection, 20% holdout for final evaluation",
        "- Permutation test shuffles labels and re-runs the full benchmark pipeline",
    ])

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[save] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze stratified CCA results")
    ap.add_argument("--derived-dir", required=True, help="Path to derived directory with results")
    ap.add_argument("--out-json", default=None, help="Output JSON path (default: derived_dir/stratified_comparison_report.json)")
    ap.add_argument("--out-md", default=None, help="Output Markdown path (default: derived_dir/stratified_comparison_report.md)")
    args = ap.parse_args()

    derived_dir = Path(args.derived_dir)
    out_json = Path(args.out_json) if args.out_json else derived_dir / "stratified_comparison_report.json"
    out_md = Path(args.out_md) if args.out_md else derived_dir / "stratified_comparison_report.md"

    print("=" * 60)
    print("Stratified CCA Results Analysis")
    print("=" * 60)

    # Collect results
    results = collect_results(derived_dir)
    if not results:
        print("[error] No results found. Run the stratified benchmark jobs first.")
        return

    print(f"[found] {len(results)} modalities")

    # Save JSON
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {out_json}")

    # Generate Markdown
    generate_markdown_report(results, out_md)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for r in results:
        p_str = f"p={r.get('perm_p_value', 'N/A'):.3f}" if r.get("perm_p_value") is not None else ""
        print(
            f"  {r['modality']:15s}: r_MDD={r['r_mdd_holdout_cc1']:+.4f}, "
            f"r_Ctrl={r['r_ctrl_holdout_cc1']:+.4f}, Δr={r['r_diff']:+.4f} {p_str}"
        )

    print("\n[done]")


if __name__ == "__main__":
    main()
