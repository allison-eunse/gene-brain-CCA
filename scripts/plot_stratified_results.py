#!/usr/bin/env python3
"""Visualization script for stratified CCA results.

Generates publication-quality figures showing:
- MDD vs Control coupling correlations
- Effect size heatmaps with p-value annotations
- Forest plots with confidence intervals
- Weight cosine similarity comparisons
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Display names for gene foundation models and modalities
FM_ORDER = ["dnabert2", "evo2", "hyenadna", "caduceus"]
FM_DISPLAY = {
    "dnabert2": "DNABERT2",
    "evo2": "Evo2",
    "hyenadna": "HyenaDNA",
    "caduceus": "Caduceus",
}
MOD_ORDER = ["schaefer7", "schaefer17", "smri", "dmri"]
MOD_DISPLAY = {
    "schaefer7": "Schaefer-7",
    "schaefer17": "Schaefer-17",
    "smri": "sMRI",
    "dmri": "dMRI",
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _parse_name(path: Path) -> tuple[str, str]:
    stem = path.stem.replace("stratified_comparison_", "", 1)
    if "_" not in stem:
        return stem, "unknown"
    fm, modality = stem.split("_", 1)
    return fm, modality


def _display_fm(fm: str) -> str:
    return FM_DISPLAY.get(fm, fm.upper())


def _display_mod(mod: str) -> str:
    return MOD_DISPLAY.get(mod, mod.upper())


def load_all_results(derived_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comp_path in sorted(derived_dir.glob("stratified_comparison_*.json")):
        fm, modality = _parse_name(comp_path)
        comp = _load_json(comp_path)
        perm_path = derived_dir / f"stratified_perm_{fm}_{modality}.json"
        perm = _load_json(perm_path) if perm_path.exists() else {}

        rows.append(
            {
                "fm": fm,
                "modality": modality,
                "r_mdd": comp.get("r_mdd_holdout_cc1"),
                "r_ctrl": comp.get("r_ctrl_holdout_cc1"),
                "r_diff": comp.get("r_diff"),
                "p_value": comp.get("perm_p_value", perm.get("p_value")),
                "null_std": comp.get("perm_null_std", perm.get("null_std")),
                "gene_cos": comp.get("gene_weight_cosine_cc1"),
                "brain_cos": comp.get("brain_weight_cosine_cc1"),
                "mdd_config": comp.get("mdd_best_config", ""),
                "ctrl_config": comp.get("ctrl_best_config", ""),
            }
        )
    return rows


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _idx(val: str, order: list[str]) -> int:
        return order.index(val) if val in order else len(order)

    return sorted(
        rows, key=lambda r: (_idx(r["fm"], FM_ORDER), _idx(r["modality"], MOD_ORDER))
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
        }
    )


def _sig_star(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _format_pval(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}"


def plot_mdd_vs_ctrl_bars(
    rows: list[dict[str, Any]], out_path: Path, ax: plt.Axes | None = None
) -> None:
    """Grouped bar chart comparing MDD vs Control holdout correlations."""
    rows = _sort_rows(rows)
    labels = [f"{_display_fm(r['fm'])}\n{_display_mod(r['modality'])}" for r in rows]
    r_mdd = np.array([r["r_mdd"] if r["r_mdd"] is not None else 0 for r in rows], dtype=float)
    r_ctrl = np.array([r["r_ctrl"] if r["r_ctrl"] is not None else 0 for r in rows], dtype=float)
    pvals = [r.get("p_value") for r in rows]

    x = np.arange(len(labels))
    width = 0.35

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 5))

    bars_mdd = ax.bar(x - width / 2, r_mdd, width, label="MDD", color="#d95f5f", 
                      edgecolor="white", linewidth=0.7, alpha=0.85)
    bars_ctrl = ax.bar(x + width / 2, r_ctrl, width, label="Control", color="#4c78a8", 
                       edgecolor="white", linewidth=0.7, alpha=0.85)
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=1, linestyle="--")

    # Add significance stars only (no individual values to avoid clutter)
    for i, p in enumerate(pvals):
        star = _sig_star(p)
        if star:
            y_max = max(r_mdd[i], r_ctrl[i])
            y_min = min(r_mdd[i], r_ctrl[i])
            y_pos = y_max + 0.02 if y_max > 0 else y_min - 0.02
            ax.text(x[i], y_pos, star, ha="center", va="bottom" if y_max > 0 else "top",
                    fontsize=14, fontweight="bold", color="#d32f2f")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Holdout Canonical Correlation (CC1)", fontsize=10)
    ax.set_title("Gene-Brain Coupling: MDD vs Control Groups", fontsize=12, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    # Set y-axis limits with padding
    ymin = min(r_mdd.min(), r_ctrl.min()) - 0.05
    ymax = max(r_mdd.max(), r_ctrl.max()) + 0.08
    ax.set_ylim(ymin, ymax)

    if standalone and out_path:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(ax.figure)


def plot_effect_heatmap(
    rows: list[dict[str, Any]], out_path: Path, ax: plt.Axes | None = None
) -> None:
    """Heatmap of effect sizes (r_diff = r_MDD - r_Ctrl) with p-value annotations."""
    rows = _sort_rows(rows)
    fm_set = [fm for fm in FM_ORDER if any(r["fm"] == fm for r in rows)]
    mod_set = [m for m in MOD_ORDER if any(r["modality"] == m for r in rows)]

    n_rows = len(fm_set)
    n_cols = len(mod_set)
    
    grid = np.full((n_rows, n_cols), np.nan, dtype=float)
    pval_grid: list[list[float | None]] = [[None for _ in mod_set] for _ in fm_set]

    for r in rows:
        if r["fm"] not in fm_set or r["modality"] not in mod_set:
            continue
        i = fm_set.index(r["fm"])
        j = mod_set.index(r["modality"])
        grid[i, j] = r["r_diff"] if r["r_diff"] is not None else np.nan
        pval_grid[i][j] = r.get("p_value")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Symmetric colorbar around 0 (use observed, non-missing cells)
    vmax = np.nanmax(np.abs(grid)) if not np.all(np.isnan(grid)) else 0.1

    # IMPORTANT: pcolormesh does NOT draw cells for NaNs (it masks them),
    # which makes the grid look "inconsistent". To enforce a full rectangular
    # grid with consistent cell sizes, we draw missing cells as 0 (neutral color)
    # and annotate them as NA.
    missing = np.isnan(grid)
    grid_plot = grid.copy()
    grid_plot[missing] = 0.0

    # Use pcolormesh for explicit cell boundaries
    x_edges = np.arange(n_cols + 1)
    y_edges = np.arange(n_rows + 1)

    im = ax.pcolormesh(
        x_edges,
        y_edges,
        grid_plot,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        edgecolors="white",
        linewidth=2,
    )
    
    # Enforce uniform cell sizes and fixed bounds
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)  # So first FM is at top
    # Keep aspect flexible so we don't crop top/bottom rows (matplotlib can
    # otherwise auto-shift limits to 0.5..n-0.5 under some layouts).
    ax.set_aspect("auto")
    
    # Center tick labels in cells
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels([_display_mod(m) for m in mod_set], fontsize=11)
    ax.set_yticklabels([_display_fm(f) for f in fm_set], fontsize=11)

    # IMPORTANT: setting y-ticks can shift view limits (esp. with inverted axes).
    # Re-assert full cell extents so first/last rows aren't cropped.
    ax.set_ylim(n_rows, 0)
    
    # Remove tick marks
    ax.tick_params(axis='both', which='both', length=0)
    
    ax.set_xlabel("Brain Modality", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Gene Foundation Model", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_title("Effect Size: Δr = r(MDD) − r(Control)", fontsize=13, fontweight="bold", pad=15)

    # Add cell annotations.
    #
    # Use DATA coordinates to define positions within each cell, then convert
    # to Axes coordinates for placement. This keeps labels aligned with cells
    # even when aspect='equal' introduces internal padding.
    def _data_to_axes(x: float, y: float) -> tuple[float, float]:
        return ax.transAxes.inverted().transform(ax.transData.transform((x, y)))

    for i in range(n_rows):
        for j in range(n_cols):
            # Cell center in DATA coordinates
            cx_d = j + 0.5
            cy_d = i + 0.5
            cx, cy = _data_to_axes(cx_d, cy_d)

            if missing[i, j]:
                ax.text(
                    cx,
                    cy,
                    "NA",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#666666",
                    style="italic",
                    transform=ax.transAxes,
                    clip_on=True,
                )
                continue

            val = grid[i, j]
            p = pval_grid[i][j]
            star = _sig_star(p)

            # Choose text color based on background intensity
            text_color = "white" if abs(val) > vmax * 0.5 else "black"

            # Draw value (upper) and p-value (lower) with generous margins
            # inside each cell (in DATA coordinates), then convert to axes coords.
            val_y_d = cy_d - 0.15  # i + 0.35
            p_y_d = cy_d + 0.25    # i + 0.75
            _, val_y = _data_to_axes(cx_d, val_y_d)
            _, p_y = _data_to_axes(cx_d, p_y_d)

            val_text = f"{val:+.3f}{(' ' + star) if star else ''}"
            ax.text(
                cx,
                val_y,
                val_text,
                ha="center",
                va="center",
                fontsize=10.5 if star else 10,
                fontweight="bold",
                color=text_color,
                transform=ax.transAxes,
                clip_on=True,
            )

            if p is not None:
                ax.text(
                    cx,
                    p_y,
                    _format_pval(p),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    style="italic",
                    alpha=0.9,
                    transform=ax.transAxes,
                    clip_on=True,
                )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δr (MDD − Control)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    if standalone and out_path:
        # Avoid tight_layout here: it can change axes geometry after we've
        # positioned annotations in Axes coordinates (computed from transData).
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(ax.figure)


def plot_forest(
    rows: list[dict[str, Any]], out_path: Path, ax: plt.Axes | None = None
) -> None:
    """Forest plot of effect sizes with 95% CI from permutation null distribution."""
    # Sort by effect size for visual clarity
    rows = sorted([r for r in rows if r["r_diff"] is not None], key=lambda r: r["r_diff"])
    labels = [f"{_display_fm(r['fm'])} / {_display_mod(r['modality'])}" for r in rows]
    diffs = np.array([r["r_diff"] for r in rows], dtype=float)
    null_std = np.array(
        [r["null_std"] if r.get("null_std") is not None else np.nan for r in rows], dtype=float
    )
    ci = 1.96 * null_std
    pvals = [r.get("p_value") for r in rows]

    y = np.arange(len(labels))
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Color points by significance
    colors = ["#d32f2f" if p is not None and p < 0.05 else "#1976d2" for p in pvals]
    markers = ["D" if p is not None and p < 0.05 else "o" for p in pvals]

    for i, (diff, c, err, marker) in enumerate(zip(diffs, colors, ci, markers)):
        ax.errorbar(diff, y[i], xerr=err if not np.isnan(err) else 0,
                    fmt=marker, color=c, ecolor="#999999", capsize=4, 
                    markersize=7, linewidth=1.8, alpha=0.85)

    ax.axvline(0, color="#444444", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δr (MDD − Control)", fontsize=10, fontweight="bold")
    ax.set_title("Effect Size with 95% CI (Permutation Null)", fontsize=12, pad=10)
    ax.grid(axis="x", alpha=0.3, linestyle=":", linewidth=0.5)

    # Add p-value text box in corner instead of inline
    sig_count = sum(1 for p in pvals if p is not None and p < 0.05)
    textstr = f"{sig_count}/{len(pvals)} significant (p<0.05)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Legend
    sig_patch = mpatches.Patch(color="#d32f2f", label="p < 0.05")
    ns_patch = mpatches.Patch(color="#1976d2", label="p ≥ 0.05")
    ax.legend(handles=[sig_patch, ns_patch], loc="lower left", frameon=True, 
              fontsize=9, framealpha=0.9)

    if standalone and out_path:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(ax.figure)


def plot_weight_cosine(
    rows: list[dict[str, Any]], out_path: Path, ax: plt.Axes | None = None
) -> None:
    """Bar chart showing weight cosine similarity between MDD and Control models."""
    rows = _sort_rows(rows)
    labels = [f"{_display_fm(r['fm'])}\n{_display_mod(r['modality'])}" for r in rows]
    gene = np.array([r["gene_cos"] if r["gene_cos"] is not None else 0 for r in rows], dtype=float)
    brain = np.array([r["brain_cos"] if r["brain_cos"] is not None else 0 for r in rows], dtype=float)

    x = np.arange(len(labels))
    width = 0.35

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(14, 6))

    bars_gene = ax.bar(x - width / 2, gene, width, label="Gene Weights", color="#43a047", 
                       edgecolor="white", linewidth=0.7, alpha=0.85)
    bars_brain = ax.bar(x + width / 2, brain, width, label="Brain Weights", color="#fb8c00", 
                        edgecolor="white", linewidth=0.7, alpha=0.85)
    
    # Reference lines
    ax.axhline(0, color="#444444", linewidth=1.0, zorder=1, linestyle="-", alpha=0.8)
    ax.axhline(0.1, color="#e53935", linewidth=1.0, linestyle="--", zorder=1, alpha=0.6, label="±0.1 threshold")
    ax.axhline(-0.1, color="#e53935", linewidth=1.0, linestyle="--", zorder=1, alpha=0.6)
    
    # Add value labels on each bar
    for i, (g_val, b_val) in enumerate(zip(gene, brain)):
        # Gene bar label
        if abs(g_val) < 0.01:
            # Very small values - place label above/below bar position
            y_pos = 0.015 if g_val >= 0 else -0.015
            va = "bottom" if g_val >= 0 else "top"
        else:
            # Larger values - place inside or outside bar
            if abs(g_val) > 0.05:
                y_pos = g_val / 2  # Inside bar
                color = "white"
            else:
                y_pos = g_val + (0.01 if g_val > 0 else -0.01)  # Outside bar
                color = "#43a047"
            va = "center" if abs(g_val) > 0.05 else ("bottom" if g_val > 0 else "top")
            color = "white" if abs(g_val) > 0.05 else "#43a047"
            ax.text(x[i] - width/2, y_pos, f"{g_val:.3f}", ha="center", va=va,
                    fontsize=7, fontweight="bold", color=color)
            continue
        ax.text(x[i] - width/2, y_pos, f"{g_val:.3f}", ha="center", va=va,
                fontsize=7, fontweight="bold", color="#43a047")
        
        # Brain bar label
        if abs(b_val) < 0.01:
            y_pos = 0.015 if b_val >= 0 else -0.015
            va = "bottom" if b_val >= 0 else "top"
            color = "#fb8c00"
        else:
            if abs(b_val) > 0.05:
                y_pos = b_val / 2
                color = "white"
                va = "center"
            else:
                y_pos = b_val + (0.01 if b_val > 0 else -0.01)
                color = "#fb8c00"
                va = "bottom" if b_val > 0 else "top"
            ax.text(x[i] + width/2, y_pos, f"{b_val:.3f}", ha="center", va=va,
                    fontsize=7, fontweight="bold", color=color)
            continue
        ax.text(x[i] + width/2, y_pos, f"{b_val:.3f}", ha="center", va=va,
                fontsize=7, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine Similarity", fontsize=10, fontweight="bold")
    ax.set_title("Weight Similarity Between MDD and Control Models", fontsize=12, pad=10)
    ax.legend(frameon=True, loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_ylim(-0.5, 0.5)  # Zoomed in range
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)
    
    # Add shaded region for "essentially zero"
    ax.axhspan(-0.1, 0.1, alpha=0.1, color="gray", zorder=0)
    
    # Add reference text
    ax.text(0.02, 0.98, "Shaded region (|cos|<0.1): essentially zero\ncos>0: similar direction\ncos<0: opposite direction",
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    if standalone and out_path:
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(ax.figure)


def plot_combined(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Combined 2x2 panel figure with all visualizations."""
    fig = plt.figure(figsize=(16, 13))

    # Create a 2x2 grid with adjusted spacing
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3,
                          left=0.08, right=0.96, top=0.93, bottom=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_mdd_vs_ctrl_bars(rows, out_path=None, ax=ax1)
    plot_effect_heatmap(rows, out_path=None, ax=ax2)
    plot_forest(rows, out_path=None, ax=ax3)
    plot_weight_cosine(rows, out_path=None, ax=ax4)

    # Add panel labels
    for i, (ax, letter) in enumerate(zip([ax1, ax2, ax3, ax4], ["A", "B", "C", "D"])):
        ax.text(-0.08, 1.05, letter, transform=ax.transAxes, fontsize=18,
                fontweight="bold", va="bottom", ha="right")

    fig.suptitle("Stratified CCA Analysis: Gene-Brain Coupling in MDD vs Controls",
                 fontsize=15, fontweight="bold", y=0.97)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot stratified CCA results")
    parser.add_argument("--derived-dir", required=True, help="Path to stratified results directory")
    parser.add_argument("--out-dir", default="figures/stratified", help="Output directory for figures")
    args = parser.parse_args()

    _configure_style()
    derived_dir = Path(args.derived_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_all_results(derived_dir)
    if not rows:
        raise SystemExit(f"No stratified_comparison_*.json found in {derived_dir}")

    print(f"Loaded {len(rows)} result files")
    print("Generating plots...")

    plot_mdd_vs_ctrl_bars(rows, out_dir / "stratified_results_bar.png")
    print(f"  ✓ {out_dir / 'stratified_results_bar.png'}")

    plot_effect_heatmap(rows, out_dir / "stratified_results_heatmap.png")
    print(f"  ✓ {out_dir / 'stratified_results_heatmap.png'}")

    plot_forest(rows, out_dir / "stratified_results_forest.png")
    print(f"  ✓ {out_dir / 'stratified_results_forest.png'}")

    plot_weight_cosine(rows, out_dir / "stratified_results_cosine.png")
    print(f"  ✓ {out_dir / 'stratified_results_cosine.png'}")

    plot_combined(rows, out_dir / "stratified_results_summary.png")
    print(f"  ✓ {out_dir / 'stratified_results_summary.png'}")

    print("\nDone! All plots generated successfully.")


if __name__ == "__main__":
    main()
