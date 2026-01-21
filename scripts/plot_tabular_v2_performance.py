from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _fmt_float(val: float) -> str:
    if val is None:
        return "NA"
    return f"{val:.4f}".rstrip("0").rstrip(".")


def _fmt_cfg(method: str, pca_dim: int, c1: float | None, c2: float | None) -> str:
    if method == "cca":
        return f"cca_pca{pca_dim}"
    c1_str = _fmt_float(float(c1)) if c1 is not None else "NA"
    c2_str = _fmt_float(float(c2)) if c2 is not None else "NA"
    return f"scca_pca{pca_dim}_c{c1_str}_{c2_str}"


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _collect_results(full_json: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for res in full_json.get("results", []):
        rows.append(
            {
                "method": res["method"],
                "pca_dim": res["gene_pca_dim"],
                "c1": res.get("c1"),
                "c2": res.get("c2"),
                "cv_mean": res["cv"]["mean_r_val_cc1"],
                "cv_std": res["cv"]["std_r_val_cc1"],
                "overfit_gap": res["cv"]["overfitting_gap"],
                "holdout": res["holdout"]["r_holdout_cc1"],
                "gen_gap": res["holdout"]["generalization_gap"],
            }
        )
    return rows


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel_scatter(ax, rows: List[Dict], best_cfg: str) -> None:
    colors = {"cca": "#1f77b4", "scca": "#ff7f0e"}
    for method in ["cca", "scca"]:
        subset = [r for r in rows if r["method"] == method]
        ax.scatter(
            [r["cv_mean"] for r in subset],
            [r["holdout"] for r in subset],
            s=30,
            c=colors[method],
            alpha=0.8,
            label=method.upper(),
            edgecolors="none",
        )

    best_row = next((r for r in rows if r["config"] == best_cfg), None)
    if best_row is not None:
        ax.scatter(
            [best_row["cv_mean"]],
            [best_row["holdout"]],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Best",
            zorder=5,
        )
        ax.annotate(
            best_cfg,
            (best_row["cv_mean"], best_row["holdout"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title("Config selection")
    ax.set_xlabel("CV mean r (CC1)")
    ax.set_ylabel("Holdout r (CC1)")
    ax.legend(frameon=False, loc="best")


def _panel_top_configs(ax, rows: List[Dict], top_n: int) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["cv_mean"], reverse=True)[:top_n]
    x = np.arange(len(rows_sorted))
    cv_means = [r["cv_mean"] for r in rows_sorted]
    cv_std = [r["cv_std"] for r in rows_sorted]
    holdout = [r["holdout"] for r in rows_sorted]
    labels = [r["config"] for r in rows_sorted]

    ax.bar(x, cv_means, yerr=cv_std, color="#4c78a8", alpha=0.8, capsize=3, label="CV mean ± SD")
    ax.scatter(x, holdout, color="#e45756", s=25, label="Holdout r (CC1)", zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title(f"Top {len(rows_sorted)} configs by CV mean")
    ax.set_ylabel("Correlation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(frameon=False, loc="best")


def _panel_gap(ax, rows: List[Dict], best_cfg: str) -> None:
    ax.scatter(
        [r["gen_gap"] for r in rows],
        [r["holdout"] for r in rows],
        s=30,
        c="#72b7b2",
        alpha=0.8,
        edgecolors="none",
    )
    best_row = next((r for r in rows if r["config"] == best_cfg), None)
    if best_row is not None:
        ax.scatter(
            [best_row["gen_gap"]],
            [best_row["holdout"]],
            s=80,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title("Generalization gap vs holdout")
    ax.set_xlabel("Generalization gap (train CC1 - holdout CC1)")
    ax.set_ylabel("Holdout r (CC1)")


def _figure_caption(fig, title: str, summary: Dict, bestfit: Dict, metadata: Dict) -> None:
    split = bestfit.get("split", {})
    n_total = split.get("n_total", "NA")
    n_train = split.get("n_train", "NA")
    n_holdout = split.get("n_holdout", "NA")
    cov_extra = metadata.get("cov_extra", None)
    cov_extra_str = cov_extra if cov_extra else "none"

    text = (
        f"{title} | best={summary['best_config']} | "
        f"CV mean±SD={summary['mean_r_val_cc1']:.3f}±{summary['std_r_val_cc1']:.3f} | "
        f"holdout={summary['r_holdout_cc1']:.3f} | "
        f"overfit_gap={summary['overfitting_gap']:.3f} | "
        f"gen_gap={summary['generalization_gap']:.3f} | "
        f"n={n_total} (train={n_train}, holdout={n_holdout}) | "
        f"cov_extra={cov_extra_str}"
    )
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8)


def plot_modality(
    full_path: Path,
    summary_path: Path,
    bestfit_path: Path,
    out_png: Path,
    title: str,
    top_n: int,
) -> None:
    full_json = _load_json(full_path)
    summary = _load_json(summary_path)[0]
    bestfit = _load_json(bestfit_path)
    rows = _collect_results(full_json)

    for r in rows:
        r["config"] = _fmt_cfg(r["method"], r["pca_dim"], r["c1"], r["c2"])

    _configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    _panel_scatter(axes[0], rows, summary["best_config"])
    _panel_top_configs(axes[1], rows, top_n)
    _panel_gap(axes[2], rows, summary["best_config"])
    _figure_caption(fig, title, summary, bestfit, full_json.get("metadata", {}))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smri-full", required=True)
    ap.add_argument("--smri-summary", required=True)
    ap.add_argument("--smri-bestfit", required=True)
    ap.add_argument("--dmri-full", required=True)
    ap.add_argument("--dmri-summary", required=True)
    ap.add_argument("--dmri-bestfit", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    plot_modality(
        Path(args.smri_full),
        Path(args.smri_summary),
        Path(args.smri_bestfit),
        out_dir / "smri_tabular_dnabert2_v2_performance.png",
        "sMRI tabular v2 + DNABERT2",
        args.top_n,
    )
    plot_modality(
        Path(args.dmri_full),
        Path(args.dmri_summary),
        Path(args.dmri_bestfit),
        out_dir / "dmri_tabular_dnabert2_v2_performance.png",
        "dMRI tabular v2 + DNABERT2",
        args.top_n,
    )


if __name__ == "__main__":
    main()
