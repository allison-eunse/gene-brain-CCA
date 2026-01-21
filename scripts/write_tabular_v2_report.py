from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def _format_float(val: float | None, ndigits: int = 3) -> str:
    if val is None:
        return "NA"
    return f"{val:.{ndigits}f}"


def _section_header(title: str) -> str:
    return f"## {title}\n"


def _modality_block(name: str, summary: Dict, bestfit: Dict) -> str:
    split = bestfit.get("split", {})
    notes = bestfit.get("notes", {})
    parsed = bestfit.get("parsed", {})
    n_total = split.get("n_total", "NA")
    n_train = split.get("n_train", "NA")
    n_holdout = split.get("n_holdout", "NA")
    pos_ratio = split.get("pos_ratio_total", "NA")

    lines = [
        _section_header(f"{name} results"),
        f"- Best config: `{summary['best_config']}`",
        f"- CV mean r (CC1): `{_format_float(summary['mean_r_val_cc1'])}` ± `{_format_float(summary['std_r_val_cc1'])}`",
        f"- Holdout r (CC1): `{_format_float(summary['r_holdout_cc1'])}`",
        f"- Overfitting gap (train - val): `{_format_float(summary['overfitting_gap'])}`",
        f"- Generalization gap (train - holdout): `{_format_float(summary['generalization_gap'])}`",
        f"- Split: n_total={n_total}, n_train={n_train}, n_holdout={n_holdout}, pos_ratio_total={pos_ratio}",
        f"- Parsed config: method={parsed.get('method')}, gene_pca_dim={parsed.get('gene_pca_dim')}, c1={parsed.get('c1')}, c2={parsed.get('c2')}",
        f"- Brain values: {notes.get('brain_values', 'NA')}",
    ]
    return "\n".join(lines) + "\n\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smri-summary", required=True)
    ap.add_argument("--smri-bestfit", required=True)
    ap.add_argument("--dmri-summary", required=True)
    ap.add_argument("--dmri-bestfit", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    smri_summary = _load_json(Path(args.smri_summary))[0]
    smri_bestfit = _load_json(Path(args.smri_bestfit))
    dmri_summary = _load_json(Path(args.dmri_summary))[0]
    dmri_bestfit = _load_json(Path(args.dmri_bestfit))

    md_lines = [
        "# DNABERT2 × Brain CCA (tabular v2) — Results summary\n",
        "This report explains the leakage-safe CCA/SCCA benchmark results and how to interpret the performance plots.\n",
        _section_header("What was evaluated"),
        "- Input: DNABERT2 gene embeddings vs sMRI/dMRI tabular features.",
        "- Preprocessing: leakage-safe residualization, standardization, and gene PCA fit on training data only.",
        "- Selection: 5-fold CV on training set; final holdout evaluation on a fixed 20% split.",
        "- Metrics: CC1 correlation (first canonical variate), reported as CV mean±SD and holdout r.",
        "\n",
        _section_header("How to read the plots"),
        "- Panel A (Config selection): higher CV mean and higher holdout r are better; points near 0 indicate weak coupling.",
        "- Panel B (Top configs): bars show CV mean±SD; red points show holdout r for the same configs.",
        "- Panel C (Gap diagnostic): large gaps with low holdout imply overfitting; small gaps with positive holdout imply better generalization.",
        "\n",
        _section_header("Interpretation guidance"),
        "- CC1 is the strongest linear coupling between gene and brain spaces after leakage-safe preprocessing.",
        "- Negative or near-zero holdout r suggests no stable coupling under the current feature set or hyperparameters.",
        "- Overfitting gap captures train→val inflation; generalization gap captures train→holdout drop.",
        "- Brain maps use loadings (correlations), not beta weights, to highlight interpretable contributors.",
        "\n",
        _section_header("Covariates and leakage safety"),
        "- Age and sex are residualized in both modalities.",
        "- sMRI additionally includes eTIV (intracranial volume) as a covariate to remove size-related variance.",
        "- All residualization parameters are fit on training data only and applied to validation/holdout splits.",
        "\n",
        _modality_block("sMRI", smri_summary, smri_bestfit),
        _modality_block("dMRI", dmri_summary, dmri_bestfit),
        _section_header("Where outputs are saved"),
        "- Performance figures: `gene-brain-cca-2/derived/figures_tabular_v2/`",
        "- Benchmarks: `gene-brain-cca-2/derived/coupling_benchmark_*_tabular_dnabert2_v2/`",
        "- Best-fit exports: `gene-brain-cca-2/derived/bestfit_*_tabular_dnabert2_v2/`",
        "\n",
        _section_header("Next checks (optional)"),
        "- Permutation testing to confirm significance of holdout r.",
        "- Sensitivity across multiple random seeds.",
        "- Compare to alternative embeddings or additional covariates (if available).",
    ]

    out_path = Path(args.out_md).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
