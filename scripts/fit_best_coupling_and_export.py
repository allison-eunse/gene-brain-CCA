#!/usr/bin/env python3
"""
Fit the best coupling config (from coupling_benchmark_summary.json) and export artifacts.

What this does
--------------
- Reproduces the same leakage-safe split used in `scripts/run_coupling_benchmark.py`
  (StratifiedShuffleSplit with seed + holdout_frac).
- Applies train-only residualization (age/sex) and standardization.
- Applies train-only gene PCA (to the best gene_pca_dim).
- Fits either CCA or SCCA (PMD) using the best_config string.
- Exports:
  - U/V canonical variates (train + holdout)
  - raw weights in PCA space + mapped-back gene weights in original feature space
  - brain loadings (correlations) in the ORIGINAL brain feature space
  - top brain features by absolute loading
  - gene-level aggregation (optional): 85,248 dims -> (n_genes, 768) -> L2/mean|w|

Why this exists
--------------
`scripts/run_coupling_benchmark.py` is optimized to search configs, not to save a full
interpretation bundle for the best run. This script is the “one best fit” exporter.

Lab compliance
--------------
Run on a compute node via Slurm for the full dataset; do not run heavy jobs on the login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

# Local SCCA implementation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "gene-brain-cca-2" / "scripts"))
from scca_pmd import SCCA_PMD  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _parse_best_config(best_config: str) -> Tuple[str, int, Optional[float], Optional[float]]:
    """
    Parse strings like:
      - cca_pca128
      - scca_pca64_c0.1_0.3
    Returns: (method, gene_pca_dim, c1, c2)
    """
    s = best_config.strip()
    if s.startswith("cca_pca"):
        gene_pca_dim = int(s.replace("cca_pca", ""))
        return "cca", gene_pca_dim, None, None
    if s.startswith("scca_pca"):
        rest = s.replace("scca_pca", "")
        # rest like "64_c0.1_0.3"
        if "_c" not in rest:
            raise ValueError(f"Unparseable best_config (missing _c): {best_config}")
        pca_str, c_str = rest.split("_c", 1)
        gene_pca_dim = int(pca_str)
        c_parts = c_str.split("_")
        if len(c_parts) != 2:
            raise ValueError(f"Unparseable best_config (c1/c2): {best_config}")
        c1 = float(c_parts[0])
        c2 = float(c_parts[1])
        return "scca", gene_pca_dim, c1, c2
    raise ValueError(f"Unknown best_config format: {best_config}")


def _build_cov(age: np.ndarray, sex: np.ndarray, extra: Optional[np.ndarray]) -> np.ndarray:
    parts = [
        np.ones(len(age), dtype=np.float32),
        age.astype(np.float32),
        sex.astype(np.float32),
    ]
    if extra is not None:
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        parts.append(extra.astype(np.float32))
    return np.column_stack(parts)


def _fit_resid_betas(C_tr: np.ndarray, X_tr: np.ndarray) -> np.ndarray:
    """
    Fit B in X ~= C @ B on TRAIN only, returning B (3 x D).
    We keep large arrays in float32; only the 3x3 solve uses float64.
    """
    CtC = (C_tr.T @ C_tr).astype(np.float64)  # 3x3
    CtX = (C_tr.T @ X_tr).astype(np.float64)  # 3xD
    B = np.linalg.solve(CtC, CtX).astype(np.float32, copy=False)  # 3xD
    return B


def _apply_resid(C: np.ndarray, X: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (X - (C @ B)).astype(np.float32, copy=False)


def _fit_standardizer(X_tr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_tr.mean(axis=0).astype(np.float32, copy=False)
    sd = X_tr.std(axis=0).astype(np.float32, copy=False)
    sd = np.where(sd == 0, 1.0, sd).astype(np.float32, copy=False)
    return mu, sd


def _apply_standardizer(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32, copy=False)


def _canonical_corrs(U: np.ndarray, V: np.ndarray, k: int) -> np.ndarray:
    k = min(k, U.shape[1], V.shape[1])
    return np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(k)], dtype=np.float32)


def _brain_loadings(Xb_tr_s: np.ndarray, V_tr: np.ndarray) -> np.ndarray:
    """
    Compute brain loadings: corr(feature_j, V_tr[:,k]) for each component k.
    Xb_tr_s is standardized (train-only).
    """
    n = Xb_tr_s.shape[0]
    Vc = V_tr - V_tr.mean(axis=0, keepdims=True)
    Vsd = Vc.std(axis=0, keepdims=True)
    Vsd = np.where(Vsd == 0, 1.0, Vsd)
    # corr = (X^T @ Vc) / ((n-1) * 1 * std(V))
    return (Xb_tr_s.T @ Vc) / (float(max(n - 1, 1)) * Vsd)


def _topk(names: List[str], values: np.ndarray, k: int = 20) -> List[Tuple[str, float]]:
    idx = np.argsort(np.abs(values))[-k:][::-1]
    return [(names[i], float(values[i])) for i in idx]


def _load_name_list(path: Optional[Path], expected_len: int, fallback_prefix: str) -> List[str]:
    if path is None:
        return [f"{fallback_prefix}_{i}" for i in range(expected_len)]
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True).astype(str)
        names = arr.tolist()
    else:
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(names) != expected_len:
        raise SystemExit(f"[ERROR] Name list len {len(names)} != expected {expected_len} ({path})")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True, help="coupling_benchmark_summary.json (single brain feature set expected)")
    ap.add_argument("--x-gene-wide", required=True)
    ap.add_argument("--x-brain", required=True)
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--cov-extra", default=None, help="Optional extra covariate(s) .npy (N,) or (N,C)")
    ap.add_argument("--brain-names", default=None, help="Optional .txt/.npy list of brain feature names (len = brain_dim)")
    ap.add_argument("--gene-names", default=None, help="Optional genes.npy or gene list .txt (len = n_genes)")
    ap.add_argument("--k", type=int, default=10, help="Number of canonical components to export")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load benchmark summary and pick best_config (single brain feature set assumed)
    summary = _load_json(Path(args.summary_json))
    if not isinstance(summary, list) or not summary:
        raise SystemExit(f"[ERROR] summary-json must be a non-empty list: {args.summary_json}")
    if len(summary) > 1:
        # Still okay, but we take the first unless user provides a filter later.
        pass
    best_config = summary[0]["best_config"]
    method, gene_pca_dim, c1, c2 = _parse_best_config(best_config)

    # Load arrays (memmap where possible)
    Xg = np.load(args.x_gene_wide, mmap_mode="r")
    Xb = np.load(args.x_brain, mmap_mode="r")
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    extra = None
    if args.cov_extra:
        extra = np.load(args.cov_extra)
        if extra.ndim == 1:
            extra = extra.reshape(-1, 1)
        if extra.shape[0] != Xg.shape[0]:
            raise SystemExit(f"[ERROR] cov-extra row mismatch: {extra.shape[0]} != {Xg.shape[0]}")
    y = np.load(args.labels)
    ids = np.load(args.ids, allow_pickle=True).astype(str)

    if not (Xg.shape[0] == Xb.shape[0] == age.shape[0] == sex.shape[0] == y.shape[0] == ids.shape[0]):
        raise SystemExit(
            "[ERROR] Row mismatch:\n"
            f"  Xg={Xg.shape}\n  Xb={Xb.shape}\n  age={age.shape}\n  sex={sex.shape}\n  y={y.shape}\n  ids={ids.shape}"
        )

    # Reproduce leakage-safe holdout split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(y)), y))

    # Brain preprocessing (resid + standardize) in ORIGINAL brain feature space
    C_tr = _build_cov(age[train_idx], sex[train_idx], extra[train_idx] if extra is not None else None)
    C_ho = _build_cov(age[hold_idx], sex[hold_idx], extra[hold_idx] if extra is not None else None)

    Xb_tr = np.asarray(Xb[train_idx], dtype=np.float32)
    B_b = _fit_resid_betas(C_tr, Xb_tr)
    Xb_tr_r = _apply_resid(C_tr, Xb_tr, B_b)
    Xb_ho_r = _apply_resid(C_ho, np.asarray(Xb[hold_idx], dtype=np.float32), B_b)
    mu_b, sd_b = _fit_standardizer(Xb_tr_r)
    Xb_tr_s = _apply_standardizer(Xb_tr_r, mu_b, sd_b)
    Xb_ho_s = _apply_standardizer(Xb_ho_r, mu_b, sd_b)

    # Gene preprocessing: resid + standardize, then PCA(train-only)
    Xg_tr = np.asarray(Xg[train_idx], dtype=np.float32)
    B_g = _fit_resid_betas(C_tr, Xg_tr)
    Xg_tr_r = _apply_resid(C_tr, Xg_tr, B_g)
    Xg_ho_r = _apply_resid(C_ho, np.asarray(Xg[hold_idx], dtype=np.float32), B_g)
    mu_g, sd_g = _fit_standardizer(Xg_tr_r)
    Xg_tr_s = _apply_standardizer(Xg_tr_r, mu_g, sd_g)
    Xg_ho_s = _apply_standardizer(Xg_ho_r, mu_g, sd_g)

    gene_pca_dim = int(min(gene_pca_dim, Xg_tr_s.shape[0], Xg_tr_s.shape[1]))
    pca = PCA(n_components=gene_pca_dim, svd_solver="randomized", random_state=args.seed)
    Xg_tr_p = pca.fit_transform(Xg_tr_s).astype(np.float32, copy=False)
    Xg_ho_p = pca.transform(Xg_ho_s).astype(np.float32, copy=False)

    kk = int(min(args.k, Xg_tr_p.shape[1], Xb_tr_s.shape[1]))

    # Fit model on TRAIN, evaluate HOLDOUT
    if method == "cca":
        model = CCA(n_components=kk, max_iter=500)
        model.fit(Xg_tr_p, Xb_tr_s)
        U_tr, V_tr = model.transform(Xg_tr_p, Xb_tr_s)
        U_ho, V_ho = model.transform(Xg_ho_p, Xb_ho_s)
        Wg_pca = model.x_weights_.astype(np.float32, copy=False)  # (gene_pca_dim, k)
        Wb = model.y_weights_.astype(np.float32, copy=False)      # (brain_dim, k)
    elif method == "scca":
        model = SCCA_PMD(latent_dimensions=kk, c=[float(c1), float(c2)], max_iter=500, tol=1e-6)
        model.fit([Xg_tr_p, Xb_tr_s])
        U_tr, V_tr = model.transform([Xg_tr_p, Xb_tr_s])
        U_ho, V_ho = model.transform([Xg_ho_p, Xb_ho_s])
        Wg_pca, Wb = model.weights
        Wg_pca = Wg_pca.astype(np.float32, copy=False)
        Wb = Wb.astype(np.float32, copy=False)
    else:
        raise SystemExit(f"[ERROR] Unknown method parsed from best_config: {method}")

    r_train = _canonical_corrs(U_tr, V_tr, kk)
    r_holdout = _canonical_corrs(U_ho, V_ho, kk)

    # Brain loadings in original (resid+z) brain feature space
    brain_load = _brain_loadings(Xb_tr_s, V_tr).astype(np.float32, copy=False)  # (brain_dim, k)

    # Map gene weights back to original standardized gene features: (p, k) = (p, d) @ (d, k)
    # PCA components_: (d, p) so transpose gives (p, d)
    gene_w_orig = (pca.components_.T @ Wg_pca).astype(np.float32, copy=False)

    # Names for reporting
    brain_names = _load_name_list(
        Path(args.brain_names).expanduser().resolve() if args.brain_names else None,
        expected_len=int(Xb.shape[1]),
        fallback_prefix="brain",
    )

    # Top brain features by CC1 loading
    top_brain_cc1 = _topk(brain_names, brain_load[:, 0], k=min(30, len(brain_names)))

    # Gene-level aggregation (optional)
    gene_level = None
    if args.gene_names:
        gene_names_path = Path(args.gene_names).expanduser().resolve()
        if gene_names_path.suffix.lower() == ".npy":
            genes = np.load(gene_names_path, allow_pickle=True).astype(str).tolist()
        else:
            genes = [ln.strip() for ln in gene_names_path.read_text().splitlines() if ln.strip()]
        if len(genes) > 0 and (len(genes) * 768 == gene_w_orig.shape[0]):
            Wg_cc1 = gene_w_orig[:, 0].reshape(len(genes), 768)
            gene_l2 = np.linalg.norm(Wg_cc1, axis=1)
            gene_meanabs = np.mean(np.abs(Wg_cc1), axis=1)
            top_idx = np.argsort(gene_l2)[-30:][::-1]
            gene_level = {
                "n_genes": int(len(genes)),
                "metric": "l2_norm_of_mapped_weight_cc1",
                "top": [(genes[i], float(gene_l2[i]), float(gene_meanabs[i])) for i in top_idx],
            }

    results: Dict[str, Any] = {
        "best_config": best_config,
        "parsed": {"method": method, "gene_pca_dim": int(gene_pca_dim), "c1": c1, "c2": c2, "k": int(kk)},
        "split": {
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "holdout_frac": float(args.holdout_frac),
            "seed": int(args.seed),
            "pos_ratio_total": float(np.mean(y)),
            "pos_ratio_train": float(np.mean(y[train_idx])),
            "pos_ratio_holdout": float(np.mean(y[hold_idx])),
        },
        "r_train": r_train.tolist(),
        "r_holdout": r_holdout.tolist(),
        "top_brain_cc1_by_loading": top_brain_cc1,
        "gene_level": gene_level,
        "notes": {
            "brain_values": "Loadings (correlations) computed on train set (residualized+standardized), not beta weights.",
            "cov_extra": args.cov_extra,
        },
    }

    # Save artifacts
    (out_dir / "bestfit_results.json").write_text(json.dumps(results, indent=2))
    np.save(out_dir / "train_ids.npy", ids[train_idx].astype(object))
    np.save(out_dir / "holdout_ids.npy", ids[hold_idx].astype(object))
    np.save(out_dir / "U_train.npy", U_tr.astype(np.float32))
    np.save(out_dir / "V_train.npy", V_tr.astype(np.float32))
    np.save(out_dir / "U_holdout.npy", U_ho.astype(np.float32))
    np.save(out_dir / "V_holdout.npy", V_ho.astype(np.float32))
    np.save(out_dir / "W_gene_pca.npy", Wg_pca.astype(np.float32))
    np.save(out_dir / "W_brain.npy", Wb.astype(np.float32))
    np.save(out_dir / "W_gene_mapped.npy", gene_w_orig.astype(np.float32))
    np.save(out_dir / "brain_loadings.npy", brain_load.astype(np.float32))

    print(f"[done] wrote {out_dir}/bestfit_results.json", flush=True)


if __name__ == "__main__":
    main()

