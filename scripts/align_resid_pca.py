#!/usr/bin/env python3
"""
Align (gene, fmri) by subject IDs, optionally filter by a valid-mask, residualize on covariates,
standardize, and PCA up to `--pca-dim` for each modality (capped by the input dimension).

Inputs:
  --ids-gene: ids_gene.npy (strings)
  --x-gene:   X_gene_*.npy  (N_gene × Dg)
  --ids-fmri: ids_fmri.npy
  --x-fmri:   X_fmri_*.npy  (N_fmri × Db)

Covariates:
  Provide --cov-age, --cov-sex (aligned to ids from nesap-genomics iids.npy),
  and optionally --cov-valid-mask to drop invalid rows before intersection.

Outputs (to --out-dir):
  ids_common.npy
  X_gene_pca{kg}.npy   where kg = min(pca_dim, gene_in_dim)
  X_fmri_pca{kb}.npy   where kb = min(pca_dim, fmri_in_dim)
  pca_info.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _to_str_ids(arr) -> np.ndarray:
    return np.asarray(arr).astype(str)


def align_by_ids(
    ids_a: np.ndarray, X_a: np.ndarray, ids_b: np.ndarray, X_b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (ids_common, Xa_aligned, Xb_aligned) with identical row order.
    Uses intersection; if duplicates exist, keeps first occurrence.
    """
    idx_a: Dict[str, int] = {}
    for i, sid in enumerate(ids_a):
        if sid not in idx_a:
            idx_a[sid] = i
    idx_b: Dict[str, int] = {}
    for i, sid in enumerate(ids_b):
        if sid not in idx_b:
            idx_b[sid] = i

    common = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    ia = np.array([idx_a[sid] for sid in common], dtype=np.int64)
    ib = np.array([idx_b[sid] for sid in common], dtype=np.int64)
    return np.array(common, dtype=object), X_a[ia], X_b[ib]


def residualize(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Residualize each column of X on covariates C using least squares.
    C should already include intercept.
    """
    # solve C * B ~= X  -> B = lstsq(C, X)
    B, *_ = np.linalg.lstsq(C, X, rcond=None)
    R = X - C @ B
    return R


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu.squeeze(), sd.squeeze()


def pca_svd(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA via SVD on centered X; returns (scores, components).
    components shape: (k, D)
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    scores = (U[:, :k] * S[:k]).astype(np.float32, copy=False)
    comps = Vt[:k].astype(np.float32, copy=False)
    return scores, comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-gene", required=True)
    ap.add_argument("--x-gene", required=True)
    ap.add_argument("--ids-fmri", required=True)
    ap.add_argument("--x-fmri", required=True)

    ap.add_argument("--cov-iids", required=True, help="iids.npy that covariates are aligned to")
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--cov-valid-mask", default=None, help="Optional valid mask npy (1D bool/int)")

    ap.add_argument("--pca-dim", type=int, default=512)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ids_gene = _to_str_ids(np.load(args.ids_gene, allow_pickle=True))
    X_gene = np.load(args.x_gene, mmap_mode="r")
    ids_fmri = _to_str_ids(np.load(args.ids_fmri, allow_pickle=True))
    X_fmri = np.load(args.x_fmri, mmap_mode="r")

    cov_iids = _to_str_ids(np.load(args.cov_iids, allow_pickle=True))
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    if args.cov_valid_mask:
        vmask = np.load(args.cov_valid_mask).astype(bool)
    else:
        vmask = np.ones(cov_iids.shape[0], dtype=bool)

    # covariate lookup by iid
    cov_idx = {sid: i for i, sid in enumerate(cov_iids) if vmask[i]}

    # align gene/fmri first
    ids_common, Xg, Xb = align_by_ids(ids_gene, np.asarray(X_gene), ids_fmri, np.asarray(X_fmri))

    # build covariate matrix C for the aligned subjects
    a = []
    s = []
    kept = []
    for sid in ids_common.astype(str):
        i = cov_idx.get(sid)
        if i is None:
            continue
        a.append(float(age[i]))
        s.append(float(sex[i]))
        kept.append(sid)
    kept = np.array(kept, dtype=object)

    # filter Xg/Xb to kept
    keep_set = set(kept.astype(str))
    mask = np.array([sid in keep_set for sid in ids_common.astype(str)], dtype=bool)
    Xg = Xg[mask]
    Xb = Xb[mask]

    C = np.column_stack([np.ones(len(kept), dtype=np.float64), np.asarray(a), np.asarray(s)]).astype(np.float64)

    # residualize + standardize
    Xg_r = residualize(Xg.astype(np.float64, copy=False), C)
    Xb_r = residualize(Xb.astype(np.float64, copy=False), C)
    Xg_z, _, _ = standardize(Xg_r)
    Xb_z, _, _ = standardize(Xb_r)

    # PCA
    kg = min(args.pca_dim, Xg_z.shape[1])
    kb = min(args.pca_dim, Xb_z.shape[1])
    Xg_p, _ = pca_svd(Xg_z, n_components=kg)
    Xb_p, _ = pca_svd(Xb_z, n_components=kb)

    np.save(out_dir / "ids_common.npy", kept)
    np.save(out_dir / f"X_gene_pca{kg}.npy", Xg_p)
    np.save(out_dir / f"X_fmri_pca{kb}.npy", Xb_p)

    info = {
        "n_common": int(len(kept)),
        "gene_in_dim": int(Xg.shape[1]),
        "fmri_in_dim": int(Xb.shape[1]),
        "gene_pca_dim": int(kg),
        "fmri_pca_dim": int(kb),
        "covariates": ["intercept", "age", "sex"],
        "used_valid_mask": bool(args.cov_valid_mask is not None),
    }
    (out_dir / "pca_info.json").write_text(json.dumps(info, indent=2))
    print(f"[save] {out_dir} (n_common={len(kept)})", flush=True)


if __name__ == "__main__":
    main()


