#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
# Use local standalone SCCA_PMD implementation (avoids cca-zoo version issues)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scca_pmd import SCCA_PMD

def _standardize_train_val(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    return X_tr_s, X_va_s


def _build_cov(age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """(N,) age/sex -> (N,3) covariate matrix with intercept."""
    return np.column_stack(
        [np.ones(len(age), dtype=np.float64), age.astype(np.float64), sex.astype(np.float64)]
    )


def _residualize_train_val(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    age_tr: np.ndarray,
    sex_tr: np.ndarray,
    age_va: np.ndarray,
    sex_va: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit linear residualization on TRAIN only (intercept+age+sex),
    apply to both train and val.
    """
    C_tr = _build_cov(age_tr, sex_tr)
    # Solve B = (C^T C)^{-1} C^T X  (3×D) – stable and faster than generic lstsq for tiny C
    CtC = C_tr.T @ C_tr  # 3×3
    CtX = C_tr.T @ X_tr.astype(np.float64, copy=False)  # 3×D
    B = np.linalg.solve(CtC, CtX)
    X_tr_r = X_tr.astype(np.float64, copy=False) - (C_tr @ B)

    C_va = _build_cov(age_va, sex_va)
    X_va_r = X_va.astype(np.float64, copy=False) - (C_va @ B)

    return X_tr_r.astype(np.float32, copy=False), X_va_r.astype(np.float32, copy=False)


def fit_scca(X, Y, k, c1, c2, max_iter=500, tol=1e-6):
    m = SCCA_PMD(latent_dimensions=k, c=[c1,c2], max_iter=max_iter, tol=tol)
    m.fit([X,Y])
    U,V = m.transform([X,Y])
    Wx,Wy = m.weights
    r = np.array([np.corrcoef(U[:,i], V[:,i])[0,1] for i in range(k)])
    return U,V,Wx,Wy,r

def top_feats(W, names, k=10):
    idx = np.argsort(np.abs(W))[-k:][::-1]
    return [(names[i], float(W[i])) for i in idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-gene", default=None, help="(Legacy) Preprocessed gene matrix (N×G)")
    ap.add_argument("--x-fmri", default=None, help="(Legacy) Preprocessed fMRI matrix (N×D)")
    ap.add_argument("--x-gene-raw", default=None, help="Raw aligned gene matrix (N×G), before residualization")
    ap.add_argument("--x-fmri-raw", default=None, help="Raw aligned fMRI matrix (N×D), before residualization")
    ap.add_argument("--cov-age", default=None, help="Aligned age covariate (N,), same row order as x-*-raw")
    ap.add_argument("--cov-sex", default=None, help="Aligned sex covariate (N,), same row order as x-*-raw")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gene-names", default=None, help="Optional genes.npy (len = gene_dim) for reporting")
    ap.add_argument("--roi-names", default=None, help="Optional ROI names .npy/.txt (len = fmri_dim) for reporting")
    ap.add_argument("--c1", type=float, default=0.3)
    ap.add_argument("--c2", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="Stratified holdout fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    if args.x_gene_raw and args.x_fmri_raw and args.cov_age and args.cov_sex:
        Xg_raw = np.load(args.x_gene_raw, mmap_mode="r")
        Xb_raw = np.load(args.x_fmri_raw, mmap_mode="r")
        age = np.load(args.cov_age)
        sex = np.load(args.cov_sex)
    elif args.x_gene and args.x_fmri:
        # Backward-compatible mode: treat x-gene/x-fmri as already residualized+scaled.
        Xg_raw = np.load(args.x_gene, mmap_mode="r")
        Xb_raw = np.load(args.x_fmri, mmap_mode="r")
        age = None
        sex = None
    else:
        raise SystemExit("[ERROR] Provide either --x-gene-raw/--x-fmri-raw/--cov-age/--cov-sex OR --x-gene/--x-fmri")

    y = np.load(args.labels)
    ids = np.load(args.ids, allow_pickle=True).astype(str)
    if (
        Xg_raw.shape[0] != Xb_raw.shape[0]
        or Xg_raw.shape[0] != y.shape[0]
        or ids.shape[0] != Xg_raw.shape[0]
    ):
        raise SystemExit(
            f"[ERROR] Row mismatch: Xg={Xg_raw.shape}, Xb={Xb_raw.shape}, y={y.shape}, ids={ids.shape}"
        )
    if age is not None and (age.shape[0] != Xg_raw.shape[0] or sex.shape[0] != Xg_raw.shape[0]):
        raise SystemExit(f"[ERROR] Covariate row mismatch: age={age.shape}, sex={sex.shape}, N={Xg_raw.shape[0]}")

    gene_names = None
    if args.gene_names:
        gene_names = np.load(args.gene_names, allow_pickle=True).astype(str)
        if gene_names.shape[0] != Xg_raw.shape[1]:
            raise SystemExit(f"[ERROR] gene-names len {gene_names.shape[0]} != gene_dim {Xg_raw.shape[1]}")
    else:
        gene_names = np.array([f"gene_{i}" for i in range(Xg_raw.shape[1])], dtype=object)

    roi_names = None
    if args.roi_names:
        rp = Path(args.roi_names)
        if rp.suffix.lower() in {".npy"}:
            roi_names = np.load(rp, allow_pickle=True).astype(str)
        else:
            roi_names = np.array([ln.strip() for ln in rp.read_text().splitlines() if ln.strip()], dtype=object)
        if roi_names.shape[0] != Xb_raw.shape[1]:
            raise SystemExit(f"[ERROR] roi-names len {roi_names.shape[0]} != fmri_dim {Xb_raw.shape[1]}")
    else:
        roi_names = np.array([f"roi_{i}" for i in range(Xb_raw.shape[1])], dtype=object)

    # 1) Create stratified holdout split (generalization test)
    if not (0.0 < args.holdout_frac < 1.0):
        raise SystemExit("[ERROR] --holdout-frac must be in (0,1)")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(y)), y))

    # 2) Cross-validation within TRAIN ONLY (tuning/stability)
    y_train = y[train_idx]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    results = {
        "params": {"c1": args.c1, "c2": args.c2, "k": args.k, "n_folds": args.n_folds, "seed": args.seed, "holdout_frac": args.holdout_frac},
        "split": {
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "pos_ratio_total": float(np.mean(y)),
            "pos_ratio_train": float(np.mean(y_train)),
            "pos_ratio_holdout": float(np.mean(y[hold_idx])),
        },
        "folds": [],
    }

    for f, (tr_rel, va_rel) in enumerate(skf.split(np.zeros(len(train_idx)), y_train)):
        tr = train_idx[tr_rel]
        va = train_idx[va_rel]

        kk = min(args.k, Xg_raw.shape[1], Xb_raw.shape[1])

        # Residualize on TRAIN only if covariates provided; else pass through.
        if age is not None:
            Xg_tr_r, Xg_va_r = _residualize_train_val(Xg_raw[tr], Xg_raw[va], age[tr], sex[tr], age[va], sex[va])
            Xb_tr_r, Xb_va_r = _residualize_train_val(Xb_raw[tr], Xb_raw[va], age[tr], sex[tr], age[va], sex[va])
        else:
            Xg_tr_r, Xg_va_r = Xg_raw[tr], Xg_raw[va]
            Xb_tr_r, Xb_va_r = Xb_raw[tr], Xb_raw[va]

        # Fold-wise standardization to avoid any distribution leakage.
        Xg_tr_s, Xg_va_s = _standardize_train_val(Xg_tr_r, Xg_va_r)
        Xb_tr_s, Xb_va_s = _standardize_train_val(Xb_tr_r, Xb_va_r)

        m = SCCA_PMD(latent_dimensions=kk, c=[args.c1, args.c2], max_iter=500, tol=1e-6)
        m.fit([Xg_tr_s, Xb_tr_s])
        U_tr, V_tr = m.transform([Xg_tr_s, Xb_tr_s])
        U_va, V_va = m.transform([Xg_va_s, Xb_va_s])
        Wx, Wy = m.weights

        r_tr = np.array([np.corrcoef(U_tr[:, i], V_tr[:, i])[0, 1] for i in range(kk)])
        r_va = np.array([np.corrcoef(U_va[:, i], V_va[:, i])[0, 1] for i in range(kk)])

        results["folds"].append(
            {
                "fold": f,
                "r_train": r_tr.tolist(),
                "r_val": r_va.tolist(),
                "sparsity_gene": float((np.abs(Wx) < 1e-3).mean()),
                "sparsity_fmri": float((np.abs(Wy) < 1e-3).mean()),
            }
        )

    # 3) Final fit on TRAIN ONLY, then evaluate HOLDOUT (one-touch generalization)
    kk = min(args.k, Xg_raw.shape[1], Xb_raw.shape[1])

    if age is not None:
        # Residualize train vs holdout using betas from train only
        Xg_tr_r, Xg_ho_r = _residualize_train_val(
            Xg_raw[train_idx], Xg_raw[hold_idx], age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx]
        )
        Xb_tr_r, Xb_ho_r = _residualize_train_val(
            Xb_raw[train_idx], Xb_raw[hold_idx], age[train_idx], sex[train_idx], age[hold_idx], sex[hold_idx]
        )
    else:
        Xg_tr_r, Xg_ho_r = Xg_raw[train_idx], Xg_raw[hold_idx]
        Xb_tr_r, Xb_ho_r = Xb_raw[train_idx], Xb_raw[hold_idx]

    Xg_tr_s, Xg_ho_s = _standardize_train_val(Xg_tr_r, Xg_ho_r)
    Xb_tr_s, Xb_ho_s = _standardize_train_val(Xb_tr_r, Xb_ho_r)

    m = SCCA_PMD(latent_dimensions=kk, c=[args.c1, args.c2], max_iter=500, tol=1e-6)
    m.fit([Xg_tr_s, Xb_tr_s])
    U_tr, V_tr = m.transform([Xg_tr_s, Xb_tr_s])
    U_ho, V_ho = m.transform([Xg_ho_s, Xb_ho_s])
    Wx, Wy = m.weights

    r_train_final = np.array([np.corrcoef(U_tr[:, i], V_tr[:, i])[0, 1] for i in range(kk)])
    r_holdout = np.array([np.corrcoef(U_ho[:, i], V_ho[:, i])[0, 1] for i in range(kk)])

    results["train_fit"] = {
        "r_train": r_train_final.tolist(),
        "r_holdout": r_holdout.tolist(),
        "sparsity_gene": float((np.abs(Wx) < 1e-3).mean()),
        "sparsity_fmri": float((np.abs(Wy) < 1e-3).mean()),
    }

    # Add interpretable top features per component
    top_gene = {}
    top_roi = {}
    for c in range(kk):
        top_gene[str(c)] = top_feats(Wx[:, c], gene_names, k=min(10, Wx.shape[0]))
        top_roi[str(c)] = top_feats(Wy[:, c], roi_names, k=min(10, Wy.shape[0]))
    results["train_fit"]["top_gene"] = top_gene
    results["train_fit"]["top_roi"] = top_roi

    out_json = Path(args.out_json)
    stem = out_json.stem
    # Save TRAIN projections/weights (holdout kept separate for leakage-proof evaluation)
    np.save(out_json.parent / f"{stem}_U_train.npy", U_tr.astype(np.float32))
    np.save(out_json.parent / f"{stem}_V_train.npy", V_tr.astype(np.float32))
    np.save(out_json.parent / f"{stem}_U_holdout.npy", U_ho.astype(np.float32))
    np.save(out_json.parent / f"{stem}_V_holdout.npy", V_ho.astype(np.float32))
    np.save(out_json.parent / f"{stem}_W_gene.npy", Wx.astype(np.float32))
    np.save(out_json.parent / f"{stem}_W_fmri.npy", Wy.astype(np.float32))
    np.save(out_json.parent / f"{stem}_train_ids.npy", ids[train_idx].astype(object))
    np.save(out_json.parent / f"{stem}_holdout_ids.npy", ids[hold_idx].astype(object))
    results["artifacts"] = {
        "U_train": f"{stem}_U_train.npy",
        "V_train": f"{stem}_V_train.npy",
        "U_holdout": f"{stem}_U_holdout.npy",
        "V_holdout": f"{stem}_V_holdout.npy",
        "W_gene": f"{stem}_W_gene.npy",
        "W_fmri": f"{stem}_W_fmri.npy",
        "train_ids": f"{stem}_train_ids.npy",
        "holdout_ids": f"{stem}_holdout_ids.npy",
    }

    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"[done] saved {args.out_json}")

if __name__ == "__main__":
    main()