#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

from sklearn.cross_decomposition import CCA

# Use local standalone SCCA_PMD implementation (avoids cca-zoo version issues)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from scca_pmd import SCCA_PMD

def auc_ap(y, p):
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
    }

def fit_predict(model, X, y, tr, va):
    clf = model
    clf.fit(X[tr], y[tr])
    p = clf.predict_proba(X[va])[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X[va])
    return p

def run_cv(X, y, model_builder, n_folds, seed):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        clf = model_builder()
        preds[va] = fit_predict(clf, X, y, tr, va)
    return preds

def model_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"))

def model_mlp():
    # Note: early_stopping uses an internal split of the TRAIN fold only, which is fine.
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
    )

def _build_cov(age: np.ndarray, sex: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(age), dtype=np.float64),
            age.astype(np.float64),
            sex.astype(np.float64),
        ]
        + ([] if extra is None else [_as_2d(extra).astype(np.float64)])
    )

def _as_2d(x: np.ndarray | None) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"cov_extra must be 1D or 2D, got shape={x.shape}")

def _residualize_train_test(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    age_tr: np.ndarray,
    sex_tr: np.ndarray,
    age_te: np.ndarray,
    sex_te: np.ndarray,
    extra_tr: np.ndarray | None = None,
    extra_te: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit residualization (intercept+age+sex+extra) on TRAIN only; apply to train and test.
    """
    C_tr = _build_cov(age_tr, sex_tr, extra_tr)
    CtC = C_tr.T @ C_tr
    CtX = C_tr.T @ X_tr.astype(np.float64, copy=False)
    B = np.linalg.solve(CtC, CtX)
    X_tr_r = X_tr.astype(np.float64, copy=False) - (C_tr @ B)

    C_te = _build_cov(age_te, sex_te, extra_te)
    X_te_r = X_te.astype(np.float64, copy=False) - (C_te @ B)

    return X_tr_r.astype(np.float32, copy=False), X_te_r.astype(np.float32, copy=False)


def _standardize_train_val(X_tr: np.ndarray, X_va: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    return X_tr_s, X_va_s

def _embed_cca_fold(
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_va: np.ndarray,
    Xb_va: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit CCA on TRAIN only; return (Z_train, Z_val) where Z = [U,V].
    Uses train-fitted standardization for Stage 1.
    """

    Xg_tr_s, Xg_va_s = _standardize_train_val(Xg_tr, Xg_va)
    Xb_tr_s, Xb_va_s = _standardize_train_val(Xb_tr, Xb_va)

    cca = CCA(n_components=k, max_iter=5000)
    cca.fit(Xg_tr_s, Xb_tr_s)
    # Compute canonical variates via learned weights to avoid sklearn transform API ambiguity.
    U_tr = Xg_tr_s @ cca.x_weights_
    V_tr = Xb_tr_s @ cca.y_weights_
    U_va = Xg_va_s @ cca.x_weights_
    V_va = Xb_va_s @ cca.y_weights_

    Z_tr = np.hstack([U_tr, V_tr])
    Z_va = np.hstack([U_va, V_va])
    return Z_tr, Z_va


def _embed_scca_fold(
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_va: np.ndarray,
    Xb_va: np.ndarray,
    k: int,
    c1: float,
    c2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit SCCA on TRAIN only; return (Z_train, Z_val) where Z = [U,V].
    Uses train-fitted standardization for Stage 1.
    """

    Xg_tr_s, Xg_va_s = _standardize_train_val(Xg_tr, Xg_va)
    Xb_tr_s, Xb_va_s = _standardize_train_val(Xb_tr, Xb_va)

    m = SCCA_PMD(latent_dimensions=k, c=[c1, c2])
    m.fit([Xg_tr_s, Xb_tr_s])
    U_tr, V_tr = m.transform([Xg_tr_s, Xb_tr_s])
    U_va, V_va = m.transform([Xg_va_s, Xb_va_s])
    Z_tr = np.hstack([U_tr, V_tr])
    Z_va = np.hstack([U_va, V_va])
    return Z_tr, Z_va


def run_cv_stage1_stage2(
    Xg: np.ndarray,
    Xb: np.ndarray,
    y: np.ndarray,
    stage1: str,
    stage2_builder,
    n_folds: int,
    seed: int,
    k: int,
    c1: float,
    c2: float,
) -> np.ndarray:
    """
    Leakage-safe CV: fit Stage 1 (CCA/SCCA) on TRAIN folds only, transform train/val,
    then fit Stage 2 predictor on train embeddings and predict on val.
    Returns out-of-fold predicted probabilities for y==1.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(Xg, y):
        Xg_tr, Xg_va = Xg[tr], Xg[va]
        Xb_tr, Xb_va = Xb[tr], Xb[va]

        kk = min(k, Xg_tr.shape[1], Xb_tr.shape[1])
        if stage1 == "cca":
            Z_tr, Z_va = _embed_cca_fold(Xg_tr, Xb_tr, Xg_va, Xb_va, kk)
        elif stage1 == "scca":
            Z_tr, Z_va = _embed_scca_fold(Xg_tr, Xb_tr, Xg_va, Xb_va, kk, c1, c2)
        else:
            raise ValueError(f"Unknown stage1: {stage1}")

        clf = stage2_builder()
        clf.fit(Z_tr, y[tr])
        if hasattr(clf, "predict_proba"):
            oof[va] = clf.predict_proba(Z_va)[:, 1]
        else:
            oof[va] = clf.decision_function(Z_va)
    return oof

def _embed_cca_train_test(
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_te: np.ndarray,
    Xb_te: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only Stage1 fit; return (Z_train, Z_test) where Z=[U,V]."""
    Z_tr, Z_te = _embed_cca_fold(Xg_tr, Xb_tr, Xg_te, Xb_te, k)
    return Z_tr, Z_te


def _embed_scca_train_test(
    Xg_tr: np.ndarray,
    Xb_tr: np.ndarray,
    Xg_te: np.ndarray,
    Xb_te: np.ndarray,
    k: int,
    c1: float,
    c2: float,
) -> tuple[np.ndarray, np.ndarray]:
    Z_tr, Z_te = _embed_scca_fold(Xg_tr, Xb_tr, Xg_te, Xb_te, k, c1, c2)
    return Z_tr, Z_te


def _fit_gene_pca_train_only(
    Xg_tr: np.ndarray,
    Xg_te: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fit PCA on TRAIN only; transform train and test.
    """
    k = int(min(n_components, Xg_tr.shape[0], Xg_tr.shape[1]))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    Xg_tr_p = pca.fit_transform(Xg_tr)
    Xg_te_p = pca.transform(Xg_te)
    meta = {
        "n_components": int(k),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return Xg_tr_p.astype(np.float32), Xg_te_p.astype(np.float32), meta


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def main():
    ap = argparse.ArgumentParser()
    # Preferred (leakage-safe) inputs:
    ap.add_argument("--x-gene-wide", default=None, help="Gene-wide matrix (N×(111*768))")
    ap.add_argument("--x-fmri-raw", default=None, help="Raw aligned fMRI matrix (N×D)")
    ap.add_argument("--cov-age", default=None, help="Aligned age covariate (N,)")
    ap.add_argument("--cov-sex", default=None, help="Aligned sex covariate (N,)")
    ap.add_argument("--cov-extra", default=None, help="Optional extra covariate (e.g., eTIV) (N,)")
    ap.add_argument("--ids", default=None, help="Optional ids_common.npy to save train/holdout IDs")

    # Legacy (not fully leakage-proof unless these were built train-only):
    ap.add_argument("--x-gene", default=None, help="Precomputed gene features (e.g., PCA512)")
    ap.add_argument("--x-fmri", default=None, help="Precomputed fMRI features")

    ap.add_argument("--labels", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--c1", type=float, default=0.3)
    ap.add_argument("--c2", type=float, default=0.3)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--holdout-frac", type=float, default=0.2, help="Holdout fraction for one-touch test set")
    ap.add_argument("--pca-components", type=int, default=512, help="Gene PCA components (train-only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    out = Path(args.out_json)

    y = np.load(args.labels).astype(np.int64).ravel()

    # -------------------------
    # 1) Stratified holdout split
    # -------------------------
    if not (0.0 < args.holdout_frac < 1.0):
        raise SystemExit("[ERROR] --holdout-frac must be in (0,1)")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.holdout_frac, random_state=args.seed)
    train_idx, hold_idx = next(sss.split(np.zeros(len(y)), y))

    ids = None
    if args.ids:
        ids = np.load(args.ids, allow_pickle=True).astype(str)
        if ids.shape[0] != len(y):
            raise SystemExit(f"[ERROR] ids len {ids.shape[0]} != labels len {len(y)}")

    # -------------------------
    # 2) Build leakage-safe features (train-only residualization + train-only PCA)
    # -------------------------
    pca_meta = None
    warnings = []

    if args.x_gene_wide and args.x_fmri_raw and args.cov_age and args.cov_sex:
        Xg_wide = np.load(args.x_gene_wide, mmap_mode="r")
        Xb_raw = np.load(args.x_fmri_raw, mmap_mode="r")
        age = np.load(args.cov_age).astype(np.float32)
        sex = np.load(args.cov_sex).astype(np.float32)
        extra = np.load(args.cov_extra).astype(np.float32) if args.cov_extra else None

        if Xg_wide.shape[0] != len(y) or Xb_raw.shape[0] != len(y):
            raise SystemExit(
                f"[ERROR] Row mismatch: Xg_wide={Xg_wide.shape}, Xb_raw={Xb_raw.shape}, y={y.shape}"
            )

        # Residualize BOTH modalities using train-only betas
        Xb_tr, Xb_ho = _residualize_train_test(
            np.asarray(Xb_raw[train_idx], dtype=np.float32),
            np.asarray(Xb_raw[hold_idx], dtype=np.float32),
            age[train_idx],
            sex[train_idx],
            age[hold_idx],
            sex[hold_idx],
            extra_tr=extra[train_idx] if extra is not None else None,
            extra_te=extra[hold_idx] if extra is not None else None,
        )
        Xg_tr_r, Xg_ho_r = _residualize_train_test(
            np.asarray(Xg_wide[train_idx], dtype=np.float32),
            np.asarray(Xg_wide[hold_idx], dtype=np.float32),
            age[train_idx],
            sex[train_idx],
            age[hold_idx],
            sex[hold_idx],
            extra_tr=extra[train_idx] if extra is not None else None,
            extra_te=extra[hold_idx] if extra is not None else None,
        )

        # Train-only PCA on gene-wide
        Xg_tr, Xg_ho, pca_meta = _fit_gene_pca_train_only(
            Xg_tr_r, Xg_ho_r, n_components=args.pca_components, seed=args.seed
        )
    elif args.x_gene and args.x_fmri:
        # Legacy: caller supplied precomputed features.
        warnings.append(
            "Using --x-gene/--x-fmri legacy inputs. Ensure PCA/residualization were fit train-only to avoid leakage."
        )
        Xg_all = np.load(args.x_gene, mmap_mode="r")
        Xb_all = np.load(args.x_fmri, mmap_mode="r")
        Xg_tr = np.asarray(Xg_all[train_idx])
        Xg_ho = np.asarray(Xg_all[hold_idx])
        Xb_tr = np.asarray(Xb_all[train_idx])
        Xb_ho = np.asarray(Xb_all[hold_idx])
    else:
        raise SystemExit(
            "[ERROR] Provide leakage-safe inputs (--x-gene-wide, --x-fmri-raw, --cov-age, --cov-sex) "
            "or legacy (--x-gene, --x-fmri)."
        )

    y_tr = y[train_idx]
    y_ho = y[hold_idx]

    results = {
        "split": {
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "pos_ratio_total": float(np.mean(y)),
            "pos_ratio_train": float(np.mean(y_tr)),
            "pos_ratio_holdout": float(np.mean(y_ho)),
            "seed": int(args.seed),
            "holdout_frac": float(args.holdout_frac),
        },
        "pca": pca_meta,
        "warnings": warnings,
        "cv": {},
        "holdout": {},
    }
    if ids is not None:
        stem = out.stem
        np.save(out.parent / f"{stem}_train_ids.npy", ids[train_idx].astype(object))
        np.save(out.parent / f"{stem}_holdout_ids.npy", ids[hold_idx].astype(object))
        results["artifacts"] = {
            "train_ids": f"{stem}_train_ids.npy",
            "holdout_ids": f"{stem}_holdout_ids.npy",
        }

    # -------------------------
    # 3) CV on TRAIN for tuning (Stage1 is fold-wise)
    # -------------------------
    for name, X in [
        ("gene_only", Xg_tr),
        ("fmri_only", Xb_tr),
        ("early_fusion", np.hstack([Xg_tr, Xb_tr])),
    ]:
        preds = run_cv(X, y_tr, model_logreg, args.n_folds, args.seed)
        results["cv"][f"{name}_logreg"] = _metrics(y_tr, preds)
        preds = run_cv(X, y_tr, model_mlp, args.n_folds, args.seed)
        results["cv"][f"{name}_mlp"] = _metrics(y_tr, preds)

    preds = run_cv_stage1_stage2(
        Xg_tr, Xb_tr, y_tr, "cca", model_logreg, args.n_folds, args.seed, args.k, args.c1, args.c2
    )
    results["cv"]["cca_joint_logreg"] = _metrics(y_tr, preds)
    preds = run_cv_stage1_stage2(
        Xg_tr, Xb_tr, y_tr, "cca", model_mlp, args.n_folds, args.seed, args.k, args.c1, args.c2
    )
    results["cv"]["cca_joint_mlp"] = _metrics(y_tr, preds)
    preds = run_cv_stage1_stage2(
        Xg_tr, Xb_tr, y_tr, "scca", model_logreg, args.n_folds, args.seed, args.k, args.c1, args.c2
    )
    results["cv"]["scca_joint_logreg"] = _metrics(y_tr, preds)
    preds = run_cv_stage1_stage2(
        Xg_tr, Xb_tr, y_tr, "scca", model_mlp, args.n_folds, args.seed, args.k, args.c1, args.c2
    )
    results["cv"]["scca_joint_mlp"] = _metrics(y_tr, preds)

    # -------------------------
    # 4) One-touch HOLDOUT evaluation (final claim)
    # -------------------------
    for name, Xtr, Xte in [
        ("gene_only", Xg_tr, Xg_ho),
        ("fmri_only", Xb_tr, Xb_ho),
        ("early_fusion", np.hstack([Xg_tr, Xb_tr]), np.hstack([Xg_ho, Xb_ho])),
    ]:
        clf = model_logreg()
        clf.fit(Xtr, y_tr)
        p = clf.predict_proba(Xte)[:, 1]
        results["holdout"][f"{name}_logreg"] = _metrics(y_ho, p)

        clf = model_mlp()
        clf.fit(Xtr, y_tr)
        p = clf.predict_proba(Xte)[:, 1]
        results["holdout"][f"{name}_mlp"] = _metrics(y_ho, p)

    # CCA holdout
    kk = min(args.k, Xg_tr.shape[1], Xb_tr.shape[1])
    Z_tr, Z_ho = _embed_cca_train_test(Xg_tr, Xb_tr, Xg_ho, Xb_ho, kk)
    clf = model_logreg()
    clf.fit(Z_tr, y_tr)
    p = clf.predict_proba(Z_ho)[:, 1]
    results["holdout"]["cca_joint_logreg"] = _metrics(y_ho, p)
    clf = model_mlp()
    clf.fit(Z_tr, y_tr)
    p = clf.predict_proba(Z_ho)[:, 1]
    results["holdout"]["cca_joint_mlp"] = _metrics(y_ho, p)

    # SCCA holdout
    Z_tr, Z_ho = _embed_scca_train_test(Xg_tr, Xb_tr, Xg_ho, Xb_ho, kk, args.c1, args.c2)
    clf = model_logreg()
    clf.fit(Z_tr, y_tr)
    p = clf.predict_proba(Z_ho)[:, 1]
    results["holdout"]["scca_joint_logreg"] = _metrics(y_ho, p)
    clf = model_mlp()
    clf.fit(Z_tr, y_tr)
    p = clf.predict_proba(Z_ho)[:, 1]
    results["holdout"]["scca_joint_mlp"] = _metrics(y_ho, p)

    out.write_text(json.dumps(results, indent=2))
    print(f"[done] {out}")

if __name__ == "__main__":
    main()