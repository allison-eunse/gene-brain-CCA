#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def to_str(x):
    return np.asarray(x).astype(str)


def load_feature_map(csv_path, id_col, target_ids, chunksize):
    header = pd.read_csv(csv_path, nrows=0)
    if id_col not in header.columns:
        raise SystemExit(f"[ERROR] {csv_path} missing id column: {id_col}")
    feat_cols = [c for c in header.columns if c != id_col]
    if not feat_cols:
        raise SystemExit(f"[ERROR] {csv_path} has no feature columns.")

    feat_map = {}
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype={id_col: str},
        low_memory=False,
    ):
        if id_col not in chunk.columns:
            raise SystemExit(f"[ERROR] {csv_path} chunk missing id column: {id_col}")
        sub = chunk[chunk[id_col].isin(target_ids)]
        if sub.empty:
            continue
        ids = sub[id_col].astype(str).to_numpy()
        vals = sub[feat_cols].to_numpy(dtype=np.float32, copy=True)
        for i, sid in enumerate(ids):
            if sid not in feat_map:
                feat_map[sid] = vals[i]
    return feat_map, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-gene", required=True, help="npy of IIDs (string-like)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--cov-valid-mask", default=None)
    ap.add_argument("--csv-a", required=True, help="First CSV to include (e.g., a2009s/skeleton)")
    ap.add_argument("--csv-b", required=True, help="Second CSV to include (e.g., aseg/weighted)")
    ap.add_argument("--id-col", default="participant.eid")
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--modality", required=True, choices=["smri", "dmri"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--save-z", action="store_true", help="Also save residualized + z-scored X")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ids_gene = to_str(np.load(args.ids_gene, allow_pickle=True))
    labels = np.load(args.labels)
    age = np.load(args.cov_age)
    sex = np.load(args.cov_sex)
    if args.cov_valid_mask:
        vmask = np.load(args.cov_valid_mask).astype(bool)
    else:
        vmask = np.ones_like(ids_gene, dtype=bool)

    if not (ids_gene.shape[0] == labels.shape[0] == age.shape[0] == sex.shape[0] == vmask.shape[0]):
        raise SystemExit(
            "[ERROR] Covariate/label length mismatch: "
            f"ids={ids_gene.shape}, labels={labels.shape}, age={age.shape}, "
            f"sex={sex.shape}, vmask={vmask.shape}"
        )

    cov_idx = {sid: i for i, sid in enumerate(ids_gene) if vmask[i]}
    target_ids = set(cov_idx.keys())

    map_a, cols_a = load_feature_map(args.csv_a, args.id_col, target_ids, args.chunksize)
    map_b, cols_b = load_feature_map(args.csv_b, args.id_col, target_ids, args.chunksize)

    common = sorted(set(map_a.keys()) & set(map_b.keys()) & set(cov_idx.keys()))
    if not common:
        raise SystemExit("[ERROR] No overlapping IDs found between gene/covariates and tabular CSVs.")

    kept = []
    X_rows = []
    cov_age = []
    cov_sex = []
    y = []
    for sid in common:
        i = cov_idx.get(sid)
        if i is None:
            continue
        kept.append(sid)
        cov_age.append(float(age[i]))
        cov_sex.append(float(sex[i]))
        y.append(labels[i])
        X_rows.append(np.concatenate([map_a[sid], map_b[sid]]))

    kept = np.array(kept, dtype=object)
    X = np.asarray(X_rows, dtype=np.float32)
    cov_age = np.asarray(cov_age, dtype=np.float32)
    cov_sex = np.asarray(cov_sex, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    np.save(out / "ids_common.npy", kept)
    np.save(out / "labels_common.npy", y)
    np.save(out / "cov_age.npy", cov_age)
    np.save(out / "cov_sex.npy", cov_sex)
    np.save(out / f"X_{args.modality}_raw.npy", X)
    cols_all = [args.id_col] + cols_a + cols_b
    (out / f"{args.modality}_columns.txt").write_text("\n".join(cols_all))

    if args.save_z:
        A = np.column_stack([np.ones(len(kept), dtype=np.float64), cov_age, cov_sex]).astype(np.float64)

        def resid(Xin, C):
            B, *_ = np.linalg.lstsq(C, Xin, rcond=None)
            return Xin - C @ B

        def zscore(Xin):
            mu = Xin.mean(0, keepdims=True)
            sd = Xin.std(0, keepdims=True)
            sd = np.where(sd == 0, 1, sd)
            return (Xin - mu) / sd

        X_r = resid(X, A)
        X_z = zscore(X_r).astype(np.float32)
        np.save(out / f"X_{args.modality}_z.npy", X_z)

    print(
        f"[save] n={len(kept)}, {args.modality}_dim={X.shape[1]} "
        f"(csv_a={len(cols_a)}, csv_b={len(cols_b)})",
        flush=True,
    )


if __name__ == "__main__":
    main()
