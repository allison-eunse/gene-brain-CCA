#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path

def to_str(x): return np.asarray(x).astype(str)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-gene", required=True)
    ap.add_argument("--x-gene", required=True)
    ap.add_argument("--ids-fmri", required=True)
    ap.add_argument("--x-fmri", required=True)
    ap.add_argument("--cov-iids", required=True)
    ap.add_argument("--cov-age", required=True)
    ap.add_argument("--cov-sex", required=True)
    ap.add_argument("--cov-valid-mask", default=None)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ids_gene = to_str(np.load(args.ids_gene, allow_pickle=True))
    Xg = np.load(args.x_gene, mmap_mode="r")
    ids_fmri = to_str(np.load(args.ids_fmri, allow_pickle=True))
    Xb = np.load(args.x_fmri, mmap_mode="r")
    cov_iids = to_str(np.load(args.cov_iids, allow_pickle=True))
    age = np.load(args.cov_age); sex = np.load(args.cov_sex)
    if args.cov_valid_mask:
        vmask = np.load(args.cov_valid_mask).astype(bool)
    else:
        vmask = np.ones_like(cov_iids, dtype=bool)
    labels = np.load(args.labels)

    if Xg.shape[0] != ids_gene.shape[0]:
        raise SystemExit(f"[ERROR] gene rows mismatch: Xg {Xg.shape} vs ids_gene {ids_gene.shape}")
    if Xb.shape[0] != ids_fmri.shape[0]:
        raise SystemExit(f"[ERROR] fmri rows mismatch: Xb {Xb.shape} vs ids_fmri {ids_fmri.shape}")

    # Fast ID -> row lookup (keep first occurrence if duplicates exist)
    idx_gene = {}
    for i, sid in enumerate(ids_gene):
        if sid not in idx_gene:
            idx_gene[sid] = i
    idx_fmri = {}
    for i, sid in enumerate(ids_fmri):
        if sid not in idx_fmri:
            idx_fmri[sid] = i

    cov_idx = {sid: i for i, sid in enumerate(cov_iids) if vmask[i]}
    common = sorted(set(idx_gene.keys()) & set(idx_fmri.keys()))
    a,s,l,ig,ib,kept = [],[],[],[],[],[]
    for sid in common:
        i = cov_idx.get(sid)
        if i is None: continue
        a.append(float(age[i])); s.append(float(sex[i])); l.append(labels[i])
        ig.append(idx_gene[sid])
        ib.append(idx_fmri[sid])
        kept.append(sid)
    kept = np.array(kept, dtype=object)
    Xg = np.asarray(Xg[np.array(ig)], dtype=np.float32)  # (N, G_gene)
    Xb = np.asarray(Xb[np.array(ib)], dtype=np.float32)  # (N, D_fmri)
    age_c = np.asarray(a, dtype=np.float32)
    sex_c = np.asarray(s, dtype=np.float32)
    y_c = np.asarray(l, dtype=np.int64)

    # Save RAW aligned arrays + covariates (for leakage-safe downstream splits)
    np.save(out/"X_gene_raw.npy", Xg)
    np.save(out/"X_fmri_raw.npy", Xb)
    np.save(out/"cov_age.npy", age_c)
    np.save(out/"cov_sex.npy", sex_c)

    A = np.column_stack([np.ones(len(kept), dtype=np.float64), age_c, sex_c]).astype(np.float64)

    # residualize
    def resid(X, C):
        B, *_ = np.linalg.lstsq(C, X, rcond=None)
        return X - C @ B
    Xg_r = resid(Xg, A); Xb_r = resid(Xb, A)

    # z-score
    def zscore(X):
        mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True); sd = np.where(sd==0,1,sd)
        return (X-mu)/sd
    Xg_z = zscore(Xg_r).astype(np.float32)
    Xb_z = zscore(Xb_r).astype(np.float32)

    np.save(out/"ids_common.npy", kept)
    np.save(out/"X_gene_z.npy", Xg_z)
    np.save(out/"X_fmri_z.npy", Xb_z)
    np.save(out/"labels_common.npy", y_c)
    print(
        f"[save] n={len(kept)}, gene_dim={Xg_z.shape[1]}, fmri_dim={Xb_z.shape[1]} (raw+cov saved)",
        flush=True,
    )

if __name__ == "__main__":
    main()