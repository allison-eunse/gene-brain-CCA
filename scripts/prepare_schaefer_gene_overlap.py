#!/usr/bin/env python3
"""
Prepare aligned Schaefer fMRI + gene embedding overlap arrays for stratified CCA.

This script aligns:
- Schaefer 7/17 fMRI summary features
- Wide gene embeddings (111 genes x 768 dims)
- Labels, covariates

Outputs separate directories for each Schaefer parcellation with all arrays aligned.

Lab server rules:
- Run via Slurm (sbatch/srun), not on login node for heavy I/O
- No manual CUDA_VISIBLE_DEVICES
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def to_str(x: np.ndarray) -> np.ndarray:
    """Convert array to string dtype."""
    return np.asarray(x).astype(str)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare Schaefer fMRI + gene overlap")
    ap.add_argument("--schaefer-fmri", required=True, help="Path to X_fmri_schaefer*.npy")
    ap.add_argument("--schaefer-meta", required=True, help="Path to meta_schaefer*_summary.json")
    ap.add_argument("--gene-wide", required=True, help="Path to X_gene_wide.npy")
    ap.add_argument("--gene-ids", required=True, help="Path to ids for gene embeddings")
    ap.add_argument("--labels", required=True, help="Path to labels_common.npy")
    ap.add_argument("--cov-age", required=True, help="Path to cov_age.npy")
    ap.add_argument("--cov-sex", required=True, help="Path to cov_sex.npy")
    ap.add_argument("--ids-common", required=True, help="Path to ids_common.npy (reference order)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(args.schaefer_meta) as f:
        meta = json.load(f)
    print(f"[load] Schaefer meta: tag={meta['tag']}, n_subjects={meta['n_subjects']}, features={meta['feature_length']}")

    # Load reference IDs (defines subject order)
    ids_common = to_str(np.load(args.ids_common, allow_pickle=True))
    print(f"[load] Reference IDs: {len(ids_common)}")

    # Load labels and covariates (aligned to ids_common)
    labels = np.load(args.labels)
    cov_age = np.load(args.cov_age)
    cov_sex = np.load(args.cov_sex)

    if len(labels) != len(ids_common):
        raise ValueError(f"Labels length {len(labels)} != ids_common length {len(ids_common)}")

    # Load gene embeddings
    X_gene = np.load(args.gene_wide, mmap_mode="r")
    ids_gene = to_str(np.load(args.gene_ids, allow_pickle=True))
    print(f"[load] Gene embeddings: {X_gene.shape}, IDs: {len(ids_gene)}")

    # Load Schaefer fMRI
    X_fmri = np.load(args.schaefer_fmri, mmap_mode="r")
    print(f"[load] Schaefer fMRI: {X_fmri.shape}")

    # Schaefer fMRI should already be aligned to ids_common (same N=4218)
    if X_fmri.shape[0] != len(ids_common):
        raise ValueError(f"Schaefer fMRI rows {X_fmri.shape[0]} != ids_common {len(ids_common)}")

    # Gene embeddings should also be aligned to ids_common
    if len(ids_gene) != len(ids_common):
        print(f"[warn] Gene IDs length {len(ids_gene)} != ids_common {len(ids_common)}, will align")
        # Build index map
        gene_id_to_idx = {sid: i for i, sid in enumerate(ids_gene)}
        keep_idx = []
        for sid in ids_common:
            if sid in gene_id_to_idx:
                keep_idx.append(gene_id_to_idx[sid])
            else:
                raise ValueError(f"ID {sid} in ids_common not found in gene IDs")
        X_gene = np.array(X_gene[keep_idx])
    else:
        # Check if they're in the same order
        if not np.array_equal(ids_gene, ids_common):
            print("[align] Reordering gene embeddings to match ids_common")
            gene_id_to_idx = {sid: i for i, sid in enumerate(ids_gene)}
            keep_idx = [gene_id_to_idx[sid] for sid in ids_common]
            X_gene = np.array(X_gene[keep_idx])
        else:
            X_gene = np.array(X_gene)

    # Materialize fMRI
    X_fmri = np.array(X_fmri, dtype=np.float32)
    X_gene = np.array(X_gene, dtype=np.float32)

    print(f"[aligned] X_fmri: {X_fmri.shape}, X_gene: {X_gene.shape}")
    print(f"[aligned] Labels: {len(labels)} (MDD={np.sum(labels == 1)}, Ctrl={np.sum(labels == 0)})")

    # Save aligned arrays
    np.save(out_dir / "X_fmri.npy", X_fmri)
    np.save(out_dir / "X_gene_wide.npy", X_gene)
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "cov_age.npy", cov_age)
    np.save(out_dir / "cov_sex.npy", cov_sex)
    np.save(out_dir / "ids.npy", ids_common)

    # Save metadata
    out_meta = {
        "tag": meta["tag"],
        "n_subjects": len(ids_common),
        "n_mdd": int(np.sum(labels == 1)),
        "n_ctrl": int(np.sum(labels == 0)),
        "fmri_dim": int(X_fmri.shape[1]),
        "gene_dim": int(X_gene.shape[1]),
        "source_schaefer_meta": args.schaefer_meta,
        "source_gene_wide": args.gene_wide,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(out_meta, f, indent=2)

    print(f"[save] {out_dir}")
    print(f"  X_fmri.npy: {X_fmri.shape}")
    print(f"  X_gene_wide.npy: {X_gene.shape}")
    print(f"  labels.npy: {labels.shape} (MDD={out_meta['n_mdd']}, Ctrl={out_meta['n_ctrl']})")
    print("[done]")


if __name__ == "__main__":
    main()
