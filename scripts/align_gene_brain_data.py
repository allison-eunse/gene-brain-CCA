#!/usr/bin/env python3
"""
Align gene embeddings with brain data by subject IDs.

This script takes:
1. Gene wide matrix (N_gene x D_gene) with corresponding IIDs
2. Brain data (N_brain x D_brain) with corresponding IDs
3. Labels and covariates

And outputs aligned versions where rows match between gene and brain.

Lab server rules:
- Run via Slurm (sbatch/srun), not on login node
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def align_data(
    gene_wide: np.ndarray,
    gene_iids: np.ndarray,
    brain_data: np.ndarray,
    brain_ids: np.ndarray,
    labels: np.ndarray,
    label_iids: np.ndarray,
    cov_age: np.ndarray,
    cov_sex: np.ndarray,
    cov_extra: np.ndarray | None = None,
) -> dict:
    """
    Align gene, brain, and label data by subject IDs.
    Returns aligned arrays and the common IDs.
    """
    # Convert all IDs to strings for consistent comparison
    gene_iids_str = np.array([str(x) for x in gene_iids])
    brain_ids_str = np.array([str(x) for x in brain_ids])
    label_iids_str = np.array([str(x) for x in label_iids])

    # Find common IDs across all three
    common_ids = set(gene_iids_str) & set(brain_ids_str) & set(label_iids_str)
    common_ids = sorted(common_ids)  # Sort for reproducibility

    if len(common_ids) == 0:
        raise ValueError("No common IDs found between gene, brain, and label data")

    print(f"  Gene subjects: {len(gene_iids_str)}")
    print(f"  Brain subjects: {len(brain_ids_str)}")
    print(f"  Label subjects: {len(label_iids_str)}")
    print(f"  Common subjects: {len(common_ids)}")

    # Create ID-to-index maps
    gene_id2idx = {str(x): i for i, x in enumerate(gene_iids_str)}
    brain_id2idx = {str(x): i for i, x in enumerate(brain_ids_str)}
    label_id2idx = {str(x): i for i, x in enumerate(label_iids_str)}

    # Get aligned indices
    gene_indices = np.array([gene_id2idx[x] for x in common_ids])
    brain_indices = np.array([brain_id2idx[x] for x in common_ids])
    label_indices = np.array([label_id2idx[x] for x in common_ids])

    # Align data
    result = {
        "X_gene_wide": gene_wide[gene_indices].astype(np.float32),
        "X_brain": brain_data[brain_indices].astype(np.float32),
        "labels": labels[label_indices],
        "cov_age": cov_age[label_indices].astype(np.float32),
        "cov_sex": cov_sex[label_indices].astype(np.float32),
        "ids_common": np.array(common_ids),
    }

    if cov_extra is not None:
        result["cov_extra"] = cov_extra[brain_indices].astype(np.float32)

    return result


def main():
    ap = argparse.ArgumentParser(description="Align gene and brain data")
    ap.add_argument("--gene-wide", required=True, help="Gene wide matrix .npy")
    ap.add_argument("--gene-iids", required=True, help="Gene subject IIDs .npy")
    ap.add_argument("--brain-data", required=True, help="Brain data .npy")
    ap.add_argument("--brain-ids", required=True, help="Brain subject IDs .npy")
    ap.add_argument("--labels", required=True, help="Labels .npy")
    ap.add_argument("--label-iids", required=True, help="Label subject IIDs .npy")
    ap.add_argument("--cov-age", required=True, help="Age covariate .npy")
    ap.add_argument("--cov-sex", required=True, help="Sex covariate .npy")
    ap.add_argument("--cov-extra", default=None, help="Extra covariate .npy (e.g., eTIV)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--modality-tag", required=True, help="Tag for output naming")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aligning data for {args.modality_tag}...")

    # Load data
    gene_wide = np.load(args.gene_wide, mmap_mode="r")
    gene_iids = np.load(args.gene_iids, allow_pickle=True)
    brain_data = np.load(args.brain_data)
    brain_ids = np.load(args.brain_ids, allow_pickle=True)
    labels = np.load(args.labels)
    label_iids = np.load(args.label_iids, allow_pickle=True)
    cov_age = np.load(args.cov_age)
    cov_sex = np.load(args.cov_sex)
    cov_extra = np.load(args.cov_extra) if args.cov_extra else None

    # Align
    result = align_data(
        gene_wide=gene_wide,
        gene_iids=gene_iids,
        brain_data=brain_data,
        brain_ids=brain_ids,
        labels=labels,
        label_iids=label_iids,
        cov_age=cov_age,
        cov_sex=cov_sex,
        cov_extra=cov_extra,
    )

    # Save aligned data
    np.save(out_dir / "X_gene_wide.npy", result["X_gene_wide"])
    np.save(out_dir / "X_brain.npy", result["X_brain"])
    np.save(out_dir / "labels.npy", result["labels"])
    np.save(out_dir / "cov_age.npy", result["cov_age"])
    np.save(out_dir / "cov_sex.npy", result["cov_sex"])
    np.save(out_dir / "ids_common.npy", result["ids_common"])

    if "cov_extra" in result:
        np.save(out_dir / "cov_extra.npy", result["cov_extra"])

    # Save metadata
    n_mdd = int(np.sum(result["labels"] == 1))
    n_ctrl = int(np.sum(result["labels"] == 0))
    meta = {
        "modality": args.modality_tag,
        "n_subjects": len(result["ids_common"]),
        "n_mdd": n_mdd,
        "n_ctrl": n_ctrl,
        "gene_dim": result["X_gene_wide"].shape[1],
        "brain_dim": result["X_brain"].shape[1],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved aligned data to {out_dir}")
    print(f"  N={meta['n_subjects']} (MDD={n_mdd}, Ctrl={n_ctrl})")
    print(f"  Gene: {result['X_gene_wide'].shape}")
    print(f"  Brain: {result['X_brain'].shape}")


if __name__ == "__main__":
    main()
