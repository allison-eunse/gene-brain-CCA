#!/usr/bin/env python3
"""
Build wide gene embedding matrices for each gene foundation model.

This script loads per-gene, per-batch embedding files and concatenates them
into a single (n_subjects, n_genes * embedding_dim) matrix for CCA/SCCA.

Gene FM models supported:
- DNABERT2_embedding
- Evo2_embedding
- HyenaDNA_embedding
- Caduceus_embedding

Lab server rules:
- Run via Slurm (sbatch/srun), not on login node
- No manual CUDA_VISIBLE_DEVICES
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

# Embedding directory patterns for each model
FM_PATTERNS = {
    "dnabert2": {
        "dir": "DNABERT2_embedding",
        "pattern": r"embeddings_(\d+)_layer_last\.npy",
    },
    "evo2": {
        "dir": "Evo2_embedding",
        "pattern": r"embeddings_(\d+)_layer_blocks_21_mlp_l3\.npy",
    },
    "hyenadna": {
        "dir": "HyenaDNA_embedding",
        "pattern": r"embeddings_(\d+)_layer-1\.npy",
    },
    "caduceus": {
        "dir": "Caduceus_embedding",
        "pattern": r"embeddings_(\d+)\.npy",
    },
}


def load_gene_embedding(gene_dir: Path, pattern: str) -> np.ndarray:
    """Load and concatenate all batch files for a gene."""
    files = list(gene_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files in {gene_dir}")

    # Parse batch indices and sort
    batches = []
    for f in files:
        match = re.match(pattern, f.name)
        if match:
            batch_idx = int(match.group(1))
            batches.append((batch_idx, f))

    if not batches:
        raise ValueError(f"No files matching pattern {pattern} in {gene_dir}")

    batches.sort(key=lambda x: x[0])

    # Load and concatenate
    arrays = [np.load(f) for _, f in batches]
    return np.concatenate(arrays, axis=0)


def build_wide_matrix(
    embedding_root: Path,
    fm_name: str,
    gene_list: list[str],
    out_dir: Path,
) -> dict:
    """Build wide gene matrix for a single FM model."""
    cfg = FM_PATTERNS[fm_name]
    fm_dir = embedding_root / cfg["dir"]
    pattern = cfg["pattern"]

    if not fm_dir.exists():
        raise FileNotFoundError(f"FM directory not found: {fm_dir}")

    print(f"Building wide matrix for {fm_name}...")
    print(f"  Source: {fm_dir}")
    print(f"  Genes: {len(gene_list)}")

    gene_embeddings = []
    valid_genes = []
    embedding_dims = []

    for gene in gene_list:
        gene_dir = fm_dir / gene
        if not gene_dir.exists():
            print(f"  [warn] Gene {gene} not found, skipping")
            continue

        try:
            emb = load_gene_embedding(gene_dir, pattern)
            gene_embeddings.append(emb)
            valid_genes.append(gene)
            embedding_dims.append(emb.shape[1])
        except Exception as e:
            print(f"  [warn] Error loading {gene}: {e}")
            continue

    if not gene_embeddings:
        raise ValueError("No valid gene embeddings loaded")

    # Check all have same n_subjects
    n_subjects = gene_embeddings[0].shape[0]
    for i, (g, e) in enumerate(zip(valid_genes, gene_embeddings)):
        if e.shape[0] != n_subjects:
            raise ValueError(f"Gene {g} has {e.shape[0]} subjects, expected {n_subjects}")

    # Concatenate horizontally
    X_wide = np.concatenate(gene_embeddings, axis=1).astype(np.float32)
    total_dim = X_wide.shape[1]

    print(f"  Result: {X_wide.shape} ({n_subjects} subjects, {total_dim} features)")
    print(f"  Valid genes: {len(valid_genes)}/{len(gene_list)}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_gene_wide_{fm_name}.npy", X_wide)

    meta = {
        "fm_model": fm_name,
        "n_subjects": n_subjects,
        "n_genes": len(valid_genes),
        "total_features": total_dim,
        "embedding_dim": embedding_dims[0] if len(set(embedding_dims)) == 1 else embedding_dims,
        "genes": valid_genes,
    }
    with open(out_dir / f"meta_gene_wide_{fm_name}.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    ap = argparse.ArgumentParser(description="Build wide gene embedding matrices")
    ap.add_argument(
        "--embedding-root",
        default="/storage/bigdata/NESAP",
        help="Root directory containing FM embedding folders",
    )
    ap.add_argument(
        "--gene-list",
        default="/storage/bigdata/NESAP/gene_list_filtered.txt",
        help="File with list of gene names",
    )
    ap.add_argument(
        "--out-dir",
        default="/storage/bigdata/UKB/fMRI/gene-brain-CCA/derived_gene_wide",
        help="Output directory for wide matrices",
    )
    ap.add_argument(
        "--fm-models",
        default="dnabert2,evo2,hyenadna,caduceus",
        help="Comma-separated list of FM models to process",
    )
    args = ap.parse_args()

    embedding_root = Path(args.embedding_root)
    out_dir = Path(args.out_dir)

    # Load gene list
    with open(args.gene_list) as f:
        gene_list = [line.strip() for line in f if line.strip()]
    print(f"Gene list: {len(gene_list)} genes")

    fm_models = [m.strip() for m in args.fm_models.split(",")]
    print(f"FM models: {fm_models}")

    results = {}
    for fm in fm_models:
        if fm not in FM_PATTERNS:
            print(f"[warn] Unknown FM model: {fm}, skipping")
            continue
        try:
            meta = build_wide_matrix(embedding_root, fm, gene_list, out_dir)
            results[fm] = meta
        except Exception as e:
            print(f"[error] Failed to build {fm}: {e}")
            results[fm] = {"error": str(e)}

    # Save summary
    with open(out_dir / "build_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nDone! Summary:")
    for fm, meta in results.items():
        if "error" in meta:
            print(f"  {fm}: ERROR - {meta['error']}")
        else:
            print(f"  {fm}: {meta['n_subjects']} x {meta['total_features']}")


if __name__ == "__main__":
    main()
