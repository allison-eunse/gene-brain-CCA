#!/usr/bin/env python3
"""
Analyze Evo2 gene weights to identify top contributing genes.
"""

import numpy as np
import json
from pathlib import Path

# Load gene names
gene_list_path = Path("/storage/bigdata/NESAP/gene_list_filtered.txt")
with open(gene_list_path) as f:
    gene_names = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(gene_names)} gene names")

# Load Evo2 dMRI weights for both groups
mdd_dir = Path("/storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/stratified_fm/stratified_evo2_dmri_mdd")
ctrl_dir = Path("/storage/bigdata/UKB/fMRI/gene-brain-CCA/gene-brain-cca-2/derived/stratified_fm/stratified_evo2_dmri_ctrl")

W_gene_mdd = np.load(mdd_dir / "W_gene.npy")
W_gene_ctrl = np.load(ctrl_dir / "W_gene.npy")

print(f"\nMDD gene weights shape: {W_gene_mdd.shape}")
print(f"Control gene weights shape: {W_gene_ctrl.shape}")

# Get first canonical component (CC1) weights
w_mdd_cc1 = W_gene_mdd[:, 0] if W_gene_mdd.ndim > 1 else W_gene_mdd
w_ctrl_cc1 = W_gene_ctrl[:, 0] if W_gene_ctrl.ndim > 1 else W_gene_ctrl

print(f"\nMDD CC1 weights shape: {w_mdd_cc1.shape}")
print(f"Control CC1 weights shape: {w_ctrl_cc1.shape}")

# Analyze MDD group
print("\n" + "="*80)
print("MDD GROUP ANALYSIS (r_holdout = -0.032)")
print("="*80)

# Get absolute values for ranking
abs_weights_mdd = np.abs(w_mdd_cc1)
top_indices_mdd = np.argsort(abs_weights_mdd)[::-1][:20]

print("\nTop 20 genes by absolute weight (MDD):")
print(f"{'Rank':<6} {'Gene':<12} {'Weight':<12} {'|Weight|':<12}")
print("-" * 50)
for rank, idx in enumerate(top_indices_mdd, 1):
    gene = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
    print(f"{rank:<6} {gene:<12} {w_mdd_cc1[idx]:>11.6f} {abs_weights_mdd[idx]:>11.6f}")

# Sparsity
zero_threshold = 1e-6
n_zero_mdd = np.sum(np.abs(w_mdd_cc1) < zero_threshold)
print(f"\nSparsity: {n_zero_mdd}/{len(w_mdd_cc1)} genes near-zero ({n_zero_mdd/len(w_mdd_cc1)*100:.1f}%)")

# Analyze Control group
print("\n" + "="*80)
print("CONTROL GROUP ANALYSIS (r_holdout = +0.103)")
print("="*80)

# Get absolute values for ranking
abs_weights_ctrl = np.abs(w_ctrl_cc1)
top_indices_ctrl = np.argsort(abs_weights_ctrl)[::-1][:20]

print("\nTop 20 genes by absolute weight (Control):")
print(f"{'Rank':<6} {'Gene':<12} {'Weight':<12} {'|Weight|':<12}")
print("-" * 50)
for rank, idx in enumerate(top_indices_ctrl, 1):
    gene = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
    print(f"{rank:<6} {gene:<12} {w_ctrl_cc1[idx]:>11.6f} {abs_weights_ctrl[idx]:>11.6f}")

# Sparsity
n_zero_ctrl = np.sum(np.abs(w_ctrl_cc1) < zero_threshold)
print(f"\nSparsity: {n_zero_ctrl}/{len(w_ctrl_cc1)} genes near-zero ({n_zero_ctrl/len(w_ctrl_cc1)*100:.1f}%)")

# Compare overlap
print("\n" + "="*80)
print("COMPARISON: MDD vs CONTROL")
print("="*80)

top10_mdd = set(top_indices_mdd[:10])
top10_ctrl = set(top_indices_ctrl[:10])
overlap = top10_mdd & top10_ctrl

print(f"\nOverlap in top 10 genes: {len(overlap)}/10")
if overlap:
    print("Shared genes:")
    for idx in sorted(overlap):
        gene = gene_names[idx] if idx < len(gene_names) else f"Gene_{idx}"
        print(f"  {gene}: MDD={w_mdd_cc1[idx]:.6f}, Ctrl={w_ctrl_cc1[idx]:.6f}")

# Weight correlation
corr = np.corrcoef(w_mdd_cc1, w_ctrl_cc1)[0, 1]
print(f"\nCorrelation between MDD and Control gene weights: r = {corr:.4f}")
print(f"(Note: This is reported as 'gene_weight_cosine_cc1 = 0.0' in the JSON)")

# Biological coherence check - look for known MDD genes
known_mdd_genes = [
    'BDNF', 'SLC6A4', 'HTR1A', 'HTR2A', 'TPH1', 'TPH2', 
    'COMT', 'MTHFR', 'NR3C1', 'FKBP5', 'CRHR1', 'DRD2',
    'DISC1', 'NTRK2', 'RELN', 'GAD1', 'CACNA1C'
]

print("\n" + "="*80)
print("BIOLOGICAL COHERENCE CHECK")
print("="*80)
print("\nKnown MDD-related genes in your 111-gene list:")

genes_in_list = []
for gene in known_mdd_genes:
    if gene in gene_names:
        idx = gene_names.index(gene)
        rank_mdd = np.where(top_indices_mdd == idx)[0]
        rank_ctrl = np.where(top_indices_ctrl == idx)[0]
        rank_mdd_str = f"#{rank_mdd[0]+1}" if len(rank_mdd) > 0 and rank_mdd[0] < 20 else "outside top 20"
        rank_ctrl_str = f"#{rank_ctrl[0]+1}" if len(rank_ctrl) > 0 and rank_ctrl[0] < 20 else "outside top 20"
        
        print(f"  {gene:<12} | MDD: {w_mdd_cc1[idx]:>8.5f} ({rank_mdd_str}), "
              f"Ctrl: {w_ctrl_cc1[idx]:>8.5f} ({rank_ctrl_str})")
        genes_in_list.append(gene)

if not genes_in_list:
    print("  None of the canonical MDD genes are in top 20 for either group")

print(f"\n{len(genes_in_list)}/{len(known_mdd_genes)} canonical MDD genes present in 111-gene list")

# Statistical summary
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

print(f"\nMDD group:")
print(f"  Mean weight: {np.mean(w_mdd_cc1):.6f}")
print(f"  Std weight:  {np.std(w_mdd_cc1):.6f}")
print(f"  Max |weight|: {np.max(abs_weights_mdd):.6f}")
print(f"  Effective N features (|w| > 0.01): {np.sum(abs_weights_mdd > 0.01)}")

print(f"\nControl group:")
print(f"  Mean weight: {np.mean(w_ctrl_cc1):.6f}")
print(f"  Std weight:  {np.std(w_ctrl_cc1):.6f}")
print(f"  Max |weight|: {np.max(abs_weights_ctrl):.6f}")
print(f"  Effective N features (|w| > 0.01): {np.sum(abs_weights_ctrl > 0.01)}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
The p = 0.049 result comes from the difference in r_holdout:
  - MDD: r = -0.032 (no coupling, negative direction)
  - Control: r = +0.103 (weak positive coupling)
  - Difference: -0.135 (p = 0.049)

Key questions:
1. Do the top genes make biological sense?
2. Is there overlap between MDD and Control top genes?
3. Are the gene weight patterns stable (high cosine similarity)?

Gene weight cosine = 0.0 suggests MDD and Control learned completely
different patterns, which is suspicious and indicates instability.
""")
