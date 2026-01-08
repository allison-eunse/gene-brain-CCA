#!/bin/bash
#SBATCH --job-name=gene_x
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Adjust these on the lab server
NESAP="/path/to/nesap-genomics-allison"
EMBED_ROOT="$NESAP/DNABERT2_embedding_merged"   # contains <GENE>/embeddings_*_layer_last.npy
GENE_LIST="$NESAP/iids_labels_covariates/gene_list_filtered.txt"
IIDS="$NESAP/iids_labels_covariates/iids.npy"
N_FILES=49

OUT_DIR="$PWD/derived/gene_x"

mkdir -p logs "$OUT_DIR"

source ~/.bashrc || true
conda activate <YOUR_ENV>

python scripts/build_x_gene.py \
  --embed-root "$EMBED_ROOT" \
  --gene-list "$GENE_LIST" \
  --iids "$IIDS" \
  --n-files "$N_FILES" \
  --reduce mean \
  --pca512 \
  --out-dir "$OUT_DIR"


