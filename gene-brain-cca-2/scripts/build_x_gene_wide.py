#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path

def read_gene_list(path: Path):
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", required=True)
    ap.add_argument("--gene-list", required=True)
    ap.add_argument("--iids", required=True)
    ap.add_argument("--ids-keep", required=True, help="ids_common.npy to subset rows AND define output row order")
    ap.add_argument("--n-files", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    embed_root = Path(args.embed_root)
    genes = read_gene_list(Path(args.gene_list))
    iids = np.load(args.iids, allow_pickle=True).astype(str)
    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str)
    iid_to_idx = {sid: i for i, sid in enumerate(iids)}
    missing = [sid for sid in ids_keep if sid not in iid_to_idx]
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} ids from --ids-keep not found in --iids (example: {missing[:3]})")
    # Global row indices in the EXACT order of ids_keep
    keep_idx = np.array([iid_to_idx[sid] for sid in ids_keep], dtype=np.int64)
    # We'll gather embeddings in sorted-index order (streaming through chunk files),
    # then unsort back to ids_keep order.
    order = np.argsort(keep_idx)
    keep_idx_sorted = keep_idx[order]
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cols = []
    for g in genes:
        gene_dir = embed_root / g
        parts = []
        offset = 0
        for k in range(1, args.n_files + 1):
            fp = gene_dir / f"embeddings_{k}_layer_last.npy"
            E = np.load(fp, mmap_mode="r")  # (n_chunk, 768)
            n_chunk = E.shape[0]
            start = offset
            end = offset + n_chunk
            offset = end

            # Pick the subset of keep_idx_sorted that falls within this chunk slice.
            lo = np.searchsorted(keep_idx_sorted, start, side="left")
            hi = np.searchsorted(keep_idx_sorted, end, side="left")
            if hi > lo:
                local_idx = (keep_idx_sorted[lo:hi] - start).astype(np.int64)
                parts.append(np.asarray(E[local_idx], dtype=np.float32))
        if offset != iids.shape[0]:
            raise SystemExit(
                f"[ERROR] Total rows across chunks for gene {g} = {offset}, expected {iids.shape[0]}.\n"
                f"Check --n-files and embed_root layout."
            )

        G_sorted = np.concatenate(parts, axis=0)  # (N_keep, 768) in keep_idx_sorted order
        if G_sorted.shape[0] != keep_idx_sorted.shape[0]:
            raise SystemExit(
                f"[ERROR] Collected {G_sorted.shape[0]} rows for gene {g}, expected {keep_idx_sorted.shape[0]}"
            )
        # Reorder back to ids_keep order
        G = G_sorted[inv_order]
        cols.append(G)
        print(f"[gene] {g}: {G.shape}", flush=True)
    X = np.stack(cols, axis=1)  # (N, G, 768)
    X = X.reshape(X.shape[0], -1)  # (N, G*768)
    np.save(out/"ids_gene_overlap.npy", ids_keep)
    np.save(out/"X_gene_wide.npy", X.astype(np.float32))
    print(f"[save] X_gene_wide {X.shape}")

if __name__ == "__main__":
    main()