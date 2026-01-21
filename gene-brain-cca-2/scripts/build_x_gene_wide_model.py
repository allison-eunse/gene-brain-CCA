#!/usr/bin/env python3
import argparse
import numpy as np
from pathlib import Path
import re
import glob


MODEL_CONFIG = {
    "caduceus": {"n_files": 58, "pattern": "embeddings_{k}.npy"},
    "hyenadna": {"n_files": 49, "pattern": "embeddings_{k}_layer-1.npy"},
    "dnabert2": {"n_files": 49, "pattern": "embeddings_{k}_layer_last.npy"},
    "evo2": {"n_files": 49, "pattern": "embeddings_{k}_layer_blocks_21_mlp_l3.npy"},
}


def read_gene_list(path: Path):
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-root", required=True)
    ap.add_argument("--gene-list", required=True)
    ap.add_argument("--iids", required=True)
    ap.add_argument("--ids-keep", required=True, help="ids_common.npy to subset rows AND define output row order")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", choices=sorted(MODEL_CONFIG.keys()))
    ap.add_argument("--n-files", type=int, default=None, help="Override chunk count for custom layouts")
    ap.add_argument("--pattern", default=None, help="Override filename pattern, use {k} for index")
    ap.add_argument("--use-glob", action="store_true", help="Ignore n-files; load all matching files per gene dir")
    args = ap.parse_args()

    if args.model is None and (args.pattern is None or (args.n_files is None and not args.use_glob)):
        raise SystemExit("[ERROR] Provide --model or both --pattern and --n-files.")

    if args.model:
        cfg = MODEL_CONFIG[args.model]
        n_files = args.n_files or cfg["n_files"]
        pattern = args.pattern or cfg["pattern"]
    else:
        n_files = args.n_files
        pattern = args.pattern

    embed_root = Path(args.embed_root)
    genes = read_gene_list(Path(args.gene_list))
    iids = np.load(args.iids, allow_pickle=True).astype(str)
    ids_keep = np.load(args.ids_keep, allow_pickle=True).astype(str)

    iid_to_idx = {sid: i for i, sid in enumerate(iids)}
    missing = [sid for sid in ids_keep if sid not in iid_to_idx]
    if missing:
        raise SystemExit(f"[ERROR] {len(missing)} ids from --ids-keep not found in --iids (example: {missing[:3]})")

    keep_idx = np.array([iid_to_idx[sid] for sid in ids_keep], dtype=np.int64)
    order = np.argsort(keep_idx)
    keep_idx_sorted = keep_idx[order]
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = []
    def iter_files(gene_dir: Path):
        if args.use_glob:
            if "{k}" in pattern:
                glob_pat = pattern.replace("{k}", "*")
            else:
                glob_pat = pattern
            files = glob.glob(str(gene_dir / glob_pat))
            if not files:
                raise SystemExit(f"[ERROR] No files matched {glob_pat} in {gene_dir}")
            def idx_key(p):
                m = re.search(r"embeddings_(\\d+)", Path(p).name)
                return int(m.group(1)) if m else 0
            files = sorted(files, key=idx_key)
            for f in files:
                yield Path(f)
        else:
            for k in range(1, n_files + 1):
                yield gene_dir / pattern.format(k=k)

    for g in genes:
        gene_dir = embed_root / g
        parts = []
        offset = 0
        for fp in iter_files(gene_dir):
            E = np.load(fp, mmap_mode="r")  # (n_chunk, dim)
            n_chunk = E.shape[0]
            start = offset
            end = offset + n_chunk
            offset = end

            lo = np.searchsorted(keep_idx_sorted, start, side="left")
            hi = np.searchsorted(keep_idx_sorted, end, side="left")
            if hi > lo:
                local_idx = (keep_idx_sorted[lo:hi] - start).astype(np.int64)
                parts.append(np.asarray(E[local_idx], dtype=np.float32))
        if offset != iids.shape[0]:
            raise SystemExit(
                f"[ERROR] Total rows across chunks for gene {g} = {offset}, expected {iids.shape[0]}.\n"
                f"Check --n-files/--pattern and embed_root layout."
            )

        G_sorted = np.concatenate(parts, axis=0)
        if G_sorted.shape[0] != keep_idx_sorted.shape[0]:
            raise SystemExit(
                f"[ERROR] Collected {G_sorted.shape[0]} rows for gene {g}, expected {keep_idx_sorted.shape[0]}"
            )
        G = G_sorted[inv_order]
        cols.append(G)
        print(f"[gene] {g}: {G.shape}", flush=True)

    X = np.stack(cols, axis=1)  # (N, G, dim)
    X = X.reshape(X.shape[0], -1)  # (N, G*dim)
    np.save(out / "ids_gene_overlap.npy", ids_keep)
    np.save(out / "X_gene_wide.npy", X.astype(np.float32))
    print(f"[save] X_gene_wide {X.shape}", flush=True)


if __name__ == "__main__":
    main()
