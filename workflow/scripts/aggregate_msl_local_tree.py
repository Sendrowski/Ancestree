"""Concatenate the per-region MSL local-tree shards into the chr1 call set.

The shards are independent by construction: each region is windowed inside
itself (``pos_offset`` makes the builder's coordinates region-relative) and
written back out in absolute chr1 coordinates, so joining them is a
concatenation in start order rather than a merge.
"""
import json

import numpy as np

try:
    calls_in = list(snakemake.input.calls)  # noqa: F821
    metas_in = list(snakemake.input.metas)  # noqa: F821
    posts_in = list(snakemake.input.posteriors)  # noqa: F821
    out_anc = snakemake.output.anc  # noqa: F821
    out_posteriors = snakemake.output.posteriors  # noqa: F821
    out_meta = snakemake.output.meta  # noqa: F821
except NameError:
    # Standalone debug defaults; the DAG supplies these through `script:`.
    import glob as _glob
    import re as _re

    def _shards(suffix):
        """Region shards in start order, excluding this script's own output."""
        paths = [q for q in _glob.glob(
            f"results/data/msl_localtree_chr1_*{suffix}")
            if "_chr1_full" not in q]

        def start(q):
            m = _re.search(r"chr1_(\d+)(M?)_", q)
            if not m:
                return 0
            return int(m.group(1)) * (1_000_000 if m.group(2) else 1)
        return sorted(paths, key=start)

    calls_in = _shards("_calls.txt")
    metas_in = _shards("_meta.json")
    posts_in = _shards("_posteriors.npz")
    out_anc = "results/data/msl_localtree_chr1_full_calls.txt"
    out_posteriors = "results/data/msl_localtree_chr1_full_posteriors.npz"
    out_meta = "results/data/msl_localtree_chr1_full_meta.json"

# Input order is genomic order: the rule expands over MSL_CHR1_CHUNK_REGIONS,
# which is start-ordered, and expand() preserves it. Deriving the order from
# the region strings would be wrong --- they mix "25M" style abbreviations with
# plain digits, so they do not sort lexically.
order = range(len(metas_in))

n_written = 0
with open(out_anc, "w") as fout:
    for i in order:
        with open(calls_in[i]) as fin:
            for line in fin:
                fout.write(line)
                n_written += 1

pos_parts, post_parts, alleles = [], [], None
for i in order:
    z = np.load(posts_in[i], allow_pickle=True)
    if len(z["pos"]):
        pos_parts.append(z["pos"])
        post_parts.append(z["posterior"])
    alleles = z["alleles"]
np.savez_compressed(
    out_posteriors,
    pos=(np.concatenate(pos_parts) if pos_parts
         else np.empty(0, dtype=np.int32)),
    posterior=(np.concatenate(post_parts) if post_parts
               else np.empty((0, 4), dtype=np.float32)),
    alleles=alleles,
)

per_chunk, n_skipped, wall = [], 0, 0.0
for i in order:
    m = json.load(open(metas_in[i]))
    per_chunk.extend(m.get("per_chunk", []))
    n_skipped += int(m.get("n_sites_skipped", 0))
    wall += float(m.get("wall_total_s", 0.0))
with open(out_meta, "w") as fh:
    json.dump({
        "mode": "local_tree_chr1",
        "n_written": n_written,
        "n_skipped": n_skipped,
        "n_chunks": len(per_chunk),
        # Sum of the shards' own wall times, i.e. the serial cost. The observed
        # wall time is the slowest shard, since they run concurrently.
        "wall_s_serial_equivalent": wall,
        "per_chunk": per_chunk,
    }, fh, indent=2)

print(f"aggregate_msl_local_tree: {n_written:,} calls from {len(order)} shards")
