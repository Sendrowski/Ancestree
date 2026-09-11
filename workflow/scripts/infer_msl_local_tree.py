"""B13 (local-tree variant): MSL chr1 polarisation via inferred local trees.

VCF-only counterpart to the fixed-tree / ARG MSL runs. For each positional
chunk it builds ingroup-only :class:`Site` records (the 174 MSL haplotypes,
outgroups are not used in local-tree mode) from the compact polymorphic npz,
runs :class:`~ancestree.local_tree_inference.LocalTreeInference` (PSMC'-style
pairwise-coalescent HMM -> per-window UPGMA -> Felsenstein kernel), and writes
per-site MAP calls in the same ``pos<TAB>allele_idx`` format as the other
modes so the 3-way report consumes it unchanged.

Per-chunk (not whole-chr1) because the pairwise HMM is O(n^2) in samples and
the local trees are windowed within a contiguous region anyway. Chunk
boundaries are arbitrary 6.25 Mb cuts, so the handful of windows straddling a
boundary are a negligible fraction of the ~1.14M SNPs.

Positions are offset to chunk-relative coordinates for the builder (whose
windows tile ``[0, sequence_length)``) and mapped back to absolute chr1
coordinates on output.

Args (module defaults glob every chunk; ``--max-chunks N`` for a timing probe).
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

# The manuscript's ensemble size lives in the sibling module, so every
# figure marginalises over the same number of genealogies.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _robustness_common import LOCAL_TREE_ENSEMBLE_MEMBERS  # noqa: E402

from ancestree import JC69, STATE_INDEX, STATES
from ancestree.local_tree_inference import LocalTreeInference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_io import sites_from_arrays  # noqa: E402

MSL_MU = 1.25e-8
MSL_REC = 1e-8


def _parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--max-chunks", type=int, default=0,
                   help="if >0, only process the first N chunks (timing probe)")
    p.add_argument("--out-stem", default="results/data/msl_localtree_chr1_full")
    return p.parse_args(argv)


try:
    # A sharded rule passes one path, and snakemake hands that over as a plain
    # str --- list() would iterate its characters, so wrap before listing.
    _npz = snakemake.input.npz  # type: ignore[name-defined]
    _meta = snakemake.input.meta  # type: ignore[name-defined]
    npz_paths = [_npz] if isinstance(_npz, str) else list(_npz)
    meta_paths = [_meta] if isinstance(_meta, str) else list(_meta)
    out_anc = snakemake.output.anc  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    out_posteriors = snakemake.output.posteriors  # type: ignore[name-defined]
    max_chunks = 0
except NameError:
    _a = _parse_args(sys.argv[1:])
    npz_paths = sorted(
        p for p in glob.glob("results/data/msl_polymorphic_chr1_*.npz")
        if not p.endswith("_meta.json")
    )
    meta_paths = [p.replace(".npz", "_meta.json") for p in npz_paths]
    out_anc = f"{_a.out_stem}_calls.txt"
    out_meta = f"{_a.out_stem}_meta.json"
    out_posteriors = f"{_a.out_stem}_posteriors.npz"
    max_chunks = _a.max_chunks


def _meta(path):
    with open(path) as f:
        return json.load(f)


order = sorted(range(len(npz_paths)), key=lambda i: int(_meta(meta_paths[i])["start"]))
npz_paths = [npz_paths[i] for i in order]
meta_paths = [meta_paths[i] for i in order]
if max_chunks > 0:
    npz_paths = npz_paths[:max_chunks]
    meta_paths = meta_paths[:max_chunks]


def build_ingroup_sites(z, start):
    """Ingroup-only, chunk-relative Site records. Returns (sites, names)."""
    ing_names = list(z["ingroup_names"].astype(str))
    sites = sites_from_arrays(
        z["pos"], z["ref_idx"], z["alt_idx"], z["ingroup"], ing_names,
        pos_offset=-start,  # chunk-relative for the local-tree windower
    )
    return sites, ing_names


# -------------------------------------------------------- per-chunk loop
Path(out_anc).parent.mkdir(parents=True, exist_ok=True)
n_written = n_skipped = 0
post_positions: list[int] = []
post_values: list[list[float]] = []
per_chunk_timings: list[dict] = []
t_all0 = time.perf_counter()

with open(out_anc, "w") as fout:
    for npz_path, meta_path in zip(npz_paths, meta_paths):
        m = _meta(meta_path)
        start, end = int(m["start"]), int(m["end"])
        z = np.load(npz_path, allow_pickle=True)
        sites, ing_names = build_ingroup_sites(z, start)
        if len(sites) < 2:
            continue
        t0 = time.perf_counter()
        inference = LocalTreeInference(
            sites, JC69(),
            mu=MSL_MU, rec_rate=MSL_REC,
            sample_names=ing_names,
            sequence_length=float(end - start),
            window="30snp", progress=False,
            n_ensemble=LOCAL_TREE_ENSEMBLE_MEMBERS,
            # Explicit, not the library default: this row is defined on the
            # single-region path, whose window grid and per-region time-grid
            # calibration differ from the segmented one.
            chunk_size=None,
        )
        nc = 0
        for site, posterior in inference.infer():
            allele = posterior.map_allele
            if allele not in STATE_INDEX:
                n_skipped += 1
                continue
            abs_pos = int(site.pos) + start  # back to absolute chr1 coords
            fout.write(f"{abs_pos}\t{STATE_INDEX[allele]}\n")
            post_positions.append(abs_pos)
            post_values.append([float(v) for v in posterior.values])
            n_written += 1
            nc += 1
        dt = time.perf_counter() - t0
        per_chunk_timings.append({
            "region": f"chr1_{start}_{end}", "n_sites": len(sites),
            "n_called": nc, "wall_s": dt,
        })
        print(
            f"local-tree chr1_{start}_{end}: {len(sites):,} SNPs, "
            f"{nc:,} calls, {dt:.1f}s",
            flush=True,
        )

t_all = time.perf_counter() - t_all0
print(
    f"infer_msl_local_tree: {n_written:,} calls across {len(per_chunk_timings)} "
    f"chunks in {t_all:.1f}s (skipped {n_skipped})",
    flush=True,
)

np.savez_compressed(
    out_posteriors,
    pos=np.asarray(post_positions, dtype=np.int32),
    posterior=np.asarray(post_values, dtype=np.float32),
    alleles=np.asarray(list(STATES)),
)

meta = {
    "mode": "local_tree_chr1",
    "n_chunks": len(per_chunk_timings),
    "n_sites_written": n_written,
    "n_sites_skipped": n_skipped,
    "mu": MSL_MU,
    "rec_rate": MSL_REC,
    "window": "30snp",
    "wall_total_s": t_all,
    "per_chunk": per_chunk_timings,
}
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"infer_msl_local_tree: wrote {out_anc} + {out_meta}", flush=True)
