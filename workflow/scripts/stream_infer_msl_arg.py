"""B13b ARG-mode streaming inference (single-process, all chr1 chunks).

Why one process for all 40 chunks?  The gamma-SMC ARG is 18 GB resident.
Parallelising chunks would multiply that and swap.  This script loads the
ARG once, then for each chunk's ingroup .npz iterates SNPs in position
order, advances a single ``tskit.Tree`` forward, builds a Site from the
haplotype states, and runs the Felsenstein kernel via the kernel-level
API of :class:`~ancestree.inference.ARGBasedInference`.

Outputs per chunk match the projection-based path so downstream
aggregation / reporting rules are unchanged:

- ``msl_arg_{region}_calls.txt``       — ``pos\\tallele_idx`` per site
- ``msl_arg_{region}_calls_meta.json`` — timings + counts
- ``msl_arg_{region}_posteriors.npz``  — ``pos`` + ``posterior`` (N,4)

After all per-chunk files are emitted, also writes
``msl_arg_chr1full_streaming_meta.json`` with the aggregated load + per-
chunk timings so the manuscript can quote a single ARG-mode wall-clock.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tskit

from ancestree import STATES

# Per-tree batching + numba-jit kernel are pulled in via the shared
# helper so the production script and the 2x2 benchmark exercise the
# exact same code path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_arg_kernel import (  # noqa: E402
    build_inference,
    infer_chunk_batched,
)


# -------------------------------------------------------- input config
try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_npz_list = list(snakemake.input.npz_files)  # type: ignore[name-defined]
    out_calls_list = list(snakemake.output.calls)  # type: ignore[name-defined]
    out_meta_list = list(snakemake.output.metas)  # type: ignore[name-defined]
    out_post_list = list(snakemake.output.posteriors)  # type: ignore[name-defined]
    out_overall_meta = snakemake.output.overall_meta  # type: ignore[name-defined]
    time_scale = float(snakemake.params.time_scale)  # type: ignore[name-defined]
except NameError:
    in_trees = (
        "external/polarbear/data/real_data/ancestral_state/PolarBEAR_gammaSMC/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned_tskit.trees"
    )
    chr1_len = 248_956_422
    step = 6_250_000
    regions = []
    for cs in range(0, chr1_len, step):
        ce = min(cs + step, chr1_len)
        def _fmt(bp):
            if bp == 0: return "0"
            return f"{bp // 1_000_000}M" if bp % 1_000_000 == 0 else str(bp)
        regions.append(f"chr1_{_fmt(cs)}_{_fmt(ce)}")
    in_npz_list = [f"results/data/msl_polymorphic_{r}.npz" for r in regions]
    out_calls_list = [f"results/data/msl_arg_{r}_calls.txt" for r in regions]
    out_meta_list = [f"results/data/msl_arg_{r}_calls_meta.json" for r in regions]
    out_post_list = [f"results/data/msl_arg_{r}_posteriors.npz" for r in regions]
    out_overall_meta = "results/data/msl_arg_chr1full_streaming_meta.json"
    # Coalescent-unit node times: one unit is 4*Ne generations, so the branch
    # scale is 4 * Ne * mu with Ne = 1e4 and mu = 1.25e-8.
    time_scale = 5.0e-4


NUC = ("A", "C", "G", "T")


# -------------------------------------------------------- load full ARG
t_overall = time.perf_counter()
t0 = time.perf_counter()
ts = tskit.load(in_trees)
sample_nodes = ts.samples()
n_haps = len(sample_nodes)
t_load = time.perf_counter() - t0
print(
    f"stream_infer_msl_arg: loaded ARG ({n_haps} samples, "
    f"{ts.num_trees:,} trees, {ts.num_edges:,} edges) in {t_load:.1f}s",
    flush=True,
)


# -------------------------------------------------------- inference setup
inference, engine, sample_names = build_inference(ts, time_scale=time_scale)


# -------------------------------------------------------- chunk loop
per_chunk_timings: dict[str, float] = {}
per_chunk_counts: dict[str, int] = {}
total_written = 0


for chunk_idx, (in_npz, out_anc, out_meta, out_post) in enumerate(
    zip(in_npz_list, out_calls_list, out_meta_list, out_post_list)
):
    region = Path(in_npz).stem.replace("msl_polymorphic_", "")
    data = np.load(in_npz, allow_pickle=True)
    positions = data["pos"]
    ingroup = data["ingroup"]
    n_snps = len(positions)
    print(
        f"\n[{chunk_idx+1:>2}/{len(in_npz_list)}] {region}  {n_snps:,} SNPs",
        flush=True,
    )

    Path(out_anc).parent.mkdir(parents=True, exist_ok=True)
    t_chunk0 = time.perf_counter()

    # Per-tree batched kernel: groups SNPs by their local-tree interval,
    # calls the Felsenstein kernel once per tree (numba JIT path if
    written_pos, map_idx, posteriors, stats = infer_chunk_batched(
        ts, positions, ingroup,
        inference, engine, sample_names, time_scale,
    )
    n_written = int(stats["n_sites_written"])
    n_skipped = int(stats["n_skipped_out_of_tree"])
    n_multi_root = int(stats["n_multi_root"])
    total_written += n_written

    with open(out_anc, "w") as fout:
        for s_pos, a_idx in zip(written_pos, map_idx):
            fout.write(f"{int(s_pos)}\t{int(a_idx)}\n")

    t_chunk = time.perf_counter() - t_chunk0
    per_chunk_timings[region] = float(t_chunk)
    per_chunk_counts[region] = int(n_written)

    Path(out_post).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_post,
        pos=written_pos,
        posterior=posteriors,
        alleles=np.asarray(list(STATES)),
    )
    meta = {
        "region": region,
        "time_scale": time_scale,
        "n_samples": int(n_haps),
        "n_sites_in_chunk": int(n_snps),
        "n_sites_written": int(n_written),
        "n_skipped_out_of_tree": int(n_skipped),
        "n_multi_root": int(n_multi_root),
        "model": "JC69",
        "recurrence": "full",
        "timings_seconds": {
            "fit_infer": float(t_chunk),
            "total": float(t_chunk),  # load shared. Per-chunk total = fit_infer
        },
        "in_trees": str(in_trees),
        "in_npz": str(in_npz),
    }
    Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"  done  {n_written:,} calls  ({t_chunk:.1f}s; "
        f"{n_multi_root} multi-root, {n_skipped} out-of-tree)",
        flush=True,
    )


# -------------------------------------------------------- overall meta
t_total = time.perf_counter() - t_overall
overall = {
    "in_trees": str(in_trees),
    "n_chunks": len(in_npz_list),
    "load_seconds": float(t_load),
    "per_chunk_seconds": per_chunk_timings,
    "per_chunk_n_sites_written": per_chunk_counts,
    "total_seconds": float(t_total),
    "total_sites_written": int(total_written),
    "time_scale": time_scale,
    "model": "JC69",
    "recurrence": "full",
}
Path(out_overall_meta).parent.mkdir(parents=True, exist_ok=True)
with open(out_overall_meta, "w") as f:
    json.dump(overall, f, indent=2)
print(
    f"\nstream_infer_msl_arg: total {total_written:,} sites written "
    f"in {t_total:.1f}s ({t_total/60:.1f} min); "
    f"per-chunk min/max/mean: "
    f"{min(per_chunk_timings.values()):.0f}/"
    f"{max(per_chunk_timings.values()):.0f}/"
    f"{sum(per_chunk_timings.values())/len(per_chunk_timings):.0f}s",
    flush=True,
)
