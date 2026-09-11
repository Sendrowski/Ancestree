"""Kernel runtime + peak-memory benchmark on a simulated tree sequence.

All three inference modes run on ONE shared dataset (17 ingroup + 3 outgroups
via a nested-split demography, 100 Mb, JC69) and polarise the SAME site set,
so wall-clock seconds and peak RSS compare directly:

- **ARG mode** (:class:`~ancestree.inference.ARGBasedInference`): reads
  the genealogy directly (all samples as tips), over ``{1, 2, 4, 8}`` workers.
- **VCF mode** (:class:`~ancestree.inference.FixedTreeInference`): fits
  the outgroup ladder, wall covering the whole flow (``fit()`` +
  per-site ``infer()``), over ``{1, 2}`` workers, that axis being the
  L-BFGS-B multi-start pool only.
- **Local-tree mode** (:class:`~ancestree.local_tree_inference.LocalTreeInference`):
  builds a genealogy from genotypes then delegates polarisation to the ARG
  pool, over ``{1, 2, 4, 8}`` numba threads at one worker and ``LOCAL_TREE_CHUNK``
  segments. Workers reach only the Felsenstein pass on this mode, not the tree
  build that dominates it, so the thread count is the axis its runtime turns on.

Each configuration runs in its OWN subprocess (one config per process), so one
mode's pool cannot deadlock a later one. Peak memory is sampled across the whole
process group while the config runs: ``getrusage`` covers the caller alone, and
a pooled configuration holds most of its memory in the children.

The dataset is reproducible from ``msprime`` with hard-coded seeds. It is
re-simulated in every worker and excluded from the per-config wall, which costs
about 100 s per configuration at 100 Mb. The bench is self-contained, with no
real-data dependency.
"""
import json
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

# Force the fork start method so FixedTreeInference's multi-start
# ProcessPoolExecutor (used in the VCF mode 2-worker config below) can
# inherit module-level globals from the parent. Under snakemake's
# script-wrap the default `spawn` method can't re-import the tempfile-
# wrapped script and the workers crash with BrokenProcessPool.
if "fork" in mp.get_all_start_methods():
    mp.set_start_method("fork", force=True)

import msprime
import tskit

# The manuscript's ensemble size lives in the sibling module, so every
# figure marginalises over the same number of genealogies.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _robustness_common import LOCAL_TREE_ENSEMBLE_MEMBERS  # noqa: E402

from ancestree import (
    JC69, FixedTreeInference, KingmanIngroupWeight, LocalTreeInference,
    Site, _jit_kernel, _smc_kernel,
)
from ancestree.inference import ARGBasedInference


# -------------------------------------------------------- input config
try:
    out_md = snakemake.output.md  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    worker_grid = list(snakemake.params.worker_grid)  # type: ignore[name-defined]
except NameError:
    out_md = "results/reports/arg_kernel_bench.md"
    out_json = "results/reports/arg_kernel_bench.json"
    mu = 1.25e-8
    worker_grid = [1, 2, 4, 8]



# Simulated-dataset parameters. A 97-haplotype ingroup keeps the n=100
# (ingroup + 3 outgroups) panel's O(n^2) local-tree HMM tractable at the
# 10 Mb length, which keeps total wall manageable on a laptop.
# Recombination rate and Ne are at standard human-scale values.
SIM_SAMPLES = 17
SIM_LENGTH = 100_000_000
SIM_REC_RATE = 1e-8
SIM_POP_SIZE = 10_000
SIM_SEED = 42
SIM_MU = mu

# Outgroup demography for VCF mode. Three single-haplotype outgroups via a
# nested-split ladder, closest at 50000 gens then 100000 and 150000. The
# ingroup population matches the ARG-mode sim's Ne so the polymorphic-site
# density per haplotype is comparable.
OUT_SPLIT_TIMES = [50_000, 100_000, 150_000]


def _simulate_with_outgroups(*, samples: int, length: float, rec_rate: float,
                             pop_size: float, mu: float, seed: int,
                             split_times: list[float]) -> tuple[
                                 tskit.TreeSequence, list[str], list[str]
                             ]:
    """Ingroup + ``len(split_times)`` outgroups via a nested-split ladder."""
    demography = msprime.Demography()
    demography.add_population(name="ingroup", initial_size=pop_size)
    for k in range(1, len(split_times) + 1):
        demography.add_population(name=f"outgroup_{k}", initial_size=pop_size)
    # Cumulative-ancestor populations: anc_{k} is the MRCA of
    # ingroup + outgroups 1..k. outgroup_{k} splits off from anc_{k-1}
    # at split_times[k-1].
    demography.add_population(name="anc_root", initial_size=pop_size)
    chain: list[str] = ["ingroup"]
    for k in range(1, len(split_times) + 1):
        anc_name = "anc_root" if k == len(split_times) else f"anc_{k}"
        if anc_name != "anc_root":
            demography.add_population(name=anc_name, initial_size=pop_size)
        demography.add_population_split(
            time=split_times[k - 1],
            ancestral=anc_name,
            derived=[chain[-1], f"outgroup_{k}"],
        )
        chain.append(anc_name)
    sample_sets = [msprime.SampleSet(samples, population="ingroup", ploidy=1)]
    for k in range(1, len(split_times) + 1):
        sample_sets.append(
            msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1),
        )
    ts = msprime.sim_ancestry(
        samples=sample_sets,
        demography=demography,
        sequence_length=length,
        recombination_rate=rec_rate,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, model=msprime.JC69(), random_seed=seed)
    # Recover ingroup / outgroup names from population metadata.
    pop_id_to_name = {
        p.id: p.metadata.get("name", f"pop_{p.id}") for p in ts.populations()
    }
    ingroup_names: list[str] = []
    outgroup_names: list[str] = []
    for ind in ts.individuals():
        pop_id = ts.node(int(ind.nodes[0])).population
        pop_name = pop_id_to_name[pop_id]
        # TskitSource derives sample names from individual `name` metadata and
        # falls back to str(node_id) when there is none, which is the case for
        # a plain msprime panel. Name them the way the source will read them.
        name = str(int(ind.nodes[0]))
        if pop_name == "ingroup":
            ingroup_names.append(name)
        elif pop_name.startswith("outgroup_"):
            outgroup_names.append(name)
    return ts, ingroup_names, outgroup_names


def _run_arg_config(ts: tskit.TreeSequence, *, use_numba: bool,
                    n_workers: int, mu: float) -> dict:
    """Time one ARG-mode ``(numba, n_workers)`` cell."""
    label = f"ARG numba={'on ' if use_numba else 'off'} / {n_workers}w"
    if False:
        return {
            "mode": "arg", "label": label, "use_numba": use_numba,
            "n_workers": n_workers, "skipped": "numba not installed",
        }
    _jit_kernel.use_numba = use_numba
    inf = ARGBasedInference(
        source=ts, model=JC69(), mu=mu,
        progress=False, n_workers=n_workers,
    )
    t0 = time.perf_counter()
    n_sites = sum(1 for _ in inf.infer())
    wall = time.perf_counter() - t0
    return {
        "mode": "arg",
        "label": label,
        "use_numba": use_numba,
        "n_workers": n_workers,
        "total_sites": int(n_sites),
        "total_wall_seconds": float(wall),
        "sites_per_second": float(n_sites / wall) if wall else 0.0,
    }


def _run_vcf_config(
    ts: tskit.TreeSequence, *, use_numba: bool, n_workers: int,
    ingroup_names: list[str], outgroup_names: list[str],
) -> dict:
    """Time one VCF-mode (FixedTreeInference) cell — fit + per-site infer.

    ``n_workers`` controls the L-BFGS-B multi-start parallelism only.
    Per-site ``infer()`` is always single-threaded. ``n_workers=1`` runs
    the fit sequentially (``parallelize=False``); ``n_workers > 1`` runs
    the multi-start replicates in parallel via the inference layer's
    ``ProcessPoolExecutor`` (``parallelize=True, n_starts=n_workers``).
    """
    label = f"VCF numba={'on ' if use_numba else 'off'} / {n_workers}w"
    if False:
        return {
            "mode": "vcf", "label": label, "use_numba": use_numba,
            "n_workers": int(n_workers),
            "skipped": "numba not installed",
        }
    _jit_kernel.use_numba = use_numba
    prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
    inf = FixedTreeInference(
        ts,
        ingroup_samples=ingroup_names,
        outgroup_samples=outgroup_names,
        model=JC69(),
        n_target_sites=int(ts.sequence_length),
        ingroup_weight=prior,
        parallelize=(n_workers > 1),
        n_starts=(n_workers if n_workers > 1 else 1),
        progress=False,
    )
    t0 = time.perf_counter()
    inf.fit()
    n_sites = sum(1 for _ in inf.infer())
    wall = time.perf_counter() - t0
    return {
        "mode": "vcf",
        "label": label,
        "use_numba": use_numba,
        "n_workers": int(n_workers),
        "total_sites": int(n_sites),
        "total_wall_seconds": float(wall),
        "sites_per_second": float(n_sites / wall) if wall else 0.0,
    }


def _run_local_tree_config(
    sites: list, names: list[str], *, use_numba: bool, n_workers: int,
    mu: float, rec_rate: float, seq_len: float, chunk_size=None,
) -> dict:
    """Time one local-tree-mode (LocalTreeInference) cell — VCF-only.

    Infers a dated local tree per window from genotypes (pairwise-coalescent
    HMM -> UPGMA), then polarises site-by-site. The numba toggle flips both
    the Felsenstein kernel (``_jit_kernel``) and the SMC-HMM kernel
    (``_smc_kernel``). ``n_workers`` is forwarded to the internal
    ARGBasedInference and parallelises the polarisation pass over the built
    trees (the single-threaded HMM + UPGMA build is serial here). On an
    in-memory panel this single-run path is faster than the chunked path,
    whose per-segment overhead only pays off when memory forces chunking
    (genome scale). Chunking additionally fans the HMM build across workers.
    """
    label = (f"LocalTree numba={'on ' if use_numba else 'off'} / {n_workers}w"
             f" / chunk={chunk_size or 'none'}")
    if False:
        return {
            "mode": "local_tree", "label": label, "use_numba": use_numba,
            "n_workers": int(n_workers), "skipped": "numba not installed",
        }
    _jit_kernel.use_numba = use_numba
    _smc_kernel.use_numba = use_numba
    inf = LocalTreeInference(
        sites, JC69(), mu=mu, rec_rate=rec_rate,
        sample_names=names, sequence_length=float(seq_len),
        window="30snp", progress=False, n_workers=n_workers,
        # Explicit, not the library default: this row reports the cost of
        # the mode as the manuscript runs it, which marginalises over an
        # ensemble rather than scoring one plug-in tree, on the single-region
        # path rather than the segmented one.
        n_ensemble=LOCAL_TREE_ENSEMBLE_MEMBERS,
        chunk_size=chunk_size,
    )
    t0 = time.perf_counter()
    n_sites = sum(1 for _ in inf.infer())
    wall = time.perf_counter() - t0
    return {
        "mode": "local_tree",
        "label": label,
        "use_numba": use_numba,
        "n_workers": int(n_workers),
        "chunk_size": chunk_size,
        "total_sites": int(n_sites),
        "total_wall_seconds": float(wall),
        "sites_per_second": float(n_sites / wall) if wall else 0.0,
    }

# -------------------------------------------------------- shared dataset
# ONE dataset for all three modes: SIM_SAMPLES ingroup plus one outgroup per
# entry of OUT_SPLIT_TIMES, via a nested-split demography. ARG and VCF mode
# polarise its full site set, and local-tree builds a genealogy over all
# samples (outgroups as tips) and polarises the SAME sites, so every row's
# wall and peak memory compare directly on identical data. The local-tree
# rows run chunked: the pairwise-HMM TMRCA matrix grows with the segment
# span, so the blow-up that forces chunking is a length effect rather than a
# per-site one.
def _build_shared():
    ts, ingroup_names, outgroup_names = _simulate_with_outgroups(
        samples=SIM_SAMPLES, length=SIM_LENGTH, rec_rate=SIM_REC_RATE,
        pop_size=SIM_POP_SIZE, mu=SIM_MU, seed=SIM_SEED,
        split_times=OUT_SPLIT_TIMES,
    )
    return ts, ingroup_names, outgroup_names


def _local_tree_sites(ts) -> tuple[list, list[str]]:
    name = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [f"tsk_{ind.id}" for ind in ts.individuals()]
    sites = [
        Site(chrom="1", pos=int(v.site.position), alleles=tuple(v.alleles),
             tip_alleles={name[int(node)]: v.alleles[v.genotypes[j]]
                          for j, node in enumerate(ts.samples())})
        for v in ts.variants()
    ]
    return sites, names


def _rss_scale() -> float:
    """Divisor taking ``ru_maxrss`` to MB (macOS reports bytes, Linux KB)."""
    return 1024 * 1024 if sys.platform == "darwin" else 1024


def _tree_rss_mb() -> float:
    """Resident set of this process and every live descendant, in MB.

    Sampled rather than read from ``getrusage``, whose ``RUSAGE_SELF`` covers
    the caller alone and whose ``RUSAGE_CHILDREN`` reports the largest single
    reaped child rather than the simultaneous total. A pooled configuration
    holds most of its memory in the children, so neither sees it.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-g", str(os.getpgrp())],
            capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return 0.0
    return sum(int(v) for v in out.split() if v.isdigit()) / 1024


def _peak_rss_mb() -> float:
    """High-water-mark RSS in MB, this process only.

    Kept for the single-process configurations. Pooled ones report
    :func:`_tree_rss_mb` sampled during the run instead, since this misses
    every child.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _rss_scale()


# -------------------------------------------------------- worker branch
# Each config runs in its own subprocess (one config per process), so the
# reported peak RSS is that config's alone and the fork-pool / executor of
# one mode can't deadlock a later one. Re-simulating per worker is cheap
# (<1 s) and deterministic (fixed seed).
if len(sys.argv) == 7 and sys.argv[1] == "--worker":
    _, _, w_mode, w_numba, w_nw, w_chunk, w_threads = sys.argv
    # The ensemble kernels are @njit(parallel=True) prange loops, so a single
    # process already spreads across every numba thread. Capping them is what
    # separates intra-kernel threading from the fork-pool worker count.
    if w_threads != "max":
        import numba
        numba.set_num_threads(int(w_threads))
    w_use_numba = (w_numba == "1")
    w_n_workers = int(w_nw)
    # Poll the whole process group while the config runs: a fork pool holds
    # its memory in children that a post-hoc reading would already have lost.
    import threading
    _peak_tree = [0.0]
    _stop = threading.Event()

    def _sample():
        # 0.05 s: the shortest configurations run for about a second, and a
        # coarser interval missed their peak entirely (a 4-worker cell once
        # read below the 2-worker one, which cannot happen).
        while not _stop.wait(0.05):
            _peak_tree[0] = max(_peak_tree[0], _tree_rss_mb())

    _sampler = threading.Thread(target=_sample, daemon=True)
    _sampler.start()
    _ts, _ingroup, _outgroup = _build_shared()
    if w_mode == "arg":
        _r = _run_arg_config(_ts, use_numba=w_use_numba,
                             n_workers=w_n_workers, mu=SIM_MU)
    elif w_mode == "vcf":
        _r = _run_vcf_config(_ts, use_numba=w_use_numba, n_workers=w_n_workers,
                             ingroup_names=_ingroup, outgroup_names=_outgroup)
    elif w_mode == "local_tree":
        _sites, _names = _local_tree_sites(_ts)
        _r = _run_local_tree_config(
            _sites, _names, use_numba=w_use_numba, n_workers=w_n_workers,
            mu=SIM_MU, rec_rate=SIM_REC_RATE, seq_len=_ts.sequence_length,
            chunk_size=(None if w_chunk == "none" else w_chunk),
        )
    else:
        raise SystemExit(f"unknown worker mode: {w_mode}")
    _stop.set()
    _sampler.join(timeout=2)
    _r["n_threads"] = w_threads
    # The sampler sees the children; getrusage sees this process's true
    # high-water mark between polls. Neither dominates, so take both.
    _r["peak_rss_mb"] = max(_peak_rss_mb(), _peak_tree[0])
    print("RESULT_JSON " + json.dumps(_r), flush=True)
    sys.exit(0)


# -------------------------------------------------------- parent driver
SUBPROC_TIMEOUT = 600  # per-config cap. A config that exceeds it is marked


def _run_subprocess(mode: str, use_numba: bool, n_workers: int,
                    chunk=None, threads="max") -> dict | None:
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", mode,
           "1" if use_numba else "0", str(n_workers), chunk or "none",
           str(threads)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=SUBPROC_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"mode": mode, "use_numba": use_numba, "n_workers": n_workers,
                "chunk_size": chunk, "n_threads": str(threads),
                "skipped": f">{SUBPROC_TIMEOUT}s"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    raise RuntimeError(
        f"no RESULT_JSON from {mode}/numba={use_numba}/{n_workers}w\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )


# Header stats from one lightweight parent simulation (config walls exclude
# this. It only labels the dataset).
t0 = time.perf_counter()
_ts_stats, _, _ = _build_shared()
N_SITES = int(_ts_stats.num_sites)
N_TREES = int(_ts_stats.num_trees)
t_sim = time.perf_counter() - t0
print(f"shared dataset: {N_SITES:,} sites, {N_TREES:,} trees "
      f"(sim wall {t_sim:.1f}s)\n", flush=True)
del _ts_stats

# numba=on first so the useful rows land even if a numba-off config is slow.
# numba is a hard dependency, so the un-JIT'd path is not a configuration the
# package can be run in and its column is gone from the table. The toggle still
# exists for the parity tests, which compare the two kernels directly.
#: Chunk span for the local-tree rows. On the single-region path n_workers
#: reaches only the Felsenstein pass, not the tree build that dominates, so a
#: worker scan there measures nothing. Chunked, whole segments fan across the
#: pool. 1 Mb over the 10 Mb panel leaves ten segments, enough to fill 8w.
LOCAL_TREE_CHUNK = "1mb"

#: Numba thread counts for the local-tree thread scan.
THREAD_GRID = (1, 2, 4, 8)

config_grid = (
    [("arg", True, nw, None, "max") for nw in worker_grid]
    + [("vcf", True, nw, None, "max") for nw in (1, 2)]
    + [("local_tree", True, 1, LOCAL_TREE_CHUNK, t) for t in THREAD_GRID]
)

results: list[dict] = []
for mode, use_numba, n_w, chunk, threads in config_grid:
    print(f"=== {mode} numba={'on ' if use_numba else 'off'} / {n_w}w"
          f" / chunk={chunk or 'none'} / threads={threads}", flush=True)
    if False:
        print("  skipped: numba not installed", flush=True)
        continue
    r = _run_subprocess(mode, use_numba, n_w, chunk, threads)
    if r is None or r.get("skipped"):
        print(f"  skipped: {(r or {}).get('skipped', 'no result')}", flush=True)
        if r is not None:
            results.append(r)
        continue
    results.append(r)
    print(f"  {r['total_sites']:,} sites in {r['total_wall_seconds']:.1f}s, "
          f"peak {r['peak_rss_mb']:.0f} MB", flush=True)


# -------------------------------------------------------- write reports
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps({
    "simulation": {
        "samples": SIM_SAMPLES, "length": SIM_LENGTH, "rec_rate": SIM_REC_RATE,
        "pop_size": SIM_POP_SIZE, "mu": SIM_MU, "seed": SIM_SEED,
        "outgroup_split_times": OUT_SPLIT_TIMES,
        "n_sites": N_SITES, "n_trees": N_TREES,
    },
    "worker_grid": worker_grid,
    "subproc_timeout_s": SUBPROC_TIMEOUT,
    "configs": results,
}, indent=2))


def _cell(mode: str, n_workers: int, use_numba: bool,
          chunk=None, threads="max") -> dict | None:
    for r in results:
        if (r["mode"] == mode and r["n_workers"] == n_workers
                and r["use_numba"] is use_numba
                and r.get("chunk_size") == chunk
                and str(r.get("n_threads", "max")) == str(threads)):
            return r
    return None


def _row(mode: str, n_workers: int, chunk=None,
         threads="max") -> tuple[str, str]:
    """(wall seconds, peak mem GB) strings for one configuration."""
    on = _cell(mode, n_workers, True, chunk, threads)

    def _w(r):
        if r is None:
            return "—"
        if r.get("skipped"):
            return r["skipped"]
        return f"{r['total_wall_seconds']:.1f}"
    peaks = [r["peak_rss_mb"] for r in (on,)
             if r is not None and "peak_rss_mb" in r]
    mem = f"{max(peaks) / 1024:.1f}" if peaks else "—"
    return _w(on), mem


row_specs = (
    [(f"ARG mode / {n_w}w", "arg", n_w, None, "max") for n_w in worker_grid]
    + [("fixed-tree mode / 1w", "vcf", 1, None, "max"),
       ("fixed-tree mode / 2w", "vcf", 2, None, "max")]
    + [(f"local-tree mode / {t} thread" + ("s" if t != 1 else ""),
        "local_tree", 1, LOCAL_TREE_CHUNK, t) for t in THREAD_GRID]
)

lines = [
    "# Kernel benchmark — all three modes on one shared dataset",
    "",
    f"Simulation (msprime, seed={SIM_SEED}): {SIM_SAMPLES} ingroup + "
    f"{len(OUT_SPLIT_TIMES)} outgroups (splits {OUT_SPLIT_TIMES} gens), "
    f"length={SIM_LENGTH:,} bp, mu={SIM_MU:g}, Ne={SIM_POP_SIZE:,}, JC69 — "
    f"{N_SITES:,} sites, {N_TREES:,} trees. Every config runs in its own "
    f"subprocess (one config per process), so peak RSS is that config's "
    f"alone; the parent sim wall is excluded.",
    "",
    "| configuration | runtime (s) | peak mem (GB) |",
    "|---|---:|---:|",
]
for label, mode, n_w, chunk, threads in row_specs:
    wall, mem = _row(mode, n_w, chunk, threads)
    lines.append(f"| {label} | {wall} | {mem} |")
lines.append("")
Path(out_md).parent.mkdir(parents=True, exist_ok=True)
Path(out_md).write_text("\n".join(lines))
print(f"\nwrote {out_md} + {out_json}", flush=True)
