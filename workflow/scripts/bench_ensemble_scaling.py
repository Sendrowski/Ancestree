"""Time and peak memory of one ensemble-mode configuration.

One cell of the (draws x workers) grid, written as its own JSON so the grid
shards across the cluster and a re-run only repeats the cells that changed.

Peak memory is ``ru_maxrss``, a high-water mark that never falls, so a cell
must own its process: measuring several configurations in one would report the
largest cell's peak for every later one. Running one cell per job gives that
for free. ``RUSAGE_SELF`` alone would understate a fork-pool run, whose work
happens in children the parent's counter never sees, so the children's peak is
added to it.

The numba kernels are compiled on a small warm-up pass before the clock
starts. They are cached to disk, so otherwise the first cell of a grid pays a
compilation cost that every later cell avoids, and the grid reads as flat in
the draws.
"""
import json
import resource
import time

import msprime

from ancestree import JC69, LocalTreeInference
from ancestree.sites import Site

try:
    n_ensemble = int(snakemake.params.n_ensemble)  # type: ignore[name-defined]
    n_workers = int(snakemake.params.n_workers)  # type: ignore[name-defined]
    n_individuals = int(snakemake.params.n_individuals)  # type: ignore[name-defined]
    sequence_length = int(snakemake.params.sequence_length)  # type: ignore[name-defined]
    chunk_size = str(snakemake.params.chunk_size)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    n_ensemble, n_workers, n_individuals = 128, 1, 50
    sequence_length, chunk_size = 2_000_000, "250kb"
    mu, rec_rate, seed = 1.25e-8, 1e-8, 3
    out_json = "results/data/ensemble_scaling_m128_w1.json"

#: Effective size of the simulated population, in diploid individuals.
POPULATION_SIZE = 1e4

ts = msprime.sim_ancestry(
    n_individuals, ploidy=2, sequence_length=sequence_length,
    recombination_rate=rec_rate, population_size=POPULATION_SIZE,
    random_seed=seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
names = [f"i{i}_h{h}" for i in range(n_individuals) for h in (0, 1)]

sites = []
for variant in ts.variants():
    alleles = variant.alleles
    if len(alleles) != 2 or any(a not in "ACGT" for a in alleles if a):
        continue
    sites.append(Site(
        chrom="1", pos=int(variant.site.position), alleles=tuple(alleles),
        tip_alleles={names[k]: alleles[variant.genotypes[k]]
                     for k in range(len(names))},
    ))

warm = LocalTreeInference(
    sites[:60], JC69(), mu=mu, rec_rate=rec_rate, sample_names=names,
    sequence_length=float(sequence_length), window="8snp",
    n_ensemble=2, n_workers=1, progress=False,
)
sum(1 for _ in warm.infer())
del warm

def _peak_mib() -> float:
    """Peak resident set of this process and its workers, in MiB."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (own + kids) / 1024.0


baseline_mib = _peak_mib()
started = time.perf_counter()
inference = LocalTreeInference(
    sites, JC69(), mu=mu, rec_rate=rec_rate, sample_names=names,
    sequence_length=float(sequence_length), window="8snp",
    n_ensemble=n_ensemble, n_workers=n_workers, chunk_size=chunk_size,
    progress=False,
)
n_scored = sum(1 for _ in inference.infer())
wall_s = time.perf_counter() - started
peak_mib = _peak_mib()

cell = {
    "n_ensemble": n_ensemble,
    "n_workers": n_workers,
    "n_individuals": n_individuals,
    "n_haplotypes": 2 * n_individuals,
    "sequence_length": sequence_length,
    "chunk_size": chunk_size,
    "n_sites": n_scored,
    "wall_s": wall_s,
    "peak_mib": peak_mib,
    "baseline_mib": baseline_mib,
}
with open(out_json, "w") as handle:
    json.dump(cell, handle, indent=2)
print(f"M={n_ensemble} workers={n_workers}: {wall_s:.1f}s, "
      f"{peak_mib:.0f} MiB peak over {n_scored:,} sites", flush=True)
