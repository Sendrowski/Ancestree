"""Outgroup-effect benchmark simulation: msprime ARG with ingroup + N outgroups.

Builds one tree sequence with an ingroup population plus a fixed
number of outgroup populations at progressively deeper divergence
times. Inference downstream subsets the saved ts via
``ts.simplify(samples=...)`` to study how posterior certainty scales
with the number of outgroups available to the Felsenstein kernel.

Run directly (writes a small default sim)::

    python workflow/scripts/simulate_baseline.py

Or via snakemake::

    snakemake -j 1 results/data/baseline_sim.trees
"""
import json
import math
from pathlib import Path

import msprime


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    n_ingroup = int(snakemake.params.n_ingroup)  # type: ignore[name-defined]
    n_outgroup_pops = int(snakemake.params.n_outgroup_pops)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    ingroup_pop_size = float(getattr(snakemake.params, "ingroup_pop_size", pop_size))  # type: ignore[name-defined]
    ingroup_growth_rate = float(getattr(snakemake.params, "ingroup_growth_rate", 0.0))  # type: ignore[name-defined]
    _flat = getattr(snakemake.params, "ingroup_growth_flat_time", None)  # type: ignore[name-defined]
    ingroup_growth_flat_time = None if _flat is None else float(_flat)
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    outgroup_split_times = list(snakemake.params.outgroup_split_times)  # type: ignore[name-defined]
    _chunk_idx = getattr(snakemake.params, "chunk_idx", None)  # type: ignore[name-defined]
    chunk_idx = None if _chunk_idx is None else int(_chunk_idx)
    n_chunks = int(getattr(snakemake.params, "n_chunks", 1))  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/baseline_sim.trees"
    out_meta = "results/data/baseline_sim_meta.json"
    n_ingroup = 20
    n_outgroup_pops = 3
    length = 5e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    ingroup_growth_rate = 0.0
    ingroup_growth_flat_time = None
    seed = 42
    outgroup_split_times = [3e5, 9e5, 1.5e6]
    chunk_idx = None
    n_chunks = 1

# Chunked mode (chunk_idx not None): split the scenario into n_chunks
# INDEPENDENT replicates of 1/n_chunks the sequence length, each with
# distinct ancestry + mutation seeds. Per-site statistics aggregate the same
# as one monolithic sim, but each chunk's downstream inference is a small
# parallel job. chunk_idx=None keeps the original single-sim behaviour (B6).
if chunk_idx is not None:
    length = length / n_chunks
    anc_seed = seed + chunk_idx
    mut_seed = seed + 100 + chunk_idx
else:
    anc_seed = mut_seed = seed

assert len(outgroup_split_times) == n_outgroup_pops, (
    f"outgroup_split_times length {len(outgroup_split_times)} != "
    f"n_outgroup_pops {n_outgroup_pops}"
)
assert outgroup_split_times == sorted(outgroup_split_times), (
    "outgroup_split_times must be sorted ascending (closest first)"
)

demography = msprime.Demography()
# Ingroup gets its own (larger) Ne so most polymorphic sites have a
# polymorphic ingroup — otherwise outgroup-private singletons dominate
# and the benchmark mostly measures "did MAP pick the ingroup-fixed
# allele", which doesn't exercise the polarisation machinery.
# A non-zero ingroup_growth_rate gives the ingroup a recent exponential
# size trajectory (growth_rate > 0 = forward growth, an excess of rare
# variants; < 0 = forward decline, an enriched high-frequency tail), pinned
# back to a constant ancestral size at ingroup_growth_flat_time so the size
# stays bounded before the first outgroup split. Rate 0 (default) reproduces
# the constant-Ne ingroup of every existing scenario byte-for-byte.
if ingroup_growth_rate != 0.0:
    demography.add_population(name="ingroup", initial_size=ingroup_pop_size,
                             growth_rate=ingroup_growth_rate)
    if ingroup_growth_flat_time is not None:
        anc_ingroup = ingroup_pop_size * math.exp(-ingroup_growth_rate * ingroup_growth_flat_time)
        demography.add_population_parameters_change(
            time=ingroup_growth_flat_time, population="ingroup",
            growth_rate=0.0, initial_size=anc_ingroup)
else:
    demography.add_population(name="ingroup", initial_size=ingroup_pop_size)
for k in range(1, n_outgroup_pops + 1):
    demography.add_population(name=f"outgroup_{k}", initial_size=pop_size)
# Cumulative ancestor populations so we can join the outgroup ladder:
# anc_{k} is the MRCA of (ingroup + outgroups 1..k). Each outgroup_{k}
# diverges from anc_{k-1} (the ancestor of ingroup + outgroups 1..k-1)
# at outgroup_split_times[k-1].
demography.add_population(name="anc_root", initial_size=pop_size)
ancestor_chain: list[str] = ["ingroup"]
for k in range(1, n_outgroup_pops + 1):
    anc_name = "anc_root" if k == n_outgroup_pops else f"anc_{k}"
    if anc_name != "anc_root":
        demography.add_population(name=anc_name, initial_size=pop_size)
    demography.add_population_split(
        time=outgroup_split_times[k - 1],
        ancestral=anc_name,
        derived=[ancestor_chain[-1], f"outgroup_{k}"],
    )
    ancestor_chain.append(anc_name)

# One haploid sample per ingroup individual. One haploid per outgroup
# population (mirrors the est-sfs convention of single-haplotype
# outgroup proxies). Ploidy=1 → tskit ind ↔ haplotype 1:1.
samples = [msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1)]
for k in range(1, n_outgroup_pops + 1):
    samples.append(msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1))

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    sequence_length=length,
    recombination_rate=rec_rate,
    random_seed=anc_seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=mut_seed)

# Resolve sample-node id ranges from the simulator. Population names
# live in population.metadata when msprime.Demography is used.
ingroup_names: list[str] = []
outgroup_names: dict[str, list[str]] = {f"outgroup_{k}": [] for k in range(1, n_outgroup_pops + 1)}
pop_id_to_name = {p.id: p.metadata.get("name", f"pop_{p.id}") for p in ts.populations()}
for ind in ts.individuals():
    pop_id = ts.node(int(ind.nodes[0])).population
    pop_name = pop_id_to_name[pop_id]
    if pop_name == "ingroup":
        ingroup_names.append(f"tsk_{ind.id}")
    elif pop_name.startswith("outgroup_"):
        outgroup_names[pop_name].append(f"tsk_{ind.id}")

meta = {
    "ingroup_names": ingroup_names,
    "outgroups_by_pop": outgroup_names,
    "outgroup_split_times": outgroup_split_times,
    "n_ingroup": n_ingroup,
    "n_outgroup_pops": n_outgroup_pops,
    "length": length,
    "mu": mu,
    "rec_rate": rec_rate,
    "pop_size": pop_size,
    "ingroup_growth_rate": ingroup_growth_rate,
    "ingroup_growth_flat_time": ingroup_growth_flat_time,
    "seed": seed,
    "chunk_idx": chunk_idx,
    "n_chunks": n_chunks,
    "n_sites": ts.num_sites,
    "n_trees": ts.num_trees,
}
Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts.dump(out_trees)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"Wrote {out_trees}: {ts.num_sites} sites, "
    f"{ts.num_samples} haplotypes ({n_ingroup} ingroup + "
    f"{n_outgroup_pops} outgroups), {ts.num_trees} local trees",
    flush=True,
)
