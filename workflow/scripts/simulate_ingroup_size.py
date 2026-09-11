"""Ingroup-size sweep benchmark simulation: msprime ARG with a large ingroup-only sample.

Builds one tree sequence with ``n_ingroup_max`` haploid samples drawn
from a single ingroup population (no outgroup populations — B7 only
tests ARG ingroup-only mode, i.e. ``n_out=0``). Inference downstream
subsamples the tree sequence via ``ts.simplify(samples=...)`` to study
how per-bin SFS accuracy scales with the number of ingroup haplotypes
available to the Felsenstein kernel.

Run directly (writes a small default sim)::

    python workflow/scripts/simulate_ingroup_size.py

Or via snakemake::

    snakemake -j 1 results/data/ingroup_size_sim.trees
"""
import json
from pathlib import Path

import msprime


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    n_ingroup_max = int(snakemake.params.n_ingroup_max)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/ingroup_size_sim.trees"
    out_meta = "results/data/ingroup_size_sim_meta.json"
    n_ingroup_max = 1000
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    seed = 42


demography = msprime.Demography()
demography.add_population(name="ingroup", initial_size=pop_size)

# One haploid per ingroup individual. Ploidy=1 keeps the tskit
# individual ↔ haplotype mapping 1:1 (mirrors B6's convention).
samples = [msprime.SampleSet(n_ingroup_max, population="ingroup", ploidy=1)]

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    sequence_length=length,
    recombination_rate=rec_rate,
    random_seed=seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)

# Resolve ingroup sample names from the simulator. With a single
# population and ploidy=1 the individual id is the haplotype id.
ingroup_names: list[str] = [f"tsk_{ind.id}" for ind in ts.individuals()]

meta = {
    "ingroup_names": ingroup_names,
    "n_ingroup_max": n_ingroup_max,
    "length": length,
    "mu": mu,
    "rec_rate": rec_rate,
    "pop_size": pop_size,
    "seed": seed,
    "n_sites": ts.num_sites,
    "n_trees": ts.num_trees,
}
Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts.dump(out_trees)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"Wrote {out_trees}: {ts.num_sites} sites, "
    f"{ts.num_samples} haplotypes ({n_ingroup_max} ingroup, no outgroups), "
    f"{ts.num_trees} local trees",
    flush=True,
)
