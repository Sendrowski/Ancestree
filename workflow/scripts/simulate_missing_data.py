"""B12 missing-data sweep simulation: msprime ARG with ingroup + 3 outgroups.

Same B6-style demography as :mod:`simulate_baseline`. No
masking is done here. The per-cell inference script applies random
outgroup-tip nulling at inference time (one mask per (missing_frac,
seed) cell), so a single ``.trees`` feeds the whole sweep.

Run directly (writes a small default sim)::

    python workflow/scripts/simulate_missing_data.py

Or via snakemake::

    snakemake -j 1 results/data/missing_data_sim.trees
"""
import json
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
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    outgroup_split_times = list(snakemake.params.outgroup_split_times)  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/missing_data_sim.trees"
    out_meta = "results/data/missing_data_sim_meta.json"
    n_ingroup = 20
    n_outgroup_pops = 3
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    seed = 42
    outgroup_split_times = [3e5, 9e5, 1.5e6]

assert len(outgroup_split_times) == n_outgroup_pops, (
    f"outgroup_split_times length {len(outgroup_split_times)} != "
    f"n_outgroup_pops {n_outgroup_pops}"
)
assert outgroup_split_times == sorted(outgroup_split_times), (
    "outgroup_split_times must be sorted ascending (closest first)"
)

demography = msprime.Demography()
demography.add_population(name="ingroup", initial_size=ingroup_pop_size)
for k in range(1, n_outgroup_pops + 1):
    demography.add_population(name=f"outgroup_{k}", initial_size=pop_size)
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

samples = [msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1)]
for k in range(1, n_outgroup_pops + 1):
    samples.append(msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1))

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    sequence_length=length,
    recombination_rate=rec_rate,
    random_seed=seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)

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
    f"{ts.num_samples} haplotypes ({n_ingroup} ingroup + "
    f"{n_outgroup_pops} outgroups), {ts.num_trees} local trees",
    flush=True,
)
