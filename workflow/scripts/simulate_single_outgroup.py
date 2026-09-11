"""Single-outgroup depth sweep: msprime ARG with an ingroup plus one outgroup.

The demographic skeleton is the ingroup population and a single outgroup
population splitting from it at ``{depth}`` generations. With one outgroup
there is no relative spacing to choose, so the depth is the only axis.

Wildcards:

- ``{depth}`` — outgroup split time in generations, parseable by ``float``
  (e.g. ``"2.5e5"``).

Run directly (defaults to ``depth=1e6``)::

    python workflow/scripts/simulate_single_outgroup.py

Or via snakemake::

    snakemake -j 1 results/data/single_outgroup_sim_d1e6.trees
"""
import json
from pathlib import Path

import msprime


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    depth_str = str(snakemake.wildcards.depth)  # type: ignore[name-defined]
    n_ingroup = int(snakemake.params.n_ingroup)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    ingroup_pop_size = float(getattr(snakemake.params, "ingroup_pop_size", pop_size))  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
except NameError:
    depth_str = "1e6"
    out_trees = f"results/data/single_outgroup_sim_d{depth_str}.trees"
    out_meta = f"results/data/single_outgroup_sim_d{depth_str}_meta.json"
    n_ingroup = 20
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    seed = 42

n_outgroup_pops = 1
depth = float(depth_str)
outgroup_split_times = [depth]

demography = msprime.Demography()
demography.add_population(name="ingroup", initial_size=ingroup_pop_size)
demography.add_population(name="outgroup_1", initial_size=pop_size)
demography.add_population(name="anc_root", initial_size=pop_size)
demography.add_population_split(
    time=depth, ancestral="anc_root", derived=["ingroup", "outgroup_1"],
)

samples = [
    msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1),
    msprime.SampleSet(1, population="outgroup_1", ploidy=1),
]

ts = msprime.sim_ancestry(
    samples=samples,
    demography=demography,
    sequence_length=length,
    recombination_rate=rec_rate,
    random_seed=seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)

ingroup_names: list[str] = []
outgroup_names: dict[str, list[str]] = {"outgroup_1": []}
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
    "depth": depth,
    "depth_label": depth_str,
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
    f"Wrote {out_trees}: depth={depth_str}, split={depth}, "
    f"{ts.num_sites} sites, {ts.num_samples} haplotypes, "
    f"{ts.num_trees} local trees",
    flush=True,
)
