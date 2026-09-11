"""Local-tree window benchmark simulation: one neutral ingroup-only ARG.

Builds a single neutral ``msprime`` ARG with known ancestral states
(JC69 mutator) used by the appendix benchmark that probes
:class:`~ancestree.local_tree_inference.LocalTreeInference` accuracy as a
function of (a) the local-tree window size in SNPs and (b) a misspecified
recombination rate.

The default panel is ingroup-only, isolating the *quality of the inferred
local genealogy* from outgroup polarisation. Passing ``n_outgroup_pops`` > 0
with matching ascending ``outgroup_split_times`` adds an outgroup ladder
(one haploid per population, demography as in ``simulate_baseline``),
used by the appendix comparison that adds outgroups to the same benchmark.
The sample count is kept modest: the local-tree pairwise-coalescent HMM is
``O(n^2)`` in haplotypes, so a large panel would make the per-cell HMM
infeasible.

Run directly (writes a small default sim)::

    python workflow/scripts/simulate_local_tree_window.py

Or via snakemake::

    snakemake -j 1 results/data/local_tree_window_sim.trees
"""
import json
from pathlib import Path

import msprime


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    n_ingroup = int(snakemake.params.n_ingroup)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    n_outgroup_pops = int(getattr(snakemake.params, "n_outgroup_pops", 0))  # type: ignore[name-defined]
    outgroup_split_times = list(getattr(snakemake.params, "outgroup_split_times", []))  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/local_tree_window_sim.trees"
    out_meta = "results/data/local_tree_window_sim_meta.json"
    n_ingroup = 20
    length = 2e7
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    seed = 42
    n_outgroup_pops = 0
    outgroup_split_times = []

outgroups_by_pop: "dict[str, list[str]]" = {}
if n_outgroup_pops <= 0:
    # Single neutral panmictic population, one haploid sample per individual
    # (ploidy=1 → tskit ind ↔ haplotype 1:1, matching the other sims).
    ts = msprime.sim_ancestry(
        samples=n_ingroup,
        sequence_length=length,
        recombination_rate=rec_rate,
        population_size=pop_size,
        ploidy=1,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, model=msprime.JC69(), random_seed=seed)
    ingroup_names = [f"tsk_{ind.id}" for ind in ts.individuals()]
else:
    # Ingroup plus an outgroup ladder: outgroup_k diverges from the ancestor
    # of (ingroup + outgroups 1..k-1) at outgroup_split_times[k-1], one
    # haploid sample per outgroup population (mirrors simulate_baseline).
    assert len(outgroup_split_times) == n_outgroup_pops
    assert outgroup_split_times == sorted(outgroup_split_times)
    dem = msprime.Demography()
    dem.add_population(name="ingroup", initial_size=pop_size)
    for k in range(1, n_outgroup_pops + 1):
        dem.add_population(name=f"outgroup_{k}", initial_size=pop_size)
    dem.add_population(name="anc_root", initial_size=pop_size)
    chain = ["ingroup"]
    for k in range(1, n_outgroup_pops + 1):
        anc = "anc_root" if k == n_outgroup_pops else f"anc_{k}"
        if anc != "anc_root":
            dem.add_population(name=anc, initial_size=pop_size)
        dem.add_population_split(time=outgroup_split_times[k - 1], ancestral=anc,
                                 derived=[chain[-1], f"outgroup_{k}"])
        chain.append(anc)
    samples = [msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1)]
    for k in range(1, n_outgroup_pops + 1):
        samples.append(msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1))
    ts = msprime.sim_ancestry(samples=samples, demography=dem,
                              sequence_length=length, recombination_rate=rec_rate,
                              random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=mu, model=msprime.JC69(), random_seed=seed)
    pop_name = {p.id: p.metadata.get("name", f"pop_{p.id}") for p in ts.populations()}
    ingroup_names = []
    outgroups_by_pop = {f"outgroup_{k}": [] for k in range(1, n_outgroup_pops + 1)}
    for ind in ts.individuals():
        nm = pop_name[ts.node(int(ind.nodes[0])).population]
        if nm == "ingroup":
            ingroup_names.append(f"tsk_{ind.id}")
        elif nm.startswith("outgroup_"):
            outgroups_by_pop[nm].append(f"tsk_{ind.id}")

meta = {
    "ingroup_names": ingroup_names,
    "outgroups_by_pop": outgroups_by_pop,
    "outgroup_split_times": outgroup_split_times,
    "n_ingroup": n_ingroup,
    "length": length,
    "mu": mu,
    "rec_rate": rec_rate,  # the TRUE recombination rate
    "pop_size": pop_size,
    "seed": seed,
    "n_sites": int(ts.num_sites),
    "n_trees": int(ts.num_trees),
}
Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts.dump(out_trees)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
_desc = "ingroup-only" if n_outgroup_pops <= 0 else f"{n_outgroup_pops} outgroup pops"
print(
    f"Wrote {out_trees}: {ts.num_sites} sites, {ts.num_samples} haplotypes "
    f"({_desc}), {ts.num_trees} local trees",
    flush=True,
)
