"""B9 model-comparison benchmark simulation: HKY mutations with non-uniform π and κ ≠ 1.

Builds one tree sequence with an ingroup population plus three outgroup
populations at progressively deeper divergence times (same demography as
B6 :mod:`simulate_baseline`), but mutates it under msprime's
:class:`msprime.HKY` model with both a transition/transversion bias
(``kappa=4``) and a non-uniform AT-rich equilibrium distribution
(``π = (0.30, 0.20, 0.20, 0.30)`` over ``(A, C, G, T)``). Downstream
inference cells then probe how badly each Ancestree substitution model
(JC69, K2, F81, HKY) recovers the simulator's truth under this combined
misspecification challenge.

The simulator's true ancestral state per site is read directly from the
ts (``ts.sites()[i].ancestral_state``) and persisted in the meta JSON
indirectly — the report scripts re-load the ts and read site truth
directly, matching the B6/B8 convention.

Run directly (writes a small default sim)::

    python workflow/scripts/simulate_model_comparison.py

Or via snakemake::

    snakemake -j 1 results/data/model_comparison_sim.trees
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
    kappa = float(snakemake.params.kappa)  # type: ignore[name-defined]
    equilibrium_frequencies = list(snakemake.params.equilibrium_frequencies)  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/model_comparison_sim.trees"
    out_meta = "results/data/model_comparison_sim_meta.json"
    n_ingroup = 20
    n_outgroup_pops = 3
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    seed = 42
    outgroup_split_times = [3e5, 9e5, 1.5e6]
    kappa = 4.0
    equilibrium_frequencies = [0.30, 0.20, 0.20, 0.30]

assert len(outgroup_split_times) == n_outgroup_pops, (
    f"outgroup_split_times length {len(outgroup_split_times)} != "
    f"n_outgroup_pops {n_outgroup_pops}"
)
assert outgroup_split_times == sorted(outgroup_split_times), (
    "outgroup_split_times must be sorted ascending (closest first)"
)
assert abs(sum(equilibrium_frequencies) - 1.0) < 1e-9, (
    f"equilibrium_frequencies must sum to 1.0, got {sum(equilibrium_frequencies)}"
)
assert len(equilibrium_frequencies) == 4, (
    f"equilibrium_frequencies must be length 4 (A,C,G,T), got {len(equilibrium_frequencies)}"
)


# ----- Demography (mirrors simulate_baseline.py exactly) -----

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

# Drop mutations under HKY(kappa, π). msprime's HKY rate matrix scales
# rates by ``mu`` directly. We pass kappa + the AT-rich equilibrium
# frequencies. The simulator records each site's true ancestral state
# in ts.sites()[i].ancestral_state (same as B6 / B8).
mutation_model = msprime.HKY(
    kappa=kappa,
    equilibrium_frequencies=equilibrium_frequencies,
)
ts = msprime.sim_mutations(ts, rate=mu, model=mutation_model, random_seed=seed)

# ----- Resolve per-population sample names (matches B6's meta schema) -----

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
    # B9-specific truth bookkeeping. The downstream report uses these to
    # label the misspecification axis it's penalising in each (model, mode)
    # cell.
    "mutation_model": "HKY",
    "kappa": kappa,
    "equilibrium_frequencies": equilibrium_frequencies,
}
Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts.dump(out_trees)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"Wrote {out_trees}: {ts.num_sites} sites, {ts.num_samples} haplotypes "
    f"({n_ingroup} ingroup + {n_outgroup_pops} outgroups), "
    f"{ts.num_trees} local trees; mutated under HKY(kappa={kappa}, "
    f"π={equilibrium_frequencies})",
    flush=True,
)
