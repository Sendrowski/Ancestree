"""Outgroup-CLADE scenario: outgroups coalesce with each other before the ingroup.

A deliberate, serious breach of the outgroup-*ladder* topology that
``FixedTreeInference``'s :class:`OutgroupLadderTree` hard-codes. Here the
three outgroup lineages form their own clade — they coalesce among
themselves at shallow/moderate times, and the whole outgroup clade joins
the ingroup only at the deepest split:

    ( ingroup , ( (out_2, out_3) , out_1 ) )

The ladder model assumes instead that each successive outgroup branches
off the ingroup lineage one at a time, so on this topology its assumed
tree is simply wrong. Local-tree mode infers the actual topology per
window, so this scenario is where its adaptivity should pay off.

Output format matches ``simulate_baseline.py`` (``ingroup_names`` +
``outgroups_by_pop`` + times) so the robustness harness loads it
unchanged. Every outgroup diverges from the ingroup at the same deep time
(they're a clade), so the closest-first ``n_out`` subset is arbitrary
among them.

Run directly::

    python workflow/scripts/simulate_outgroup_clade.py
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
    among_outgroup_times = list(snakemake.params.among_outgroup_times)  # type: ignore[name-defined]
    ingroup_outgroup_split = float(snakemake.params.ingroup_outgroup_split)  # type: ignore[name-defined]
    _chunk_idx = getattr(snakemake.params, "chunk_idx", None)  # type: ignore[name-defined]
    chunk_idx = None if _chunk_idx is None else int(_chunk_idx)
    n_chunks = int(getattr(snakemake.params, "n_chunks", 1))  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/outgroup_clade_sim.trees"
    out_meta = "results/data/outgroup_clade_sim_meta.json"
    n_ingroup = 20
    n_outgroup_pops = 3
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    seed = 42
    # among-outgroup coalescences (shallow), then the clade joins the
    # ingroup at ingroup_outgroup_split (deep). All << the deep split.
    among_outgroup_times = [5e4, 1e5]  # (out2,out3) then +out1
    ingroup_outgroup_split = 3e5  # whole clade ↔ ingroup
    chunk_idx = None
    n_chunks = 1

# Chunked mode: see simulate_baseline.py — n_chunks independent
# replicates at 1/n_chunks the length with distinct seeds. Chunk_idx=None
# keeps the single-sim behaviour.
if chunk_idx is not None:
    length = length / n_chunks
    anc_seed = seed + chunk_idx
    mut_seed = seed + 100 + chunk_idx
else:
    anc_seed = mut_seed = seed

assert n_outgroup_pops >= 2, "outgroup-clade scenario needs >=2 outgroups"
assert len(among_outgroup_times) == n_outgroup_pops - 1
assert max(among_outgroup_times) < ingroup_outgroup_split, (
    "outgroups must coalesce with each other strictly before the ingroup"
)

# Caterpillar clade among the outgroups: outgroup_1 and outgroup_2 coalesce
# first, then each successive outgroup joins that growing clade, all before
# the whole clade joins the ingroup at the deep split:
#   ( ingroup , ( … ((o1,o2),o3) … , o_n ) )
demography = msprime.Demography()
demography.add_population(name="ingroup", initial_size=ingroup_pop_size)
for k in range(1, n_outgroup_pops + 1):
    demography.add_population(name=f"outgroup_{k}", initial_size=pop_size)
# n-1 internal among-outgroup ancestors out_anc_2 … out_anc_n + the root.
for k in range(2, n_outgroup_pops + 1):
    demography.add_population(name=f"out_anc_{k}", initial_size=pop_size)
demography.add_population(name="anc_root", initial_size=pop_size)
among_sorted = sorted(among_outgroup_times)
# out_anc_2 = (outgroup_1, outgroup_2). Out_anc_k = (out_anc_{k-1}, outgroup_k)
demography.add_population_split(time=among_sorted[0], ancestral="out_anc_2",
                                derived=["outgroup_1", "outgroup_2"])
for k in range(3, n_outgroup_pops + 1):
    demography.add_population_split(
        time=among_sorted[k - 2], ancestral=f"out_anc_{k}",
        derived=[f"out_anc_{k - 1}", f"outgroup_{k}"],
    )
demography.add_population_split(
    time=ingroup_outgroup_split, ancestral="anc_root",
    derived=["ingroup", f"out_anc_{n_outgroup_pops}"],
)

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
    # all outgroups diverge from the ingroup at the same deep split (clade)
    "outgroup_split_times": [ingroup_outgroup_split] * n_outgroup_pops,
    "among_outgroup_times": among_sorted,
    "ingroup_outgroup_split": ingroup_outgroup_split,
    "topology": "outgroup_clade",
    "n_ingroup": n_ingroup,
    "n_outgroup_pops": n_outgroup_pops,
    "length": length,
    "mu": mu,
    "rec_rate": rec_rate,
    "pop_size": pop_size,
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
    f"Wrote {out_trees}: {ts.num_sites} sites, outgroup CLADE "
    f"({n_outgroup_pops} outgroups, among-outgroup "
    f"{among_sorted[0]:g}…{among_sorted[-1]:g}, "
    f"ingroup split {ingroup_outgroup_split:g})",
    flush=True,
)
