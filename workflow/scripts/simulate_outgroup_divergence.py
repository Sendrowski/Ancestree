"""B8 outgroup-divergence sweep: msprime ARG at one (depth, spacing) cell.

Same demographic skeleton as B6 (``simulate_baseline``): an
ingroup population plus three outgroup populations diverging from a
ladder of ancestor populations. What this script sweeps is the
*placement* of the three outgroup splits — both the deepest-outgroup
split time (``depth``) and the relative spacing of the closer two
(``spacing``, see :mod:`_outgroup_divergence_common`).

Wildcards:

- ``{depth}`` — deepest-outgroup split time, parseable by ``float``
  (e.g. ``"1.5e6"``).
- ``{spacing}`` — one of ``linear|geometric|compact_near|compact_far``.

Run directly (defaults to ``depth=1.5e6, spacing=geometric``)::

    python workflow/scripts/simulate_outgroup_divergence.py

Or via snakemake::

    snakemake -j 1 results/data/outgroup_divergence_sim_d1.5e6_sgeometric.trees
"""
import json
import sys
from pathlib import Path

import msprime

# Snakemake's script: directive prepends the script directory to sys.path,
# so the underscore-prefixed common helper resolves cleanly. When run
# standalone (python workflow/scripts/simulate_outgroup_divergence.py)
# the same prepend pattern keeps the import working.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _outgroup_divergence_common import compute_split_times  # noqa: E402


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    depth_str = str(snakemake.wildcards.depth)  # type: ignore[name-defined]
    spacing = str(snakemake.wildcards.spacing)  # type: ignore[name-defined]
    n_ingroup = int(snakemake.params.n_ingroup)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    ingroup_pop_size = float(getattr(snakemake.params, "ingroup_pop_size", pop_size))  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
except NameError:
    depth_str = "1.5e6"
    spacing = "geometric"
    out_trees = f"results/data/outgroup_divergence_sim_d{depth_str}_s{spacing}.trees"
    out_meta = f"results/data/outgroup_divergence_sim_d{depth_str}_s{spacing}_meta.json"
    n_ingroup = 20
    length = 1e6
    mu = 1.25e-8
    rec_rate = 1e-8
    pop_size = 3e4
    ingroup_pop_size = 3e4
    seed = 42

n_outgroup_pops = 3  # fixed for B8
depth = float(depth_str)
outgroup_split_times = compute_split_times(depth, spacing)

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
    "depth": depth,
    "depth_label": depth_str,
    "spacing": spacing,
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
    f"Wrote {out_trees}: depth={depth_str}, spacing={spacing}, "
    f"splits={outgroup_split_times}, {ts.num_sites} sites, "
    f"{ts.num_samples} haplotypes, {ts.num_trees} local trees",
    flush=True,
)
