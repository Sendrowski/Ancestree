"""Single-outgroup depth sweep: ARG-mode inference for one ``{depth}`` cell.

Loads the matching ``single_outgroup_sim_d{depth}.trees`` (ingroup plus one
outgroup population) and runs
:class:`~ancestree.inference.ARGBasedInference` over it. The per-site JSON
schema matches the B8 ARG cells so the report can reuse the same site
iteration and scoring logic.

Wildcards:

- ``{depth}`` — outgroup split time in generations, e.g. ``"2.5e5"``.

Run directly (defaults to ``depth=1e6``)::

    python workflow/scripts/infer_single_outgroup_arg.py
"""
import json
from pathlib import Path

import tskit

from ancestree import ARGBasedInference, JC69


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    depth_str = str(snakemake.wildcards.depth)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    depth_str = "1e6"
    in_trees = f"results/data/single_outgroup_sim_d{depth_str}.trees"
    in_meta = f"results/data/single_outgroup_sim_d{depth_str}_meta.json"
    out_json = f"results/data/single_outgroup_arg_d{depth_str}.json"
    mu = 1.25e-8


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
split_times: list[float] = list(meta["outgroup_split_times"])
outgroup_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
outgroup_names = [name for pop in outgroup_pops for name in outgroups_by_pop[pop]]

ts_full = tskit.load(in_trees)
sample_id_to_node = {f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts_full.individuals()}
keep_sample_names = ingroup_names + outgroup_names
keep_node_ids = [sample_id_to_node[s] for s in keep_sample_names]
# filter_sites=False keeps every site (incl. outgroup-private) so the site
# set matches across depths. Ingroup-polymorphic filtering is done in the
# report stage.
ts_sub = ts_full.simplify(samples=keep_node_ids, filter_sites=False)
sample_map = {s: i for i, s in enumerate(keep_sample_names)}

print(
    f"Single-outgroup ARG: depth={depth_str}, "
    f"{ts_sub.num_samples} haplotypes, {ts_sub.num_sites} sites, "
    f"{ts_sub.num_trees} local trees",
    flush=True,
)

inference = ARGBasedInference(
    ts_sub, JC69(),
    mu=mu,
    sample_map=sample_map,
    progress=False,
)

truth_by_pos: dict[int, str] = {
    int(site.position): site.ancestral_state for site in ts_full.sites()
}
ingroup_name_set = set(ingroup_names)

results: dict[int, dict] = {}
for site, posterior in inference.infer():
    truth = truth_by_pos.get(int(site.pos))
    # Ingroup polymorphism: more than one distinct non-missing tip allele
    # across the ingroup samples.
    ingroup_allele_counts: dict[str, int] = {}
    for s, a in site.tip_alleles.items():
        if s in ingroup_name_set and a is not None:
            ingroup_allele_counts[a] = ingroup_allele_counts.get(a, 0) + 1
    ingroup_polymorphic = len(ingroup_allele_counts) > 1
    if ingroup_polymorphic:
        minor_count = int(sorted(ingroup_allele_counts.values(), reverse=True)[1])
    else:
        minor_count = 0
    results[int(site.pos)] = {
        "map_allele": posterior.map_allele,
        "max_prob": float(posterior.max_prob),
        "posterior": [float(p) for p in posterior.values],
        "alleles": list(site.alleles),
        "truth": truth,
        "map_correct": truth is not None and posterior.map_allele == truth,
        "ingroup_polymorphic": bool(ingroup_polymorphic),
        "minor_count": minor_count,
    }

inference_meta = {
    "depth": float(depth_str),
    "depth_label": depth_str,
    "n_out": 1,
    "outgroup_split_times": split_times,
    "outgroup_pops_used": outgroup_pops,
    "outgroup_names_used": outgroup_names,
    "n_ingroup": len(ingroup_names),
    "n_sites_inferred": len(results),
    "mu": mu,
    "mode": "arg",
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
