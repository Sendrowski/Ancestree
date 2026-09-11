"""Outgroup-effect benchmark inference: ARGBasedInference with N outgroups in the local tree.

Loads the full simulated ts (ingroup + ``n_outgroup_pops`` outgroups)
and simplifies it down to ingroup + the first ``n_out`` outgroups
(closest-first), then runs :class:`ARGBasedInference` and dumps per-site
posteriors plus the simulator's truth ancestral state for downstream
comparison.

Wildcards:

- ``{n_out}`` ∈ ``{0, 1, 2, ...}`` — number of outgroups included in
  the local tree fed to Felsenstein.

Run directly (defaults to n_out=2)::

    python workflow/scripts/infer_baseline.py

Or via snakemake::

    snakemake -j 1 results/data/baseline_n2.json
"""
import json
from pathlib import Path

import tskit

from ancestree import (
    ARGBasedInference,
    JC69,
    MajorityOutgroupInference,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/baseline_sim.trees"
    in_meta = "results/data/baseline_sim_meta.json"
    out_json = "results/data/baseline_n2.json"
    n_out = 2
    mu = 1e-8


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
# Closest-first by construction in the simulator (outgroup_1 = nearest).
ordered_outgroup_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
selected_outgroup_names: list[str] = []
for pop_name in ordered_outgroup_pops[:n_out]:
    selected_outgroup_names.extend(outgroups_by_pop[pop_name])

ts_full = tskit.load(in_trees)
# Map sample id → tskit node id (one haploid sample per individual under
# the simulator's ploidy=1, so the mapping is 1:1 by individual id).
sample_id_to_node: dict[str, int] = {}
for ind in ts_full.individuals():
    sid = f"tsk_{ind.id}"
    sample_id_to_node[sid] = int(ind.nodes[0])

keep_sample_names = ingroup_names + selected_outgroup_names
keep_node_ids = [sample_id_to_node[s] for s in keep_sample_names]
ts_sub = ts_full.simplify(samples=keep_node_ids, filter_sites=False)
# After simplify the kept samples are re-numbered 0..len(keep)-1 in
# the same order they were passed in. Rebuild the name-to-node map.
sample_map = {s: i for i, s in enumerate(keep_sample_names)}

print(
    f"Outgroup-effect inference: n_out={n_out}, "
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

# Pre-index truth ancestral states by site position for cheap lookup.
truth_by_pos: dict[int, str] = {
    int(site.position): site.ancestral_state for site in ts_full.sites()
}

results: dict[int, dict] = {}
sites_seen: list = []
for site, posterior in inference.infer():
    truth = truth_by_pos.get(int(site.pos))
    results[int(site.pos)] = {
        "map_allele": posterior.map_allele,
        "max_prob": float(posterior.max_prob),
        "posterior": [float(p) for p in posterior.values],
        "alleles": list(site.alleles),
        "truth": truth,
        "map_correct": truth is not None and posterior.map_allele == truth,
    }
    sites_seen.append(site)

# B11: optional baseline_map pass — MajorityOutgroupInference on the same
# sites. Only meaningful when there's at least one outgroup. The n_out=0
# cell skips it cleanly (the baseline class requires >=1 outgroup).
if selected_outgroup_names:
    baseline = MajorityOutgroupInference(
        sites_seen, selected_outgroup_names, for_comparison_only=True,
    )
    for site, post_b in baseline.infer():
        entry = results.get(int(site.pos))
        if entry is not None:
            entry["baseline_map"] = post_b.map_allele

inference_meta = {
    "n_out": n_out,
    "outgroup_pops_used": ordered_outgroup_pops[:n_out],
    "outgroup_names_used": selected_outgroup_names,
    "n_ingroup": len(ingroup_names),
    "n_sites_inferred": len(results),
    "mu": mu,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
