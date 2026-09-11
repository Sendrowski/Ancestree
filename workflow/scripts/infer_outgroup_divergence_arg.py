"""B8 outgroup-divergence sweep: ARG-mode inference for one (depth, spacing, n_out) cell.

Loads the matching ``outgroup_divergence_sim_d{depth}_s{spacing}.trees``
(ingroup + 3 outgroup populations at the sweep's split-time triple),
subsets the outgroups down to ``{n_out}`` populations with
:func:`_outgroup_divergence_common.select_outgroup_pops`, and runs
:class:`~ancestree.inference.ARGBasedInference` on the simplified tree
sequence. Mirrors the per-site JSON schema of B6's ``baseline_n3.json``
so the report script can reuse the same site iteration / accuracy logic.

Wildcards:

- ``{depth}`` — e.g. ``"1.5e6"``.
- ``{spacing}`` — one of ``linear|geometric|compact_near|compact_far``.
- ``{n_out}`` — one of ``1|2|3`` outgroup populations.

Run directly (defaults to ``depth=1.5e6, spacing=geometric, n_out=3``)::

    python workflow/scripts/infer_outgroup_divergence_arg.py
"""
import json
import sys
from pathlib import Path

import tskit

from ancestree import ARGBasedInference, JC69, MajorityOutgroupInference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _outgroup_divergence_common import select_outgroup_pops  # noqa: E402


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    depth_str = str(snakemake.wildcards.depth)  # type: ignore[name-defined]
    spacing = str(snakemake.wildcards.spacing)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    depth_str = "1.5e6"
    spacing = "geometric"
    n_out = 3
    in_trees = f"results/data/outgroup_divergence_sim_d{depth_str}_s{spacing}.trees"
    in_meta = f"results/data/outgroup_divergence_sim_d{depth_str}_s{spacing}_meta.json"
    out_json = (
        f"results/data/outgroup_divergence_arg_d{depth_str}_s{spacing}_n{n_out}.json"
    )
    mu = 1.25e-8


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
all_split_times: list[float] = list(meta["outgroup_split_times"])
ordered_outgroup_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
selected_outgroup_pops = select_outgroup_pops(ordered_outgroup_pops, n_out)
selected_outgroup_names: list[str] = []
for pop_name in selected_outgroup_pops:
    selected_outgroup_names.extend(outgroups_by_pop[pop_name])
selected_split_times = [
    all_split_times[ordered_outgroup_pops.index(pop)] for pop in selected_outgroup_pops
]

ts_full = tskit.load(in_trees)
sample_id_to_node: dict[str, int] = {}
for ind in ts_full.individuals():
    sid = f"tsk_{ind.id}"
    sample_id_to_node[sid] = int(ind.nodes[0])

keep_sample_names = ingroup_names + selected_outgroup_names
keep_node_ids = [sample_id_to_node[s] for s in keep_sample_names]
# filter_sites=False keeps every site (incl. outgroup-private) so the
# site set matches across cells. Ingroup-polymorphic filtering is done
# in the report stage.
ts_sub = ts_full.simplify(samples=keep_node_ids, filter_sites=False)
sample_map = {s: i for i, s in enumerate(keep_sample_names)}

print(
    f"Outgroup-divergence ARG: depth={depth_str}, spacing={spacing}, "
    f"n_out={n_out} ({','.join(selected_outgroup_pops)}), "
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
sites_seen: list = []
for site, posterior in inference.infer():
    truth = truth_by_pos.get(int(site.pos))
    # Ingroup polymorphism: more than one distinct non-missing tip allele
    # across the ingroup samples. Restricting the accuracy summary to
    # ingroup-polymorphic sites matches B6's reporting convention.
    ingroup_allele_counts: dict[str, int] = {}
    for s, a in site.tip_alleles.items():
        if s in ingroup_name_set and a is not None:
            ingroup_allele_counts[a] = ingroup_allele_counts.get(a, 0) + 1
    ingroup_polymorphic = len(ingroup_allele_counts) > 1
    # Minor allele count: second-largest tip count. 0 if monomorphic.
    if ingroup_polymorphic:
        sorted_counts = sorted(ingroup_allele_counts.values(), reverse=True)
        minor_count = int(sorted_counts[1])
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
    sites_seen.append(site)

# B11: optional baseline_map pass — MajorityOutgroupInference on the same
# sites. The baseline needs at least one outgroup, which B8 always has.
if selected_outgroup_names and sites_seen:
    baseline = MajorityOutgroupInference(
        sites_seen, selected_outgroup_names, for_comparison_only=True,
    )
    for site, post_b in baseline.infer():
        entry = results.get(int(site.pos))
        if entry is not None:
            entry["baseline_map"] = post_b.map_allele

inference_meta = {
    "depth": float(depth_str),
    "depth_label": depth_str,
    "spacing": spacing,
    "n_out": n_out,
    "outgroup_split_times": selected_split_times,
    "all_outgroup_split_times": all_split_times,
    "outgroup_pops_used": selected_outgroup_pops,
    "outgroup_names_used": selected_outgroup_names,
    "n_ingroup": len(ingroup_names),
    "n_sites_inferred": len(results),
    "mu": mu,
    "mode": "arg",
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
