"""B8 outgroup-divergence sweep: fixed-tree (VCF) inference for one (depth, spacing, n_out) cell.

Fixed-tree mirror of :mod:`infer_outgroup_divergence_arg`. Loads the
matching ``outgroup_divergence_sim_d{depth}_s{spacing}.trees``, subsets
the outgroups down to ``{n_out}`` populations with
:func:`_outgroup_divergence_common.select_outgroup_pops`, writes a
temporary VCF over (ingroup + selected outgroups), and runs
:class:`~ancestree.inference.FixedTreeInference` (via
:meth:`Inference.from_fixed_tree`) with those outgroups in the ladder.
Per-site JSON schema matches B6's ``baseline_vcf_n3.json`` so the
report can reuse the same accuracy aggregation.

Wildcards:

- ``{depth}`` — e.g. ``"1.5e6"``.
- ``{spacing}`` — one of ``linear|geometric|compact_near|compact_far``.
- ``{n_out}`` — one of ``1|2|3`` outgroup populations.

Run directly (defaults to ``depth=1.5e6, spacing=geometric, n_out=3``)::

    python workflow/scripts/infer_outgroup_divergence_vcf.py
"""
import json
import sys
import tempfile
from pathlib import Path

import tskit

from ancestree import (
    Inference,
    JC69,
    KingmanIngroupWeight,
    MajorityOutgroupInference,
    TskitLocalTree,
)

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
        f"results/data/outgroup_divergence_vcf_d{depth_str}_s{spacing}_n{n_out}.json"
    )
    mu = 1.25e-8


ts = tskit.load(in_trees)
with open(in_meta) as f:
    meta = json.load(f)

ingroup_names = list(meta["ingroup_names"])
by_pop = meta["outgroups_by_pop"]
# Closest-first ordering matches the simulator's "outgroup_1 is nearest"
# convention, and fixes the ladder direction for OutgroupLadderTree.
all_split_times = list(meta["outgroup_split_times"])
ordered_pops = sorted(by_pop.keys(), key=lambda s: int(s.split("_")[1]))
selected_pops = select_outgroup_pops(ordered_pops, n_out)
outgroup_names = [name for pop in selected_pops for name in by_pop[pop]]
selected_split_times = [
    all_split_times[ordered_pops.index(pop)] for pop in selected_pops
]
nodes = TskitLocalTree.sample_map_from_individuals(ts) or {
    f"tsk_{int(s)}": int(s) for s in ts.samples()
}

ordered = ingroup_names + outgroup_names
work = Path(tempfile.mkdtemp(prefix="outgroup_div_vcf_"))
in_vcf = work / "snps.vcf"
with open(in_vcf, "w") as f:
    ts.write_vcf(
        f,
        contig_id="1",
        individual_names=ordered,
        individuals=[nodes[n] for n in ordered],
        allow_position_zero=True,
    )

ingroup_weight = KingmanIngroupWeight(ingroup_samples=ingroup_names)
truth = {int(s.position): s.ancestral_state for s in ts.sites()}

vcf_inf = Inference.from_fixed_tree(
    in_vcf,
    ingroup_samples=ingroup_names,
    outgroup_samples=outgroup_names,
    model=JC69(),
    n_target_sites=int(ts.sequence_length),
    ingroup_weight=ingroup_weight,
    parallelize=False,
    progress=False,
)
vcf_inf.fit()
fitted_outgroup_divergence = (
    vcf_inf.outgroup_divergence_mle.tolist()
    if vcf_inf.outgroup_divergence_mle is not None else None
)

ingroup_name_set = set(ingroup_names)

per_site: list[dict] = []
sites_seen: list = []
for site, post_s in vcf_inf.infer():
    pos = int(site.pos)
    t = truth.get(pos)
    if t is None:
        continue
    ingroup_allele_counts: dict[str, int] = {}
    for s, a in site.tip_alleles.items():
        if s in ingroup_name_set and a is not None:
            ingroup_allele_counts[a] = ingroup_allele_counts.get(a, 0) + 1
    ingroup_polymorphic = len(ingroup_allele_counts) > 1
    if ingroup_polymorphic:
        sorted_counts = sorted(ingroup_allele_counts.values(), reverse=True)
        minor_count = int(sorted_counts[1])
    else:
        minor_count = 0
    per_site.append({
        "pos": pos,
        "map_allele": post_s.map_allele,
        "max_prob": float(post_s.max_prob),
        "p_true": float(post_s[t]),
        "posterior": [float(x) for x in post_s.values],
        "alleles": list(post_s.alleles) if hasattr(post_s, "alleles") else None,
        "truth": t,
        "map_correct": post_s.map_allele == t,
        "ingroup_polymorphic": bool(ingroup_polymorphic),
        "minor_count": minor_count,
    })
    sites_seen.append(site)

# B11: optional baseline_map pass — MajorityOutgroupInference on the same
# sites. Stored under "baseline_map" alongside the model-based MAP so the
# report can compute agreement rates.
if outgroup_names and sites_seen:
    baseline = MajorityOutgroupInference(
        sites_seen, outgroup_names, for_comparison_only=True,
    )
    pos_to_entry = {entry["pos"]: entry for entry in per_site}
    for site, post_b in baseline.infer():
        entry = pos_to_entry.get(int(site.pos))
        if entry is not None:
            entry["baseline_map"] = post_b.map_allele

n = len(per_site)
hits = sum(int(s["map_correct"]) for s in per_site)
soft = sum(s["p_true"] for s in per_site)

result = {
    "depth": float(depth_str),
    "depth_label": depth_str,
    "spacing": spacing,
    "n_out": n_out,
    "outgroup_split_times": selected_split_times,
    "all_outgroup_split_times": all_split_times,
    "outgroup_pops_used": selected_pops,
    "n_sites": n,
    "accuracy": (hits / n) if n else 0.0,
    "mean_p_true": (soft / n) if n else 0.0,
    "mean_max_prob": (sum(s["max_prob"] for s in per_site) / n) if n else 0.0,
    "outgroup_names_used": outgroup_names,
    "ingroup_weight": ingroup_weight.__class__.__name__,
    "mode": "vcf",
    "fitted_outgroup_divergence": fitted_outgroup_divergence,
    "per_site": per_site,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)

print(
    f"Outgroup-divergence VCF: depth={depth_str}, spacing={spacing}, "
    f"n_out={n_out} ({','.join(selected_pops)}), "
    f"{n} sites, accuracy={result['accuracy']:.3f}, "
    f"mean_p_true={result['mean_p_true']:.3f}",
    flush=True,
)
