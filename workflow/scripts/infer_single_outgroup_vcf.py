"""Single-outgroup depth sweep: fixed-tree (VCF) inference for one ``{depth}`` cell.

Fixed-tree mirror of :mod:`infer_single_outgroup_arg`. Loads the matching
``single_outgroup_sim_d{depth}.trees``, writes a temporary VCF over the
ingroup plus the one outgroup, and runs
:class:`~ancestree.inference.FixedTreeInference` via
:meth:`Inference.from_fixed_tree() <ancestree.inference.Inference.from_fixed_tree>`.
The per-site JSON schema matches the B8 VCF cells.

Wildcards:

- ``{depth}`` — outgroup split time in generations, e.g. ``"2.5e5"``.

Run directly (defaults to ``depth=1e6``)::

    python workflow/scripts/infer_single_outgroup_vcf.py
"""
import json
import tempfile
from pathlib import Path

import tskit

from ancestree import (
    Inference,
    JC69,
    KingmanIngroupWeight,
    TskitLocalTree,
)


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
    out_json = f"results/data/single_outgroup_vcf_d{depth_str}.json"
    mu = 1.25e-8


ts = tskit.load(in_trees)
with open(in_meta) as f:
    meta = json.load(f)

ingroup_names = list(meta["ingroup_names"])
by_pop = meta["outgroups_by_pop"]
split_times = list(meta["outgroup_split_times"])
outgroup_pops = sorted(by_pop.keys(), key=lambda s: int(s.split("_")[1]))
outgroup_names = [name for pop in outgroup_pops for name in by_pop[pop]]
nodes = TskitLocalTree.sample_map_from_individuals(ts) or {
    f"tsk_{int(s)}": int(s) for s in ts.samples()
}

ordered = ingroup_names + outgroup_names
work = Path(tempfile.mkdtemp(prefix="single_outgroup_vcf_"))
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
        minor_count = int(sorted(ingroup_allele_counts.values(), reverse=True)[1])
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

n = len(per_site)
hits = sum(int(s["map_correct"]) for s in per_site)
soft = sum(s["p_true"] for s in per_site)

result = {
    "depth": float(depth_str),
    "depth_label": depth_str,
    "n_out": 1,
    "outgroup_split_times": split_times,
    "outgroup_pops_used": outgroup_pops,
    "outgroup_names_used": outgroup_names,
    "n_sites": n,
    "accuracy": (hits / n) if n else 0.0,
    "mean_p_true": (soft / n) if n else 0.0,
    "mean_max_prob": (sum(s["max_prob"] for s in per_site) / n) if n else 0.0,
    "ingroup_weight": ingroup_weight.__class__.__name__,
    "mode": "vcf",
    "fitted_outgroup_divergence": fitted_outgroup_divergence,
    "per_site": per_site,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)

print(
    f"Single-outgroup VCF: depth={depth_str}, {n} sites, "
    f"accuracy={result['accuracy']:.3f}, "
    f"mean_p_true={result['mean_p_true']:.3f}",
    flush=True,
)
