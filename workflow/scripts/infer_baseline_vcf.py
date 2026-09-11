"""Outgroup-effect benchmark, fixed-tree mirror of :mod:`infer_baseline`.

Same simulation (``baseline_sim.trees`` — ingroup + 3 outgroup
populations), same closest-first outgroup subset for each ``n_out``,
but the inference path is the EST-SFS-style fixed-tree mode rather than
ARG-based. The :class:`~ancestree.priors.KingmanIngroupWeight` is
used at every ``n_out`` so the comparison across cells is fair:
empirically Stage 1's stationary-prior ML fit drives the deepest branch
rate ``K_0`` toward saturation when more than one outgroup is in the
ladder on these data, which collapses the posterior to ~uniform.

``n_out=0`` is a degenerate "fixed-tree" cell: there's no outgroup
ladder to fit, so the per-site posterior is just the (normalised)
Kingman prior on the ingroup site-frequency spectrum. It is still emitted
through this rule, giving the report the full $n_{\\mathrm{out}} \\in
\\{0,1,2,3\\}$ row set.

Wildcards:

- ``{n_out}`` ∈ ``{0, 1, 2, 3}``

Run directly (defaults to n_out=2)::

    python workflow/scripts/infer_baseline_vcf.py
"""
# No `from __future__ import annotations`: snakemake's script: directive
# prepends a preamble, which pushes it off the top of the file and makes it a
# SyntaxError. Python 3.11 evaluates `X | Y` natively, so it buys nothing here.
import json
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


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/baseline_sim.trees"
    in_meta = "results/data/baseline_sim_meta.json"
    out_json = "results/data/baseline_vcf_n2.json"
    n_out = 2
    mu = 1e-8


ts = tskit.load(in_trees)
with open(in_meta) as f:
    meta = json.load(f)

ingroup_names = list(meta["ingroup_names"])
# Different meta schemas live alongside: baseline_sim_meta.json uses
# `outgroups_by_pop` keyed by population (one haplotype per outgroup pop, by
# convention closest-first when iterating the dict insertion order);
# estsfs_sim_meta.json uses a flat `outgroup_names` list.
if "outgroup_names" in meta:
    all_outgroup_names = list(meta["outgroup_names"])
else:
    by_pop = meta["outgroups_by_pop"]
    all_outgroup_names = [name for hs in by_pop.values() for name in hs]
outgroup_names = all_outgroup_names[:n_out]
nodes = TskitLocalTree.sample_map_from_individuals(ts) or {
    f"tsk_{int(s)}": int(s) for s in ts.samples()
}

# Dump VCF for FixedTreeInference. Reuses the closest-first sample order
# the meta file declares, so n_out's prefix slice matches what the ARG
# cell sees.
ordered = ingroup_names + outgroup_names
work = Path(tempfile.mkdtemp(prefix="outgroup_vcf_"))
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

# Native path: FixedTreeInference handles outgroup_samples=[] as
# "no ladder, posterior = normalised prior on the ingroup".
vcf_inf = Inference.from_fixed_tree(
    in_vcf,
    ingroup_samples=ingroup_names,
    outgroup_samples=outgroup_names,
    model=JC69(),
    n_target_sites=int(ts.sequence_length),
    ingroup_weight=ingroup_weight,
    # match the ARG cells + est-sfs/PolarBEAR baselines
    parallelize=False,
    progress=False,
)
vcf_inf.fit()
fitted_outgroup_divergence = (
    vcf_inf.outgroup_divergence_mle.tolist()
    if n_out > 0 and vcf_inf.outgroup_divergence_mle is not None else None
)

per_site: list[dict] = []
sites_seen: list = []
for site, post_s in vcf_inf.infer():
    pos = int(site.pos)
    t = truth.get(pos)
    if t is None:
        continue
    values = [float(v) for v in post_s.values]
    p_t = float(post_s[t]) if t in post_s else 0.0
    per_site.append({
        "pos": pos,
        "map_allele": post_s.map_allele,
        "max_prob": float(post_s.max_prob),
        "posterior": values,
        "alleles": list(post_s.alleles),
        "brier": sum(v * v for v in values) - 2.0 * p_t + 1.0,
        "truth": t,
        "map_correct": post_s.map_allele == t,
    })
    sites_seen.append(site)

# B11: optional baseline_map pass — MajorityOutgroupInference on the same
# sites. Only meaningful when there's at least one outgroup.
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
brier = sum(s["brier"] for s in per_site)

result = {
    "n_out": n_out,
    "n_sites": n,
    "accuracy": (hits / n) if n else 0.0,
    "mean_brier": (brier / n) if n else 0.0,
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
    f"Outgroup-effect VCF: n_out={n_out}, {n} sites, "
    f"accuracy={result['accuracy']:.3f}, mean_brier={result['mean_brier']:.3f}",
)
