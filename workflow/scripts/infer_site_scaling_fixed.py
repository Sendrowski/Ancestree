"""B14: FixedTreeInference parameter recovery vs n_sites.

Reads the site-scaling ARG, slices the ingroup down to the first
``n_ingroup_used`` haplotypes (matches the sim's max of 20) and the
outgroup list down to the closest ``n_out`` outgroups, then randomly
sub-samples the polymorphic-site catalog to ``n_sites`` with the
per-cell RNG ``seed`` and fits :class:`FixedTreeInference` under the
requested substitution model. The fitted branch rates ``K_i`` and (when
``model = HKY``) ``kappa`` are written to JSON for the report to compare
against the reference / full-data fit.

To preserve the polymorphism rate across cells, ``n_target_sites`` is
scaled in proportion with the kept-site count:
``n_target_eff = round(L_full * n_sites / |all polymorphic sites|)``.
Without this scaling the optimiser would see an artificially sparse
polymorphic spectrum and bias every ``K_i`` toward zero.

Wildcards:

- ``{model}`` ∈ ``{JC, HKY}`` — substitution model. HKY fits ``kappa``
  jointly via the :attr:`SubstitutionModel.free_params` protocol.
- ``{n_out}`` ∈ ``{1, 2, 3}`` — closest-first outgroup subset.
- ``{n_sites}`` — number of polymorphic sites to keep before the fit.
  The literal ``all`` is a sentinel that fits on every polymorphic site.
  The report uses these reference cells as the asymptotic truth.
- ``{seed}`` — per-cell RNG seed for the site sub-sample.

Run directly (defaults to JC + n_out=3 + n_sites=1000 + seed=100)::

    python workflow/scripts/infer_site_scaling_fixed.py
"""
import json
import time
from pathlib import Path

import numpy as np
import tskit

from ancestree import (
    BaseComposition,
    FixedTreeInference,
    HKY,
    JC69,
    KingmanIngroupWeight,
    OutgroupLadderTree,
    STATES,
    Site,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    model_name = snakemake.wildcards.model  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    # ``n_sites`` and ``seed`` are wildcards on the sweep rule but params
    # on the reference rule (single full-data fit per (model, n_out)).
    n_sites_wc = (
        snakemake.wildcards.get("n_sites")  # type: ignore[name-defined]
        if "n_sites" in snakemake.wildcards.keys()  # type: ignore[name-defined]
        else str(snakemake.params.n_sites)  # type: ignore[name-defined]
    )
    seed = int(
        snakemake.wildcards.seed  # type: ignore[name-defined]
        if "seed" in snakemake.wildcards.keys()  # type: ignore[name-defined]
        else snakemake.params.seed  # type: ignore[name-defined]
    )
except NameError:
    in_trees = "results/data/site_scaling_sim.trees"
    in_meta = "results/data/site_scaling_sim_meta.json"
    out_json = "results/data/site_scaling_fixed_JC_n3_s1000_seed100.json"
    model_name = "JC"
    n_out = 3
    n_sites_wc = "1000"
    seed = 100


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroup_names: list[str] = list(meta["outgroup_names"])[:n_out]
length_full = float(meta["length"])

ts = tskit.load(in_trees)
sample_id_per_node: dict[int, str] = {
    int(node_id): f"tsk_{ind.id}"
    for ind in ts.individuals()
    for node_id in ind.nodes
}
sample_nodes = ts.samples()

ingroup_set = set(ingroup_names)
outgroup_set = set(outgroup_names)
keep_ids = ingroup_set | outgroup_set

all_sites: list[Site] = []
polymorphic_ingroup_mask: list[bool] = []
for variant in ts.variants():
    alleles = tuple(variant.alleles)
    gens = variant.genotypes
    tip_alleles: dict[str, str | None] = {}
    for sample_idx, node_id in enumerate(sample_nodes):
        sid = sample_id_per_node.get(int(node_id))
        if sid is None or sid not in keep_ids:
            continue
        g = int(gens[sample_idx])
        if g < 0:
            tip_alleles[sid] = None
        else:
            a = alleles[g]
            tip_alleles[sid] = a if a in STATES else None
    ingroup_alleles = {tip_alleles[s] for s in ingroup_names if tip_alleles.get(s) is not None}
    polymorphic_ingroup_mask.append(len(ingroup_alleles) >= 2)
    all_sites.append(Site(
        chrom="1", pos=int(variant.site.position),
        alleles=tuple(a for a in alleles if a in STATES),
        tip_alleles=tip_alleles,
    ))

n_total = len(all_sites)
n_ingroup_poly = sum(polymorphic_ingroup_mask)
print(
    f"B14 fixed-tree cell: model={model_name}, n_out={n_out}, "
    f"n_sites_wc={n_sites_wc}, seed={seed}; "
    f"{n_total} polymorphic sites total, {n_ingroup_poly} ingroup-polymorphic",
    flush=True,
)

if n_sites_wc == "all":
    kept = all_sites
    n_sites_kept = n_total
else:
    n_sites_target = int(n_sites_wc)
    if n_sites_target > n_total:
        raise ValueError(
            f"n_sites={n_sites_target} exceeds available polymorphic sites "
            f"({n_total})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_sites_target, replace=False)
    idx.sort()
    kept = [all_sites[i] for i in idx]
    n_sites_kept = n_sites_target

n_target_eff = max(n_sites_kept, int(round(length_full * n_sites_kept / n_total)))

tree = OutgroupLadderTree(ingroup_names, outgroup_names)
if model_name == "JC":
    model = JC69()
elif model_name == "HKY":
    model = HKY(fit_kappa=True)
else:
    raise ValueError(f"Unknown model: {model_name!r}")

prior = KingmanIngroupWeight(ingroup_names)
bc = BaseComposition.from_n_target_sites(int(n_target_eff))

inference = FixedTreeInference(
    kept, model, base_composition=bc, ingroup_weight=prior, tree=tree,
    n_starts=20, parallelize=False, seed=seed,
)
_t0 = time.perf_counter()
params = inference.fit()
fit_seconds = float(time.perf_counter() - _t0)

result = {
    "model": model_name,
    "n_out": n_out,
    "n_sites_wc": n_sites_wc,
    "n_sites_kept": n_sites_kept,
    "n_total_polymorphic": n_total,
    "n_ingroup_poly_full": n_ingroup_poly,
    "n_target_eff": int(n_target_eff),
    "length_full": length_full,
    "seed": seed,
    "outgroup_names_used": outgroup_names,
    "params_mle": params,
    "outgroup_divergence": (
        inference.outgroup_divergence_mle.tolist()
        if inference.outgroup_divergence_mle is not None else None
    ),
    "log_likelihood_mle": inference.log_likelihood_mle,
    "fit_seconds": fit_seconds,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: divergence={result['outgroup_divergence']}, "
    f"kappa={params.get('kappa')}, fit_seconds={fit_seconds:.2f}",
    flush=True,
)
