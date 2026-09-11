"""B15: AdaptiveIngroupWeight pi_i recovery vs n_sites.

Same ARG as :mod:`infer_site_scaling_fixed`. Fixed at ``n_out = 3``
(closest-first three outgroups, so the tree-rate fit has plenty of
ladder signal) and substitution model = JC69 (the simulator). The cell
sub-samples the ingroup down to ``n_sub`` haplotypes (so ``n_obs ==
subsample_size`` and there is no hypergeometric down-projection in the
prior's MLE) before fitting, and sweeps the polymorphic site count.
This isolates the bin-count scaling: the adaptive prior fits
``floor(n_sub/2)`` free parameters by symmetry, so the total site
budget required to localise every bin scales with ``n_sub`` — modulo
diminishing returns from statistical strength shared via the
canonical-bin parameterisation (manuscript §3.6).

The reference / truth for each bin is the Kingman closed form
``(n_sub - j) / n_sub`` — the manuscript section establishes that the
adaptive prior converges to Kingman in the neutral large-data limit, so
this benchmark times that convergence.

Wildcards:

- ``{n_sub}`` ∈ ``{5, 10, 20}`` — number of ingroup haplotypes used in
  the fit and AdaptiveIngroupWeight.subsample_size simultaneously.
- ``{n_sites}`` — polymorphic-site count to keep (literal ``all``
  reserved for the reference cell, used by the report only as a sanity
  asymptote).
- ``{seed}`` — per-cell RNG seed for the ingroup + site sub-samples.

Run directly::

    python workflow/scripts/infer_site_scaling_adaptive.py
"""
import json
import time
import warnings
from pathlib import Path

import numpy as np
import tskit

from ancestree import (
    AdaptiveIngroupWeight,
    BaseComposition,
    FixedTreeInference,
    JC69,
    OutgroupLadderTree,
    STATES,
    Site,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    n_sub = int(snakemake.wildcards.n_sub)  # type: ignore[name-defined]
    n_sites_wc = snakemake.wildcards.n_sites  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.seed)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/site_scaling_sim.trees"
    in_meta = "results/data/site_scaling_sim_meta.json"
    out_json = "results/data/site_scaling_adaptive_sub20_s1000_seed100.json"
    n_sub = 20
    n_sites_wc = "1000"
    seed = 100


with open(in_meta) as f:
    meta = json.load(f)
full_ingroup_names: list[str] = list(meta["ingroup_names"])
outgroup_names: list[str] = list(meta["outgroup_names"])[:3]
length_full = float(meta["length"])

if n_sub > len(full_ingroup_names):
    raise ValueError(
        f"n_sub={n_sub} exceeds simulated ingroup size "
        f"({len(full_ingroup_names)})"
    )
ingroup_names = full_ingroup_names[:n_sub]

ts = tskit.load(in_trees)
sample_id_per_node: dict[int, str] = {
    int(node_id): f"tsk_{ind.id}"
    for ind in ts.individuals()
    for node_id in ind.nodes
}
sample_nodes = ts.samples()

keep_ids = set(ingroup_names) | set(outgroup_names)
all_sites: list[Site] = []
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
    ingroup_alleles = {
        tip_alleles.get(s) for s in ingroup_names
        if tip_alleles.get(s) is not None
    }
    if len(ingroup_alleles) < 2:
        continue  # only ingroup-polymorphic sites feed the adaptive fit
    all_sites.append(Site(
        chrom="1", pos=int(variant.site.position),
        alleles=tuple(a for a in alleles if a in STATES),
        tip_alleles=tip_alleles,
    ))

n_total = len(all_sites)
print(
    f"B15 adaptive cell: n_sub={n_sub}, n_sites_wc={n_sites_wc}, seed={seed}; "
    f"{n_total} polymorphic sites total",
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
model = JC69()
prior = AdaptiveIngroupWeight(
    ingroup_names, subsample_size=n_sub, min_bin_n_sites=0, seed=seed,
)
bc = BaseComposition.from_n_target_sites(int(n_target_eff))

inference = FixedTreeInference(
    kept, model, base_composition=bc, ingroup_weight=prior, tree=tree,
    n_starts=20, parallelize=False, seed=seed,
)
_t0 = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    params = inference.fit()
fit_seconds = float(time.perf_counter() - _t0)

kingman_truth = {j: (n_sub - j) / n_sub for j in range(n_sub + 1)}
fitted_pi = {int(j): float(p) for j, p in prior.pi.items()}
n_sites_per_bin = {int(j): float(c) for j, c in prior.n_sites_per_bin.items()}

result = {
    "n_sub": n_sub,
    "n_sites_wc": n_sites_wc,
    "n_sites_kept": n_sites_kept,
    "n_total_polymorphic": n_total,
    "n_target_eff": int(n_target_eff),
    "length_full": length_full,
    "seed": seed,
    "params_mle": params,
    "outgroup_divergence": (
        inference.outgroup_divergence_mle.tolist()
        if inference.outgroup_divergence_mle is not None else None
    ),
    "log_likelihood_mle": inference.log_likelihood_mle,
    "pi_fit": fitted_pi,
    "pi_kingman": kingman_truth,
    "n_sites_per_bin": n_sites_per_bin,
    "n_sites_fallback_kingman": int(prior.n_sites_fallback_kingman),
    "fit_seconds": fit_seconds,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: pi[1]={fitted_pi.get(1):.4f} vs Kingman {kingman_truth[1]:.4f}; "
    f"fit_seconds={fit_seconds:.2f}",
    flush=True,
)
