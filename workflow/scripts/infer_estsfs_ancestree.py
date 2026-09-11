"""B3 Ancestree-side inference: FixedTreeInference on the est-sfs topology.

One cell per ``(model, prior)`` combo, driven by snakemake wildcards.
Builds an :class:`OutgroupLadderTree` over the simulated ingroup +
outgroup haplotype samples (the simulator uses ``ploidy=1`` so each
sample is one haplotype, one tip in the Felsenstein tree), fits its
branch rates (plus kappa when ``model=K2``) under the requested
polarization prior, and writes per-site posteriors to JSON.

Wildcards:

- ``{model}`` ∈ ``{JC, K2}`` — substitution model. K2 fits kappa
  jointly via the :attr:`SubstitutionModel.free_params` protocol.
- ``{prior}`` ∈ ``{kingman, adaptive}`` — root-state prior over the
  ingroup MRCA.

Run directly (defaults to JC+Kingman)::

    python workflow/scripts/infer_estsfs_ancestree.py

Or via snakemake::

    snakemake -j 1 results/data/estsfs_ancestree_JC_kingman.json
"""
import json
import time
from pathlib import Path

import tskit

from ancestree import (
    AdaptiveIngroupWeight,
    BaseComposition,
    FixedTreeInference,
    GTR,
    JC69,
    K2,
    KingmanIngroupWeight,
    OutgroupLadderTree,
    STATES,
    Site,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    sim_name = snakemake.wildcards.sim  # type: ignore[name-defined]
    model_name = snakemake.wildcards.model  # type: ignore[name-defined]
    prior_name = snakemake.wildcards.prior  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/estsfs_pi_sim.trees"
    in_meta = "results/data/estsfs_pi_sim_meta.json"
    out_json = "results/data/estsfs_pi_ancestree_jc_JC_adaptive_n3.json"
    sim_name = "jc"
    model_name = "JC"
    prior_name = "adaptive"
    n_out = 3
    length = 2e6


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
# Subset to first n_out outgroups (closest-first order from the simulator).
outgroup_names: list[str] = list(meta["outgroup_names"])[:n_out]

ts = tskit.load(in_trees)

# Map each tskit sample node → caller-facing sample id. With ploidy=1
# the simulator's per-individual labelling (tsk_0, tsk_1, …) is the
# sample id directly — no haplotype splitting needed.
sample_id_per_node: dict[int, str] = {
    int(node_id): f"tsk_{ind.id}"
    for ind in ts.individuals()
    for node_id in ind.nodes
}

print(
    f"B3 Ancestree ({model_name} + {prior_name}) on {in_trees}: "
    f"{ts.num_sites} sites, ingroup={len(ingroup_names)} haps, "
    f"outgroup={len(outgroup_names)} haps",
    flush=True,
)

sample_nodes = ts.samples()
sites: list[Site] = []
for variant in ts.variants():
    alleles = tuple(variant.alleles)
    gens = variant.genotypes
    tip_alleles: dict[str, str | None] = {}
    for sample_idx, node_id in enumerate(sample_nodes):
        sid = sample_id_per_node.get(int(node_id))
        if sid is None:
            continue
        g = int(gens[sample_idx])
        if g < 0:
            tip_alleles[sid] = None
        else:
            a = alleles[g]
            tip_alleles[sid] = a if a in STATES else None
    sites.append(Site(
        chrom="1", pos=int(variant.site.position),
        alleles=tuple(a for a in alleles if a in STATES),
        tip_alleles=tip_alleles,
    ))

tree = OutgroupLadderTree(ingroup_names, outgroup_names)
# Fixed-tree mode: FixedTreeInference's K_i parameters are already in
# expected substitutions per site (matches fastDFE's est-sfs convention).
# `mu` lives only on ARGBasedInference; FixedTreeInference doesn't scale
# branch lengths by it.
if model_name == "JC":
    model = JC69()
elif model_name == "K2":
    model = K2(fit_kappa=True)  # kappa defaults to 2.0 (initial guess)
elif model_name == "R6":
    # R6 (= GTR with 6 free exchangeability rates + empirical π) matches
    # EST-SFS's headline configuration. Starting all six at 1.0 collapses
    # initially to F81. The joint MLE then fits 5 rates with rate_AG held
    # as the reference scale (rate_AG = 1).
    model = GTR(fit_rates=True)
else:
    raise ValueError(f"Unknown model: {model_name!r}")

if prior_name == "kingman":
    prior = KingmanIngroupWeight(ingroup_names)
elif prior_name == "adaptive":
    # subsample_size=n_ingroup mirrors fastDFE's n_ingroups=n_ingroup setting
    # in infer_estsfs_fastdfe.py — same canonical bin space, apples-to-apples
    # pi comparison. Drop this to let Ancestree default (min(n_ingroup, 11))
    # if a separately-baselined run is wanted.
    prior = AdaptiveIngroupWeight(
        ingroup_names, subsample_size=len(ingroup_names),
    )
else:
    raise ValueError(f"Unknown prior: {prior_name!r}")

# The composition spans the whole target region; the monomorphic weight is
# derived from it against the polymorphic count.
n_polymorphic = len(sites)
bc = BaseComposition.from_n_target_sites(int(length))

inference = FixedTreeInference(
    sites, model, base_composition=bc, ingroup_weight=prior, tree=tree,
    # parallelize=False under snakemake (ProcessPool inside a snakemake
    # subprocess is brittle. Each cell is already its own snakemake job).
    # n_starts=10 to match EST-SFS's nrandom=10 default, so both tools
    # use the same number of independent ML initialisations. (The n=2 fit
    # with K1/K2 free has a flat likelihood ridge along K_1+K_2=const, so
    # several starts are needed to avoid a swapped K_1 > K_2 solution.)
    n_starts=10, parallelize=False,
)
# Wall-clock for the inference step only (post data loading, pre output write);
# reported in inference_meta for the manuscript runtime column.
_t_inference_start = time.perf_counter()
params = inference.fit()

out: dict[int, dict] = {}
for site, posterior in inference.infer():
    out[site.pos] = {
        "map_allele": posterior.map_allele,
        "max_prob": posterior.max_prob,
        "posterior": posterior.values.tolist(),
        "alleles": list(site.alleles),
    }
inference_seconds = float(time.perf_counter() - _t_inference_start)

inference_meta = {
    "params_mle": params,
    "outgroup_divergence": inference.outgroup_divergence_mle.tolist()
        if inference.outgroup_divergence_mle is not None else None,
    "log_likelihood_mle": inference.log_likelihood_mle,
    "sim": sim_name,
    "model": model_name,
    "prior": prior_name,
    "n_out": n_out,
    "outgroup_names_used": outgroup_names,
    "n_polymorphic_sites": n_polymorphic,
    "n_target_sites": int(length),
    "inference_seconds": inference_seconds,
}
if isinstance(prior, AdaptiveIngroupWeight):
    inference_meta["adaptive_pi"] = {str(k): v for k, v in prior.pi.items()}
    inference_meta["adaptive_n_sites_per_bin"] = {
        str(k): v for k, v in prior.n_sites_per_bin.items()
    }

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": out, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(out)} sites", flush=True)
