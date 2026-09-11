"""B12 missing-data sweep inference: random outgroup-tip masking per cell.

Loads :mod:`simulate_missing_data`'s full sim (ingroup + 3 outgroups),
materialises the polymorphic-site list via
:class:`~ancestree.sources.TskitSource`, randomly nulls out
``missing_frac`` of the outgroup tip slots per site (per-cell RNG
seeded with ``seed``), then runs
:class:`~ancestree.inference.FixedTreeInference` (HKY + empirical π
from the unmasked sites + Kingman prior). Per-site JSON schema mirrors
B7's so the report can stratify accuracy by folded SFS bin with the
same code path.

Wildcards:

- ``{missing_frac}`` — fraction of outgroup tips to mask per site (e.g.
  ``"0.0"`` for no masking, ``"0.5"`` for half).
- ``{seed}`` — per-cell masking RNG seed.

Run directly (defaults to ``missing_frac=0.25, seed=100``)::

    python workflow/scripts/infer_missing_data.py

Or via snakemake::

    snakemake -j 1 results/data/missing_data_f0.25_seed100.json
"""
import json
from pathlib import Path

import numpy as np
import tskit

from ancestree import (
    BaseComposition,
    FixedTreeInference,
    HKY,
    KingmanIngroupWeight,
    OutgroupLadderTree,
    Site,
    TskitSource,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    missing_frac = float(snakemake.wildcards.missing_frac)  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.seed)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/missing_data_sim.trees"
    in_meta = "results/data/missing_data_sim_meta.json"
    missing_frac = 0.25
    seed = 100
    out_json = f"results/data/missing_data_f{missing_frac}_seed{seed}.json"
    mu = 1.25e-8


if not 0.0 <= missing_frac <= 1.0:
    raise ValueError(f"missing_frac must be in [0, 1]; got {missing_frac}")


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
ordered_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
outgroup_names: list[str] = [name for pop in ordered_pops for name in outgroups_by_pop[pop]]
n_outgroups = len(outgroup_names)

ts = tskit.load(in_trees)

# Materialise the polymorphic-site stream once. Empirical π / κ caches
# come from the *unmasked* sites — the masking is downstream of the
# composition fit, so per-cell calibration stays stable. msprime's
# individuals don't carry name metadata, so we build the sample_map
# explicitly using the simulator's "tsk_{individual_id}" convention
# (matches what the rest of the B6-family benchmarks use).
sample_map = {
    f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()
}
source = TskitSource(ts, chrom="1", sample_map=sample_map)
all_sites_unmasked = list(source)
bc = BaseComposition.from_polymorphic_sites(all_sites_unmasked)
kappa_init = min(max(bc.kappa_estimate, 0.1), 50.0)

# Apply the per-cell mask. We mask outgroup tip slots only — the
# ingroup-tip frequency signal that drives the Kingman prior is left
# intact so the per-cell variance only reflects outgroup-evidence loss.
rng = np.random.default_rng(seed)
n_masked_total = 0
n_obs_total = 0
patched_sites: list[Site] = []
for site in all_sites_unmasked:
    tip_alleles = dict(site.tip_alleles)
    if missing_frac > 0.0:
        mask = rng.random(n_outgroups) < missing_frac
        for og, drop in zip(outgroup_names, mask):
            n_obs_total += 1
            # Count only slots that actually had a non-None observation
            # pre-mask so the "empirical_mask_rate" denominator is honest.
            if tip_alleles.get(og) is not None:
                if drop:
                    tip_alleles[og] = None
                    n_masked_total += 1
    else:
        n_obs_total += sum(
            1 for og in outgroup_names if tip_alleles.get(og) is not None
        )
    patched_sites.append(Site(
        chrom=site.chrom, pos=site.pos, alleles=site.alleles,
        tip_alleles=tip_alleles,
        local_tree_handle=site.local_tree_handle,
        info=site.info,
    ))

print(
    f"B12 missing_data inference: missing_frac={missing_frac}, seed={seed}, "
    f"{len(patched_sites)} polymorphic sites, "
    f"masked {n_masked_total}/{n_obs_total} outgroup tip slots "
    f"(empirical rate "
    f"{n_masked_total / n_obs_total if n_obs_total else 0.0:.3f})",
    flush=True,
)


# Build the FixedTreeInference on the masked sites + an HKY/Kingman pair.
# Construction order matters: pass the masked sites in directly so the
# prior log_probs cache is computed from them (preserves ingroup signal
# under masking, which only touches outgroups anyway, but the symmetry
# is the right invariant to lock in).
tree = OutgroupLadderTree(ingroup_names, outgroup_names)
model = HKY(kappa=kappa_init, fit_kappa=False)
# The weight takes only the ingroup. The base composition goes to the
# inference below, which is where it is read (it was a parameter of this class
# before the prior / ingroup-weight split).
prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
vcf_inf = FixedTreeInference(
    patched_sites, model, tree=tree,
    base_composition=bc,
    n_target_sites=int(ts.sequence_length),
    ingroup_weight=prior,
    parallelize=False,
    progress=False,
)
vcf_inf.fit()
fitted_outgroup_divergence = (
    vcf_inf.outgroup_divergence_mle.tolist()
    if vcf_inf.outgroup_divergence_mle is not None else None
)


truth_by_pos: dict[int, str] = {
    int(site.position): site.ancestral_state for site in ts.sites()
}

ingroup_name_set = set(ingroup_names)


def _ingroup_stats(site_tip_alleles) -> tuple[int, bool, int]:
    """Return (derived_count_proxy, ingroup_polymorphic, minor_count).

    ``derived_count_proxy`` = minor allele count over the ingroup —
    matches B7's folded-bin axis so :mod:`report_missing_data` can reuse
    the same SFS-bin code path.
    """
    counts: dict[str, int] = {}
    for s, a in site_tip_alleles.items():
        if s in ingroup_name_set and a is not None:
            counts[a] = counts.get(a, 0) + 1
    if len(counts) <= 1:
        return 0, False, 0
    sorted_counts = sorted(counts.values(), reverse=True)
    return int(sorted_counts[1]), True, int(sorted_counts[1])


results: dict[int, dict] = {}
for site, posterior in vcf_inf.infer():
    pos = int(site.pos)
    truth = truth_by_pos.get(pos)
    derived_count, ingroup_poly, minor_count = _ingroup_stats(site.tip_alleles)
    n_outgroup_observed = sum(
        1 for og in outgroup_names if site.tip_alleles.get(og) is not None
    )
    results[pos] = {
        "map_allele": posterior.map_allele,
        "max_prob": float(posterior.max_prob),
        "posterior": [float(p) for p in posterior.values],
        "alleles": list(site.alleles),
        "truth": truth,
        "map_correct": truth is not None and posterior.map_allele == truth,
        "ingroup_polymorphic": bool(ingroup_poly),
        "minor_count": minor_count,
        "derived_count": derived_count,  # matches B7 folded-bin axis
        "n_outgroup_observed": n_outgroup_observed,
    }


inference_meta = {
    "missing_frac": missing_frac,
    "seed": seed,
    "n_ingroup": len(ingroup_names),
    "n_outgroups": n_outgroups,
    "outgroup_names": outgroup_names,
    "n_sites_inferred": len(results),
    "n_outgroup_slots_total": n_obs_total,
    "n_outgroup_slots_masked": n_masked_total,
    "empirical_mask_rate": (
        n_masked_total / n_obs_total if n_obs_total else 0.0
    ),
    "mu": mu,
    "model": "HKY",
    "kappa_init": kappa_init,
    "prior": prior.__class__.__name__,
    "fitted_outgroup_divergence": fitted_outgroup_divergence,
    "mode": "vcf",
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
