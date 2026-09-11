"""Ingroup-size sweep benchmark inference: ARGBasedInference with n_out=0 only.

Loads the simulated ts (ingroup-only, ``n_ingroup_max`` haplotypes),
randomly subsamples it to ``n_ingroup`` haplotypes using a per-cell RNG
seeded with ``seed``, then runs :class:`ARGBasedInference` in
ingroup-only mode (no outgroups) and dumps per-site posteriors plus the
simulator's truth ancestral state and the per-site derived allele count
(needed by the report to stratify accuracy by SFS bin).

Wildcards:

- ``{n_ingroup}`` — number of ingroup haplotypes kept after
  subsampling.
- ``{seed}`` — per-cell subsampling RNG seed.

Run directly (defaults to n_ingroup=20, seed=100)::

    python workflow/scripts/infer_ingroup_size.py

Or via snakemake::

    snakemake -j 1 results/data/ingroup_size_n20_seed100.json
"""
import json
from pathlib import Path

import numpy as np
import tskit

from ancestree import ARGBasedInference, JC69

# B11 note: MajorityOutgroupInference is the baseline_check comparator,
# but B7 is the ingroup-only sweep (n_out=0 by design) — there are no
# outgroups to vote, so the baseline isn't applicable here and the
# per-cell JSON omits the ``baseline_map`` field. The B7 report should
# treat it as "n/a".


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    n_ingroup = int(snakemake.wildcards.n_ingroup)  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.seed)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/ingroup_size_sim.trees"
    in_meta = "results/data/ingroup_size_sim_meta.json"
    out_json = "results/data/ingroup_size_n20_seed100.json"
    n_ingroup = 20
    seed = 100
    mu = 1.25e-8


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
n_ingroup_max = int(meta["n_ingroup_max"])
assert n_ingroup <= n_ingroup_max, (
    f"n_ingroup={n_ingroup} exceeds simulated n_ingroup_max={n_ingroup_max}"
)

ts_full = tskit.load(in_trees)
# Sample id → tskit node id (ploidy=1, so one node per individual).
sample_id_to_node: dict[str, int] = {}
for ind in ts_full.individuals():
    sid = f"tsk_{ind.id}"
    sample_id_to_node[sid] = int(ind.nodes[0])

# Deterministic random subsample of the ingroup down to ``n_ingroup``.
rng = np.random.default_rng(seed)
chosen_idx = rng.choice(len(ingroup_names), size=n_ingroup, replace=False)
chosen_idx.sort()  # keep id order stable for nicer JSON / debugging
keep_sample_names = [ingroup_names[i] for i in chosen_idx]
keep_node_ids = [sample_id_to_node[s] for s in keep_sample_names]
ts_sub = ts_full.simplify(samples=keep_node_ids, filter_sites=False)
# After simplify the kept samples are renumbered 0..n_ingroup-1 in the
# order we passed them. Rebuild the name-to-node map.
sample_map = {s: i for i, s in enumerate(keep_sample_names)}

print(
    f"Ingroup-size inference: n_ingroup={n_ingroup}, seed={seed}, "
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
for site, posterior in inference.infer():
    truth = truth_by_pos.get(int(site.pos))
    # Per-site derived-allele count over the subsampled ingroup.
    # When truth is known, derived = alleles \ {truth}. We count tips
    # carrying any non-truth allele. Sites with no truth allele get
    # ``derived_count = None`` and are dropped by the report's SFS
    # binning. Monomorphic sites (post-simplify) get derived_count=0.
    if truth is not None:
        derived_count = sum(
            1 for a in site.tip_alleles.values()
            if a is not None and a != truth
        )
    else:
        derived_count = None
    results[int(site.pos)] = {
        "map_allele": posterior.map_allele,
        "max_prob": float(posterior.max_prob),
        "posterior": [float(p) for p in posterior.values],
        "alleles": list(site.alleles),
        "truth": truth,
        "map_correct": truth is not None and posterior.map_allele == truth,
        "derived_count": derived_count,
    }

inference_meta = {
    "n_ingroup": n_ingroup,
    "seed": seed,
    "ingroup_names_used": keep_sample_names,
    "n_sites_inferred": len(results),
    "mu": mu,
    "mode": "arg_ingroup_only",
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
