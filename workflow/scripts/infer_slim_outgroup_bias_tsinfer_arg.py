"""SLiM-outgroup-bias benchmark: ARGBasedInference on a tsinfer-inferred ARG.

The tsinfer ARG was built from the ingroup VCF only (no outgroups), so
the inference runs on a strictly ingroup-only topology. ``mu = mu_slim`` is
passed because tsinfer outputs branches in (per-construction) generation units
consistent with the SLiM scale given via the ``recombination_rate`` flag.

Truth ancestral states are looked up from the simulator-truth ts
(``_chunk{c}.trees``) keyed by site position. Sites tsinfer keeps that
overlap with truth-known sites are scored. Others are skipped.

This script is invoked ONCE per ``(chunk_idx, f_del)`` cell (no n_out
dimension — tsinfer-ARG inference is ingroup-only by construction).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import tskit

from ancestree import ARGBasedInference, JC69

STATES = ("A", "C", "G", "T")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI fallback when not driven by snakemake."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-inferred", required=True)
    p.add_argument("--in-truth-trees", required=True)
    p.add_argument("--in-meta", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--f-del", type=float, required=True)
    p.add_argument("--mu-slim", type=float, required=True)
    return p.parse_args(argv)


try:
    in_inferred = snakemake.input.inferred  # type: ignore[name-defined]
    in_truth_trees = snakemake.input.truth_trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    f_del = float(snakemake.wildcards.f_del)  # type: ignore[name-defined]
    mu_slim = float(snakemake.params.mu_slim)  # type: ignore[name-defined]
except NameError:
    _args = _parse_args(sys.argv[1:])
    in_inferred = _args.in_inferred
    in_truth_trees = _args.in_truth_trees
    in_meta = _args.in_meta
    out_json = _args.out_json
    f_del = _args.f_del
    mu_slim = _args.mu_slim


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
n_ingroup = len(ingroup_names)

# Load truth ts to build a position -> ancestral-state map.
ts_truth = tskit.load(in_truth_trees)
truth_by_pos: dict[int, str] = {}
for site in ts_truth.sites():
    pos = int(site.position)
    anc = site.ancestral_state
    if anc in STATES:
        truth_by_pos[pos] = anc

ts_inferred = tskit.load(in_inferred)
# The builder dates the ARG, so mu is per generation; a chunk tsdate could not
# date carries a rate fitted against the ARG's own time axis instead.
_side_meta_path = str(in_inferred) + "_meta.json"
_side_meta = {}
if os.path.exists(_side_meta_path):
    with open(_side_meta_path) as f:
        _side_meta = json.load(f)
mu_use = float(_side_meta.get("mu", mu_slim))
mu_matches_time_units = (
    ts_inferred.time_units == tskit.TIME_UNITS_UNCALIBRATED)
print(
    f"tsinfer-ARG inference: f_del={f_del}, "
    f"{ts_inferred.num_samples} haplotypes, {ts_inferred.num_sites} sites, "
    f"{ts_inferred.num_trees} local trees, mu={mu_slim:g}, "
    f"truth-site-count={len(truth_by_pos)}",
    flush=True,
)

# tsinfer assigns sample ids 0..n_ingroup-1 in the order add_site saw
# them (ingroup_names order, matching build_tsinfer_arg.py).
sample_map = {ingroup_names[i]: i for i in range(min(n_ingroup, ts_inferred.num_samples))}
if ts_inferred.num_samples != n_ingroup:
    print(
        f"  WARNING: tsinfer ts has {ts_inferred.num_samples} samples "
        f"but meta has {n_ingroup} ingroup; using min mapping.",
        flush=True,
    )

# Pre-compute per-site ingroup counts from the inferred ts so we can
# write ingroup_minor / ingroup_derived counts in the same schema as
# the other modes.
ingroup_counts_by_pos: dict[int, dict[str, int]] = {}
n_ingroup_samples = ts_inferred.num_samples
for var in ts_inferred.variants():
    pos = int(var.site.position)
    alleles = list(var.alleles)
    gens = var.genotypes
    counts: dict[str, int] = {}
    for i in range(n_ingroup_samples):
        g = int(gens[i])
        if g < 0:
            continue
        a = alleles[g]
        if a is None or a not in STATES:
            continue
        counts[a] = counts.get(a, 0) + 1
    ingroup_counts_by_pos[pos] = counts


inference = ARGBasedInference(
    ts_inferred, JC69(),
    mu=mu_use,
    sample_map=sample_map,
    progress=False,
    mu_matches_time_units=mu_matches_time_units,
)

per_site: list[dict] = []
n_skipped_no_truth = 0
_t0 = time.perf_counter()
for site, post in inference.infer():
    pos = int(site.pos)
    truth = truth_by_pos.get(pos)
    if truth is None:
        n_skipped_no_truth += 1
        continue
    counts = ingroup_counts_by_pos.get(pos, {})
    if counts:
        major_count = max(counts.values())
    else:
        major_count = n_ingroup
    minor_count = n_ingroup - major_count
    derived_count = int(n_ingroup - counts.get(truth, 0))
    posterior_values = [float(p) for p in post.values]
    posterior_alleles = list(post.alleles)
    per_site.append({
        "pos": pos,
        "alleles": list(site.alleles),
        "map_allele": post.map_allele,
        "max_prob": float(post.max_prob),
        "posterior": posterior_values,
        "posterior_alleles": posterior_alleles,
        "truth": truth,
        "map_correct": post.map_allele == truth,
        "ingroup_major_count": int(major_count),
        "ingroup_minor_count": int(minor_count),
        "ingroup_derived_count": derived_count,
    })
infer_seconds = float(time.perf_counter() - _t0)

# Pull tsinfer-build timings from the side-meta if present.
tsinfer_meta_path = Path(str(in_inferred) + "_meta.json")
tsinfer_build_meta: dict = {}
if tsinfer_meta_path.exists():
    with open(tsinfer_meta_path) as f:
        tsinfer_build_meta = json.load(f)

result = {
    "f_del": f_del,
    "n_out": 0,  # ingroup-only sentinel for the report aggregator
    "chunk_idx": meta.get("chunk_idx"),
    "ingroup_size": n_ingroup,
    "outgroup_names_used": [],
    "mode": "arg_tsinfer",
    "mu": mu_slim,
    "fit_seconds": 0.0,
    "infer_seconds": infer_seconds,
    "n_sites": len(per_site),
    "n_skipped_no_truth": n_skipped_no_truth,
    "tsinfer_build_meta": tsinfer_build_meta,
    "per_site": per_site,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: {len(per_site)} sites scored "
    f"(skipped {n_skipped_no_truth} without truth), "
    f"infer={infer_seconds:.2f}s",
    flush=True,
)
