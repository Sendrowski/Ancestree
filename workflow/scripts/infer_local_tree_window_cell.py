"""One (window, rec_rate, mu, switch) cell of the local-tree window benchmark.

Builds the ingroup :class:`~ancestree.sites.Site` list from the simulated
ARG's genotypes (no truth allele consumed — VCF-only), runs
:class:`~ancestree.local_tree_inference.LocalTreeInference` at the cell's
window size, assumed recombination rate and assumed mutation rate, then
scores every polarised site against the msprime ground-truth ancestral
state. The assumed ``mu`` is given as a multiple of the true simulation
``mu``. It enters both the emission rate ``lambda_i = 2*mu*B*t_i`` and the
time-grid calibration, so a misspecified ``mu`` rescales inferred TMRCAs.

The optional ``switch`` param injects phasing error into the genotypes fed
to inference (truth unchanged): consecutive ingroup haplotypes are paired
into pseudo-diploids and a per-pair orientation flips with probability
``switch`` at each heterozygous site — the same switch-error process as
:func:`_robustness_common.Scenario._run_local_tree` (0 = perfectly phased,
0.5 = fully random). Ingroup allele *frequencies* are preserved, so the
truth and per-site scoring are unaffected.

A special ``rec_rate`` sentinel of ``"true_arg"`` instead scores the
genuine simulated tree sequence through
:class:`~ancestree.inference.ARGBasedInference` (the achievable
true-genealogy ceiling); ``window`` is ignored in that mode.

Reuses :func:`_robustness_common._sumsq` so the per-site Brier term is
computed identically to the robustness heatmap.

Run directly with the standalone defaults::

    python workflow/scripts/infer_local_tree_window_cell.py

Or via snakemake (one job per grid cell)::

    snakemake -j 4 results/data/local_tree_window_w50snp_r1.0.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tskit

# Make the sibling _robustness_common importable when run as a snakemake
# script (the script is copied to a tempdir) or standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _robustness_common import (  # noqa: E402
    _sumsq, _evenly_spread_outgroups, LOCAL_TREE_ENSEMBLE_MEMBERS)

from ancestree import ARGBasedInference, JC69, STATES, Site  # noqa: E402
from ancestree import LocalTreeInference  # noqa: E402


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    window = str(snakemake.params.window)  # type: ignore[name-defined]
    rec_rate_spec = str(snakemake.params.rec_rate)  # type: ignore[name-defined]
    mu_spec = str(snakemake.params.mu)  # type: ignore[name-defined]
    switch_spec = str(snakemake.params.switch)  # type: ignore[name-defined]
    n_out = int(getattr(snakemake.params, "n_out", 0))  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/local_tree_window_sim.trees"
    in_meta = "results/data/local_tree_window_sim_meta.json"
    out_json = "results/data/local_tree_window_w50snp_r1.0.json"
    window = "50snp"
    rec_rate_spec = "1.0"
    mu_spec = "1.0"
    switch_spec = "0.0"
    n_out = 0


meta = json.loads(Path(in_meta).read_text())
ingroup_names = list(meta["ingroup_names"])
mu_true = float(meta["mu"])  # the TRUE simulation mutation rate
r_true = float(meta["rec_rate"])
ts = tskit.load(str(in_trees))

# Per-site truth + node→name map (single haplotype per individual).
name_for_node = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
node_for_name = {v: k for k, v in name_for_node.items()}
truth_by_pos = {int(s.position): s.ancestral_state for s in ts.sites()}

# Outgroup ladder (n_out=0 keeps the ingroup-only behaviour: keep_names is the
# whole panel and every truth-bearing site is scored). With n_out>0 the kept
# panel is the ingroup plus the n_out evenly-spread outgroups, and only the
# ingroup-polymorphic sites are scored, so the target matches the n_out=0 run.
ingroup_set = set(ingroup_names)
if n_out > 0:
    ordered = sorted(meta["outgroups_by_pop"].keys(), key=lambda s: int(s.split("_")[1]))
    all_outgroup_names = [nm for p in ordered for nm in meta["outgroups_by_pop"][p]]
    keep_names = ingroup_names + _evenly_spread_outgroups(all_outgroup_names, n_out)
    score_pos: "set[int] | None" = set()
    for v in ts.variants():
        ing = {v.alleles[v.genotypes[j]] for j, nd in enumerate(ts.samples())
               if name_for_node[int(nd)] in ingroup_set}
        if len(ing) > 1:
            score_pos.add(int(v.site.position))
else:
    keep_names = ingroup_names
    score_pos = None
keep_set = set(keep_names)


def _score(infer_iter) -> list[tuple[int, float, float]]:
    """Score a polariser's ``(Site, Posterior)`` stream against truth.

    :param infer_iter: Iterable of ``(Site, Posterior)`` from an inference.
    :return: Per-site ``(map_hit, p_true, sum_sq)`` triples (scored sites
        only); ``sum_sq`` lets the report recover the Brier score.
    """
    recs: list[tuple[int, float, float]] = []
    for site, post in infer_iter:
        pos = int(site.pos)
        if score_pos is not None and pos not in score_pos:
            continue
        t = truth_by_pos.get(pos)
        if t is None or t not in STATES:
            continue
        recs.append((int(post.map_allele == t), float(post[t]), _sumsq(post)))
    return recs


# Assumed mutation rate: a multiple of the true mu (1.0 = well-specified).
mu_used = float(mu_spec) * mu_true
# Injected phasing switch-error rate (0.0 = perfectly phased).
switch_rate = float(switch_spec)

t0 = time.perf_counter()
if rec_rate_spec == "true_arg":
    # True-ARG ceiling: score the genuine simulated genealogy directly
    # (always with the correct mu — this is the reference, not a cell).
    if n_out > 0:
        keep_nodes = [node_for_name[n] for n in keep_names]
        ts_ceil = ts.simplify(samples=keep_nodes, filter_sites=False)
        sample_map = {n: i for i, n in enumerate(keep_names)}
    else:
        ts_ceil = ts
        sample_map = {f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()}
    inf = ARGBasedInference(
        ts_ceil, JC69(), mu=mu_true, sample_map=sample_map,
        # Named for the same reason the local-tree branch names them: the
        # ceiling has to answer the question the cells it bounds answer, and
        # an unnamed panel is read at the panel root rather than at the
        # ingroup MRCA.
        ingroup_samples=[n for n in sample_map if n in ingroup_set] or None,
        outgroup_samples=[n for n in sample_map if n not in ingroup_set] or None,
        progress=False,
    )
    records = _score(inf.infer())
    rec_rate_used = r_true
    mu_used = mu_true
else:
    # Inferred local trees at this cell's window + (mis)specified rates.
    rec_rate_used = float(rec_rate_spec) * r_true
    # Pair consecutive ingroup haplotypes into pseudo-diploids. A per-pair
    # orientation is carried along the genome and flips with probability
    # switch_rate at each het site (the _run_local_tree switch-error scheme).
    dip_pairs = [(ingroup_names[2 * k], ingroup_names[2 * k + 1])
                 for k in range(len(ingroup_names) // 2)]
    rng = np.random.default_rng(12345)
    orient = [0] * len(dip_pairs)
    sites: list[Site] = []
    for v in ts.variants():
        pos = int(v.site.position)
        if truth_by_pos.get(pos) not in STATES:
            continue
        tip_alleles = {name_for_node[int(node)]: v.alleles[v.genotypes[j]]
                       for j, node in enumerate(ts.samples())
                       if name_for_node[int(node)] in keep_set}
        if n_out > 0 and len(set(tip_alleles.values())) < 2:
            continue  # monomorphic among the kept panel: carries no signal
        if switch_rate > 0.0:
            for pi, (a, b) in enumerate(dip_pairs):
                if tip_alleles.get(a) == tip_alleles.get(b):
                    continue  # homozygous: phase irrelevant
                if rng.random() < switch_rate:
                    orient[pi] ^= 1  # switch error from here on
                if orient[pi]:
                    tip_alleles[a], tip_alleles[b] = tip_alleles[b], tip_alleles[a]
        sites.append(Site(chrom="1", pos=pos,
                          alleles=tuple(v.alleles), tip_alleles=tip_alleles))
    # Block tracks the window (default): each sweep point is one tree per the
    # window's SNPs, so the sweep varies local-tree resolution directly (rather
    # than blocks-averaged-per-tree at a fixed emission grid). chunk_size bounds
    # peak memory at the fine-window end (small blocks -> many blocks on 100 Mb).
    inf = LocalTreeInference(
        sites, JC69(), mu=mu_used, rec_rate=rec_rate_used,
        sample_names=keep_names,
        sequence_length=float(ts.sequence_length),
        window=window, chunk_size="10mb", progress=False,
        # Name the panel's halves so the default focal node ("ingroup_mrca")
        # resolves to the ingroup's own ancestor. Without them the ingroup is
        # taken to be the whole panel and the posterior is read at the panel
        # root instead --- the outgroups are in the panel here, so that is a
        # different question from the one the heatmap's local-tree row answers.
        ingroup_samples=ingroup_names,
        outgroup_samples=[n for n in keep_names if n not in ingroup_set],
        # The manuscript's ensemble size, shared with the heatmap rows so a
        # window chosen here is chosen under the setting the figures use.
        n_ensemble=LOCAL_TREE_ENSEMBLE_MEMBERS,
    )
    records = _score(inf.infer())
runtime = time.perf_counter() - t0

n = len(records)
mean_map = sum(r[0] for r in records) / n if n else float("nan")
mean_ptrue = sum(r[1] for r in records) / n if n else float("nan")
# Brier = 1 - 2*p_true + sum_sq, averaged over sites (lower is better).
mean_brier = (
    sum(1.0 - 2.0 * r[1] + r[2] for r in records) / n if n else float("nan")
)

payload = {
    "window": window,
    "n_out": n_out,
    "rec_rate_spec": rec_rate_spec,
    "rec_rate_used": rec_rate_used,
    "r_true": r_true,
    "mu_spec": mu_spec,
    "mu_used": mu_used,
    "mu_true": mu_true,
    "switch_spec": switch_spec,
    "switch_rate": switch_rate,
    "n_sites": n,
    "mean_brier": mean_brier,
    "mean_map": mean_map,
    "mean_ptrue": mean_ptrue,
    "runtime": runtime,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(
    f"Wrote {out_json}: window={window} rec={rec_rate_spec} mu={mu_spec} "
    f"switch={switch_spec} "
    f"n={n} Brier={mean_brier:.4f} MAP={mean_map:.4f} "
    f"P(true)={mean_ptrue:.4f} ({runtime:.1f}s)",
    flush=True,
)
