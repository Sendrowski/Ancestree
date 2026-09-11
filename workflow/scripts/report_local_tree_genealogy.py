"""Local-tree genealogy-recovery benchmark: how well are the inferred local
trees themselves recovered, separate from polarisation accuracy.

The window-bench accuracy tables score only the final ancestral-allele call
(Brier / MAP / P(true)). They cannot say whether a residual error is the
*genealogy* being mis-inferred or the *kernel* mis-polarising a good tree.
This script measures the genealogy directly: it rebuilds the inferred local
trees from the same simulated genotypes (perfect phasing, well-specified
rates) and compares them, position by position, against the true simulated
local trees, sweeping the window size.

Three families of metric (inferred vs. true, across the genome):

- **TMRCA rank** — Spearman rho between inferred and true pairwise TMRCAs:
  does the genealogy order coalescences correctly even if absolute times drift?
- **TMRCA bias** — mean log-ratio ``log(t_hat / t_true)`` (0 = unbiased) and the
  slope of a ``t_hat ~ t_true`` regression: is the PSMC'-HMM time grid
  systematically compressed or inflated?
- **Tree distance** — normalised Robinson-Foulds (topology only) and
  Kendall-Colijn at ``lambda=1`` (branch-length / time aware), span-weighted
  along the genome.

Output: ``results/reports/local_tree_genealogy.{json,md}``.
"""
import json
import os
from pathlib import Path

import numpy as np
import tskit

from ancestree import Site, STATES, JC69, ARGBasedInference
from ancestree.local_tree_inference import (LocalTreeBuilder,
                                            LocalTreeInference)

#: Members drawn per window for the ensemble columns. Matches the
#: manuscript's local-tree configuration.
ENSEMBLE_MEMBERS = int(os.environ.get("ANCESTREE_ENSEMBLE_MEMBERS", "128"))

WINDOWS = [1, 2, 5, 10, 25, 50, 100, 200, 400, 1000, 2000]
MAX_TREES_PER_WINDOW = 1500  # cap genome positions sampled per window
N_PAIRS = 80  # random haplotype pairs for TMRCA stats
SEED = 7


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = getattr(snakemake.output, "md", None)  # type: ignore[name-defined]
    n_out = int(getattr(snakemake.params, "n_out", 0))  # type: ignore[name-defined]
    run_tsinfer = bool(getattr(snakemake.params, "run_tsinfer", True))  # type: ignore[name-defined]
    shard = getattr(snakemake.params, "shard", None)  # type: ignore[name-defined]
    shard_inputs = list(getattr(snakemake.input, "shards", []) or [])  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/local_tree_window_sim.trees"
    in_meta = "results/data/local_tree_window_sim_meta.json"
    out_json = "results/reports/local_tree_genealogy.json"
    out_md = "results/reports/local_tree_genealogy.md"
    n_out = 0
    run_tsinfer = True
    shard = None
    shard_inputs = []


with open(in_meta) as f:
    meta = json.load(f)
mu_true = float(meta["mu"])
r_true = float(meta["rec_rate"])
ingroup_names = list(meta["ingroup_names"])
n_ing = len(ingroup_names)

ts = tskit.load(str(in_trees))
name_for_node = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
node_for_name = {v: k for k, v in name_for_node.items()}
truth_states = {int(s.position): s.ancestral_state for s in ts.sites()}

# Outgroup ladder: n_out=0 keeps the ingroup-only panel. N_out>0 adds the
# evenly-spread outgroups to both the inferred and the (simplified) true
# tree, so sample indices align. TMRCA pairs are always drawn among the
# ingroup (indices 0..n_ing-1, since keep_names lists the ingroup first).
if n_out > 0:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _robustness_common import _evenly_spread_outgroups  # noqa: E402
    _ordered = sorted(meta["outgroups_by_pop"].keys(), key=lambda s: int(s.split("_")[1]))
    _all_og = [nm for p in _ordered for nm in meta["outgroups_by_pop"][p]]
    keep_names = ingroup_names + _evenly_spread_outgroups(_all_og, n_out)
    truth_ts = ts.simplify(samples=[node_for_name[x] for x in keep_names],
                           filter_sites=False)
else:
    keep_names = ingroup_names
    truth_ts = ts
keep_set = set(keep_names)
ingroup_set = set(ingroup_names)
n_keep = len(keep_names)

# Build the (perfectly phased, well-specified) Site stream once. With outgroups
# the genealogy Brier is scored over the ingroup-polymorphic sites (same target
# as the n_out=0 figures), and sites monomorphic among the kept panel are
# dropped (they carry no genealogy signal and would bloat the builder).
ingroup_poly: "set[int] | None" = None if n_out <= 0 else set()
sites: list[Site] = []
for v in ts.variants():
    pos = int(v.site.position)
    if truth_states.get(pos) not in STATES:
        continue
    tip_alleles = {name_for_node[int(node)]: v.alleles[v.genotypes[j]]
                   for j, node in enumerate(ts.samples())
                   if name_for_node[int(node)] in keep_set}
    if n_out > 0:
        if len({tip_alleles[nm] for nm in ingroup_names}) > 1:
            ingroup_poly.add(pos)
        if len(set(tip_alleles.values())) < 2:
            continue  # monomorphic among the kept panel
    sites.append(Site(chrom="1", pos=pos, alleles=tuple(v.alleles),
                      tip_alleles=tip_alleles))
print(f"genealogy bench: {len(sites):,} sites, n_keep={n_keep} (n_out={n_out})", flush=True)

rng = np.random.default_rng(SEED)
all_pairs = [(i, j) for i in range(n_ing) for j in range(i + 1, n_ing)]
pair_idx = rng.choice(len(all_pairs), size=min(N_PAIRS, len(all_pairs)),
                      replace=False)
pairs = [all_pairs[k] for k in pair_idx]

rf_max = 2.0 * (n_keep - 2)  # max RF for two rooted binary trees on n_keep leaves


def _metrics_vs_truth(inf_ts) -> dict:
    """Span-weighted genealogy-recovery metrics of ``inf_ts`` vs. the true
    ``ts`` (RF, KC, TMRCA rank rho, log-bias, regression slope)."""
    rf_acc = kc_acc = span_acc = 0.0
    t_true_all: list[float] = []
    t_inf_all: list[float] = []

    # NB: tskit's ``trees()`` reuses one mutable Tree per iteration, so the
    # stream must be processed in place (never ``list(...)``-materialised).
    stride = max(1, inf_ts.num_trees // MAX_TREES_PER_WINDOW)
    for ti, inf_tree in enumerate(inf_ts.trees(sample_lists=True)):
        if ti % stride:
            continue
        # tsinfer can leave a tree uncoalesced (multiple roots), which
        # rf_distance / kc_distance reject. Skip those intervals.
        if inf_tree.num_roots != 1:
            continue
        left, right = inf_tree.interval.left, inf_tree.interval.right
        span = float(right - left)
        if span <= 0:
            continue
        mid = 0.5 * (left + right)
        true_tree = truth_ts.at(mid, sample_lists=True)
        if true_tree.num_roots != 1:
            continue
        rf_acc += span * (inf_tree.rf_distance(true_tree) / rf_max)
        kc_acc += span * inf_tree.kc_distance(true_tree, 1.0)
        span_acc += span
        for i, j in pairs:
            tt = true_tree.tmrca(i, j)
            ti_ = inf_tree.tmrca(i, j)
            if tt > 0 and ti_ > 0:
                t_true_all.append(tt)
                t_inf_all.append(ti_)

    t_true_arr = np.asarray(t_true_all)
    t_inf_arr = np.asarray(t_inf_all)
    # Spearman rho via rank-Pearson (no scipy dependency).
    rt = np.argsort(np.argsort(t_true_arr))
    ri = np.argsort(np.argsort(t_inf_arr))
    spearman = float(np.corrcoef(rt, ri)[0, 1])
    log_bias = float(np.mean(np.log(t_inf_arr) - np.log(t_true_arr)))
    slope = float(np.polyfit(t_true_arr, t_inf_arr, 1)[0])
    return {
        "any_tree_compared": bool(span_acc > 0),
        "rf_norm": rf_acc / span_acc if span_acc else float("nan"),
        "kc_lambda1": kc_acc / span_acc if span_acc else float("nan"),
        "tmrca_spearman": spearman,
        "tmrca_log_bias": log_bias,
        "tmrca_slope": slope,
        "n_tmrca_points": int(t_true_arr.size),
    }


def _brier_of_posteriors(stream) -> float:
    """Mean genealogy Brier over a ``(Site, Posterior)`` stream, scored against
    the true ancestral state on the same target sites as the figures (ingroup-
    polymorphic when ``n_out>0``)."""
    s = n = 0
    for site, post in stream:
        pos = int(site.pos)
        if ingroup_poly is not None and pos not in ingroup_poly:
            continue
        t = truth_states.get(pos)
        if t not in STATES:
            continue
        s += 1.0 - 2.0 * float(post[t]) + sum(p * p for p in post.values)
        n += 1
    return s / n if n else float("nan")


def _window_recovery(window_snps: int) -> dict:
    # Bake genotypes so the built ARG is self-describing and the kernel can
    # re-polarise it for the Brier overlay. Baking adds only site/mutation
    # tables, leaving node topology and times (and thus the genealogy metrics)
    # unchanged. Both metrics and Brier come from this one build on the
    # genealogy sim, so the overlay is self-consistent at every window --- the
    # 1- and 2-SNP windows the 100 Mb window bench cannot reach included.
    builder = LocalTreeBuilder(
        sites, mu=mu_true, rec_rate=r_true, sample_names=keep_names,
        sequence_length=float(ts.sequence_length), window=f"{window_snps}snp",
        bake_genotypes=True,
    )
    inf_ts = builder.to_tree_sequence()  # the builder's single plug-in tree
    # The genealogy metrics describe the reconstructed tree, so they are taken
    # from this build. The Brier is NOT: ensemble mode is what local-tree mode
    # runs by default and what the manuscript reports, and it marginalises over
    # drawn genealogies rather than scoring this one. Scoring the plug-in tree
    # reported the mode at its worse estimator against a tsinfer reference that
    # was handed the true ancestral allele.
    ens = LocalTreeInference(
        builder.sites, JC69(), mu=mu_true, rec_rate=r_true,
        sample_names=list(builder.sample_names),
        sequence_length=float(builder.sequence_length),
        window=f"{window_snps}snp", n_ensemble=ENSEMBLE_MEMBERS,
        ensemble_seed=1, progress=False,
        # Named so the default focal node resolves to the ingroup's own
        # ancestor. An unnamed panel is read at the panel root, which with
        # outgroups present is a different question.
        ingroup_samples=[n for n in builder.sample_names if n in ingroup_set] or None,
        outgroup_samples=[n for n in builder.sample_names
                          if n not in ingroup_set] or None,
    )
    out = {"window_snps": window_snps,
           **_metrics_vs_truth(inf_ts),
           "mean_brier": _brier_of_posteriors(ens.infer())}
    out.update(_ensemble_recovery(ens, window_snps))
    return out


def _ensemble_recovery(inf, window_snps: int) -> dict:
    """The same genealogy metrics over the ENSEMBLE the mode actually scores.

    The point tree above is the posterior-mean UPGMA build --- the plug-in
    estimator. Ensemble mode does not score it: it marginalises over
    genealogies drawn from the pairwise-HMM posterior, so a figure describing
    only the point tree describes a tree the default mode never uses. Each
    member is measured against the truth and the metrics are averaged, with
    the spread kept so the member-to-member variability is visible.

    :param inf: The ensemble inference whose members are measured. The
        caller's instance is reused: a second, argument-identical build would
        repeat the pairwise-HMM pass and all ENSEMBLE_MEMBERS draws.
    :param window_snps: Window width, for the record.
    :return: ``ens_<metric>`` means plus ``ens_<metric>_sd`` and the member
        count, prefixed so they sit beside the point-tree columns.
    """
    per_member: list[dict] = []
    for _interval, members in inf.to_tree_sequence():
        per_member.extend(_metrics_vs_truth(member) for member in members)
    if not per_member:
        return {"ens_members": 0}
    keys = [k for k, v in per_member[0].items() if isinstance(v, (int, float))]
    out: dict = {"ens_members": len(per_member)}
    for k in keys:
        vals = np.array([m[k] for m in per_member], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[f"ens_{k}"] = float(vals.mean()) if vals.size else float("nan")
        out[f"ens_{k}_sd"] = float(vals.std()) if vals.size else float("nan")
    return out


def _tsinfer_recovery() -> dict:
    """tsinfer ARG built from the same genotypes with the CORRECT (simulator)
    ancestral allele (tskit's ``variant.alleles[0]`` is the true ancestral
    state), as a genealogy-inference reference independent of window size."""
    import tsinfer
    # Build from the kept panel (truth_ts is the ingroup, plus the n_out
    # outgroups when n_out>0. It equals ts for n_out=0), so the tsinfer samples
    # align with the local-tree builder's and with truth_ts for the metrics.
    sd = tsinfer.SampleData(sequence_length=float(truth_ts.sequence_length))
    with sd:
        for v in truth_ts.variants():
            al = v.alleles
            if len(al) != 2 or any(a not in STATES for a in al if a is not None):
                continue
            if len(set(v.genotypes)) < 2:
                continue
            sd.add_site(v.site.position, v.genotypes, al)  # allele 0 = ancestral
    inf_ts = tsinfer.infer(sd, recombination_rate=r_true).simplify()
    # tsinfer leaves trees undated. Date them properly with the standard
    # tsinfer->tsdate pipeline (variational gamma) at the true mutation rate, so
    # node times land in generations and the absolute-time metrics (slope,
    # log-bias, KC) reflect a real dated ARG rather than a single-parameter
    # rate rescale. preprocess_ts handles tsinfer's unary / multi-root output.
    import tsdate
    inf_ts = tsdate.date(tsdate.preprocess_ts(inf_ts), mutation_rate=mu_true)
    metrics = _metrics_vs_truth(inf_ts)
    # Downstream polarisation Brier on the (now generation-calibrated) tsinfer
    # ARG, comparable to the local-tree Brier overlay.
    sm = {f"tsk_{i}": i for i in range(inf_ts.num_samples)}
    # The tsinfer samples follow keep_names, so index i is keep_names[i].
    # Naming the halves reads the posterior at the ingroup MRCA rather than at
    # the panel root, which with outgroups present is a different question.
    ing = [f"tsk_{i}" for i, n in enumerate(keep_names) if n in ingroup_set]
    outg = [f"tsk_{i}" for i, n in enumerate(keep_names) if n not in ingroup_set]
    pol = ARGBasedInference(inf_ts, JC69(), mu=mu_true, sample_map=sm,
                            ingroup_samples=ing or None,
                            outgroup_samples=outg or None,
                            progress=False)
    metrics["mean_brier"] = _brier_of_posteriors(pol.infer())
    return metrics


# One shard per window, plus one for the tsinfer reference. Each is an
# independent job. The merge concatenates them in WINDOWS order. A single
# job over all eleven windows at ENSEMBLE_MEMBERS draws exceeded the SLURM
# walltime, and the windows share no state.
if shard is not None:
    if shard == "tsinfer":
        payload = {"tsinfer": _tsinfer_recovery()}
    else:
        payload = {"window": _window_recovery(int(shard))}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"genealogy shard {shard}: wrote {out_json}", flush=True)
    raise SystemExit(0)

if shard_inputs:
    by_window, tsinfer_ref = {}, None
    for path in shard_inputs:
        with open(path) as f:
            d = json.load(f)
        if "tsinfer" in d:
            tsinfer_ref = d["tsinfer"]
        else:
            by_window[int(d["window"]["window_snps"])] = d["window"]
    results = [by_window[w] for w in WINDOWS if w in by_window]
else:
    results = []
    for w in WINDOWS:
        r = _window_recovery(w)
        results.append(r)
        print(
            f"  window={w:>4} SNPs  rho={r['tmrca_spearman']:.3f}  "
            f"logbias={r['tmrca_log_bias']:+.3f}  slope={r['tmrca_slope']:.3f}  "
            f"RF={r['rf_norm']:.3f}  KC={r['kc_lambda1']:.3g}",
            flush=True,
        )
    tsinfer_ref = _tsinfer_recovery() if run_tsinfer else None

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"meta": meta, "windows": results, "tsinfer": tsinfer_ref},
              f, indent=2)

lines = [
    "# Local-tree genealogy recovery (inferred vs. true local trees)",
    "",
    "| window (SNPs) | TMRCA rank rho | TMRCA log-bias | TMRCA slope | "
    "RF (norm) | KC (lambda=1) |",
    "|---|---|---|---|---|---|",
]
for r in results:
    lines.append(
        f"| {r['window_snps']} | {r['tmrca_spearman']:.3f} | "
        f"{r['tmrca_log_bias']:+.3f} | {r['tmrca_slope']:.3f} | "
        f"{r['rf_norm']:.3f} | {r['kc_lambda1']:.3g} |"
    )
if out_md is not None:
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
print(f"genealogy bench: wrote {out_json} + {out_md}", flush=True)
