"""Assemble the site panel of one simulated replicate, with its truth.

The panel has the layout of ``build_epo_panel.py``, so the same inference
steps score it, and adds the simulated state at the two nodes a site is read
at: the common ancestor of the sample (``truth_mrca``) and its common ancestor
with the two closest outgroups (``truth_node``), the node EPO reports.
``concordant`` marks the sites whose local tree places that second ancestor
below the third outgroup, as the species tree does.

Outgroup bases are set missing position by position in the pattern of an
analysed chr1 chunk, so the replicate carries the alignment gaps of the data.
The fixed-tree runs scale the species tree by the substitution rate fitted by
maximum likelihood to the reference and outgroup columns of every position,
once over all sites and once per rate class, as the data are.
"""
import json
import sys
from pathlib import Path

import msprime
import numpy as np
import tskit
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import REFERENCE, SPECIES  # noqa: E402

from ancestree import STATES, BaseComposition, FixedTree, HKY, Site  # noqa: E402
from ancestree.likelihood import Likelihood  # noqa: E402

try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_gaps = snakemake.input.gaps  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
except NameError:
    seed = 1
    in_trees = f"results/data/epo_simulation_seed{seed}.trees"
    in_meta = f"results/data/epo_simulation_seed{seed}_meta.json"
    in_gaps = f"results/data/epo_simulation_gaps_seed{seed}.npz"
    out_npz = f"results/data/epo_simulation_panel_seed{seed}.npz"
    out_meta = f"results/data/epo_simulation_panel_seed{seed}_meta.json"

ts = tskit.load(in_trees)
with open(in_meta) as f:
    cfg = json.load(f)
n_hap, length, period = int(cfg["n_haplotypes"]), int(cfg["length"]), int(cfg["period"])
samples = list(ts.samples())
sample_nodes = samples[:n_hap]
pan_nodes = samples[n_hap + 1:n_hap + 3]
gorilla_node = samples[n_hap + 3]
BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


def state_at(tree, site, node) -> int:
    """Simulated base at ``node``."""
    by_node = {m.node: m for m in site.mutations}
    while node != tskit.NULL:
        if node in by_node:
            return BASE[by_node[node].derived_state]
        node = tree.parent(node)
    return BASE[site.ancestral_state]


gaps = np.load(in_gaps)["gaps"]
assert gaps.shape == (length, len(SPECIES))
pos, tips, truth_node, truth_mrca, concordant = [], [], [], [], []
# Reference and outgroup column of every position with a simulated site.
columns, column_pos = [], []
discordant_span = 0.0
for tree in ts.trees():
    mrca = tree.mrca(*sample_nodes)
    node = tree.mrca(*sample_nodes, *pan_nodes)
    ok = not tree.is_descendant(gorilla_node, node)
    if not ok:
        discordant_span += tree.span
    for site in tree.sites():
        row = [state_at(tree, site, s) for s in samples]
        for j in np.flatnonzero(gaps[int(site.position)]):
            row[n_hap + 1 + j] = -1
        columns.append(row[n_hap:])
        column_pos.append(int(site.position))
        if len(set(row[:n_hap] + row[n_hap + 1:]) - {-1}) < 2:
            continue  # no second base in the sample or an aligned outgroup
        pos.append(int(site.position) + 1)
        tips.append(row)
        truth_node.append(state_at(tree, site, node))
        truth_mrca.append(state_at(tree, site, mrca))
        concordant.append(ok)
tips = np.asarray(tips, dtype=np.int8).reshape(-1, len(samples))
pos = np.asarray(pos, dtype=np.int32)

fast_positions = np.arange(0, length, period)
model = dict(kappa=float(cfg["kappa"]), pi=list(cfg["pi"]))
# Mutations that keep the base are not substitutions.
P = np.asarray(msprime.HKY(kappa=model["kappa"], equilibrium_frequencies=model["pi"])
               .transition_matrix)
non_silent = float((np.asarray(model["pi"]) * (1.0 - np.diag(P))).sum())
slow, ratio = float(cfg["mu"]) * non_silent, float(cfg["fast_ratio"])
mean_rate = slow * (period - 1 + ratio) / period
generations = cfg["splits"]


def tree_newick() -> str:
    """The species tree in generations, the human tip the reference."""
    t = generations
    cb = f"(pan_troglodytes:{t['CB']},pan_paniscus:{t['CB']})"
    hcb = f"({cb}:{t['HCB'] - t['CB']},{REFERENCE}:{t['HCB']})"
    hcbg = f"({hcb}:{t['HCBG'] - t['HCB']},gorilla_gorilla:{t['HCBG']})"
    ga = f"({hcbg}:{t['GA'] - t['HCBG']},pongo_abelii:{t['GA']})"
    ape = f"({ga}:{t['APE'] - t['GA']},nomascus_leucogenys:{t['APE']})"
    mf = f"(macaca_mulatta:{t['MF']},macaca_fascicularis:{t['MF']})"
    mfv = f"({mf}:{t['MFV'] - t['MF']},chlorocebus_sabaeus:{t['MFV']})"
    cat = f"({ape}:{t['CAT'] - t['APE']},{mfv}:{t['CAT'] - t['MFV']})"
    return f"({cat}:{t['ROOT'] - t['CAT']},microcebus_murinus:{t['ROOT']});"


def class_columns() -> dict[bool, dict[tuple, float]]:
    """Site count of each distinct reference and outgroup column, per rate
    class. A position without a simulated site carries one base, drawn from
    the stationary frequencies, on every aligned tip."""
    is_fast = np.zeros(length, bool)
    is_fast[fast_positions] = True
    cols = np.asarray(columns, np.int8).reshape(-1, 1 + len(SPECIES))
    at = np.asarray(column_pos, np.int64)
    no_site = np.ones(length, bool)
    no_site[at] = False
    out = {}
    for cls in (False, True):
        unique, count = np.unique(cols[is_fast[at] == cls], axis=0, return_counts=True)
        weights = {tuple(int(a) for a in u): float(c) for u, c in zip(unique, count)}
        patterns, n = np.unique(gaps[no_site & (is_fast == cls)], axis=0, return_counts=True)
        for gap, c in zip(patterns, n):
            for b, p in enumerate(model["pi"]):
                key = (b, *(-1 if g else b for g in gap))
                weights[key] = weights.get(key, 0.0) + c * p
        out[cls] = weights
    return out


def fitted_scale(weights: dict[tuple, float], guess: float) -> float:
    """ML substitution rate per generation of the species tree for the
    columns weighted."""
    names = [REFERENCE] + SPECIES
    keys = list(weights)
    w = np.array([weights[k] for k in keys])
    sites = [Site(chrom="1", pos=i + 1,
                  alleles=tuple(dict.fromkeys(STATES[a] for a in key if a >= 0)),
                  tip_alleles={n: STATES[a] if a >= 0 else None for n, a in zip(names, key)})
             for i, key in enumerate(keys)]
    engine = Likelihood(HKY(kappa=model["kappa"]), base_composition=BaseComposition.from_n_target_sites(
        length, stationary=np.asarray(model["pi"])))
    log_pi = np.log(model["pi"])

    def negative_log_likelihood(log_scale):
        tree = FixedTree.from_newick(tree_newick(), time_scale=float(np.exp(log_scale)))
        return -float((w * logsumexp(engine.log_likelihoods(tree, sites) + log_pi, axis=1)).sum())

    best = minimize_scalar(negative_log_likelihood, bounds=(np.log(guess / 20), np.log(guess * 20)),
                           method="bounded", options={"xatol": 1e-4})
    return float(np.exp(best.x))


by_class = class_columns()
scale_slow = fitted_scale(by_class[False], slow)
scale_fast = fitted_scale(by_class[True], slow * ratio)
pooled = dict(by_class[False])
for key, c in by_class[True].items():
    pooled[key] = pooled.get(key, 0.0) + c
scale_one = fitted_scale(pooled, mean_rate)

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_npz, pos=pos, sample=tips[:, :n_hap], reference=tips[:, n_hap],
    species=tips[:, n_hap + 1:], fast=(pos - 1) % period == 0,
    access=np.zeros((0, 2), np.int64),
    fast_start=fast_positions + 1, fast_end=fast_positions + 2,
    truth_node=np.asarray(truth_node, np.int8), truth_mrca=np.asarray(truth_mrca, np.int8),
    concordant=np.asarray(concordant, bool))
assert tips.shape[1] == n_hap + 1 + len(SPECIES)
meta = dict(
    start=0, sequence_length=length + 1, n_haplotypes=n_hap, n_rows=int(len(pos)),
    n_eligible=length, n_eligible_fast=int(len(fast_positions)),
    mu=mean_rate, rate_ratio=ratio, **model,
    tree=tree_newick(),
    fixed_tree_runs=[
        dict(name="one_rate", **model, scale_slow=scale_one, scale_fast=scale_one),
        dict(name="class_rate", **model, scale_slow=scale_slow, scale_fast=scale_fast),
    ],
    non_silent_fraction=non_silent,
    discordant_fraction=discordant_span / length,
)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"build_epo_simulation_panel: {len(pos):,} rows, discordant fraction "
      f"{meta['discordant_fraction']:.3f}, fitted rate per generation {scale_slow:.3g} "
      f"and {scale_fast:.3g} (true {slow:.3g} and {slow * ratio:.3g}), pooled {scale_one:.3g}",
      flush=True)
