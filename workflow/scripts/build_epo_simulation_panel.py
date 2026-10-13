"""Assemble the site panel of one simulated replicate, with its truth.

The panel has the layout of ``build_epo_panel.py``, so the same inference
steps score it, and adds the simulated state at the two nodes a site is read
at: the common ancestor of the sample (``truth_mrca``) and its common ancestor
with the two closest outgroups (``truth_node``), the node EPO reports.
``concordant`` marks the sites whose local tree places that second ancestor
below the third outgroup, as the species tree does.

Outgroup bases are set missing in the pattern of a real chr1 panel, row for
row, so the replicate carries the alignment gaps of the data it is matched to.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tskit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import REFERENCE, SPECIES  # noqa: E402

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
    in_gaps = "results/data/epo_panel_chr1_100M_106250000.npz"
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


gaps = np.load(in_gaps)["species"] < 0
pos, tips, truth_node, truth_mrca, concordant = [], [], [], [], []
discordant_span = 0.0
n_variant = 0
for tree in ts.trees():
    mrca = tree.mrca(*sample_nodes)
    node = tree.mrca(*sample_nodes, *pan_nodes)
    ok = not tree.is_descendant(gorilla_node, node)
    if not ok:
        discordant_span += tree.span
    for site in tree.sites():
        row = [state_at(tree, site, s) for s in samples]
        if len(set(row[:n_hap] + row[n_hap + 1:])) < 2:
            continue  # no second base in the sample or an outgroup
        for j in np.flatnonzero(gaps[n_variant % len(gaps)]):
            row[n_hap + 1 + j] = -1
        n_variant += 1
        if len(set(row[:n_hap] + row[n_hap + 1:]) - {-1}) < 2:
            continue  # the second base sat in a gap
        pos.append(int(site.position) + 1)
        tips.append(row)
        truth_node.append(state_at(tree, site, node))
        truth_mrca.append(state_at(tree, site, mrca))
        concordant.append(ok)
tips = np.asarray(tips, dtype=np.int8).reshape(-1, len(samples))
pos = np.asarray(pos, dtype=np.int32)

fast_positions = np.arange(0, length, period)
slow, ratio = float(cfg["mu"]), float(cfg["fast_ratio"])
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


Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_npz, pos=pos, sample=tips[:, :n_hap], reference=tips[:, n_hap],
    species=tips[:, n_hap + 1:], fast=(pos - 1) % period == 0,
    access=np.zeros((0, 2), np.int64),
    fast_start=fast_positions + 1, fast_end=fast_positions + 2,
    truth_node=np.asarray(truth_node, np.int8), truth_mrca=np.asarray(truth_mrca, np.int8),
    concordant=np.asarray(concordant, bool))
assert tips.shape[1] == n_hap + 1 + len(SPECIES)
model = dict(kappa=float(cfg["kappa"]), pi=list(cfg["pi"]))
meta = dict(
    start=0, sequence_length=length + 1, n_haplotypes=n_hap, n_rows=int(len(pos)),
    n_eligible=length, n_eligible_fast=int(len(fast_positions)),
    mu=mean_rate, rate_ratio=ratio, **model,
    tree=tree_newick(),
    fixed_tree_runs=[
        dict(name="one_rate", **model, scale_slow=mean_rate, scale_fast=mean_rate),
        dict(name="class_rate", **model, scale_slow=slow, scale_fast=slow * ratio),
    ],
    discordant_fraction=discordant_span / length,
)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"build_epo_simulation_panel: {len(pos):,} rows, discordant fraction "
      f"{meta['discordant_fraction']:.3f}", flush=True)
