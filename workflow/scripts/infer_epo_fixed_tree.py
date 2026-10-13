"""Fixed-tree posteriors at the node EPO reports, on a panel's species tree.

The human tip is the reference sequence and every outgroup one tip, so the
posterior is read at the parent of the human tip. Each run listed in the panel
metadata gives one row-aligned posterior matrix. A row at which the reference
and every aligned outgroup carry one base takes that base with certainty.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import REFERENCE, SPECIES, posterior_matrix, sites_from_rows  # noqa: E402

from ancestree import BaseComposition, FixedTree, FixedTreeInference, HKY  # noqa: E402
from ancestree.focal import FocalNode  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
except NameError:
    region = "chr1_143750000_150M"
    in_npz = f"results/data/epo_panel_{region}.npz"
    in_meta = f"results/data/epo_panel_{region}_meta.json"
    out_npz = f"results/data/epo_fixed_tree_{region}.npz"

panel = np.load(in_npz)
with open(in_meta) as f:
    meta = json.load(f)
n_rows = len(panel["pos"])
alleles = np.column_stack([panel["reference"], panel["species"]])
sites, rows = sites_from_rows(panel["pos"], alleles, [REFERENCE] + SPECIES)
fast = panel["fast"][rows] if len(rows) else np.zeros(0, bool)

out = {}
for run in meta["fixed_tree_runs"]:
    composition = BaseComposition.from_n_target_sites(
        max(int(meta["n_eligible"]), 1), stationary=np.asarray(run["pi"]))
    posteriors = [None] * len(sites)
    for is_fast, scale in ((False, run["scale_slow"]), (True, run["scale_fast"])):
        select = np.flatnonzero(fast == is_fast)
        if not len(select):
            continue
        inference = FixedTreeInference(
            [sites[i] for i in select], HKY(kappa=run["kappa"]),
            tree=FixedTree.from_newick(meta["tree"], time_scale=float(scale)),
            base_composition=composition, ingroup_samples=[REFERENCE],
            focal=FocalNode("ingroup_mrca", coalescences=1),
            parallelize=False, progress=False)
        for i, (_, posterior) in zip(select, inference.infer()):
            posteriors[i] = posterior
    out[run["name"]] = posterior_matrix(
        n_rows, rows, posteriors, default=panel["reference"]).astype(np.float32)

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, **out)
print(f"infer_epo_fixed_tree: {len(sites):,} of {n_rows:,} rows scored -> {out_npz}", flush=True)
