"""Fixed-tree posteriors of the ingroup sample on a fitted outgroup ladder.

The sample's haplotypes form the ingroup, weighted by the Kingman spectrum,
and one outgroup per nested lineage forms the ladder, whose branch rates are
fitted separately for each site class. The posterior is read at the ingroup
MRCA (``sample_mrca``) and at its first join with the ladder, the ancestor it
shares with chimpanzee (``sample_node``). A row that does not vary within the
panel takes its one base with certainty.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import LADDER, SPECIES, posterior_matrix, sites_from_rows  # noqa: E402

from ancestree import (  # noqa: E402
    BaseComposition, FixedTreeInference, HKY, KingmanIngroupWeight, OutgroupLadderTree,
)
from ancestree.focal import FocalNode  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
except NameError:
    region = "chr1_93750000_100M"
    in_npz = f"results/data/epo_panel_{region}.npz"
    in_meta = f"results/data/epo_panel_{region}_meta.json"
    out_npz = f"results/data/epo_fixed_tree_sample_{region}.npz"

panel = np.load(in_npz)
with open(in_meta) as f:
    meta = json.load(f)
n_rows = len(panel["pos"])
tips = [f"h{i}" for i in range(panel["sample"].shape[1])]
columns = [SPECIES.index(s) for s in LADDER]
alleles = np.column_stack([panel["sample"], panel["species"][:, columns]])
sites, rows = sites_from_rows(panel["pos"], alleles, tips + LADDER)
fast = panel["fast"][rows] if len(rows) else np.zeros(0, bool)
n_class = {False: meta["n_eligible"] - meta["n_eligible_fast"], True: meta["n_eligible_fast"]}
NODES = {"sample_mrca": FocalNode("ingroup_mrca"),
         "sample_node": FocalNode("ingroup_mrca", coalescences=1)}

t0 = time.perf_counter()
posteriors = {name: [None] * len(sites) for name in NODES}
rates = {}
for is_fast in (False, True):
    select = np.flatnonzero(fast == is_fast)
    if not len(select):
        continue
    subset = [sites[i] for i in select]
    for name, focal in NODES.items():
        inference = FixedTreeInference(
            subset, HKY(kappa=float(meta["kappa"])),
            tree=OutgroupLadderTree(tips, LADDER),
            base_composition=BaseComposition.from_n_target_sites(
                max(int(n_class[is_fast]), 1), stationary=np.asarray(meta["pi"])),
            n_target_sites=max(int(n_class[is_fast]), 1),
            ingroup_weight=KingmanIngroupWeight(ingroup_samples=tips),
            focal=focal, n_starts=4, parallelize=False, progress=False)
        fitted = inference.fit()
        rates[("CpG" if is_fast else "non-CpG") + ", " + name] = str(fitted)
        for i, (_, posterior) in zip(select, inference.infer()):
            posteriors[name][i] = posterior

default = panel["sample"][:, 0] if n_rows else np.zeros(0, np.int8)
out = {name: posterior_matrix(n_rows, rows, post, default=default).astype(np.float32)
       for name, post in posteriors.items()}
Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, rates=json.dumps(rates), seconds=time.perf_counter() - t0, **out)
print(f"infer_epo_fixed_tree_sample: {len(sites):,} of {n_rows:,} rows in "
      f"{time.perf_counter() - t0:.0f}s, rates {rates} -> {out_npz}", flush=True)
