"""Spectra and divergence counts of one region under every polarisation.

Reads the region's panel and the posterior matrices of its inference runs and
reduces each to the unfolded spectrum of the sample and the number of sites
the sample has fixed for a derived base, by CpG context, from MAP estimates
and from posterior weights. The Ensembl EPO call is reduced the same way.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import summarise  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_fixed_tree = snakemake.input.fixed_tree  # type: ignore[name-defined]
    in_fixed_tree_sample = snakemake.input.fixed_tree_sample  # type: ignore[name-defined]
    in_fixed_tree_reference_ladder = snakemake.input.fixed_tree_reference_ladder  # type: ignore[name-defined]
    in_local_tree = dict(zip(snakemake.params.runs, snakemake.input.local_tree))  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    region = "chr1_143750000_150M"
    in_npz = f"results/data/epo_panel_{region}.npz"
    in_meta = f"results/data/epo_panel_{region}_meta.json"
    in_fixed_tree = f"results/data/epo_fixed_tree_{region}.npz"
    in_fixed_tree_sample = f"results/data/epo_fixed_tree_sample_{region}.npz"
    in_fixed_tree_reference_ladder = f"results/data/epo_fixed_tree_reference_ladder_{region}.npz"
    in_local_tree = {r: f"results/data/epo_local_tree_{region}_{r}.npz" for r in (
        "reference_node_class_rate_out9", "sample_node_class_rate_out9",
        "sample_mrca_class_rate_out9", "sample_mrca_one_rate_out9")}
    out_json = f"results/data/epo_spectra_{region}.json"

panel = np.load(in_npz)
with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
sample = panel["sample"]
counts = np.stack([(sample == base).sum(1) for base in range(4)], axis=1)
# Polymorphic sites whose EPO base is neither sample allele have no EPO
# polarisation and are left out of every spectrum, so all share one site set.
present = counts > 0
unpolarised = (present.sum(1) == 2) & ~present[np.arange(len(sample)), panel["epo"] % 4]
classes = {"non-CpG": ~panel["fast"] & ~unpolarised, "CpG": panel["fast"] & ~unpolarised}

result = dict(
    n_eligible={"non-CpG": meta["n_eligible"] - meta["n_eligible_fast"],
                "CpG": meta["n_eligible_fast"]},
    n_without_epo_call=meta["n_without_epo_call"],
    n_haplotypes=meta["n_haplotypes"], kappa=meta["kappa"], methods={}, seconds={})

# The EPO call as a point mass on its base. Upper case marks high confidence.
n = int(meta["n_haplotypes"])
epo = np.zeros((len(sample), 4))
epo[np.arange(len(sample)), panel["epo"] % 4] = 1.0
result["methods"]["epo"] = summarise(epo, counts, classes, n)
result["methods"]["epo_high_confidence"] = summarise(
    epo, counts, classes, n, called=panel["epo"] < 4)

fixed_tree = np.load(in_fixed_tree)
for name in fixed_tree.files:
    result["methods"]["fixed_tree_reference_" + name] = summarise(
        fixed_tree[name].astype(float), counts, classes, n)
fixed_tree_sample = np.load(in_fixed_tree_sample)
for name in ("sample_mrca", "sample_node"):
    result["methods"]["fixed_tree_" + name] = summarise(
        fixed_tree_sample[name].astype(float), counts, classes, n)
result["methods"]["fixed_tree_reference_ladder_node"] = summarise(
    np.load(in_fixed_tree_reference_ladder)["reference_node"].astype(float), counts, classes, n)
for run, path in in_local_tree.items():
    z = np.load(path)
    result["methods"]["local_tree_" + run] = summarise(z["post"].astype(float), counts, classes, n)
    result["seconds"]["local_tree_" + run] = float(z["seconds"])

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(f"summarise_epo_region: {len(sample):,} rows -> {out_json}", flush=True)
