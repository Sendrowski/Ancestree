"""Spectra and divergence counts of one simulated replicate against truth.

Reduces the simulated states at the two nodes, and the posterior matrices of
every inference run, to the unfolded spectrum of the sample and the number of
sites it has fixed for a derived base, by mutation-rate class, over all sites
and over the sites of discordant local trees alone. Each run is also scored
per site by its Brier score against the simulated state at the node it is
read at, over the sites with two alleles in the sample.
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
    in_local_tree = dict(zip(snakemake.params.runs, snakemake.input.local_tree))  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    import glob
    seed = 1
    in_npz = f"results/data/epo_simulation_panel_seed{seed}.npz"
    in_meta = f"results/data/epo_simulation_panel_seed{seed}_meta.json"
    in_fixed_tree = f"results/data/epo_simulation_fixed_tree_seed{seed}.npz"
    in_fixed_tree_sample = f"results/data/epo_simulation_fixed_tree_sample_seed{seed}.npz"
    in_local_tree = {
        Path(p).stem.split(f"seed{seed}_", 1)[1]: p
        for p in sorted(glob.glob(f"results/data/epo_simulation_local_tree_seed{seed}_*.npz"))}
    out_json = f"results/data/epo_simulation_spectra_seed{seed}.json"

panel = np.load(in_npz)
with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
sample = panel["sample"]
counts = np.stack([(sample == base).sum(1) for base in range(4)], axis=1)
discordant = ~panel["concordant"]
classes = {"normal": ~panel["fast"], "hypermutable": panel["fast"]}
classes.update({name + ", discordant": mask & discordant for name, mask in list(classes.items())})


def one_hot(states: np.ndarray) -> np.ndarray:
    out = np.zeros((len(states), 4))
    out[np.arange(len(states)), states] = 1.0
    return out


n = int(meta["n_haplotypes"])
result = dict(
    n_eligible={"normal": meta["n_eligible"] - meta["n_eligible_fast"],
                "hypermutable": meta["n_eligible_fast"]},
    n_haplotypes=meta["n_haplotypes"], discordant_fraction=meta["discordant_fraction"],
    methods={
        "truth_node": summarise(one_hot(panel["truth_node"]), counts, classes, n),
        "truth_mrca": summarise(one_hot(panel["truth_mrca"]), counts, classes, n),
    })
posteriors = {}
fixed_tree = np.load(in_fixed_tree)
for name in fixed_tree.files:
    posteriors["fixed_tree_reference_" + name] = fixed_tree[name].astype(float)
fixed_tree_sample = np.load(in_fixed_tree_sample)
for name in ("sample_mrca", "sample_node"):
    posteriors["fixed_tree_" + name] = fixed_tree_sample[name].astype(float)
for run, path in in_local_tree.items():
    posteriors["local_tree_" + run] = np.load(path)["post"].astype(float)
for method, post in posteriors.items():
    result["methods"][method] = summarise(post, counts, classes, n)

# Per-site Brier score against the truth at the run's own node.
two_alleles = (counts > 0).sum(1) == 2
result["brier"] = {}
for method in result["methods"]:
    if method.startswith("truth"):
        continue
    truth = one_hot(panel["truth_mrca" if "_mrca" in method else "truth_node"])
    score = ((posteriors[method] - truth) ** 2).sum(1)
    result["brier"][method] = {
        name: dict(total=float(score[mask & two_alleles].sum()),
                   n=int((mask & two_alleles).sum()))
        for name, mask in classes.items()}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(f"summarise_epo_simulation: {len(sample):,} rows -> {out_json}", flush=True)
