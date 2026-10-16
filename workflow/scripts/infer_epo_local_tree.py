"""Local-tree posteriors of one run on a panel.

The run name ``<panel>_<node>_<rate>_out<n>`` selects the human tips (the
sample or the reference sequence), the node (the ancestor of those tips, or
their ancestor with the Pan species), a single mutation rate or one rate per
site class, and the outgroup subset. A row that does not vary within the run's
panel takes its one base with certainty.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import (  # noqa: E402
    PAN, REFERENCE, SPECIES, interval_rate_map, local_tree_posteriors, parse_run,
    posterior_matrix, sites_from_rows,
)

from ancestree import BaseComposition, HKY  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_rec_map = snakemake.input.get("rec_map")  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    run = str(snakemake.wildcards.epo_run)  # type: ignore[name-defined]
    n_ensemble = int(snakemake.params.n_ensemble)  # type: ignore[name-defined]
    threads = int(snakemake.threads)  # type: ignore[name-defined]
except NameError:
    region, run = "chr1_143750000_150M", "sample_mrca_class_rate_out9"
    in_npz = f"results/data/epo_panel_{region}.npz"
    in_meta = f"results/data/epo_panel_{region}_meta.json"
    in_rec_map = "external/genetic_maps/HapMapII_GRCh38/genetic_map_Hg38_chr1.txt"
    out_npz = f"results/data/epo_local_tree_{region}_{run}.npz"
    n_ensemble, threads = 64, 2

import numba  # noqa: E402

numba.set_num_threads(max(1, min(threads, numba.config.NUMBA_NUM_THREADS)))

spec = parse_run(run)
panel = np.load(in_npz)
with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
n_rows = len(panel["pos"])
length = int(meta["sequence_length"])

species_names = list(spec["outgroups"])
columns = [SPECIES.index(s) for s in species_names]
if spec["panel"] == "sample":
    tips = [f"h{i}" for i in range(panel["sample"].shape[1])]
    human = panel["sample"]
else:
    tips = [REFERENCE]
    human = panel["reference"][:, None]
alleles = np.column_stack([human, panel["species"][:, columns]])
names = tips + species_names
sites, rows = sites_from_rows(panel["pos"], alleles, names)
ingroup = tips + ([p for p in PAN if p in species_names] if spec["node"] == "node" else [])

t0 = time.perf_counter()
posteriors = []
if sites:
    kwargs = dict(
        model=HKY(kappa=float(meta["kappa"])),
        base_composition=BaseComposition.from_n_target_sites(
            max(int(meta["n_eligible"]), 1), stationary=np.asarray(meta["pi"])),
        rec_rate=1e-8, sequence_length=length, n_ensemble=n_ensemble)
    if len(panel["access"]):
        kwargs["accessibility"] = [tuple(map(int, iv)) for iv in panel["access"]]
    if in_rec_map:
        import msprime

        rec = msprime.RateMap.read_hapmap(
            in_rec_map, position_col=1, rate_col=2,
            sequence_length=max(int(meta["start"]) + length, 248_956_423))
        kwargs["recombination_map"] = rec.slice(
            left=int(meta["start"]), right=int(meta["start"]) + length, trim=True)
    mu, ratio = float(meta["mu"]), float(meta["rate_ratio"])
    if spec["class_rate"]:
        # The mean over the eligible positions stays mu.
        share = meta["n_eligible_fast"] / max(meta["n_eligible"], 1)
        slow = mu / (1.0 - share + share * ratio)
        kwargs["mu"] = slow
        kwargs["mutation_map"] = interval_rate_map(
            panel["fast_start"], panel["fast_end"], length, slow * ratio, slow)
    else:
        kwargs["mu"] = mu
    posteriors = local_tree_posteriors(sites, names, ingroup, **kwargs)

# A row this panel does not vary at carries one base at every tip.
default = np.where(human[:, 0] >= 0, human[:, 0], 0)
post = posterior_matrix(n_rows, rows, posteriors, default=default)
Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, post=post.astype(np.float32),
                    seconds=time.perf_counter() - t0, n_scored=len(sites))
print(f"infer_epo_local_tree: {run}, {len(sites):,} of {n_rows:,} rows in "
      f"{time.perf_counter() - t0:.0f}s -> {out_npz}", flush=True)
