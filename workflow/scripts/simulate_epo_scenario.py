"""Simulate one replicate of the scenario matched to the EPO comparison.

A sample of ingroup haplotypes and one further ingroup haplotype as the
reference, with nine outgroups on the topology and depths of the Ensembl
10-primate tree. Substitutions follow the Ortheus default model, HKY at 40% GC
with an expected transition/transversion ratio of 2, and every ``period``-th
site mutates ``fast_ratio`` times faster.
"""
import json
from pathlib import Path

import msprime
import numpy as np

try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.epo_seed)  # type: ignore[name-defined]
    cfg = dict(snakemake.params.config)  # type: ignore[name-defined]
except NameError:
    seed = 1
    out_trees = f"results/data/epo_simulation_seed{seed}.trees"
    out_meta = f"results/data/epo_simulation_seed{seed}_meta.json"
    cfg = dict(
        length=2_000_000, n_haplotypes=20, mu=1.25e-8, rec_rate=1e-8, fast_ratio=10.0, period=20,
        kappa=25.0 / 6.0, pi=[0.3, 0.2, 0.2, 0.3],
        splits=dict(CB=220_000, HCB=528_000, HCBG=676_000, GA=1_355_000, APE=1_568_000,
                    MF=226_000, MFV=937_000, CAT=2_444_000, ROOT=6_080_000),
        ingroup_size=20_000, ancestral_size=30_000, human_pan_ancestral_size=60_000)

#: Population of each outgroup, in the order of ``_epo_common.SPECIES``.
POPULATIONS = ["C", "B", "G", "O", "N", "M", "F", "V", "X"]
SPLITS = cfg["splits"]

dem = msprime.Demography()
dem.add_population(name="H", initial_size=cfg["ingroup_size"])
for name in POPULATIONS:
    dem.add_population(name=name, initial_size=cfg["ancestral_size"])
for name in SPLITS:
    dem.add_population(name=name, initial_size=(
        cfg["human_pan_ancestral_size"] if name == "HCB" else cfg["ancestral_size"]))
for ancestral, derived in (
        ("CB", ["C", "B"]), ("HCB", ["H", "CB"]), ("HCBG", ["HCB", "G"]), ("GA", ["HCBG", "O"]),
        ("APE", ["GA", "N"]), ("MF", ["M", "F"]), ("MFV", ["MF", "V"]),
        ("CAT", ["APE", "MFV"]), ("ROOT", ["CAT", "X"])):
    dem.add_population_split(time=SPLITS[ancestral], derived=derived, ancestral=ancestral)
dem.sort_events()

length, period = int(cfg["length"]), int(cfg["period"])
# Samples: the ingroup haplotypes, the reference, then one haplotype per outgroup.
ts = msprime.sim_ancestry(
    samples=[msprime.SampleSet(cfg["n_haplotypes"] + 1, population="H", ploidy=1)]
    + [msprime.SampleSet(1, population=p, ploidy=1) for p in POPULATIONS],
    demography=dem, sequence_length=length, recombination_rate=cfg["rec_rate"],
    random_seed=seed)
edges = np.unique(np.concatenate([np.arange(0, length + 1, period), np.arange(1, length, period)]))
rates = np.where(edges[:-1] % period == 0, cfg["mu"] * cfg["fast_ratio"], cfg["mu"])
ts = msprime.sim_mutations(
    ts, rate=msprime.RateMap(position=edges.astype(float), rate=rates), random_seed=seed + 1,
    model=msprime.HKY(kappa=cfg["kappa"], equilibrium_frequencies=cfg["pi"]))

Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts.dump(out_trees)
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(dict(seed=seed, n_sites=int(ts.num_sites), n_trees=int(ts.num_trees), **cfg), f, indent=2)
print(f"simulate_epo_scenario: seed {seed}, {ts.num_sites:,} sites, {ts.num_trees:,} trees", flush=True)
