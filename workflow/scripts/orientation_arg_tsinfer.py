"""Build one tsinfer + tsdate ARG under a single orientation scheme.

The panel is the 20 ingroup haplotypes, the genotypes are the shared site set,
and the only thing the scheme changes is which allele is handed to tsinfer as
ancestral. The reference arm passes no ancestral allele at all, leaving
tsinfer's own default in force, so it is bit-identical to a run that knows
nothing about the orientation machinery.

Wildcard: ``{scheme}``. Run via snakemake::

    snakemake -j 1 results/data/orientation_arg_tsinfer_random5.trees
"""
import json
import os
import sys
import time

import tsdate
import tsinfer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import load_orientation, load_sites  # noqa: E402


try:
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    in_map = str(snakemake.input.orientation)  # type: ignore[name-defined]
    out_trees = str(snakemake.output.trees)  # type: ignore[name-defined]
    scheme = str(snakemake.wildcards.scheme)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    scheme = "random5"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    in_map = f"{DATA}/orientation_map_{scheme}.json"
    out_trees = f"{DATA}/orientation_arg_tsinfer_{scheme}.trees"


meta = json.loads(open(in_meta).read())
mu = float(meta["mu"])
rec_rate = float(meta["rec_rate"])

sites = load_sites(in_sites)
pos = sites["pos"]
genotypes = sites["genotypes"]
alleles = sites["alleles"]
anc_idx, _ = load_orientation(in_map, pos.size)

t0 = time.perf_counter()
sd = tsinfer.SampleData(sequence_length=float(meta["length"]))
with sd:
    for k in range(pos.size):
        al = (str(alleles[k, 0]), str(alleles[k, 1]))
        if scheme == "true_aa":
            sd.add_site(float(pos[k]), genotypes[k], al)
        else:
            sd.add_site(float(pos[k]), genotypes[k], al,
                        ancestral_allele=int(anc_idx[k]))

arg = tsinfer.infer(sd, recombination_rate=rec_rate).simplify()
arg = tsdate.date(tsdate.preprocess_ts(arg), mutation_rate=mu)
arg.dump(out_trees)
print(f"tsinfer/{scheme}: num_trees={arg.num_trees} "
      f"num_samples={arg.num_samples} runtime={time.perf_counter() - t0:.1f}s",
      flush=True)
