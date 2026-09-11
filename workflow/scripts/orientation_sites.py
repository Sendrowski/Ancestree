"""Build the shared site set the orientation-damage experiment scores on.

A site enters if it is biallelic, both alleles are nucleotides, and it
segregates among the 20 ingroup haplotypes. The genotypes stored here are the
ingroup slice only: the outgroups of the reference panel exist so the
ancestral-allele calls of the fixed-tree schemes have something to condition
on, and never reach the ARG tools.

Run via snakemake::

    snakemake -j 1 results/data/orientation_sites.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import N_OUT_REFERENCE, OrientationPanel  # noqa: E402


try:
    in_trees = str(snakemake.input.trees)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    out_npz = str(snakemake.output.npz)  # type: ignore[name-defined]
    n_out_reference = int(snakemake.params.n_out_reference)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    in_trees = f"{DATA}/local_tree_genealogy_og_sim_20mb.trees"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    out_npz = f"{DATA}/orientation_sites.npz"
    n_out_reference = N_OUT_REFERENCE


panel = OrientationPanel(in_trees, in_meta,
                         n_out_reference=n_out_reference).load()
sites = panel.shared_sites()
pos = sites["pos"]
if not np.all(np.diff(pos) > 0):
    raise ValueError("shared-site positions are not strictly ascending")

np.savez_compressed(out_npz, **sites)
print(json.dumps({
    "n_sites_shared": int(pos.size),
    "n_ingroup_haplotypes": panel.n_ingroup,
    "outgroups_in_orientation_panel": panel.outgroup_names,
    "sequence_length": panel.sequence_length,
    "rf_max": panel.rf_max,
    "mean_derived_allele_count": float(sites["daf"].mean()),
}, indent=1), flush=True)
