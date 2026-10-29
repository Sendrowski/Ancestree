"""Regenerate the msprime inputs committed in this directory.

msprime reproduces a seed only within one platform: at 1.4.2 the same seed
gives arm64 macOS and x86-64 Linux different mutation positions and
genotypes. The tests that pin values to these draws therefore read the
committed files, and the pinned values are those of the arm64 macOS draw.
Regenerating on another platform changes the inputs, so the pinned values in
``testing/test_local_tree_geometry.py`` would have to be regenerated with
them.

- ``geometry_rows.npz``: the biallelic A/C/G/T sites of a diploid panel of 4
  individuals over 80 kb, read by ``testing/test_local_tree_geometry.py``.
  ``position`` holds the simulated integer positions, ``alleles`` the
  ``(n_sites, 2)`` alleles, ancestral first, and ``genotypes`` the
  ``(n_sites, 8)`` haplotype genotypes.
- ``continuous_genome.trees``: a continuous-genome JC69 tree sequence of 5
  diploids over 10 kb, read by ``testing/test_posterior.py`` and
  ``testing/test_writers.py``.

Run from the repository root::

    python testing/fixtures/simulated/generate.py
"""
from pathlib import Path

import msprime
import numpy as np

OUT = Path(__file__).resolve().parent


def geometry_rows():
    """Write ``geometry_rows.npz``."""
    ts = msprime.sim_ancestry(
        4, ploidy=2, sequence_length=80_000, recombination_rate=1e-8,
        population_size=1e4, random_seed=11)
    ts = msprime.sim_mutations(ts, rate=3e-8, random_seed=11)
    position, alleles, genotypes = [], [], []
    for variant in ts.variants():
        if (len(variant.alleles) != 2
                or any(a not in "ACGT" for a in variant.alleles)):
            continue
        position.append(int(variant.site.position))
        alleles.append(variant.alleles)
        genotypes.append(variant.genotypes)
    np.savez_compressed(
        OUT / "geometry_rows.npz", position=np.asarray(position, np.int64),
        alleles=np.asarray(alleles, dtype="<U1"),
        genotypes=np.asarray(genotypes, np.int8))


def continuous_genome():
    """Write ``continuous_genome.trees``."""
    ts = msprime.sim_ancestry(5, sequence_length=10_000,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=3, discrete_genome=False)
    ts = msprime.sim_mutations(ts, rate=5e-8, random_seed=4,
                               model=msprime.JC69(), discrete_genome=False)
    ts.dump(OUT / "continuous_genome.trees")


if __name__ == "__main__":
    geometry_rows()
    continuous_genome()
