"""Simulate the species alignment the sequence-alignments docs page reads.

Four species, one haploid sequence each, under the species tree
``(((W,X),Y),Z)``. The branch from the ``(W,X)`` split to the ``(W,X,Y)`` split
is short in coalescent units, so incomplete lineage sorting makes the local
gene tree of ``W``, ``X`` and ``Y`` discordant with the species tree along a
large share of the sequence. Mutations follow JC69.

Writes the alignment as one FASTA record per species and the tree sequence
holding the true mutations, with the species names in its individual metadata.
Rule ``docs_alignment`` runs this script, which also runs directly::

    python workflow/scripts/simulate_docs_alignment.py
"""
import gzip
from pathlib import Path

import msprime
import numpy as np
import tskit

try:
    OUT_FASTA = Path(snakemake.output.fasta)  # type: ignore[name-defined]
    OUT_TREES = Path(snakemake.output.trees)  # type: ignore[name-defined]
except NameError:
    STATIC = Path(__file__).resolve().parents[2] / "docs" / "_static"
    OUT_FASTA, OUT_TREES = STATIC / "alignment.fa.gz", STATIC / "alignment.trees"

#: Effective size of every population.
NE = 10_000
#: Split times in generations: (W,X), then Y, then the outgroup Z. The
#: 0.2 coalescent units between the first two leave about half of the gene
#: trees of W, X and Y discordant with the species tree.
T_WX, T_WXY, T_ROOT = 20_000, 24_000, 100_000
LENGTH = 100_000
MU = 2.5e-7
REC_RATE = 2.5e-8
SEED = 7
SPECIES = ["W", "X", "Y", "Z"]


def main() -> None:
    demography = msprime.Demography()
    for name in [*SPECIES, "WX", "WXY", "ROOT"]:
        demography.add_population(name=name, initial_size=NE)
    demography.add_population_split(time=T_WX, derived=["W", "X"], ancestral="WX")
    demography.add_population_split(time=T_WXY, derived=["WX", "Y"], ancestral="WXY")
    demography.add_population_split(time=T_ROOT, derived=["WXY", "Z"], ancestral="ROOT")
    ts = msprime.sim_ancestry(
        samples={name: 1 for name in SPECIES}, demography=demography,
        ploidy=1, sequence_length=LENGTH, recombination_rate=REC_RATE,
        random_seed=SEED)
    ts = msprime.sim_mutations(ts, rate=MU, model=msprime.JC69(),
                               random_seed=SEED)

    # One sample per population, so each individual is named after its species.
    tables = ts.dump_tables()
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    rows = list(tables.individuals)
    tables.individuals.clear()
    for ind, row in zip(ts.individuals(), rows):
        population = ts.node(ind.nodes[0]).population
        tables.individuals.append(row.replace(
            metadata={"name": ts.population(population).metadata["name"]}))
    ts = tables.tree_sequence()

    # The monomorphic columns take a random root sequence.
    rng = np.random.default_rng(SEED)
    root = "".join(rng.choice(list("ACGT"), LENGTH))
    names = [ts.individual(ts.node(u).individual).metadata["name"]
             for u in ts.samples()]
    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_FASTA, "wt") as f:
        for name, sequence in zip(names, ts.alignments(reference_sequence=root)):
            f.write(f">{name}\n")
            for i in range(0, LENGTH, 60):
                f.write(sequence[i:i + 60] + "\n")
    ts.dump(OUT_TREES)
    print(f"Wrote {OUT_FASTA} and {OUT_TREES}: {ts.num_trees} trees, "
          f"{ts.num_sites} sites")


if __name__ == "__main__":
    main()
