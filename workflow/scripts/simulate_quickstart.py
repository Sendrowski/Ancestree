"""Simulate the small tree sequence used by the quickstart notebook.

Two ingroup subclades (``A``, ``B`` — 3 samples each, named ``i0..i5``)
plus 2 outgroups (``O`` — ``o0..o1``), under a 3-population demography
with low recombination + low mutation. The result is a multi-tree,
multi-site ARG suitable for end-to-end recovery-rate comparisons.

Under the seed-42 sim the leftmost site sits at position 556, with a
single mutation on the edge above subclade ``B``'s MRCA — the kind of
clean one-subclade-derived pattern the quickstart's section-1
walkthrough uses to demonstrate polarisation::

    i0..i2 → G   (ancestral)
    i3..i5 → C   (derived)
    o0..o1 → G   (ancestral)

The output is a ``.trees`` file with sample names baked into individual
metadata, loaded back via :meth:`~ancestree.trees.TskitLocalTree.from_tskit_tree`.

Run directly::

    python workflow/scripts/simulate_quickstart.py
"""
from pathlib import Path

import msprime
import tskit


OUT = Path(__file__).resolve().parents[2] / "docs" / "_static" / "quickstart.trees"


def main() -> None:
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=10_000)
    demography.add_population(name="B", initial_size=10_000)
    demography.add_population(name="O", initial_size=10_000)
    demography.add_population(name="AB", initial_size=10_000)
    demography.add_population(name="ANC", initial_size=10_000)
    demography.add_population_split(time=2_000, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=20_000, derived=["AB", "O"], ancestral="ANC")

    # Seed 42 chosen so the first local tree has subclades A, B, and O each
    # monophyletic — required for the section-1 hand-polarisation illustration
    # to read cleanly. Other seeds break monophyly via early recombinations.
    ts = msprime.sim_ancestry(
        samples={"A": 3, "B": 3, "O": 2},
        demography=demography,
        ploidy=1,
        sequence_length=5e4,
        recombination_rate=1e-8,
        random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=5e-8, random_seed=42)

    # Group sample node ids by population, then assign caller-facing names.
    pop_id = {p.metadata["name"]: p.id for p in ts.populations()}
    by_pop: dict[str, list[int]] = {"A": [], "B": [], "O": []}
    for s in ts.samples():
        name = next(n for n, pid in pop_id.items() if pid == ts.node(s).population)
        if name in by_pop:
            by_pop[name].append(int(s))
    sample_name: dict[int, str] = {}
    for i, s in enumerate(by_pop["A"]):
        sample_name[s] = f"i{i}"
    for i, s in enumerate(by_pop["B"]):
        sample_name[s] = f"i{3 + i}"
    for i, s in enumerate(by_pop["O"]):
        sample_name[s] = f"o{i}"

    tables = ts.dump_tables()

    # Bake names into individual metadata. Ploidy=1 ⇒ one node per individual.
    # IndividualTableRow has no .nodes, so resolve nodes via the ts-level iterator.
    ind_to_nodes = {ind.id: list(ind.nodes) for ind in ts.individuals()}
    rows = list(tables.individuals)
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.individuals.clear()
    for ind_id, ind in enumerate(rows):
        nodes = ind_to_nodes.get(ind_id, [])
        name = sample_name.get(int(nodes[0])) if nodes else None
        tables.individuals.add_row(
            flags=ind.flags,
            location=ind.location,
            parents=ind.parents,
            metadata={"name": name} if name else {},
        )

    ts_out = tables.tree_sequence()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    ts_out.dump(OUT)

    names = [sample_name[s] for s in ts_out.samples()]
    print(
        f"Wrote {OUT}: {ts_out.num_samples} samples, "
        f"{ts_out.num_trees} trees, {ts_out.num_sites} sites"
    )
    print(f"Sample names: {names}")
    print(
        f"First site: pos={ts_out.site(0).position:.1f} "
        f"ancestral={ts_out.site(0).ancestral_state}"
    )


if __name__ == "__main__":
    main()
