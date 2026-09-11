"""Equivalence tests across the two Felsenstein paths in :mod:`~ancestree.likelihood`.

The kernel has two code paths that compute the same thing:

1. **Generic**, :meth:`Likelihood.log_likelihoods` packs any
   :class:`~ancestree.trees.Tree` into dense CSR arrays and dispatches to
   the ``@njit``-compiled
   :func:`~ancestree._jit_kernel.felsenstein_postorder_dense`.
2. **tskit-native**, :meth:`Likelihood.log_likelihoods_tskit_native`
   reads those dense arrays straight from tskit buffers (no
   :class:`~ancestree.trees.TskitLocalTree` shim) and calls the same
   compiled kernel.

Without these tests the two can silently drift apart on refactor. Each
test runs the same input through both and asserts numerical agreement.
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import GTR, HKY, JC69, K2, Likelihood, OutgroupLadderTree
from ancestree.sites import BaseComposition, Site


# ----------------------------------------------------------------- fixtures


def _make_ladder_tree(n_outgroups: int = 3) -> OutgroupLadderTree:
    """Outgroup-ladder tree with 3 ingroup haplotypes + ``n_outgroups`` outgroups."""
    ingroup = [f"ing_{i}" for i in range(3)]
    outgroup = [f"og_{i}" for i in range(n_outgroups)]
    tree = OutgroupLadderTree(ingroup, outgroup)
    # Set per-branch rates to something biologically reasonable so
    # transition probabilities are not near-degenerate (which would mask
    # cross-path numerical drift behind matrix saturation).
    tree.set_params(np.full(tree.n_params, 0.05))
    return tree


def _make_sites() -> list[Site]:
    """A small mixed-pattern site set: biallelic, multi-allelic, missing-tip."""
    return [
        # Biallelic, all tips called.
        Site(
            chrom="1", pos=100, alleles=("A", "G"),
            tip_alleles={
                "ing_0": "A", "ing_1": "A", "ing_2": "G",
                "og_0": "A", "og_1": "G", "og_2": "G",
            },
        ),
        # Multi-allelic (A/C/T) with one missing ingroup tip.
        Site(
            chrom="1", pos=200, alleles=("A", "C", "T"),
            tip_alleles={
                "ing_0": "A", "ing_1": "C", "ing_2": None,
                "og_0": "T", "og_1": "T", "og_2": "T",
            },
        ),
        # Biallelic with a tip outside the alphabet (treated as missing).
        Site(
            chrom="1", pos=300, alleles=("C", "T"),
            tip_alleles={
                "ing_0": "C", "ing_1": "T", "ing_2": "N",
                "og_0": "C", "og_1": "C", "og_2": "T",
            },
        ),
    ]


_BC_SKEWED = BaseComposition.from_n_target_sites(1000)
_BC_SKEWED.counts.update({"A": 300, "C": 200, "G": 200, "T": 300})


MODEL_CASES = [
    ("JC69", JC69(), None),
    ("K2(kappa=2)", K2(kappa=2.0), None),
    ("HKY(kappa=2, skewed pi)", HKY(kappa=2.0), _BC_SKEWED),
    ("GTR(asymmetric rates, skewed pi)",
     GTR(rates=[1.0, 4.0, 0.7, 0.5, 4.0, 1.3]), _BC_SKEWED),
]


# ----------------------------------------------- generic vs tskit-native fast path


@pytest.mark.parametrize(
    "label,model,base_composition",
    MODEL_CASES,
    ids=[c[0] for c in MODEL_CASES],
)
def test_generic_matches_tskit_native(label, model, base_composition):
    """``log_likelihoods`` and ``log_likelihoods_tskit_native`` must agree
    on a tskit-backed local tree.

    Builds a tiny msprime ARG, grabs the first local tree, evaluates the
    same site batch through both entry points, and asserts agreement.
    Tolerance: ``atol=1e-12`` (same kernel, only the packing differs).
    """
    import msprime

    from ancestree import STATE_INDEX, STATES, TskitLocalTree

    ts = msprime.sim_ancestry(
        samples=5, sequence_length=2_000, recombination_rate=0,
        random_seed=42, ploidy=1, population_size=1_000,
    )
    ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=42)
    assert not (ts.num_sites == 0), "seed produced no mutations on this small sim"

    tree = next(ts.trees())
    # Collect every variant whose site lives inside this local tree.
    sites = []
    for variant in ts.variants():
        if not (tree.interval.left <= variant.site.position < tree.interval.right):
            continue
        alleles = tuple(variant.alleles)
        # Skip rare non-ACGT mutations that slip through msprime.JC69.
        if any(a not in STATES for a in alleles):
            continue
        tip_alleles: dict[str, "str | None"] = {}
        for sample_idx, node_id in enumerate(ts.samples()):
            g = int(variant.genotypes[sample_idx])
            tip_alleles[f"tsk_{node_id}"] = alleles[g] if alleles[g] in STATES else None
        sites.append(Site(
            chrom="1", pos=int(variant.site.position),
            alleles=alleles, tip_alleles=tip_alleles,
        ))
    assert not (not sites), "no biallelic/SNP sites in the first local tree"

    sample_map = {f"tsk_{int(n)}": int(n) for n in ts.samples()}
    local = TskitLocalTree(ts, position=float(tree.interval.left), sample_map=sample_map)

    lik = Likelihood(model, base_composition=base_composition)
    log_L_generic = lik.log_likelihoods(local, sites)

    # Pack the dense int8 / int32 buffers the tskit-native fast path expects.
    B = len(sites)
    n_samples = len(sample_map)
    tip_states = np.full((B, n_samples), -1, dtype=np.int8)
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    sample_id_per_node = {int(n): f"tsk_{int(n)}" for n in sample_nodes}
    for b, site in enumerate(sites):
        for col, node_id in enumerate(sample_nodes):
            allele = site.tip_alleles.get(sample_id_per_node[int(node_id)])
            if allele is None:
                continue
            idx = STATE_INDEX.get(allele)
            if idx is not None:
                tip_states[b, col] = idx
    node_times = np.asarray(ts.tables.nodes.time, dtype=np.float64)

    log_L_native = lik.log_likelihoods_tskit_native(
        tree, tip_states, sample_nodes, node_times,
        time_scale=local.time_scale,
    )

    assert log_L_generic.shape == log_L_native.shape == (B, model.n_states)
    np.testing.assert_allclose(log_L_generic, log_L_native, atol=1e-12)


def test_native_path_rejects_multi_root_tree():
    """The tskit-native fast path assumes a single root, so a
    multi-root tree is refused rather than dropping the other roots' samples."""
    import msprime
    ts = msprime.sim_ancestry(
        samples=8, sequence_length=1e3, recombination_rate=1e-5,
        population_size=1e4, end_time=1.0, random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=1)
    tree = next((t for t in ts.trees() if t.num_roots > 1), None)
    assert not (tree is None), "could not induce a multi-root tree"
    lik = Likelihood(JC69())
    n = ts.num_samples
    tip_states = np.zeros((1, n), dtype=np.int8)
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    node_times = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    with pytest.raises(ValueError, match="single-root"):
        lik.log_likelihoods_tskit_native(
            tree, tip_states, sample_nodes, node_times, time_scale=1e-6,
        )
