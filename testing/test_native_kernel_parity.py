"""Parity of the native single-root ARG kernel with the generic path.

``ARGBasedInference._infer_range`` routes single-root local trees through
:meth:`~ancestree.likelihood.Likelihood.log_likelihoods_tskit_native` (int8
tip-state packing) instead of building a
:class:`~ancestree.trees.TskitLocalTree` and calling the dict-based
:meth:`~ancestree.likelihood.Likelihood.log_likelihoods`. Both dispatch to
the same ``felsenstein_postorder_dense`` kernel, so with numba active the two
paths are bit-identical. These tests pin that, and the ``id_map_scratch``
buffer reuse of the native path.
"""
import numpy as np
import pytest

from ancestree import _jit_kernel
from ancestree import HKY, STATE_INDEX, STATES
from ancestree.inference import ARGBasedInference
from ancestree.likelihood import Likelihood
from ancestree.sites import BaseComposition
from ancestree.trees import TskitLocalTree
from ancestree import models as M

import msprime


def _build_ts(seed=42):
    ts = msprime.sim_ancestry(
        samples=40, sequence_length=1_000_000, recombination_rate=1e-8,
        population_size=1e4, random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=2e-8, random_seed=seed)


def _models():
    return [
        ("JC69", M.JC69()),
        ("HKY", M.HKY(kappa=3.0)),
        ("GTR", M.GTR(rates=np.array([1.0, 2.0, 0.5, 0.8, 1.5, 1.0]))),
    ]


@pytest.mark.parametrize("name,model", _models(), ids=[m[0] for m in _models()])
def test_native_matches_generic_per_tree(name, model):
    """Native kernel output equals the generic ``_compute_posteriors`` path."""
    ts = _build_ts()
    inf = ARGBasedInference(ts, model, mu=2e-8, chrom="1")
    engine = Likelihood(
        inf.model,
        base_composition=inf.base_composition,
    )
    sample_nodes = ts.samples()
    node_times = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    id_map_scratch = np.full(int(ts.num_nodes), -1, dtype=np.int32)
    state_index = {s: i for i, s in enumerate(inf.model.states)}
    bound_mask = np.array(
        [int(n) in inf._node_to_sample for n in sample_nodes], dtype=bool,
    )
    by_site = {v.site.id: v for v in ts.variants()}

    n_single = 0
    for tree in ts.trees():
        if not tree.num_sites or tree.num_roots != 1:
            continue
        vlist = [by_site[s.id] for s in tree.sites()]
        sites = [inf._parse_variant(v, sample_nodes) for v in vlist]
        tree_mu = inf._mu_for_interval(
            float(tree.interval.left), float(tree.interval.right),
        )
        lt = TskitLocalTree.from_tskit_tree(tree, sample_map=inf.sample_map)
        lt.time_scale = tree_mu
        old = inf._compute_posteriors(engine, lt, sites)

        tip = inf._pack_tip_states(vlist, state_index, bound_mask)
        log_L = engine.log_likelihoods_tskit_native(
            tree, tip, sample_nodes, node_times, time_scale=tree_mu,
            id_map_scratch=id_map_scratch,
        )
        log_post = log_L + inf.prior.log_probs(sites)
        new = inf._normalise_log_post(log_post, n_states=inf.model.n_states)

        n_single += 1
        # Both paths call the same compiled kernel on the same dense arrays,
        # so this is bit-exact, not merely close.
        assert np.array_equal(old, new), (
            f"{name}: native path diverged from generic at tree "
            f"{tree.index}"
        )
    assert n_single > 0


def test_pack_tip_states_missing_and_unbound():
    """Missing genotypes, non-ACGT alleles, and unbound columns all map to -1."""
    ts = _build_ts()
    inf = ARGBasedInference(ts, M.JC69(), mu=2e-8, chrom="1")
    sample_nodes = ts.samples()
    state_index = {s: i for i, s in enumerate(inf.model.states)}
    # Drop one sample from the map so its column is "unbound".
    dropped = int(sample_nodes[0])
    inf._node_to_sample.pop(dropped, None)
    bound_mask = np.array(
        [int(n) in inf._node_to_sample for n in sample_nodes], dtype=bool,
    )
    variants = list(ts.variants())[:5]
    tip = inf._pack_tip_states(variants, state_index, bound_mask)
    assert tip.dtype == np.int8
    assert tip.shape == (5, sample_nodes.shape[0])
    # Unbound column is fully missing.
    assert np.all(tip[:, 0] == -1)
    # Every non-missing entry is a valid state index.
    assert np.all((tip == -1) | ((tip >= 0) & (tip < inf.model.n_states)))


def test_pack_tip_states_sentinel_branches():
    """Directly exercise the missing-genotype, non-ACGT, and empty-allele branches.

    The msprime fixtures carry no missing data and only A/C/G/T alleles, so the
    ``g < 0`` sentinel gather and the ``state_index.get(a, -1)`` non-ACGT/empty
    branches never fire there. Drive them with a hand-built variant.
    """
    class _FakeVariant:
        def __init__(self, alleles, genotypes):
            self.alleles = alleles
            self.genotypes = genotypes

    inf = ARGBasedInference.__new__(ARGBasedInference)  # no I/O; only method under test
    state_index = {"A": 0, "C": 1, "G": 2, "T": 3}
    # 4 sample columns, all bound. Alleles: A (valid), N (non-ACGT), "" (empty).
    v = _FakeVariant(alleles=("A", "N", ""), genotypes=np.array([0, 1, 2, -1]))
    bound_mask = np.ones(4, dtype=bool)
    tip = inf._pack_tip_states([v], state_index, bound_mask)
    # col0 -> "A" (0), col1 -> "N" missing, col2 -> "" missing, col3 -> g<0 missing.
    np.testing.assert_array_equal(tip, np.array([[0, -1, -1, -1]], dtype=np.int8))


class TestImpossibleRowStaysFinite:
    """A row the data forbid for every state must come back zero, not NaN.

    The traversal rescales each row by its max. When a row underflows to zero
    it redoes that row in log space, where ``-inf`` is both a value (a state
    the data forbid) and the sentinel meaning every state is forbidden. That
    makes the kernel depend on IEEE infinity semantics, which ``fastmath``
    explicitly licenses the compiler to discard: with the flag set LLVM folds
    the ``row_max_log == -inf`` test away, the all-forbidden row falls through
    to ``exp(-inf - -inf)``, and NaN reaches the posterior.

    The flag is applied per compiled function and re-applied to inlined
    callees, so a caller carrying it reinstates the bug in the inlined body.
    Both entry points are therefore exercised.
    """

    #: Two tips carrying different states down zero-length branches: no state
    #: at the root can produce both, so every state is forbidden.
    @staticmethod
    def _contradiction():
        n_nodes, S = 3, 4
        P = np.stack([np.eye(S)] * n_nodes)
        partials = np.zeros((n_nodes, 1, S))
        partials[0] = 1.0
        partials[1, 0, 0] = 1.0
        partials[2, 0, 1] = 1.0
        return (1, S, 0, np.array([0], np.int32),
                np.array([0, 2, 2, 2], np.int32), np.array([1, 2], np.int32),
                P, partials)

    def test_dense_entry_point(self):
        B, S, root, order, offsets, flat, P, partials = self._contradiction()
        rp, scale = _jit_kernel.felsenstein_postorder_dense(
            B, S, root, order, offsets, flat, P, partials)
        assert not np.isnan(rp).any(), f"NaN in root partial: {rp}"
        assert not np.isnan(scale).any(), f"NaN in scale: {scale}"
        assert np.array_equal(rp, np.zeros((B, S)))

    def test_core_entry_point(self):
        B, S, root, order, offsets, flat, P, partials = self._contradiction()
        rp = _jit_kernel.postorder_core(
            B, S, root, order, offsets, flat, P,
            np.arange(P.shape[0], dtype=np.int32), 0, partials,
            np.zeros(B), np.empty((B, S)), np.empty(S))
        assert not np.isnan(rp).any(), f"NaN in root partial: {rp}"
        assert np.array_equal(rp, np.zeros((B, S)))

    def test_a_possible_row_is_untouched(self):
        """The guard must not fire on data that are merely improbable."""
        B, S, root, order, offsets, flat, P, partials = self._contradiction()
        partials[2, 0, 1] = 0.0
        partials[2, 0, 0] = 1.0  # both tips agree on state 0
        rp, scale = _jit_kernel.felsenstein_postorder_dense(
            B, S, root, order, offsets, flat, P, partials)
        assert np.array_equal(rp, np.array([[1.0, 0.0, 0.0, 0.0]]))
        assert scale[0] == 0.0


def test_focal_node_absent_from_the_tree_is_rejected():
    """A focal node this tree does not contain must fail, not read elsewhere.

    ``id_map`` is pre-filled with -1, so an unknown focal node maps to
    ``root_dense == -1``, which would index the last dense node and report a
    confident posterior at an unrelated node.
    """
    from ancestree.focal import ResolvedFocal

    ts = _build_ts()
    inf = ARGBasedInference(ts, M.JC69(), mu=2e-8, chrom="1")
    tree = ts.first()
    sample_nodes = ts.samples()
    node_times = ts.tables.nodes.time
    state_index = {s: i for i, s in enumerate(inf.model.states)}
    bound_mask = np.array([True] * len(sample_nodes))
    variants = [next(ts.variants())]
    tip = inf._pack_tip_states(variants, state_index, bound_mask)
    engine = Likelihood(M.JC69())

    # A node of the tree SEQUENCE that this local tree does not contain, the
    # reachable case, since ARG mode re-resolves the focal node per tree and
    # most nodes belong to other trees (1135 of 1294 here). An out-of-range id
    # is caught by tskit instead. This one reaches the kernel.
    in_tree = {int(n) for n in tree.postorder()}
    absent = next(n for n in range(int(ts.num_nodes)) if n not in in_tree)
    with pytest.raises(ValueError, match="not a node of this tree"):
        engine.log_likelihoods_tskit_native(
            tree, tip, sample_nodes, node_times, time_scale=2e-8,
            focal=ResolvedFocal(node=absent, tau=0.0),
        )


def _pack_all_tip_states(ts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack the dense buffers the native path consumes from a haploid-ordered ARG.

    :return: ``(tip_states (B, n_samples) int8, sample_nodes int32, node_times f64)``
        where ``B`` is the number of ACGT-only sites in ``ts``.
    """
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    node_times = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    rows = []
    for variant in ts.variants():
        alleles = tuple(variant.alleles)
        if any(a not in STATES for a in alleles):
            continue
        row = np.full(sample_nodes.shape[0], -1, dtype=np.int8)
        for col in range(sample_nodes.shape[0]):
            idx = STATE_INDEX.get(alleles[int(variant.genotypes[col])])
            if idx is not None:
                row[col] = idx
        rows.append(row)
    tip_states = (
        np.stack(rows).astype(np.int8)
        if rows
        else np.empty((0, sample_nodes.shape[0]), dtype=np.int8)
    )
    return tip_states, sample_nodes, node_times


@pytest.fixture(scope="module")
def native_inputs(small_ts):
    """Model, packed dense buffers, and the list of single-root local trees."""
    bc = BaseComposition.from_n_target_sites(1000)
    bc.counts.update({"A": 300, "C": 200, "G": 200, "T": 300})
    lik = Likelihood(HKY(kappa=2.5), base_composition=bc)
    tip_states, sample_nodes, node_times = _pack_all_tip_states(small_ts)
    trees = [t.copy() for t in small_ts.trees() if t.num_roots == 1]
    assert len(trees) >= 2, "fixture must yield >=2 single-root local trees"
    n_nodes = int(small_ts.first().parent_array.shape[0])
    return lik, tip_states, sample_nodes, node_times, trees, n_nodes


def test_scratch_matches_fresh_alloc_bitwise(native_inputs):
    """Shared-scratch results are bit-identical to the fresh-alloc path,
    and the scratch is restored to all ``-1`` after every call."""
    lik, tip_states, sample_nodes, node_times, trees, n_nodes = native_inputs
    scratch = np.full(n_nodes, -1, dtype=np.int32)

    for tree in trees:
        expected = lik.log_likelihoods_tskit_native(
            tree, tip_states, sample_nodes, node_times,
            time_scale=1e-8, id_map_scratch=None,
        )
        got = lik.log_likelihoods_tskit_native(
            tree, tip_states, sample_nodes, node_times,
            time_scale=1e-8, id_map_scratch=scratch,
        )
        np.testing.assert_array_equal(got, expected)
        assert np.all(scratch == -1), "scratch not fully reset to -1 after call"


def test_no_leakage_between_trees_on_shared_buffer(native_inputs):
    """treeA-then-treeB on one shared buffer gives the same treeB result as
    a treeB call on a pristine buffer (no cross-tree id_map leakage)."""
    lik, tip_states, sample_nodes, node_times, trees, n_nodes = native_inputs
    tree_a, tree_b = trees[0], trees[1]

    fresh = np.full(n_nodes, -1, dtype=np.int32)
    ref_b = lik.log_likelihoods_tskit_native(
        tree_b, tip_states, sample_nodes, node_times,
        time_scale=1e-8, id_map_scratch=fresh,
    )

    shared = np.full(n_nodes, -1, dtype=np.int32)
    lik.log_likelihoods_tskit_native(
        tree_a, tip_states, sample_nodes, node_times,
        time_scale=1e-8, id_map_scratch=shared,
    )
    after_a = lik.log_likelihoods_tskit_native(
        tree_b, tip_states, sample_nodes, node_times,
        time_scale=1e-8, id_map_scratch=shared,
    )
    np.testing.assert_array_equal(after_a, ref_b)
    assert np.all(shared == -1)


def _recombining_ts(seed: int = 1):
    """An ARG with many local trees, so consecutive trees differ by an edge swap."""
    ts = msprime.sim_ancestry(
        samples=10, sequence_length=5000, recombination_rate=1e-5,
        population_size=1e4, random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=1e-5, random_seed=seed)


def _chunk_bounds(num_trees: int, n_chunks: int) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` tree-index ranges covering ``[0, num_trees)``."""
    edges = np.linspace(0, num_trees, n_chunks + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if a < b]


def _walk_chunks(ts, bounds):
    """Reach every local tree by seeking to each chunk's start and stepping on.

    :return: ``{tree_index: postorder}`` over the single-root trees.
    """
    seen = {}
    for start, stop in bounds:
        tree = ts.at_index(start)
        while tree.index != -1 and tree.index < stop:
            if tree.num_roots == 1:
                seen[tree.index] = np.asarray(tree.postorder(), dtype=np.int64)
            tree.next()
    return seen


def _score_chunks(engine, ts, bounds, tip_states, sample_nodes, node_times):
    """Score every single-root local tree, reaching it via ``bounds``.

    :return: ``{tree_index: (B, n_states) log-likelihood}``.
    """
    scratch = np.full(int(ts.num_nodes), -1, dtype=np.int32)
    out = {}
    for start, stop in bounds:
        tree = ts.at_index(start)
        while tree.index != -1 and tree.index < stop:
            if tree.num_roots == 1:
                out[tree.index] = engine.log_likelihoods_tskit_native(
                    tree, tip_states, sample_nodes, node_times,
                    time_scale=1e-8, id_map_scratch=scratch,
                )
            tree.next()
    return out


def test_native_scoring_is_independent_of_the_tree_seeking_path():
    """One local tree, reached by two different walks, must score bit-identically.

    tskit's ``left_child`` / ``right_sib`` linkage records edge-insertion
    history, so :meth:`tskit.Tree.postorder` lists a node's children, and the
    internal nodes themselves, in an order that depends on where the walk was
    seeded. Packing the kernel's CSR and traversal order straight from that
    listing reassociates both the product over child messages and the running
    sum of per-node log-scales, so ARG-mode posteriors moved by a few ULP with
    the number of fork-pool workers, which sets how the tree range is
    partitioned.
    """
    ts = _recombining_ts()
    one = _chunk_bounds(int(ts.num_trees), 1)
    many = _chunk_bounds(int(ts.num_trees), 7)

    # The property is only under test where the two walks really do list the
    # tree differently.
    po_one = _walk_chunks(ts, one)
    po_many = _walk_chunks(ts, many)
    assert po_one.keys() == po_many.keys()
    n_reordered = sum(
        1 for i in po_one if not np.array_equal(po_one[i], po_many[i])
    )
    assert n_reordered > 0, (
        "fixture no longer exercises the property: both walks list every "
        "local tree in the same order"
    )

    # Eight sites, scored against every local tree.
    tip_states, sample_nodes, node_times = _pack_all_tip_states(ts)
    tip_states = tip_states[:8]

    engine = Likelihood(M.JC69())
    scored_one = _score_chunks(
        engine, ts, one, tip_states, sample_nodes, node_times)
    scored_many = _score_chunks(
        engine, ts, many, tip_states, sample_nodes, node_times)

    assert scored_one.keys() == scored_many.keys()
    for index in scored_one:
        assert np.array_equal(scored_one[index], scored_many[index]), (
            f"local tree {index} scored differently depending on the walk that "
            f"reached it"
        )
