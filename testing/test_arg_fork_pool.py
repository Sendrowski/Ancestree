"""Fork-pool parallel-walk tests for :class:`~ancestree.inference.ARGBasedInference`.

The library exposes ``n_workers`` on :class:`ARGBasedInference`. With
``n_workers > 1`` and a ``fork``-capable platform, the local trees are
partitioned into roughly equal-site-count chunks and each chunk runs in
a forked child that inherits the parent's ARG via copy-on-write. The
emitted ``(Site, Posterior)`` stream must be:

1. **Bit-identical** to the single-worker walk. Dense slots are numbered by
   ascending node id and the post-order is built from the parent array, so
   the child messages are multiplied in one order whatever tree index the
   walk started from. Per-site posterior values are compared exactly.
2. **In tskit-natural variant order**, chunks are dispatched in
   tree-index order and the parent yields per-chunk results in chunk
   order, so the final stream is sorted by position.

Skipped on Windows / spawn-only platforms (the library falls back to
single-worker there. Nothing to compare).
"""
from __future__ import annotations

import logging
import multiprocessing as mp

import msprime
import numpy as np
import pytest

import ancestree.inference as inference_mod
from ancestree import ARGBasedInference, JC69
from ancestree.inference import _arg_infer_chunk_worker
from ancestree.settings import Settings


FORK_AVAILABLE = "fork" in mp.get_all_start_methods()

pytestmark = pytest.mark.skipif(
    not FORK_AVAILABLE,
    reason="fork start method unavailable on this platform (Windows)",
)


@pytest.fixture(scope="module")
def fork_pool_ts():
    """msprime ARG sized to populate multiple local trees with sites.

    ``recombination_rate=1e-5`` over ``length=5_000`` yields ~3.4k local
    trees, ``mu=1e-5`` puts ~3.7k SNPs on them, enough that a 4-way
    fork pool actually gets 4 non-empty chunks.
    """
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=5_000,
        recombination_rate=1e-5,
        population_size=1e4,
        random_seed=1,
    )
    return msprime.sim_mutations(ts, rate=1e-5, random_seed=1)


def _run(ts, n_workers):
    inference = ARGBasedInference(
        ts,
        JC69(),
        mu=1e-8,
        progress=False,
        n_workers=n_workers,
    )
    return list(inference.infer())


def test_single_vs_fork_pool_identical(fork_pool_ts):
    """``n_workers=1`` and ``n_workers=4`` agree to floating-point tolerance.

    Not bit-exact. Both paths batch the sites of one local tree together, so
    the batches themselves are identical; what differs is the order in which
    a node's children are visited. ``_build_tree_chunks`` partitions the
    trees by worker count and ``_infer_range`` seeds its walk with
    ``ts.at_index(tree_start)``, and tskit's ``left_child``/``right_sib``
    linkage makes ``postorder()`` list a node's siblings in an order that
    depends on where the walk started. The kernel does not read that order:
    it numbers dense slots by ascending node id, so every walk multiplies the
    child messages in the same sequence and the results match bit for bit.
    """
    single = _run(fork_pool_ts, n_workers=1)
    forked = _run(fork_pool_ts, n_workers=4)
    assert len(single) == len(forked)
    for (s_site, s_post), (f_site, f_post) in zip(single, forked):
        assert int(s_site.pos) == int(f_site.pos)
        np.testing.assert_array_equal(s_post.values, f_post.values)


def test_fork_pool_preserves_order(fork_pool_ts):
    """Site positions are emitted in the same order as the single-worker walk."""
    single_positions = [int(s.pos) for s, _ in _run(fork_pool_ts, n_workers=1)]
    forked_positions = [int(s.pos) for s, _ in _run(fork_pool_ts, n_workers=4)]
    assert single_positions == forked_positions
    # Sanity: positions should be sorted ascending (tskit-natural variant order).
    assert single_positions == sorted(single_positions)


@pytest.fixture(scope="module")
def topology_only_arg_with_injected_sites():
    """A topology-only ARG with synthetic sites added afterwards.

    This is the shape of a gamma-SMC ARG, whose site and mutation tables are
    empty because the SNPs live in a separate VCF. The msprime topology has
    its site and mutation tables cleared, then sparse sites are added with a
    mutation on every other sample so the kernel has work to do per site.
    """
    ts = msprime.sim_ancestry(
        samples=20,
        sequence_length=20_000,
        recombination_rate=1e-5,
        population_size=1e4,
        random_seed=7,
    )
    # Strip any mutations (msprime.sim_ancestry adds none, but be explicit).
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    # Add ~200 sites at sparse positions. Place a mutation on every
    # other sample at each site so genotypes are non-trivial.
    rng = np.random.default_rng(0)
    sample_nodes = ts.samples()
    positions = np.sort(rng.uniform(0, ts.sequence_length, size=200))
    # Deduplicate to integer positions (tskit forbids two sites at the
    # same position).
    positions = np.unique(positions.astype(int))
    for pos in positions:
        sid = tables.sites.add_row(position=float(pos), ancestral_state="A")
        for i, n in enumerate(sample_nodes):
            if i % 2:
                tables.mutations.add_row(site=sid, node=int(n), derived_state="C")
    tables.sort()
    return tables.tree_sequence(), len(positions)


def test_fork_pool_topology_only_arg_with_injected_sites(
    topology_only_arg_with_injected_sites,
):
    """The fork pool emits every site of a topology-only ARG with added sites.

    The site count matches under both the single-worker and the 4-worker
    walk, and the two walks emit numerically equal posteriors site by site
    (to floating-point tolerance, since chunking can perturb the reduction
    order).
    """
    ts, n_expected = topology_only_arg_with_injected_sites
    assert ts.num_sites == n_expected, "fixture preconditions"

    single = _run(ts, n_workers=1)
    forked = _run(ts, n_workers=4)
    assert len(single) == n_expected
    assert len(forked) == n_expected
    for (s_site, s_post), (f_site, f_post) in zip(single, forked):
        assert int(s_site.pos) == int(f_site.pos)
        np.testing.assert_array_equal(s_post.values, f_post.values)


def test_build_tree_chunks_matches_legacy_per_tree_walk():
    """``_build_tree_chunks`` yields a valid, balanced, contiguous partition.

    The vectorised ``searchsorted`` split is compared against a per-tree
    greedy walk. It must lose no sites and stay balanced, but need not be
    identical to the greedy walk, since the two distribute the remainder
    differently.
    """
    from ancestree import ARGBasedInference, JC69

    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=5_000,
        recombination_rate=1e-5,
        population_size=1e4,
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-5, random_seed=1)

    inf = ARGBasedInference(
        source=ts, model=JC69(), mu=1e-8,
         progress=False, n_workers=4,
    )
    chunks = inf._build_tree_chunks(4)

    per_tree_sites = np.fromiter(
        (t.num_sites for t in ts.trees()),
        dtype=np.int64, count=ts.num_trees,
    )
    total_sites = int(per_tree_sites.sum())

    # Contiguous cover of [0, num_trees): no gaps, no overlaps, no empties.
    assert chunks[0][0] == 0
    assert chunks[-1][1] == ts.num_trees
    for (a, b), (c, _d) in zip(chunks, chunks[1:]):
        assert a < b and b == c

    # No sites lost, and each chunk stays near the per-chunk target. The
    # vectorised searchsorted split need not reproduce the legacy greedy
    # walk's exact per-chunk counts (it spreads the remainder more evenly),
    # so balance is asserted to a tolerance rather than as bit-identical counts.
    counts = [int(per_tree_sites[a:b].sum()) for a, b in chunks]
    assert sum(counts) == total_sites
    target = total_sites / 4
    assert max(abs(c - target) for c in counts) <= 0.05 * total_sites


def test_fork_pool_propagates_uniform_fallback_count(fork_pool_ts, caplog, monkeypatch):
    """Uniform-fallback counts incremented in forked children reach the parent.

    The parent sums them and emits the summary warning under
    ``n_workers > 1``.
    """
    import logging

    orig = ARGBasedInference._normalise_log_post

    def _bump(self, log_post, *, n_states):
        # Force a fallback count on every batch (patched before the pool forks,
        # so children inherit it via copy-on-write).
        self._n_uniform_fallback = getattr(self, "_n_uniform_fallback", 0) + 1
        return orig(self, log_post, n_states=n_states)

    monkeypatch.setattr(ARGBasedInference, "_normalise_log_post", _bump)
    inf = ARGBasedInference(
        fork_pool_ts, JC69(), mu=1e-8,
        progress=False, n_workers=4,
    )
    with caplog.at_level(logging.WARNING, logger="ancestree.ARGBasedInference"):
        list(inf.infer())
    assert any(
        "degenerate log-posterior" in r.message for r in caplog.records
    ), "fork-pool dropped the uniform-fallback summary warning"


#: Diagnostic counters every walk maintains, compared across execution paths.
COUNTER_NAMES = (
    "_n_uniform_fallback",
    "_n_ingroup_monomorphic",
    "_n_uncoalesced_segments",
    "_n_ingroup_non_monophyletic",
    "_n_focal_multiroot_fallback",
    "_n_unrepresentable_sites",
    "_n_unrepresentable_tips",
)


def _counters(ts, n_workers):
    """Every diagnostic counter after a walk, before the summary clears them."""
    inference = ARGBasedInference(
        ts, JC69(), mu=1e-8, progress=False, n_workers=n_workers,
    )
    # Quiet keeps the summary from clearing the counters it reports.
    inference._quiet = True
    list(inference.infer())
    return {name: int(getattr(inference, name)) for name in COUNTER_NAMES}


def test_fork_pool_reports_the_serial_counters(fork_pool_ts):
    """Every diagnostic counter survives the trip through the fork pool.

    The chunk worker reset and returned four of the five, so a fork-pool run
    reported no uncoalesced segments whatever the serial walk had counted.
    """
    serial = _counters(fork_pool_ts, 1)
    assert serial["_n_ingroup_monomorphic"] > 0, "fixture preconditions"
    assert _counters(fork_pool_ts, 4) == serial


@pytest.fixture(scope="module")
def indel_fork_pool_ts(fork_pool_ts):
    """``fork_pool_ts`` with every fifth mutation rewritten to an indel.

    The rewritten allele is outside A/C/G/T, so the sites carrying it are
    what the unrepresentable-allele counters must tally, spread over enough
    local trees to reach several fork-pool chunks.
    """
    import tskit

    tables = fork_pool_ts.dump_tables()
    derived = tskit.unpack_strings(tables.mutations.derived_state,
                                   tables.mutations.derived_state_offset)
    derived = ["AT" if i % 5 == 0 else d for i, d in enumerate(derived)]
    packed, offset = tskit.pack_strings(derived)
    tables.mutations.set_columns(
        site=tables.mutations.site, node=tables.mutations.node,
        time=tables.mutations.time, derived_state=packed,
        derived_state_offset=offset, parent=tables.mutations.parent)
    return tables.tree_sequence()


def test_fork_pool_reports_the_unrepresentable_allele_counts(indel_fork_pool_ts):
    """Sites and tips outside the A/C/G/T alphabet survive the fork pool.

    A counter the chunk worker neither resets nor returns is accumulated in
    the child and discarded when it exits, so a parallel run reported none of
    them however many the serial walk had counted.
    """
    serial = _counters(indel_fork_pool_ts, 1)
    assert serial["_n_unrepresentable_sites"] > 0, "fixture preconditions"
    assert serial["_n_unrepresentable_tips"] > 0, "fixture preconditions"
    assert _counters(indel_fork_pool_ts, 4) == serial


def test_fork_pool_clamps_chunks_to_num_trees():
    """More workers than local trees must not produce out-of-range windows.

    ``_build_tree_chunks`` clamps the chunk count to ``num_trees``.
    """
    # Non-recombining region → exactly one local tree, plus some sites.
    ts = msprime.sim_ancestry(
        samples=5, ploidy=1, sequence_length=1000,
        recombination_rate=0, population_size=1e4, random_seed=3,
    )
    ts = msprime.sim_mutations(ts, rate=1e-2, random_seed=3)
    assert ts.num_trees == 1 and ts.num_sites > 0

    inf = ARGBasedInference(ts, JC69(), mu=1e-2, progress=False, n_workers=4)
    # Chunk builder collapses to a single covering chunk.
    assert inf._build_tree_chunks(4) == [(0, 1)]
    # And infer() runs to completion (single-worker fallback) instead of crashing.
    out_parallel = {int(s.pos): p.map_allele for s, p in inf.infer()}

    serial = {int(s.pos): p.map_allele
              for s, p in ARGBasedInference(
                  ts, JC69(), mu=1e-2, progress=False, n_workers=1).infer()}
    assert out_parallel == serial


# ----------------------------------------------- fork-pool worker guard
def test_fork_worker_without_parent_inference_raises():
    # _ARG_INFERENCE_FOR_FORK is None outside an active parallel infer().
    with pytest.raises(RuntimeError, match="fork did not propagate"):
        _arg_infer_chunk_worker((0, 1))


def test_chunk_worker_refuses_to_run_without_a_parent(monkeypatch):
    monkeypatch.setattr(inference_mod, "_ARG_INFERENCE_FOR_FORK", None)
    with pytest.raises(RuntimeError, match="without an inference object"):
        _arg_infer_chunk_worker((0, 1))


def test_chunk_worker_reproduces_the_serial_walk(small_ts, monkeypatch):
    """The fork-pool worker, run in-process, returns the serial posteriors
    and integer diagnostic counts."""
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
    serial = [(s.pos, p.values) for s, p in inf.infer()]
    monkeypatch.setattr(inference_mod, "_ARG_INFERENCE_FOR_FORK", inf)
    results, *counts = _arg_infer_chunk_worker((0, small_ts.num_trees))
    assert [(s.pos, v.tolist()) for s, v in results] == \
        [(pos, v.tolist()) for pos, v in serial]
    assert len(counts) == 7 and all(isinstance(c, int) for c in counts)


def test_infer_range_over_an_empty_window_yields_nothing(small_ts):
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
    assert list(inf._infer_range(2, 2)) == []
    assert list(inf._infer_range(3, 1)) == []


# ----------------------------------------------------- serial fallbacks


def test_fork_unsafe_threading_layer_falls_back_to_serial(small_ts, monkeypatch, caplog):
    """When the numba threading layer is not fork-safe the request for
    workers is ignored, with a warning, and the serial walk runs."""
    monkeypatch.setattr(Settings, "_fork_is_safe", staticmethod(lambda: False))
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False, n_workers=2)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        results = list(inf.infer())
    assert len(results) == small_ts.num_sites
    assert any("Ignoring the parallelization request" in r.message
               for r in caplog.records)


def test_fork_pool_without_fork_runs_serially(small_ts, monkeypatch, caplog):
    monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False, n_workers=2)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        results = list(inf._infer_fork_pool())
    assert len(results) == small_ts.num_sites
    assert any("'fork' start method is unavailable" in r.message
               for r in caplog.records)


# ------------------------------------------------------ chunk geometry


@pytest.fixture(scope="module")
def few_trees_ts():
    """An ARG with five local trees, every one of them carrying sites."""
    ts = msprime.sim_ancestry(samples=4, sequence_length=2_000,
                              recombination_rate=4e-8, population_size=1e4,
                              random_seed=2)
    ts = msprime.sim_mutations(ts, rate=5e-7, random_seed=2)
    assert ts.num_trees == 5 and all(t.num_sites > 0 for t in ts.trees())
    return ts


def test_chunks_without_sites_cover_every_tree(few_trees_ts):
    empty = few_trees_ts.delete_sites(range(few_trees_ts.num_sites))
    inf = ARGBasedInference(empty, JC69(), mu=1e-8, progress=False)
    assert inf._build_tree_chunks(4) == [(0, empty.num_trees)]


def test_chunks_stay_strictly_increasing_when_sites_cluster(few_trees_ts):
    """Per-tree site counts of ``1, 0, 0, 10, 0`` put both boundaries of a
    three-way split on the fourth tree; the chunks are still contiguous,
    non-empty and cover every tree."""
    ts = few_trees_ts
    keep_per_tree = {0: 1, 3: 10}
    kept: list[int] = []
    for tree in ts.trees():
        kept += [s.id for s in tree.sites()][:keep_per_tree.get(tree.index, 0)]
    shaped = ts.delete_sites([s.id for s in ts.sites() if s.id not in kept])
    assert [t.num_sites for t in shaped.trees()] == [1, 0, 0, 10, 0]
    inf = ARGBasedInference(shaped, JC69(), mu=1e-8, progress=False)
    chunks = inf._build_tree_chunks(3)
    assert chunks == [(0, 3), (3, 4), (4, 5)]


def test_fork_pool_with_fewer_chunks_than_its_window(few_trees_ts, monkeypatch):
    """Fewer chunks than in-flight slots: the submission window closes early
    and the stream still matches the serial walk."""
    monkeypatch.setattr(ARGBasedInference, "FORK_CHUNKS_PER_WORKER", 1)
    inf = ARGBasedInference(few_trees_ts, JC69(), mu=1e-8, progress=False,
                            n_workers=2)
    assert 1 < len(inf._build_tree_chunks(2)) < 4
    serial = ARGBasedInference(few_trees_ts, JC69(), mu=1e-8, progress=False)
    got = [(int(s.pos), p.values) for s, p in inf.infer()]
    expected = [(int(s.pos), p.values) for s, p in serial.infer()]
    assert [pos for pos, _ in got] == [pos for pos, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        np.testing.assert_array_equal(a, b)
