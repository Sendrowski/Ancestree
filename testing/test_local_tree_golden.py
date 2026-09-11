"""Reference values for the local-tree HMM, which nothing else pins.

The rest of the local-tree suite asserts relations -- segment-reset
consistency, chunked-versus-whole agreement, plug-in-versus-cached equality --
all of which hold under a change that shifts every TMRCA by a common factor or
redraws the ensemble while keeping it self-consistent. These pin the numbers
themselves, so a change to the emission, the time grid, the forward-backward
recursion or the FFBS draw has to be deliberate.

Regenerate deliberately, never to make a red test green: the stored matrix is
the output of this exact construction at the commit that introduced it.
"""
from pathlib import Path

import numpy as np

import ancestree as anc

FIXTURE = Path(__file__).parent / "fixtures" / "local_tree" / "hmm_block_tmrcas.npy"

N_HAP, N_BLOCKS, BLOCK_SIZE, N_SITES = 4, 20, 2000, 240
SEED = 20260810


def _panel():
    """The fixed panel the stored matrix was computed from."""
    rng = np.random.default_rng(SEED)
    pos = np.sort(rng.choice(N_BLOCKS * BLOCK_SIZE, size=N_SITES, replace=False))
    block_of_site = (pos // BLOCK_SIZE).astype(np.int64)
    genotypes = (rng.random((N_SITES, N_HAP)) < 0.25).astype(np.int8)
    return genotypes, block_of_site


def _hmm():
    return anc.PairwiseCoalescentHMM(
        n_haplotypes=N_HAP, mu=1.25e-8, rec_rate=1e-8,
        block_size=BLOCK_SIZE, n_time_bins=32)


def test_block_tmrcas_match_the_reference():
    g, bos = _panel()
    got = _hmm().block_tmrcas(g, bos, N_BLOCKS)
    want = np.load(FIXTURE)
    assert got.shape == want.shape
    np.testing.assert_allclose(
        got, want, rtol=1e-9, atol=1e-6,
        err_msg="local-tree HMM posterior-mean TMRCAs moved; regenerate the "
                "fixture only if the change was intended")


def test_the_reference_is_physically_sensible():
    """Guards the fixture itself: a degenerate matrix would pin nothing."""
    want = np.load(FIXTURE)
    assert np.all(np.isfinite(want))
    assert np.all(want > 0.0), "a zero TMRCA is the absorbing-underflow failure"
    # Not collapsed onto a single value, which a star tree would give.
    assert want.std() > 0.0
    assert want.max() / want.min() > 1.5


def test_block_tmrcas_are_deterministic():
    g, bos = _panel()
    first = _hmm().block_tmrcas(g, bos, N_BLOCKS)
    second = _hmm().block_tmrcas(g, bos, N_BLOCKS)
    np.testing.assert_array_equal(first, second)


NODE_TIMES = Path(__file__).parent / "fixtures" / "local_tree" / "upgma_node_times.npy"


def _builder():
    """The same fixed panel, as Site records for the full build path."""
    rng = np.random.default_rng(SEED)
    pos = np.sort(rng.choice(N_BLOCKS * BLOCK_SIZE, size=N_SITES, replace=False))
    g = rng.random((N_SITES, N_HAP)) < 0.25
    names = [f"h{i}" for i in range(N_HAP)]
    sites = [
        anc.Site(chrom="1", pos=int(p), alleles=("A", "C"),
                tip_alleles={n: ("A", "C")[int(b)] for n, b in zip(names, row)})
        for p, row in zip(pos, g)
    ]
    return anc.LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(N_BLOCKS * BLOCK_SIZE), block_size=BLOCK_SIZE,
        window="8snp")


def test_upgma_node_times_match_the_reference():
    """Pins the stretch the TMRCA golden leaves free.

    block_tmrcas -> per-window condensation -> UPGMA linkage -> node times
    is pinned to values, since no relational assertion constrains the node
    time scale.
    """
    ts = _builder().to_tree_sequence()
    got = np.asarray(ts.tables.nodes.time)
    want = np.load(NODE_TIMES)
    assert got.shape == want.shape, f"{got.shape} against {want.shape}"
    np.testing.assert_allclose(
        got, want, rtol=1e-9, atol=1e-6,
        err_msg="UPGMA node times moved; regenerate the fixture only if the "
                "change was intended")


def test_the_node_time_reference_is_sensible():
    want = np.load(NODE_TIMES)
    assert np.all(np.isfinite(want))
    assert (want == 0.0).sum() == N_HAP, "sample nodes must sit at time 0"
    assert want.max() > 0.0


# --------------------------------------------------- independent derivation
def test_node_times_come_from_the_linkage_heights():
    """UPGMA node times must equal the linkage matrix heights, scaled by mu.

    Derived from the linkage rather than from a stored array: the fixture in
    this file is the builder's own output, so it pins that the numbers do not
    move but not that they are the right ones. scipy's linkage column 2 is the
    merge height in generations, and the builder's only job is to place a node
    at each of them.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    import ancestree._upgma as upgma

    rng = np.random.default_rng(4)
    n = 6
    # A random ultrametric-ish distance matrix. The assertion does not depend
    # on it being ultrametric, only on average linkage being applied to it.
    d = rng.uniform(1e3, 1e5, size=(n, n))
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")

    ts = upgma.build_windowed_tree_sequence(
        n_samples=n, sequence_length=1000.0,
        window_intervals=[(0.0, 1000.0)], window_linkages=[z],
    )
    times = np.asarray(ts.tables.nodes.time)
    assert (times == 0.0).sum() == n, "sample nodes must sit at time 0"
    internal = np.sort(times[times > 0.0])
    want = np.sort(z[:, 2])
    assert internal.shape == want.shape, (internal.shape, want.shape)
    # The builder may bump a tie to keep a parent strictly above its children,
    # so equality is asserted up to that bump rather than exactly.
    np.testing.assert_allclose(internal, want, rtol=1e-9, atol=1e-6)
