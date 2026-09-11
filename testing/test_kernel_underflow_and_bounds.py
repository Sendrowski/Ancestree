"""Two numba kernels that failed silently: a cross-member write and an
absorbing zero.

Neither raised. The FFBS draw and the window reduction that follows it both
run ``parallel=True`` over members, so an index one past a member's row lands
in the next member's window rather than out of bounds. An odd block count is
where that happens: the final window holds a single block, at the index just
past the ``K // 2`` full ones. ``_pair_posterior_mean_impl`` mapped a zero
forward normaliser to ``inv = 0``, zeroing the row, which forces ``S = 0`` at
every later block: one underflowing block collapsed the posterior mean of the
entire segment to zero, and UPGMA then merged that pair first in every local
tree of the segment.
"""
import numpy as np

from ancestree import _ensemble, _smc_kernel


def test_odd_block_count_keeps_members_independent():
    """Member b's window must not receive member b-1's contribution."""
    P, K, T, M = 1, 7, 3, 3
    rng = np.random.default_rng(0)
    alpha = rng.random((P, K, T)) + 0.1
    alpha /= alpha.sum(axis=2, keepdims=True)
    # r must vary along the block axis: constant r makes an index shift on
    # that axis a no-op, which is how an r[k+1] -> r[k] off-by-one
    # is caught here.
    r = np.linspace(0.2, 0.8, K)[:, None] * np.ones((1, T))
    t_rep = np.array([100.0, 100.0, 10000.0])
    needed = np.arange((K + 1) // 2, dtype=np.int64)
    out = np.zeros((M, needed.shape[0], P))
    pi = np.full((P, T), 1.0 / T)

    # stride=1 makes every block its own checkpoint, so alpha is read as the
    # forward matrix and nothing is replayed: counts, lam and esc go unused.
    counts = np.zeros((P, K), dtype=np.int64)
    lam = np.ones((K, T))
    path = np.empty((M, P, K), dtype=np.int8)
    _ensemble.ffbs_paths(counts, lam, np.log(lam), alpha, 1, K, r, pi,
                         np.ones((P, K)), 1, np.arange(P, dtype=np.uint64),
                         path)
    blk_lo = 2 * needed
    blk_hi = np.minimum(blk_lo + 2, K)
    # Zero-width bins take the draw out of it, so a window mean is exactly a
    # mean of bin values and the leak below is the only thing under test.
    _ensemble.paths_to_condensed(
        path, t_rep, t_rep, np.ones(P), needed, blk_lo, blk_hi, 0,
        np.arange(P, dtype=np.uint64), 0, 0, out)

    # A full window averages exactly two block draws, each drawn from t_rep,
    # so its value must be one of the pairwise means. A third contribution
    # leaking in from the neighbouring member lands outside that set.
    allowed = {0.5 * (a + b) for a in t_rep for b in t_rep}
    full = out[:, :K // 2, :].ravel()
    bad = [v for v in full
           if not any(abs(v - w) < 1e-9 for w in allowed)]
    assert not bad, (
        f"window means {bad[:4]} are not means of two draws from {t_rep}; "
        f"a neighbouring member's seed leaked in (allowed: {sorted(allowed)})")
    assert np.all(np.isfinite(out))


def test_a_single_underflowing_block_does_not_zero_the_segment():
    """One block with no emission support must not collapse the whole track."""
    n, T = 12, 16
    lam = np.geomspace(1e-4, 1e-1, T)
    log_lam = np.log(lam)
    pi = np.full(T, 1.0 / T)
    r = np.full(T, 0.9)
    step = np.ones(n - 1)
    escale = np.ones(n)
    counts = np.zeros(n, dtype=np.int64)
    # A single block whose count is far outside what any time bin explains.
    counts[n // 2] = 100_000
    mask = np.ones(n, dtype=np.bool_)

    t_rep = 1.0 / lam
    pmean = _smc_kernel._pair_posterior_mean_impl(
        counts, t_rep, lam, log_lam, r, pi, step, mask, escale)

    assert np.all(np.isfinite(pmean))
    # Blocks after the offending one must retain a positive TMRCA estimate.
    tail = pmean[n // 2 + 1:]
    assert np.all(tail > 0.0), (
        f"posterior means collapsed to zero after the underflowing block: "
        f"{pmean}")
