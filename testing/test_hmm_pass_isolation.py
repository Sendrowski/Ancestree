"""The forward and backward passes in isolation.

``_pair_posterior_mean_impl`` is undecorated so it can be called directly,
which makes the recursion testable without going through a builder.

These cover the ordinary track and an extreme-count block, and assert that no
computed block returns 0.0, which ``block_tmrcas`` uses as its "never written"
sentinel. They do NOT exercise the normaliser-underflow branch in either
pass: no input constructed so far drives ``bs`` or ``s0`` to zero, so those
resets are guards by symmetry with the forward case rather than tested paths.
"""
import numpy as np
import pytest

from ancestree import _ensemble
from ancestree._smc_kernel import _pair_posterior_mean_impl

T, N = 16, 12


def _grid():
    lam = np.geomspace(1e-4, 1e-1, T)
    t_rep = 1.0 / lam
    return lam, np.log(lam), t_rep


def _inputs(bad_block=None, bad_count=100_000):
    lam, log_lam, t_rep = _grid()
    counts = np.zeros(N, dtype=np.int64)
    if bad_block is not None:
        counts[bad_block] = bad_count
    pi = np.full(T, 1.0 / T)
    r = np.full(T, 0.9)
    step = np.ones(N - 1)
    mask = np.ones(N, dtype=np.bool_)
    escale = np.ones(N)
    return counts, t_rep, lam, log_lam, r, pi, step, mask, escale


def test_an_ordinary_track_gives_finite_positive_means():
    pmean = _pair_posterior_mean_impl(*_inputs())
    assert np.all(np.isfinite(pmean))
    assert np.all(pmean > 0.0)


@pytest.mark.parametrize("bad", [0, N // 2, N - 1])
def test_an_underflowing_block_does_not_zero_its_neighbours(bad):
    """A dead block must not propagate, in either direction."""
    pmean = _pair_posterior_mean_impl(*_inputs(bad_block=bad))
    assert np.all(np.isfinite(pmean)), pmean
    others = np.delete(pmean, bad)
    assert np.all(others > 0.0), (
        f"block {bad} zeroed its neighbours: {pmean}")


def test_no_block_returns_the_never_written_sentinel():
    """block_tmrcas uses 0.0 for "never written", so a live block must not."""
    pmean = _pair_posterior_mean_impl(*_inputs(bad_block=N // 2))
    assert np.all(pmean != 0.0), (
        "a computed block returned 0.0, which collides with the sentinel")


def _smc_filtered(counts, lam, log_lam, r, pi):
    """Scaled forward rows from a dense reference, independent of both kernels.

    Built from the transition the kernels claim to implement, with the reset
    row ``nu_j`` proportional to ``pi_j (1 - r_j)``.

    :return: ``(n_blocks, T)`` scaled filtered distributions.
    """
    n, T_ = counts.shape[0], pi.shape[0]
    nu = pi * (1.0 - r)
    nu = nu / nu.sum()
    A = (1.0 - r)[:, None] * nu[None, :]
    A[np.diag_indices(T_)] += r

    def emit(k):
        ll = -lam + counts[k] * log_lam
        return np.exp(ll - ll.max())

    alpha = np.zeros((n, T_))
    alpha[0] = pi * emit(0)
    alpha[0] /= alpha[0].sum()
    for k in range(1, n):
        alpha[k] = (alpha[k - 1] @ A) * emit(k)
        alpha[k] /= alpha[k].sum()
    return alpha


def _smc_smoothed_mean(counts, t_rep, lam, log_lam, r, pi):
    """Dense forward-backward posterior-mean TMRCA per block.

    The quantity ``_pair_posterior_mean_impl`` returns, built here from the
    transition matrix directly so the JIT kernel has an independent reference.
    Written for the unmasked, unit-``escale``, unit-``step`` inputs
    :func:`_inputs` supplies.
    """
    n, T_ = counts.shape[0], t_rep.shape[0]
    A = np.outer(1.0 - r, pi)
    A[np.diag_indices(T_)] += r

    def emit(k):
        ll = -lam + counts[k] * log_lam
        return np.exp(ll - ll.max())

    alpha = _smc_filtered(counts, lam, log_lam, r, pi)
    beta = np.ones((n, T_))
    for k in range(n - 2, -1, -1):
        beta[k] = A @ (emit(k + 1) * beta[k + 1])
        beta[k] /= beta[k].sum()

    gamma = alpha * beta
    return (gamma * t_rep).sum(axis=1) / gamma.sum(axis=1)


def test_the_smc_posterior_mean_matches_a_dense_forward_backward():
    """The JIT kernel's smoothed mean must match a dense reference.

    ``_pair_posterior_mean_impl`` carries its own forward and backward
    recursions inline, and the surrounding tests assert only that its output is
    finite and positive, so a wrong recursion would pass them. This compares
    the quantity it returns against the same quantity computed from the
    transition matrix.
    """
    counts, t_rep, lam, log_lam, r, pi, step, mask, escale = _inputs()
    got = _pair_posterior_mean_impl(counts, t_rep, lam, log_lam, r, pi,
                                    step, mask, escale)
    want = _smc_smoothed_mean(counts, t_rep, lam, log_lam, r, pi)
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def test_the_ensemble_forward_matches_the_smc_forward():
    """Two implementations of one recursion must agree, block by block.

    Both kernels carry their own copy of the forward recursion, so they can
    drift apart. The ensemble kernel is compared here against a dense
    reference built in this file from the transition matrix itself. The SMC
    kernel is held to the same reference by
    :func:`test_the_smc_posterior_mean_matches_a_dense_forward_backward`, so
    the two are pinned to one another through it.
    """
    counts, t_rep, lam, log_lam, r, pi, step, mask, escale = _inputs()
    ref = _smc_filtered(counts, lam, log_lam, r, pi)
    assert ref.shape == (counts.shape[0], T)

    # The ensemble kernels take per-block lam/log_lam/r and a per-pair prior.
    lam2 = np.tile(lam, (N, 1))
    log_lam2 = np.tile(log_lam, (N, 1))
    r2 = np.tile(r, (N, 1))
    esc = np.ones(counts.shape[0])
    a_prev, a_cur = np.empty(T), np.empty(T)
    e, nu = np.empty(T), np.empty(T)

    _ensemble._forward_init(counts, lam2, log_lam2, pi, esc, a_prev, e, T)
    np.testing.assert_allclose(a_prev, ref[0], rtol=1e-10, atol=1e-12)
    for k in range(1, counts.shape[0]):
        _ensemble._forward_step(counts, lam2, log_lam2, r2, pi, esc, k,
                                a_prev, a_cur, e, nu, T)
        np.testing.assert_allclose(
            a_cur, ref[k], rtol=1e-10, atol=1e-12,
            err_msg=f"ensemble forward diverged from the reference at block {k}")
        a_prev[:] = a_cur
