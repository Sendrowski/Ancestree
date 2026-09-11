"""The pairwise HMM must not move the TMRCA marginal in the absence of data.

The transition is ``A[i, j] = r_i d_ij + (1 - r_i) nu_j``, with ``i``, ``j``
indexing the discretised TMRCA bins, ``r_i = exp(-2 rho t_i B)`` the probability
of no recombination across one block given bin ``i``, ``rho`` the per-site
per-generation recombination rate, ``t_i`` the bin's representative time in
generations, ``B`` the block size in bp, and ``nu`` the row a recombination
resets to.

Resetting to the prior ``pi`` itself does not leave ``pi`` invariant, because
``r_i`` depends on depth: the chain then relaxes toward ``pi_j / (1 - r_j)``,
which up-weights shallow bins, and the posterior mean TMRCA decays along the
sequence toward the present. On a 32-bin grid over [300, 3e6] generations that
decay was a factor of 3.4 with no data at all. The coalescent's marginal TMRCA
is the same at every position, so these tests pin the invariance directly.
"""
import numpy as np
import pytest

from ancestree._smc_kernel import pair_posterior_mean_tmrca

T = 32
LO, HI = 300.0, 3.0e6
RHO, BLOCK, MU = 1e-8, 500, 1e-8


def _grid(t_bar):
    """The prior, its bin times and the per-block no-recombination probability.

    :param t_bar: Genome-average TMRCA for the pair, in generations.
    :return: ``(t_rep, pi, r, lam)``.
    """
    edges = np.exp(np.linspace(np.log(LO), np.log(HI), T + 1))
    t_rep = np.sqrt(edges[:-1] * edges[1:])
    pi = np.exp(-edges[:-1] / t_bar) - np.exp(-edges[1:] / t_bar)
    pi /= pi.sum()
    r = np.exp(-2.0 * RHO * t_rep * BLOCK)
    lam = 2.0 * MU * t_rep * BLOCK
    return t_rep, pi, r, lam


@pytest.mark.parametrize("t_bar", [2e4, 2e5])
def test_no_data_leaves_the_posterior_at_the_prior_mean(t_bar):
    """With every block masked the posterior mean must equal the prior mean.

    Masked blocks emit uniformly, so nothing but the transition can move the
    marginal. Any drift along the 400 blocks is the transition failing to hold
    its own prior.
    """
    t_rep, pi, r, lam = _grid(t_bar)
    n = 400
    pm = pair_posterior_mean_tmrca(
        np.zeros(n, np.int64), t_rep, lam, np.log(lam), r, pi,
        mask=np.zeros(n, bool))
    prior_mean = float(pi @ t_rep)
    assert np.allclose(pm, prior_mean, rtol=1e-10), (
        f"posterior mean TMRCA drifted from {pm[0]:.1f} to {pm[-1]:.1f} "
        f"generations against a prior mean of {prior_mean:.1f}")


def test_the_reset_row_leaves_the_prior_invariant():
    """``pi A = pi`` for the reset row the kernel uses, and not for ``pi``.

    Stated on the transition matrix itself, so a change to the kernel's reset
    row is caught here even if the forward-backward masks it.
    """
    from ancestree._smc_kernel import _reset_row

    t_rep, pi, r, _ = _grid(2e4)
    # Built by the kernel, not by the test: an arbitrarily wrong reset row in
    # the kernel must fail here, which re-deriving nu inline cannot detect.
    nu = np.empty_like(pi)
    _reset_row(pi, r, nu)
    assert np.isclose(nu.sum(), 1.0)
    good = pi * r + nu * float(pi @ (1.0 - r))
    assert np.max(np.abs(good - pi) / pi) < 1e-12
    # Resetting to the prior instead is what the invariance rules out.
    bad = pi * r + pi * (1.0 - float(pi @ r))
    assert np.max(np.abs(bad - pi) / pi) > 0.1


def test_a_wider_step_is_still_stationary():
    """Masked gaps and rate maps widen a step, and ``nu`` tracks the width.

    ``r_i`` is raised to the step width, so a reset row derived from the
    one-block ``r`` would only be stationary for adjacent uniform blocks.
    """
    t_rep, pi, r, lam = _grid(2e4)
    n = 200
    step = np.full(n - 1, 7.5)
    pm = pair_posterior_mean_tmrca(
        np.zeros(n, np.int64), t_rep, lam, np.log(lam), r, pi,
        step=step, mask=np.zeros(n, bool))
    assert np.allclose(pm, float(pi @ t_rep), rtol=1e-10)
