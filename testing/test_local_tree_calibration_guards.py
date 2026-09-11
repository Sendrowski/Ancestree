"""Degenerate inputs to the local-tree calibration that must not pass silently.

An over-deep time grid would wrap its int8 bin indices, a per-bin weight fit
with a flat objective would return its own random start, and uncalled blocks
would deflate the pair prior mean.
"""
import numpy as np
import pytest

import ancestree as anc
from ancestree.priors import _fit_pi_bin


def test_a_time_grid_deeper_than_int8_is_refused():
    """Bin indices are stored int8, and 128+ would wrap negative and invert TMRCAs."""
    with pytest.raises(ValueError, match="n_time_bins"):
        anc.PairwiseCoalescentHMM(n_haplotypes=4, mu=1e-8, rec_rate=1e-8,
                                 n_time_bins=150)


def test_the_default_grid_is_accepted():
    hmm = anc.PairwiseCoalescentHMM(n_haplotypes=4, mu=1e-8, rec_rate=1e-8,
                                   n_time_bins=32)
    assert hmm.n_time_bins == 32


def test_a_flat_per_bin_objective_returns_the_symmetric_default():
    """Identical major/minor likelihoods carry no information about pi.

    L-BFGS-B terminates at x0 on a flat objective, so the fit must return the
    symmetric default rather than its random start.
    """
    flat = [(0.5, 0.5, 1.0)] * 160
    fits = {_fit_pi_bin(flat, 4, seed) for seed in (0, 1, 7, 13)}
    assert fits == {0.5}, f"fit is seed-dependent on a flat objective: {fits}"


def test_a_real_per_bin_objective_still_moves_off_the_default():
    """The guard must not swallow a genuine fit."""
    informative = [(0.9, 0.1, 1.0)] * 200
    fit = _fit_pi_bin(informative, 4, 0)
    assert 0.0 < fit < 1.0
    assert abs(fit - 0.5) > 1e-6, "an informative bin must not return 0.5"


def test_the_per_bin_fit_uses_the_hypergeometric_weights():
    """``_fit_pi_bin`` maximises the weighted per-site log-likelihood.

    The weights are each site's hypergeometric sub-sampling mass. The data
    here is chosen so the weighted and unweighted objectives separate
    completely: the heavily-weighted sites favour the major allele, the
    lightly-weighted ones the minor, so the weighted fit goes to the upper
    bound while the unweighted fit sits at the symmetric default.
    """
    from ancestree.priors import _fit_pi_bin

    data = [(0.9, 0.1, 10.0)] * 3 + [(0.1, 0.9, 0.5)] * 3
    assert _fit_pi_bin(data, 4, 0) == pytest.approx(1.0, abs=1e-6)
    unweighted = [(major, minor, 1.0) for major, minor, _ in data]
    assert _fit_pi_bin(unweighted, 4, 0) == pytest.approx(0.5, abs=1e-6)


def test_uncalled_blocks_do_not_deflate_the_pair_prior_mean():
    """``t_bar`` is the mean over blocks the pair is actually called in.

    A block carrying no called site for the pair has a zero count and says
    nothing about coalescent time. Averaged in, it would scale the pair's
    prior mean by its called fraction.
    """

    rng = np.random.default_rng(0)
    n_blocks, true_t, mu, rec, block = 400, 20_000.0, 1.25e-8, 1e-8, 2000
    hmm = anc.PairwiseCoalescentHMM(
        n_haplotypes=2, mu=mu, rec_rate=rec, block_size=block)
    base = rng.poisson(2 * mu * block * true_t, size=(1, n_blocks)).astype(float)

    def t_bar(uncalled_fraction):
        counts, called = base.copy(), np.ones((1, n_blocks))
        k = int(uncalled_fraction * n_blocks)
        if k:
            counts[0, :k] = 0.0
            called[0, :k] = 0.0
        return float(np.asarray(
            hmm.calibrate(counts, called_frac=called)[-1]).ravel()[0])

    reference = t_bar(0.0)
    # Half the blocks uncalled must not halve the prior mean. The bound is
    # loose enough for the sampling noise of a shorter average.
    for fraction, floor in ((0.10, 0.95), (0.25, 0.90), (0.50, 0.85)):
        ratio = t_bar(fraction) / reference
        assert ratio > floor, (
            f"{fraction:.0%} uncalled deflated t_bar to {ratio:.3f}")
