"""Degenerate inputs to the local-tree calibration that must not pass silently.

An over-deep time grid would wrap its int8 bin indices, a per-bin weight fit
with a flat objective would return its own random start, and uncalled blocks
would deflate the pair prior mean.
"""
import logging

import numpy as np
import pytest

import ancestree as anc
from ancestree import PairwiseCoalescentHMM
from ancestree.priors import _fit_pi_bin
from testing._helpers import MU, REC, toy_builder, toy_inference, toy_sites


def test_a_time_grid_deeper_than_int8_is_refused():
    """Bin indices are stored int8, and 128+ would wrap negative and invert TMRCAs."""
    with pytest.raises(ValueError, match="n_time_bins"):
        anc.PairwiseCoalescentHMM(n_haplotypes=4, mu=1e-8, rec_rate=1e-8,
                                 n_time_bins=150)


def test_the_default_grid_is_accepted():
    hmm = anc.PairwiseCoalescentHMM(n_haplotypes=4, mu=1e-8, rec_rate=1e-8,
                                   n_time_bins=32)
    assert hmm.n_time_bins == 32


@pytest.mark.parametrize("grid, message", [
    ([100.0], "at least 2 edges"),
    (np.ones((2, 2)), "at least 2 edges"),
    ([0.0, 10.0], "finite and positive"),
    ([1.0, np.inf], "finite and positive"),
    ([10.0, 5.0], "strictly ascending"),
    ([10.0, 10.0], "strictly ascending"),
])
def test_check_time_grid_rejects_malformed_edges(grid, message):
    """A grid must be one strictly ascending, positive, finite dimension."""
    with pytest.raises(ValueError, match=message):
        PairwiseCoalescentHMM._check_time_grid(grid)


def test_explicit_time_grid_sets_the_bin_count():
    """An explicit grid overrides ``n_time_bins`` on every constructor."""
    grid = np.array([10.0, 100.0, 1000.0, 10_000.0])
    hmm = PairwiseCoalescentHMM(2, mu=MU, rec_rate=REC, time_grid=grid,
                                n_time_bins=32)
    assert hmm.n_time_bins == 3
    sites, names = toy_sites(range(0, 2000, 20))
    b = toy_builder(sites, names, 2000.0, time_grid=grid)
    assert b.n_time_bins == 3
    inf = toy_inference(sites, names, time_grid=grid, n_time_bins=32)
    assert inf.n_time_bins == 3
    np.testing.assert_array_equal(inf.time_grid, grid)


def test_time_grid_without_differences_takes_the_default_span():
    """All-zero counts calibrate the grid over ``[1e2, 1e5]`` generations."""
    hmm = PairwiseCoalescentHMM(2, mu=MU, rec_rate=REC, n_time_bins=4)
    edges, t_rep = hmm._calibrate_time_grid(np.zeros((1, 5)))
    assert edges[0] == pytest.approx(1e2)
    assert edges[-1] == pytest.approx(1e5)
    assert edges.shape == (5,) and t_rep.shape == (4,)


def test_time_grid_deep_end_is_pushed_above_a_shallow_floor():
    """When the deep end falls under the floor it is set to a hundred floors."""
    hmm = PairwiseCoalescentHMM(2, mu=1e-2, rec_rate=REC, block_size=10,
                                n_time_bins=4)
    # crude TMRCA of a one-difference block is 1 / (2 mu block) = 5 generations,
    # below the 10-generation floor.
    edges, _t_rep = hmm._calibrate_time_grid(np.array([[1, 0, 1, 0]]))
    assert edges[0] == pytest.approx(10.0)
    assert edges[-1] == pytest.approx(1000.0)


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


def test_calibrate_gives_a_never_called_pair_the_median_prior_mean(caplog):
    """A pair with no jointly called block carries the panel's median mean."""
    hmm = PairwiseCoalescentHMM(3, mu=1e-3, rec_rate=REC, block_size=10)
    counts = np.array([[0, 0, 0, 0],
                       [2, 4, 2, 4],
                       [1, 1, 1, 1]], dtype=np.int64)
    called = np.ones_like(counts, dtype=np.float64)
    called[0] = 0.0
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        _edges, _t_rep, _lam, _log_lam, _r, t_bar = hmm.calibrate(
            counts, called_frac=called)
    per_block = 2.0 * hmm.mu * hmm.block_size
    np.testing.assert_allclose(t_bar[1], 3.0 / per_block)
    np.testing.assert_allclose(t_bar[2], 1.0 / per_block)
    assert t_bar[0] == pytest.approx(float(np.median(t_bar[1:])))
    assert any("never called at the same site" in r.getMessage()
               for r in caplog.records)


def test_calibrate_with_every_pair_blind_falls_back_to_the_grid_mean(caplog):
    """With no pair called anywhere the mean of the grid representatives is used."""
    hmm = PairwiseCoalescentHMM(2, mu=1e-3, rec_rate=REC, block_size=10,
                                n_time_bins=4)
    counts = np.zeros((1, 4), dtype=np.int64)
    called = np.zeros((1, 4), dtype=np.float64)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        _edges, t_rep, _lam, _log_lam, _r, t_bar = hmm.calibrate(
            counts, called_frac=called)
    assert t_bar.shape == (1,)
    assert t_bar[0] == pytest.approx(float(np.mean(t_rep)))


def test_coalescent_prior_falls_back_to_uniform_for_a_non_finite_mean():
    """A non-finite pair mean gives the uniform prior over the bins."""
    hmm = PairwiseCoalescentHMM(2, mu=MU, rec_rate=REC, n_time_bins=5)
    edges = np.geomspace(10.0, 1e4, 6)
    pi = hmm._coalescent_prior(edges, np.nan)
    np.testing.assert_allclose(pi, np.full(5, 0.2))
    finite = hmm._coalescent_prior(edges, 100.0)
    assert finite.sum() == pytest.approx(1.0)
    assert finite[0] > finite[-1]
