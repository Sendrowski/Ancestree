"""Tests for substitution models: rate matrix structure, transition probabilities,
limit.

The model carries only structural parameters (``κ`` for K2). Rate scaling
(``μ``) and base composition (``π``) are supplied by the caller. Tests
here therefore feed ``mu * t`` directly as the branch length argument and
pass ``pi`` to ``Q(pi=...)`` / ``transition_probs(t, pi=...)`` when
checking F81 / HKY shapes.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pytest
import scipy.linalg

from ancestree import Site
from ancestree.models import _KAPPA_CLAMP, F81, GTR, HKY, JC69, K2


class TestJC69:
    def test_q_rows_sum_to_zero(self):
        Q = JC69().Q()
        assert np.allclose(Q.sum(axis=1), 0)

    def test_q_off_diagonals_equal(self):
        Q = JC69().Q()
        off = Q[~np.eye(4, dtype=bool)]
        assert np.allclose(off, off[0])

    def test_stationary_is_uniform(self):
        np.testing.assert_allclose(JC69().stationary(), [0.25] * 4)

    def test_stationary_ignores_pi(self):
        """The symmetric models are uniform regardless of any
        supplied ``pi`` (a non-uniform pi must not leak through)."""
        np.testing.assert_allclose(
            JC69().stationary(pi=[0.1, 0.2, 0.3, 0.4]), [0.25] * 4
        )
        np.testing.assert_allclose(
            K2(kappa=3.0).stationary(pi=[0.1, 0.2, 0.3, 0.4]), [0.25] * 4
        )

    def test_transition_probs_full_matches_analytic(self):
        """Standard JC69 closed form: P(t)[i,i] = 1/4 + 3/4 * exp(-4t/3).

        Per-state escape rate is r_i = 1 (mu lives in the branch-length
        scaling, so t is fed directly as expected substitutions per site).
        """
        t = 0.5
        P = JC69().transition_probs(t)
        expected_same = 0.25 + 0.75 * np.exp(-4 * t / 3)
        expected_diff = 0.25 - 0.25 * np.exp(-4 * t / 3)
        np.testing.assert_allclose(np.diag(P), expected_same, atol=1e-12)
        np.testing.assert_allclose(P[~np.eye(4, dtype=bool)], expected_diff, atol=1e-12)

    def test_transition_probs_rows_sum_to_one(self):
        P = JC69().transition_probs(0.3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)



    def test_transition_probs_batched_t(self):
        ts = np.array([0.1, 0.3, 1.0])
        P = JC69().transition_probs(ts)
        assert P.shape == (3, 4, 4)
        np.testing.assert_allclose(P.sum(axis=2), 1.0)
        for i, t in enumerate(ts):
            np.testing.assert_allclose(
                P[i], JC69().transition_probs(float(t))
            )


class TestK2:
    def test_q_rows_sum_to_zero(self):
        Q = K2(kappa=2.5).Q()
        assert np.allclose(Q.sum(axis=1), 0)

    def test_diagonal_matches_one(self):
        """Construction guarantees r_i = 1 for every row, independent of kappa.

        Per-data ``μ`` scaling is applied in the branch-length argument
        (or via the tree's :attr:`~ancestree.trees.Tree.time_scale`). The
        model itself normalises the per-state escape rate to 1.
        """
        for kappa in (0.5, 1.0, 2.0, 5.0):
            Q = K2(kappa=kappa).Q()
            np.testing.assert_allclose(-np.diag(Q), 1.0)

    def test_kappa_one_collapses_to_jc(self):
        Q_k2 = K2(kappa=1.0).Q()
        Q_jc = JC69().Q()
        np.testing.assert_allclose(Q_k2, Q_jc)

    def test_transitions_faster_than_transversions_for_large_kappa(self):
        Q = K2(kappa=5.0).Q()
        # A=0, C=1, G=2, T=3
        ag_transition = Q[0, 2]
        ac_transversion = Q[0, 1]
        assert ag_transition > ac_transversion

    def test_transition_probs_rows_sum_to_one(self):
        P = K2(kappa=2.0).transition_probs(0.3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            K2(kappa=0)
        with pytest.raises(ValueError):
            K2(kappa=-1.0)


def test_gtr_does_not_mutate_caller_rates_array():
    """GTR copies the caller's rates ndarray on ingest rather than
    aliasing it, so fixing a rate (or an optimiser step) cannot corrupt the
    caller-owned array."""
    from ancestree.models import GTR

    r = np.ones(6, dtype=float)
    m = GTR(rates=r, fit_rates=True)
    m.set_free_params({"rate_AC": 5.0})
    # The model's own buffer reflects the update...
    assert m.rates[0] == 5.0
    # ...but the caller's array is untouched.
    np.testing.assert_array_equal(r, np.ones(6))


class TestFullModeEigPath:
    """The ``full`` kernel computes ``exp(Q·t)`` from the eigendecomposition of
    ``Q``, vectorised over the branch array. It must agree with
    :func:`scipy.linalg.expm` to machine precision across all models and stay a
    valid (non-negative, row-stochastic) matrix at ``t = 0``."""

    def _models(self):
        from ancestree.models import F81, GTR, HKY
        pi = np.array([0.1, 0.2, 0.3, 0.4])
        return [
            (JC69(), None), (K2(kappa=3.1), None), (F81(), pi),
            (HKY(kappa=2.7), pi),
            (GTR(rates=[0.5, 1.2, 0.8, 1.0, 0.9, 1.1]), pi),
        ]

    def test_matches_scipy_expm_array_and_scalar(self):
        import scipy.linalg
        rng = np.random.default_rng(0)
        t = np.abs(rng.normal(scale=0.05, size=200))
        for m, pi in self._models():
            Q = m.Q(pi=pi)
            ref = np.stack([scipy.linalg.expm(Q * tk) for tk in t])
            got = m.transition_probs(t, pi=pi)
            np.testing.assert_allclose(got, ref, atol=1e-13)
            # scalar (cached) path agrees too
            np.testing.assert_allclose(
                m.transition_probs(0.037, pi=pi),
                scipy.linalg.expm(Q * 0.037), atol=1e-13,
            )

    def test_zero_branch_is_clean_identity(self):
        """``exp(Q·0)`` must be the exact identity with no negative
        reconstruction noise, since rows feed straight into samplers and logs."""
        for m, pi in self._models():
            P = m.transition_probs(np.array([0.0, 0.0]), pi=pi)
            assert np.all(P >= 0.0)
            np.testing.assert_allclose(P[0], np.eye(m.n_states), atol=1e-13)

    def test_rows_sum_to_one(self):
        rng = np.random.default_rng(3)
        t = np.abs(rng.normal(scale=0.1, size=50))
        for m, pi in self._models():
            P = m.transition_probs(t, pi=pi)
            np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)



def test_negative_branch_length_raises():
    """A negative t yields a non-stochastic exp(Qt), so it is refused rather
    than allowed to corrupt the likelihood."""
    with pytest.raises(ValueError, match="t >= 0"):
        JC69().transition_probs(-0.1)
    with pytest.raises(ValueError, match="t >= 0"):
        JC69().transition_probs(np.array([0.1, -0.2, 0.3]))


class TestTransitionCache:
    def test_scalar_call_is_cached(self):
        """A repeat scalar call should be served from the cache."""
        m = JC69()
        assert m.cache_info().currsize == 0
        m.transition_probs(0.5)
        info1 = m.cache_info()
        assert info1.misses == 1
        assert info1.hits == 0
        assert info1.currsize == 1
        m.transition_probs(0.5)
        info2 = m.cache_info()
        assert info2.hits == 1
        assert info2.misses == 1
        assert info2.currsize == 1

    def test_a_new_t_misses_and_a_repeat_hits(self):
        m = JC69()
        m.transition_probs(0.5)
        m.transition_probs(0.5)  # same t
        m.transition_probs(0.6)  # different t
        info = m.cache_info()
        assert info.misses == 2
        assert info.hits == 1
        assert info.currsize == 2

    def test_array_t_bypasses_cache(self):
        """Batched ``t`` should NOT touch the scalar cache."""
        m = JC69()
        m.transition_probs(np.array([0.1, 0.2, 0.3]))
        info = m.cache_info()
        assert info.misses == 0
        assert info.hits == 0
        assert info.currsize == 0

    def test_clear_cache_resets_state(self):
        m = JC69()
        m.transition_probs(0.5)
        m.transition_probs(0.6)
        assert m.cache_info().currsize == 2
        m.clear_cache()
        info = m.cache_info()
        assert info.currsize == 0
        assert info.hits == 0
        assert info.misses == 0

    def test_separate_instances_have_separate_caches(self):
        m1 = JC69()
        m2 = JC69()
        m1.transition_probs(0.5)
        m2.transition_probs(0.5)
        assert m1.cache_info().currsize == 1
        assert m2.cache_info().currsize == 1
        assert m1.cache_info().hits == 0
        m1.transition_probs(0.5)
        assert m1.cache_info().hits == 1
        assert m2.cache_info().hits == 0

    def test_cached_value_matches_uncached(self):
        m = JC69()
        cached = m.transition_probs(0.42)  # caches
        # Compare to a direct, uncached call via the array path on the same t.
        uncached = m.transition_probs(np.array([0.42]))[0]
        np.testing.assert_allclose(cached, uncached, atol=1e-15)


class TestK2EstimateKappaFromData:
    """K2.estimate_kappa_from_data, 2·Ts/Tv low-divergence MLE."""

    @staticmethod
    def _site(maj_a: str, maj_b: str, n: int = 5) -> Site:
        """Tiny site where samples_a all carry maj_a and samples_b all maj_b."""
        ta = {f"a{i}": maj_a for i in range(n)}
        tb = {f"b{i}": maj_b for i in range(n)}
        return Site(
            chrom="1", pos=1,
            alleles=tuple(sorted({maj_a, maj_b})),
            tip_alleles={**ta, **tb},
        )

    def test_pure_transitions(self):
        sites = [self._site("A", "G") for _ in range(10)]
        k = K2.estimate_kappa_from_data(
            sites, [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)],
        )
        # 10 Ts, 0 Tv: clamped to the upper bound, which K2 accepts.
        assert k == _KAPPA_CLAMP[1]
        K2(kappa=k)

    def test_pure_transversions(self):
        sites = [self._site("A", "C") for _ in range(10)]
        k = K2.estimate_kappa_from_data(
            sites, [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)],
        )
        # 0 Ts, 10 Tv: clamped to the lower bound, which K2 accepts.
        assert k == _KAPPA_CLAMP[0]
        K2(kappa=k)

    def test_one_to_one_ratio(self):
        # 5 Ts (A↔G) + 10 Tv (5×A↔C, 5×A↔T) → kappa = 2·5/10 = 1
        sites = (
            [self._site("A", "G")] * 5
            + [self._site("A", "C")] * 5
            + [self._site("A", "T")] * 5
        )
        k = K2.estimate_kappa_from_data(
            sites, [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)],
        )
        assert k == 1.0

    def test_no_differences_raises(self):
        sites = [self._site("A", "A") for _ in range(5)]
        with pytest.raises(ValueError, match="no pairwise differences"):
            K2.estimate_kappa_from_data(
                sites, [f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)],
            )


PI = np.array([0.1, 0.2, 0.3, 0.4])


class _NearDefectiveModel(JC69):
    """JC69 whose ``Q`` is a valid generator (rows sum to zero) but
    non-diagonalisable (a Jordan block for eigenvalue -3), so
    ``_expm_via_eig`` gives an ill-conditioned eigenvector matrix and returns
    ``None``, forcing the :func:`scipy.linalg.expm` fallback."""

    def Q(self, *, pi=None):  # noqa: D102 (see class docstring)
        return np.array(
            [[-3.0, 3.0, 0.0, 0.0],
             [0.0, -3.0, 3.0, 0.0],
             [0.0, 0.0, -3.0, 3.0],
             [0.0, 0.0, 0.0, 0.0]]
        )


class TestFittedModelPickling:
    """``pickle.loads(pickle.dumps(model))`` must reproduce ``P(t)`` bit-for-bit
    after the strip-and-rebuild-cache round trip, for a fitted HKY and GTR."""

    def test_hky_pickle_roundtrip_bit_identical(self):
        m = HKY(kappa=2.0, fit_kappa=True)
        m.set_free_params({"kappa": 3.3})  # "fit" the model
        m.transition_probs(0.37, pi=PI)  # prime the scalar cache
        clone = pickle.loads(pickle.dumps(m))
        assert clone.kappa == m.kappa
        # The cache closure is stripped on pickling and rebuilt on load.
        assert clone.cache_info().currsize == 0
        t_scalar = 0.53
        t_arr = np.array([0.05, 0.37, 1.2])
        for t in (t_scalar, t_arr):
            orig = m.transition_probs(t, pi=PI)
            got = clone.transition_probs(t, pi=PI)
            assert np.array_equal(orig, got)  # bit-identical, not just close

    def test_gtr_pickle_roundtrip_bit_identical(self):
        m = GTR(rates=[0.5, 1.2, 0.8, 1.0, 0.9, 1.1], fit_rates=True)
        m.set_free_params({"rate_AC": 0.7, "rate_GT": 1.4})  # "fit" the model
        m.transition_probs(0.21, pi=PI)  # prime the scalar cache
        clone = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(clone.rates, m.rates)
        assert clone.fit_mask == m.fit_mask
        assert clone.cache_info().currsize == 0
        t_scalar = 0.61
        t_arr = np.array([0.02, 0.21, 0.9])
        for t in (t_scalar, t_arr):
            orig = m.transition_probs(t, pi=PI)
            got = clone.transition_probs(t, pi=PI)
            assert np.array_equal(orig, got)


class TestScipyExpmFallback:
    """``_transition_probs_full`` on a near-defective ``Q`` (where
    ``_expm_via_eig`` returns ``None``) must fall back to
    :func:`scipy.linalg.expm`, giving a row-stochastic result."""

    def test_eig_path_returns_none_for_this_q(self):
        m = _NearDefectiveModel()
        Q = m.Q()
        assert m._expm_via_eig(Q, np.array([0.5, 1.0])) is None

    def test_array_t_fallback_matches_scipy_and_rows_sum_to_one(self):
        m = _NearDefectiveModel()
        Q = m.Q()
        t = np.array([0.05, 0.5, 1.0, 2.5])
        P = m._transition_probs_full(t)
        assert P.shape == (4, 4, 4)
        # Rows are valid probability vectors.
        np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)
        # Matches the per-element scipy.linalg.expm reference exactly.
        ref = np.stack([scipy.linalg.expm(Q * float(tk)) for tk in t])
        np.testing.assert_array_equal(P, ref)

    def test_scalar_t_fallback_matches_scipy(self):
        m = _NearDefectiveModel()
        Q = m.Q()
        P = m._transition_probs_full(0.75)
        assert P.shape == (4, 4)
        np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)
        np.testing.assert_array_equal(P, scipy.linalg.expm(Q * 0.75))


class TestArrayTNonSymmetricModels:
    """The array-``t`` transition path for F81 / HKY / GTR, whose ``Q``
    depends on ``pi``, equals the stack of per-scalar (cached) calls."""

    def _models(self):
        return [
            (F81(), PI),
            (HKY(kappa=2.7), PI),
            (GTR(rates=[0.5, 1.2, 0.8, 1.0, 0.9, 1.1]), PI),
        ]
    def test_array_t_matches_per_scalar(self):
        ts = np.array([0.03, 0.15, 0.4, 1.1])
        for m, pi in self._models():
            P = m.transition_probs(ts, pi=pi)
            assert P.shape == (4, 4, 4)
            np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)
            for i, t in enumerate(ts):
                per_scalar = m.transition_probs(float(t), pi=pi)
                np.testing.assert_allclose(P[i], per_scalar, atol=1e-13)


# ----------------------------------------------------------- rate normalisation
class TestRateNormalisation:
    """One unit of branch length is one expected substitution per site."""

    PIS = (
        (0.25, 0.25, 0.25, 0.25),
        (0.40, 0.15, 0.15, 0.30),
        (0.10, 0.40, 0.40, 0.10),
        (0.70, 0.10, 0.10, 0.10),
    )

    MODELS = (
        JC69(),
        K2(kappa=3.0),
        F81(),
        HKY(kappa=3.0),
        GTR([0.5, 3.0, 0.7, 0.9, 2.5, 1.1]),
    )

    @pytest.mark.parametrize("pi", PIS)
    @pytest.mark.parametrize("model", MODELS)
    def test_expected_substitutions_per_unit_t_is_one(self, model, pi):
        """``Σ_i π_i (-Q_ii) = 1``, with ``π`` the model's own stationary vector.

        ``π_i`` is the equilibrium frequency of state ``i`` and ``-Q_ii`` its
        escape rate per unit ``t``, so the sum is the expected substitutions
        per site per unit ``t`` at equilibrium. JC69 and K2 are uniform
        whatever ``π`` is passed, and the sum runs over that uniform vector.
        """
        pi_arr = np.asarray(pi, dtype=float)
        Q = model.Q(pi=pi_arr)
        stationary = model.stationary(pi=pi_arr)
        np.testing.assert_allclose(
            float(stationary @ -np.diag(Q)), 1.0, atol=1e-12,
        )


# ------------------------------------------------------------------- models
class TestModelValidation:

    def test_hky_set_free_params_rejects_nonpositive_kappa(self):
        m = HKY(fit_kappa=True)
        with pytest.raises(ValueError, match="kappa must be positive"):
            m.set_free_params({"kappa": -0.5})

    def test_gtr_fit_rates_wrong_length(self):
        with pytest.raises(ValueError, match="length 6"):
            GTR(fit_rates=[True] * 5)

    def test_gtr_bad_rate_bounds(self):
        with pytest.raises(ValueError, match="rate_bounds"):
            GTR(fit_rates=True, rate_bounds=(0.0, 1.0))


class TestWarnIfBoundsHit:
    """The post-fit bound-proximity warnings on the fittable models (emitted
    via the per-class logger at WARNING level)."""

    def test_k2_kappa_lower_and_upper_bound(self, caplog):
        m = K2(fit_kappa=True, kappa_bounds=(0.5, 5.0))
        m.kappa = 0.5
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "lower bound" in caplog.text
        caplog.clear()
        m.kappa = 5.0
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "upper bound" in caplog.text

    def test_hky_kappa_lower_and_upper_bound(self, caplog):
        m = HKY(fit_kappa=True, kappa_bounds=(0.5, 5.0))
        m.kappa = 0.5
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "lower bound" in caplog.text
        caplog.clear()
        m.kappa = 5.0
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "upper bound" in caplog.text

    def test_gtr_rate_lower_and_upper_bound(self, caplog):
        m = GTR(fit_rates=True, rate_bounds=(0.5, 5.0))
        m.rates[0] = 0.5
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "lower bound" in caplog.text
        caplog.clear()
        m.rates[0] = 5.0
        with caplog.at_level(logging.WARNING):
            m.warn_if_bounds_hit()
        assert "upper bound" in caplog.text

    def test_no_warning_when_not_fitting(self, caplog):
        # With the fit flags off, warn_if_bounds_hit short-circuits (no warning).
        with caplog.at_level(logging.WARNING):
            K2().warn_if_bounds_hit()
            HKY().warn_if_bounds_hit()
        assert not caplog.records
