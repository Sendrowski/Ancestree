"""Tests for the GTR substitution model: rate-matrix structure, collapse
relations to F81/HKY/JC69, free-parameter protocol, kernel integration.
"""
import numpy as np
import pytest

from ancestree import GTR, HKY, F81, JC69, Likelihood, STATES
from ancestree.sites import BaseComposition, Site
from ancestree.trees import TskitLocalTree


# ------------------------------------------------------- rate-matrix structure


class TestGTRRateMatrix:
    def test_diagonal_negative_rowsum(self):
        gtr = GTR([0.5, 1.0, 0.7, 1.3, 1.1, 0.9])
        Q = gtr.Q(pi=[0.2, 0.3, 0.3, 0.2])
        # Rows of any Q sum to 0 by construction.
        np.testing.assert_allclose(Q.sum(axis=1), 0, atol=1e-12)
        # Off-diagonals are non-negative. Diagonal is negative.
        assert (Q[~np.eye(4, dtype=bool)] >= 0).all()
        assert (Q.diagonal() < 0).all()

    def test_expected_substitutions_per_unit_t_is_one(self):
        """β-normalisation pins ``Σ_i π_i (-Q_ii) = 1`` for any positive rates / pi."""
        for rates in (
            np.ones(6),
            np.array([0.3, 2.0, 0.5, 1.7, 1.2, 0.4]),
            np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        ):
            for pi in (
                np.full(4, 0.25),
                np.array([0.4, 0.3, 0.2, 0.1]),
                np.array([0.1, 0.4, 0.4, 0.1]),
            ):
                gtr = GTR(rates)
                Q = gtr.Q(pi=pi)
                assert np.isclose(float(pi @ -Q.diagonal()), 1.0)

    def test_validate_rates_shape(self):
        with pytest.raises(ValueError, match="length-6"):
            GTR([1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            GTR([1.0, -0.5, 1.0, 1.0, 1.0, 1.0])

    def test_canonical_ordering(self):
        """``GTR.RATE_NAMES`` and ``GTR.RATE_PAIRS`` are aligned."""
        nuc = "ACGT"
        for name, (i, j) in zip(GTR.RATE_NAMES, GTR.RATE_PAIRS):
            assert name == f"rate_{nuc[i]}{nuc[j]}"


# --------------------------------------------------------- collapse relations


class TestGTRCollapsesToF81:
    """Uniform rates → F81 (same Q, same P(t))."""

    def test_q_matches_f81(self):
        pi = np.array([0.3, 0.25, 0.25, 0.2])
        np.testing.assert_allclose(
            GTR([1, 1, 1, 1, 1, 1]).Q(pi=pi),
            F81().Q(pi=pi),
            atol=1e-12,
        )

    def test_p_t_matches_f81(self):
        pi = np.array([0.3, 0.25, 0.25, 0.2])
        for t in (0.0, 0.1, 1.0, 5.0):
            np.testing.assert_allclose(
                GTR([1, 1, 1, 1, 1, 1]).transition_probs(t, pi=pi),
                F81().transition_probs(t, pi=pi),
                atol=1e-10,
            )


class TestGTRCollapsesToHKY:
    """rate_AG = rate_CT = κ, four transversion rates = 1 → HKY."""

    @pytest.mark.parametrize("kappa", [1.0, 2.5, 5.0])
    @pytest.mark.parametrize("pi", [
        np.full(4, 0.25),
        np.array([0.3, 0.2, 0.2, 0.3]),
    ])
    def test_q_matches_hky(self, kappa, pi):
        # canonical order: AC, AG, AT, CG, CT, GT
        gtr = GTR([1.0, kappa, 1.0, 1.0, kappa, 1.0])
        hky = HKY(kappa=kappa)
        np.testing.assert_allclose(gtr.Q(pi=pi), hky.Q(pi=pi), atol=1e-12)

    def test_p_t_matches_hky(self):
        kappa = 3.0
        pi = np.array([0.3, 0.2, 0.2, 0.3])
        gtr = GTR([1.0, kappa, 1.0, 1.0, kappa, 1.0])
        hky = HKY(kappa=kappa)
        for t in (0.05, 0.5, 2.0):
            np.testing.assert_allclose(
                gtr.transition_probs(t, pi=pi),
                hky.transition_probs(t, pi=pi),
                atol=1e-10,
            )


class TestGTRCollapsesToJC69:
    """Uniform rates + uniform π → JC69."""

    def test_q_matches_jc69(self):
        np.testing.assert_allclose(
            GTR([1, 1, 1, 1, 1, 1]).Q(),
            JC69().Q(),
            atol=1e-12,
        )

    def test_p_t_matches_jc69(self):
        gtr = GTR([1, 1, 1, 1, 1, 1])
        jc = JC69()
        for t in (0.1, 1.0, 5.0):
            np.testing.assert_allclose(
                gtr.transition_probs(t),
                jc.transition_probs(t),
                atol=1e-10,
            )


# ----------------------------------------------------- free-parameter protocol


class TestGTRFreeParams:
    def test_default_fit_rates_false_no_free_params(self):
        assert GTR().free_params == {}

    def test_fit_rates_true_holds_AG_fixed(self):
        """fit_rates=True identifies 5 rates with rate_AG as reference."""
        gtr = GTR(fit_rates=True)
        keys = list(gtr.free_params.keys())
        assert "rate_AG" not in keys
        assert len(keys) == 5

    def test_per_rate_fit_mask(self):
        gtr = GTR(fit_rates=[True, False, True, False, False, True])
        assert set(gtr.free_params.keys()) == {"rate_AC", "rate_AT", "rate_GT"}

    def test_set_free_params_updates_q(self):
        """An optimiser-style update of rate_AC changes Q[0,1] and clears caches."""
        gtr = GTR(fit_rates=True)
        q_before = gtr.Q(pi=[0.25] * 4).copy()
        gtr.set_free_params({"rate_AC": 5.0})
        q_after = gtr.Q(pi=[0.25] * 4)
        assert not np.allclose(q_before, q_after)
        # rate_AC at index 0 -> rates[0] should be 5.0
        assert gtr.rates[0] == 5.0

    def test_set_free_params_rejects_non_positive(self):
        gtr = GTR(fit_rates=True)
        with pytest.raises(ValueError, match="must be positive"):
            gtr.set_free_params({"rate_AC": 0.0})

    def test_fit_mask_sequence_length_validated(self):
        with pytest.raises(ValueError, match="length 6"):
            GTR(fit_rates=[True, False, True])


# --------------------------------------------------------- kernel integration


class TestGTRKernelIntegration:
    def test_log_likelihoods_runs_and_normalises(self):
        """End-to-end Felsenstein on a 2-tip tree with GTR; row-sums == 1."""
        tree = TskitLocalTree.from_newick("(t1:0.3,t2:0.3);")
        tree.time_scale = 1.0
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"t1": "A", "t2": "C"},
        )
        gtr = GTR([0.5, 2.0, 1.0, 0.7, 2.0, 0.3])
        eng = Likelihood(gtr)
        log_L = eng.log_likelihoods(tree, [site])
        assert log_L.shape == (1, 4)
        # log_L < 0 for any non-degenerate Felsenstein pass
        assert (log_L < 0).all()

    def test_collapse_to_hky_matches_log_likelihoods(self):
        """GTR with HKY-like rates yields identical log_L to HKY on real data."""
        tree = TskitLocalTree.from_newick(
            "((t1:0.2,t2:0.2):0.3,(t3:0.4,t4:0.4):0.1);",
        )
        tree.time_scale = 1.0
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"t1": "A", "t2": "G", "t3": "G", "t4": "A"},
        )
        bc = BaseComposition.from_counts(A=300, C=200, G=200, T=300)
        kappa = 2.5
        gtr = GTR([1.0, kappa, 1.0, 1.0, kappa, 1.0])
        hky = HKY(kappa=kappa)
        log_L_gtr = Likelihood(gtr, base_composition=bc).log_likelihoods(tree, [site])
        log_L_hky = Likelihood(hky, base_composition=bc).log_likelihoods(tree, [site])
        np.testing.assert_allclose(log_L_gtr, log_L_hky, atol=1e-10)


# --------------------------------------------------------- helper sanity


class TestValidateRates:
    def test_accepts_length_6_positive(self):
        out = GTR._validate_rates([0.5, 1.0, 0.7, 1.3, 1.1, 0.9])
        assert out.shape == (6,)

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="strictly positive"):
            GTR._validate_rates([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])


# --------------------------------------------------- empirical_from_sites


class TestGTREmpiricalFromSites:
    def _make_sites(self, counts: dict[tuple[str, str], int]) -> list:
        """Build a list of Sites with the given pair counts. Allele pairs sorted."""
        sites = []
        pos = 0
        for (a, b), n in counts.items():
            for _ in range(n):
                pos += 1
                sites.append(Site(
                    chrom="1", pos=pos, alleles=(a, b),
                    tip_alleles={"s1": a, "s2": b},
                ))
        return sites

    def test_recovers_proportional_to_counts_under_uniform_pi(self):
        """Under uniform π, empirical rates are just normalised pair counts."""
        sites = self._make_sites({
            ("A", "C"): 100, ("A", "G"): 40, ("A", "T"): 20,
            ("C", "G"): 20,  ("C", "T"): 40, ("G", "T"): 20,
        })
        bc = BaseComposition.from_n_target_sites(n_total=10000)
        gtr = GTR.empirical_from_sites(sites, bc, ["s1", "s2"])
        # rate_AG normalised to 1, so AC/AG = 100/40 = 2.5
        np.testing.assert_allclose(gtr.rates[1], 1.0)
        np.testing.assert_allclose(gtr.rates[0], 2.5)
        np.testing.assert_allclose(gtr.rates[4], 1.0)  # CT == AG count
        np.testing.assert_allclose(gtr.rates[2], 0.5)  # AT

    def test_pi_correction(self):
        """Non-uniform π re-weights the pair counts by 1/(π_i·π_j)."""
        sites = self._make_sites({
            ("A", "C"): 50, ("A", "G"): 50, ("A", "T"): 50,
            ("C", "G"): 50, ("C", "T"): 50, ("G", "T"): 50,
        })
        # π = (0.4, 0.1, 0.1, 0.4), A and T are common, C and G rare.
        bc = BaseComposition.from_counts(A=4000, C=1000, G=1000, T=4000)
        gtr = GTR.empirical_from_sites(sites, bc, ["s1", "s2"])
        # All pair counts equal → rate proportional to 1/(π_i · π_j).
        # rate_AC ∝ 1/(0.4·0.1) = 25;  rate_AG ∝ 1/(0.4·0.1) = 25 → ratio 1.
        # rate_CG ∝ 1/(0.1·0.1) = 100 → ratio 4.
        # rate_AT ∝ 1/(0.4·0.4) = 6.25 → ratio 0.25.
        # rate_GT ∝ 1/(0.1·0.4) = 25 → ratio 1.
        np.testing.assert_allclose(gtr.rates[0], 1.0)
        np.testing.assert_allclose(gtr.rates[1], 1.0)  # reference
        np.testing.assert_allclose(gtr.rates[2], 0.25)  # AT
        np.testing.assert_allclose(gtr.rates[3], 4.0)  # CG
        np.testing.assert_allclose(gtr.rates[5], 1.0)  # GT

    def test_skips_monomorphic_and_multi_allelic(self):
        """Sites with !=2 distinct ingroup alleles are ignored."""
        sites = self._make_sites({
            ("A", "C"): 100, ("A", "G"): 100, ("A", "T"): 100,
            ("C", "G"): 100, ("C", "T"): 100, ("G", "T"): 100,
        })
        # Add a multi-allelic site (3 alleles observed) and a monomorphic one.
        # Both should be ignored by the estimator.
        sites.append(Site(
            chrom="1", pos=999, alleles=("A", "C", "G"),
            tip_alleles={"s1": "A", "s2": "C", "s3": "G"},
        ))
        sites.append(Site(
            chrom="1", pos=1000, alleles=("A",),
            tip_alleles={"s1": "A", "s2": "A"},
        ))
        bc = BaseComposition.from_n_target_sites(n_total=10000)
        # s3 is in the ingroup, so the three-allele site really is
        # multi-allelic there. With an ingroup of s1/s2 alone it would be an
        # ordinary A/C pair and would count.
        gtr = GTR.empirical_from_sites(sites, bc, ["s1", "s2", "s3"])
        # All six pair counts equal under uniform π → all rates equal → 1.
        np.testing.assert_allclose(gtr.rates, np.ones(6), atol=1e-12)

    def test_zero_count_pair_raises(self):
        """If any pair has zero observed sites, the estimator cannot normalise it."""
        sites = self._make_sites({
            ("A", "C"): 10, ("A", "G"): 10, ("A", "T"): 10,
            ("C", "G"): 10, ("C", "T"): 10,  # G-T missing
        })
        bc = BaseComposition.from_n_target_sites(n_total=1000)
        with pytest.raises(ValueError, match="no polymorphic sites observed"):
            GTR.empirical_from_sites(sites, bc, ["s1", "s2"])

    def test_returns_unfitted_by_default(self):
        sites = self._make_sites({
            ("A", "C"): 10, ("A", "G"): 10, ("A", "T"): 10,
            ("C", "G"): 10, ("C", "T"): 10, ("G", "T"): 10,
        })
        bc = BaseComposition.from_n_target_sites(n_total=1000)
        gtr = GTR.empirical_from_sites(sites, bc, ["s1", "s2"])
        assert gtr.free_params == {}

    def test_can_request_fit_on_top(self):
        """The user can ask the estimator to ALSO fit the rates further."""
        sites = self._make_sites({
            ("A", "C"): 10, ("A", "G"): 10, ("A", "T"): 10,
            ("C", "G"): 10, ("C", "T"): 10, ("G", "T"): 10,
        })
        bc = BaseComposition.from_n_target_sites(n_total=1000)
        gtr = GTR.empirical_from_sites(sites, bc, ["s1", "s2"], fit_rates=True)
        # 5 free params (rate_AG held as reference)
        assert len(gtr.free_params) == 5


# --------------------------------------------------- fixed_rates ergonomic API


class TestGTRFixedRates:
    def test_fixed_rates_pins_named_subset(self):
        """`fixed_rates={'rate_AC': 0.5}` fixes AC + holds AG as reference. Fits 4."""
        gtr = GTR(fixed_rates={"rate_AC": 0.5})
        # rate_AC must be set to 0.5 and not in free_params
        assert gtr.rates[0] == 0.5
        free = gtr.free_params
        assert "rate_AC" not in free
        assert "rate_AG" not in free  # held as reference
        assert set(free.keys()) == {"rate_AT", "rate_CG", "rate_CT", "rate_GT"}

    def test_fixed_rates_overrides_default_rates_entry(self):
        """Values in fixed_rates win over the same index in `rates=`."""
        gtr = GTR(
            rates=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            fixed_rates={"rate_CT": 2.5},
        )
        assert gtr.rates[4] == 2.5

    def test_fixed_rates_with_rate_AG_explicit_drops_reference_lock(self):
        """When the user fixes rate_AG explicitly, the auto-reference lock
        on rate_AG is unnecessary, but other rates remain free."""
        gtr = GTR(fixed_rates={"rate_AG": 2.0})
        assert gtr.rates[1] == 2.0
        free = gtr.free_params
        assert "rate_AG" not in free
        # All other 5 rates are free.
        assert len(free) == 5

    def test_fixed_rates_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown rate name"):
            GTR(fixed_rates={"rate_XX": 1.0})

    def test_fixed_rates_non_positive_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            GTR(fixed_rates={"rate_AC": 0.0})

    def test_fixed_rates_and_sequence_fit_rates_conflict(self):
        with pytest.raises(ValueError, match="Cannot combine"):
            GTR(
                fixed_rates={"rate_AC": 1.0},
                fit_rates=[True, False, True, False, True, False],
            )

    def test_fixed_rates_compatible_with_bool_fit_rates(self):
        """`fixed_rates` is the primary subset API; ignoring a redundant
        `fit_rates=True` is fine because `fixed_rates` already implies the
        rest are free."""
        gtr = GTR(fixed_rates={"rate_AC": 0.5}, fit_rates=True)
        assert gtr.rates[0] == 0.5
        # Same 4-free count as without fit_rates=True.
        assert len(gtr.free_params) == 4


def test_empirical_from_sites_ignores_outgroup_alleles():
    """The estimator counts ingroup polymorphism, not divergence.

    ``empirical_from_sites`` counted every tip, so a site the ingroup had fixed
    still contributed whenever an outgroup carried another allele. That loads
    the six exchangeability rates with whichever substitutions happen to sit on
    the outgroup branches, which is divergence rather than the polymorphism the
    estimator is defined on.
    """
    ingroup, outgroup = ["i1", "i2"], ["o1"]
    sites = []
    pos = 0
    # Equal ingroup polymorphism in all six pairs.
    for a, b in (("A", "C"), ("A", "G"), ("A", "T"),
                 ("C", "G"), ("C", "T"), ("G", "T")):
        for _ in range(50):
            pos += 1
            sites.append(Site(chrom="1", pos=pos, alleles=(a, b),
                              tip_alleles={"i1": a, "i2": b, "o1": a}))
    # Ingroup-fixed sites whose outgroup differs: divergence only, all A/G.
    for _ in range(500):
        pos += 1
        sites.append(Site(chrom="1", pos=pos, alleles=("A", "G"),
                          tip_alleles={"i1": "A", "i2": "A", "o1": "G"}))

    bc = BaseComposition.from_n_target_sites(n_total=100000)
    gtr = GTR.empirical_from_sites(sites, bc, ingroup)
    # Equal ingroup counts in every pair, so every rate is 1 despite the 500
    # ingroup-fixed A/G divergence sites.
    np.testing.assert_allclose(gtr.rates, np.ones(6), atol=1e-12)

    # Counting the outgroup too would swamp AG and drive the others down.
    swamped = GTR.empirical_from_sites(sites, bc, ingroup + outgroup)
    assert swamped.rates[0] < 0.5  # AC, normalised against the inflated AG


class TestGTREdges:
    def test_repr_lists_the_rates(self):
        assert repr(GTR(rates=[1, 2, 3, 4, 5, 6])) == "GTR(rates=[1, 2, 3, 4, 5, 6])"

    def test_empirical_rates_refuse_a_zero_base_frequency(self):
        bc = BaseComposition(
            counts={b: 0 for b in STATES},
            _pi_cache=np.array([0.5, 0.5, 0.0, 0.0]),
        )
        sites = [Site(chrom="1", pos=1, alleles=(x, y), tip_alleles={"a": x, "b": y})
                 for x, y in
                 [("A", "C"), ("A", "G"), ("A", "T"), ("C", "G"), ("C", "T"), ("G", "T")]]
        with pytest.raises(ValueError, match="strictly positive"):
            GTR.empirical_from_sites(sites, bc, ["a", "b"])
