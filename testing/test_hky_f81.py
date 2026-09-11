"""Tests for the F81 and HKY substitution models.

F81/HKY carry only structural parameters (``κ`` for HKY, none for F81).
Per-data ``π`` lives on :class:`~ancestree.sites.BaseComposition` and is
passed to ``Q(pi=...)`` / ``transition_probs(t, pi=...)`` at call time;
``BaseComposition.from_polymorphic_sites`` is the empirical entry point.
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import (
    BaseComposition,
    F81,
    FixedTreeInference,
    HKY,
    JC69,
    OutgroupLadderTree,
    STATES,
    Site,
)

from testing._helpers import no_counts as _no_counts


# ---------------------------------------------------------- F81 rate matrix


class TestF81RateMatrix:
    def test_q_rows_sum_to_zero(self):
        Q = F81().Q(pi=(0.3, 0.2, 0.2, 0.3))
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-14)

    def test_pi_is_stationary(self):
        pi = np.array([0.3, 0.2, 0.2, 0.3])
        Q = F81().Q(pi=pi)
        np.testing.assert_allclose(pi @ Q, 0.0, atol=1e-14)

    def test_off_diagonals_proportional_to_pi(self):
        """``Q[i,j] / Q[k,j]`` is constant across ``i,k`` (depends only on ``j``)."""
        pi = (0.4, 0.1, 0.2, 0.3)
        Q = F81().Q(pi=pi)
        for j in range(4):
            col = [Q[i, j] for i in range(4) if i != j]
            np.testing.assert_allclose(col, col[0], atol=1e-14)

    def test_transition_probs_rows_sum_to_one(self):
        P = F81().transition_probs(0.3, pi=(0.4, 0.1, 0.2, 0.3))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)

    def test_closed_form_matches_expm(self):
        """Custom closed-form ``P(t)`` matches ``scipy.linalg.expm(Q t)``."""
        import scipy.linalg
        pi = (0.4, 0.1, 0.2, 0.3)
        m = F81()
        for t in (0.01, 0.1, 0.5, 2.0):
            P = m.transition_probs(t, pi=pi)
            P_ref = scipy.linalg.expm(m.Q(pi=pi) * t)
            np.testing.assert_allclose(P, P_ref, atol=1e-12)

    def test_uniform_pi_matches_jc69(self):
        """F81 with uniform ``π`` is JC69."""
        Q_f81 = F81().Q()
        Q_jc = JC69().Q()
        np.testing.assert_allclose(Q_f81, Q_jc)

    def test_invalid_pi_raises(self):
        with pytest.raises(ValueError, match="length-4"):
            F81().Q(pi=(0.5, 0.5))
        with pytest.raises(ValueError, match="strictly positive"):
            F81().Q(pi=(0.0, 0.4, 0.3, 0.3))
        with pytest.raises(ValueError, match="must sum to 1"):
            F81().Q(pi=(0.3, 0.3, 0.3, 0.3))


# ---------------------------------------------------------- HKY rate matrix


class TestHKYRateMatrix:
    def test_q_rows_sum_to_zero(self):
        Q = HKY(kappa=3.0).Q(pi=(0.3, 0.2, 0.2, 0.3))
        np.testing.assert_allclose(Q.sum(axis=1), 0.0, atol=1e-14)

    def test_pi_is_stationary(self):
        pi = np.array([0.4, 0.1, 0.2, 0.3])
        Q = HKY(kappa=4.0).Q(pi=pi)
        np.testing.assert_allclose(pi @ Q, 0.0, atol=1e-14)

    def test_transitions_faster_than_transversions(self):
        Q = HKY(kappa=5.0).Q(pi=(0.25, 0.25, 0.25, 0.25))
        # A=0, C=1, G=2, T=3
        ag_transition = Q[0, 2]
        ac_transversion = Q[0, 1]
        assert ag_transition > ac_transversion

    def test_kappa_one_collapses_to_f81(self):
        pi = (0.3, 0.2, 0.2, 0.3)
        Q_hky = HKY(kappa=1.0).Q(pi=pi)
        Q_f81 = F81().Q(pi=pi)
        np.testing.assert_allclose(Q_hky, Q_f81, atol=1e-14)
        P_hky = HKY(kappa=1.0).transition_probs(0.3, pi=pi)
        P_f81 = F81().transition_probs(0.3, pi=pi)
        np.testing.assert_allclose(P_hky, P_f81, atol=1e-12)

    def test_expected_substitutions_per_unit_t_equals_one(self):
        """``Σ_i π_i (-Q_ii) = 1``: one unit of ``t`` is one substitution."""
        pi = np.array([0.4, 0.1, 0.2, 0.3])
        Q = HKY(kappa=3.0).Q(pi=pi)
        r = -np.diag(Q)
        np.testing.assert_allclose(float(pi @ r), 1.0, atol=1e-12)
        # The per-state escape rates spread around the mean of one.
        assert float(r.min()) < 1.0 < float(r.max())

    def test_transition_probs_rows_sum_to_one(self):
        P = HKY(kappa=2.5).transition_probs(0.3, pi=(0.4, 0.1, 0.2, 0.3))
        np.testing.assert_allclose(P.sum(axis=1), 1.0)

    def test_per_state_escape_proportional_to_msprime(self):
        """The per-state escape rates are msprime's up to the overall scale.

        msprime's ``HKY.transition_matrix`` gives ``-Q_ii = 1 - P[i, i]``
        under its own convention, which fixes ``max_i (-Q_ii) = 1``. The two
        rate matrices therefore differ by a single factor, here the expected
        substitutions per unit ``t`` that msprime's matrix implies,
        ``Σ_i π_i (1 - P[i, i])``.
        """
        import msprime
        pi = np.array([0.4, 0.1, 0.1, 0.4])
        kappa = 4.0
        Q = HKY(kappa=kappa).Q(pi=pi)
        ms = msprime.HKY(kappa=kappa, equilibrium_frequencies=list(pi))
        P = np.array(ms.transition_matrix)
        msprime_escape = 1.0 - np.diag(P)
        np.testing.assert_allclose(
            -np.diag(Q), msprime_escape / float(pi @ msprime_escape), atol=1e-10,
        )

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            HKY(kappa=0)
        with pytest.raises(ValueError):
            HKY(kappa=-1.0)


# ---------------------------------------------------------- BaseComposition empirical helpers


def _site_from_tip_alleles(tip_alleles: dict[str, str], pos: int = 1) -> Site:
    return Site(
        chrom="1", pos=pos,
        alleles=tuple(sorted(set(tip_alleles.values()))),
        tip_alleles=tip_alleles,
    )


class TestEmpiricalPi:
    def test_recovers_handcrafted_counts(self):
        # Build sites whose tip-allele counts give pi=(0.4, 0.1, 0.2, 0.3).
        # 40 A, 10 C, 20 G, 30 T → total 100.
        tips: dict[str, str] = {}
        for i in range(40):
            tips[f"a{i}"] = "A"
        for i in range(10):
            tips[f"c{i}"] = "C"
        for i in range(20):
            tips[f"g{i}"] = "G"
        for i in range(30):
            tips[f"t{i}"] = "T"
        site = _site_from_tip_alleles(tips)
        bc = BaseComposition.from_polymorphic_sites([site])
        np.testing.assert_allclose(bc.pi, [0.4, 0.1, 0.2, 0.3], atol=1e-12)

    def test_empirical_pi_floors_zero_entries(self):
        """A base unobserved across every tip is floored, not zero."""
        site = _site_from_tip_alleles({"a1": "A", "a2": "A", "g1": "G"})
        bc = BaseComposition.from_polymorphic_sites([site])
        assert np.all(bc.pi > 0)
        # Floored entries are tiny but real. Sums to 1.
        np.testing.assert_allclose(bc.pi.sum(), 1.0)

    def test_empirical_pi_skips_missing(self):
        site = _site_from_tip_alleles({"a1": "A", "n1": "N", "g1": "G"})  # type: ignore[arg-type]
        bc = BaseComposition.from_polymorphic_sites([site])
        assert bc.pi[STATES.index("A")] > bc.pi[STATES.index("C")]


class TestEmpiricalKappa:
    def test_pure_transitions_pegs_to_upper(self):
        sites = [
            _site_from_tip_alleles({"x": "A", "y": "G"}),  # transition
            _site_from_tip_alleles({"x": "C", "y": "T"}),  # transition
        ]
        bc = BaseComposition.from_polymorphic_sites(sites)
        # 0 transversions → empirical formula degenerates. Falls back to
        # the upper-bound default (50.0).
        assert bc.kappa_estimate == 50.0

    def test_pure_transversions_pegs_to_lower(self):
        sites = [
            _site_from_tip_alleles({"x": "A", "y": "C"}),
            _site_from_tip_alleles({"x": "A", "y": "T"}),
        ]
        bc = BaseComposition.from_polymorphic_sites(sites)
        # 2·0/2 = 0 → clamped to the kappa lower bound (0.1).
        assert bc.kappa_estimate == 0.1

    def test_one_to_one_ratio(self):
        # 5 Ts (A↔G) + 10 Tv (5×A↔C, 5×A↔T) → κ̂ = 2·5/10 = 1
        sites = (
            [_site_from_tip_alleles({"x": "A", "y": "G"})] * 5
            + [_site_from_tip_alleles({"x": "A", "y": "C"})] * 5
            + [_site_from_tip_alleles({"x": "A", "y": "T"})] * 5
        )
        bc = BaseComposition.from_polymorphic_sites(sites)
        assert bc.kappa_estimate == 1.0

    def test_three_to_one_ratio(self):
        # 9 Ts + 6 Tv → κ̂ = 2·9/6 = 3
        sites = (
            [_site_from_tip_alleles({"x": "A", "y": "G"})] * 9
            + [_site_from_tip_alleles({"x": "A", "y": "C"})] * 6
        )
        bc = BaseComposition.from_polymorphic_sites(sites)
        assert bc.kappa_estimate == pytest.approx(3.0)


class TestCalibrationCap:
    """``max_sites`` lazily truncates the input so streaming sources stay cheap."""

    def _shuffled_dataset(self, n: int):
        # 80% A/A monomorphic, 15% A/G transitions, 5% A/C transversions,
        # randomly interleaved so file-order truncation gets a representative
        # slice. Both classes are present so kappa_estimate takes its
        # 2*n_ts/n_tv branch: with transitions alone it returns the 50.0 upper
        # bound whatever the data, and the comparison below would hold trivially.
        import random
        sites = (
            [_site_from_tip_alleles({"x": "A", "y": "A"})] * (n * 4 // 5)
            + [_site_from_tip_alleles({"x": "A", "y": "G"})] * (n * 3 // 20)
            + [_site_from_tip_alleles({"x": "A", "y": "C"})] * (n // 20)
        )
        random.Random(0).shuffle(sites)
        return sites

    def test_max_sites_truncates_when_dataset_exceeds_cap(self):
        sites = self._shuffled_dataset(20_000)
        full = BaseComposition.from_polymorphic_sites(sites)
        capped = BaseComposition.from_polymorphic_sites(sites, max_sites=1_000)
        # Guard the kappa assertion against comparing a clamp to itself.
        assert 0.1 < full.kappa_estimate < 50.0
        assert 0.1 < capped.kappa_estimate < 50.0
        # Same shuffled distribution → both estimates close, but capped
        # consumed only 5% of the data.
        np.testing.assert_allclose(capped.pi, full.pi, atol=0.03)
        assert capped.kappa_estimate == pytest.approx(full.kappa_estimate, rel=0.4)

    def test_max_sites_consumes_lazily_from_generator(self):
        """Generator inputs only have the first ``max_sites`` items pulled."""
        all_sites = self._shuffled_dataset(20_000)
        consumed = [0]
        def gen():
            for s in all_sites:
                consumed[0] += 1
                yield s
        BaseComposition.from_polymorphic_sites(gen(), max_sites=500)
        assert consumed[0] == 500  # exact: only 500 items pulled


# ---------------------------------------------------------- kernel integration


class TestKernelIntegration:
    """End-to-end posterior differs between models in the expected direction."""

    def test_transition_bias_amplifies_transition_partner(self):
        """At a polymorphic outgroup pair (A↔G), HKY with high κ puts more
        posterior mass on the transition partner than JC69 does."""
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))  # K1, K2 (Kdeep/2 each)
        # Construct a single A-vs-G outgroup-pair site.
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"o1": "A", "o2": "G"},
        )
        # JC69 reference posterior.
        inf_jc = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        post_jc = next(inf_jc.infer())[1]
        # HKY with strong transition bias.
        inf_hky = FixedTreeInference(
            [site], HKY(kappa=10.0), _no_counts(), tree=tree,
            fit_required=False,
        )
        post_hky = next(inf_hky.infer())[1]
        # Both A and G are observed alleles. With κ>>1, A↔G is a single
        # transition (likely), while reaching A or G from C/T requires a
        # transversion (suppressed). So HKY should concentrate posterior
        # on {A, G} more than JC69 does.
        ag_mass_jc = post_jc["A"] + post_jc["G"]
        ag_mass_hky = post_hky["A"] + post_hky["G"]
        assert ag_mass_hky > ag_mass_jc + 1e-3

    def test_f81_uniform_pi_matches_jc69_posterior(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.075, 0.075]))  # K1, K2 (Kdeep/2 each)
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"o1": "A", "o2": "C"},
        )
        inf_jc = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        inf_f81 = FixedTreeInference(
            [site], F81(), _no_counts(), tree=tree,
            fit_required=False,
        )
        post_jc = next(inf_jc.infer())[1]
        post_f81 = next(inf_f81.infer())[1]
        for s in STATES:
            np.testing.assert_allclose(post_jc[s], post_f81[s], atol=1e-10)


# ---------------------------------------------------------- free-parameter protocol


class TestFreeParams:
    def test_f81_no_free_params(self):
        assert F81().free_params == {}

    def test_hky_default_no_free_params(self):
        assert HKY().free_params == {}

    def test_hky_fit_kappa_exposes_kappa(self):
        free = HKY(kappa=2.5, fit_kappa=True).free_params
        assert list(free) == ["kappa"]
        initial, (lo, hi) = free["kappa"]
        assert initial == 2.5
        assert 0 < lo < hi

    def test_set_free_params_updates_kappa(self):
        m = HKY(kappa=2.0, fit_kappa=True)
        m.set_free_params({"kappa": 5.0})
        assert m.kappa == 5.0


# ---------------------------------------------------------- MLE recovery


def _simulate_outgroup_sites(
    tree, model, n_sites: int, *, pi: np.ndarray, seed: int,
) -> list[Site]:
    """Top-down sample tip alleles from ``model`` on ``tree`` (stationary root).

    Branch-length scaling lives on the tree. ``pi`` is the per-data base
    composition fed into ``model.Q(pi=...)`` / ``transition_probs``.
    """
    rng = np.random.default_rng(seed)
    P_by_node = {
        node: model.transition_probs(tree.branch_length(node), pi=pi)
        for node in tree.postorder() if node != tree.root
    }
    sites: list[Site] = []
    for i in range(n_sites):
        state_at: dict[int, int] = {tree.root: int(rng.choice(4, p=pi))}
        stack = [tree.root]
        while stack:
            parent = stack.pop()
            for child in tree.children(parent):
                state_at[child] = int(
                    rng.choice(4, p=P_by_node[child][state_at[parent]])
                )
                stack.append(child)
        tip_alleles = {
            sid: STATES[state_at[tree.tip_for_sample(sid)]]
            for sid in tree.outgroup_samples
        }
        sites.append(Site(
            chrom="1", pos=i + 1,
            alleles=tuple(sorted(set(tip_alleles.values()))),
            tip_alleles=tip_alleles,
        ))
    return sites


class TestHKYMLERecovery:
    def test_kappa_mle_recovers_within_25pct(self):
        ingroup = [f"i{i}" for i in range(10)]
        tree = OutgroupLadderTree(ingroup, ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.10, 0.10]))  # K1, K2 (Kdeep/2 each)
        true_kappa = 5.0
        pi = np.full(4, 0.25)
        sites = _simulate_outgroup_sites(
            tree, HKY(kappa=true_kappa), 800, pi=pi, seed=17,
        )
        tree.set_params(np.full(tree.n_params, 0.1))  # K1, K2
        model = HKY(kappa=2.0, fit_kappa=True)
        bc = _no_counts()
        inf = FixedTreeInference(
            sites, model, bc, tree=tree,
            n_target_sites=len(sites),
            n_starts=2, parallelize=False,
        )
        params = inf.fit()
        assert "kappa" in params
        kappa_mle = inf.model_params_mle["kappa"]
        rel_err = abs(kappa_mle - true_kappa) / true_kappa
        assert rel_err < 0.25, (
            f"κ recovery off: true={true_kappa}, mle={kappa_mle}, "
            f"rel_err={rel_err:.3f}"
        )


# ---------------------------------------------------------- StationaryPrior wiring


class TestStationaryPriorWithNonUniform:
    """StationaryPrior reads pi from BaseComposition. Verify F81/HKY plug in."""

    def test_stationary_prior_picks_up_bc_pi(self):
        from ancestree.priors import StationaryPrior
        pi = np.array([0.4, 0.1, 0.2, 0.3])
        # Build a BaseComposition whose pi equals pi (counts proportional).
        bc = BaseComposition.from_counts(A=400, C=100, G=200, T=300)
        np.testing.assert_allclose(bc.pi, pi, atol=1e-12)
        prior = StationaryPrior(F81(), bc)
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"o1": "A", "o2": "G"},
        )
        log_prior = prior.log_probs([site])[0]
        np.testing.assert_allclose(np.exp(log_prior), pi, atol=1e-12)
