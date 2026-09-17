"""Tests for FixedTreeInference."""
import gc
import logging
import multiprocessing as mp
import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
import ancestree as anc
import ancestree.inference as inference_mod

from ancestree import (
    AdaptiveIngroupWeight,
    BaseComposition,
    FixedTreeInference,
    HKY,
    OutgroupLadderTree,
    JC69,
    STATES,
    Site,
    TskitLocalTree,
)
from ancestree.priors import KingmanIngroupWeight
from ancestree.settings import Settings
from ancestree.sites import SiteSource
from testing._helpers import no_counts as _no_counts


def _simulate_sites(
    tree: OutgroupLadderTree,
    model: JC69,
    n_sites: int,
    *,
    seed: int,
) -> list[Site]:
    """Sample N polymorphic-or-monomorphic Sites from the tree.

    For each site: draw a root state uniformly, then top-down sample each
    branch's child state from ``model.transition_probs(branch_length)``.
    Record the resulting outgroup tip alleles into a :class:`Site`.
    """
    rng = np.random.default_rng(seed)

    # Precompute per-branch transition matrices (root has no branch).
    P_by_node = {
        node: model.transition_probs(tree.branch_length(node))
        for node in tree.postorder() if node != tree.root
    }

    # Top-down walk via a stack (no recursion, preorder by parent then children).
    sites: list[Site] = []
    for i in range(n_sites):
        state_at: dict[int, int] = {tree.root: int(rng.integers(0, 4))}
        stack = [tree.root]
        while stack:
            parent = stack.pop()
            for child in tree.children(parent):
                state_at[child] = int(rng.choice(4, p=P_by_node[child][state_at[parent]]))
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


# ------------------------------------------------------------------- construction


class TestConstruction:
    def test_base_composition_defaults_to_no_counts(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        inf = FixedTreeInference([], JC69(), None, tree=tree, fit_required=False)
        assert inf.base_composition.n_total == 0

    def test_no_counts_satisfies_require(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        inf = FixedTreeInference(
            [], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        assert inf.base_composition.n_total == 0

    def test_no_counts_without_target_raises_in_fit_mode(self):
        """no_counts() + fit_required=True + no n_target_sites → ValueError."""
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with pytest.raises(ValueError, match="monomorphic-site calibration"):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
            )

    def test_bad_initial_rates_shape(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])  # n_params=3 (K0, K1, K2)
        with pytest.raises(ValueError, match="length-3"):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
                initial_rates=np.array([1.0, 2.0, 3.0, 4.0]),
                fit_required=False,
            )


# ----------------------------------------------------------- fit-before-infer guard


class TestFitBeforeInfer:
    def test_infer_before_fit_raises(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"o1": "A", "o2": "A"},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            n_target_sites=1,
        )
        with pytest.raises(RuntimeError, match="before fit"):
            list(inf.infer())

    def test_accessors_none_before_fit(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        inf = FixedTreeInference(
            [], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        assert inf.params_mle is None
        assert inf.log_likelihood_mle is None
        assert inf.outgroup_divergence_mle is None

    def _unfitted(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        sites = [
            Site(chrom="1", pos=i + 1, alleles=("A", "T"),
                 tip_alleles={"i1": "T", "o1": "A", "o2": "A"})
            for i in range(3)
        ]
        return FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree, n_target_sites=3,
        )

    def test_summary_auto_fits_unfitted_instance(self):
        """summary() documents that it triggers fitting, so it must
        auto-fit rather than raise on an unfitted, fit_required instance."""
        inf = self._unfitted()
        assert inf.params_mle is None
        summ = inf.summary()  # must not raise
        assert summ.n_sites == 3
        assert inf.params_mle is not None  # fit actually ran

    def test_grade_auto_fits_unfitted_instance(self):
        inf = self._unfitted()
        _, _, n = inf.grade({1: "A", 2: "A", 3: "A"})  # must not raise
        assert n == 3
        assert inf.params_mle is not None


# -------------------------------------------------------------------- fit recovery


class TestFitRecovery2Outgroups:
    """2-outgroup fit: the sum ``K_1 + K_2`` (the O_1–O_2 distance through n_1)
    should recover.

    Under the EST-SFS-style 2n−1 parameterisation, ``K_1`` and ``K_2`` are
    both exposed as free params, but with no ingroup prior the per-site
    likelihood depends only on ``K_1+K_2`` (Chapman–Kolmogorov:
    ``P(O_1, O_2) = (1/4) P_{K_1+K_2}(O_1, O_2)``), so the *individual*
    branch lengths are unidentifiable, the optimizer typically parks one
    of ``K_1, K_2`` at the lower bound and absorbs the rest into the other.
    Only the sum is recoverable here. Identifying the individual branches
    requires an informative ingroup prior or ≥3 outgroups.
    """

    def _setup(self, true_sum: float, n_sites: int, seed: int):
        ingroup = ["i0", "i1"]
        outgroup = ["o1", "o2"]
        model = JC69()

        # Build with truth, simulate, then reset to a neutral start.
        # Split the K_1+K_2 sum symmetrically. Only the sum is identifiable
        # under the uniform-marginal stationary prior used here. K_0 is
        # left at 0 in the truth setup (its value cannot move the per-tip
        # probabilities under reversibility + stationary prior).
        tree = OutgroupLadderTree(ingroup, outgroup)
        tree.set_params(np.array([0.0, true_sum / 2, true_sum / 2]))
        sites = _simulate_sites(tree, model, n_sites, seed=seed)
        tree.set_params(np.full(tree.n_params, 0.05))

        inf = FixedTreeInference(
            sites, model, _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        return inf

    def test_outgroup_pair_distance_recovered(self):
        """``K_1 + K_2`` (the O_1–O_2 distance through n_1) recovers within 10%."""
        true_sum = 0.18
        inf = self._setup(true_sum, n_sites=4000, seed=42)
        params = inf.fit()
        fitted_sum = params["K1"] + params["K2"]
        rel_err = abs(fitted_sum - true_sum) / true_sum
        assert rel_err < 0.10, (
            f"truth K_1+K_2={true_sum:.4f}, "
            f"fitted K_1+K_2={fitted_sum:.4f}, rel_err={rel_err:.3f}"
        )

    def test_fit_populates_accessors(self):
        inf = self._setup(true_sum=0.18, n_sites=2000, seed=7)
        params = inf.fit()
        assert inf.params_mle == params
        assert inf.log_likelihood_mle is not None
        assert np.isfinite(inf.log_likelihood_mle)
        assert inf.outgroup_divergence_mle is not None
        assert inf.outgroup_divergence_mle.shape == (2,)


# ---------------------------------------------------------- HKY fit recovery


def _simulate_sites_with_pi(
    tree: OutgroupLadderTree,
    model,
    n_sites: int,
    *,
    pi,
    seed: int,
) -> list[Site]:
    """Simulator that respects a non-uniform stationary ``pi``.

    Root state drawn from ``pi`` (the equilibrium). Branches propagate via
    ``model.transition_probs(branch_length, pi=pi)``. Mirrors
    :func:`_simulate_sites` but passes ``pi`` to the kernel so HKY/F81
    samples match what the MLE will be fitting under.
    """
    rng = np.random.default_rng(seed)
    pi_arr = np.asarray(pi, dtype=float)
    P_by_node = {
        node: model.transition_probs(tree.branch_length(node), pi=pi_arr)
        for node in tree.postorder() if node != tree.root
    }
    sites: list[Site] = []
    for i in range(n_sites):
        state_at: dict[int, int] = {tree.root: int(rng.choice(4, p=pi_arr))}
        stack = [tree.root]
        while stack:
            parent = stack.pop()
            for child in tree.children(parent):
                state_at[child] = int(rng.choice(4, p=P_by_node[child][state_at[parent]]))
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


class TestFitRecoveryHKY3Outgroups:
    """3-outgroup HKY fit: every branch rate ``K_1..K_4`` AND the
    transition/transversion ratio ``kappa`` should recover from a
    sufficiently large simulated dataset.

    With ``n_out >= 3`` the OutgroupLadderTree's branch rates are
    individually identifiable (the closer outgroups break the
    uniform-marginal at the deepest sister pair's parent). Tests that
    the new pi-as-BaseComposition / fit_kappa=True / no-mu API
    recovers truth at the same fidelity as the historical JC69 fit
    on a much simpler tree.
    """

    def _setup(
        self,
        true_K: "tuple[float, float, float, float]",
        *,
        true_kappa: float,
        pi: "tuple[float, float, float, float]",
        n_sites: int,
        seed: int,
    ):
        ingroup = ["i0", "i1"]
        outgroup = ["o1", "o2", "o3"]
        truth_model = HKY(kappa=true_kappa)

        tree = OutgroupLadderTree(ingroup, outgroup)
        # Set ladder with K_0 = 0 then K_1..K_4 from the test's true_K.
        tree.set_params(np.array([0.0] + list(true_K)))
        sites = _simulate_sites_with_pi(
            tree, truth_model, n_sites, pi=pi, seed=seed,
        )
        # Reset branch rates so the MLE starts from a neutral seed.
        tree.set_params(np.full(tree.n_params, 0.05))

        # Pass empirical-from-data pi via the BaseComposition, supply
        # ``fit_kappa=True`` so kappa enters the joint L-BFGS-B fit.
        # ``n_target_sites=len(sites)`` declares the genome length: every
        # simulated position is already in `sites` (polymorphic + the
        # simulator's incidentally-monomorphic patterns), so no
        # additional monomorphic weight is needed.
        # Pin K_0 = 0 via fixed_params so the test's per-branch K_i
        # recovery is not sensitive to where the optimiser lands on the
        # K_0 / terminal-K reversibility ridge (the library leaves K_0
        # free by default for parity with EST-SFS; here the test covers the
        # canonical-parameterisation recovery).
        bc = BaseComposition.from_polymorphic_sites(sites)
        fit_model = HKY(kappa=2.0, fit_kappa=True)  # neutral init
        inf = FixedTreeInference(
            sites, fit_model, base_composition=bc, tree=tree,
            n_target_sites=len(sites),
            fixed_params={"K0": 0.0},
        )
        return inf, bc

    def test_branch_rates_and_kappa_recover(self):
        true_K = (0.04, 0.03, 0.05, 0.05)  # K_1..K_4 (K_0 left at 0 in setup)
        true_kappa = 4.0
        pi = (0.30, 0.20, 0.20, 0.30)  # AT-rich
        # Path lengths under the simulator (K_0 = 0):
        #   d_1 = K_1 = 0.04
        #   d_2 = K_2 + K_3 = 0.08
        #   d_3 = K_2 + K_4 = 0.08
        true_d = (0.04, 0.08, 0.08)
        inf, bc = self._setup(
            true_K, true_kappa=true_kappa, pi=pi, n_sites=8000, seed=42,
        )
        params = inf.fit()

        # Path lengths I→O_k are identifiable. Per-branch K_i sit on a
        # K_0 / terminal-K ridge (the marginal likelihood is invariant
        # to redistribution along that ridge under reversibility).
        # Verify the identifiable quantity, path lengths, rather than
        # asserting on individual K_i, which is sensitive to where the
        # optimiser lands on the ridge.
        fitted_d = inf.outgroup_divergence_mle
        for k, (true_v, fitted) in enumerate(zip(true_d, fitted_d), start=1):
            rel_err = abs(fitted - true_v) / true_v
            assert rel_err < 0.30, (
                f"d_{k}: truth={true_v:.4f}, fitted={fitted:.4f}, "
                f"rel_err={rel_err:.3f}"
            )

        # kappa: empirical Ts/Tv estimate (from bc.kappa_estimate) plus
        # L-BFGS-B refinement should land within 30% of truth.
        fitted_kappa = params["kappa"]
        rel_err_k = abs(fitted_kappa - true_kappa) / true_kappa
        assert rel_err_k < 0.30, (
            f"kappa: truth={true_kappa}, empirical={bc.kappa_estimate:.3f}, "
            f"fitted={fitted_kappa:.3f}, rel_err={rel_err_k:.3f}"
        )

    def test_empirical_pi_close_to_truth(self):
        """``BaseComposition.from_polymorphic_sites`` recovers the simulator's
        stationary within sampling noise on 8 000 sites."""
        true_K = (0.04, 0.03, 0.05, 0.05)
        pi = (0.30, 0.20, 0.20, 0.30)
        _, bc = self._setup(
            true_K, true_kappa=4.0, pi=pi, n_sites=8000, seed=7,
        )
        np.testing.assert_allclose(bc.pi, pi, atol=0.04)


# -------------------------------------------------------------------- infer shape


class TestInfer:
    def test_posteriors_sum_to_one(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 100, seed=11)
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        inf.fit()
        for _, posterior in inf.infer():
            assert np.isclose(posterior.values.sum(), 1.0, atol=1e-9)
            assert (posterior.values >= 0).all()

    def test_infer_emits_one_per_site(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 50, seed=3)
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        inf.fit()
        emitted = list(inf.infer())
        assert len(emitted) == 50
        for in_site, (out_site, _) in zip(sites, emitted):
            assert out_site is in_site


# ------------------------------------------------------- non-monotone-divergence


class TestNonMonotoneDivergenceWarning:
    # The "warns when out-of-order" case is hard to construct in a
    # closed-form test: for n=2 the topology is symmetric and the
    # optimizer can always swap labels to land in the monotone half-
    # plane, suppressing the warning. The warning will fire in practice
    # for n >= 3 (asymmetric ladder) or whenever the data + bounds
    # genuinely pin the fit into a non-monotone configuration. The
    # straightforward unit test below verifies the silent path.

    def test_no_warning_when_order_matches(self, caplog):
        ingroup = [f"i{i}" for i in range(8)]
        outgroup = ["o_close", "o_far"]
        truth_tree = OutgroupLadderTree(ingroup, outgroup)
        truth_tree.set_params(np.array([0.0, 0.175, 0.175]))
        sites = _simulate_sites(truth_tree, JC69(), 1000, seed=51)
        tree = OutgroupLadderTree(ingroup, outgroup)
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            n_starts=2, parallelize=False,
        )
        with caplog.at_level(logging.WARNING):
            inf.fit()
        assert "monotonically increasing" not in caplog.text


# ---------------------------------------------------- redundant-outgroup warning


class TestRedundantOutgroupWarning:
    def test_warns_on_near_identical_outgroups(self, caplog):
        sites = [Site(
            chrom="1", pos=i + 1, alleles=("A",),
            tip_alleles={"o1": "A", "o2": "A"},
        ) for i in range(200)]
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with caplog.at_level(logging.WARNING):
            FixedTreeInference(
                sites, JC69(), _no_counts(), tree=tree,
                n_target_sites=len(sites),
                n_starts=1, parallelize=False,
            )
        assert "near-redundant" in caplog.text

    def test_no_warning_on_diverged_outgroups(self, caplog):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.10, 0.10]))
        sites = _simulate_sites(tree, JC69(), 500, seed=99)
        with caplog.at_level(logging.WARNING):
            FixedTreeInference(
                sites, JC69(), _no_counts(), tree=tree,
                n_target_sites=len(sites),
                n_starts=1, parallelize=False,
            )
        assert "near-redundant" not in caplog.text

    def test_threshold_zero_disables(self, caplog):
        sites = [Site(
            chrom="1", pos=i + 1, alleles=("A",),
            tip_alleles={"o1": "A", "o2": "A"},
        ) for i in range(100)]
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with caplog.at_level(logging.WARNING):
            FixedTreeInference(
                sites, JC69(), _no_counts(), tree=tree,
                n_target_sites=len(sites),
                outgroup_similarity_threshold=0.0,
                n_starts=1, parallelize=False,
            )
        assert "near-redundant" not in caplog.text


# ----------------------------------------------------------- auto outgroup order


def _simulate_sites_with_ingroup(
    tree: OutgroupLadderTree,
    model: JC69,
    n_sites: int,
    ingroup_branch_length: float,
    *,
    seed: int,
) -> list[Site]:
    """Same as `_simulate_sites` but also draws an ingroup haplotype per site
    by mutating the root state down a single ``ingroup_branch_length`` branch.
    Ingroup tip alleles are stored under the names in ``tree.ingroup_samples``.
    """
    rng = np.random.default_rng(seed)
    P_by_node = {
        node: model.transition_probs(tree.branch_length(node))
        for node in tree.postorder() if node != tree.root
    }
    P_ingroup = model.transition_probs(ingroup_branch_length)
    sites: list[Site] = []
    for i in range(n_sites):
        root_state = int(rng.integers(0, 4))
        state_at: dict[int, int] = {tree.root: root_state}
        stack = [tree.root]
        while stack:
            parent = stack.pop()
            for child in tree.children(parent):
                state_at[child] = int(rng.choice(4, p=P_by_node[child][state_at[parent]]))
                stack.append(child)
        tip_alleles: dict[str, str | None] = {
            sid: STATES[state_at[tree.tip_for_sample(sid)]]
            for sid in tree.outgroup_samples
        }
        for sid in tree.ingroup_samples:
            ingroup_state = int(rng.choice(4, p=P_ingroup[root_state]))
            tip_alleles[sid] = STATES[ingroup_state]
        sites.append(Site(
            chrom="1", pos=i + 1,
            alleles=tuple(sorted(set(v for v in tip_alleles.values() if v))),
            tip_alleles=tip_alleles,
        ))
    return sites


class TestAutoOutgroupOrder:
    def test_orders_3outgroups_closest_first(self):
        """3-outgroup truth: O_close (K=0.02), O_mid (K=0.10), O_far (K=0.30).
        Supply in scrambled order. Auto-ordering should put closest first."""
        ingroup = [f"i{i}" for i in range(10)]
        unordered = ["O_far", "O_close", "O_mid"]
        # Build a "truth tree" in the canonical closest-first order to sim.
        truth_tree = OutgroupLadderTree(ingroup, ["O_close", "O_mid", "O_far"])
        # External params for n=3: (K_1, K_2, K_3, K_4) map to
        # n_1→O_close, n_1→n_2, n_2→O_mid, n_2→O_far.
        truth_tree.set_params(np.array([0.0, 0.02, 0.03, 0.10, 0.30]))
        sites = _simulate_sites_with_ingroup(
            truth_tree, JC69(), n_sites=2000,
            ingroup_branch_length=0.005, seed=17,
        )
        tree = OutgroupLadderTree.with_outgroup_order_from_sites(
            ingroup, unordered, sites,
        )
        assert list(tree.outgroup_samples) == ["O_close", "O_mid", "O_far"]

    def test_raises_when_no_ingroup_tips(self):
        sites = [Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"o1": "A", "o2": "A"},  # no ingroup keys
        )]
        with pytest.raises(ValueError, match="no Site carries ingroup"):
            OutgroupLadderTree.with_outgroup_order_from_sites(
                ["i1"], ["o1", "o2"], sites,
            )

    def test_raises_when_outgroup_unobserved(self):
        sites = [Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"i1": "A", "o1": "A"},  # o2 missing
        )]
        with pytest.raises(ValueError, match="no sites jointly observed"):
            OutgroupLadderTree.with_outgroup_order_from_sites(
                ["i1"], ["o1", "o2"], sites,
            )


# ----------------------------------------------------------------- fit logging


# --------------------------------------------------------------- fixed params


class TestFixedParams:
    def test_unknown_name_raises(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with pytest.raises(ValueError, match="not in tree.param_names"):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
                fixed_params={"K99": 0.5},
                fit_required=False,
            )

    def test_all_fixed_raises(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])  # n_params=3 (K0, K1, K2)
        with pytest.raises(ValueError, match="All parameters are fixed"):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
                fixed_params={"K0": 0.0, "K1": 0.15, "K2": 0.15},
                fit_required=False,
            )

    def test_fixed_value_held_through_fit(self):
        """A fixed param is exactly its supplied value in the MLE dict. The
        free params still optimise and recover truth (within tolerance).

        Uses n=3 outgroups so the ``K0, K1, K2, K3, K4`` ladder is well-
        identifiable with one branch held fixed and the rest free.
        """
        tree = OutgroupLadderTree(["i1"], ["o1", "o2", "o3"])
        true_K0, true_K1, true_K2, true_K3, true_K4 = (
            0.0, 0.005, 0.012, 0.018, 0.025,
        )
        tree.set_params(np.array([true_K0, true_K1, true_K2, true_K3, true_K4]))
        sites = _simulate_sites(tree, JC69(), 2000, seed=33)
        tree.set_params(np.full(tree.n_params, 0.05))
        # Fix K1 at truth so K2's recovery is meaningful.
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            fixed_params={"K1": true_K1},
        )
        params = inf.fit()
        assert params["K1"] == pytest.approx(true_K1, rel=1e-12)
        assert abs(params["K2"] - true_K2) / true_K2 < 0.30


# -------------------------------------------------------- multistart + parallel


class TestMultistart:
    def test_n_starts_must_be_positive(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with pytest.raises(ValueError, match="n_starts must be >= 1"):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
                n_starts=0,
                fit_required=False,
            )

    def test_multistart_finds_at_least_as_good_as_single(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 600, seed=51)
        tree.set_params(np.full(tree.n_params, 0.1))  # K1, K2

        single = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            n_starts=1, seed=1,
        )
        single.fit()
        log_L_single = single.log_likelihood_mle

        # Fresh tree (mutated by single.fit()) before multistart.
        tree2 = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree2.set_params(np.full(tree2.n_params, 0.1))  # K1, K2
        multi = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree2,
            n_target_sites=len(sites),
            n_starts=5, seed=1,
        )
        multi.fit()

        # The first start matches the single-start, so multi.log_L >= single.log_L
        assert multi.log_likelihood_mle >= log_L_single - 1e-6

    def test_parallel_matches_sequential(self):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 400, seed=77)
        tree.set_params(np.full(tree.n_params, 0.1))  # K1, K2

        seq = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            n_starts=3, seed=5, parallelize=False,
        )
        seq.fit()
        log_L_seq = seq.log_likelihood_mle

        tree2 = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree2.set_params(np.full(tree2.n_params, 0.1))  # K1, K2
        par = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree2,
            n_target_sites=len(sites),
            n_starts=3, seed=5, parallelize=True, n_workers=2,
        )
        par.fit()
        assert par.log_likelihood_mle == pytest.approx(log_L_seq, rel=1e-9)

    def test_same_seed_run_to_run_reproducible(self):
        """Two identical multistart fits give bit-identical MLEs.

        The extra starts are drawn from ``np.random.default_rng(seed)``, so a
        fixed seed must reproduce the fit exactly (sequential path, no parallel
        nondeterminism), guards against an accidental unseeded RNG.
        """
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 500, seed=63)

        def fit():
            t = OutgroupLadderTree(["i1"], ["o1", "o2"])
            t.set_params(np.full(t.n_params, 0.1))  # K1, K2
            inf = FixedTreeInference(
                sites, JC69(), _no_counts(), tree=t,
                n_target_sites=len(sites),
                n_starts=6, seed=17, parallelize=False,
            )
            inf.fit()
            return inf

        a, b = fit(), fit()
        assert a.log_likelihood_mle == b.log_likelihood_mle
        assert a.params_mle.keys() == b.params_mle.keys()
        for k in a.params_mle:
            assert a.params_mle[k] == b.params_mle[k], k


# ---------------------------------------------- alt substitution model (K2)


class TestSubstitutionModel:
    def test_k2_fit_smoke(self):
        from ancestree import K2
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        # Simulate under K2 (kappa=2), fit with K2, verify no crash and
        # MLE log-likelihood is finite. Strict recovery vs JC69 is not
        # the point here, exercising the model interface is.
        sites = _simulate_sites(tree, K2(kappa=2.0), 400, seed=88)
        tree.set_params(np.full(tree.n_params, 0.1))  # K1, K2
        inf = FixedTreeInference(
            sites, K2(kappa=2.0), _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        inf.fit()
        assert np.isfinite(inf.log_likelihood_mle)


class TestFitKappa:
    def test_jc69_has_no_free_params(self):
        """Default JC69 has no fittable internal params."""
        assert JC69().free_params == {}

    def test_k2_default_has_no_free_params(self):
        """K2 defaults to fixed kappa."""
        from ancestree import K2
        assert K2(kappa=2.0).free_params == {}

    def test_k2_fit_kappa_exposes_free_param(self):
        """K2(fit_kappa=True) advertises kappa via the protocol."""
        from ancestree import K2
        free = K2(kappa=2.5, fit_kappa=True).free_params
        assert list(free) == ["kappa"]
        initial, (lo, hi) = free["kappa"]
        assert initial == 2.5
        assert 0 < lo < hi

    def test_kappa_mle_in_params(self):
        from ancestree import K2
        ingroup = [f"i{i}" for i in range(10)]
        tree = OutgroupLadderTree(ingroup, ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, K2(kappa=4.0), 600, seed=12)
        tree.set_params(np.full(tree.n_params, 0.1))  # K1, K2
        model = K2(kappa=2.0, fit_kappa=True)
        inf = FixedTreeInference(
            sites, model, _no_counts(), tree=tree,
            n_target_sites=len(sites),
            n_starts=2, parallelize=False,
        )
        params = inf.fit()
        assert "kappa" in params
        assert inf.model_params_mle is not None
        assert "kappa" in inf.model_params_mle
        kappa_mle = inf.model_params_mle["kappa"]
        assert 0.1 <= kappa_mle <= 50.0
        # The model instance carries the MLE kappa
        assert model.kappa == kappa_mle


# ------------------------------------------------------------------- logging


class TestFitLogging:
    def test_logs_mle_summary_at_info(self, caplog):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.09, 0.09]))
        sites = _simulate_sites(tree, JC69(), 300, seed=23)
        # Reset before fit so the optimiser does not start at truth.
        tree.set_params(np.full(tree.n_params, 0.05))
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        with caplog.at_level(logging.INFO, logger="ancestree.inference"):
            inf.fit()
        text = caplog.text
        assert "L-BFGS-B" in text and "converged" in text
        assert "MLE branch rates" in text
        assert "Ingroup MRCA → outgroup divergences" in text
        assert "o1=" in text
        assert "o2=" in text

    def test_info_visible_by_default(self):
        """ancestree package logger carries an INFO StreamHandler so the
        post-fit summary is visible out of the box."""
        ancestree_logger = logging.getLogger("ancestree")
        assert ancestree_logger.level == logging.INFO
        assert any(isinstance(h, logging.StreamHandler) for h in ancestree_logger.handlers)

    def test_silenceable_via_setLevel(self, caplog):
        """Caller opts out: setLevel(WARNING) on the ancestree logger."""
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        sites = _simulate_sites(tree, JC69(), 200, seed=5)
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
        )
        prior_level = logging.getLogger("ancestree").level
        try:
            logging.getLogger("ancestree").setLevel(logging.WARNING)
            with caplog.at_level(logging.WARNING, logger="ancestree.inference"):
                inf.fit()
            info_records = [r for r in caplog.records if r.levelno == logging.INFO]
            assert info_records == []
        finally:
            logging.getLogger("ancestree").setLevel(prior_level)


# ------------------------------------------------------ the deep readout


class TestDeepReadout:
    """Monoallelic-ingroup sites read at the ladder's deepest join.

    The ladder's own root is the ingroup MRCA, the derived side of the single
    informative substitution for a fixed-derived ingroup, so reading there
    reports the ingroup's allele back. Moving the focal node to the deepest
    join recovers the ancestral allele. Polyallelic sites are untouched.
    """

    @staticmethod
    def _inf(site, focal=None):
        from ancestree.focal import FocalNode
        from ancestree.priors import NoIngroupWeight
        focal = focal or FocalNode("ingroup_mrca", fraction=1.0)
        tree = OutgroupLadderTree(["i1", "i2", "i3"], ["o1", "o2", "o3"])
        # External params for n=3: (K1, K2, K3, K4), K1 = O_1's branch,
        # K2 = ladder backbone n_1→n_2, K3 = n_2→O_2, K4 = n_2→O_3.
        tree.set_params(np.array([0.0, 0.004, 0.008, 0.021, 0.021]))
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree, focal=focal,
            ingroup_weight=NoIngroupWeight(), fit_required=False,
            progress=False,
        )
        inf._params_mle = tree.param_vector  # simulate a completed fit
        return inf

    def test_fixed_derived_recovers_via_deep_outgroups(self):
        """Ingroup + closest outgroup share derived G; the two deeper outgroups = A.

        Read at the deepest join, O2 and O3 recover the ancestral A despite
        the ingroup and the closest outgroup carrying G.
        """
        site = Site(chrom="1", pos=1, alleles=("A", "G"),
                    tip_alleles={"i1": "G", "i2": "G", "i3": "G",
                                 "o1": "G", "o2": "A", "o3": "A"})
        (post,) = [p for _, p in self._inf(site).infer()]
        assert post.map_allele == "A"

    def test_polyallelic_ingroup_untouched(self):
        """A polyallelic ingroup leaves the collapsed tip missing, so it
        marginalises out and the outgroups alone answer the site.

        Ingroup polymorphic for ``A``/``G`` with all outgroups ``A``, the
        ancestral allele is unambiguously ``A``.
        """
        site = Site(chrom="1", pos=1, alleles=("A", "G"),
                    tip_alleles={"i1": "A", "i2": "A", "i3": "G",
                                 "o1": "A", "o2": "A", "o3": "A"})
        (post,) = [p for _, p in self._inf(site).infer()]
        assert post.map_allele == "A"
        assert post["A"] > 0.9


class TestSourceDispatch:
    """``FixedTreeInference`` accepts non-tree sources (paths, tree sequences) and dispatches."""

    def test_trees_path_yields_one_posterior_per_site(self, small_ts, tmp_path):
        """A ``.trees`` path source materialises through
        :class:`~ancestree.sources.TskitSource` and yields one posterior per site
        (covers the ``.trees`` dispatch tail of ``SiteSource.resolve``).
        """
        trees_path = tmp_path / "src.trees"
        small_ts.dump(str(trees_path))
        # Pick a small set of node ids as ingroup / outgroup samples.
        sample_ids = [str(int(s)) for s in small_ts.samples()]
        ingroup = sample_ids[:6]
        outgroup = sample_ids[6:8]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inf = FixedTreeInference(
                str(trees_path),
                ingroup_samples=ingroup,
                outgroup_samples=outgroup,
                model=JC69(),
                n_target_sites=int(small_ts.sequence_length),
                progress=False,
                n_starts=1, parallelize=False,
            )
        inf.fit()
        emitted = list(inf.infer())
        assert len(emitted) == int(small_ts.num_sites)

    def test_tskit_source_builds_outgroup_ladder_tree(self, small_ts):
        """A ``tskit.TreeSequence`` source + ``ingroup/outgroup_samples=``
        auto-constructs an :class:`OutgroupLadderTree` (covers
        ``inference.py`` lines 798-817, the source-but-no-tree branch).
        """
        sample_ids = [str(int(s)) for s in small_ts.samples()]
        ingroup = sample_ids[:6]
        outgroup = sample_ids[6:8]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inf = FixedTreeInference(
                small_ts,
                ingroup_samples=ingroup,
                outgroup_samples=outgroup,
                model=JC69(),
                n_target_sites=int(small_ts.sequence_length),
                progress=False,
                fit_required=False,
            )
        assert isinstance(inf.tree, OutgroupLadderTree)
        assert list(inf.tree.outgroup_samples) == outgroup
        assert list(inf.tree.ingroup_samples) == ingroup

    def test_vcf_path_fits_and_infers(self, small_ts, tmp_path):
        """A VCF path source materialises through
        :class:`~ancestree.sources.CyVCF2Source` and fits + infers end-to-end
        (the CLI's ``fixed-tree`` entry point, covers the ``CyVCF2Source``
        dispatch tail of ``SiteSource.resolve``).
        """
        from ancestree.sources import CyVCF2Source
        vcf_path = tmp_path / "src.vcf"
        with open(vcf_path, "w") as f:
            small_ts.write_vcf(f, contig_id="1", allow_position_zero=True)
        hap_names = CyVCF2Source(vcf_path).samples()
        ingroup, outgroup = hap_names[:-4], hap_names[-4:]
        n_src_sites = len(list(CyVCF2Source(vcf_path)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            inf = FixedTreeInference(
                str(vcf_path),
                ingroup_samples=ingroup,
                outgroup_samples=outgroup,
                model=JC69(),
                n_target_sites=int(small_ts.sequence_length),
                progress=False,
                n_starts=1, parallelize=False,
            )
            inf.fit()  # fit-quality warnings irrelevant here (dispatch test)
        emitted = list(inf.infer())
        assert len(emitted) == n_src_sites > 0
        for _s, post in emitted:
            np.testing.assert_allclose(post.values.sum(), 1.0, atol=1e-9)

    @pytest.mark.filterwarnings(
        "ignore::zarr.errors.UnstableSpecificationWarning"
    )
    def test_vcz_path_builds_outgroup_ladder_tree(self, small_ts, tmp_path):
        """A ``.vcz`` path source materialises through
        :class:`~ancestree.sources.VcfZarrSource` and dispatches to an
        :class:`OutgroupLadderTree` (covers the ``VcfZarrSource`` branch of
        ``SiteSource.resolve``).
        """
        from testing._helpers import ts_to_vcz
        vcz_path = tmp_path / "src.vcz"
        names = ts_to_vcz(small_ts, vcz_path)
        ingroup, outgroup = names[:-4], names[-4:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf = FixedTreeInference(
                str(vcz_path),
                ingroup_samples=ingroup,
                outgroup_samples=outgroup,
                model=JC69(),
                n_target_sites=int(small_ts.sequence_length),
                progress=False, fit_required=False,
            )
        assert isinstance(inf.tree, OutgroupLadderTree)
        assert list(inf.tree.outgroup_samples) == outgroup
        emitted = list(inf.infer())
        assert len(emitted) > 0
        for _s, post in emitted:
            np.testing.assert_allclose(post.values.sum(), 1.0, atol=1e-9)

    def test_to_arg_unavailable(self, tmp_path):
        """Fixed-tree mode has no tree sequence, so the inherited base
        ``to_arg`` raises rather than writing a ``.trees`` file."""
        ingroup = [f"i{k}" for k in range(4)]
        tree = OutgroupLadderTree(ingroup, ["o1", "o2"])
        tree.set_params(np.array([0.0, 0.025, 0.025]))
        inf = FixedTreeInference(
            [], JC69(), _no_counts(), tree=tree, fit_required=False,
            progress=False,
        )
        with pytest.raises(ValueError, match="no tree sequence"):
            inf.to_arg(str(tmp_path / "x.trees"))

    def test_tree_as_source_rejected(self):
        tree = OutgroupLadderTree(["i0"], ["o1", "o2"])
        with pytest.raises(TypeError, match="site data"):
            FixedTreeInference(tree, JC69())

    def test_unsupported_source_type(self):
        with pytest.raises(TypeError, match="unsupported source type"):
            FixedTreeInference(123, JC69())


# ------------------------------------------------------------- baseline-check SFS bin


class TestBaselineCheckSFSBins:
    """``baseline_check=True`` log includes the per-SFS-bin disagreement tail."""

    def test_logs_largest_disagreement_at_sfs_bin(self, caplog):
        """Sites spanning multiple folded-SFS bins should produce the
        ``largest disagreement at SFS bin`` substring in the INFO line
        (covers ``inference.py`` lines 1856-1868).
        """
        ingroup = [f"i{k}" for k in range(4)]  # n=4 → bins 0, 1, 2 possible
        outgroup = ["o1", "o2"]
        tree = OutgroupLadderTree(ingroup, outgroup)
        tree.set_params(np.array([0.0, 0.025, 0.025]))
        # Mix of singletons (bin 1) and doubletons (bin 2) so per_bin_total
        # has >= 2 distinct entries.
        sites = []
        # Singletons (bin 1): one ingroup carries the minor allele.
        for k in range(3):
            sites.append(Site(
                chrom="1", pos=k + 1, alleles=("A", "T"),
                tip_alleles={
                    "i0": "A", "i1": "A", "i2": "A", "i3": "T",
                    "o1": "A", "o2": "A",
                },
            ))
        # Doubletons (bin 2): two ingroup carry the minor allele.
        for k in range(3):
            sites.append(Site(
                chrom="1", pos=10 + k, alleles=("A", "T"),
                tip_alleles={
                    "i0": "A", "i1": "A", "i2": "T", "i3": "T",
                    "o1": "A", "o2": "A",
                },
            ))
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            fit_required=False, baseline_check=True,
            progress=False, n_starts=1, parallelize=False,
        )
        with caplog.at_level(logging.INFO, logger="ancestree.inference"):
            list(inf.infer())
        text = caplog.text
        assert "largest disagreement at SFS bin" in text, (
            f"expected per-SFS-bin tail in INFO log; got:\n{text}"
        )


class TestNoOutgroupMonomorphic:
    """No-outgroup mode: monoallelic ingroup -> point mass on observed allele."""

    def _infer_one(self, site, ingroup):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inf = FixedTreeInference(
                [site], ingroup_samples=ingroup, outgroup_samples=[],
                model=JC69(), progress=False,
            )
        (post,) = [p for _, p in inf.infer()]
        return post

    def test_monoallelic_gets_point_mass(self):
        ingroup = ["i1", "i2", "i3"]
        site = Site(chrom="1", pos=1, alleles=("G",),
                    tip_alleles={"i1": "G", "i2": "G", "i3": "G"})
        post = self._infer_one(site, ingroup)
        assert post["G"] == 1.0
        assert post.max_prob == 1.0

    def test_polymorphic_still_uses_prior(self):
        """A segregating ingroup must keep the (non-degenerate) SFS prior."""
        ingroup = ["i1", "i2", "i3"]
        site = Site(chrom="1", pos=2, alleles=("A", "G"),
                    tip_alleles={"i1": "A", "i2": "A", "i3": "G"})
        post = self._infer_one(site, ingroup)
        assert post.max_prob < 1.0
        assert post["A"] > post["G"]  # major allele favoured ancestral


# ------------------------------------------------------------------- streaming


def _synthetic_source_sites(n_ingroup=12, n_out=3, n_sites=300, *, seed=0):
    """Build a list of ingroup+outgroup :class:`Site` records for the source
    (path/SiteSource) construction form, with biallelic ingroup AFS at random
    derived-allele frequencies and the ancestral allele in every outgroup."""
    rng = np.random.default_rng(seed)
    ing = [f"i{k}" for k in range(n_ingroup)]
    outs = [f"o{k}" for k in range(n_out)]
    sites = []
    for p in range(1, n_sites + 1):
        a, b = (str(x) for x in rng.choice(list("ACGT"), size=2, replace=False))
        freq = int(rng.integers(1, n_ingroup))  # 1..n-1 derived copies
        col = [a] * (n_ingroup - freq) + [b] * freq
        rng.shuffle(col)
        ta = {s: col[k] for k, s in enumerate(ing)}
        for j, s in enumerate(outs):
            # Outgroups mostly carry the ancestral allele, but each diverges
            # independently with a small, distinct rate so they are not
            # near-redundant (which would peg their branch rates at the bound).
            ta[s] = b if rng.random() < 0.05 * (j + 1) else a
        sites.append(Site(
            chrom="1", pos=p, alleles=tuple(sorted({a, b})), tip_alleles=ta,
        ))
    return sites, ing, outs


class TestStreaming:
    def test_stream_matches_materialised(self):
        """stream=True must give bit-identical fits and posteriors to the
        materialised path on the same data."""
        from ancestree import KingmanIngroupWeight
        sites, ing, outs = _synthetic_source_sites()
        kw = dict(
            ingroup_samples=ing, outgroup_samples=outs,
            n_target_sites=20_000, n_starts=2, parallelize=False, seed=1,
        )
        mat = FixedTreeInference(
            sites, model=JC69(),
            ingroup_weight=KingmanIngroupWeight(ingroup_samples=ing), **kw,
        )
        mat.fit()
        mpost = {s.pos: p.values for s, p in mat.infer()}

        strm = FixedTreeInference(
            sites, model=JC69(),
            ingroup_weight=KingmanIngroupWeight(ingroup_samples=ing),
            stream=True, **kw,
        )
        strm.fit()
        spost = {s.pos: p.values for s, p in strm.infer()}

        assert np.allclose(
            list(mat.params_mle.values()), list(strm.params_mle.values()),
            atol=1e-10,
        )
        assert set(mpost) == set(spost)
        for k in mpost:
            assert np.allclose(mpost[k], spost[k], atol=1e-12)

    def test_stream_matches_materialised_adaptive(self):
        """The adaptive prior streams too: its deduped incremental fit gives
        per-bin π and per-site posteriors bit-identical to the materialised
        fit."""
        from ancestree import AdaptiveIngroupWeight
        sites, ing, outs = _synthetic_source_sites(
            n_ingroup=16, n_sites=600, seed=3,
        )
        kw = dict(
            ingroup_samples=ing, outgroup_samples=outs,
            n_target_sites=20_000, n_starts=2, parallelize=False, seed=1,
            subsample_size=12,
        )

        def run(stream):
            inf = FixedTreeInference(
                sites, model=JC69(),
                ingroup_weight=AdaptiveIngroupWeight(
                    ingroup_samples=ing, subsample_size=12, parallelize=False,
                ),
                stream=stream, **kw,
            )
            inf.fit()
            return inf.ingroup_weight, {s.pos: p.values for s, p in inf.infer()}

        mat_prior, mpost = run(False)
        strm_prior, spost = run(True)
        assert set(mat_prior._pi) == set(strm_prior._pi)
        for k in mat_prior._pi:
            assert abs(mat_prior._pi[k] - strm_prior._pi[k]) < 1e-12
        assert set(mpost) == set(spost)
        for k in mpost:
            assert np.allclose(mpost[k], spost[k], atol=1e-12)

    def test_tree_as_source_rejected(self):
        """A tree passed as ``source`` is rejected, it goes via ``tree=``."""
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        with pytest.raises(TypeError, match="pass a pre-built"):
            FixedTreeInference(tree, JC69(), _no_counts())

    def test_streaming_construction_reads_source_once(self):
        """Construction iterates a streaming source once.

        The polymorphic projection runs once and reports the site count.
        With the redundant-outgroup data-quality pass disabled
        (``outgroup_similarity_threshold=0``), the source is iterated exactly
        once during construction.
        """
        from ancestree.sites import SiteSource

        sites, ing, outs = _synthetic_source_sites()
        all_samples = list(ing) + list(outs)

        class CountingSource(SiteSource):
            def __init__(self, records):
                self._records = records
                self.scans = 0

            def __iter__(self):
                self.scans += 1
                yield from self._records

            def samples(self):
                return list(all_samples)

        src = CountingSource(sites)
        FixedTreeInference(
            src, model=JC69(),
            ingroup_samples=ing, outgroup_samples=outs,
            n_target_sites=20_000, n_starts=1, parallelize=False, seed=1,
            stream=True, outgroup_similarity_threshold=0.0,
        )
        assert src.scans == 1, (
            f"expected one construction scan, got {src.scans}"
        )


class TestDeepReadoutFitNotRequired:
    """The deep reading must work in fit_required=False mode (pre-parameterised
    tree), not only after fit(). It was once gated on the MLE fit-state, so a
    monoallelic-ingroup site whose closest outgroup shares the derived allele
    was mis-called as that derived allele."""

    def _infer_map(self, tip_alleles):
        from ancestree.focal import FocalNode
        tree = OutgroupLadderTree(["i1"], ["o1", "o2", "o3"])
        tree.set_params(np.array([0.01, 0.02, 0.04, 0.06, 0.5]))
        site = Site(chrom="1", pos=1,
                    alleles=tuple(sorted(set(tip_alleles.values()))),
                    tip_alleles=tip_alleles)
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            focal=FocalNode("ingroup_mrca", fraction=1.0),
            n_starts=1, parallelize=False, fit_required=False,
        )
        ((_, post),) = list(inf.infer())
        return post.map_allele

    def test_closest_outgroup_shares_derived_allele(self):
        # ingroup fixed 'T'. Closest outgroup also 'T', deeper outgroups 'A'.
        # The deep-root ancestral state is 'A'. The ingroup-rooted call (bug)
        # would return 'T'.
        m = self._infer_map({"i1": "T", "o1": "T", "o2": "A", "o3": "A"})
        assert m == "A", f"expected deep-root ancestral 'A', got {m!r}"

    def test_all_outgroups_ancestral_unaffected(self):
        m = self._infer_map({"i1": "T", "o1": "A", "o2": "A", "o3": "A"})
        assert m == "A"


class TestRedundantOutgroupCheckStreaming:
    """The redundant-outgroup data-quality warning also fires in
    streaming mode (the default VCF path), where self.sites is empty."""

    def test_warns_on_identical_outgroups_via_vcf_stream(self, tmp_path, caplog):
        # Two outgroup samples that are byte-identical across all sites -> the
        # check should flag them as near-redundant.
        n = 60
        lines = [
            "##fileformat=VCFv4.2", "##contig=<ID=1>",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ti1\to1\to2",
        ]
        for k in range(n):
            ref, alt = ("A", "T")
            # ingroup varies. O1 and o2 carry the identical allele every site.
            ig = "1" if k % 2 else "0"
            lines.append(f"1\t{k+1}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t{ig}\t0\t0")
        vcf = tmp_path / "redundant.vcf"
        vcf.write_text("\n".join(lines) + "\n")

        with caplog.at_level(logging.WARNING, logger="ancestree.FixedTreeInference"):
            FixedTreeInference(
                str(vcf), model=JC69(), ingroup_samples=["i1"],
                outgroup_samples=["o1", "o2"], n_target_sites=n,
                stream=True,
            )
        assert any("near-redundant" in r.message for r in caplog.records), \
            "streaming redundancy check did not fire"


def test_initial_rates_array_not_mutated_by_fixed_params():
    """Passing initial_rates as a float64 ndarray together with
    fixed_params must not overwrite the caller's array in place."""
    tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
    n = tree.n_params
    rates = np.full(n, 0.05, dtype=float)
    original = rates.copy()
    FixedTreeInference(
        [], JC69(), _no_counts(), tree=tree, fit_required=False,
        initial_rates=rates, fixed_params={tree.param_names[0]: 1e-3},
    )
    np.testing.assert_array_equal(rates, original)  # caller's buffer untouched


class TestBothSiteClassesArePinned:
    """Golden posteriors for the two readings fixed-tree mode produces.

    The ladder is one tree and the focal node is a position on it. At the
    ingroup MRCA the site-frequency prior applies and a site the ingroup has
    fixed returns its own allele. At the deepest join the ingroup enters as a
    tip under the stationary prior, which is the reading such a site needs.
    Both are pinned because a change to the ladder's topology or branch
    bookkeeping moves one without touching the other. The tolerance is tighter
    than the one-site rounding in apportioning the region total across bases,
    so how the monomorphic weight is split also moves them.
    """

    ING = [f"i{k}" for k in range(6)]
    OUT = ["o1", "o2", "o3"]
    RATES = [0.004, 0.008, 0.006, 0.014, 0.020]

    SPECS = [
        ("poly_1_of_6", "GAAAAA", "AAA"),
        ("poly_3_of_6", "GGGAAA", "AAA"),
        ("poly_5_of_6", "GGGGGA", "AAA"),
        ("mono_ancestral", "AAAAAA", "AAA"),
        ("mono_derived", "GGGGGG", "AAA"),
        ("mono_split_outgroups", "GGGGGG", "GAA"),
    ]

    #: P(A) at the ingroup MRCA. The polymorphic values are invariant to the
    #: ladder carrying the collapsed ingroup as a tip. The monomorphic ones are
    #: degenerate, the site reading back its own allele.
    AT_INGROUP_MRCA = {
        "poly_1_of_6": 0.9997319583628013,
        "poly_3_of_6": 0.9986612272014174,
        "poly_5_of_6": 0.9933417913217492,
        "mono_ancestral": 1.0,
        "mono_derived": 0.0,
        "mono_split_outgroups": 0.0,
    }

    #: P(A) at the panel's own ancestor, on the deepest outgroup's branch.
    #: The three polymorphic values separate, in order of derived count: the
    #: ingroup enters as a likelihood on its own MRCA, so its allele
    #: frequency propagates to whatever node the run reports at.
    AT_DEEP_JOIN = {
        "poly_1_of_6": 0.9999746789196151,
        "poly_3_of_6": 0.9999746281055778,
        "poly_5_of_6": 0.9999743756593787,
        "mono_ancestral": 0.9999746916401543,
        "mono_derived": 0.999927234314803,
        "mono_split_outgroups": 0.9883371491627605,
    }

    def _run(self, focal=None):
        sites = [
            Site(chrom="1", pos=pos, alleles=("A", "G"),
                 tip_alleles={**dict(zip(self.ING, ingroup)),
                              **dict(zip(self.OUT, outgroup))})
            for pos, (_, ingroup, outgroup) in enumerate(self.SPECS, start=1)
        ]
        tree = OutgroupLadderTree(self.ING, self.OUT)
        tree.set_params(self.RATES)
        inference = FixedTreeInference(
            sites, JC69(), n_target_sites=10_000, tree=tree, focal=focal,
            fit_required=False, n_starts=1, parallelize=False, progress=False,
        )
        return {self.SPECS[site.pos - 1][0]: float(post["A"])
                for site, post in inference.infer()}

    @pytest.mark.parametrize("label", list(AT_INGROUP_MRCA))
    def test_ingroup_mrca_reading_is_pinned(self, label):
        assert self._run()[label] == pytest.approx(
            self.AT_INGROUP_MRCA[label], rel=1e-8, abs=1e-13,
        )

    @pytest.mark.parametrize("label", list(AT_DEEP_JOIN))
    def test_deep_reading_is_pinned(self, label):
        from ancestree.focal import FocalNode
        got = self._run(FocalNode("ingroup_mrca", fraction=1.0))
        assert got[label] == pytest.approx(
            self.AT_DEEP_JOIN[label], rel=1e-8, abs=1e-13,
        )

    def test_a_fixed_site_needs_the_deeper_node_to_say_anything(self):
        from ancestree.focal import FocalNode
        shallow = self._run()
        deep = self._run(FocalNode("ingroup_mrca", fraction=1.0))
        # At the ingroup MRCA the site restates its own allele. Deeper, the
        # outgroups overturn it.
        assert shallow["mono_derived"] == pytest.approx(0.0, abs=1e-10)
        assert deep["mono_derived"] > 0.99


class TestFitPriorAppliedOnce:
    """The config objective's prior slot holds the ingroup term alone.

    With no ingroup weight the slot stays empty, since ``_neg_log_likelihood``
    adds the stationary log-prior itself. Flat under a uniform composition,
    so only a skewed one exposes a double count.
    """

    @staticmethod
    def _sites(n=60):
        rng = np.random.default_rng(0)
        ing = [f"i{i}" for i in range(4)]
        out = ["o0", "o1"]
        sites = []
        for k in range(n):
            a, b = ("A", "C") if k % 2 else ("G", "T")
            tips = {s: (a if rng.random() < 0.7 else b) for s in ing}
            tips.update({o: (a if rng.random() < 0.8 else b) for o in out})
            sites.append(anc.Site(chrom="1", pos=k + 1, alleles=(a, b),
                                 tip_alleles=tips))
        return sites, ing, out

    def test_no_ingroup_weight_leaves_the_slot_empty(self):
        from ancestree.inference import FixedTreeInference
        from ancestree.priors import NoIngroupWeight

        sites, ing, out = self._sites()
        bc = anc.BaseComposition.from_counts(A=4000, C=1000, G=1000, T=4000)
        inf = FixedTreeInference(
            sites, HKY(), bc, tree=anc.OutgroupLadderTree(ing, out),
            ingroup_weight=NoIngroupWeight(), progress=False,
        )
        rows = np.asarray(inf._fit_log_prior)
        # Mono-allelic-ingroup configs carry a delta. The rest must be flat.
        plain = np.isfinite(rows).all(axis=1)
        assert plain.any()
        assert np.allclose(rows[plain], 0.0)
        assert not np.allclose(rows[plain], np.log(bc.pi))


class TestIngroupWeightSlotIsTyped:
    """The ingroup-weight slot accepts only an IngroupWeight.

    A ``StationaryPrior`` passed here would be seeded on the ingroup MRCA as
    a likelihood rather than applied once at the readout, which under an
    empirical composition is a different model.
    """

    @staticmethod
    def _kwargs():
        sites = [anc.Site(chrom="1", pos=1, alleles=("A", "C"),
                         tip_alleles={"i0": "A", "i1": "C", "o0": "A"})]
        return dict(
            sites=sites, model=JC69(),
            base_composition=anc.BaseComposition.from_n_target_sites(1000),
            tree=anc.OutgroupLadderTree(["i0", "i1"], ["o0"]), progress=False,
        )

    def test_a_stationary_prior_is_rejected(self):
        from ancestree.inference import FixedTreeInference
        from ancestree.priors import StationaryPrior

        kw = self._kwargs()
        with pytest.raises(TypeError, match="ingroup_weight must be"):
            FixedTreeInference(
                kw["sites"], kw["model"], kw["base_composition"],
                tree=kw["tree"], progress=False,
                ingroup_weight=StationaryPrior(JC69()),
            )

    def test_a_flat_weight_is_accepted(self):
        from ancestree.inference import FixedTreeInference

        kw = self._kwargs()
        inf = FixedTreeInference(
            kw["sites"], kw["model"], kw["base_composition"],
            tree=kw["tree"], progress=False,
            ingroup_weight=anc.NoIngroupWeight(),
        )
        assert isinstance(inf.ingroup_weight, anc.NoIngroupWeight)


class TestIngroupIdsResolvePerIndividual:
    """The config projection resolves an ingroup id the way count_alleles does.

    ``Site.count_alleles`` matches an id as a tip or as the individual a tip
    belongs to, and the projection behind the branch-rate fit must do the
    same, or a diploid panel named per individual lands in the zero-AFS
    "no called ingroup" cell.
    """

    @staticmethod
    def _site():
        return anc.Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0_h0": "A", "i0_h1": "A",
                         "i1_h0": "C", "i1_h1": "A", "o0_h0": "A"},
        )

    def test_individual_and_haplotype_naming_project_alike(self):
        bc = anc.BaseComposition.no_counts()
        site = self._site()
        by_hap, _ = bc._project_polymorphic(
            [site], ["i0_h0", "i0_h1", "i1_h0", "i1_h1"], ["o0_h0"], 2)
        by_ind, _ = bc._project_polymorphic(
            [site], ["i0", "i1"], ["o0_h0"], 2)
        assert by_ind == by_hap
        # The zero-AFS marker is the cell an exact lookup would collapse onto.
        assert ((0, 0, 0, 0), ("A",)) not in by_ind

    def test_the_polymorphic_filter_accepts_an_individual_id(self):
        from ancestree.sites import PolymorphicSiteFilter

        site = self._site()
        assert PolymorphicSiteFilter(
            [site], samples=["i0", "i1"]).accepts(site)


class TestNamingTheIngroupEitherWayAgrees:
    """Naming individuals and naming their haplotypes give the same fit.

    The sub-sample size is bounded by the haplotypes the ids resolve to, not
    by the number of ids, so a diploid panel named per individual projects
    onto the full spectrum.
    """

    @staticmethod
    def _panel(n_sites=300):
        rng = np.random.default_rng(7)
        haps = [f"i{i}_h{h}" for i in range(4) for h in (0, 1)]
        out = ["o0_h0", "o1_h0"]
        sites = []
        for k in range(n_sites):
            a, b = ("A", "C") if k % 2 else ("G", "T")
            tips = {s: (a if rng.random() < 0.8 else b) for s in haps}
            tips.update({o: (a if rng.random() < 0.9 else b) for o in out})
            sites.append(anc.Site(chrom="1", pos=k + 1, alleles=(a, b),
                                 tip_alleles=tips))
        return sites, haps, out

    def _fit(self, sites, ingroup, outgroup):
        from ancestree.inference import FixedTreeInference

        inference = FixedTreeInference(
            sites, JC69(), anc.BaseComposition.from_n_target_sites(200_000),
            tree=anc.OutgroupLadderTree(ingroup, outgroup),
            n_starts=1, seed=1, progress=False,
        )
        inference.fit()
        params = inference.params_mle
        values = list(params.values()) if isinstance(params, dict) else params
        return inference.subsample_size, np.asarray(values, dtype=float)

    def test_the_two_namings_give_the_same_rates(self):
        sites, haps, out = self._panel()
        by_hap = self._fit(sites, haps, out)
        by_ind = self._fit(sites, ["i0", "i1", "i2", "i3"], out)
        assert by_hap[0] == by_ind[0] == 8
        np.testing.assert_allclose(by_ind[1], by_hap[1], rtol=0, atol=0)

    def test_a_subsample_beyond_the_haplotypes_is_refused(self):
        from ancestree.inference import FixedTreeInference

        sites, _, out = self._panel(n_sites=20)
        with pytest.raises(ValueError, match=r"in \[2, 8\]"):
            FixedTreeInference(
                sites, JC69(), anc.BaseComposition.from_n_target_sites(1000),
                tree=anc.OutgroupLadderTree(["i0", "i1", "i2", "i3"], out),
                subsample_size=9, progress=False,
            )


class TestNoOutgroupModeIsUsable:
    """The constructor supports ``outgroup_samples=[]``, and so must the object."""

    @staticmethod
    def _inference():
        sites = [
            anc.Site(chrom="1", pos=p, alleles=("A", "C"),
                    tip_alleles={"i0": "A", "i1": "C"})
            for p in range(1, 20)
        ]
        return anc.FixedTreeInference(
            sites, JC69(), ingroup_samples=["i0", "i1"], outgroup_samples=[],
        )

    def test_repr_does_not_raise(self):
        assert "FixedTreeInference(" in repr(self._inference())

    def test_model_params_mle_is_none(self):
        assert self._inference().model_params_mle is None


def _biallelic_site(pos, ingroup, outgroup, *, minor="T", major="A"):
    """A biallelic site: first ingroup sample carries the minor allele, the
    rest (ingroup + outgroup) carry the major allele."""
    tip = {s: major for s in list(ingroup) + list(outgroup)}
    tip[ingroup[0]] = minor
    return Site(
        chrom="1", pos=pos, alleles=(major, minor), tip_alleles=tip,
    )


def _fittable_instance(**overrides):
    """A small, valid fitting-mode FixedTreeInference (2 outgroups)."""
    ingroup = ["i0", "i1"]
    outgroup = ["o1", "o2"]
    tree = OutgroupLadderTree(ingroup, outgroup)
    sites = [
        Site(chrom="1", pos=i + 1, alleles=("A", "T"),
             tip_alleles={"i0": "T", "i1": "A", "o1": "A", "o2": "A"})
        for i in range(3)
    ]
    kwargs = dict(tree=tree, n_target_sites=3, progress=False, n_starts=2)
    kwargs.update(overrides)
    return FixedTreeInference(sites, JC69(), _no_counts(), **kwargs)


class TestSubsampleSizeReconciliation:
    """A supplied AFS-aware prior must agree on ``subsample_size``.

    The prior carrying a ``subsample_size`` attribute is
    :class:`AdaptiveIngroupWeight`, so the mismatch guard is exercised with
    the adaptive prior.
    """

    ingroup = [f"i{i}" for i in range(8)]
    outgroup = ["o1", "o2"]

    def test_mismatch_raises(self):
        prior = AdaptiveIngroupWeight(self.ingroup, subsample_size=7)
        tree = OutgroupLadderTree(self.ingroup, self.outgroup)
        site = _biallelic_site(1, self.ingroup, self.outgroup)
        with pytest.raises(ValueError, match="subsample_size mismatch"):
            FixedTreeInference(
                [site], JC69(), _no_counts(), tree=tree,
                subsample_size=5, ingroup_weight=prior,
                fit_required=False, progress=False,
            )

    def test_agreeing_does_not_raise(self):
        prior = AdaptiveIngroupWeight(self.ingroup, subsample_size=7)
        tree = OutgroupLadderTree(self.ingroup, self.outgroup)
        site = _biallelic_site(1, self.ingroup, self.outgroup)
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            subsample_size=7, ingroup_weight=prior,
            fit_required=False, progress=False,
        )
        assert inf.subsample_size == 7


class TestOptimizerTotalFailure:
    def test_all_starts_fail_raises(self, monkeypatch):
        def _fake_minimize(fun, x0, **kwargs):
            return OptimizeResult(
                x=np.asarray(x0, dtype=float),
                fun=float("inf"),
                success=False,
                message="mock non-convergence",
            )

        monkeypatch.setattr(inference_mod, "minimize", _fake_minimize)
        inf = _fittable_instance()
        with pytest.raises(RuntimeError, match="failed to converge"):
            inf.fit()


class TestNoOutgroupMode:
    def test_streaming_unsupported(self):
        with pytest.raises(NotImplementedError, match="no-outgroup"):
            FixedTreeInference([], JC69(), outgroup_samples=[], stream=True)

    def test_ingroup_required(self):
        with pytest.raises(ValueError, match="ingroup_samples is required"):
            FixedTreeInference([], JC69(), outgroup_samples=[])


def test_a_list_source_is_written_from_the_sites(tmp_path):
    import cyvcf2

    tree = OutgroupLadderTree(["i0"], ["o1", "o2"])
    site = Site(chrom="1", pos=1, alleles=("A",),
                tip_alleles={"x": "A", "o1": "A", "o2": "A"})
    inf = FixedTreeInference(
        [site], JC69(), _no_counts(), tree=tree, fit_required=False,
    )
    assert inf.to_vcf(tmp_path / "out.vcf") == 1
    assert cyvcf2.VCF(str(tmp_path / "out.vcf")).samples == ["o1", "o2", "i0"]


def _fit_with_both_pools_requested():
    """Fit with the multistart pool and the adaptive per-bin pool both asked for."""
    ingroup = [f"i{i}" for i in range(8)]
    outgroup = ["o1", "o2"]
    tree = OutgroupLadderTree(ingroup, outgroup)
    rng = np.random.default_rng(5)
    sites = []
    for i in range(300):
        k = int(rng.integers(1, len(ingroup)))
        carriers = set(rng.choice(len(ingroup), size=k, replace=False).tolist())
        tip_alleles = {sid: ("T" if j in carriers else "A")
                       for j, sid in enumerate(ingroup)}
        tip_alleles.update({"o1": "A", "o2": "A"})
        sites.append(Site(chrom="1", pos=i + 1, alleles=("A", "T"),
                          tip_alleles=tip_alleles))
    prior = AdaptiveIngroupWeight(ingroup, n_runs=2, seed=3, parallelize=True)
    inf = FixedTreeInference(
        sites, JC69(), _no_counts(), tree=tree, n_target_sites=len(sites),
        ingroup_weight=prior, n_starts=2, parallelize=True, progress=False,
    )
    return inf.fit(), prior.fitted


class TestFittingInsideADaemonicWorker:
    """A fit driven from inside someone else's pool must fall back to serial.

    ``multiprocessing`` forbids a daemonic process from having children, so
    both worker pools the fit opens, the one over the L-BFGS-B starts and the
    adaptive prior's one over the spectrum bins, raised ``AssertionError:
    daemonic processes are not allowed to have children`` whenever a caller ran
    the inference inside a pool of their own.
    """

    def test_a_fit_inside_a_pool_worker_returns_results(self):
        import multiprocessing as mp

        with mp.get_context("fork").Pool(1) as pool:
            params, prior_fitted = pool.apply(_fit_with_both_pools_requested)
        assert {"K1", "K2"} <= set(params)
        assert all(np.isfinite(v) for v in params.values())
        assert prior_fitted is True


class _PairSource(SiteSource):
    """Re-iterable stream of biallelic sites over two outgroups and an ingroup.

    ``o1`` is uncalled at ``missing_at``, carries the allele ``o0`` does not at
    ``diff_at``, and matches ``o0`` everywhere else.
    """

    def __init__(self, n_sites: int, diff_at=(), missing_at=(),
                 census_at: int | None = None) -> None:
        self.n_sites = n_sites
        self.diff_at = set(diff_at)
        self.missing_at = set(missing_at)
        self.census_at = census_at
        self.n_yielded = 0
        self.live_sites: int | None = None

    def samples(self):
        return ["o0", "o1", "i0", "i1"]

    def __iter__(self):
        for pos in range(self.n_sites):
            if pos in self.missing_at:
                o1 = None
            elif pos in self.diff_at:
                o1 = "C"
            else:
                o1 = "A"
            self.n_yielded += 1
            if self.n_yielded == self.census_at:
                self.live_sites = sum(
                    1 for o in gc.get_objects() if type(o) is Site)
            yield Site(
                chrom="1", pos=pos + 1, alleles=("A", "C"),
                tip_alleles={"o0": "A", "o1": o1,
                             "i0": "A", "i1": "C" if pos % 2 else "A"},
            )


def _stream_inference(source, **kwargs):
    return FixedTreeInference(
        source, JC69(), ingroup_samples=["i0", "i1"],
        outgroup_samples=["o0", "o1"], n_target_sites=10_000,
        fit_required=False, progress=False, stream=True, **kwargs,
    )


class TestRedundantOutgroupCheckStreams:
    """The pre-fit redundancy check reads its prefix without holding it.

    Materialising the capped 50,000-site prefix retains one Site record per
    site, each carrying one freshly-minted key string per haplotype, so peak
    resident memory grew with the panel rather than staying flat in it: 1098 MB
    at 206 haplotypes over 60,000 records against 142 MB with the check
    disabled, and 2021 MB at 406 haplotypes.
    """

    def test_the_prefix_is_not_held_while_the_stream_is_read(self):
        # The constructor reads the source twice, for the config projection
        # and then for this check, so the census lands in the second pass.
        source = _PairSource(n_sites=200, diff_at=[0], census_at=380)
        _stream_inference(source)
        assert source.live_sites is not None
        assert source.live_sites < 20

    def test_the_pair_counts_match_the_whole_prefix(self, caplog):
        source = _PairSource(n_sites=200, diff_at=[60],
                             missing_at=range(50))
        with caplog.at_level(logging.WARNING):
            _stream_inference(source)
        assert "differ at only 1/150" in caplog.text

    def test_the_prefix_is_capped(self, caplog, monkeypatch):
        monkeypatch.setattr(
            FixedTreeInference, "_REDUNDANT_OUTGROUP_SAMPLE_CAP", 60)
        source = _PairSource(n_sites=200, diff_at=[100])
        with caplog.at_level(logging.WARNING):
            _stream_inference(source)
        assert "differ at only 0/60" in caplog.text

    def test_a_prefix_below_the_minimum_is_not_reported(self, caplog):
        source = _PairSource(n_sites=40)
        with caplog.at_level(logging.WARNING):
            _stream_inference(source)
        assert "near-redundant" not in caplog.text


def _weighted_sites(n_sites: int, ingroup: list[str]) -> list[Site]:
    """Sites cycling through polymorphic, ingroup-monomorphic and all-missing."""
    sites = []
    for pos in range(n_sites):
        kind = pos % 5
        if kind == 3:  # monomorphic within the ingroup
            tips = {sid: "A" for sid in ingroup}
        elif kind == 4:  # no ingroup allele called at all
            tips = {sid: None for sid in ingroup}
        else:
            tips = {sid: ("C" if (pos + j) % 3 == 0 else "A")
                    for j, sid in enumerate(ingroup)}
        tips.update({"o0": "A", "o1": "C" if pos % 7 == 0 else "A"})
        sites.append(Site(chrom="1", pos=pos + 1, alleles=("A", "C"),
                          tip_alleles=tips))
    return sites


class TestIngroupCountsAreTalliedOnce:
    """One allele tally per site feeds the ingroup weight and the diagnostic.

    ``Site.count_alleles`` ran twice per site on the identical sample list,
    once inside ``KingmanIngroupWeight`` and once for the ingroup-monomorphic
    diagnostic: 40,000 calls over 20,000 sites at 100 haplotypes, half of the
    profiled ``infer()`` time.
    """

    INGROUP = [f"i{k}" for k in range(6)]

    def _inference(self, **kwargs):
        return FixedTreeInference(
            _weighted_sites(200, self.INGROUP), JC69(),
            ingroup_samples=self.INGROUP, outgroup_samples=["o0", "o1"],
            n_target_sites=10_000, fit_required=False, progress=False,
            baseline_check=False, **kwargs,
        )

    @staticmethod
    def _run(inf):
        """Posteriors and the ingroup-monomorphic tally, read before the
        end-of-run summary clears it."""
        inf._log_uniform_fallback_summary = lambda: None
        posteriors = [np.asarray(p.values) for _, p in inf.infer()]
        return posteriors, inf._n_ingroup_monomorphic

    def test_the_shared_tally_reproduces_a_separate_one(self, monkeypatch):
        shared_post, shared_mono = self._run(self._inference())
        monkeypatch.setattr(
            FixedTreeInference, "_shared_ingroup_counts",
            lambda self, batch: None)
        own_post, own_mono = self._run(self._inference())
        assert shared_mono == own_mono
        assert np.abs(np.array(shared_post) - np.array(own_post)).max() == 0.0

    def test_an_all_missing_ingroup_site_is_counted_as_monomorphic(self):
        """The tally cannot be read off the weight's log-probability row.

        ``KingmanIngroupWeight`` returns four finite entries both for a site
        with no called ingroup allele, which the tally counts, and for a
        four-allele ingroup, which it does not.
        """
        inf = self._inference()
        _, n_mono = self._run(inf)
        sites = _weighted_sites(200, self.INGROUP)
        all_missing = [s for s in sites
                       if not s.count_alleles(self.INGROUP)]
        rows = anc.KingmanIngroupWeight(self.INGROUP).log_probs(all_missing)
        assert len(all_missing) == 40
        assert np.isfinite(rows).all()
        assert n_mono == 80  # the 40 monomorphic plus the 40 all-missing

    def test_counts_are_shared_only_for_a_matching_kingman_weight(self):
        batch = _weighted_sites(4, self.INGROUP)
        assert self._inference()._shared_ingroup_counts(batch) is not None
        for weight in (anc.NoIngroupWeight(),
                       anc.AdaptiveIngroupWeight(self.INGROUP),
                       anc.KingmanIngroupWeight(self.INGROUP[:3])):
            inf = self._inference(ingroup_weight=weight)
            assert inf._shared_ingroup_counts(batch) is None

    def test_the_shared_counts_equal_a_direct_tally(self):
        batch = _weighted_sites(10, self.INGROUP)
        shared = self._inference()._shared_ingroup_counts(batch)
        assert shared == [s.count_alleles(self.INGROUP) for s in batch]


class TestAnIngroupEmptiedByTheOutgroupListIsRefused:
    """Naming every panel sample as an outgroup leaves nothing to report at.

    The guard once read the class-level ``focal`` default because it ran
    before the constructor parsed the argument, took the ``panel_root``
    exemption, and let the run reach an ``IndexError`` in
    ``BaseComposition._finalize_configs``.
    """

    @staticmethod
    def _sites():
        return [anc.Site(chrom="1", pos=i, alleles=("A", "C"),
                         tip_alleles={"o1": "A", "o2": "C"})
                for i in range(1, 30)]

    @pytest.mark.parametrize("focal", ["ingroup_mrca", "panel_root"])
    def test_it_raises_whatever_the_run_reports_at(self, focal):
        from ancestree.trees import OutgroupLadderTree

        with pytest.raises(ValueError, match="the ingroup is empty"):
            anc.FixedTreeInference(
                self._sites(), anc.JC69(),
                anc.BaseComposition.from_counts(A=100, C=80, G=80, T=100),
                tree=OutgroupLadderTree([], ["o1", "o2"]), focal=focal,
                fit_required=False)


class TestTheNoOutgroupModeStillRecordsProvenance:
    """``provenance()`` reads the tree, which this mode does not build.

    It is what every annotated write records, so an exception here takes out
    ``to_vcf``, ``to_zarr`` and ``to_arg`` in that mode.
    """

    def test_provenance_names_the_ingroup(self):
        site = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                        tip_alleles={"i1": "A", "i2": "G"})
        inference = anc.FixedTreeInference(
            [site], anc.JC69(), ingroup_samples=["i1", "i2"],
            outgroup_samples=[])
        record = inference.provenance()
        assert record["parameters"]["ingroup_samples"] == ["i1", "i2"]
        assert record["parameters"]["outgroup_samples"] == []


# ---------------------------------------- constructor guards and fit paths


def _ladder_sites(n: int, outgroups=("o1", "o2"), pattern=None) -> list[Site]:
    """``n`` polymorphic sites over ``outgroups``, alleles cycling A/C.

    :param n: Number of sites.
    :param outgroups: Outgroup ids carried by every site.
    :param pattern: Optional callable ``(i, outgroup) -> allele or None``.
    """
    sites = []
    for i in range(n):
        if pattern is None:
            tips = {o: ("A" if (i + k) % 2 == 0 else "C")
                    for k, o in enumerate(outgroups)}
        else:
            tips = {o: pattern(i, o) for o in outgroups}
        alleles = tuple(sorted({a for a in tips.values() if a}))
        sites.append(Site(chrom="1", pos=i + 1, alleles=alleles, tip_alleles=tips))
    return sites


def _fixed_tree(sites, **kwargs) -> FixedTreeInference:
    """A fit-ready fixed-tree inference over ``i1`` and two outgroups."""
    return FixedTreeInference(
        sites, JC69(), _no_counts(),
        tree=OutgroupLadderTree(["i1"], ["o1", "o2"]),
        n_target_sites=1_000, progress=False, baseline_check=False, **kwargs,
    )


def test_provenance_reports_an_ascertained_composition_as_empirical():
    """A composition tallied from polymorphic sites carries no counts, only pi."""
    sites = _ladder_sites(4)
    bc = BaseComposition.from_polymorphic_sites(sites)
    assert bc.n_total == 0
    inf = FixedTreeInference(
        sites, JC69(), bc, tree=OutgroupLadderTree(["i1"], ["o1", "o2"]),
        n_target_sites=1_000, progress=False, baseline_check=False)
    assert inf.provenance()["parameters"]["base_composition"] == "empirical"
    assert _fixed_tree(sites).provenance()["parameters"]["base_composition"] == "uniform"


def test_fixed_tree_needs_names_or_a_ladder():
    with pytest.raises(ValueError, match="ingroup_samples and outgroup_samples"):
        FixedTreeInference([], JC69(), _no_counts(), ingroup_samples=["i1"])


def test_fixed_tree_refuses_an_ingroup_weight_as_prior():
    with pytest.raises(TypeError, match="prior must be a StationaryPrior"):
        _fixed_tree(_ladder_sites(4), prior=KingmanIngroupWeight(["i1"]))


def test_subsample_size_is_bounded_by_the_ingroup():
    sites = _ladder_sites(4, outgroups=("i1", "o1", "o2"))
    with pytest.raises(ValueError, match=r"subsample_size must be in \[2, 1\]"):
        _fixed_tree(sites, subsample_size=3)
    with pytest.raises(ValueError, match="subsample_size must be >= 2"):
        _fixed_tree(sites, subsample_size=1)


def test_infer_site_needs_an_ingroup_mrca_to_seed_the_weight(small_ts):
    """A fixed-tree run carries an ingroup weight that a tree without an
    ingroup MRCA cannot host."""
    inf = FixedTreeInference(
        [], JC69(), _no_counts(), tree=OutgroupLadderTree(["i1"], ["o1", "o2"]),
        fit_required=False, focal="panel_root",
    )
    assert isinstance(inf.ingroup_weight, KingmanIngroupWeight)
    tree = TskitLocalTree.from_tskit_tree(small_ts.first())
    site = Site(chrom="1", pos=1, alleles=("A",), tip_alleles={"0": "A"})
    with pytest.raises(ValueError, match="no ingroup MRCA"):
        inf.infer_site(tree, site)


def test_bound_warning_is_silent_before_the_fit(caplog):
    inf = _fixed_tree(_ladder_sites(4))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        inf._warn_if_branch_rates_at_bounds()
    assert not caplog.records


def test_redundancy_check_skips_a_pair_never_observed_together(caplog):
    """Two identical outgroups are reported. A third that is never observed
    alongside them is not compared."""
    def pattern(i, o):
        if o == "o3":
            return None
        return "A" if i % 2 == 0 else "C"

    sites = _ladder_sites(60, outgroups=("o1", "o2", "o3"), pattern=pattern)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        FixedTreeInference(
            sites, JC69(), _no_counts(),
            tree=OutgroupLadderTree(["i1"], ["o1", "o2", "o3"]),
            n_target_sites=1_000, progress=False, baseline_check=False,
        )
    reported = [r for r in caplog.records if "near-redundant" in r.message]
    assert len(reported) == 1
    assert "'o1' and 'o2'" in reported[0].message


def test_sample_presence_check_is_a_no_op_without_names():
    site = Site(chrom="1", pos=1, alleles=("A",), tip_alleles={"o1": "A"})
    assert FixedTreeInference._check_samples_present(None, [site]) is None
    assert FixedTreeInference._check_samples_present([], [site],
                                                     ingroup_samples=[]) is None


def test_no_outgroup_mode_has_nothing_to_fit_and_no_ladder(caplog):
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"i1": "A", "i2": "C"})
    inf = FixedTreeInference([site], JC69(), _no_counts(), ingroup_samples=["i1", "i2"],
                             outgroup_samples=[], progress=False)
    assert inf.fit() == {}
    assert inf._focal_tree is None
    assert len(list(inf.infer())) == 1
    empty = FixedTreeInference([], JC69(), _no_counts(), ingroup_samples=["i1"],
                               outgroup_samples=[], progress=False)
    assert list(empty.infer()) == []


def test_monotone_divergence_check_needs_two_outgroups(caplog):
    inf = FixedTreeInference(
        _ladder_sites(4, outgroups=("o1",)), JC69(), _no_counts(),
        tree=OutgroupLadderTree(["i1"], ["o1"]), n_target_sites=1_000,
        progress=False, baseline_check=False,
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        inf._warn_if_outgroup_divergences_non_monotone()
    assert not caplog.records


def test_objective_is_zero_without_configs():
    inf = _fixed_tree([], fit_required=False)
    inf._fit_configs = []
    assert inf._neg_log_likelihood(np.array(inf._x0)[inf._free_indices]) == 0.0


class TestRunParallelFallbacks:
    """Every serial fallback of the multi-start dispatcher returns one
    optimiser result per start."""

    @pytest.fixture
    def inf(self):
        return _fixed_tree(_ladder_sites(6), n_starts=2)

    def _check(self, inf):
        starts = inf._generate_starts()
        results = inf._run_parallel(starts)
        assert len(results) == len(starts)
        assert all(np.isfinite(r.fun) for r in results)

    def test_single_worker(self, inf, monkeypatch):
        monkeypatch.setattr(Settings, "parallelize", False)
        self._check(inf)

    def test_fork_unsafe_layer(self, inf, monkeypatch, caplog):
        monkeypatch.setattr(Settings, "_fork_is_safe", staticmethod(lambda: False))
        inf.n_workers = 2
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            self._check(inf)
        assert any("Ignoring the parallelization request" in r.message
                   for r in caplog.records)

    def test_no_fork_start_method(self, inf, monkeypatch, caplog):
        monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])
        inf.n_workers = 2
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            self._check(inf)
        assert any("running the 2 starts serially" in r.message
                   for r in caplog.records)
