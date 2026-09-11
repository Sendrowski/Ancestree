"""Tests for KingmanIngroupWeight and AdaptiveIngroupWeight."""
import logging

import numpy as np
import pytest

from ancestree import (
    AdaptiveIngroupWeight,
    FixedTreeInference,
    GTR,
    OutgroupLadderTree,
    JC69,
    KingmanIngroupWeight,
    STATE_INDEX,
    STATES,
    Site,
    StationaryPrior,
)
from ancestree.priors import IngroupWeight, _fit_pi_bin
from ancestree.settings import Settings

from testing._helpers import no_counts as _no_counts, panel_site, panel_ts


def _simulate_sites_with_ingroup_sfs(
    tree: OutgroupLadderTree,
    model: JC69,
    n_sites: int,
    *,
    ingroup_branch_length: float,
    seed: int,
) -> list[Site]:
    """Simulate sites where each ingroup haplotype is an independent draw
    from a JC69 branch of length ``ingroup_branch_length`` rooted at the
    inference root I. Produces realistic-looking ingroup SFS variation
    (close to neutral Kingman SFS) and outgroup tip patterns under the
    tree's branch rates.
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
            alleles=tuple(sorted({v for v in tip_alleles.values() if v})),
            tip_alleles=tip_alleles,
        ))
    return sites


# ----------------------------------------------------------------- prior class


class TestKingmanFormula:
    def test_biallelic_singleton(self):
        """n=4, i=1: P(major)=3/4, P(minor)=1/4."""
        prior = KingmanIngroupWeight(["i0", "i1", "i2", "i3"])
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": "A", "i1": "A", "i2": "A", "i3": "C"},
        )
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(0.75)
        assert p[STATE_INDEX["C"]] == pytest.approx(0.25)
        assert p[STATE_INDEX["G"]] == 0.0
        assert p[STATE_INDEX["T"]] == 0.0

    def test_biallelic_even_split(self):
        """n=4, i=2: P(major)=P(minor)=0.5."""
        prior = KingmanIngroupWeight(["i0", "i1", "i2", "i3"])
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": "A", "i1": "A", "i2": "C", "i3": "C"},
        )
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(0.5)
        assert p[STATE_INDEX["C"]] == pytest.approx(0.5)

    def test_monomorphic_ingroup_is_a_delta(self):
        """Mono-allelic ingroup gives a delta on the observed allele.

        The vector is the likelihood of the ingroup alleles given the state
        at the ingroup MRCA, seeded at that node, so a delta states an
        observation rather than ruling an ancestral state out. The outgroups
        and the branch above the MRCA still move mass onto alternatives.
        """
        prior = KingmanIngroupWeight(["i0", "i1", "i2"])
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"i0": "A", "i1": "A", "i2": "A"},
        )
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(1.0)
        for b in ("C", "G", "T"):
            assert p[STATE_INDEX[b]] == pytest.approx(0.0)

    def test_multiallelic_count_proportional(self):
        """3 segregating alleles → count-proportional, no fallback to uniform."""
        # Counts (2, 1, 1) → probs (0.5, 0.25, 0.25)
        prior = KingmanIngroupWeight(["i0", "i1", "i2", "i3"])
        site = Site(
            chrom="1", pos=1, alleles=("A", "C", "G"),
            tip_alleles={"i0": "A", "i1": "A", "i2": "C", "i3": "G"},
        )
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(0.5)
        assert p[STATE_INDEX["C"]] == pytest.approx(0.25)
        assert p[STATE_INDEX["G"]] == pytest.approx(0.25)
        assert p[STATE_INDEX["T"]] == 0.0

    def test_multiallelic_extreme(self):
        """Counts (18, 1, 1) in n=20 → majority dominates: (0.9, 0.05, 0.05)."""
        ingroup = [f"i{i}" for i in range(20)]
        prior = KingmanIngroupWeight(ingroup)
        alleles_per = ["A"] * 18 + ["C", "G"]
        site = Site(
            chrom="1", pos=1, alleles=("A", "C", "G"),
            tip_alleles={f"i{i}": a for i, a in enumerate(alleles_per)},
        )
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(0.9)
        assert p[STATE_INDEX["C"]] == pytest.approx(0.05)
        assert p[STATE_INDEX["G"]] == pytest.approx(0.05)

    def test_missing_ingroup_uniform(self):
        """No ingroup tips observed → uniform fallback."""
        prior = KingmanIngroupWeight(["i0", "i1"])
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"o1": "A"},  # no ingroup keys
        )
        p = np.exp(prior.log_probs([site])[0])
        np.testing.assert_allclose(p, np.full(4, 0.25))

    def test_skip_missing_individuals_in_ingroup(self):
        """Some ingroup individuals missing → only observed ones count."""
        prior = KingmanIngroupWeight(["i0", "i1", "i2", "i3"])
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": "A", "i1": "A", "i2": "C", "i3": None},
        )
        # n_obs=3, i=1 → P(major=A) = 2/3, P(minor=C) = 1/3
        p = np.exp(prior.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(2 / 3)
        assert p[STATE_INDEX["C"]] == pytest.approx(1 / 3)

    def test_empty_ingroup_raises(self):
        with pytest.raises(ValueError, match="at least one ingroup sample"):
            KingmanIngroupWeight([])


# --------------------------------------------------- integration with inference


class TestKingmanInIdentifiability:
    def test_1outgroup_warning_suppressed_with_kingman(self, caplog):
        """With a Kingman prior the 1-outgroup case is identifiable, so the
        unidentifiability warning should not fire."""
        tree = OutgroupLadderTree(["i0", "i1", "i2"], ["o1"])
        with caplog.at_level(logging.WARNING):
            FixedTreeInference(
                [], JC69(), _no_counts(), tree=tree,
                ingroup_weight=KingmanIngroupWeight(["i0", "i1", "i2"]),
                fit_required=False,
                n_starts=1, parallelize=False,
            )
        assert "unidentifiable" not in caplog.text

    def test_K0_is_a_free_parameter(self):
        """All ladder branches K_0..K_{2n-2} are exposed as free parameters
        (EST-SFS-style 2n-1 convention). Under the stationary prior K_0
        would be unidentifiable by detailed balance, but the
        AFS-informed Kingman prior breaks the invariance and K_0
        becomes identifiable in practice (matches EST-SFS's free fit
        to within sampling noise on shared inputs).
        """
        tree = OutgroupLadderTree(["i0", "i1"], ["o1", "o2"])
        assert "K0" in tree.param_names
        assert tree.n_params == 3  # 2n-1 for n=2
        tree3 = OutgroupLadderTree(["i0"], ["o1", "o2", "o3"])
        assert "K0" in tree3.param_names
        assert tree3.n_params == 5  # 2n-1 for n=3

    def test_per_site_prior_reflects_sfs_class(self):
        """The Kingman prior's per-site log-probs reflect the ingroup SFS
        class, the per-site quantity FixedTreeInference evaluates lazily
        (``self.prior.log_probs(batch)``) during inference."""
        ingroup = ["i0", "i1", "i2", "i3"]
        sites = [
            # Site A: A=3, C=1 → P(A)=3/4
            Site(chrom="1", pos=1, alleles=("A", "C"),
                 tip_alleles={"i0": "A", "i1": "A", "i2": "A", "i3": "C",
                              "o1": "A", "o2": "A"}),
            # Site B: A=1, C=3 → P(A)=1/4
            Site(chrom="1", pos=2, alleles=("A", "C"),
                 tip_alleles={"i0": "A", "i1": "C", "i2": "C", "i3": "C",
                              "o1": "A", "o2": "A"}),
        ]
        p = np.exp(KingmanIngroupWeight(ingroup).log_probs(sites))
        assert p[0, STATE_INDEX["A"]] == pytest.approx(0.75)
        assert p[0, STATE_INDEX["C"]] == pytest.approx(0.25)
        assert p[1, STATE_INDEX["A"]] == pytest.approx(0.25)
        assert p[1, STATE_INDEX["C"]] == pytest.approx(0.75)


# ----------------------------------------------------- AdaptiveIngroupWeight


class TestAdaptiveBeforeFit:
    def test_defaults_to_kingman_baseline(self):
        """Before fit(), Adaptive == Kingman per-site log-probs."""
        ingroup = [f"i{i}" for i in range(10)]
        adaptive = AdaptiveIngroupWeight(ingroup)
        kingman = KingmanIngroupWeight(ingroup)
        sites = [
            Site(chrom="1", pos=1, alleles=("A", "C"),
                 tip_alleles={**{f"i{i}": "A" for i in range(7)},
                              **{f"i{i}": "C" for i in range(7, 10)}}),
            Site(chrom="1", pos=2, alleles=("A", "C"),
                 tip_alleles={**{f"i{i}": "A" for i in range(5)},
                              **{f"i{i}": "C" for i in range(5, 10)}}),
        ]
        np.testing.assert_allclose(
            adaptive.log_probs(sites), kingman.log_probs(sites),
        )

    def test_fitted_flag(self):
        ingroup = ["i0", "i1"]
        adaptive = AdaptiveIngroupWeight(ingroup)
        assert adaptive.fitted is False

    def test_empty_ingroup_raises(self):
        with pytest.raises(ValueError, match="at least one ingroup sample"):
            AdaptiveIngroupWeight([])


class TestAdaptiveFit:
    def test_fit_via_inference_updates_pi(self):
        """FixedTreeInference.fit() drives prior.fit() and updates π."""
        ingroup = [f"i{i}" for i in range(8)]
        outgroup = ["o1", "o2"]
        truth_tree = OutgroupLadderTree(ingroup, outgroup)
        truth_tree.set_params(np.array([0.0, 0.05, 0.05]))

        rng = np.random.default_rng(33)
        P = {n: JC69().transition_probs(truth_tree.branch_length(n))
             for n in truth_tree.postorder() if n != truth_tree.root}
        P_ingroup = JC69().transition_probs(0.05)
        sites: list[Site] = []
        for i in range(1500):
            root = int(rng.integers(0, 4))
            state = {truth_tree.root: root}
            stack = [truth_tree.root]
            while stack:
                p_node = stack.pop()
                for c in truth_tree.children(p_node):
                    state[c] = int(rng.choice(4, p=P[c][state[p_node]]))
                    stack.append(c)
            tip_alleles = {
                sid: STATES[state[truth_tree.tip_for_sample(sid)]]
                for sid in outgroup
            }
            for sid in ingroup:
                tip_alleles[sid] = STATES[int(rng.choice(4, p=P_ingroup[root]))]
            sites.append(Site(
                chrom="1", pos=i + 1,
                alleles=tuple(sorted({v for v in tip_alleles.values() if v})),
                tip_alleles=tip_alleles,
            ))

        tree = OutgroupLadderTree(ingroup, outgroup)
        adaptive = AdaptiveIngroupWeight(ingroup, n_runs=2, seed=7)
        kingman_before = dict(adaptive.pi)
        FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            ingroup_weight=adaptive, n_starts=1, parallelize=False,
        ).fit()

        assert adaptive.fitted is True
        # At least one fittable bin should have updated to a non-Kingman value
        n = 8
        any_updated = any(
            abs(adaptive.pi[i] - kingman_before[i]) > 1e-3
            for i in range(1, (n + 1) // 2)
        )
        assert any_updated, "Adaptive.fit did not move any π_i from the Kingman default"
        # All π_i still in [0, 1]
        for i, pi in adaptive.pi.items():
            assert 0.0 <= pi <= 1.0, f"π_{i}={pi} out of [0, 1]"
        # Folded symmetry preserved: π_{n-i} == 1 - π_i for fittable bins
        for i in range(1, (n + 1) // 2):
            assert adaptive.pi[n - i] == pytest.approx(1 - adaptive.pi[i])


class TestAdaptiveUnstablePiWarning:
    """_warn_on_unstable_pi fires when per-bin n is low or fits peg at bounds."""

    @staticmethod
    def _make_inference_with_sites(sites, ingroup, outgroup, *, min_bin_n_sites=20):
        from ancestree import FixedTreeInference, JC69
        tree = OutgroupLadderTree(ingroup, outgroup)
        prior = AdaptiveIngroupWeight(
            ingroup, n_runs=2, seed=0, min_bin_n_sites=min_bin_n_sites,
        )
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_target_sites=len(sites),
            ingroup_weight=prior, n_starts=1, parallelize=False,
        )
        return inf, prior

    def test_warns_on_low_per_bin_counts(self, caplog):
        """8 ingroup haps × 100 sites → most bins below default min=20."""
        ingroup = [f"i{i}" for i in range(8)]
        outgroup = ["o1", "o2"]
        truth_tree = OutgroupLadderTree(ingroup, outgroup)
        truth_tree.set_params(np.array([0.0, 0.05, 0.05]))
        sites = _simulate_sites_with_ingroup_sfs(
            truth_tree, JC69(), 100,
            ingroup_branch_length=0.05, seed=11,
        )
        inf, _ = self._make_inference_with_sites(sites, ingroup, outgroup)
        with caplog.at_level(logging.WARNING):
            inf.fit()
        assert "noisy on this dataset" in caplog.text

    def test_warning_names_problem_bins(self, caplog):
        """Warning text should specify which bins are problematic."""
        ingroup = [f"i{i}" for i in range(8)]
        outgroup = ["o1", "o2"]
        truth_tree = OutgroupLadderTree(ingroup, outgroup)
        truth_tree.set_params(np.array([0.0, 0.05, 0.05]))
        sites = _simulate_sites_with_ingroup_sfs(
            truth_tree, JC69(), 100,
            ingroup_branch_length=0.05, seed=11,
        )
        inf, _ = self._make_inference_with_sites(sites, ingroup, outgroup)
        with caplog.at_level(logging.WARNING):
            inf.fit()
        # Find the AdaptiveIngroupWeight noisy-bins warning. Other
        # warnings (e.g. the post-fit branch-rate-bound warning when K_0
        # hits 1e-9 on this tiny dataset) may fire as well. Filter to
        # the Adaptive prior message under test.
        adaptive_msgs = [
            r.getMessage() for r in caplog.records
            if "min_bin_n_sites" in r.getMessage()
            or "pegged" in r.getMessage()
        ]
        assert adaptive_msgs, (
            f"no AdaptiveIngroupWeight diagnostic warning in: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        msg = adaptive_msgs[0]
        # Must explain WHY it fired (low n or pegged) and reference Kingman
        # as the suggested fallback.
        assert "min_bin_n_sites" in msg or "pegged" in msg
        assert "KingmanIngroupWeight" in msg

    # NOTE on the "silent" path:
    # The test simulator here does not generate a clean Kingman SFS, so
    # silence cannot be asserted reliably at any n_sites threshold. The silent
    # case is exercised empirically by the report_pi_comparison snakemake
    # rule (n_ingroup=10, L=2Mb, ~1100 sites in fittable bins, all bins
    # above the default 20-site threshold), no warning fires there.


class TestAdaptiveParallelMatchesSequential:
    """parallelize=True dispatches per-bin L-BFGS-B across workers but
    must return the same fitted π_i as sequential (per-bin seeds are
    deterministic from self.seed)."""

    @pytest.mark.slow
    def test_parallel_matches_sequential(self):
        ingroup = [f"i{i}" for i in range(10)]
        outgroup = ["o1", "o2"]
        truth_tree = OutgroupLadderTree(ingroup, outgroup)
        truth_tree.set_params(np.array([0.0, 0.025, 0.025]))

        sites = _simulate_sites_with_ingroup_sfs(
            truth_tree, JC69(), 600,
            ingroup_branch_length=0.05, seed=99,
        )

        # Same data + same seed → fits must be identical regardless of mode.
        from ancestree import FixedTreeInference, JC69 as _JC
        seq_prior = AdaptiveIngroupWeight(
            ingroup, n_runs=4, seed=42, parallelize=False,
        )
        par_prior = AdaptiveIngroupWeight(
            ingroup, n_runs=4, seed=42, parallelize=True,
        )
        for prior in (seq_prior, par_prior):
            tree = OutgroupLadderTree(ingroup, outgroup)
            FixedTreeInference(
                sites, _JC(), _no_counts(), tree=tree,
                n_target_sites=len(sites),
                ingroup_weight=prior, n_starts=1, parallelize=False,
            ).fit()
        assert seq_prior.pi == par_prior.pi


class TestAdaptiveMultiAllelicFallback:
    def test_multi_allelic_uses_kingman_baseline(self):
        """Adaptive's per-bin params apply only to biallelic. Multi-allelic
        sites use the count-proportional baseline regardless of fit."""
        ingroup = [f"i{i}" for i in range(10)]
        adaptive = AdaptiveIngroupWeight(ingroup)
        # Pretend a fit happened (mutate π for bin 3 to a non-Kingman value)
        adaptive._pi[3] = 0.99
        # A multi-allelic site (3 segregating), Adaptive should ignore the
        # bin-3 entry and return count-proportional probs.
        alleles_per = ["A"] * 6 + ["C"] * 3 + ["G"]
        site = Site(
            chrom="1", pos=1, alleles=("A", "C", "G"),
            tip_alleles={f"i{i}": a for i, a in enumerate(alleles_per)},
        )
        p = np.exp(adaptive.log_probs([site])[0])
        assert p[STATE_INDEX["A"]] == pytest.approx(0.6)
        assert p[STATE_INDEX["C"]] == pytest.approx(0.3)
        assert p[STATE_INDEX["G"]] == pytest.approx(0.1)


class TestSubsampleSizeDefault:
    """Default subsample_size = min(n_ingroup, 11) with INFO log on first fit."""

    def test_default_caps_at_eleven(self):
        prior = AdaptiveIngroupWeight([f"i{i}" for i in range(40)])
        assert prior.subsample_size == 11

    def test_default_matches_small_ingroup(self):
        prior = AdaptiveIngroupWeight([f"i{i}" for i in range(8)])
        assert prior.subsample_size == 8

    def test_explicit_too_large_raises(self):
        with pytest.raises(ValueError, match=r"subsample_size must be in"):
            AdaptiveIngroupWeight(
                [f"i{i}" for i in range(10)], subsample_size=20,
            )

    def test_explicit_too_small_raises(self):
        with pytest.raises(ValueError, match=r"subsample_size must be in"):
            AdaptiveIngroupWeight(
                [f"i{i}" for i in range(10)], subsample_size=1,
            )

    def test_default_emits_info_log(self, caplog):
        """The first fit() with a default-derived subsample_size logs an INFO line."""
        ingroup = [f"i{i}" for i in range(40)]
        prior = AdaptiveIngroupWeight(ingroup)
        # Build a tiny sites array. The fit content does not matter for the log.
        sites: list[Site] = []
        log_L = np.zeros((0, 4))
        with caplog.at_level(logging.INFO, logger="ancestree"):
            prior.fit(sites, log_L)
        assert any("subsample size defaulted" in r.message for r in caplog.records)

    def test_explicit_size_suppresses_info_log(self, caplog):
        ingroup = [f"i{i}" for i in range(40)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=11)
        with caplog.at_level(logging.INFO, logger="ancestree"):
            prior.fit([], np.zeros((0, 4)))
        assert not any("subsample size defaulted" in r.message for r in caplog.records)


class TestSubsampleProjection:
    """Hypergeometric projection in fit() and _log_prior_one()."""

    def test_full_obs_site_with_ssize_eq_nobs_is_deterministic(self):
        """When n_obs == subsample_size, hypergeometric weights collapse to a delta:
        the site contributes weight 1 to its observed bin and 0 elsewhere."""
        ingroup = [f"i{i}" for i in range(8)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=8)
        # 8 ingroup haps, 2 minor → bin 2 in subsample space.
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={**{f"i{i}": "A" for i in range(6)},
                         **{f"i{i}": "C" for i in range(6, 8)}},
        )
        log_L = np.zeros((1, 4))
        log_L[0, STATE_INDEX["A"]] = np.log(0.7)
        log_L[0, STATE_INDEX["C"]] = np.log(0.3)
        prior.fit([site], log_L)
        # Bin 2 got the full weight. Other (fittable) bins got nothing.
        assert prior.n_sites_per_bin.get(2, 0) == pytest.approx(1.0)
        assert prior.n_sites_per_bin.get(1, 0) == pytest.approx(0.0)

    def test_projection_spreads_mass_across_reachable_bins(self):
        """A site with n_obs > subsample_size spreads contribution across
        bins reachable by hypergeometric subsampling."""
        ingroup = [f"i{i}" for i in range(20)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=11)
        # 20 ingroup haps, 5 minor, projected to 11 lands in bins 0..5.
        # Hypergeometric: hypergeom.pmf(j; M=20, n=5, N=11) for j in 0..5.
        tip = {f"i{i}": "A" for i in range(15)}
        tip.update({f"i{i}": "C" for i in range(15, 20)})
        site = Site(chrom="1", pos=1, alleles=("A", "C"), tip_alleles=tip)
        log_L = np.zeros((1, 4))
        prior.fit([site], log_L)
        # Total weight summed across bins = 1 (probability mass conserved).
        total = sum(prior.n_sites_per_bin.values())
        assert total == pytest.approx(1.0, rel=1e-6)
        # Mass spread across bins 0..5 (some may be tiny).
        assert sum(prior.n_sites_per_bin.get(j, 0) for j in range(6)) == pytest.approx(1.0, rel=1e-6)

    def test_low_obs_site_falls_back_to_kingman(self):
        """Sites with n_obs < subsample_size are excluded from the fit
        and counted in n_sites_fallback_kingman."""
        ingroup = [f"i{i}" for i in range(20)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=11)
        # Construct a site where only 5 ingroups are called (n_obs=5 < 11).
        tip: dict[str, str | None] = {f"i{i}": None for i in range(20)}
        tip["i0"] = tip["i1"] = tip["i2"] = "A"
        tip["i3"] = tip["i4"] = "C"
        site = Site(chrom="1", pos=1, alleles=("A", "C"), tip_alleles=tip)
        prior.fit([site], np.zeros((1, 4)))
        assert prior.n_sites_fallback_kingman == 1
        assert sum(prior.n_sites_per_bin.values()) == pytest.approx(0.0)

    def test_apply_low_obs_uses_count_proportional(self):
        """A site with n_obs < subsample_size receives count-proportional
        Kingman fallback at apply time (subsample-invariant)."""
        ingroup = [f"i{i}" for i in range(20)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=11)
        # 3 A + 2 C observed, rest missing → 3/5 and 2/5.
        tip: dict[str, str | None] = {f"i{i}": None for i in range(20)}
        for i in range(3):
            tip[f"i{i}"] = "A"
        for i in range(3, 5):
            tip[f"i{i}"] = "C"
        site = Site(chrom="1", pos=1, alleles=("A", "C"), tip_alleles=tip)
        log_prior = prior.log_probs([site])[0]
        assert log_prior[STATE_INDEX["A"]] == pytest.approx(np.log(3 / 5))
        assert log_prior[STATE_INDEX["C"]] == pytest.approx(np.log(2 / 5))


class TestSubsampleApplyConsistency:
    """The fit-then-apply round trip for a single site with n_obs==subsample_size
    yields the same prior values as the pre-subsampling code path."""

    def test_full_obs_apply_matches_pi_directly(self):
        ingroup = [f"i{i}" for i in range(8)]
        prior = AdaptiveIngroupWeight(ingroup, subsample_size=8)
        # 8 ingroup, 3 minor → bin 3. Mutate π for bin 3 to a non-Kingman value.
        prior._pi[3] = 0.85
        prior._pi[5] = 0.15  # symmetric pair
        prior.fitted = True
        tip = {**{f"i{i}": "A" for i in range(5)},
               **{f"i{i}": "C" for i in range(5, 8)}}
        site = Site(chrom="1", pos=1, alleles=("A", "C"), tip_alleles=tip)
        log_prior = prior.log_probs([site])[0]
        # hypergeom.pmf with n_obs == subsample_size collapses to a delta at
        # j == k → projection gives exactly π_3 for major (A) and (1-π_3) for minor (C).
        assert log_prior[STATE_INDEX["A"]] == pytest.approx(np.log(0.85))
        assert log_prior[STATE_INDEX["C"]] == pytest.approx(np.log(0.15))


def test_adaptive_prior_folds_upper_half_bins():
    from ancestree.priors import AdaptiveIngroupWeight

    ingroup = [f"i{k}" for k in range(7)]
    prior = AdaptiveIngroupWeight(ingroup, subsample_size=5, seed=0)
    # 4 major (A) + 3 minor (T) among 7 ingroup: the hypergeometric projection
    # to a subsample of 5 spreads mass up to bin j=3, which is the upper half.
    tip = {f"i{k}": ("A" if k < 4 else "T") for k in range(7)}
    site = Site(chrom="1", pos=1, alleles=("A", "T"), tip_alleles=tip)
    logL = np.full((1, 4), np.log(1e-9))
    logL[0, STATE_INDEX["A"]] = np.log(0.7)  # p_major
    logL[0, STATE_INDEX["T"]] = np.log(0.1)  # p_minor

    prior.begin_fit()
    prior.accumulate([site], logL)

    # No upper-half bin (j >= (5+1)//2 = 3) is left as its own fit bin.
    assert all(j < (5 + 1) // 2 for j in prior._acc_bins), dict(prior._acc_bins)
    # Bin 3's mass folded into its mirror bin 2 with p_major / p_minor swapped.
    folded = [
        k for k in prior._acc_bins[2]
        if abs(k[0] - 0.1) < 1e-9 and abs(k[1] - 0.7) < 1e-9
    ]
    assert folded, "upper-bin mass was not folded into the lower mirror bin"


def test_adaptive_warning_pools_mirror_bin(caplog):
    # The instability warning uses the pooled (bin + fold mirror) sample size,
    # so a bin whose single weight is below the threshold but whose pooled
    # weight is above it does not fire a spurious warning.
    import logging

    from ancestree.priors import AdaptiveIngroupWeight

    prior = AdaptiveIngroupWeight(
        [f"i{k}" for k in range(10)], subsample_size=10, min_bin_n_sites=20, seed=0,
    )
    prior.min_bin_n_sites = 20
    prior._n_sites_per_bin = {j: 12.0 for j in range(11)}  # single-bin 12 < 20
    prior._pi = {j: 0.5 for j in range(11)}  # interior, not pegged
    with caplog.at_level(logging.WARNING):
        prior._warn_on_unstable_pi()
    assert "min_bin_n_sites" not in caplog.text  # 12 + 12 (mirror) = 24 >= 20


def _biallelic_sites(n_ingroup, n_sites, seed):
    """Build ``n_sites`` biallelic ingroup-only sites with varied minor counts.

    Each site partitions the ``n_ingroup`` haplotypes into a "major" (``A``)
    and "minor" (``C``) block. The minor count cycles through ``1..n-1`` so
    several SFS bins get populated. ``count_alleles`` sees exactly two alleles
    per site, so every site enters the per-bin fit.
    """
    rng = np.random.default_rng(seed)
    sites = []
    for i in range(n_sites):
        n_minor = 1 + (i % (n_ingroup - 1))
        tip_alleles = {}
        for h in range(n_ingroup):
            tip_alleles[f"i{h}"] = "C" if h >= n_ingroup - n_minor else "A"
        sites.append(Site(
            chrom="1", pos=i + 1, alleles=("A", "C"), tip_alleles=tip_alleles,
        ))
    # Random but fixed per-site, per-state log-likelihoods feeding the mixture.
    log_L = rng.uniform(-6.0, 0.0, size=(n_sites, len(STATES)))
    return sites, log_L


class TestAdaptiveStreamedFit:
    def test_streamed_equals_all_at_once(self):
        """begin_fit / accumulate (2 batches) / end_fit equals a single fit(), bit-identical."""
        ingroup = [f"i{h}" for h in range(10)]
        sites, log_L = _biallelic_sites(10, 240, seed=11)

        # All-at-once.
        prior_all = AdaptiveIngroupWeight(ingroup, n_runs=3, seed=5)
        prior_all.fit(sites, log_L)

        # Streamed in two contiguous batches (same overall site order, so the
        # dedup accumulators receive keys in identical first-seen order).
        k = 137
        prior_stream = AdaptiveIngroupWeight(ingroup, n_runs=3, seed=5)
        prior_stream.begin_fit()
        prior_stream.accumulate(sites[:k], log_L[:k])
        prior_stream.accumulate(sites[k:], log_L[k:])
        prior_stream.end_fit()

        assert prior_stream.fitted is True
        assert prior_all.fitted is True
        # Bit-identical π across every bin.
        assert prior_stream.pi == prior_all.pi
        # Per-bin site mass and fallback count also match exactly.
        assert prior_stream.n_sites_per_bin == prior_all.n_sites_per_bin
        assert prior_stream.n_sites_fallback_kingman == prior_all.n_sites_fallback_kingman


class TestFitPiBinEmpty:
    def test_empty_data_returns_symmetric_default(self):
        """_fit_pi_bin([], ...) short-circuits to the 0.5 symmetric default."""
        assert _fit_pi_bin([], n_runs=4, seed=0) == 0.5
        # Independent of n_runs / seed since it never enters the optimiser.
        assert _fit_pi_bin([], n_runs=1, seed=99) == 0.5


def test_the_ingroup_weight_projects_a_subsample():
    """The hypergeometric projection is what a sub-sample size means.

    A point mass at the observed count would reverse the direction of the
    prior at intermediate frequencies, so this is asserted at
    ``n_obs > subsample_size``, where the projection is not an identity.
    """
    ingroup = [f"i{k}" for k in range(20)]
    weight = AdaptiveIngroupWeight(ingroup_samples=ingroup, subsample_size=6)
    alleles = ("A", "C")
    site = Site(chrom="1", pos=1, alleles=alleles,
                tip_alleles={name: (alleles[1] if k < 5 else alleles[0])
                             for k, name in enumerate(ingroup)})
    logp = np.asarray(weight.log_probs([site])[0])
    probs = np.exp(logp - logp.max())
    probs /= probs.sum()
    major, minor = probs[0], probs[1]
    assert major > minor, (
        f"the majority allele is disfavoured (major {major:.3f} against "
        f"minor {minor:.3f}); the sub-sample projection is not applied")
    assert major == pytest.approx(0.75, rel=1e-3)


def test_a_thin_bin_is_named_in_the_unstable_pi_warning(caplog):
    """The low-count clause fires on its own, not only beside a pegged bin."""
    ingroup = [f"i{k}" for k in range(8)]
    weight = AdaptiveIngroupWeight(ingroup_samples=ingroup, subsample_size=8,
                                   min_bin_n_sites=20)
    alleles = ("A", "C")
    sites = []
    for pos in range(6):
        minor = 1 + (pos % 3)
        sites.append(Site(
            chrom="1", pos=pos, alleles=alleles,
            tip_alleles={n: (alleles[1] if k < minor else alleles[0])
                         for k, n in enumerate(ingroup)}))
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        try:
            weight.fit(sites, np.zeros((len(sites), 4)))
        except Exception:                     # fitting may decline outright
            pass
    assert "min_bin_n_sites" in caplog.text, (
        f"no low-count warning naming min_bin_n_sites; got {caplog.text!r}")


class TestPriorReprsAndGuards:
    def test_the_base_ingroup_weight_has_no_weight(self):
        _, ids = panel_ts()
        with pytest.raises(NotImplementedError):
            IngroupWeight().log_probs([panel_site(ids)])

    def test_reprs_name_the_identifying_fields(self):
        assert repr(StationaryPrior(JC69())) == "StationaryPrior(model=JC69)"
        assert repr(KingmanIngroupWeight(["a", "b"])) == (
            "KingmanIngroupWeight(n_ingroup=2)")
        assert repr(AdaptiveIngroupWeight(["a", "b", "c"], subsample_size=2)) == (
            "AdaptiveIngroupWeight(n_ingroup=3, subsample_size=2)")
        assert "rates=[1, 2, 3, 4, 5, 6]" in repr(GTR(rates=[1, 2, 3, 4, 5, 6]))


def _sites_with_minor_counts(ingroup, minor_counts) -> list[Site]:
    """One ``A/T`` site per entry of ``minor_counts``, both outgroups ``A``."""
    sites = []
    for pos, k in enumerate(minor_counts, start=1):
        tips = {s: ("T" if j < k else "A") for j, s in enumerate(ingroup)}
        tips.update({"o1": "A", "o2": "A"})
        sites.append(Site(chrom="1", pos=pos, alleles=("A", "T"),
                          tip_alleles=tips))
    return sites


class TestAdaptiveFitFallbacks:
    ingroup = [f"i{k}" for k in range(8)]

    @staticmethod
    def _log_L(n_sites: int) -> np.ndarray:
        """Per-site log-likelihoods favouring ``A`` over ``T``."""
        row = np.log([0.6, 0.1, 0.1, 0.2])
        return np.repeat(row[None, :], n_sites, axis=0)

    def test_a_pool_that_may_not_fork_falls_back_to_the_serial_fit(
            self, monkeypatch, caplog):
        sites = _sites_with_minor_counts(self.ingroup, [1, 1, 2, 2, 3, 3, 1, 2])
        serial = AdaptiveIngroupWeight(self.ingroup, n_runs=2, seed=3)
        serial.fit(sites, self._log_L(len(sites)))

        monkeypatch.setattr("ancestree.priors.os.cpu_count", lambda: 4)
        monkeypatch.setattr(Settings, "_fork_is_safe", staticmethod(lambda: False))

        def _no_pool(*args, **kwargs):
            raise AssertionError("a process pool was started despite the fallback")

        monkeypatch.setattr("ancestree.priors.ProcessPoolExecutor", _no_pool)
        parallel = AdaptiveIngroupWeight(self.ingroup, n_runs=2, seed=3,
                                         parallelize=True)
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            parallel.fit(sites, self._log_L(len(sites)))
        assert any("Ignoring the parallelization request" in r.message
                   for r in caplog.records)
        assert parallel.fitted
        assert parallel.pi == serial.pi

    def test_a_subsample_of_two_has_no_bin_to_fit_or_warn_about(self, caplog):
        prior = AdaptiveIngroupWeight(self.ingroup[:2], subsample_size=2)
        sites = _sites_with_minor_counts(self.ingroup[:2], [1, 1, 1])
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            prior.fit(sites, self._log_L(len(sites)))
        assert prior.fitted
        # The two endpoints are deterministic and the middle bin is its own
        # fold mirror, so the Kingman default survives the fit untouched.
        assert prior.pi == {0: 1.0, 1: 0.5, 2: 0.0}
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
