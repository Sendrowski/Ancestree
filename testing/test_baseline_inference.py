"""Tests for the rule-based sanity-check baseline inference.

Covers :class:`MajorityOutgroupInference` (the baseline itself) and the
``baseline_check=True`` plumbing on :class:`FixedTreeInference` /
:class:`ARGBasedInference`.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from ancestree import (
    ARGBasedInference,
    FixedTreeInference,
    JC69,
    MajorityOutgroupInference,
    OutgroupLadderTree,
    Site,
)
from ancestree.inference import Inference

from testing._helpers import QUICKSTART_TREES
from testing._helpers import no_counts as _no_counts, post


# ---------------------------------------------------------------- the rule itself


class TestMajorityOutgroupRule:
    """The deterministic per-site behaviour of the baseline rule."""

    def test_unanimous_outgroups_yield_their_allele_as_map(self):
        site = Site(
            chrom="1", pos=1, alleles=("A", "T"),
            tip_alleles={"o1": "A", "o2": "A", "o3": "A"},
        )
        inf = MajorityOutgroupInference(
            [site], ["o1", "o2", "o3"], for_comparison_only=True,
        )
        ((_, post),) = list(inf.infer())
        assert post.map_allele == "A"
        assert post["A"] == pytest.approx(0.95)
        # remaining 0.05 spread evenly across the other 3 states
        assert post["C"] == pytest.approx(0.05 / 3)
        assert post["G"] == pytest.approx(0.05 / 3)
        assert post["T"] == pytest.approx(0.05 / 3)
        assert np.isclose(post.values.sum(), 1.0)

    def test_majority_wins_when_one_dissenter(self):
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"o1": "A", "o2": "A", "o3": "G"},
        )
        inf = MajorityOutgroupInference(
            [site], ["o1", "o2", "o3"], for_comparison_only=True,
        )
        ((_, post),) = list(inf.infer())
        assert post.map_allele == "A"

    def test_tie_break_picks_closest_outgroup_allele(self):
        """Two-way 1-1 tie: the allele carried by the *first* listed (= closest) outgroup wins."""
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"o_close": "G", "o_far": "A"},
        )
        # o_close lists first → its allele G wins the tie.
        inf_close_first = MajorityOutgroupInference(
            [site], ["o_close", "o_far"], for_comparison_only=True,
        )
        ((_, post_g),) = list(inf_close_first.infer())
        assert post_g.map_allele == "G"
        # Reverse the order, so o_far is "closest" by the rule and A wins.
        inf_far_first = MajorityOutgroupInference(
            [site], ["o_far", "o_close"], for_comparison_only=True,
        )
        ((_, post_a),) = list(inf_far_first.infer())
        assert post_a.map_allele == "A"

    def test_tie_break_reads_every_haplotype_of_a_diploid_outgroup(self):
        """A tie must be broken by the earliest outgroup that carries a tied
        allele on any of its haplotypes, not by alphabetical order.

        The counting rule matches every haplotype of a named individual and
        skips alleles outside A/C/G/T, so a diploid outgroup whose first
        haplotype is uncalled still contributes its second haplotype to the
        counts. A tie-break that inspects only the first matching haplotype
        finds an uncalled allele, matches no outgroup at all, and falls
        through to the alphabetically first tied allele, contradicting the
        rule the class documents.
        """
        # o0 and o1 are diploid, each with an uncalled first haplotype, so the
        # counts are one A and one C and both alleles are tied.
        panel_a = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"o0_h0": "N", "o0_h1": "A",
                         "o1_h0": "N", "o1_h1": "C"},
        )
        ((_, post_a),) = list(MajorityOutgroupInference(
            [panel_a], ["o0", "o1"], for_comparison_only=True,
        ).infer())
        assert post_a.map_allele == "A"
        # Swapping the two alleles between the individuals moves the winner to
        # C, which alphabetical fallback ordering would never select.
        panel_c = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"o0_h0": "N", "o0_h1": "C",
                         "o1_h0": "N", "o1_h1": "A"},
        )
        ((_, post_c),) = list(MajorityOutgroupInference(
            [panel_c], ["o0", "o1"], for_comparison_only=True,
        ).infer())
        assert post_c.map_allele == "C"
        assert post_c["C"] == pytest.approx(0.95)
        # Listing o1 first makes A the earliest carried tied allele.
        ((_, post_swapped),) = list(MajorityOutgroupInference(
            [panel_c], ["o1", "o0"], for_comparison_only=True,
        ).infer())
        assert post_swapped.map_allele == "A"

    def test_no_outgroup_data_yields_uniform(self):
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"o1": None, "o2": None},
        )
        inf = MajorityOutgroupInference(
            [site], ["o1", "o2"], for_comparison_only=True,
        )
        ((_, post),) = list(inf.infer())
        assert np.allclose(post.values, 0.25)

    def test_confidence_kwarg_controls_map_mass(self):
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"o1": "A"},
        )
        inf = MajorityOutgroupInference(
            [site], ["o1"], confidence=0.7, for_comparison_only=True,
        )
        ((_, post),) = list(inf.infer())
        assert post["A"] == pytest.approx(0.7)
        assert post["C"] == pytest.approx(0.3 / 3)


class TestMajorityOutgroupConfidence:
    def test_confidence_above_one(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            MajorityOutgroupInference([], [], confidence=1.5)

    def test_confidence_zero(self):
        with pytest.raises(ValueError, match="confidence must be in"):
            MajorityOutgroupInference([], [], confidence=0.0)


# ----------------------------- majority-outgroup no-outgroup fallback --------
class TestMajorityNoOutgroupFallback:
    def _call(self, site):
        inf = MajorityOutgroupInference(
            [site], outgroup_samples=[], ingroup_samples=["i1", "i2", "i3", "i4"],
            for_comparison_only=True,
        )
        ((_, post),) = list(inf.infer())
        return post

    def test_unique_major_ingroup_allele(self):
        post = self._call(Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"i1": "A", "i2": "A", "i3": "A", "i4": "G"},
        ))
        assert post.map_allele == "A"
        assert post["A"] == pytest.approx(0.95)

    def test_tie_spreads_uniform_over_tied(self):
        post = self._call(Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"i1": "A", "i2": "A", "i3": "G", "i4": "G"},
        ))
        assert post["A"] == pytest.approx(0.5)
        assert post["G"] == pytest.approx(0.5)
        assert post["C"] == 0.0

    def test_no_called_ingroup_alleles_is_uniform(self):
        post = self._call(Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"i1": "N", "i2": "N", "i3": "N", "i4": "N"},
        ))
        np.testing.assert_allclose(post.values, 0.25, atol=1e-12)


# ---------------------------------------------------------------- standalone warning


class TestStandaloneWarning:
    def test_for_comparison_only_false_emits_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            MajorityOutgroupInference([], ["o1"])
        assert "consistency check" in caplog.text

    def test_for_comparison_only_true_suppresses_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            MajorityOutgroupInference([], ["o1"], for_comparison_only=True)
        assert "consistency check" not in caplog.text


# --------------------------------------------------- baseline_check plumbing (FTI)


class TestFixedTreeBaselineCheck:
    """End-to-end: FixedTreeInference(baseline_check=True) emits the INFO log."""

    def _build(self, baseline_check: bool, caplog: pytest.LogCaptureFixture):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        # K_0 > 0: with a zero branch above it the ingroup tip would sit on the
        # deep node itself and its own allele would be the answer there.
        tree.set_params(np.array([0.02, 0.05, 0.05]))
        # A few hand-crafted polymorphic sites with unanimous outgroups.
        # Real inference (Stage-1 stationary prior under JC69) should pick the
        # outgroup's allele → 100% agreement with the baseline.
        sites = [
            Site(
                chrom="1", pos=1, alleles=("A", "T"),
                tip_alleles={"i1": "T", "o1": "A", "o2": "A"},
            ),
            Site(
                chrom="1", pos=2, alleles=("G", "C"),
                tip_alleles={"i1": "C", "o1": "G", "o2": "G"},
            ),
            Site(
                chrom="1", pos=3, alleles=("T", "A"),
                tip_alleles={"i1": "A", "o1": "T", "o2": "T"},
            ),
        ]
        from ancestree.focal import FocalNode
        inf = FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            # The ingroup is one haplotype, so every site is monoallelic to it;
            # the outgroups only carry the answer at the deeper node.
            focal=FocalNode("ingroup_mrca", fraction=1.0),
            n_starts=1, parallelize=False, fit_required=False,
            baseline_check=baseline_check,
        )
        return inf

    def test_baseline_check_true_logs_agreement(self, caplog):
        caplog.set_level(logging.INFO, logger="ancestree.inference")
        inf = self._build(baseline_check=True, caplog=caplog)
        # Drain the iterator to trigger the post-iteration hook.
        list(inf.infer())
        msgs = [r.message for r in caplog.records]
        baseline_lines = [m for m in msgs if "baseline" in m and "agreement" in m]
        assert baseline_lines, (
            f"Expected a 'baseline ... agreement' INFO line; got {msgs}"
        )
        assert "MajorityOutgroupInference" in baseline_lines[0]

    def test_baseline_check_false_silent(self, caplog):
        caplog.set_level(logging.INFO, logger="ancestree.inference")
        inf = self._build(baseline_check=False, caplog=caplog)
        list(inf.infer())
        assert not any(
            "baseline" in r.message and "agreement" in r.message
            for r in caplog.records
        )

    def test_baseline_agrees_when_outgroups_carry_the_signal(self, caplog):
        """When the real inference is well-specified and outgroups are unambiguous,
        agreement should be 100% on the toy sites."""
        caplog.set_level(logging.INFO, logger="ancestree.inference")
        inf = self._build(baseline_check=True, caplog=caplog)
        list(inf.infer())
        line = next(
            r.message for r in caplog.records
            if "baseline" in r.message and "agreement" in r.message
        )
        assert "1.000" in line, line


# ------------------------------------ baseline_check in ARG / local-tree modes


class TestGeneralizedBaselineCheck:
    """The sanity check generalised to ARG mode: off by default (no outgroup
    designation is required there), opt-in via ``outgroup_samples`` +
    ``baseline_check=True``."""

    @pytest.fixture(scope="class")
    @classmethod
    def arg_ts(cls):
        import msprime
        ts = msprime.sim_ancestry(
            samples=6, ploidy=1, sequence_length=2e4,
            recombination_rate=1e-8, population_size=1e4, random_seed=7,
        )
        return msprime.sim_mutations(ts, rate=1e-7, random_seed=7)

    def test_off_by_default(self, arg_ts, caplog):
        caplog.set_level(logging.INFO)
        list(ARGBasedInference(arg_ts, JC69(), mu=1e-7, progress=False).infer())
        assert not any("baseline" in r.message for r in caplog.records)

    def test_opt_in_logs_agreement(self, arg_ts, caplog):
        caplog.set_level(logging.INFO)
        names = [str(int(s)) for s in arg_ts.samples()]
        inf = ARGBasedInference(
            arg_ts, JC69(), mu=1e-7, progress=False,
            outgroup_samples=names[-2:], baseline_check=True,
        )
        list(inf.infer())
        assert any(
            "baseline" in r.message and "agreement" in r.message
            for r in caplog.records
        )

    def test_enabled_without_outgroups_skips(self, arg_ts, caplog):
        caplog.set_level(logging.INFO)
        inf = ARGBasedInference(
            arg_ts, JC69(), mu=1e-7, progress=False, baseline_check=True,
        )
        list(inf.infer())
        assert any(
            "baseline" in r.message.lower() and "skipped" in r.message
            for r in caplog.records
        )


class TestBaselineMemoryGate:
    """The baseline check stays bounded in memory: it only buffers when it
    will actually run (INFO on + outgroups), and caps the buffered sample."""

    def _inf(self, n_sites):
        tree = OutgroupLadderTree(["i1"], ["o1", "o2"])
        # K_0 > 0: with a zero branch above it the ingroup tip would sit on the
        # deep node itself and its own allele would be the answer there.
        tree.set_params(np.array([0.02, 0.05, 0.05]))
        sites = [
            Site(chrom="1", pos=i + 1, alleles=("A", "T"),
                 tip_alleles={"i1": "T", "o1": "A", "o2": "A"})
            for i in range(n_sites)
        ]
        return FixedTreeInference(
            sites, JC69(), _no_counts(), tree=tree,
            n_starts=1, parallelize=False, fit_required=False, baseline_check=True,
        )

    def test_info_disabled_does_not_run(self, caplog):
        """With INFO disabled the gate short-circuits, so the whole
        stream is not buffered and no baseline line is emitted."""
        inf = self._inf(6)
        with caplog.at_level(logging.WARNING, logger="ancestree.FixedTreeInference"):
            list(inf.infer())
        assert not any("baseline" in r.message for r in caplog.records)

    def test_buffer_capped_reports_leading_sample(self, caplog, monkeypatch):
        """Beyond the buffer cap the agreement is reported over a leading
        sample (so streaming memory stays bounded), and the line says so."""
        monkeypatch.setattr(FixedTreeInference, "_BASELINE_CHECK_MAX_SITES", 4)
        inf = self._inf(10)
        caplog.set_level(logging.INFO, logger="ancestree.FixedTreeInference")
        list(inf.infer())
        line = next(
            r.message for r in caplog.records
            if "baseline" in r.message and "agreement" in r.message
        )
        assert "leading 4-site sample" in line


# ------------------------------------------------- the agreement summary


def test_baseline_agreement_refuses_misaligned_streams():
    """Streams whose sites differ in position cannot be compared."""
    real = [(Site(chrom="1", pos=1, alleles=("A",), tip_alleles={}), post("A", 0.9))]
    baseline = [(Site(chrom="1", pos=2, alleles=("A",), tip_alleles={}), post("A", 0.9))]
    assert Inference._summarise_baseline_agreement(
        real, baseline, ingroup_samples=None) is None
    aligned = [(Site(chrom="1", pos=1, alleles=("A",), tip_alleles={}), post("A", 0.9))]
    assert "1.000 on 1" in Inference._summarise_baseline_agreement(
        real, aligned, ingroup_samples=None)


def test_baseline_without_designated_samples_names_none():
    """A mode without designated outgroups yields empty sample tuples, so the
    consistency check is skipped."""
    inf = MajorityOutgroupInference([], ["o1"], for_comparison_only=True)
    assert Inference._baseline_outgroup_samples(inf) == ()
    assert Inference._baseline_ingroup_samples(inf) == ()


def test_the_majority_outgroup_rule_is_graded_at_the_arg_root():
    """The rule resolves no ingroup, so its truth stays at the ARG root."""
    import tskit

    from ancestree.posterior import Grade
    from ancestree.sites import SiteSource

    ts = tskit.load(QUICKSTART_TREES)
    rule = MajorityOutgroupInference(
        list(SiteSource.resolve(ts)), ["o0", "o1"], for_comparison_only=True)
    assert rule.grade(ts) == Grade(rule.infer(), Grade.truth_mapping(ts))
