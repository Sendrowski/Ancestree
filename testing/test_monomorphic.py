"""Tests for BaseComposition and monomorphic-site likelihood support."""

import numpy as np
import pytest
import ancestree as anc
from ancestree.priors import StationaryPrior

from ancestree import (
    BaseComposition,
    JC69,
    OutgroupLadderTree,
    Site,
)

from testing._helpers import no_counts as _no_counts
from testing.test_likelihood import _ManualTree


def _two_tip_tree(branch_length: float) -> _ManualTree:
    """Tiny tree: root → (tipA, tipB), each child branch of length t."""
    return _ManualTree(
        children_map={2: [0, 1]},
        branch_lengths={0: branch_length, 1: branch_length},
        root_id=2,
        tip_to_sample={0: "tipA", 1: "tipB"},
    )


def _three_tip_tree(t_left: float, t_right: float, t_inner: float) -> _ManualTree:
    """Tree:  root → (tipA, inner → (tipB, tipC))."""
    return _ManualTree(
        children_map={4: [0, 3], 3: [1, 2]},
        branch_lengths={0: t_left, 1: t_right, 2: t_right, 3: t_inner},
        root_id=4,
        tip_to_sample={0: "tipA", 1: "tipB", 2: "tipC"},
    )


# --------------------------------------------------------------- BaseComposition


class TestBaseComposition:
    def test_from_counts(self):
        bc = BaseComposition.from_counts(A=1, C=2, G=3, T=4)
        assert dict(bc.counts) == {"A": 1, "C": 2, "G": 3, "T": 4}
        assert bc.n_total == 10

    def test_from_counts_defaults_zero(self):
        bc = BaseComposition.from_counts()
        assert bc.n_total == 0
        assert all(v == 0 for v in bc.counts.values())

    def test_no_counts_is_zero_placeholder(self):
        bc = _no_counts()
        assert bc.n_total == 0
        assert bc.counts == {"A": 0, "C": 0, "G": 0, "T": 0}

    def test_from_n_target_sites_uniform_default(self):
        bc = BaseComposition.from_n_target_sites(400)
        assert bc.counts == {"A": 100, "C": 100, "G": 100, "T": 100}
        assert bc.n_total == 400

    def test_from_n_target_sites_preserves_total_with_rounding(self):
        bc = BaseComposition.from_n_target_sites(7)
        assert bc.n_total == 7

    def test_from_n_target_sites_with_stationary(self):
        bc = BaseComposition.from_n_target_sites(
            1000, stationary=np.array([0.3, 0.2, 0.3, 0.2]),
        )
        assert bc.n_total == 1000
        assert bc.counts["A"] == 300
        assert bc.counts["C"] == 200
        assert bc.counts["G"] == 300

    def test_from_n_target_sites_rejects_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            BaseComposition.from_n_target_sites(-1)

    def test_from_n_target_sites_rejects_bad_stationary(self):
        with pytest.raises(ValueError, match="length-4"):
            BaseComposition.from_n_target_sites(100, stationary=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="sum to 1"):
            BaseComposition.from_n_target_sites(
                100, stationary=np.array([0.5, 0.5, 0.5, 0.5]),
            )

    def test_from_n_target_sites_skewed_stationary_no_negative_count(self):
        """Largest-remainder apportionment keeps every count non-negative and
        the total exact on a skewed simplex."""
        bc = BaseComposition.from_n_target_sites(
            50, stationary=np.array([0.215, 0.772, 0.013, 0.0]),
        )
        counts = [bc.counts[s] for s in ("A", "C", "G", "T")]
        assert all(c >= 0 for c in counts)
        assert sum(counts) == 50

    def test_from_n_target_sites_apportionment_is_exact_and_nonneg(self):
        """Counts sum to n_total and stay non-negative across skewed simplices."""
        for stat in ([0.97, 0.01, 0.01, 0.01], [0.0, 0.0, 0.5, 0.5],
                     [0.3, 0.3, 0.3, 0.1]):
            for n in (1, 7, 50, 999):
                bc = BaseComposition.from_n_target_sites(n, stationary=np.array(stat))
                c = [bc.counts[s] for s in ("A", "C", "G", "T")]
                assert all(x >= 0 for x in c) and sum(c) == n

    def test_validation_missing_keys(self):
        with pytest.raises(ValueError, match="missing"):
            BaseComposition(counts={"A": 1, "C": 2, "G": 3})

    def test_validation_extra_keys(self):
        with pytest.raises(ValueError, match="unexpected"):
            BaseComposition(counts={"A": 1, "C": 2, "G": 3, "T": 4, "N": 5})

    def test_validation_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            BaseComposition(counts={"A": -1, "C": 0, "G": 0, "T": 0})

    def test_from_fasta_single_contig(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">contig1\nACGTACGT\nACGTNN\n")
        bc = BaseComposition.from_fasta(f)
        # 3xA + 3xC + 3xG + 3xT, two Ns skipped
        assert bc.counts == {"A": 3, "C": 3, "G": 3, "T": 3}

    def test_from_fasta_case_insensitive(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">contig1\naCgTaCgT\n")
        bc = BaseComposition.from_fasta(f)
        assert bc.counts == {"A": 2, "C": 2, "G": 2, "T": 2}

    def test_from_fasta_multi_contig(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">contig1\nACGT\n>contig2\nAAAA\n")
        bc = BaseComposition.from_fasta(f)
        assert bc.counts == {"A": 5, "C": 1, "G": 1, "T": 1}

    def test_from_fasta_specific_contig(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">contig1\nAAAA\n>contig2\nCCCC\n")
        bc = BaseComposition.from_fasta(f, contig="contig2")
        assert bc.counts == {"A": 0, "C": 4, "G": 0, "T": 0}

    def test_from_fasta_missing_contig_raises(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">contig1\nAAAA\n")
        with pytest.raises(ValueError, match="does not contain contig 'nope'"):
            BaseComposition.from_fasta(f, contig="nope")

    def test_from_fasta_skips_iupac_and_gaps(self, tmp_path):
        f = tmp_path / "test.fasta"
        f.write_text(">x\nACGT-RYNW\n")
        bc = BaseComposition.from_fasta(f)
        assert bc.counts == {"A": 1, "C": 1, "G": 1, "T": 1}

    # 5. from_fasta tolerates a bare ">" header instead of crashing.
    def test_from_fasta_tolerates_empty_header(self, tmp_path):
        fa = tmp_path / "x.fa"
        fa.write_text(">\nACGT\n>chr1\nAAAA\n")
        bc = BaseComposition.from_fasta(str(fa))  # must not raise IndexError
        assert sum(dict(bc.counts).values()) == 8

    def test_base_composition_counts_exact_on_off_normalized_stationary(self):
        # A stationary vector summing slightly above 1 (within tolerance) must still
        # apportion to exactly n_total.
        from ancestree.sites import BaseComposition

        bc = BaseComposition.from_n_target_sites(
            100_000_000, stationary=np.array([0.25001, 0.25, 0.25, 0.25]),
        )
        assert sum(int(v) for v in dict(bc.counts).values()) == 100_000_000

    def test_a_zero_base_count_still_gives_a_usable_composition(self):
        """The floor keeps a state that was never observed representable.

        Without it an unobserved base gets probability zero, which is not merely
        a ``-inf`` log-prior: the substitution model refuses the composition
        outright, so the run dies rather than down-weighting the state.
        """
        composition = BaseComposition.from_counts(A=1000, C=0, G=500, T=800)
        pi = np.asarray(composition.pi)
        assert (pi > 0.0).all(), f"pi has a zero entry: {pi.tolist()}"
        prior = StationaryPrior(anc.F81(), composition)
        logp = np.asarray(prior.log_probs([Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": "A"})])[0])
        assert np.isfinite(logp).all(), f"non-finite log-prior: {logp.tolist()}"


# --------------------------------------------------------- BaseComposition.require


class TestBaseCompositionRequire:
    def test_passes_through(self):
        bc = BaseComposition.from_counts(A=1, C=1, G=1, T=1)
        assert BaseComposition.require(bc) is bc

    def test_raises_on_none(self):
        with pytest.raises(ValueError, match="requires a BaseComposition"):
            BaseComposition.require(None)

    def test_error_mentions_no_counts_placeholder(self):
        with pytest.raises(ValueError, match=r"\.no_counts\(\)"):
            BaseComposition.require(None)

    def test_no_counts_satisfies_require(self):
        bc = _no_counts()
        assert BaseComposition.require(bc) is bc

    def test_custom_context_label(self):
        with pytest.raises(ValueError, match="my custom mode"):
            BaseComposition.require(None, context="my custom mode")


# ---------------------------------------------------------------- Site.monomorphic


class TestSiteMonomorphic:
    def test_construction(self):
        s = Site.monomorphic(base="A", samples=["s1", "s2", "s3"])
        assert s.alleles == ("A",)
        assert s.tip_alleles == {"s1": "A", "s2": "A", "s3": "A"}

    def test_passes_through_optional_metadata(self):
        s = Site.monomorphic(
            base="C", samples=["s1"], chrom="chr1", pos=42,
            info={"src": "test"},
        )
        assert s.chrom == "chr1"
        assert s.pos == 42
        assert s.info == {"src": "test"}


# NOTE: TestLogLikelihoodMonomorphic was removed when
# Likelihood.log_likelihood_monomorphic + _log_L_uniform_leaves were
# deleted. The monomorphic contribution is now absorbed into the
# unified config-multiplicity fit (see BaseComposition.project_sites_to_configs)
# and exercised end-to-end by the B3 + B7 simulation benchmarks.


class TestCompositionSpansTheRegion:
    """A composition and ``n_target_sites`` are one convention.

    ``counts`` covers the whole target region and the monomorphic weight is
    derived from it against the polymorphic count, so handing a composition
    straight to the inference weights each polymorphic position once.
    """

    @staticmethod
    def _sites(n=120):
        rng = np.random.default_rng(3)
        ing = ["i0", "i1", "i2", "i3"]
        out = ["o0", "o1"]
        sites = []
        for k in range(n):
            a, b = ("A", "C") if k % 2 else ("G", "T")
            tips = {s: (a if rng.random() < 0.8 else b) for s in ing}
            tips.update({o: (a if rng.random() < 0.9 else b) for o in out})
            sites.append(Site(chrom="1", pos=k + 1, alleles=(a, b),
                                 tip_alleles=tips))
        return sites, ing, out

    def test_monomorphic_counts_net_the_polymorphic_sites(self):
        bc = BaseComposition.from_n_target_sites(1000)
        assert bc.n_total == 1000
        assert sum(bc.monomorphic_counts(40).values()) == 960

    def test_a_region_shorter_than_its_variants_is_refused(self):
        bc = BaseComposition.from_n_target_sites(100)
        with pytest.raises(ValueError, match="length of the target region"):
            bc.monomorphic_counts(500)

    def test_both_routes_give_the_same_config_weights(self):
        from ancestree.inference import FixedTreeInference

        sites, ing, out = self._sites()
        length = 50_000
        by_target = FixedTreeInference(
            sites, JC69(), BaseComposition.no_counts(),
            tree=OutgroupLadderTree(ing, out), n_target_sites=length,
            fit_required=False, progress=False,
        )
        by_composition = FixedTreeInference(
            sites, JC69(), BaseComposition.from_n_target_sites(length),
            tree=OutgroupLadderTree(ing, out),
            fit_required=False, progress=False,
        )
        left = {
            tuple(sorted(c.tip_alleles.items())): w
            for c, w in zip(by_target._fit_configs, by_target._fit_weights)
        }
        right = {
            tuple(sorted(c.tip_alleles.items())): w
            for c, w in zip(
                by_composition._fit_configs, by_composition._fit_weights)
        }
        assert set(left) == set(right)
        assert max(abs(left[k] - right[k]) for k in left) == 0.0


class TestConfigWeightsCountEachSiteOnce:
    """BaseComposition.counts is a whole-region count, not a monomorphic one.

    The public projection enters every polymorphic site once, as a projected
    config, and nets it out of the monomorphic boundary mass.
    """

    ING = [f"i{i}" for i in range(4)]
    OUT = ["o0", "o1"]

    def _sites(self):
        return [
            anc.Site(chrom="1", pos=10, alleles=("A", "C"),
                    tip_alleles={**{s: "A" for s in self.ING}, "i0": "C",
                                 "o0": "A", "o1": "A"}),
            anc.Site(chrom="1", pos=20, alleles=("C", "G"),
                    tip_alleles={**{s: "C" for s in self.ING}, "i1": "G",
                                 "o0": "C", "o1": "C"}),
        ]

    def test_the_weights_sum_to_the_region_size(self):
        bc = BaseComposition.from_counts(A=250, C=250, G=250, T=250)
        _, weights = bc.project_sites_to_configs(
            self._sites(), self.ING, self.OUT, 4)
        assert float(weights.sum()) == pytest.approx(float(bc.n_total))


