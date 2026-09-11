"""Site-level helpers that the inference layer depends on."""
from collections import Counter

import msprime
import numpy as np
import pytest

from ancestree import BaseComposition, Site
from ancestree.sites import SiteSource, SiteTable
from ancestree.sources import CyVCF2Source
from testing._helpers import QUICKSTART_TREES


class TestIndividualNamesMatchHaplotypes:
    """An id may name an individual or a haplotype, and must mean the same.

    A VCF reader splits a diploid into ``S0_h0`` / ``S0_h1``, but a user names
    individuals. Matching only exact tip ids made ``S0`` count nothing, so the
    ingroup weight fell back to uniform and put posterior mass on alleles no
    tip carried, with no error and no warning.
    """

    @staticmethod
    def _site():
        return Site(chrom="1", pos=1, alleles=("A", "C"),
                    tip_alleles={"i0_h0": "A", "i0_h1": "A",
                                 "i1_h0": "C", "i1_h1": "A"})

    def test_individual_id_counts_both_haplotypes(self):
        s = self._site()
        assert s.count_alleles(["i0"]) == s.count_alleles(["i0_h0", "i0_h1"])
        assert sum(s.count_alleles(["i0"]).values()) == 2

    def test_mixed_naming_agrees_with_full_haplotype_naming(self):
        s = self._site()
        assert (s.count_alleles(["i0", "i1_h0", "i1_h1"])
                == s.count_alleles(["i0_h0", "i0_h1", "i1_h0", "i1_h1"]))

    def test_an_unrelated_id_still_counts_nothing(self):
        """The match must not become so loose that a typo silently succeeds."""
        assert self._site().count_alleles(["i2"]) == Counter()

    def test_a_haplotype_id_does_not_pull_in_its_siblings(self):
        """Naming one haplotype counts that one only."""
        assert sum(self._site().count_alleles(["i0_h0"]).values()) == 1


class TestColumnarRoundTripKeepsMissingTips:
    """A rebuilt site keeps a key for every haplotype in the panel.

    ``Site.tip_alleles`` maps every sample to its allele or to ``None``, and
    every source backend emits it that way. Dropping the key for an uncalled
    haplotype made ``PolymorphicSiteFilter.accepts`` raise on a correctly
    spelled sample id, since its guard against a mistyped id tests membership.
    """

    NAMES = ("h0", "h1", "h2")

    @classmethod
    def _table(cls):
        from ancestree.sites import SiteTable
        site = Site(chrom="1", pos=30, alleles=("A", "C"),
                    tip_alleles={"h0": "A", "h1": None, "h2": "C"})
        return SiteTable.from_sites([site], cls.NAMES)

    def test_an_uncalled_tip_keeps_its_key(self):
        rebuilt = self._table()[0]
        assert tuple(rebuilt.tip_alleles) == self.NAMES
        assert rebuilt.tip_alleles["h1"] is None

    def test_the_ingroup_filter_accepts_a_panel_with_an_uncalled_tip(self):
        from ancestree import PolymorphicSiteFilter
        rebuilt = self._table()[0]
        assert PolymorphicSiteFilter(samples=list(self.NAMES)).accepts(rebuilt)


class TestPhaseArgumentsAreRefusedWhereTheyCannotApply:
    """A tree sequence carries its own phase, so accepting these discards them.

    ``sample_filter``, ``chrom_filter`` and ``ploidy`` were refused while
    ``phased`` and ``phase_seed`` were taken and dropped, which is the silent
    outcome the refusal exists to prevent.
    """

    @staticmethod
    def _ts():
        import tskit

        return tskit.load(QUICKSTART_TREES)

    @pytest.mark.parametrize("kwargs", [
        {"phased": True},
        {"phase_seed": 7},
        {"ploidy": 2},
    ])
    def test_a_tree_sequence_refuses_them(self, kwargs):
        from ancestree.sites import SiteSource

        with pytest.raises(ValueError, match="cannot be applied"):
            SiteSource.resolve(self._ts(), **kwargs)

    def test_a_tree_sequence_with_none_of_them_resolves(self):
        from ancestree.sites import SiteSource

        assert SiteSource.resolve(self._ts()) is not None


class TestAlleleCategories:
    """Three allele categories: A/C/G/T, a no-call, and everything else.

    An allele outside A/C/G/T was one undifferentiated bucket, so a site whose
    every allele fell in it was scored at the prior and its four-way tie broke
    to ``"A"``, which was then written as the ancestral state at 0.25
    confidence at a site no sample carried ``A`` at.
    """

    @staticmethod
    def _site(alleles, tip_alleles):
        return Site(chrom="1", pos=1, alleles=alleles, tip_alleles=tip_alleles)

    def test_n_is_a_no_call_and_not_unrepresentable(self):
        s = self._site(("N", "C"), {"a": "N", "b": "C"})
        assert not s.has_unrepresentable_allele
        assert s.n_unrepresentable_tips() == 0

    def test_lowercase_n_is_a_no_call_too(self):
        s = self._site(("n", "C"), {"a": "n", "b": "C"})
        assert not s.has_unrepresentable_allele

    @pytest.mark.parametrize("allele", [None, "", ".", "N"])
    def test_every_no_call_spelling_is_a_no_call(self, allele):
        s = self._site(("A",), {"a": allele, "b": "A"})
        assert s.n_unrepresentable_tips() == 0

    @pytest.mark.parametrize("allele", ["AT", "*", "<DEL>"])
    def test_an_allele_outside_the_alphabet_is_unrepresentable(self, allele):
        s = self._site(("A", allele), {"a": "A", "b": allele})
        assert s.has_unrepresentable_allele
        assert s.n_unrepresentable_tips() == 1

    def test_a_site_with_one_acgt_allele_is_representable(self):
        assert self._site(("N", "C"), {"a": "N", "b": "C"}).has_representable_allele
        assert self._site(("AT", "C"), {"a": "AT", "b": "C"}).has_representable_allele

    def test_a_site_with_no_acgt_allele_is_not_representable(self):
        assert not self._site(
            ("AT", "ATT"), {"a": "AT", "b": "ATT"}).has_representable_allele
        assert not self._site(("N", "."), {"a": "N", "b": "."}).has_representable_allele

    def test_soft_masked_alleles_stay_representable(self):
        assert self._site(("a", "c"), {"a": "a", "b": "c"}).has_representable_allele


class TestIsCalled:
    """``Site.is_called`` recognises the no-call spellings in any case."""

    @pytest.mark.parametrize("allele", [None, "", ".", "N", "n"])
    def test_missing_spellings_are_not_called(self, allele):
        assert Site.is_called(allele) is False

    def test_non_strings_are_not_called(self):
        assert Site.is_called(0) is False

    @pytest.mark.parametrize("allele", ["A", "a", "T"])
    def test_bases_are_called(self, allele):
        assert Site.is_called(allele) is True


class TestBaseCompositionEdges:
    """Blank FASTA lines, count-free κ and a stream with no called base."""

    def test_from_fasta_skips_blank_lines(self, tmp_path):
        path = tmp_path / "ref.fa"
        path.write_text(">c1 first\nACGT\n\nAC\n\n>c2\n\nGG\n")
        bc = BaseComposition.from_fasta(path)
        assert dict(bc.counts) == {"A": 2, "C": 2, "G": 3, "T": 1}

    def test_kappa_estimate_without_ts_tv_counts_is_the_default(self):
        bc = BaseComposition.from_counts(A=10, C=10, G=10, T=10)
        assert bc.n_ts == 0 and bc.n_tv == 0
        assert bc.kappa_estimate == 2.0

    def test_from_polymorphic_sites_with_no_called_base_is_uniform(self):
        sites = [
            Site(chrom="1", pos=1, alleles=("N",), tip_alleles={"a": None, "b": "N"}),
            Site(chrom="1", pos=2, alleles=(".",), tip_alleles={"a": ".", "b": None}),
        ]
        bc = BaseComposition.from_polymorphic_sites(sites)
        assert bc.n_ts == 0 and bc.n_tv == 0
        np.testing.assert_array_equal(bc.pi, np.full(4, 0.25))


class TestSiteTableIndexing:
    """Slices stay tables and negative indices count from the end."""

    @pytest.fixture
    def table(self):
        sites = [
            Site(chrom="1", pos=10, alleles=("A", "C"), tip_alleles={"h0": "A", "h1": "C"}),
            Site(chrom="1", pos=20, alleles=("G",), tip_alleles={"h0": "G", "h1": "G"}),
            Site(chrom="1", pos=30, alleles=("A", "T"), tip_alleles={"h0": "T", "h1": "A"}),
        ]
        return SiteTable.from_sites(sites)

    def test_slice_returns_a_table_over_the_selected_rows(self, table):
        tail = table[1:]
        assert isinstance(tail, SiteTable)
        assert len(tail) == 2
        assert [s.pos for s in tail] == [20, 30]
        assert tail[0].tip_alleles == {"h0": "G", "h1": "G"}

    def test_negative_index_counts_from_the_end(self, table):
        assert table[-1].pos == 30
        assert table[-3].pos == table[0].pos == 10


@pytest.fixture(scope="module")
def vcf_path(tmp_path_factory):
    """A haploid four-sample VCF on contig ``1``."""
    ts = msprime.sim_ancestry(
        samples=4, ploidy=1, sequence_length=2e4, population_size=1e4,
        random_seed=11,
    )
    ts = msprime.sim_mutations(ts, rate=5e-8, random_seed=11)
    path = tmp_path_factory.mktemp("resolve") / "h.vcf"
    with open(path, "w") as f:
        ts.write_vcf(f, contig_id="1")
    return path


class TestResolvePathArguments:
    """The filter arguments reach the VCF backend and are refused for VCZ."""

    def test_every_argument_is_forwarded_to_the_vcf_source(self, vcf_path):
        src = SiteSource.resolve(
            str(vcf_path), sample_filter=["tsk_0", "tsk_2"], chrom_filter="1",
            ploidy=1, phased=True, phase_seed=7,
        )
        assert isinstance(src, CyVCF2Source)
        assert src.samples() == ["tsk_0", "tsk_2"]
        assert src.ploidy == 1
        assert src._chrom_filter == "1"
        assert src._phased is True
        assert src._phase_seed == 7
        first = next(iter(src))
        assert set(first.tip_alleles) == {"tsk_0", "tsk_2"}

    def test_vcz_refuses_a_ploidy_override(self, tmp_path):
        with pytest.raises(ValueError, match="ploidy is read from the store"):
            SiteSource.resolve(str(tmp_path / "absent.vcz"), ploidy=2)
