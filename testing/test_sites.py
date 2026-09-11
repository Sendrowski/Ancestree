"""Site-level helpers that the inference layer depends on."""
from collections import Counter

import pytest

from ancestree import Site


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

        return tskit.load("docs/_static/quickstart.trees")

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
