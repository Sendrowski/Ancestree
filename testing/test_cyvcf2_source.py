"""Tests for :class:`CyVCF2Source`.

Round-trip: simulate a small ``tskit.TreeSequence``, write it to VCF via
``ts.write_vcf()``, read it back via :class:`CyVCF2Source`, and check
the recovered :class:`Site` records match :class:`TskitSource`'s output
on the same data.
"""
from __future__ import annotations

from pathlib import Path

import msprime
import pytest
import ancestree as anc

from ancestree.sources import CyVCF2Source, TskitSource

import cyvcf2


@pytest.fixture(scope="module")
def haploid_ts():
    """Small haploid msprime ARG (one allele per sample)."""
    ts = msprime.sim_ancestry(
        samples=8, ploidy=1,
        sequence_length=2e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=42,
    )
    return msprime.sim_mutations(ts, rate=5e-8, random_seed=42)


@pytest.fixture(scope="module")
def diploid_ts():
    """Small diploid msprime ARG (two haplotypes per VCF sample)."""
    ts = msprime.sim_ancestry(
        samples=5, ploidy=2,
        sequence_length=2e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=43,
    )
    return msprime.sim_mutations(ts, rate=5e-8, random_seed=43)


def _write_vcf(ts, path: Path, contig: str = "1") -> None:
    """Write ``ts`` to ``path`` with a single contig label."""
    with open(path, "w") as f:
        ts.write_vcf(f, contig_id=contig)


class TestHaploidVCF:
    """Round-trip a ploidy=1 simulation through VCF and CyVCF2Source."""

    @pytest.fixture
    def vcf_path(self, haploid_ts, tmp_path_factory):
        p = tmp_path_factory.mktemp("hap") / "h.vcf"
        _write_vcf(haploid_ts, p)
        return p

    def test_autodetects_haploid_ploidy(self, vcf_path):
        """Auto-detect should report ploidy=1 from gt structure."""
        src = CyVCF2Source(vcf_path)
        assert src.ploidy == 1

    def test_samples_match_vcf_sample_names(self, vcf_path):
        """For haploid, sample ids pass through unchanged (no _h0 suffix)."""
        src = CyVCF2Source(vcf_path)
        vcf = cyvcf2.VCF(str(vcf_path))
        try:
            assert src.samples() == list(vcf.samples)
        finally:
            vcf.close()

    def test_site_count_matches_polymorphic_sites(self, vcf_path, haploid_ts):
        """Every polymorphic SNP in the ts should show up in the VCF stream."""
        # tskit emits one VCF record per site by default. All our sims are A/C/G/T SNPs.
        sites = list(CyVCF2Source(vcf_path))
        assert len(sites) == haploid_ts.num_sites

    def test_tip_alleles_agree_with_tskit_source(self, vcf_path, haploid_ts):
        """Per-site allele observations should match the original tskit view."""
        vcf_sites = list(CyVCF2Source(vcf_path))
        ts_sites = list(TskitSource(haploid_ts))
        assert len(vcf_sites) == len(ts_sites)
        # tskit names samples "tsk_0", "tsk_1", ... in write_vcf; TskitSource
        # defaults to "0", "1", ... (node ids). Map by position instead.
        ts_by_pos = {s.pos: s for s in ts_sites}
        for vs in vcf_sites:
            ts = ts_by_pos[vs.pos]
            # Allele sets should agree (order may differ if REF/ALT reorder).
            assert set(vs.alleles) == set(ts.alleles)


class TestDiploidVCF:
    """ploidy=2 should auto-split each VCF sample into _h0 / _h1 tips."""

    @pytest.fixture
    def vcf_path(self, diploid_ts, tmp_path_factory):
        p = tmp_path_factory.mktemp("dip") / "d.vcf"
        _write_vcf(diploid_ts, p)
        return p

    def test_autodetects_diploid_ploidy(self, vcf_path):
        src = CyVCF2Source(vcf_path)
        assert src.ploidy == 2

    def test_samples_are_haplotype_split(self, vcf_path):
        src = CyVCF2Source(vcf_path)
        samples = src.samples()
        assert all(s.endswith("_h0") or s.endswith("_h1") for s in samples)
        # Each VCF sample contributes exactly 2 hap ids.
        assert len(samples) == 2 * 5

    def test_each_site_has_2n_tip_alleles(self, vcf_path):
        src = CyVCF2Source(vcf_path)
        first = next(iter(src))
        assert len(first.tip_alleles) == 2 * 5


class TestSampleFilter:
    """``sample_filter`` restricts to a subset of VCF samples."""

    @pytest.fixture
    def vcf_path(self, haploid_ts, tmp_path_factory):
        p = tmp_path_factory.mktemp("filt") / "h.vcf"
        _write_vcf(haploid_ts, p)
        return p

    def test_filter_restricts_samples(self, vcf_path):
        src = CyVCF2Source(vcf_path, sample_filter=["tsk_0", "tsk_2"])
        assert src.samples() == ["tsk_0", "tsk_2"]
        first = next(iter(src))
        assert set(first.tip_alleles.keys()) == {"tsk_0", "tsk_2"}

    def test_the_panel_follows_the_filter_order(self, vcf_path):
        """The panel order must be the caller's, not the VCF's.

        cyvcf2 returns the kept samples in file order whatever order they
        were requested in, while VcfZarrSource keeps the requested one, so
        the two backends emitted the same panel in different orders. Each
        sample's own alleles must travel with its name under the reorder.
        """
        forward = CyVCF2Source(vcf_path, sample_filter=["tsk_0", "tsk_2"])
        reverse = CyVCF2Source(vcf_path, sample_filter=["tsk_2", "tsk_0"])
        assert forward.samples() == ["tsk_0", "tsk_2"]
        assert reverse.samples() == ["tsk_2", "tsk_0"]
        assert ([dict(s.tip_alleles) for s in forward]
                == [dict(s.tip_alleles) for s in reverse])
        full = [dict(s.tip_alleles) for s in CyVCF2Source(vcf_path)]
        for site, whole in zip(reverse, full):
            assert site.tip_alleles == {k: whole[k] for k in ("tsk_2", "tsk_0")}

    def test_missing_filter_sample_raises(self, vcf_path):
        """A sample_filter name absent from the VCF raises KeyError
        (cyvcf2 only warns), for parity with VcfZarrSource."""
        with pytest.raises(KeyError, match="not present"):
            CyVCF2Source(vcf_path, sample_filter=["tsk_0", "nonexistent"])

    def test_a_repeated_filter_sample_raises(self, vcf_path):
        """A repeated sample_filter entry raises rather than duplicating a tip.

        htslib de-duplicates the sample list it is handed, so a filter naming
        one sample twice gave a panel one entry longer than the genotype rows
        behind it: samples() reported the duplicate tip id while the sites
        carried it once, and the panel length disagreed with tip_alleles.
        """
        with pytest.raises(ValueError, match="more than once"):
            CyVCF2Source(vcf_path, sample_filter=["tsk_0", "tsk_0", "tsk_2"])


class TestNonSNPSkip:
    """Records with non-A/C/G/T alleles (indels, complex) are skipped."""

    def test_skips_indel_record(self, tmp_path):
        vcf_text = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
            "1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0\t1\n"
            "1\t20\t.\tAT\tA\t.\tPASS\t.\tGT\t0\t1\n"  # indel: skipped
            "1\t30\t.\tC\tT\t.\tPASS\t.\tGT\t1\t0\n"
        )
        p = tmp_path / "mixed.vcf"
        p.write_text(vcf_text)
        sites = list(CyVCF2Source(p))
        assert [s.pos for s in sites] == [10, 30]


class TestAlleleCaseFolding:
    """Soft-masked (lowercase) REF/ALT must be upper-cased so the kernel
    sees A/C/G/T and does not marginalise the tips. ``N`` and other
    non-alphabet characters stay rejected (mapped to None)."""

    def test_lowercase_ref_alt_uppercased(self, tmp_path):
        vcf_text = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
            "1\t10\t.\ta\tg\t.\tPASS\t.\tGT\t0\t1\n"  # soft-masked
        )
        p = tmp_path / "soft.vcf"
        p.write_text(vcf_text)
        sites = list(CyVCF2Source(p))
        assert len(sites) == 1
        assert sites[0].alleles == ("A", "G")
        assert sites[0].tip_alleles == {"s0": "A", "s1": "G"}

    def test_N_allele_record_skipped(self, tmp_path):
        # `N` in REF/ALT means "any base", not a real polymorphism
        # callable for ancestral-allele inference. Treat as non-SNP.
        vcf_text = (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
            "1\t10\t.\tA\tN\t.\tPASS\t.\tGT\t0\t1\n"
            "1\t20\t.\tC\tT\t.\tPASS\t.\tGT\t1\t0\n"
        )
        p = tmp_path / "withN.vcf"
        p.write_text(vcf_text)
        sites = list(CyVCF2Source(p))
        assert [s.pos for s in sites] == [20]


class TestSubPloidyGenotypeCalls:
    """A short (sub-ploidy) genotype call marginalises the absent haplotypes.

    They are padded with ``None`` rather than fabricated by replicating the
    observed allele, which would double-count one observation as two tips.
    """

    def test_haps_from_genotype_pads_missing_haps_with_none(self):
        H = CyVCF2Source._haps_from_genotype
        alleles = ["A", "T", "C"]
        assert H([0, True], 2, alleles) == ["A", None]
        assert H([0, 1, True], 2, alleles) == ["A", "T"]
        assert H([0, 2, 1, False], 2, alleles) == ["A", "C"]
        assert H(None, 2, alleles) == [None, None]


_MIXED_PLOIDY_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t5\t.\tG\tC\t.\tPASS\t.\tGT\t.\t0/1
1\t20\t.\tC\tG\t.\tPASS\t.\tGT\t0/1\t0|0
"""

_TRIPLOID_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t0|1|1\t1/1/0
"""


class TestPloidyResolution:
    """Ploidy comes from every sample of the peeked record.

    A single-dot missing call or a haploid chrY/chrM row at the head of a
    file must not set the whole panel to ploidy 1.
    """

    def test_a_missing_first_call_does_not_halve_the_panel(self, tmp_path):
        path = tmp_path / "mixed.vcf"
        path.write_text(_MIXED_PLOIDY_VCF)
        source = CyVCF2Source(str(path))
        assert source.ploidy == 2
        assert source.samples() == ["s1_h0", "s1_h1", "s2_h0", "s2_h1"]
        sites = list(source)
        assert dict(sites[-1].tip_alleles) == {
            "s1_h0": "C", "s1_h1": "G", "s2_h0": "C", "s2_h1": "C"}

    def test_triploid_genotypes_are_decoded(self, tmp_path):
        """Every haplotype of a ploidy-3 record reaches the panel.

        A decoder that stops at the second index drops one haplotype per
        sample, and one that reads the unphased call in written order makes
        the first haplotype the alternate carrier at every such site. The
        unphased assignment is fixed by the deterministic permutation of
        ``(phase_seed, pos, sample, ploidy)``, so the whole panel is pinned.
        """
        path = tmp_path / "triploid.vcf"
        path.write_text(_TRIPLOID_VCF)
        source = CyVCF2Source(str(path))
        assert source.ploidy == 3
        assert len(source.samples()) == 6
        site = next(iter(source))
        assert dict(site.tip_alleles) == {
            "s1_h0": "A", "s1_h1": "T", "s1_h2": "T",
            "s2_h0": "A", "s2_h1": "T", "s2_h2": "T"}


# ------------------------------------------ CyVCF2Source._haps_from_genotype
class TestHapsFromGenotype:
    """Decoding a ``genotypes`` row: indices in, alleles out.

    The row is per-haplotype allele indices followed by a phase flag, so the
    cases are a missing index, a short row and a long one. Non-canonical
    alleles never reach here: a record carrying one is skipped upstream.
    """

    H = staticmethod(CyVCF2Source._haps_from_genotype)
    ALLELES = ["A", "G"]

    def test_missing_row_is_none(self):
        assert self.H(None, 2, self.ALLELES) == [None, None]

    def test_negative_indices_are_none(self):
        assert self.H([-1, -1, False], 2, self.ALLELES) == [None, None]

    def test_phased_diploid(self):
        assert self.H([0, 1, True], 2, self.ALLELES) == ["A", "G"]

    def test_short_row_padded_with_none(self):
        # One haplotype at ploidy 2: the absent one is marginalised (None),
        # not fabricated by replicating the observed allele.
        assert self.H([0, True], 2, self.ALLELES) == ["A", None]

    def test_long_row_truncated(self):
        assert self.H([0, 1, 1, False], 2, self.ALLELES) == ["A", "G"]

    def test_index_past_the_allele_list_is_none(self):
        assert self.H([0, 5, False], 2, self.ALLELES) == ["A", None]

    def test_arbitrary_ploidy(self):
        assert self.H([0, 1, 1, True], 3, self.ALLELES) == ["A", "G", "G"]

    def test_an_absent_allele_keeps_its_slot(self):
        """An allele the record does not carry must not shift the indices.

        Empty alleles were dropped from the list before decoding, so every
        allele after the gap moved down one index and each genotype pointing
        past the gap resolved to the neighbouring base. The tip then carried
        a real but wrong allele, with nothing to flag it.
        """
        assert self.H([0, 2, True], 2, ["A", None, "G"]) == ["A", "G"]
        assert self.H([1, 0, True], 2, ["A", None, "G"]) == [None, "A"]


class TestPloidyTruncationKeepsTheLeadingCalls:
    """The unphased draw permutes haplotype order, it does not select alleles.

    Truncation to the resolved ploidy keeps the leading calls whatever the
    draw, so an unphased ``0/1/1`` read at ploidy 2 is always heterozygous.
    """

    @staticmethod
    def _decode(order):
        return CyVCF2Source._haps_from_genotype(
            [0, 1, 1, False], 2, ["A", "T"], order=order)

    def test_the_kept_alleles_do_not_depend_on_the_draw(self):
        assert sorted(self._decode(None)) == sorted(self._decode([1, 0]))

    def test_the_leading_calls_are_the_ones_kept(self):
        assert sorted(self._decode(None)) == ["A", "T"]

    def test_a_truncated_record_is_reported(self, tmp_path, caplog):
        import logging

        vcf = tmp_path / "mixed.vcf"
        rows = ["1\t100\t.\tA\tT\t.\t.\t.\tGT\t0|1\t0|1",
                "1\t200\t.\tA\tT\t.\t.\t.\tGT\t0/1/1\t0|1"]
        vcf.write_text(
            "##fileformat=VCFv4.2\n##contig=<ID=1,length=100000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
            + "\n".join(rows) + "\n")
        with caplog.at_level(logging.WARNING, logger="ancestree.CyVCF2Source"):
            list(anc.CyVCF2Source(str(vcf)))
        assert any("carries 3 haplotypes" in r.getMessage()
                   for r in caplog.records)


def test_cyvcf2_source_sites_only_vcf(tmp_path):
    from ancestree.sources import CyVCF2Source

    vcf = tmp_path / "sites.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\t.\t.\n"
    )
    src = CyVCF2Source(str(vcf))
    assert src.ploidy == 2
