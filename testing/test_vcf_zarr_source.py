"""Tests for :class:`VcfZarrSource`.

Builds a small VCZ-format zarr store by hand (matching the layout
``bio2zarr`` writes) and checks that the reader recovers the expected
sites, samples, and per-haplotype tip alleles.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from ancestree.sites import SiteSource
from ancestree.sources import CyVCF2Source, VcfZarrSource
from testing._helpers import write_vcz

import zarr

# The hand-built VCZ fixtures use fixed-length string arrays (bio2zarr's
# layout). Zarr v3 has no V3 spec for them yet and warns. The format mandates
# the dtype, so the notice is not actionable, silence it for this module only
# (the category resolves because zarr is imported above).
pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.errors.UnstableSpecificationWarning"
)


@pytest.fixture
def diploid_store(tmp_path):
    """A 3-variant, 2-sample, diploid VCZ store with one missing genotype.

    The calls are written as phased, so each haplotype keeps the position it
    was written at and the decoded tips are a fixed expectation.
    """
    p = tmp_path / "dip.vcz"
    # site 0: A/C, gt = [[0,0], [0,1]]  -> s0: A/A, s1: A/C
    # site 1: G/T, gt = [[1,1], [-1,0]] -> s0: T/T, s1: missing/G
    # site 2: C/G/T (multi-allelic), gt = [[0,1], [2,0]]
    gt = np.array(
        [
            [[0, 0], [0, 1]],
            [[1, 1], [-1, 0]],
            [[0, 1], [2, 0]],
        ],
        dtype=np.int8,
    )
    write_vcz(
        p,
        positions=[100, 200, 300],
        contigs_per_variant=[0, 0, 0],
        contig_ids=["chr1"],
        alleles=[["A", "C"], ["G", "T"], ["C", "G", "T"]],
        sample_ids=["s0", "s1"],
        genotypes=gt,
        phased=True,
    )
    return p


class TestDiploidVcz:
    """Round-trip a hand-crafted diploid VCZ store."""

    def test_ploidy_and_sample_split(self, diploid_store):
        src = VcfZarrSource(diploid_store)
        assert src.ploidy == 2
        assert src.samples() == ["s0_h0", "s0_h1", "s1_h0", "s1_h1"]

    def test_site_count_and_positions(self, diploid_store):
        sites = list(VcfZarrSource(diploid_store))
        assert [s.pos for s in sites] == [100, 200, 300]

    def test_chrom_resolves_via_contig_id(self, diploid_store):
        first = next(iter(VcfZarrSource(diploid_store)))
        assert first.chrom == "chr1"

    def test_tip_alleles_first_site(self, diploid_store):
        sites = list(VcfZarrSource(diploid_store))
        s0 = sites[0]  # A/C, gt = [[0,0],[0,1]]
        assert s0.alleles == ("A", "C")
        assert s0.tip_alleles == {"s0_h0": "A", "s0_h1": "A", "s1_h0": "A", "s1_h1": "C"}

    def test_missing_genotype_becomes_none(self, diploid_store):
        sites = list(VcfZarrSource(diploid_store))
        s1 = sites[1]  # gt = [[1,1],[-1,0]]
        assert s1.tip_alleles["s1_h0"] is None

    def test_multiallelic_kept(self, diploid_store):
        sites = list(VcfZarrSource(diploid_store))
        s2 = sites[2]  # C/G/T, gt = [[0,1],[2,0]]
        assert s2.alleles == ("C", "G", "T")
        assert s2.tip_alleles == {"s0_h0": "C", "s0_h1": "G", "s1_h0": "T", "s1_h1": "C"}


class TestSampleFilter:
    """``sample_filter`` restricts which VCZ samples appear in tip_alleles."""

    def test_filter_to_one_sample(self, diploid_store):
        src = VcfZarrSource(diploid_store, sample_filter=["s1"])
        assert src.samples() == ["s1_h0", "s1_h1"]
        first = next(iter(src))
        assert set(first.tip_alleles.keys()) == {"s1_h0", "s1_h1"}

    def test_unknown_sample_raises(self, diploid_store):
        with pytest.raises(KeyError):
            VcfZarrSource(diploid_store, sample_filter=["nonexistent"])

    def test_a_repeated_filter_sample_raises(self, diploid_store):
        """A repeated sample_filter entry raises rather than duplicating a tip.

        The kept names were taken from the caller's list verbatim, so a filter
        naming one sample twice put the same haplotype tip ids in the panel
        twice while each site carried them once, and the panel length
        disagreed with tip_alleles.
        """
        with pytest.raises(ValueError, match="more than once"):
            VcfZarrSource(diploid_store, sample_filter=["s0", "s0", "s1"])


class TestHaploidVcz:
    """Haploid stores keep sample names unchanged (no _h suffix)."""

    def test_no_hap_suffix(self, tmp_path):
        p = tmp_path / "hap.vcz"
        gt = np.array(
            [[[0], [1]], [[1], [0]]],
            dtype=np.int8,
        )
        write_vcz(
            p,
            positions=[50, 60],
            contigs_per_variant=[0, 0],
            contig_ids=["chr1"],
            alleles=[["A", "T"], ["C", "G"]],
            sample_ids=["h0", "h1"],
            genotypes=gt,
        )
        src = VcfZarrSource(p)
        assert src.ploidy == 1
        assert src.samples() == ["h0", "h1"]
        first = next(iter(src))
        assert first.tip_alleles == {"h0": "A", "h1": "T"}


class TestNonSNPSkip:
    """Sites with non-A/C/G/T alleles are skipped."""

    def test_indel_skipped(self, tmp_path):
        p = tmp_path / "indel.vcz"
        # A non-ACGT allele is enough to exercise the skip.
        gt = np.array([[[0], [0]], [[1], [0]]], dtype=np.int8)
        write_vcz(
            p,
            positions=[10, 20],
            contigs_per_variant=[0, 0],
            contig_ids=["chr1"],
            alleles=[["A", "N"], ["C", "T"]],  # site 0 has an N → skipped
            sample_ids=["s0", "s1"],
            genotypes=gt,
        )
        sites = list(VcfZarrSource(p))
        assert [s.pos for s in sites] == [20]


def test_resolve_routes_zarr_extension_to_vcf_zarr_source(tmp_path):
    """A VCF-Zarr store with a `.zarr` extension (accepted by the
    CLI guards and to_zarr) routes to VcfZarrSource, not to
    cyvcf2, which cannot open a Zarr directory."""

    p = tmp_path / "store.zarr"
    write_vcz(
        p,
        positions=[10, 20, 30],
        contigs_per_variant=[0, 0, 0],
        contig_ids=["1"],
        alleles=[["A", "T"], ["C", "G"], ["A", "G"]],
        sample_ids=["s0", "s1"],
        genotypes=np.array(
            [[[0], [1]], [[1], [0]], [[0], [0]]], dtype=np.int8
        ),
    )
    src = SiteSource.resolve(str(p))
    assert isinstance(src, VcfZarrSource)
    # And it actually reads (re-iterable, yields the three sites).
    assert len(list(src)) == 3


class TestZarrDecode:
    def test_bytes_decoded_to_ascii(self):
        assert VcfZarrSource._decode(b"A") == "A"

    def test_str_passthrough(self):
        assert VcfZarrSource._decode("G") == "G"

    def test_other_stringified(self):
        assert VcfZarrSource._decode(3) == "3"


class TestAbsentPhaseArray:
    """A store carrying no ``call_genotype_phased`` reads as unphased.

    ``VcfZarrSource`` left the per-call phase undefined when the array was
    absent and then took every genotype in the order written, while
    ``CyVCF2Source`` reads a record with no phase flag as unphased and draws
    the haplotype order. The same unphased data therefore produced different
    tips, and only one warning, depending on which backend read it, with the
    VCZ path building the reference-carrier clade the randomisation exists
    to break.
    """

    N_SITES = 20
    POSITIONS = [100 + 10 * i for i in range(N_SITES)]

    @pytest.fixture
    def unphased_pair(self, tmp_path):
        """The same unphased het panel as a VCZ store and as a VCF."""
        store = tmp_path / "unphased.vcz"
        write_vcz(
            store,
            positions=self.POSITIONS,
            contigs_per_variant=[0] * self.N_SITES,
            contig_ids=["1"],
            alleles=[["A", "G"]] * self.N_SITES,
            sample_ids=["s0", "s1"],
            genotypes=np.zeros((self.N_SITES, 2, 2), dtype=np.int8) + np.array(
                [0, 1], dtype=np.int8),
        )
        vcf = tmp_path / "unphased.vcf"
        rows = "\n".join(
            f"1\t{pos}\t.\tA\tG\t.\t.\t.\tGT\t0/1\t0/1"
            for pos in self.POSITIONS
        )
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=100000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
            "\ts0\ts1\n" + rows + "\n"
        )
        return store, str(vcf)

    def test_the_two_backends_assign_the_same_haplotypes(self, unphased_pair):
        store, vcf = unphased_pair
        from_vcz = {s.pos: dict(s.tip_alleles) for s in VcfZarrSource(store)}
        from_vcf = {s.pos: dict(s.tip_alleles) for s in CyVCF2Source(vcf)}
        assert from_vcz == from_vcf

    def test_the_order_is_drawn_rather_than_taken_as_written(self, unphased_pair):
        """Guard the parity test above against passing on two clades."""
        store, _ = unphased_pair
        first_haplotype = {
            alleles["s0_h0"]
            for alleles in ({s.pos: dict(s.tip_alleles)
                             for s in VcfZarrSource(store)}).values()
        }
        assert first_haplotype == {"A", "G"}

    def test_the_same_notice_is_reported(self, unphased_pair, caplog):
        import logging

        store, vcf = unphased_pair
        messages = []
        for source, logger in ((VcfZarrSource(store), "ancestree.VcfZarrSource"),
                               (CyVCF2Source(vcf), "ancestree.CyVCF2Source")):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger=logger):
                list(source)
            messages.append([r.getMessage() for r in caplog.records
                             if "Unphased heterozygous" in r.getMessage()])
        assert messages[0] and messages[0] == messages[1]

    @pytest.fixture
    def phased_store(self, tmp_path):
        """The same het panel, with every call declared phased."""
        store = tmp_path / "phased.vcz"
        write_vcz(
            store,
            positions=self.POSITIONS,
            contigs_per_variant=[0] * self.N_SITES,
            contig_ids=["1"],
            alleles=[["A", "G"]] * self.N_SITES,
            sample_ids=["s0", "s1"],
            genotypes=np.zeros((self.N_SITES, 2, 2), dtype=np.int8) + np.array(
                [0, 1], dtype=np.int8),
            phased=True,
        )
        return store

    def test_a_written_phase_array_stays_authoritative(self, phased_store):
        assert all(s.tip_alleles["s0_h0"] == "A" and s.tip_alleles["s0_h1"] == "G"
                   for s in VcfZarrSource(phased_store))

    def test_forcing_phased_true_overrides_an_unphased_store(self, unphased_pair):
        store, _ = unphased_pair
        assert all(s.tip_alleles["s0_h0"] == "A"
                   for s in VcfZarrSource(store, phased=True))

    def test_forcing_phased_false_ignores_the_written_array(
            self, phased_store, unphased_pair):
        store, _ = unphased_pair
        forced = [dict(s.tip_alleles) for s in VcfZarrSource(phased_store,
                                                             phased=False)]
        drawn = [dict(s.tip_alleles) for s in VcfZarrSource(store)]
        written = [dict(s.tip_alleles) for s in VcfZarrSource(phased_store)]
        assert forced == drawn
        assert forced != written


def test_an_absent_allele_keeps_the_genotype_indices_aligned(tmp_path):
    """An empty entry in ``variant_allele`` must not shift the alleles after it.

    Empty entries were dropped before the genotypes were decoded, which is
    safe only for the trailing padding ``bio2zarr`` writes. A gap anywhere
    earlier moved every later allele down one index, so a genotype pointing
    past the gap silently resolved to the neighbouring base instead of to a
    missing tip.
    """
    p = tmp_path / "gap.vcz"
    write_vcz(
        p,
        positions=[100],
        contigs_per_variant=[0],
        contig_ids=["1"],
        alleles=[["A", "", "G"]],
        sample_ids=["s0"],
        genotypes=np.array([[[0, 2]]], dtype=np.int8),
        phased=True,
    )
    site = next(iter(VcfZarrSource(p)))
    assert site.alleles == ("A", "G")
    assert site.tip_alleles == {"s0_h0": "A", "s0_h1": "G"}


def test_the_panel_follows_the_filter_order(diploid_store):
    """Both backends order the panel by ``sample_filter``, not by the file."""
    reverse = VcfZarrSource(diploid_store, sample_filter=["s1", "s0"])
    assert reverse.samples() == ["s1_h0", "s1_h1", "s0_h0", "s0_h1"]
    forward = VcfZarrSource(diploid_store, sample_filter=["s0", "s1"])
    assert ([dict(s.tip_alleles) for s in reverse]
            == [dict(s.tip_alleles) for s in forward])


class TestVcfZarrSource:
    """The repr, the contig filter, records without a reference allele,
    malformed stores and the backend guard."""

    def test_vcf_zarr_source(self, tmp_path):
        path = str(tmp_path / "s.vcz")
        write_vcz(
            path, positions=[100, 200], contigs_per_variant=[0, 0],
            contig_ids=["1"], alleles=[["A", "C"], ["A", "C"]],
            sample_ids=["s1"], genotypes=np.zeros((2, 1, 2), dtype=np.int8),
        )
        assert repr(VcfZarrSource(path)) == f"VcfZarrSource(path={path!r}, n_samples=2)"

    def test_the_contig_filter_drops_other_contigs(self, tmp_path):
        path = str(tmp_path / "s.vcz")
        write_vcz(
            path, positions=[100, 150, 200], contigs_per_variant=[0, 1, 0],
            contig_ids=["1", "2"],
            alleles=[["A", "C"], ["G", "T"], ["A", "G"]],
            sample_ids=["s1"], genotypes=np.zeros((3, 1, 1), dtype=np.int8),
        )
        assert [(s.chrom, s.pos) for s in VcfZarrSource(path, chrom_filter="2")] \
            == [("2", 150)]
        assert [(s.chrom, s.pos) for s in VcfZarrSource(path)] \
            == [("1", 100), ("2", 150), ("1", 200)]

    def test_a_record_without_a_reference_allele_is_skipped(self, tmp_path):
        path = str(tmp_path / "s.vcz")
        write_vcz(
            path, positions=[100, 200, 300], contigs_per_variant=[0, 0, 0],
            contig_ids=["1"], alleles=[["A", "C"], [], ["", "C"]],
            sample_ids=["s1"], genotypes=np.zeros((3, 1, 1), dtype=np.int8),
        )
        sites = list(VcfZarrSource(path))
        assert [s.pos for s in sites] == [100]
        assert sites[0].tip_alleles == {"s1": "A"}

    def test_a_two_dimensional_genotype_array_is_refused(self, tmp_path):
        path = str(tmp_path / "s.vcz")
        write_vcz(
            path, positions=[100], contigs_per_variant=[0], contig_ids=["1"],
            alleles=[["A", "C"]], sample_ids=["s1"],
            genotypes=np.zeros((1, 1), dtype=np.int8),
        )
        with pytest.raises(ValueError, match="Expected call_genotype"):
            VcfZarrSource(path)

    def test_a_missing_backend_is_reported_at_construction(
            self, diploid_store, monkeypatch):
        monkeypatch.setitem(sys.modules, "zarr", None)
        with pytest.raises(ImportError, match="VcfZarrSource requires zarr"):
            VcfZarrSource(diploid_store)
