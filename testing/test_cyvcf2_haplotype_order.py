"""The cyvcf2 reader's per-haplotype assignment.

Pins which allele lands on which haplotype of a diploid individual, as a
dict comparison against the tskit reading of the same data, so a reversed
haplotype order cannot pass on allele sets alone.
"""
import gzip
from collections import Counter

import pytest

import ancestree as anc

from ancestree.sources import CyVCF2Source


@pytest.fixture
def diploid_vcf(tmp_path):
    """Two diploid samples: a hom-ref, a phased het, and a missing call."""
    p = tmp_path / "dip.vcf.gz"
    body = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
        "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0|0\t0|1\n"
        "1\t20\t.\tG\tT\t.\tPASS\t.\tGT\t1|1\t.|0\n"
    )
    with gzip.open(p, "wt") as fh:
        fh.write(body)
    return str(p)


def test_the_phased_het_assigns_each_haplotype_its_own_allele(diploid_vcf):
    """s1 is 0|1, so _h0 carries REF and _h1 carries ALT, not the reverse."""
    sites = list(anc.CyVCF2Source(diploid_vcf))
    s0 = sites[0]
    assert s0.alleles == ("A", "C")
    assert s0.tip_alleles == {
        "s0_h0": "A", "s0_h1": "A", "s1_h0": "A", "s1_h1": "C",
    }, f"haplotype assignment is {s0.tip_alleles}"


def test_a_missing_haplotype_is_none_and_its_partner_is_kept(diploid_vcf):
    """s1 is .|0 at the second site: _h0 missing, _h1 the reference allele."""
    sites = list(anc.CyVCF2Source(diploid_vcf))
    s1 = sites[1]
    assert s1.tip_alleles["s0_h0"] == "T"
    assert s1.tip_alleles["s0_h1"] == "T"
    assert s1.tip_alleles["s1_h0"] is None
    assert s1.tip_alleles["s1_h1"] == "G"


def test_the_sample_order_follows_the_header(diploid_vcf):
    src = anc.CyVCF2Source(diploid_vcf)
    assert src.samples() == ["s0_h0", "s0_h1", "s1_h0", "s1_h1"]


class TestUnphasedOrderIsIndependentOfThePanel:
    """The unphased-het haplotype order must not move with ``sample_filter``.

    ``CyVCF2Source._phase_permutation`` was keyed on the sample's ordinal position in the
    already-filtered panel, so subsetting or reordering a panel reassigned the
    same records' heterozygous alleles to opposite haplotype tips at roughly
    half of the unphased het sites. Each haplotype is a tip, so the pairwise
    difference counts, the sampled TMRCAs, the window topologies and the
    emitted posteriors all moved with a parameter that names no data.
    """

    @pytest.fixture
    def unphased_vcf(self, tmp_path):
        """Three diploid samples, all unphased heterozygous at every site."""
        p = tmp_path / "unphased.vcf"
        rows = "\n".join(
            f"1\t{100 + i * 10}\t.\tA\tG\t.\t.\t.\tGT\t0/1\t0/1\t0/1"
            for i in range(40)
        )
        p.write_text(
            "##fileformat=VCFv4.2\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "##contig=<ID=1,length=1000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
            "\ts0\ts1\ts2\n" + rows + "\n"
        )
        return str(p)

    @staticmethod
    def _decode(path, **kwargs):
        """Position to tip-allele mapping for one reader configuration."""
        return {s.pos: dict(s.tip_alleles)
                for s in anc.CyVCF2Source(path, **kwargs)}

    def test_the_permutation_actually_fires(self, unphased_vcf):
        """Guard the two tests below against passing vacuously."""
        decoded = self._decode(unphased_vcf)
        reversed_calls = sum(
            1 for alleles in decoded.values()
            for s in ("s0", "s1", "s2") if alleles[f"{s}_h0"] == "G"
        )
        assert 0 < reversed_calls < 3 * len(decoded)

    def test_subsetting_the_panel_keeps_each_call_s_order(self, unphased_vcf):
        full = self._decode(unphased_vcf)
        subset = self._decode(unphased_vcf, sample_filter=["s1", "s2"])
        for pos, alleles in subset.items():
            for tip, allele in alleles.items():
                assert full[pos][tip] == allele

    def test_reordering_the_panel_keeps_each_call_s_order(self, unphased_vcf):
        forward = self._decode(unphased_vcf, sample_filter=["s1", "s2"])
        reverse = self._decode(unphased_vcf, sample_filter=["s2", "s1"])
        assert forward == reverse


class TestTheDrawCoversEveryOrdering:
    """The unphased draw is uniform over the orderings, at every ploidy.

    ``CyVCF2Source._phase_permutation`` took a ``ploidy`` argument and ignored it, and both
    readers applied its result by reversing the written haplotype order. A
    reversal reaches two of the ``ploidy!`` orderings, so the per-site
    randomisation the readers document held for diploids only, and a triploid
    or tetraploid panel kept most of the reference-carrier structure the draw
    exists to break.
    """

    def test_the_draw_is_a_permutation_of_the_ploidy(self):
        for ploidy in (1, 2, 3, 4):
            drawn = CyVCF2Source._phase_permutation(0, 100, "s0", ploidy)
            assert sorted(int(x) for x in drawn) == list(range(ploidy))

    def test_every_triploid_ordering_is_reachable(self):
        seen = {tuple(int(x) for x in CyVCF2Source._phase_permutation(0, pos, "s0", 3))
                for pos in range(500)}
        assert len(seen) == 6

    def test_the_same_key_draws_the_same_order(self):
        """Reproducibility across runs is the point of ``phase_seed``."""
        assert (list(CyVCF2Source._phase_permutation(3, 77, "s0", 4))
                == list(CyVCF2Source._phase_permutation(3, 77, "s0", 4)))

    def test_the_seed_moves_the_order(self):
        """Every ordering is reachable by varying ``phase_seed`` alone."""
        seen = {tuple(CyVCF2Source._phase_permutation(seed, 77, "s0", 4))
                for seed in range(200)}
        assert len(seen) == 24

    def test_a_triploid_reader_reaches_every_ordering(self, tmp_path):
        """The reader wires the draw through, not just the helper."""
        positions = [100 + 10 * i for i in range(80)]
        rows = "\n".join(
            f"1\t{pos}\t.\tA\tG,T\t.\t.\t.\tGT\t0/1/2" for pos in positions
        )
        vcf = tmp_path / "triploid.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=100000>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\n"
            + rows + "\n"
        )
        orderings = {
            tuple(site.tip_alleles[f"s0_h{h}"] for h in range(3))
            for site in anc.CyVCF2Source(str(vcf))
        }
        assert len(orderings) == 6


class TestTheDrawIsIndependentOfNumpy:
    """The haplotype order is fixed by the key alone, in integer arithmetic.

    The draw seeded a ``numpy.random.Generator`` with the key and took its
    permutation. numpy gives no compatibility guarantee on that bit stream,
    while the project pins only ``numpy >= 1.24``, so a numpy upgrade would
    have silently reassigned the haplotypes of every unphased heterozygote
    and changed the stored ``AA`` and ``AA_prob``, against a docstring
    promising reproducibility across runs.
    """

    #: Orderings the unranking produces, keyed on ``(seed, pos, sample, ploidy)``.
    GOLDEN = {
        (0, 100, "s0", 1): [0],
        (0, 100, "s0", 2): [1, 0],
        (0, 100, "s0", 3): [0, 1, 2],
        (0, 100, "s0", 4): [1, 0, 3, 2],
        (7, 12345, "sample_A", 3): [2, 0, 1],
    }

    def test_a_fixed_key_draws_a_fixed_order(self):
        for key, expected in self.GOLDEN.items():
            assert CyVCF2Source._phase_permutation(*key) == expected, key

    def test_the_result_is_always_a_permutation(self):
        for ploidy in (1, 2, 3, 4, 5):
            for pos in range(400):
                drawn = CyVCF2Source._phase_permutation(0, pos, "s0", ploidy)
                assert sorted(drawn) == list(range(ploidy))
                assert all(type(h) is int for h in drawn)

    def test_a_haploid_genotype_keeps_its_single_haplotype(self):
        for seed in range(20):
            for pos in range(50):
                assert CyVCF2Source._phase_permutation(seed, pos, "s0", 1) == [0]

    def test_the_draw_is_close_to_uniform(self):
        """Unranking a 64-bit key modulo ``ploidy!`` stays near-uniform."""
        for ploidy, n_orderings in ((2, 2), (3, 6)):
            counts = Counter(
                tuple(CyVCF2Source._phase_permutation(0, pos, "s0", ploidy))
                for pos in range(20000)
            )
            assert len(counts) == n_orderings
            expected = 20000 / n_orderings
            chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
            # 99.9th percentile of chi-square on n_orderings - 1 df.
            assert chi2 < {2: 10.83, 6: 20.52}[n_orderings]
