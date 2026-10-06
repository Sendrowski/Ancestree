"""The ``AA`` / ``AA_prob`` / ``AA_post`` annotations are readable by the sfsutils parser.

sfsutils is a test dependency rather than a runtime one: it is the downstream
consumer whose expectations these annotations exist to meet, so the suite fails
rather than skips when it is absent.
"""
import pytest
import sfsutils

import ancestree as anc


@pytest.fixture(scope="module")
def annotated(tmp_path_factory, small_ts):
    """The same calls written to all three output formats."""
    out = tmp_path_factory.mktemp("sfs")
    inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
    inference.to_vcf(out / "a.vcf.gz")
    inference.to_zarr(out / "a.vcz")
    inference.to_arg(out / "a.trees")
    return out


def _spectrum(source, n, **kwargs):
    """The unfolded SFS sfsutils parses from ``source``."""
    parsed = sfsutils.Parser(source=source, n=n, **kwargs).parse()
    return [int(x) for x in parsed.data.iloc[:, 0].tolist()]


class TestSFSUtilsCompatibility:
    """What the writers emit is what the downstream parser expects."""

    def test_vcf_and_zarr_agree(self, annotated, small_ts):
        """The ``AA`` INFO tag and the ``variant_AA`` array give one spectrum.

        Both are read through sfsutils' default ``info_ancestral='AA'``, so a
        rename on either side would show up here.
        """
        n = small_ts.num_samples
        vcf = _spectrum(str(annotated / "a.vcf.gz"), n, skip_non_polarized=True)
        vcz = _spectrum(str(annotated / "a.vcz"), n, skip_non_polarized=True)
        assert sum(vcf) > 0
        assert vcf == vcz

    def test_tree_sequence_agrees(self, annotated, small_ts):
        """A tree sequence polarises off allele 0, which is the MAP call.

        sfsutils reads no ``AA`` tag from a tree sequence. It takes the
        ancestral state as allele 0, which is what the writer sets. This only
        matches the VCF spectrum because annotation re-derives the mutations
        against the new ancestral state. Leaving them untouched shifts sites
        into the monomorphic bin.
        """
        import tskit

        n = small_ts.num_samples
        vcf = _spectrum(str(annotated / "a.vcf.gz"), n, skip_non_polarized=True)
        trees = _spectrum(
            tskit.load(str(annotated / "a.trees")), n, skip_non_polarized=False,
        )
        assert vcf == trees

    def test_probability_field_is_the_expected_name(self):
        """sfsutils' probabilistic polarisation reads the tag Ancestree writes."""
        import inspect

        defaults = inspect.signature(sfsutils.Parser.__init__).parameters
        assert defaults["info_ancestral"].default == "AA"
        assert defaults["info_ancestral_prob"].default == "AA_prob"
        assert defaults["info_ancestral_post"].default == "AA_post"


class TestPosteriorAndConfidenceGiveOneSpectrum:
    """sfsutils polarises by ``AA_post`` where present, else by ``AA`` / ``AA_prob``.

    With ``n`` equal to the number of haplotypes the projection is the
    identity, so a site with ``k`` derived copies and ancestral probability
    ``p`` adds ``p`` to bin ``k`` and ``1 - p`` to bin ``n - k``.
    """

    N = 6
    #: ``(pos, ref, alt, genotypes, posterior over A, C, G, T)``.
    TWO_BASE_SITES = [
        (100, "A", "C", "0/1\t0/0\t0/0", [0.9, 0.1, 0.0, 0.0]),
        (200, "A", "C", "0/1\t1/1\t1/1", [0.3, 0.7, 0.0, 0.0]),
        (300, "A", "G", "0/1\t0/1\t0/0", [0.2, 0.0, 0.8, 0.0]),
        (400, "C", "T", "0/0\t0/1\t0/0", [0.0, 1.0, 0.0, 0.0]),
        (500, "A", "C", "1/1\t0/1\t0/0", [0.55, 0.45, 0.0, 0.0]),
    ]
    TWO_BASE_SPECTRUM = [0.0, 2.6, 0.2, 1.0, 0.8, 0.4, 0.0]
    #: A third of the mass sits on bases the site does not carry.
    THREE_BASE_SITES = [(100, "A", "C", "0/1\t0/0\t0/0", [0.5, 0.3, 0.2, 0.0])]

    @staticmethod
    def _spectra(tmp_path, sites):
        """The spectra parsed from one set of sites written with and without ``AA_post``."""
        import numpy as np

        from ancestree.posterior import Posterior

        template = tmp_path / "t.vcf"
        template.write_text(
            "##fileformat=VCFv4.2\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "##contig=<ID=1,length=100000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
            + "".join(f"1\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t{gts}\n"
                      for pos, ref, alt, gts, _ in sites))
        pairs = [
            (anc.Site(chrom="1", pos=pos, alleles=(ref, alt), tip_alleles={}),
             Posterior(alleles=("A", "C", "G", "T"), values=np.array(post)))
            for pos, ref, alt, _, post in sites
        ]
        spectra = {}
        for store_posterior in (True, False):
            out = str(tmp_path / f"post_{store_posterior}.vcf")
            anc.VCFWriter(str(template), out).write(
                iter(pairs), store_posterior=store_posterior)
            parsed = sfsutils.Parser(
                source=out, n=TestPosteriorAndConfidenceGiveOneSpectrum.N,
                polarize_probabilistically=True).parse()
            spectra[store_posterior] = parsed.data.iloc[:, 0].to_numpy()
        return spectra[True], spectra[False]

    def test_a_two_base_posterior_matches_the_confidence(self, tmp_path):
        """Mass on the site's two alleles only makes the renormalisation the identity."""
        from_post, from_prob = self._spectra(tmp_path, self.TWO_BASE_SITES)
        # cyvcf2 hands out Float INFO values in single precision.
        assert from_post == pytest.approx(from_prob, abs=1e-6)
        assert from_post == pytest.approx(self.TWO_BASE_SPECTRUM, abs=1e-6)

    def test_mass_on_a_third_base_separates_them(self, tmp_path):
        """``AA_post`` renormalises over the site's alleles, ``AA_prob`` is the four-state maximum.

        Guards the test above against a parser that ignores ``AA_post``.
        """
        from_post, from_prob = self._spectra(tmp_path, self.THREE_BASE_SITES)
        assert from_post[[1, 5]] == pytest.approx([0.625, 0.375], abs=1e-6)
        assert from_prob[[1, 5]] == pytest.approx([0.5, 0.5], abs=1e-6)
