"""The ``AA`` / ``AA_prob`` annotations are readable by the sfsutils parser.

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
