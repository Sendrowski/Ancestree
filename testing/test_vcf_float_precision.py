"""``AA_prob`` and ``AA_post`` across the three VCF destinations.

htslib formats a float into text at six significant digits, so a near-certain
call written as a float reads back as exactly 1 and disagrees with the maximum
of the posterior beside it. Full-mantissa text avoids that in a text VCF. BCF
holds typed binary, where the same text becomes a character field under a
``Type=Float`` declaration and reaches the reader as a string.
"""
import os
import tempfile

import numpy as np
import pytest

import ancestree as anc
from ancestree.posterior import Posterior
from ancestree.writers import VCFWriter


SRC = "docs/_static/quickstart.vcf.gz"
NEAR_CERTAIN = 0.9999999


def _calls(n=3):
    sites = list(anc.CyVCF2Source(SRC))[:n]
    post = Posterior(list("ACGT"), np.array([NEAR_CERTAIN, 1e-7, 1e-7, 1e-7]))
    return sites, [(s, post) for s in sites]


def _write(ext, **kwargs):
    sites, pairs = _calls()
    out = tempfile.mktemp(suffix=ext)
    VCFWriter(SRC, out).write(iter(pairs), **kwargs)
    return out


@pytest.mark.parametrize("ext", [".vcf", ".vcf.gz", ".bcf"])
def test_the_confidence_reads_back_as_a_float(ext):
    """``Annotation.aa_prob`` is declared ``float | None`` in every format."""
    out = _write(ext, store_posterior=(ext != ".bcf"))
    try:
        first = next(iter(anc.Reader(out).annotations()))
        assert isinstance(first.aa_prob, float), (
            f"{ext} gave {type(first.aa_prob).__name__}: {first.aa_prob!r}")
    finally:
        os.unlink(out)


@pytest.mark.parametrize("ext", [".vcf", ".vcf.gz", ".bcf"])
def test_a_near_certain_call_is_not_rounded_to_one(ext):
    out = _write(ext, store_posterior=(ext != ".bcf"))
    try:
        first = next(iter(anc.Reader(out).annotations()))
        assert first.aa_prob is not None and first.aa_prob < 1.0, (
            f"{ext} rounded {NEAR_CERTAIN} to {first.aa_prob!r}")
        assert first.aa_prob == pytest.approx(NEAR_CERTAIN, abs=1e-7)
    finally:
        os.unlink(out)


def test_the_confidence_matches_the_posterior_it_summarises():
    out = _write(".vcf")
    try:
        first = next(iter(anc.Reader(out).annotations()))
        assert first.posterior is not None
        assert first.aa_prob == pytest.approx(max(first.posterior.values()))
    finally:
        os.unlink(out)


def test_bcf_refuses_the_posterior_rather_than_dropping_it():
    """cyvcf2 cannot write a multi-valued Type=Float INFO field to BCF."""
    _, pairs = _calls()
    out = tempfile.mktemp(suffix=".bcf")
    try:
        with pytest.raises(ValueError, match="store_posterior"):
            VCFWriter(SRC, out).write(iter(pairs), store_posterior=True)
    finally:
        if os.path.exists(out):
            os.unlink(out)
