"""Degenerate inputs to FixedTreeInference and ARGBasedInference.

The contract for pathological inputs:

- All-missing tips (ingroup or outgroup): the likelihood marginalises via
  all-ones partials, and the prior falls back to uniform when no ingroup is
  observed.
- Empty sites list: ``.infer()`` yields nothing.
- Prior assigning ``-inf`` to every state: the ingroup node is marginalised
  out with a WARNING log, and the outgroup decides.
- Single ingroup sample (n=1): monomorphic by construction, so the prior is
  a delta on the observed allele.
"""
from __future__ import annotations

import logging

import msprime
import numpy as np
import pytest
import ancestree as anc

from ancestree import (
    AdaptiveIngroupWeight,
    ARGBasedInference,
    FixedTreeInference,
    JC69,
    KingmanIngroupWeight,
    OutgroupLadderTree,
    Site,
)
from ancestree.priors import IngroupWeight

from testing._helpers import no_counts as _no_counts


class TestAllMissingIngroup:
    """A site with no observed ingroup alleles gives a uniform prior."""

    def test_adaptive_uniform_when_ingroup_all_missing(self):
        prior = AdaptiveIngroupWeight(["i0", "i1", "i2"])
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": None, "i1": None, "i2": None},
        )
        log_prior = prior.log_probs([site])[0]
        np.testing.assert_allclose(np.exp(log_prior), [0.25, 0.25, 0.25, 0.25])


class TestEmptyInputs:
    """Empty sites list should yield nothing without errors."""

    def test_fixed_tree_inference_empty(self):
        tree = OutgroupLadderTree.from_newick(
            "((i0,i1):0.05,O1:0.10);",
            ingroup_samples=["i0", "i1"], outgroup_samples=["O1"],
        )
        inf = FixedTreeInference(
            [], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        assert list(inf.infer()) == []

    def test_arg_inference_empty(self):
        # A ts with no mutations → no sites.
        ts = msprime.sim_ancestry(
            samples=4, ploidy=1, sequence_length=1e3, random_seed=1,
            population_size=1e4,
        )
        assert ts.num_sites == 0
        results = list(
            ARGBasedInference(ts, JC69(), mu=1.0, progress=False).infer()
        )
        assert results == []

    def test_an_empty_source_warns_once_and_names_the_filters(self, caplog):
        """A contig label that matches nothing has to be visible.

        ``chrom_filter="chrZZZ"`` left every record at INFO, so a run over a
        VCF whose contigs are named ``1`` while the workflow passes ``chr1``
        wrote a syntactically valid unannotated file and exited 0 with nothing
        on any warning channel.
        """
        tree = OutgroupLadderTree.from_newick(
            "((i0,i1):0.05,O1:0.10);",
            ingroup_samples=["i0", "i1"], outgroup_samples=["O1"],
        )
        inf = FixedTreeInference(
            [], JC69(), _no_counts(), tree=tree, fit_required=False,
        )
        inf._chrom_filter = "chrZZZ"
        inf._sample_filter = ["i0", "i1", "O1"]
        with caplog.at_level(logging.WARNING,
                             logger="ancestree.FixedTreeInference"):
            assert list(inf.infer()) == []
            assert list(inf.infer()) == []
        warned = [r for r in caplog.records if "read no sites" in r.message]
        assert len(warned) == 1
        assert "chrom_filter='chrZZZ'" in warned[0].message
        assert "3 sample(s)" in warned[0].message

    def test_a_non_empty_source_does_not_warn(self, caplog):
        """The warning keys on the source, not on any downstream filtering."""
        ts = msprime.sim_ancestry(
            samples=4, ploidy=1, sequence_length=1e4, random_seed=3,
            population_size=1e4,
        )
        ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
        assert ts.num_sites > 0
        with caplog.at_level(logging.WARNING):
            list(ARGBasedInference(ts, JC69(), mu=1.0, progress=False).infer())
        assert not [r for r in caplog.records if "read no sites" in r.message]


class TestAllMissingOutgroup:
    """With every outgroup tip missing the likelihood is flat and the prior decides."""

    def test_fixed_tree_all_outgroup_missing(self):
        tree = OutgroupLadderTree.from_newick(
            "((i0,i1):0.05,O1:0.10);",
            ingroup_samples=["i0", "i1"], outgroup_samples=["O1"],
        )
        # Ingroup observation makes biallelic A/C; outgroup is missing.
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"i0": "A", "i1": "C", "O1": None},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            ingroup_weight=KingmanIngroupWeight(["i0", "i1"]),
            fit_required=False,
        )
        for _, post in inf.infer():
            # Kingman prior with 1 A + 1 C is 0.5 / 0.5 and the likelihood is
            # uniform (outgroup uninformative), so the posterior is the prior.
            assert post["A"] == pytest.approx(0.5)
            assert post["C"] == pytest.approx(0.5)
            assert post["G"] == 0
            assert post["T"] == 0


class _NoMassPrior(IngroupWeight):
    """Returns ``-inf`` for every state, so every row is degenerate."""

    def __init__(self):
        """Stateless."""

    def log_probs(self, sites):
        """Constant ``-inf`` matrix."""
        return np.full((len(sites), 4), -np.inf)


class TestPathologicalPrior:
    """A prior with -inf at every state marginalises the ingroup out and warns."""

    def test_all_inf_prior_marginalises_the_ingroup_out(self, caplog):
        class BadPrior(IngroupWeight):
            """Returns log P = -inf for every state at every site."""

            def __init__(self):
                """Stateless. Nothing to configure."""

            def log_probs(self, sites):
                """Constant ``-inf`` matrix."""
                return np.full((len(sites), 4), -np.inf)

        tree = OutgroupLadderTree.from_newick(
            "((i0,i1):0.05,O1:0.10);",
            ingroup_samples=["i0", "i1"], outgroup_samples=["O1"],
        )
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"O1": "A"},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            ingroup_weight=BadPrior(), fit_required=False,
        )
        with caplog.at_level(logging.WARNING, logger="ancestree.likelihood"):
            posteriors = [(s, p) for s, p in inf.infer()]
        assert len(posteriors) == 1
        _, post = posteriors[0]
        # The prior is the ingroup's likelihood on its own MRCA, so a row with
        # no mass marginalises the ingroup out and warns, rather than forcing
        # a degenerate posterior: the outgroup still decides.
        assert "no probability mass" in caplog.text
        assert np.isclose(post.values.sum(), 1.0)
        assert post.map_allele == "A"  # O1's allele, the only observation


def test_a_degenerate_prior_row_leaves_the_outgroup_deciding():
    """A degenerate prior row marginalises the node out rather than blanking the site.

    The posterior must be peaked, not uniform, and is pinned to a value so
    that a flat vector cannot pass on the ``argmax`` tie-break alone.
    """
    tree = OutgroupLadderTree.from_newick(
        "((i0,i1):0.05,O1:0.10);",
        ingroup_samples=["i0", "i1"], outgroup_samples=["O1"])
    site = Site(chrom="1", pos=1, alleles=("A", "C"), tip_alleles={"O1": "A"})
    inference = anc.FixedTreeInference(
        [site], anc.JC69(), _no_counts(), tree=tree,
        ingroup_weight=_NoMassPrior(), fit_required=False)
    (_site, post), = inference.infer()
    values = np.asarray(post.values)
    assert values.max() > 0.5, (
        f"the posterior is flat ({values.round(3).tolist()}); a degenerate "
        f"prior row blanked the site instead of marginalising the node out")
    assert values[0] == pytest.approx(0.864048064808, rel=1e-6)


class TestSingleIngroupSample:
    """n_ingroup == 1 is degenerate (always monomorphic) but must not raise."""

    def test_kingman_n1(self):
        prior = KingmanIngroupWeight(["only"])
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"only": "A"},
        )
        log_prior = prior.log_probs([site])[0]
        # n_ingroup == 1 is trivially mono-allelic, so a delta on its allele.
        np.testing.assert_allclose(np.exp(log_prior), [1.0, 0.0, 0.0, 0.0])

    def test_adaptive_n1_pre_fit(self):
        prior = AdaptiveIngroupWeight(["only"])
        # subsample_size clamps to 1 = n_ingroup. Construction must not raise.
        site = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"only": "A"},
        )
        log_prior = prior.log_probs([site])[0]
        np.testing.assert_allclose(np.exp(log_prior), [1.0, 0.0, 0.0, 0.0])
