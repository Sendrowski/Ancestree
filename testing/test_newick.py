"""Tests for OutgroupLadderTree.from_newick + FixedTreeInference(fit_required=False)."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import ancestree as anc

from ancestree import (
    FixedTreeInference,
    JC69,
    OutgroupLadderTree,
    Site,
    TskitLocalTree,
)

from testing._helpers import no_counts as _no_counts


class TestFromNewickTopology:
    """Topology + ordering extraction from a typical species-tree Newick."""

    def test_three_outgroups(self):
        nwk = "((((tsk_0,tsk_1,tsk_2):0.05,O_1:0.10):0.08,O_2:0.20):0.15,O_3:0.40);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1", "tsk_2"],
            outgroup_samples=["O_1", "O_2", "O_3"],
        )
        assert tree.outgroup_samples == ("O_1", "O_2", "O_3")
        assert tree.n_outgroups == 3
        # External params map 1-to-1 to all 2n-1=5 internal ladder branches:
        #   K_0 = 0.05 (I → n_1)
        #   K_1 = 0.10 (n_1 → O_1)
        #   K_2 = 0.08 (n_1 → n_2)
        #   K_3 = 0.20 (n_2 → O_2)
        #   K_4 = 0.15 + 0.40 = 0.55 (folded degree-2 Newick root → O_3)
        np.testing.assert_allclose(tree.param_vector, [0.05, 0.10, 0.08, 0.20, 0.55])

    def test_two_outgroups(self):
        nwk = "(((tsk_0,tsk_1):0.05,O_1:0.10):0.08,O_2:0.30);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"],
            outgroup_samples=["O_1", "O_2"],
        )
        assert tree.outgroup_samples == ("O_1", "O_2")
        # External params for n=2 (EST-SFS-style 2n-1): (K0, K1, K2) map 1-to-1
        # to K^int_0 (= I → n_1 = 0.05), K^int_1 (= n_1 → O_1 = 0.10), and
        # K^int_2 (= the degree-2 folded n_1 → O_2 backbone = 0.08+0.30 = 0.38).
        np.testing.assert_allclose(tree.param_vector, [0.05, 0.10, 0.38])

    def test_one_outgroup(self):
        nwk = "((tsk_0,tsk_1):0.05,O_1:0.20);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"], outgroup_samples=["O_1"],
        )
        assert tree.outgroup_samples == ("O_1",)
        # K_0 = 0.05 + 0.20 (both edges at the Newick root)
        np.testing.assert_allclose(tree.param_vector, [0.25])

    def test_single_ingroup_sample(self):
        nwk = "((tsk_0:0.05,O_1:0.10):0.08,O_2:0.30);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0"], outgroup_samples=["O_1", "O_2"],
        )
        assert tree.outgroup_samples == ("O_1", "O_2")

    def test_outgroup_divergences_from_full_ladder(self):
        """Every ``K^int_0..K^int_{2n-2}`` is exposed individually (no pinning);
        ``outgroup_divergence()`` sums the appropriate ladder edges.
        """
        nwk = "((((tsk_0,tsk_1):0.05,O_1:0.10):0.08,O_2:0.20):0.15,O_3:0.40);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"],
            outgroup_samples=["O_1", "O_2", "O_3"],
        )
        # K_i extracted above: K_0=0.05, K_1=0.10, K_2=0.08, K_3=0.20, K_4=0.55.
        expected = np.array([
            0.05 + 0.10,  # I → O_1 = K_0 + K_1
            0.05 + 0.08 + 0.20,  # I → O_2 = K_0 + K_2 + K_3
            0.05 + 0.08 + 0.55,  # I → O_3 = K_0 + K_2 + K_4
        ])
        np.testing.assert_allclose(tree.outgroup_divergence(), expected)

    def test_prunes_extra_leaves(self):
        """Newick may contain leaves not listed as ingroup or outgroup, they
        get pruned out, and degree-2 internals are collapsed with branch-
        length summation."""
        # Newick has 4 outgroups but only 2 are listed. The others (O_2, O_3) are pruned.
        nwk = "((((tsk_0,tsk_1):0.05,O_1:0.10):0.08,O_2:0.20):0.15,O_3:0.40);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"],
            outgroup_samples=["O_1"],
        )
        assert tree.outgroup_samples == ("O_1",)


class TestFromNewickErrors:
    def test_missing_ingroup_sample(self):
        nwk = "(((A,B):0.1,O_1:0.2):0.3,O_2:0.4);"
        with pytest.raises(ValueError, match="not found"):
            OutgroupLadderTree.from_newick(
                nwk, ingroup_samples=["A", "B", "ZZZ"],
                outgroup_samples=["O_1", "O_2"],
            )

    def test_missing_outgroup_sample(self):
        nwk = "(((A,B):0.1,O_1:0.2):0.3,O_2:0.4);"
        with pytest.raises(ValueError, match="not found"):
            OutgroupLadderTree.from_newick(
                nwk, ingroup_samples=["A", "B"],
                outgroup_samples=["O_1", "ZZZ"],
            )

    def test_overlap_between_ingroup_outgroup(self):
        nwk = "((A,B):0.1,O_1:0.2);"
        with pytest.raises(ValueError, match="appear in both"):
            OutgroupLadderTree.from_newick(
                nwk, ingroup_samples=["A", "B"], outgroup_samples=["A"],
            )

    def test_empty_outgroup_rejected(self):
        nwk = "(tsk_0,tsk_1);"
        with pytest.raises(ValueError, match="must be non-empty"):
            OutgroupLadderTree.from_newick(
                nwk, ingroup_samples=["tsk_0", "tsk_1"], outgroup_samples=[],
            )

    def test_paraphyletic_ingroup_rejected(self):
        """Ingroup samples cannot be split across non-sister clades."""
        nwk = "(((tsk_0:0.1,O_1:0.1):0.1,(tsk_1:0.1,O_2:0.1):0.1):0.1,O_3:0.5);"
        with pytest.raises(ValueError, match="non-ingroup leaves found"):
            OutgroupLadderTree.from_newick(
                nwk, ingroup_samples=["tsk_0", "tsk_1"],
                outgroup_samples=["O_1", "O_2", "O_3"],
            )

    def test_undated_newick_raises(self):
        """A Newick without branch lengths is rejected.

        The parser cannot distinguish an absent length from an explicit zero,
        so the check is on the derived K vector: a non-positive ladder branch
        means no divergence, which is never a usable tree. Callers who want
        the rates fitted should omit the Newick and build the ladder from the
        sample names instead.
        """
        with pytest.raises(ValueError, match="must be dated"):
            OutgroupLadderTree.from_newick(
                "(((tsk_0,tsk_1),O_1),O_2);", ingroup_samples=["tsk_0", "tsk_1"],
                outgroup_samples=["O_1", "O_2"],
            )

    def test_partially_dated_newick_raises(self):
        """One missing length is enough. It names the offending branch."""
        with pytest.raises(ValueError, match=r"but K\d+(, K\d+)* came out non-positive"):
            OutgroupLadderTree.from_newick(
                "(((tsk_0,tsk_1),O_1):0.1,O_2:0.3);",
                ingroup_samples=["tsk_0", "tsk_1"],
                outgroup_samples=["O_1", "O_2"],
            )

    def test_more_than_one_tree(self):
        with pytest.raises(ValueError, match="exactly one tree"):
            OutgroupLadderTree.from_newick("(a:1,b:1):0;(c:1,d:1):0;", ["a"], ["b"])

    def test_samples_missing_from_newick(self):
        with pytest.raises(ValueError, match="not found as"):
            OutgroupLadderTree.from_newick("(a:1,b:1):0;", ["x"], ["y"])


class TestFromNewickLadderViolations:
    """Topologies that are not an outgroup ladder are refused by name."""

    def test_two_siblings_at_one_step(self):
        nwk = "((i0:0.1,i1:0.1):0.2,o1:0.3,o2:0.4);"
        with pytest.raises(ValueError, match="expected exactly 1 sibling"):
            OutgroupLadderTree.from_newick(nwk, ["i0", "i1"], ["o1", "o2"])

    def test_outgroups_grouped_as_a_clade(self):
        nwk = "((i0:0.1,i1:0.1):0.2,(o1:0.1,o2:0.1):0.2);"
        with pytest.raises(ValueError, match="must be a leaf"):
            OutgroupLadderTree.from_newick(nwk, ["i0", "i1"], ["o1", "o2"])


class TestNewickImportGuard:
    """Both Newick constructors name the missing package in their error."""

    @pytest.fixture
    def without_newick(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "newick", None)

    def test_local_tree(self, without_newick):
        with pytest.raises(ImportError, match="pip install newick"):
            TskitLocalTree.from_newick("(a:1,b:1);")

    def test_ladder(self, without_newick):
        with pytest.raises(ImportError, match="pip install newick"):
            OutgroupLadderTree.from_newick(
                "((i0:1,i1:1):1,o1:1);", ["i0", "i1"], ["o1"],
            )


class TestNonMonophyleticIngroupRejected:
    """The MRCA-subtree check is the sole guard against a split ingroup."""

    @staticmethod
    def _build(newick: str):
        return anc.OutgroupLadderTree.from_newick(
            newick, ingroup_samples=["i1", "i2"], outgroup_samples=["o1", "o2"],
        )

    @pytest.mark.parametrize("newick", [
        "((i1:1,(o1:0.5,i2:0.5):0.5):1,o2:2);",  # outgroup nested in the ingroup
        "((i1:1,o1:1):1,(i2:1,o2:1):1);",  # ingroup split by an outgroup
    ])
    def test_non_monophyletic_ingroup_raises(self, newick):
        with pytest.raises(ValueError, match="monophyletic"):
            self._build(newick)

    def test_well_formed_ladder_still_builds(self):
        assert self._build("(((i1:1,i2:1):1,o1:2):1,o2:3);").n_outgroups == 2


class TestFitRequiredFalse:
    """FixedTreeInference(fit_required=False) skips MLE; uses tree's branch rates."""

    def test_infer_without_fit_works(self):
        nwk = "(((tsk_0,tsk_1,tsk_2):0.05,O_1:0.10):0.08,O_2:0.30);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1", "tsk_2"],
            outgroup_samples=["O_1", "O_2"],
        )
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"O_1": "A", "O_2": "A"},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        # No .fit(), should run.
        results = list(inf.infer())
        assert len(results) == 1
        _, post = results[0]
        # Two outgroups both A → MAP should be A.
        assert post.map_allele == "A"
        assert post.max_prob > 0.5

    def test_default_still_requires_fit(self):
        """The default fit_required=True keeps the existing RuntimeError contract."""
        nwk = "(((tsk_0,tsk_1):0.05,O_1:0.10):0.08,O_2:0.30);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"],
            outgroup_samples=["O_1", "O_2"],
        )
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"O_1": "A", "O_2": "C"},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            n_target_sites=1,
        )
        with pytest.raises(RuntimeError, match="called before fit"):
            list(inf.infer())

    def test_newick_branch_rates_are_used_verbatim(self):
        """Posterior with fit_required=False matches a direct kernel call at the same rates."""
        nwk = "(((tsk_0,tsk_1):0.05,O_1:0.10):0.08,O_2:0.30);"
        tree = OutgroupLadderTree.from_newick(
            nwk, ingroup_samples=["tsk_0", "tsk_1"],
            outgroup_samples=["O_1", "O_2"],
        )
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"O_1": "A", "O_2": "C"},
        )
        inf = FixedTreeInference(
            [site], JC69(), _no_counts(), tree=tree,
            fit_required=False,
        )
        _, post = next(iter(inf.infer()))

        # Compare against hand-running the Likelihood engine on the same tree.
        from ancestree.likelihood import Likelihood
        from scipy.special import logsumexp
        log_L = Likelihood(JC69()).log_likelihoods(tree, [site])[0]
        log_prior = np.full(4, -np.log(4))
        log_post = log_L + log_prior
        log_post -= logsumexp(log_post)
        expected = np.exp(log_post)
        np.testing.assert_allclose(post.values, expected, atol=1e-10)
