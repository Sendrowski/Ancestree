"""Tests for :class:`ancestree.trees.OutgroupLadderTree`.

Covers:

- Topology shape for ``n = 1, 2, 3`` outgroups (node count, child–parent
  edges, parameter cardinality).
- Sample-id ↔ tip-node mapping (ingroup samples are metadata only and
  resolve to ``None``. Outgroup samples resolve to their tip nodes).
- Free branch-rate parameter machinery (``n_params``, ``param_names``,
  ``param_vector`` round-trip via ``set_params``).
- ``outgroup_divergence()`` agreement with fastDFE's convention.
- Kernel sanity: Felsenstein on an outgroup-only Site returns sensible
  posteriors over candidate ingroup-MRCA ancestral states.
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import JC69, OutgroupLadderTree, Likelihood, Posterior, STATES, Site
from ancestree.focal import FocalNode
from ancestree.trees import RerootedTree


def _three_outgroup_tree(n_ingroup: int = 4) -> OutgroupLadderTree:
    return OutgroupLadderTree(
        ingroup_samples=[f"i{k}" for k in range(n_ingroup)],
        outgroup_samples=["O1", "O2", "O3"],
    )


def _two_outgroup_tree() -> OutgroupLadderTree:
    """A two-outgroup ladder with divergences 0.2 and 0.6.

    The backbone runs ``I -> n_1 -> O_2`` with branch lengths ``0.1`` and
    ``0.5``, the branch above ``O_1`` is ``0.1``, and the readout may be
    placed up to ``0.3`` above the ingroup MRCA.
    """
    return OutgroupLadderTree.from_divergences(
        ["i0", "i1"], ["o1", "o2"], [0.2, 0.6],
    )


class TestTopologyShape:
    def test_one_outgroup_has_two_nodes(self):
        t = OutgroupLadderTree(["i0", "i1", "i2"], ["O1"])
        # 1 root (I) + 1 outgroup tip = 2 nodes. No ladder. The ingroup is
        # seeded on I rather than represented by a tip.
        assert t.n_nodes == 2
        assert t.n_tips() == 1
        assert t.n_outgroups == 1
        assert t.n_ingroup == 3

    def test_two_outgroups_have_four_nodes(self):
        t = OutgroupLadderTree(["i0", "i1", "i2"], ["O1", "O2"])
        # 1 I + 2 outgroup tips + 1 ladder internal = 4
        assert t.n_nodes == 4
        assert t.n_outgroups == 2

    def test_three_outgroups_have_six_nodes(self):
        t = _three_outgroup_tree()
        # 1 I + 3 outgroup tips + 2 ladder internals = 6
        assert t.n_nodes == 6
        assert t.n_outgroups == 3

    def test_root_is_the_ingroup_mrca_above_the_ladder(self):
        """I is the root and carries the ladder as its only child. The
        ingroup is seeded on I rather than hanging off it as a tip."""
        for n_out in (2, 3, 4):
            t = OutgroupLadderTree(["i0"], [f"O{k+1}" for k in range(n_out)])
            assert t.root == t.ingroup_mrca
            assert len(list(t.children(t.root))) == 1

    def test_root_carries_the_single_outgroup_directly(self):
        t = OutgroupLadderTree(["i0"], ["O1"])
        kids = list(t.children(t.root))
        assert kids == [t.tip_for_sample("O1")]

    def test_postorder_children_before_parents(self):
        t = _three_outgroup_tree()
        seen: set[int] = set()
        for node in t.postorder():
            for child in t.children(node):
                assert child in seen, f"postorder violated at node {node}"
            seen.add(node)

    def test_postorder_visits_every_node_exactly_once(self):
        t = _three_outgroup_tree()
        po = list(t.postorder())
        assert len(po) == t.n_nodes
        assert len(set(po)) == t.n_nodes


class TestSampleMapping:
    def test_outgroup_samples_resolve_to_tips(self):
        t = OutgroupLadderTree(["i0", "i1"], ["chimp", "gorilla"])
        for sid in ("chimp", "gorilla"):
            assert t.tip_for_sample(sid) is not None

    def test_ingroup_samples_are_metadata_only_not_tips(self):
        t = OutgroupLadderTree(["alpha", "beta"], ["X", "Y"])
        # Ingroup samples are stored as metadata but do not resolve to
        # tip nodes, their data is handled by the Kingman prior, not
        # by Felsenstein on individual tips.
        assert t.tip_for_sample("alpha") is None
        assert t.tip_for_sample("beta") is None
        # But still accessible as metadata
        assert "alpha" in t.ingroup_samples
        assert t.n_ingroup == 2

    def test_unknown_sample_returns_none(self):
        t = OutgroupLadderTree(["i0"], ["O1"])
        assert t.tip_for_sample("missing") is None


class TestParameterMachinery:
    """All ``2n − 1`` ladder branches are exposed as free parameters
    ``(K0, K1, …, K_{2n-2})`` (EST-SFS-style, matching Keightley &
    Jackson 2018), no pinning of K_0.
    """

    @pytest.mark.parametrize("n_out, expected_params", [
        (1, ("K0",)),
        (2, ("K0", "K1", "K2")),
        (3, ("K0", "K1", "K2", "K3", "K4")),
        (5, ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8")),
    ])
    def test_identifiable_param_names(self, n_out, expected_params):
        t = OutgroupLadderTree(["i0"], [f"O{k+1}" for k in range(n_out)])
        assert t.n_params == len(expected_params)
        assert t.param_names == expected_params

    def test_param_vector_round_trips_via_set_params(self):
        t = _three_outgroup_tree()
        new_rates = np.linspace(0.01, 0.05, t.n_params)
        t.set_params(new_rates)
        np.testing.assert_allclose(t.param_vector, new_rates)

    def test_set_params_rejects_wrong_length(self):
        t = _three_outgroup_tree()
        with pytest.raises(ValueError, match="length"):
            t.set_params(np.zeros(t.n_params + 1))

    def test_param_vector_is_a_copy(self):
        t = _three_outgroup_tree()
        v = t.param_vector
        v[0] = 999.0
        assert t.param_vector[0] != 999.0


class TestOutgroupDivergence:
    def test_one_outgroup_divergence_equals_k0(self):
        t = OutgroupLadderTree(["i0", "i1"], ["O1"])
        t.set_params(np.array([0.003]))
        np.testing.assert_allclose(t.outgroup_divergence(), [0.003])

    def test_two_outgroup_divergences_match_fastdfe_convention(self):
        """For n=2 the external params ``(K0, K1, K2)`` are the three ladder
        branches. ``outgroup_divergence()`` returns ``[K0+K1, K0+K2]``.
        """
        t = OutgroupLadderTree(["i0"], ["O1", "O2"])
        t.set_params(np.array([0.0, 0.003, 0.003]))  # K0, K1, K2
        np.testing.assert_allclose(t.outgroup_divergence(), [0.003, 0.003])

    def test_three_outgroup_divergences_chain_correctly(self):
        """For n=3 the external params ``(K0, K1, K2, K3, K4)`` map 1-to-1
        to the five internal ladder branches.
        """
        t = _three_outgroup_tree()
        K0, K1, K2, K3, K4 = 0.001, 0.002, 0.0015, 0.003, 0.0055
        t.set_params(np.array([K0, K1, K2, K3, K4]))
        np.testing.assert_allclose(
            t.outgroup_divergence(),
            [K0 + K1, K0 + K2 + K3, K0 + K2 + K4],
        )


# ------------------------------------------------------ OutgroupLadderTree
class TestLadderFromDivergences:
    def test_divergences_round_trip_for_every_outgroup_count(self):
        """The realised I -> O_k path equals the requested divergence.

        With one outgroup there is no join node, so the single branch is the
        whole path and must carry the full divergence, not the half the
        multi-outgroup split uses.
        """
        for divs in ([0.11], [0.05, 0.11], [0.008, 0.024, 0.04]):
            outs = [f"o{i + 1}" for i in range(len(divs))]
            t = OutgroupLadderTree.from_divergences(["i1", "i2"], outs, divs)
            parent = {c: n for n in t.postorder() for c in t.children(n)}
            for k, want in enumerate(divs):
                node, got = t.tip_for_sample(outs[k]), 0.0
                while node in parent:
                    got += t.branch_length(node)
                    node = parent[node]
                assert got == pytest.approx(want), (len(divs), outs[k], got, want)

    def test_single_outgroup_valid_and_properties(self):
        t = OutgroupLadderTree.from_divergences(["i1"], ["o1"], [1.0])
        assert t.n_outgroups == 1
        assert t.n_tips() == 1  # the outgroup. The ingroup is not a tip
        assert t.ingroup_samples == ("i1",)
        assert t.outgroup_samples == ("o1",)
        assert t.n_nodes == len(t.postorder())

    def test_divergence_length_mismatch(self):
        with pytest.raises(ValueError, match="divergences for"):
            OutgroupLadderTree.from_divergences(["i1"], ["o1", "o2"], [1.0])

    def test_nonpositive_divergences(self):
        with pytest.raises(ValueError, match="positive and increasing"):
            OutgroupLadderTree.from_divergences(["i1"], ["o1"], [0.0])


class TestKernelOnOutgroupOnlySite:
    """The kernel runs on Sites carrying outgroup observations only.

    The Felsenstein root partial is exactly the per-state likelihood of the
    observed outgroup tips conditional on the ingroup MRCA state, what est-sfs
    calls ``p_1`` (if MAP allele is major-ingroup) and ``p_2`` (if MAP allele
    is minor-ingroup). :class:`FixedTreeInference` combines these with the
    Kingman prior.
    """

    def test_all_outgroups_agree_strongly_supports_their_allele(self):
        """When every outgroup observes the same allele on short branches,
        posterior at I (= root) should put nearly all mass on that allele."""
        t = OutgroupLadderTree(["i0", "i1"], ["O1", "O2"])
        t.set_params(np.array([0.0, 1e-4, 1e-4])) 
        site = Site(
            chrom="1", pos=1, alleles=("G",),
            tip_alleles={"O1": "G", "O2": "G"},
        )
        log_L = Likelihood(JC69()).log_likelihoods(t, [site])[0]
        post = np.exp(log_L - np.logaddexp.reduce(log_L))
        assert STATES[int(np.argmax(post))] == "G"
        assert post[STATES.index("G")] > 0.99

    def test_split_outgroups_give_intermediate_posterior(self):
        """One outgroup says A, the other says C, on equal short branches.
        Posterior should be split between A and C, with the other two
        states essentially zero."""
        t = OutgroupLadderTree(["i0"], ["O1", "O2"])
        t.set_params(np.array([0.0, 1e-3, 1e-3])) 
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"O1": "A", "O2": "C"},
        )
        log_L = Likelihood(JC69()).log_likelihoods(t, [site])[0]
        post = Posterior(alleles=STATES, values=np.exp(log_L - np.logaddexp.reduce(log_L)))
        assert post["A"] + post["C"] > 0.99
        assert post["G"] < 0.01 and post["T"] < 0.01

    def test_missing_outgroup_marginalises_out(self):
        """A site with all outgroups missing should give a uniform posterior
        (the kernel learns nothing about the root state)."""
        t = OutgroupLadderTree(["i0"], ["O1", "O2"])
        t.set_params(np.array([0.0, 0.01, 0.01])) 
        site = Site(
            chrom="1", pos=1, alleles=(),
            tip_alleles={"O1": None, "O2": None},
        )
        log_L = Likelihood(JC69()).log_likelihoods(t, [site])[0]
        post = np.exp(log_L - np.logaddexp.reduce(log_L))
        # Uniform: every state ≈ 0.25
        np.testing.assert_allclose(post, 0.25, atol=1e-12)


class TestConstructorValidation:
    def test_empty_outgroups_raises(self):
        with pytest.raises(ValueError, match="outgroup"):
            OutgroupLadderTree(["i0"], [])


class TestDrawText:
    """``draw_text`` should produce ASCII output naming the outgroup samples."""

    def test_includes_outgroup_sample_ids_and_root_label(self):
        t = _three_outgroup_tree()
        t.set_params(np.array([0.0, 0.01, 0.02, 0.03, 0.04]))
        text = t.draw_text()
        assert text  # non-empty
        # Each outgroup sample id appears verbatim as a tip label.
        for sid in ("O1", "O2", "O3"):
            assert sid in text, f"missing outgroup {sid!r} in:\n{text}"
        # Root carries the conventional ``I`` label for the ingroup MRCA.
        assert "I" in text

    def test_no_branch_lengths_option(self):
        t = OutgroupLadderTree(["i0"], ["O1", "O2"])
        t.set_params(np.array([0.0, 0.05, 0.05]))
        text = t.draw_text(show_branch_lengths=False)
        assert "[bl=" not in text
        assert "O1" in text and "O2" in text


class TestRerootedView:
    """Placement, delegation and validation of a re-rooted ladder view."""

    def test_placement_on_the_deepest_branch(self):
        ladder = _two_outgroup_tree()
        view = ladder.at_focal(FocalNode("ingroup_mrca", depth=0.2))
        assert isinstance(view, RerootedTree)
        o2 = ladder.tip_for_sample("o2")
        anchor, tau = view.placement
        assert anchor == o2
        assert tau == pytest.approx(0.4)
        assert repr(view) == f"RerootedTree(anchor={o2}, tau=0.4)"

    def test_placement_stacks_through_a_view_of_a_view(self):
        """A depth past the panel root adds its remainder to the deep end."""
        ladder = _two_outgroup_tree()
        deep = ladder.as_deep_rooted()
        anchor, tau = deep.placement
        assert anchor == ladder.tip_for_sample("o2")
        assert tau == pytest.approx(0.3)
        beyond = ladder.at_focal(FocalNode("ingroup_mrca", depth=0.5))
        assert isinstance(beyond, RerootedTree)
        assert beyond.placement[0] == anchor
        assert beyond.placement[1] == pytest.approx(0.5)

    def test_rejects_a_focal_node_outside_the_tree(self):
        with pytest.raises(ValueError, match="not in this tree"):
            RerootedTree(_two_outgroup_tree(), 99)

    def test_rejects_tau_past_the_branch_above_focal(self):
        ladder = _two_outgroup_tree()
        o1 = ladder.tip_for_sample("o1")
        with pytest.raises(ValueError, match="exceeds the branch above"):
            RerootedTree(ladder, o1, 10.0)

    def test_time_scale_writes_through_to_the_source(self):
        ladder = _two_outgroup_tree()
        view = RerootedTree(ladder, ladder.tip_for_sample("o1"), 0.05)
        view.time_scale = 2.5
        assert ladder.time_scale == 2.5
        assert view.time_scale == 2.5

    def test_delegates_panel_and_deep_rooting_to_the_source(self):
        ladder = _two_outgroup_tree()
        view = RerootedTree(ladder, ladder.tip_for_sample("o1"), 0.05)
        assert view.ingroup_samples == ("i0", "i1")
        assert view.outgroup_samples == ("o1", "o2")
        assert view.ingroup_mrca == ladder.ingroup_mrca
        assert view.n_tips() == ladder.n_tips() == 2
        assert view.as_deep_rooted().placement == ladder.as_deep_rooted().placement


class TestLadderShape:
    """Repr, placement and divergence ordering of the ladder itself."""

    def test_repr_counts_both_panels(self):
        assert repr(_two_outgroup_tree()) == "OutgroupLadderTree(n_ingroup=2, n_outgroups=2)"

    def test_placement_is_the_ingroup_mrca(self):
        ladder = _two_outgroup_tree()
        assert ladder.placement == (ladder.ingroup_mrca, 0.0)

    def test_ordering_skips_sites_without_an_ingroup_allele(self):
        """A site no ingroup tip is called at counts for no outgroup.

        Without the skip the uncalled site would score as a disagreement for
        ``o1`` alone and put ``o2`` first.
        """
        sites = [
            Site(chrom="1", pos=1, alleles=("A", "C"),
                 tip_alleles={"i0": "A", "i1": "A", "o1": "A", "o2": "C"}),
            Site(chrom="1", pos=2, alleles=("C",),
                 tip_alleles={"i0": None, "i1": None, "o1": "C", "o2": None}),
            Site(chrom="1", pos=3, alleles=("C", "T"),
                 tip_alleles={"i0": "C", "i1": "C", "o1": "T", "o2": "C"}),
        ]
        order = OutgroupLadderTree.order_outgroups_by_divergence(
            sites, ["i0", "i1"], ["o1", "o2"],
        )
        assert order == ["o1", "o2"]
