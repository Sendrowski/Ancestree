"""Tests for `TskitLocalTree`: post-order completeness, single-root requirement,
sample mapping, and the `from_tskit_tree` constructor.
"""
from __future__ import annotations

import msprime
import numpy as np
import pytest
import ancestree as anc

from ancestree.trees import Tree, TskitLocalTree


def test_postorder_visits_every_node(small_ts):
    tree = TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)
    po = list(tree.postorder())
    assert len(po) == tree.n_nodes
    assert tree.root in po


def test_postorder_children_before_parents(small_ts):
    tree = TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)
    seen: set[int] = set()
    for n in tree.postorder():
        for c in tree.children(n):
            assert c in seen, "post-order must visit children before parents"
        seen.add(n)


def test_n_tips_matches_num_samples(small_ts):
    tree = TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)
    assert tree.n_tips() == small_ts.num_samples


def test_default_sample_map_uses_str_of_int(small_ts):
    tree = TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)
    for s in small_ts.samples():
        node = tree.tip_for_sample(str(int(s)))
        assert node == int(s)
        assert tree.branch_length(node) >= 0


def test_custom_sample_map_respected(small_ts):
    mapping = {f"name_{int(s)}": int(s) for s in small_ts.samples()}
    tree = TskitLocalTree(small_ts, position=0.0, sample_map=mapping)
    for name, node in mapping.items():
        assert tree.tip_for_sample(name) == node
    assert tree.tip_for_sample("missing") is None


def test_sample_map_rejects_duplicate_individual_names():
    import msprime
    import tskit

    from ancestree.trees import TskitLocalTree

    ts = msprime.sim_ancestry(2, ploidy=1, sequence_length=10, random_seed=1)
    tables = ts.dump_tables()
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    tables.individuals.clear()
    tables.individuals.add_row(metadata={"name": "A"})
    tables.individuals.add_row(metadata={"name": "A"})  # same name, distinct ind
    ts2 = tables.tree_sequence()

    with pytest.raises(ValueError, match="duplicate haplotype name"):
        TskitLocalTree.sample_map_from_individuals(ts2)


def test_from_tskit_tree_matches_position_constructor(small_ts):
    pos = small_ts.sequence_length / 2
    t_pos = TskitLocalTree(small_ts, position=pos)
    raw_tree = small_ts.at(pos)
    t_direct = TskitLocalTree.from_tskit_tree(raw_tree)
    assert t_pos.root == t_direct.root
    assert tuple(t_pos.postorder()) == tuple(t_direct.postorder())


def test_multi_root_raises_without_explicit_root():
    """A multi-root local tree raises unless an explicit `root` is supplied;
    with `root=...` the wrapper acts on that root's subtree."""
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=1e3,
        recombination_rate=1e-5,
        population_size=1e4,
        end_time=1.0,  # very recent → most trees have multiple roots
        random_seed=1,
    )
    pos = None
    for tree in ts.trees():
        if tree.num_roots > 1:
            pos = tree.interval.left
            break
    assert not (pos is None), "could not induce a multi-root tree in the fixture"
    with pytest.raises(ValueError, match="num_roots"):
        TskitLocalTree(ts, position=pos)
    # With an explicit root the wrapper succeeds.
    one_root = int(next(iter(ts.at(pos).roots)))
    local = TskitLocalTree(ts, position=pos, root=one_root)
    assert local.root == one_root


def test_from_newick_ultrametric():
    """`TskitLocalTree.from_newick` parses an ultrametric Newick into a
    single-rooted local tree whose tip set and branch lengths match the input."""
    # Ultrametric: every leaf is at depth 0.4 from the root.
    tree = TskitLocalTree.from_newick(
        "((((i0:0.1,i1:0.1):0.05,(i2:0.1,i3:0.1):0.05):0.05,o1:0.2):0.2,o2:0.4);"
    )
    assert tree.n_tips() == 6
    assert set(tree._sample_to_node) == {"i0", "i1", "i2", "i3", "o1", "o2"}
    for name, bl in {"i0": 0.1, "i1": 0.1, "i2": 0.1, "i3": 0.1, "o1": 0.2, "o2": 0.4}.items():
        assert tree.branch_length(tree.tip_for_sample(name)) == pytest.approx(bl)


def test_from_newick_rejects_multi_root():
    with pytest.raises(ValueError, match="single rooted"):
        TskitLocalTree.from_newick("a:0.1; b:0.2;")


# --------------------------------------------------- time_scale setter guard
def test_time_scale_must_be_positive():
    tree = TskitLocalTree.from_newick("((a:1,b:1):1,c:2):0;")
    with pytest.raises(ValueError, match="time_scale must be positive"):
        tree.time_scale = -1.0


class TestFromNewickBranchLengths:

    def test_zero_length_internal_branch_raises_value_error(self):
        with pytest.raises(ValueError, match="positive branch length"):
            anc.TskitLocalTree.from_newick("((A:0,B:0):1,C:1);")

    def test_positive_branch_lengths_still_build(self):
        tree = anc.TskitLocalTree.from_newick("((A:1,B:1):1,C:2);")
        assert tree.n_tips() == 3


class TestTreeDefaults:
    """A tree that names no ingroup answers the base-class defaults."""

    @pytest.fixture
    def tree(self, small_ts):
        return TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)

    def test_no_ingroup_mrca(self, tree):
        assert tree.ingroup_mrca is None

    def test_no_ingroup_or_outgroup_samples(self, tree):
        assert tree.ingroup_samples == ()
        assert tree.outgroup_samples == ()

    def test_no_deep_rooting(self, tree):
        assert tree.as_deep_rooted() is None

    def test_repr_shows_tip_count_and_root(self, tree):
        assert repr(tree) == (
            f"TskitLocalTree(n_tips={tree.n_tips()}, root={tree.root})"
        )


# ────────────────────────────────────────── OutgroupLadderTree, deep readout


def _ladder(n_outgroups: int = 3):
    from ancestree import OutgroupLadderTree
    return OutgroupLadderTree.from_divergences(
        [f"i{k}" for k in range(4)],
        [f"o{k + 1}" for k in range(n_outgroups)],
        [0.01 * (k + 1) for k in range(n_outgroups)],
    )


def test_the_ladder_has_no_ingroup_tip():
    tree = _ladder()
    # The ingroup enters as one likelihood vector seeded on its MRCA, so no
    # node of the ladder stands for it and its ids resolve to no tip.
    assert tree.root == tree.ingroup_mrca
    assert tree.n_tips() == tree.n_outgroups
    assert tree.tip_for_sample("i0") is None


def test_the_deep_readout_splits_the_deepest_branch():
    tree = _ladder()
    deep = tree.as_deep_rooted()
    assert deep.root != tree.root
    # The panel's ancestor sits on the deepest outgroup's branch, so reading
    # there adds the point itself and leaves a bifurcating root, not the
    # trifurcation that rooting at the join below it would give.
    assert set(deep.postorder()) > set(tree.postorder())
    assert len(list(deep.children(deep.root))) == 2


def test_from_divergences_round_trips():
    from ancestree import OutgroupLadderTree
    div = [0.01, 0.02, 0.03]
    tree = OutgroupLadderTree.from_divergences(["i0"], ["o1", "o2", "o3"], div)
    np.testing.assert_allclose(tree.outgroup_divergence(), div, atol=1e-12)


def test_the_readout_slides_continuously_between_the_two_ends():
    from ancestree import JC69, Likelihood, STATES
    from ancestree.focal import FocalNode
    from ancestree.sites import Site

    tree = _ladder()
    site = Site(chrom="1", pos=1, alleles=("A", "G"),
                tip_alleles={"o1": "G", "o2": "A", "o3": "A"})
    engine = Likelihood(JC69())
    # The ingroup is fixed for G, entering as a delta on its own MRCA.
    seed = {tree.ingroup_mrca: np.array([[0.0, 0.0, 1.0, 0.0]])}

    def p_of_a(view):
        log_l = np.asarray(
            engine.log_likelihoods(view, [site], node_seeds=seed)
        )[0]
        w = np.exp(log_l - log_l.max()) * JC69().stationary()
        return (w / w.sum())[STATES.index("A")]

    seen = [p_of_a(tree.at_focal(FocalNode("ingroup_mrca", fraction=f)))
            for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    # At the ingroup MRCA the site restates its own allele. Support for the
    # outgroups' allele rises monotonically toward the deepest join.
    assert seen[0] == pytest.approx(0.0, abs=1e-12)
    assert all(b > a for a, b in zip(seen, seen[1:]))
    assert seen[-1] == pytest.approx(p_of_a(tree.as_deep_rooted()))





def test_draw_text_honours_show_branch_lengths(small_ts):
    """TskitLocalTree.draw_text honours show_branch_lengths."""
    tree = TskitLocalTree(small_ts, position=small_ts.sequence_length / 2)
    with_bl = tree.draw_text(show_branch_lengths=True)
    without_bl = tree.draw_text(show_branch_lengths=False)
    assert "[bl=" in with_bl
    assert "[bl=" not in without_bl


class _PlainTree(Tree):
    """A four-tip tree with no backend renderer, exercising the base ASCII renderer."""

    _CHILDREN = {6: (5, 3), 5: (4, 2), 4: (0, 1)}

    @property
    def root(self) -> int:
        return 6

    def children(self, node):
        return self._CHILDREN.get(node, ())

    def branch_length(self, node):
        return 0.0 if node == self.root else 0.5

    def postorder(self):
        return [0, 1, 4, 2, 5, 3, 6]

    def tip_for_sample(self, sample_id):
        return {"a": 0, "b": 1, "o1": 2, "o2": 3}.get(sample_id)

    def n_tips(self) -> int:
        return 4


def test_default_ascii_tree_render_with_branch_lengths():
    """The base renderer draws internal-node labels and [bl=...] overlays."""
    txt = _PlainTree().draw_text(show_branch_lengths=True)
    assert isinstance(txt, str) and txt


class TestTheTreeABCCoversSampleForTip:
    """A subclass implementing the declared contract must not crash on naming.

    ``sample_for_tip`` is declared on the ``Tree`` ABC with a default of
    ``None``, the value the concrete trees give for a tip they cannot name,
    since inverting the map needs each subclass's own sample list.
    """

    def test_the_abc_declares_it(self):
        assert hasattr(Tree, "sample_for_tip")

    def test_it_is_not_abstract(self):
        """Declaring it must not force every existing subclass to implement it."""
        assert "sample_for_tip" not in Tree.__abstractmethods__

    def test_a_minimal_subclass_names_no_tip_rather_than_raising(self):
        tree = _PlainTree()
        assert tree.sample_for_tip(0) is None

    def test_rerooting_a_minimal_subclass_still_resolves_names(self):
        from ancestree.trees import RerootedTree

        rerooted = RerootedTree(_PlainTree(), 5, 0.0)
        assert rerooted.sample_for_tip(0) is None
        assert rerooted.draw_text()

    def test_the_concrete_trees_still_name_their_tips(self):
        """The base default must not shadow a real implementation."""
        from ancestree.trees import OutgroupLadderTree

        ladder = OutgroupLadderTree(["i1"], ["o1", "o2"])
        tip = ladder.tip_for_sample("o1")
        assert ladder.sample_for_tip(tip) == "o1"
