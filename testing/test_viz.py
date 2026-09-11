"""Smoke tests for `Tree.draw_text` / `Tree.draw_svg`: both the tskit
delegation path (on `TskitLocalTree`) and the generic ASCII fallback (on a
hand-built `_ManualTree`).

The exact string output is not pinned (brittle). The tests check that the
renderer runs, includes the expected sample/allele tokens, and raises in the
documented case.
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import Posterior, STATES
from ancestree.focal import FocalNode
from ancestree.plotting import FocalTreePlot
from ancestree.sites import Site
from ancestree.trees import OutgroupLadderTree, TskitLocalTree
from testing._helpers import panel_site, panel_ts


def _polytomy_tree():
    from testing.test_likelihood import _polytomy_tree as p
    return p()


def _ladder() -> OutgroupLadderTree:
    """Six ingroup haplotypes below three outgroups, ``K0..K4`` set explicitly."""
    ladder = OutgroupLadderTree([f"i{k}" for k in range(6)], ["o1", "o2", "o3"])
    ladder.set_params([0.004, 0.008, 0.006, 0.014, 0.020])
    return ladder


def test_tskit_draw_text_runs(small_ts):
    tree = TskitLocalTree(small_ts, position=0.0)
    out = tree.draw_text()
    assert isinstance(out, str) and len(out) > 0


def test_tskit_draw_text_with_site_inlines_alleles(small_ts):
    tree = TskitLocalTree(small_ts, position=0.0)
    samples = [str(int(s)) for s in small_ts.samples()]
    site = Site(
        chrom="1", pos=1, alleles=("A", "C"),
        tip_alleles={s: ("A" if i % 2 == 0 else "C") for i, s in enumerate(samples)},
    )
    out = tree.draw_text(site=site)
    for s in samples:
        assert f"{s}=" in out


def test_tskit_draw_text_with_posterior_shows_map(small_ts):
    tree = TskitLocalTree(small_ts, position=0.0)
    posterior = Posterior(alleles=STATES, values=np.array([0.7, 0.2, 0.05, 0.05]))
    out = tree.draw_text(posterior=posterior)
    assert "MAP=A" in out
    assert "0.700" in out


def test_generic_draw_text_runs():
    """Default ASCII renderer used by Tree subclasses that do not override
    (NexusTree / OutgroupLadderTree once they exist)."""
    tree = _polytomy_tree()
    out = tree.draw_text()
    assert isinstance(out, str)
    for n in ("0", "1", "2"):
        assert n in out


def test_generic_draw_text_with_site_inlines_alleles():
    tree = _polytomy_tree()
    site = Site(
        chrom="1", pos=1, alleles=("A", "C", "G"),
        tip_alleles={"A": "A", "B": "C", "C": "G"},
    )
    out = tree.draw_text(site=site)
    assert "A=A" in out
    assert "B=C" in out
    assert "C=G" in out


def test_tskit_draw_svg_runs(small_ts):
    tree = TskitLocalTree(small_ts, position=0.0)
    out = tree.draw_svg()
    assert "<svg" in out


def test_generic_draw_svg_runs_on_a_non_tskit_backend():
    """Every Tree renders through tskit, not only the tskit-backed one."""
    out = _polytomy_tree().draw_svg()
    assert "<svg" in out


def test_generic_draw_svg_marks_the_focal_position():
    """A focal part-way along a branch is drawn on that branch."""
    from ancestree._viz import FOCAL_COLOUR, TreeRenderer

    tree = _polytomy_tree()
    on_node = TreeRenderer.draw_svg(tree, focal_at=(tree.root, 0.0))
    assert FOCAL_COLOUR in on_node
    kid = next(iter(tree.children(tree.root)))
    mid = TreeRenderer.draw_svg(
        tree, focal_at=(kid, 0.5 * tree.branch_length(kid)))
    # The spliced point is an extra node, so the mid-branch mark carries one
    # more node than the on-node one.
    assert mid.count("class=\"") > on_node.count("class=\"")


class TestRendererLabels:
    def test_a_tip_the_site_did_not_observe_is_marked_unknown(self):
        ts, ids = panel_ts()
        tree = TskitLocalTree(ts, position=0.0)
        site = panel_site(ids)
        del site.tip_alleles[str(ids["b"])]
        out = tree.draw_text(site=site)
        assert f"{ids['b']}=?" in out
        assert f"{ids['a']}=A" in out

    def test_the_svg_root_carries_the_map_state(self):
        posterior = Posterior(alleles=STATES,
                              values=np.array([0.7, 0.2, 0.05, 0.05]))
        svg = _ladder().draw_svg(posterior=posterior)
        assert "MAP=A 0.7000" in svg


class TestFocalTreePlotAnchors:
    """Where the focal marker lands on a genealogy drawn as a canvas."""

    def test_the_ingroup_mrca_anchor_needs_an_ingroup(self):
        ts, _ = panel_ts()
        plot = FocalTreePlot(TskitLocalTree(ts, position=0.0))
        with pytest.raises(ValueError, match="ingroup_samples="):
            plot.focal_point()

    @pytest.mark.parametrize("ingroup", [("a", "b", "c", "d"), ("a", "c")])
    def test_the_anchor_is_the_mrca_of_the_ingroup_tips(self, ingroup):
        ts, ids = panel_ts()
        plot = FocalTreePlot(TskitLocalTree(ts, position=0.0),
                             ingroup_samples=[str(ids[s]) for s in ingroup])
        # The canvas is six units tall, so a node at time 3 sits at y = 3.
        assert plot.focal_point()[1] == pytest.approx(3.0)

    def test_a_single_ingroup_tip_anchors_on_the_node_above_it(self):
        ts, ids = panel_ts()
        plot = FocalTreePlot(TskitLocalTree(ts, position=0.0),
                             ingroup_samples=[str(ids["a"])])
        assert plot.focal_point()[1] == pytest.approx(1.0)

    def test_a_fraction_above_a_root_anchor_stays_on_the_root(self):
        ts, ids = panel_ts()
        everyone = [str(ids[s]) for s in ("a", "b", "c", "d", "o1", "o2")]
        plot = FocalTreePlot(TskitLocalTree(ts, position=0.0),
                             FocalNode("ingroup_mrca", fraction=0.5),
                             ingroup_samples=everyone)
        root = FocalTreePlot(TskitLocalTree(ts, position=0.0), "panel_root")
        assert plot.focal_point() == pytest.approx(root.focal_point())
        assert plot.focal_point()[1] == pytest.approx(6.0)

    def test_a_depth_on_a_genealogy_is_measured_from_the_sample_plane(self):
        ts, ids = panel_ts()
        plot = FocalTreePlot(TskitLocalTree(ts, position=0.0),
                             FocalNode("ingroup_mrca", depth=4.5),
                             ingroup_samples=[str(ids[s]) for s in "abcd"])
        # Half-way up the branch from the ingroup MRCA (time 3) to the root
        # (time 6).
        assert plot.focal_point()[1] == pytest.approx(4.5)

    def test_one_coalescence_lands_on_the_first_join_of_a_ladder(self):
        ladder = _ladder()
        by_join = FocalTreePlot(ladder, FocalNode("ingroup_mrca", coalescences=1))
        # K0 is the branch from the ingroup MRCA to the first join, so a
        # depth of exactly K0 names the same node.
        by_depth = FocalTreePlot(ladder, FocalNode("ingroup_mrca", depth=0.004))
        below = FocalTreePlot(ladder).focal_point()[1]
        above = FocalTreePlot(ladder, "panel_root").focal_point()[1]
        assert by_join.focal_point() == pytest.approx(by_depth.focal_point())
        assert below < by_join.focal_point()[1] < above
