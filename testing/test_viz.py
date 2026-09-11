"""Smoke tests for `Tree.draw_text` / `Tree.draw_svg`: both the tskit
delegation path (on `TskitLocalTree`) and the generic ASCII fallback (on a
hand-built `_ManualTree`).

The exact string output is not pinned (brittle). The tests check that the
renderer runs, includes the expected sample/allele tokens, and raises in the
documented case.
"""
from __future__ import annotations

import numpy as np

from ancestree import Posterior, STATES
from ancestree.sites import Site
from ancestree.trees import TskitLocalTree


def _polytomy_tree():
    from testing.test_likelihood import _polytomy_tree as p
    return p()


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
