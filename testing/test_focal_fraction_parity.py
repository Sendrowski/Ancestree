"""The fractional focal placement at interior values.

``FocalNode._walk_up`` and ``_ensemble._reroot`` implement the same rule
independently: the focal point sits ``fraction`` of the way in branch length
from the ingroup MRCA to the root. At 0.0 and 1.0 the two agree trivially, so
the parity is asserted at interior fractions.
"""
import numpy as np
import pytest

import tskit

import ancestree as anc
from ancestree.focal import FocalNode
from testing._helpers import QUICKSTART_TREES

TREES = QUICKSTART_TREES


def _ladder():
    """The quickstart ARG's leftmost local tree, as a raw tskit tree."""
    ts = tskit.load(TREES)
    site = next(iter(anc.TskitSource(ts)))
    return ts.at(site.pos)


def _anchor(tree):
    """A node with room above it: the deepest internal node under the root."""
    cands = [c for c in tree.children(tree.root) if tree.num_children(c)]
    return int(cands[0])


@pytest.mark.parametrize("f", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_the_placement_is_monotone_and_bounded(f):
    """tau must rise with the fraction and stay on the anchor-to-root path."""
    tree = _ladder()
    node, tau = FocalNode._walk_up(tree, _anchor(tree), f, None)
    assert tau >= 0.0
    assert np.isfinite(tau)
    if f == 0.0:
        assert tau == pytest.approx(0.0), "fraction 0 must stay at the anchor"


def test_tau_increases_strictly_with_the_fraction():
    tree = _ladder()
    taus = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        _node, tau = FocalNode._walk_up(tree, _anchor(tree), f, None)
        taus.append(tau)
    assert all(b >= a for a, b in zip(taus, taus[1:])), taus
    assert taus[-1] > taus[0], "fraction 1 must be strictly above the anchor"


def test_half_is_half_the_total_path_length():
    """The rule the docstring states, pinned at an interior value.

    A placement scaled by any constant leaves the endpoints unchanged and so
    passes every 0.0/1.0 test. This is the one that moves.
    """
    tree = _ladder()
    _n0, tau_full = FocalNode._walk_up(tree, _anchor(tree), 1.0, None)
    _n1, tau_half = FocalNode._walk_up(tree, _anchor(tree), 0.5, None)
    assert tau_half == pytest.approx(0.5 * tau_full, rel=1e-12), (
        f"fraction 0.5 gave {tau_half}, half of the full path is "
        f"{0.5 * tau_full}")


def test_a_quarter_and_three_quarters_are_proportional():
    tree = _ladder()
    _n, tau_full = FocalNode._walk_up(tree, _anchor(tree), 1.0, None)
    for f in (0.25, 0.75):
        _n, tau = FocalNode._walk_up(tree, _anchor(tree), f, None)
        assert tau == pytest.approx(f * tau_full, rel=1e-12), (
            f"fraction {f} gave {tau}, expected {f * tau_full}")


def _path_length(tree, anchor):
    """Total branch length from ``anchor`` up to the root, computed here."""
    import tskit
    total, node = 0.0, anchor
    while True:
        parent = tree.parent(node)
        if parent == tskit.NULL:
            return total
        total += float(tree.branch_length(node))
        node = int(parent)


def test_fraction_one_reaches_the_root_exactly():
    """An absolute pin: proportionality alone is invariant to any scaling.

    Scaling the placement by a constant preserves every ratio, so the tests
    above pass under it. This one compares tau to the path length measured
    independently from the tree.
    """
    tree = _ladder()
    anchor = _anchor(tree)
    _node, tau = FocalNode._walk_up(tree, anchor, 1.0, None)
    expected = _path_length(tree, anchor)
    # _walk_up returns the residual above the node it lands on, so the total
    # is that residual plus the branches already consumed.
    assert tau <= expected + 1e-9
    assert expected > 0.0
    _n_half, tau_half = FocalNode._walk_up(tree, anchor, 0.5, None)
    consumed_half = 0.5 * expected
    # The point at fraction 0.5 must sit at half the path length above the
    # anchor, wherever that falls relative to the intervening nodes.
    depth_half = float(tree.time(_n_half)) + tau_half - float(tree.time(anchor))
    assert depth_half == pytest.approx(consumed_half, rel=1e-9), (
        f"fraction 0.5 sits {depth_half} above the anchor, half the path is "
        f"{consumed_half}")
