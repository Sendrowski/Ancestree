"""The kernel against a hand-computed JC69 likelihood.

``test_matches_brute_force`` enumerates root states, but both sides call the
same ``transition_probs``, so an error in P(t) itself satisfies the comparison.
These values come from the closed form instead, written out here:

    P_same(d) = 1/4 + 3/4 exp(-4d/3)
    P_diff(d) = 1/4 - 1/4 exp(-4d/3)

with ``d`` the branch length in expected substitutions per site. For a root
with two tips at distances d1 and d2 observing alleles a and b,

    L(root = s) = P_{s,a}(d1) * P_{s,b}(d2)

which is evaluated below without touching the package's own matrices.
"""
import math

import numpy as np
import pytest

import ancestree as anc

STATES = ("A", "C", "G", "T")


def _p_same(d):
    return 0.25 + 0.75 * math.exp(-4.0 * d / 3.0)


def _p_diff(d):
    return 0.25 - 0.25 * math.exp(-4.0 * d / 3.0)


def _hand_likelihood(d1, d2, a, b):
    """L(root = s) for every s, from the closed form alone."""
    out = []
    for s in STATES:
        p1 = _p_same(d1) if s == a else _p_diff(d1)
        p2 = _p_same(d2) if s == b else _p_diff(d2)
        out.append(p1 * p2)
    return np.array(out)


@pytest.mark.parametrize("d1,d2", [(0.1, 0.1), (0.05, 0.3), (0.5, 0.5)])
def test_two_tip_likelihood_matches_the_closed_form(d1, d2):
    import tskit

    # Branch lengths are node-time differences, so the tips are placed below
    # the root by d1 and d2 rather than both at zero, which would give the
    # kernel two branches of max(d1, d2) whatever the parameters say.
    root_time = max(d1, d2)
    tables = tskit.TableCollection(sequence_length=2.0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=root_time - d1)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=root_time - d2)
    root = tables.nodes.add_row(time=root_time)
    tables.edges.add_row(left=0.0, right=2.0, parent=root, child=0)
    tables.edges.add_row(left=0.0, right=2.0, parent=root, child=1)
    tables.sort()
    ts = tables.tree_sequence()

    tree = anc.TskitLocalTree(ts, position=0.0)
    assert tree.branch_length(tree.tip_for_sample("0")) == pytest.approx(d1)
    assert tree.branch_length(tree.tip_for_sample("1")) == pytest.approx(d2)
    site = anc.Site(chrom="1", pos=0, alleles=("A", "C"),
                   tip_alleles={"0": "A", "1": "C"})

    lik = anc.Likelihood(anc.JC69(),
                        base_composition=anc.BaseComposition.no_counts())
    got = lik.log_likelihoods(tree, [site])[0]

    want = np.log(_hand_likelihood(d1, d2, "A", "C"))
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)
