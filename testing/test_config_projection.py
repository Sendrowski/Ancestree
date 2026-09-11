"""Tests for :meth:`BaseComposition.project_sites_to_configs`.

Directly exercises the AFS-histogram projection kernel on small, hand-built
site sets, asserting the returned ``(config, weight)`` cells match values
computed by hand from the multivariate-hypergeometric soft projection. Also
pins the documented error path and the boundary/degenerate branches
(monomorphic-source counts, under-target ``n_called``, no called ingroup,
multi-allelic skip).
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import STATE_INDEX
from ancestree.sites import BaseComposition, Site


def _cells(configs, weights, sub_ids, outgroup_samples):
    """Reconstruct the ``{(sub_afs, og_pattern): weight}`` histogram.

    Rebuilds each returned synthetic Site's sub-sample AFS (counted over the
    first ``subsample_size`` ingroup ids) and outgroup pattern, so the
    returned list can be compared cell-by-cell against a hand-computed dict.
    """
    cells: dict[tuple[tuple[int, ...], tuple[str | None, ...]], float] = {}
    for site, w in zip(configs, weights):
        afs = [0, 0, 0, 0]
        for sid in sub_ids:
            a = site.tip_alleles.get(sid)
            if a is not None:
                afs[STATE_INDEX[a]] += 1
        og_pattern = tuple(site.tip_alleles.get(o) for o in outgroup_samples)
        cells[(tuple(afs), og_pattern)] = float(w)
    return cells


INGROUP = ["i0", "i1", "i2", "i3"]
OUTGROUP = ["o0"]


def _assert_cells(actual, expected):
    assert set(actual) == set(expected)
    for key, val in expected.items():
        assert actual[key] == pytest.approx(val)


def test_single_biallelic_site_hypergeometric_projection():
    """One 2-A/2-G ingroup site, projected to subsample_size=2.

    HG mass over the three reachable sub-AFS, denom = C(4,2) = 6:
      (A=0,G=2): C(2,0)C(2,2)/6 = 1/6
      (A=1,G=1): C(2,1)C(2,1)/6 = 4/6
      (A=2,G=0): C(2,2)C(2,0)/6 = 1/6
    All paired with the site's outgroup allele ('A',). Masses sum to 1.
    """
    bc = BaseComposition.no_counts()
    site = Site("1", 1, ("A", "G"),
                {"i0": "A", "i1": "A", "i2": "G", "i3": "G", "o0": "A"})
    configs, weights = bc.project_sites_to_configs([site], INGROUP, OUTGROUP, 2)

    expected = {
        ((0, 0, 2, 0), ("A",)): 1 / 6,
        ((1, 0, 1, 0), ("A",)): 4 / 6,
        ((2, 0, 0, 0), ("A",)): 1 / 6,
    }
    _assert_cells(_cells(configs, weights, INGROUP[:2], OUTGROUP), expected)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_two_biallelic_sites_share_and_sum_configs():
    """Two sites projected together: shared sub-AFS cells accumulate.

    Site1 = 2A/2G -> {(0,2):1/6, (1,1):4/6, (2,0):1/6}.
    Site2 = 3A/1G -> denom C(4,2)=6, k_A in {1,2}:
      (A=1,G=1): C(3,1)C(1,1)/6 = 3/6
      (A=2,G=0): C(3,2)C(1,0)/6 = 3/6
    Summed over the shared ('A',) outgroup pattern. Total mass = 2 sites.
    """
    bc = BaseComposition.no_counts()
    s1 = Site("1", 1, ("A", "G"),
              {"i0": "A", "i1": "A", "i2": "G", "i3": "G", "o0": "A"})
    s2 = Site("1", 2, ("A", "G"),
              {"i0": "A", "i1": "A", "i2": "A", "i3": "G", "o0": "A"})
    configs, weights = bc.project_sites_to_configs([s1, s2], INGROUP, OUTGROUP, 2)

    expected = {
        ((0, 0, 2, 0), ("A",)): 1 / 6,
        ((1, 0, 1, 0), ("A",)): 4 / 6 + 3 / 6,
        ((2, 0, 0, 0), ("A",)): 1 / 6 + 3 / 6,
    }
    _assert_cells(_cells(configs, weights, INGROUP[:2], OUTGROUP), expected)
    assert float(weights.sum()) == pytest.approx(2.0)


def test_full_projection_when_subsample_equals_called():
    """subsample_size == n_called leaves the AFS unchanged (denom = 1)."""
    bc = BaseComposition.no_counts()
    site = Site("1", 1, ("A", "G"),
                {"i0": "A", "i1": "A", "i2": "G", "i3": "G", "o0": "A"})
    configs, weights = bc.project_sites_to_configs([site], INGROUP, OUTGROUP, 4)

    expected = {((2, 0, 2, 0), ("A",)): 1.0}
    _assert_cells(_cells(configs, weights, INGROUP[:4], OUTGROUP), expected)


def test_monomorphic_counts_project_to_boundary_config():
    """Per-base ``counts`` become boundary configs with all-same outgroup.

    Five monomorphic-A sites in the composition, no polymorphic input:
    one cell at sub-AFS (subsample_size, 0, 0, 0) with an all-'A' outgroup
    pattern and weight equal to the raw count.
    """
    bc = BaseComposition.from_counts(A=5)
    configs, weights = bc.project_sites_to_configs([], INGROUP, OUTGROUP, 2)

    expected = {((2, 0, 0, 0), ("A",)): 5.0}
    _assert_cells(_cells(configs, weights, INGROUP[:2], OUTGROUP), expected)


def test_under_target_called_count_groups_on_full_afs():
    """n_called < subsample_size falls back to the raw full-n AFS cell."""
    bc = BaseComposition.no_counts()
    # Only one ingroup tip called. The rest missing.
    site = Site("1", 1, ("A",), {"i0": "A", "o0": "C"})
    configs, weights = bc.project_sites_to_configs([site], INGROUP, OUTGROUP, 2)

    expected = {((1, 0, 0, 0), ("C",)): 1.0}
    _assert_cells(_cells(configs, weights, INGROUP[:2], OUTGROUP), expected)


def test_no_called_ingroup_groups_on_outgroup_only():
    """No called ingroup info -> zero-AFS marker keyed on outgroup pattern."""
    bc = BaseComposition.no_counts()
    site = Site("1", 1, ("A",), {"o0": "C"})
    configs, weights = bc.project_sites_to_configs([site], INGROUP, OUTGROUP, 2)

    expected = {((0, 0, 0, 0), ("C",)): 1.0}
    _assert_cells(_cells(configs, weights, INGROUP[:2], OUTGROUP), expected)


def test_multiallelic_site_is_skipped():
    """>2 distinct ingroup alleles contribute nothing to the histogram."""
    bc = BaseComposition.no_counts()
    site = Site("1", 1, ("A", "C", "G"),
                {"i0": "A", "i1": "C", "i2": "G", "i3": "A", "o0": "A"})
    configs, weights = bc.project_sites_to_configs([site], INGROUP, OUTGROUP, 2)

    assert configs == []
    assert weights.shape == (0,)
    assert weights.dtype == np.float64


@pytest.mark.parametrize("bad", [0, -1])
def test_subsample_size_below_one_raises(bad):
    """The projection needs at least one sub-sampled haplotype.

    It carries no upper bound: an id may name an individual, so the length of
    the id list is not the number of ingroup haplotypes. The inference bounds
    the target against the haplotypes the panel resolves to.
    """
    bc = BaseComposition.no_counts()
    with pytest.raises(ValueError, match=r"subsample_size must be >= 1"):
        bc.project_sites_to_configs([], INGROUP, OUTGROUP, bad)
