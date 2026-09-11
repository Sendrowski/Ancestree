"""Tests for the UPGMA window builder in :mod:`ancestree._upgma`."""
from __future__ import annotations

import numpy as np
import pytest

from ancestree._upgma import WindowedUpgmaTreeSequence


def test_upgma_deep_tree_no_recursion_error():
    # A deep (near-caterpillar) UPGMA topology is emitted iteratively, so it
    # does not overflow the interpreter recursion limit.
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    from ancestree._upgma import build_windowed_tree_sequence

    n = 1200
    idx = np.arange(n)
    d = np.maximum(idx[:, None], idx[None, :]).astype(float)  # d(i,j) = max(i,j)
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="average")  # chains into a caterpillar
    ts = build_windowed_tree_sequence(n, 1.0, [(0.0, 1.0)], [z])
    assert ts.num_samples == n


# --------------------------------------------------- UPGMA window builder
class TestWindowedUpgma:
    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 2 samples"):
            WindowedUpgmaTreeSequence(1, 10.0, [(0.0, 10.0)], [np.zeros(0)])

    def test_sample_names_length_mismatch_raises(self):
        # 2 samples → one condensed pairwise TMRCA per window.
        with pytest.raises(ValueError, match="sample_names length"):
            WindowedUpgmaTreeSequence(
                2, 10.0, [(0.0, 10.0)], [np.array([1.0])], sample_names=["only-one"],
            )


class TestWindowedUpgmaGuards:
    def test_window_lists_must_align(self):
        with pytest.raises(ValueError, match="must have the same length"):
            WindowedUpgmaTreeSequence(
                2, 10.0, [(0.0, 5.0), (5.0, 10.0)], [np.array([1.0])])

    def test_a_condensed_vector_must_hold_every_pair(self):
        with pytest.raises(ValueError, match=r"expected n\*\(n-1\)/2 = 3"):
            WindowedUpgmaTreeSequence(3, 10.0, [(0.0, 10.0)], [np.array([1.0])])
