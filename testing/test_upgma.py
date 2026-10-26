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


def _clades(z, n):
    """The leaf set of every merge of the linkage ``z`` over ``n`` leaves."""
    members = [frozenset([i]) for i in range(n)]
    for a, b, _height, _size in z:
        members.append(members[int(a)] | members[int(b)])
    return set(members[n:])


class TestUpgmaBatchMatchesScipy:
    """The unweighted :func:`~ancestree._ensemble.upgma_batch` is scipy's
    average linkage: the same merge heights, bit for bit, and the same
    clades, ties included."""

    @pytest.mark.parametrize("ties", [False, True], ids=["distinct", "ties"])
    @pytest.mark.parametrize("seed", range(25))
    def test_heights_and_clades(self, seed, ties):
        from scipy.cluster.hierarchy import linkage

        from ancestree._ensemble import upgma_batch

        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 16))
        n_pairs = n * (n - 1) // 2
        d = (rng.integers(0, 4, n_pairs).astype(float) if ties
             else rng.exponential(1.0, n_pairs))
        expected = linkage(d, method="average")
        z = np.empty((1, n - 1, 4))
        upgma_batch(d[None, :].copy(), n, z, np.empty((0, n_pairs)))
        np.testing.assert_array_equal(z[0, :, 2], expected[:, 2])
        np.testing.assert_array_equal(z[0, :, 3], expected[:, 3])
        assert _clades(z[0], n) == _clades(expected, n)


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
