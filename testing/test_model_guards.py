"""Model inputs that produced NaN matrices or aliased state.

Each reached the kernel silently: an all-NaN P(t) enters Felsenstein as a
valid array, and a shared cached matrix lets one caller's in-place edit change
another's answer.
"""
import numpy as np
import pytest

import ancestree as anc


def test_an_infinite_branch_length_is_refused():
    """t = +inf passes a t >= 0 test and gives NaN rows under mode='full'."""
    with pytest.raises(ValueError, match="finite t"):
        anc.JC69().transition_probs(np.inf)


def test_an_infinite_kappa_is_refused():
    """An infinite kappa gives an all-NaN Q, so the constructor refuses it."""
    with pytest.raises(ValueError, match="positive and finite"):
        anc.K2(kappa=np.inf)


def test_the_cached_matrix_cannot_be_mutated():
    """The cache is shared across every call at one key."""
    m = anc.JC69()
    P = m.transition_probs(0.1)
    with pytest.raises(ValueError):
        P[0, 0] = 0.5
    assert m.transition_probs(0.1)[0, 0] == pytest.approx(P[0, 0])


def test_pi_is_renormalised_so_rows_sum_to_one():
    """isclose admits a residual that F81's closed form carries into the rows."""
    pi = np.array([0.25, 0.25, 0.25, 0.250003])  # sums to 1 + 3e-6
    P = anc.F81().transition_probs(0.2, pi=pi)
    sums = np.asarray(P).sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-12)
