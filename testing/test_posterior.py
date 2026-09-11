"""Tests for :class:`ancestree.Posterior`."""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import Posterior, STATES
from ancestree.posterior import InferenceSummary


def _p(a=0.7, c=0.1, g=0.1, t=0.1) -> Posterior:
    return Posterior(alleles=STATES, values=np.array([a, c, g, t], dtype=float))


def test_dict_like_indexing():
    p = _p()
    assert p["A"] == pytest.approx(0.7)
    assert p["C"] == pytest.approx(0.1)
    assert isinstance(p["A"], float)


def test_unknown_allele_raises():
    p = _p()
    with pytest.raises(KeyError):
        _ = p["N"]


def test_iter_yields_alleles():
    p = _p()
    assert list(p) == list(STATES)


def test_len():
    assert len(_p()) == 4


def test_contains():
    p = _p()
    assert "A" in p
    assert "N" not in p


def test_items():
    p = _p(0.4, 0.3, 0.2, 0.1)
    items = list(p.items())
    assert items == [("A", 0.4), ("C", 0.3), ("G", 0.2), ("T", 0.1)]


def test_to_dict():
    p = _p(0.4, 0.3, 0.2, 0.1)
    d = p.to_dict()
    assert d == {"A": 0.4, "C": 0.3, "G": 0.2, "T": 0.1}
    assert isinstance(d, dict)


def test_map_allele_and_max_prob():
    p = _p(0.1, 0.7, 0.1, 0.1)
    assert p.map_allele == "C"
    assert p.max_prob == pytest.approx(0.7)


def test_values_is_ndarray():
    p = _p()
    assert isinstance(p.values, np.ndarray)
    np.testing.assert_allclose(p.values, [0.7, 0.1, 0.1, 0.1])


def test_repr_includes_all_alleles():
    p = _p(0.4, 0.3, 0.2, 0.1)
    r = repr(p)
    for allele in STATES:
        assert allele in r


def test_frozen_cannot_reassign_fields():
    p = _p()
    with pytest.raises((AttributeError, TypeError)):
        p.alleles = ("A", "C")  # type: ignore[misc]


def test_equality_and_hashing():
    """``Posterior`` is ``eq=False`` with explicit ``__eq__`` and ``__hash__``.

    The generated dataclass ``__eq__`` would compare its ndarray field
    elementwise and return an array, so the class defines both dunders itself.
    """
    p1 = _p()
    p2 = _p()
    p3 = _p(a=0.4, c=0.4)

    assert p1 == p2
    assert not (p1 == p3)
    assert p1 != p3
    assert p1 != "not a posterior"

    assert hash(p1) == hash(p2)
    assert len({p1, p2}) == 1
    assert len({p1, p3}) == 2


def test_posterior_hash_consistent_across_dtypes():
    # Equal Posteriors must hash equally: __eq__ (np.array_equal) is dtype- and
    # signed-zero-agnostic, so __hash__ must be too.
    from ancestree.posterior import Posterior

    p64 = Posterior(("A", "C", "G", "T"), np.array([1., 0., 0., 0.], dtype=np.float64))
    p32 = Posterior(("A", "C", "G", "T"), np.array([1., 0., 0., 0.], dtype=np.float32))
    assert p64 == p32 and hash(p64) == hash(p32) and len({p64, p32}) == 1

    pz = Posterior(("A", "C"), np.array([0.0, 1.0]))
    nz = Posterior(("A", "C"), np.array([-0.0, 1.0]))
    assert pz == nz and hash(pz) == hash(nz)


def test_inference_summary_zero_sites_str():
    summ = InferenceSummary(0, {}, float("nan"), {}, float("nan"))
    assert str(summ) == "InferenceSummary: 0 sites"
