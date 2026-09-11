"""Tests for rate-map support in :class:`~ancestree.inference.ARGBasedInference`.

The library accepts any object with a ``get_cumulative_mass(position)``
method as ``mu``, that is the method ``msprime.RateMap`` exposes, so
the standard rate-map class works without ``msprime`` being a runtime
dep. Coverage:

1. Uniform ``msprime.RateMap`` produces bit-identical posteriors to the
   scalar ``mu`` path.
2. Non-uniform ``msprime.RateMap`` produces posteriors that differ from
   a scalar at the mean rate, confirming the per-tree query actually
   flows through to the kernel.
3. A duck-typed shim (a tiny user-defined class with only
   ``mean_rate_in_interval``) is accepted by the constructor and produces
   the same output as a matched scalar.
4. Non-numeric, non-rate-map ``mu`` raises ``TypeError`` with a clear
   message naming the expected contract.
"""
import msprime
import numpy as np
import pytest

from ancestree import ARGBasedInference, JC69, TskitLocalTree


def _small_arg():
    """msprime ARG with enough recombination to give multiple local trees,
    and enough mutations for the inference to do real work.
    """
    import msprime
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=2_000,
        recombination_rate=1e-5,
        population_size=1e4,
        random_seed=1,
    )
    return msprime.sim_mutations(ts, rate=1e-5, random_seed=1)


def test_uniform_rate_map_matches_scalar_mu():
    """A uniform ``msprime.RateMap`` at rate r must produce per-site
    posteriors that are bit-identical to the scalar ``mu=r`` path.
    """
    import msprime
    ts = _small_arg()
    mu = 1.25e-8
    seq_len = ts.sequence_length

    inf_scalar = ARGBasedInference(
        ts, JC69(), mu=mu, progress=False,
    )
    inf_map = ARGBasedInference(
        ts, JC69(), mu=msprime.RateMap.uniform(seq_len, mu),
         progress=False,
    )

    scalar = list(inf_scalar.infer())
    mapped = list(inf_map.infer())

    assert len(scalar) == len(mapped) > 0
    for (s_site, s_post), (m_site, m_post) in zip(scalar, mapped):
        assert int(s_site.pos) == int(m_site.pos)
        np.testing.assert_allclose(s_post.values, m_post.values, atol=1e-12)


def test_nonuniform_rate_map_differs_from_uniform_scalar():
    """A non-uniform ``msprime.RateMap`` (10× hotter in the second half)
    must yield posteriors that differ from a scalar at the average rate
    at sites inside the hot region.
    """
    import msprime
    ts = _small_arg()
    seq_len = ts.sequence_length
    mu_low = 1e-8
    mu_high = 1e-7

    rm = msprime.RateMap(
        position=[0.0, seq_len / 2, seq_len],
        rate=[mu_low, mu_high],
    )
    mu_mean = (mu_low + mu_high) / 2

    inf_scalar = ARGBasedInference(
        ts, JC69(), mu=mu_mean, progress=False,
    )
    inf_map = ARGBasedInference(
        ts, JC69(), mu=rm, progress=False,
    )

    scalar_by_pos = {int(s.pos): p.values for s, p in inf_scalar.infer()}
    map_by_pos = {int(s.pos): p.values for s, p in inf_map.infer()}
    shared = set(scalar_by_pos) & set(map_by_pos)
    assert shared, "no shared positions between scalar and rate-map runs"

    # At least one site in the hot region must produce a numerically
    # different posterior. The rate-map path is taking the per-tree
    # mean-rate query, not the global average.
    differs = any(
        pos > seq_len / 2 and not np.allclose(
            scalar_by_pos[pos], map_by_pos[pos], rtol=0, atol=1e-12,
        )
        for pos in shared
    )
    assert differs, (
        "non-uniform RateMap produced identical posteriors to the matched-"
        "average scalar; the per-tree rate query is not taking effect."
    )


def test_duck_typed_rate_map_accepted():
    """Any object exposing ``get_cumulative_mass(position)`` must be
    accepted as ``mu``, that is the contract the library documents, and
    it is what lets ``msprime.RateMap`` work without ``msprime`` being a
    runtime dependency.
    """
    ts = _small_arg()
    rate = 1.25e-8

    class _DuckRateMap:
        """Uniform rate. Cumulative mass is just rate × position."""
        def get_cumulative_mass(self, position):
            return rate * position

    inf_duck = ARGBasedInference(
        ts, JC69(), mu=_DuckRateMap(),
        progress=False,
    )
    inf_scalar = ARGBasedInference(
        ts, JC69(), mu=rate, progress=False,
    )

    duck = list(inf_duck.infer())
    scalar = list(inf_scalar.infer())
    assert len(duck) == len(scalar) > 0
    for (d_site, d_post), (s_site, s_post) in zip(duck, scalar):
        assert int(d_site.pos) == int(s_site.pos)
        np.testing.assert_allclose(d_post.values, s_post.values, atol=1e-12)


def test_invalid_mu_type_rejected():
    """Non-float, non-rate-map ``mu`` values must raise TypeError with a
    clear message naming the expected contract.
    """
    ts = _small_arg()
    with pytest.raises(TypeError, match="get_cumulative_mass"):
        ARGBasedInference(ts, JC69(), mu="not a rate", progress=False)


def test_rate_map_scales_a_supplied_tree_by_its_interval(small_ts):
    """``infer_site`` on a tskit-backed tree reads the rate map over the
    tree's own interval, so it agrees with a scalar run at that rate."""
    length = small_ts.sequence_length
    rate_map = msprime.RateMap(position=[0.0, length / 2, length],
                               rate=[1e-8, 4e-8])
    mapped = ARGBasedInference(small_ts, JC69(), mu=rate_map, progress=False)
    scalar = ARGBasedInference(small_ts, JC69(), mu=4e-8, progress=False)
    tree = small_ts.at(0.75 * length)
    assert tree.interval.left >= length / 2
    site = next(s for s, _ in scalar.infer() if s.pos >= tree.interval.left)
    values = mapped.infer_site(TskitLocalTree.from_tskit_tree(tree), site).values
    expected = scalar.infer_site(TskitLocalTree.from_tskit_tree(tree), site).values
    np.testing.assert_allclose(values, expected, rtol=1e-12)
    assert not np.allclose(values, 0.25)
