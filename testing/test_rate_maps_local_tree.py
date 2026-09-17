"""The zero-rate and missing-interval guards, exercised through a real map.

These construct a mutation map with a zero-rate interval and a recombination
map with a masked one, so the guards are reached.

Without the emission floor, a zero-rate block loses its ``-s*lam`` term and the
emission becomes monotone in the coalescent rate, so any block carrying a
difference maximises at the deepest time bin. Without the NaN mask, a masked
interior interval contributes no cumulative mass, its step is 0, and ``r ** 0``
is 1, propagating the pairwise state across the gap as perfect linkage.
"""
import msprime
import numpy as np
import pytest

import ancestree as anc
from ancestree import LocalTreeInference
from testing._helpers import toy_inference, toy_sites

L = 40_000.0
BLOCK = 2000
NAMES = [f"h{i}" for i in range(4)]


def _sites(rng, n=200):
    pos = np.sort(rng.choice(int(L), size=n, replace=False))
    out = []
    for p in pos:
        g = rng.random(len(NAMES)) < 0.3
        alleles = ("A", "C")
        out.append(anc.Site(chrom="1", pos=int(p), alleles=alleles,
                           tip_alleles={s: alleles[int(b)]
                                        for s, b in zip(NAMES, g)}))
    return out


def _builder(**kw):
    rng = np.random.default_rng(11)
    return anc.LocalTreeBuilder(
        _sites(rng), mu=1.25e-8, rec_rate=1e-8, sample_names=NAMES,
        sequence_length=L, block_size=BLOCK, **kw)


def test_a_zero_rate_mutation_interval_is_floored_not_zero():
    """escale stays positive, or the block would maximise at the deepest bin."""
    mm = msprime.RateMap(position=[0.0, 6000.0, L], rate=[0.0, 1.25e-8])
    b = _builder(mutation_map=mm)
    _g, _bos, _pos, n_blocks = b._genotype_matrix()
    step, block_mask, escale = b._block_geometry(n_blocks)
    zero_blocks = escale[:3]
    assert np.all(zero_blocks > 0.0), (
        f"zero-rate blocks have escale {zero_blocks}; a zero scale makes the "
        "emission monotone in the coalescent rate")


def test_a_zero_rate_region_does_not_pin_tmrca_to_the_deepest_bin():
    """A zero-rate block must not carry the segment's deepest TMRCA.

    Flooring the emission scale alone does not achieve this: at any floor
    small enough to mean "no mutation here" the emission stays monotone in
    the coalescent rate. The block is masked uninformative instead.
    """
    mm = msprime.RateMap(position=[0.0, 6000.0, L], rate=[0.0, 1.25e-8])
    b = _builder(mutation_map=mm)
    g, _pos, bos, n_blocks = b._genotype_matrix()
    step, block_mask, escale = b._block_geometry(n_blocks)
    hmm = anc.PairwiseCoalescentHMM(n_haplotypes=len(NAMES), mu=1.25e-8,
                                   rec_rate=1e-8, block_size=BLOCK)
    t = hmm.block_tmrcas(g, bos, n_blocks, step=step, block_mask=block_mask,
                         emit_scale=escale)
    assert np.all(t[:, :3] < t.max())


def test_a_masked_recombination_interval_is_not_perfect_linkage():
    """A missing interior interval invalidates its step rather than giving r**0."""
    rm = msprime.RateMap(position=[0.0, 14000.0, 22000.0, L],
                         rate=[1e-8, np.nan, 1e-8])
    b = _builder(recombination_map=rm)
    _g, _bos, _pos, n_blocks = b._genotype_matrix()
    step, block_mask, escale = b._block_geometry(n_blocks)
    # A step whose blocks straddle the masked interval carries no cumulative
    # mass, so without the guard it reads as 0 and r ** 0 == 1 is perfect
    # linkage. The guard invalidates it, and an invalid step falls back to the
    # reference rate of 1.0.
    masked = step[7:10]
    assert np.all(masked > 0.5), (
        f"steps across the masked interval are {masked}; a near-zero step is "
        "perfect linkage across the gap")


def test_map_provenance_reports_map_paths_and_summaries():
    """A rate map is summarised, and a caller-supplied path recorded."""
    sites, names = toy_sites(range(0, 1000, 20))
    rate_map = msprime.RateMap(position=[0.0, 400.0, 1000.0],
                               rate=[1e-8, 3e-8])
    inf = toy_inference(sites, names, n_ensemble=None, sequence_length=1000.0,
                        recombination_map=rate_map,
                        accessibility=[(0.0, 500.0), (600.0, 1000.0)])
    inf._map_paths = {"recombination_map": "/maps/rec.txt",
                      "accessibility": "/maps/mask.bed"}
    out = inf._map_provenance()
    assert out["recombination_map_intervals"] == 2
    assert out["recombination_map_mean_rate"] == pytest.approx(
        float(rate_map.mean_rate))
    assert out["recombination_map_path"] == "/maps/rec.txt"
    assert out["accessibility_path"] == "/maps/mask.bed"
    assert out["accessibility_intervals"] == 2
    assert "mutation_map_intervals" not in out


def test_rate_map_provenance_tolerates_a_map_without_a_mean_rate():
    """A map whose mean rate is undefined still reports its interval count."""

    class _NoMean:
        rate = np.array([1e-8, 2e-8, 3e-8])

        @property
        def mean_rate(self):
            raise ZeroDivisionError("empty map")

    out = LocalTreeInference._rate_map_provenance("mutation_map", _NoMean())
    assert out == {"mutation_map_intervals": 3}
    assert LocalTreeInference._rate_map_provenance("mutation_map", None) == {}
