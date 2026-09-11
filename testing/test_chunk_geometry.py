"""Chunked geometry that the block-aligned cases cannot expose.

The suite's other chunking tests use a window that is an exact multiple of the
block size and a chunk size that is a multiple of the window, which is the one
configuration in which several geometry defects are invisible. Everything here
is deliberately misaligned.
"""
import msprime
import numpy as np
import pytest

from ancestree import JC69, LocalTreeInference
from ancestree.sites import Site

#: Not a multiple of BLOCK_BP, so the window must be snapped up to one.
WINDOW_BP = 2500
BLOCK_BP = 1000
SEQUENCE_LENGTH = 200_000


def _sites(sequence_length=SEQUENCE_LENGTH, seed=7, drop=None):
    """A small phased panel. ``drop`` removes a ``(lo, hi)`` stretch of sites."""
    ts = msprime.sim_ancestry(
        5, ploidy=2, sequence_length=sequence_length, recombination_rate=1e-8,
        population_size=1e4, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=seed)
    names = [f"i{i}_h{h}" for i in range(5) for h in (0, 1)]
    out = []
    for variant in ts.variants():
        alleles = variant.alleles
        if len(alleles) != 2 or any(a not in "ACGT" for a in alleles if a):
            continue
        pos = int(variant.site.position)
        if drop is not None and drop[0] <= pos < drop[1]:
            continue
        out.append(Site(
            chrom="1", pos=pos, alleles=tuple(alleles),
            tip_alleles={names[k]: alleles[variant.genotypes[k]]
                         for k in range(len(names))}))
    return out, names


def _run(sites, names, **kwargs):
    """Posteriors keyed by position."""
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(SEQUENCE_LENGTH), window=WINDOW_BP,
        block_size=BLOCK_BP, n_ensemble=None, progress=False, **kwargs)
    return {site.pos: post.to_dict() for site, post in inference.infer()}


def _max_abs_difference(a: dict, b: dict) -> float:
    """Largest per-allele posterior difference over the shared sites."""
    worst = 0.0
    for pos, left in a.items():
        right = b[pos]
        for allele in set(left) | set(right):
            worst = max(worst,
                        abs(left.get(allele, 0.0) - right.get(allele, 0.0)))
    return worst


def test_an_unaligned_window_is_snapped_on_the_chunked_path():
    """The chunked window width is a whole number of blocks, as unchunked."""
    sites, names = _sites()
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(SEQUENCE_LENGTH), window=WINDOW_BP,
        block_size=BLOCK_BP, chunk_size="100kb", progress=False)
    inference._resolve_segmentation_params()
    assert inference._wbp % inference.block_size == 0
    assert inference._wbp == 3000


def test_one_segment_reproduces_the_unchunked_run_exactly():
    """A window that is not a block multiple is snapped on both paths.

    A chunk spanning the whole region differs from the unchunked build only
    through the window grid, so once the grid matches the two are identical.
    """
    sites, names = _sites()
    plain = _run(sites, names, chunk_size=None)
    chunked = _run(sites, names, chunk_size="1mb")
    assert set(plain) == set(chunked)
    worst = _max_abs_difference(plain, chunked)
    assert worst == 0.0, f"max |dposterior| = {worst:.3e}"


def test_a_window_straddling_a_chunk_boundary_is_built_whole():
    """A straddling window is condensed from every block it covers.

    Clipped to the core instead, such a window is built from the fraction of
    its blocks the core happens to hold, which moves the calls it scores. The
    window here spans twenty blocks and the chunk boundary cuts it, so the
    truncated form is far from the unchunked answer.
    """
    sites, names = _sites()

    def run(**kwargs):
        inference = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=float(SEQUENCE_LENGTH), window=20_000,
            block_size=BLOCK_BP, n_ensemble=None, progress=False, **kwargs)
        return {site.pos: post.to_dict() for site, post in inference.infer()}

    plain = run(chunk_size=None)
    chunked = run(chunk_size=37_000, halo=40_000)
    assert set(plain) == set(chunked)
    per_site = [
        max(abs(plain[p].get(a, 0.0) - chunked[p].get(a, 0.0))
            for a in set(plain[p]) | set(chunked[p]))
        for p in plain
    ]
    flips = sum(max(plain[p], key=plain[p].get)
                != max(chunked[p], key=chunked[p].get) for p in plain)
    assert flips == 0, f"{flips} MAP calls moved"
    assert float(np.median(per_site)) < 1e-4


def test_narrow_windows_agree_on_the_call_across_chunk_sizes():
    """Multi-segment runs agree on the call, not to the last bit.

    Each segment calibrates its own prior mean TMRCA from the blocks it sees,
    so a genuinely multi-segment run carries a small chunk dependence that the
    window grid does not account for.
    """
    sites, names = _sites()
    plain = _run(sites, names, chunk_size=None)
    for chunk_size in ("100kb", 50_000):
        chunked = _run(sites, names, chunk_size=chunk_size, halo=40_000)
        assert set(plain) == set(chunked)
        flips = sum(max(plain[p], key=plain[p].get)
                    != max(chunked[p], key=chunked[p].get) for p in plain)
        assert flips / len(plain) < 0.05, (
            f"chunk_size={chunk_size}: {flips}/{len(plain)} MAP calls moved")


def test_a_gap_wider_than_the_halo_still_materialises():
    """A core reaching past its segment's axis is clamped, not a ValueError."""
    sites, names = _sites(drop=(60_000, 160_000))
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=None, window=WINDOW_BP, block_size=BLOCK_BP,
        chunk_size=50_000, halo=10_000, n_ensemble=None, progress=False)
    groups = list(inference.to_tree_sequence())
    assert groups
    for (lo, hi), members in groups:
        assert hi >= lo
        assert all(m.num_trees > 0 for m in members)


def _two_contig_sites():
    """The same panel laid on two contigs, so positions collide across them."""
    sites, names = _sites(sequence_length=60_000)
    from dataclasses import replace
    return sites + [replace(s, chrom="2") for s in sites], names


def test_multi_contig_tmrcas_refuse_rather_than_collide():
    """Blocks from two contigs must not be glued onto one axis."""
    sites, names = _two_contig_sites()
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=None, window=WINDOW_BP, block_size=BLOCK_BP,
        chunk_size=100_000, n_ensemble=None, progress=False)
    with pytest.raises(NotImplementedError, match="single contig"):
        inference.pairwise_tmrcas()


def test_multi_contig_tree_sequences_refuse_rather_than_collide():
    """Genealogies from two contigs must not be reported on one axis."""
    sites, names = _two_contig_sites()
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=None, window=WINDOW_BP, block_size=BLOCK_BP,
        chunk_size=100_000, n_ensemble=None, progress=False)
    with pytest.raises(NotImplementedError, match="single contig"):
        list(inference.to_tree_sequence())


def test_the_time_grid_reaches_below_one_difference_per_block():
    """The grid floor is not stopped by the empty-block filter.

    A panel spanning several depths has most of its non-empty blocks in the
    deep cross pairs, so a floor taken from the pooled non-empty cells sits
    above the recent coalescences of the ingroup and pins them to the first
    bin. Worse, discarding the empty blocks floors any estimate at one
    difference per block, ``1 / (2 mu B)``, however shallow the data are. The
    floor is therefore read from per-pair means over every block, which the
    empty ones inform.
    """
    from ancestree.local_tree_inference import PairwiseCoalescentHMM

    mu, block_size = 1.25e-8, 2000
    hmm = PairwiseCoalescentHMM(4, mu=mu, rec_rate=1e-8,
                                block_size=block_size, n_time_bins=32)
    per_block_rate = 2.0 * mu * block_size
    n_blocks = 200
    # One shallow pair (mostly empty blocks) against three deep ones.
    counts = np.zeros((4, n_blocks), dtype=float)
    counts[0, ::80] = 1.0                       # shallow: ~1 difference in 80
    counts[1:, :] = 6.0                         # deep: every block informative

    edges, _ = hmm._calibrate_time_grid(counts)
    one_difference = 1.0 / per_block_rate
    assert edges[0] < one_difference, (
        f"grid floor {edges[0]:.0f} sits at or above the one-difference-per-"
        f"block depth {one_difference:.0f}; a statistic taken over the "
        f"non-empty blocks alone cannot reach below it")
    typical = float(np.median(counts.mean(axis=1) / per_block_rate))
    assert edges[0] < typical, (
        f"grid floor {edges[0]:.0f} sits above the typical pair depth "
        f"{typical:.0f}")


def test_a_group_keeps_its_frame_when_consumed_out_of_order():
    """Materialising the groups first must not restamp their coordinates.

    Each group's genealogies are placed on the genome axis lazily. Reading the
    placement from the loop's own variables would give a group whatever
    segment happened to be current when it was finally consumed, so a caller
    that collects the groups before consuming them would silently receive the
    wrong stretch.
    """
    sites, names = _sites()
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(SEQUENCE_LENGTH), window=WINDOW_BP,
        block_size=BLOCK_BP, chunk_size=50_000, n_ensemble=None,
        progress=False)
    groups = list(inference.to_tree_sequence())
    assert len(groups) > 1
    for (lo, hi), members in groups:
        for member in members:
            covered = [t.interval for t in member.trees() if t.num_edges]
            assert covered, f"stretch [{lo}, {hi}) came back empty"
            assert covered[0].left == pytest.approx(lo)
            assert covered[-1].right == pytest.approx(hi)


def test_one_duplicated_sample_does_not_set_the_grid_floor():
    """A single near-clone pair must not move the whole panel's time grid.

    The floor is read from the median over pairs of each pair's mean, not from
    the shallowest pair, so a duplicated or closely related sample cannot drag
    the grid down for everyone. Taken at the minimum instead, one such pair
    moved the floor by two thirds and onto its hard clamp.
    """
    from ancestree.local_tree_inference import PairwiseCoalescentHMM

    mu, block_size, n_blocks = 1.25e-8, 2000, 200
    rng = np.random.default_rng(3)

    def floor_of(counts):
        hmm = PairwiseCoalescentHMM(6, mu=mu, rec_rate=1e-8,
                                    block_size=block_size, n_time_bins=32)
        return hmm._calibrate_time_grid(counts)[0][0]

    # A panel of ordinary pairs, then the same panel plus one near-clone pair
    # carrying a single difference over the whole region.
    ordinary = rng.poisson(3.0, size=(15, n_blocks)).astype(float)
    clone = np.zeros((1, n_blocks))
    clone[0, 0] = 1.0
    with_clone = np.vstack([ordinary, clone])

    before, after = floor_of(ordinary), floor_of(with_clone)
    shift = abs(after - before) / before
    assert shift < 0.05, (
        f"one near-clone pair moved the grid floor by {100 * shift:.1f}% "
        f"({before:.1f} -> {after:.1f} generations)")
