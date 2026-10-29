"""Peak memory of ensemble scoring must not track the segment's window count.

The scoring scratch is ``member_chunk x windows x ...``. Left unbounded it grows
with the number of windows in a segment, and at ``window="1snp"`` that is one
window per SNP, gigabytes of scratch for a dataset of a few megabytes.
:func:`score_ensemble` rebuilds the transition matrices per window in-thread,
leaving the condensed pairwise TMRCAs and ``branch_t`` the largest terms, and
:data:`ancestree._ensemble.MAX_SCRATCH_BYTES` caps them by scoring windows in
blocks, so peak memory is flat in window count.

Peak RSS is measured in a child process (``ru_maxrss`` of RUSAGE_CHILDREN),
because numpy allocates outside the Python allocator and ``tracemalloc`` would
not see the arrays that matter.
"""
import multiprocessing as mp
import sys

import numpy as np
import pytest

from ancestree._ensemble import MAX_SCRATCH_BYTES, _window_block
from testing._helpers import ts_sites

#: Generous beside the ~256 MB scratch bound, but far below the multi-GB the
#: unbounded scratch would reach.
PEAK_RSS_LIMIT_MB = 1200


def _sites(seed=3, n=8, L=40_000):
    """A panel dense enough to tile many windows."""
    import msprime
    ts = msprime.sim_ancestry(samples=n, ploidy=1, sequence_length=L,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=2.5e-6, random_seed=seed)
    names = [f"s{i}" for i in range(n)]
    return ts_sites(ts, names), names, float(ts.sequence_length)


def _run_posteriors(sites, names, L):
    """Posteriors keyed by position, for a fixed ensemble and seed."""
    from ancestree import JC69, LocalTreeInference
    inf = LocalTreeInference(sites, JC69(), mu=2.5e-8, rec_rate=1e-8,
                             sample_names=names, sequence_length=L,
                             window="1snp", n_ensemble=8, ensemble_seed=17,
                             progress=False)
    return {int(s.pos): np.asarray(post.values) for s, post in inf.infer()}


def _one_run(window, n_ensemble, n_sites, q):
    """Infer in a child process so its peak RSS can be read after it exits.

    ``n_sites`` is the target number of SEGREGATING sites, which is what sets
    the window count. The mutation rate is raised to hit it in a tractable
    span: at the reference rate this asked for 160 kb and yielded 178 sites,
    far too few for the scratch to be visible above the ~150 MB import floor.
    """
    import resource

    import msprime
    from ancestree import JC69, LocalTreeInference

    L = n_sites * 40
    ts = msprime.sim_ancestry(samples=8, ploidy=1, sequence_length=L,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=3)
    ts = msprime.sim_mutations(ts, rate=2.5e-6, random_seed=3)
    names = [f"s{i}" for i in range(8)]
    sites = ts_sites(ts, names)
    inf = LocalTreeInference(sites, JC69(), mu=2.5e-8, rec_rate=1e-8,
                             sample_names=names, sequence_length=float(L),
                             window=window, n_ensemble=n_ensemble,
                             # Pinned, so the two runs differ ONLY in window
                             # count. Left to default it is derived from the
                             # window, and the HMM's retained forward matrix
                             # -- (n_pairs, n_blocks, n_time_bins), hundreds of
                             # MB, bounded by chunk_size rather than by the
                             # scoring budget -- would move with it and swamp
                             # the thing under test.
                             block_size=500, progress=False,
                             chunk_size=None)
    scored = sum(1 for _ in inf.infer())
    n_windows = len(inf.segmentation.contig(sites[0].chrom).windows)
    # The child reports its OWN peak: RUSAGE_CHILDREN in the parent is a
    # high-water mark across every child ever spawned, so a delta there reads
    # zero for any run that stays below an earlier one -- and the assertion
    # would pass without measuring anything.
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    q.put((scored, peak_kib, n_windows))


def _peak_rss_mb(window, n_ensemble, n_sites):
    """Peak RSS of a child that runs the inference, in MB."""
    ctx = mp.get_context("spawn")  # a fresh interpreter, so the parent's
    q = ctx.Queue()  # already-imported arrays are not inherited
    proc = ctx.Process(target=_one_run, args=(window, n_ensemble, n_sites, q))
    proc.start()
    scored, peak_kib, n_windows = q.get(timeout=900)
    proc.join()
    assert proc.exitcode == 0, f"child exited {proc.exitcode}"
    assert scored > 0
    return peak_kib / 1024.0, n_windows  # ru_maxrss is KiB on Linux


def test_block_size_shrinks_as_the_ensemble_grows():
    """The bound is on bytes, so a bigger member chunk means fewer windows."""
    small = _window_block(100_000, 1, 20, 190)
    large = _window_block(100_000, 32, 20, 190)
    assert large < small
    for members in (1, 8, 32, 512):
        blk = _window_block(100_000, members, 20, 190)
        nslot = 2 * 20 + 1
        # branch_t: the dominant packing array, since the transition
        # matrices are never stored.
        assert blk * members * nslot * 8 <= MAX_SCRATCH_BYTES or blk == 256


def test_block_size_counts_the_condensed_tmrcas():
    """The ``(member, window, pair)`` float64 TMRCAs are scratch too. Sizing
    the block by the packing arrays alone let them reach 2 GB at 200
    haplotypes and one member."""
    n_hap, n_pairs = 200, 19900
    for members in (1, 2, 8):
        blk = _window_block(100_000, members, n_hap, n_pairs)
        assert blk * members * n_pairs * 8 <= MAX_SCRATCH_BYTES or blk == 256


def test_block_size_never_starves_the_parallel_loop():
    """Blocking is a memory measure. It must not shrink below the floor."""
    assert _window_block(100_000, 4096, 200, 19900) >= 256


def test_block_size_is_capped_by_the_window_count():
    assert _window_block(10, 8, 20, 190) == 10


@pytest.mark.skipif(sys.platform != "linux", reason="ru_maxrss units are Linux")
def test_peak_memory_stays_within_budget():
    """A gross ceiling on the peak memory of a realistic run.

    Deliberately a loose absolute: a large share of any run is the interpreter
    plus numpy/numba, and the window count cannot be varied independently of the
    block grid (a window never spans less than a block), so a slope test
    cannot isolate the scoring scratch through RSS. The sharp statement about
    the scratch is :func:`test_posterior_is_independent_of_the_window_block`,
    which exercises the blocking loop directly.
    """
    peak, n_windows = _peak_rss_mb("1snp", 64, 8000)
    assert n_windows > 500, f"too few windows ({n_windows}) to be meaningful"
    assert peak < PEAK_RSS_LIMIT_MB, (
        f"peak RSS {peak:.0f} MB over {n_windows} windows exceeds "
        f"{PEAK_RSS_LIMIT_MB} MB")


def test_posterior_is_independent_of_the_window_block():
    """Blocking is a memory measure: it must not touch the numbers.

    The suite never reaches a second block on its own -- the budget allows
    ~16k windows per block and no test segment is a tenth of that -- so the
    loop, and the short final block in particular, are only exercised here.
    Both matter: a short final block cannot slice the wide buffers, because
    the non-contiguous slice makes the kernels' reshape hand back a COPY they
    then write into, silently corrupting every posterior in that block.
    """
    from ancestree import _ensemble

    sites, names, L = _sites()
    ref = _run_posteriors(sites, names, L)
    for block in (1, 3, 7, 64):  # 3 and 7 leave a short final block
        monkey = lambda W, M, n, P, b=block: min(b, W)  # noqa: E731
        orig = _ensemble._window_block
        _ensemble._window_block = monkey
        try:
            got = _run_posteriors(sites, names, L)
        finally:
            _ensemble._window_block = orig
        assert got.keys() == ref.keys()
        for pos in ref:
            assert np.array_equal(ref[pos], got[pos]), (
                f"window block {block} changed the posterior at {pos}")


def _hmm_counts(n_hap=30, n_blocks=4000, seed=0):
    """Poisson block counts in int16, as the ensemble holds them."""
    from ancestree import PairwiseCoalescentHMM
    hmm = PairwiseCoalescentHMM(n_hap, mu=1e-8, rec_rate=1e-8, block_size=1000)
    rng = np.random.default_rng(seed)
    n_pairs = n_hap * (n_hap - 1) // 2
    counts = rng.poisson(3.0, size=(n_pairs, n_blocks)).astype(np.int16)
    return hmm, counts, rng


@pytest.mark.parametrize("missing", [False, True])
def test_calibrate_holds_one_float_copy_of_the_counts(monkeypatch, missing):
    """Calibration scratch is one float64 copy of the counts plus row chunks.

    Each division, column selection and mask product used to allocate a full
    ``(n_pairs, n_blocks)`` float64 temporary, taking the transient peak to
    about 20 times the int16 counts (1.4 GB at 60 haplotypes and 20,000
    blocks).
    """
    import tracemalloc

    from ancestree import _ensemble
    monkeypatch.setattr(_ensemble, "ROW_CHUNK_BYTES", 2**20, raising=False)
    hmm, counts, rng = _hmm_counts()
    n_blocks = counts.shape[1]
    kw = {}
    if missing:
        kw = dict(called=rng.integers(1, 5, size=counts.shape).astype(np.int16),
                  block_sites=np.full(n_blocks, 4, dtype=np.int64))
    emit_scale = np.full(n_blocks, 0.9)
    block_mask = np.ones(n_blocks, dtype=bool)
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        hmm._calibrate(counts, emit_scale=emit_scale, block_mask=block_mask, **kw)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    # The float64 copy alone is 4 times the int16 counts.
    assert peak < 6 * counts.nbytes, (
        f"calibrate peaked at {peak / counts.nbytes:.1f}x the int16 counts")


def test_a_fully_called_panel_allocates_no_exposure_array():
    """Without a missing call every exposure is 1 and no array holds it."""
    from ancestree import PairwiseCoalescentHMM
    rng = np.random.default_rng(1)
    n_hap, n_blocks = 6, 50
    genotypes = rng.integers(0, 2, size=(4 * n_blocks, n_hap)).astype(np.int8)
    block_of_site = np.repeat(np.arange(n_blocks), 4)
    hmm = PairwiseCoalescentHMM(n_hap, mu=1e-8, rec_rate=1e-8, block_size=100)
    _counts, called, block_sites = hmm._pair_block_counts(
        genotypes, block_of_site, n_blocks, dtype=np.int16)
    assert called is None
    assert block_sites.shape == (n_blocks,)
