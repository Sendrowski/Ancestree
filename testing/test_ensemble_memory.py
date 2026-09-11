"""Peak memory of ensemble scoring must not track the segment's window count.

The scoring scratch is ``member_chunk x windows x ...``. Left unbounded it grows
with the number of windows in a segment, and at ``window="1snp"`` that is one
window per SNP, gigabytes of scratch for a dataset of a few megabytes.
:func:`score_ensemble` rebuilds the transition matrices per window in-thread,
leaving ``branch_t`` the largest term, and
:data:`ancestree._ensemble.MAX_SCRATCH_BYTES` caps it by scoring windows in
blocks, so peak memory is flat in window count.

Peak RSS is measured in a child process (``ru_maxrss`` of RUSAGE_CHILDREN),
because numpy allocates outside the Python allocator and ``tracemalloc`` would
not see the arrays that matter.
"""
import multiprocessing as mp
import resource
import sys

import numpy as np
import pytest

from ancestree._ensemble import MAX_SCRATCH_BYTES, _window_block
from testing._helpers import canonical_alleles

#: Generous beside the ~256 MB scratch bound, but far below the multi-GB the
#: unbounded scratch would reach.
PEAK_RSS_LIMIT_MB = 1200

#: How much peak RSS may grow between a few-window and a many-window run over
#: the SAME sites. Blocking caps the scratch, so the honest expectation is
#: ~0. This leaves room for allocator noise without admitting growth,
#: which reached gigabytes.
WINDOW_SLOPE_LIMIT_MB = 350


def _sites(seed=3, n=8, L=40_000):
    """A panel dense enough to tile many windows."""
    import msprime
    from ancestree.sites import Site
    ts = msprime.sim_ancestry(samples=n, ploidy=1, sequence_length=L,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=2.5e-6, random_seed=seed)
    names = [f"s{i}" for i in range(n)]
    out = [Site(chrom="1", pos=int(v.site.position), alleles=canonical_alleles(v.alleles),
                tip_alleles={names[i]: v.alleles[g]
                             for i, g in enumerate(v.genotypes)})
           for v in ts.variants()
           if len(v.alleles) == 2 and all(a in "ACGT" for a in v.alleles if a)]
    return out, names, float(ts.sequence_length)


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
    import msprime
    from ancestree import JC69, LocalTreeInference
    from ancestree.sites import Site

    L = n_sites * 40
    ts = msprime.sim_ancestry(samples=8, ploidy=1, sequence_length=L,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=3)
    ts = msprime.sim_mutations(ts, rate=2.5e-6, random_seed=3)
    names = [f"s{i}" for i in range(8)]
    sites = [Site(chrom="1", pos=int(v.site.position), alleles=canonical_alleles(v.alleles),
                  tip_alleles={names[i]: v.alleles[g]
                               for i, g in enumerate(v.genotypes)})
             for v in ts.variants()
             if len(v.alleles) == 2 and all(a in "ACGT" for a in v.alleles if a)]
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
                             # Reads the builder's window grid directly, which
                             # the segmented path resolves per segment.
                             chunk_size=None)
    scored = sum(1 for _ in inf.infer())
    n_windows = len(getattr(inf.builder, "_window_intervals")())
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
    small = _window_block(100_000, 1, 20)
    large = _window_block(100_000, 32, 20)
    assert large < small
    for members in (1, 8, 32, 512):
        blk = _window_block(100_000, members, 20)
        nslot = 2 * 20 + 1
        # branch_t: the dominant per-window array, since the transition
        # matrices are never stored.
        assert blk * members * nslot * 8 <= MAX_SCRATCH_BYTES or blk == 256


def test_block_size_never_starves_the_parallel_loop():
    """Blocking is a memory measure. It must not shrink below the floor."""
    assert _window_block(100_000, 4096, 200) >= 256


def test_block_size_is_capped_by_the_window_count():
    assert _window_block(10, 8, 20) == 10


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
        monkey = lambda W, M, n, b=block: min(b, W)  # noqa: E731
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
