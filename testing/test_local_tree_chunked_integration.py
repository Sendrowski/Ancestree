"""Slow integration tests for the chunked ``LocalTreeInference`` path.

Simulate several msprime datasets and drive the *whole* chunked pipeline
(stream → per-segment HMM + UPGMA + Felsenstein) end to end, checking that it

* recovers the simulator's ancestral allele well above chance,
* gives results that do not depend on the chunking / parallelism bookkeeping
  (chunk size, worker count are memory / speed knobs, not modelling choices),
* handles the two discontinuity regimes the finite-Δ kernel was built for:
  **large gaps** (independent contigs) and a **callability mask** over otherwise
  gap-free data.

These re-simulate per test and run the full pipeline, so they are marked
``slow`` (deselected by default). Run with ``pytest -m slow`` or
``pytest -m slow testing/test_local_tree_chunked_integration.py``.
"""
from __future__ import annotations

import logging
import multiprocessing

import numpy as np
import msprime
import pytest

import ancestree.local_tree_inference as lti
from ancestree import (JC69, LocalTreeBuilder, LocalTreeInference, STATES,
                       Site)
from ancestree.settings import Settings
from testing._helpers import canonical_alleles, toy_chunked_inference, toy_sites


# --------------------------------------------------------------- simulation
def _sim_io(length, n_in, n_out, seed, rec=1e-8, mu=1.25e-8, split=2e5):
    """One contig: ingroup + a diverged outgroup population (haploid)."""
    dem = msprime.Demography()
    dem.add_population(name="ingroup", initial_size=1e4)
    dem.add_population(name="outgroup", initial_size=1e4)
    dem.add_population(name="anc", initial_size=1e4)
    dem.add_population_split(time=split, ancestral="anc",
                            derived=["ingroup", "outgroup"])
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(n_in, population="ingroup", ploidy=1),
                 msprime.SampleSet(n_out, population="outgroup", ploidy=1)],
        demography=dem, sequence_length=length, recombination_rate=rec,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=mu, random_seed=seed)


def _contig_sites(ts, chrom):
    """``(sites, truth, panel)`` for one contig.

    Sites are ingroup-polymorphic (the benchmark convention), tagged with
    ``chrom`` and carrying every sample's allele. ``truth`` is the simulator
    ancestral state keyed by ``(chrom, pos)``. ``panel`` is ingroup + outgroup
    haplotype names (outgroups inform the rooting).
    """
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    pop_of = {f"tsk_{ind.id}": ts.node(int(ind.nodes[0])).population
              for ind in ts.individuals()}
    ing_pop = [p.id for p in ts.populations()
               if p.metadata.get("name") == "ingroup"][0]
    ingroup = [n for n, p in pop_of.items() if p == ing_pop]
    panel = ingroup + [n for n, p in pop_of.items() if p != ing_pop]
    poly = set(ingroup)
    sites, truth = [], {}
    for v in ts.variants():
        anc = v.site.ancestral_state
        if anc not in STATES:
            continue
        ta = {nm[int(node)]: v.alleles[v.genotypes[j]]
              for j, node in enumerate(ts.samples())}
        present = {a for s, a in ta.items() if s in poly and a in STATES}
        if len(present) < 2:
            continue
        sites.append(Site(chrom=chrom, pos=int(v.site.position),
                          alleles=canonical_alleles(v.alleles), tip_alleles=ta))
        truth[(chrom, int(v.site.position))] = anc
    return sites, truth, panel


def _multicontig(n_contigs, length, n_in, n_out):
    """Concatenate ``n_contigs`` independent contigs (true large gaps between
    them), labelled ``"1"``..., contiguous and position-sorted within each."""
    sites, truth, panel = [], {}, None
    for i in range(1, n_contigs + 1):
        ts = _sim_io(length=length, n_in=n_in, n_out=n_out, seed=i)
        s, t, panel = _contig_sites(ts, str(i))
        sites += s
        truth.update(t)
    return sites, truth, panel


def _accuracy(results, truth):
    """Fraction of polarised sites whose MAP allele matches the truth."""
    hit = tot = 0
    for s, post in results:
        t = truth.get((s.chrom, s.pos))
        if t is None:
            continue
        tot += 1
        hit += int(post.map_allele == t)
    return hit / max(1, tot)


_COMMON = dict(mu=1.25e-8, rec_rate=1e-8, window="20kb", block_size=1000,
               progress=False)


# ------------------------------------------------------ large gaps (contigs)
@pytest.mark.slow
def test_chunked_multicontig_recovery_stable_across_chunk_size():
    """Three independent contigs (large gaps): recovery is high and essentially
    chunk-size-independent, a small chunk that slices each contig matches a
    chunk large enough to hold a whole contig."""
    sites, truth, names = _multicontig(n_contigs=3, length=150_000,
                                       n_in=10, n_out=2)

    def run(cs):
        return list(LocalTreeInference(
            sites, JC69(), sample_names=names, chunk_size=cs,
            **_COMMON).infer())

    small = run(40_000)  # slices each 150 kb contig into ~4 cores
    big = run("5mb")  # one chunk per contig (no interior slicing)
    assert len(small) == len(sites) and len(big) == len(sites)
    a_small, a_big = _accuracy(small, truth), _accuracy(big, truth)
    assert a_small > 0.75, a_small
    assert a_big > 0.75, a_big
    assert abs(a_small - a_big) < 0.05  # recovery ~ chunk-size independent


@pytest.mark.slow
def test_chunked_workers_match_serial_end_to_end():
    """The fork-pool path is pure bookkeeping: n_workers>1 reproduces the serial
    run site-for-site across a realistic two-contig simulation."""
    sites, truth, names = _multicontig(n_contigs=2, length=120_000,
                                       n_in=8, n_out=2)
    kw = dict(sample_names=names, chunk_size=40_000, halo="auto", **_COMMON)
    serial = list(LocalTreeInference(sites, JC69(), n_workers=1, **kw).infer())
    par = list(LocalTreeInference(sites, JC69(), n_workers=2, **kw).infer())
    assert [(s.chrom, s.pos) for s, _ in serial] \
        == [(s.chrom, s.pos) for s, _ in par]
    for (_, a), (_, b) in zip(serial, par):
        np.testing.assert_allclose(a.values, b.values)
    assert _accuracy(serial, truth) > 0.75


# ------------------------------------------------- mask over gap-free data
@pytest.mark.slow
def test_chunked_accessibility_mask_applied_and_preserves_recovery():
    """A callability mask over otherwise gap-free data: the masked region's
    calls change (its blocks emit uniformly) while recovery in the accessible
    flanks is preserved, and every input site is still emitted."""
    ts = _sim_io(length=300_000, n_in=10, n_out=2, seed=11)
    sites, truth, names = _contig_sites(ts, "1")
    L = ts.sequence_length
    lo, hi = L / 3.0, 2.0 * L / 3.0  # declare the middle third masked
    access = [(0.0, lo), (hi, L)]
    kw = dict(sample_names=names, chunk_size=80_000, **_COMMON)

    masked = list(LocalTreeInference(
        sites, JC69(), accessibility=access, **kw).infer())
    unmasked = {(s.chrom, s.pos): p.values
                for s, p in LocalTreeInference(sites, JC69(), **kw).infer()}

    assert len(masked) == len(sites)
    # the mask actually does something inside the masked region
    inside = [float(np.abs(p.values - unmasked[(s.chrom, s.pos)]).max())
              for s, p in masked if lo <= s.pos < hi]
    assert inside and max(inside) > 1e-6
    # recovery in the accessible flanks stays high
    flank = [(s, p) for s, p in masked if s.pos < lo or s.pos >= hi]
    assert _accuracy(flank, truth) > 0.75


def test_chunked_sub_block_exposure_matches_the_whole_region():
    """Each segment reads the callable base-pair fraction of its own stretch.

    The mask is intersected with the segment and shifted to a local zero, so a
    fraction taken before that shift, or against the whole region, rescales
    the Poisson emission of every block in the chunked path while the run
    still looks successful.
    """
    names = [f"h{i}" for i in range(4)]
    rng = np.random.default_rng(0)
    length, block_size = 40_000, 1000
    sites = [
        Site(chrom="1", pos=pos, alleles=("A", "T"),
             tip_alleles={n: ("A" if rng.random() < 0.7 else "T")
                          for n in names})
        for pos in range(0, length, 50)
    ]
    # Callable width cycles 1000, 800, 600, 400, 200 bp per 1000 bp block, so
    # the fractions differ from block to block and a misplaced origin shows.
    widths = [1000.0, 800.0, 600.0, 400.0, 200.0]
    access = [(float(b * block_size),
               float(b * block_size) + widths[b % len(widths)])
              for b in range(length // block_size)]
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(length), window=2000, block_size=block_size,
        accessibility=access, chunk_size=10_000, halo=1000, progress=False)
    whole = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=float(length), block_size=block_size,
        accessibility=access).geometry.accessible_fraction()

    n_segments = n_compared = 0
    for builder in inference.segments():
        geometry = builder.geometry
        origin, span = geometry.origin, geometry.length
        n_segments += 1
        n_seg_blocks = geometry.n_blocks
        frac = geometry.accessible_fraction()
        covered = sum(min(a1, origin + span) - max(a0, origin)
                      for a0, a1 in access
                      if min(a1, origin + span) > max(a0, origin))
        assert frac.sum() * block_size == pytest.approx(covered), origin
        if origin % block_size == 0:
            b0 = origin // block_size
            np.testing.assert_allclose(
                frac[:min(n_seg_blocks, len(whole) - b0)],
                whole[b0:b0 + n_seg_blocks])
            n_compared += 1
    assert n_segments > 1, "expected a genuinely multi-segment run"
    assert n_compared > 1, "no segment started on a block boundary to compare"


# ------------------------------------------------------------- small config
@pytest.mark.slow
def test_chunked_small_config_runs_and_recovers():
    """A small panel + short region with a single outgroup still streams, emits
    every site, and recovers above chance under aggressive chunking."""
    ts = _sim_io(length=80_000, n_in=4, n_out=1, seed=21)
    sites, truth, names = _contig_sites(ts, "1")
    out = list(LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window="50snp", block_size=500, chunk_size=20_000, halo="auto",
        progress=False).infer())
    assert len(out) == len(sites)
    assert _accuracy(out, truth) > 0.5  # well above 0.25 random


# ------------------------------------------------------ segment workers
def test_segment_worker_requires_the_module_global(monkeypatch):
    """The fork entrypoint refuses to run without the inherited instance."""
    monkeypatch.setattr(lti, "_LOCAL_TREE_FOR_FORK", None)
    with pytest.raises(RuntimeError, match="_LOCAL_TREE_FOR_FORK"):
        lti._segment_worker(None)


def test_segment_worker_processes_the_inherited_instance(monkeypatch):
    """The fork entrypoint delegates to the parent's ``_process_segment``."""
    sites, names = toy_sites(range(0, 1500, 20))
    inf = toy_chunked_inference(sites, names, n_ensemble=None)
    work_units = list(inf.segmentation.units())
    assert work_units
    monkeypatch.setattr(lti, "_LOCAL_TREE_FOR_FORK", inf)
    rows, counts = lti._segment_worker(work_units[0])
    direct, direct_counts = inf._process_segment(work_units[0])
    assert counts == direct_counts
    assert [s.pos for s, _ in rows] == [s.pos for s, _ in direct]
    for (_, a), (_, b) in zip(rows, direct):
        np.testing.assert_allclose(a, b)


def test_chunked_workers_fall_back_to_serial_on_an_unsafe_fork(monkeypatch, caplog):
    """An unsafe numba layer runs the segments in-process with a warning."""
    sites, names = toy_sites(range(0, 3000, 20))
    kw = dict(chunk_size=1000, halo=400, n_ensemble=None)
    serial = list(toy_chunked_inference(sites, names, n_workers=1, **kw).infer())
    monkeypatch.setattr(Settings, "_fork_is_safe", staticmethod(lambda: False))
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        par = list(toy_chunked_inference(sites, names, n_workers=2, **kw).infer())
    assert any("not fork-safe" in r.getMessage() for r in caplog.records)
    assert [s.pos for s, _ in par] == [s.pos for s, _ in serial]
    for (_, a), (_, b) in zip(par, serial):
        np.testing.assert_allclose(a.values, b.values)


def test_chunked_workers_fall_back_to_serial_without_a_fork_method(monkeypatch, caplog):
    """A platform without the ``fork`` start method runs single-threaded."""
    sites, names = toy_sites(range(0, 3000, 20))
    kw = dict(chunk_size=1000, halo=400, n_ensemble=None)
    serial = list(toy_chunked_inference(sites, names, n_workers=1, **kw).infer())
    monkeypatch.setattr(multiprocessing, "get_all_start_methods",
                        lambda: ["spawn"])
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        par = list(toy_chunked_inference(sites, names, n_workers=2, **kw).infer())
    assert any("'fork' start method is unavailable" in r.getMessage()
               for r in caplog.records)
    assert [s.pos for s, _ in par] == [s.pos for s, _ in serial]
    for (_, a), (_, b) in zip(par, serial):
        np.testing.assert_allclose(a.values, b.values)


# ---------------------------------------------------- core-only diagnostics
def _halo_diagnostics(sites, panel, halo, caplog):
    """The per-tree diagnostics of a chunked plug-in run with ``halo``.

    :return: ``(uncoalesced-tree warnings, non-monophyletic trees, trees
        without an ingroup MRCA)``.
    """
    caplog.clear()
    inf = LocalTreeInference(
        sites, JC69(), sample_names=panel, ingroup_samples=panel[:6],
        outgroup_samples=panel[6:], sequence_length=200_000,
        chunk_size="60kb", halo=halo, n_ensemble=None, mu=1.25e-8,
        rec_rate=1e-8, progress=False)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        list(inf.infer())
    params = inf.provenance()["parameters"]
    return (sum("carried no coalescence" in r.getMessage()
                for r in caplog.records),
            params["n_trees_ingroup_non_monophyletic"],
            params["n_segments_without_ingroup_mrca"])


def test_chunked_tree_diagnostics_count_core_trees_only(caplog):
    """Each segment tallies the trees of its core alone, so the counts do not
    move with the halo. The edge-less trees of the halo were counted as
    uncoalesced, more of them the wider the halo."""
    ts = _sim_io(200_000, 6, 2, 3)
    sites, _, panel = _contig_sites(ts, "1")
    narrow = _halo_diagnostics(sites, panel, "5kb", caplog)
    assert narrow == _halo_diagnostics(sites, panel, "20kb", caplog)
    assert narrow[0] == 0 and narrow[2] == 0


def test_chunked_ensemble_uniform_fallback_counts_core_sites_only(
        monkeypatch, caplog):
    """A site given the uniform fallback is counted once, in the segment
    whose core holds it. The halo sites of every segment used to count too."""
    from ancestree._ensemble import SegmentEnsemble

    scored = SegmentEnsemble.posterior

    def all_fallback(self, *args, **kwargs):
        post = scored(self, *args, **kwargs)
        self.uniform_fallback = np.ones(len(post), dtype=bool)
        return post

    monkeypatch.setattr(SegmentEnsemble, "posterior", all_fallback)
    sites, names = toy_sites(range(0, 3000, 20))
    inf = toy_chunked_inference(sites, names, chunk_size=1000, halo=400,
                                n_ensemble=2)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        n = len(list(inf.infer()))
    assert any(r.getMessage().startswith(f"{n} site(s) had a degenerate")
               for r in caplog.records)
