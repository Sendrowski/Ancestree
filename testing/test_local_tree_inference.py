"""Tests for :class:`~ancestree.local_tree_inference.LocalTreeInference`.

Covers the three stages of the VCF-only local-tree mode: the pairwise
PSMC′ HMM (does its posterior-mean TMRCA track truth?), the per-window
UPGMA tree-sequence assembly (is it a valid, tiling, sites-free
``tskit.TreeSequence``?), and end-to-end polarisation (sensible accuracy
from genotypes alone, with outgroups helping). Also asserts that no
ancestral-state truth is read.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import msprime

import pytest
import tskit
import ancestree as anc

from ancestree import (
    ARGBasedInference,
    JC69,
    LocalTreeBuilder,
    LocalTreeInference,
    PairwiseCoalescentHMM,
    PairwiseTmrcas,
    STATES,
    Site,
)
import ancestree._smc_kernel as smc
from ancestree.sites import SiteTable
from testing._helpers import (
    MU,
    REC,
    canonical_alleles,
    toy_builder,
    toy_chunked_inference,
    toy_inference,
    toy_sites,
)


def _sim(samples=10, length=200_000, mu=1.25e-8, rec=1e-8, N=1e4, seed=7):
    ts = msprime.sim_ancestry(
        samples=samples, ploidy=1, sequence_length=length,
        recombination_rate=rec, population_size=N, random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=mu, random_seed=seed)


def _sites_and_truth(ts, sample_names):
    """Build Sites (polymorphic across ``sample_names``) + truth map.

    The polymorphism filter is restricted to ``sample_names`` (the
    ingroup), matching the benchmark convention, outgroup-private sites
    where the ingroup is monomorphic are excluded.
    """
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    panel = set(sample_names)
    sites, truth = [], {}
    for v in ts.variants():
        anc = v.site.ancestral_state
        if anc not in STATES:
            continue
        ta = {nm[int(node)]: v.alleles[v.genotypes[j]]
              for j, node in enumerate(ts.samples())}
        alleles_present = {a for s, a in ta.items()
                           if s in panel and a in STATES}
        if len(alleles_present) < 2:
            continue
        sites.append(Site(chrom="1", pos=int(v.site.position),
                          alleles=canonical_alleles(v.alleles),
                          tip_alleles=ta))
        truth[int(v.site.position)] = anc
    return sites, truth


# --------------------------------------------------------------- HMM core
def test_seg_block_bounds_cover_final_block():
    """The segment bounds partition [0, n_blocks) exactly.

    ``n_blocks`` uses ceil(), so the final, ragged block must be covered for
    a sequence_length whose remainder mod block_size is <= block_size/2.
    """
    # L=50000, block_size=4000 -> 12.5 -> ceil n_blocks=13. Old round(12.5)=12
    # (banker's) left block 12 uncovered.
    ts = _sim(samples=4, length=50_000, seed=2)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=50_000, block_size=4000,
    )
    _g, _pos, _bos, nb = builder._genotype_matrix()
    assert nb == 13
    covered = set()
    for b0, b1 in builder._seg_block_bounds(nb):
        covered.update(range(b0, b1))
    assert covered == set(range(nb)), f"blocks {set(range(nb)) - covered} uncovered"
    # End to end: every per-block TMRCA is written (no leftover zero column).
    pair_block_t = builder._pairwise_block_tmrcas(_g, _bos, nb)
    assert (pair_block_t[:, -1] > 0).all(), "final block left unwritten (zero)"


def test_hmm_posterior_mean_tracks_truth():
    """Pairwise HMM posterior-mean TMRCA correlates with true TMRCA."""
    ts = _sim(samples=2, length=500_000, rec=2e-8, seed=3)
    # true pairwise TMRCA per block
    block = 2000
    n_blocks = int(np.ceil(ts.sequence_length / block))
    true_t = np.zeros(n_blocks)
    for tree in ts.trees():
        t = tree.tmrca(0, 1)
        lo = int(tree.interval.left // block)
        hi = int(np.ceil(tree.interval.right / block))
        true_t[lo:hi] = t

    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    # For n=2, polymorphic == the pair differs, which is exactly the het
    # signal. Build the genotype matrix via the builder helper.
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=2e-8, sample_names=names,
        sequence_length=ts.sequence_length, block_size=block,
    )
    g, _pos, block_of_site, nb = builder._genotype_matrix()
    hmm = PairwiseCoalescentHMM(2, mu=1.25e-8, rec_rate=2e-8, block_size=block)
    pmean = hmm.block_tmrcas(g, block_of_site, nb)[0]  # only pair

    # Rank-correlate on blocks where the truth is defined. Spearman (not
    # Pearson) is the right metric: TMRCA spans orders of magnitude, so
    # raw-value Pearson is dominated by a few deep blocks. Single-pair
    # Spearman is ~0.75 across seeds (min ~0.68); 0.6 is a meaningful
    # bar with margin.
    from scipy.stats import spearmanr
    m = true_t[:nb] > 0
    rho = spearmanr(pmean[m], true_t[:nb][m]).correlation
    assert rho > 0.6, f"HMM TMRCA rank-correlation with truth too low: {rho:.3f}"


def test_inferred_tmrca_unbiased_across_deciles():
    """Inferred per-block TMRCA is not systematically biased across the
    true-TMRCA range. Binary per-block het emission saturates at deep
    times (block ~always het), crushing deep TMRCAs ~3x (ratio ~0.32),
    the *harmful* direction, since deep branches drive outgroup-based
    polarisation. The count-based (Poisson) emission removes that crush.
    Assert every quintile's median inferred/true ∈ [0.5, 2.5]: the lower
    bound is the meaningful one (catches the deep crush the binary emission
    produced). The upper bound is a loose guard. A residual *recent*
    over-estimation (~2x) remains and is benign, a near-uniform scaling of
    recent ingroup TMRCAs cancels in UPGMA topology, so it does not affect
    polarisation (ISM-bi accuracy is unchanged by it).
    """
    from itertools import combinations
    ts = _sim(samples=8, length=1_000_000, rec=1e-8, N=3e4, seed=1)
    block = 2000
    nb = int(np.ceil(ts.sequence_length / block))
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    node_of = {nm[int(s)]: int(s) for s in ts.samples()}
    pairs = list(combinations(range(len(names)), 2))

    true = np.full((len(pairs), nb), np.nan)
    for b in range(nb):
        c = (b + 0.5) * block
        if c >= ts.sequence_length:
            continue
        tr = ts.at(c)
        for pi, (a, bb) in enumerate(pairs):
            true[pi, b] = tr.tmrca(node_of[names[a]], node_of[names[bb]])

    sites, _ = _sites_and_truth(ts, names)
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, block_size=block,
        bake_genotypes=False,
    )
    g, _pos, bos, nbk = builder._genotype_matrix()
    hmm = PairwiseCoalescentHMM(len(names), mu=1.25e-8, rec_rate=1e-8,
                               block_size=block)
    inferred = hmm.block_tmrcas(g, bos, nbk)

    t = true[:, :nbk].ravel()
    i = inferred.ravel()
    m = np.isfinite(t) & (t > 0)
    t, i = t[m], i[m]
    qs = np.quantile(t, np.linspace(0, 1, 6))
    worst = []
    for k in range(5):
        sel = (t >= qs[k]) & (t <= qs[k + 1])
        ratio = np.median(i[sel] / t[sel])
        worst.append((qs[k], qs[k + 1], ratio))
    msg = "  ".join(f"[{lo:.0f},{hi:.0f}]:{r:.2f}" for lo, hi, r in worst)
    for lo, hi, r in worst:
        assert 0.5 < r < 2.5, f"TMRCA bias by true quintile: {msg}"


def test_pairwise_tmrca_matrix_tracks_truth():
    """Aggregated inferred pairwise TMRCAs track truth better than a
    single pair, this is what actually drives UPGMA tree quality.

    Correlates the inferred per-pair mean TMRCA against the true mean
    pairwise TMRCA across a 10-haplotype panel. Averaging over pairs
    lifts the correlation well above the single-pair ~0.6.
    """
    ts = _sim(samples=10, length=1_000_000, rec=1e-8, seed=4)
    block = 2000
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    from itertools import combinations
    pairs = list(combinations(range(len(names)), 2))

    # True per-pair mean TMRCA (genome-averaged).
    node_of = {f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()}
    true_mean = np.zeros(len(pairs))
    for tree in ts.trees():
        span = tree.interval.right - tree.interval.left
        for pi, (a, b) in enumerate(pairs):
            true_mean[pi] += tree.tmrca(node_of[names[a]], node_of[names[b]]) * span
    true_mean /= ts.sequence_length

    sites, _ = _sites_and_truth(ts, names)
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, block_size=block, progress=False,
        chunk_size=None,  # reads the builder's genotype matrix directly
    )
    g, pos, bos, nbk = inf.builder._genotype_matrix()
    hmm = PairwiseCoalescentHMM(len(names), mu=1.25e-8, rec_rate=1e-8,
                               block_size=block)
    inferred_mean = hmm.block_tmrcas(g, bos, nbk).mean(axis=1)
    r = np.corrcoef(inferred_mean, true_mean)[0, 1]
    assert r > 0.7, f"aggregated pairwise-TMRCA correlation too low: {r:.3f}"


# --------------------------------------------------------- tree sequence
def test_builder_writes_self_contained_trees(tmp_path, caplog):
    """LocalTreeBuilder.write yields a tiling, genotype-baked, loadable
    ``.trees`` that ARGBasedInference consumes directly.

    Self-describing includes the time units. The builder's node times are in
    generations, but the emitted tables left ``time_units`` at tskit's
    ``unknown`` default, so a file written here and read back declared nothing
    about the scale the ``mu`` it is scored with has to match.
    """
    ts = _sim(samples=8, length=100_000, seed=11)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    out = tmp_path / "inferred.trees"
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, window="20kb",
    )
    inferred = builder.to_tree_sequence()
    builder.write(out)
    assert out.exists()
    reloaded = tskit.load(str(out))
    assert reloaded.num_trees == inferred.num_trees
    assert reloaded.num_samples == 8
    # genotypes baked in as sites (default), one per input site
    assert reloaded.num_sites == len(sites)
    # sample names recoverable from individual metadata (self-describing)
    smap = {reloaded.individual(reloaded.node(int(s)).individual).metadata["name"]
            for s in reloaded.samples()}
    assert smap == set(names)
    # intervals tile [0, L) with no gaps. Every local tree single-rooted
    rights = [t.interval.right for t in reloaded.trees()]
    lefts = [t.interval.left for t in reloaded.trees()]
    assert lefts[0] == 0.0 and rights[-1] == reloaded.sequence_length
    for r_prev, l_next in zip(rights[:-1], lefts[1:]):
        assert r_prev == l_next
    for t in reloaded.trees():
        assert t.num_roots == 1
    # the branch-length scale mu is quoted against is declared, so a
    # per-generation rate is the right one and no units warning fires
    assert reloaded.time_units == "generations"
    # ARGBasedInference runs on the file directly (the point of baking)
    with caplog.at_level(logging.WARNING, logger="ancestree.ARGBasedInference"):
        arg = ARGBasedInference(reloaded, JC69(), mu=1.25e-8,
                                 progress=False)
    assert not [r for r in caplog.records if "time units" in r.message]
    assert sum(1 for _ in arg.infer()) == len(sites)


def test_prebuilt_tree_sequence_infer():
    """LocalTreeInference over a pre-built, genotype-baked tree sequence
    polarises without rebuilding, and infer() does not depend on attributes
    only the genotype-building path sets (sample_names / window)."""
    ts = _sim(samples=8, length=100_000, seed=13)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length,
    )
    baked = builder.to_tree_sequence()  # bake_genotypes default -> self-describing
    lti = LocalTreeInference(baked, JC69(), mu=1.25e-8,
                              progress=False)
    # sample names recovered from the ARG
    assert set(lti.sample_names) == set(names)
    # infer() streams a posterior per baked site (would AttributeError before)
    assert sum(1 for _ in lti.infer()) == len(sites)


class TestLocalTreeReprOnPrebuiltInput:

    def test_repr_does_not_raise(self):
        tables = tskit.TableCollection(sequence_length=100.0)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        root = tables.nodes.add_row(flags=0, time=1.0)
        for tip in range(4):
            tables.edges.add_row(left=0, right=100.0, parent=root, child=tip)
        tables.sort()
        inference = anc.LocalTreeInference(
            tables.tree_sequence(), JC69(), mu=1.25e-8,
        )
        assert "LocalTreeInference(" in repr(inference)


def test_reprs_name_the_configuration():
    """The HMM and the builder render their identifying fields."""
    hmm = PairwiseCoalescentHMM(3, mu=MU, rec_rate=REC, n_time_bins=8)
    assert repr(hmm) == "PairwiseCoalescentHMM(n_haplotypes=3, n_time_bins=8)"
    sites, names = toy_sites(range(0, 2000, 20))
    b = toy_builder(sites, names, 2000.0, n_time_bins=8)
    assert repr(b) == "LocalTreeBuilder(window_bp=200, n_time_bins=8)"


def test_builder_topology_only_when_not_baking():
    """bake_genotypes=False yields a sites-free topology tree sequence."""
    ts = _sim(samples=6, length=60_000, seed=12)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, bake_genotypes=False,
    )
    topo = builder.to_tree_sequence()
    assert topo.num_sites == 0
    assert topo.num_samples == 6


def test_builder_accepts_a_site_table_and_tallies_its_alleles():
    """A ``SiteTable`` is held as given, its unrepresentable alleles tallied."""
    sites, names = toy_sites(range(0, 1000, 50))
    sites[3] = Site(chrom="1", pos=sites[3].pos, alleles=("A", "AT"),
                    tip_alleles=dict(sites[3].tip_alleles))
    table = SiteTable.from_sites(sites, names)
    b = toy_builder(table, names, 1000.0)
    assert b.sites is table
    assert b._n_unrepresentable_sites == 1
    assert b._n_unrepresentable_tips == 0
    assert len(b.sites) == len(sites)


def test_genotype_matrix_follows_a_reordered_panel():
    """Columns follow ``sample_names``, an unknown name reading as uncalled."""
    sites, names = toy_sites(range(0, 1000, 50))
    table = SiteTable.from_sites(sites, names)
    order = ["h3", "h1", "hx"]
    b = toy_builder(table, order, 1000.0, validate_coverage=False)
    g, _pos, _bos, _nb = b._genotype_matrix()
    assert g.shape == (len(sites), 3)
    np.testing.assert_array_equal(g[:, 0], table.genotypes[:, 3])
    np.testing.assert_array_equal(g[:, 1], table.genotypes[:, 1])
    assert (g[:, 2] == -1).all()


def test_genotype_matrix_refuses_a_sample_observed_nowhere():
    """With coverage validation on, a never-called column is an error."""
    sites, names = toy_sites(range(0, 1000, 50))
    table = SiteTable.from_sites(sites, names)
    b = toy_builder(table, ["h0", "h1", "hx"], 1000.0)
    with pytest.raises(ValueError, match=r"no observed genotype.*hx"):
        b._genotype_matrix()


def test_pairwise_block_tmrcas_are_cached():
    """A second call returns the array of the first without a rerun."""
    sites, names = toy_sites(range(0, 1000, 20))
    b = toy_builder(sites, names, 1000.0)
    g, _pos, bos, nb = b._genotype_matrix()
    first = b._pairwise_block_tmrcas(g, bos, nb)
    assert b._pairwise_block_tmrcas(g, bos, nb) is first
    assert first.shape == (6, nb)


def test_uncalled_site_is_left_out_of_the_trees_and_scored_flat(caplog):
    """A site with no called tip is not baked, and streams at the flat posterior."""
    sites, names = toy_sites(range(0, 2000, 20))
    blank = Site(chrom="1", pos=1010, alleles=("A", "C"),
                 tip_alleles={n: None for n in names})
    sites = sorted(sites + [blank], key=lambda s: s.pos)
    b = toy_builder(sites, names, 2000.0)
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        ts = b.to_tree_sequence()
    assert ts.num_sites == len(sites) - 1
    assert 1010.0 not in set(ts.sites_position)
    assert any(r.getMessage().startswith(
        f"1 of {len(sites)} sites carry no called genotype")
        for r in caplog.records)

    inf = toy_inference(sites, names, n_ensemble=None)
    out = list(inf.infer())
    assert [s.pos for s, _ in out] == [s.pos for s in sites]
    by_pos = {s.pos: p for s, p in out}
    np.testing.assert_allclose(by_pos[1010].values, np.full(4, 0.25))
    assert tuple(by_pos[1010].alleles) == tuple(STATES)
    peaked = [p for s, p in out if s.pos != 1010]
    assert max(p.max_prob for p in peaked) > 0.25


# --------------------------------------------------------- end to end
def test_polarisation_accuracy_and_outgroup_gain():
    """Genotype-only polarisation is well above chance. Outgroups help."""
    # ingroup + a diverged outgroup population
    dem = msprime.Demography()
    dem.add_population(name="ingroup", initial_size=1e4)
    dem.add_population(name="outgroup", initial_size=1e4)
    dem.add_population(name="anc", initial_size=1e4)
    dem.add_population_split(time=2e5, ancestral="anc",
                             derived=["ingroup", "outgroup"])
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(10, population="ingroup", ploidy=1),
                 msprime.SampleSet(2, population="outgroup", ploidy=1)],
        demography=dem, sequence_length=300_000,
        recombination_rate=1e-8, random_seed=5,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=5)

    pop_of = {f"tsk_{ind.id}": ts.node(int(ind.nodes[0])).population
              for ind in ts.individuals()}
    ing_pop = [p.id for p in ts.populations()
               if p.metadata.get("name") == "ingroup"][0]
    ingroup = [n for n, p in pop_of.items() if p == ing_pop]
    outgroup = [n for n, p in pop_of.items() if p != ing_pop]

    sites, truth = _sites_and_truth(ts, ingroup)  # ingroup-poly filter

    def accuracy(panel):
        inf = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=panel,
            sequence_length=ts.sequence_length, window="20kb", progress=False,
        )
        h = n = 0
        for site, post in inf.infer():
            t = truth.get(site.pos)
            if t is None:
                continue
            n += 1
            h += int(post.map_allele == t)
        return h / n

    acc_ingroup = accuracy(ingroup)
    acc_with_out = accuracy(ingroup + outgroup)
    assert acc_ingroup > 0.65, f"ingroup-only accuracy too low: {acc_ingroup:.3f}"
    # outgroups inform rooting → should help (or at least not hurt).
    assert acc_with_out >= acc_ingroup, (
        f"outgroups hurt: {acc_ingroup:.3f} -> {acc_with_out:.3f}"
    )


def test_runs_from_genotypes_only():
    """No ancestral state is consumed, Sites carry only tip alleles."""
    ts = _sim(samples=6, length=50_000, seed=2)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, progress=False,
    )
    posteriors = list(inf.infer())
    assert len(posteriors) > 0
    for site, post in posteriors:
        assert abs(float(np.sum(post.values)) - 1.0) < 1e-9


def test_runs_from_vcf_path(tmp_path):
    """A VCF path is read as genotypes. The haplotype panel is derived from it."""
    from ancestree import Inference

    ts = _sim(samples=6, length=50_000, seed=2)
    names = [f"n{i}" for i in range(int(ts.num_samples))]
    vcf = tmp_path / "snps.vcf"
    with open(vcf, "w") as f:
        ts.write_vcf(f, individual_names=names, allow_position_zero=True)

    # No sample_names passed, they must be derived from the VCF.
    inf = Inference.from_local_tree(
        vcf, model=JC69(), mu=1.25e-8, rec_rate=1e-8,
        sequence_length=ts.sequence_length, progress=False,
    )
    assert list(inf.sample_names) == names
    posteriors = list(inf.infer())
    assert len(posteriors) > 0
    for _site, post in posteriors:
        assert abs(float(np.sum(post.values)) - 1.0) < 1e-9


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
def test_runs_from_vcz_path(tmp_path):
    """A VCZ path is read as genotypes. The panel is derived from the store."""
    from ancestree import Inference
    from testing._helpers import ts_to_vcz

    ts = _sim(samples=6, length=50_000, seed=2)
    vcz = tmp_path / "snps.vcz"
    names = ts_to_vcz(ts, vcz)

    # No sample_names passed, they must be derived from the VCZ store.
    inf = Inference.from_local_tree(
        str(vcz), model=JC69(), mu=1.25e-8, rec_rate=1e-8,
        sequence_length=ts.sequence_length, progress=False,
    )
    assert list(inf.sample_names) == names
    posteriors = list(inf.infer())
    assert len(posteriors) > 0
    for _site, post in posteriors:
        assert abs(float(np.sum(post.values)) - 1.0) < 1e-9


@pytest.mark.parametrize("chunk_size", [None, "50kb"])
def test_local_tree_non_default_contig(tmp_path, chunk_size):
    import cyvcf2
    import msprime

    from ancestree.inference import Inference

    ts = msprime.sim_ancestry(
        samples=8, ploidy=1, sequence_length=1e5, recombination_rate=1e-8,
        population_size=1e4, random_seed=4,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=4)
    assert ts.num_sites > 0
    vcf = tmp_path / "panel.vcf"
    with open(vcf, "w") as f:
        ts.write_vcf(f, contig_id="chr2")

    kw = dict(mu=1e-8, rec_rate=1e-8, progress=False)
    if chunk_size is None:
        kw["sequence_length"] = float(ts.sequence_length)
    else:
        kw["chunk_size"] = chunk_size
    out = tmp_path / f"out_{chunk_size}.vcf"
    n = Inference.from_local_tree(str(vcf), **kw).to_vcf(str(out))

    assert n > 0
    recs = list(cyvcf2.VCF(str(out)))
    assert recs and all(r.CHROM == "chr2" for r in recs)
    assert any(r.INFO.get("AA") is not None for r in recs)


# ----------------------------------------------------- pairwise-TMRCA export
def test_pairwise_tmrcas_export(tmp_path):
    """`pairwise_tmrcas()` returns a labelled (n_pairs, n_blocks) matrix and
    writes it to .npz / .csv / .tsv. The cached matrix is shared with
    `to_tree_sequence`."""
    import csv
    ts = _sim(samples=6, length=120_000, seed=7)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, window="20kb", block_size=1000,
        progress=False,
    )
    pt = inf.pairwise_tmrcas()
    n_hap = len(names)
    n_pairs = n_hap * (n_hap - 1) // 2
    assert isinstance(pt, PairwiseTmrcas)
    assert pt.tmrca.shape == (n_pairs, pt.block_midpoints.shape[0])
    assert len(pt.pairs) == n_pairs
    assert pt.pairs[0] == (names[0], names[1])  # combinations order
    assert pt.pairs[-1] == (names[-2], names[-1])
    assert np.all(np.isfinite(pt.tmrca)) and np.all(pt.tmrca > 0)
    # block midpoints: half a block in, one block apart, inside [0, L]
    assert pt.block_midpoints[0] == 500.0
    assert np.all(np.diff(pt.block_midpoints) > 0)
    assert pt.block_midpoints[-1] <= ts.sequence_length
    # cached: a second call returns the same underlying array object
    assert inf.pairwise_tmrcas().tmrca is pt.tmrca

    # long-form CSV / TSV: header + one row per pair × block
    pcsv = tmp_path / "t.csv"
    inf.write_pairwise_tmrcas(pcsv)
    rows = list(csv.reader(open(pcsv)))
    assert rows[0] == ["hap_i", "hap_j", "block_midpoint", "tmrca"]
    assert len(rows) - 1 == n_pairs * pt.tmrca.shape[1]
    ptsv = tmp_path / "t.tsv"
    inf.write_pairwise_tmrcas(ptsv)
    assert next(csv.reader(open(ptsv), delimiter="\t"))[0] == "hap_i"

    # NPZ round-trips the raw arrays
    pnpz = tmp_path / "t.npz"
    inf.write_pairwise_tmrcas(pnpz)
    z = np.load(pnpz)
    np.testing.assert_array_equal(z["tmrca"], pt.tmrca)
    np.testing.assert_array_equal(z["block_midpoints"], pt.block_midpoints)
    np.testing.assert_array_equal(z["sample_names"], np.asarray(names))

    with pytest.raises(ValueError, match="suffix"):
        inf.write_pairwise_tmrcas(tmp_path / "t.bogus")


def test_pairwise_tmrcas_unavailable_paths(tmp_path):
    """`pairwise_tmrcas()` errors clearly on the chunked and pre-built paths."""
    ts = _sim(samples=6, length=120_000, seed=7)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)

    # One segment spans the whole input and reproduces the unsegmented
    # build, so the per-block TMRCAs are well defined and are returned.
    one_segment = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window="20kb", block_size=1000, chunk_size="1mb", progress=False,
    )
    assert one_segment.pairwise_tmrcas() is not None

    # Several segments are glued into the same genome-wide matrix, so the
    # chunk size does not change what the caller receives.
    segmented = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window="20kb", block_size=1000, chunk_size="40kb", progress=False,
    )
    many = segmented.pairwise_tmrcas()
    one = one_segment.pairwise_tmrcas()
    assert many.pairs == one.pairs
    assert many.tmrca.shape[0] == one.tmrca.shape[0]
    assert np.all(np.diff(many.block_midpoints) > 0)

    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, window="20kb", block_size=1000,
    )
    tpath = tmp_path / "local.trees"
    builder.write(tpath)
    prebuilt = LocalTreeInference(tpath, JC69(), mu=1.25e-8, progress=False)
    with pytest.raises(NotImplementedError, match="pre-built"):
        prebuilt.pairwise_tmrcas()


def test_run_to_run_determinism():
    """The same genotypes give bit-identical posteriors on a repeat run.

    Local-tree inference draws no randomness (HMM, UPGMA and the Felsenstein
    pass are all deterministic), so two independent runs over the same sites
    must agree exactly, this guards against an accidental RNG, dict-ordering
    or thread-race dependency creeping in.
    """
    ts = _sim(samples=8, length=120_000, seed=2)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              sequence_length=ts.sequence_length, window="20kb", progress=False)
    a = list(LocalTreeInference(sites, JC69(), **kw).infer())
    b = list(LocalTreeInference(sites, JC69(), **kw).infer())
    assert [s.pos for s, _ in a] == [s.pos for s, _ in b] and len(a) > 0
    for (sa, pa), (sb, pb) in zip(a, b):
        assert sa.pos == sb.pos and pa.map_allele == pb.map_allele
        np.testing.assert_array_equal(pa.values, pb.values)


def test_ensemble_row_count_mismatch_is_an_error(monkeypatch):
    """An ensemble returning the wrong number of rows is refused."""
    from ancestree._ensemble import SegmentEnsemble
    sites, names = toy_sites(range(0, 1000, 50))
    inf = toy_inference(sites, names, n_ensemble=2, sequence_length=1000.0)
    monkeypatch.setattr(
        SegmentEnsemble, "posterior",
        lambda self, *a, **k: np.full((len(sites) - 1, 4), 0.25))
    with pytest.raises(RuntimeError, match="must pair one to one"):
        list(inf.infer())


def test_baseline_check_compares_against_the_named_outgroups(caplog):
    """The baseline hook reports the outgroups and logs the agreement line."""
    sites, names = toy_sites(range(0, 2000, 20))
    inf = toy_inference(sites, names, n_ensemble=None, outgroup_samples=["h3"],
                        baseline_check=True)
    assert inf._baseline_outgroup_samples() == ("h3",)
    assert inf._baseline_ingroup_samples() == ("h0", "h1", "h2")
    with caplog.at_level(logging.INFO, logger="ancestree"):
        out = list(inf.infer())
    assert len(out) == len(sites)
    assert any("agree" in r.getMessage().lower() for r in caplog.records)


def test_to_tree_sequence_yields_the_plug_in_tree_once_without_an_ensemble():
    """Without an ensemble the single group holds the plug-in tree sequence."""
    sites, names = toy_sites(range(0, 1000, 20))
    inf = toy_inference(sites, names, n_ensemble=None, sequence_length=1000.0)
    groups = list(inf.to_tree_sequence())
    assert len(groups) == 1
    (interval, members), = groups
    assert interval == (0.0, 1000.0)
    members = list(members)
    assert len(members) == 1
    assert members[0] is inf.point_tree_sequence()
    assert members[0].num_sites == len(sites)


# --------------------------------------------------------- provenance
def test_provenance_of_an_unchunked_ensemble_run():
    """The ensemble record carries the builder's widths and the draw settings."""
    sites, names = toy_sites(range(0, 1000, 20))
    inf = toy_inference(sites, names, n_ensemble=3, ensemble_seed=5,
                        member_chunk=2, sequence_length=1000.0,
                        outgroup_samples=["h3"])
    params = inf._provenance_parameters()
    assert params["model"] == "JC69"
    assert params["prior"] == "StationaryPrior"
    assert params["mu"] == pytest.approx(MU)
    assert params["rec_rate"] == pytest.approx(REC)
    assert params["window_bp"] == 200
    assert params["block_size"] == 50
    assert params["n_time_bins"] == 32
    assert params["n_ensemble"] == 3
    assert params["ensemble_seed"] == 5
    assert params["member_chunk"] == 2
    assert inf.provenance()["parameters"]["outgroup_samples"] == ["h3"]
    assert params["focal"] == "ingroup_mrca"


def test_provenance_of_an_unchunked_plug_in_run():
    """The plug-in record reads model, prior and focal from the point ARG."""
    sites, names = toy_sites(range(0, 1000, 20))
    inf = toy_inference(sites, names, n_ensemble=None, sequence_length=1000.0)
    params = inf._provenance_parameters()
    assert params["model"] == "JC69"
    assert params["prior"] == "StationaryPrior"
    assert params["mu"] == pytest.approx(MU)
    assert params["window_bp"] == 200
    assert params["block_size"] == 50
    assert "n_ensemble" not in params
    assert inf._arg is not None


# --------------------------------------------------------- HMM correctness
def _bruteforce_pmean(counts, t_rep, lam, log_lam, r, pi,
                      step=None, mask=None, escale=None):
    """Reference O(T²) full-matrix forward-backward posterior-mean TMRCA.

    The kernel uses the structured ``A[i,j] = r_i·δ_ij + (1-r_i)·ν_j``
    recursion in O(T), with ``ν_j ∝ π_j (1-r_j)`` the reset row that leaves
    ``π`` invariant. This builds the dense transition matrix and runs a
    textbook scaled forward-backward, so equality validates the structured
    algebra exactly. Emission is the Poisson likelihood of ``counts[k]``.
    """
    T, n = len(t_rep), len(counts)
    step = np.ones(n - 1) if step is None else np.asarray(step, float)
    mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    escale = np.ones(n) if escale is None else np.asarray(escale, float)

    def trans(width):
        """Dense transition over a step of ``width`` nominal blocks."""
        rk = r ** width
        nu = pi * (1.0 - rk)
        nu = nu / nu.sum()
        A = (1.0 - rk)[:, None] * nu[None, :]
        A[np.diag_indices(T)] += rk
        return A

    def emit(k):
        if not mask[k]:
            return np.ones(T)  # uninformative block
        ll = -escale[k] * lam + counts[k] * log_lam
        return np.exp(ll - ll.max())

    alpha = np.zeros((n, T))
    alpha[0] = pi * emit(0)
    alpha[0] /= alpha[0].sum()
    for k in range(1, n):
        alpha[k] = (alpha[k - 1] @ trans(step[k - 1])) * emit(k)
        alpha[k] /= alpha[k].sum()
    beta = np.zeros((n, T))
    beta[-1] = 1.0
    for k in range(n - 2, -1, -1):
        beta[k] = trans(step[k]) @ (emit(k + 1) * beta[k + 1])
        beta[k] /= beta[k].sum()
    g = alpha * beta
    g /= g.sum(axis=1, keepdims=True)
    return g @ t_rep


def test_hmm_matches_bruteforce_forward_backward():
    """Structured O(T) kernel == dense O(T²) forward-backward (exact)."""
    rng = np.random.default_rng(7)
    t_rep = np.geomspace(1e2, 1e5, 24)
    lam = 2 * 1.25e-8 * 2000 * t_rep
    log_lam = np.log(lam)
    r = np.exp(-1e-8 * 2000 * t_rep)
    pi = np.full(24, 1.0 / 24)
    for _ in range(5):
        counts = rng.poisson(rng.uniform(0.05, 0.6), size=200).astype(np.int64)
        ref = _bruteforce_pmean(counts, t_rep, lam, log_lam, r, pi)
        got = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi)
        np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def test_hmm_matches_bruteforce_with_step_mask_and_exposure():
    """The same equality on the finite-step, masked and rate-scaled branches.

    The default-argument comparison exercises none of them, so a defect in the
    step exponent, the uniform emission a masked block takes, or the exposure
    factor would have gone unseen.
    """
    rng = np.random.default_rng(11)
    n, T_ = 120, 24
    t_rep = np.geomspace(1e2, 1e5, T_)
    lam = 2 * 1.25e-8 * 2000 * t_rep
    log_lam = np.log(lam)
    r = np.exp(-1e-8 * 2000 * t_rep)
    pi = np.full(T_, 1.0 / T_)
    counts = rng.poisson(0.3, size=n).astype(np.int64)
    # A masked gap, a wider step across it, and a non-uniform exposure.
    step = rng.uniform(0.5, 6.0, size=n - 1)
    mask = np.ones(n, bool)
    mask[40:50] = False
    escale = rng.uniform(0.3, 3.0, size=n)
    ref = _bruteforce_pmean(counts, t_rep, lam, log_lam, r, pi,
                            step=step, mask=mask, escale=escale)
    got = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi,
                                        step=step, mask=mask, escale=escale)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("true_t", [5.0e3, 2.0e4, 6.0e4])
def test_hmm_recovers_constant_tmrca(true_t):
    """A constant-density sequence (no recombination) concentrates the
    posterior near the TMRCA implied by the per-block difference count.

    Graded against the grid mean rather than a fixed window: the kernel's own
    dead-block fallback IS the grid mean, so a window wide enough to contain it
    is passed by a kernel that ignores the data entirely. Requiring the
    estimate to beat that fallback, and requiring three generating TMRCAs to
    come out ordered, is what distinguishes inference from the fallback.
    """
    t_rep = np.geomspace(1e2, 1e5, 40)
    block, mu = 2000, 1.25e-8
    lam = 2 * mu * block * t_rep
    log_lam = np.log(lam)
    r = np.exp(-1e-8 * block * t_rep)
    pi = np.full(40, 1.0 / 40)
    lam_true = 2 * mu * block * true_t  # expected diffs/block at true_t
    rng = np.random.default_rng(0)
    counts = rng.poisson(lam_true, size=4000).astype(np.int64)  # iid, no recomb
    pm = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi)
    got = float(np.median(pm))
    assert 0.5 * true_t < got < 2.0 * true_t, got
    # Strictly closer to the truth than the uninformative fallback.
    grid_mean = float(t_rep.mean())
    assert abs(got - true_t) < abs(grid_mean - true_t), (got, grid_mean)


def test_hmm_separates_two_generating_tmrcas():
    """Estimates must order with the TMRCA that generated the counts.

    A kernel returning any constant passes a per-value tolerance. Ordering
    across generating values cannot be faked by one.
    """
    t_rep = np.geomspace(1e2, 1e5, 40)
    block, mu = 2000, 1.25e-8
    lam = 2 * mu * block * t_rep
    log_lam = np.log(lam)
    r = np.exp(-1e-8 * block * t_rep)
    pi = np.full(40, 1.0 / 40)
    rng = np.random.default_rng(0)
    got = []
    for true_t in (5.0e3, 2.0e4, 6.0e4):
        counts = rng.poisson(2 * mu * block * true_t, size=4000).astype(np.int64)
        got.append(float(np.median(
            smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi))))
    assert got[0] < got[1] < got[2], got


# --------------------------------------------------------- topology
def _splits(tree, n):
    """Non-trivial clades (as frozensets of sample ids) of a tskit tree."""
    out = set()
    for u in tree.nodes():
        if tree.is_sample(u):
            continue
        clade = frozenset(tree.samples(u))
        if 2 <= len(clade) <= n - 1:
            out.add(clade)
    return out


def _rf(t1, t2, n):
    """Robinson-Foulds distance (symmetric difference of split sets)."""
    s1, s2 = _splits(t1, n), _splits(t2, n)
    return len(s1 ^ s2)


def test_inferred_topology_beats_chance():
    """Inferred window-tree topologies match the true local trees far
    better than a label-permuted baseline (Robinson-Foulds)."""
    ts = _sim(samples=8, length=600_000, rec=1e-8, seed=9)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    node_of = {f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    n = len(names)
    # True ts in canonical sample order so leaf ids 0..n-1 == names order.
    true_ts = ts.simplify(samples=[node_of[x] for x in names])

    sites, _ = _sites_and_truth(ts, names)
    builder = LocalTreeBuilder(
        sites, mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, window="20kb",
        bake_genotypes=False,
    )
    inferred = builder.to_tree_sequence()

    rng = np.random.default_rng(0)
    real, perm = [], []
    for itree in inferred.trees():
        mid = 0.5 * (itree.interval.left + itree.interval.right)
        ttree = true_ts.at(mid)
        real.append(_rf(itree, ttree, n))
        # permuted baseline: relabel the true tree's samples at random,
        # which preserves its shape but destroys correspondence.
        perm_map = dict(zip(range(n), rng.permutation(n)))
        permuted = {frozenset(perm_map[s] for s in cl)
                    for cl in _splits(ttree, n)}
        isplits = _splits(itree, n)
        perm.append(len(isplits ^ permuted))
    assert np.mean(real) < 0.7 * np.mean(perm), (
        f"inferred topology not better than chance: "
        f"RF_real={np.mean(real):.2f} vs RF_perm={np.mean(perm):.2f}"
    )


def test_block_tmrcas_segment_reset():
    """`seg_bounds` severs the HMM at segment breaks (exact Δ→∞ reset).

    Two segments on one axis: the first all-different (deep TMRCA), the
    second all-identical (recent). Without segments the chain carries the
    deep signal across the boundary and decays into the second segment;
    with `seg_bounds` splitting at the boundary the second segment resets to
    the prior, so its first block is recent and the segment is flat.
    """
    nb, k = 24, 12
    g = np.zeros((nb, 2), dtype=np.int8)
    g[:k] = (0, 1)  # segment A: pair differs every block (deep)
    g[k:] = (0, 0)  # segment B: pair identical (recent)
    bos = np.arange(nb)
    hmm = PairwiseCoalescentHMM(2, mu=1.25e-8, rec_rate=1e-8, block_size=1)

    joint = hmm.block_tmrcas(g, bos, nb)[0]
    split = hmm.block_tmrcas(g, bos, nb, seg_bounds=[(0, k), (k, nb)])[0]

    # Backward compatibility: a single full-span segment == the default.
    same = hmm.block_tmrcas(g, bos, nb, seg_bounds=[(0, nb)])[0]
    assert np.allclose(joint, same)

    # Decoupling: the reset removes the deep carry-over at the boundary, so
    # segment B starts (and stays) more recent under the split than the joint
    # pass, and is essentially flat (no boundary gradient).
    assert split[k] < joint[k]
    assert np.ptp(split[k:]) < np.ptp(joint[k:])
    assert split[k:].max() <= joint[k:].max() + 1e-9


def test_segment_aware_windows_and_bounds():
    """`segment_breaks` give per-segment, runt-free windows and block bounds.

    Windows never cross a break, are equal-width within a segment (no tiny
    end window), and the segment block-bounds partition ``[0, n_blocks)``.
    """
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=10000, block_size=100, window=1000,
        segment_breaks=[3050, 6000],  # 3050 snaps to the 3000 block edge
    )
    segs = b._segment_intervals()
    assert segs == [(0.0, 3000.0), (3000.0, 6000.0), (6000.0, 10000.0)]

    wins = b._window_intervals()
    # every window lies strictly within one segment (crosses no break)
    for left, right in wins:
        seg = [s for s in segs if s[0] <= left and right <= s[1]]
        assert len(seg) == 1, (left, right)
    # equal width within each segment (no runt remainder)
    for s0, s1 in segs:
        widths = [round(r - l, 6) for l, r in wins if s0 <= l < s1]
        assert max(widths) - min(widths) < 1e-6

    bounds = b._seg_block_bounds(100)
    assert bounds == [(0, 30), (30, 60), (60, 100)]
    # contiguous partition of [0, n_blocks)
    assert bounds[0][0] == 0 and bounds[-1][1] == 100
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1))


def test_no_segment_breaks_covers_axis():
    """Default (no breaks): one segment, windows cover [0, L)."""
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=9000, block_size=100, window=1000,
    )
    assert b._segment_intervals() == [(0.0, 9000.0)]
    wins = b._window_intervals()
    assert wins[0][0] == 0.0 and wins[-1][1] == 9000.0
    assert all(wins[i][1] == wins[i + 1][0] for i in range(len(wins) - 1))


def test_chunked_multicontig_decoupling_exact():
    """Chunked path: contigs are independent and coordinate-stable.

    Running two contigs together yields exactly the per-site posteriors of
    running each alone (so contig B never influences contig A), in genomic
    order, with original positions preserved.
    """
    from ancestree import JC69
    a, names = toy_sites(range(1000, 1000 + 80 * 100, 100), seed=1)
    b, _ = toy_sites(range(500, 500 + 60 * 100, 100), seed=2, chrom="2")
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              window=500, block_size=100, chunk_size="1mb")

    out_ab = list(LocalTreeInference(a + b, JC69(), **kw).infer())
    out_a = list(LocalTreeInference(a, JC69(), **kw).infer())
    out_b = list(LocalTreeInference(b, JC69(), **kw).infer())

    assert len(out_ab) == len(a) + len(b)  # all core sites emitted
    assert [s.pos for s, _ in out_ab] == [s.pos for s, _ in out_a + out_b]
    assert [s.chrom for s, _ in out_ab] == ["1"] * len(a) + ["2"] * len(b)
    for (_, p_join), (_, p_sep) in zip(out_ab, out_a + out_b):
        assert np.allclose(p_join.values, p_sep.values)


def test_chunked_halo_covers_all_core_sites():
    """Slicing a single contig (chunk_size << span) still emits every site once,
    in order, with halo overlap discarded (no duplicates, no gaps)."""
    from ancestree import JC69
    a, names = toy_sites(range(0, 4000, 20), seed=3)  # 200 sites, ~4 kb
    out = list(LocalTreeInference(
        a, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=300, block_size=50, chunk_size=1000, halo=500,  # forces slicing
    ).infer())
    positions = [s.pos for s, _ in out]
    assert positions == [s.pos for s in a]  # each site once, in order, no halo dup


def test_chunked_parallel_matches_serial():
    """The fork-pool segment path (n_workers>1) is identical to serial."""
    from ancestree import JC69
    a, names = toy_sites(range(0, 6000, 20), seed=7)  # one sliced contig
    a2, _ = toy_sites(range(100, 100 + 50 * 20, 20), seed=8, chrom="2")
    src = a + a2
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              window=300, block_size=50, chunk_size=1500, halo=400)
    serial = list(LocalTreeInference(src, JC69(), n_workers=1, **kw).infer())
    par = list(LocalTreeInference(src, JC69(), n_workers=3, **kw).infer())
    assert [s.pos for s, _ in serial] == [s.pos for s, _ in par]
    assert [s.chrom for s, _ in serial] == [s.chrom for s, _ in par]
    for (_, ps), (_, pp) in zip(serial, par):
        assert np.allclose(ps.values, pp.values)


def test_to_arg_writes_pseudo_arg(tmp_path):
    """``to_arg`` (base ``Inference``) writes the inferred pseudo-ARG tree
    sequence with ancestral states annotated from the MAP allele."""
    ts = _sim(samples=6, length=50_000, seed=2)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=ts.sequence_length, progress=False,
    )
    out = tmp_path / "pseudo.trees"
    n = inf.to_arg(str(out))
    assert n > 0
    annotated = tskit.load(str(out))
    assert all(s.ancestral_state in STATES for s in annotated.sites())


@pytest.mark.parametrize("name", ["panel.vcf.gz", "panel.gvcf.gz"])
def test_a_vcf_source_is_annotated_whatever_its_suffix(tmp_path, name):
    import shutil

    from testing._helpers import DEMO_VCF

    path = str(tmp_path / name)
    shutil.copy(DEMO_VCF, path)
    inf = LocalTreeInference(path, mu=5e-8, rec_rate=1e-8,
                             sequence_length=1e6, progress=False)
    assert inf._output_template(None, "vcf", "out.vcf", False) == (path, None)


def test_to_vcf_from_sites_annotates_every_record(tmp_path):
    """``to_vcf`` on a sites source writes one annotated record per site."""
    import cyvcf2
    sites, names = toy_sites(range(20, 1000, 20), chrom="chr7")
    inf = toy_inference(sites, names, n_ensemble=None, sequence_length=1000.0)
    out = tmp_path / "out.vcf"
    n = inf.to_vcf(str(out))
    assert n == len(sites)
    recs = list(cyvcf2.VCF(str(out)))
    assert len(recs) == len(sites)
    assert all(r.CHROM == "chr7" for r in recs)
    assert all(r.INFO.get("AA") in STATES for r in recs)


def _stitch_inputs(seed=12, samples=8, length=60_000):
    """A simulated single-contig ARG → (Site list, haplotype names)."""
    ts = _sim(samples=samples, length=length, seed=seed)
    names = [f"n{i}" for i in range(int(ts.num_samples))]
    nm = {int(s): names[i] for i, s in enumerate(ts.samples())}
    sites = []
    for v in ts.variants():
        ta = {nm[int(s)]: v.alleles[g] for s, g in zip(ts.samples(), v.genotypes)}
        if len(set(ta.values())) < 2:
            continue
        sites.append(Site(chrom="1", pos=int(v.site.position),
                          alleles=canonical_alleles(v.alleles), tip_alleles=ta))
    return sites, names


def test_chunked_to_tree_sequence_is_valid():
    """The chunked path stitches a single, valid genome-wide tree sequence:
    one shared sample set (names preserved), tiling edges, baked sites."""
    sites, names = _stitch_inputs()
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              window=300, block_size=100)
    cts = LocalTreeInference(sites, JC69(), chunk_size=8000, halo=2000,
                             **kw).point_tree_sequence()
    assert cts.num_samples == len(names)
    got = [cts.individual(cts.node(s).individual).metadata["name"]
           for s in cts.samples()]
    assert got == names
    # Every input site appears once, in order, at its genome position.
    assert [int(s.position) for s in cts.sites()] == [s.pos for s in sites]
    cts.tables.tree_sequence()  # re-validates the table collection


def test_chunked_vs_unchunked_tree_sequence_agree():
    """Chunked-stitched and unchunked tree sequences carry identical sites /
    genotypes and yield the same MAP ancestral calls (genealogies agree)."""
    sites, names = _stitch_inputs(seed=13)
    seq_len = max(s.pos for s in sites) + 1000
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              window=300, block_size=100)
    cts = LocalTreeInference(sites, JC69(), chunk_size=8000, halo=2000,
                             **kw).point_tree_sequence()
    uts = LocalTreeInference(sites, JC69(), sequence_length=seq_len,
                             **kw).point_tree_sequence()
    # Observed data is deterministic: same site positions and same decoded
    # per-haplotype nucleotides (compare decoded alleles, not the 0/1 index
    # matrix, which depends on the independently-baked ancestral state).
    assert [s.position for s in cts.sites()] == [s.position for s in uts.sites()]

    def decoded(t):
        return [tuple(v.alleles[g] for g in v.genotypes) for v in t.variants()]
    assert decoded(cts) == decoded(uts)

    def maps(t):
        return [p.map_allele
                for _s, p in ARGBasedInference(t, JC69(), mu=1.25e-8).infer()]
    agree = np.mean([a == b for a, b in zip(maps(cts), maps(uts))])
    assert agree >= 0.9, f"chunked vs unchunked MAP agreement too low: {agree:.3f}"


def test_chunked_stitched_similar_to_single_chunk():
    """Multi-segment stitched ≈ single-chunk (one segment) tree sequence.

    Both go through the segmented stitcher, so this isolates the effect of
    cutting the genome into chunks. Per-window UPGMA topology is locally noisy
    (sensitive to the exact windowing), so similarity is asserted in
    aggregate: identical site / sample structure and a near-identical
    genome-averaged pairwise-divergence matrix (branch-mode).
    """
    import itertools
    sites, names = _stitch_inputs(seed=15, samples=16, length=150_000)
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              window=400, block_size=100)
    single = LocalTreeInference(sites, JC69(), chunk_size=10**9, halo=4000,
                                **kw).point_tree_sequence()  # one segment
    multi = LocalTreeInference(sites, JC69(), chunk_size=20000, halo=6000,
                               **kw).point_tree_sequence()  # ~8 segments stitched
    assert single.num_samples == multi.num_samples == len(names)
    assert [s.position for s in single.sites()] == [s.position for s in multi.sites()]
    assert single.sequence_length == multi.sequence_length
    # inferred node times are in generations, and the stitched output says so
    assert single.time_units == multi.time_units == "generations"

    ss = [[int(s)] for s in single.samples()]
    idx = list(itertools.combinations(range(len(ss)), 2))
    d_single = single.divergence(ss, indexes=idx, mode="branch")
    d_multi = multi.divergence(ss, indexes=idx, mode="branch")
    r = np.corrcoef(d_single, d_multi)[0, 1]
    rel = np.median(np.abs(d_single - d_multi) / ((d_single + d_multi) / 2))
    assert r > 0.95, f"divergence correlation too low: {r:.3f}"
    assert rel < 0.2, f"median relative divergence difference too high: {rel:.3f}"


def test_chunked_to_arg_round_trip(tmp_path):
    """Chunked ``to_arg`` writes a genome-wide ``.trees`` whose annotated
    ancestral states match the chunked ``infer()`` MAP alleles."""
    import tskit
    sites, names = _stitch_inputs(seed=14)
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=300, block_size=100, chunk_size=8000, halo=2000,
    )
    map_by_pos = {s.pos: p.map_allele for s, p in inf.infer()}
    out = tmp_path / "chunked.trees"
    n = inf.to_arg(str(out))
    assert n == len(sites) > 0
    annotated = tskit.load(str(out))
    for site in annotated.sites():
        assert site.ancestral_state == map_by_pos[int(site.position)]


def test_chunked_tree_sequence_multicontig_raises():
    """A genome-wide tree sequence needs a single coordinate axis, so a
    multi-contig chunked source is rejected (use VCF/VCZ output)."""
    a, names = toy_sites(range(0, 6000, 20), seed=7)
    b, _ = toy_sites(range(0, 6000, 20), seed=8, chrom="2")
    inf = LocalTreeInference(
        a + b, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=300, block_size=50, chunk_size=1500, halo=400,
    )
    with pytest.raises(NotImplementedError, match="single contig"):
        inf.point_tree_sequence()


class _ReIterSource:
    """Re-iterable site source that counts how many times it is streamed."""
    def __init__(self, sites):
        self._sites = sites
        self.iters = 0
    def __iter__(self):
        self.iters += 1
        return iter(self._sites)


def test_chunked_streaming_reiterable_source():
    """A re-iterable source is streamed (not materialised) and matches a list."""
    from ancestree import JC69
    a, names = toy_sites(range(0, 3000, 30), seed=11)
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names, window="50snp",
              block_size=50, chunk_size=1000, halo=300)
    src = _ReIterSource(a)
    out = list(LocalTreeInference(src, JC69(), **kw).infer())
    base = list(LocalTreeInference(a, JC69(), **kw).infer())
    assert [s.pos for s, _ in out] == [s.pos for s, _ in base]
    for (_, po), (_, pb) in zip(out, base):
        assert np.allclose(po.values, pb.values)
    # window="50snp" streams once to size the global window and once to process
    assert src.iters >= 2


def test_chunked_rejects_unsorted():
    """Out-of-order positions within a contig are rejected on the chunked path."""
    from ancestree import JC69, Site
    names = ["h0", "h1"]
    def site(p):
        return Site(chrom="1", pos=p, alleles=("A", "C"),
                    tip_alleles={"h0": "A", "h1": "C"})
    bad = [site(100), site(50)]  # descending -> invalid
    with pytest.raises(ValueError, match="sorted"):
        list(LocalTreeInference(bad, JC69(), mu=1.25e-8, rec_rate=1e-8,
             sample_names=names, window=200, block_size=50,
             chunk_size=1000).infer())


def test_auto_halo_is_grounded_and_clamped():
    """The 'auto' halo resolves to a positive value clamped to [block, chunk/2]."""
    from ancestree import JC69
    a, names = toy_sites(range(0, 5000, 25), seed=4)
    inf = LocalTreeInference(
        a, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=300, block_size=50, chunk_size=2000, halo="auto",
    )
    inf._resolve_segmentation_params()
    assert inf.block_size <= inf._halo_value <= inf._chunk_size // 2
    assert inf._wbp >= inf.block_size


# ----------------------------------------- finite-Δ kernel: step & mask
def _kernel_grid(block=1000, T=24):
    t_rep = np.geomspace(1e2, 1e5, T)
    lam = 2 * 1.25e-8 * block * t_rep
    return t_rep, lam, np.log(lam), np.exp(-1e-8 * block * t_rep), np.full(T, 1.0 / T)


def test_kernel_default_step_mask_is_identity():
    """Explicit step=ones / mask=all-true == the bare 6-arg kernel, bit-exact."""
    t_rep, lam, log_lam, r, pi = _kernel_grid()
    counts = np.random.default_rng(3).poisson(0.3, size=120).astype(np.int64)
    base = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi)
    got = smc.pair_posterior_mean_tmrca(
        counts, t_rep, lam, log_lam, r, pi,
        step=np.ones(len(counts) - 1), mask=np.ones(len(counts), dtype=bool),
    )
    np.testing.assert_array_equal(base, got)


def test_kernel_masked_block_emits_uniformly():
    """A non-callable block ignores its count, and an inaccessible run of
    count-0 blocks does not pull the TMRCA toward the present the way an
    accessible (monomorphic) run does."""
    t_rep, lam, log_lam, r, pi = _kernel_grid()
    n = 60
    c1 = np.full(n, 3, dtype=np.int64)
    c2 = c1.copy(); c2[30] = 99  # differ only at the masked block
    mask = np.ones(n, dtype=bool); mask[30] = False
    p1 = smc.pair_posterior_mean_tmrca(c1, t_rep, lam, log_lam, r, pi, mask=mask)
    p2 = smc.pair_posterior_mean_tmrca(c2, t_rep, lam, log_lam, r, pi, mask=mask)
    np.testing.assert_allclose(p1, p2)  # the masked count is irrelevant

    counts = np.full(n, 4, dtype=np.int64); counts[20:40] = 0
    accessible = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi)
    m2 = np.ones(n, dtype=bool); m2[20:40] = False
    masked = smc.pair_posterior_mean_tmrca(
        counts, t_rep, lam, log_lam, r, pi, mask=m2)
    # the masked gap keeps the deep flanking signal. The accessible zeros dip.
    assert masked[20:40].mean() > accessible[20:40].mean()


def test_kernel_large_step_approximates_segment_reset():
    """A huge inter-block step ≈ a full Δ→∞ reset at that boundary, matching an
    independent two-segment run."""
    nb, k = 24, 12
    t_rep, lam, log_lam, r, pi = _kernel_grid(block=1, T=32)
    counts = np.zeros(nb, dtype=np.int64); counts[:k] = 1  # deep then recent
    seg = np.concatenate([
        smc.pair_posterior_mean_tmrca(counts[:k], t_rep, lam, log_lam, r, pi),
        smc.pair_posterior_mean_tmrca(counts[k:], t_rep, lam, log_lam, r, pi),
    ])
    step = np.ones(nb - 1); step[k - 1] = 1e12  # ~unlinked boundary
    big = smc.pair_posterior_mean_tmrca(
        counts, t_rep, lam, log_lam, r, pi, step=step)
    np.testing.assert_allclose(big, seg, rtol=1e-3, atol=1.0)


# ---------------------------------- recombination map & accessibility wiring
def test_accessibility_block_fraction_reaches_the_emission_scale():
    """A block's callable base-pair fraction, and the emission scale it sets.

    Block 1 ([100, 200) bp) is half accessible and block 4 ([400, 500) bp) a
    fifth, so the Poisson mean of a block is the accessible share of the
    differences a fully accessible block would carry. A block a fifth callable
    still emits, and the mask is pinned as a literal: a raised callable floor
    drops partially callable blocks out of the likelihood, so they carry only
    recombination distance while the run still looks successful.
    """
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=1000, block_size=100,
        accessibility=[(150, 420), (800, 1000)],
    )
    frac = b._accessible_fraction(10)
    np.testing.assert_allclose(
        frac, [0.0, 0.5, 1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0])
    _step, mask, escale = b._block_geometry(10)
    np.testing.assert_array_equal(
        mask,
        [False, True, True, True, True, False, False, False, True, True])
    np.testing.assert_allclose(escale[mask], frac[mask])


def test_a_block_below_the_accessible_floor_does_not_emit():
    """A block below the callable floor emits uniformly.

    A block whose expected difference count is a small fraction of a full
    block's carries that fraction of the evidence, while the calibration
    divides its count by the fraction, so it would take on many times a full
    block's leverage on the time grid and on the per-pair prior mean. The two
    partially callable blocks are 2.5% and 10% callable, bracketing the floor:
    the first is silent, the second emits, and the emission scale tracks the
    callable fraction on either side.
    """
    bs = 1000
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=3 * bs, block_size=bs,
        accessibility=[(0.0, 25.0),  # 2.5% callable
                       (bs, bs + 100.0),  # 10% callable
                       (2 * bs, 3 * bs)],  # fully accessible
    )
    _step, mask, escale = b._block_geometry(3)
    np.testing.assert_array_equal(mask, [False, True, True])
    np.testing.assert_allclose(escale, [0.025, 0.1, 1.0])


def test_a_truncated_final_block_carries_the_exposure_its_span_affords():
    """A short final block emits at its span's exposure, or not at all.

    The emission mean is ``2*mu*B*t_i`` for every block, so a block spanning
    ``S_b`` of the block width ``B`` contributes ``S_b/B``. Two questions use
    that number differently. ``calibrate`` divides each block's difference
    count by its exposure, so a block far below the width carries ``B/S_b``
    times a full block's weight in the shared time grid and would dominate it;
    :data:`_ACCESS_FRAC_FLOOR` bounds that on the accessibility axis and on
    this one alike. Whether the base pairs a block does have are callable is
    the separate question the mask answers against its own span.
    """
    # 40 bp of a 1000 bp width is 0.04, below the floor: no emission.
    tiny = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=10040, block_size=1000,
        accessibility=[(0, 10040)],
    )
    np.testing.assert_allclose(tiny._accessible_fraction(11), np.ones(11))
    _step, mask, _escale = tiny._block_geometry(11)
    assert mask is not None and not mask[-1] and mask[:-1].all()

    # 400 bp of 1000 is 0.4, above the floor: emits at 0.4.
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=10400, block_size=1000,
        accessibility=[(0, 10400)],
    )
    _step, mask, escale = b._block_geometry(11)
    np.testing.assert_array_equal(mask, np.ones(11, dtype=bool))
    expected = np.ones(11)
    expected[-1] = 400 / 1000
    np.testing.assert_allclose(escale, expected)

    # With no mask the same region gives the same exposure.
    bare = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=10400, block_size=1000,
    )
    np.testing.assert_allclose(bare._block_geometry(11)[2], expected)

    # The final block holds 40 bp, of which 20 are accessible.
    half = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=10040, block_size=1000,
        accessibility=[(0, 10000), (10000, 10020)],
    )
    np.testing.assert_allclose(half._accessible_fraction(11)[-1], 0.5)

def _gap_intervals(length, gap=500.0, fraction=0.5, offset=250.0):
    """Accessible intervals leaving ``fraction`` of the bp in ``gap`` bp holes.

    The holes repeat every ``gap / fraction`` bp from ``offset``, a period
    that does not follow the HMM block grid, so each block is partly callable
    rather than wholly callable or wholly masked.
    """
    period = gap / fraction
    acc: list[tuple[float, float]] = []
    prev, x = 0.0, offset
    while x < length:
        if x > prev:
            acc.append((prev, x))
        prev = x + gap
        x += period
    if prev < length:
        acc.append((prev, float(length)))
    return acc


def test_sub_block_gaps_leave_the_posterior_tmrca_unbiased():
    """Callability holes finer than one block do not shift the TMRCAs.

    A hole emits no records, so its base pairs leave the pairwise difference
    count and the per-pair called fraction together. Scoring the block over
    its full width then reads the missing differences as a recent coalescence
    and shrinks every posterior mean by about ``1 / (1 - masked fraction)``.
    The exposure the Poisson mean is written over is the callable base-pair
    fraction of the block, which is asserted here against the same
    simulations run without any mask.
    """
    n_blocks, block_size, n_hap = 400, 2000, 6
    mu, rec, ne = 1.25e-8, 1e-8, 1e4
    length = n_blocks * block_size
    grid = np.geomspace(50.0, 3e6, 33)
    acc = _gap_intervals(length, fraction=0.5)
    names = [f"h{i}" for i in range(n_hap)]
    builder = LocalTreeBuilder(
        [], mu=mu, rec_rate=rec, sample_names=names, sequence_length=length,
        block_size=block_size, accessibility=acc)
    _step, mask, escale = builder._block_geometry(n_blocks)
    assert 0.4 < float(np.mean(escale)) < 0.6, (
        f"the mask is meant to leave about half of each block callable, got a "
        f"mean callable fraction of {float(np.mean(escale)):.3f}")

    def mean_tmrca(pos, gen, block_mask, emit_scale):
        hmm = PairwiseCoalescentHMM(
            n_hap, mu=mu, rec_rate=rec, block_size=block_size, time_grid=grid)
        t = hmm.block_tmrcas(gen, (pos // block_size).astype(np.int64),
                             n_blocks, block_mask=block_mask,
                             emit_scale=emit_scale)
        return float(t.mean())

    ratios, no_exposure_ratios = [], []
    for seed in (1, 2, 3):
        ts = _sim(samples=n_hap, length=length, mu=mu, rec=rec, N=ne, seed=seed)
        pos = np.array([v.site.position for v in ts.variants()])
        gen = np.array([v.genotypes for v in ts.variants()], dtype=np.int8)
        keep = np.zeros(pos.size, dtype=bool)
        for a0, a1 in acc:
            keep |= (pos >= a0) & (pos < a1)
        assert 0.3 < keep.mean() < 0.7, (
            f"the mask should drop about half the sites, it dropped "
            f"{1 - keep.mean():.2f}")
        full = mean_tmrca(pos, gen, None, None)
        ratios.append(mean_tmrca(pos[keep], gen[keep], mask, escale) / full)
        # Power check: with the callable fraction dropped from the exposure,
        # the same data reads far too recent.
        no_exposure_ratios.append(
            mean_tmrca(pos[keep], gen[keep], mask, None) / full)

    assert max(abs(r - 1.0) for r in ratios) < 0.15, (
        f"sub-block gaps moved the posterior mean TMRCA: per-seed ratios to "
        f"the unmasked run {np.round(ratios, 3).tolist()}")
    assert abs(float(np.mean(ratios)) - 1.0) < 0.08, (
        f"mean ratio to the unmasked run {float(np.mean(ratios)):.3f}")
    assert float(np.mean(no_exposure_ratios)) < 0.8, (
        f"the simulation cannot detect a dropped exposure term: it gives a "
        f"mean ratio of {float(np.mean(no_exposure_ratios)):.3f}")


def test_recombination_map_step_tracks_local_rate():
    """`_block_geometry` step reads the map's local rate in nominal-block units."""
    rm = msprime.RateMap(position=[0, 300, 600, 900], rate=[1e-8, 5e-8, 1e-8])
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=900, block_size=100, recombination_map=rm,
    )
    step, _, _ = b._block_geometry(9)
    assert step is not None and len(step) == 8
    assert step[3] == pytest.approx(5.0, rel=0.05)  # inside the 5× hotspot
    assert step[0] == pytest.approx(1.0, rel=0.05)  # 1× flank


def test_recombination_map_uniform_matches_constant():
    """A uniform RateMap at rate == rec_rate reproduces the no-map build exactly."""
    from ancestree import JC69
    a, names = toy_sites(range(0, 5000, 25), seed=6)
    L = 5000
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
              sequence_length=L, window=300, block_size=50, progress=False)
    base = list(LocalTreeInference(a, JC69(), **kw).infer())
    rm = msprime.RateMap(position=[0, L], rate=[1e-8])
    mapped = list(LocalTreeInference(a, JC69(), recombination_map=rm, **kw).infer())
    for (_, pb), (_, pm) in zip(base, mapped):
        np.testing.assert_allclose(pb.values, pm.values, rtol=1e-6, atol=1e-9)


def test_the_chunked_maps_reach_the_builder():
    """A non-uniform map changes the chunked result, and matches unchunked.

    ``_slice_map`` can be correct while nothing passes its output on: with
    both maps replaced by ``None`` the rest of the local-tree suite still
    passes. This pins the maps by their effect on the posteriors.
    """
    from ancestree import JC69

    a, names = toy_sites(range(0, 6000, 20), seed=9)
    rm = msprime.RateMap(position=[0, 2000, 4000, 6000], rate=[1e-9, 8e-8, 1e-9])
    mm = msprime.RateMap(position=[0, 3000, 6000], rate=[2e-9, 6e-8])

    def run(chunk_size, maps):
        inference = LocalTreeInference(
            a, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            window=400, block_size=50, progress=False,
            sequence_length=6000, chunk_size=chunk_size,
            recombination_map=rm if maps else None,
            mutation_map=mm if maps else None)
        return np.array([p.values for _, p in inference.infer()])

    with_maps = run(1500, True)
    without = run(1500, False)
    unchunked = run(None, True)

    assert with_maps.shape == without.shape == unchunked.shape
    # The maps must matter on the chunked path.
    assert not np.allclose(with_maps, without, atol=1e-9), (
        "the sliced maps never reached the builder")
    # Per-segment estimates differ from a single-segment run by up to ~0.2 in
    # probability here, so the two are compared on the call they make.
    agree = np.mean(with_maps.argmax(1) == unchunked.argmax(1))
    assert agree > 0.9, f"chunked and unchunked agree on only {agree:.2%}"


def test_chunked_recombination_map_slices_per_segment():
    """Each segment reads the rate its own stretch carries in the global map.

    A chunk size that leaves one segment exercises no slicing at all, so this
    runs several segments and compares each sliced map against the global one
    at the same genome position.
    """
    from ancestree import JC69
    a, names = toy_sites(range(0, 6000, 20), seed=9)
    rm = msprime.RateMap(position=[0, 2000, 4000, 6000], rate=[1e-8, 4e-8, 1e-8])
    inference = LocalTreeInference(
        a, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=400, block_size=50, progress=False, recombination_map=rm,
        sequence_length=6000, chunk_size=1500)
    inference._resolve_segmentation_params()
    n_segments = 0
    for work_unit in inference._stream_segments():
        _seg, _core_lo, _core_hi, origin, span = work_unit
        n_segments += 1
        sliced = inference._slice_map(rm, origin, span)
        assert sliced is not None
        for local in (1.0, min(span, 6000 - origin) - 1.0):
            if local < 0:
                continue
            assert sliced.get_rate(local) == rm.get_rate(origin + local), (
                origin, local)
    assert n_segments > 1, "expected a genuinely multi-segment run"


# ----------------------------------------------------- mutation map (emission)
def test_kernel_emit_scale_default_is_identity():
    """escale=ones reproduces the bare kernel bit-exactly, and a higher escale
    shifts the inferred TMRCA shallower (excess diffs read as mutation)."""
    t_rep, lam, log_lam, r, pi = _kernel_grid()
    counts = np.random.default_rng(5).poisson(0.4, size=100).astype(np.int64)
    base = smc.pair_posterior_mean_tmrca(counts, t_rep, lam, log_lam, r, pi)
    ident = smc.pair_posterior_mean_tmrca(
        counts, t_rep, lam, log_lam, r, pi,
        escale=np.ones(len(counts)))
    np.testing.assert_array_equal(base, ident)
    # double the local rate everywhere → same counts explained by 2× mutation,
    # so the posterior-mean TMRCA roughly halves.
    hot = smc.pair_posterior_mean_tmrca(
        counts, t_rep, lam, log_lam, r, pi,
        escale=np.full(len(counts), 2.0))
    assert hot.mean() < base.mean()


def test_mutation_map_emit_scale_tracks_local_rate():
    """`_block_geometry` emit_scale reads the map's local rate over μ."""
    mm = msprime.RateMap(position=[0, 300, 600, 900],
                         rate=[1.25e-8, 5e-8, 1.25e-8])
    b = LocalTreeBuilder(
        [], mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
        sequence_length=900, block_size=100, mutation_map=mm,
    )
    _, _, escale = b._block_geometry(9)
    assert escale is not None and len(escale) == 9
    assert escale[4] == pytest.approx(5e-8 / 1.25e-8, rel=0.05)  # hotspot block
    assert escale[0] == pytest.approx(1.0, rel=0.05)  # reference flank


def test_mutation_map_uniform_matches_constant():
    """A uniform RateMap at rate == mu reproduces the no-map build.

    Asserted on the PLUG-IN path, where it is a statement about the arithmetic
    and holds to machine precision. The ensemble cannot be held to it: a
    uniform map divides the counts by a scale that is 1.0 only to within 1e-14,
    and that perturbation is enough to move a drawn genealogy across a sampling
    boundary, which changes the posterior by far more than the perturbation.
    Requiring exactness there would only be satisfiable by having the ensemble
    ignore the map in its calibration, which is precisely the bug that let
    accessibility and mutation maps reach the emission but not the time grid.
    """
    from ancestree import JC69
    a, names = toy_sites(range(0, 5000, 25), seed=6)
    L = 5000
    kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names, n_ensemble=None,
              sequence_length=L, window=300, block_size=50, progress=False)
    base = list(LocalTreeInference(a, JC69(), **kw).infer())
    mm = msprime.RateMap(position=[0, L], rate=[1.25e-8])
    mapped = list(LocalTreeInference(a, JC69(), mutation_map=mm, **kw).infer())
    for (_, pb), (_, pm) in zip(base, mapped):
        np.testing.assert_allclose(pb.values, pm.values, rtol=1e-9, atol=1e-12)


def test_ensemble_applies_the_mutation_map_to_its_calibration():
    """The ensemble's time grid must see the map, not only the emission.

    Deriving the grid from raw counts leaves ``mutation_map`` and
    ``accessibility`` applying to the Poisson emission alone, which moves the
    grid's deep edge without moving the emission. Measured on the grid itself
    rather than by inspecting the source.
    """

    from ancestree.local_tree_inference import PairwiseCoalescentHMM

    n_blocks, T = 12, 16
    counts = np.tile(np.arange(1, n_blocks + 1, dtype=np.int64), (3, 1))
    hmm = PairwiseCoalescentHMM(n_haplotypes=3, mu=1.25e-8, rec_rate=1e-8,
                                block_size=2000, n_time_bins=T)

    flat = np.ones(n_blocks)
    contrast = np.full(n_blocks, 1.0)
    contrast[:4] = 5.0  # a 5x rate contrast over the first third

    *_, t_bar_flat = hmm.calibrate(counts, emit_scale=flat)
    *_, t_bar_scaled = hmm.calibrate(counts, emit_scale=contrast)

    # Blocks at 5x the reference rate carry proportionally fewer coalescent
    # units per difference, so the calibrated per-pair mean must drop.
    assert float(t_bar_scaled[0]) < float(t_bar_flat[0]), (
        f"the rate contrast did not reach the calibration: {t_bar_flat[0]} "
        f"against {t_bar_scaled[0]}")


def test_chunked_mutation_map_slices_per_segment():
    """Each segment reads the mutation rate its own stretch carries.

    Runs several segments, since a chunk size that leaves one exercises no
    slicing.
    """
    from ancestree import JC69
    a, names = toy_sites(range(0, 6000, 20), seed=9)
    mm = msprime.RateMap(position=[0, 2000, 4000, 6000],
                         rate=[1.25e-8, 4e-8, 1.25e-8])
    inference = LocalTreeInference(
        a, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        window=400, block_size=50, progress=False, mutation_map=mm,
        sequence_length=6000, chunk_size=1500)
    inference._resolve_segmentation_params()
    n_segments = 0
    for work_unit in inference._stream_segments():
        _seg, _core_lo, _core_hi, origin, span = work_unit
        n_segments += 1
        sliced = inference._slice_map(mm, origin, span)
        assert sliced is not None
        for local in (1.0, min(span, 6000 - origin) - 1.0):
            if local < 0:
                continue
            assert sliced.get_rate(local) == mm.get_rate(origin + local), (
                origin, local)
    assert n_segments > 1, "expected a genuinely multi-segment run"


# --------------------------------- end-to-end ground truth under perturbation
# The wiring tests above pin the kernel/builder plumbing for accessibility,
# missing data and multi-contig/hotspot maps in isolation. These three run the
# full genotype->tree->polarisation pipeline against msprime truth and assert
# the inferred ancestral state stays well above chance when the data is
# degraded the way real data is: masked-out gaps, missing genotype calls, and
# several contigs scanned together over a recombination-hotspot map. The bar is
# graceful degradation, not invariance, each perturbation is allowed to shave
# accuracy, just not collapse it. Baselines on these sims sit near 0.78, so a
# 0.62 floor leaves margin against seed noise while still catching a defect
# that breaks the perturbation path.
def _panel_accuracy(sites, truth, panel, seq_length, **kw):
    """Fraction of polymorphic sites whose MAP allele matches truth.

    ``truth`` is keyed by ``(chrom, pos)`` so it survives multiple contigs
    sharing a coordinate range.
    """
    inf = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=panel,
        sequence_length=seq_length, window="20kb", progress=False, **kw,
    )
    h = n = 0
    for site, post in inf.infer():
        t = truth.get((site.chrom, site.pos))
        if t is None:
            continue
        n += 1
        h += int(post.map_allele == t)
    return h / n, n


def _panel_sim_sites(samples=12, length=400_000, seed=7, chrom="1",
                     name_pool=None):
    """One panel's worth of (sites, truth, names), truth keyed by (chrom, pos).

    When ``name_pool`` is given, the panel's haplotypes are renamed onto it so
    independent sims can be concatenated as separate contigs of one sample set.
    """
    ts = _sim(samples=samples, length=length, seed=seed)
    raw = {ind.id: f"tsk_{ind.id}" for ind in ts.individuals()}
    names = name_pool if name_pool is not None else [raw[i] for i in range(samples)]
    remap = {raw[i]: names[i] for i in range(samples)}
    node_name = {int(ind.nodes[0]): remap[f"tsk_{ind.id}"]
                 for ind in ts.individuals()}
    panel = set(names)
    sites, truth = [], {}
    for v in ts.variants():
        anc = v.site.ancestral_state
        if anc not in STATES:
            continue
        ta = {node_name[int(node)]: v.alleles[v.genotypes[j]]
              for j, node in enumerate(ts.samples())}
        present = {a for s, a in ta.items() if s in panel and a in STATES}
        if len(present) < 2:
            continue
        sites.append(Site(chrom=chrom, pos=int(v.site.position),
                          alleles=canonical_alleles(v.alleles), tip_alleles=ta))
        truth[(chrom, int(v.site.position))] = anc
    return sites, truth, names, ts.sequence_length


def test_ground_truth_under_accessibility_mask():
    """Masking out the middle third of the genome degrades accuracy gracefully.

    The inaccessible gap drops those blocks' het signal from the pairwise HMM
    (they emit uniformly), so the inferred trees lose information there, yet
    sites are still polarised and accuracy holds well above chance.
    """
    sites, truth, names, L = _panel_sim_sites(samples=12, length=400_000, seed=7)
    base, _ = _panel_accuracy(sites, truth, names, L)
    masked, n = _panel_accuracy(
        sites, truth, names, L,
        accessibility=[(0, 0.35 * L), (0.65 * L, L)],
    )
    assert n > 100, f"too few sites to judge: {n}"
    assert masked > 0.62, f"accuracy collapsed under accessibility mask: {masked:.3f}"
    assert masked >= base - 0.15, (
        f"accessibility mask hurt too much: {base:.3f} -> {masked:.3f}"
    )


def test_ground_truth_under_missing_genotypes():
    """Randomly dropping ~20% of genotype calls degrades accuracy gracefully.

    Missing tips contribute an all-ones partial (no constraint) at their sites
    and are skipped in the pairwise het counts, so both the trees and the
    per-site polarisation lose information, accuracy dips but stays well above
    chance.
    """
    from dataclasses import replace
    sites, truth, names, L = _panel_sim_sites(samples=12, length=400_000, seed=7)
    base, _ = _panel_accuracy(sites, truth, names, L)
    rng = np.random.default_rng(0)
    holey = []
    for s in sites:
        ta = dict(s.tip_alleles)
        for k in list(ta):
            if rng.random() < 0.20:
                ta[k] = None
        holey.append(replace(s, tip_alleles=ta))
    miss, n = _panel_accuracy(holey, truth, names, L)
    assert n > 100, f"too few sites to judge: {n}"
    assert miss > 0.62, f"accuracy collapsed under missing genotypes: {miss:.3f}"
    assert miss >= base - 0.15, (
        f"missing genotypes hurt too much: {base:.3f} -> {miss:.3f}"
    )


def test_ground_truth_multicontig_with_hotspot_map():
    """Two contigs scanned together over a recombination-hotspot map.

    Independent sims become contigs ``"1"`` and ``"2"`` of one sample set, run
    in a single chunked call (``chunk_size`` engages the multi-contig path) with
    a 20x central hotspot recombination map. Per-contig truth is recovered well
    above chance, confirming the contigs stay decoupled and the hotspot map is
    sliced correctly per segment.
    """
    s1, t1, names, L = _panel_sim_sites(samples=12, length=400_000, seed=7,
                                        chrom="1")
    s2, t2, _, L2 = _panel_sim_sites(samples=12, length=400_000, seed=23,
                                     chrom="2", name_pool=names)
    assert L == L2
    sites = s1 + s2
    truth = {**t1, **t2}
    hotspot = msprime.RateMap(
        position=[0, 0.45 * L, 0.55 * L, L], rate=[1e-8, 2e-7, 1e-8],
    )
    acc, n = _panel_accuracy(
        sites, truth, names, L,
        chunk_size="1mb", recombination_map=hotspot,
    )
    assert n > 200, f"too few sites to judge: {n}"
    assert acc > 0.62, f"multi-contig + hotspot accuracy collapsed: {acc:.3f}"
    # per-contig: neither contig is dragged down by the other
    for chrom, tt in (("1", t1), ("2", t2)):
        inf = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=L, window="20kb", chunk_size="1mb",
            recombination_map=hotspot, progress=False,
        )
        h = m = 0
        for site, post in inf.infer():
            if site.chrom != chrom:
                continue
            g = tt.get((chrom, site.pos))
            if g is None:
                continue
            m += 1
            h += int(post.map_allele == g)
        assert h / m > 0.6, f"contig {chrom} accuracy too low: {h / m:.3f}"


def test_chunk_size_convergence_with_halo():
    """Interior calls converge to the unchunked build to a small bounded
    residual.

    Away from the artificial chunk cut, a sliced build with a generous halo
    tracks the unchunked one closely. (Exact agreement at *true* contig breaks
    is covered by ``test_chunked_multicontig_decoupling_exact``. The small
    residual here is per-chunk recalibration of the time grid / per-pair prior,
    not cross-chunk leakage, so it is bounded but not strictly monotone in the
    halo.)"""
    from ancestree import JC69
    ts = _sim(samples=8, length=60_000, seed=21)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)
    L = ts.sequence_length
    common = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
                  window="30snp", block_size=500, progress=False,
                  # chunking invariance is a property of the deterministic
                  # estimator. An ensemble adds Monte Carlo noise that would
                  # mask it, and is covered by its own tests.
                  n_ensemble=None)
    ref = {s.pos: p.values for s, p in LocalTreeInference(
        sites, JC69(), sequence_length=L, **common).infer()}

    cs = 25_000
    errs = []
    for s, p in LocalTreeInference(
            sites, JC69(), chunk_size=cs, halo=8000, **common).infer():
        d = s.pos % cs
        if min(d, cs - d) > 4000:  # interior: away from chunk cuts
            errs.append(float(np.abs(p.values - ref[s.pos]).max()))
    assert errs, "no interior sites sampled"
    assert float(np.mean(errs)) < 0.15  # interior calls ≈ unchunked

    # The residual is per-chunk recalibration, so it vanishes as the chunk
    # spans the whole region: a single full-length chunk reproduces the
    # unchunked build to a window-grid rounding tolerance (no recalibration
    # is left to drift).
    big = {s.pos: p.values for s, p in LocalTreeInference(
        sites, JC69(), chunk_size=int(L), halo=8000, **common).infer()}
    full = float(np.max([np.abs(big[pos] - ref[pos]).max() for pos in ref]))
    assert full < 0.03


def test_unmatched_sample_name_raises():
    """A ``sample_name`` absent from every site's ``tip_alleles`` is refused.

    It is the signature of a name mismatch (e.g. ``tskit.simplify``
    re-indexing samples). Read as an all-missing tip it would collapse that
    haplotype's inferred TMRCA, so the genotype build rejects it.
    """
    ts = _sim(samples=6, length=150_000, seed=11)
    names = [f"tsk_{i}" for i in range(6)]
    sites, _ = _sites_and_truth(ts, names)
    # Declare a sample no site knows about (mimics simplify's rename).
    bad_names = names[:-1] + ["tsk_renamed"]
    with pytest.raises(ValueError, match="no observed genotype"):
        inf = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=bad_names,
            sequence_length=ts.sequence_length, block_size=2000, progress=False,
        )
        list(inf.infer())


def test_hmm_recombination_uses_total_pair_branch_length(monkeypatch):
    """The no-recombination transition uses the same total
    pairwise branch length (2·t) as the Poisson mutation emission. The
    ``r`` array passed into the kernel must equal
    exp(-2·rec_rate·block_size·t_rep), not the single-lineage exp(-1·…)."""
    import ancestree.local_tree_inference as lti

    captured = {}

    def _spy(counts_k, t_rep, lam, log_lam, r, pi, **kw):
        captured["t_rep"] = np.asarray(t_rep).copy()
        captured["r"] = np.asarray(r).copy()
        # Return a benign per-block TMRCA (shape matches the counts slice).
        return np.zeros_like(counts_k, dtype=np.float64)

    monkeypatch.setattr(lti, "pair_posterior_mean_tmrca", _spy)

    rec_rate, block_size = 1e-8, 2000
    hmm = PairwiseCoalescentHMM(
        n_haplotypes=2, mu=1e-8, rec_rate=rec_rate, block_size=block_size,
        n_time_bins=16,
    )
    n_sites = 40
    rng = np.random.default_rng(0)
    genotypes = rng.integers(0, 2, size=(n_sites, 2)).astype(np.int8)
    block_of_site = (np.arange(n_sites) // 4).astype(np.int64)
    n_blocks = int(block_of_site.max()) + 1
    hmm.block_tmrcas(genotypes, block_of_site, n_blocks)

    expected = np.exp(-2.0 * rec_rate * block_size * captured["t_rep"])
    np.testing.assert_allclose(captured["r"], expected, rtol=1e-12)
    # Guard against the single-lineage convention.
    single = np.exp(-1.0 * rec_rate * block_size * captured["t_rep"])
    assert not np.allclose(captured["r"], single)


def test_hmm_recombination_scaling_matches_smc_ground_truth(monkeypatch):
    """Ground truth: the recombination rate acting on a pair is ``rho`` times
    the *total* pairwise branch length ``2T`` (a breakpoint can fall on either
    of the pair's two lineages back to the MRCA). Under msprime's SMC
    coalescent, the model this PSMC′-style HMM implements, where every
    recombination changes the marginal tree, the empirical per-bp TMRCA
    transition rate is therefore ``2*rho*E[T]``. Confirmed here: (a) that physical
    rate from msprime, and (b) that the HMM's no-recombination transition
    encodes the same ``2*rho*t`` hazard, i.e. a factor of 2, not 1.

    Run under the SMC (not Hudson / SMC′): there, recombinations that
    re-coalesce to the same time leave the TMRCA track unchanged, hiding events
    and pulling the measured ratio below 2, so only the SMC cleanly exposes
    the recombination *event* rate the HMM models.
    """
    Ne, rho, L = 1e4, 1e-8, 5_000_000

    # (a) msprime SMC ground truth: empirical TMRCA-transition rate / (rho*E[T]).
    ts = msprime.sim_ancestry(
        samples=2, ploidy=1, sequence_length=L, recombination_rate=rho,
        population_size=Ne, random_seed=4, discrete_genome=False, model="smc",
    )
    changes, integral_t, prev = 0, 0.0, None
    for tree in ts.trees():
        t = tree.tmrca(0, 1)
        integral_t += t * tree.span
        if prev is not None and t != prev:
            changes += 1
        prev = t
    mean_t = integral_t / L
    emp_ratio = (changes / L) / (rho * mean_t)
    # Robustly ~2 (see scratch sweep: mean 2.03, sd 0.08 over 10 seeds);
    # the window 1.6–2.4 excludes the single-lineage value of 1.
    assert 1.6 < emp_ratio < 2.4, emp_ratio

    # (b) The HMM's implied per-generation recombination hazard at TMRCA t is
    # -ln(r)/block_size = 2*rho*t. Its ratio to rho*t must match the ~2 above.
    import ancestree.local_tree_inference as lti
    captured = {}

    def _spy(counts_k, t_rep, lam, log_lam, r, pi, **kw):
        captured["t_rep"] = np.asarray(t_rep).copy()
        captured["r"] = np.asarray(r).copy()
        return np.zeros_like(counts_k, dtype=np.float64)

    monkeypatch.setattr(lti, "pair_posterior_mean_tmrca", _spy)
    block_size = 2000
    hmm = PairwiseCoalescentHMM(
        n_haplotypes=2, mu=1e-8, rec_rate=rho, block_size=block_size,
        n_time_bins=16,
    )
    rng = np.random.default_rng(1)
    G = rng.integers(0, 2, size=(40, 2)).astype(np.int8)
    block_of_site = (np.arange(40) // 4).astype(np.int64)
    hmm.block_tmrcas(G, block_of_site, int(block_of_site.max()) + 1)

    hmm_ratio = -np.log(captured["r"]) / (block_size * rho * captured["t_rep"])
    np.testing.assert_allclose(hmm_ratio, 2.0, rtol=1e-9)  # HMM uses 2T
    # The HMM's scaling agrees with the SMC ground-truth event rate.
    assert abs(float(hmm_ratio.mean()) - emp_ratio) < 0.4


def test_recombination_step_falls_back_to_reference_beyond_map_end():
    """Blocks beyond the recombination map's end fall back to
    the reference rate (step=1.0) rather than freezing the chain with step=0
    (r**0=1), mirroring the mutation-map branch."""
    from ancestree.local_tree_inference import LocalTreeBuilder
    b = LocalTreeBuilder.__new__(LocalTreeBuilder)
    b.block_size = 100
    b.rec_rate = 1e-8
    b.accessibility = None
    b.mutation_map = None
    b.mu = 1e-8
    b.sequence_length = 500.0
    # Map covers only [0, 250). Rate is 2x the reference so in-map steps == 2.0.
    b.recombination_map = msprime.RateMap(position=[0, 250], rate=[2e-8])
    step, _mask, _escale = b._block_geometry(n_blocks=5)
    # mids = [50,150,250,250,250] clipped to L=250 -> diff = [100,100,0,0].
    # In-map steps (k=0,1) carry the 2x rate. Off-map steps (k=2,3) fall back.
    np.testing.assert_allclose(step[:2], 2.0, rtol=1e-9)
    np.testing.assert_allclose(step[2:], 1.0, rtol=1e-9)  # was 0.0 before fix


def test_uniform_map_step_is_one_when_blocks_do_not_divide_the_sequence():
    """A uniform map gives step == 1.0 on every step, including
    the short one into a block truncated by the sequence end.

    The step's recombination mass must be divided by the block's actual
    width, not the nominal one, or the ragged final block reads as a cold
    spot that FFBS carries backwards through the whole chain.
    """
    from ancestree.local_tree_inference import LocalTreeBuilder
    b = LocalTreeBuilder.__new__(LocalTreeBuilder)
    b.block_size, b.rec_rate = 2824, 1e-8
    b.accessibility = b.mutation_map = None
    b.mu = 1e-8
    L = 60000.0  # 60000 / 2824 = 21.2 blocks: does NOT divide
    b.sequence_length = L
    b.recombination_map = msprime.RateMap(position=[0.0, L], rate=[b.rec_rate])
    n_blocks = int(np.ceil(L / b.block_size))
    step, _mask, _escale = b._block_geometry(n_blocks=n_blocks)
    assert len(step) == n_blocks - 1
    np.testing.assert_allclose(step, 1.0, rtol=1e-9)



def test_hmm_pair_prior_mean_excludes_masked_blocks(monkeypatch):
    """The per-pair coalescent prior mean (t_bar) averages over
    callable blocks only when an accessibility mask is set. Inaccessible blocks
    carry zero counts and would otherwise drag the prior toward the present."""

    captured = []

    hmm = PairwiseCoalescentHMM(
        n_haplotypes=2, mu=1e-8, rec_rate=1e-8, block_size=100, n_time_bins=16,
    )
    orig_prior = hmm._coalescent_prior

    def _spy(edges, t_bar):
        captured.append(t_bar)
        return orig_prior(edges, t_bar)

    monkeypatch.setattr(hmm, "_coalescent_prior", _spy)

    # 8 blocks of 100 bp. First 4 accessible & divergent (many diffs). Last 4
    # inaccessible with zero counts.
    n_blocks = 8
    bs = 100
    rng = np.random.default_rng(0)
    rows = []
    for b in range(n_blocks):
        pos0 = b * bs
        n_diff = 10 if b < 4 else 0
        for _ in range(n_diff):
            rows.append(pos0 + rng.integers(0, bs))
    sites = np.sort(np.array(rows, dtype=int))
    genotypes = np.zeros((len(sites), 2), dtype=np.int8)
    genotypes[:, 1] = 1  # every site differs between the pair
    block_of_site = (sites // bs).astype(np.int64)
    mask = np.array([True] * 4 + [False] * 4)

    hmm.block_tmrcas(genotypes, block_of_site, n_blocks, block_mask=mask)
    t_masked = captured[-1]

    captured.clear()
    hmm.block_tmrcas(genotypes, block_of_site, n_blocks, block_mask=None)
    t_unmasked = captured[-1]

    # Masking out the zero-count inaccessible tail must NOT pull the prior mean
    # down. The callable-only mean is strictly larger.
    assert t_masked > t_unmasked


# --------------------------------------------- candidate-neighbour restriction
def _candidate_names(ts):
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    return [nm[int(s)] for s in ts.samples()]


def test_one_segment_reproduces_the_unsegmented_build():
    """An input inside one ``chunk_size`` is segmented in name only.

    The segment's extent, and the block width the SNP density sets, follow
    the region's coordinates rather than the positions of the sites that
    happen to fall in it. Read off the data instead, the window grid and
    every block boundary shift, and the two paths disagree on roughly four
    per cent of the calls. Checked in both estimators, since the ensemble is
    the default.
    """
    ts = _sim(samples=16, length=1_000_000, seed=5)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)

    def run(chunk_size, n_ensemble):
        inference = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=float(ts.sequence_length), window="8snp",
            n_ensemble=n_ensemble, chunk_size=chunk_size, progress=False)
        return {int(s.pos): np.asarray(p.values) for s, p in inference.infer()}

    for n_ensemble in (None, 8):
        whole = run(None, n_ensemble)
        one_segment = run("10mb", n_ensemble)   # far exceeds the 1 Mb input
        assert whole.keys() == one_segment.keys() and whole
        for pos, values in whole.items():
            np.testing.assert_array_equal(
                values, one_segment[pos],
                err_msg=f"segmented build diverged at {pos} "
                        f"(n_ensemble={n_ensemble})")


def test_a_chunk_outside_the_mask_scores_rather_than_raises():
    """A segment the mask excludes is uninformative, not an error.

    The slicer hands such a segment an empty interval list, which the
    builder read as a mask that marks nothing callable and refused. The
    same data and mask built as one region succeed, so the chunked path
    dropped a legitimate input, and skipping the segment instead would
    silently lose its sites.
    """
    names = [f"h{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    sites = [
        Site(chrom="1", pos=pos, alleles=("A", "T"),
             tip_alleles={n: ("A" if rng.random() < 0.7 else "T")
                          for n in names})
        for pos in range(0, 20_000, 50)
    ]
    # The middle chunk falls wholly between the two accessible intervals.
    accessible = [(0, 6000), (14000, 20000)]
    common = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names, window=400,
                  block_size=100, accessibility=accessible, progress=False,
                  n_ensemble=None)

    whole = list(LocalTreeInference(
        sites, JC69(), sequence_length=20_000.0, chunk_size=None,
        **common).infer())
    chunked = list(LocalTreeInference(
        sites, JC69(), chunk_size=4000, halo=500, **common).infer())

    assert len(whole) == len(sites)
    assert [s.pos for s, _ in chunked] == [s.pos for s, _ in whole]


def test_a_whole_input_outside_the_mask_still_raises():
    """An unsegmented build whose mask excludes everything is a mistake."""
    names = [f"h{i}" for i in range(4)]
    sites = [Site(chrom="1", pos=pos, alleles=("A", "T"),
                  tip_alleles={n: "A" for n in names})
             for pos in range(0, 2000, 50)]
    inference = LocalTreeInference(
        sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
        sequence_length=2000.0, window=400, block_size=100,
        accessibility=[(50_000, 60_000)], chunk_size=None, progress=False)
    with pytest.raises(ValueError, match="no block over this region is callable"):
        list(inference.infer())


def test_segments_read_their_calls_on_one_grid():
    """Every segment reads on the grid laid over the whole region.

    Tiled per segment, the window grid follows that segment's own length and
    site density, and a halo that padded the span moved it again, so the same
    position fell in a different window depending on where the chunk
    boundaries landed. Sites were then scored on different trees genome-wide,
    not merely near the joins.

    Agreement is measured on the posteriors rather than on MAP calls: a flip
    count only reports sites whose top two alleles were near-tied, so it hides
    both the size of a disagreement and its absence.
    """
    ts = _sim(samples=16, length=1_000_000, seed=5)
    nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
    names = [nm[int(s)] for s in ts.samples()]
    sites, _ = _sites_and_truth(ts, names)

    def run(chunk_size, halo=None):
        extra = {} if halo is None else {"halo": halo}
        inference = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=float(ts.sequence_length), window="8snp",
            n_ensemble=None, chunk_size=chunk_size, progress=False, **extra)
        return {int(s.pos): np.asarray(p.values) for s, p in inference.infer()}

    whole = run(None)
    for halo in (1_000, 25_000):
        segmented = run("250kb", halo)
        shared = sorted(set(whole) & set(segmented))
        assert len(shared) == len(whole)
        gap = np.array([np.abs(whole[p] - segmented[p]).max() for p in shared])
        # The typical site agrees to several decimals. The upper tail is not
        # pinned: how much of it the shared grid removes depends on the panel,
        # from most of it on a sparse one to a third of the mean on this
        # denser fixture, and the rest is the segmented forward-backward.
        assert np.median(gap) < 1e-4, f"median gap {np.median(gap):.2e}"


class TestStreamingAlongTheGenome:
    """Both streams iterate the genome, whatever the input was chunked into.

    The chunked and unchunked paths differ only in how many groups arrive,
    not in what a group contains or what the genome-wide form looks like, so a
    caller need not know which path ran.
    """

    @staticmethod
    def _panel():
        ts = _sim(samples=12, length=1_000_000, seed=5)
        nm = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
        names = [nm[int(s)] for s in ts.samples()]
        sites, _ = _sites_and_truth(ts, names)
        return sites, names, float(ts.sequence_length)

    def _inference(self, chunk_size, n_ensemble=None):
        sites, names, length = self._panel()
        return LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=length, window="8snp", n_ensemble=n_ensemble,
            chunk_size=chunk_size, progress=False)

    def test_the_glued_tmrcas_do_not_depend_on_the_chunk_size(self):
        """One region, one segment and several give the same matrix shape."""
        reference = self._inference(None).pairwise_tmrcas()
        for chunk_size in ("10mb", "250kb", "100kb"):
            glued = self._inference(chunk_size).pairwise_tmrcas()
            assert glued.pairs == reference.pairs
            assert glued.sample_names == reference.sample_names
            assert glued.tmrca.shape == reference.tmrca.shape, chunk_size
            assert np.all(np.diff(glued.block_midpoints) > 0), chunk_size
            assert np.all(np.isfinite(glued.tmrca)) and np.all(glued.tmrca > 0)
            np.testing.assert_allclose(
                glued.block_midpoints, reference.block_midpoints, rtol=1e-9)
            # Per-segment estimates legitimately differ from a whole-region
            # run (median 3-5% here), so the values are pinned by their
            # agreement in genomic order rather than elementwise: a segment
            # glued back to front measures 0.00 against the reference's 0.99.
            r = np.corrcoef(np.log(glued.tmrca.ravel()),
                            np.log(reference.tmrca.ravel()))[0, 1]
            assert r > 0.9, f"chunk_size={chunk_size}, correlation {r:.4f}"

    def test_the_tmrca_stream_covers_the_genome_once(self):
        """Segments tile the blocks: none repeated from a halo, none lost."""
        for chunk_size in (None, "10mb", "250kb"):
            parts = list(self._inference(chunk_size).iter_pairwise_tmrcas())
            assert parts
            mids = np.concatenate([p.block_midpoints for p in parts])
            assert len(np.unique(mids)) == len(mids), chunk_size
            glued = self._inference(chunk_size).pairwise_tmrcas()
            assert len(mids) == glued.tmrca.shape[1], chunk_size

    def test_the_ensemble_stream_yields_every_draw_for_each_stretch(self):
        """All draws arrive together, a stretch at a time."""
        for chunk_size, expected_groups in ((None, 1), ("10mb", 1), ("250kb", 4)):
            groups = [(iv, list(m)) for iv, m in self._inference(
                chunk_size, n_ensemble=8).to_tree_sequence()]
            assert len(groups) == expected_groups, chunk_size
            assert {len(members) for _, members in groups} == {8}, chunk_size

    def test_the_ensemble_stream_tiles_the_region(self):
        """The stretches cover the input once, in order, without overlap."""
        groups = list(self._inference(
            "250kb", n_ensemble=4).to_tree_sequence())
        bounds = [interval for interval, _ in groups]
        assert bounds == sorted(bounds)
        for (_, prev_hi), (next_lo, _) in zip(bounds, bounds[1:]):
            assert next_lo == prev_hi, bounds
        assert bounds[0][0] == 0.0

    def test_streamed_genealogies_sit_on_the_genome_axis(self):
        """A segment's trees are returned where they belong, and only those.

        A segment is built on its own axis starting at zero and reaches past
        the stretch it emits by the halo. Returned as built, every segment's
        trees claimed to start at position zero and to cover ground the next
        segment covered too, so a caller unioning them would double-count.
        """
        groups = [(iv, list(m)) for iv, m in self._inference(
            "250kb", n_ensemble=2).to_tree_sequence()]
        assert len(groups) > 1
        for (lo, hi), members in groups:
            for member in members:
                covered = [t.interval for t in member.trees() if t.num_edges]
                assert covered, f"stretch [{lo}, {hi}) came back empty"
                assert covered[0].left == pytest.approx(lo), (lo, hi)
                assert covered[-1].right == pytest.approx(hi), (lo, hi)

    def test_plug_in_mode_streams_a_single_genealogy_per_stretch(self):
        """Without an ensemble each stretch carries one tree sequence."""
        groups = [(iv, list(m)) for iv, m in self._inference(
            "250kb", n_ensemble=None).to_tree_sequence()]
        assert {len(members) for _, members in groups} == {1}


class TestAnExplicitHaloIsFlooredAtTheBlockWidth:
    """A halo narrower than the block must not fork the block grid.

    ``_resolve_segmentation_params`` floored the halo at ``block_size`` only on
    the ``"auto"`` branch, so an explicit narrower value left a segment's
    truncated final block with its midpoint inside the core. That block was
    emitted, and the next segment emitted the full-width block covering the
    same span, so ``block_midpoints`` stopped being a global grid and one span
    carried two different TMRCA estimates. ``block_size`` is resolved from the
    data's SNP density, so a caller cannot know in advance what to clear.
    """

    MU = 1.25e-8
    LENGTH = 300_000

    @pytest.fixture(scope="class")
    def sites(self):
        ts = msprime.sim_ancestry(
            samples=6, ploidy=1, population_size=1e4,
            sequence_length=self.LENGTH, recombination_rate=1e-8,
            random_seed=11,
        )
        ts = msprime.sim_mutations(ts, rate=self.MU, random_seed=11)
        names = [str(i) for i in range(ts.num_samples)]
        return names, [
            Site(chrom="1", pos=int(v.site.position), alleles=canonical_alleles(v.alleles),
                 tip_alleles={names[i]: v.alleles[g]
                              for i, g in enumerate(v.genotypes)})
            for v in ts.variants()
        ]

    def _midpoints(self, sites, halo):
        """Block midpoints and the resolved block width for one halo."""
        names, records = sites
        inf = LocalTreeInference(
            records, JC69(), mu=self.MU, rec_rate=1e-8, sample_names=names,
            sequence_length=self.LENGTH, chunk_size=50_000, halo=halo,
            n_ensemble=None, progress=False,
        )
        return np.asarray(inf.pairwise_tmrcas().block_midpoints), inf.block_size

    def test_a_narrow_halo_gives_the_same_grid_as_auto(self, sites):
        narrow, block = self._midpoints(sites, 500)
        auto, _ = self._midpoints(sites, "auto")
        assert 500 < block, "fixture no longer exercises a sub-block halo"
        np.testing.assert_allclose(narrow, auto)

    def test_a_narrow_halo_emits_each_block_once(self, sites):
        midpoints, _ = self._midpoints(sites, 500)
        assert len(np.unique(np.round(midpoints, 3))) == len(midpoints)

    def test_raising_the_halo_is_reported(self, sites, caplog):
        with caplog.at_level(logging.WARNING):
            self._midpoints(sites, 500)
        assert any("narrower than the resolved block width" in r.message
                   for r in caplog.records)


class TestInferSiteAgreesWithInferInLocalTreeMode:
    """The single-site entry point must answer as the batch path does.

    ``LocalTreeInference`` subclasses ``Inference`` rather than
    ``ARGBasedInference`` and inherited neither the rate scaling nor the focal
    resolution, so branch lengths in generations reached the kernel unscaled
    and every transition saturated. At the default focal it raised. At
    ``focal="panel_root"`` it silently returned a flat posterior.
    """

    MU = 1e-8

    @pytest.fixture(scope="class")
    def panel(self):
        ts = msprime.sim_mutations(
            msprime.sim_ancestry(
                samples=8, ploidy=1, population_size=1e4,
                sequence_length=50_000, recombination_rate=1e-8,
                random_seed=4),
            rate=self.MU, random_seed=4)
        names = [str(i) for i in range(ts.num_samples)]
        sites = [
            Site(chrom="1", pos=int(v.site.position), alleles=canonical_alleles(v.alleles),
                 tip_alleles={names[i]: v.alleles[g]
                              for i, g in enumerate(v.genotypes)})
            for v in ts.variants()
        ]
        return names, sites

    def _inference(self, panel):
        names, sites = panel
        return LocalTreeInference(
            sites, JC69(), mu=self.MU, rec_rate=1e-8, sample_names=names,
            sequence_length=50_000, n_ensemble=None, progress=False)

    def test_the_rate_reaches_the_supplied_tree(self, panel):
        """Guard the equality below against passing on an unscaled tree."""
        from ancestree.trees import TskitLocalTree

        names, sites = panel
        inference = self._inference(panel)
        list(inference.infer())
        tsq = inference.point_tree_sequence()
        tree = TskitLocalTree(tsq, tsq.sequence_length / 2,
                              sample_map={n: i for i, n in enumerate(names)})
        assert tree.time_scale == 1.0
        inference.infer_site(tree, sites[0])
        assert tree.time_scale == pytest.approx(self.MU)

    def test_infer_site_matches_infer(self, panel):
        from ancestree.trees import TskitLocalTree

        names, sites = panel
        inference = self._inference(panel)
        batch = {int(s.pos): p.values for s, p in inference.infer()}
        tsq = inference.point_tree_sequence()
        tree = TskitLocalTree(tsq, tsq.sequence_length / 2,
                              sample_map={n: i for i, n in enumerate(names)})
        compared = 0
        for site in sites:
            expected = batch.get(int(site.pos))
            interval = tree._tree.interval
            if expected is None or not interval.left <= site.pos < interval.right:
                continue
            np.testing.assert_allclose(
                inference.infer_site(tree, site).values, expected, atol=1e-12)
            compared += 1
        assert compared


SAMPLES = 8
INGROUP = tuple(f"s{i}" for i in range(6))
OUTGROUP = ("s6", "s7")
SAMPLE_MAP = {f"s{i}": i for i in range(SAMPLES)}


class TestSegmentDiagnosticsCountCoreSitesOnly:
    """A segment emits its core rows, so it reports on its core rows.

    Halos overlap, so counters accumulated over core plus halo would tally
    each halo site once per segment covering it.
    """

    @staticmethod
    def _reported(caplog, vcf, **kwargs):
        """Sites emitted, and the ingroup-monomorphic total the run reports."""
        import logging

        caplog.clear()
        with caplog.at_level(logging.INFO, logger="ancestree"):
            inf = anc.LocalTreeInference(
                vcf, JC69(), mu=2e-7, rec_rate=1e-8, sequence_length=100_000,
                ingroup_samples=INGROUP, outgroup_samples=OUTGROUP, **kwargs)
            n = sum(1 for _ in inf.infer())
        tallies = [int(m.group(1).replace(",", ""))
                   for r in caplog.records
                   for m in [re.search(r"([\d,]+) site\(s\) are monomorphic",
                                       r.getMessage())] if m]
        assert tallies, "the run reported no monomorphic tally"
        return n, tallies[-1]

    def test_every_chunking_reports_the_same_tally(self, tmp_path, caplog):
        ts = msprime.sim_ancestry(samples=SAMPLES, sequence_length=100_000,
                                  ploidy=1, recombination_rate=1e-8,
                                  population_size=1e4, random_seed=11)
        ts = msprime.sim_mutations(ts, rate=2e-7, random_seed=11)
        vcf = tmp_path / "halo.vcf"
        with open(vcf, "w") as fh:
            ts.write_vcf(fh, individual_names=list(SAMPLE_MAP),
                         position_transform="legacy")
        plain = self._reported(caplog, str(vcf), chunk_size=None, n_ensemble=None)
        wide = self._reported(caplog, str(vcf), chunk_size=20_000, halo=15_000,
                              n_ensemble=None)
        narrow = self._reported(caplog, str(vcf), chunk_size=20_000, halo=2_000,
                                n_ensemble=None)
        assert plain[0] == wide[0] == narrow[0], "site counts differ"
        assert plain[1] <= plain[0], (
            f"unchunked reports {plain[1]} monomorphic of {plain[0]} emitted")
        assert plain[1] == wide[1] == narrow[1], (
            f"the tally depends on the halo: unchunked {plain[1]}, "
            f"halo=15000 {wide[1]}, halo=2000 {narrow[1]}")


class TestUnrepresentableAllelesReachTheLocalTreeSummary:
    """Alleles outside A/C/G/T are reported by the local-tree modes too.

    A segment worker reports its diagnostics as a fixed-width tuple, so a
    counter the tuple omits reads as zero however many such alleles the panel
    carries.
    """

    _REPORTED = re.compile(
        r"([\d,]+) site\(s\) carry an allele outside .*? ([\d,]+) tip\(s\)",
        re.S)

    @classmethod
    def _reported(cls, caplog, inference):
        """Run a walk and read the totals it reports.

        :return: ``(n_emitted, n_unrepresentable_sites,
            n_unrepresentable_tips)``.
        """
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            n = sum(1 for _ in inference.infer())
        hits = [m for r in caplog.records
                for m in [cls._REPORTED.search(r.getMessage())] if m]
        if not hits:
            return n, 0, 0
        assert len(hits) == 1, f"{len(hits)} summaries for one walk"
        return (n, int(hits[0].group(1).replace(",", "")),
                int(hits[0].group(2).replace(",", "")))

    @staticmethod
    def _ts():
        """Four samples on one tree with three sites.

        Position 1 is a readable ``A``/``T``. Position 3 puts one ``AT`` tip on
        a readable site. Position 5 carries ``AT`` / ``ATT`` alone.
        """
        tables = tskit.TableCollection(sequence_length=10.0)
        samples = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                   for _ in range(4)]
        root = tables.nodes.add_row(flags=0, time=1.0)
        for child in samples:
            tables.edges.add_row(left=0, right=10.0, parent=root, child=child)
        readable = tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.mutations.add_row(site=readable, node=samples[0],
                                 derived_state="T")
        mixed = tables.sites.add_row(position=3.0, ancestral_state="A")
        tables.mutations.add_row(site=mixed, node=samples[0],
                                 derived_state="AT")
        opaque = tables.sites.add_row(position=5.0, ancestral_state="AT")
        tables.mutations.add_row(site=opaque, node=samples[0],
                                 derived_state="ATT")
        tables.sort()
        return tables.tree_sequence()

    @staticmethod
    def _panel(n_sites=120, step=50):
        """Biallelic haplotypes with an indel allele at every tenth site."""
        sites, names = toy_sites(range(1000, 1000 + n_sites * step, step),
                                 seed=5)
        for k in range(0, len(sites), 10):
            site = sites[k]
            tips = dict(site.tip_alleles)
            tips[names[0]] = "AT"
            sites[k] = Site(chrom=site.chrom, pos=site.pos,
                            alleles=tuple(site.alleles) + ("AT",),
                            tip_alleles=tips)
        return sites, names

    def test_a_pre_built_genealogy_reports_what_arg_mode_reports(self, caplog):
        ts = self._ts()
        arg = self._reported(
            caplog, ARGBasedInference(ts, JC69(), mu=1e-8, progress=False))
        local = self._reported(
            caplog, LocalTreeInference(ts, JC69(), mu=1e-8, progress=False))
        assert arg == (3, 2, 5), "fixture preconditions"
        assert local == arg

    def test_a_chunked_run_reports_what_an_unchunked_one_reports(self, caplog):
        sites, names = self._panel()
        kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
                  window=500, block_size=100, progress=False)
        single = self._reported(caplog, LocalTreeInference(
            sites, JC69(), chunk_size=10 ** 9, **kw))
        chunked = self._reported(caplog, LocalTreeInference(
            sites, JC69(), chunk_size=1000, halo=500, **kw))
        assert single == (len(sites), 12, 12), "fixture preconditions"
        assert chunked == single

    def test_the_single_build_path_reports_the_sites(self, caplog):
        """The unchunked path reports the tips the chunked one reports.

        The single build encodes the panel into a columnar table, whose
        genotypes hold an allele outside the alphabet as a no-call, so a tally
        taken after that encoding reads zero tips for a panel whose chunked
        run reports one per affected site.
        """
        sites, names = self._panel()
        kw = dict(mu=1.25e-8, rec_rate=1e-8, sample_names=names,
                  window=500, block_size=100, progress=False)
        built = self._reported(caplog, LocalTreeInference(
            sites, JC69(), sequence_length=10_000, chunk_size=None, **kw))
        chunked = self._reported(caplog, LocalTreeInference(
            sites, JC69(), chunk_size=1000, halo=500, **kw))
        assert built[:2] == (len(sites), 12), "fixture preconditions"
        assert built == chunked
