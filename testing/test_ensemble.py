"""Ensemble sampling in local-tree mode.

Covers the two paths that reach the sampler (single-region and segmented),
reproducibility, the non-uniform-map support, and that the plug-in estimate is
recovered exactly at ``n_ensemble=1``.
"""
import itertools
import os
import pathlib
import tempfile

import msprime
import numpy as np
import tskit
import pytest
import ancestree as anc

import ancestree._ensemble as ensemble_module
from ancestree import JC69, LocalTreeBuilder, LocalTreeInference
from ancestree._ensemble import SegmentEnsemble, _pair_stream_keys
from ancestree.sites import Site


#: Panels committed rather than simulated at import. msprime reproduces a seed
#: only within one platform: at 1.4.2 the same seed gives arm64 macOS and
#: x86-64 Linux the same site count and the same first nineteen sites, then
#: different mutation positions and genotypes, which moves every value the
#: golden assertions pin. See :func:`_sites`.
FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "ensemble"


def _sites(seed=7, n=8, L=60_000):
    """A small panel with enough SNPs to tile several windows.

    :param seed: Selects the committed panel, not a live simulation.
    :param n: Haplotype count the panel was built with.
    :param L: Sequence length the panel was built with.
    :return: ``(sites, names, sequence_length)``.
    """
    ts = tskit.load(str(FIXTURES / f"ensemble_panel_s{seed}.trees"))
    assert ts.num_samples == n and ts.sequence_length == L, (
        f"the committed panel for seed {seed} is {ts.num_samples} haplotypes "
        f"over {ts.sequence_length:g} bp, not {n} over {L:g}")
    names = [f"s{i}" for i in range(n)]
    out = []
    for v in ts.variants():
        al = v.alleles
        if len(al) != 2 or any(a not in "ACGT" for a in al if a):
            continue
        out.append(Site(chrom="1", pos=int(v.site.position), alleles=tuple(al),
                        tip_alleles={names[i]: al[g]
                                     for i, g in enumerate(v.genotypes)}))
    return out, names, float(ts.sequence_length)


def _run(sites, names, L, **kw):
    kw.setdefault("sequence_length", L)
    inf = LocalTreeInference(sites, JC69(), mu=2.5e-8, rec_rate=1e-8,
                             sample_names=names, window="8snp",
                             progress=False, **kw)
    return {int(s.pos): np.asarray(p.values) for s, p in inf.infer()}


def test_none_is_the_plug_in_estimator():
    """None draws no genealogies: the posterior-mean tree, deterministically."""
    sites, names, L = _sites()
    a = _run(sites, names, L, n_ensemble=None)
    b = _run(sites, names, L, n_ensemble=None)
    assert a.keys() == b.keys()
    assert all(np.array_equal(a[p], b[p]) for p in a)


def test_size_one_is_a_draw_not_the_mean():
    """1 is a genuine sample size, so it must differ from the plug-in mean."""
    sites, names, L = _sites()
    plug = _run(sites, names, L, n_ensemble=None)
    one = _run(sites, names, L, n_ensemble=1)
    assert plug.keys() == one.keys()
    assert max(np.abs(plug[p] - one[p]).max() for p in plug) > 1e-9


def test_ensemble_changes_the_posterior():
    """Sampling genealogies must actually move the answer."""
    sites, names, L = _sites()
    point = _run(sites, names, L, n_ensemble=None)
    ens = _run(sites, names, L, n_ensemble=8)
    assert point.keys() == ens.keys()
    assert max(np.abs(point[p] - ens[p]).max() for p in point) > 1e-6


def test_reproducible_under_a_fixed_seed():
    sites, names, L = _sites()
    a = _run(sites, names, L, n_ensemble=8, ensemble_seed=3)
    b = _run(sites, names, L, n_ensemble=8, ensemble_seed=3)
    assert all(np.array_equal(a[p], b[p]) for p in a)


def test_seed_changes_the_draws():
    sites, names, L = _sites()
    a = _run(sites, names, L, n_ensemble=8, ensemble_seed=1)
    b = _run(sites, names, L, n_ensemble=8, ensemble_seed=2)
    assert max(np.abs(a[p] - b[p]).max() for p in a) > 1e-9


def _last_block_quantiles(ens, n_members, seed):
    """Mid-CDF quantile of each member's last-block bin, centred per pair.

    The last block is drawn first, from the filtered distribution there, so
    under independent streams two such arrays are uncorrelated.

    :return: ``(n_members, n_pairs)`` quantiles minus their per-pair mean.
    """
    path = np.empty((n_members, ens.alpha_ck.shape[0], ens.n_blocks), np.int8)
    ens._paths_into(path, n_members, 0, seed)
    a = ens.alpha_ck[:, -1]
    cum = np.cumsum(a, axis=1) / a.sum(axis=1, keepdims=True)
    lower = np.concatenate([np.zeros((len(a), 1)), cum[:, :-1]], axis=1)
    mid = (lower + cum) / 2
    q = np.take_along_axis(mid, path[:, :, -1].T.astype(np.int64), axis=1).T
    return q - q.mean(axis=0)


def _stream_correlation(x, y):
    """Pearson correlation of two centred quantile arrays, and its 4 SE bound."""
    r = float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    return r, 4.0 / np.sqrt(x.size)


def _segment_ensembles(monkeypatch, sites, names, L, **kw):
    """The :class:`SegmentEnsemble` instances a chunked run scores with."""
    seen = []

    class Spy(SegmentEnsemble):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen.append(self)

    monkeypatch.setattr(ensemble_module, "SegmentEnsemble", Spy)
    _run(sites, names, L, n_ensemble=2, chunk_size="20kb", halo="5kb", **kw)
    return seen


def test_segments_draw_from_independent_streams(monkeypatch):
    """Two segments of one contig share no random numbers.

    Keyed on the member and pair alone, every segment replayed the same
    uniforms, so the last-block quantiles of two segments correlated at 0.98.
    """
    sites, names, L = _sites()
    ens = _segment_ensembles(monkeypatch, sites, names, L)
    assert len(ens) >= 2
    r, tol = _stream_correlation(_last_block_quantiles(ens[0], 128, 0),
                                 _last_block_quantiles(ens[1], 128, 0))
    assert abs(r) < tol, (r, tol)


def test_contigs_draw_from_independent_streams():
    """The same panel on two contigs gives two different ensembles."""
    sites, names, L = _sites()
    twin = [Site(chrom="2", pos=s.pos, alleles=s.alleles,
                 tip_alleles=s.tip_alleles) for s in sites]
    inf = _inference(sites + twin, names, L, n_ensemble=8, chunk_size="20kb",
                     halo="5kb")
    post = {(s.chrom, int(s.pos)): np.asarray(p.values)
            for s, p in inf.infer()}
    assert max(np.abs(post[("1", s.pos)] - post[("2", s.pos)]).max()
               for s in sites) > 1e-6


def test_consecutive_seeds_share_no_members(monkeypatch):
    """Member ``b`` at seed ``s`` and member ``b - 1`` at ``s + 1`` differ.

    With member streams at ``seed + b`` the two were the same draw, so
    replicate runs at consecutive seeds shared all but one member.
    """
    sites, names, L = _sites()
    ens = _segment_ensembles(monkeypatch, sites, names, L)[0]
    q0 = _last_block_quantiles(ens, 129, 0)
    q1 = _last_block_quantiles(ens, 129, 1)
    r, tol = _stream_correlation(q0[1:], q1[:-1])
    assert abs(r) < tol, (r, tol)


def test_posteriors_are_normalised():
    sites, names, L = _sites()
    for size in (None, 1, 8):
        post = _run(sites, names, L, n_ensemble=size)
        tot = np.array([v.sum() for v in post.values()])
        assert np.allclose(tot, 1.0)
        assert all((v >= 0).all() for v in post.values())


def test_segmented_path_reaches_the_sampler():
    """chunk_size routes through _process_segment. It must ensemble too."""
    sites, names, L = _sites()
    point = _run(sites, names, L, n_ensemble=None, chunk_size=int(L))
    ens = _run(sites, names, L, n_ensemble=8, chunk_size=int(L))
    assert point.keys() == ens.keys()
    assert max(np.abs(point[p] - ens[p]).max() for p in point) > 1e-6


def test_non_uniform_maps_are_supported():
    """A rate map must not be rejected, and must change the result."""
    sites, names, L = _sites()
    flat = msprime.RateMap(position=[0.0, L], rate=[1e-8])
    hot = msprime.RateMap(position=[0.0, L / 2, L], rate=[1e-9, 5e-8])
    a = _run(sites, names, L, n_ensemble=8, recombination_map=flat)
    b = _run(sites, names, L, n_ensemble=8, recombination_map=hot)
    assert a.keys() == b.keys()
    assert max(np.abs(a[p] - b[p]).max() for p in a) > 1e-9


def test_uniform_map_matches_no_map():
    """A uniform map is the constant-rate case and must agree with it."""
    sites, names, L = _sites()
    flat = msprime.RateMap(position=[0.0, L], rate=[1e-8])
    a = _run(sites, names, L, n_ensemble=8)
    b = _run(sites, names, L, n_ensemble=8, recombination_map=flat)
    assert max(np.abs(a[p] - b[p]).max() for p in a) < 1e-9


def test_rejects_a_non_positive_size():
    sites, names, L = _sites()
    with pytest.raises(ValueError, match="positive sample size"):
        _run(sites, names, L, n_ensemble=0)


# ---------------------------------------------- materialising the ensemble

def _inference(sites, names, L, **kw):
    kw.setdefault("sequence_length", L)
    return LocalTreeInference(sites, JC69(), mu=2.5e-8, rec_rate=1e-8,
                              sample_names=names, window="8snp",
                              progress=False, **kw)


def _topology(ts):
    """A hashable signature of a tree sequence's topology and node times."""
    return (ts.tables.nodes.time.round(6).tobytes(),
            ts.tables.edges.child.tobytes(),
            ts.tables.edges.parent.tobytes())


def test_tree_sequences_yields_one_without_an_ensemble():
    """Ensemble off: the iterator carries exactly the plug-in tree."""
    sites, names, L = _sites()
    inf = _inference(sites, names, L, n_ensemble=None)
    got = list(inf.tree_sequences())
    assert len(got) == 1
    (lo, hi), members = got[0]
    members = list(members)
    assert (lo, hi) == (0.0, float(L))
    assert len(members) == 1
    assert _topology(members[0]) == _topology(inf.point_tree_sequence())


def test_tree_sequences_yields_every_member():
    """Ensemble on: one tree sequence per member, none of them the plug-in."""
    sites, names, L = _sites()
    inf = _inference(sites, names, L, n_ensemble=8)
    groups = list(inf.tree_sequences())
    assert len(groups) == 1
    members = list(groups[0][1])
    assert len(members) == 8
    # The drawn genealogies are not the point estimate, and differ from
    # each other, otherwise the ensemble would be marginalising over one
    # repeated tree.
    point = _topology(inf.point_tree_sequence())
    assert all(_topology(m) != point for m in members)
    assert len({_topology(m) for m in members}) > 1


def test_materialised_members_are_reproducible():
    """Fixing ensemble_seed fixes the drawn genealogies."""
    sites, names, L = _sites()
    def topologies(seed):
        return [_topology(m)
                for _interval, members in
                _inference(sites, names, L, n_ensemble=4,
                           ensemble_seed=seed).tree_sequences()
                for m in list(members)]

    a, b, c = topologies(3), topologies(3), topologies(4)
    assert a == b
    assert a != c


def test_member_chunk_does_not_change_the_posterior():
    """Chunking bounds memory. It must not move the answer beyond rounding.

    It reorders the running log-sum-exp, so this is a tolerance, not equality.
    """
    sites, names, L = _sites()
    a = _run(sites, names, L, n_ensemble=8, member_chunk=8)
    b = _run(sites, names, L, n_ensemble=8, member_chunk=2)
    assert max(np.abs(a[p] - b[p]).max() for p in a) < 1e-12


def test_a_member_chunk_that_does_not_divide_is_reduced():
    """An ensemble size the chunk does not divide must still run.

    The chunk is a memory bound, and draws stream from ``seed + start``, so
    reducing it to a divisor cannot reach the result. Refusing instead aborted
    the run after the forward pass had already been paid for.
    """
    sites, names, L = _sites()
    post = _run(sites, names, L, n_ensemble=10, member_chunk=4)
    assert post, "no posteriors returned"
    assert all(np.all(np.isfinite(v)) for v in post.values())


def test_chunked_ensemble_materialises_a_stretch_at_a_time():
    """The chunked path yields its draws per segment rather than refusing."""
    sites, names, L = _sites()
    inf = _inference(sites, names, L, n_ensemble=8, chunk_size="20kb",
                     sequence_length=None)
    groups = [(iv, list(members)) for iv, members in inf.tree_sequences()]
    assert len(groups) > 1, "expected more than one segment at 20kb chunks"
    for (lo, hi), members in groups:
        assert hi > lo
        assert len(members) == 8
    # The stretches tile the genome in order and never overlap.
    bounds = [iv for iv, _ in groups]
    assert bounds == sorted(bounds)
    assert all(a[1] <= b[0] for a, b in zip(bounds, bounds[1:]))
    # The plug-in stitch is still available there.
    assert inf.point_tree_sequence().num_trees > 0


def test_builder_write_round_trips():
    """LocalTreeBuilder.write() dumps a loadable .trees (it is not the
    inference class, and does not have point_tree_sequence)."""
    import tskit
    from ancestree import LocalTreeBuilder
    sites, names, L = _sites()
    b = LocalTreeBuilder(sites, mu=2.5e-8, rec_rate=1e-8, sample_names=names,
                         sequence_length=L)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.trees")
        b.write(p)
        assert tskit.load(p).num_samples == len(names)


def test_base_composition_reaches_the_ensemble_posterior():
    """Ensemble mode must honour ``base_composition``.

    It feeds pi into Q and supplies the root prior, so a supplied composition
    must move the ensemble posterior as it moves the plug-in one. Invisible
    under JC69/K2, whose pi is uniform.
    """
    from ancestree import HKY
    from ancestree.sites import BaseComposition
    sites, names, L = _sites(seed=11, n=8, L=120_000)
    a_rich = BaseComposition(counts={"A": 7000, "C": 1000, "G": 1000, "T": 1000})
    t_rich = BaseComposition(counts={"A": 1000, "C": 1000, "G": 1000, "T": 7000})

    def run(bc):
        inf = LocalTreeInference(
            sites, HKY(kappa=3.0), mu=2.5e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=float(L), window="4snp", base_composition=bc,
            n_ensemble=8, progress=False)
        return np.array([[p[b] for b in "ACGT"] for _s, p in inf.infer()])

    assert np.abs(run(a_rich) - run(t_rich)).max() > 1e-3


def test_checkpoint_stride_engages_only_above_the_budget():
    """The HMM forward matrix is retained whole until it would be too large.

    Checkpointing is not free -- the backward pass runs once per member chunk,
    so its replay is paid per chunk (+30% measured on a 128-member cell), not
    once. Below the budget ``stride == 1`` retains every block and the replay
    rebuilds nothing, which is the un-checkpointed cost through one code path.
    """
    from ancestree._ensemble import MAX_HMM_ALPHA_BYTES, _ckpt_stride
    # The retained checkpoints are float64, so the budget is 8 bytes an entry.
    small = 40 * 39 // 2  # 40 haplotypes: 143 MB, fits
    assert _ckpt_stride(714, small, 32) == 1
    big = 174 * 173 // 2  # 174 haplotypes: 2.8 GB, does not
    stride = _ckpt_stride(714, big, 32)
    assert stride > 1
    n_ckpt = int(np.ceil(714 / stride))
    assert big * n_ckpt * 32 * 8 < MAX_HMM_ALPHA_BYTES


class TestEnsembleKernelInvariants:
    """Three guarantees the ensemble kernels state and nothing pinned.

    ``ffbs_paths`` keys each member's stream on its global index, ``upgma_batch``
    reimplements scipy's average linkage in numba, and ``hmm_forward_ckpt``
    claims a checkpoint holds bit-for-bit what the full forward pass would.
    All three hold. None was checked, and the checkpointed branch only engages
    above the memory budget, which no test reaches.
    """

    P, K, T = 3, 12, 8

    @classmethod
    def _hmm_inputs(cls):
        """``lam`` / ``log_lam`` / ``r`` are per block, ``pi`` per pair."""
        rng = np.random.default_rng(4)
        counts = rng.integers(0, 4, size=(cls.P, cls.K)).astype(np.int16)
        base_lam = np.linspace(0.05, 1.5, cls.T)
        lam = np.repeat(base_lam[None, :], cls.K, axis=0)
        log_lam = np.log(lam)
        r = np.repeat(np.linspace(0.9, 0.2, cls.T)[None, :], cls.K, axis=0)
        pi = np.full((cls.P, cls.T), 1.0 / cls.T)
        called = np.ones((cls.P, cls.K), dtype=np.int16)
        block_sites = np.ones(cls.K, dtype=np.int64)
        return counts, lam, log_lam, r, pi, called, block_sites

    def _draw(self, n_members, seed0, stride):
        from ancestree._ensemble import ffbs_paths, hmm_forward_ckpt

        counts, lam, log_lam, r, pi, called, block_sites = self._hmm_inputs()
        n_ckpt = int(np.ceil(self.K / stride))
        alpha_ck = np.empty((self.P, n_ckpt, self.T), dtype=np.float64)
        hmm_forward_ckpt(counts, lam, log_lam, r, pi, called, block_sites,
                         stride, alpha_ck)
        path = np.empty((n_members, self.P, self.K), dtype=np.int8)
        # Unnamed panel: the kernel keys each pair's stream by index, which
        # is what these invariants are stated over.
        pair_key = np.arange(self.P, dtype=np.uint64)
        ffbs_paths(counts, lam, log_lam, alpha_ck, stride, self.K,
                   r, pi, called, block_sites, np.uint64(seed0), 0, pair_key,
                   path)
        return path

    def test_a_smaller_ensemble_is_a_prefix_of_a_larger_one(self):
        big = self._draw(5, seed0=7, stride=1)
        for size in (2, 3):
            np.testing.assert_array_equal(self._draw(size, 7, 1), big[:size])

    def test_the_checkpoint_stride_does_not_move_the_draws(self):
        reference = self._draw(4, seed0=7, stride=1)
        for stride in (2, 3, 4):
            np.testing.assert_array_equal(
                self._draw(4, 7, stride), reference)

    def test_equal_exposure_weights_give_the_plain_linkage(self):
        from ancestree._ensemble import upgma_batch

        rng = np.random.default_rng(12)
        for n in (5, 8, 12):
            condensed = (rng.random(n * (n - 1) // 2) + 0.1)[None, :]
            plain = np.empty((1, n - 1, 4))
            weighted = np.empty((1, n - 1, 4))
            upgma_batch(condensed.copy(), n, plain, np.empty((0, condensed.size)))
            upgma_batch(condensed.copy(), n, weighted, np.full((1, condensed.size), 0.37))
            np.testing.assert_allclose(weighted, plain, rtol=1e-12)

    def test_a_haplotype_without_exposure_leaves_the_others_tree_unchanged(self):
        """The called haplotypes merge at the heights they merge at without it.

        With the plain average, the random distances of a haplotype uncalled
        over a window entered every cluster mean it took part in, so it moved
        the haplotypes that do have data. With every deeper outgroup uncalled
        over a stretch, a rare ingroup allele then received tens of percent of
        ancestral probability against a simulated rate below one in a thousand.
        """
        from itertools import combinations

        from ancestree._ensemble import _WEIGHT_FLOOR, upgma_batch

        rng = np.random.default_rng(13)
        n = 9
        pairs = list(combinations(range(n), 2))
        condensed = rng.random(len(pairs)) + 0.1
        blind = np.array([n - 1 in pair for pair in pairs])
        weights = np.where(blind, _WEIGHT_FLOOR, 1.0)[None, :]
        # A draw that would make it the first merge under the plain average.
        condensed[np.flatnonzero(blind)[0]] = 1e-3
        full = np.empty((1, n - 1, 4))
        upgma_batch(condensed[None, :].copy(), n, full, weights)
        called = np.empty((1, n - 2, 4))
        upgma_batch(condensed[None, ~blind].copy(), n - 1, called,
                    np.empty((0, int((~blind).sum()))))
        heights = np.sort(full[0, :, 2])
        for h in called[0, :, 2]:
            assert np.isclose(heights, h, rtol=1e-6).any(), (
                f"merge height {h} of the called haplotypes is absent from "
                f"the tree that includes the uncalled one: {heights}")

    def test_the_numba_upgma_matches_scipy(self):
        from scipy.cluster.hierarchy import linkage

        from ancestree._ensemble import upgma_batch

        rng = np.random.default_rng(11)
        for n in (5, 8, 12):
            condensed = rng.random(n * (n - 1) // 2) + 0.1
            out = np.empty((1, n - 1, 4), dtype=np.float64)
            upgma_batch(condensed[None, :].copy(), n, out,
                        np.empty((0, condensed.size)))
            want = linkage(condensed.copy(), method="average")
            # The whole linkage, not only column 2: two different topologies
            # can merge at the same multiset of heights, so sorted heights
            # alone leave the structure unpinned. Columns are
            # (child_a, child_b, height, cluster_size), and each row's two
            # children are order-independent.
            np.testing.assert_allclose(
                np.sort(out[0][:, :2], axis=1),
                np.sort(want[:, :2], axis=1), rtol=0, atol=0)
            np.testing.assert_allclose(
                out[0][:, 2], want[:, 2], rtol=0, atol=1e-10)
            np.testing.assert_allclose(
                out[0][:, 3], want[:, 3], rtol=0, atol=0)


#: Ensemble posteriors at the first four sites, ``n_ensemble=8``,
#: ``ensemble_seed=0``, from the fixture built by :func:`_sites`.
_GOLDEN_ENSEMBLE = {
    405: (0.34757032814482136, 2.393030473429052e-05, 0.6523818112457099, 2.393030473429052e-05),
    1061: (2.393030473429052e-05, 0.34757032814482164, 2.393030473429052e-05, 0.6523818112457098),
    1976: (0.13835192908031924, 1.3160654847827645e-05, 0.8616217496099852, 1.3160654847827691e-05),
    1984: (0.8558404746350377, 1.9305255162784716e-05, 1.9305255162784716e-05, 0.14412091485463677),
}


def test_ensemble_posterior_matches_the_reference_values():
    """The marginalised posterior is pinned to values, not only to relations.

    Every other ensemble assertion holds under any monotone reweighting of
    the members, so the member weighting and the branch scale are only
    constrained here.
    """
    sites, names, L = _sites()
    post = _run(sites, names, L, n_ensemble=8, ensemble_seed=0)
    for pos, want in _GOLDEN_ENSEMBLE.items():
        np.testing.assert_allclose(
            post[pos], want, rtol=1e-9, atol=1e-12,
            err_msg=f"ensemble posterior moved at position {pos}")


N_SITES, N_HAP = 400, 6


SEQUENCE_LENGTH = 40_000.0


SAMPLES = [f"h{i}" for i in range(N_HAP)]


def _sites_with_missing(missing_fraction=0.0, seed=5):
    """A panel with a controllable fraction of uncalled genotypes."""
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(int(SEQUENCE_LENGTH), N_SITES,
                                   replace=False))
    out = []
    for pos in positions:
        alleles = ("A", "C")
        tips = {}
        for name in SAMPLES:
            if rng.random() < missing_fraction:
                continue                      # uncalled: absent from tip_alleles
            tips[name] = alleles[int(rng.random() < 0.35)]
        if len(tips) < 2:
            continue
        out.append(Site(chrom="1", pos=int(pos), alleles=alleles,
                        tip_alleles=tips))
    return out


def _posteriors(sites, model):
    inference = anc.Inference.from_local_tree(
        sites, model=model, mu=2.5e-8, rec_rate=1e-8, sample_names=SAMPLES,
        sequence_length=SEQUENCE_LENGTH, window="8snp", n_ensemble=16,
        ensemble_seed=3, progress=False,
        chunk_size=None)
    return {s.pos: np.asarray(p.values) for s, p in inference.infer()}


#: Ensemble posteriors on a panel with 35% uncalled genotypes under JC69.
#: Pins the per-block called-fraction exposure in the ensemble's
#: own forward pass.
_GOLDEN_MISSING = {
    36: (0.22698099192213617, 0.7723311203870444, 0.0003439438454096634, 0.0003439438454096639),
    40: (0.9998755891704991, 4.147027650029285e-05, 4.147027650029287e-05, 4.147027650029287e-05),
    47: (0.9372296710681701, 0.06260012290458843, 8.510301362076841e-05, 8.510301362076841e-05),
    109: (0.7290978093181899, 0.2698542749899058, 0.0005239578459520828, 0.0005239578459520834),
}


def test_the_ensemble_honours_the_per_block_exposure():
    """Pinned values on a panel with uncalled genotypes.

    A pair called over part of a block has had proportionally less
    opportunity to accumulate differences, so the block's Poisson exposure is
    scaled by the called fraction, and a block spanning less than the block
    width by that span as a fraction of it. A cell below
    :data:`_ACCESS_FRAC_FLOOR` leaves the shared time grid, since the
    calibration divides by that fraction. The values are pinned because no
    relational assertion constrains those scales.
    """
    post = _posteriors(_sites_with_missing(missing_fraction=0.35), anc.JC69())
    for pos, want in _GOLDEN_MISSING.items():
        np.testing.assert_allclose(
            post[pos], want, rtol=1e-9, atol=1e-12,
            err_msg=f"ensemble posterior moved at position {pos}")


def test_halving_the_exposure_doubles_the_inferred_tmrca():
    """The per-block called fraction scales the Poisson exposure.

    A pair called over half of a block has had half the opportunity to
    accumulate differences there, so the same difference count implies twice
    the coalescence time. Holding the count fixed and varying only how much of
    the block is called isolates that scale: without it the two runs agree.
    """
    from ancestree.local_tree_inference import PairwiseCoalescentHMM

    mu, block_size, n_blocks, n_hap = 1.25e-8, 4000, 40, 4
    per_block = 40
    block_of_site = np.repeat(np.arange(n_blocks), per_block)

    def genotypes(called):
        g = np.zeros((n_blocks * per_block, n_hap), dtype=np.int8)
        for k in range(n_blocks):
            lo = k * per_block
            g[lo:lo + 2, 1] = 1                    # two differences, pair (0, 1)
            g[lo + called:lo + per_block, 0] = -1   # the rest uncalled
            g[lo + called:lo + per_block, 1] = -1
        return g

    hmm = PairwiseCoalescentHMM(n_hap, mu=mu, rec_rate=1e-8,
                                block_size=block_size, n_time_bins=32)
    full = hmm.block_tmrcas(genotypes(per_block), block_of_site, n_blocks)
    half = hmm.block_tmrcas(genotypes(per_block // 2), block_of_site, n_blocks)
    ratio = float(half[0].mean() / full[0].mean())
    assert 1.7 < ratio < 2.3, (
        f"halving the called fraction changed the inferred TMRCA by "
        f"{ratio:.3f}x, expected about 2; the per-block exposure scale is "
        f"not reaching the emission")


class TestSegmentEnsemble:
    @staticmethod
    def _builder():
        sites, names, L = _sites()
        return LocalTreeBuilder(sites, mu=2.5e-8, rec_rate=1e-8,
                                sample_names=names, sequence_length=L), names

    def test_an_unnamed_panel_keys_its_streams_by_pair_index(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        keys = _pair_stream_keys(None, pairs)
        assert keys.dtype == np.uint64
        assert keys.tolist() == [0, 1, 2]
        named = _pair_stream_keys(["s0", "s1", "s2"], pairs)
        assert len(set(named.tolist())) == 3 and named.tolist() != [0, 1, 2]

    def test_a_pair_below_the_grid_floor_draws_inside_the_lowest_bin(self):
        """The within-bin draw of a pair with ``t_bar`` below the grid floor
        follows the floored prior the chain was built on.

        A haplotype duplicated in the panel has ``t_bar = 0`` with its copy.
        The chain prior floors that at the lowest edge, but the draw used the
        raw mean, an ``Exp`` of mean zero, and so put every lowest-bin draw
        exactly on the lowest edge.
        """
        sites, names, L = _sites()
        sites = [Site(chrom=s.chrom, pos=s.pos, alleles=s.alleles,
                      tip_alleles={**s.tip_alleles, "dup": s.tip_alleles["s0"]})
                 for s in sites]
        names = names + ["dup"]
        builder = LocalTreeBuilder(sites, mu=2.5e-8, rec_rate=1e-8,
                                   sample_names=names, sequence_length=L)
        ens = SegmentEnsemble(builder, JC69(), len(names), None)
        pair = list(itertools.combinations(range(len(names)), 2)).index(
            (0, len(names) - 1))
        draws = np.concatenate([c[:, :, pair].copy() for c in ens.draw_chunks(
            64, member_chunk=16, seed=1)])
        lo, hi = ens.edge_lo[0], ens.edge_hi[0]
        bin0 = draws[draws < hi]
        assert bin0.size > draws.size // 2
        assert np.all(bin0 > lo), "lowest-bin draws sit on the lowest edge"
        # A window mean below hi is a run of lowest-bin draws, each from
        # Exp(mean lo) truncated to [lo, hi), of mean 2 lo - w / expm1(w / lo)
        # with w = hi - lo.
        w = hi - lo
        assert bin0.mean() > 0.9 * (2 * lo - w / np.expm1(w / lo))

    def test_a_named_ingroup_marks_only_those_haplotypes(self):
        builder, names = self._builder()
        ens = SegmentEnsemble(builder, JC69(), len(names), [0, 2])
        assert ens.is_ingroup.tolist() == [i in (0, 2) for i in range(len(names))]

    def test_an_empty_ingroup_is_refused(self):
        builder, names = self._builder()
        with pytest.raises(ValueError, match="the ingroup is empty"):
            SegmentEnsemble(builder, JC69(), len(names), [])

    def test_a_model_without_a_real_eigendecomposition_is_refused(self):
        class _NoEig(JC69):
            def real_eig(self, *, pi=None):
                return None

        builder, names = self._builder()
        ens = SegmentEnsemble(builder, _NoEig(), len(names), None)
        with pytest.raises(ValueError, match="no real eigendecomposition"):
            ens.posterior(0.0, 2)

    def test_the_path_budget_shrinks_the_chunk_to_a_divisor(self, monkeypatch):
        builder, names = self._builder()
        ens = SegmentEnsemble(builder, JC69(), len(names), None)
        unbounded = ens.posterior(0.0, 4, member_chunk=4, seed=1)
        assert ens._path_chunk(4, 4) == 4

        per_member = ens.alpha_ck.shape[0] * ens.n_blocks
        monkeypatch.setattr(ensemble_module, "MAX_PATH_BYTES", 3 * per_member)
        assert ens._path_chunk(4, 4) == 2
        assert ens._path_chunk(4, 9) == 3
        bounded = ens.posterior(0.0, 4, member_chunk=4, seed=1)
        np.testing.assert_allclose(bounded, unbounded, rtol=1e-12, atol=1e-15)
