"""Pairwise TMRCA under the truth, the plug-in estimate and the ensemble.

Local-tree mode reduces the pairwise-coalescent HMM to a genealogy in one of
two ways. The plug-in path takes the posterior mean

.. math::

    \\hat{t}_{pw} \;=\; \\sum_{k \\in w} \\frac{1}{|w|}
        \\sum_{i} \\gamma_{ki}(p)\; t_i,

and the ensemble path draws :math:`B` joint paths by forward-filtering
backward-sampling and reads each one off the same grid,

.. math::

    t^{(b)}_{pw} \;=\; \\sum_{k \\in w} \\frac{1}{|w|}\; t_{x^{(b)}_{pk}},
    \\qquad x^{(b)}_{p\\cdot} \\sim p(x_{p\\cdot} \\mid y_{p\\cdot}).

Here :math:`p` indexes the haplotype pairs, :math:`w` a window, :math:`k` the
blocks it spans, :math:`|w|` their number, :math:`i` a bin of the discretised
time grid, :math:`t_i` that bin's representative TMRCA in generations,
:math:`\\gamma_{ki}(p)` the HMM's posterior probability that pair :math:`p`
occupies bin :math:`i` over block :math:`k`, :math:`x^{(b)}_{pk}` the bin the
:math:`b`-th draw assigns there, and :math:`y_{p\\cdot}` the per-block counts of
differences between the pair.

The two differ in shape, not only in noise: averaging over :math:`\\gamma`
before agglomerating pulls every window toward the middle of the grid, while a
draw lands on one bin. This writes all three distributions, over the same
windows and the same pairs, against the simulated truth
:math:`t_{pw} = \\sum_{\\tau} (|\\tau \\cap w| / |w|)\\, t_{p\\tau}`, the true
pairwise TMRCA averaged over the local trees :math:`\\tau` a window spans,
weighted by the span each covers.

One chunk per job. Nothing per-window is retained: the marginal histograms and
the joint (truth, estimate) histograms are the only accumulators, so the cost
of pooling the whole simulation is flat in memory and the axes are shared
across chunks by construction rather than by agreement.
"""
import json
from itertools import combinations

import numpy as np
import tskit

import ancestree as anc
from ancestree.models import JC69

try:  # snakemake execution
    TREES = snakemake.input.trees
    OUT_NPZ = snakemake.output.npz
    MU = snakemake.params.mu
    REC_RATE = snakemake.params.rec_rate
    WINDOW = snakemake.params.window
    SEGMENT = snakemake.params.segment
    N_MEMBERS = int(snakemake.params.n_members)
    MEMBER_CHUNK = int(snakemake.params.member_chunk)
    SEED = int(snakemake.params.seed)
    FLOOR_SCALE = float(getattr(snakemake.params, "floor_scale", 1.0))
except NameError:  # direct execution
    TREES = "results/data/baseline10_chunk0.trees"
    OUT_NPZ = "results/data/tmrca_distributions_baseline_chunk0.npz"
    MU = 1.25e-8
    REC_RATE = 1e-8
    WINDOW = "8snp"
    #: Segment length the HMM is run over. Bounds the forward matrix and the
    #: truth pass; the chunks are longer than this, so most carry several.
    SEGMENT = "2mb"
    #: Ensemble size B.
    N_MEMBERS = 128
    MEMBER_CHUNK = 8
    SEED = 1000
    #: How much deeper than calibration to put the grid floor. The coalescent
    #: prior folds everything below the floor into the lowest bin, so a floor
    #: at calibration's own value piles that tail up there. 1.0 leaves the
    #: calibrated grid alone.
    FLOOR_SCALE = 10.0

#: The shared TMRCA axis, in generations. Fixed rather than derived from the
#: data so shards taken from different chunks are summable: a per-chunk axis
#: calibrated from that chunk's own grid would put every shard on its own bins.
LO, HI = 1e1, 1e7
#: Marginal-histogram resolution, in bins per decade. The plot rebins from
#: here, so this only has to be finer than anything the figure wants.
PER_DECADE = 100
#: Resolution of the joint (true, estimated) histogram the calibration panel
#: reads its per-band quantiles from.
JOINT_TRUE_PER_DECADE = 20
JOINT_EST_PER_DECADE = 40


def _axis(per_decade):
    """:return: Log-spaced bin edges over ``[LO, HI]`` at that resolution."""
    return np.geomspace(LO, HI, int(round(np.log10(HI / LO) * per_decade)) + 1)


def _ingroup(ts):
    """:return: The ingroup sample nodes, in ``ts.samples()`` order."""
    name = {p.id: (p.metadata or {}).get("name") for p in ts.populations()}
    return [int(n) for n in ts.samples()
            if name[ts.node(int(n)).population] == "ingroup"]


def _deepen_floor(builder, scale):
    """The segment's own calibrated grid, with its floor divided by ``scale``.

    Calibration is per segment and stays that way: only the floor moves, and
    bins are added rather than widened, so the step is the calibrated one.

    :param builder: The segment's :class:`LocalTreeBuilder`.
    :param scale: Factor to lower the floor by. ``1.0`` returns ``None``.
    :return: Bin edges in generations, or ``None`` to leave calibration alone.
    """
    if scale == 1.0:
        return None
    from ancestree.local_tree_inference import PairwiseCoalescentHMM
    g, _pos, block_of_site, n_blocks = builder._genotype_matrix()
    hmm = PairwiseCoalescentHMM(
        g.shape[1], mu=builder.mu, rec_rate=builder.rec_rate,
        block_size=builder.block_size, n_time_bins=builder.n_time_bins)
    counts, _called = hmm._pair_block_counts(g, block_of_site, n_blocks)
    edges, _ = hmm._calibrate_time_grid(counts)
    step = np.log10(edges[1] / edges[0])
    # Calibration puts the floor at the median over pairs of each pair's mean
    # crude TMRCA divided by 200, clamped at 10 generations. The clamp holds
    # for the lowered floor too.
    lo = max(edges[0] / scale, 10.0)
    n = int(round(np.log10(edges[-1] / lo) / step))
    return np.geomspace(lo, edges[-1], n + 1)


def _true_window_tmrca(ts, node_pairs, edges):
    """Span-weighted true pairwise TMRCA per window.

    :param ts: The panel tree sequence.
    :param node_pairs: ``(node_a, node_b)`` per pair, in kernel order.
    :param edges: ``(n_windows + 1,)`` window boundaries in base pairs.
    :return: ``(n_windows, n_pairs)`` generations, ``nan`` where a window is
        covered by no tree.
    """
    n_win = len(edges) - 1
    acc = np.zeros((n_win, len(node_pairs)))
    span = np.zeros(n_win)
    for tree in ts.trees():
        left, right = tree.interval
        if right <= edges[0] or left >= edges[-1]:
            continue
        w0 = max(int(np.searchsorted(edges, left, side="right")) - 1, 0)
        w1 = min(int(np.searchsorted(edges, right, side="left")) - 1,
                 n_win - 1)
        if w1 < w0:
            continue
        t = np.array([tree.tmrca(a, b) for a, b in node_pairs])
        for w in range(w0, w1 + 1):
            overlap = min(right, edges[w + 1]) - max(left, edges[w])
            if overlap > 0.0:
                acc[w] += overlap * t
                span[w] += overlap
    out = np.full_like(acc, np.nan)
    covered = span > 0.0
    out[covered] = acc[covered] / span[covered, None]
    return out


class Accumulator:
    """The three marginals, the two joints, and their log moments."""

    def __init__(self):
        self.edges = _axis(PER_DECADE)
        self.true_edges = _axis(JOINT_TRUE_PER_DECADE)
        self.est_edges = _axis(JOINT_EST_PER_DECADE)
        self.hist = {k: np.zeros(len(self.edges) - 1, dtype=np.int64)
                     for k in ("truth", "point", "draws")}
        self.joint = {k: np.zeros((len(self.true_edges) - 1,
                                   len(self.est_edges) - 1), dtype=np.int64)
                      for k in ("point", "draw_mean")}
        # (count, sum log10 t, sum (log10 t)^2), so the pooled mean and spread
        # are exact rather than read back off the binned axis.
        self.moments = {k: np.zeros(3) for k in ("truth", "point", "draws")}
        self.t_rep = []
        self.n_windows = 0
        self.n_pairs = 0

    def add_marginal(self, key, values):
        """Bin ``values`` (generations) into the ``key`` marginal."""
        v = values[np.isfinite(values) & (values > 0.0)].ravel()
        self.hist[key] += np.histogram(v, bins=self.edges)[0]
        lg = np.log10(v)
        self.moments[key] += (lg.size, lg.sum(), (lg ** 2).sum())

    def add_joint(self, key, truth, estimate):
        """Bin matched ``(truth, estimate)`` pairs, both in generations."""
        ok = (np.isfinite(truth) & (truth > 0.0)
              & np.isfinite(estimate) & (estimate > 0.0))
        self.joint[key] += np.histogram2d(
            truth[ok].ravel(), estimate[ok].ravel(),
            bins=[self.true_edges, self.est_edges])[0].astype(np.int64)

    def save(self, path, meta):
        """Write the accumulators and ``meta`` as JSON in the same archive."""
        np.savez_compressed(
            path, bin_edges=self.edges, true_edges=self.true_edges,
            est_edges=self.est_edges,
            hist_truth=self.hist["truth"], hist_point=self.hist["point"],
            hist_draws=self.hist["draws"],
            joint_point=self.joint["point"],
            joint_draw_mean=self.joint["draw_mean"],
            moments_truth=self.moments["truth"],
            moments_point=self.moments["point"],
            moments_draws=self.moments["draws"],
            t_rep=np.concatenate(self.t_rep) if self.t_rep else np.zeros(0),
            n_windows=self.n_windows, n_pairs=self.n_pairs,
            meta=json.dumps(meta))


def main() -> None:
    """Accumulate one chunk's three distributions and write the shard."""
    source_ts = tskit.load(TREES)
    ts = source_ts.simplify(samples=_ingroup(source_ts), filter_sites=False)
    n_hap = ts.num_samples
    names = [f"s{i}" for i in range(n_hap)]
    sample_map = {name: int(node) for name, node in zip(names, ts.samples())}
    samples = list(ts.samples())
    node_pairs = [(samples[i], samples[j])
                  for i, j in combinations(range(n_hap), 2)]

    model = JC69()
    inference = anc.LocalTreeInference(
        anc.TskitSource(ts, sample_map=sample_map), model,
        mu=MU, rec_rate=REC_RATE, sample_names=names,
        window=WINDOW, chunk_size=SEGMENT, progress=False,
    )
    inference._resolve_segmentation_params()

    from ancestree._ensemble import SegmentEnsemble
    acc = Accumulator()
    acc.n_pairs = len(node_pairs)
    for si, work_unit in enumerate(inference._stream_segments()):
        # A work unit is (sites, core_lo, core_hi, origin, span). The builder
        # shifts its sites to a local zero at ``origin``, so its windows are
        # segment-local while the tree sequence is read in absolute
        # coordinates. ``core_lo``/``core_hi`` bound the part this segment
        # owns; the halo either side is shared with its neighbour.
        _seg, core_lo, core_hi, origin, _span = work_unit
        builder = inference._segment_builder(work_unit)
        builder.time_grid = _deepen_floor(builder, FLOOR_SCALE)
        ens = SegmentEnsemble(builder, model, n_hap, None)
        intervals = builder._window_intervals()
        edges = origin + np.array(
            [lo for lo, _ in intervals] + [intervals[-1][1]], float)
        centre = 0.5 * (edges[:-1] + edges[1:])
        owned = (centre >= core_lo) & (centre < core_hi)

        genotypes, _pos0, block_of_site, n_blocks = builder._genotype_matrix()
        block_t = builder._pairwise_block_tmrcas(
            genotypes, block_of_site, n_blocks)
        keep = owned[ens.needed]
        rows = ens.needed[keep]
        point = np.stack([block_t[:, ens.blk_lo[w]:ens.blk_hi[w]].mean(axis=1)
                          for w in rows])
        truth = _true_window_tmrca(ts, node_pairs, edges)[rows]

        acc.add_marginal("truth", truth)
        acc.add_marginal("point", point)
        acc.add_joint("point", truth, point)
        acc.t_rep.append(np.asarray(ens.t_rep, float))
        acc.n_windows += len(rows)

        draw_mean = np.zeros_like(point)
        drawn = 0
        for chunk in ens.draw_chunks(N_MEMBERS, member_chunk=MEMBER_CHUNK,
                                     seed=SEED):
            acc.add_marginal("draws", chunk[:, keep])
            draw_mean += chunk[:, keep].sum(axis=0)
            drawn += chunk.shape[0]
        draw_mean /= drawn
        acc.add_joint("draw_mean", truth, draw_mean)
        print(f"segment {si}: {len(rows)} windows of {len(ens.needed)} kept "
              f"(core {core_lo:.0f}-{core_hi:.0f}, origin {origin:.0f}), "
              f"{drawn} members", flush=True)
        del ens, builder, block_t, point, truth, draw_mean

    acc.save(OUT_NPZ, {"trees": TREES, "mu": MU, "rec_rate": REC_RATE,
                       "window": WINDOW, "segment": SEGMENT,
                       "n_members": N_MEMBERS, "seed": SEED,
                       "n_hap": int(n_hap),
                       "sequence_length": float(ts.sequence_length)})
    print(f"wrote {OUT_NPZ}: {acc.n_windows} windows x {acc.n_pairs} pairs")


main()
