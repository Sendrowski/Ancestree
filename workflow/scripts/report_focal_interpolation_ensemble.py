"""Local-tree accuracy when the likelihood is marginalised over a tree ensemble.

The ``local_tree`` mode of
:mod:`workflow.scripts.report_focal_interpolation` scores one point estimate of
the local genealogy: the per-pair TMRCA posterior mean out of the
PSMC'-style HMM, agglomerated per window. Where that estimate mis-resolves the
deepest split inside the ingroup, the derived-allele carrier set stops being a
clade and the Felsenstein kernel is confidently wrong. Marginalising instead
over :math:`B` genealogies drawn from the same HMM posterior replaces the plug-in
likelihood with

.. math::

    P(s \\mid x) \\;\\propto\\; \\pi(s)\\,
        \\frac{1}{B}\\sum_{b=1}^{B} P(x \\mid s, \\tau_b),

where :math:`s` is the ancestral state at the focal node (one of the four
nucleotides), :math:`x` the observed tip alleles at a site, :math:`\\pi(s)` the
root prior (uniform under JC69 with no base composition), :math:`\\tau_b` the
:math:`b`-th sampled genealogy for the window carrying the site, and :math:`B`
the ensemble size. Each :math:`\\tau_b` is a joint draw of the per-block TMRCA
path for every haplotype pair, taken by forward-filtering backward-sampling from
the HMM's own forward matrix, condensed to per-window means and agglomerated by
the same average-linkage rule the point estimate uses.

The draws do not depend on where the posterior is read, so the ensemble is built
once per outgroup count and every focal position is evaluated by re-rooting each
member. Everything below the re-rooting reproduces, element for element, the
arrays :meth:`Likelihood.log_likelihoods_tskit_native()
<ancestree.likelihood.Likelihood.log_likelihoods_tskit_native>` builds from the
tskit tree that :func:`~ancestree._upgma.build_windowed_tree_sequence` would have
produced, so the ``fraction = 0`` column is directly comparable with the
``local_tree`` rows of the point-estimate scan.

Emits the same row schema as that scan, under the mode name ``local_tree_ens``.
"""
import json
from itertools import combinations

import numpy as np
import tskit

import ancestree as anc
from ancestree.focal import FocalNode
from ancestree.models import JC69
from ancestree.sites import Site

try:  # snakemake execution
    TREES = snakemake.input.trees
    OUT_JSON = snakemake.output.json
    OUT_DIAGNOSTICS = snakemake.output.diagnostics
    FRACTIONS = list(snakemake.params.fractions)
    N_OUT = int(snakemake.wildcards.n_out)
    MU = snakemake.params.mu
    REC_RATE = snakemake.params.rec_rate
    WINDOW = snakemake.params.window
    N_MEMBERS = int(snakemake.params.n_members)
    N_SUBSETS = int(snakemake.params.n_subsets)
    MEMBER_CHUNK = int(snakemake.params.member_chunk)
    SEED = int(snakemake.params.seed)
except NameError:  # direct execution
    TREES = "results/data/baseline10_chunk0.trees"
    OUT_JSON = "results/reports/focal_ens_shard_n1.json"
    OUT_DIAGNOSTICS = "results/reports/focal_ens_shard_n1_diagnostics.json"
    FRACTIONS = [round(k / 40, 3) for k in range(41)]
    N_OUT = 1
    MU = 1.25e-8
    REC_RATE = 1e-8
    WINDOW = "8snp"
    #: Ensemble size B in the marginalisation above.
    N_MEMBERS = 128
    #: Disjoint sub-ensembles the members are split into, so the Monte-Carlo
    #: spread of each reported figure can be read off.
    N_SUBSETS = 4
    #: Members held in memory at once. The transition matrices dominate.
    MEMBER_CHUNK = 8
    SEED = 1000

# The sampler and its compiled kernels now live in the package. This scan
# drives them rather than carrying its own copy.
from ancestree._ensemble import (  # noqa: E402
    S_STATES, ffbs_paths, hmm_forward_ckpt, pack_base_all, paths_to_condensed,
    reroot_all, score_ensemble, upgma_batch,
)


def _panel(ts, n_out):
    """Ingroup nodes and an outgroup subset spread across split depths.

    :param ts: The simulated tree sequence.
    :param n_out: Number of outgroups to retain.
    :return: ``(ingroup_nodes, outgroup_nodes)``.
    """
    name = {p.id: (p.metadata or {}).get("name") for p in ts.populations()}
    ingroup = [int(n) for n in ts.samples()
               if name[ts.node(int(n)).population] == "ingroup"]
    outgroups = [int(n) for n in ts.samples()
                 if str(name[ts.node(int(n)).population]).startswith("outgroup")]
    outgroups.sort(key=lambda n: int(name[ts.node(n).population].split("_")[1]))
    if n_out >= len(outgroups):
        chosen = outgroups
    else:
        index = np.linspace(0, len(outgroups) - 1, n_out).round().astype(int)
        chosen = [outgroups[i] for i in dict.fromkeys(index)]
    return ingroup, chosen


def _truth_at(tree, site, node, height=0.0):
    """Simulated state at a point ``height`` above ``node``.

    :param tree: The local tree covering the site.
    :param site: The :class:`tskit.Site` carrying the mutations.
    :param node: Node immediately below the point.
    :param height: Distance above ``node``, in the tree's own time units.
    :return: The simulated state at that point.
    """
    point_time = tree.time(node) + height
    state, best = site.ancestral_state, np.inf
    for mutation in site.mutations:
        on_path = (mutation.node == node
                   or tree.is_descendant(node, mutation.node))
        if not on_path:
            continue
        time = float(mutation.time)
        if np.isnan(time):
            time = tree.time(mutation.node)
        if time <= point_time:
            continue
        if time < best:
            best, state = time, mutation.derived_state
    return state


class _GenotypeSource(anc.SiteSource):
    """Re-iterable genotype stream over a panel's variants.

    :param ts: The simplified panel tree sequence.
    :param names: ``{node: sample name}``.
    :param samples: Panel node ids, in ``ts.samples()`` order.
    """

    def __init__(self, ts, names, samples):
        self._ts = ts
        self._names = [names[int(n)] for n in samples]

    def samples(self) -> list:
        """:return: Sample names, in genotype-column order."""
        return list(self._names)

    def __iter__(self):
        """:return: One :class:`~ancestree.sites.Site` per variant."""
        for variant in self._ts.variants():
            yield Site(
                chrom="1", pos=int(variant.site.position),
                alleles=tuple(a for a in variant.alleles if a),
                tip_alleles={name: variant.alleles[g]
                             for name, g in zip(self._names, variant.genotypes)},
            )


class EnsembleFocalScan:
    """Marginalised local-tree accuracy across the focal-node sweep.

    :param source: The simulated tree sequence the panel is drawn from.
    :param n_out: Number of outgroups in the panel.
    :param mu: Per-site per-generation mutation rate.
    :param rec_rate: Per-site per-generation recombination rate.
    :param window: Local-tree window specification.
    :param n_members: Ensemble size B.
    :param n_subsets: Disjoint sub-ensembles the members are split into.
    :param member_chunk: Members packed and scored at once.
    :param seed: Base seed. Member ``b`` draws from stream ``seed + b``.
    """

    def __init__(self, source, n_out, *, mu, rec_rate, window, n_members,
                 n_subsets, member_chunk, seed):
        self.source = source
        self.n_out = int(n_out)
        self.mu = float(mu)
        self.rec_rate = float(rec_rate)
        self.window = window
        self.n_members = int(n_members)
        self.n_subsets = int(n_subsets)
        self.member_chunk = int(member_chunk)
        self.seed = int(seed)
        if self.n_members % (self.n_subsets * self.member_chunk):
            raise ValueError(
                "n_members must be a whole number of chunks and of subsets; "
                f"got {n_members}, {n_subsets} subsets, chunk {member_chunk}")
        self.model = JC69()

    # ------------------------------------------------------------ panel
    def build_panel(self):
        """Simplify to the panel and index its sites.

        :return: ``None``. Fills the panel attributes used by everything else.
        """
        ingroup, outgroups = _panel(self.source, self.n_out)
        self.ts = self.source.simplify(samples=ingroup + outgroups,
                                       filter_sites=False)
        self.samples = list(self.ts.samples())
        self.names = {int(n): f"s{i}" for i, n in enumerate(self.samples)}
        self.ing_nodes = self.samples[:len(ingroup)]
        self.n_ing = len(self.ing_nodes)
        self.n_hap = len(self.samples)
        # The ingroup occupies the leading columns, matching sample_names.
        self.is_ingroup = np.arange(self.n_hap) < self.n_ing
        self.sample_names = [self.names[int(n)] for n in self.samples]

        positions, classes = [], []
        tip_states = np.full((self.ts.num_sites, self.n_hap), -1, dtype=np.int8)
        ing_set = set(self.ing_nodes)
        for i, variant in enumerate(self.ts.variants()):
            positions.append(int(variant.site.position))
            for h, g in enumerate(variant.genotypes):
                allele = variant.alleles[g]
                if allele:
                    tip_states[i, h] = anc.STATE_INDEX.get(allele, -1)
            alleles = {variant.alleles[g]
                       for node, g in zip(self.samples, variant.genotypes)
                       if int(node) in ing_set and variant.alleles[g]}
            if len(alleles) > 1:
                classes.append("polymorphic")
            elif alleles and next(iter(alleles)) == variant.site.ancestral_state:
                classes.append("fixed_ancestral")
            elif alleles:
                classes.append("fixed_derived")
            else:
                classes.append("")
        self.positions = np.asarray(positions, dtype=np.int64)
        self.site_class = np.asarray(classes)
        self.tip_states = tip_states
        self.n_sites = len(positions)

    # -------------------------------------------------------------- HMM
    def build_hmm(self):
        """Segment the panel, run the forward pass, and keep the point estimate.

        :return: ``None``. Fills the HMM attributes and the window index.
        """
        inference = anc.LocalTreeInference(
            _GenotypeSource(self.ts, self.names, self.samples), self.model,
            mu=self.mu, rec_rate=self.rec_rate,
            sample_names=self.sample_names,
            ingroup_samples=self.sample_names[:self.n_ing],
            outgroup_samples=self.sample_names[self.n_ing:],
            window=self.window, chunk_size="2mb",
            progress=False,
        )
        inference._resolve_segmentation_params()
        segments = list(inference._stream_segments())
        if len(segments) != 1:
            raise ValueError(
                f"the ensemble path assumes one segment; got {len(segments)}")
        builder = inference._segment_builder(segments[0])
        if (builder.recombination_map is not None
                or builder.accessibility is not None
                or builder.mutation_map is not None
                or builder.segment_breaks):
            raise ValueError("the ensemble path assumes a plain uniform map")
        self.builder = builder

        genotypes, _pos0, block_of_site, n_blocks = builder._genotype_matrix()
        intervals = builder._window_intervals()
        self.intervals = intervals

        from ancestree.local_tree_inference import PairwiseCoalescentHMM
        hmm = PairwiseCoalescentHMM(
            self.n_hap, mu=builder.mu, rec_rate=builder.rec_rate,
            block_size=builder.block_size, n_time_bins=builder.n_time_bins)
        pairs = list(combinations(range(self.n_hap), 2))
        counts = np.zeros((len(pairs), n_blocks), dtype=np.int64)
        for p, (a, b) in enumerate(pairs):
            ga, gb = genotypes[:, a], genotypes[:, b]
            differ = (ga != gb) & (ga >= 0) & (gb >= 0)
            if differ.any():
                np.add.at(counts[p], block_of_site[differ], 1)
        edges, t_rep = hmm._calibrate_time_grid(counts)
        # Per-block (n_blocks, T), the shape the kernels index as
        # lam[k, i] and r[k + 1, i]. This path rejects non-uniform maps above,
        # so every block carries the same rate and the broadcast is exactly
        # what SegmentEnsemble builds with trivial block geometry.
        lam = np.repeat((2.0 * hmm.mu * hmm.block_size * t_rep)[None, :],
                        n_blocks, axis=0)
        log_lam = np.log(np.clip(lam, 1e-300, None))
        r = np.exp(-np.repeat(
            (2.0 * hmm.rec_rate * hmm.block_size * t_rep)[None, :],
            n_blocks, axis=0))
        t_bar = counts.mean(axis=1) / (2.0 * hmm.mu * hmm.block_size)
        prior = np.stack([hmm._coalescent_prior(edges, t_bar[p])
                          for p in range(len(pairs))])

        self.alpha = np.empty((len(pairs), n_blocks, hmm.n_time_bins),
                              dtype=np.float32)
        # stride=1 checkpoints every block, so alpha_ck IS the full forward
        # matrix, bit-for-bit what the removed hmm_forward stored.
        esc = np.ones(counts.shape, np.float64)
        hmm_forward_ckpt(counts, lam, log_lam, r, prior, esc, 1, self.alpha)
        self.hmm_r, self.hmm_pi, self.t_rep = r, prior, t_rep
        self.log_lo = np.log10(edges[:-1])
        self.log_w = np.log10(edges[1:]) - self.log_lo
        self.edge_lo = edges[:-1].copy()
        self.edge_hi = edges[1:].copy()
        self.t_bar = np.asarray(t_bar, np.float64)
        self.hmm_counts, self.hmm_lam, self.hmm_log_lam = counts, lam, log_lam
        self.hmm_esc = esc
        self.n_blocks = n_blocks
        # Keyed by column index: this panel is passed positionally throughout.
        self.pair_key = np.arange(len(pairs), dtype=np.uint64)

        # Windows carrying at least one site, and the site rows in each.
        # By containing interval, not positions // window_bp: _window_intervals
        # tiles a segment into round(seg_len / window_bp) equal windows, so the
        # real step is only window_bp when the segment divides evenly, and the
        # two drift apart near window edges (see SegmentEnsemble.__init__).
        _edges = np.array([lo for lo, _ in intervals] + [intervals[-1][1]],
                          dtype=float)
        win_of_site = np.clip(
            np.searchsorted(_edges, self.positions, side="right") - 1,
            0, len(intervals) - 1).astype(np.int64)
        self.needed = np.unique(win_of_site)
        order = np.argsort(win_of_site, kind="stable")
        self.row_idx = order.astype(np.int64)
        counts_per_win = np.bincount(
            np.searchsorted(self.needed, win_of_site[order]),
            minlength=len(self.needed))
        self.row_off = np.zeros(len(self.needed) + 1, dtype=np.int64)
        self.row_off[1:] = np.cumsum(counts_per_win)

        # The point estimate the non-marginalised mode scores.
        block_t = builder._pairwise_block_tmrcas(
            genotypes, block_of_site, n_blocks)
        bounds = [builder._blocks_for_window(l, r_, n_blocks)
                  for l, r_ in intervals]
        self.blk_lo = np.array([b0 for b0, _ in bounds], dtype=np.int64)
        self.blk_hi = np.array([b1 for _, b1 in bounds], dtype=np.int64)
        self.point_condensed = np.ascontiguousarray(np.stack(
            [block_t[:, b0:b1].mean(axis=1) for b0, b1 in
             (bounds[w] for w in self.needed)]))

    # ------------------------------------------------------------ truth
    def truth_tables(self, fractions):
        """Simulated states at each grading point, as state indices.

        :param fractions: Focal positions to resolve on the true trees.
        :return: ``(moving, at_mrca, at_root)``; ``moving`` is
            ``(len(fractions), n_sites) int8``, the others ``(n_sites,) int8``.
            ``-1`` marks a state outside the model alphabet.
        """
        index = {int(p): i for i, p in enumerate(self.positions)}
        moving = np.full((len(fractions), self.n_sites), -1, dtype=np.int8)
        at_mrca = np.full(self.n_sites, -1, dtype=np.int8)
        at_root = np.full(self.n_sites, -1, dtype=np.int8)
        def point_at(tree, fraction):
            return FocalNode(
                "ingroup_mrca", fraction=(fraction if fraction > 0 else None),
            ).resolve(tree, ingroup_nodes=self.ing_nodes)

        for tree in self.ts.trees():
            if not tree.num_sites:
                continue
            points = [point_at(tree, f) for f in fractions]
            shallow = point_at(tree, 0.0)
            deep = point_at(tree, 1.0)
            for site in tree.sites():
                row = index[int(site.position)]
                at_mrca[row] = anc.STATE_INDEX.get(
                    _truth_at(tree, site, shallow.node, shallow.tau), -1)
                at_root[row] = anc.STATE_INDEX.get(
                    _truth_at(tree, site, deep.node, deep.tau), -1)
                for fi, point in enumerate(points):
                    moving[fi, row] = anc.STATE_INDEX.get(
                        _truth_at(tree, site, point.node, point.tau), -1)
        return moving, at_mrca, at_root

    # --------------------------------------------------------- scoring
    def _new_accumulator(self, n_frac, n_sub):
        """Zeroed posterior accumulator and its per-site log offset."""
        acc = np.zeros((n_frac, n_sub, self.n_sites, S_STATES))
        ref = np.full((n_frac, n_sub, self.n_sites), -1e300)
        return acc, ref

    def _score_chunk(self, condensed, fractions, out, arrays):
        """Log-likelihoods for one member chunk at every focal position.

        :param condensed: ``(M, W, n_pairs)`` sampled window-mean TMRCAs.
        :param fractions: Focal positions to evaluate.
        :param out: ``(M, n_sites, 4)`` scratch for one position's result.
        :param arrays: The per-chunk buffers from :meth:`_chunk_buffers`.
        :return: Generator of ``(fraction_index, out)``.
        """
        (Z, branch_t, parent0, sample_dense, fmrca, pidx, cofs, cflat, poi,
         root) = arrays
        n = self.n_hap
        n_all = 2 * n - 1
        nslot = n_all + 2
        M, W = condensed.shape[0], condensed.shape[1]
        # The kernel rebuilds each edge's exp(Q*t) from the eigendecomposition
        # rather than reading a materialised (M, W, nslot, 4, 4) array.
        lam, V, Vinv = self.model.real_eig()

        upgma_batch(condensed.reshape(M * W, -1), n, Z.reshape(M * W, n - 1, 4))
        pack_base_all(Z.reshape(M * W, n - 1, 4), n, self.is_ingroup, self.mu,
                      branch_t.reshape(M * W, nslot),
                      parent0.reshape(M * W, n_all),
                      sample_dense.reshape(M * W, n), fmrca.reshape(M * W))

        for fi, fraction in enumerate(fractions):
            reroot_all(parent0.reshape(M * W, n_all),
                       branch_t.reshape(M * W, nslot), fmrca.reshape(M * W),
                       n, float(fraction), -1, pidx.reshape(M * W, nslot),
                       cofs.reshape(M * W, nslot + 1),
                       cflat.reshape(M * W, nslot), poi.reshape(M * W, n),
                       root.reshape(M * W))
            score_ensemble(branch_t, lam, V, Vinv,
                           pidx, poi, cofs, cflat, root, sample_dense,
                           self.tip_states, self.row_off, self.row_idx,
                           nslot, S_STATES, out)
            yield fi, out

    def _chunk_buffers(self, M, W):
        """Allocate the per-chunk packing and transition buffers."""
        n = self.n_hap
        n_all = 2 * n - 1
        nslot = n_all + 2
        return (
            np.empty((M, W, n - 1, 4)),
            np.empty((M, W, nslot)),
            np.empty((M, W, n_all), np.int32),
            np.empty((M, W, n), np.int32),
            np.empty((M, W), np.int32),
            np.empty((M, W, nslot), np.int32),
            np.empty((M, W, nslot + 1), np.int32),
            np.empty((M, W, nslot), np.int32),
            np.empty((M, W, n), np.int32),
            np.empty((M, W), np.int64),
        )

    @staticmethod
    def _accumulate(out, acc, ref):
        """Fold one chunk's log-likelihoods into a running ``sum_b P(x|s,tau_b)``.

        :param out: ``(M, n_sites, 4)`` log-likelihoods.
        :param acc: ``(n_sites, 4)`` running sum, in units of ``exp(-ref)``.
        :param ref: ``(n_sites,)`` current log offset, updated in place.
        """
        chunk = out.max(axis=(0, 2))
        new = np.maximum(ref, chunk)
        acc *= np.exp(ref - new)[:, None]
        acc += np.exp(out - new[None, :, None]).sum(axis=0)
        ref[:] = new

    @staticmethod
    def _posterior(acc):
        """Normalise an accumulator to a posterior, uniform where it is empty."""
        total = acc.sum(axis=1, keepdims=True)
        good = total[:, 0] > 0
        post = np.full_like(acc, 1.0 / S_STATES)
        post[good] = acc[good] / total[good]
        return post

    def run(self, fractions, *, progress=print):
        """Score the ensemble, and the point estimate, over the whole sweep.

        :param fractions: Focal positions to evaluate.
        :param progress: Callable taking one string.
        :return: ``(ensemble_posteriors, subset_posteriors, point_posteriors)``
            of shapes ``(F, n_sites, 4)``, ``(F, n_subsets, n_sites, 4)`` and
            ``(F, n_sites, 4)``.
        """
        W = len(self.needed)
        M = self.member_chunk
        n_frac = len(fractions)
        acc, ref = self._new_accumulator(n_frac, self.n_subsets)
        arrays = self._chunk_buffers(M, W)
        out = np.empty((M, self.n_sites, S_STATES))

        point = np.empty((n_frac, self.n_sites, S_STATES))
        point_arrays = self._chunk_buffers(1, W)
        point_out = np.empty((1, self.n_sites, S_STATES))
        for fi, res in self._score_chunk(self.point_condensed[None], fractions,
                                         point_out, point_arrays):
            row_max = res[0].max(axis=1, keepdims=True)
            probs = np.exp(res[0] - row_max)
            point[fi] = probs / probs.sum(axis=1, keepdims=True)
        del point_arrays, point_out
        progress("point estimate scored")

        P = self.alpha.shape[0]
        draws = np.empty((M, W, P))
        path = np.empty((M, P, self.n_blocks), dtype=np.int8)
        per_subset = self.n_members // self.n_subsets
        for start in range(0, self.n_members, M):
            ffbs_paths(self.hmm_counts, self.hmm_lam, self.hmm_log_lam,
                       self.alpha, 1, self.n_blocks, self.hmm_r, self.hmm_pi,
                       self.hmm_esc, self.seed + start, self.pair_key, path)
            paths_to_condensed(path, self.edge_lo, self.edge_hi, self.t_bar,
                               self.needed,
                               self.blk_lo, self.blk_hi, 0, self.pair_key,
                               self.seed, start, draws)
            sub = start // per_subset
            for fi, res in self._score_chunk(draws, fractions, out, arrays):
                self._accumulate(res, acc[fi, sub], ref[fi, sub])
            progress(f"members {start}-{start + M - 1} scored")
        del arrays, out, draws, path
        self.alpha = None

        subset_post = np.empty((n_frac, self.n_subsets, self.n_sites, S_STATES))
        full = np.empty((n_frac, self.n_sites, S_STATES))
        for fi in range(n_frac):
            for sub in range(self.n_subsets):
                subset_post[fi, sub] = self._posterior(acc[fi, sub])
            top = ref[fi].max(axis=0)
            merged = (acc[fi] * np.exp(ref[fi] - top[None])[:, :, None]).sum(0)
            full[fi] = self._posterior(merged)
        return full, subset_post, point


def _rows(post, moving, at_mrca, at_root, site_class, fractions, n_out, mode):
    """Per-class mean Brier scores for one mode's posteriors.

    :param post: ``(F, n_sites, 4)`` posteriors.
    :param moving: ``(F, n_sites)`` truth at the reporting point.
    :param at_mrca: ``(n_sites,)`` truth at the ingroup MRCA.
    :param at_root: ``(n_sites,)`` truth at the panel root.
    :param site_class: ``(n_sites,)`` class label per site.
    :param fractions: Focal positions, parallel to ``post``.
    :param n_out: Outgroup count, recorded on every row.
    :param mode: Mode name, recorded on every row.
    :return: List of row dicts.
    """
    rows = []
    for fi, fraction in enumerate(fractions):
        gradings = (("moving", moving[fi]), ("at_mrca", at_mrca),
                    ("at_root", at_root))
        for label in ("polymorphic", "fixed_ancestral", "fixed_derived"):
            in_class = site_class == label
            for grading, table in gradings:
                keep = in_class & (table >= 0)
                probs = post[fi][keep]
                hit = np.zeros_like(probs)
                hit[np.arange(probs.shape[0]), table[keep]] = 1.0
                score = ((probs - hit) ** 2).sum(axis=1)
                rows.append({
                    "mode": mode, "n_out": int(n_out),
                    "fraction": float(fraction), "site_class": label,
                    "grading": grading, "n_sites": int(keep.sum()),
                    "mean_brier": (float(score.mean()) if score.size else None),
                })
    return rows


def main() -> None:
    """Sweep the focal node under ensemble marginalisation and write JSON."""
    source = tskit.load(TREES)
    n_out = N_OUT
    scan = EnsembleFocalScan(
        source, n_out, mu=MU, rec_rate=REC_RATE, window=WINDOW,
        n_members=N_MEMBERS, n_subsets=N_SUBSETS,
        member_chunk=MEMBER_CHUNK, seed=SEED)
    scan.build_panel()
    print(f"n_out={n_out}: {scan.n_hap} haplotypes, {scan.n_sites} sites",
          flush=True)
    scan.build_hmm()
    print(f"n_out={n_out}: {len(scan.needed)} windows, "
          f"alpha {scan.alpha.nbytes / 1e9:.2f} GB", flush=True)
    moving, at_mrca, at_root = scan.truth_tables(FRACTIONS)
    print(f"n_out={n_out}: truth tables built", flush=True)

    full, subsets, point = scan.run(
        FRACTIONS,
        progress=lambda m: print(f"n_out={n_out}: {m}", flush=True))
    rows = _rows(full, moving, at_mrca, at_root, scan.site_class,
                 FRACTIONS, n_out, "local_tree_ens")
    # The plug-in estimate down the same path: identical to the
    # ``local_tree`` rows of the point-estimate scan, so any drift between
    # the two is visible without a second run.
    diagnostics = _rows(point, moving, at_mrca, at_root, scan.site_class,
                        FRACTIONS, n_out, "local_tree_point")
    for sub in range(scan.n_subsets):
        diagnostics += _rows(
            subsets[:, sub], moving, at_mrca, at_root, scan.site_class,
            FRACTIONS, n_out, f"local_tree_ens_subset{sub}")

    meta = {"trees": TREES, "mu": MU, "rec_rate": REC_RATE, "window": WINDOW,
            "n_members": N_MEMBERS, "n_subsets": N_SUBSETS, "seed": SEED,
            "n_out": n_out}
    with open(OUT_JSON, "w") as handle:
        json.dump({**meta, "rows": rows}, handle, indent=1)
    with open(OUT_DIAGNOSTICS, "w") as handle:
        json.dump({**meta, "rows": diagnostics}, handle, indent=1)
    print(f"wrote {OUT_JSON} and {OUT_DIAGNOSTICS}")


main()
