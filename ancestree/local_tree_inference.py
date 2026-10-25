"""Local-tree inference from genotypes: genealogy estimation + ancestral-allele calling.

The two main public classes, backed by the lower-level
:class:`~ancestree.local_tree_inference.PairwiseCoalescentHMM` and :class:`~ancestree.local_tree_inference.PairwiseTmrcas`:

- :class:`~ancestree.local_tree_inference.LocalTreeBuilder`, the genealogy estimator. Infers local
  dated trees along the genome from genotypes alone (a discretized
  PSMC′-style pairwise-TMRCA HMM, condensed per window into one ultrametric
  tree) and emits a self-describing :class:`tskit.TreeSequence` (sample names
  + the observed genotypes baked in as sites).
  :meth:`LocalTreeBuilder.write() <ancestree.local_tree_inference.LocalTreeBuilder.write>`
  dumps a ``.trees`` file that can be fed straight to
  :class:`~ancestree.inference.ARGBasedInference`.
- :class:`~ancestree.local_tree_inference.LocalTreeInference`, which builds the
  trees from genotypes via :class:`~ancestree.local_tree_inference.LocalTreeBuilder`
  and scores the per-site ancestral-allele posterior over an ensemble of drawn
  genealogies, or on the single plug-in genealogy through
  :class:`~ancestree.inference.ARGBasedInference` when the ensemble is off.
  Implements the
  :class:`~ancestree.inference.Inference` contract
  (:meth:`Inference.infer() <ancestree.inference.Inference.infer>` yields
  ``(Site, Posterior)``).

The genealogy estimator follows the SMC family in reading local trees from
pairwise coalescence times, using a discretized-time HMM, not a
continuous approximation. It requires no ancestral-allele call and assumes
phased haplotypes. Outgroups are supported, not required: they are extra panel
haplotypes the local tree places, so ILS is accommodated by the inferred tree
and not by an assumed outgroup-ladder topology.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, compress
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tskit

from ancestree import DEFAULT_MU, DEFAULT_REC_RATE, STATES
from tqdm import tqdm

from ancestree._smc_kernel import pair_posterior_mean_tmrca
from ancestree._upgma import (
    WindowedUpgmaTreeSequence,
)
from ancestree.focal import FocalNode
from ancestree.inference import (
    _ZERO_RATE_FLOOR, ARGBasedInference, Inference,
)
from ancestree.models import JC69, SubstitutionModel
from ancestree.posterior import Posterior
from ancestree.priors import StationaryPrior
from ancestree.settings import Settings
from ancestree.sites import (BaseComposition, Site, SiteSource,
                             SiteTable, _PanelSites, _path_format)
from ancestree._repr import ReprMixin

# Below this many SNPs per block (panel-wide average), a large fraction of
# blocks carry no informative sites and the per-block pairwise-TMRCA estimates
# become noise-dominated. The resolution warning fires.
_MIN_SNPS_PER_BLOCK = 2.0


def _length_spec_error(name: str, spec: object, *, snp: bool) -> ValueError:
    """The error a malformed length spec raises, listing the accepted forms.

    :param name: Parameter the spec was given for, named in the message.
    :param spec: The rejected value.
    :param snp: Whether the parameter also takes a ``"<N>snp"`` spec.
    :return: The exception for the caller to raise.
    """
    forms = ("an integer number of base pairs, or a 'bp' / 'kb' / 'mb' "
             "suffixed string such as '10mb'")
    if snp:
        forms = ("an integer number of base pairs, a 'bp' / 'kb' / 'mb' "
                 "suffixed string such as '10mb', or a '<N>snp' string such "
                 "as '8snp'")
    return ValueError(f"{name} must be {forms}, got {spec!r}")


def _parse_bp(spec: "int | str", *, name: str = "length",
              snp: bool = False) -> int:
    """Parse a bp length: an int or a ``bp`` / ``kb`` / ``mb`` suffixed string.

    :param spec: The width to parse.
    :param name: Parameter the spec was given for, named in the error message.
    :param snp: Whether the calling parameter also takes a ``"<N>snp"`` spec,
        which the error message then lists.
    :return: The width in base pairs.
    :raises ValueError: If ``spec`` is not one of the accepted forms, or is not
        a positive width.
    """
    try:
        if isinstance(spec, str):
            s = spec.strip().lower()
            if s.endswith("kb"):
                bp = int(round(float(s[:-2]) * 1000))
            elif s.endswith("mb"):
                bp = int(round(float(s[:-2]) * 1_000_000))
            elif s.endswith("bp"):
                bp = int(s[:-2])
            else:
                bp = int(s)
        else:
            bp = int(spec)
    except (TypeError, ValueError):
        raise _length_spec_error(name, spec, snp=snp) from None
    if bp <= 0:
        raise ValueError(f"{name} must be a positive width, got {spec!r}")
    return bp


def _width_bp(spec: "int | str", n_sites: int, span_bp: float,
              *, name: str) -> int:
    """A window or block width in bp. An ``"<N>snp"`` spec is N sites' worth
    of span at the data's mean SNP density. A bp / int spec is its parsed
    width.

    :param spec: The width spec.
    :param n_sites: Sites the density is taken over.
    :param span_bp: Span the density is taken over.
    :param name: Parameter the spec was given for, named in the error message.
    :return: The width in base pairs.
    :raises ValueError: If ``spec`` is not one of the accepted forms.
    """
    if isinstance(spec, str) and spec.strip().lower().endswith("snp"):
        try:
            n_target = int(spec.strip().lower()[:-3])
        except ValueError:
            raise _length_spec_error(name, spec, snp=True) from None
        if n_target <= 0:
            raise _length_spec_error(name, spec, snp=True)
        return max(1, int(round(n_target * span_bp / max(1, n_sites))))
    return _parse_bp(spec, name=name, snp=True)


# Target SNPs per HMM emission block when ``block_size`` is left unset. A block
# finer than the window gives the pairwise HMM several emission units per output
# tree.
_DEFAULT_BLOCK_SNPS = 4.0


def _default_block_bp(n_sites: int, span_bp: float, window_bp: int) -> int:
    """Density-adaptive default block width in bp: about
    ``_DEFAULT_BLOCK_SNPS`` SNPs at the data's mean density, sized so a whole
    number of blocks spans the window exactly, never coarser than the window
    and at least 1 bp.
    """
    target = _DEFAULT_BLOCK_SNPS * span_bp / max(1, n_sites)
    n_blocks = max(1, int(round(window_bp / max(target, 1.0))))
    return max(1, -(-int(window_bp) // n_blocks))


# Set by LocalTreeInference._infer_segmented just before forking the segment
# pool. Children inherit it via copy-on-write fork (no pickling of the input).
_LOCAL_TREE_FOR_FORK: "LocalTreeInference | None" = None


def _segment_worker(spec):
    """Fork-pool entrypoint: process one segment spec on the inherited instance."""
    inst = _LOCAL_TREE_FOR_FORK
    if inst is None:
        raise RuntimeError(
            "_LOCAL_TREE_FOR_FORK not set in the parent; fork did not "
            "propagate the module global."
        )
    return inst._process_segment(spec)


#: Deepest time grid the sampled-path storage can address. Bin indices are
#: ``int8`` (see ``SegmentEnsemble``).
MAX_TIME_BINS: int = int(np.iinfo(np.int8).max)


class PairwiseCoalescentHMM(ReprMixin):
    """Discretized PSMC′-style estimator of local pairwise TMRCAs.

    Estimates, for every haplotype pair, the posterior-mean coalescence
    time (in generations) within each fixed-width block along the
    sequence. The hidden state is a discretized coalescent time
    (``n_time_bins`` log-spaced bins). The emission is the count of
    pairwise differences per block as ``Poisson(2·μ·t·block_size)``, which,
    unlike a binary heterozygosity flag, does not saturate at deep
    TMRCAs. The transition resets the TMRCA to the prior with per-block
    probability ``1 - e^{-2ρ t · block_size}`` (recombination), giving the
    standard ``O(n_blocks · T)`` forward-backward.

    The shared time grid is calibrated from the per-block difference
    counts (top bin reaches the deepest observed divergence, so outgroup
    pairs are not pinned to the last bin). The prior / reset distribution
    is per-pair coalescent: each pair's local times are taken
    ``Exponential`` with mean equal to that pair's own genome-average
    TMRCA, so ingroup pairs receive a recent-concentrated prior and outgroup pairs a
    deep one. No demographic truth is read.

    :param n_haplotypes: Number of panel haplotypes.
    :param mu: Per-site per-generation mutation rate.
    :param rec_rate: Per-site per-generation recombination rate.
    :param block_size: Block width in bp (HMM emission resolution).
    :param n_time_bins: Number of discretized coalescent-time bins.
    :param time_grid: Explicit bin edges in generations, ascending and strictly
        positive, replacing the data-calibrated grid and overriding
        ``n_time_bins``. At most ``MAX_TIME_BINS + 1`` edges may be given.
    :raises ValueError: If fewer than two haplotypes are given, if ``mu`` is
        not a positive finite rate, or if ``n_time_bins`` is outside
        ``[1, MAX_TIME_BINS]``.
    """

    @staticmethod
    def _check_time_grid(time_grid):
        """Validate an explicit time grid, or pass ``None`` through.

        :param time_grid: Bin edges in generations, ascending, or ``None`` to
            calibrate from the data.
        :return: The edges as a float array, or ``None``.
        :raises ValueError: If the edges are not one strictly ascending,
            positive, finite dimension of at least two entries.
        """
        if time_grid is None:
            return None
        edges = np.asarray(time_grid, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                f"time_grid must be a 1-D array of at least 2 edges, got shape "
                f"{edges.shape}")
        if not np.all(np.isfinite(edges)) or edges[0] <= 0.0:
            raise ValueError(
                "time_grid edges must be finite and positive (the grid is "
                f"log-spaced), got first edge {edges[0]}")
        if not np.all(np.diff(edges) > 0):
            raise ValueError("time_grid edges must be strictly ascending")
        return edges

    @staticmethod
    def _check_n_time_bins(n_time_bins) -> None:
        """Reject a time grid the sampled-path storage cannot address.

        :param n_time_bins: The requested grid depth.
        :raises ValueError: If it is outside ``[1, MAX_TIME_BINS]``.
        """
        if not 1 <= int(n_time_bins) <= MAX_TIME_BINS:
            raise ValueError(
                f"n_time_bins must be in [1, {MAX_TIME_BINS}], got {n_time_bins}"
            )


    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_haplotypes": self.n_haplotypes, "n_time_bins": self.n_time_bins}

    def __init__(
        self,
        n_haplotypes: int,
        *,
        mu: float,
        rec_rate: float,
        block_size: int = 2000,
        n_time_bins: int = 32,
        time_grid: "np.ndarray | None" = None,
    ) -> None:
        if n_haplotypes < 2:
            raise ValueError(f"need >= 2 haplotypes, got {n_haplotypes}")
        if not np.isfinite(mu):
            raise ValueError(f"mu must be positive and finite, got {mu}")
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        self.n_haplotypes = int(n_haplotypes)
        self.mu = float(mu)
        self.rec_rate = float(rec_rate)
        self.block_size = int(block_size)
        self.time_grid = PairwiseCoalescentHMM._check_time_grid(time_grid)
        if self.time_grid is not None:
            n_time_bins = self.time_grid.size - 1
        PairwiseCoalescentHMM._check_n_time_bins(n_time_bins)
        self.n_time_bins = int(n_time_bins)

    def calibrate(self, counts, *, emit_scale=None, block_mask=None,
                  called=None, block_sites=None):
        """Time grid, per-bin rates and per-pair prior mean, from block counts.

        :param counts: ``(n_pairs, n_blocks)`` pairwise difference counts.
        :param emit_scale: ``(n_blocks,)`` Poisson exposure scale of each
            block, the callable base-pair fraction of the block times its local
            mutation rate over the reference ``mu`` (dimensionless, ``1`` for a
            fully accessible block at the reference rate). ``None`` is uniform.
            Counts are divided by it.
        :param block_mask: Callable blocks. ``None`` is all.
        :param called: ``(n_pairs, n_blocks)`` number of the block's sites at
            which the pair is called. ``None`` is fully called. Counts are
            divided by the called fraction ``called / block_sites``.
        :param block_sites: ``(n_blocks,)`` site count each ``called`` entry is
            out of. Required with ``called``.
        :return: ``(edges, t_rep, lam, log_lam, r, t_bar_per_pair)``.
        """
        from ancestree._ensemble import row_chunks
        n_pairs, n_blocks = counts.shape
        if emit_scale is None:
            es = None
            informative = np.ones(n_blocks, dtype=bool)
        else:
            es = np.asarray(emit_scale, dtype=np.float64)
            # A block at the emission-scale floor carries no information
            # about coalescent time and is excluded from the calibration.
            informative = es > _EMIT_SCALE_FLOOR
            es = np.where(informative, es, 1.0)
        grid_cols = informative
        if block_mask is not None:
            grid_cols = grid_cols & block_mask
        if not grid_cols.any():
            grid_cols = informative if informative.any() \
                else np.ones(n_blocks, dtype=bool)
        cols = np.flatnonzero(grid_cols)

        # Effective counts over the grid columns, built a chunk of pairs at a
        # time. The prior mean averages over the blocks the pair is called in.
        # Column-major, the layout a column selection gives, so each pair's
        # blocks are summed in order.
        eff = np.empty((n_pairs, cols.size), dtype=np.float64, order="F")
        own = np.empty(n_pairs, dtype=np.float64)
        n_seen = np.full(n_pairs, cols.size, dtype=np.int64)
        for a, b in row_chunks(n_pairs, cols.size):
            e = counts[a:b, cols].astype(np.float64)
            if es is not None:
                e /= es[cols]
            if called is None:
                own[a:b] = e.mean(axis=1)
            else:
                cf = called[a:b, cols] / block_sites[cols]
                seen = cf > _ACCESS_FRAC_FLOOR
                e /= np.where(seen, cf, 1.0)
                n_seen[a:b] = seen.sum(axis=1)
                own[a:b] = (e * seen).sum(axis=1) / np.maximum(n_seen[a:b], 1)
            eff[a:b] = e
        per_block_rate = 2.0 * self.mu * self.block_size
        t_bar = np.where(n_seen > 0, own, np.nan) / per_block_rate

        edges, t_rep = self._calibrate_time_grid(eff, overwrite_input=True)
        lam = 2.0 * self.mu * self.block_size * t_rep
        log_lam = np.log(np.clip(lam, 1e-300, None))
        r = np.exp(-2.0 * self.rec_rate * self.block_size * t_rep)
        # A pair never called alongside itself has no average of its own; its
        # counts are all zero, so the plain mean would pin it at the grid
        # floor and make it the first UPGMA merge in every window.
        blind = ~np.isfinite(t_bar)
        if blind.any():
            fallback = (float(np.median(t_bar[~blind])) if (~blind).any()
                        else float(np.mean(t_rep)))
            t_bar = np.where(blind, fallback, t_bar)
            self._log.warning(
                "%d haplotype pair(s) are never called at the same site, "
                "so they carry the panel's median coalescence time rather "
                "than one of their own.", int(blind.sum()))
        return edges, t_rep, lam, log_lam, r, t_bar

    def _calibrate_time_grid(
        self, counts: np.ndarray, overwrite_input: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the shared log-spaced grid (bin edges + representatives).

        Calibrated from the per-block difference counts, not a binary
        heterozygosity flag: ``crude TMRCA = count / (2·μ·block)``. Because the
        count does not saturate, the top bin reaches the genuinely deep
        coalescences (outgroup pairs) and does not cap at the
        one-difference ceiling.

        :param counts: ``(n_pairs, n_blocks)`` per-block difference counts,
            divided by any per-block mutation-rate scale.
        :param overwrite_input: Use ``counts``, a contiguous float64 array, as
            scratch, leaving its contents undefined.
        :return: ``(edges[T+1], t_rep[T])``, bin edges and geometric-mid
            representatives, in generations.
        """
        if self.time_grid is not None:
            edges = self.time_grid
            return edges, np.sqrt(edges[:-1] * edges[1:])
        from ancestree._ensemble import row_chunks
        per_block_rate = 2.0 * self.mu * self.block_size
        crude = np.divide(counts, per_block_rate,
                          out=counts if overwrite_input else None)
        # Shallow end: the median over pairs of each pair's mean over all its
        # blocks, empty ones included, divided by 200.
        per_pair = np.atleast_1d(
            crude.mean(axis=1) if crude.ndim == 2 else crude.mean())
        # The nonzero entries, compacted in order to the front of ``crude``.
        flat = crude.ravel(order="K")
        n_nonzero = 0
        for a, b in row_chunks(flat.size, 1):
            chunk = flat[a:b]
            v = chunk[chunk > 0]
            flat[n_nonzero:n_nonzero + v.size] = v
            n_nonzero += v.size
        nonzero = flat[:n_nonzero]
        if nonzero.size == 0:
            t_lo, t_hi = 1e2, 1e5
        else:
            # robust deep end (0.5% quantile guard against a lone outlier)
            t_hi = float(np.quantile(nonzero, 0.995, overwrite_input=True)) \
                * 2.0
            shallow = per_pair[per_pair > 0]
            t_mid = (float(np.median(shallow)) if shallow.size
                     else float(np.median(nonzero)))
            t_lo = max(t_mid / 200.0, 10.0)
            if t_hi <= t_lo:
                t_hi = t_lo * 100.0
        edges = np.geomspace(t_lo, t_hi, self.n_time_bins + 1)
        t_rep = np.sqrt(edges[:-1] * edges[1:])  # geometric bin midpoints
        return edges, t_rep

    def _coalescent_prior(self, edges: np.ndarray, t_bar: float) -> np.ndarray:
        """Per-pair coalescent prior: ``Exp(1/t_bar)`` mass per grid bin.

        A pair whose genome-average TMRCA is ``t_bar`` has its local
        coalescence times distributed (under a constant-size coalescent)
        as ``Exponential`` with mean ``t_bar``. Discretising that over the
        shared grid gives a per-pair prior concentrated on recent bins for
        ingroup pairs and on deep bins for outgroup pairs.

        The grid is closed at both ends: mass below ``edges[0]`` joins the first
        bin and mass above ``edges[-1]`` joins the last, so the bins carry the
        whole distribution.
        """
        t_bar = max(float(t_bar), float(edges[0]))
        surv = np.exp(-edges / t_bar)  # survival at each edge
        pi = surv[:-1] - surv[1:]  # mass within each bin
        pi[0] += 1.0 - surv[0]
        pi[-1] += surv[-1]
        s = pi.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full(self.n_time_bins, 1.0 / self.n_time_bins)
        return pi / s

    def _pair_block_counts(
        self, genotypes: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
        dtype: "np.dtype | type" = np.int64,
    ) -> "tuple[np.ndarray, np.ndarray | None, np.ndarray]":
        """Per-block pairwise differences and the exposure they were counted over.

        :param genotypes: ``(n_sites, n_haplotypes)`` int8 allele indices
            (0..3), ``-1`` for missing.
        :param block_of_site: ``(n_sites,)`` block index per site.
        :param n_blocks: Total number of blocks.
        :param dtype: Integer dtype the count arrays are held in, which bounds
            their memory. Wide enough to hold the busiest block's site count.
        :return: ``(counts, called, block_sites)``. ``counts`` and ``called``
            are ``(n(n-1)/2, n_blocks)`` in ``dtype``, with one row per
            haplotype pair in :func:`itertools.combinations` order: the
            pairwise differences per block, and the number of the block's sites
            at which both haplotypes are called. ``block_sites`` is the
            ``(n_blocks,)`` int64 site count of each block, so the count's
            exposure is ``called / block_sites``. ``called`` is ``None`` when
            no call is missing, meaning an exposure of 1 everywhere.
        """
        pairs = list(combinations(range(self.n_haplotypes), 2))
        counts = np.zeros((len(pairs), n_blocks), dtype=dtype)
        sites_per_block = np.bincount(block_of_site, minlength=n_blocks)
        has_sites = sites_per_block > 0
        # A block listing no site says nothing about a pair's coverage, so it
        # carries its nearest listed neighbour's exposure, the left one on a
        # tie, rather than a full block's: otherwise a pair uncalled over a
        # stretch reads it as a run of zero differences.
        source = np.arange(n_blocks)
        if has_sites.any() and not has_sites.all():
            idx = np.flatnonzero(has_sites)
            empty = np.flatnonzero(~has_sites)
            right = np.searchsorted(idx, empty)
            left = idx[np.maximum(right - 1, 0)]
            right = idx[np.minimum(right, idx.size - 1)]
            take_left = (left < empty) & ((right < empty)
                                          | (empty - left <= right - empty))
            source[~has_sites] = np.where(take_left, left, right)
        block_sites = np.maximum(sites_per_block[source], 1).astype(np.int64)
        any_missing = bool((genotypes < 0).any())
        called = np.zeros((len(pairs), n_blocks), dtype=dtype) \
            if any_missing else None
        for p, (a, b) in enumerate(pairs):
            ga, gb = genotypes[:, a], genotypes[:, b]
            if any_missing:
                both = (ga >= 0) & (gb >= 0)
                differ = (ga != gb) & both
                called[p] = np.bincount(block_of_site[both],
                                        minlength=n_blocks)[source]
            else:
                differ = ga != gb
            if differ.any():
                counts[p] = np.bincount(block_of_site[differ],
                                        minlength=n_blocks).astype(dtype)
        return counts, called, block_sites

    def block_tmrcas(
        self, genotypes: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
        seg_bounds: "Sequence[tuple[int, int]] | None" = None,
        step: "np.ndarray | None" = None,
        block_mask: "np.ndarray | None" = None,
        emit_scale: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Posterior-mean TMRCA per (pair, block).

        :param genotypes: ``(n_sites, n_haplotypes)`` int8 allele indices
            (0..3), ``-1`` for missing.
        :param block_of_site: ``(n_sites,)`` block index per site.
        :param n_blocks: Total number of blocks.
        :param seg_bounds: Optional ``(start, end)`` block-index ranges that
            partition ``[0, n_blocks)`` in order. The forward-backward runs
            independently within each range (the ``Δ → ∞`` recombination
            reset), while the time grid and per-pair prior stay global.
            Defaults to one segment over all blocks.
        :param step: Optional ``(n_blocks-1,)`` genetic distance of each
            inter-block step in nominal-block units, the local recombination
            rate over the interval divided by ``rec_rate·block_size``.
            ``None`` is a uniform map (all ones).
        :param block_mask: Optional ``(n_blocks,)`` bool of callable blocks.
            ``None`` is all callable. A non-callable block emits uniformly,
            while recombination still accumulates across it.
        :param emit_scale: Optional ``(n_blocks,)`` Poisson exposure scale of
            each block, the callable base-pair fraction of the block times its
            local mutation rate over the reference ``μ``. ``None`` is all ones
            (fully accessible at the reference rate). The calibration uses the
            same scaling.
        :return: ``(n(n-1)/2, n_blocks)`` posterior-mean TMRCA (generations),
            one row per haplotype pair in :func:`itertools.combinations` order.
        """
        counts, called, block_sites = self._pair_block_counts(
            genotypes, block_of_site, n_blocks, dtype=np.int64)
        self.called, self.block_sites = called, block_sites

        if seg_bounds is None:
            seg_bounds = [(0, n_blocks)]

        edges, t_rep, lam, log_lam, r, t_bar_all = self.calibrate(
            counts, emit_scale=emit_scale, block_mask=block_mask,
            called=called, block_sites=block_sites)

        out = np.zeros((counts.shape[0], n_blocks), dtype=np.float64)
        for p in range(counts.shape[0]):
            pi = self._coalescent_prior(edges, t_bar_all[p])
            # Run the forward--backward separately on each contiguous segment
            # so the chain resets to the prior at every break (no carry-over).
            for s0, s1 in seg_bounds:
                if s1 > s0:
                    seg_step = None if step is None else step[s0:s1 - 1]
                    seg_mask = None if block_mask is None else block_mask[s0:s1]
                    seg_esc = None if called is None \
                        else called[p, s0:s1] / block_sites[s0:s1]
                    if emit_scale is not None:
                        seg_esc = emit_scale[s0:s1] if seg_esc is None \
                            else emit_scale[s0:s1] * seg_esc
                    out[p, s0:s1] = pair_posterior_mean_tmrca(
                        counts[p, s0:s1], t_rep, lam, log_lam, r, pi,
                        step=seg_step, mask=seg_mask, escale=seg_esc,
                    )
        return out


@dataclass(frozen=True)
class PairwiseTmrcas:
    """Inferred per-block pairwise TMRCAs from the local-tree HMM.

    The raw posterior-mean coalescence times the pairwise-coalescent HMM
    produces, ahead of their condensation into per-window local trees. The
    estimator's most direct intermediate, exposed for inspection and debugging
    (e.g. checking whether a pair's TMRCA track follows the expected
    recombination-driven structure, or why a window's tree is implausible).
    """

    #: Panel haplotype names, in genotype-column order.
    sample_names: tuple[str, ...]
    #: The ``(name_i, name_j)`` haplotype pairs, one per row of :attr:`tmrca`:
    #: all ``n(n-1)/2`` of them, in :func:`itertools.combinations` order over
    #: :attr:`sample_names`.
    pairs: tuple[tuple[str, str], ...]
    #: ``(n_blocks,)`` bp midpoint of each HMM block.
    block_midpoints: np.ndarray
    #: ``(n_pairs, n_blocks)`` posterior-mean TMRCA in generations.
    tmrca: np.ndarray

    def write(self, path: "str | os.PathLike") -> None:
        """Write to ``path``. Format chosen by suffix.

        ``.npz`` dumps the raw arrays (``sample_names``, ``pairs``,
        ``block_midpoints``, ``tmrca``). ``.csv`` writes a comma-separated, and
        ``.tsv`` / ``.txt`` a tab-separated, long table with columns
        ``hap_i, hap_j, block_midpoint, tmrca`` (one row per pair × block).

        :param path: Output path. Suffix selects the format.
        :raises ValueError: On an unrecognised suffix.
        """
        s = str(path).lower()
        if s.endswith(".npz"):
            self.to_npz(path)
        elif s.endswith((".tsv", ".txt")):
            self.to_csv(path, sep="\t")
        elif s.endswith(".csv"):
            self.to_csv(path, sep=",")
        else:
            raise ValueError(
                f"unrecognised suffix for pairwise-TMRCA output: {path!r}; "
                "use .npz, .csv, .tsv or .txt"
            )

    def to_npz(self, path: "str | os.PathLike") -> None:
        """Dump the raw arrays to a NumPy ``.npz`` archive, keyed by field name."""
        np.savez(
            path,
            sample_names=np.asarray(self.sample_names),
            pairs=np.asarray(self.pairs),
            block_midpoints=self.block_midpoints,
            tmrca=self.tmrca,
        )

    def to_csv(self, path: "str | os.PathLike", *, sep: str = ",") -> None:
        """Write a long ``hap_i, hap_j, block_midpoint, tmrca`` table."""
        import csv

        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=sep)
            w.writerow(["hap_i", "hap_j", "block_midpoint", "tmrca"])
            mids = self.block_midpoints
            for (hi, hj), row in zip(self.pairs, self.tmrca):
                for mid, t in zip(mids, row):
                    w.writerow([hi, hj, f"{mid:g}", f"{t:.6g}"])


#: Smallest emission scale a block may carry. Zero would make the emission
#: monotone in the coalescent rate, sending any block with a difference to
#: the deepest time bin.
_EMIT_SCALE_FLOOR: float = 1e-12

#: Smallest accessible base-pair fraction at which a block still emits. A block
#: below it contributes an expected difference count proportional to that
#: fraction, so its likelihood is negligible, while the calibration divides its
#: count by the fraction and would give it up to ``1 / _ACCESS_FRAC_FLOOR``
#: times a full block's leverage on the time grid and on the per-pair prior
#: mean. Below the floor a block emits uniformly and only carries
#: recombination distance.
_ACCESS_FRAC_FLOOR: float = 0.05


class LocalTreeBuilder(ReprMixin):
    """Infer local dated trees from genotypes → a :class:`tskit.TreeSequence`.

    Runs :class:`~ancestree.local_tree_inference.PairwiseCoalescentHMM`, then condenses each window's pairwise
    TMRCAs into one ultrametric local tree by UPGMA. The emitted tree sequence
    is self-describing:
    sample names live in individual metadata and the observed genotypes are
    baked in as sites (minimal :meth:`tskit.Tree.map_mutations` placement per
    site), so the ``.trees`` file can be handed straight to
    :class:`~ancestree.inference.ARGBasedInference`.

    :param sites: Polymorphic :class:`~ancestree.sites.Site` records (a
        list or any :class:`~ancestree.sites.SiteSource`). Only genotypes
        + positions are read. Ancestral state is never consumed. Sites must
        be supplied in sorted genomic order (ascending position within each
        contig): the pairwise-HMM blocking and the streaming / chunked path
        assume monotonic positions and do not sort the input themselves.
    :param mu: Per-site per-generation mutation rate.
    :param rec_rate: Per-site per-generation recombination rate.
    :param sample_names: Panel haplotype names (tip order). Each must be a
        key of some site's
        :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`, and a
        site missing the key reads it as uncalled.
    :param sequence_length: Region length in bp (tskit coordinates).
    :param window: Local-tree window in bp, an int or a string with a
        ``bp`` / ``kb`` / ``mb`` suffix, or ``"<N>snp"`` to size the window so
        it averages N SNPs at the data's mean density (default ``"8snp"``).
        One tree per window, shared by its SNPs. The primary resolution parameter.
        A window cannot be finer than one block, so it is floored to
        ``block_size`` and snapped up to a whole number of blocks.
        :class:`~ancestree.local_tree_inference.LocalTreeInference` warns when a spec is floored this way.
    :param block_size: HMM emission block width, an int / bp string, or a
        ``"<N>snp"`` spec (N SNPs' worth of span at the data's mean density, the
        natural density-invariant unit). ``None`` (default) sizes the block to a
        few SNPs' worth of span (well below the window), so the HMM emits several
        blocks per output tree. Set explicitly to override the density-adaptive
        default.
    :param n_time_bins: Discretized coalescent-time bins.
    :param time_grid: Explicit bin edges in generations, ascending and strictly
        positive, replacing the data-calibrated grid and overriding
        ``n_time_bins``. At most ``MAX_TIME_BINS + 1`` edges may be given.
    :param bake_genotypes: When ``True`` (default), the observed genotypes
        are written into the tree sequence as sites (minimal
        :meth:`tskit.Tree.map_mutations` placement per site), making the ``.trees`` file
        self-contained and directly consumable by
        :class:`~ancestree.inference.ARGBasedInference`. Set ``False`` for
        a smaller, topology-only tree sequence, which carries no genotypes and
        so cannot be annotated on its own.
    :param segment_breaks: Optional bp positions at which the coordinate axis
        is severed into independent segments (e.g.\\ contig boundaries when
        several contigs are packed into one region). The pairwise HMM is reset
        at each break (no information crosses it) and windows are tiled per
        segment so none straddles a break. Breaks are snapped to the block
        grid. Defaults to a single contiguous segment.
    :param recombination_map: Optional :class:`msprime.RateMap` (in this
        region's bp coordinates) driving the HMM's TMRCA-reset rate. When
        given, the per-block transition uses the map's local genetic distance
        in place of the constant ``rec_rate``, so recombination hotspots
        reset the local TMRCA more readily and cold spots less. ``rec_rate``
        then acts only as the reference scale (it cancels from the transition)
        and the deserts/hotspots are read from the map.
    :param accessibility: Optional sequence of half-open ``(start, end)`` bp
        intervals marking the callable genome (a BED-style accessibility mask).
        Each block's Poisson emission is scaled by its accessible base-pair
        fraction, and a block whose fraction does not exceed 0.05 emits
        uniformly in the HMM. See :meth:`PairwiseCoalescentHMM.block_tmrcas`.
        Defaults to ``None`` (every block callable).
    :param mutation_map: Optional :class:`msprime.RateMap` (in this region's bp
        coordinates) giving the local per-site mutation rate, with ``mu`` the
        reference rate the map is expressed against. The HMM emission of a
        block uses the map's mean rate over it (see
        :meth:`PairwiseCoalescentHMM.block_tmrcas`), and the kernel scales
        the branches of each site by the map's rate at that site. Both take
        ``mu`` over an interval the map leaves undefined (NaN) and past its
        end. Defaults to ``None`` (a uniform rate).
    :param validate_coverage: When ``True`` (default), raise if any declared
        sample is observed at zero sites, the signature of a ``sample_names`` /
        :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>` key
        mismatch. Per-chunk sub-builders pass ``False`` and the chunked driver
        checks the full panel once.
    :param window_anchor: Optional window intervals in the builder's own frame,
        supplied by a segmented run so every segment reads on one grid laid
        over the whole region.
    :param position_base: Coordinate of the region's first base in the sites'
        positions, ``1`` for VCF / VCF-Zarr sites. The region's bases then lie
        on ``[position_base, sequence_length + position_base)``, the axis the
        tree sequence is laid on. Defaults to ``0``.
    :raises ValueError: If ``mu`` or ``rec_rate`` is not positive, if the
        sites span more than one contig, if fewer than two samples are given,
        if ``n_time_bins`` is outside ``[1, MAX_TIME_BINS]``, or if a site lies
        off the axis.
    """

    #: Set by a caller that does its own user-facing logging, so a builder it
    #: drives per segment stays silent.
    _quiet: bool = False

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"window_bp": self.window_bp, "n_time_bins": self.n_time_bins}

    def __init__(
        self,
        sites: "Iterable[Site] | SiteSource",
        *,
        mu: float,
        rec_rate: float,
        sample_names: Sequence[str],
        sequence_length: float,
        window: "int | str" = "8snp",
        block_size: "int | str | None" = None,
        n_time_bins: int = 32,
        time_grid: "np.ndarray | None" = None,
        bake_genotypes: bool = True,
        segment_breaks: "Sequence[float] | None" = None,
        recombination_map=None,
        accessibility: "Sequence[tuple[float, float]] | None" = None,
        mutation_map=None,
        validate_coverage: bool = True,
        window_anchor: "Sequence[tuple[float, float]] | None" = None,
        position_base: int = 0,
    ) -> None:
        if not np.isfinite(mu):
            raise ValueError(f"mu must be positive and finite, got {mu}")
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if rec_rate <= 0:
            raise ValueError(f"rec_rate must be positive, got {rec_rate}")
        self._n_unrepresentable_sites = 0
        self._n_unrepresentable_tips = 0
        if isinstance(sites, SiteTable):
            self.sites = sites
            for _ in self._tally_unrepresentable(sites):
                pass
        else:
            # Columnar: from_sites consumes the input one site at a time.
            self.sites = SiteTable.from_sites(
                self._tally_unrepresentable(sites), sample_names)
        _chroms = self.sites.chroms
        if len(_chroms) > 1:
            raise ValueError(
                f"LocalTreeBuilder builds a single contiguous coordinate axis, "
                f"but the sites span {len(_chroms)} contigs "
                f"({sorted(_chroms)}). Local trees are per-contig: pass one "
                f"contig at a time, or drive multi-contig data through "
                f"LocalTreeInference(chunk_size=...), which severs contigs "
                f"automatically."
            )
        self.mu = float(mu)
        self.rec_rate = float(rec_rate)
        self.sample_names: list[str] = list(sample_names)
        if len(self.sample_names) < 2:
            raise ValueError(f"need >= 2 samples, got {len(self.sample_names)}")
        self.sequence_length = float(sequence_length)
        self.position_base = int(position_base)
        self._check_on_axis(self.sites.pos, self.axis_length)
        _, self.block_size, self.window_bp = self._resolve_widths(
            window, block_size, len(self.sites), self._site_extent)
        self.time_grid = PairwiseCoalescentHMM._check_time_grid(time_grid)
        if self.time_grid is not None:
            n_time_bins = self.time_grid.size - 1
        PairwiseCoalescentHMM._check_n_time_bins(n_time_bins)
        self.n_time_bins = int(n_time_bins)
        self.bake_genotypes = bool(bake_genotypes)
        self._validate_coverage = bool(validate_coverage)
        # Windows this build reads on, in its own frame, or None to tile.
        self._window_anchor = window_anchor
        self.segment_breaks: list[float] = (
            list(segment_breaks) if segment_breaks else []
        )
        self.recombination_map = recombination_map
        self.accessibility: "list[tuple[float, float]] | None" = (
            list(accessibility) if accessibility is not None else None
        )
        self.mutation_map = mutation_map
        self._ts = None
        self._pair_block_t: "np.ndarray | None" = None
        self._pair_called: "np.ndarray | None" = None
        self._pair_block_sites: "np.ndarray | None" = None

    @property
    def axis_length(self) -> float:
        """Length of the tree sequence's axis, which holds every base of the
        region: :attr:`sequence_length` plus :attr:`position_base`."""
        return self.sequence_length + self.position_base

    @staticmethod
    def _check_on_axis(positions: np.ndarray, axis: float) -> None:
        """Refuse site positions off the axis ``[0, axis)``.

        :param positions: Site positions on the axis.
        :param axis: Length of the axis, :attr:`axis_length`.
        :raises ValueError: If a position is negative or not below ``axis``.
        """
        positions = np.asarray(positions)
        oob = (positions < 0) | (positions >= axis)
        if np.any(oob):
            bad = positions[oob]
            raise ValueError(
                f"{bad.size} site position(s) lie outside the axis [0, {axis:g}) "
                f"of the declared region, e.g. {bad[:5].tolist()}. The region's "
                f"bases lie below sequence_length plus the source's position "
                f"base (1 for VCF / VCF-Zarr, 0 otherwise). Pass a "
                f"sequence_length that contains every site, or restrict the "
                f"sites to the region.")

    def _tally_unrepresentable(self, sites: "Iterable[Site]") -> "Iterator[Site]":
        """Pass sites through, tallying alleles outside the alphabet.

        A site counts once when its allele list carries such an allele or one
        of its tips is observed at one. The tips are counted individually.

        :param sites: Iterable of :class:`~ancestree.sites.Site`.
        :return: The same sites, in order.
        """
        for site in sites:
            n_tips = site.n_unrepresentable_tips()
            self._n_unrepresentable_tips += n_tips
            if n_tips or site.has_unrepresentable_allele:
                self._n_unrepresentable_sites += 1
            yield site

    @property
    def _site_extent(self) -> float:
        """The bp from the first site to the last, inclusive, over which the
        SNP density is taken, or :attr:`sequence_length` without sites."""
        pos = self.sites.pos
        return (float(pos[-1] - pos[0] + 1) if len(pos)
                else self.sequence_length)

    @staticmethod
    def _resolve_widths(window, block_size, n_sites: int,
                        span_bp: float) -> tuple[int, int, int]:
        """The requested window, the block and the window the HMM uses, in bp.

        :param window: The window spec.
        :param block_size: The block spec, or ``None`` for the default.
        :param n_sites: Sites the density is taken over.
        :param span_bp: Span the density is taken over.
        :return: ``(requested, block, window)``, the window rounded up to a
            whole number of blocks.
        """
        requested = _width_bp(window, n_sites, span_bp, name="window")
        block = (_default_block_bp(n_sites, span_bp, requested)
                 if block_size is None
                 else _width_bp(block_size, n_sites, span_bp, name="block_size"))
        return requested, block, LocalTreeBuilder._snap_to_block(requested, block)

    @staticmethod
    def _snap_to_block(window_bp: int, block_size: int) -> int:
        """Round a window width up to a whole number of blocks.

        :param window_bp: The requested width.
        :param block_size: Block width in base pairs.
        :return: The window width the HMM uses, at least one block.
        """
        bp = max(window_bp, block_size)
        return int(np.ceil(bp / block_size) * block_size)

    def _genotype_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Build the ``(n_sites, n_hap)`` allele-index matrix + block ids.

        :return: ``(genotypes, positions, block_of_site, n_blocks)`` where
            ``positions`` is each site's tskit coordinate, its own
            ``site.pos`` on ``[0, axis_length)``, baked and emitted at the
            same coordinate so the round-trip is position-stable.
        """
        n_sites = len(self.sites)
        # The table holds exactly this matrix, in this column order.
        if tuple(self.sites.sample_names) == tuple(self.sample_names):
            g = self.sites.genotypes.copy()
        else:  # a builder re-ordered or subset the panel
            n_hap = len(self.sample_names)
            src = {n: i for i, n in enumerate(self.sites.sample_names)}
            g = np.full((n_sites, n_hap), -1, dtype=np.int8)
            for hi, name in enumerate(self.sample_names):
                j = src.get(name)
                if j is not None:
                    g[:, hi] = self.sites.genotypes[:, j]
        positions = self.sites.pos.astype(np.float64)
        block_of_site = (self.sites.pos // self.block_size).astype(np.int64)
        if n_sites and self._validate_coverage:
            seen = (g >= 0).any(axis=0)
            if not seen.all():
                missing = [self.sample_names[i]
                           for i in np.nonzero(~seen)[0]]
                LocalTreeInference._raise_unobserved_samples(missing)
        n_blocks = max(1, int(np.ceil(self.sequence_length / self.block_size)))
        # The last block holds the bases past its nominal end up to the axis's.
        np.minimum(block_of_site, n_blocks - 1, out=block_of_site)
        return g, positions, block_of_site, n_blocks

    def _segment_intervals(self) -> list[tuple[float, float]]:
        """Contiguous ``(start, end)`` bp segments, breaks snapped to blocks.

        ``segment_breaks`` partition ``[0, sequence_length)``. Snapping each to
        the block grid keeps every block within a single segment, so the HMM
        reset and the window tiling agree.
        """
        bs = self.block_size
        cuts = {0.0, float(self.sequence_length)}
        for b in self.segment_breaks:
            snapped = round(float(b) / bs) * bs
            if 0.0 < snapped < self.sequence_length:
                cuts.add(float(snapped))
        edges = sorted(cuts)
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)
                if edges[i + 1] > edges[i]]

    def _seg_block_bounds(self, n_blocks: int) -> list[tuple[int, int]]:
        """Block-index ranges per segment, for :meth:`block_tmrcas`.

        Starts floor down, ends ceil up, matching the ``ceil``-based
        ``n_blocks`` so the segments partition ``[0, n_blocks)`` exactly and
        the final block is always covered. Internal segment breaks are snapped
        to block multiples upstream, so floor and ceil agree there (no overlap).
        """
        out = []
        for s0, s1 in self._segment_intervals():
            b0 = min(n_blocks, max(0, int(s0 // self.block_size)))
            b1 = min(n_blocks, max(0, int(np.ceil(s1 / self.block_size))))
            if b1 > b0:
                out.append((b0, b1))
        return out

    def _window_intervals(self) -> list[tuple[float, float]]:
        """Tile each segment into equal-width windows (no runt remainder).

        Each segment of length ``Lₛ`` is split into ``max(1, round(Lₛ / w))``
        windows of width ``Lₛ / n`` (``w`` = target window bp), so every window
        is close to the target and no segment leaves a tiny end window.
        Windows never cross a segment break, and the last one reaches
        :attr:`axis_length`.
        """
        w = self.window_bp
        iv: list[tuple[float, float]] = []
        if self._window_anchor is not None:
            return list(self._window_anchor)
        for s0, s1 in self._segment_intervals():
            seg_len = s1 - s0
            n = max(1, int(round(seg_len / w)))
            step = seg_len / n
            for i in range(n):
                left = s0 + i * step
                right = s1 if i == n - 1 else s0 + (i + 1) * step
                iv.append((float(left), float(right)))
        if iv:
            iv[-1] = (iv[-1][0], self.axis_length)
        return iv

    def _block_geometry(
        self, n_blocks: int,
    ) -> "tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]":
        """Per-block HMM geometry: ``(step, block_mask, emit_scale)``.

        ``step[k]`` is the genetic distance of the step from block ``k`` to
        ``k+1`` in nominal-block units, the recombination map's local mass
        over that bp interval divided by the reference ``rec_rate·block_size``;
        ``None`` when no map is set (uniform).

        ``emit_scale[b]`` is the Poisson exposure scale of block ``b``: the
        product of ``a_b·(S_b/B)``, the accessible base pairs of block ``b`` as
        a fraction of the nominal width ``B`` of :attr:`block_size`, for a
        block spanning ``S_b`` base pairs and an accessible fraction ``a_b`` of
        that span (dimensionless, in ``[0, 1]``, and ``1`` with no mask on a
        full-width block), and ``μ_b / μ``, the mutation map's local per-bp
        rate over the block divided by the reference rate (dimensionless, and
        ``1`` with no map). The emission mean of block ``b`` in time bin ``i``
        is then ``a_b·(S_b/B)·(μ_b/μ)·2·μ·B·t_i`` differences, for a bin
        representative time ``t_i`` in generations, so the exposure is measured
        against the same ``B`` the mean is built from.
        ``None`` when neither an accessibility mask nor a mutation map is set.

        ``block_mask[b]`` is ``True`` when block ``b`` emits: more than
        :data:`_ACCESS_FRAC_FLOOR` of its own base pairs are accessible and its
        local mutation rate is positive. ``None`` when neither an
        accessibility mask nor a zero-rate mutation-map block constrains it.

        :param n_blocks: Number of blocks the region is cut into.
        :return: ``(step, block_mask, emit_scale)``, each either an array over
            blocks or ``None`` for its neutral default.
        """
        bs = self.block_size
        step = None
        if self.recombination_map is not None:
            rm = self.recombination_map
            edges = self._block_edges(n_blocks)
            L = float(getattr(rm, "sequence_length", edges[-1]))
            # Midpoint of each block's extent within the map, so the block
            # holding the map's end is truncated there.
            edges = np.minimum(edges, L)
            lo, hi = edges[:-1], edges[1:]
            # A block is in the map when it has extent there and the map
            # reports a rate (a missing interval reports NaN).
            in_map = hi > lo
            with np.errstate(invalid="ignore"):
                probe = np.asarray(
                    rm.get_rate(np.clip(0.5 * (lo + hi), 0.0, L - 1e-9)),
                    dtype=np.float64)
            in_map &= ~np.isnan(probe)
            mids = np.where(in_map, 0.5 * (lo + hi), lo)
            cum = np.asarray(rm.get_cumulative_mass(np.clip(mids, 0.0, L)),
                             dtype=np.float64)
            gdist = np.diff(cum)  # (n_blocks-1,)
            span = np.diff(mids)  # bp the step actually spans
            # A step is only a real measurement when BOTH its blocks lie inside
            # the map. Otherwise its gdist is a spurious 0, not a cold spot.
            valid = in_map[:-1] & in_map[1:] & (span > 0)
            # Divide by the bp the step spans, so a truncated final block gets
            # the same per-bp rate.
            ref = self.rec_rate * np.clip(span, 1e-300, None)
            step = np.nan_to_num(gdist / ref, nan=1.0, posinf=1e6, neginf=0.0)
            step = np.clip(step, 0.0, None)
            # Steps beyond the map take the reference rate (1.0), so the
            # TMRCA chain keeps decorrelating.
            step = np.where(valid, step, 1.0)
        block_mask = None
        access_frac = None
        if self.accessibility is not None:
            access_frac = self._accessible_fraction(n_blocks)
            block_mask = access_frac > _ACCESS_FRAC_FLOOR
        emit_scale = None
        uninformative = None
        if self.mutation_map is not None:
            mm = self._kernel_mu()
            L = float(mm.sequence_length)
            edges = np.minimum(self._block_edges(n_blocks), L)
            lo, hi = edges[:-1], edges[1:]
            valid = hi > lo  # False past the map's end
            width = np.clip(hi - lo, 1e-300, None)
            rate = (np.asarray(mm.get_cumulative_mass(hi), dtype=np.float64)
                    - np.asarray(mm.get_cumulative_mass(lo), dtype=np.float64)
                    ) / width  # local per-bp rate
            emit_scale = np.nan_to_num(
                rate / self.mu, nan=1.0, posinf=1e6, neginf=0.0,
            )
            # Blocks beyond the map take the reference rate (1.0).
            uninformative = valid & (emit_scale <= _EMIT_SCALE_FLOOR)
            emit_scale = np.where(
                valid, np.clip(emit_scale, _EMIT_SCALE_FLOOR, None), 1.0)
            if uninformative.any():
                if block_mask is None:
                    block_mask = np.ones(n_blocks, dtype=bool)
                block_mask = block_mask & ~uninformative
                if not getattr(self, "_warned_zero_rate", False):
                    logging.getLogger(
                        f"ancestree.{type(self).__name__}"
                    ).warning(
                        "mutation-map rate is zero over %d block(s); they "
                        "carry no information about coalescent time, so "
                        "their sites are scored prior-dominated.",
                        int(uninformative.sum()),
                    )
                    self._warned_zero_rate = True
        # lambda_i is 2·mu·block_size·t_i for every block, so exposure is
        # measured against the nominal width B: a block spanning S_b < B bp
        # offers S_b/B of one whether or not a mask is given, which is what
        # keeps the masked and unmasked paths on one scale.
        trunc = np.diff(self._block_edges(n_blocks)) / float(bs)
        exposure = trunc if access_frac is None else access_frac * trunc
        # calibrate divides each block's count by its exposure, so a block
        # whose exposure is tiny carries 1/exposure times a full block's
        # weight in the shared time grid. The same floor bounds both axes.
        if not np.all(trunc > _ACCESS_FRAC_FLOOR):
            if block_mask is None:
                block_mask = np.ones(n_blocks, dtype=bool)
            block_mask = block_mask & (trunc > _ACCESS_FRAC_FLOOR)
        # A zero-rate mutation map masks blocks too, but warns and continues,
        # so it is not what this refusal is about.
        callable_mask = block_mask
        if (callable_mask is not None and uninformative is not None
                and uninformative.any()):
            callable_mask = callable_mask | uninformative
        if (callable_mask is not None and not callable_mask.any()
                and self._validate_coverage):
            raise ValueError(
                f"no block over this region is callable ({n_blocks} block(s) "
                f"of {bs} bp), so every block would emit uniformly and the "
                f"returned TMRCAs would carry no positional information while "
                f"still looking like a successful run. A block emits when more "
                f"than {_ACCESS_FRAC_FLOOR:.3g} of its own base pairs are "
                f"accessible and it spans more than that fraction of the block "
                f"width. Check the region length, the block size, and that "
                f"any mask is in the same bp coordinates as the sites.")
        if access_frac is not None or not np.allclose(trunc, 1.0):
            scale = exposure if emit_scale is None else emit_scale * exposure
            emit_scale = np.clip(scale, _EMIT_SCALE_FLOOR, None)
        return step, block_mask, emit_scale

    def _site_rate_scale(self, positions: np.ndarray) -> "np.ndarray | None":
        """Mutation rate at each site relative to :attr:`mu`.

        :param positions: Site coordinates in region coordinates ``[0, L)``.
        :return: ``(n_sites,)`` positive multipliers, a zero rate floored, or
            ``None`` without a :attr:`mutation_map`.
        """
        if self.mutation_map is None:
            return None
        mm = self._kernel_mu()
        x = np.clip(np.asarray(positions, dtype=np.float64), 0.0,
                    np.nextafter(float(mm.sequence_length), 0.0))
        rate = np.asarray(mm.get_rate(x), dtype=np.float64)
        return np.maximum(rate, _ZERO_RATE_FLOOR) / float(self.mu)

    def _kernel_mu(self):
        """The rate the HMM and the kernel score this builder's region at.

        :return: :attr:`mu`, or the :attr:`mutation_map` with the intervals
            it leaves undefined, and the stretch from its end to
            :attr:`axis_length`, at :attr:`mu`.
        """
        if self.mutation_map is None:
            return self.mu
        return self._filled(self.mutation_map, self.mu, float(self.axis_length))

    @staticmethod
    def _filled(mm, mu: float, end: float):
        """``mm`` with the intervals it leaves undefined, and the stretch from
        its end to ``end``, at ``mu``.

        :param mm: An :class:`msprime.RateMap`.
        :param mu: The rate filled in.
        :param end: The axis end the map is extended to.
        :return: ``mm`` itself when it is defined up to ``end``, else the
            filled map.
        """
        position = np.asarray(mm.position, dtype=np.float64)
        rate = np.asarray(mm.rate, dtype=np.float64)
        if not np.isnan(rate).any() and position[-1] >= end:
            return mm
        rate = np.where(np.isnan(rate), mu, rate)
        if position[-1] < end:
            position = np.append(position, end)
            rate = np.append(rate, mu)
        from ancestree._maps import MapFiles
        return MapFiles.msprime().RateMap(position=position, rate=rate)

    def _block_edges(self, n_blocks: int) -> np.ndarray:
        """Base-pair edges of the blocks on the axis.

        Block ``b`` spans ``[b·B, (b+1)·B)`` for a block width ``B`` of
        :attr:`block_size`, and a block reaching :attr:`sequence_length` ends
        at :attr:`axis_length`, so the last block holds the region's last base.

        :param n_blocks: Number of blocks the region is cut into.
        :return: ``(n_blocks + 1,)`` float64 edges, non-decreasing.
        """
        edges = np.arange(n_blocks + 1, dtype=np.float64) * self.block_size
        return np.where(edges >= self.sequence_length, self.axis_length, edges)

    def _accessible_fraction(self, n_blocks: int) -> np.ndarray:
        """Callable base-pair fraction of each block.

        :param n_blocks: Number of blocks the region is cut into, spanning the
            base pairs given by :meth:`_block_edges`.
        :return: ``(n_blocks,)`` float64 in ``[0, 1]``, the base pairs of block
            ``b`` covered by :attr:`accessibility`, given on the axis, divided
            by that block's own span.
        """
        edges = self._block_edges(n_blocks)
        end = edges[-1]
        iv = np.asarray(self.accessibility, dtype=np.float64).reshape(-1, 2)
        lo = np.clip(iv[:, 0], 0.0, end)
        hi = np.clip(iv[:, 1], 0.0, end)
        keep = hi > lo
        if not keep.any():
            return np.zeros(n_blocks, dtype=np.float64)
        lo, hi = lo[keep], hi[keep]
        order = np.argsort(lo, kind="stable")
        lo, hi = lo[order], hi[order]
        # Merge overlapping intervals, so a base pair two of them cover counts
        # once.
        starts = np.empty(lo.size, dtype=bool)
        starts[0] = True
        starts[1:] = lo[1:] > np.maximum.accumulate(hi)[:-1]
        at = np.flatnonzero(starts)
        lo, hi = lo[at], np.maximum.reduceat(hi, at)
        # Covered bp below x is sum_i [min(hi_i, x) - min(lo_i, x)], evaluated
        # at every block boundary and differenced.
        below = np.empty((2, n_blocks + 1), dtype=np.float64)
        for row, bound in enumerate((lo, hi)):
            cum = np.concatenate(([0.0], np.cumsum(bound)))
            j = np.searchsorted(bound, edges, side="right")
            below[row] = cum[j] + (bound.size - j) * edges
        span = np.clip(np.diff(edges), 1e-300, None)
        return np.clip(np.diff(below[1] - below[0]) / span, 0.0, 1.0)

    def _blocks_for_window(
        self, left: float, right: float, n_blocks: int,
    ) -> tuple[int, int]:
        """Block index range ``[b0, b1)`` covered by ``[left, right)``."""
        b0 = int(left // self.block_size)
        b1 = int(np.ceil(right / self.block_size))
        b0 = max(0, min(b0, n_blocks - 1))
        b1 = max(b0 + 1, min(b1, n_blocks))
        return b0, b1

    def to_tree_sequence(self):
        """Build (and cache) the self-describing local-tree sequence.

        :return: A :class:`tskit.TreeSequence`, one local tree per window,
            branch lengths in generations, sample names in individual
            metadata, observed genotypes baked in as sites.
        :raises ValueError: If two sites share a position, which tskit forbids.
            Multi-allelic sites split across records at one position are the
            usual cause.
        """
        if self._ts is not None:
            return self._ts
        import tskit

        g, pos0, block_of_site, n_blocks = self._genotype_matrix()
        # tskit requires unique site positions.
        if pos0.shape[0] > 1:
            uniq_pos, counts = np.unique(pos0, return_counts=True)
            if np.any(counts > 1):
                dup = uniq_pos[counts > 1]
                raise ValueError(
                    f"LocalTreeBuilder: {int((counts > 1).sum())} site "
                    f"position(s) are duplicated (e.g. {float(dup[0])}); tskit "
                    f"requires unique site positions. De-duplicate the sites "
                    f"(one record per position) before building; multi-allelic "
                    f"sites split across records at the same position are the "
                    f"usual cause."
                )
        pair_block_t = self._pairwise_block_tmrcas(g, block_of_site, n_blocks)

        intervals = self._window_intervals()
        n = len(self.sample_names)
        # Exact UPGMA over the full pairwise-TMRCA matrix (combinations order).
        bounds = [self._blocks_for_window(l, r, n_blocks) for l, r in intervals]
        condensed = [pair_block_t[:, b0:b1].mean(axis=1) for b0, b1 in bounds]
        from ancestree._ensemble import window_exposure
        topo = WindowedUpgmaTreeSequence(
            n, self.axis_length,
            intervals, condensed, sample_names=self.sample_names,
            window_weights=window_exposure(
                self._pair_called, self._pair_block_sites,
                np.array([b0 for b0, _ in bounds], dtype=np.int64),
                np.array([b1 for _, b1 in bounds], dtype=np.int64)),
        ).build()
        if not self.bake_genotypes:
            self._ts = topo
            return self._ts

        # Bake the genotypes in as sites, each with a minimal map_mutations
        # placement, so ts.variants() reproduces them.
        tables = topo.dump_tables()
        alleles = list(STATES)
        order = np.argsort(pos0, kind="stable")
        si = 0
        n_uncalled = 0
        n_sites = pos0.shape[0]
        for tree in topo.trees():
            right = tree.interval.right
            while si < n_sites:
                idx = order[si]
                p = pos0[idx]
                if p >= right:
                    break
                row = g[idx].astype(np.int8)
                if not (row >= 0).any():
                    # map_mutations rejects an all-missing column, and emitting
                    # the site without a mutation would read as every tip
                    # carrying the ancestral allele. It is left out; callers
                    # pair by position, not by order.
                    n_uncalled += 1
                    si += 1
                    continue
                anc, muts = tree.map_mutations(row, alleles)
                site_id = tables.sites.add_row(position=float(p),
                                               ancestral_state=anc)
                mut_ids: list[int] = []
                for mut in muts:
                    parent = (mut_ids[mut.parent]
                              if mut.parent != tskit.NULL else tskit.NULL)
                    mut_ids.append(tables.mutations.add_row(
                        site=site_id, node=mut.node,
                        derived_state=mut.derived_state, parent=parent,
                    ))
                si += 1
        if n_uncalled:
            self._log.warning(
                "%d of %d sites carry no called genotype in the panel; they "
                "are emitted without a mutation and score at the prior",
                n_uncalled, n_sites)
        tables.sort()
        tables.time_units = "generations"
        self._ts = tables.tree_sequence()
        return self._ts

    def _pairwise_block_tmrcas(
        self, g: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
    ) -> np.ndarray:
        """Run (and cache) the pairwise-coalescent HMM.

        :return: ``(n_rows, n_blocks)`` posterior-mean TMRCA matrix, with
            ``n_rows = n(n-1)/2`` in :func:`itertools.combinations` order.
            Cached, so :meth:`to_tree_sequence` and :meth:`pairwise_tmrcas`
            share one run.
        """
        if self._pair_block_t is not None:
            return self._pair_block_t
        n_hap = len(self.sample_names)
        if not self._quiet:
            n_all = n_hap * (n_hap - 1) // 2
            self._log.info(
                "Running pairwise-coalescent HMM (%s pairs × %s blocks)",
                f"{n_all:,}", f"{n_blocks:,}",
            )
        hmm = PairwiseCoalescentHMM(
            n_hap, mu=self.mu, rec_rate=self.rec_rate,
            block_size=self.block_size, n_time_bins=self.n_time_bins,
            time_grid=self.time_grid,
        )
        step, block_mask, emit_scale = self._block_geometry(n_blocks)
        self._pair_block_t = hmm.block_tmrcas(
            g, block_of_site, n_blocks,
            seg_bounds=self._seg_block_bounds(n_blocks),
            step=step, block_mask=block_mask, emit_scale=emit_scale,
        )
        self._pair_called, self._pair_block_sites = hmm.called, hmm.block_sites
        return self._pair_block_t

    def pairwise_tmrcas(self) -> PairwiseTmrcas:
        """Inferred per-block pairwise TMRCAs (the raw HMM output).

        Runs (and caches) the pairwise-coalescent HMM and returns its
        posterior-mean TMRCA matrix with pair and block labels, the
        intermediate that :meth:`to_tree_sequence` condenses into per-window
        trees. Exposed for inspection and debugging of genealogy quality;
        :meth:`PairwiseTmrcas.write` dumps it to file.

        All ``n(n-1)/2`` pairs, in :func:`itertools.combinations` order.

        :return: A :class:`~ancestree.local_tree_inference.PairwiseTmrcas` record.
        """
        g, _pos0, block_of_site, n_blocks = self._genotype_matrix()
        pair_block_t = self._pairwise_block_tmrcas(g, block_of_site, n_blocks)
        names = self.sample_names
        row_pairs = (list(combinations(range(len(names)), 2)))
        pairs = tuple((names[a], names[b]) for a, b in row_pairs)
        starts = np.arange(n_blocks) * self.block_size
        ends = np.minimum(starts + self.block_size, self.sequence_length)
        midpoints = 0.5 * (starts + ends)
        return PairwiseTmrcas(tuple(names), pairs, midpoints, pair_block_t)

    def write(self, path: "str | os.PathLike") -> None:
        """Dump the inferred local-tree sequence to a ``.trees`` file."""
        self.to_tree_sequence().dump(str(path))


class LocalTreeInference(Inference):
    """Infer ancestral alleles from genotypes via inferred local trees.

    Builds the local trees from genotypes with
    :class:`~ancestree.local_tree_inference.LocalTreeBuilder` and scores each
    site over an ensemble of genealogies drawn from the HMM posterior. With
    ``n_ensemble=None`` it scores the single plug-in genealogy through
    :class:`~ancestree.inference.ARGBasedInference`.

    .. code-block:: python

        import ancestree as anc

        inf = anc.LocalTreeInference(
            "snps.vcf.gz", anc.JC69(), mu=1e-8, rec_rate=1e-8,
            outgroup_samples=["o1", "o2"])
        inf.to_vcf("snps.annotated.vcf.gz")

    :param source: Genotype data the trees are inferred from: a VCF / BCF /
        VCZ path, a :class:`~ancestree.sites.SiteSource`, or polymorphic
        :class:`~ancestree.sites.Site` records. A ``.trees`` path or
        :class:`tskit.TreeSequence` is read for its genotypes alone, with a
        warning that :meth:`Inference.from_arg() <ancestree.inference.Inference.from_arg>`
        scores its genealogy. When building from sites,
        they must be in sorted genomic order (ascending position within each
        contig. See :class:`~ancestree.local_tree_inference.LocalTreeBuilder`). A VCF/VCZ path's haplotype
        panel is read from the file, so ``sample_names`` may be omitted.
    :param model: Substitution model for the kernel. Defaults to
        :class:`~ancestree.models.JC69`.
    :param mu: Per-site per-generation rate (also the branch
        :attr:`Tree.time_scale <ancestree.trees.Tree.time_scale>`, since
        inferred branches are in generations). Species-specific: omitting it falls back to ``1e-8`` and logs
        a warning.
    :param rec_rate: Recombination rate, likewise falling back to
        ``1e-8`` with a warning.
    :param sample_names: Panel haplotype names (tip order). Required when
        building from :class:`~ancestree.sites.Site` records. Read from the
        source for a VCF / BCF / VCZ path, a tree sequence or a
        :class:`~ancestree.sites.SiteSource`. A haplotype uncalled at every
        site is left out of the panel.
    :param sample_filter: Restrict a VCF / VCZ read to these sample names,
        each of which must be in ``ingroup_samples`` or ``outgroup_samples``.
    :param chrom_filter: Restrict a VCF / VCZ read to this contig.
    :param ploidy: Haplotypes read per VCF sample. ``None`` (default) detects
        it from the first record.
    :param phased: Whether the genotypes are phased, so haplotype ``k`` of an
        individual is the same lineage at every site. ``None`` (default) reads
        the per-record phase flag.
    :param phase_seed: Seed for the per-site haplotype-order randomisation
        applied to unphased heterozygotes, so a run is reproducible.
    :param sequence_length: Region length ``L`` in bp, which the local-tree
        windows tile as ``[0, L)``, or a ``{contig: L}`` mapping giving each
        contig its own. For 1-based input (VCF / VCZ) the axis is
        ``[0, L + 1)``, so a site at ``POS == L`` lies on it. A single length
        applies to every contig. Defaults to a tree-sequence source's own
        length, or to the contig lengths of a VCF header or VCZ store. Under
        ``chunk_size`` the length is what lays one window grid over a contig
        and not one per segment, and a contig with none is tiled to its last
        site.
    :param window: Local-tree window spec (the primary resolution parameter). See
        :class:`~ancestree.local_tree_inference.LocalTreeBuilder`.
    :param block_size: HMM emission block width: int / bp string / ``"<N>snp"``;
        ``None`` (default) sizes the block to a few SNPs' worth of span (several
        blocks per tree). See :class:`~ancestree.local_tree_inference.LocalTreeBuilder`.
    :param n_time_bins: Discretized coalescent-time bins.
    :param time_grid: Explicit bin edges in generations, ascending and strictly
        positive, replacing the data-calibrated grid and overriding
        ``n_time_bins``. At most ``MAX_TIME_BINS + 1`` edges may be given.
    :param prior: Optional root :class:`~ancestree.priors.StationaryPrior`;
        ``None`` (default) applies the π of ``base_composition``, uniform
        when none is supplied.
    :param base_composition: Optional base composition for the prior.
    :param progress: Show a tqdm bar, over segments on the chunked path and
        over trees on a single-region build.
    :param n_workers: Fork-pool workers. On the single-region path this
        parallelises only the Felsenstein pass, not the tree construction that dominates
        this mode's runtime. With ``chunk_size`` set it instead fans
        whole segments across the pool, so peak memory scales with
        ``n_workers``. Requires the ``fork`` start method (falls back to
        single-threaded with a warning).
    :param chunk_size: Segment span (bp int or ``"5mb"``-style string), raised
        to the resolved block width when narrower, which shifts the posteriors
        as any change of segmentation does, or
        ``None`` to build the whole input as one region. Chunked, one
        contiguous segment is processed at a time, so peak memory tracks the
        segment span and ``n_workers`` fans whole segments out. Contigs are
        independent segments, and a contig longer than ``chunk_size`` is
        sliced into cores of that size. A chunked and an unchunked pass agree
        closely, not exactly. Requires sites in sorted genomic order.
    :param halo: Overlap (bp int / string, or ``"auto"``) added on the
        contig-interior side of each chunk slice and then discarded, so the
        HMM decorrelates from the cut before any emitted site. ``"auto"`` uses
        the recombination correlation length. Raised to the widest window and
        to the block width, so every window overlapping a core lies whole
        inside its segment. Only relevant with ``chunk_size``.
    :param n_ensemble: Genealogies per window to marginalise the posterior
        over, drawn from the pairwise HMM's posterior. ``None`` is the plug-in
        estimate, one agglomerated tree per window. Cost is linear in this
        value.
    :param ensemble_seed: Base seed for the genealogy draws. Each member,
        segment, contig and haplotype pair draws from its own stream, hashed
        from this seed, so runs at different seeds share no draws.
    :param member_chunk: Genealogies drawn and scored at once, which sets peak
        memory. The chunk used is the largest divisor of ``n_ensemble`` no
        greater than this.
    :param recombination_map: Optional HapMap file path or
        :class:`msprime.RateMap` (in the input's bp coordinates) driving the
        HMM's TMRCA-reset rate in place of the constant ``rec_rate``.
    :param accessibility: Optional BED file path, or sequence of half-open
        ``(start, end)`` intervals (in the input's bp coordinates), marking
        the callable genome.
    :param mutation_map: Optional bedGraph file path or
        :class:`msprime.RateMap` (in the input's bp coordinates) giving the
        local mutation rate, with ``mu`` the reference rate. The HMM emission
        of a block uses the map's mean rate over it, and the kernel scales the
        branches of each site by the map's rate at that site.

        Each of the three also accepts a ``{contig: map}`` dict, whose entry
        for the sites' contig applies. A file, plain or gzip-compressed, is
        read into one entry per contig, its 0-based BED and bedGraph intervals
        placed on the 1-based positions of a VCF or VCF-Zarr source. A list of
        sites carries no coordinate base and is read as 0-based. See
        :class:`~ancestree.local_tree_inference.LocalTreeBuilder`. On the
        chunked path each map is sliced per segment.
    :param outgroup_samples: Sample ids treated as outgroups, used by the
        ``baseline_check`` comparison. Defaults to the panel samples outside
        ``ingroup_samples``. With both lists named, samples in neither are
        dropped before the local trees are inferred, or refused where
        ``sample_names`` names them. With only outgroups named, so are the
        other haplotypes of an outgroup individual.
    :param ingroup_samples: Sample ids making up the ingroup. Stratifies the
        baseline comparison by folded-SFS bin, and defines the ingroup whose
        MRCA ``focal="ingroup_mrca"`` reports at. Defaults to the panel
        samples whose individual is not an outgroup.
    :param focal: Node to report the posterior at, forwarded to the underlying
        :class:`~ancestree.inference.ARGBasedInference`. ``None`` (default) is
        the ingroup MRCA.
    :param baseline_check: INFO-log MAP agreement with the majority-outgroup
        rule (:class:`~ancestree.inference.MajorityOutgroupInference`) as a
        consistency check. Off by default. Set ``True`` together with
        ``outgroup_samples`` or ``ingroup_samples`` to enable.
    :param chrom: Contig label of every emitted site. ``None`` keeps the
        source's own.
    :raises ValueError: If ``member_chunk``, ``n_ensemble`` or ``n_workers`` is
        below 1, if ``n_time_bins`` is outside ``[1, MAX_TIME_BINS]``, if
        ``sample_names`` is needed and absent, if a named sample is absent
        from the panel or named in both lists, if ``sample_names`` names a
        sample that ``outgroup_samples`` drops or that no site's
        ``tip_alleles`` holds, if every ingroup sample is uncalled at
        every site, or if a site lies off the axis.
    """

    @staticmethod
    def _raise_unobserved_samples(missing: "list[str]") -> None:
        """Raise the standard 'sample observed at zero sites' error."""
        shown = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"{len(missing)} sample(s) have no observed genotype at any site: "
            f"[{shown}{', ...' if len(missing) > 10 else ''}]. This usually means "
            "sample_names do not match the keys of the sites' tip_alleles (e.g. "
            "names re-indexed by tskit.simplify); each declared sample must appear "
            "at ≥ 1 site."
        )

    @staticmethod
    def _uncalled_tips(sites, sample_names) -> "list[str]":
        """Full-panel coverage check: the samples that every site carries
        with no allele, such as an outgroup with no aligned base or the
        padded slot of a haploid sample in a diploid file. Early-exits once
        every sample is called, so the overhead is a handful of sites for
        well-formed input.

        :param sites: The sites, re-iterable.
        :param sample_names: The panel.
        :return: The uncalled samples, in panel order.
        :raises ValueError: If a sample is a key of no site's ``tip_alleles``.
        """
        need = set(sample_names)
        seen: set = set()
        keyed: set = set()
        any_site = False
        for site in sites:
            any_site = True
            seen.update(n for n, a in site.tip_alleles.items()
                        if a is not None and n in need)
            if len(seen) >= len(need):
                return []
            if len(keyed) < len(need):
                keyed |= need & site.tip_alleles.keys()
        if not any_site:
            return []
        absent = [n for n in sample_names if n not in keyed]
        if absent:
            LocalTreeInference._raise_unobserved_samples(absent)
        return [n for n in sample_names if n not in seen]

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"model": type(self.model), "mu": self.mu, "rec_rate": self.rec_rate,
                "window": self.window}

    def __init__(
        self,
        source: "Iterable[Site] | SiteSource | str | os.PathLike",
        model: SubstitutionModel | None = None,
        *,
        mu: float | None = None,
        rec_rate: float | None = None,
        sample_names: Sequence[str] | None = None,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        ploidy: int | None = None,
        phased: bool | None = None,
        phase_seed: int | None = None,
        sequence_length: "float | Mapping[str, float] | None" = None,
        window: "int | str" = "8snp",
        block_size: "int | str | None" = None,
        n_time_bins: int = 32,
        time_grid: "np.ndarray | None" = None,
        prior: "StationaryPrior | None" = None,
        base_composition: BaseComposition | None = None,
        progress: bool = True,
        n_workers: int = 1,
        chunk_size: "int | str | None" = "10mb",
        halo: "int | str" = "auto",
        n_ensemble: "int | None" = 64,
        ensemble_seed: int = 0,
        member_chunk: int = 8,
        recombination_map=None,
        accessibility=None,
        mutation_map=None,
        outgroup_samples: Sequence[str] | None = None,
        ingroup_samples: Sequence[str] | None = None,
        baseline_check: bool = False,
        focal: "FocalNode | str | None" = None,
        chrom: str | None = None,
    ) -> None:
        import tskit

        if mu is None:
            mu = DEFAULT_MU
            self._log.warning(
                "LocalTreeInference: no mu given, falling back to %g per site "
                "per generation. That is a human / great-ape figure, so pass "
                "mu= for your species.", DEFAULT_MU,
            )
        self.builder: LocalTreeBuilder | None = None
        self._arg: ARGBasedInference | None = None
        # The rate infer() scores each contig at, filled on first use.
        self._kernel_mus: dict = {}
        self.chrom = chrom
        self._segmented = False
        self._announced = False
        # The cached genome-wide TS on the chunked path, with its contig.
        self._stitched: "tuple[tskit.TreeSequence, str] | None" = None
        # Retained for the segmented path / provenance. Constructed here rather
        # than defaulted in the signature: models carry mutable state that would
        # otherwise be shared across inferences.
        model = model if model is not None else JC69()
        self.model = model
        if isinstance(mu, (int, float)):
            if not (np.isfinite(mu) and mu > 0):
                raise ValueError(f"mu must be positive, got {mu}")
            mu = float(mu)
        self.mu = mu
        self._require_stationary_prior(prior)
        self.prior = prior
        self.base_composition = base_composition
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1; got {n_workers}")
        self.n_workers = int(n_workers)
        self.progress = bool(progress)
        self.n_ensemble = None if n_ensemble is None else int(n_ensemble)
        self.ensemble_seed = int(ensemble_seed)
        self.member_chunk = int(member_chunk)
        if self.member_chunk < 1:
            raise ValueError("member_chunk must be a positive number of "
                             "genealogies drawn at once")
        if self.n_ensemble is not None and self.n_ensemble < 1:
            raise ValueError(
                "n_ensemble must be a positive sample size, or None for the "
                "plug-in estimate")
        self.baseline_check = bool(baseline_check)
        self._outgroup_samples: tuple[str, ...] = tuple(outgroup_samples or ())
        self._ingroup_samples: tuple[str, ...] = tuple(ingroup_samples or ())
        self.focal = FocalNode.parse(focal)
        self._note_unnamed_ingroup()

        from ancestree.sources import TskitSource

        source = SiteSource.resolve(
            source, sample_filter=sample_filter, chrom_filter=chrom_filter,
            ploidy=ploidy, phased=phased, phase_seed=phase_seed)
        if isinstance(source, TskitSource):
            self._log.warning(
                "LocalTreeInference infers local trees from the genotypes of "
                "the tree sequence and ignores its genealogy, which ARG mode "
                "(Inference.from_arg(), the `arg` subcommand) scores "
                "directly.")
            if sequence_length is None:
                sequence_length = source._ts.sequence_length
        # BED and bedGraph files are 0-based, read onto the sites' coordinates.
        base = self._position_base = getattr(source, "position_base", 0)
        self._map_paths: dict[str, str] = {}
        self.recombination_map = self._load_map(
            recombination_map, "recombination_map")
        self.accessibility = self._load_map(accessibility, "accessibility", base)
        self.mutation_map = self._load_map(mutation_map, "mutation_map", base)
        if ingroup_samples and outgroup_samples:
            self._check_filter_labelled(sample_filter, ingroup_samples,
                                        outgroup_samples)

        explicit_panel = sample_names is not None
        if sample_names is None and callable(getattr(source, "samples", None)):
            sample_names = list(source.samples())
        self._set_input_paths(source)
        if sequence_length is None:
            sequence_length = Inference._contig_lengths(self) or None
        self._declared_length = sequence_length

        if self.n_ensemble is not None and self.focal.depth is not None:
            raise NotImplementedError(
                "ensemble mode cannot place the focal node by absolute depth: "
                "the kernel reroots on each draw's own branch lengths and is "
                "not given that draw's anchor height. Use fraction= or "
                "coalescences=, or n_ensemble=None to score the plug-in "
                "genealogy.")
        # Building from genotypes: sample_names is required, rec_rate falls back.
        if sample_names is None:
            raise ValueError("building from genotypes requires sample_names")
        if rec_rate is None:
            rec_rate = DEFAULT_REC_RATE
            self._log.warning(
                "LocalTreeInference: no rec_rate given, falling back to %g per "
                "site per generation. That is a human / great-ape figure "
                "driving the HMM's TMRCA-reset transitions, so pass rec_rate= "
                "for your species.", DEFAULT_REC_RATE,
            )
        if not (np.isfinite(rec_rate) and float(rec_rate) > 0):
            raise ValueError(
                f"rec_rate must be positive, got {rec_rate}")
        self.rec_rate = float(rec_rate)
        self.window = window
        # The raw spec. None and "<N>snp" resolve from the SNP density later.
        self.block_size = block_size
        self.time_grid = PairwiseCoalescentHMM._check_time_grid(time_grid)
        if self.time_grid is not None:
            n_time_bins = self.time_grid.size - 1
        PairwiseCoalescentHMM._check_n_time_bins(n_time_bins)
        self.n_time_bins = int(n_time_bins)
        # Every width is parsed before the source is touched, so a malformed
        # spec is reported without a streaming pre-pass. A "<N>snp" spec
        # resolves against the real density later.
        if chunk_size is not None:
            self._chunk_size = _parse_bp(chunk_size, name="chunk_size")
            LocalTreeBuilder._resolve_widths(window, block_size, 1, 1.0)
            if not (isinstance(halo, str) and halo.strip().lower() == "auto"):
                _parse_bp(halo, name="halo")
        elif sequence_length is None:
            raise ValueError(
                "building requires sequence_length (--sequence-length), or "
                "chunk_size (--chunk-size) for the chunked path, which "
                "derives it per segment"
            )

        self.sample_names = list(
            self._resolve_panel(sample_names, explicit=explicit_panel))
        # A one-shot iterator is materialised, a list or SiteSource is
        # re-iterated lazily.
        if hasattr(source, "__next__"):
            source = list(source)
        # A tip uncalled at every site is no lineage, and as a tip the
        # pairwise HMM would give it the panel's median coalescence time.
        uncalled = set(self._uncalled_tips(source, self.sample_names))
        if uncalled:
            self._log.info(
                "Leaving out %d haplotype(s) that no call reaches, such as the "
                "second slot of a haploid sample in a diploid file: %s",
                len(uncalled),
                ", ".join(n for n in self.sample_names if n in uncalled))
            self.sample_names = [n for n in self.sample_names if n not in uncalled]
            ingroup = self._resolved_ingroup
            self._resolved_ingroup = tuple(
                n for n in ingroup if n not in uncalled)
            if ingroup and not self._resolved_ingroup:
                raise ValueError(
                    f"every ingroup sample is uncalled at every site: "
                    f"{list(ingroup)[:10]}")
            self._resolved_outgroups = tuple(
                n for n in self._resolved_outgroups if n not in uncalled)
        if len(self.sample_names) < len(sample_names):
            source = _PanelSites(source, self.sample_names)
        if not self._resolved_ingroup:
            self._refuse_empty_ingroup(len(self.sample_names))
        self._warn_single_ingroup(len(self._resolved_ingroup))

        if chunk_size is not None:
            # Chunked / segmented build path: defer all work to infer(), which
            # processes one contiguous segment at a time (bounded memory) and
            # severs contigs / slices large contigs with a halo overlap.
            self._segmented = True
            self._source: "Iterable[Site] | SiteSource" = source
            self._halo = ("auto" if isinstance(halo, str)
                          and halo.strip().lower() == "auto" else halo)
            return

        self._log_start()
        first = next(iter(source), None)
        maps = self._maps_on(str(first.chrom)) if first is not None else {}
        sequence_length = self._declared_on(
            None if first is None else first.chrom)
        if sequence_length is None:
            raise ValueError(
                f"sequence_length gives no length for contig "
                f"{getattr(first, 'chrom', None)!r}")
        self.builder = LocalTreeBuilder(
            source, mu=mu, rec_rate=rec_rate, sample_names=self.sample_names,
            sequence_length=sequence_length, window=window,
            block_size=block_size, n_time_bins=n_time_bins,
            time_grid=time_grid, position_base=base, **maps,
        )
        # The builder resolved a tracked (None) block_size from the SNP density.
        self.block_size = self.builder.block_size
        _n = len(self.builder.sites)
        _span = self.builder._site_extent
        self._warn_resolution(requested_bp=_width_bp(window, _n, _span,
                                                     name="window"),
                              n_sites=_n, span_bp=_span)
        self._arg_model = model
        self._arg_kwargs = dict(
            mu=mu, prior=prior,
            base_composition=base_composition, progress=progress,
            n_workers=n_workers, focal=focal,
            ingroup_samples=self._resolved_ingroup,
            outgroup_samples=self._resolved_outgroups or None,
        )

    def _kernel_mu(self, site: Site, x: float):
        """The rate :meth:`infer` scores ``site`` at: :attr:`mu`, or the
        :attr:`mutation_map` of its contig with every stretch it leaves
        undefined at :attr:`mu`.

        :param site: The site scored.
        :param x: Its position on the tree's axis.
        :return: A scalar rate or a rate map.
        """
        contig = str(site.chrom)
        if contig not in self._kernel_mus:
            if self.builder is not None:
                mu = self.builder._kernel_mu()
            else:
                from ancestree._maps import MapFiles

                mm = MapFiles.on_contig(self.mutation_map, contig,
                                        "mutation_map")
                mu = self.mu if mm is None else LocalTreeBuilder._filled(
                    mm, self.mu, float(self._axis_end(contig) or mm.sequence_length))
            self._kernel_mus[contig] = mu
        mu = self._kernel_mus[contig]
        if isinstance(mu, float) or x < float(mu.sequence_length):
            return mu
        return self.mu

    def _point_arg(self) -> "ARGBasedInference":
        """The plug-in (point-estimate) genealogy wrapped for kernel scoring.

        Built on first use, since building it runs the HMM and bakes the
        genotypes into a tree sequence.
        """
        if self._arg is None:
            if self._segmented:
                raise NotImplementedError(
                    "the plug-in ARG is per-segment on the chunked path "
                    "(chunk_size set); there is no single genome-wide one to "
                    "build. Use point_tree_sequence() for the stitched trees.")
            self._arg = ARGBasedInference(
                self.builder.to_tree_sequence(), self._arg_model,
                **{**self._arg_kwargs, "mu": self.builder._kernel_mu()})
            self._arg._quiet = True  # parent does the user-facing logging
        return self._arg

    def _log_start(self) -> None:
        """Announce the run once, before any tree building begins."""
        if self._announced:
            return
        self._announced = True
        self._log.info(
            "Inferring local trees from genotypes (%d samples, window=%s)",
            len(self.sample_names), self.window,
        )

    def infer(self) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(Site, Posterior)`` for every site, in genomic order."""
        self._log_start()
        yield from self._with_baseline_check(self._emitted(self._infer_impl()))
        if self._arg is not None:
            self._add_counts(self._arg._diagnostic_counts())
            self._arg._reset_counts()
        self._log_uniform_fallback_summary()

    def _infer_impl(self) -> Iterator[tuple[Site, Posterior]]:
        """Per-site posterior stream (segmented or single-pass build)."""
        if self._segmented:
            yield from self._infer_segmented()
        elif self._ensemble_mode():
            yield from self._infer_ensemble()
        else:
            yield from self._plug_in_stream()

    def _add_builder_unrepresentable(self) -> None:
        """Add the builder's tally of alleles outside the alphabet.

        The builder counts over the source records, whose unrepresentable tips
        the columnar encoding stores as no-calls.
        """
        self._n_unrepresentable_sites += self.builder._n_unrepresentable_sites
        self._n_unrepresentable_tips += self.builder._n_unrepresentable_tips

    def _plug_in_stream(self) -> Iterator[tuple[Site, Posterior]]:
        """Plug-in posteriors, one per site in the source's own order.

        A site with no called genotype anywhere is absent from the baked tree
        sequence and is reported at the flat posterior, as on the ensemble
        path, so the stream pairs one to one with the source sites.
        """
        sites = self.builder.sites
        self._add_builder_unrepresentable()
        scored = {int(ts_site.pos): post
                  for ts_site, post in self._point_arg().infer()}
        if len(scored) == len(sites):
            yield from ((s, scored[int(s.pos)]) for s in sites)
            return
        n_states = len(self.model.states)
        flat = np.full(n_states, 1.0 / n_states)
        for site in sites:
            post = scored.get(int(site.pos))
            yield site, (post if post is not None
                         else Posterior(tuple(self.model.states), flat))

    def _score_ensemble(self, builder, sites, keep=None,
                        segment=("", 0)) -> np.ndarray:
        """Prior-weighted per-site posteriors for one builder's sites.

        The whole of ensemble scoring, shared by the single-region and
        segmented paths, which differ only in the builder and sites they pass.

        :param builder: The :class:`~ancestree.local_tree_inference.LocalTreeBuilder` covering ``sites``.
        :param sites: Those sites, in order.
        :param keep: Boolean mask over ``sites`` of the rows returned and
            tallied, ``None`` for all.
        :param segment: ``(contig, core start)`` keying the segment's draws.
        :return: ``(n_kept, 4)`` posteriors.
        """

        from ancestree._ensemble import SegmentEnsemble
        ens = SegmentEnsemble(
            builder, self.model, len(self.sample_names),
            self._ingroup_indices(),
            pi=self._pi(), segment=segment)
        frac, n_coal = self._ensemble_placement()
        post = ens.posterior(frac, self.n_ensemble, n_coal=n_coal,
                             member_chunk=self.member_chunk,
                             seed=self.ensemble_seed)
        if len(post) != len(sites):
            raise RuntimeError(
                f"ensemble returned {len(post)} posteriors for {len(sites)} "
                f"sites; they must pair one to one")
        uniform = ens.uniform_fallback
        if keep is not None:
            post, uniform = post[keep], uniform[keep]
            sites = list(compress(sites, keep))
        self._n_uniform_fallback += int(uniform.sum())
        return self._weight_by_prior(post, sites)

    def _region_segment(self) -> tuple[str, int]:
        """The unsegmented region's stream key, that of a run chunked into
        one segment: its contig, at core start ``0``."""
        sites = self.builder.sites
        return (sites[0].chrom if sites else "", 0)

    def _infer_ensemble(self) -> Iterator[tuple[Site, Posterior]]:
        """Single-region ensemble path: one builder, no segmentation."""
        sites = self.builder.sites
        post = self._score_ensemble(self.builder, sites,
                                    segment=self._region_segment())
        self._count_ingroup_monomorphic(sites)
        self._add_builder_unrepresentable()
        for row, site in enumerate(sites):
            yield site, Posterior(tuple(STATES), post[row])

    # ----------------------------------------------------- chunked build path
    def _warn_resolution(self, *, requested_bp, n_sites, span_bp) -> None:
        """Warn when the window floored to ``block_size``, or blocks are
        SNP-starved.

        ``requested_bp`` is the pre-clamp window width (``None`` to skip the
        clamp check); ``n_sites`` / ``span_bp`` are the panel-wide SNP count and
        genome span used for the per-block SNP density, and ``0`` for either
        skips the density check only.
        """
        log = self._log
        if requested_bp is not None and requested_bp < self.block_size:
            log.warning(
                "Window %r resolved to ~%d bp but was floored to "
                "block_size=%d bp: a window cannot be finer than one block, so "
                "the effective window is %d bp. Lower block_size for finer "
                "local trees.",
                self.window, int(requested_bp), self.block_size,
                self.block_size)
        if span_bp <= 0 or n_sites <= 0:
            return
        snps_per_block = self.block_size * n_sites / span_bp
        if snps_per_block < _MIN_SNPS_PER_BLOCK:
            log.warning(
                "A block size of %d bp averages only %.1f SNPs per block (< %.0f): "
                "many blocks carry no informative sites, so the per-block "
                "pairwise-TMRCA estimates are noise-dominated. Consider a "
                "larger block_size.",
                self.block_size, snps_per_block, _MIN_SNPS_PER_BLOCK)

    def _resolve_segmentation_params(self) -> None:
        """Resolve ``self.block_size``, ``self._wbp`` (window bp) and
        ``self._halo_value`` (halo bp).

        Takes at most one streaming pre-pass over the source, needed because a
        default or ``"<N>snp"`` ``block_size`` and a ``"<N>snp"`` window are
        both sized from the global SNP density, the sites per bp between each
        contig's first and last site. The resolved block width is written back
        to ``self.block_size``, which every segment builder then shares. An
        ``"auto"`` halo is grounded in the recombination correlation length: it
        estimates per-bp pairwise diversity ``π`` from the site allele
        frequencies, converts to a mean pairwise TMRCA ``E[t] = π / (2μ)``,
        whence the bp over which the HMM's TMRCA decorrelates is
        ``L_c ≈ 1 / (r · E[t])``, and takes ``halo = 5·L_c`` (≈ e⁻⁵ residual
        correlation), capped at ``chunk_size/2``. Either halo is raised to the
        widest window of the global grid and to ``block_size``.
        """
        if getattr(self, "_wbp", None) is not None:
            return
        from collections import Counter
        w = self.window
        need_window = isinstance(w, str) and w.strip().lower().endswith("snp")
        need_halo = isinstance(self._halo, str) and self._halo == "auto"
        # A default (unset) or "<N>snp" block_size is sized from the SNP
        # density, which needs a site count even for a window given in bp.
        need_density = self.block_size is None or isinstance(self.block_size, str)

        n_sites = 0
        span = 0
        pairdiff = 0.0
        if need_window or need_halo or need_density:
            cur = None
            p0 = last = 0
            for s in self._source:
                n_sites += 1
                if s.chrom != cur:
                    if cur is not None:
                        span += last - p0 + 1
                    cur, p0 = s.chrom, s.pos
                last = s.pos
                if need_halo:
                    vals = [a for a in s.tip_alleles.values() if a is not None]
                    N = len(vals)
                    if N >= 2:
                        same = sum(c * (c - 1) for c in Counter(vals).values())
                        pairdiff += 1.0 - same / (N * (N - 1))
            if cur is not None:
                span += last - p0 + 1

        requested, self.block_size, self._wbp = LocalTreeBuilder._resolve_widths(
            w, self.block_size, n_sites, span)
        chunk = getattr(self, "_chunk_size", None)
        if chunk is not None and chunk < self.block_size:
            self._log.warning(
                "The chunk size of %d bp is narrower than the resolved block "
                "width of %d bp, raising it to the block width, below which a "
                "segment holds no whole emission block and the run splits into "
                "one HMM pass per block",
                chunk, self.block_size)
            self._chunk_size = self.block_size
        self._warn_resolution(requested_bp=requested, n_sites=n_sites,
                              span_bp=span)

        # At least one block, so a segment's truncated final block is never
        # emitted beside the next segment's, and at least the widest window,
        # so every window overlapping a core lies inside its segment.
        declared = self._declared_length
        ends = (declared.values() if isinstance(declared, Mapping)
                else [declared])
        floor = max(self.block_size, int(np.ceil(max(
            (self._widest_window(float(end)) for end in ends if end),
            default=0.0))))
        if not need_halo:
            halo = _parse_bp(self._halo, name="halo")
            if halo < floor:
                self._log.warning(
                    "The halo of %d bp is narrower than the widest window or "
                    "block of %d bp, raising it to that width so no window is "
                    "cut at a segment edge", halo, floor)
                halo = floor
            self._halo_value = halo
            return
        pi = pairdiff / max(1, span)  # per-bp pairwise diversity
        e_t = pi / (2.0 * self.mu) if pi > 0 else 0.0
        if e_t > 0 and self.rec_rate > 0:
            halo = int(np.ceil(5.0 / (self.rec_rate * e_t)))  # 5 × L_c
        else:
            halo = 10 * self._wbp  # fallback if no diversity
        self._halo_value = max(floor, min(halo, self._chunk_size // 2))

    def _declared_on(self, chrom: "str | None") -> "float | None":
        """The declared length of a contig.

        :param chrom: The contig.
        :return: Its entry in a ``{contig: length}`` ``sequence_length``, the
            single length declared for every contig, or ``None`` when none is
            declared.
        """
        end = self._declared_length
        return end.get(str(chrom)) if isinstance(end, Mapping) else end

    def _axis_end(self, chrom: "str | None") -> "int | None":
        """End of a contig's declared region on the sites' axis.

        :param chrom: The contig.
        :return: Its declared length plus the source's
            :attr:`SiteSource.position_base <ancestree.sites.SiteSource.position_base>`,
            so the region's last base lies below it, or ``None`` when no length
            is declared.
        """
        end = self._declared_on(chrom)
        return int(end) + self._position_base if end else None

    def _stream_segments(self):
        """Stream the source once, yielding segment work-units.

        ``(seg_sites, core_lo, core_hi, origin, span, chrom)``: sites are
        streamed in sorted order through a sliding buffer (holding the sites
        within ~``chunk_size + 2·halo`` bp), so neither the full input nor a
        whole contig is materialised. Each contiguous contig is an independent
        segment. A contig longer than ``chunk_size`` is sliced into cores
        extended by a ``halo`` overlap on its interior sides (discarded after
        building). The cores tile the contig up to the declared region's end,
        or to its last site when none is declared, so a core may hold no site.
        ``origin`` shifts the segment to a local 0 so the HMM block count
        tracks the span.
        """
        bs, cs = self.block_size, self._chunk_size
        halo = self._halo_value
        base = self._position_base

        def make(buf, core_lo, core_hi, chrom):
            """One work unit: the core plus its halo."""
            end = self._declared_on(chrom)
            end = int(end) if end else None
            lo, hi = core_lo - halo, core_hi + halo
            seg = [s for s in buf if lo <= s.pos < hi]
            origin = (max(0, lo) // bs) * bs
            # Reaches the segment's last site, or the halo's end within the
            # declared region.
            last = seg[-1].pos + 1 if seg else core_hi
            reach = (origin + ((last - 1 - origin) // bs + 1) * bs
                     if end is None else min(end, hi))
            return (seg, core_lo, core_hi, origin, reach - origin, chrom)

        def flush(buf, core_lo, chrom):
            """Emit the work units from ``core_lo`` to the end of the axis."""
            stop = self._axis_end(chrom) or buf[-1].pos + 1
            # The cores tile the stop - base bases on the axis, and the last
            # one reaches stop.
            while core_lo < stop - base:
                core_hi = core_lo + cs
                yield make(buf, core_lo,
                           core_hi if core_hi < stop - base else stop, chrom)
                core_lo = core_hi

        cur = None
        buf: list[Site] = []
        core_lo = 0
        last = -1
        axis_end = None
        for s in self._source:
            if cur is None or s.chrom != cur:
                if cur is not None:
                    yield from flush(buf, core_lo, cur)
                cur, buf, core_lo, last = s.chrom, [], 0, s.pos
                axis_end = self._axis_end(cur)
            if s.pos < 0 or (axis_end is not None and s.pos >= axis_end):
                LocalTreeBuilder._check_on_axis(
                    np.array([s.pos]), axis_end or np.inf)
            if s.pos < last:
                raise ValueError(
                    "chunked LocalTreeInference requires sorted sites; saw "
                    f"{cur}:{s.pos} after {cur}:{last}"
                )
            last = s.pos
            buf.append(s)
            while s.pos >= core_lo + cs + halo:  # right halo fully buffered
                yield make(buf, core_lo, core_lo + cs, cur)
                core_lo += cs
                cut = core_lo - halo  # drop sites before next halo
                k = 0
                while k < len(buf) and buf[k].pos < cut:
                    k += 1
                if k:
                    del buf[:k]
        if cur is not None:
            yield from flush(buf, core_lo, cur)

    def _load_map(self, value, name: str, base: int = 0):
        """A map as passed, read from its file where a path is given.

        :param value: A map, a ``{contig: map}`` dict, a file path, or
            ``None``.
        :param name: The parameter: ``recombination_map`` reads a HapMap file,
            ``accessibility`` a BED file and ``mutation_map`` a bedGraph.
        :param base: Coordinate of a contig's first base in the sites, onto
            which the 0-based BED and bedGraph intervals are shifted.
        :return: The map, one entry per contig where read from a file.
        """
        if not isinstance(value, (str, os.PathLike)):
            return list(value) if name == "accessibility" and isinstance(
                value, (list, tuple)) else value
        from ancestree._maps import MapFiles

        self._map_paths[name] = str(value)
        if name == "recombination_map":
            return MapFiles.read_hapmap(value)
        if name == "accessibility":
            return MapFiles.read_bed(value, base)
        return MapFiles.read_bedgraph(value, default_rate=float(self.mu), base=base)

    def _maps_on(self, contig: str) -> dict:
        """The recombination map, accessibility mask and mutation map for
        ``contig``.

        :param contig: The contig of the sites.
        :return: ``{parameter: map}``, each ``None`` where unset.
        """
        from ancestree._maps import MapFiles

        return {name: MapFiles.on_contig(getattr(self, name), contig, name)
                for name in ("recombination_map", "accessibility",
                             "mutation_map")}

    def _slice_map(self, rate_map, origin: int, span: int):
        """Sub-map over ``[origin, origin+span)`` shifted to a local ``0``.

        The builder reads per-block rates as cumulative-mass differences over
        each block, so the slice's constant offset cancels.

        :param rate_map: The map to slice, or ``None``.
        :param origin: Left edge of the segment, in base pairs.
        :param span: Width of the slice from ``origin``, in base pairs.
        :return: The sub-map, or ``None`` when unset or outside the map.
        """
        if rate_map is None:
            return None
        length = float(getattr(rate_map, "sequence_length", origin + span))
        right = min(float(origin + span), length)
        if right <= float(origin):
            return None
        return rate_map.slice(left=float(origin), right=right, trim=True)

    @staticmethod
    def _slice_accessibility(accessibility, origin: int, span: int):
        """Accessible intervals intersected with the segment, shifted to 0.

        :param accessibility: The contig's intervals, or ``None``.
        :param origin: Left edge of the segment, in base pairs.
        :param span: Width of the segment's axis, in base pairs: its span,
            plus the position base when it holds the region's last base.
        :return: The shifted intervals, or ``None`` when unset.
        """
        if accessibility is None:
            return None
        out: list[tuple[float, float]] = []
        for a0, a1 in accessibility:
            lo = max(float(a0), float(origin))
            hi = min(float(a1), float(origin + span))
            if hi > lo:
                out.append((lo - origin, hi - origin))
        return out

    def _window_count(self, end: float) -> int:
        """Windows the grid over a region of ``end`` bp holds, each close to
        the resolved window width."""
        return max(1, int(round(end / float(self._wbp))))

    def _widest_window(self, end: float) -> float:
        """Width of the widest window in the grid over a region of ``end`` bp,
        the last one, which reaches past the region's last base to the axis's
        end."""
        n = self._window_count(end)
        return float(int(end) + self._position_base) - (n - 1) * (end / n)

    def _global_windows(self, chrom: str) -> "list[tuple[float, float]]":
        """The window grid over a contig's whole region, in genome coordinates.

        Laid once per contig so every segment reads its calls on the same
        tiling. Tiled per segment instead, the grid follows each segment's own
        length and site density, so the same position falls in a different
        window depending on where the chunk boundaries landed.

        :param chrom: The contig.
        :return: ``(left, right)`` windows tiling the contig's axis, empty
            when it has no declared length and each segment tiles its own core.
        """
        cached = getattr(self, "_windows_cache", None)
        if cached is not None and cached[0] == chrom:
            return cached[1]
        end = self._declared_on(chrom)
        windows = []
        if end:
            end = float(end)
            n = self._window_count(end)
            step = end / n
            # The last window reaches the end of the axis, past the region's
            # last base.
            windows = [
                (i * step,
                 float(self._axis_end(chrom)) if i == n - 1 else (i + 1) * step)
                for i in range(n)]
        self._windows_cache = (chrom, windows)
        return windows

    def _segment_builder(self, work_unit) -> "LocalTreeBuilder":
        """Configure the :class:`~ancestree.local_tree_inference.LocalTreeBuilder` for one segment work-unit.

        Sites are shifted to a local ``0`` (``origin``) so the HMM block count
        tracks the segment span. The recombination / mutation / accessibility
        maps are sliced to the segment's axis and shifted likewise. A segment
        whose halo reaches the declared region's end holds its last base, at
        the source's position base past the span.
        """
        seg, _core_lo, _core_hi, origin, span, chrom = work_unit
        shifted = [replace(s, pos=int(s.pos - origin)) for s in seg]
        maps = self._maps_on(str(chrom))
        end = self._axis_end(chrom)
        base = (self._position_base
                if end is not None and _core_hi + self._halo_value >= end else 0)
        axis = span + base
        builder = LocalTreeBuilder(
            shifted, mu=self.mu, rec_rate=self.rec_rate,
            sample_names=self.sample_names, sequence_length=span,
            window=self._wbp, block_size=self.block_size,
            n_time_bins=self.n_time_bins,
            time_grid=self.time_grid,
            recombination_map=self._slice_map(
                maps["recombination_map"], origin, axis),
            accessibility=self._slice_accessibility(
                maps["accessibility"], origin, axis),
            mutation_map=self._slice_map(maps["mutation_map"], origin, axis),
            validate_coverage=False,
            position_base=base,
            # A window overlapping this core is clipped to the segment's axis.
            window_anchor=([
                (max(l - origin, 0.0), min(r - origin, float(axis)))
                for l, r in self._global_windows(chrom)
                if r > float(_core_lo) and l < float(_core_hi)
            ] or None),
        )
        builder._quiet = True  # per-segment: parent logs the segmented build
        return builder

    @staticmethod
    def _same_contig(seen: "str | None", chrom: str, what: str) -> str:
        """The one contig an output laid on a single coordinate axis may span.

        :param seen: Contig accepted so far, or ``None`` at the first segment.
        :param chrom: The current segment's contig.
        :param what: Caller name, for the message.
        :return: The contig every later segment must also carry.
        :raises NotImplementedError: If ``chrom`` differs from ``seen``.
        """
        if seen is not None and chrom != seen:
            raise NotImplementedError(
                f"LocalTreeInference: {what} on the chunked path supports a "
                f"single contig; the source spans {seen!r} and {chrom!r}. "
                f"Positions from different contigs would collide on one axis. "
                f"Use VCF / VCZ output for multi-contig data."
            )
        return chrom if seen is None else seen

    def _process_segment(self, work_unit):
        """Build one segment work-unit's trees and run the kernel over them.

        Returns ``([(orig_site, posterior values)], counts)`` for the segment's
        core sites, the halo discarded. Raw value arrays keep the fork-pool
        payload small. The parent rewraps them in
        :class:`~ancestree.posterior.Posterior` and sums the counts, which a
        forked worker cannot report any other way.

        :param work_unit: One ``_stream_segments`` work-unit.
        :return: ``(rows, (n_ingroup_non_monophyletic, n_focal_multiroot_fallback,
            n_uniform_fallback, n_ingroup_monomorphic, n_uncoalesced_segments,
            n_unrepresentable_sites, n_unrepresentable_tips))``.
        """
        seg, core_lo, core_hi, _origin, _span, _chrom = work_unit
        if self.n_ensemble is not None:
            uniform_before = self._n_uniform_fallback
            rows = self._process_segment_ensemble(work_unit)
            uniform = self._n_uniform_fallback - uniform_before
            self._n_uniform_fallback = uniform_before
            # The remaining three counters are per-tree, and a window here
            # carries n_ensemble of them, so they have no single value to
            # report.
            before = self._site_counts_before()
            core = [site for site, _ in rows]
            self._count_ingroup_monomorphic(core)
            self._count_unrepresentable(core)
            counted, unrep_sites, unrep_tips = self._drain_site_counts(before)
            return rows, self._counts_in_order({
                "_n_uniform_fallback": uniform,
                "_n_ingroup_monomorphic": counted,
                "_n_unrepresentable_sites": unrep_sites,
                "_n_unrepresentable_tips": unrep_tips,
            })
        core = [site for site in seg if core_lo <= site.pos < core_hi]
        if not core:
            return [], self._counts_in_order({})
        segment = self._segment_builder(work_unit)
        origin = work_unit[3]
        ts = segment.to_tree_sequence()
        # Clipped to the core, so the per-tree tallies count no halo tree.
        ts = ts.keep_intervals(
            [[max(0.0, core_lo - origin), min(ts.sequence_length, core_hi - origin)]],
            simplify=False, record_provenance=False)
        arg = ARGBasedInference(
            ts, self.model,
            mu=segment._kernel_mu(), prior=self.prior,
            base_composition=self.base_composition, progress=False,
            n_workers=1, focal=self.focal,
            ingroup_samples=self._resolved_ingroup,
            outgroup_samples=self._resolved_outgroups or None,
        )
        arg._quiet = True
        # Paired by position in the segment's frame (pos - origin). A site
        # absent from the baked tree sequence takes the flat posterior.
        scored = {int(ts_site.pos): np.asarray(post.values)
                  for ts_site, post in arg.infer()}
        n_states = len(self.model.states)
        flat = np.full(n_states, 1.0 / n_states)
        out = [(site, scored.get(int(site.pos - origin), flat)) for site in core]
        # Tallied over the core rows the segment emits, never the halo.
        before = self._site_counts_before()
        self._count_ingroup_monomorphic(core)
        self._count_unrepresentable(core)
        counted, unrep_sites, unrep_tips = self._drain_site_counts(before)
        return out, self._counts_in_order({
            "_n_ingroup_non_monophyletic": arg._n_ingroup_non_monophyletic,
            "_n_focal_multiroot_fallback": arg._n_focal_multiroot_fallback,
            "_n_uniform_fallback": arg._n_uniform_fallback,
            "_n_ingroup_monomorphic": counted,
            "_n_uncoalesced_segments": arg._n_uncoalesced_segments,
            "_n_unrepresentable_sites": unrep_sites,
            "_n_unrepresentable_tips": unrep_tips,
        })

    def _process_segment_ensemble(self, work_unit) -> list[tuple[Site, np.ndarray]]:
        """Score one segment's core sites against an ensemble of genealogies.

        The plug-in path scores the single agglomerated tree per window. Here
        each window's posterior is instead marginalised over
        :paramref:`n_ensemble <ancestree.local_tree_inference.LocalTreeInference.n_ensemble>` genealogies drawn from the same pairwise-HMM
        posterior, which is what removes the confidently-wrong calls where the
        point estimate mis-resolves a split inside the ingroup.
        """
        seg, core_lo, core_hi, _origin, _span, chrom = work_unit
        keep = np.fromiter((core_lo <= s.pos < core_hi for s in seg), bool,
                           len(seg))
        post = self._score_ensemble(self._segment_builder(work_unit), seg, keep,
                                    segment=(chrom, core_lo))
        return list(zip(compress(seg, keep), post))

    def _weight_by_prior(self, post, sites):
        """Multiply marginalised likelihoods by the root prior and renormalise.

        :param post: ``(n_sites, S)`` normalised likelihoods.
        :param sites: The sites those rows correspond to, in the same order.
        :return: ``(n_sites, S)`` prior-weighted posteriors.
        """
        prior = self.prior
        if prior is None:
            if self.base_composition is None:
                return post  # uniform prior: weighting is a no-op
            prior = StationaryPrior(self.model, self.base_composition)
        with np.errstate(divide="ignore"):
            log_post = np.log(np.asarray(post, dtype=float))
        log_post = log_post + prior.log_probs(list(sites))
        return self._normalise_log_post(log_post, n_states=self.model.n_states)

    def _ensemble_fraction(self) -> float:
        """Where along the anchor-to-root path the ensemble reads.

        The ensemble kernel places the readout by fraction (``_reroot``), so an
        intermediate ``fraction`` is honoured as given. ``panel_root`` is the
        far end whatever else is set.

        """
        spec = self.focal
        if not self._focal_anchored_at_ingroup_mrca:
            return 1.0
        return float(spec.fraction or 0.0)

    def _ensemble_placement(self) -> "tuple[float, int]":
        """Where the ensemble reads, as ``(fraction, coalescences)``.

        ``coalescences`` is topological -- the k-th join above the anchor --
        so it is well defined in every drawn genealogy even though the drawn
        heights differ, and it is passed through. ``-1``
        means "not a coalescence placement", and the fraction applies.

        :return: ``(fraction, n_coal)``. Exactly one is meaningful.
        """
        spec = self.focal
        if self._focal_anchored_at_ingroup_mrca and spec.coalescences is not None:
            return 0.0, int(spec.coalescences)
        return self._ensemble_fraction(), -1

    def _stitch_segment_tree_sequence(self) -> "tuple[tskit.TreeSequence, str]":
        """Assemble a genome-wide tree sequence from the chunked segments.

        Streams one segment at a time (so the HMM's peak memory still tracks a
        single segment span), builds each segment's local-tree sequence, clips
        it to its core interval, shifts it back to genome coordinates, and
        appends nodes / edges / baked sites into one growing table collection.
        The result tiles the covered genome with the per-window local trees,
        the same pseudo-ARG the non-chunked path produces, but reconstructed
        without ever holding the whole-genome HMM TMRCA matrix.

        Single contig only: a :class:`tskit.TreeSequence` has one coordinate
        axis, so a multi-contig source cannot be represented as one tree
        sequence (use VCF / VCZ output instead).

        :return: The stitched genome-wide pseudo-ARG tree sequence, and the
            contig it spans.
        :raises NotImplementedError: If the source spans more than one contig.
        :raises ValueError: If the source yields no segments.
        """
        import tskit
        self._resolve_segmentation_params()
        n = len(self.sample_names)
        NULL = tskit.NULL

        internal_times: list[np.ndarray] = []
        e_left, e_right, e_parent, e_child = [], [], [], []
        s_pos: list[np.ndarray] = []
        s_anc: list[str] = []
        m_site, m_node, m_parent = [], [], []
        m_derived: list[str] = []

        internal_base = n  # next global internal-node id
        site_base = 0
        mut_base = 0
        seq_len = 0.0
        seen_chrom = None

        for wu in self._stream_segments():
            seg, core_lo, core_hi, origin, _span, chrom = wu
            seen_chrom = self._same_contig(
                seen_chrom, chrom, "point_tree_sequence() / to_arg()")
            seg_ts = self._segment_builder(wu).to_tree_sequence()
            lo = max(0.0, core_lo - origin)
            hi = min(seg_ts.sequence_length, core_hi - origin)
            if hi <= lo:
                # An empty core: with no declared length the segment's axis
                # ends at its last site, which lies in the left halo. It carries
                # no core sites, so it contributes nothing to the stitched output.
                continue
            seg_ts = seg_ts.keep_intervals(
                [[lo, hi]], simplify=False, record_provenance=False,
            )
            t = seg_ts.tables

            # Node remap: shared samples [0, n) keep their ids. Each segment's
            # internal nodes get fresh contiguous global ids.
            n_nodes = t.nodes.num_rows
            node_map = np.arange(n_nodes, dtype=np.int64)
            n_int = n_nodes - n
            node_map[n:] = internal_base + np.arange(n_int)
            internal_times.append(np.asarray(t.nodes.time[n:]))
            internal_base += n_int

            # Edges: shift back to genome coordinates, remap node ids.
            e_left.append(np.asarray(t.edges.left) + origin)
            e_right.append(np.asarray(t.edges.right) + origin)
            e_parent.append(node_map[np.asarray(t.edges.parent)])
            e_child.append(node_map[np.asarray(t.edges.child)])

            # Sites (core-only after keep_intervals): shift positions.
            ns = t.sites.num_rows
            if ns:
                s_pos.append(np.asarray(t.sites.position) + origin)
                s_anc.extend(t.sites[i].ancestral_state for i in range(ns))
            site_map = site_base + np.arange(ns, dtype=np.int64)
            site_base += ns

            # Mutations: remap site / node / parent (parent stays within-site).
            nm = t.mutations.num_rows
            if nm:
                m_site.append(site_map[np.asarray(t.mutations.site)])
                m_node.append(node_map[np.asarray(t.mutations.node)])
                mp = np.asarray(t.mutations.parent)
                m_parent.append(np.where(mp == NULL, NULL, mut_base + mp))
                m_derived.extend(t.mutations[i].derived_state for i in range(nm))
            mut_base += nm

            seq_len = max(seq_len, float(origin) + hi)

        if seen_chrom is None:
            raise ValueError("no segments to build a tree sequence from")

        out = tskit.TableCollection(sequence_length=seq_len)
        out.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        for name in self.sample_names:
            ind = out.individuals.add_row(metadata={"name": str(name)})
            out.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0, individual=ind)
        for tt in np.concatenate(internal_times):
            out.nodes.add_row(time=float(tt))
        out.edges.set_columns(
            left=np.concatenate(e_left), right=np.concatenate(e_right),
            parent=np.concatenate(e_parent).astype(np.int32),
            child=np.concatenate(e_child).astype(np.int32),
        )
        if s_anc:
            positions = np.concatenate(s_pos)
            for pos, anc in zip(positions, s_anc):
                out.sites.add_row(position=float(pos), ancestral_state=anc)
        if m_derived:
            ms = np.concatenate(m_site)
            mn = np.concatenate(m_node)
            mpar = np.concatenate(m_parent)
            for i, derived in enumerate(m_derived):
                out.mutations.add_row(
                    site=int(ms[i]), node=int(mn[i]),
                    derived_state=derived, parent=int(mpar[i]),
                )
        out.sort()
        out.time_units = "generations"
        return out.tree_sequence(), seen_chrom

    def _infer_segmented(self) -> Iterator[tuple[Site, Posterior]]:
        """Posteriors for a chunked run, one segment at a time."""
        # Resolve window + halo once (workers inherit them via fork).
        self._resolve_segmentation_params()
        states = self.model.states
        n_workers = Settings._resolve_n_workers(self.n_workers)
        if n_workers > 1 and not Settings._fork_pool_ok(
                self._log, "genomic segments", n_workers):
            n_workers = 1
        if n_workers > 1:
            import multiprocessing as mp
            if "fork" in mp.get_all_start_methods():
                global _LOCAL_TREE_FOR_FORK
                _LOCAL_TREE_FOR_FORK = self
                try:
                    ctx = mp.get_context("fork")
                    with ctx.Pool(
                            processes=n_workers,
                            initializer=Settings._cap_worker_threads,
                            initargs=(n_workers,)) as pool:
                        # imap streams work-units to workers in genomic order.
                        for seg_out, counts in pool.imap(
                                _segment_worker, self._segments_with_bar()):
                            self._add_counts(counts)
                            for orig, values in seg_out:
                                yield orig, Posterior(alleles=states, values=values)
                finally:
                    _LOCAL_TREE_FOR_FORK = None
                return
            self._log.warning(
                "LocalTreeInference(n_workers=%s): the 'fork' start method is "
                "unavailable on this platform; running the chunked path "
                "single-threaded.", self.n_workers,
            )
        for wu in self._segments_with_bar():
            seg_out, counts = self._process_segment(wu)
            self._add_counts(counts)
            for orig, values in seg_out:
                yield orig, Posterior(alleles=states, values=values)

    def _segments_with_bar(self):
        """The segment work-units, under a progress bar when requested."""
        # A segment holding no site in its core has no posterior to report.
        segments = (wu for wu in self._stream_segments()
                    if any(wu[1] <= site.pos < wu[2] for site in wu[0]))
        if not self.progress:
            return segments
        end = self._declared_length
        total = (max(1, -(-int(end) // int(self._chunk_size)))
                 if end and not isinstance(end, Mapping)
                 and self._chunk_size else None)
        return tqdm(segments, total=total, desc="LocalTreeInference",
                    unit=" segments", disable=Settings.disable_pbar)

    def _site_counts_before(self) -> tuple[int, int, int]:
        """The per-site counters, read before a segment tallies its core rows.

        :return: ``(n_ingroup_monomorphic, n_unrepresentable_sites,
            n_unrepresentable_tips)``.
        """
        return (self._n_ingroup_monomorphic, self._n_unrepresentable_sites,
                self._n_unrepresentable_tips)

    def _drain_site_counts(
        self, before: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        """One segment's per-site tallies, taken off the live counters.

        The segment reports them to the parent, which rolls them in through
        :meth:`~ancestree.inference.Inference._add_counts`.

        :param before: The reading :meth:`_site_counts_before` returned.
        :return: ``(n_ingroup_monomorphic, n_unrepresentable_sites,
            n_unrepresentable_tips)`` accumulated since that reading.
        """
        delta = (self._n_ingroup_monomorphic - before[0],
                 self._n_unrepresentable_sites - before[1],
                 self._n_unrepresentable_tips - before[2])
        (self._n_ingroup_monomorphic, self._n_unrepresentable_sites,
         self._n_unrepresentable_tips) = before
        return delta

    _MODE = "local-tree"

    def _provenance_parameters(self) -> dict[str, object]:
        """Local-tree run parameters.

        Model, prior and mu, the contig, the tree-building parameters (rate,
        window, block size, time bins), and in ensemble mode the member count,
        seed and chunk. On the chunked path the contig is known once
        :meth:`point_tree_sequence` has run.
        """
        params: dict[str, object] = {
            "model": self._model_name(self.model),
            "prior": type(self.prior).__name__ if self.prior is not None
            else "StationaryPrior",
            "mu": float(self.mu) if isinstance(self.mu, float) else "variable",
            "rec_rate": float(self.rec_rate),
            **self._model_provenance(),
            "window": self.window,
            "n_time_bins": int(self.n_time_bins),
            **self.focal.provenance(),
        }
        if self._segmented:
            self._resolve_segmentation_params()
            params.update({
                # The configured spec (None is density-adaptive).
                "block_size": self.block_size,
                "halo": self._halo,
                "chunk_size": int(self._chunk_size),
                # The spec snapped up to a whole number of blocks.
                "window_bp": int(self._wbp),
            })
            contig = None if self._stitched is None else self._stitched[1]
        else:
            params.update({"block_size": int(self.builder.block_size),
                           "window_bp": int(self.builder.window_bp)})
            contig = next(iter(self.builder.sites.chroms), None)
        # The contig of every emitted site, where the run lays them on one.
        params["chrom"] = self.chrom if self.chrom is not None else contig
        if self._arg is not None:
            params.update(self._arg._focal_provenance())
        params.update(self._ensemble_provenance())
        params.update(self._map_provenance())
        return params

    def _ensemble_mode(self) -> bool:
        """Whether this run marginalises over an ensemble."""
        return self.n_ensemble is not None

    def _map_provenance(self) -> dict:
        """Which maps applied.

        Each map is summarised from the object itself, per contig where it
        was given per contig, with the file it was read from.

        :return: Provenance entries for the supplied maps.
        """
        out: dict = {}
        mask = self.accessibility
        if mask is not None:
            per_contig = mask.values() if isinstance(mask, dict) else [mask]
            out["accessibility_intervals"] = sum(len(m) for m in per_contig)
        for name in ("recombination_map", "mutation_map"):
            value = getattr(self, name)
            if isinstance(value, dict):
                out[f"{name}_contigs"] = len(value)
                out[f"{name}_intervals"] = sum(len(m.rate) for m in value.values())
            else:
                out.update(self._rate_map_provenance(name, value))
        out.update({f"{name}_path": path
                    for name, path in self._map_paths.items()})
        return out

    @staticmethod
    def _rate_map_provenance(name: str, rate_map) -> dict:
        """A rate map summarised for the provenance record.

        :param name: The parameter the map was passed as.
        :param rate_map: An :class:`msprime.RateMap`, or ``None``.
        :return: Interval count and mean rate, or an empty mapping when no
            map was supplied.
        """
        if rate_map is None:
            return {}
        entry: dict = {f"{name}_intervals": int(len(rate_map.rate))}
        try:
            entry[f"{name}_mean_rate"] = float(rate_map.mean_rate)
        except Exception:
            pass
        return entry

    def _ensemble_provenance(self) -> dict[str, object]:
        """The ensemble's own parameters, or nothing when not in that mode.

        The draws are a function of the seed, the member count and the HMM
        forward pass, so the three together reproduce a run.
        """
        if not self._ensemble_mode():
            return {}
        return {
            "n_ensemble": int(self.n_ensemble),
            "ensemble_seed": int(self.ensemble_seed),
            "member_chunk": int(self.member_chunk),
        }

    def _ingroup_indices(self):
        """Panel positions of the ingroup, resolved by name.

        The ensemble locates the ingroup MRCA by membership, so the ingroup is
        not required to be the leading block of ``sample_names``.

        :return: Indices into ``sample_names``, or ``None`` for the whole panel.
        """
        wanted = self._baseline_ingroup_samples()
        if len(wanted) == len(self.sample_names):
            return None
        pos = {name: i for i, name in enumerate(self.sample_names)}
        return [pos[name] for name in wanted]

    def tree_sequences(
        self,
    ) -> "Iterator[tuple[tuple[float, float], Iterator[tskit.TreeSequence]]]":
        """The inferred genealogies, one group per stretch of the genome.

        Each item is ``(interval, members)``: the stretch in genome
        coordinates and an iterator over the genealogies covering it. In
        ensemble mode ``members`` yields the :paramref:`n_ensemble <ancestree.local_tree_inference.LocalTreeInference.n_ensemble>` draws
        :meth:`infer` marginalises over, otherwise the single plug-in tree
        sequence. An unchunked run yields one group spanning the region; a
        chunked run yields one per segment. The draws are a pure function of
        :paramref:`ensemble_seed <ancestree.local_tree_inference.LocalTreeInference.ensemble_seed>` and the HMM forward pass, so re-iterating
        reproduces them.

        Both axes stream: a genome-wide pass holds one stretch, and within a
        stretch a draw is built when it is pulled, so peak memory holds one
        genealogy, not all :paramref:`n_ensemble <ancestree.local_tree_inference.LocalTreeInference.n_ensemble>` of them. Call ``list()``
        on a group to keep its draws. A group's iterator is consumed once, and
        must be consumed before the next group is pulled.

        A draw is taken per segment, so the member at one index in this group
        and at the same index in the next are separate samples, not one
        genealogy running the length of the genome. Call
        :meth:`point_tree_sequence` for a single genome-wide plug-in tree.

        For 1-based input (VCF / VCZ) the last interval ends at
        ``sequence_length + 1``.

        :return: Iterator of ``(interval, iterator of tskit.TreeSequence)``.
        """
        if not self._segmented:
            span = float(self.builder.axis_length)
            # The unsegmented plug-in tree is the cached one infer() scored.
            members = (self._ensemble_members(
                           self.builder, segment=self._region_segment())
                       if self._ensemble_mode()
                       else iter([self.point_tree_sequence()]))
            yield (0.0, span), members
            return
        self._resolve_segmentation_params()
        seen_chrom = None
        for work_unit in self._stream_segments():
            seg, core_lo, core_hi, origin, _span, chrom = work_unit
            seen_chrom = self._same_contig(
                seen_chrom, chrom, "tree_sequences()")
            end = float(self._axis_end(chrom) or 0.0)
            members = self._ensemble_members(self._segment_builder(work_unit),
                                             segment=(chrom, core_lo))
            core = (float(core_lo), float(core_hi))
            yield core, self._genome_frame_iter(members, origin, end, core)

    def _genome_frame_iter(self, members, origin, region_end, core):
        """Place a segment's genealogies on the genome axis, one at a time.

        :param members: The segment's genealogies, on the segment's own axis.
        :param origin: Genome coordinate the segment's zero sits at.
        :param region_end: End of the genome axis (the declared length plus
            the position base), or 0 when undeclared.
        :param core: ``(lo, hi)`` stretch this segment emits.
        :return: Iterator of :class:`tskit.TreeSequence` on the genome axis.
        """
        for member in members:
            yield self._to_genome_frame(member, origin, region_end, core)

    @staticmethod
    def _to_genome_frame(ts, origin, region_end, core):
        """Shift a segment's tree sequence into genome coordinates.

        A segment is built on its own axis starting at zero, and reaches past
        the stretch it emits by the halo.

        :param ts: The segment's tree sequence, on the segment's axis.
        :param origin: Genome coordinate the segment's zero sits at.
        :param region_end: End of the genome axis (the declared length plus
            the position base), or 0 when undeclared.
        :param core: ``(lo, hi)`` stretch this segment emits, in genome
            coordinates.
        :return: The tree sequence on the genome's axis, trimmed to ``core``.
        """
        import numpy as np

        tables = ts.dump_tables()
        tables.sequence_length = max(float(region_end),
                                     float(origin + ts.sequence_length))
        edges = tables.edges.copy()
        tables.edges.clear()
        tables.edges.set_columns(
            left=np.asarray(edges.left) + origin,
            right=np.asarray(edges.right) + origin,
            parent=edges.parent, child=edges.child,
        )
        if tables.sites.num_rows:
            sites = tables.sites.copy()
            positions = np.asarray(sites.position) + origin
            mutations = tables.mutations.copy()
            tables.sites.clear()
            tables.mutations.clear()
            tables.sites.set_columns(
                position=positions,
                ancestral_state=sites.ancestral_state,
                ancestral_state_offset=sites.ancestral_state_offset,
            )
            tables.mutations.set_columns(
                site=mutations.site, node=mutations.node,
                parent=mutations.parent,
                derived_state=mutations.derived_state,
                derived_state_offset=mutations.derived_state_offset,
            )
        tables.sort()
        # A gap in the sites wider than the halo leaves the core reaching past
        # the segment's own axis, so the kept interval is clamped to the axis.
        lo = max(0.0, min(float(core[0]), tables.sequence_length))
        hi = max(lo, min(float(core[1]), tables.sequence_length))
        return tables.tree_sequence().keep_intervals(
            [[lo, hi]], simplify=False)

    def _ensemble_members(self, builder, segment=("", 0)):
        """The genealogies one builder's region contributes, lazily.

        :param builder: The :class:`~ancestree.local_tree_inference.LocalTreeBuilder`
            covering the region.
        :param segment: ``(contig, core start)`` keying the region's draws.
        :return: Iterator of :class:`tskit.TreeSequence`, the
            ``n_ensemble`` draws in ensemble mode, else the one plug-in tree
            sequence.
        """
        if not self._ensemble_mode():
            return iter([builder.to_tree_sequence()])
        from ancestree._ensemble import SegmentEnsemble

        ens = SegmentEnsemble(builder, self.model, len(self.sample_names),
                              self._ingroup_indices(), segment=segment)
        return ens.iter_tree_sequences(
            self.n_ensemble, member_chunk=self.member_chunk,
            seed=self.ensemble_seed)

    def point_tree_sequence(self):
        """The single plug-in (point-estimate) local-tree sequence.

        On the chunked path the genome-wide tree sequence is stitched (and
        cached) from the per-segment builds by
        ``_stitch_segment_tree_sequence()``, single contig only. The HMM is
        re-run per segment to assemble the topology, so this is independent of
        (and does not cache) the kernel pass in :meth:`infer`.

        :return: The local-tree sequence (a pseudo-ARG, see :meth:`to_arg`),
            of length ``sequence_length + 1`` for 1-based input (VCF / VCZ).
        :raises NotImplementedError: On the chunked path with a multi-contig
            source (no single coordinate axis to represent it on).
        """
        if self._segmented:
            if self._stitched is None:
                self._stitched = self._stitch_segment_tree_sequence()
            return self._stitched[0]
        return self._point_arg().ts

    def pairwise_tmrcas(self) -> PairwiseTmrcas:
        """Inferred per-block pairwise TMRCAs (the raw HMM output, for debugging).

        Delegates to :meth:`LocalTreeBuilder.pairwise_tmrcas`. The chunked path
        runs the HMM per segment and glues the per-block records in genome
        order, so the record covers the whole region either way.

        :return: A :class:`~ancestree.local_tree_inference.PairwiseTmrcas` record.
        :raises NotImplementedError: When a chunked source spans more than one
            contig (the blocks of
            different contigs would collide on one coordinate axis).
        :raises ValueError: If a chunked source yields no segments.
        """
        if getattr(self, "_tmrcas_cache", None) is not None:
            return self._tmrcas_cache
        parts = list(self.iter_pairwise_tmrcas())
        if parts:
            if len(parts) == 1:
                self._tmrcas_cache = parts[0]
                return self._tmrcas_cache
            order = np.argsort(np.concatenate(
                [p.block_midpoints for p in parts]))
            self._tmrcas_cache = PairwiseTmrcas(
                sample_names=parts[0].sample_names,
                pairs=parts[0].pairs,
                block_midpoints=np.concatenate(
                    [p.block_midpoints for p in parts])[order],
                tmrca=np.concatenate([p.tmrca for p in parts], axis=1)[:, order],
            )
            return self._tmrcas_cache
        raise ValueError("no segments to report pairwise TMRCAs for")

    def iter_pairwise_tmrcas(self) -> "Iterator[PairwiseTmrcas]":
        """Per-block pairwise TMRCAs, one record per stretch of the genome.

        A single region yields one record; a chunked build yields one per
        segment, trimmed to the blocks whose calls it emits so the halo is not
        reported twice. :meth:`pairwise_tmrcas` concatenates the stream, which
        is the whole matrix and therefore the thing chunking exists to avoid
        holding: iterate instead where memory is the constraint.

        :return: Iterator of :class:`~ancestree.local_tree_inference.PairwiseTmrcas`, in genome order.
        """
        if not self._segmented:
            yield self.builder.pairwise_tmrcas()
            return
        self._resolve_segmentation_params()
        seen_chrom = None
        for work_unit in self._stream_segments():
            seg, core_lo, core_hi, origin, _span, chrom = work_unit
            seen_chrom = self._same_contig(
                seen_chrom, chrom, "pairwise_tmrcas()")
            part = self._segment_builder(work_unit).pairwise_tmrcas()
            mids = np.asarray(part.block_midpoints, dtype=float) + origin
            core = (mids >= float(core_lo)) & (mids < float(core_hi))
            if not core.any():
                continue
            yield PairwiseTmrcas(
                sample_names=part.sample_names,
                pairs=part.pairs,
                block_midpoints=mids[core],
                tmrca=np.asarray(part.tmrca)[:, core],
            )

    def write_pairwise_tmrcas(self, path: "str | os.PathLike") -> None:
        """Write the inferred per-block pairwise TMRCAs to ``path``.

        Format by suffix: ``.npz`` (raw arrays), ``.csv`` / ``.tsv`` (a long
        ``hap_i, hap_j, block_midpoint, tmrca`` table). See
        :meth:`pairwise_tmrcas` and :meth:`PairwiseTmrcas.write`.

        :param path: Output path. Suffix selects the format.
        """
        self.pairwise_tmrcas().write(path)

    def _contig_lengths(self) -> dict[str, int]:
        """The contig lengths the input declares, the only one keyed under
        ``chrom`` when set."""
        lengths = super()._contig_lengths()
        if self.chrom is None:
            return lengths
        return ({self.chrom: next(iter(lengths.values()))}
                if len(lengths) == 1 else {})

    def _emitted(self, pairs):
        """``pairs`` as this mode reports them: without the tree handle of a
        tree-sequence source, which the inferred trees do not share, and on
        contig ``chrom`` where set.

        :param pairs: The ``(Site, Posterior)`` stream.
        :return: Generator over the reported pairs.
        :raises ValueError: If ``chrom`` is set and the sites lie on more than
            one contig.
        """
        first = None
        for site, posterior in pairs:
            if self.chrom is not None:
                if first is None:
                    first = site.chrom
                elif site.chrom != first:
                    raise ValueError(
                        f"chrom={self.chrom!r} would put the sites of contigs "
                        f"{first!r} and {site.chrom!r} on one contig. Leave "
                        f"chrom unset for a source spanning several contigs.")
                site = replace(site, chrom=self.chrom, local_tree_handle=None)
            elif site.local_tree_handle is not None:
                site = replace(site, local_tree_handle=None)
            yield site, posterior

    def _output_template(self, given, fmt, output, restrict_samples):
        """As :meth:`Inference._output_template`, refusing a relabelled run.

        :raises ValueError: If ``chrom`` renames the sites of an output that
            annotates a file, whose records keep their own contig names.
        """
        template, samples = super()._output_template(
            given, fmt, output, restrict_samples)
        if template is None or self.chrom is None:
            return template, samples
        sites = self._source if self._segmented else self.builder.sites
        first = next(iter(sites), None)
        if first is not None and str(first.chrom) != self.chrom:
            raise ValueError(
                f"chrom={self.chrom!r} renames the sites of contig "
                f"{first.chrom!r}, so {output} cannot annotate {template}, "
                f"whose records keep their own contig names. Write an output "
                f"of another format, or leave chrom (--chrom) unset.")
        return template, samples

    def _source_tree_sequence(
        self, restrict_samples: bool = False,
    ) -> "tskit.TreeSequence":
        """The local-tree sequence that
        :meth:`LocalTreeInference.to_arg() <ancestree.local_tree_inference.LocalTreeInference.to_arg>`
        annotates.

        The plug-in genealogy is resolved as ARG mode resolves it. On the
        chunked path it is the stitch from
        :meth:`LocalTreeInference.point_tree_sequence() <ancestree.local_tree_inference.LocalTreeInference.point_tree_sequence>`,
        which holds only the panel.

        :param restrict_samples: Whether to hold only the samples the
            inference used.
        :return: The tree sequence to annotate.
        :raises NotImplementedError: On the chunked path with a multi-contig source.
        """
        if not self._segmented:
            return self._point_arg()._source_tree_sequence(restrict_samples)
        return self.point_tree_sequence()
