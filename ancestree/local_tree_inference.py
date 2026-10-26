"""Local-tree inference from genotypes: genealogy estimation and ancestral-allele inference.

- :class:`~ancestree.local_tree_inference.LocalTreeBuilder` infers dated
  local trees along the genome from phased genotypes, with a discretized
  PSMC′-style pairwise-TMRCA HMM
  (:class:`~ancestree.local_tree_inference.PairwiseCoalescentHMM`) condensed
  per window into one ultrametric tree. It emits a :class:`tskit.TreeSequence`
  carrying the sample names and the observed genotypes as sites.
  :meth:`LocalTreeBuilder.write() <ancestree.local_tree_inference.LocalTreeBuilder.write>`
  writes a ``.trees`` file that
  :class:`~ancestree.inference.ARGBasedInference` accepts.
- :class:`~ancestree.local_tree_inference.LocalTreeInference` builds these
  trees and scores the per-site ancestral-allele posterior over an ensemble of
  drawn genealogies, or on the single plug-in genealogy when the ensemble is
  off. :meth:`Inference.infer() <ancestree.inference.Inference.infer>` yields
  ``(Site, Posterior)``.

Outgroups are optional. They are extra panel haplotypes the local tree
places, so incomplete lineage sorting is absorbed by the inferred tree and not
by an assumed outgroup-ladder topology.
"""
from __future__ import annotations

import math
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import cached_property
from itertools import combinations, compress
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import tskit

from ancestree import DEFAULT_MU, DEFAULT_REC_RATE, STATES
from tqdm import tqdm

from ancestree._geometry import (
    _ACCESS_FRAC_FLOOR, _EMIT_SCALE_FLOOR, BuildGeometry, ContigGeometry,
    Segmentation, _check_on_axis, _filled, _parse_bp,
)
from ancestree._smc_kernel import pair_posterior_mean_tmrcas
from ancestree._upgma import (
    WindowedUpgmaTreeSequence,
)
from ancestree.focal import FocalNode
from ancestree.inference import ARGBasedInference, Inference, _Tally
from ancestree.likelihood import _normalise
from ancestree.models import JC69, SubstitutionModel
from ancestree.posterior import Posterior
from ancestree.priors import StationaryPrior
from ancestree.settings import Settings, _ForkPool
from ancestree.sites import (BaseComposition, Site, SiteSource,
                             SiteTable, _PanelSites, _path_format)
from ancestree._repr import ReprMixin

#: Deepest time grid the sampled-path storage can address. Bin indices are
#: ``int8`` (see ``SegmentEnsemble``).
MAX_TIME_BINS: int = int(np.iinfo(np.int8).max)


@dataclass(frozen=True)
class HmmInputs:
    """The prepared inputs of the pairwise-coalescent HMM over one region.

    Rows are haplotype pairs in :func:`itertools.combinations` order and
    columns blocks.
    """

    #: ``(P, K)`` pairwise differences per pair and block.
    counts: np.ndarray
    #: ``(P, K)`` number of the block's sites at which the pair is called, or
    #: ``None`` when no call is missing.
    called: "np.ndarray | None"
    #: ``(K,)`` site count each ``called`` entry is out of.
    block_sites: np.ndarray
    #: ``(T + 1,)`` time-bin edges in generations.
    edges: np.ndarray
    #: ``(T,)`` representative time of each bin in generations.
    t_rep: np.ndarray
    #: ``(T,)`` Poisson mean ``2·μ·t_i·B`` of a full block per bin.
    lam: np.ndarray
    #: ``(T,)`` ``log(lam)``.
    log_lam: np.ndarray
    #: ``(T,)`` probability of no recombination over one nominal block per bin.
    r: np.ndarray
    #: ``(P,)`` per-pair prior mean TMRCA in generations.
    t_bar: np.ndarray
    #: ``(P, T)`` per-pair coalescent prior over the bins.
    prior: np.ndarray
    #: ``(K - 1,)`` genetic distance of each inter-block step in nominal-block
    #: units, or ``None`` for all ones.
    step: "np.ndarray | None"
    #: ``(K,)`` bool of callable blocks, or ``None`` for all.
    block_mask: "np.ndarray | None"
    #: ``(K,)`` Poisson exposure scale of each block, or ``None`` for all ones.
    emit_scale: "np.ndarray | None"


class PairwiseCoalescentHMM(ReprMixin):
    """Discretized PSMC′-style estimator of local pairwise TMRCAs.

    Estimates, for every haplotype pair, the posterior-mean coalescence
    time in generations within each fixed-width block. The hidden state is
    the pair's coalescence time :math:`t` in generations, discretized into
    ``n_time_bins`` log-spaced bins. The emission is the block's count of
    pairwise differences,

    .. math::

        d \\sim \\mathrm{Poisson}(2 \\mu t B),

    and between consecutive blocks the time resets to the prior with
    probability :math:`1 - e^{-2 \\rho t B}`. Here :math:`d` is the number
    of differences in the block, :math:`\\mu` the per-site per-generation
    mutation rate ``mu``, :math:`\\rho` the per-site per-generation
    recombination rate ``rec_rate`` and :math:`B` the block width
    ``block_size`` in bp.

    The time grid is calibrated from the per-block difference counts, its top
    bin reaching the deepest observed divergence. The prior is per pair, an
    exponential with mean equal to the pair's genome-average TMRCA.

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
        """The shared log-spaced time grid, calibrated from the crude TMRCAs
        ``count / (2·μ·B)`` for a per-bp mutation rate ``μ`` and block width
        ``B`` bp.

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
        """Per-pair coalescent prior, the ``Exp(1/t_bar)`` mass per bin of
        ``edges`` for a pair with genome-average TMRCA ``t_bar`` (generations).
        The tails below ``edges[0]`` and above ``edges[-1]`` fold into the end
        bins.
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
        :param dtype: Integer dtype of the count arrays, wide enough for the
            busiest block's site count.
        :return: ``(counts, called, block_sites)``. ``counts`` and ``called``
            are ``(n(n-1)/2, n_blocks)`` in ``dtype``, rows in
            :func:`itertools.combinations` order: the pairwise differences and
            the sites at which both haplotypes are called (``None`` when no
            call is missing). ``block_sites`` is the ``(n_blocks,)`` int64 site
            count, so the exposure is ``called / block_sites``.
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

    def prepare(
        self, genotypes: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
        step: "np.ndarray | None" = None,
        block_mask: "np.ndarray | None" = None,
        emit_scale: "np.ndarray | None" = None,
    ) -> HmmInputs:
        """The block counts, calibration and per-pair prior the HMM runs on.

        Counts are held in ``int16``, or ``int32`` where the busiest block
        holds more sites than ``int16`` can count.

        :param genotypes: ``(n_sites, n_haplotypes)`` int8 allele indices
            (0..3), ``-1`` for missing.
        :param block_of_site: ``(n_sites,)`` block index per site.
        :param n_blocks: Total number of blocks.
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
        :return: The :class:`HmmInputs`.
        """
        busiest = (int(np.bincount(block_of_site, minlength=n_blocks).max())
                   if block_of_site.size else 0)
        dtype = np.int16 if busiest <= np.iinfo(np.int16).max else np.int32
        counts, called, block_sites = self._pair_block_counts(
            genotypes, block_of_site, n_blocks, dtype=dtype)
        edges, t_rep, lam, log_lam, r, t_bar = self.calibrate(
            counts, emit_scale=emit_scale, block_mask=block_mask,
            called=called, block_sites=block_sites)
        prior = np.stack([self._coalescent_prior(edges, t) for t in t_bar])
        return HmmInputs(counts, called, block_sites, edges, t_rep, lam,
                         log_lam, r, t_bar, prior, step, block_mask,
                         emit_scale)

    @staticmethod
    def posterior_means(inputs: HmmInputs) -> np.ndarray:
        """Posterior-mean TMRCA per (pair, block) by forward-backward.

        :param inputs: The :class:`HmmInputs` of :meth:`prepare`.
        :return: ``(n(n-1)/2, n_blocks)`` posterior-mean TMRCA (generations),
            one row per haplotype pair in :func:`itertools.combinations` order.
        """
        return pair_posterior_mean_tmrcas(
            inputs.counts, inputs.t_rep, inputs.lam, inputs.log_lam, inputs.r,
            inputs.prior, step=inputs.step, mask=inputs.block_mask,
            called=inputs.called, block_sites=inputs.block_sites,
            emit_scale=inputs.emit_scale)

    def block_tmrcas(
        self, genotypes: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
        step: "np.ndarray | None" = None,
        block_mask: "np.ndarray | None" = None,
        emit_scale: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Posterior-mean TMRCA per (pair, block): :meth:`posterior_means` of
        :meth:`prepare`, whose parameters it takes.

        :return: ``(n(n-1)/2, n_blocks)`` posterior-mean TMRCA (generations),
            one row per haplotype pair in :func:`itertools.combinations` order.
        """
        return self.posterior_means(self.prepare(
            genotypes, block_of_site, n_blocks, step=step,
            block_mask=block_mask, emit_scale=emit_scale))


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
        ``block_size``. It spans a whole number of blocks:
        ``ceil(N_window / N_block)`` when both are ``"<N>snp"`` specs, otherwise
        the requested width rounded up.
        :class:`~ancestree.local_tree_inference.LocalTreeInference` warns when a spec is floored this way.
    :param block_size: HMM emission block width, an int / bp string, or a
        ``"<N>snp"`` spec (N SNPs' worth of span at the data's mean density, the
        natural density-invariant unit). ``None`` (default) sizes the block to a
        few SNPs' worth of span (well below the window), so the HMM emits several
        blocks per output tree. An ``"<N>snp"`` window then holds ``N / 4``
        blocks rounded half up, at least one, at every density. Set explicitly to override the
        density-adaptive default.
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
    :param geometry: A resolved
        :class:`~ancestree._geometry.BuildGeometry`, which a chunked run
        supplies per segment, setting the axis, the widths, the windows and
        the maps in place of ``sequence_length``, ``window``, ``block_size``,
        the three maps and ``position_base``.
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
        sequence_length: "float | None" = None,
        window: "int | str" = "8snp",
        block_size: "int | str | None" = None,
        n_time_bins: int = 32,
        time_grid: "np.ndarray | None" = None,
        bake_genotypes: bool = True,
        recombination_map=None,
        accessibility: "Sequence[tuple[float, float]] | None" = None,
        mutation_map=None,
        validate_coverage: bool = True,
        geometry: "BuildGeometry | None" = None,
        position_base: int = 0,
    ) -> None:
        if not np.isfinite(mu):
            raise ValueError(f"mu must be positive and finite, got {mu}")
        if mu <= 0:
            raise ValueError(f"mu must be positive, got {mu}")
        if rec_rate <= 0:
            raise ValueError(f"rec_rate must be positive, got {rec_rate}")
        # Columnar: from_sites consumes the input one site at a time.
        self.sites = (sites if isinstance(sites, SiteTable)
                      else SiteTable.from_sites(sites, sample_names))
        _chroms = self.sites.chroms
        if len(_chroms) > 1:
            raise ValueError(
                f"LocalTreeBuilder builds a single contiguous coordinate axis, "
                f"but the sites span {len(_chroms)} contigs "
                f"({sorted(_chroms)}). Local trees are per-contig: pass one "
                f"contig at a time, or drive multi-contig data through "
                f"LocalTreeInference, which builds each contig on its own."
            )
        self.mu = float(mu)
        self.rec_rate = float(rec_rate)
        self.sample_names: list[str] = list(sample_names)
        if len(self.sample_names) < 2:
            raise ValueError(f"need >= 2 samples, got {len(self.sample_names)}")
        if geometry is None:
            if sequence_length is None:
                raise ValueError(
                    "LocalTreeBuilder needs sequence_length, the region "
                    "length in bp, when no geometry is passed")
            sequence_length = math.ceil(sequence_length)
            pos = self.sites.pos
            extent = (float(pos[-1] - pos[0] + 1) if len(pos)
                      else float(sequence_length))
            _, block, window_bp = ContigGeometry.resolve_widths(
                window, block_size, len(pos), extent)
            geometry = BuildGeometry(
                sequence_length, block=block, window=window_bp,
                position_base=position_base,
                recombination_map=recombination_map,
                accessibility=accessibility, mutation_map=mutation_map,
                data=(float(pos[0]), float(pos[-1] + 1)) if len(pos) else None)
        #: The axis, blocks, windows and maps this build reads.
        self.geometry = geometry
        self.sequence_length, self.position_base = (geometry.length,
                                                    geometry.position_base)
        self.block_size, self.window_bp = geometry.block, geometry.window
        _check_on_axis(self.sites.pos, geometry.axis_length)
        self.time_grid = PairwiseCoalescentHMM._check_time_grid(time_grid)
        if self.time_grid is not None:
            n_time_bins = self.time_grid.size - 1
        PairwiseCoalescentHMM._check_n_time_bins(n_time_bins)
        self.n_time_bins = int(n_time_bins)
        self.bake_genotypes = bool(bake_genotypes)
        self._validate_coverage = bool(validate_coverage)
        self._ts = None
        self._pair_block_t: "np.ndarray | None" = None
        self._pair_called: "np.ndarray | None" = None
        self._pair_block_sites: "np.ndarray | None" = None

    @property
    def axis_length(self) -> float:
        """Length of the tree sequence's axis, which holds every base of the
        region: :attr:`sequence_length` plus :attr:`position_base`."""
        return self.geometry.axis_length

    def _genotype_matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Build the ``(n_sites, n_hap)`` allele-index matrix + block ids.

        :return: ``(genotypes, positions, block_of_site, n_blocks)``, with
            ``positions`` each site's ``site.pos`` on ``[0, axis_length)``.
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
        if n_sites and self._validate_coverage:
            seen = (g >= 0).any(axis=0)
            if not seen.all():
                missing = [self.sample_names[i]
                           for i in np.nonzero(~seen)[0]]
                LocalTreeInference._raise_unobserved_samples(missing)
        return (g, positions, self.geometry.block_of(self.sites.pos),
                self.geometry.n_blocks)

    def _hmm_inputs(self):
        """Per-block ``(step, block_mask, emit_scale)`` of
        :meth:`BuildGeometry.hmm_inputs() <ancestree._geometry.BuildGeometry.hmm_inputs>`."""
        return self.geometry.hmm_inputs(self.rec_rate, self.mu, log=self._log,
                                        validate=self._validate_coverage)

    def prepared_hmm(self) -> "tuple[np.ndarray, np.ndarray, HmmInputs]":
        """The genotypes and the HMM inputs prepared from them.

        :return: ``(genotypes, positions, inputs)``: the ``(n_sites, n_hap)``
            allele-index matrix, each site's position on the axis, and the
            :class:`HmmInputs` of
            :meth:`PairwiseCoalescentHMM.prepare() <ancestree.local_tree_inference.PairwiseCoalescentHMM.prepare>`.
        """
        g, positions, block_of_site, n_blocks = self._genotype_matrix()
        return g, positions, self._prepare(g, block_of_site, n_blocks)

    def _prepare(self, g, block_of_site, n_blocks) -> HmmInputs:
        """The :class:`HmmInputs` of genotypes ``g`` on this build's blocks."""
        step, block_mask, emit_scale = self._hmm_inputs()
        return PairwiseCoalescentHMM(
            len(self.sample_names), mu=self.mu, rec_rate=self.rec_rate,
            block_size=self.block_size, n_time_bins=self.n_time_bins,
            time_grid=self.time_grid,
        ).prepare(g, block_of_site, n_blocks, step=step,
                  block_mask=block_mask, emit_scale=emit_scale)

    def to_tree_sequence(self):
        """Build (and cache) the self-describing local-tree sequence.

        :return: A :class:`tskit.TreeSequence` on the builder's axis, one local
            tree per window over the data stretch (see
            :attr:`BuildGeometry.tree_span <ancestree._geometry.BuildGeometry.tree_span>`)
            and no edges elsewhere, branch lengths in generations, sample names
            in individual metadata, and the observed genotypes baked in as
            sites.
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
        geometry = self.geometry
        trees = geometry.tree_windows
        intervals = [geometry.windows[w] for w in trees]
        lo, hi = (b[trees] for b in geometry.window_blocks)
        # Exact UPGMA over the full pairwise-TMRCA matrix (combinations order).
        condensed = [pair_block_t[:, b0:b1].mean(axis=1) for b0, b1 in zip(lo, hi)]
        from ancestree._ensemble import window_exposure
        upgma = WindowedUpgmaTreeSequence(
            len(self.sample_names), self.axis_length,
            intervals, condensed, sample_names=self.sample_names,
            window_weights=window_exposure(
                self._pair_called, self._pair_block_sites, lo, hi),
        )
        topo = upgma.build()
        span = geometry.tree_span
        if span is not None and intervals and span != (intervals[0][0],
                                                       intervals[-1][1]):
            topo = topo.keep_intervals([span], simplify=False,
                                       record_provenance=False)
        if not self.bake_genotypes:
            self._ts = topo
            return self._ts

        # Bake the genotypes in as sites, each with a minimal map_mutations
        # placement on its window's own tree, so ts.variants() reproduces them.
        tables = topo.dump_tables()
        alleles = list(STATES)
        n = len(self.sample_names)
        order = np.argsort(pos0, kind="stable")
        # Only the sites under a tree are baked.
        lo_site, hi_site = (0, 0) if span is None else np.searchsorted(
            pos0[order], span, side="left")
        window_of = np.searchsorted([iv[0] for iv in intervals],
                                    pos0[order], side="right") - 1
        n_uncalled = 0
        w_tree = tree = None
        for si in range(lo_site, hi_site):
            idx = order[si]
            row = g[idx].astype(np.int8)
            if not (row >= 0).any():
                # map_mutations rejects an all-missing column, and emitting
                # the site without a mutation would read as every tip
                # carrying the ancestral allele. It is left out; callers
                # pair by position, not by order.
                n_uncalled += 1
                continue
            w = int(window_of[si])
            if w != w_tree:
                tree, w_tree = upgma.window_tree(w), w
            shift = w * (n - 1)
            anc, muts = tree.map_mutations(row, alleles)
            site_id = tables.sites.add_row(position=float(pos0[idx]),
                                           ancestral_state=anc)
            mut_ids: list[int] = []
            for mut in muts:
                parent = (mut_ids[mut.parent]
                          if mut.parent != tskit.NULL else tskit.NULL)
                mut_ids.append(tables.mutations.add_row(
                    site=site_id,
                    node=mut.node if mut.node < n else mut.node + shift,
                    derived_state=mut.derived_state, parent=parent,
                ))
        if n_uncalled:
            self._log.warning(
                "%d of %d sites carry no called genotype in the panel; they "
                "are left out of the tree sequence and score at the prior",
                n_uncalled, pos0.shape[0])
        tables.sort()
        tables.time_units = "generations"
        self._ts = tables.tree_sequence()
        return self._ts

    def _pairwise_block_tmrcas(
        self, g: np.ndarray, block_of_site: np.ndarray, n_blocks: int,
    ) -> np.ndarray:
        """Run (and cache) the pairwise-coalescent HMM.

        :return: ``(n(n-1)/2, n_blocks)`` posterior-mean TMRCA matrix, rows in
            :func:`itertools.combinations` order.
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
        inputs = self._prepare(g, block_of_site, n_blocks)
        self._pair_block_t = PairwiseCoalescentHMM.posterior_means(inputs)
        self._pair_called, self._pair_block_sites = (inputs.called,
                                                     inputs.block_sites)
        return self._pair_block_t

    def pairwise_tmrcas(self) -> PairwiseTmrcas:
        """Inferred per-block pairwise TMRCAs (the raw HMM output).

        Runs (and caches) the pairwise-coalescent HMM and returns its
        posterior-mean TMRCA matrix with pair and block labels, the
        intermediate that :meth:`to_tree_sequence` condenses into per-window
        trees. Exposed for inspection and debugging of genealogy quality;
        :meth:`PairwiseTmrcas.write` dumps it to file.

        All ``n(n-1)/2`` pairs, in :func:`itertools.combinations` order, over
        the blocks overlapping
        :attr:`BuildGeometry.tree_span <ancestree._geometry.BuildGeometry.tree_span>`,
        at genome coordinates. A block straddling the edge of a chunked core
        is reported by the core holding its midpoint clipped to the data
        stretch.

        :return: A :class:`~ancestree.local_tree_inference.PairwiseTmrcas` record.
        """
        g, _pos0, block_of_site, n_blocks = self._genotype_matrix()
        pair_block_t = self._pairwise_block_tmrcas(g, block_of_site, n_blocks)
        names = self.sample_names
        pairs = tuple((names[a], names[b])
                      for a, b in combinations(range(len(names)), 2))
        geometry = self.geometry
        edges = geometry.block_edges
        lo, hi = edges[:-1], edges[1:]
        mids = 0.5 * (lo + hi)
        span, core = geometry.tree_span, geometry.core
        s0, s1 = geometry.stretch
        owner = np.clip(mids, s0, np.nextafter(s1, -np.inf))
        keep = (np.zeros(n_blocks, dtype=bool) if span is None else
                (hi > span[0]) & (lo < span[1])
                & (owner >= core[0]) & (owner < core[1]))
        return PairwiseTmrcas(tuple(names), pairs,
                              geometry.origin + mids[keep],
                              pair_block_t[:, keep])

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
        length, or to the contig lengths of a VCF header or VCZ store. A
        contig with no length ends at its last site. Under ``chunk_size`` the
        length is what lays one window grid over a contig and not one per
        segment.
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
    :param progress: Show a tqdm bar over segments.
    :param n_workers: Fork-pool workers, across which whole segments are
        fanned, so peak memory scales with ``n_workers``. Requires the
        ``fork`` start method (falls back to single-threaded with a warning).
    :param chunk_size: Segment span (bp int or ``"5mb"``-style string), raised
        to the resolved block width when narrower, which shifts the posteriors
        as any change of segmentation does, or ``None`` to build each contig
        as one segment. One contiguous segment is processed at a time, so
        peak memory tracks the segment span. Contigs are independent
        segments, and a contig longer than ``chunk_size`` is sliced into
        cores of that size. A chunked and an unchunked pass agree closely,
        not exactly. Requires sites in sorted genomic order.
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
        :class:`~ancestree.local_tree_inference.LocalTreeBuilder`. Each map is
        sliced per segment.
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

    #: Where the contig lengths came from when the input declared them, named
    #: when a site lies past one, or ``None`` when ``sequence_length`` was
    #: passed.
    _length_source: "str | None" = None

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
        """The samples every site carries with no allele, stopping once every
        sample is called.

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
        # The rate infer() scores each contig at, filled on first use.
        self._kernel_mus: dict = {}
        self.chrom = chrom
        self._announced = False
        # The cached genome-wide TS.
        self._stitched: "tskit.TreeSequence | None" = None
        # Constructed here rather than defaulted in the signature: models carry
        # mutable state that would otherwise be shared across inferences.
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
                self._length_source = "the tree sequence"
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
        self._input_source = source
        if sequence_length is None:
            sequence_length = Inference._contig_lengths(self) or None
            if sequence_length is not None:
                self._length_source = {
                    "trees": "the tree sequence", "vcf": "the VCF header",
                    "vcz": "the VCF Zarr store"}[source.format]
        # A partial base is a whole one, on both paths.
        self._declared_length = (
            {c: math.ceil(v) for c, v in sequence_length.items()}
            if isinstance(sequence_length, Mapping)
            else None if sequence_length is None
            else math.ceil(sequence_length))

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
        ContigGeometry.resolve_widths(window, block_size, 1, 1.0)
        self._chunk_size = (None if chunk_size is None
                            else _parse_bp(chunk_size, name="chunk_size"))
        auto_halo = isinstance(halo, str) and halo.strip().lower() == "auto"
        if chunk_size is not None and not auto_halo:
            _parse_bp(halo, name="halo")

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

        # All work is deferred to infer(), which builds one contiguous segment
        # at a time.
        self._source: "Iterable[Site] | SiteSource" = source
        self._halo = "auto" if auto_halo else halo

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
            geometry = self.segmentation.contig(contig)
            self._kernel_mus[contig] = _filled(
                geometry.mutation_map, self.mu, float(geometry.axis_end))
        mu = self._kernel_mus[contig]
        if isinstance(mu, float) or x < float(mu.sequence_length):
            return mu
        return self.mu

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
        yield from self._with_baseline_check(
            self._emitted(self._infer_segmented()))
        self._log_uniform_fallback_summary()

    def _score_ensemble(self, builder, sites, keep=None,
                        segment=("", 0)) -> "tuple[np.ndarray, _Tally]":
        """Prior-weighted per-site posteriors for one builder's sites.

        :param builder: The :class:`~ancestree.local_tree_inference.LocalTreeBuilder` covering ``sites``.
        :param sites: Those sites, in order.
        :param keep: Boolean mask over ``sites`` of the rows returned and
            tallied, ``None`` for all.
        :param segment: ``(contig, core start)`` keying the segment's draws.
        :return: ``(n_kept, 4)`` posteriors and the uniform-fallback tally.
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
        post, n_bad = self._weight_by_prior(post, sites)
        return post, _Tally(uniform_fallback=int(uniform.sum()) + n_bad)

    # ------------------------------------------------------------- segments
    @cached_property
    def segmentation(self) -> Segmentation:
        """The resolved block, window, chunk and halo, and each contig's
        geometry.

        Resolved on first use, which reads the source once when a width, a
        contig's length or the halo depends on the sites.

        :return: The :class:`~ancestree._geometry.Segmentation`.
        """
        return Segmentation(
            self._source, window=self.window, block_size=self.block_size,
            chunk=self._chunk_size, halo=self._halo,
            declared=self._declared_length, length_source=self._length_source,
            position_base=self._position_base, maps=self._maps_on, mu=self.mu,
            rec_rate=self.rec_rate, ingroup=self._resolved_ingroup,
            log=self._log)

    def segments(self) -> "Iterator[LocalTreeBuilder]":
        """The builder of every segment that carries a tree, in genome order.

        Each builds its segment in a frame of its own, placed at genome
        coordinate ``geometry.origin``, with trees over its core's part of the
        contig's data stretch. Nothing is built until a builder method runs,
        so iterating holds one segment's sites at a time.

        :return: Iterator of
            :class:`~ancestree.local_tree_inference.LocalTreeBuilder`.
        """
        for sites, geometry in self.segmentation.units():
            if geometry.tree_span is not None:
                yield self._segment_builder(sites, geometry)

    def _one_contig(self, what: str) -> "Iterator[LocalTreeBuilder]":
        """:meth:`segments`, for an output laid on one coordinate axis.

        :param what: Caller name, for the message.
        :return: Iterator of the builders.
        :raises NotImplementedError: If the segments span two contigs.
        """
        seen = None
        for builder in self.segments():
            chrom = builder.geometry.contig
            if seen is not None and chrom != seen:
                raise NotImplementedError(
                    f"LocalTreeInference: {what} supports a single contig; "
                    f"the source spans {seen!r} and {chrom!r}. Positions from "
                    f"different contigs would collide on one axis. Use VCF / "
                    f"VCZ output for multi-contig data.")
            seen = chrom
            yield builder

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

    def _segment_builder(self, sites, geometry) -> "LocalTreeBuilder":
        """The :class:`~ancestree.local_tree_inference.LocalTreeBuilder` of one
        segment, its sites shifted to the frame of ``geometry``.

        :param sites: The segment's sites, in genome coordinates.
        :param geometry: The segment's
            :class:`~ancestree._geometry.BuildGeometry`.
        :return: The builder.
        """
        origin = geometry.origin
        builder = LocalTreeBuilder(
            [replace(s, pos=int(s.pos - origin)) for s in sites] if origin
            else sites,
            mu=self.mu, rec_rate=self.rec_rate,
            sample_names=self.sample_names, n_time_bins=self.n_time_bins,
            time_grid=self.time_grid, validate_coverage=False,
            geometry=geometry)
        builder._quiet = True  # per-segment: parent logs the segmented build
        return builder

    def _process_segment(self, work_unit):
        """Build one segment's trees and score its core sites, the halo
        discarded.

        :param work_unit: One ``(sites, geometry)`` of
            :meth:`Segmentation.units() <ancestree._geometry.Segmentation.units>`.
        :return: ``(rows, tally)``: ``(site, posterior values)`` per core site,
            in order, and the segment's diagnostic
            :class:`~ancestree.inference._Tally`.
        """
        seg, geometry = work_unit
        core_lo, core_hi = geometry.genome_core
        if self.n_ensemble is not None:
            rows, tally = self._process_segment_ensemble(work_unit)
            # The per-tree counts have no single value over the n_ensemble
            # trees of a window, so they are not reported.
            core = [site for site, _ in rows]
            return rows, (tally + self._count_ingroup_monomorphic(core)
                          + self._count_unrepresentable(core))
        core = [site for site in seg if core_lo <= site.pos < core_hi]
        if not core:
            return [], _Tally()
        segment = self._segment_builder(*work_unit)
        origin = geometry.origin
        # The trees lie on the core alone, so the per-tree tallies count no
        # halo tree.
        arg = ARGBasedInference(
            segment.to_tree_sequence(), self.model,
            mu=geometry.kernel_mu(self.mu), prior=self.prior,
            base_composition=self.base_composition, progress=False,
            n_workers=1, focal=self.focal,
            ingroup_samples=self._resolved_ingroup,
            outgroup_samples=self._resolved_outgroups or None,
        )
        arg._quiet = True
        # Paired by position in the segment's frame (pos - origin). A site
        # absent from the baked tree sequence scores at the prior.
        scored = {int(ts_site.pos): np.asarray(post.values)
                  for ts_site, post in arg.infer()}
        n_states = len(self.model.states)
        flat = np.full((1, n_states), 1.0 / n_states)
        out = [(site, scored.get(int(site.pos - origin))) for site in core]
        out = [(site, self._weight_by_prior(flat, [site])[0][0]
                if values is None else values) for site, values in out]
        # The per-site counts are tallied over the core rows the segment
        # emits, never the halo.
        per_tree = replace(arg._tally, ingroup_monomorphic=0,
                           unrepresentable_sites=0, unrepresentable_tips=0)
        return out, (per_tree + self._count_ingroup_monomorphic(core)
                     + self._count_unrepresentable(core))

    def _process_segment_ensemble(
        self, work_unit,
    ) -> "tuple[list[tuple[Site, np.ndarray]], _Tally]":
        """Score one segment's core sites, marginalised over
        :paramref:`n_ensemble <ancestree.local_tree_inference.LocalTreeInference.n_ensemble>`
        genealogies drawn from the pairwise-HMM posterior.

        :param work_unit: One ``(sites, geometry)`` of
            :meth:`Segmentation.units() <ancestree._geometry.Segmentation.units>`.
        :return: ``(rows, tally)`` as :meth:`_process_segment`.
        """
        seg, geometry = work_unit
        core_lo, core_hi = geometry.genome_core
        keep = np.fromiter((core_lo <= s.pos < core_hi for s in seg), bool,
                           len(seg))
        post, tally = self._score_ensemble(
            self._segment_builder(*work_unit), seg, keep,
            segment=(geometry.contig, core_lo))
        return list(zip(compress(seg, keep), post)), tally

    def _weight_by_prior(self, post, sites):
        """Multiply marginalised likelihoods by the root prior and renormalise.

        :param post: ``(n_sites, S)`` normalised likelihoods.
        :param sites: The sites those rows correspond to, in the same order.
        :return: ``(n_sites, S)`` prior-weighted posteriors and the number of
            rows that took the uniform fallback.
        """
        prior = self.prior
        if prior is None:
            if self.base_composition is None:
                return post, 0  # uniform prior: weighting is a no-op
            prior = StationaryPrior(self.model, self.base_composition)
        with np.errstate(divide="ignore"):
            log_post = np.log(np.asarray(post, dtype=float))
        log_post = log_post + prior.log_probs(list(sites))
        return _normalise(log_post, self.model.n_states)

    def _ensemble_fraction(self) -> float:
        """Fraction along the anchor-to-root path the ensemble reads at, ``1``
        for a focal node not anchored at the ingroup MRCA."""
        spec = self.focal
        if not self._focal_anchored_at_ingroup_mrca:
            return 1.0
        return float(spec.fraction or 0.0)

    def _ensemble_placement(self) -> "tuple[float, int]":
        """Where the ensemble reads.

        :return: ``(fraction, n_coal)``, ``n_coal`` the join count above the
            anchor or ``-1`` when ``fraction`` places the readout.
        """
        spec = self.focal
        if self._focal_anchored_at_ingroup_mrca and spec.coalescences is not None:
            return 0.0, int(spec.coalescences)
        return self._ensemble_fraction(), -1

    def _stitch_segment_tree_sequence(self) -> "tskit.TreeSequence":
        """The genome-wide tree sequence over the data stretch of a single
        contig, stitched from the segments built one at a time.

        :return: The stitched pseudo-ARG tree sequence.
        :raises NotImplementedError: If the source spans more than one contig.
        :raises ValueError: If the source yields no segments.
        """
        parts = []
        chrom = None
        for builder in self._one_contig("point_tree_sequence() / to_arg()"):
            chrom = builder.geometry.contig
            parts.append((builder.to_tree_sequence(), builder.geometry.origin))
        if chrom is None:
            raise ValueError("no segments to build a tree sequence from")
        return self._stitch(parts, self.segmentation.contig(chrom).axis_end)

    def _stitch(self, parts, length: float) -> "tskit.TreeSequence":
        """Tree sequences built in segment frames, placed on one genome axis.

        :param parts: ``(tree sequence, origin)`` pairs, each tree sequence
            holding the panel as its first nodes and its frame's zero at genome
            coordinate ``origin``.
        :param length: Length of the genome axis.
        :return: One tree sequence sharing the panel's sample nodes, every
            part's other nodes, edges, sites and mutations shifted onto it.
        """
        import tskit
        n = len(self.sample_names)
        out = tskit.TableCollection(sequence_length=length)
        out.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        for name in self.sample_names:
            ind = out.individuals.add_row(metadata={"name": str(name)})
            out.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0, individual=ind)
        for ts, origin in parts:
            t = ts.tables
            node_map = np.arange(t.nodes.num_rows, dtype=np.int32)
            node_map[n:] += out.nodes.num_rows - n
            out.nodes.append_columns(
                flags=np.zeros(t.nodes.num_rows - n, dtype=np.uint32),
                time=t.nodes.time[n:])
            out.edges.append_columns(
                left=t.edges.left + origin, right=t.edges.right + origin,
                parent=node_map[t.edges.parent], child=node_map[t.edges.child])
            site0, mut0 = out.sites.num_rows, out.mutations.num_rows
            out.sites.append_columns(
                position=t.sites.position + origin,
                ancestral_state=t.sites.ancestral_state,
                ancestral_state_offset=t.sites.ancestral_state_offset)
            parent = t.mutations.parent
            out.mutations.append_columns(
                site=t.mutations.site + site0,
                node=node_map[t.mutations.node],
                parent=np.where(parent == tskit.NULL, tskit.NULL, parent + mut0),
                time=t.mutations.time,
                derived_state=t.mutations.derived_state,
                derived_state_offset=t.mutations.derived_state_offset)
        out.sort()
        out.time_units = parts[0][0].time_units if parts else "generations"
        return out.tree_sequence()

    def _infer_segmented(self) -> Iterator[tuple[Site, Posterior]]:
        """Posteriors, one segment at a time."""
        # Resolves the segmentation before any worker forks, so they inherit it.
        units = self._segments_with_bar()
        states = self.model.states
        pool = _ForkPool(Settings._resolve_n_workers(self.n_workers),
                         self._log, "genomic segments")
        for seg_out, tally in pool.map(self._process_segment, units):
            self._tally += tally
            for orig, values in seg_out:
                yield orig, Posterior(alleles=states, values=values)

    def _segments_with_bar(self):
        """The segment work-units, under a progress bar when requested."""
        # A segment holding no site in its core has no posterior to report.
        segmentation = self.segmentation
        segments = (wu for wu in segmentation.units()
                    if any(wu[1].genome_core[0] <= site.pos
                           < wu[1].genome_core[1] for site in wu[0]))
        if not self.progress:
            return segments
        end, chunk = self._declared_length, segmentation.chunk
        total = (max(1, -(-int(end) // int(chunk)))
                 if end and not isinstance(end, Mapping) and chunk else None)
        return tqdm(segments, total=total, desc="LocalTreeInference",
                    unit=" segments", disable=Settings.disable_pbar)

    _MODE = "local-tree"

    def _provenance_parameters(self) -> dict[str, object]:
        """Local-tree run parameters: model, prior, rates, tree-building
        widths, contig, ingroup size, maps and ensemble settings."""
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
        segmentation = self.segmentation
        chunked = segmentation.chunk is not None
        contigs = segmentation.contigs
        # The width of the windows tiling each contig, L / n.
        widths = {c: float(segmentation.contig(c).length)
                  / len(segmentation.contig(c).windows) for c in contigs}
        params.update({
            "block_size": int(segmentation.block),
            "halo": int(segmentation.halo) if chunked else None,
            "chunk_size": int(segmentation.chunk) if chunked else None,
            "window_bp": (next(iter(widths.values()))
                          if len(set(widths.values())) == 1
                          else widths or None),
        })
        params["chrom"] = (self.chrom if self.chrom is not None
                           else contigs[0] if len(contigs) == 1
                           else contigs or None)
        params["n_ingroup"] = len(self._resolved_ingroup)
        params.update(self._ensemble_provenance())
        params.update(self._map_provenance())
        return params

    def _ensemble_mode(self) -> bool:
        """Whether this run marginalises over an ensemble."""
        return self.n_ensemble is not None

    def _map_provenance(self) -> dict:
        """Summary of the maps applied, with the files they were read from.

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
        """The ensemble's parameters, empty outside ensemble mode."""
        if not self._ensemble_mode():
            return {}
        return {
            "n_ensemble": int(self.n_ensemble),
            "ensemble_seed": int(self.ensemble_seed),
            "member_chunk": int(self.member_chunk),
        }

    def _ingroup_indices(self):
        """Panel positions of the ingroup, resolved by name.

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
        sequence. There is one group per core, or one per contig when
        unchunked. A contig's genealogies span the windows from the one
        holding its first site to the one holding its last, and the rest of
        the contig carries no trees. The draws are fixed by
        :paramref:`ensemble_seed <ancestree.local_tree_inference.LocalTreeInference.ensemble_seed>`.

        Each draw is built when it is pulled. A group's iterator is consumed
        once, and must be consumed before the next group is pulled. Call
        ``list()`` on a group to keep its draws. Draws are taken per segment,
        so members at the same index in two groups are independent samples.
        :meth:`point_tree_sequence` gives a single genome-wide plug-in tree.

        For 1-based input (VCF / VCZ) the last interval ends at
        ``sequence_length + 1``.

        :return: Iterator of ``(interval, iterator of tskit.TreeSequence)``.
        """
        for builder in self._one_contig("tree_sequences()"):
            geometry = builder.geometry
            core = geometry.genome_core
            members = self._ensemble_members(
                builder, segment=(geometry.contig, core[0]))
            length = self.segmentation.contig(geometry.contig).axis_end
            # Bound at the yield, so a group consumed later keeps its own frame.
            yield (tuple(map(float, core)),
                   map(lambda m, o=geometry.origin, n=length:
                       self._stitch([(m, o)], n), members))

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

        The genome-wide tree sequence is stitched (and cached) from the
        per-segment builds, single contig only. The HMM is
        re-run per segment to assemble the topology, so this is independent of
        (and does not cache) the kernel pass in :meth:`infer`.

        :return: The local-tree sequence (a pseudo-ARG, see :meth:`to_arg`),
            of length ``sequence_length + 1`` for 1-based input (VCF / VCZ).
        :raises NotImplementedError: With a multi-contig source (no single
            coordinate axis to represent it on).
        """
        if self._stitched is None:
            self._stitched = self._stitch_segment_tree_sequence()
        return self._stitched

    def pairwise_tmrcas(self) -> PairwiseTmrcas:
        """Inferred per-block pairwise TMRCAs (the raw HMM output, for debugging).

        Runs :meth:`LocalTreeBuilder.pairwise_tmrcas` per segment and glues
        the per-block records in genome order, so the record covers the blocks
        of the data stretch.

        :return: A :class:`~ancestree.local_tree_inference.PairwiseTmrcas` record.
        :raises NotImplementedError: When the source spans more than one
            contig (the blocks of different contigs would collide on one
            coordinate axis).
        :raises ValueError: If the source yields no segments.
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

        One record per segment,
        trimmed to the blocks whose calls it emits so the halo is not
        reported twice. :meth:`pairwise_tmrcas` concatenates the stream, which
        is the whole matrix and therefore the thing chunking exists to avoid
        holding: iterate instead where memory is the constraint.

        :return: Iterator of :class:`~ancestree.local_tree_inference.PairwiseTmrcas`, in genome order.
        """
        for builder in self._one_contig("pairwise_tmrcas()"):
            part = builder.pairwise_tmrcas()
            if part.block_midpoints.size:
                yield part

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
        """``pairs`` without a source tree handle, and on contig ``chrom``
        where set.

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
        first = next(iter(self._source), None)
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
        """The panel-only
        :meth:`LocalTreeInference.point_tree_sequence() <ancestree.local_tree_inference.LocalTreeInference.point_tree_sequence>`
        that
        :meth:`LocalTreeInference.to_arg() <ancestree.local_tree_inference.LocalTreeInference.to_arg>`
        annotates.

        :param restrict_samples: Whether to hold only the samples the
            inference used.
        :return: The tree sequence to annotate.
        :raises NotImplementedError: With a multi-contig source.
        """
        return self.point_tree_sequence()
