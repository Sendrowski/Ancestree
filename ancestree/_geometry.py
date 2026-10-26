"""Block, window and segment geometry of local-tree builds.

A contig of region length ``L`` bp, read from a source whose first base sits at
position ``b`` (``1`` for VCF / VCF-Zarr, ``0`` otherwise), lies on the axis
``[0, L + b)``, so its last base ``L + b - 1`` lies on it. Blocks of width
``B`` bp tile the axis from 0, the last ending at the axis end. Windows tile it
into ``n = max(1, round(L / w))`` windows of ``L / n`` bp for a window width
``w`` bp, the last ending at the axis end.

Trees cover a contig's data stretch: the whole windows from the one holding
its first site to the one holding its last. A stretch without sites inside it
is tiled, and everything outside it is left without trees.

:class:`Segmentation` resolves the widths, chunk and halo of a run once and
holds a :class:`ContigGeometry` per contig, which lays the window grid, the
cores and their segments over the contig. :class:`BuildGeometry` is what one
build reads, one segment shifted to a local 0: its blocks, windows, map slices
and the windows that get trees.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ancestree._repr import ReprMixin

#: Target SNPs per HMM emission block when ``block_size`` is left unset. A block
#: finer than the window gives the pairwise HMM several emission units per
#: output tree.
_DEFAULT_BLOCK_SNPS = 4

#: Below this many SNPs per block (panel-wide average), a large fraction of
#: blocks carry no informative sites and the per-block pairwise-TMRCA estimates
#: become noise-dominated. The resolution warning fires.
_MIN_SNPS_PER_BLOCK = 2.0

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


def _snp_count(spec: "int | str", *, name: str) -> "int | None":
    """The N of an ``"<N>snp"`` spec, or ``None`` for a bp / int spec.

    :param spec: The width spec.
    :param name: Parameter the spec was given for, named in the error message.
    :return: The SNP count, or ``None``.
    :raises ValueError: If ``spec`` ends in ``snp`` without a positive count.
    """
    if not (isinstance(spec, str) and spec.strip().lower().endswith("snp")):
        return None
    try:
        n_target = int(spec.strip().lower()[:-3])
    except ValueError:
        raise _length_spec_error(name, spec, snp=True) from None
    if n_target <= 0:
        raise _length_spec_error(name, spec, snp=True)
    return n_target


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
    n_target = _snp_count(spec, name=name)
    if n_target is None:
        return _parse_bp(spec, name=name, snp=True)
    return max(1, int(round(n_target * span_bp / max(1, n_sites))))


def _default_block_bp(n_sites: int, span_bp: float, window_bp: int,
                      n_window: "int | None" = None) -> int:
    """Density-adaptive default block width in bp, at least 1, sized so a
    whole number of blocks spans the window: ``N / _DEFAULT_BLOCK_SNPS``
    rounded half up for an ``"<N>snp"`` window, otherwise the count nearest to
    ``_DEFAULT_BLOCK_SNPS`` SNPs per block at the mean density, at least one.

    :param n_sites: Sites the density is taken over.
    :param span_bp: Span the density is taken over.
    :param window_bp: The requested window width.
    :param n_window: The window's SNP count, or ``None`` for a bp window.
    :return: The block width.
    """
    if n_window is not None:
        n_blocks = max(1, (2 * n_window + _DEFAULT_BLOCK_SNPS)
                       // (2 * _DEFAULT_BLOCK_SNPS))
    else:
        target = _DEFAULT_BLOCK_SNPS * span_bp / max(1, n_sites)
        n_blocks = max(1, int(round(window_bp / max(target, 1.0))))
    return max(1, -(-int(window_bp) // n_blocks))


def _check_on_axis(positions, axis: float) -> None:
    """Refuse site positions off the axis ``[0, axis)``.

    :param positions: Site positions on the axis.
    :param axis: Length of the axis.
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


def _tile(lo: float, hi: float, width: int) -> np.ndarray:
    """``[lo, hi)`` cut into the whole number of equal windows nearest to
    ``hi - lo`` over ``width``, at least one.

    :param lo: Left end.
    :param hi: Right end.
    :param width: Target window width in bp.
    :return: ``(n, 2)`` float64 ``(left, right)`` windows in order, the last
        ending at ``hi``.
    """
    n = max(1, int(round((hi - lo) / width)))
    edges = lo + np.arange(n + 1) * ((hi - lo) / n)
    edges[-1] = hi
    return np.column_stack((edges[:-1], edges[1:]))


def _filled(mm, mu: float, end: float):
    """``mm`` with the intervals it leaves undefined, and the stretch from
    its end to ``end``, at ``mu``.

    :param mm: An :class:`msprime.RateMap`, or ``None``.
    :param mu: The rate filled in.
    :param end: The axis end the map is extended to.
    :return: ``mu`` without a map, ``mm`` itself when it is defined up to
        ``end``, else the filled map.
    """
    if mm is None:
        return mu
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


class BuildGeometry(ReprMixin):
    """The blocks, windows and maps one local-tree build reads, in its frame.

    The frame is the build's own axis ``[0, L_f + b_f)`` for a frame length
    ``L_f`` bp and a frame position base ``b_f``, placed at genome coordinate
    ``origin``. Blocks of width ``B`` bp tile the axis from 0, and the block
    reaching ``L_f`` ends at the axis end. The HMM runs over the blocks
    overlapping the data stretch ``stretch``, so block ``k`` of the build
    spans ``[(k_0 + k)·B, (k_0 + k + 1)·B)`` for the first such
    block ``k_0`` of the axis, :attr:`first_block`. Everything is resolved on
    construction.

    :param length: Frame length ``L_f`` in bp.
    :param block: Block width ``B`` in bp.
    :param window: Window width in bp, a whole number of blocks.
    :param position_base: Frame position base ``b_f``.
    :param windows: ``(left, right)`` windows in the frame, or ``None`` to tile
        the frame with windows of ``window`` bp.
    :param recombination_map: :class:`msprime.RateMap` on the frame, or ``None``.
    :param accessibility: Half-open callable ``(start, end)`` intervals on the
        frame, or ``None`` for all callable. The HMM reads only their part
        within ``data``.
    :param mutation_map: :class:`msprime.RateMap` on the frame, or ``None``.
    :param core: ``(lo, hi)`` stretch of the frame the build reports trees and
        blocks for, or ``None`` for the whole axis.
    :param data: ``(lo, hi)`` extent of the contig's sites on the frame, from
        the first site to one past the last, a position of which a window must
        hold to get a tree, or ``None`` when there is no site, so no window
        gets one.
        The HMM reads the frame outside it as not callable.
    :param stretch: ``(lo, hi)`` data stretch on the frame, from the left edge
        of the contig's window holding the first site to the right edge of
        the one holding the last, or ``None`` to take it from ``windows`` and
        ``data``.
        The HMM runs over the blocks overlapping it.
    :param origin: Genome coordinate of the frame's 0.
    :param contig: Contig the frame lies on.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"origin": self.origin, "length": self.length,
                "block": self.block, "window": self.window}

    def __init__(self, length: float, *, block: int, window: int,
                 position_base: int = 0, windows=None, recombination_map=None,
                 accessibility=None, mutation_map=None, core=None, data=None,
                 stretch=None, origin: int = 0,
                 contig: "str | None" = None) -> None:
        self.length = float(length)
        self.position_base = int(position_base)
        self.block = bs = int(block)
        self.window = int(window)
        self.recombination_map = recombination_map
        self.accessibility = (None if accessibility is None
                              else list(accessibility))
        self.mutation_map = mutation_map
        self.origin = origin
        self.contig = contig
        self.data = data
        self._warned_zero_rate = False
        #: Length of the frame's axis, :attr:`length` plus :attr:`position_base`.
        self.axis_length = self.length + self.position_base
        self.core = (0.0, self.axis_length) if core is None else core
        #: :attr:`core` in genome coordinates.
        self.genome_core = (origin + self.core[0], origin + self.core[1])
        n = max(1, int(np.ceil(self.length / bs)))
        #: Last block of the axis, which holds the bases up to the axis end.
        self._last_block = n - 1
        if windows is None:
            windows = _tile(0.0, self.length, self.window)
            windows[-1, 1] = self.axis_length
        #: ``(n_windows, 2)`` float64 ``(left, right)`` windows in the frame,
        #: the last reaching the axis end.
        self.windows = iv = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
        lo = np.clip(np.floor_divide(iv[:, 0], bs).astype(np.int64), 0, n - 1)
        hi = np.maximum(lo + 1, np.minimum(
            np.ceil(iv[:, 1] / bs).astype(np.int64), n))
        lo_, hi_ = (0.0, 0.0) if data is None else data
        if stretch is None and data is not None:
            held = (iv[:, 1] > lo_) & (iv[:, 0] <= hi_ - 1)
            stretch = ((min(lo_, iv[held, 0].min()), max(hi_, iv[held, 1].max()))
                       if held.any() else data)
        k0, k1 = 0, n
        if stretch is not None:
            s0 = max(0.0, float(stretch[0]))
            s1 = min(self.axis_length, float(stretch[1]))
            if s1 > s0:
                k0 = min(int(s0 // bs), n - 1)
                k1 = min(int(np.ceil(s1 / bs)), n)
        #: First block of the axis the HMM runs over.
        self.first_block = k0
        #: Blocks the HMM runs over, at least one.
        self.n_blocks = k1 - k0
        edges = np.arange(k0, k1 + 1, dtype=np.float64) * bs
        #: ``(n_blocks + 1,)`` float64 edges of the HMM's blocks, the edge at
        #: ``L_f`` moved to the axis end.
        self.block_edges = np.where(edges >= self.length, self.axis_length,
                                    edges)
        lo = np.clip(lo - k0, 0, self.n_blocks - 1)
        #: ``(lo, hi)`` int64 range ``[lo, hi)`` of the HMM's blocks each
        #: window covers.
        self.window_blocks = (lo, np.clip(hi - k0, lo + 1, self.n_blocks))
        #: Indices of the windows that get trees: those overlapping
        #: :attr:`core` and holding a position of :attr:`data`.
        self.tree_windows = np.flatnonzero(
            (iv[:, 1] > self.core[0]) & (iv[:, 0] < self.core[1])
            & (iv[:, 1] > lo_) & (iv[:, 0] <= hi_ - 1))
        #: ``(lo, hi)`` stretch of the frame carrying trees, the tree windows
        #: within :attr:`core` and the axis, or ``None`` when empty.
        self.tree_span = None
        if self.tree_windows.size:
            left = max(0.0, float(self.core[0]), iv[self.tree_windows[0], 0])
            right = min(self.axis_length, float(self.core[1]),
                        iv[self.tree_windows[-1], 1])
            if right > left:
                self.tree_span = (float(left), float(right))

    def block_of(self, positions) -> np.ndarray:
        """The HMM block holding each position.

        :param positions: Positions on the frame, within the HMM's blocks.
        :return: ``(n,)`` int64 block indices into :attr:`block_edges`.
        """
        k = np.asarray(positions) // self.block
        return (np.minimum(k.astype(np.int64), self._last_block)
                - self.first_block)

    def kernel_mu(self, mu: float):
        """The rate the HMM and the kernel score the frame at.

        :param mu: The reference rate.
        :return: ``mu``, or :attr:`mutation_map` with the intervals it leaves
            undefined, and the stretch from its end to the axis end, at ``mu``.
        """
        return _filled(self.mutation_map, mu, float(self.axis_length))

    @cached_property
    def callable_intervals(self) -> "np.ndarray | None":
        """``(n, 2)`` float64 intervals the HMM reads as callable:
        :attr:`accessibility` (the whole axis when unset) within :attr:`data`,
        or ``None`` when neither restricts the axis."""
        if self.data is None:
            return (None if self.accessibility is None else
                    np.asarray(self.accessibility,
                               dtype=np.float64).reshape(-1, 2))
        iv = (np.array([[0.0, self.axis_length]]) if self.accessibility is None
              else np.asarray(self.accessibility,
                              dtype=np.float64).reshape(-1, 2))
        return np.column_stack((np.maximum(iv[:, 0], float(self.data[0])),
                                np.minimum(iv[:, 1], float(self.data[1]))))

    def accessible_fraction(self) -> np.ndarray:
        """``(n_blocks,)`` float64 in ``[0, 1]``: the base pairs of each block
        covered by :attr:`callable_intervals`, divided by that block's span."""
        edges = self.block_edges
        end = edges[-1]
        iv = self.callable_intervals
        if iv is None:
            return np.ones(self.n_blocks, dtype=np.float64)
        lo = np.clip(iv[:, 0], 0.0, end)
        hi = np.clip(iv[:, 1], 0.0, end)
        keep = hi > lo
        if not keep.any():
            return np.zeros(self.n_blocks, dtype=np.float64)
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
        below = np.empty((2, self.n_blocks + 1), dtype=np.float64)
        for row, bound in enumerate((lo, hi)):
            cum = np.concatenate(([0.0], np.cumsum(bound)))
            j = np.searchsorted(bound, edges, side="right")
            below[row] = cum[j] + (bound.size - j) * edges
        span = np.clip(np.diff(edges), 1e-300, None)
        return np.clip(np.diff(below[1] - below[0]) / span, 0.0, 1.0)

    def hmm_inputs(self, rec_rate: float, mu: float, *, log,
                   validate: bool = True):
        """Per-block HMM inputs: ``(step, block_mask, emit_scale)``.

        ``step[k]`` is the genetic distance of the step from block ``k`` to
        ``k+1`` in nominal-block units, the recombination map's local mass
        over that bp interval divided by the reference ``rec_rate·B``;
        ``None`` when no map is set (uniform).

        ``emit_scale[k]`` is the Poisson exposure scale of block ``k``: the
        product of ``a_k·(S_k/B)``, the accessible base pairs of block ``k`` as
        a fraction of the nominal width ``B``, for a block spanning ``S_k``
        base pairs and an accessible fraction ``a_k`` of that span
        (dimensionless, in ``[0, 1]``, the share of the span within
        :attr:`callable_intervals`), and ``μ_k / μ``, the mutation map's local
        per-bp rate over the block divided by the reference rate ``μ``
        (dimensionless, and ``1`` with no map). The emission mean of block
        ``k`` in time bin ``i`` is then ``a_k·(S_k/B)·(μ_k/μ)·2·μ·B·t_i``
        differences, for a bin representative time ``t_i`` in generations.
        ``None`` when there are no sites, no mask and no map, and every block
        is full width.

        ``block_mask[k]`` is ``True`` when block ``k`` emits: more than
        :data:`_ACCESS_FRAC_FLOOR` of its own base pairs are in
        :attr:`callable_intervals`, it spans more than that fraction of ``B``,
        and its local mutation rate is positive. ``None`` when no block is masked.

        :param rec_rate: Reference recombination rate per bp per generation.
        :param mu: Reference mutation rate per bp per generation.
        :param log: Logger the zero-rate warning goes to, once per frame, or
            ``None`` for none.
        :param validate: Whether to refuse a frame with no callable block.
        :return: ``(step, block_mask, emit_scale)``, each an array over blocks
            or ``None`` for its neutral default.
        :raises ValueError: If ``validate`` and no block is callable.
        """
        bs = self.block
        n_blocks = self.n_blocks
        step = None
        if self.recombination_map is not None:
            rm = self.recombination_map
            edges = self.block_edges
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
            ref = rec_rate * np.clip(span, 1e-300, None)
            step = np.nan_to_num(gdist / ref, nan=1.0, posinf=1e6, neginf=0.0)
            step = np.clip(step, 0.0, None)
            # Steps beyond the map take the reference rate (1.0), so the
            # TMRCA chain keeps decorrelating.
            step = np.where(valid, step, 1.0)
        block_mask = None
        access_frac = None
        if self.callable_intervals is not None:
            access_frac = self.accessible_fraction()
            block_mask = access_frac > _ACCESS_FRAC_FLOOR
        emit_scale = None
        uninformative = None
        if self.mutation_map is not None:
            mm = self.kernel_mu(mu)
            L = float(mm.sequence_length)
            edges = np.minimum(self.block_edges, L)
            lo, hi = edges[:-1], edges[1:]
            valid = hi > lo  # False past the map's end
            width = np.clip(hi - lo, 1e-300, None)
            rate = (np.asarray(mm.get_cumulative_mass(hi), dtype=np.float64)
                    - np.asarray(mm.get_cumulative_mass(lo), dtype=np.float64)
                    ) / width  # local per-bp rate
            emit_scale = np.nan_to_num(
                rate / mu, nan=1.0, posinf=1e6, neginf=0.0,
            )
            # Blocks beyond the map take the reference rate (1.0).
            uninformative = valid & (emit_scale <= _EMIT_SCALE_FLOOR)
            emit_scale = np.where(
                valid, np.clip(emit_scale, _EMIT_SCALE_FLOOR, None), 1.0)
            if uninformative.any():
                if block_mask is None:
                    block_mask = np.ones(n_blocks, dtype=bool)
                block_mask = block_mask & ~uninformative
                if log is not None and not self._warned_zero_rate:
                    log.warning(
                        "mutation-map rate is zero over %d block(s); they "
                        "carry no information about coalescent time, so "
                        "their sites are scored prior-dominated.",
                        int(uninformative.sum()),
                    )
                    self._warned_zero_rate = True
        # lambda_i is 2·mu·B·t_i for every block, so exposure is measured
        # against the nominal width B: a block spanning S_b < B bp offers S_b/B
        # of one whether or not a mask is given, which keeps the masked and
        # unmasked paths on one scale.
        trunc = np.diff(self.block_edges) / float(bs)
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
        # Nor is a data stretch narrower than the floor without a mask.
        if self.accessibility is None and access_frac is not None:
            callable_mask = callable_mask | (
                (access_frac > 0.0) & (trunc > _ACCESS_FRAC_FLOOR))
        if (callable_mask is not None and not callable_mask.any()
                and validate):
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


class ContigGeometry(ReprMixin):
    """The window grid, chunked cores and segments of one contig.

    Cores of ``chunk`` bp tile the contig from 0. The segment of a core
    extends it by ``halo`` bp on each side, starts on the block grid and ends
    at the halo's end or the region's last base, and is built in its own frame
    shifted to a local 0. A segment reads the contig's window grid, so every
    segment calls a position in the same window.

    :param length: Region length ``L`` in bp.
    :param position_base: Coordinate of the region's first base in the sites'
        positions.
    :param block: Block width in bp.
    :param window: Window width in bp, a whole number of blocks.
    :param chunk: Core width in bp, ``0`` for one core over the contig.
    :param halo: Overlap added on each side of a core, in bp.
    :param recombination_map: :class:`msprime.RateMap` on the contig axis.
    :param accessibility: Callable intervals on the contig axis.
    :param mutation_map: :class:`msprime.RateMap` on the contig axis.
    :param contig: The contig.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"contig": self.contig, "length": self.length,
                "block": self.block, "window": self.window}

    def __init__(self, length, *, position_base: int, block: int, window: int,
                 chunk: int = 0, halo: int = 0, recombination_map=None,
                 accessibility=None, mutation_map=None,
                 contig: "str | None" = None) -> None:
        self.length = length
        self.position_base = int(position_base)
        #: End of the contig's axis, ``L`` plus the position base.
        self.axis_end = int(length) + self.position_base
        self.block, self.window = int(block), int(window)
        self.chunk, self.halo = int(chunk) or self.axis_end, int(halo)
        self.recombination_map = recombination_map
        self.accessibility = accessibility
        self.mutation_map = mutation_map
        self.contig = contig

    @cached_property
    def windows(self) -> np.ndarray:
        """``(n, 2)`` float64 window grid over the contig's axis."""
        iv = _tile(0.0, float(self.length), self.window)
        iv[-1, 1] = self.axis_end
        return iv

    def cores(self, start: int) -> "list[tuple[int, int]]":
        """The cores from ``start`` to the end of the region.

        :param start: Left edge of the first core.
        :return: ``(lo, hi)`` cores, the last reaching the axis end.
        """
        out = []
        while start < int(self.length):
            hi = start + self.chunk
            out.append((start, hi if hi < int(self.length) else self.axis_end))
            start = hi
        return out

    def check_callable(self, mu: float, data) -> None:
        """Refuse a contig its mask and mutation map leave without a callable
        block, as an unchunked build of it refuses.

        :param mu: Reference mutation rate per bp per generation.
        :param data: ``(lo, hi)`` genome extent of the contig's sites, to
            infinity while the last is unread.
        :raises ValueError: If no block of the contig is callable.
        """
        if self.accessibility is None and self.mutation_map is None:
            return
        # The blocks overlapping the data stretch, in a frame of their own.
        origin = (int(data[0]) // self.block) * self.block
        end = min(float(data[1]), float(self.axis_end))
        span = int(np.ceil((end - origin) / self.block)) * self.block
        axis = min(span, self.axis_end - origin)
        BuildGeometry(
            axis, block=self.block, window=self.window,
            accessibility=self._slice_intervals(self.accessibility, origin,
                                                axis),
            mutation_map=self._slice_map(self.mutation_map, origin, axis),
            data=(data[0] - origin, end - origin),
        ).hmm_inputs(1.0, mu, log=None)

    def origin(self, core_lo: int) -> int:
        """Genome coordinate of the frame origin of the core from ``core_lo``.

        :param core_lo: Left edge of the core.
        :return: Its halo's left end, snapped down to the block grid.
        """
        return (max(0, core_lo - self.halo) // self.block) * self.block

    def segment(self, core_lo: int, core_hi: int, data) -> BuildGeometry:
        """The frame a core's segment is built in.

        :param core_lo: Left edge of the core.
        :param core_hi: Right edge of the core.
        :param data: ``(lo, hi)`` genome extent of the contig's sites, from the
            first to one past the last, or to infinity while the last is
            unread.
        :return: The segment's :class:`BuildGeometry`.
        """
        bs = self.block
        hi = core_hi + self.halo
        origin = self.origin(core_lo)
        span = min(int(self.length), hi) - origin
        axis = span + (self.position_base if hi >= self.axis_end else 0)
        # A window overlapping the core is clipped to the segment's axis.
        iv = self.windows
        a = np.searchsorted(iv[:, 1], core_lo, side="right")
        b = np.searchsorted(iv[:, 0], core_hi, side="left")
        windows = np.column_stack((np.maximum(iv[a:b, 0] - origin, 0.0),
                                   np.minimum(iv[a:b, 1] - origin, float(axis))))
        # The windows holding the first and the last site.
        first = min(np.searchsorted(iv[:, 1], data[0], side="right"),
                    len(iv) - 1)
        last = max(np.searchsorted(iv[:, 0], data[1] - 1, side="right") - 1,
                   first)
        return BuildGeometry(
            span, block=bs, window=self.window, position_base=axis - span,
            windows=windows,
            recombination_map=self._slice_map(self.recombination_map,
                                              origin, axis),
            accessibility=self._slice_intervals(self.accessibility, origin,
                                                axis),
            mutation_map=self._slice_map(self.mutation_map, origin, axis),
            core=(core_lo - origin, core_hi - origin),
            data=(data[0] - origin, data[1] - origin),
            stretch=(iv[first, 0] - origin, iv[last, 1] - origin),
            origin=origin, contig=self.contig)

    @staticmethod
    def _slice_map(rate_map, origin: int, span: int):
        """Sub-map over ``[origin, origin+span)`` shifted to a local ``0``.

        A build reads per-block rates as cumulative-mass differences over each
        block, so the slice's constant offset cancels.

        :param rate_map: The map to slice, or ``None``.
        :param origin: Left edge of the segment, in base pairs.
        :param span: Width of the slice from ``origin``, in base pairs.
        :return: The sub-map, or ``None`` when unset or when the map defines
            no rate over the slice.
        """
        if rate_map is None:
            return None
        length = float(getattr(rate_map, "sequence_length", origin + span))
        right = min(float(origin + span), length)
        if right <= float(origin):
            return None
        position = np.asarray(rate_map.position, dtype=np.float64)
        defined = ((position[1:] > float(origin)) & (position[:-1] < right)
                   & ~np.isnan(np.asarray(rate_map.rate, dtype=np.float64)))
        if not defined.any():
            return None
        return rate_map.slice(left=float(origin), right=right, trim=True)

    @staticmethod
    def _slice_intervals(accessibility, origin: int, span: int):
        """Accessible intervals intersected with the segment, shifted to 0.

        :param accessibility: The contig's intervals, or ``None``.
        :param origin: Left edge of the segment, in base pairs.
        :param span: Width of the segment's axis, in base pairs.
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

    @staticmethod
    def resolve_widths(window, block_size, n_sites: int,
                       span_bp: float) -> tuple[int, int, int]:
        """The requested window, the block and the window the HMM uses, in bp.

        :param window: The window spec.
        :param block_size: The block spec, or ``None`` for the default.
        :param n_sites: Sites the density is taken over.
        :param span_bp: Span the density is taken over.
        :return: ``(requested, block, window)``, the window rounded up to a
            whole number of blocks, at least one. Two ``"<N>snp"`` specs give
            ``ceil(N_window / N_block)`` blocks.
        """
        requested = _width_bp(window, n_sites, span_bp, name="window")
        n_window = _snp_count(window, name="window")
        if block_size is None:
            block = _default_block_bp(n_sites, span_bp, requested, n_window)
            n_block = None
        else:
            block = _width_bp(block_size, n_sites, span_bp, name="block_size")
            n_block = _snp_count(block_size, name="block_size")
        if n_window is not None and n_block is not None:
            n_blocks = -(-n_window // n_block)
        else:
            n_blocks = max(1, -(-requested // block))
        return requested, block, n_blocks * block


class Segmentation(ReprMixin):
    """The block, window, chunk and halo of a local-tree run, resolved once,
    and the :class:`ContigGeometry` of each contig.

    Resolving reads the sites once when a width is sized from the SNP
    density, the sites per bp between each contig's first and last site (a
    default or ``"<N>snp"`` block, a ``"<N>snp"`` window), when a contig
    declares no length and so ends at its last site, or when the halo is
    ``"auto"``. An ``"auto"`` halo is ``5·L_c``, capped at half the chunk, for
    the bp ``L_c ≈ 1 / (r·E[t])`` over which the HMM's TMRCA decorrelates, with
    ``r`` the recombination rate per bp per generation and ``E[t] = π / (2μ)``
    the mean pairwise TMRCA in generations for the ingroup's per-bp pairwise
    diversity ``π`` and mutation rate ``μ``. Either halo is raised to the
    widest window of every contig with sites and to the block width.

    :param sites: The sites, re-iterable, in sorted order within each contig.
    :param window: The window spec.
    :param block_size: The block spec, or ``None`` for the default.
    :param chunk: Core width in bp, or ``None`` for one core per contig.
    :param halo: Overlap in bp or ``"auto"``, read only with ``chunk``.
    :param declared: The declared lengths: a ``{contig: L}`` mapping, one ``L``
        for every contig, or ``None``.
    :param length_source: Where ``declared`` came from, named when a site lies
        past a length, or ``None`` when it was passed.
    :param position_base: Coordinate of a contig's first base in the sites.
    :param maps: Callable giving the ``{parameter: map}`` of a contig.
    :param mu: Reference mutation rate per bp per generation.
    :param rec_rate: Recombination rate per bp per generation.
    :param ingroup: Ingroup sample names, whose diversity sizes an ``"auto"``
        halo.
    :param log: Logger the resolution warnings go to.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"block": self.block, "window": self.window,
                "chunk": self.chunk, "halo": self.halo}

    def __init__(self, sites, *, window, block_size, chunk, halo, declared,
                 length_source, position_base: int, maps, mu: float,
                 rec_rate: float, ingroup, log) -> None:
        from collections import Counter
        from collections.abc import Mapping

        self.sites = sites
        self.declared = declared
        self.length_source = length_source
        self.position_base = int(position_base)
        self.mu = float(mu)
        self._maps = maps
        self._contigs: dict = {}
        #: Last site of each contig read so far, in input order.
        self.last_sites: dict[str, int] = {}
        self._read_all = False
        need_halo = chunk is not None and halo == "auto"
        per_contig = declared is None or isinstance(declared, Mapping)
        n_sites = span = 0
        pairdiff = 0.0
        if (per_contig or need_halo or block_size is None
                or _snp_count(window, name="window") is not None
                or isinstance(block_size, str)):
            cur = None
            p0 = last = 0
            for s in sites:
                n_sites += 1
                if s.chrom != cur:
                    if cur is not None:
                        span += last - p0 + 1
                    cur, p0 = s.chrom, s.pos
                last = s.pos
                if last > self.last_sites.get(str(cur), -1):
                    self.last_sites[str(cur)] = last
                if need_halo:
                    vals = [a for a in map(s.tip_alleles.get, ingroup)
                            if a is not None]
                    N = len(vals)
                    if N >= 2:
                        same = sum(c * (c - 1) for c in Counter(vals).values())
                        pairdiff += 1.0 - same / (N * (N - 1))
            if cur is not None:
                span += last - p0 + 1
            self._read_all = True
        requested, block, self.window = ContigGeometry.resolve_widths(
            window, block_size, n_sites, span)
        #: Block width in bp.
        self.block = block
        if chunk is not None and chunk < block:
            log.warning(
                "The chunk size of %d bp is narrower than the resolved block "
                "width of %d bp, raising it to the block width, below which a "
                "segment holds no whole emission block and the run splits into "
                "one HMM pass per block", chunk, block)
            chunk = block
        #: Core width in bp, ``None`` for one core per contig.
        self.chunk = chunk
        if requested < block:
            log.warning(
                "Window %r resolved to ~%d bp but was floored to "
                "block_size=%d bp: a window cannot be finer than one block, so "
                "the effective window is %d bp. Lower block_size for finer "
                "local trees.", window, int(requested), block, block)
        if span > 0 and n_sites > 0 and block * n_sites / span \
                < _MIN_SNPS_PER_BLOCK:
            log.warning(
                "A block size of %d bp averages only %.1f SNPs per block (< "
                "%.0f): many blocks carry no informative sites, so the "
                "per-block pairwise-TMRCA estimates are noise-dominated. "
                "Consider a larger block_size.",
                block, block * n_sites / span, _MIN_SNPS_PER_BLOCK)
        #: Halo in bp, ``0`` unchunked.
        self.halo = 0
        if chunk is None:
            return
        # At least one block, so a segment's truncated final block is never
        # emitted beside the next segment's, and at least the widest window,
        # so every window overlapping a core lies inside its segment.
        ends = ([self.declared_on(c) or last + 1
                 for c, last in self.last_sites.items()]
                if per_contig else [declared])
        floor = max(block, int(np.ceil(max(
            (float(np.ptp(ContigGeometry(
                end, position_base=self.position_base, block=block,
                window=self.window).windows[-1])) for end in ends if end),
            default=0.0))))
        if not need_halo:
            self.halo = _parse_bp(halo, name="halo")
            if self.halo < floor:
                log.warning(
                    "The halo of %d bp is narrower than the widest window or "
                    "block of %d bp, raising it to that width so no window is "
                    "cut at a segment edge", self.halo, floor)
                self.halo = floor
            return
        pi = pairdiff / max(1, span)
        e_t = pi / (2.0 * self.mu) if pi > 0 else 0.0
        auto = (int(np.ceil(5.0 / (rec_rate * e_t)))
                if e_t > 0 and rec_rate > 0 else 10 * self.window)
        self.halo = max(floor, min(auto, chunk // 2))

    def declared_on(self, chrom) -> "float | None":
        """The declared length of a contig.

        :param chrom: The contig.
        :return: Its entry in a ``{contig: length}`` declaration, the single
            length declared for every contig, or ``None`` when none is.
        """
        from collections.abc import Mapping

        end = self.declared
        return end.get(str(chrom)) if isinstance(end, Mapping) else end

    def contig(self, chrom) -> "ContigGeometry":
        """The geometry of a contig, laid once.

        :param chrom: The contig.
        :return: Its :class:`ContigGeometry`, at the resolved widths, chunk and
            halo, with its maps, ending at its declared length or else at its
            last site.
        """
        chrom = str(chrom)
        if chrom not in self._contigs:
            length = self.declared_on(chrom)
            if not length:
                if chrom not in self.last_sites:
                    self._read_last_sites()
                length = self.last_sites[chrom] + 1
            self._contigs[chrom] = ContigGeometry(
                length, position_base=self.position_base, block=self.block,
                window=self.window, chunk=self.chunk or 0, halo=self.halo,
                contig=chrom, **self._maps(chrom))
        return self._contigs[chrom]

    @property
    def contigs(self) -> list[str]:
        """The contigs holding sites, in input order, reading the sites once
        if resolving did not."""
        if not self._read_all:
            self._read_last_sites()
        return list(self.last_sites)

    def _read_last_sites(self) -> None:
        """Read every contig's last site into :attr:`last_sites`."""
        for s in self.sites:
            if s.pos > self.last_sites.get(str(s.chrom), -1):
                self.last_sites[str(s.chrom)] = s.pos
        self._read_all = True

    def _refuse_off_axis(self, site, axis_end: float) -> None:
        """Refuse a site past the end of a contig whose length the input
        declares.

        :param site: A site off its contig's axis.
        :param axis_end: End of that axis.
        :raises ValueError: If the length came from the input and the site lies
            at or past ``axis_end``.
        """
        if self.length_source is not None and site.pos >= axis_end:
            raise ValueError(
                f"{self.length_source} gives contig {site.chrom} length "
                f"{int(self.declared_on(site.chrom)):,}, but a site lies at "
                f"{int(site.pos):,}; fix the header or pass sequence_length")

    def units(self):
        """Stream the sites once, yielding ``(sites, geometry)`` per core.

        Sites are streamed in sorted order through a sliding buffer holding
        the sites within about ``chunk + 2·halo`` bp, so neither the full
        input nor a whole contig is materialised. Each contig is cut into the
        cores of :meth:`ContigGeometry.cores`, so a core may hold no site.
        ``sites`` are those of the core's segment and ``geometry`` its
        :meth:`ContigGeometry.segment`, whose trees cover the core's part of
        the contig's data stretch.

        :return: Iterator of ``(sites, geometry)``.
        :raises ValueError: If a site lies off its contig's axis or out of
            order.
        """
        halo = self.halo

        def unit(contig, buf, core_lo, core_hi, data):
            """One work unit: the core plus its halo, from the segment's
            origin on the block grid."""
            geometry = contig.segment(core_lo, core_hi, data)
            seg = [s for s in buf
                   if geometry.origin <= s.pos < core_hi + halo]
            return seg, geometry

        cur = contig = None
        buf: list = []
        core_lo = 0
        first = last = -1
        for s in self.sites:
            if contig is None or s.chrom != cur:
                if contig is not None:
                    for lo, hi in contig.cores(core_lo):
                        yield unit(contig, buf, lo, hi, (first, last + 1))
                cur, buf, core_lo, first = s.chrom, [], 0, s.pos
                last = s.pos
                contig = self.contig(cur)
                contig.check_callable(self.mu, (
                    first, self.last_sites.get(str(cur), np.inf - 1) + 1))
            if not 0 <= s.pos < contig.axis_end:
                self._refuse_off_axis(s, contig.axis_end)
                _check_on_axis(np.array([s.pos]), contig.axis_end)
            if s.pos < last:
                raise ValueError(
                    "LocalTreeInference requires sorted sites; saw "
                    f"{s.chrom}:{s.pos} after {s.chrom}:{last}")
            last = s.pos
            buf.append(s)
            cs = contig.chunk
            while s.pos >= core_lo + cs + halo:  # right halo fully buffered
                yield unit(contig, buf, core_lo, core_lo + cs,
                           (first, np.inf))
                core_lo += cs
                # Drop sites before the next segment's origin.
                cut = contig.origin(core_lo)
                k = 0
                while k < len(buf) and buf[k].pos < cut:
                    k += 1
                if k:
                    del buf[:k]
        if contig is not None:
            for lo, hi in contig.cores(core_lo):
                yield unit(contig, buf, lo, hi, (first, last + 1))
