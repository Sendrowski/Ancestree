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

:class:`ContigGeometry` lays the window grid, the chunked cores and their
segments over one contig. :class:`BuildGeometry` is what one build reads, the
whole region or one segment shifted to a local 0: its blocks, windows, map
slices and the windows that get trees.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

from ancestree._repr import ReprMixin

#: Target SNPs per HMM emission block when ``block_size`` is left unset. A block
#: finer than the window gives the pairwise HMM several emission units per
#: output tree.
_DEFAULT_BLOCK_SNPS = 4

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
    ``origin``. Block ``k`` spans ``[k·B, (k+1)·B)`` for the block width ``B``,
    and the block reaching ``L_f`` ends at the axis end. Everything is resolved
    on construction.

    :param length: Frame length ``L_f`` in bp.
    :param block: Block width ``B`` in bp.
    :param window: Window width in bp, a whole number of blocks.
    :param position_base: Frame position base ``b_f``.
    :param windows: ``(left, right)`` windows in the frame, or ``None`` to tile
        each segment between ``segment_breaks`` with windows of ``window`` bp.
    :param segment_breaks: Positions at which the HMM resets and window tiling
        restarts, snapped to the block grid.
    :param recombination_map: :class:`msprime.RateMap` on the frame, or ``None``.
    :param accessibility: Half-open callable ``(start, end)`` intervals on the
        frame, or ``None`` for all callable. The HMM reads only their part
        within ``data``.
    :param mutation_map: :class:`msprime.RateMap` on the frame, or ``None``.
    :param core: ``(lo, hi)`` stretch of the frame the build reports trees and
        blocks for, or ``None`` for the whole axis.
    :param data: ``(lo, hi)`` extent of the contig's sites on the frame, from
        the first site to one past the last, which a window must overlap to
        get a tree, or ``None`` when there is no site, so no window gets one.
        The HMM reads the frame outside it as not callable.
    :param origin: Genome coordinate of the frame's 0.
    :param contig: Contig the frame lies on.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"origin": self.origin, "length": self.length,
                "block": self.block, "window": self.window}

    def __init__(self, length: float, *, block: int, window: int,
                 position_base: int = 0, windows=None, segment_breaks=(),
                 recombination_map=None, accessibility=None,
                 mutation_map=None, core=None, data=None, origin: int = 0,
                 contig: "str | None" = None) -> None:
        self.length = float(length)
        self.position_base = int(position_base)
        self.block = bs = int(block)
        self.window = int(window)
        self.segment_breaks = list(segment_breaks) if segment_breaks else []
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
        #: Blocks the frame is cut into, at least one.
        self.n_blocks = n = max(1, int(np.ceil(self.length / bs)))
        edges = np.arange(n + 1, dtype=np.float64) * bs
        #: ``(n_blocks + 1,)`` float64 block edges, the last at the axis end.
        self.block_edges = np.where(edges >= self.length, self.axis_length,
                                    edges)
        # The segments between breaks, snapped to the block grid so every
        # block lies in one.
        cuts = sorted({0.0, self.length} | {
            float(round(float(b) / bs) * bs) for b in self.segment_breaks
            if 0.0 < round(float(b) / bs) * bs < self.length})
        segments = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
        #: Block-index ranges, one per segment, partitioning ``[0, n_blocks)``.
        self.seg_bounds = [
            (b0, b1) for b0, b1 in
            ((min(n, max(0, int(s0 // bs))), min(n, max(0, int(np.ceil(s1 / bs)))))
             for s0, s1 in segments) if b1 > b0]
        if windows is None:
            windows = np.concatenate([_tile(s0, s1, self.window)
                                      for s0, s1 in segments])
            windows[-1, 1] = self.axis_length
        #: ``(n_windows, 2)`` float64 ``(left, right)`` windows in the frame.
        #: Tiled, none crosses a segment break and the last reaches the axis
        #: end.
        self.windows = iv = np.asarray(windows, dtype=np.float64).reshape(-1, 2)
        lo = np.clip(np.floor_divide(iv[:, 0], bs).astype(np.int64), 0, n - 1)
        hi = np.ceil(iv[:, 1] / bs).astype(np.int64)
        #: ``(lo, hi)`` int64 block range ``[lo, hi)`` each window covers.
        self.window_blocks = (lo, np.maximum(lo + 1, np.minimum(hi, n)))
        lo_, hi_ = (0.0, 0.0) if data is None else data
        #: Indices of the windows that get trees: those overlapping both
        #: :attr:`core` and :attr:`data`.
        self.tree_windows = np.flatnonzero(
            (iv[:, 1] > self.core[0]) & (iv[:, 0] < self.core[1])
            & (iv[:, 1] > lo_) & (iv[:, 0] < hi_))
        #: ``(lo, hi)`` stretch of the frame carrying trees, the tree windows
        #: within :attr:`core` and the axis, or ``None`` when empty.
        self.tree_span = None
        if self.tree_windows.size:
            left = max(0.0, float(self.core[0]), iv[self.tree_windows[0], 0])
            right = min(self.axis_length, float(self.core[1]),
                        iv[self.tree_windows[-1], 1])
            if right > left:
                self.tree_span = (float(left), float(right))

    def kernel_mu(self, mu: float):
        """The rate the HMM and the kernel score the frame at.

        :param mu: The reference rate.
        :return: ``mu``, or :attr:`mutation_map` with the intervals it leaves
            undefined, and the stretch from its end to the axis end, at ``mu``.
        """
        return _filled(self.mutation_map, mu, float(self.axis_length))

    def site_rate_scale(self, positions: np.ndarray,
                        mu: float) -> "np.ndarray | None":
        """Mutation rate at each frame position relative to ``mu``.

        :param positions: Positions on the frame.
        :param mu: The reference rate.
        :return: ``(n,)`` positive multipliers, a zero rate floored, or
            ``None`` without a :attr:`mutation_map`.
        """
        if self.mutation_map is None:
            return None
        from ancestree.inference import _ZERO_RATE_FLOOR
        mm = self.kernel_mu(mu)
        x = np.clip(np.asarray(positions, dtype=np.float64), 0.0,
                    np.nextafter(float(mm.sequence_length), 0.0))
        rate = np.asarray(mm.get_rate(x), dtype=np.float64)
        return np.maximum(rate, _ZERO_RATE_FLOOR) / float(mu)

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
        :param log: Logger the zero-rate warning goes to, once per frame.
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
                if not self._warned_zero_rate:
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
    :param chunk: Core width in bp.
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
        self.chunk, self.halo = int(chunk), int(halo)
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
        origin = (max(0, core_lo - self.halo) // bs) * bs
        span = min(int(self.length), hi) - origin
        axis = span + (self.position_base if hi >= self.axis_end else 0)
        # A window overlapping the core is clipped to the segment's axis.
        iv = self.windows
        a = np.searchsorted(iv[:, 1], core_lo, side="right")
        b = np.searchsorted(iv[:, 0], core_hi, side="left")
        windows = np.column_stack((np.maximum(iv[a:b, 0] - origin, 0.0),
                                   np.minimum(iv[a:b, 1] - origin, float(axis))))
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
