"""Shared helper for the B8 outgroup-divergence sweep scripts.

Defines the four spacing strategies that map a deepest-outgroup split
time ``d`` to a triple of split times ``(t1, t2, t3)`` for outgroups
``(outgroup_1, outgroup_2, outgroup_3)`` (closest-first, all in
generations). Centralised here so the simulate / infer / report
scripts can't drift from one another.

Strategies:

- ``linear``: evenly spaced on the divergence axis: ``(d/3, 2d/3, d)``.
- ``geometric``: log-spaced (each outgroup ~3x deeper): ``(d/9, d/3, d)``.
- ``compact_near``: closer two bunched near the ingroup: ``(d/10, d/5, d)``.
- ``compact_far``: closer two bunched near the deepest split: ``(2d/3, 5d/6, d)``.
"""
from __future__ import annotations


SPACING_STRATEGIES: tuple[str, ...] = (
    "linear", "geometric", "compact_near", "compact_far",
)


def compute_split_times(depth: float, spacing: str) -> list[float]:
    """Return the three outgroup split times (generations) for a given
    deepest-outgroup ``depth`` and ``spacing`` strategy.

    :param depth: Deepest-outgroup split time, in generations.
    :param spacing: One of :data:`SPACING_STRATEGIES`.
    :return: List of three split times, sorted ascending (closest first).
    :raises ValueError: If ``spacing`` is not a recognised strategy.
    """
    d = float(depth)
    if spacing == "linear":
        times = [d / 3.0, 2.0 * d / 3.0, d]
    elif spacing == "geometric":
        times = [d / 9.0, d / 3.0, d]
    elif spacing == "compact_near":
        times = [d / 10.0, d / 5.0, d]
    elif spacing == "compact_far":
        times = [2.0 * d / 3.0, 5.0 * d / 6.0, d]
    else:
        raise ValueError(
            f"Unknown spacing {spacing!r}; "
            f"expected one of {list(SPACING_STRATEGIES)}"
        )
    # Float arithmetic may give a not-quite-sorted result on extreme
    # depth scales. Sort defensively before returning.
    return sorted(times)


#: Outgroup counts the sweep subsets the simulated three outgroups down to.
OUTGROUP_COUNTS: tuple[int, ...] = (1, 2, 3)


def select_outgroup_pops(ordered_pops: list[str], n_out: int) -> list[str]:
    """Subset an ordered (closest-first) outgroup-population list down to
    ``n_out`` populations.

    ``n_out == 3`` keeps all three populations, ``n_out == 2`` keeps the
    first and the last (shallowest and deepest split), and ``n_out == 1``
    keeps the middle one.

    :param ordered_pops: Outgroup population names sorted by split time
        ascending (closest to the ingroup first).
    :param n_out: Number of outgroup populations to retain, in ``{1, 2, 3}``.
    :return: The retained population names, in the input order.
    :raises ValueError: If ``n_out`` is not in ``{1, 2, 3}`` or exceeds the
        number of available populations.
    """
    n = int(n_out)
    if n not in OUTGROUP_COUNTS:
        raise ValueError(
            f"n_out={n_out!r}; expected one of {list(OUTGROUP_COUNTS)}"
        )
    if n > len(ordered_pops):
        raise ValueError(
            f"n_out={n} exceeds the {len(ordered_pops)} available "
            f"outgroup populations {ordered_pops}"
        )
    if n == len(ordered_pops):
        return list(ordered_pops)
    if n == 2:
        return [ordered_pops[0], ordered_pops[-1]]
    return [ordered_pops[len(ordered_pops) // 2]]
