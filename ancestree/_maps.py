"""Readers for the genome map files local-tree mode accepts.

A BED accessibility mask, a bedGraph mutation map and a HapMap recombination
map carry a contig column, so each is read into one entry per contig. Each
may be plain text or gzip-compressed.
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np

#: Floor for a rate-map mu that is zero or undefined at a site: branch lengths
#: stay near zero and the posterior is prior-dominated.
_ZERO_RATE_FLOOR: float = 1e-12


def site_rates(positions: np.ndarray, mu) -> "tuple[np.ndarray, int]":
    """The rate each site is scored at: ``mu``, or a rate-map ``mu`` at the
    site's position, floored at :data:`_ZERO_RATE_FLOOR` where it is zero or
    undefined.

    :param positions: Site positions (bp).
    :param mu: A scalar rate or a rate map.
    :return: ``(rates, n_floored)``, the ``(n_sites,)`` positive rates and the
        number of sites floored.
    """
    if isinstance(mu, float):
        return np.full(len(positions), mu), 0
    if hasattr(mu, "get_rate"):
        end = float(getattr(mu, "sequence_length", np.inf))
        rate = np.asarray(mu.get_rate(
            np.clip(positions, 0.0, np.nextafter(end, 0.0))), dtype=float)
    else:
        mass = (np.asarray(mu.get_cumulative_mass(positions + 1.0), float)
                - np.asarray(mu.get_cumulative_mass(positions), float))
        # The difference of two cumulative masses carries their rounding.
        rate = np.array([float(f"{m:.9g}") for m in mass])
    bad = ~(rate > 0.0)
    return np.where(bad, _ZERO_RATE_FLOOR, rate), int(bad.sum())


class RateClasses:
    """Per-site rates grouped into the classes the kernel scores.

    Iterating yields ``(rate, rows)`` per class. More than :attr:`MAX_CLASSES`
    distinct rates are binned geometrically over their range, a bin taking
    the geometric mean of its sites.

    :param rate: ``(n_sites,)`` positive rate at each site, absolute or
        relative to a reference rate.
    """

    #: Most classes the sites of one batch are scored in.
    MAX_CLASSES = 16

    def __init__(self, rate) -> None:
        rate = np.asarray(rate, dtype=np.float64)
        self.rates, self._class_of_site = np.unique(rate, return_inverse=True)
        if self.rates.size > self.MAX_CLASSES:
            log_rate = np.log(rate)
            edges = np.linspace(log_rate.min(), log_rate.max(),
                                self.MAX_CLASSES + 1)
            binned = np.clip(np.searchsorted(edges, log_rate, side="right") - 1,
                             0, self.MAX_CLASSES - 1)
            used, self._class_of_site = np.unique(binned, return_inverse=True)
            self.rates = np.array(
                [np.exp(log_rate[binned == b].mean()) for b in used])

    def __len__(self) -> int:
        return int(self.rates.size)

    def __iter__(self):
        for c, rate in enumerate(self.rates):
            yield float(rate), np.flatnonzero(self._class_of_site == c)


class MapFiles:
    """Read accessibility masks and rate maps, and pick a contig's entry."""

    @staticmethod
    def msprime():
        """Import ``msprime``, the ``maps`` extra.

        :return: The module.
        :raises ImportError: If msprime is not installed.
        """
        try:
            import msprime
        except ImportError as e:
            raise ImportError(
                "rate maps require msprime; install it with "
                "`pip install \"ancestree-popgen[maps]\"`.") from e
        return msprime

    @staticmethod
    def _open(path: "str | os.PathLike"):
        """A text handle on a plain or gzip-compressed file.

        :param path: The file.
        :return: The open handle.
        """
        with open(path, "rb") as fh:
            compressed = fh.read(2) == b"\x1f\x8b"
        if compressed:
            return gzip.open(path, "rt", encoding="utf-8")
        return open(path, encoding="utf-8")

    @staticmethod
    def _rows(path: "str | os.PathLike", n_fields: int):
        """The data rows of a BED-like file.

        :param path: The file.
        :param n_fields: Fields a row needs, shorter rows being skipped.
        :return: Generator over the split rows, skipping ``track``, ``browser``,
            comment and blank lines.
        """
        with MapFiles._open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split()
                if len(fields) >= n_fields:
                    yield fields

    @staticmethod
    def read_bed(path: "str | os.PathLike",
                 base: int = 0) -> dict[str, list[tuple[float, float]]]:
        """Half-open ``(start, end)`` intervals of a BED file, per contig.

        :param path: The BED file, 0-based half-open.
        :param base: Coordinate of a contig's first base in the returned
            intervals, ``1`` to match the positions of a VCF.
        :return: ``{contig: intervals}``, in file order.
        """
        intervals: dict[str, list[tuple[float, float]]] = {}
        for f in MapFiles._rows(path, 3):
            intervals.setdefault(f[0], []).append(
                (float(f[1]) + base, float(f[2]) + base))
        return intervals

    @staticmethod
    def read_bedgraph(path: "str | os.PathLike",
                      default_rate: float, base: int = 0) -> dict:
        """One :class:`msprime.RateMap` per contig of a 4-column bedGraph
        (contig, start, end, rate).

        Each contig's rows are sorted, an overlap is clipped to the preceding
        interval, and any gap, including one before the first interval, takes
        ``default_rate``.

        :param path: The bedGraph file.
        :param default_rate: Rate of the gaps.
        :param base: Coordinate of a contig's first base in the returned maps,
            ``1`` to match the positions of a VCF.
        :return: ``{contig: RateMap}`` covering ``[0, last end]``.
        :raises ValueError: If the file yields no usable interval.
        """
        msprime = MapFiles.msprime()
        rows: dict[str, list[tuple[float, float, float]]] = {}
        for f in MapFiles._rows(path, 4):
            rows.setdefault(f[0], []).append(
                (float(f[1]) + base, float(f[2]) + base, float(f[3])))
        maps = {}
        for contig, intervals in rows.items():
            position, rate = [0.0], []
            for start, end, r in sorted(intervals):
                start = max(start, position[-1])
                if end <= start:
                    continue
                if start > position[-1]:
                    position.append(start)
                    rate.append(default_rate)
                position.append(end)
                rate.append(r)
            if len(position) > 1:
                maps[contig] = msprime.RateMap(position=position, rate=rate)
        if not maps:
            raise ValueError(f"no usable intervals in the mutation map {path}")
        return maps

    @staticmethod
    def read_hapmap(path: "str | os.PathLike") -> dict:
        """One :class:`msprime.RateMap` per contig of a HapMap recombination
        map (contig, position, rate in cM/Mb, map position in cM), with a
        header line.

        :param path: The HapMap file.
        :return: ``{contig: RateMap}``, each read by
            :meth:`msprime.RateMap.read_hapmap` from that contig's rows.
        :raises ValueError: If the file holds no data row.
        """
        msprime = MapFiles.msprime()
        with MapFiles._open(path) as fh:
            lines = [line for line in fh if line.strip()]
        header, rows = lines[:1], lines[1:]
        per_contig: dict[str, list[str]] = {}
        for line in rows:
            per_contig.setdefault(line.split()[0], []).append(line)
        if not per_contig:
            raise ValueError(f"no data rows in the recombination map {path}")
        return {contig: msprime.RateMap.read_hapmap(
                    io.StringIO("".join(header + contig_rows)))
                for contig, contig_rows in per_contig.items()}

    @staticmethod
    def on_contig(value, contig: str, name: str):
        """The entry of a per-contig map for ``contig``, or ``value`` itself.

        A contig name matches with or without a ``chr`` prefix, so ``chr1``
        and ``1`` name the same contig.

        :param value: A map, a ``{contig: map}`` dict, or ``None``.
        :param contig: The contig of the sites.
        :param name: The parameter the map was passed as, named in the error.
        :return: The map for ``contig``.
        :raises ValueError: If a per-contig dict has no entry for ``contig``.
        """
        if not isinstance(value, dict):
            return value
        bare = contig[3:] if contig.startswith("chr") else contig
        for key in (contig, bare, f"chr{bare}"):
            if key in value:
                return value[key]
        raise ValueError(
            f"{name} has no entry for contig {contig!r}. It covers "
            f"{sorted(value)[:5]}")
