"""Readers for the genome map files local-tree mode accepts.

A BED accessibility mask, a bedGraph mutation map and a HapMap recombination
map carry a contig column, so each is read into one entry per contig.
"""
from __future__ import annotations

import io
import os


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
    def _rows(path: "str | os.PathLike", n_fields: int):
        """The data rows of a BED-like file.

        :param path: The file.
        :param n_fields: Fields a row needs, shorter rows being skipped.
        :return: Generator over the split rows, skipping ``track``, ``browser``,
            comment and blank lines.
        """
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.split()
                if len(fields) >= n_fields:
                    yield fields

    @staticmethod
    def read_bed(path: "str | os.PathLike") -> dict[str, list[tuple[float, float]]]:
        """Half-open ``(start, end)`` intervals of a BED file, per contig.

        :param path: The BED file, 0-based half-open.
        :return: ``{contig: intervals}``, in file order.
        """
        intervals: dict[str, list[tuple[float, float]]] = {}
        for f in MapFiles._rows(path, 3):
            intervals.setdefault(f[0], []).append((float(f[1]), float(f[2])))
        return intervals

    @staticmethod
    def read_bedgraph(path: "str | os.PathLike",
                      default_rate: float) -> dict:
        """One :class:`msprime.RateMap` per contig of a 4-column bedGraph
        (contig, start, end, rate).

        Each contig's rows are sorted, an overlap is clipped to the preceding
        interval, and any gap, including one before the first interval, takes
        ``default_rate``.

        :param path: The bedGraph file.
        :param default_rate: Rate of the gaps.
        :return: ``{contig: RateMap}`` covering ``[0, last end]``.
        :raises ValueError: If the file yields no usable interval.
        """
        msprime = MapFiles.msprime()
        rows: dict[str, list[tuple[float, float, float]]] = {}
        for f in MapFiles._rows(path, 4):
            rows.setdefault(f[0], []).append(
                (float(f[1]), float(f[2]), float(f[3])))
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
        with open(path) as fh:
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
