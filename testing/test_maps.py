"""Unit tests for :class:`ancestree._maps.MapFiles`, the readers behind the
``--accessibility``, ``--mutation-map`` and ``--recombination-map`` options.
Both BED-like formats carry a contig column, so each is read into one entry
per contig and :meth:`ancestree._maps.MapFiles.on_contig` picks the one the
sites live on.
"""
import sys

import msprime
import numpy as np
import pytest

from ancestree import DEFAULT_MU
from ancestree._maps import MapFiles


class TestReadBed:
    def test_parses_and_skips_noise(self, tmp_path):
        bed = tmp_path / "mask.bed"
        bed.write_text(
            "# a comment\n"
            "track name=foo\n"
            "browser dense\n"
            "chr1\t100\t200\n"
            "chr1\t300\t450\n"
            "too short\n"  # < 3 fields -> skipped
            "\n"
        )
        assert MapFiles.read_bed(str(bed)) == {
            "chr1": [(100.0, 200.0), (300.0, 450.0)]}

    def test_keeps_each_contig_apart(self, tmp_path):
        bed = tmp_path / "mask.bed"
        bed.write_text("chr1\t0\t10\nchr2\t5\t25\nchr1\t40\t50\n")
        assert MapFiles.read_bed(str(bed)) == {
            "chr1": [(0.0, 10.0), (40.0, 50.0)],
            "chr2": [(5.0, 25.0)],
        }


class TestReadBedgraph:
    def test_fills_gaps_and_sorts(self, tmp_path):
        # Out-of-order rows with a gap before the first interval. The gap
        # (and the [0, first_start) prefix) must be filled with default_rate.
        bg = tmp_path / "rates.bedgraph"
        bg.write_text(
            "# header\n"
            "chr1\t300\t400\t2e-8\n"
            "chr1\t100\t200\t1e-8\n"
        )
        rm = MapFiles.read_bedgraph(str(bg), default_rate=5e-9)["chr1"]
        assert rm.position[0] == 0.0
        assert rm.position[-1] == 400.0
        # interval rates appear in sorted order, gaps carry the default
        assert 1e-8 in rm.rate and 2e-8 in rm.rate and 5e-9 in rm.rate

    def test_one_map_per_contig(self, tmp_path):
        bg = tmp_path / "rates.bedgraph"
        bg.write_text("chr1\t0\t100\t1e-8\nchr2\t0\t50\t3e-8\n")
        maps = MapFiles.read_bedgraph(str(bg), default_rate=5e-9)
        assert set(maps) == {"chr1", "chr2"}
        assert maps["chr1"].position[-1] == 100.0
        assert maps["chr2"].position[-1] == 50.0

    def test_no_usable_intervals_raises(self, tmp_path):
        bg = tmp_path / "empty.bedgraph"
        bg.write_text("# only a comment\nchr1\t10\n")  # second row too short
        with pytest.raises(ValueError, match="no usable intervals"):
            MapFiles.read_bedgraph(str(bg), default_rate=1e-8)

    def test_overlap_and_zero_length(self, tmp_path):
        # An overlapping interval (start < current frontier → clipped) and a
        # zero-length interval (end <= start after clip → skipped).
        bg = tmp_path / "r.bedgraph"
        bg.write_text(
            "chr1\t0\t200\t1e-8\n"
            "chr1\t100\t300\t2e-8\n"  # overlaps [0,200): clipped to start at 200
            "chr1\t300\t300\t9e-8\n"  # zero-length: skipped
        )
        rm = MapFiles.read_bedgraph(str(bg), default_rate=5e-9)["chr1"]
        assert rm.position[0] == 0.0
        assert rm.position[-1] == 300.0


class TestMutationMapGapRate:
    """``--mutation-map`` gaps are filled with a real rate, not NaN.

    ``--mu`` defaults to ``None`` at the CLI (the API supplies the fallback and
    warns), so the gap-filling rate has to resolve to ``DEFAULT_MU`` here:
    msprime turns a ``None`` rate into ``NaN``, which would silently poison
    every site in the gap.
    """

    def _bedgraph(self, tmp_path):
        # A deliberate gap over [0, 100): the map starts at 100.
        p = tmp_path / "mu.bedGraph"
        p.write_text("1\t100\t200\t2e-8\n")
        return str(p)

    def test_gap_uses_default_mu_when_mu_omitted(self, tmp_path):
        rm = MapFiles.read_bedgraph(
            self._bedgraph(tmp_path), default_rate=DEFAULT_MU)["1"]
        assert not np.isnan(rm.rate).any()
        assert rm.rate[0] == pytest.approx(DEFAULT_MU)

    def test_none_default_rate_would_be_nan(self, tmp_path):
        """A ``None`` default rate must be resolved before it reaches the map."""
        rm = MapFiles.read_bedgraph(
            self._bedgraph(tmp_path), default_rate=None)["1"]
        assert np.isnan(rm.rate[0])


class TestOnContig:
    def test_a_plain_map_passes_through(self):
        rate_map = msprime.RateMap(position=[0.0, 10.0], rate=[1e-8])
        assert MapFiles.on_contig(rate_map, "chr1", "mutation_map") is rate_map

    def test_none_passes_through(self):
        assert MapFiles.on_contig(None, "chr1", "accessibility") is None

    def test_a_dict_is_indexed_by_contig(self):
        assert MapFiles.on_contig({"chr1": [(0.0, 5.0)]}, "chr1",
                                  "accessibility") == [(0.0, 5.0)]

    def test_a_missing_contig_names_what_is_covered(self):
        with pytest.raises(ValueError, match=r"accessibility has no entry for "
                                             r"contig 'chr9'.*'chr1'"):
            MapFiles.on_contig({"chr1": []}, "chr9", "accessibility")


def test_msprime_import_names_the_extra(monkeypatch):
    """A missing ``msprime`` raises with the install hint."""
    monkeypatch.setitem(sys.modules, "msprime", None)
    with pytest.raises(ImportError,
                       match=r"ancestree-popgen\[maps\]"):
        MapFiles.msprime()


def test_msprime_import_returns_the_module():
    assert MapFiles.msprime() is msprime
