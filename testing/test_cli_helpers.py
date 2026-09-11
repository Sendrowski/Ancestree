"""Unit tests for the pure ``ancestree.cli`` helpers (model / prior factories,
logging setup, and the BED / bedGraph parsers). These exercise the argument-
plumbing branches without running any inference, so they stay fast.
"""
import logging

import pytest

from ancestree.cli import (
    _build_model,
    _build_ingroup_weight,
    _build_prior,
    _configure_logging,
    _read_bed_intervals,
    _read_ratemap_bedgraph,
)
from ancestree.models import JC69, K2, F81, HKY, GTR
from ancestree.priors import (
    AdaptiveIngroupWeight,
    KingmanIngroupWeight,
    StationaryPrior,
)


class TestBuildModel:
    @pytest.mark.parametrize(
        "name,cls",
        [("JC69", JC69), ("K2", K2), ("F81", F81), ("HKY", HKY), ("GTR", GTR)],
    )
    def test_each_model_name(self, name, cls):
        m = _build_model(name, fit_kappa=False, fit_rates=False)
        assert isinstance(m, cls)

    def test_name_is_case_insensitive(self):
        assert isinstance(_build_model("hky", fit_kappa=False, fit_rates=False), HKY)

    def test_fit_flags_forwarded(self):
        # K2/HKY accept fit_kappa; GTR accepts fit_rates. Building with the
        # flags set must not raise and must give a fittable parameter.
        k2 = _build_model("K2", fit_kappa=True, fit_rates=False)
        gtr = _build_model("GTR", fit_kappa=False, fit_rates=True)
        assert k2.free_params
        assert gtr.free_params

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown --model"):
            _build_model("NOTAMODEL", fit_kappa=False, fit_rates=False)


class TestBuildPrior:
    def test_stationary(self):
        p = _build_prior("uniform", model=JC69())
        assert isinstance(p, StationaryPrior)

    def test_kingman_with_ingroup(self):
        p = _build_ingroup_weight("kingman", ingroup_samples=["a", "b"])
        assert isinstance(p, KingmanIngroupWeight)

    def test_adaptive_with_ingroup(self):
        p = _build_ingroup_weight("adaptive", ingroup_samples=["a", "b"])
        assert isinstance(p, AdaptiveIngroupWeight)

    def test_adaptive_parallelize_reaches_prior(self):
        """``--parallelize`` reaches the adaptive prior's per-bin fit, and the
        default matches :class:`AdaptiveIngroupWeight`'s own ``False``."""
        default = _build_ingroup_weight(
            "adaptive", ingroup_samples=["a", "b"],
        )
        assert default.parallelize is False
        parallel = _build_ingroup_weight(
            "adaptive", ingroup_samples=["a", "b"], parallelize=True,
        )
        assert parallel.parallelize is True

    @pytest.mark.parametrize("name", ["kingman", "adaptive"])
    def test_requires_ingroup(self, name):
        with pytest.raises(ValueError, match="requires --ingroup"):
            _build_ingroup_weight(name, ingroup_samples=None)

    def test_unknown_prior_raises(self):
        with pytest.raises(ValueError, match="Unknown --prior"):
            _build_prior("nope", model=JC69())

    def test_unknown_ingroup_weight_raises(self):
        with pytest.raises(ValueError, match="Unknown --ingroup-weight"):
            _build_ingroup_weight("nope", ingroup_samples=["a", "b"])


class TestConfigureLogging:
    def test_quiet_sets_warning(self):
        _configure_logging(verbosity=0, quiet=True)
        assert logging.getLogger("ancestree").level == logging.WARNING

    def test_verbose_sets_debug(self):
        _configure_logging(verbosity=1, quiet=False)
        assert logging.getLogger("ancestree").level == logging.DEBUG

    def test_default_sets_info(self):
        _configure_logging(verbosity=0, quiet=False)
        assert logging.getLogger("ancestree").level == logging.INFO


class TestReadBedIntervals:
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
        assert _read_bed_intervals(str(bed)) == [(100.0, 200.0), (300.0, 450.0)]


class TestReadRatemapBedgraph:
    def test_fills_gaps_and_sorts(self, tmp_path):
        # Out-of-order rows with a gap before the first interval. The gap
        # (and the [0, first_start) prefix) must be filled with default_rate.
        bg = tmp_path / "rates.bedgraph"
        bg.write_text(
            "# header\n"
            "chr1\t300\t400\t2e-8\n"
            "chr1\t100\t200\t1e-8\n"
        )
        rm = _read_ratemap_bedgraph(str(bg), default_rate=5e-9)
        assert rm.position[0] == 0.0
        assert rm.position[-1] == 400.0
        # interval rates appear in sorted order, gaps carry the default
        assert 1e-8 in rm.rate and 2e-8 in rm.rate and 5e-9 in rm.rate

    def test_no_usable_intervals_raises(self, tmp_path):
        bg = tmp_path / "empty.bedgraph"
        bg.write_text("# only a comment\nchr1\t10\n")  # second row too short
        with pytest.raises(SystemExit, match="no usable intervals"):
            _read_ratemap_bedgraph(str(bg), default_rate=1e-8)

    def test_overlap_and_zero_length(self, tmp_path):
        # An overlapping interval (start < current frontier → clipped) and a
        # zero-length interval (end <= start after clip → skipped).
        bg = tmp_path / "r.bedgraph"
        bg.write_text(
            "chr1\t0\t200\t1e-8\n"
            "chr1\t100\t300\t2e-8\n"  # overlaps [0,200): clipped to start at 200
            "chr1\t300\t300\t9e-8\n"  # zero-length: skipped
        )
        rm = _read_ratemap_bedgraph(str(bg), default_rate=5e-9)
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
        import numpy as np

        from ancestree import DEFAULT_MU
        from ancestree.cli import _read_ratemap_bedgraph

        rm = _read_ratemap_bedgraph(self._bedgraph(tmp_path), default_rate=DEFAULT_MU)
        assert not np.isnan(rm.rate).any()
        assert rm.rate[0] == pytest.approx(DEFAULT_MU)

    def test_none_default_rate_would_be_nan(self, tmp_path):
        """A ``None`` default rate must be resolved before it reaches the map."""
        import numpy as np

        from ancestree.cli import _read_ratemap_bedgraph

        rm = _read_ratemap_bedgraph(self._bedgraph(tmp_path), default_rate=None)
        assert np.isnan(rm.rate[0])


class TestLocalTreeOmitsFitToggles:
    """``local-tree`` must not expose ``--fit-kappa`` / ``--fit-rates``.

    :class:`~ancestree.local_tree_inference.LocalTreeInference` has no ``fit``
    method, so the toggles would mark model parameters free that nothing ever
    optimises. Exposing them also left the handler reading ``args.fit_kappa``,
    which raised ``AttributeError`` once the flags were dropped from only one
    of the two places.
    """

    def _sub(self, name):
        import argparse

        from ancestree.cli import build_parser

        sa = next(a for a in build_parser()._actions
                  if isinstance(a, argparse._SubParsersAction))
        return sa.choices[name]

    def test_flags_absent_from_local_tree(self):
        dests = {a.dest for a in self._sub("local-tree")._actions}
        assert "fit_kappa" not in dests
        assert "fit_rates" not in dests

    def test_flags_present_on_fixed_tree(self):
        dests = {a.dest for a in self._sub("fixed-tree")._actions}
        assert {"fit_kappa", "fit_rates"} <= dests

    def test_handler_does_not_read_the_missing_flags(self):
        """The handler must not reference attributes the parser does not set."""
        import inspect

        from ancestree import cli

        src = inspect.getsource(cli._run_local_tree)
        assert "args.fit_kappa" not in src
        assert "args.fit_rates" not in src


class TestLoggerAnnotationsResolve:
    """``_log`` return annotations must resolve under ``get_type_hints``.

    Declaring ``-> "logging.Logger"`` while importing ``logging`` inside the
    property body raises ``NameError`` for anything that introspects the
    annotation, including sphinx-autodoc-typehints.
    """

    @pytest.mark.parametrize("dotted", [
        "ancestree.sites.SiteSource",
        "ancestree.writers.Writer",
        "ancestree.priors.IngroupWeight",
    ])
    def test_log_annotation_resolves(self, dotted):
        import importlib
        import logging as _logging
        from typing import get_type_hints

        mod, _, name = dotted.rpartition(".")
        cls = getattr(importlib.import_module(mod), name)
        hints = get_type_hints(cls._log.fget)
        assert hints["return"] is _logging.Logger


class TestSpeciesTreeOrdersTheLadder:
    """The ladder order must come from the Newick topology, not from the order
    ``--outgroups`` happens to be typed in.

    Passing the outgroups in any order other than closest-first would otherwise
    fit the wrong ladder, silently, while the Newick that specifies the correct
    one is parsed and discarded.
    """

    def test_topology_wins_over_argument_order(self):
        from ancestree.trees import OutgroupLadderTree

        ingroup = ["i0", "i1"]
        # o3 is the closest outgroup, o1 the most distant.
        newick = "((((i0,i1):0.05,o3:0.1):0.1,o2:0.2):0.1,o1:0.4);"
        from_newick = OutgroupLadderTree.from_newick(
            newick, ingroup_samples=ingroup, outgroup_samples=["o1", "o2", "o3"],
        )
        assert from_newick.outgroup_samples == ("o3", "o2", "o1")
        # Building from the name list alone keeps the supplied order, which is
        # why the handler must pass the parsed tree through.
        assert OutgroupLadderTree(
            ingroup, ["o1", "o2", "o3"]
        ).outgroup_samples == ("o1", "o2", "o3")

    def test_handler_passes_the_parsed_tree(self):
        import inspect

        from ancestree import cli

        src = inspect.getsource(cli._run_fixed_tree)
        # A source-text check: it proves the argument is named, not that the
        # parsed tree is the one constructed. The behavioural half is
        # test_topology_wins_over_argument_order above, which pins the ladder
        # order the Newick produces.
        assert "tree=tree" in src, "the parsed Newick must reach FixedTreeInference"


class TestSpeciesTreeIsUsedVerbatim:
    """``--species-tree`` supplies the tree, so the ML fit is skipped.

    Without it the ladder comes from ``--outgroups`` and its rates are fitted.
    The two are distinguishable in the handler by ``fit_required``.
    """

    def test_handler_gates_the_fit_on_the_flag(self):
        import inspect

        from ancestree import cli

        src = inspect.getsource(cli._run_fixed_tree)
        assert "fit_required=args.species_tree is None" in src
        assert "if args.species_tree is None:" in src

    def test_species_tree_is_optional(self):
        import argparse

        from ancestree.cli import build_parser

        sa = next(a for a in build_parser()._actions
                  if isinstance(a, argparse._SubParsersAction))
        action = next(a for a in sa.choices["fixed-tree"]._actions
                      if a.dest == "species_tree")
        assert not action.required
        assert action.default is None


def test_colored_formatter_preserves_message_text():
    # The name field carries the full dotted logger name, and the logger name
    # inside the message text is left verbatim.
    import logging

    import ancestree

    fmt = ancestree._ColoredFormatter("%(levelname)s:%(name)s: %(message)s")
    rec = logging.LogRecord(
        "ancestree.inference", logging.INFO, __file__, 1,
        "ancestree.inference failed to load", None, None,
    )
    out = fmt.format(rec)
    assert "ancestree.inference failed to load" in out  # message verbatim
    assert "INFO:ancestree.inference:" in out  # full name in the name field


def test_cli_clean_error_on_missing_file(tmp_path, capsys):
    # The console entry point reports an operational error as a clean message,
    # not a raw traceback.
    from ancestree.cli import main

    with pytest.raises(SystemExit) as exc:
        main([
            "local-tree", "--vcf", str(tmp_path / "no_such.vcf"),
            "--sequence-length", "1000", "--out", str(tmp_path / "out.vcf"),
        ])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "ancestree: error:" in err
    assert "Traceback" not in err
