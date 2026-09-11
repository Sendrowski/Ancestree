"""Unit tests for the pure ``ancestree.cli`` helpers (model / prior factories,
logging setup, the BED / bedGraph parsers, the empirical composition tally,
the ``--debug`` probe and the entry points around :func:`ancestree.cli.main`).
These exercise the argument-plumbing branches without running any inference,
so they stay fast.
"""
import logging
import runpy
import sys

import msprime
import numpy as np
import pytest

from ancestree import STATES, TRANSITION_PAIRS
from ancestree.cli import (
    _build_model,
    _build_ingroup_weight,
    _build_prior,
    _configure_logging,
    _debug_requested,
    _empirical_composition,
    _import_msprime,
    _read_bed_intervals,
    _read_ratemap_bedgraph,
    _warn_model_defaults,
    main,
)
from ancestree.models import JC69, K2, F81, HKY, GTR, _KAPPA_CLAMP
from ancestree.priors import (
    AdaptiveIngroupWeight,
    KingmanIngroupWeight,
    StationaryPrior,
)

from testing._helpers import ladder_panel


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


def test_none_ingroup_weight_needs_no_ingroup():
    """``--ingroup-weight none`` builds without an ingroup, in any case."""
    from ancestree.priors import NoIngroupWeight

    assert isinstance(_build_ingroup_weight("none", ingroup_samples=None),
                      NoIngroupWeight)
    assert isinstance(_build_ingroup_weight("NONE", ingroup_samples=None),
                      NoIngroupWeight)


@pytest.mark.parametrize("model, composition, kappa", [
    ("JC69", False, False),
    ("F81", True, False),
    ("K2", False, True),
    ("HKY", True, True),
    ("gtr", True, False),
])
def test_warn_model_defaults_names_each_free_parameter(
        caplog, model, composition, kappa):
    """Each unsettable model parameter is reported once, under the mode."""
    with caplog.at_level(logging.WARNING, logger="ancestree.cli"):
        _warn_model_defaults("arg", model)
    messages = [r.getMessage() for r in caplog.records]
    assert sum("uniform base frequencies" in m for m in messages) == int(composition)
    assert sum("default kappa" in m for m in messages) == int(kappa)
    assert all(m.startswith("arg: model") for m in messages)


@pytest.mark.filterwarnings("ignore::Warning:zarr")
class TestEmpiricalComposition:
    """``_empirical_composition`` tallies the input's own variants."""

    @staticmethod
    def _expected(alleles, gt, n_sites):
        """Tip-allele tally and Ts/Tv counts over the first ``n_sites`` sites."""
        pi_counts = {b: 0 for b in STATES}
        n_ts = n_tv = 0
        for i in range(n_sites):
            present = set()
            for g in gt[i]:
                pi_counts[alleles[i][g]] += 1
                present.add(alleles[i][g])
            if len(present) == 2:
                if tuple(present) in TRANSITION_PAIRS:
                    n_ts += 1
                else:
                    n_tv += 1
        arr = np.array([pi_counts[b] for b in STATES], dtype=float)
        pi = np.maximum(arr / arr.sum(), 1e-6)
        pi /= pi.sum()
        kappa = float(np.clip(2.0 * n_ts / n_tv, *_KAPPA_CLAMP))
        return pi, kappa

    @pytest.mark.parametrize("which", ["vcf", "vcz"])
    def test_pi_and_kappa_match_a_direct_tally(self, ladder_panel, which):
        vcf, vcz, _nwk, alleles, gt = ladder_panel
        path = vcf if which == "vcf" else vcz
        bc = _empirical_composition(str(path), None)
        pi, kappa = self._expected(alleles, gt, len(alleles))
        np.testing.assert_allclose(bc.pi, pi, rtol=1e-9)
        assert bc.kappa_estimate == pytest.approx(kappa)

    def test_max_sites_caps_the_tally(self, ladder_panel):
        vcf, _vcz, _nwk, alleles, gt = ladder_panel
        bc = _empirical_composition(str(vcf), 40)
        pi, kappa = self._expected(alleles, gt, 40)
        np.testing.assert_allclose(bc.pi, pi, rtol=1e-9)
        assert bc.kappa_estimate == pytest.approx(kappa)
        assert bc.n_ts + bc.n_tv <= 40


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


def test_import_msprime_names_the_extra(monkeypatch):
    """A missing ``msprime`` exits with the install hint."""
    monkeypatch.setitem(sys.modules, "msprime", None)
    with pytest.raises(SystemExit, match=r"pip install ancestree\[maps\]"):
        _import_msprime()


def test_import_msprime_returns_the_module():
    assert _import_msprime() is msprime


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


class TestDebugRequested:
    """``--debug`` is read back through the parser, abbreviation included."""

    def test_a_parsable_command_line_reads_the_flag(self):
        argv = ["arg", "--trees", "x.trees", "--out", "y.vcf"]
        assert _debug_requested(argv) is False
        assert _debug_requested(["--debug", *argv]) is True

    def test_an_unparsable_command_line_falls_back_to_the_prefix(self):
        assert _debug_requested(["--deb"]) is True
        assert _debug_requested([]) is False
        assert _debug_requested(["fixed-tree", "--debug"]) is True

    def test_sys_argv_is_read_when_no_vector_is_given(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["ancestree", "--debug", "arg",
                                          "--trees", "x.trees", "--out", "y.vcf"])
        assert _debug_requested(None) is True


def test_main_reraises_operational_errors_under_debug(tmp_path, capsys):
    """With ``--debug`` the exception propagates uncaught."""
    argv = ["local-tree", "--vcf", str(tmp_path / "no_such.vcf"),
            "--sequence-length", "1000", "--out", str(tmp_path / "out.vcf")]
    with pytest.raises(OSError, match="no_such.vcf"):
        main(["--debug", *argv])
    assert "ancestree: error:" not in capsys.readouterr().err


@pytest.mark.filterwarnings("ignore::RuntimeWarning:runpy")
def test_module_entry_point_runs_main(monkeypatch, capsys):
    """``python -m ancestree.cli`` reaches :func:`main`.

    The module is executed a second time in this process, which ``runpy``
    reports as a re-execution of an imported module.
    """
    from ancestree import __version__

    monkeypatch.setattr(sys, "argv", ["ancestree", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ancestree.cli", run_name="__main__", alter_sys=True)
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out
