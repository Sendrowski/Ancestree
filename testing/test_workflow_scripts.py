"""Tests for the scripts under ``workflow/scripts``.

``_singer_seeds`` is the single answer to which SINGER chains a cell uses,
imported by both the Snakefile and the scoring merge, so the two cannot
diverge on a seed recorded as aborting.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "workflow" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import _singer_seeds as seeds  # noqa: E402


class TestWorkflowScriptConstructorArity:
    """Workflow scripts call BaseComposition constructors with their
    required arguments.

    ``n_total`` is a required positional of
    ``BaseComposition.from_n_target_sites``, and nothing imports the scripts,
    so a bare call is checked statically.
    """

    def test_from_n_target_sites_is_never_called_bare(self):
        import ast

        scripts = sorted(SCRIPTS.glob("*.py"))
        assert scripts, f"no workflow scripts found under {SCRIPTS}"
        bare = []
        for path in scripts:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if (isinstance(fn, ast.Attribute)
                        and fn.attr == "from_n_target_sites"
                        and not node.args and not node.keywords):
                    bare.append(f"{path.name}:{node.lineno}")
        assert not bare, f"bare from_n_target_sites() calls: {bare}"


def test_the_aborting_file_parses():
    """A malformed line must fail here, not during DAG construction."""
    entries = seeds.aborting_seeds()
    assert entries, "no aborting seeds recorded; the file or its path is wrong"
    for scenario, method, n_out, chunk, seed in entries:
        assert isinstance(scenario, str) and scenario
        assert isinstance(method, str) and method
        assert all(isinstance(v, int) for v in (n_out, chunk, seed))


def test_a_recorded_seed_is_never_chosen():
    """A chain recorded as aborting cannot yield an ARG, so it is never selected."""
    aborting = seeds.aborting_seeds()
    assert aborting
    for scenario, method, n_out, chunk, _ in sorted(aborting)[:20]:
        chosen = seeds.runnable_seeds(scenario, method, n_out, chunk)
        assert len(chosen) == seeds.N_CHAINS
        assert chosen == sorted(chosen), "seeds must be ascending"
        for s in chosen:
            assert (scenario, method, n_out, chunk, s) not in aborting, (
                f"chose recorded-aborting seed {s} for {scenario} {method} "
                f"n_out={n_out} chunk={chunk}")


def test_choosing_is_a_pure_function_of_the_record():
    """Two calls must agree, or the DAG and the merge can still diverge."""
    args = ("cpg_hypermut", "singer_arg_panel_ft", 10, 2)
    assert seeds.runnable_seeds(*args) == seeds.runnable_seeds(*args)


def test_it_refuses_rather_than_returning_short():
    """A chunk with too few runnable seeds must raise, not score short.

    A shard averaged over fewer chains carries a noisier posterior than the
    cells it is compared against, so silently returning what is available
    would put an incomparable number in the heatmap.
    """
    with pytest.raises(ValueError, match="runnable"):
        seeds.runnable_seeds("cpg_hypermut", "singer_arg_panel_ft", 10, 2,
                             n_seeds=seeds.MAX_SEED + 1)
