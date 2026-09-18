"""Every CLI flag states the default of the constructor parameter it feeds.

A flag that spells out its own default is a second source of truth, and the
command line and the Python API then drift apart.
"""
import inspect

import pytest

from ancestree import LocalTreeInference
from ancestree.cli import _lib_default, build_parser

#: ``argparse`` destination -> constructor parameter it configures.
LOCAL_TREE_FLAGS = {
    "chunk_size": "chunk_size",
    "halo": "halo",
    "ensemble_size": "n_ensemble",
    "member_chunk": "member_chunk",
    "window": "window",
    "block_size": "block_size",
    "n_time_bins": "n_time_bins",
    "n_workers": "n_workers",
    "phase_seed": "phase_seed",
    "ploidy": "ploidy",
    "phased": "phased",
    "ensemble_seed": "ensemble_seed",
}


def _parsed_local_tree():
    """Parse a minimal ``local-tree`` command line, leaving every flag unset."""
    return build_parser().parse_args(
        ["local-tree", "--vcf", "in.vcf", "--out", "out.vcf", "--mu", "1e-8"])


def test_lib_default_reads_the_constructor_and_rejects_unknown_names():
    """A flag default is the constructor's own, and an unknown name is an error."""
    from ancestree.inference import FixedTreeInference

    expected = inspect.signature(FixedTreeInference).parameters["n_starts"].default
    assert _lib_default(FixedTreeInference, "n_starts") == expected
    with pytest.raises(KeyError, match="takes no parameter 'no_such_flag'"):
        _lib_default(FixedTreeInference, "no_such_flag")


@pytest.mark.parametrize("dest,param", sorted(LOCAL_TREE_FLAGS.items()))
def test_flag_default_is_the_constructor_default(dest, param):
    """A flag left unset carries the library's own default."""
    args = _parsed_local_tree()
    expected = inspect.signature(LocalTreeInference).parameters[param].default
    assert getattr(args, dest) == expected, (
        f"--{dest.replace('_', '-')} defaults to {getattr(args, dest)!r} but "
        f"LocalTreeInference({param}=) defaults to {expected!r}. Source the "
        f"flag from _lib_default() rather than restating the value."
    )


def test_every_mapped_parameter_exists():
    """The map names real constructor parameters, so a rename cannot rot it."""
    parameters = inspect.signature(LocalTreeInference).parameters
    unknown = sorted(set(LOCAL_TREE_FLAGS.values()) - set(parameters))
    assert not unknown, f"no such LocalTreeInference parameter(s): {unknown}"


def test_every_mapped_flag_exists():
    """The map names real flags, so a removed flag fails here."""
    args = _parsed_local_tree()
    missing = sorted(d for d in LOCAL_TREE_FLAGS if not hasattr(args, d))
    assert not missing, f"no such local-tree flag(s): {missing}"


def test_chunking_is_on_by_default():
    """The command line bounds memory out of the box, as the API does."""
    assert _parsed_local_tree().chunk_size is not None


@pytest.mark.parametrize("spec", ["none", "off", "NONE"])
def test_chunking_can_be_turned_off(spec):
    """``None`` is reachable from the command line, as it is from the API."""
    args = build_parser().parse_args(
        ["local-tree", "--vcf", "in.vcf", "--out", "out.vcf", "--mu", "1e-8",
         "--chunk-size", spec])
    assert args.chunk_size is None


@pytest.mark.parametrize("spec", ["0", "-1", "banana"])
def test_a_chunk_size_that_is_not_a_width_is_rejected(spec):
    """A bad width fails at parse time rather than deep in the builder."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["local-tree", "--vcf", "in.vcf", "--out", "out.vcf",
             "--mu", "1e-8", "--chunk-size", spec])
