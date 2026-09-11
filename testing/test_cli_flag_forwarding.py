"""Flags that reach the inference, not merely the parser.

``test_cli_library_defaults`` pins that a flag's default equals the
constructor's, which holds even if the flag is never forwarded. These pin the
forwarding itself: every flag below can be unwired from its ``run`` dispatcher
without any other test in the suite noticing.

The dispatchers import their classes inside the function, so the patches target
the defining module rather than ``ancestree.cli``.
"""
import functools

import pytest

from ancestree.cli import build_parser, run


def _parse(argv):
    return build_parser().parse_args(argv)


@pytest.mark.parametrize("flag,value,dest,expected", [
    ("--focal", "panel-root", "focal", "panel-root"),
    ("--focal-fraction", "0.25", "focal_fraction", 0.25),
    ("--focal-coalescences", "2", "focal_coalescences", 2),
    ("--focal-depth", "1500", "focal_depth", 1500.0),
])
def test_the_parser_carries_the_focal_flags(flag, value, dest, expected):
    """Each focal flag parses to the destination the dispatcher reads."""
    args = _parse(["arg", "--trees", "x.trees", "--out", "o.vcf", flag, value])
    assert getattr(args, dest) == expected


@pytest.mark.parametrize("extra,anchor,attr,value", [
    (["--focal", "panel-root"], "panel_root", None, None),
    (["--focal-coalescences", "2"], "ingroup_mrca", "coalescences", 2),
    (["--focal-fraction", "0.25"], "ingroup_mrca", "fraction", 0.25),
    (["--focal-depth", "1500"], "ingroup_mrca", "depth", 1500.0),
])
def test_the_arg_dispatcher_forwards_the_focal_spec(monkeypatch, extra, anchor,
                                                    attr, value):
    """``--focal`` and its placement reach :class:`ARGBasedInference`."""
    import ancestree.inference as inference_module

    seen = {}

    # functools.wraps keeps the constructor signature visible to
    # inspect.signature, which build_parser reads for the library defaults.
    @functools.wraps(inference_module.ARGBasedInference.__init__)
    def stub_init(self, *args, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(
        inference_module.ARGBasedInference, "__init__", stub_init)
    with pytest.raises(SystemExit):
        run(["arg", "--trees", "x.trees", "--out", "o.vcf", *extra])

    focal = seen.get("focal")
    assert focal is not None, "the focal spec never reached the inference"
    assert focal.anchor == anchor, focal
    if attr is not None:
        assert getattr(focal, attr) == value, focal


def test_the_local_tree_dispatcher_forwards_the_sample_filter(monkeypatch):
    """``--samples`` reaches the source builder rather than being dropped."""
    from ancestree.sites import SiteSource

    seen = {}

    def stub(path, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(SiteSource, "resolve", staticmethod(stub))
    with pytest.raises(SystemExit):
        run(["local-tree", "--vcf", "in.vcf", "--out", "o.vcf",
             "--mu", "1e-8", "--samples", "h0,h1"])
    assert seen.get("sample_filter") == ["h0", "h1"], seen
