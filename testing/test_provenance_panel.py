"""Provenance must record what determines the answer, not only the settings.

``focal`` names the node a run reports at. The panel is what places that node,
so an annotated output that records one without the other cannot be traced
back. Rate maps and the accessibility mask decide which blocks carry
information, and after masking a zero-rate interval they change the TMRCAs, so
their source is recorded where a path was supplied.
"""
import tskit

import ancestree as anc

TREES = "docs/_static/quickstart.trees"
ING = [f"i{i}" for i in range(6)]
OUT = ["o0", "o1"]


def _params(**kw):
    ts = tskit.load(TREES)
    inf = anc.Inference.from_local_tree(
        list(anc.TskitSource(ts)), model=anc.JC69(), mu=5e-8, rec_rate=1e-8,
        sample_names=ING + OUT, sequence_length=float(ts.sequence_length),
        progress=False, **kw)
    return inf.provenance()["parameters"]


def test_the_panel_that_places_the_focal_node_is_recorded():
    p = _params(ingroup_samples=ING, outgroup_samples=OUT)
    assert p["ingroup_samples"] == ING
    assert p["outgroup_samples"] == OUT
    assert "focal" in p, "focal is recorded without the panel that places it"


def test_the_ensemble_and_hmm_settings_are_recorded():
    p = _params(ingroup_samples=ING, outgroup_samples=OUT)
    for key in ("n_ensemble", "ensemble_seed", "member_chunk",
                "block_size", "n_time_bins", "rec_rate", "window_bp"):
        assert key in p, f"{key} missing from provenance"


def test_an_accessibility_mask_is_recorded():
    p = _params(ingroup_samples=ING, outgroup_samples=OUT,
                accessibility=[(0.0, 10_000.0), (20_000.0, 30_000.0)])
    assert p["accessibility_intervals"] == 2
