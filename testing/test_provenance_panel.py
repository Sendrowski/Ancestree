"""Provenance must record what determines the answer, not only the settings.

``focal`` names the node a run reports at. The panel is what places that node,
so an annotated output that records one without the other cannot be traced
back. Rate maps and the accessibility mask decide which blocks carry
information, and after masking a zero-rate interval they change the TMRCAs, so
their source is recorded where a path was supplied.
"""
import numpy as np
import tskit

import ancestree as anc
from ancestree import ARGBasedInference, BaseComposition, GTR, JC69
from testing._helpers import QUICKSTART_TREES

TREES = QUICKSTART_TREES
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


def test_model_provenance_serialises_every_parameter_kind(small_ts):
    """Array-valued parameters become lists, other objects pass through, and
    a composition whose ``pi`` cannot be read leaves the entry out."""
    composition = BaseComposition(counts={"A": 30, "C": 20, "G": 20, "T": 30})
    inf = ARGBasedInference(small_ts, GTR(), mu=1e-8, progress=False,
                            base_composition=composition)
    entry = inf._model_provenance()
    assert entry["model_rates"] == [float(v) for v in GTR().rates]
    np.testing.assert_allclose(entry["base_composition_pi"], composition.pi)

    class _Tagged(JC69):
        @property
        def _repr_params(self):
            return {"tag": "custom"}

    class _Unreadable:
        @property
        def pi(self):
            raise RuntimeError("no pi")

    inf.model = _Tagged()
    inf.base_composition = _Unreadable()
    entry = inf._model_provenance()
    assert entry == {"model_tag": "custom"}


def test_empty_provenance_writes_no_record(small_ts, tmp_path):
    """``provenance={}`` writes no record at all."""
    out = tmp_path / "bare.trees"
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
    n = inf.to_arg(str(out), provenance={})
    assert n == small_ts.num_sites
    written = tskit.load(str(out))
    assert written.num_provenances == small_ts.num_provenances


def test_panel_root_run_adds_no_focal_totals_to_a_supplied_record(small_ts, tmp_path):
    """A run reporting at the panel root has no focal diagnostics to merge
    into the caller's record once the walk has finished."""
    record = {"software": "x", "parameters": {"note": 1}}
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                            focal="panel_root")
    inf.to_arg(str(tmp_path / "root.trees"), provenance=record)
    assert record["parameters"] == {"note": 1}


def test_every_mode_records_the_panel_it_used(small_ts):
    """ARG mode records the ingroup and outgroups it resolved, as local-tree
    mode does, so a reader can grade an annotated output without being told
    the panel again."""
    names = [f"n{i}" for i in range(small_ts.num_samples)]
    inf = ARGBasedInference(
        small_ts, JC69(), mu=1e-8, progress=False,
        sample_map={n: i for i, n in enumerate(names)},
        outgroup_samples=names[-1:])
    p = inf.provenance()["parameters"]
    assert p["outgroup_samples"] == names[-1:]
    assert p["ingroup_samples"] == names[:-1]
    assert p["panel_samples"] == names


def test_an_unchunked_run_records_the_model_parameters():
    """The model's own parameters and the base composition are recorded on
    every path, not only the chunked one."""
    ts = tskit.load(TREES)
    sites = list(anc.TskitSource(ts))
    composition = BaseComposition(counts={"A": 30, "C": 20, "G": 20, "T": 30})
    params = {}
    for chunk_size in (None, 50_000):
        inf = anc.Inference.from_local_tree(
            sites, model=anc.HKY(kappa=4.0), mu=5e-8, rec_rate=1e-8,
            sample_names=ING + OUT, sequence_length=float(ts.sequence_length),
            base_composition=composition, ingroup_samples=ING,
            outgroup_samples=OUT, chunk_size=chunk_size, progress=False)
        params[chunk_size] = inf._provenance_parameters()
    for key in ("model_kappa", "base_composition_pi"):
        assert key in params[None], f"{key} missing from the unchunked record"
        assert key in params[50_000], f"{key} missing from the chunked record"
