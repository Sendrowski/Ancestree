"""Every panel sample is ingroup or outgroup.

With both lists named, a sample in neither is dropped, or raises when the
caller chose the panel explicitly. With one list named, the other is the rest
of the panel.
"""
from __future__ import annotations

import numpy as np
import pytest
import tskit

import ancestree as anc
from ancestree.local_tree_inference import LocalTreeInference
from ancestree.trees import TskitLocalTree
from testing._helpers import DEMO_VCF, QUICKSTART_TREES

ING = ["i0", "i1", "i2", "i3"]
OUT = ["o0"]


def _posteriors(ts, focal, **kw):
    """Posterior matrix of an ARG run over ``ts``."""
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=5e-8, focal=focal,
                                 progress=False, **kw)
    return np.array([p.values for _s, p in inf.infer()])


def _simplified():
    """The quickstart ARG simplified to ``ING + OUT``, with its sample map."""
    ts = tskit.load(QUICKSTART_TREES)
    full = TskitLocalTree.default_sample_map(ts)
    nodes = sorted(full[s] for s in ING + OUT)
    rank = {n: i for i, n in enumerate(nodes)}
    return (ts.simplify(samples=nodes, filter_sites=False,
                        record_provenance=False),
            {s: rank[full[s]] for s in ING + OUT})


@pytest.mark.parametrize("focal", ["ingroup_mrca", "panel_root"])
def test_arg_drops_samples_in_neither_list(focal):
    """Unlabelled haplotypes used to stay in the tree as observed tips and
    moved the posterior. The run must equal one on the ARG without them."""
    ts = tskit.load(QUICKSTART_TREES)
    got = _posteriors(ts, focal, ingroup_samples=ING, outgroup_samples=OUT)
    small, smap = _simplified()
    want = _posteriors(small, focal, ingroup_samples=ING,
                       outgroup_samples=OUT, sample_map=smap)
    np.testing.assert_allclose(got, want, atol=1e-12)
    kept = _posteriors(ts, focal, ingroup_samples=ING)
    assert np.max(np.abs(kept - got)) > 1e-3


def test_arg_explicit_sample_map_with_unlabelled_sample_raises():
    ts = tskit.load(QUICKSTART_TREES)
    smap = TskitLocalTree.default_sample_map(ts)
    with pytest.raises(ValueError, match="in neither"):
        anc.Inference.from_arg(ts, mu=5e-8, sample_map=smap,
                               ingroup_samples=ING, outgroup_samples=OUT)


def test_ingroup_alone_makes_the_rest_outgroups():
    inf = anc.Inference.from_arg(tskit.load(QUICKSTART_TREES), mu=5e-8,
                                 ingroup_samples=ING)
    assert inf._baseline_outgroup_samples() == ("i4", "i5", "o0", "o1")


def test_local_tree_panel_is_the_union_of_both_lists():
    inf = LocalTreeInference(DEMO_VCF, mu=5e-8, rec_rate=1e-8,
                             ingroup_samples=["i0", "i1"],
                             outgroup_samples=["o1"], progress=False)
    assert inf.sample_names == ["i0_h0", "i0_h1", "i1_h0", "i1_h1",
                                "o1_h0", "o1_h1"]


def test_local_tree_explicit_sample_names_with_unlabelled_sample_raises():
    with pytest.raises(ValueError, match="in neither"):
        LocalTreeInference(DEMO_VCF, mu=5e-8, rec_rate=1e-8,
                           sample_names=["i0_h0", "i1_h0", "o1_h0"],
                           ingroup_samples=["i0"], outgroup_samples=["o1"],
                           progress=False)


def test_fixed_tree_sample_filter_with_unlabelled_sample_raises():
    with pytest.raises(ValueError, match="in neither"):
        anc.Inference.from_fixed_tree(
            DEMO_VCF, ingroup_samples=["i0", "i1"],
            outgroup_samples=["o1_h0"], sample_filter=["i0", "i1", "o1", "i2"],
            n_target_sites=100_000, progress=False)


def test_a_sample_in_both_lists_raises():
    with pytest.raises(ValueError, match="in both"):
        anc.Inference.from_arg(tskit.load(QUICKSTART_TREES), mu=5e-8,
                               ingroup_samples=ING, outgroup_samples=["i0"])
    with pytest.raises(ValueError, match="in both"):
        anc.Inference.from_fixed_tree(
            DEMO_VCF, ingroup_samples=["i0", "i1"],
            outgroup_samples=["i0_h0"], n_target_sites=100_000,
            progress=False)


def test_local_tree_sites_hold_only_the_panel(tmp_path):
    """Sites emitted from a store carry only the panel's tips and alleles, so
    a template written from the inferred trees takes every posterior."""
    import bio2zarr.vcf as bio2zarr_vcf

    store = str(tmp_path / "demo.vcz")
    bio2zarr_vcf.convert([DEMO_VCF], store, show_progress=False)
    inf = LocalTreeInference(store, mu=5e-8, rec_rate=1e-8,
                             ingroup_samples=["i0", "i1"],
                             outgroup_samples=["o1"], sequence_length=2e5,
                             chunk_size=None, n_ensemble=None, progress=False)
    assert inf._input_store_path == store
    res = list(inf.infer())
    panel = set(inf.sample_names)
    assert all(set(site.tip_alleles) <= panel for site, _ in res)
    assert inf.to_vcf(str(tmp_path / "out.vcf"), posteriors=res) == len(res)
