"""A one-sample ingroup, where the MRCA is that sample's own tip.

The run reports at that tip and warns: the tip is both the query node and an
observation, so the posterior is a point mass on the allele it carries. The
batch walk and ``infer_site`` resolve the focal node through separate code
paths and must agree.
"""
import logging

import msprime

import ancestree as anc
from ancestree.focal import FocalNode
from ancestree.models import JC69
from ancestree.trees import TskitLocalTree


SAMPLE_MAP = {f"s{i}": i for i in range(4)}


def _ts():
    ts = msprime.sim_ancestry(samples=4, sequence_length=100, ploidy=1,
                              random_seed=7)
    return msprime.sim_mutations(ts, rate=0.02, random_seed=3)


def _inference(ingroup, outgroup, focal=None):
    kwargs = {} if focal is None else {"focal": focal}
    return anc.ARGBasedInference(
        _ts(), JC69(), mu=0.02, sample_map=SAMPLE_MAP,
        ingroup_samples=ingroup, outgroup_samples=outgroup, **kwargs)


def _map_calls(inf):
    """MAP allele per site from the batch walk and from ``infer_site``."""
    ts = inf.ts if hasattr(inf, "ts") else _ts()
    batch, single = {}, {}
    for site, post in inf.infer():
        batch[int(site.pos)] = post.map_allele
        tree = TskitLocalTree(ts, int(site.pos), SAMPLE_MAP)
        single[int(site.pos)] = inf.infer_site(tree, site).map_allele
    return batch, single


def test_a_lone_ingroup_sample_gives_the_same_call_from_both_entry_points():
    inf = _inference(("s0",), ("s1", "s2", "s3"))
    batch, single = _map_calls(inf)
    assert batch and batch == single, (
        f"batch walk and infer_site disagree: {batch} vs {single}")


def test_a_lone_ingroup_sample_is_reported_at_its_own_tip(caplog):
    """Every call is the sample's own allele, and the run says so."""
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        inf = _inference(("s0",), ("s1", "s2", "s3"))
    assert any("ingroup is a single sample" in r.getMessage()
               for r in caplog.records)
    calls = [(site.tip_alleles["s0"], post.map_allele, post.max_prob)
             for site, post in inf.infer()]
    assert calls, "no sites scored"
    assert all(own == called and prob > 0.999 for own, called, prob in calls)


def test_several_ingroup_samples_do_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        _inference(("s0", "s1"), ("s2", "s3"))
    assert not any("ingroup is a single sample" in r.getMessage()
                   for r in caplog.records)


def test_two_ingroup_samples_are_unaffected():
    inf = _inference(("s0", "s1"), ("s2", "s3"))
    batch, single = _map_calls(inf)
    assert batch and batch == single


def test_a_placement_above_a_lone_tip_still_re_roots():
    """A fraction walks up to an internal node."""
    plain = _inference(("s0",), ("s1", "s2", "s3"))
    placed = _inference(("s0",), ("s1", "s2", "s3"),
                        focal=FocalNode(anchor="ingroup_mrca", fraction=0.5))
    at_tip = {int(s.pos): p.to_dict() for s, p in plain.infer()}
    lifted = {int(s.pos): p.to_dict() for s, p in placed.infer()}
    assert at_tip.keys() == lifted.keys()
    assert at_tip != lifted, (
        "a fractional placement above the tip must not read at the tip")


def test_local_tree_mode_warns_on_a_lone_ingroup_sample(caplog):
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        anc.LocalTreeInference(
            _ts(), JC69(), mu=0.02, rec_rate=1e-3,
            ingroup_samples=["0"],
            outgroup_samples=["1", "2", "3"])
    assert sum("ingroup is a single sample" in r.getMessage()
               for r in caplog.records) == 1
