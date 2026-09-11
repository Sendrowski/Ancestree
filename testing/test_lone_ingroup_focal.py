"""A one-sample ingroup, where the MRCA is that sample's own tip.

Re-rooting at a tip with no offset makes it both the query node and an
observation, so the posterior collapses onto the allele it carries and the run
hands the input back as an ancestral call. The batch walk and ``infer_site``
resolve the focal node through separate code paths, and a guard added to one of
them left the other returning that point mass.
"""
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


def test_a_lone_ingroup_sample_is_not_called_as_its_own_allele():
    """The point mass the degenerate rooting produces, stated directly."""
    inf = _inference(("s0",), ("s1", "s2", "s3"))
    observed = {}
    for site, post in inf.infer():
        observed[int(site.pos)] = (site.tip_alleles["s0"], post.max_prob)
    assert observed, "no sites scored"
    assert not all(p == 1.0 for _, p in observed.values()), (
        f"every call is a point mass on s0's own allele: {observed}")


def test_two_ingroup_samples_are_unaffected():
    inf = _inference(("s0", "s1"), ("s2", "s3"))
    batch, single = _map_calls(inf)
    assert batch and batch == single


def test_a_placement_above_a_lone_tip_still_re_roots():
    """A fraction walks up to an internal node, so it is not degenerate."""
    plain = _inference(("s0",), ("s1", "s2", "s3"))
    placed = _inference(("s0",), ("s1", "s2", "s3"),
                        focal=FocalNode(anchor="ingroup_mrca", fraction=0.5))
    at_root = {int(s.pos): p.to_dict() for s, p in plain.infer()}
    lifted = {int(s.pos): p.to_dict() for s, p in placed.infer()}
    assert at_root.keys() == lifted.keys()
    assert at_root != lifted, (
        "a fractional placement above the tip must not fall back to the "
        "tree's own rooting")
