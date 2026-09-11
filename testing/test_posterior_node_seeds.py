"""A ladder tree carries its ingroup only through node seeds.

OutgroupLadderTree collapses the ingroup to a single node with no tips beneath
it, so Likelihood.posterior sees no ingroup allele unless the caller injects the
ingroup weight's partial likelihood at that node. Without it the outgroups carry
the entire call, and the returned distribution is normalised and plausible.
"""
import numpy as np

import ancestree as anc


INGROUP = [f"i{i}" for i in range(10)]
OUTGROUP = ["o0", "o1"]


def _setup():
    tree = anc.OutgroupLadderTree(ingroup_samples=INGROUP, outgroup_samples=OUTGROUP)
    site = anc.Site(
        chrom="1", pos=1, alleles=("A", "C"),
        tip_alleles={**{s: ("A" if i < 9 else "C") for i, s in enumerate(INGROUP)},
                     "o0": "C", "o1": "C"},
    )
    lik = anc.Likelihood(anc.JC69(), base_composition=anc.BaseComposition.no_counts())
    weight = anc.KingmanIngroupWeight(ingroup_samples=INGROUP)
    seeds = {tree.ingroup_mrca: np.exp(weight.log_probs([site]))}
    return lik, tree, site, seeds


def test_seeding_the_ingroup_mrca_changes_the_call():
    """9 of 10 ingroup samples carry A; the two outgroups carry C."""
    lik, tree, site, seeds = _setup()
    seeded = lik.posterior(tree, site, node_seeds=seeds)
    assert seeded.map_allele == "A", (
        f"the ingroup majority must decide this site, got {seeded.values}")
    assert seeded["A"] > 0.8


def test_the_unseeded_result_is_outgroup_only():
    """Pins the trap: without seeds the ingroup contributes nothing."""
    lik, tree, site, _ = _setup()
    bare = lik.posterior(tree, site)
    assert bare.map_allele == "C"
    # The three states no outgroup carries sit at the prior, which is the tell
    # that no ingroup evidence entered the kernel.
    assert abs(bare["A"] - bare["G"]) < 1e-9
    assert abs(bare["A"] - bare["T"]) < 1e-9
