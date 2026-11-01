"""Repolarising a site rebuilds that site's mutations alone.

tskit requires the mutation-time column to be all known or all unknown, so a
rebuilt mutation takes the midpoint of its branch and every other mutation
keeps its input time. Recomputing the whole column from the topology instead
moved the times of 119 of 120 untouched mutations by a median 37%.
"""
import msprime
import numpy as np
import tskit

import ancestree as anc


def test_untouched_sites_keep_their_input_mutation_times(tmp_path):
    ts = msprime.sim_ancestry(
        samples=6, ploidy=2, sequence_length=20_000, population_size=10_000,
        recombination_rate=1e-8, random_seed=3)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
    assert ts.num_sites > 20
    before = {int(m.id): (int(m.site), m.node, m.time) for m in ts.mutations()}
    assert not any(tskit.is_unknown_time(t) for _, _, t in before.values())

    flip = int(next(iter(ts.sites())).id)
    pairs = []
    for site in ts.sites():
        alt = [a for a in "ACGT" if a != site.ancestral_state]
        want = alt[0] if int(site.id) == flip else site.ancestral_state
        vals = np.array([1.0 if b == want else 0.0 for b in "ACGT"])
        rec = anc.Site(chrom="1", pos=int(site.position),
                       alleles=(site.ancestral_state,), tip_alleles={})
        pairs.append((rec, anc.Posterior(alleles=tuple("ACGT"), values=vals)))

    out_path = tmp_path / "annotated.trees"
    assert anc.TskitWriter(ts, out_path).write(pairs) > 0
    out = tskit.load(str(out_path))

    kept = sorted((s, node, t) for s, node, t in before.values() if s != flip)
    assert sorted((int(m.site), m.node, m.time) for m in out.mutations()
                  if m.site != flip) == kept
    tree = out.at(out.site(flip).position)
    for m in out.site(flip).mutations:
        parent = tree.parent(m.node)
        assert out.nodes_time[m.node] <= m.time
        assert parent < 0 or m.time < out.nodes_time[parent]
