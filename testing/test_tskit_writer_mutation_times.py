"""Repolarising one site rewrites every mutation time.

tskit requires the mutation-time column to be all known or all unknown, so a
single flipped site forces ``compute_mutation_times`` over the whole column.
That derives times from the topology rather than recovering the input's, which
means an ARG round-tripped through ``to_arg()`` loses its mutation times as
soon as anything is repolarised. The behaviour is forced by the constraint;
this test states it so it is not mistaken for an accident.
"""
import msprime
import numpy as np
import tskit

import ancestree as anc


def test_untouched_sites_lose_their_input_mutation_times(tmp_path):
    ts = msprime.sim_ancestry(
        samples=6, ploidy=2, sequence_length=20_000, population_size=10_000,
        recombination_rate=1e-8, random_seed=3)
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
    assert ts.num_sites > 20
    before = {int(m.site): m.time for m in ts.mutations()}
    assert not any(tskit.is_unknown_time(t) for t in before.values())

    # Flip a single site: every other site's mutation time must be inspected.
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
    n = anc.TskitWriter(ts, out_path).write(pairs)
    assert n > 0, "writer annotated no sites"
    out = tskit.load(str(out_path))

    after = {int(m.site): m.time for m in out.mutations()}
    changed = [s for s in before
               if s in after and abs(before[s] - after[s]) > 1e-9]
    assert len(changed) > 1, (
        "expected the whole mutation-time column to be recomputed, not only "
        f"the flipped site; changed={len(changed)}")
