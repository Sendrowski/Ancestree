"""ARG mode must fold allele case, as every source path does.

A tree sequence built off a soft-masked reference carries lowercase alleles.
All three source backends upper-case them. The ARG path read them raw, so a
lowercase allele missed the state index and every tip at that site read as
missing, leaving the posterior equal to the prior.
"""
import msprime
import numpy as np

import ancestree as anc


def _lowercased_ts():
    ts = msprime.sim_ancestry(samples=4, ploidy=1, sequence_length=5_000,
                              population_size=10_000, random_seed=5)
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=5)
    assert ts.num_sites > 3
    t = ts.dump_tables()
    t.sites.clear()
    for s in ts.sites():
        t.sites.add_row(position=s.position,
                        ancestral_state=s.ancestral_state.lower())
    t.mutations.clear()
    for m in ts.mutations():
        t.mutations.add_row(site=m.site, node=m.node,
                            derived_state=m.derived_state.lower(),
                            parent=m.parent, time=m.time)
    t.sort()
    return t.tree_sequence()


def test_lowercase_alleles_still_reach_the_kernel():
    ts = _lowercased_ts()
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=1e-8, progress=False)
    posts = [p for _s, p in inf.infer()]
    assert posts, "no sites inferred"
    # A site whose tips all read as missing returns the prior, which for JC69
    # with a uniform composition is flat. At least one call must be decisive.
    sharpest = max(float(np.max(p.values)) for p in posts)
    assert sharpest > 0.30, (
        f"every posterior is near-uniform (max mass {sharpest:.3f}); "
        "lowercase alleles are being read as missing")


class TestTheFoldReachesEveryScoringPath:
    """Soft-masked input must give the same answer as the same data uppercased.

    The fold once reached the ingroup tally but not the generic kernel, so a
    lowercase site scored from the ingroup counts alone: an uninformative flat
    posterior turned into a confident call carrying no outgroup evidence.
    """

    INGROUP = [f"i{j}" for j in range(6)]
    OUTGROUP = ["o1", "o2"]

    @classmethod
    def _panel(cls, lower: bool):
        f = (lambda a: a.lower()) if lower else (lambda a: a)
        rng = np.random.default_rng(0)
        sites = []
        for pos in range(1, 201):
            g = rng.integers(0, 2, len(cls.INGROUP))
            tips = {name: f("A" if g[j] == 0 else "C")
                    for j, name in enumerate(cls.INGROUP)}
            tips.update({name: f("A") for name in cls.OUTGROUP})
            sites.append(anc.Site(chrom="1", pos=pos,
                                  alleles=(f("A"), f("C")), tip_alleles=tips))
        return sites

    @classmethod
    def _run(cls, sites):
        composition = anc.BaseComposition.from_polymorphic_sites(sites)
        inference = anc.FixedTreeInference(
            sites, anc.JC69(), composition, ingroup_samples=cls.INGROUP,
            outgroup_samples=cls.OUTGROUP, fit_required=False)
        return (np.array([p.values for _, p in inference.infer()]),
                np.asarray(composition.pi))

    def test_lowercase_scores_exactly_as_uppercase(self):
        upper, pi_upper = self._run(self._panel(lower=False))
        lower, pi_lower = self._run(self._panel(lower=True))
        np.testing.assert_array_equal(upper, lower)
        np.testing.assert_array_equal(pi_upper, pi_lower)
