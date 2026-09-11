"""The ensemble must resolve the ingroup by name, not by column position.

It located the ingroup MRCA from a COUNT, assuming the ingroup occupied
genotype columns [0, n_ing). Two documented usages break that assumption:
naming only ``outgroup_samples`` (the ingroup is then everything else), and
naming an ingroup that is not the leading block of the panel. Both scored at
the wrong node while the plug-in path scored correctly, so the two estimators
answered different questions on identical input.
"""
import msprime
import numpy as np

import ancestree as anc

ING = [f"i{i}" for i in range(6)]
OUT = [f"o{i}" for i in range(3)]


def _panel(order):
    """A split panel, emitted in the requested column order."""
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=10_000)
    dem.add_population(name="O", initial_size=10_000)
    dem.add_population(name="ANC", initial_size=10_000)
    dem.add_population_split(time=20_000, derived=["A", "O"], ancestral="ANC")
    ts = msprime.sim_ancestry(
        samples={"A": len(ING), "O": len(OUT)}, demography=dem, ploidy=1,
        sequence_length=40_000, recombination_rate=1e-8, random_seed=11)
    ts = msprime.sim_mutations(ts, rate=2.5e-8, random_seed=11)
    names = ING + OUT
    sites = []
    for v in ts.variants():
        alleles = tuple(a.upper() for a in v.alleles if a)
        if len(alleles) < 2:
            continue
        tips = {names[j]: v.alleles[g].upper()
                for j, g in enumerate(v.genotypes) if v.alleles[g]}
        sites.append(anc.Site(chrom="1", pos=int(v.site.position),
                             alleles=alleles,
                             tip_alleles={n: tips[n] for n in order}))
    return sites, order


def _run(order, **kw):
    sites, names = _panel(order)
    inf = anc.Inference.from_local_tree(
        sites, model=anc.JC69(), mu=2.5e-8, rec_rate=1e-8,
        sample_names=names, sequence_length=40_000.0,
        window="8snp", ensemble_seed=3, progress=False, **kw)
    return np.array([p.values for _s, p in inf.infer()])


def test_naming_only_outgroups_matches_naming_the_ingroup():
    """The documented shorthand must reach the same node as the explicit form."""
    order = ING + OUT
    implicit = _run(order, outgroup_samples=OUT)
    explicit = _run(order, ingroup_samples=ING, outgroup_samples=OUT)
    assert implicit.shape == explicit.shape
    diff = float(np.max(np.abs(implicit - explicit)))
    assert diff < 1e-9, (
        f"naming only the outgroups scores a different node (max |diff| "
        f"{diff:.3f}); the ingroup is being taken as the leading columns")


def test_an_ingroup_that_is_not_first_is_still_found():
    """Column order must not change which clade's MRCA is reported.

    Not bit-identical: UPGMA breaks ties by index, so reordering the panel can
    pick a different member of a tied pair and shift that site's posterior.
    Those shifts are confined to the few sites carrying a tie, so the bulk of
    the panel is compared rather than the worst site alone. The positional bug
    this guards against moved calls by ~0.9 across the panel, which neither
    bound below tolerates.
    """
    leading = _run(ING + OUT, ingroup_samples=ING, outgroup_samples=OUT)
    shuffled = _run(OUT + ING, ingroup_samples=ING, outgroup_samples=OUT)
    per_site = np.abs(leading - shuffled).max(axis=1)
    bulk = float(np.quantile(per_site, 0.95))
    worst = float(per_site.max())
    assert bulk < 0.05, (
        f"moving the ingroup off the leading columns changed the posterior at "
        f"more than a tied site or two (95th percentile |diff| {bulk:.3f}); "
        f"the ingroup is being taken positionally")
    assert worst < 0.5, (
        f"a site moved by {worst:.3f}, which is the scale of reading the "
        f"ingroup positionally rather than of a broken tie")


def test_the_ensemble_draws_do_not_follow_the_column_order(caplog):
    """Sampled genealogies are keyed on sample names, not column position.

    Keyed on the pair's index in the condensed distance matrix, a permuted
    panel handed every pair a different RNG stream, so the drawn genealogies
    changed with an arbitrary bookkeeping choice. The residue here is the
    UPGMA tie-breaking, which is index-ordered inside SciPy.
    """
    leading = _run(ING + OUT, ingroup_samples=ING, outgroup_samples=OUT)
    shuffled = _run(OUT + ING, ingroup_samples=ING, outgroup_samples=OUT)
    per_site = np.abs(leading - shuffled).max(axis=1)
    assert float(per_site.max()) < 0.02, (
        f"the ensemble moved by {per_site.max():.4f} under a column "
        f"permutation; the draws are following the panel's column order")
