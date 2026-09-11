"""The ingroup-polymorphic site filter, as an input stream and as a grade filter.

Recovery on the quickstart panel is capped by identifiability, not by the
kernel: where the ingroup is fixed for one allele and both outgroups carry the
other, the mutation sits either on the branch above the ingroup MRCA or on the
branch above the outgroup MRCA, and a single outgroup population cannot say
which. These tests pin the split so a defect in the kernel cannot hide
behind that ceiling.
"""
import tskit

import ancestree as anc


INGROUP = [f"i{i}" for i in range(6)]
OUTGROUP = ["o0", "o1"]


def _ts():
    return tskit.load("docs/_static/quickstart.trees")


def test_the_two_criteria_partition_the_stream():
    ts = _ts()
    src = list(anc.TskitSource(ts))
    poly = list(anc.PolymorphicSiteFilter(src, samples=INGROUP))
    mono = list(anc.PolymorphicSiteFilter(src, samples=INGROUP, keep="monomorphic"))
    assert len(poly) + len(mono) == len(src)
    assert not {s.pos for s in poly} & {s.pos for s in mono}


def test_a_site_segregating_only_in_the_outgroups_is_monomorphic():
    """The class the ingroup filter exists to drop: an SFS gets nothing from it."""
    site = anc.Site(
        chrom="1", pos=1,
        alleles=("G", "T"),
        tip_alleles={**{s: "G" for s in INGROUP}, "o0": "T", "o1": "T"},
    )
    assert not anc.PolymorphicSiteFilter(samples=INGROUP).accepts(site)
    assert anc.PolymorphicSiteFilter(samples=INGROUP, keep="monomorphic").accepts(site)
    # ...while over the whole panel that same site is segregating.
    assert anc.PolymorphicSiteFilter().accepts(site)


def test_an_unbound_filter_is_a_predicate_until_called():
    keep = anc.PolymorphicSiteFilter(samples=INGROUP)
    try:
        list(keep)
    except ValueError as exc:
        assert "no source bound" in str(exc)
    else:
        raise AssertionError("iterating an unbound filter must raise")
    bound = keep(list(anc.TskitSource(_ts())))
    assert list(bound)


def test_it_rejects_an_unknown_criterion():
    try:
        anc.PolymorphicSiteFilter(keep="segregating")
    except ValueError as exc:
        assert "polymorphic" in str(exc)
    else:
        raise AssertionError("an unknown keep= must raise")


def test_the_inverted_filter_is_the_complement():
    ts = _ts()
    keep = anc.PolymorphicSiteFilter(samples=INGROUP)
    sites = list(anc.TskitSource(ts))
    assert all(keep.accepts(s) != (~keep).accepts(s) for s in sites)
    assert (~~keep).keep == keep.keep


def test_a_callable_filters_a_grade():
    ts = _ts()
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=5e-8, progress=False)
    keep = anc.PolymorphicSiteFilter(samples=INGROUP)
    res = list(inf.infer())
    assert anc.Grade(res, ts, filter=keep.accepts) == anc.Grade(res, ts, filter=keep)
    assert anc.Grade(res, ts, filter=lambda s: True) == anc.Grade(res, ts)


def test_filtering_at_grade_decomposes_the_same_run():
    """Grading filtered and unfiltered must partition one run's sites exactly."""
    ts = _ts()
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=5e-8, progress=False)
    keep = anc.PolymorphicSiteFilter(samples=INGROUP)
    drop = anc.PolymorphicSiteFilter(samples=INGROUP, keep="monomorphic")

    whole = inf.grade(ts)
    poly = inf.grade(ts, filter=keep)
    mono = inf.grade(ts, filter=drop)

    assert poly.n_sites + mono.n_sites == whole.n_sites
    # The identifiable class is polarised near-perfectly. The other is not.
    assert poly.map_recovery > 0.95
    assert mono.map_recovery < poly.map_recovery
    # Their site-weighted mean reproduces the undecomposed figure.
    pooled = (poly.map_recovery * poly.n_sites
              + mono.map_recovery * mono.n_sites) / whole.n_sites
    assert abs(pooled - whole.map_recovery) < 1e-9


def test_it_reports_the_source_panel_not_the_counted_subset():
    """Consumers build their tip panel from samples(). Narrowing it loses outgroups."""
    src = list(anc.TskitSource(_ts()))
    keep = anc.PolymorphicSiteFilter(src, samples=INGROUP)
    assert set(OUTGROUP) <= set(keep.samples()), (
        f"outgroups missing from samples(): {keep.samples()}")


def test_an_unknown_sample_id_raises_rather_than_emptying_the_stream():
    """A typo would otherwise count zero alleles and read as 0% accuracy."""
    site = anc.Site(chrom="1", pos=1, alleles=("G", "T"),
                   tip_alleles={s: "G" for s in INGROUP})
    try:
        anc.PolymorphicSiteFilter(samples=["nope"]).accepts(site)
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("an absent sample id must raise")


def test_a_one_shot_iterable_survives_a_second_pass():
    """FixedTreeInference walks its source three times."""
    src = list(anc.TskitSource(_ts()))
    keep = anc.PolymorphicSiteFilter((s for s in src), samples=INGROUP)
    first, second = list(keep), list(keep)
    assert first and len(first) == len(second)
