"""End-to-end tests for ``ARGBasedInference``: shape, normalisation, the
constructor guards, ``infer_site``, and agreement with
``tskit.Tree.map_mutations`` (the PolarBEAR parsimony pick) under JC69 with a
uniform prior on non-homoplastic sites.

Agreement with PolarBEAR itself is covered by ``test_polarbear_agreement.py``.
"""
from __future__ import annotations

import logging

import msprime

import numpy as np
import pytest
import tskit

from ancestree import (
    ARGBasedInference,
    JC69,
    KingmanIngroupWeight,
    MajorityOutgroupInference,
    STATE_INDEX,
    STATES,
    Site,
)
from ancestree.trees import TskitLocalTree
import ancestree as anc
from testing._helpers import canonical_alleles


def test_returns_one_posterior_per_site(small_ts):
    results = list(ARGBasedInference(small_ts, JC69(), mu=1.0).infer())
    assert len(results) == small_ts.num_sites


def test_trees_path_source(small_ts, tmp_path):
    """A ``.trees`` path source is loaded via ``tskit.load`` and yields one
    posterior per site (covers the str/PathLike dispatch in ``__init__``)."""
    p = tmp_path / "src.trees"
    small_ts.dump(str(p))
    results = list(ARGBasedInference(str(p), JC69(), mu=1.0).infer())
    assert len(results) == small_ts.num_sites


def test_to_arg_round_trip(small_ts, tmp_path):
    """``to_arg`` (hoisted to the base ``Inference``) writes an annotated
    ``.trees`` whose sites carry an ``ancestral_state`` from the MAP allele."""
    import tskit
    out = tmp_path / "annot.trees"
    n = ARGBasedInference(small_ts, JC69(), mu=1e-8).to_arg(str(out))
    assert n == small_ts.num_sites
    annotated = tskit.load(str(out))
    assert all(s.ancestral_state in STATES for s in annotated.sites())


def test_negative_mu_rejected(small_ts):
    """``mu`` must be strictly positive. The constructor guards against
    accidentally passing zero or a negative rate."""
    with pytest.raises(ValueError, match="mu must be positive"):
        ARGBasedInference(small_ts, JC69(), mu=-1.0)


@pytest.mark.parametrize("mu", [float("nan"), float("inf")])
def test_non_finite_mu_rejected(small_ts, mu):
    """``nan <= 0`` and ``inf <= 0`` are both False, so a positivity test alone
    admitted them and the failure came much later from the branch-length
    scaling, naming ``time_scale`` and ``t``, symbols the caller never set. A
    rate computed as ``theta / (4 * Ne)`` with ``Ne = 0``, or read from a table
    with a missing entry, arrives this way.
    """
    with pytest.raises(ValueError, match="mu must be positive and finite"):
        ARGBasedInference(small_ts, JC69(), mu=mu)


# ------------------------------------------------- ARGBasedInference guards
def test_missing_mu_falls_back_to_the_default_with_a_warning(small_ts, caplog):
    from ancestree import DEFAULT_MU
    with caplog.at_level(logging.WARNING, logger="ancestree"):
        inf = ARGBasedInference(small_ts, JC69(), progress=False)
    assert inf.mu == DEFAULT_MU
    assert any("no mu given" in r.message for r in caplog.records)


class TestArgGuards:
    def test_prior_wrong_type_rejected(self, small_ts):
        with pytest.raises(TypeError, match="prior must be a StationaryPrior"):
            ARGBasedInference(small_ts, JC69(), mu=1e-8, prior="not-a-prior")

    def test_ingroup_conditioned_prior_rejected(self, small_ts):
        with pytest.raises(TypeError, match="conditions on a single shared ingroup-MRCA"):
            ARGBasedInference(
                small_ts, JC69(), mu=1e-8,
                prior=KingmanIngroupWeight(["a", "b"]),
            )

    def test_nonpositive_n_workers_rejected(self, small_ts):
        with pytest.raises(ValueError, match="n_workers must be >= 1"):
            ARGBasedInference(small_ts, JC69(), mu=1e-8, n_workers=0)


def test_posteriors_sum_to_one(small_ts):
    for _site, post in ARGBasedInference(small_ts, JC69(), mu=1.0).infer():
        assert len(post) == 4
        np.testing.assert_allclose(post.values.sum(), 1.0, atol=1e-12)
        assert np.all(post.values >= 0)


def test_map_recovers_msprime_truth(small_ts):
    """End-to-end accuracy: Ancestree JC69 + uniform prior
    should recover the actual ancestral allele (msprime ground truth) at the
    vast majority of non-homoplastic sites.

    Full agreement with `tskit.Tree.map_mutations` is not required: when the
    parsimony score is tied between two candidate roots (which is common at
    sites where a single mutation on either side of the root explains the
    pattern), `map_mutations` breaks ties alphabetically while Ancestree uses
    branch-length asymmetry to pick the more probable root. Ancestree can
    therefore do *better* than tskit's parsimony on tied sites, which is a
    feature, not a bug.
    """
    results = dict(
        ((s.pos, s.chrom), p)
        for s, p in ARGBasedInference(
            small_ts,
            JC69(),
            mu=1e-8,  # branch lengths are in generations
        ).infer()
    )

    n_checked = 0
    n_correct = 0
    variants_by_site = {v.site.id: v for v in small_ts.variants()}
    for tree in small_ts.trees():
        if tree.num_roots != 1:
            continue
        for ts_site in tree.sites():
            variant = variants_by_site[ts_site.id]
            genotypes = variant.genotypes
            alleles = list(variant.alleles)
            if any(g < 0 for g in genotypes):
                continue
            if any(alleles[g] not in STATE_INDEX for g in genotypes):
                continue
            states = [STATE_INDEX[alleles[g]] for g in genotypes]
            _anc, muts = tree.map_mutations(states, alleles=STATES)
            if len(muts) > 1:
                continue  # homoplastic: parsimony reports >1 mutation, skip

            truth = ts_site.ancestral_state
            ancestree_post = results[(int(ts_site.position), "1")]
            n_checked += 1
            if ancestree_post.map_allele == truth:
                n_correct += 1

    assert n_checked > 0, "found no non-homoplastic sites to compare"
    accuracy = n_correct / n_checked
    # 0.75 is a deliberately loose floor for this unit test on a tiny n=10
    # fixture with a uniform prior. With uniform prior, the kernel can only
    # rely on branch-length asymmetry, which is noisy at small sample sizes.
    # The deeper accuracy analysis (and comparison to PolarBEAR) lives in
    # `test_polarbear_agreement.py`.
    assert accuracy > 0.75, (
        f"Ancestree MAP accuracy on non-homoplastic sites is {accuracy:.3f} "
        f"({n_correct}/{n_checked}); expected > 0.75."
    )


def test_summary_aggregates_over_pass(small_ts):
    """`Inference.summary()` reduces a full pass to consistent aggregate stats."""
    from ancestree import InferenceSummary

    s = ARGBasedInference(small_ts, JC69(), mu=1.0).summary()
    assert isinstance(s, InferenceSummary)
    assert s.n_sites == small_ts.num_sites
    # the MAP-allele histogram partitions the sites and uses only real alleles,
    # ordered by descending count
    assert sum(s.map_alleles.values()) == s.n_sites
    assert set(s.map_alleles).issubset(set(STATES))
    counts = list(s.map_alleles.values())
    assert counts == sorted(counts, reverse=True)
    # MAP-prob quantiles are a probability and non-decreasing in the quantile
    qs = [s.max_prob_quantiles[q] for q in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert qs == sorted(qs)
    assert 0.0 <= s.mean_max_prob <= 1.0
    assert all(0.0 <= v <= 1.0 for v in qs)
    # entropy is in [0, log2(4)] bits and the str view is informative
    assert 0.0 <= s.mean_entropy_bits <= 2.0 + 1e-9
    assert "sites" in str(s)


def test_zero_rate_ratemap_floors_instead_of_crashing(caplog):
    """A rate-map mu that integrates to zero over a local tree's
    span is floored (prior-dominated posterior), not fed to the
    positive-only time_scale setter where it aborts the whole walk."""
    import msprime
    ts = msprime.sim_ancestry(
        samples=5, ploidy=1, sequence_length=2000,
        recombination_rate=0, population_size=1e4, random_seed=3,
    )
    ts = msprime.sim_mutations(ts, rate=1e-2, random_seed=3)
    assert ts.num_sites > 0
    # A fully zero-rate map: every local tree's span integrates to zero mass.
    rmap = msprime.RateMap(position=[0, 2000], rate=[0.0])
    inf = ARGBasedInference(ts, JC69(), mu=rmap, progress=False)
    # _mu_for_interval floors the zero-mass interval (deterministic).
    assert inf._mu_for_interval(0.0, 2000.0) == pytest.approx(1e-12)
    with caplog.at_level(logging.WARNING, logger="ancestree.ARGBasedInference"):
        out = list(inf.infer())
    assert len(out) == ts.num_sites  # ran to completion, no crash
    assert any("zero" in r.message.lower() for r in caplog.records)


def test_ratemap_nonzero_interval_unaffected():
    import msprime
    ts = msprime.sim_ancestry(
        samples=4, ploidy=1, sequence_length=2000,
        recombination_rate=0, population_size=1e4, random_seed=5,
    )
    ts = msprime.sim_mutations(ts, rate=1e-2, random_seed=5)
    rmap = msprime.RateMap(position=[0, 1000, 2000], rate=[0.0, 1e-8])
    inf = ARGBasedInference(ts, JC69(), mu=rmap, progress=False)
    assert inf._mu_for_interval(0.0, 500.0) == pytest.approx(1e-12)  # zero region -> floor
    assert inf._mu_for_interval(1000.0, 2000.0) == pytest.approx(1e-8)  # genuine rate


class TestTimeUnits:
    """The declared time units of the tree sequence gate the run.

    ``mu`` multiplies branch lengths, so it is a rate per unit of whatever the
    tree sequence measures time in. A production ARG carried ``unknown`` units
    and was scored with a per-generation rate while its times were in
    coalescent units, which understates every branch by a factor of ``4 N_e``
    and drives the model to the ``mu -> 0`` limit, where the posterior is the
    prior. Neither the unknown nor the uncalibrated case said anything.
    """

    @staticmethod
    def _ts(units):
        import msprime

        ts = msprime.sim_ancestry(
            samples=4, ploidy=1, sequence_length=500,
            population_size=1e4, random_seed=7,
        )
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=7)
        tables = ts.dump_tables()
        tables.time_units = units
        return tables.tree_sequence()

    def test_uncalibrated_is_refused(self):
        import tskit

        with pytest.raises(ValueError, match="uncalibrated"):
            ARGBasedInference(
                self._ts(tskit.TIME_UNITS_UNCALIBRATED), JC69(),
                mu=1e-8, progress=False,
            )

    def test_unknown_warns_and_states_the_assumption(self, caplog):
        import tskit

        with caplog.at_level(
                logging.WARNING, logger="ancestree.ARGBasedInference"):
            ARGBasedInference(
                self._ts(tskit.TIME_UNITS_UNKNOWN), JC69(),
                mu=1e-8, progress=False,
            )
        assert any("generation" in r.message for r in caplog.records)

    def test_a_fitted_rate_may_declare_it_matches(self):
        """A rate fitted to the ARG itself is per that ARG's own time axis.

        An undated ARG has no calibration, but a rate derived from it, such as
        segregating sites over total branch length, is expressed in its units
        by construction. The benchmark scores undated tsinfer ARGs that way
        when tsdate cannot date them, so the refusal has to be answerable.
        """
        import tskit

        ts = self._ts(tskit.TIME_UNITS_UNCALIBRATED)
        mu_eff = ts.num_sites / sum(
            t.total_branch_length * t.span for t in ts.trees())
        inference = ARGBasedInference(
            ts, JC69(), mu=mu_eff, progress=False,
            mu_matches_time_units=True,
        )
        assert sum(1 for _ in inference.infer()) == ts.num_sites

    def test_declared_generations_are_quiet(self, caplog):
        with caplog.at_level(
                logging.WARNING, logger="ancestree.ARGBasedInference"):
            ARGBasedInference(
                self._ts("generations"), JC69(), mu=1e-8, progress=False)
        assert not [r for r in caplog.records if "time units" in r.message]


def test_a_sample_in_both_groups_is_refused():
    """A sample named as ingroup and outgroup would be read as both.

    It would serve as an outgroup tip in the Felsenstein pass and as an
    ingroup haplotype in the ingroup weight at the same time. from_newick
    already refused it. The constructor the inference actually uses did not.
    """
    from ancestree import OutgroupLadderTree

    with pytest.raises(ValueError, match="both"):
        OutgroupLadderTree(["a", "b", "o1"], ["o1"])
    OutgroupLadderTree(["a", "b"], ["o1"])


def test_the_branch_fit_uses_the_supplied_root_prior():
    """Rates are fitted under the prior the posteriors are reported under.

    The objective read the composition-derived stationary vector directly
    and never consulted ``prior=``, so a supplied prior left the fitted
    branch rates bit-identical while moving the posteriors computed from
    them.
    """
    import msprime

    from ancestree import FixedTreeInference
    from ancestree.sites import BaseComposition, Site
    from ancestree.priors import StationaryPrior

    ts = msprime.sim_ancestry(
        6, ploidy=1, sequence_length=60_000, recombination_rate=1e-8,
        population_size=1e4, random_seed=4)
    ts = msprime.sim_mutations(ts, rate=2e-7, random_seed=4)
    names = [f"tsk_{i}" for i in range(ts.num_samples)]
    sites = []
    for v in ts.variants():
        alleles = v.alleles
        if len(alleles) != 2 or any(a not in "ACGT" for a in alleles if a):
            continue
        sites.append(Site(
            chrom="1", pos=int(v.site.position), alleles=tuple(alleles),
            tip_alleles={names[i]: alleles[g]
                         for i, g in enumerate(v.genotypes)}))

    def divergences(prior):
        inference = FixedTreeInference(
            sites, JC69(), BaseComposition.from_n_target_sites(60_000),
            ingroup_samples=names[:4], outgroup_samples=names[4:],
            prior=prior, progress=False)
        inference.fit()
        return np.asarray(inference.tree.outgroup_divergence())

    skewed = StationaryPrior(
        JC69(), BaseComposition.from_counts(A=700, C=100, G=150, T=50))
    assert not np.allclose(divergences(None), divergences(skewed))


def test_infer_site_agrees_with_infer_on_an_empirical_composition():
    """The single-site entry point computes what the batch pass computes.

    ``infer_site`` rebuilt a StationaryPrior from the base composition while
    the batch path used the model's stationary vector unless per-base counts
    were supplied, so the two reported different posteriors, and different
    MAP alleles, for the same site and tree.
    """
    import msprime

    from ancestree import FixedTreeInference
    from ancestree.sites import BaseComposition, Site

    ts = msprime.sim_ancestry(
        6, ploidy=1, sequence_length=60_000, recombination_rate=1e-8,
        population_size=1e4, random_seed=4)
    ts = msprime.sim_mutations(ts, rate=2e-7, random_seed=4)
    names = [f"tsk_{i}" for i in range(ts.num_samples)]
    sites = []
    for v in ts.variants():
        alleles = v.alleles
        if len(alleles) != 2 or any(a not in "ACGT" for a in alleles if a):
            continue
        sites.append(Site(
            chrom="1", pos=int(v.site.position), alleles=tuple(alleles),
            tip_alleles={names[i]: alleles[g]
                         for i, g in enumerate(v.genotypes)}))

    inference = FixedTreeInference(
        sites, JC69(), BaseComposition.from_polymorphic_sites(sites),
        ingroup_samples=names[:4], outgroup_samples=names[4:],
        fit_required=False, progress=False)
    batch = {int(s.pos): np.asarray(p.values) for s, p in inference.infer()}
    assert batch
    for site in sites:
        one = np.asarray(
            inference.infer_site(inference._focal_tree, site).values)
        np.testing.assert_allclose(one, batch[int(site.pos)], rtol=0, atol=0)


def test_the_wrappers_forward_every_constructor_argument(small_ts):
    """``from_*`` must offer what the constructor offers, on the same terms.

    A wrapper that respells the signature drops a parameter from the
    documented entry point: from_fixed_tree defaulted baseline_check to False
    while the constructor and its own docstring said True, and from_arg
    omitted mu_matches_time_units entirely, so an uncalibrated ARG could not
    be scored through it at all.
    """
    import inspect

    from ancestree import FixedTreeInference, Inference, OutgroupLadderTree

    for wrapper in (Inference.from_arg, Inference.from_fixed_tree):
        kinds = [p.kind
                 for p in inspect.signature(wrapper).parameters.values()]
        assert kinds == [inspect.Parameter.VAR_POSITIONAL,
                         inspect.Parameter.VAR_KEYWORD], (
            f"{wrapper.__name__} respells the constructor signature")

    inference = Inference.from_arg(small_ts, JC69(), mu=1e-8, progress=False)
    assert isinstance(inference, ARGBasedInference)
    assert inference.mu == 1e-8

    fixed = Inference.from_fixed_tree(
        [], JC69(), tree=OutgroupLadderTree(["i1"], ["o1", "o2"]),
        fit_required=False, progress=False)
    assert isinstance(fixed, FixedTreeInference)
    default = inspect.signature(
        FixedTreeInference.__init__).parameters["baseline_check"].default
    assert fixed.baseline_check is default


class TestTheAlphabetIsCheckedOverTheTreeSequence:
    """One unreadable site is missing data, a whole unreadable panel is refused.

    The alphabet check runs over the tree sequence, not per local tree, and
    applies to multi-root trees as well as single-root ones.
    """

    KWARGS = dict(sample_map={f"s{i}": i for i in range(6)},
                  ingroup_samples=tuple(f"s{i}" for i in range(4)),
                  outgroup_samples=("s4", "s5"))

    @staticmethod
    def _nucleotide():
        ts = msprime.sim_ancestry(samples=6, sequence_length=100_000, ploidy=1,
                                  recombination_rate=2e-8, population_size=1e4,
                                  random_seed=42)
        return msprime.sim_mutations(ts, rate=2e-8, random_seed=7,
                                     model=msprime.JC69())

    def _score(self, ts):
        return sum(1 for _ in anc.ARGBasedInference(
            ts, JC69(), mu=2e-8, **self.KWARGS).infer())

    def test_one_non_nucleotide_site_does_not_abort_the_walk(self):
        """The site is the only one on its local tree."""
        ts = self._nucleotide()
        lone = [s.id for t in ts.trees() if len(list(t.sites())) == 1
                for s in t.sites()]
        assert lone, "fixture has no single-site local tree"
        target = lone[0]

        tables = ts.dump_tables()
        anc = tskit.unpack_strings(tables.sites.ancestral_state,
                                   tables.sites.ancestral_state_offset)
        anc[target] = "N"
        packed, offset = tskit.pack_strings(anc)
        tables.sites.set_columns(position=tables.sites.position,
                                 ancestral_state=packed,
                                 ancestral_state_offset=offset)
        der = tskit.unpack_strings(tables.mutations.derived_state,
                                   tables.mutations.derived_state_offset)
        der = ["-" if int(site) == target else d
               for d, site in zip(der, tables.mutations.site)]
        dpacked, doffset = tskit.pack_strings(der)
        tables.mutations.set_columns(
            site=tables.mutations.site, node=tables.mutations.node,
            time=tables.mutations.time, derived_state=dpacked,
            derived_state_offset=doffset, parent=tables.mutations.parent)
        dirty = tables.tree_sequence()
        assert self._score(dirty) == dirty.num_sites

    def test_a_binary_alphabet_is_refused_even_when_trees_are_multi_root(self):
        ts = msprime.sim_ancestry(samples=6, sequence_length=2000, ploidy=1,
                                  population_size=1e4, end_time=50,
                                  random_seed=5)
        ts = msprime.sim_mutations(ts, rate=1e-3, random_seed=5,
                                   model=msprime.BinaryMutationModel())
        assert max(t.num_roots for t in ts.trees()) > 1, "fixture is single-root"
        with pytest.raises(ValueError, match="maps to a model state"):
            self._score(ts)


class TestUnrepresentableAllelesAreCounted:
    """Alleles outside A/C/G/T are tallied and reported once per run.

    They were marginalised in silence, so a run could drop most of its tips
    to an indel or a symbolic allele and say nothing about it, leaving the
    resulting calls indistinguishable from calls made on readable data.
    """

    #: Sites carrying an unrepresentable allele in ``_ts``, and the tips
    #: observed at one.
    EXPECTED_SITES = 2
    EXPECTED_TIPS = 5

    @staticmethod
    def _ts():
        """Four samples on one tree with three sites.

        Position 1 is a readable ``A``/``T``. Position 3 puts one ``AT`` tip
        on a readable site. Position 5 carries ``AT`` / ``ATT`` alone, so all
        four of its tips are unrepresentable.
        """
        tables = tskit.TableCollection(sequence_length=10.0)
        samples = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                   for _ in range(4)]
        root = tables.nodes.add_row(flags=0, time=1.0)
        for child in samples:
            tables.edges.add_row(left=0, right=10.0, parent=root, child=child)
        readable = tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.mutations.add_row(site=readable, node=samples[0],
                                 derived_state="T")
        mixed = tables.sites.add_row(position=3.0, ancestral_state="A")
        tables.mutations.add_row(site=mixed, node=samples[0],
                                 derived_state="AT")
        opaque = tables.sites.add_row(position=5.0, ancestral_state="AT")
        tables.mutations.add_row(site=opaque, node=samples[0],
                                 derived_state="ATT")
        tables.sort()
        return tables.tree_sequence()

    @staticmethod
    def _counters(source, **kwargs):
        """Run a walk and read its counters before the summary clears them."""
        inference = ARGBasedInference(
            source, JC69(), mu=1e-8, progress=False, **kwargs)
        inference._quiet = True
        list(inference.infer())
        return (int(inference._n_unrepresentable_sites),
                int(inference._n_unrepresentable_tips))

    def test_the_single_arg_walk_counts_sites_and_tips(self):
        assert self._counters(self._ts()) == (
            self.EXPECTED_SITES, self.EXPECTED_TIPS)

    def test_the_marginalised_walk_counts_the_same(self):
        """A posterior sample scores every draw over the same panel, so the
        run's total is what one draw saw and not the sum over draws."""
        ts = self._ts()
        assert self._counters([ts, ts]) == (
            self.EXPECTED_SITES, self.EXPECTED_TIPS)

    def test_a_readable_panel_counts_nothing(self, small_ts):
        assert self._counters(small_ts) == (0, 0)

    def test_the_run_reports_the_counts_once(self, caplog):
        inference = ARGBasedInference(
            self._ts(), JC69(), mu=1e-8, progress=False)
        with caplog.at_level(logging.WARNING,
                             logger="ancestree.ARGBasedInference"):
            list(inference.infer())
        reported = [r.getMessage() for r in caplog.records
                    if "outside the A/C/G/T alphabet" in r.getMessage()]
        assert len(reported) == 1
        assert f"{self.EXPECTED_SITES} site(s)" in reported[0]
        assert f"{self.EXPECTED_TIPS} tip(s)" in reported[0]


# ------------------------------------------------------------- infer_site()
def test_infer_site_single_tree_posterior():
    tree = TskitLocalTree.from_newick("((a:1,b:1):1,(c:1,d:1):1):0;")
    site = Site(chrom="1", pos=1, alleles=("A", "G"),
                tip_alleles={"a": "A", "b": "A", "c": "G", "d": "G"})
    # MajorityOutgroupInference carries a model + (no) prior, so the base
    # Inference.infer_site path runs the kernel + StationaryPrior fallback.
    inf = MajorityOutgroupInference([], [], for_comparison_only=True)
    post = inf.infer_site(tree, site)
    assert len(post) == 4
    np.testing.assert_allclose(post.values.sum(), 1.0, atol=1e-12)
    assert np.all(post.values >= 0)


def test_infer_site_applies_mu_scaling(small_ts):
    """``ARGBasedInference.infer_site`` scales branch lengths by ``mu``
    like the ``infer()`` walk does.

    ARG branch lengths are in generations and the kernel needs
    ``t = mu * length``. At the tree's default ``time_scale == 1.0``,
    ``exp(Qt)`` saturates to the stationary distribution and the posterior
    collapses to the prior regardless of the data.
    """
    from ancestree.sources import TskitSource

    mu = 1e-8
    inf = ARGBasedInference(small_ts, JC69(), mu=mu, progress=False)

    # Authoritative posteriors from the full walk (sets time_scale = mu itself).
    walk = {int(s.pos): post for s, post in inf.infer()}

    # Same first site, hand-driven through infer_site on an un-scaled tree.
    site = next(iter(TskitSource(small_ts)))
    tree = TskitLocalTree(small_ts, position=site.pos)
    assert tree.time_scale == 1.0  # as built
    post = inf.infer_site(tree, site)

    # mu is applied to the supplied tree,
    assert tree.time_scale == mu
    # the posterior is informative rather than the uniform prior,
    assert post.max_prob > 0.25
    # and it agrees with the full-walk posterior for that site.
    np.testing.assert_allclose(
        post.values, walk[int(site.pos)].values, atol=1e-9,
    )


def test_infer_site_all_neg_inf_falls_back_to_uniform(monkeypatch):
    """When the prior excludes every observed allele (all-``-inf``
    ``log_post``) ``infer_site`` routes through the shared normaliser and
    falls back to uniform, not to NaN posteriors."""
    from ancestree.priors import StationaryPrior

    tree = TskitLocalTree.from_newick("((a:1,b:1):1,(c:1,d:1):1):0;")
    site = Site(chrom="1", pos=1, alleles=("A", "G"),
                tip_alleles={"a": "A", "b": "A", "c": "G", "d": "G"})
    inf = MajorityOutgroupInference([], [], for_comparison_only=True)
    # self.prior is None here, so infer_site builds a StationaryPrior. Force it
    # to exclude every state so log_post is all -inf.
    monkeypatch.setattr(
        StationaryPrior, "log_probs",
        lambda self, sites: [np.full(4, -np.inf)],
    )
    post = inf.infer_site(tree, site)
    assert not np.any(np.isnan(post.values))
    np.testing.assert_allclose(post.values.sum(), 1.0)
    np.testing.assert_allclose(post.values, 0.25)


# ------------------------------------------- infer_site() honours the focal
@pytest.fixture(scope="module")
def split_ts():
    """A two-population split, so the ingroup MRCA sits below the panel root."""
    import msprime

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=1e4)
    demography.add_population(name="B", initial_size=1e4)
    demography.add_population(name="C", initial_size=1e4)
    demography.add_population_split(time=2e5, derived=["A", "B"], ancestral="C")
    ts = msprime.sim_ancestry(
        samples={"A": 3, "B": 2}, demography=demography, ploidy=1,
        sequence_length=20_000, recombination_rate=0, random_seed=7,
    )
    return msprime.sim_mutations(ts, rate=2e-7, random_seed=7)


class TestInferSiteReportsAtTheFocalNode:
    """``infer_site`` answers about the same node as ``infer``.

    Both resolve the configured focal node. The two coincide whenever the
    ingroup spans the whole panel, so the fixture names an ingroup narrower
    than the panel.
    """

    MU = 2e-7
    INGROUP = ["0", "1", "2"]
    OUTGROUP = ["3", "4"]

    @staticmethod
    def _sample_map(split_ts):
        return {str(i): i for i in range(split_ts.num_samples)}

    def test_the_focal_node_differs_from_the_root(self, split_ts):
        """Guard the test below against passing because the nodes coincide."""
        sample_map = self._sample_map(split_ts)
        tree = split_ts.at(split_ts.sequence_length / 2)
        mrca = tree.mrca(*[sample_map[s] for s in self.INGROUP])
        assert mrca != tree.root

    def test_infer_site_matches_infer_at_the_ingroup_mrca(self, split_ts):
        sample_map = self._sample_map(split_ts)
        inf = ARGBasedInference(
            split_ts, JC69(), mu=self.MU, sample_map=sample_map,
            focal="ingroup_mrca", ingroup_samples=self.INGROUP,
            outgroup_samples=self.OUTGROUP, progress=False,
        )
        batch = {int(site.pos): post for site, post in inf.infer()}
        assert batch

        tree = TskitLocalTree(
            split_ts, split_ts.sequence_length / 2, sample_map=sample_map)
        names = {i: str(i) for i in range(split_ts.num_samples)}
        compared = 0
        for var in split_ts.variants():
            expected = batch.get(int(var.site.position))
            if expected is None:
                continue
            site = Site(
                chrom="1", pos=int(var.site.position),
                alleles=canonical_alleles(var.alleles),
                tip_alleles={names[i]: var.alleles[g]
                             for i, g in enumerate(var.genotypes)},
            )
            np.testing.assert_allclose(
                inf.infer_site(tree, site).values, expected.values, atol=1e-12)
            compared += 1
        assert compared > 100


# ---------------------------------------------- the run's own bookkeeping


def test_reprs_show_the_configuration(small_ts):
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
    assert repr(inf).startswith("ARGBasedInference(") and "mu=" in repr(inf)
    baseline = MajorityOutgroupInference([], ["o1"], confidence=0.8,
                                         for_comparison_only=True)
    assert repr(baseline) == "MajorityOutgroupInference(confidence=0.8)"


def test_ingroup_monomorphic_tally_counts_the_sites_itself(small_ts):
    """Without precomputed counts the tally reads each site's ingroup alleles."""
    names = list(TskitLocalTree.default_sample_map(small_ts))
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                            outgroup_samples=names[-2:])
    ingroup = names[:-2]
    mono = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={**{s: "A" for s in ingroup}, names[-1]: "C"})
    poly = Site(chrom="1", pos=2, alleles=("A", "C"),
                tip_alleles={**{s: "A" for s in ingroup}, ingroup[0]: "C"})
    inf._count_ingroup_monomorphic([mono, poly, mono])
    assert inf._n_ingroup_monomorphic == 2


def test_all_outgroup_panel_leaves_nothing_to_resolve(small_ts):
    """Naming every panel sample as an outgroup empties the ingroup, which the
    ingroup-MRCA focal node needs and the panel root does not."""
    names = list(TskitLocalTree.default_sample_map(small_ts))
    with pytest.raises(ValueError, match="has no node to resolve"):
        ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                          outgroup_samples=names, focal="ingroup_mrca")
    at_root = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                                outgroup_samples=names, focal="panel_root")
    assert at_root._resolved_ingroup == ()
    assert len(list(at_root.infer())) == small_ts.num_sites


