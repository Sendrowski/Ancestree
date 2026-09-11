"""ARGBasedInference over a posterior sample of ARGs.

``source`` accepts either a single tree sequence (or path) or an iterable of
them. An iterable is a posterior sample -- SINGER's MCMC draws, or local-tree
mode's materialised ensemble -- and the per-site posteriors are averaged over
it rather than taken from one sampled history.

Kept deliberately small: an 8-haplotype, ~30 kb panel, so the whole module runs
in seconds.
"""
import msprime
import numpy as np
import pytest
import tskit

import ancestree as anc
from ancestree.models import JC69
from ancestree.settings import Settings

MU = 2.5e-8


def _ts(seed, n=8, L=30_000):
    ts = msprime.sim_ancestry(samples=n, ploidy=1, sequence_length=L,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=seed)
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed)


def _retimed(ts, factor):
    """The same ARG over the same sites, with node times scaled.

    Independent simulations share no site positions, so they cannot stand in
    for an ARG posterior: SINGER's draws are different genealogies over one
    fixed set of sites. Rescaling the internal node times keeps the sites and
    topology fixed while changing every branch length, which is what makes the
    per-draw posteriors differ.
    """
    tables = ts.dump_tables()
    times = tables.nodes.time.copy()
    times[times > 0] *= factor
    tables.nodes.time = times
    # Mutation times are recorded too. Scale them by the same factor, or they
    # end up below their own node and tskit rejects the tables.
    mut_times = tables.mutations.time.copy()
    finite = np.isfinite(mut_times)
    mut_times[finite] *= factor
    tables.mutations.time = mut_times
    tables.sort()
    return tables.tree_sequence()


def _post(source):
    """{pos: posterior vector} from one ARGBasedInference run."""
    inf = anc.ARGBasedInference(source, anc.JC69(), mu=MU, progress=False)
    return {int(s.pos): np.asarray(p.values, dtype=float)
            for s, p in inf.infer()}


def test_single_tree_sequence_is_unchanged():
    """A bare tree sequence must not take the marginalising path."""
    ts = _ts(1)
    direct = _post(ts)
    assert direct, "expected some scored sites"
    assert all(np.isclose(v.sum(), 1.0) for v in direct.values())


def test_one_element_iterable_matches_the_single_arg():
    """Averaging over one draw is that draw."""
    ts = _ts(1)
    assert _keys_and_values_equal(_post(ts), _post([ts]))


def test_repeated_draws_match_the_single_arg():
    """Averaging identical draws changes nothing -- pins the accumulator."""
    ts = _ts(1)
    assert _keys_and_values_equal(_post(ts), _post([ts, ts, ts]))


def test_identical_draws_reproduce_the_single_draw_posterior():
    """Averaging a genealogy against itself must change nothing."""
    a = _ts(1)
    pa, both = _post(a), _post([a, a, a])
    shared = set(pa) & set(both)
    assert shared, "the draws should share sites"
    for pos in shared:
        assert np.allclose(both[pos], pa[pos], atol=1e-12)


def test_draws_are_combined_in_likelihood_space():
    """The marginal averages LIKELIHOODS, not normalised posteriors.

    P(x | s) ~= (1/M) sum_m P(x | s, G_m), with the prior applied once, so a
    draw that explains the site poorly contributes less than one that explains
    it well. Averaging the normalised posteriors would weight the two equally,
    which is only right for exact posterior samples; MCMC and HMM draws are
    not that. Pinned because the two agree whenever the draws agree, so only a
    site where they disagree separates them.
    """
    a = _ts(1)
    b = _retimed(a, 1.7)
    pa, pb, both = _post(a), _post(b), _post([a, b])
    shared = set(pa) & set(pb) & set(both)
    assert shared, "the two draws should share sites"
    disagree = [k for k in shared if not np.allclose(pa[k], pb[k], atol=1e-6)]
    assert disagree, "the draws must actually disagree, or the test proves nothing"
    for pos in shared:
        row = both[pos]
        assert np.all(row >= 0.0) and np.isclose(row.sum(), 1.0)
        # Bounded by the two draws state by state: a weighted average of them
        # cannot leave the interval either endpoint defines.
        lo = np.minimum(pa[pos], pb[pos]) - 1e-9
        hi = np.maximum(pa[pos], pb[pos]) + 1e-9
        assert np.all((row >= lo) & (row <= hi)), (pos, row, pa[pos], pb[pos])
    # On at least one disagreeing site the likelihood average must differ from
    # the plain posterior mean, which is what distinguishes the estimators.
    assert any(
        not np.allclose(both[k], 0.5 * (pa[k] + pb[k]), atol=1e-9)
        for k in disagree
    ), "indistinguishable from averaging normalised posteriors"


def test_a_site_missing_from_one_draw_averages_over_the_rest():
    """Sites are averaged over the draws that carry them, not over all draws.

    Otherwise a tool whose ARG stops short would drag every other site's
    posterior toward uniform.
    """
    a, b = _ts(1), _ts(2)  # independent sims: their site sets barely overlap
    pa, pb, both = _post(a), _post(b), _post([a, b])
    only_a = set(pa) - set(pb)
    assert only_a, "the two draws happen to share every site"
    pos = next(iter(only_a))
    assert np.allclose(both[pos], pa[pos], atol=1e-12)


def test_generator_input_is_accepted():
    """A one-shot generator works: draws are consumed once, lazily."""
    ts = _ts(1)
    assert _keys_and_values_equal(_post(ts), _post(t for t in [ts]))


def test_empty_iterable_is_rejected():
    with pytest.raises(ValueError, match="empty iterable"):
        anc.ARGBasedInference([], anc.JC69(), mu=MU, progress=False)


def _keys_and_values_equal(x, y):
    if set(x) != set(y):
        return False
    return all(np.allclose(x[k], y[k], atol=1e-12) for k in x)


class TestOutgroupNamesAreValidated:
    """An unmatched outgroup name is refused, not dropped.

    Dropped, the ingroup would become the whole panel and
    focal="ingroup_mrca" would resolve to the panel root, a different
    estimand.
    """

    @staticmethod
    def _ts():
        import msprime

        return msprime.sim_ancestry(
            samples=4, ploidy=1, sequence_length=100,
            population_size=1e4, random_seed=1,
        )

    def test_a_mistyped_outgroup_raises(self):
        with pytest.raises(ValueError, match="outgroup sample"):
            anc.ARGBasedInference(
                self._ts(), JC69(), mu=1e-8,
                outgroup_samples=["O_TYPO"], progress=False,
            )._baseline_ingroup_samples()

    def test_a_real_outgroup_is_excluded_from_the_ingroup(self):
        inference = anc.ARGBasedInference(
            self._ts(), JC69(), mu=1e-8,
            outgroup_samples=["3"], progress=False,
        )
        assert inference._baseline_ingroup_samples() == ("0", "1", "2")


class TestMultiRootDrawsPath:
    """An ARG passed as a posterior sample scores like the ARG itself.

    For a multi-root segment the kernel runs per root and averages the
    normalised posteriors, and the draws path must see every root's
    contribution. A worker pool stashes into its own address space, so the
    result must not depend on n_workers either.
    """

    @staticmethod
    def _decapitated():
        import msprime

        ts = msprime.sim_ancestry(
            6, sequence_length=1e4, recombination_rate=1e-8,
            population_size=1e4, random_seed=7,
        )
        ts = msprime.sim_mutations(
            ts, rate=5e-7, model=msprime.JC69(), random_seed=3)
        return ts.decapitate(1.0e4)

    @staticmethod
    def _posteriors(source, **kwargs):
        inference = anc.ARGBasedInference(
            source, JC69(), mu=1e-8, progress=False, **kwargs)
        return {float(s.pos): p.values.copy() for s, p in inference.infer()}

    def test_a_one_element_sample_matches_the_bare_arg(self):
        ts = self._decapitated()
        assert any(t.num_roots > 1 for t in ts.trees())
        bare = self._posteriors(ts)
        as_draws = self._posteriors([ts])
        assert set(bare) == set(as_draws)
        for key in bare:
            np.testing.assert_allclose(
                as_draws[key], bare[key], rtol=0, atol=1e-12)

    def test_the_draws_path_does_not_depend_on_worker_count(self):
        ts = self._decapitated()
        serial = self._posteriors([ts])
        parallel = self._posteriors([ts], n_workers=2)
        for key in serial:
            np.testing.assert_allclose(
                parallel[key], serial[key], rtol=0, atol=1e-12)


SAMPLES = 8
INGROUP = tuple(f"s{i}" for i in range(6))
OUTGROUP = ("s6", "s7")
SAMPLE_MAP = {f"s{i}": i for i in range(SAMPLES)}


@pytest.fixture
def arg():
    ts = msprime.sim_ancestry(samples=SAMPLES, sequence_length=100_000, ploidy=1,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=3)
    return msprime.sim_mutations(ts, rate=1e-7, random_seed=3)


@pytest.fixture
def restore_parallelize():
    before = Settings.parallelize
    yield
    Settings.parallelize = before


class TestMarginalisationIgnoresTheParallelismSwitch:
    """The ARG-draw combine happens in likelihood space, not posterior space.

    Settings.parallelize promotes a worker count of 1 to the CPU count, and a
    forked worker fills its own copy of the per-site likelihood sink, so the
    performance switch must not change the estimator.
    """

    @staticmethod
    def _draws(ts):
        for factor in (1.0, 0.05, 20.0):
            t = ts.dump_tables()
            t.nodes.time = ts.tables.nodes.time * factor
            t.mutations.time = np.full(len(t.mutations), tskit.UNKNOWN_TIME)
            t.sort()
            t.build_index()
            t.compute_mutation_parents()
            yield t.tree_sequence()

    def _posteriors(self, ts):
        inf = anc.ARGBasedInference(
            list(self._draws(ts)), JC69(), mu=4e-6, sample_map=SAMPLE_MAP,
            ingroup_samples=INGROUP, outgroup_samples=OUTGROUP, n_workers=1)
        return {int(s.pos): np.asarray(p.values) for s, p in inf.infer()}

    def test_the_estimator_is_the_same_with_the_switch_on(
            self, arg, restore_parallelize):
        Settings.parallelize = None
        serial = self._posteriors(arg)
        Settings.parallelize = True
        parallel = self._posteriors(arg)
        shared = sorted(set(serial) & set(parallel))
        assert shared, "no sites scored"
        worst = max(float(np.abs(serial[k] - parallel[k]).max()) for k in shared)
        assert worst == pytest.approx(0.0, abs=1e-12), (
            f"parallelize changed the posterior at {len(shared)} sites, "
            f"worst {worst:.3e}")


def test_marginalising_over_draws_ignores_the_worker_count():
    """The estimator must not change with a performance setting.

    Marginalising over an ARG posterior averages per-draw likelihoods, with
    the prior applied once, and a fork pool filling its own copy of the
    likelihood sink must give the same result.
    """
    import msprime
    import numpy as np

    from ancestree import ARGBasedInference, JC69

    base = msprime.sim_ancestry(
        6, ploidy=1, sequence_length=50_000, recombination_rate=1e-8,
        population_size=1e4, random_seed=11)
    draws = []
    for factor in (1.0, 1.6, 0.7):
        tables = base.dump_tables()
        tables.nodes.time = tables.nodes.time * factor
        tables.sort()
        draws.append(msprime.sim_mutations(
            tables.tree_sequence(), rate=1e-7, random_seed=11))

    def posteriors(n_workers):
        inference = ARGBasedInference(
            draws, JC69(), mu=2.5e-8, progress=False, n_workers=n_workers)
        return {int(s.pos): np.asarray(p.values) for s, p in inference.infer()}

    serial, pooled = posteriors(1), posteriors(4)
    assert set(serial) == set(pooled) and serial
    for pos, values in serial.items():
        np.testing.assert_array_equal(values, pooled[pos])


class TestTheDrawsAreWalkedSiteSynchronously:
    """Draws are merged on site position, not concatenated draw by draw.

    A site closes once every draw's walk has passed it, so the accumulator
    holds the positional skew between the walks rather than the genome, and
    the per-draw likelihood sink holds the local tree being walked rather
    than every site scored so far.
    """

    @staticmethod
    def _draws():
        base = _ts(5, n=8, L=40_000)
        return [base, _retimed(base, 1.7), _retimed(base, 0.4)]

    @staticmethod
    def _spy(monkeypatch, record):
        original = anc.inference.Inference._stash_log_L

        def stash(self, sites, log_L):
            original(self, sites, log_L)
            record.append((len(sites), len(self._log_L_sink)))

        monkeypatch.setattr(anc.inference.Inference, "_stash_log_L", stash)

    def test_the_sink_holds_one_local_tree_at_a_time(self, monkeypatch):
        """Rows are taken out of the sink, so it cannot grow to the genome."""
        record: list = []
        self._spy(monkeypatch, record)
        draws = self._draws()
        inference = anc.ARGBasedInference(draws, anc.JC69(), mu=MU,
                                          progress=False)
        pairs = list(inference.infer())
        assert pairs, "expected some scored sites"
        widest_tree = max(n for n, _ in record)
        deepest_sink = max(held for _, held in record)
        assert deepest_sink <= widest_tree, (
            f"the sink reached {deepest_sink} rows where the widest local "
            f"tree carries {widest_tree} sites")
        assert sum(d.num_sites for d in draws) > 10 * widest_tree, (
            "fixture preconditions: the genome must be many trees wide")

    def test_a_site_is_emitted_before_the_draws_are_fully_walked(
            self, monkeypatch):
        """The first site closes after a few local trees, not after M genomes."""
        record: list = []
        self._spy(monkeypatch, record)
        draws = self._draws()
        total_sites = sum(d.num_sites for d in draws)
        inference = anc.ARGBasedInference(draws, anc.JC69(), mu=MU,
                                          progress=False)
        stream = inference.infer()
        next(stream)
        scored_at_first_emission = sum(n for n, _ in record)
        assert scored_at_first_emission < total_sites / 4, (
            f"{scored_at_first_emission} of {total_sites} draw-sites were "
            f"scored before the first site was emitted")
        list(stream)
        assert sum(n for n, _ in record) >= total_sites


def test_a_site_is_averaged_over_exactly_the_draws_that_carry_it():
    """Restricting the sample to the draws carrying a site does not move it.

    The mixture is uniform over the draws that carry the site, so dropping a
    draw that lacks it must reproduce the same posterior bit for bit.
    """
    base = _ts(6, n=8, L=40_000)
    positions = sorted(s.position for s in base.sites())
    n = len(positions)
    assert n > 20, "fixture preconditions"

    def subset(ts, keep):
        tables = ts.dump_tables()
        tables.delete_sites(
            [s.id for s in ts.sites() if s.position not in keep])
        tables.sort()
        return tables.tree_sequence()

    draws = [
        subset(base, set(positions[:int(0.8 * n)])),
        subset(_retimed(base, 1.9), set(positions[int(0.1 * n):])),
        subset(_retimed(base, 0.35), set(positions[::2])),
    ]
    carried = [{s.position for s in d.sites()} for d in draws]
    assert len({frozenset(c) for c in carried}) == 3, "fixture preconditions"

    def run(source):
        inference = anc.ARGBasedInference(source, anc.JC69(), mu=MU,
                                          progress=False)
        return {s.local_tree_handle: np.asarray(p.values, dtype=float)
                for s, p in inference.infer()}

    full = run(draws)
    union = set().union(*carried)
    assert set(full) == {float(p) for p in union}
    partial = [p for p in union if not all(p in c for c in carried)]
    assert partial, "fixture preconditions: some site must be missing a draw"
    for pos in partial:
        holders = [d for d, c in zip(draws, carried) if pos in c]
        np.testing.assert_array_equal(
            full[float(pos)], run(holders)[float(pos)])


def test_a_one_shot_generator_source_refuses_a_second_pass():
    """The draws of a generator source are spent by the first walk."""
    ts = _ts(1)
    inference = anc.ARGBasedInference(
        (t for t in [ts, _retimed(ts, 1.4)]), anc.JC69(), mu=MU,
        progress=False)
    assert list(inference.infer())
    with pytest.raises(ValueError, match="one-shot iterator"):
        list(inference.infer())


class TestTheMixtureIsTheAverageLikelihood:
    """The marginalised posterior is pinned to a hand-computed mixture.

    Every other assertion on this path brackets the mixture between the
    individual draws, which holds for any positive weights, so none of them
    distinguishes an equal-weight average from a reweighting that returns one
    draw. Reverting the per-draw evidence offset, or dropping the accumulator
    so only the last draw survives, leaves them all passing.
    """

    @staticmethod
    def _decapitated(ts, time):
        """``ts`` with the edges above ``time`` removed, giving a forest."""
        return ts.decapitate(time)

    @staticmethod
    def _single_root_log_L(ts, model, mu, positions):
        """Each site's log likelihood row, scored independently of the walk.

        Reading the row the walk stashes would make the comparison
        self-referential: reverting the per-draw evidence offset would move
        the expectation with it. Restricted to single-root draws, where the
        row is one Felsenstein call and needs no forest arithmetic.
        """
        from ancestree.likelihood import Likelihood
        from ancestree.sources import TskitSource
        from ancestree.trees import TskitLocalTree

        engine = Likelihood(model)
        sample_map = TskitLocalTree.default_sample_map(ts)
        by_pos = {int(s.pos): s for s in TskitSource(ts, sample_map=sample_map)}
        rows = {}
        for pos in sorted(positions & set(by_pos)):
            tree = ts.at(pos)
            assert tree.num_roots == 1, "helper assumes a coalesced draw"
            local = TskitLocalTree.from_tskit_tree(tree, sample_map=sample_map)
            local.time_scale = mu
            rows[pos] = engine.log_likelihoods(local, [by_pos[pos]])[0]
        return rows, by_pos

    def test_the_mixture_equals_the_average_of_the_draw_likelihoods(self):
        """Two coalesced draws of the same sites, differing in branch length.

        Pins the mixing arithmetic itself: equal weights and the division by
        the number of draws carrying the site. The evidence scale that puts a
        multi-root draw on the same footing is pinned separately below.
        """
        from scipy.special import logsumexp

        full = _ts(seed=11, n=6, L=20_000)
        assert full.num_sites > 5
        other = _retimed(full, 2.5)
        model, mu = JC69(), 5e-8

        inference = anc.ARGBasedInference(
            [full, other], model, mu=mu, progress=False)
        mixed = {s.pos: p.values for s, p in inference.infer()}
        positions = set(mixed)
        rows_a, by_pos = self._single_root_log_L(full, model, mu, positions)
        rows_b, _ = self._single_root_log_L(other, model, mu, positions)

        shared = sorted(set(rows_a) & set(rows_b))
        assert len(shared) > 3
        differing = 0
        for pos in shared:
            stacked = np.vstack([rows_a[pos], rows_b[pos]])
            log_mean = logsumexp(stacked, axis=0) - np.log(2)
            log_post = log_mean + inference.prior.log_probs([by_pos[pos]])[0]
            want = np.exp(log_post - logsumexp(log_post))
            np.testing.assert_allclose(mixed[pos], want, atol=1e-10,
                                       err_msg=f"mixture moved at {pos}")
            differing += not np.allclose(rows_a[pos], rows_b[pos])
        # The draws must actually disagree, or any weighting would pass.
        assert differing > 3

    def test_a_multi_root_draw_carries_its_own_evidence_scale(self):
        """A draw's stashed row implies its marginal likelihood, not 1.

        ``sum_s pi_s exp(row_s)`` is ``P(data | draw)``. A multi-root branch
        stashing a normalised posterior would make that exactly 1 while a
        coalesced draw's is orders of magnitude smaller, so mixing the two in
        likelihood space would return the multi-root draw alone.
        """
        from scipy.special import logsumexp

        full = _ts(seed=11, n=6, L=20_000)
        cut = self._decapitated(full, 2_000)
        assert any(t.num_roots > 1 for t in cut.trees())

        inference = anc.ARGBasedInference(cut, JC69(), mu=5e-8, progress=False)
        inference._log_L_sink = sink = {}
        log_pi = np.log(np.full(4, 0.25))
        checked = 0
        for site, _ in inference.infer():
            key = (site.local_tree_handle if site.local_tree_handle is not None
                   else site.pos)
            row = sink.pop(key, None)
            if row is None:
                continue
            z = float(logsumexp(row + log_pi))
            assert not np.isclose(z, 0.0, atol=1e-9), (
                f"draw implies P(data)=1 at {site.pos}: the row is a "
                f"normalised posterior, not a likelihood")
            checked += 1
        assert checked > 3

    def test_a_site_carried_by_one_draw_is_that_draw_alone(self):
        """A ragged draw set averages each site over the draws holding it."""
        full = _ts(seed=13, n=6, L=20_000)
        tables = full.dump_tables()
        keep = [i for i, p in enumerate(tables.sites.position)
                if i % 2 == 0]
        tables.delete_sites([i for i in range(full.num_sites)
                             if i not in keep])
        thinned = tables.tree_sequence()
        assert thinned.num_sites < full.num_sites

        alone = {s.pos: p.values for s, p in anc.ARGBasedInference(
            full, JC69(), mu=5e-8, progress=False).infer()}
        mixed = {s.pos: p.values for s, p in anc.ARGBasedInference(
            [full, thinned], JC69(), mu=5e-8, progress=False).infer()}
        assert set(mixed) == set(alone)
        only_full = set(alone) - {int(p) for p in thinned.sites_position}
        assert only_full
        for pos in only_full:
            np.testing.assert_allclose(mixed[pos], alone[pos], atol=1e-12)
