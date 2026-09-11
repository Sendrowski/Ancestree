"""Multi-root (partial-coalescence) local-tree handling.

Time-cap an msprime simulation so coalescence is incomplete. Verify that
:class:`~ancestree.trees.TskitLocalTree` accepts an explicit ``root`` for the
multi-rooted local tree, and that :class:`~ancestree.inference.ARGBasedInference`
emits a posterior at every site by averaging across the roots' subtrees
(uniform-prior marginalisation).
"""
import logging
import multiprocessing as mp

import msprime
import numpy as np
import pytest
import tskit

from ancestree import ARGBasedInference, JC69, Site, TskitLocalTree
from ancestree.focal import ResolvedFocal
from ancestree.likelihood import Likelihood
from ancestree.trees import RerootedTree, Tree


@pytest.fixture(scope="module")
def multi_root_ts():
    """ARG with end_time=20 → very few coalescences → multi-root local trees."""
    ts = msprime.sim_ancestry(
        samples=6, sequence_length=1e4, recombination_rate=1e-7,
        population_size=1e4, end_time=20, random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-6, random_seed=1)
    return ts


class TestTskitLocalTreeWithRoot:
    def test_rejects_multi_root_without_explicit_root(self, multi_root_ts):
        ts = multi_root_ts
        tree = next(t for t in ts.trees() if t.num_roots > 1)
        with pytest.raises(ValueError, match="num_roots"):
            TskitLocalTree.from_tskit_tree(tree)

    def test_rejects_root_not_in_tree(self, multi_root_ts):
        ts = multi_root_ts
        tree = next(t for t in ts.trees() if t.num_roots > 1)
        with pytest.raises(ValueError, match="not a root"):
            TskitLocalTree.from_tskit_tree(tree, root=999_999)

    def test_explicit_root_restricts_view(self, multi_root_ts):
        ts = multi_root_ts
        tree = next(t for t in ts.trees() if t.num_roots > 1)
        for r in tree.roots:
            local = TskitLocalTree.from_tskit_tree(tree, root=int(r))
            assert local.root == int(r)
            # Postorder is only the subtree of r.
            assert int(r) in local.postorder()
            for node in local.postorder():
                assert tree.is_descendant(int(node), int(r)) or int(node) == int(r)
            # Samples in OTHER roots' subtrees are reported as missing.
            for sample_id, tip_node in local._sample_to_node.items():
                assert tree.is_descendant(int(tip_node), int(r)) or int(tip_node) == int(r)

    def test_n_tips_restricted_to_root_subtree(self, multi_root_ts):
        """``n_tips()`` counts only this view's subtree, not
        every sample in the (multi-rooted) local tree."""
        ts = multi_root_ts
        tree = next(t for t in ts.trees() if t.num_roots > 1)
        per_root = [
            TskitLocalTree.from_tskit_tree(tree, root=int(r)).n_tips()
            for r in tree.roots
        ]
        # Each restricted view sees fewer tips than the whole tree...
        assert all(n < tree.num_samples() for n in per_root)
        # ...and together they partition all samples exactly once.
        assert sum(per_root) == tree.num_samples()

    def test_draw_on_multi_root_view_does_not_crash(self, multi_root_ts):
        """``draw_text``/``draw_svg`` with a posterior render on a multi-rooted
        local-tree view without raising on ``tskit.Tree.root``."""
        from ancestree import JC69
        from ancestree.posterior import Posterior
        ts = multi_root_ts
        tree = next(t for t in ts.trees() if t.num_roots > 1)
        local = TskitLocalTree.from_tskit_tree(tree, root=int(tree.roots[0]))
        post = Posterior(alleles=JC69().states, values=np.array([0.7, 0.1, 0.1, 0.1]))
        text = local.draw_text(posterior=post)
        assert isinstance(text, str) and text
        svg = local.draw_svg(posterior=post)
        assert isinstance(svg, str) and svg


class TestARGBasedInferenceMarginalisesRoots:
    def test_emits_one_posterior_per_site(self, multi_root_ts):
        ts = multi_root_ts
        inf = ARGBasedInference(ts, JC69(), mu=1.0, progress=False)
        results = list(inf.infer())
        assert len(results) == ts.num_sites
        for _, post in results:
            assert np.isclose(post.values.sum(), 1.0, atol=1e-9)
            assert (post.values >= 0).all()

    def test_averaged_posterior_matches_manual_per_root_average(self, multi_root_ts):
        """Smoke check that the in-loop average equals an out-of-loop average."""
        ts = multi_root_ts
        # Pick a multi-root tree with at least one site.
        tree = next(t for t in ts.trees() if t.num_roots > 1 and t.num_sites > 0)
        site_id = next(iter(tree.sites())).id

        # In-loop result via the full pipeline.
        inf = ARGBasedInference(ts, JC69(), mu=1.0, progress=False)
        site_pos_to_post = {int(s.pos): p.values for s, p in inf.infer()}

        # Manual recompute: average per-root posteriors.
        from ancestree import TskitSource
        from ancestree.likelihood import Likelihood
        from ancestree.priors import StationaryPrior
        from scipy.special import logsumexp
        model = JC69()
        engine = Likelihood(model)
        prior = StationaryPrior(model)
        site = next(s for s in TskitSource(ts) if s.pos == ts.site(site_id).position)
        per_root = []
        for r in tree.roots:
            local = TskitLocalTree.from_tskit_tree(tree, root=int(r))
            log_L = engine.log_likelihoods(local, [site])[0]
            log_post = log_L + prior.log_probs([site])[0]
            log_post -= logsumexp(log_post)
            per_root.append(np.exp(log_post))
        manual = np.mean(per_root, axis=0)

        np.testing.assert_allclose(
            site_pos_to_post[int(site.pos)], manual, rtol=1e-9, atol=1e-12,
        )


class TestMultiRootReportsAtThePrincipalRoot:
    """A segment's several roots are several nodes, not several estimates.

    The emitted vector is the equal-weight mean of the per-root posteriors,
    the marginal under a uniform prior over which root is meant. That prior is
    not updated on the data, since which node a caller is asking about is a
    choice the sequence carries no information about.

    tskit reports an uncoalesced lineage as a root of its own. Such a root is
    its own ancestor, so its "ancestral" allele is just the tip's observed one,
    and it contributed a point mass on that allele at weight 1/num_roots. Those
    roots are excluded here.

    The fixture keeps the per-root posteriors away from uniform, where any
    weighting would agree.
    """

    MU = 0.3

    @staticmethod
    def _two_root_ts(n_small, n_large):
        """Two roots over disjoint samples. The small block carries ``A``.

        The site is ancestrally ``G`` with one mutation to ``A`` on the small
        root, so the small block's tips read ``A`` and the large block's ``G``.
        """
        import tskit

        n = n_small + n_large
        tables = tskit.TableCollection(sequence_length=2.0)
        for _ in range(n):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        small = tables.nodes.add_row(time=1.0)
        large = tables.nodes.add_row(time=1.0)
        for c in range(n_small):
            tables.edges.add_row(left=0, right=2, parent=small, child=c)
        for c in range(n_small, n):
            tables.edges.add_row(left=0, right=2, parent=large, child=c)
        site = tables.sites.add_row(position=1.0, ancestral_state="G")
        tables.mutations.add_row(site=site, node=small, derived_state="A")
        tables.sort()
        return tables.tree_sequence(), n

    def test_two_coalesced_roots_enter_at_equal_weight(self):
        """Both blocks coalesce, so both count, each with weight 1/2."""
        from scipy.special import logsumexp

        from ancestree.likelihood import Likelihood
        from ancestree.priors import StationaryPrior

        ts, n = self._two_root_ts(2, 8)
        sample_map = {str(i): i for i in range(n)}
        inf = ARGBasedInference(ts, JC69(), mu=self.MU,
                                sample_map=sample_map, progress=False)
        out = list(inf.infer())
        assert len(out) == 1
        site = out[0][0]

        engine, prior = Likelihood(JC69()), StationaryPrior(JC69())
        per_root = []
        for r in ts.first().roots:
            local = TskitLocalTree.from_tskit_tree(
                ts.first(), sample_map=sample_map, root=int(r))
            local.time_scale = self.MU
            log_post = (engine.log_likelihoods(local, [site])[0]
                        + prior.log_probs([site])[0])
            per_root.append(np.exp(log_post - logsumexp(log_post)))

        np.testing.assert_allclose(
            out[0][1].values, np.mean(per_root, axis=0), rtol=1e-9, atol=1e-12)

    @staticmethod
    def _tip_and_clade_ts(n_clade):
        """A sample with no parent at all, beside a coalesced clade.

        Sample 0 carries no edge, so tskit reports it as a root that is itself
        a sample tip: an uncoalesced lineage that is its own ancestor. That is
        the case the exclusion targets, and it is not the same shape as a
        unary internal root, which sits above its sample at positive time and
        whose posterior is informative.
        """
        import tskit

        n = 1 + n_clade
        tables = tskit.TableCollection(sequence_length=2.0)
        for _ in range(n):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        clade = tables.nodes.add_row(time=1.0)
        for c in range(1, n):
            tables.edges.add_row(left=0, right=2, parent=clade, child=c)
        site = tables.sites.add_row(position=1.0, ancestral_state="G")
        tables.mutations.add_row(site=site, node=0, derived_state="A")
        tables.sort()
        return tables.tree_sequence(), n

    def test_the_lone_sample_really_is_a_root(self):
        """Guard: the fixture must contain the shape the exclusion targets."""
        ts, _ = self._tip_and_clade_ts(6)
        tree = ts.first()
        tips = [r for r in tree.roots
                if tree.is_sample(r) and tree.num_children(r) == 0]
        assert tips == [0]

    def test_a_lone_uncoalesced_tip_does_not_enter_the_call(self):
        """One sample apart from a coalesced clade must not dilute it."""
        ts, n = self._tip_and_clade_ts(6)
        sample_map = {str(i): i for i in range(n)}
        inf = ARGBasedInference(ts, JC69(), mu=self.MU,
                                sample_map=sample_map, progress=False)
        post = list(inf.infer())[0][1]
        # The lone tip is its own ancestor, so its posterior is a point mass on
        # its own allele. Entering the mixture it would pull the call halfway
        # towards A; excluded, the coalesced clade decides.
        assert post.map_allele == "G"
        assert float(post["G"]) > 0.9
        assert float(post["A"]) < 0.05
        assert inf._n_uncoalesced_segments == 0

    def test_a_unary_internal_root_still_counts(self):
        """A unary root is not a lone tip: it is informative and must stay in.

        The exclusion first tested ``num_samples(r) > 1``, which also dropped a
        unary internal root, so an ARG keeping unary nodes (msprime
        ``record_full_arg`` or ``end_time``, tsinfer and SINGER output) had
        whole segments collapse to the bare prior.
        """
        import numpy as np
        import tskit

        tables = tskit.TableCollection(sequence_length=2.0)
        for _ in range(2):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        a = tables.nodes.add_row(time=1.0)
        b = tables.nodes.add_row(time=1.0)
        tables.edges.add_row(left=0, right=2, parent=a, child=0)
        tables.edges.add_row(left=0, right=2, parent=b, child=1)
        site = tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=a, derived_state="G")
        tables.mutations.add_row(site=site, node=b, derived_state="G")
        tables.sort()
        ts = tables.tree_sequence()
        tree = ts.first()
        assert all(tree.num_children(r) == 1 and not tree.is_sample(r)
                   for r in tree.roots)

        inf = ARGBasedInference(ts, JC69(), mu=0.3,
                                sample_map={str(i): i for i in range(2)},
                                progress=False)
        post = list(inf.infer())[0][1]
        assert inf._n_uncoalesced_segments == 0
        assert post.map_allele == "G"
        assert not np.allclose(post.values, 0.25)

    def test_an_uncoalesced_tree_returns_the_prior(self):
        """No coalescence anywhere means the tree constrains no ancestral node."""
        import tskit

        tables = tskit.TableCollection(sequence_length=2.0)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.sort()
        ts = tables.tree_sequence()
        sample_map = {str(i): i for i in range(4)}
        inf = ARGBasedInference(ts, JC69(), mu=self.MU,
                                sample_map=sample_map, progress=False)
        inf._quiet = True
        out = list(inf.infer())
        assert inf._n_uncoalesced_segments == 1
        for _site, post in out:
            np.testing.assert_allclose(post.values, np.full(4, 0.25),
                                       atol=1e-12)

    def test_the_uncoalesced_count_reaches_the_summary(self, caplog):
        """The count is reported to the user, as its siblings are.

        A run whose segments returned the bare prior was otherwise silent
        about it.
        """
        import logging

        import tskit

        tables = tskit.TableCollection(sequence_length=2.0)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
        tables.sites.add_row(position=1.0, ancestral_state="A")
        tables.sort()
        inf = ARGBasedInference(tables.tree_sequence(), JC69(), mu=self.MU,
                                sample_map={str(i): i for i in range(4)},
                                progress=False)
        with caplog.at_level(logging.WARNING,
                             logger="ancestree.ARGBasedInference"):
            list(inf.infer())
        assert any("1 local tree(s) carried no coalescence" in r.message
                   for r in caplog.records)
        # The summary clears the per-walk counters it reports.
        assert inf._n_uncoalesced_segments == 0


# ------------------------------------------ diagnostic counters across paths

#: Diagnostic counters every walk maintains, compared across execution paths.
COUNTER_NAMES = (
    "_n_uniform_fallback",
    "_n_ingroup_monomorphic",
    "_n_uncoalesced_segments",
    "_n_ingroup_non_monophyletic",
    "_n_focal_multiroot_fallback",
)


def _counters(inference) -> dict:
    """The five diagnostic counters of a finished walk.

    :param inference: A quiet inference whose ``infer()`` has been drained,
        so the summary has not cleared its counters.
    :return: ``{counter name: value}``.
    """
    return {name: int(getattr(inference, name)) for name in COUNTER_NAMES}


@pytest.fixture(scope="module")
def alternating_coalescence_ts():
    """Local trees alternating between a coalesced clade and no coalescence.

    One internal node parents every sample on the even-numbered unit
    intervals only, so the odd ones carry no edge and tskit reports each
    sample as a root of its own. Every interval holds one site, since the
    walk skips a tree with no site and would never reach the uncoalesced
    branch otherwise.

    :return: ``(tree sequence, number of uncoalesced local trees)``.
    """
    import tskit

    n_samples, n_windows = 4, 12
    tables = tskit.TableCollection(sequence_length=float(n_windows))
    for _ in range(n_samples):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    anc = tables.nodes.add_row(time=1.0)
    for w in range(0, n_windows, 2):
        for child in range(n_samples):
            tables.edges.add_row(left=float(w), right=float(w + 1),
                                 parent=anc, child=child)
    for w in range(n_windows):
        site = tables.sites.add_row(position=w + 0.5, ancestral_state="A")
        if w % 2 == 0:
            tables.mutations.add_row(site=site, node=0, derived_state="C")
    tables.sort()
    return tables.tree_sequence(), n_windows // 2


def _quiet_walk(ts, **kwargs):
    """A drained quiet walk over ``ts``.

    :param ts: Tree sequence, or a list of them for the marginalised path.
    :return: ``(inference, [(Site, Posterior), …])``.
    """
    inf = ARGBasedInference(
        ts, JC69(), mu=0.1, progress=False,
        sample_map={str(i): i for i in range(4)}, **kwargs,
    )
    inf._quiet = True
    return inf, list(inf.infer())


def test_the_fork_pool_reports_the_uncoalesced_count(alternating_coalescence_ts):
    """A fork-pool walk reports the same counters as the serial walk.

    The chunk worker returned four of the five counters, so a run with
    ``n_workers > 1`` reported no uncoalesced segments where the serial walk
    reported them all.
    """
    ts, n_uncoalesced = alternating_coalescence_ts
    serial, serial_out = _quiet_walk(ts, n_workers=1)
    forked, forked_out = _quiet_walk(ts, n_workers=3)
    assert serial._n_uncoalesced_segments == n_uncoalesced
    assert _counters(forked) == _counters(serial)
    assert [int(s.pos) for s, _ in forked_out] == [int(s.pos) for s, _ in serial_out]


def test_marginalising_over_one_draw_reports_the_serial_counters(
    alternating_coalescence_ts,
):
    """A one-draw posterior sample is the single-ARG walk, counters included.

    ``_infer_marginalised`` carried only two of the five counters out of its
    per-draw clones, so the uniform-fallback, ingroup-monomorphic and
    uncoalesced counts came back at zero.
    """
    ts, _ = alternating_coalescence_ts
    serial, _ = _quiet_walk(ts)
    marginal, _ = _quiet_walk([ts])
    assert serial._n_ingroup_monomorphic > 0, "fixture preconditions"
    assert _counters(marginal) == _counters(serial)


def test_marginalising_over_repeated_draws_keeps_the_site_counts(
    alternating_coalescence_ts,
):
    """Per-site counts do not multiply with the number of draws.

    Per-site counts are tallied over the sites the merge emits, so repeated
    draws of one site set give a single draw's count. The genealogy-level
    counts do add up over draws.
    """
    ts, n_uncoalesced = alternating_coalescence_ts
    serial, _ = _quiet_walk(ts)
    marginal, _ = _quiet_walk([ts, ts])
    assert marginal._n_ingroup_monomorphic == serial._n_ingroup_monomorphic
    assert marginal._n_uncoalesced_segments == 2 * n_uncoalesced


# ------------------------------------------- hand-built multi-root trees


def _mutate(tables, site_positions, node_per_site, derived="C"):
    """Add one A-to-``derived`` site per position, mutated on the given node.

    :param tables: The table collection to extend.
    :param site_positions: Site positions along the sequence.
    :param node_per_site: The node carrying the mutation at each site.
    :param derived: The derived state written at every site.
    """
    for position, node in zip(site_positions, node_per_site):
        site = tables.sites.add_row(position=position, ancestral_state="A")
        tables.mutations.add_row(site=site, node=node, derived_state=derived)


def _three_root_ts() -> tskit.TreeSequence:
    """One local tree with three roots: a clade over samples 0-2 whose
    ingroup ``{0, 1}`` coalesces at node 6, a clade over samples 3-4 under
    node 8, and sample 5 as an uncoalesced root of its own.
    """
    tables = tskit.TableCollection(sequence_length=100.0)
    tables.time_units = "generations"
    for _ in range(6):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    n6 = tables.nodes.add_row(flags=0, time=1.0)
    n7 = tables.nodes.add_row(flags=0, time=2.0)
    n8 = tables.nodes.add_row(flags=0, time=1.5)
    for parent, child in ((n6, 0), (n6, 1), (n7, n6), (n7, 2), (n8, 3), (n8, 4)):
        tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=child)
    _mutate(tables, [10.0, 20.0, 30.0], [n6, 3, 5])
    tables.sort()
    return tables.tree_sequence()


def _uncoalesced_then_coalesced_ts() -> tskit.TreeSequence:
    """Two local trees: no edge at all over ``[0, 50)``, so every sample is
    its own root there, and a fully coalesced tree over ``[50, 100)``.
    """
    tables = tskit.TableCollection(sequence_length=100.0)
    tables.time_units = "generations"
    for _ in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    n4 = tables.nodes.add_row(flags=0, time=1.0)
    n5 = tables.nodes.add_row(flags=0, time=1.5)
    n6 = tables.nodes.add_row(flags=0, time=3.0)
    for parent, child in ((n4, 0), (n4, 1), (n5, 2), (n5, 3), (n6, n4), (n6, n5)):
        tables.edges.add_row(left=50.0, right=100.0, parent=parent, child=child)
    _mutate(tables, [10.0, 20.0, 60.0, 70.0], [0, 1, n4, 2])
    tables.sort()
    return tables.tree_sequence()


class _BareTree(Tree):
    """A tree backed by nothing: no tskit tree and no focal resolution."""

    @property
    def root(self) -> int:
        return 0

    def children(self, node: int):
        return ()

    def branch_length(self, node: int) -> float:
        return 0.0

    def postorder(self):
        return (0,)

    def tip_for_sample(self, sample_id: str):
        return None

    def n_tips(self) -> int:
        return 1


class _WholeMultiRootTree(_BareTree):
    """A caller-supplied tree exposing every tip of a multi-root tskit tree."""

    def __init__(self, ts_tree):
        self._tree = ts_tree

    @property
    def tskit_tree(self):
        return self._tree

    @property
    def root(self) -> int:
        return 7

    def tip_for_sample(self, sample_id: str):
        return int(sample_id)


# ------------------------------------------------------------ focal views


def test_focal_view_refuses_a_tree_it_cannot_resolve_on(small_ts):
    """A tree with neither a focal view nor a tskit backing cannot report at
    a non-root focal node."""
    names = list(TskitLocalTree.default_sample_map(small_ts))
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                            outgroup_samples=names[-2:])
    with pytest.raises(ValueError, match="offers no way to resolve it"):
        inf._focal_view(_BareTree())


def test_tskit_focal_view_needs_an_ingroup_tip(small_ts):
    """A tskit tree naming its tips differently from the run's panel holds
    no ingroup tip to resolve the focal node against."""
    names = list(TskitLocalTree.default_sample_map(small_ts))
    inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False,
                            outgroup_samples=names[-2:])
    renamed = {f"x{n}": node for n, node in
               TskitLocalTree.default_sample_map(small_ts).items()}
    tree = TskitLocalTree.from_tskit_tree(small_ts.first(), sample_map=renamed)
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={s: "A" for s in renamed})
    with pytest.raises(ValueError, match="none of the ingroup samples are tips"):
        inf.infer_site(tree, site)


def test_tskit_focal_view_keeps_a_tree_the_ingroup_straddles():
    """An ingroup spanning several roots has no MRCA, so a caller-supplied
    tree is read at its own rooting."""
    ts = _three_root_ts()
    straddling = ARGBasedInference(ts, JC69(), mu=0.1, progress=False,
                                   ingroup_samples=["0", "3"])
    tree = _WholeMultiRootTree(ts.first())
    assert straddling._focal_view(tree) is tree


class TestMultiRootWithFocal:
    """A multi-root segment whose ingroup coalesces under one root reports
    at the ingroup MRCA of that root's subtree alone."""

    MU = 0.1

    def _manual(self, ts, site_index, prior):
        """The posterior at node 6 of the holder subtree, re-rooted there."""
        tree = ts.first()
        local = TskitLocalTree.from_tskit_tree(
            tree, sample_map=TskitLocalTree.default_sample_map(ts), root=7)
        local.time_scale = self.MU
        view = RerootedTree(local, 6, 0.0)
        site = next(s for i, (s, _) in enumerate(ARGBasedInference(
            ts, JC69(), mu=self.MU, progress=False).infer()) if i == site_index)
        log_post = Likelihood(JC69()).log_likelihoods(view, [site])[0] \
            + prior.log_probs([site])[0]
        return np.exp(log_post - np.logaddexp.reduce(log_post))

    def test_single_arg_reports_at_the_holder_subtree(self):
        ts = _three_root_ts()
        inf = ARGBasedInference(ts, JC69(), mu=self.MU, progress=False,
                                ingroup_samples=["0", "1"])
        results = list(inf.infer())
        assert len(results) == 3
        for i, (_, posterior) in enumerate(results):
            np.testing.assert_allclose(
                posterior.values, self._manual(ts, i, inf.prior), rtol=1e-10)
        # The clade-defining site pulls the ingroup MRCA toward its allele.
        assert results[0][1].map_allele == "C"

    def test_posterior_sample_of_identical_draws_matches_the_single_arg(self):
        """The draws path stashes each root's evidence into the likelihood
        rows it merges; identical draws must reproduce the single-ARG call."""
        ts = _three_root_ts()
        single = ARGBasedInference(ts, JC69(), mu=self.MU, progress=False,
                                   ingroup_samples=["0", "1"])
        draws = ARGBasedInference([ts, ts], JC69(), mu=self.MU, progress=False,
                                  ingroup_samples=["0", "1"])
        expected = [(s.pos, p.values) for s, p in single.infer()]
        got = [(s.pos, p.values) for s, p in draws.infer()]
        assert [pos for pos, _ in got] == [pos for pos, _ in expected]
        for (_, a), (_, b) in zip(got, expected):
            np.testing.assert_allclose(a, b, rtol=1e-10)
        assert draws.provenance()["parameters"]["n_draws"] == 2

    def test_root_holding_focal(self):
        tree = _three_root_ts().first()
        assert ARGBasedInference._root_holding_focal(tree, ResolvedFocal(6)) == 7
        assert ARGBasedInference._root_holding_focal(tree, ResolvedFocal(8)) == 8
        # A node under no root of this tree falls back to the first root.
        tables = tree.tree_sequence.dump_tables()
        detached_node = tables.nodes.add_row(flags=0, time=5.0)
        detached = tables.tree_sequence().first()
        assert ARGBasedInference._root_holding_focal(
            detached, ResolvedFocal(detached_node),
        ) == int(detached.roots[0])


class TestUncoalescedSegment:
    """A segment without any edge constrains no ancestral node."""

    def test_sites_take_the_prior_and_the_walk_continues(self, caplog):
        ts = _uncoalesced_then_coalesced_ts()
        inf = ARGBasedInference(ts, JC69(), mu=0.1, progress=True,
                                focal="panel_root")
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            results = list(inf.infer())
        assert [int(s.pos) for s, _ in results] == [10, 20, 60, 70]
        for _, posterior in results[:2]:
            np.testing.assert_allclose(posterior.values, 0.25)
        for _, posterior in results[2:]:
            assert not np.allclose(posterior.values, 0.25)
        assert any("carried no coalescence" in r.message for r in caplog.records)

    def test_posterior_sample_carries_the_tip_evidence(self):
        """Each bare-tip root contributes its allele's prior mass to the
        stashed evidence; identical draws reproduce the single walk."""
        ts = _uncoalesced_then_coalesced_ts()
        single = ARGBasedInference(ts, JC69(), mu=0.1, progress=False,
                                   focal="panel_root")
        draws = ARGBasedInference([ts, ts], JC69(), mu=0.1, progress=False,
                                  focal="panel_root")
        expected = [(int(s.pos), p.values) for s, p in single.infer()]
        got = [(int(s.pos), p.values) for s, p in draws.infer()]
        assert [pos for pos, _ in got] == [pos for pos, _ in expected]
        for (_, a), (_, b) in zip(got, expected):
            np.testing.assert_allclose(a, b, rtol=1e-10)


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(),
                    reason="fork start method unavailable")
def test_fork_pool_with_one_chunk_runs_inline():
    """A single-tree ARG gives one chunk, which is walked in the parent."""
    ts = _three_root_ts()
    inf = ARGBasedInference(ts, JC69(), mu=0.1, progress=False, n_workers=2,
                            focal="panel_root")
    serial = ARGBasedInference(ts, JC69(), mu=0.1, progress=False,
                               focal="panel_root")
    got = [(int(s.pos), p.values) for s, p in inf._infer_fork_pool()]
    expected = [(int(s.pos), p.values) for s, p in serial.infer()]
    assert [pos for pos, _ in got] == [pos for pos, _ in expected]
    for (_, a), (_, b) in zip(got, expected):
        np.testing.assert_array_equal(a, b)
