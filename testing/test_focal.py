"""The focal node: resolution, re-rooting, and the edge split.

The kernel gets the marginal at an interior node by re-rooting there, which is
only legitimate because the substitution models are time-reversible. These tests
check that claim two independent ways: against brute-force marginalisation over
every internal-node assignment, and against an inside-outside pass written
separately (the pre-order recursion, rather than a second rooted pruning).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import tskit

import ancestree as anc
from ancestree.priors import KingmanIngroupWeight
import ancestree._jit_kernel as jk
from ancestree.focal import FocalNode, ResolvedFocal
from ancestree.models import JC69, K2, F81, HKY, GTR
from ancestree.sites import BaseComposition
from testing._helpers import canonical_alleles

#: Skewed enough that a non-reversible slip shows up in the third decimal.
COMPOSITION = BaseComposition(counts={"A": 400, "C": 100, "G": 100, "T": 400})

#: Branch lengths are in generations. This scales them to subs/site.
TIME_SCALE = 0.05

#: ``pi`` is the raw base-composition vector the models take. The
#: :class:`BaseComposition` itself goes to the :class:`Likelihood`.
MODELS = [
    (JC69(), None), (K2(kappa=2.5), None), (F81(), COMPOSITION.pi),
    (HKY(kappa=2.7), COMPOSITION.pi), (GTR(), COMPOSITION.pi),
]


def _panel():
    """Six tips: an ingroup of four over two clades, plus two outgroups.

    Deliberately non-ultrametric, and built as a table collection rather than
    through ``from_newick``, which would force ultrametricity and silently
    rewrite these branch lengths.
    """
    tables = tskit.TableCollection(sequence_length=10.0)
    a, b, c, d = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                  for _ in range(4)]
    o1, o2 = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
              for _ in range(2)]
    ab = tables.nodes.add_row(flags=0, time=1.0)
    cd = tables.nodes.add_row(flags=0, time=1.5)
    ingroup_mrca = tables.nodes.add_row(flags=0, time=3.0)
    outgroup_mrca = tables.nodes.add_row(flags=0, time=2.0)
    root = tables.nodes.add_row(flags=0, time=6.0)
    for parent, child in (
        (ab, a), (ab, b), (cd, c), (cd, d), (ingroup_mrca, ab),
        (ingroup_mrca, cd), (outgroup_mrca, o1), (outgroup_mrca, o2),
        (root, ingroup_mrca), (root, outgroup_mrca),
    ):
        tables.edges.add_row(left=0, right=10.0, parent=parent, child=child)
    tables.sort()
    return tables.tree_sequence(), [a, b, c, d], ingroup_mrca


TREE_SEQUENCE, INGROUP_NODES, INGROUP_MRCA = _panel()
#: One G among the ingroup, both outgroups T: informative at every node.
OBSERVED = {0: "A", 1: "A", 2: "G", 3: "A", 4: "T", 5: "T"}
STATE_INDEX = {s: i for i, s in enumerate(anc.STATES)}


def _tip_states() -> np.ndarray:
    """The one-site batch the native kernel consumes."""
    return np.array(
        [[STATE_INDEX[OBSERVED[int(n)]] for n in TREE_SEQUENCE.samples()]],
        dtype=np.int8,
    )


def _posterior(model, pi, focal: ResolvedFocal | None) -> np.ndarray:
    """Normalised posterior at ``focal`` from the native kernel."""
    likelihood = anc.Likelihood(
        model, base_composition=(None if pi is None else COMPOSITION),
    )
    log_l = np.asarray(likelihood.log_likelihoods_tskit_native(
        TREE_SEQUENCE.first(), _tip_states(),
        np.asarray(TREE_SEQUENCE.samples(), dtype=np.int32),
        TREE_SEQUENCE.tables.nodes.time, TIME_SCALE, focal=focal,
    )).ravel()
    weighted = np.exp(log_l - log_l.max()) * model.stationary(pi=pi)
    return weighted / weighted.sum()


def _edges_with_split(tree, node: int, tau: float):
    """The panel's edges, with the branch above ``node`` cut at ``tau``.

    :param tree: The tree to read.
    :param node: Node below the cut.
    :param tau: Distance above ``node`` at which to cut, in the tree's own
        branch units. Zero leaves the edge whole.
    :return: ``(edges, split)`` where ``edges`` are ``(parent, child, length)``
        triples and ``split`` is the id of the node inserted at ``tau``, or
        ``None`` when no cut was made.
    """
    edges = [(int(tree.parent(u)), int(u), float(tree.branch_length(u)))
             for u in tree.nodes() if tree.parent(u) != tskit.NULL]
    if tau == 0.0:
        return edges, None
    split = max(int(u) for u in tree.nodes()) + 1
    parent = int(tree.parent(node))
    length = float(tree.branch_length(node))
    edges = [e for e in edges if e[1] != int(node)]
    edges += [(split, int(node), tau), (parent, split, length - tau)]
    return edges, split


def _brute_force(model, pi, focal_node: int, tau: float = 0.0) -> np.ndarray:
    """P(focal = s) by summing over every internal-node state assignment.

    :param focal_node: Node the marginal is taken at, or the node below the
        readout point when ``tau`` is positive.
    :param tau: Distance above ``focal_node``, in the tree's own branch units.
        A positive value cuts the branch there and marginalises at the cut,
        which enumerates one extra node.
    """
    tree = TREE_SEQUENCE.first()
    root = int(tree.root)
    edges, split = _edges_with_split(tree, focal_node, tau)
    tips = {int(u) for u in TREE_SEQUENCE.samples()}
    internals = sorted(({root} | {c for _p, c, _l in edges}) - tips)
    probs = {(p, c): model.transition_probs(length * TIME_SCALE,
                                            pi=pi)
             for p, c, length in edges}
    stationary = model.stationary(pi=pi)
    out = np.zeros(len(anc.STATES))
    for assignment in itertools.product(range(len(anc.STATES)),
                                        repeat=len(internals)):
        state = dict(zip(internals, assignment))
        joint = stationary[state[root]]
        for parent, child, _length in edges:
            below = state.get(child, STATE_INDEX.get(OBSERVED.get(child, "")))
            joint *= probs[parent, child][state[parent], below]
        out[state[focal_node if split is None else split]] += joint
    return out / out.sum()


def _inside_outside(model, pi, focal_node: int) -> np.ndarray:
    """P(focal = s) from a pre-order outside pass times the inside partials.

    An independent derivation of the same quantity: the kernel re-roots, this
    propagates a message downward from the root instead. Ported from the
    prototype in the RIKEN talk's figure script.
    """
    tree = TREE_SEQUENCE.first()
    n_states = len(anc.STATES)

    def transition(node):
        return model.transition_probs(
            tree.branch_length(node) * TIME_SCALE, pi=pi,
        )

    inside: dict[int, np.ndarray] = {}

    def visit_up(node):
        if tree.num_children(node) == 0:
            vector = np.zeros(n_states)
            vector[STATE_INDEX[OBSERVED[int(node)]]] = 1.0
        else:
            vector = np.ones(n_states)
            for child in tree.children(node):
                vector = vector * (transition(child) @ visit_up(child))
        inside[int(node)] = vector
        return vector

    visit_up(tree.root)

    outside = {int(tree.root): model.stationary(pi=pi).copy()}

    def visit_down(node):
        for child in tree.children(node):
            message = outside[int(node)].copy()
            for sibling in tree.children(node):
                if sibling != child:
                    message = message * (transition(sibling) @ inside[int(sibling)])
            outside[int(child)] = transition(child).T @ message
            visit_down(child)

    visit_down(tree.root)
    joint = outside[focal_node] * inside[focal_node]
    return joint / joint.sum()


class TestResolution:
    """`FocalNode` names a node as a rule, resolved per tree."""

    def test_ingroup_mrca_differs_from_the_panel_root(self):
        tree = TREE_SEQUENCE.first()
        resolved = FocalNode("ingroup_mrca").resolve(
            tree, ingroup_nodes=INGROUP_NODES,
        )
        assert resolved.node == INGROUP_MRCA
        assert resolved.node != tree.root
        assert resolved.ingroup_is_monophyletic is True

    def test_panel_root_anchor_needs_no_ingroup(self):
        tree = TREE_SEQUENCE.first()
        assert FocalNode("panel_root").resolve(tree).node == tree.root

    def test_ingroup_anchor_without_an_ingroup_raises(self):
        with pytest.raises(ValueError, match="needs ingroup_nodes"):
            FocalNode("ingroup_mrca").resolve(TREE_SEQUENCE.first())

    def test_postorder_ranks_agree_with_the_pairwise_fold(self):
        tree = TREE_SEQUENCE.first()
        ranks = {int(n): i for i, n in enumerate(tree.postorder())}
        with_ranks = FocalNode("ingroup_mrca").resolve(
            tree, ingroup_nodes=INGROUP_NODES, postorder_rank=ranks,
        )
        without = FocalNode("ingroup_mrca").resolve(
            tree, ingroup_nodes=INGROUP_NODES,
        )
        assert with_ranks.node == without.node

    def test_rejects_a_bad_spec(self):
        with pytest.raises(ValueError, match="unknown focal anchor"):
            FocalNode("nonsense")
        with pytest.raises(ValueError, match="at most one placement"):
            FocalNode("panel_root", fraction=0.5, depth=1.0)
        with pytest.raises(ValueError, match=r"fraction must be in \[0, 1\]"):
            FocalNode("panel_root", fraction=1.5)


class TestRerootingIsExact:
    """Re-rooting must reproduce the marginal, not approximate it."""

    @pytest.mark.parametrize("model,pi", MODELS)
    @pytest.mark.parametrize("at_ingroup", [True, False])
    def test_matches_brute_force(self, model, pi, at_ingroup):
        tree = TREE_SEQUENCE.first()
        node = INGROUP_MRCA if at_ingroup else int(tree.root)
        got = _posterior(model, pi, ResolvedFocal(node, 0.0))
        assert got == pytest.approx(_brute_force(model, pi, node), abs=1e-12)

    @pytest.mark.parametrize("model,pi", MODELS)
    @pytest.mark.parametrize("tau_fraction", [0.25, 0.5, 0.75])
    def test_matches_brute_force_inside_an_edge(self, model, pi, tau_fraction):
        """The absolute check at a point the kernel has to synthesise.

        A readout strictly inside an edge is the one placement no node-anchored
        comparison reaches: the kernel builds a split node carrying ``tau``
        below and ``L - tau`` above. Endpoint agreement and monotonicity are
        both invariant to a reparametrisation of ``tau`` that fixes 0 and
        ``L``, so the marginal is enumerated at the cut instead.
        """
        tree = TREE_SEQUENCE.first()
        tau = tau_fraction * tree.branch_length(INGROUP_MRCA)
        got = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, tau))
        assert got == pytest.approx(
            _brute_force(model, pi, INGROUP_MRCA, tau), abs=1e-12,
        )

    @pytest.mark.parametrize("model,pi", MODELS)
    def test_matches_an_independent_inside_outside_pass(self, model, pi):
        got = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, 0.0))
        want = _inside_outside(model, pi, INGROUP_MRCA)
        assert got == pytest.approx(want, abs=1e-12)

    @pytest.mark.parametrize("model,pi", MODELS)
    def test_focal_at_the_root_is_the_unrooted_default(self, model, pi):
        tree = TREE_SEQUENCE.first()
        at_root = _posterior(model, pi, ResolvedFocal(int(tree.root), 0.0))
        default = _posterior(model, pi, None)
        assert at_root == pytest.approx(default, abs=0.0)


class TestEdgeSplit:
    """A focal point part-way along an edge, and its two endpoints."""

    @staticmethod
    def _edge_above_the_ingroup() -> tuple[float, int]:
        tree = TREE_SEQUENCE.first()
        # ``tau`` is in the tree's own branch units, as ``FocalNode`` emits it;
        # the kernel applies ``time_scale`` itself.
        return (tree.branch_length(INGROUP_MRCA), int(tree.parent(INGROUP_MRCA)))

    @pytest.mark.parametrize("model,pi", MODELS)
    def test_tau_zero_is_the_node_itself(self, model, pi):
        """Splitting an edge at zero must not move the reporting position.

        The endpoint that pins this is the other one: the same walk at the
        full edge length lands on the parent, so tau=0 must differ from it by
        the whole edge rather than being an alias for it.
        """
        length, parent = self._edge_above_the_ingroup()
        at_node = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, 0.0))
        at_parent = _posterior(model, pi, ResolvedFocal(parent, 0.0))
        assert length > 0.0
        assert at_node != pytest.approx(at_parent, abs=1e-9), (
            "tau=0 gives the parent's posterior, so the split is not anchored "
            "at the node")
        walked = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, length))
        assert walked == pytest.approx(at_parent, abs=1e-12)

    @pytest.mark.parametrize("model,pi", MODELS)
    def test_tau_at_the_full_edge_is_the_parent_node(self, model, pi):
        length, parent = self._edge_above_the_ingroup()
        walked = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, length))
        assert walked == pytest.approx(
            _posterior(model, pi, ResolvedFocal(parent, 0.0)), abs=1e-12,
        )

    def test_interpolates_monotonically_between_the_endpoints(self):
        length, parent = self._edge_above_the_ingroup()
        model, pi = JC69(), None
        below = _posterior(model, pi, ResolvedFocal(INGROUP_MRCA, 0.0))
        above = _posterior(model, pi, ResolvedFocal(parent, 0.0))
        previous = below[STATE_INDEX["A"]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            current = _posterior(
                model, pi, ResolvedFocal(INGROUP_MRCA, fraction * length),
            )[STATE_INDEX["A"]]
            assert current < previous
            previous = current
        assert previous == pytest.approx(above[STATE_INDEX["A"]], abs=1e-12)


class TestPlacementForms:
    """The three ways of placing the readout point above the anchor."""

    @staticmethod
    def _resolve(**kwargs):
        return FocalNode("ingroup_mrca", **kwargs).resolve(
            TREE_SEQUENCE.first(), ingroup_nodes=INGROUP_NODES,
        )

    def test_at_most_one_placement(self):
        for pair in itertools.combinations(
            ({"fraction": 0.5}, {"coalescences": 1},
             {"depth": 1.0}), 2,
        ):
            with pytest.raises(ValueError, match="at most one placement"):
                FocalNode("ingroup_mrca", **{**pair[0], **pair[1]})

    def test_negative_placements_are_rejected(self):
        with pytest.raises(ValueError, match="coalescences must be"):
            FocalNode("ingroup_mrca", coalescences=-1)
        with pytest.raises(ValueError, match="depth must be"):
            FocalNode("ingroup_mrca", depth=-1.0)

    def test_coalescences_land_on_a_node(self):
        resolved = self._resolve(coalescences=1)
        assert resolved.node == int(TREE_SEQUENCE.first().parent(INGROUP_MRCA))
        assert resolved.tau == 0.0

    def test_coalescences_clamp_at_the_root(self):
        root = int(TREE_SEQUENCE.first().root)
        assert self._resolve(coalescences=99).node == root

    def test_depth_is_absolute_and_bounded_below_by_the_anchor(self):
        tree = TREE_SEQUENCE.first()
        anchor_time = float(tree.time(INGROUP_MRCA))
        # Shallower than the ingroup MRCA: clamped to it.
        shallow = self._resolve(depth=0.5 * anchor_time)
        assert (shallow.node, shallow.tau) == (INGROUP_MRCA, 0.0)
        # Between the MRCA and the root: lands exactly at that depth.
        want = anchor_time + 0.5 * float(tree.branch_length(INGROUP_MRCA))
        landed = self._resolve(depth=want)
        assert float(tree.time(landed.node)) + landed.tau == pytest.approx(want)

    def test_depth_beyond_the_root_extends_above_it(self):
        tree = TREE_SEQUENCE.first()
        beyond = float(tree.time(tree.root)) + 5.0
        resolved = self._resolve(depth=beyond)
        assert resolved.node == int(tree.root)
        assert resolved.tau == pytest.approx(5.0)

    def test_the_ladder_measures_depth_from_its_own_root(self):
        ladder = anc.OutgroupLadderTree(["i1", "i2"], ["o1", "o2"])
        ladder.set_params([0.01, 0.02, 0.03])
        # The backbone is a single edge of length K_0. Half way along it the
        # view is re-rooted, at the ends it is the ladder or the deepest join.
        assert ladder.at_focal(FocalNode("ingroup_mrca", depth=0.0)) is ladder
        mid = ladder.at_focal(FocalNode("ingroup_mrca", depth=0.005))
        assert mid.root != ladder.root

    def test_the_ladder_accepts_coalescences(self):
        ladder = anc.OutgroupLadderTree(["i1", "i2"], ["o1", "o2", "o3"])
        ladder.set_params([0.01, 0.02, 0.03, 0.04, 0.05])
        views = [ladder.at_focal(FocalNode("ingroup_mrca", coalescences=k))
                 for k in (0, 1, 2)]
        assert len({v.root for v in views}) == 3
        # Past the last join the readout lands on the panel's own ancestor.
        assert (ladder.at_focal(FocalNode("ingroup_mrca", coalescences=99)).root
                == ladder.at_focal(FocalNode("panel_root")).root)

    def test_the_ladder_endpoints_are_the_two_anchors(self):
        ladder = anc.OutgroupLadderTree(["i1", "i2"], ["o1", "o2", "o3"])
        ladder.set_params([0.01, 0.02, 0.03, 0.04, 0.05])
        deep = ladder.as_deep_rooted()
        at_root = ladder.at_focal(FocalNode("ingroup_mrca", fraction=1.0))
        assert at_root.root == deep.root
        # The shallow end is the ingroup MRCA, where the ingroup's own
        # likelihood vector is seeded.
        at_ingroup = ladder.at_focal(FocalNode("ingroup_mrca", fraction=0.0))
        assert at_ingroup.root == ladder.ingroup_mrca

    def test_provenance_records_the_placement(self):
        ts = _two_population_ts()
        names, ingroup, outgroup = TestInferenceWiring._panel(ts)
        inference = anc.ARGBasedInference(
            ts, JC69(), mu=1.25e-8, sample_map={v: k for k, v in names.items()},
            progress=False, ingroup_samples=ingroup, outgroup_samples=outgroup,
            focal=FocalNode("ingroup_mrca", coalescences=1),
        )
        assert inference._focal_provenance()["focal_coalescences"] == 1


class TestPathParity:
    """The two re-rooting implementations must agree.

    The native path permutes parent pointers on the packed dense arrays. The
    generic path builds a :class:`~ancestree.trees.RerootedTree` and runs the
    ordinary traversal over it. They are independent code, and comparing them
    is what caught ``tau`` being interpreted in different units.
    """

    @pytest.mark.parametrize("model,pi", MODELS)
    @pytest.mark.parametrize("tau_fraction", [0.0, 0.5, 1.0])
    def test_native_matches_the_rerooted_tree(self, model, pi, tau_fraction):
        from ancestree.trees import RerootedTree

        tree = TREE_SEQUENCE.first()
        length = tree.branch_length(INGROUP_MRCA)
        focal = ResolvedFocal(INGROUP_MRCA, tau_fraction * length)
        native = _posterior(model, pi, focal)

        site = anc.Site(
            chrom="1", pos=5, alleles=("A", "G", "T"),
            tip_alleles={f"s{int(n)}": OBSERVED[int(n)]
                         for n in TREE_SEQUENCE.samples()},
        )
        base = anc.TskitLocalTree(
            TREE_SEQUENCE, position=5.0,
            sample_map={f"s{int(n)}": int(n) for n in TREE_SEQUENCE.samples()},
        )
        base.time_scale = TIME_SCALE
        likelihood = anc.Likelihood(
            model, base_composition=(None if pi is None else COMPOSITION),
        )
        log_l = np.asarray(likelihood.log_likelihoods(
            RerootedTree(base, focal.node, focal.tau), [site],
        )).ravel()
        weighted = np.exp(log_l - log_l.max()) * model.stationary(pi=pi)
        assert native == pytest.approx(weighted / weighted.sum(), abs=1e-12)


class TestEnsembleReroot:
    """The ensemble kernel re-roots on its own packed arrays.

    :func:`ancestree._ensemble._reroot` is a third implementation, reached by
    every local-tree ensemble posterior: it reverses dense parent pointers in
    place and, for a point strictly inside an edge, synthesises a split node in
    the two trailing ``branch_t`` slots. Neither the placement it computes nor
    the posterior it produces is compared to the canonical path anywhere else.
    """

    #: Panel haplotypes. The packing requires a binary tree of ``2n - 1`` nodes.
    N_HAP = len(INGROUP_NODES) + 2

    #: Generations to expected substitutions, as ``_pack_base`` applies it.
    MU = TIME_SCALE

    @staticmethod
    def _anchors():
        """The ingroup MRCA, and a node two edges below the root.

        The MRCA is a child of the root here, so a fraction placed from it
        never leaves the first edge and no parent pointer above it is reversed.
        The deeper anchor is what exercises the walk.
        """
        tree = TREE_SEQUENCE.first()
        return {"ingroup_mrca": INGROUP_MRCA,
                "two_edges_down": int(tree.parent(TREE_SEQUENCE.samples()[0]))}

    @classmethod
    def _packed(cls):
        """The panel as the dense post-order arrays the ensemble packs into."""
        tree = TREE_SEQUENCE.first()
        n_all, nslot = 2 * cls.N_HAP - 1, 2 * cls.N_HAP + 1
        dense = {int(u): d for d, u in enumerate(tree.postorder())}
        parent0 = np.full(n_all, -1, np.int32)
        branch_t = np.zeros(nslot)
        for node, d in dense.items():
            parent = tree.parent(node)
            if parent != tskit.NULL:
                parent0[d] = dense[int(parent)]
                branch_t[d] = tree.branch_length(node) * cls.MU
        samples = np.array([dense[int(u)] for u in TREE_SEQUENCE.samples()],
                           np.int32)
        return dense, parent0, branch_t, samples

    @classmethod
    def _reroot_at(cls, anchor: int, fraction: float):
        """Run the ensemble re-rooting, returning it and its traversal arrays."""
        from ancestree._ensemble import _reroot

        n = cls.N_HAP
        n_all, nslot = 2 * n - 1, 2 * n + 1
        dense, parent0, branch_t, samples = cls._packed()
        pidx = np.empty(nslot, np.int32)
        cofs = np.empty(nslot + 1, np.int32)
        cflat = np.empty(nslot, np.int32)
        poi = np.empty(n, np.int32)
        root = _reroot(
            parent0, branch_t, dense[int(anchor)], n, float(fraction), -1,
            pidx, cofs, cflat, poi,
            np.empty(nslot, np.int64), np.empty(n_all, np.int64),
            np.empty(nslot, np.int64), np.empty(2 * nslot, np.int64),
            np.empty(2 * nslot, np.int64),
        )
        return dense, branch_t, samples, pidx, cofs, cflat, poi, int(root)

    @classmethod
    def _posterior(cls, model, pi, anchor: int, fraction: float) -> np.ndarray:
        """Normalised posterior at the ensemble's focal point, for one member."""
        from ancestree._ensemble import S_STATES, score_ensemble

        n = cls.N_HAP
        nslot = 2 * n + 1
        (_dense, branch_t, samples, pidx, cofs, cflat, poi,
         root) = cls._reroot_at(anchor, fraction)
        lam, V, Vinv = model.real_eig(pi=pi)
        out = np.zeros((1, 1, S_STATES))
        score_ensemble(
            branch_t.reshape(1, 1, nslot), lam, V, Vinv,
            pidx.reshape(1, 1, nslot), poi.reshape(1, 1, n),
            cofs.reshape(1, 1, nslot + 1), cflat.reshape(1, 1, nslot),
            np.array([[root]], np.int64), samples.reshape(1, 1, n),
            _tip_states(), np.array([0, 1], np.int64), np.array([0], np.int64),
            nslot, S_STATES, out,
        )
        log_l = out[0, 0]
        weighted = np.exp(log_l - log_l.max()) * model.stationary(pi=pi)
        return weighted / weighted.sum()

    @classmethod
    def _assert_parity(cls, model, pi, anchor_name, fraction):
        """Ensemble and canonical posteriors at the same focal point.

        The canonical side resolves the placement with
        :meth:`FocalNode._walk_up() <ancestree.focal.FocalNode._walk_up>`, so this pins the two placements
        against each other as well as the two re-rootings.
        """
        from ancestree.focal import FocalNode

        anchor = cls._anchors()[anchor_name]
        node, tau = FocalNode._walk_up(TREE_SEQUENCE.first(), anchor, fraction, None)
        assert cls._posterior(model, pi, anchor, fraction) == pytest.approx(
            _posterior(model, pi, ResolvedFocal(node, tau)), abs=1e-12,
        )

    #: JC69 for the uniform case and HKY for a skewed ``pi``. The re-rooting
    #: never reads ``Q``: the model reaches it only through the ``P(t)`` the
    #: ensemble rebuilds per edge, which
    #: ``test_ensemble_kernel_parity.test_real_eig_reconstructs_transition_probs``
    #: covers across the full set.
    PARITY_MODELS = [(JC69(), None), (HKY(kappa=2.7), COMPOSITION.pi)]

    @pytest.mark.parametrize("model,pi", PARITY_MODELS)
    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_matches_the_canonical_kernel(self, model, pi, fraction):
        self._assert_parity(model, pi, "two_edges_down", fraction)

    def test_matches_the_canonical_kernel_at_the_ingroup_mrca(self):
        """The production anchor, whose path to the root is a single edge."""
        self._assert_parity(JC69(), None, "ingroup_mrca", 0.5)

    @pytest.mark.parametrize("fraction,inside", [
        (0.0, False), (0.5, True), (1.0, False),
    ])
    def test_the_split_node_is_used_only_inside_an_edge(self, fraction, inside):
        """Guards the parity above against silently taking the no-split path."""
        n_all = 2 * self.N_HAP - 1
        anchor = self._anchors()["two_edges_down"]
        dense, _bt, _s, _p, _c, _cf, _poi, root = self._reroot_at(
            anchor, fraction)
        if inside:
            assert root == n_all, "no split node was built"
        elif fraction == 0.0:
            assert root == dense[int(anchor)]
        else:
            assert root == dense[int(TREE_SEQUENCE.first().root)]

    def test_the_deeper_anchor_reverses_a_parent_pointer(self):
        """The walk must cross a node, not stay on the anchor's own edge."""
        from ancestree.focal import FocalNode

        tree = TREE_SEQUENCE.first()
        anchor = self._anchors()["two_edges_down"]
        assert tree.parent(tree.parent(anchor)) != tskit.NULL, (
            "the deeper anchor is a child of the root, so the walk is trivial")
        node, _tau = FocalNode._walk_up(tree, anchor, 0.75, None)
        assert node != anchor, "fraction 0.75 did not leave the anchor's edge"


class TestPostorderFromParents:
    """The re-rooted traversal is derived from the parents, not read off tskit."""

    def test_matches_tskit_postorder_on_the_original_rooting(self):
        tree = TREE_SEQUENCE.first()
        order = list(tree.postorder())
        rank = {int(n): i for i, n in enumerate(order)}
        parents = np.full(len(order), -1, dtype=np.int32)
        for node in order:
            parent = tree.parent(node)
            if parent != tskit.NULL:
                parents[rank[int(node)]] = rank[int(parent)]
        got = jk.postorder_csr_from_parents(
            parents, rank[int(tree.root)], len(order),
        )[2]
        want = np.array(
            [rank[int(n)] for n in order if tree.num_children(n) > 0],
            dtype=np.int32,
        )
        assert np.array_equal(got, want)


def _two_population_ts(seed: int = 5, split_time: float = 2e5):
    """An ARG with a 4-haplotype ingroup and 2 outgroups.

    :param seed: msprime seed, for both ancestry and mutations.
    :param split_time: Population split, in generations. The default is deep
        enough that the ingroup is monophyletic on every tree.
    """
    import msprime

    demography = msprime.Demography()
    for name in ("A", "B", "AB"):
        demography.add_population(name=name, initial_size=1e4)
    demography.add_population_split(time=split_time, derived=["A", "B"],
                                    ancestral="AB")
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(4, population="A"),
                 msprime.SampleSet(2, population="B")],
        demography=demography, sequence_length=2e5,
        recombination_rate=1e-8, random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=1.25e-8, random_seed=seed)


def test_naming_zero_coalescences_reports_at_the_anchor():
    """``coalescences=0`` names the anchor itself, not its parent."""
    from ancestree.focal import FocalNode

    import msprime

    # Seed 11 leaves two coalescences between the ingroup MRCA and the root.
    # On a shallower tree every placement clamps to the same node.
    ts = msprime.sim_ancestry(8, ploidy=1, sequence_length=1000,
                              random_seed=11)
    tree = ts.first()
    ingroup_nodes = [0, 1]
    anchor = FocalNode("ingroup_mrca").resolve(
        tree, ingroup_nodes=ingroup_nodes)
    at_zero = FocalNode("ingroup_mrca", coalescences=0).resolve(
        tree, ingroup_nodes=ingroup_nodes)
    assert at_zero.node == anchor.node, (
        f"coalescences=0 moved the readout from {anchor.node} to "
        f"{at_zero.node}; the walk is one step too high")
    # And one coalescence up is the anchor's parent, not its grandparent.
    at_one = FocalNode("ingroup_mrca", coalescences=1).resolve(
        tree, ingroup_nodes=ingroup_nodes)
    assert at_one.node == tree.parent(anchor.node), (
        f"coalescences=1 reported {at_one.node}, expected the anchor's "
        f"parent {tree.parent(anchor.node)}")


def test_a_paraphyletic_ingroup_is_flagged_and_counted():
    """The anchor subtending an outgroup tip is reported, not passed over.

    The flag reaches no posterior, only a counter and the output provenance,
    so it is asserted ``False`` on a paraphyletic fixture here.
    """
    import tskit

    from ancestree.focal import FocalNode

    # ((i0,(o1,i1)),o2): the ingroup MRCA subtends the outgroup o1.
    tables = tskit.TableCollection(sequence_length=100.0)
    for _ in range(4):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0.0)
    inner = tables.nodes.add_row(time=1.0)     # (o1, i1)
    anchor = tables.nodes.add_row(time=2.0)    # (i0, inner)
    root = tables.nodes.add_row(time=3.0)      # (anchor, o2)
    for parent, child in ((inner, 1), (inner, 3), (anchor, 0), (anchor, inner),
                          (root, anchor), (root, 2)):
        tables.edges.add_row(left=0.0, right=100.0, parent=parent, child=child)
    tables.sort()
    tree = tables.tree_sequence().first()

    ingroup_nodes = [0, 1]                     # i0, i1. O1 is node 3, o2 is 2
    resolved = FocalNode("ingroup_mrca").resolve(
        tree, ingroup_nodes=ingroup_nodes)
    assert int(tree.num_samples(resolved.node)) > len(ingroup_nodes), (
        "fixture is monophyletic; it cannot exercise the flag")
    assert resolved.ingroup_is_monophyletic is False, (
        f"the anchor subtends {tree.num_samples(resolved.node)} samples for a "
        f"{len(ingroup_nodes)}-haplotype ingroup, but was reported as "
        f"monophyletic")


class TestInferenceWiring:
    """The focal node reaches the kernel from both genealogy modes."""

    @staticmethod
    def _panel(ts):
        names = {int(n): f"s{int(n)}" for n in ts.samples()}
        ingroup = [names[int(n)] for n in ts.samples()
                   if ts.node(int(n)).population == 0]
        outgroup = [names[int(n)] for n in ts.samples()
                    if ts.node(int(n)).population == 1]
        return names, ingroup, outgroup

    def test_arg_mode_reports_at_a_different_node(self):
        ts = _two_population_ts()
        names, ingroup, outgroup = self._panel(ts)
        sample_map = {v: k for k, v in names.items()}
        calls = {}
        for focal in ("panel_root", "ingroup_mrca"):
            inference = anc.ARGBasedInference(
                ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
                ingroup_samples=ingroup, outgroup_samples=outgroup, focal=focal,
            )
            calls[focal] = {s.pos: p.map_allele for s, p in inference.infer()}
        differing = sum(1 for pos, allele in calls["panel_root"].items()
                        if calls["ingroup_mrca"].get(pos) != allele)
        # The two nodes are separated by the ingroup-outgroup divergence, so a
        # substantial minority of sites must move. Equality would mean the
        # focal node never reached the kernel.
        assert differing > 0.05 * len(calls["panel_root"])

    def test_arg_mode_notes_when_the_ingroup_is_the_whole_panel(self, caplog):
        # Nothing declared, so the ingroup is every sample and its MRCA is the
        # panel root. That is a defensible default but easy to mistake for a
        # reading at the ingroup's own ancestor, so it is said out loud.
        ts = _two_population_ts()
        with caplog.at_level("INFO"):
            inference = anc.ARGBasedInference(
                ts, JC69(), mu=1.25e-8, progress=False, focal="ingroup_mrca",
            )
        assert "the ingroup is the whole panel" in caplog.text
        tree = ts.first()
        resolved = inference.focal.resolve(
            tree, ingroup_nodes=inference._ingroup_nodes,
        )
        assert resolved.node == int(tree.root)

    def test_the_non_monophyly_count_is_per_walk(self, caplog):
        """Each pass reports its own trees, not the running total.

        The counter was summed across calls, so grading one instance four
        times reported four times the trees the ARG has, eventually more
        trees than it holds.
        """
        import re

        # A shallow split, so the ingroup is frequently paraphyletic and the
        # counter has something to count. _two_population_ts splits at 20 Ne.
        ts = _two_population_ts(split_time=5e3)
        names, ingroup, outgroup = self._panel(ts)
        sample_map = {v: k for k, v in names.items()}
        inference = anc.ARGBasedInference(
            ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
            ingroup_samples=ingroup, outgroup_samples=outgroup,
        )
        counts = []
        for _ in range(3):
            with caplog.at_level("INFO"):
                caplog.clear()
                list(inference.infer())
            found = re.findall(r"(\d+) tree\(s\) where the ingroup", caplog.text)
            counts.append(int(found[0]) if found else 0)
        assert counts[0], "the fixture must have a non-monophyletic tree to count"
        assert len(set(counts)) == 1, f"the count moved across walks: {counts}"
        assert counts[0] <= ts.num_trees

    def test_local_tree_forwards_focal_on_every_path(self):
        ts = _two_population_ts()
        names, ingroup, outgroup = self._panel(ts)
        sites = [
            anc.Site(chrom="1", pos=int(v.site.position),
                    alleles=canonical_alleles(v.alleles[:2]),
                    tip_alleles={names[int(s)]: v.alleles[g]
                                 for s, g in zip(ts.samples(), v.genotypes)})
            for v in ts.variants()
            if len(v.alleles) >= 2 and all(a in anc.STATES for a in v.alleles)
        ]

        def run(focal, **kwargs):
            inference = anc.LocalTreeInference(
                sites, JC69(), mu=1.25e-8, rec_rate=1e-8,
                sample_names=[names[int(n)] for n in ts.samples()],
                sequence_length=ts.sequence_length, progress=False,
                ingroup_samples=ingroup, outgroup_samples=outgroup,
                focal=focal, **kwargs,
            )
            return {s.pos: p.map_allele for s, p in inference.infer()}

        at_root = run("panel_root")
        at_ingroup = run("ingroup_mrca")
        differing = sum(1 for pos, allele in at_root.items()
                        if at_ingroup.get(pos) != allele)
        assert differing > 0.05 * len(at_root)

        # The chunked path builds its own inner ARG per segment. Without the
        # forwarding it would silently keep the panel root.
        chunked = run("ingroup_mrca", chunk_size=50_000)
        agreeing = sum(1 for pos, allele in at_ingroup.items()
                       if chunked.get(pos) == allele)
        assert agreeing > 0.95 * len(at_ingroup)


class TestInferSiteAcceptsAReRootedView:
    """A RerootedTree reports at its focal point, so ``infer_site`` accepts it.

    A re-rooted view of a fixed-tree ladder has neither the focal node as its
    root, nor ``at_focal``, nor a tskit tree, and must still resolve.
    """

    TREES = "docs/_static/quickstart.trees"
    ING = [f"i{i}" for i in range(6)]
    OUT = ["o0", "o1"]

    def _inference(self, focal=None):
        src = list(anc.TskitSource(tskit.load(self.TREES)))
        kwargs = {} if focal is None else {"focal": focal}
        inf = anc.FixedTreeInference(
            src, JC69(), n_target_sites=100_000, ingroup_samples=self.ING,
            outgroup_samples=self.OUT, **kwargs)
        inf.fit()
        return inf, src

    @pytest.mark.parametrize("focal", [
        FocalNode("ingroup_mrca", fraction=1.0),
        FocalNode("ingroup_mrca", coalescences=1),
        FocalNode("panel_root"),
    ])
    def test_a_placement_matches_the_batch_answer(self, focal):
        inf, src = self._inference(focal)
        site = src[0]
        batch = {int(s.pos): np.asarray(p.values) for s, p in inf.infer()}
        one = np.asarray(inf.infer_site(inf._focal_tree, site).values)
        assert np.abs(one - batch[int(site.pos)]).max() == pytest.approx(0.0, abs=1e-12)

    def test_a_deep_rooted_ladder_view_is_accepted(self):
        inf, src = self._inference()
        posterior = inf.infer_site(inf._focal_tree.as_deep_rooted(), src[0])
        assert float(np.asarray(posterior.values).sum()) == pytest.approx(1.0)


class TestDiagnosticsAndTraps:
    """The focal node must fail loudly rather than quietly do nothing."""

    def test_unmatched_ingroup_names_raise(self):
        ts = _two_population_ts()
        sample_map = {f"s{int(n)}": int(n) for n in ts.samples()}
        with pytest.raises(ValueError, match="not in the panel"):
            anc.ARGBasedInference(
                ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
                ingroup_samples=["typo1", "typo2"], focal="ingroup_mrca",
            )

    def test_one_unmatched_ingroup_name_raises(self):
        """A partly-mistyped list must not silently shrink the ingroup.

        Dropping the unmatched name leaves a smaller ingroup whose MRCA sits
        deeper, which changes every posterior and has no other symptom.
        """
        ts = _two_population_ts()
        sample_map = {f"s{int(n)}": int(n) for n in ts.samples()}
        with pytest.raises(ValueError, match="typo1"):
            anc.ARGBasedInference(
                ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
                ingroup_samples=["s0", "s1", "typo1"], focal="ingroup_mrca",
            )

    def test_an_ingroup_named_per_individual_matches_both_haplotypes(self):
        """``i0`` designates ``i0_h0`` and ``i0_h1``, as it does for outgroups."""
        ts = _two_population_ts()
        names = [f"i{int(n) // 2}_h{int(n) % 2}" for n in ts.samples()]
        sample_map = {nm: int(n) for nm, n in zip(names, ts.samples())}
        inf = anc.ARGBasedInference(
            ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
            ingroup_samples=["i0", "i1"], focal="ingroup_mrca",
        )
        assert len(inf._ingroup_nodes) == 4

    def test_offsets_on_the_root_anchor_are_rejected(self):
        # They move from the anchor toward the root, so on the root itself they
        # would silently do nothing while still demanding an ingroup.
        for placement in ({"fraction": 0.5}, {"coalescences": 2},
                          {"depth": 10.0}):
            with pytest.raises(ValueError, match="does nothing"):
                FocalNode("panel_root", **placement)

    def test_non_monophyly_is_counted_and_survives_the_fork_pool(self):
        ts = _two_population_ts()
        names = [f"s{int(n)}" for n in ts.samples()]
        sample_map = {name: int(n) for name, n in zip(names, ts.samples())}
        counts = []
        for workers in (1, 2):
            inference = anc.ARGBasedInference(
                ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
                ingroup_samples=names[:4], outgroup_samples=names[4:],
                focal="ingroup_mrca", n_workers=workers,
            )
            list(inference.infer())
            counts.append(inference._n_ingroup_non_monophyletic)
        # Incremented in a forked child, so without the return-value plumbing
        # the parent's copy would stay at zero.
        assert counts[0] == counts[1]

    def test_summary_names_the_focal_node(self, caplog):
        ts = _two_population_ts()
        names = [f"s{int(n)}" for n in ts.samples()]
        sample_map = {name: int(n) for name, n in zip(names, ts.samples())}
        inference = anc.ARGBasedInference(
            ts, JC69(), mu=1.25e-8, sample_map=sample_map, progress=False,
            ingroup_samples=names[:4], outgroup_samples=names[4:],
            focal="ingroup_mrca",
        )
        with caplog.at_level("INFO"):
            list(inference.infer())
        assert "ingroup_mrca" in caplog.text


class TestBenchmarkTruthAtFocal:
    """The harness must read the simulated truth at the node being scored."""

    def test_truth_moves_with_the_focal_node(self):
        import sys

        sys.path.insert(0, "workflow/scripts")
        from _robustness_common import _truth_at_node

        # One mutation above the ingroup MRCA: the deep state is A, the state
        # at the ingroup MRCA is T.
        tables = tskit.TableCollection(sequence_length=10.0)
        ingroup = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                   for _ in range(2)]
        outgroup = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        mrca = tables.nodes.add_row(flags=0, time=1.0)
        root = tables.nodes.add_row(flags=0, time=3.0)
        for child in ingroup:
            tables.edges.add_row(left=0, right=10.0, parent=mrca, child=child)
        tables.edges.add_row(left=0, right=10.0, parent=root, child=mrca)
        tables.edges.add_row(left=0, right=10.0, parent=root, child=outgroup)
        site = tables.sites.add_row(position=5.0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=mrca, derived_state="T")
        tables.sort()
        ts = tables.tree_sequence()
        tree = ts.first()

        assert ts.site(0).ancestral_state == "A"
        assert _truth_at_node(ts.site(0), tree, ingroup) == "T"

    def test_falls_back_to_the_deep_state_without_an_mrca(self):
        import sys

        sys.path.insert(0, "workflow/scripts")
        from _robustness_common import _truth_at_node

        tables = tskit.TableCollection(sequence_length=10.0)
        a = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        b = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=5.0, ancestral_state="C")
        tables.sort()
        ts = tables.tree_sequence()
        # No edges, so the two samples are separate roots with no MRCA.
        assert _truth_at_node(ts.site(0), ts.first(), [a, b]) == "C"

    def test_the_fixed_class_is_graded_at_a_node_that_does_not_move(self):
        """The grading node must not deepen as outgroups are added.

        Otherwise the n_out axis conflates how much the outgroups buy with how
        much harder the question became, and at n_out = 0 it is degenerate: the
        panel root collapses onto the ingroup MRCA, so the truth is the
        ingroup's own fixed allele and every method scores perfectly on a
        question with no content.
        """
        import sys

        sys.path.insert(0, "workflow/scripts")
        from _robustness_common import _split_truth_for_panel

        # Ingroup, a near outgroup, and a far one. A mutation on the branch
        # between the two outgroup joins is seen only from the far root, so the
        # near and far panels disagree unless the grading node is pinned.
        tables = tskit.TableCollection(sequence_length=10.0)
        i0, i1 = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                  for _ in range(2)]
        near = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        far = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        mrca = tables.nodes.add_row(flags=0, time=1.0)
        join_near = tables.nodes.add_row(flags=0, time=2.0)
        join_far = tables.nodes.add_row(flags=0, time=4.0)
        for child in (i0, i1):
            tables.edges.add_row(left=0, right=10.0, parent=mrca, child=child)
        tables.edges.add_row(left=0, right=10.0, parent=join_near, child=mrca)
        tables.edges.add_row(left=0, right=10.0, parent=join_near, child=near)
        tables.edges.add_row(left=0, right=10.0, parent=join_far, child=join_near)
        tables.edges.add_row(left=0, right=10.0, parent=join_far, child=far)
        site = tables.sites.add_row(position=2.0, ancestral_state="A")
        # Above the near join, so only the far root still carries "A".
        tables.mutations.add_row(site=site, node=join_near, derived_state="T")
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()

        ingroup = [i0, i1]
        full = ingroup + [near, far]
        # The primitive still grades at whatever panel it is handed, and the
        # two panels genuinely disagree, which is why the callers must pin
        # it rather than pass their own selection.
        assert _split_truth_for_panel(ts, ingroup, ingroup + [near])[2] == "T"
        assert _split_truth_for_panel(ts, ingroup, full)[2] == "A"

    def test_the_split_table_grades_each_class_at_its_own_node(self):
        import sys

        sys.path.insert(0, "workflow/scripts")
        from _robustness_common import _split_truth_for_panel

        # Two sites on one tree: one polymorphic within the ingroup, one the
        # ingroup has fixed by a mutation above its MRCA.
        tables = tskit.TableCollection(sequence_length=10.0)
        i0, i1 = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                  for _ in range(2)]
        out = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        mrca = tables.nodes.add_row(flags=0, time=1.0)
        root = tables.nodes.add_row(flags=0, time=3.0)
        for child in (i0, i1):
            tables.edges.add_row(left=0, right=10.0, parent=mrca, child=child)
        tables.edges.add_row(left=0, right=10.0, parent=root, child=mrca)
        tables.edges.add_row(left=0, right=10.0, parent=root, child=out)
        fixed = tables.sites.add_row(position=2.0, ancestral_state="A")
        tables.mutations.add_row(site=fixed, node=mrca, derived_state="T")
        poly = tables.sites.add_row(position=6.0, ancestral_state="A")
        tables.mutations.add_row(site=poly, node=mrca, derived_state="T",
                                 time=1.5)
        tables.mutations.add_row(site=poly, node=i0, derived_state="G",
                                 time=0.5)
        tables.sort()
        tables.build_index()
        tables.compute_mutation_parents()
        ts = tables.tree_sequence()

        truth = _split_truth_for_panel(ts, [i0, i1], [i0, i1, out])
        # Fixed within the ingroup, so graded at the panel root.
        assert truth[2] == "A"
        # The mutation above the ingroup MRCA is the one a simplified panel
        # would have deleted. Reading the unsimplified tree keeps it.
        assert _split_truth_for_panel(ts, [i0, i1], [i0, i1])[2] == "T"
        # Polymorphic within the ingroup, so graded at the ingroup MRCA.
        assert truth[6] == "T"


class TestFocalProvenance:
    """What the run records about the node it reported at."""

    @staticmethod
    def _inference(**kwargs):
        ts = _two_population_ts()
        names, ingroup, outgroup = TestInferenceWiring._panel(ts)
        return anc.ARGBasedInference(
            ts, JC69(), mu=1.25e-8,
            sample_map={v: k for k, v in names.items()},
            progress=False, ingroup_samples=ingroup,
            outgroup_samples=outgroup, **kwargs,
        )

    def test_counts_are_withheld_until_the_walk_finishes(self):
        # A streaming writer emits its header first, so reading the counters
        # then would report zero non-monophyletic trees whatever the truth.
        inference = self._inference(focal="ingroup_mrca")
        before = inference._focal_provenance()
        assert "n_trees_ingroup_non_monophyletic" not in before

        list(inference.infer())
        after = inference._focal_provenance()
        assert after["n_trees_ingroup_non_monophyletic"] >= 0
        assert after["n_segments_without_ingroup_mrca"] == 0

    def test_the_root_focal_records_only_its_name(self):
        assert self._inference(focal="panel_root")._focal_provenance() == {
            "focal": "panel_root",
        }


class TestAboveTheRoot:
    """A depth beyond the tree's own root, where there is no edge to split.

    The posterior there is closed-form: the root posterior pushed forward by
    ``P(tau)``. Nothing is learned by going higher, but the reading has to be
    available so a fixed depth means the same thing on every local tree, even
    the shallow ones.
    """

    @pytest.mark.parametrize("model,pi", MODELS)
    @pytest.mark.parametrize("tau", [0.05, 0.4, 3.0])
    def test_matches_the_closed_form(self, model, pi, tau):
        at_root = _posterior(model, pi, None)
        pushed = model.transition_probs(
            tau * TIME_SCALE, pi=pi,
        ).T @ at_root
        pushed = pushed / pushed.sum()
        got = _posterior(
            model, pi, ResolvedFocal(int(TREE_SEQUENCE.first().root), tau),
        )
        assert got == pytest.approx(pushed, abs=1e-12)

    def test_the_two_kernel_paths_agree(self):
        from ancestree.trees import RerootedTree, TskitLocalTree

        tree = TREE_SEQUENCE.first()
        names = {f"s{i}": int(n) for i, n in enumerate(TREE_SEQUENCE.samples())}
        wrapped = TskitLocalTree.from_tskit_tree(tree, sample_map=names)
        wrapped.time_scale = TIME_SCALE
        site = anc.Site(
            chrom="1", pos=1, alleles=tuple(sorted(set(OBSERVED.values()))),
            tip_alleles={f"s{i}": OBSERVED[int(n)]
                         for i, n in enumerate(TREE_SEQUENCE.samples())},
        )
        engine = anc.Likelihood(JC69())
        for tau in (0.05, 0.4, 3.0):
            view = RerootedTree(wrapped, wrapped.root, tau)
            generic = np.asarray(engine.log_likelihoods(view, [site]))[0]
            generic = np.exp(generic - generic.max())
            native = _posterior(JC69(), None, ResolvedFocal(int(tree.root), tau))
            generic = generic * JC69().stationary()
            assert generic / generic.sum() == pytest.approx(native, abs=1e-12)


class TestPlottingUtilities:
    """The focal node drawn on a tree, and the posterior along the backbone."""

    @staticmethod
    def _ladder_and_site():
        ingroup = [f"i{k}" for k in range(6)]
        ladder = anc.OutgroupLadderTree(ingroup, ["o1", "o2", "o3"])
        ladder.set_params([0.004, 0.008, 0.006, 0.014, 0.020])
        site = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                       tip_alleles={**{s: "G" for s in ingroup},
                                    "o1": "G", "o2": "A", "o3": "A"})
        return ladder, site

    def test_the_marker_moves_with_the_placement(self):
        from ancestree.plotting import FocalTreePlot

        ladder, _ = self._ladder_and_site()
        heights = [
            FocalTreePlot(ladder, FocalNode("ingroup_mrca", fraction=f))
            .focal_point()[1]
            for f in (0.0, 0.5, 1.0)
        ]
        # The canvas puts the deepest node at the top, so the marker rises.
        assert heights[0] < heights[1] < heights[2]
        assert FocalTreePlot(ladder, "panel_root").focal_point()[1] == heights[2]

    def test_the_sweep_ends_on_the_ingroups_allele_and_the_outgroups(self):
        from ancestree.plotting import FocalSweep

        ladder, site = self._ladder_and_site()
        values = FocalSweep(ladder, site, JC69(),
                            fractions=(0.0, 1.0)).posteriors()
        assert values.sum(axis=1) == pytest.approx(1.0)
        a = anc.STATES.index("A")
        # The ingroup's own allele at its MRCA; the outgroups' allele deeper.
        assert values[0, a] == pytest.approx(0.0, abs=1e-9)
        assert values[1, a] > 0.98

    def test_the_sweep_matches_the_inference_under_the_same_prior(self):
        from ancestree.plotting import FocalSweep

        ladder, site = self._ladder_and_site()
        focal = FocalNode("ingroup_mrca", fraction=1.0)
        inference = anc.FixedTreeInference(
            [site], JC69(), n_target_sites=10_000, tree=ladder, focal=focal,
            fit_required=False, progress=False, baseline_check=False,
        )
        ((_, post),) = list(inference.infer())
        # FixedTreeInference derives a base composition from the data and uses
        # it as the stationary prior, so the sweep has to be given the same one
        # to be comparable. The likelihoods themselves are identical.
        sweep = FocalSweep(ladder, site, JC69(),
                           base_composition=inference.base_composition,
                           fractions=(1.0,))
        got = sweep.posteriors()[0, anc.STATES.index("A")]
        assert got == pytest.approx(float(post["A"]), abs=1e-14)

    def test_a_row_of_panels_is_one_svg_document(self):
        from ancestree.plotting import FocalTreePlot

        ladder, site = self._ladder_and_site()
        svg = FocalTreePlot.draw_svg_row(
            ladder, [("default", None), ("root", "panel_root")], site=site)
        assert svg.startswith("<svg") and svg.count("<svg") == 3
        assert "#focal-panel-1 .node" in svg

    def test_drawing_needs_no_display(self):
        import matplotlib
        matplotlib.use("Agg")
        from ancestree.plotting import FocalSweep, FocalTreePlot

        ladder, site = self._ladder_and_site()
        assert FocalTreePlot(ladder).draw(site=site) is not None
        assert FocalSweep(ladder, site, JC69(), fractions=(0.0, 0.5)).draw()

    def test_a_single_outgroup_never_reads_at_its_tip(self):
        """Rooting at the tip would return that sample's own allele.

        With one outgroup the backbone has no coalescence above the ingroup
        MRCA, only the outgroup's tip, so the deep end is a point on the branch
        between them.
        """
        ingroup = [f"i{k}" for k in range(6)]
        ladder = anc.OutgroupLadderTree(ingroup, ["o1"])
        ladder.set_params([0.02])
        deep = ladder.at_focal(FocalNode("ingroup_mrca", fraction=1.0))
        assert deep.root != ladder.tip_for_sample("o1")
        engine = anc.Likelihood(JC69())
        for outgroup_allele, expected in (("G", 1.0), ("A", 0.5)):
            site = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                           tip_alleles={**{s: "G" for s in ingroup},
                                        "o1": outgroup_allele})
            seeds = {ladder.ingroup_mrca: np.exp(
                KingmanIngroupWeight(ingroup).log_probs([site])
            )}
            log_l = np.asarray(
                engine.log_likelihoods(deep, [site], node_seeds=seeds)
            )[0]
            w = np.exp(log_l - log_l.max()) * JC69().stationary()
            assert (w / w.sum())[anc.STATES.index("G")] == pytest.approx(
                expected, abs=0.01,
            )

    @staticmethod
    def _ladder(params=(0.01, 0.02, 0.03, 0.05, 0.07)):
        ingroup = [f"i{k}" for k in range(6)]
        ladder = anc.OutgroupLadderTree(ingroup, ["o1", "o2", "o3"])
        ladder.set_params(list(params))
        return ladder, ingroup

    @classmethod
    def _posterior_at(cls, ladder, ingroup, depth):
        """Posterior at ``depth``, with the ingroup seeded on allele A."""
        site = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                       tip_alleles={**{s: "A" for s in ingroup},
                                    "o1": "G", "o2": "G", "o3": "G"})
        seed = np.zeros((1, 4))
        seed[0, anc.STATES.index("A")] = 1.0
        view = ladder.at_focal(FocalNode("ingroup_mrca", depth=depth))
        log_l = np.asarray(anc.Likelihood(JC69()).log_likelihoods(
            view, [site], node_seeds={ladder.ingroup_mrca: seed}))
        w = np.exp(log_l - log_l.max(axis=1, keepdims=True)) * 0.25
        return (w / w.sum(axis=1, keepdims=True)).ravel()

    def test_depth_past_the_panel_root_extends_above_it(self):
        """A depth beyond the panel's ancestor reads above it, not an error.

        The backbone is spent at the panel root, going further along it
        heads back down towards the deepest outgroup, so the view grows one
        node above, carrying the overshoot. Refusing would make an absolute
        depth unusable across trees of differing height, which is what it is
        for, and would contradict ARG mode, where ``FocalNode._walk_up`` hands the
        root back with the overshoot as ``tau``.
        """
        ladder, _ = self._ladder()
        total = ladder._backbone_limit
        at_root = ladder.at_focal(FocalNode("ingroup_mrca", depth=total))
        above = ladder.at_focal(FocalNode("ingroup_mrca", depth=total * 3))
        assert above.n_nodes == at_root.n_nodes + 1, (
            "reading above the panel root should add exactly one node")

    def test_depth_is_continuous_at_the_panel_root(self):
        """Landing exactly on the panel root is the fraction=1.0 view."""
        ladder, ingroup = self._ladder()
        total = ladder._backbone_limit
        deep = ladder.at_focal(FocalNode("ingroup_mrca", fraction=1.0))
        exact = ladder.at_focal(FocalNode("ingroup_mrca", depth=total))
        assert exact.n_nodes == deep.n_nodes
        np.testing.assert_allclose(
            self._posterior_at(ladder, ingroup, total),
            self._posterior_at(ladder, ingroup, total * (1 - 1e-12)),
            atol=1e-9)

    def test_reading_higher_decays_towards_the_stationary_prior(self):
        """Above the root the posterior is the root's pushed forward.

        Each extra unit of depth is another ``P(t)`` applied, so the readout
        loses information monotonically and tends to the model's stationary
        distribution, uniform under JC69. That is what makes the extension
        meaningful rather than a silent clamp, which would report the panel
        root's confident answer at every depth.
        """
        ladder, ingroup = self._ladder()
        total = ladder._backbone_limit
        gaps = [abs(self._posterior_at(ladder, ingroup, d).max() - 0.25)
                for d in (total, total * 1.5, total * 5, 1.0)]
        assert gaps == sorted(gaps, reverse=True), (
            f"posterior should lose information with depth, got {gaps}")
        far = self._posterior_at(ladder, ingroup, 20.0)
        np.testing.assert_allclose(far, np.full(4, 0.25), atol=1e-6)

    def test_one_absolute_depth_serves_ladders_of_different_heights(self):
        """The motivating case: a fixed depth across trees that differ in height.

        A depth chosen for the deepest ladder overshoots the shallow ones. Every
        ladder must answer, or the depth placement cannot be used across a set
        at all.
        """
        shallow, ing_s = self._ladder((0.001, 0.002, 0.003, 0.004, 0.005))
        deep, ing_d = self._ladder((0.05, 0.10, 0.15, 0.20, 0.30))
        depth = 0.2
        assert depth > shallow._backbone_limit
        for ladder, ingroup in ((shallow, ing_s), (deep, ing_d)):
            p = self._posterior_at(ladder, ingroup, depth)
            assert p.shape == (4,)
            assert p.sum() == pytest.approx(1.0)

    def test_the_focal_view_follows_a_later_fit(self):
        ingroup = [f"i{k}" for k in range(6)]
        sites = [
            anc.Site(chrom="1", pos=pos, alleles=("A", "G"),
                    tip_alleles={**{s: "A" for s in ingroup},
                                 "o1": "A", "o2": "A", "o3": "A"})
            for pos in range(1, 40)
        ]
        inference = anc.FixedTreeInference(
            sites, JC69(), ingroup_samples=ingroup,
            outgroup_samples=["o1", "o2", "o3"], n_target_sites=10_000,
            progress=False, baseline_check=False,
            focal=FocalNode("ingroup_mrca", fraction=1.0),
        )
        seeded = dict(inference._focal_tree._branch)
        inference.fit()
        # fit() sets the rates on the ladder in place, so a view taken at
        # construction would still carry the seed.
        assert dict(inference._focal_tree._branch) != seeded


class TestPanelRootOnAMultiRootTree:
    """``Grade.truth_at_focal`` at the panel root, with no ingroup resolved.

    The ``"panel_root"`` anchor is exempt from the guard that refuses an
    ingroup matching no sample of the truth, so the resolved ingroup node set
    is legitimately empty there.
    """

    @staticmethod
    def _forest(join_the_roots: bool):
        """Four samples in two pairs, optionally joined under one root.

        :param join_the_roots: Whether to add the node joining the two pairs.
        :return: A tree sequence carrying one site, ancestral state ``"A"``,
            with a mutation on the first pair's parent.
        """
        tables = tskit.TableCollection(sequence_length=10.0)
        tips = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                for _ in range(4)]
        left = tables.nodes.add_row(flags=0, time=1.0)
        right = tables.nodes.add_row(flags=0, time=1.0)
        for parent, children in ((left, tips[:2]), (right, tips[2:])):
            for child in children:
                tables.edges.add_row(left=0, right=10.0, parent=parent,
                                     child=child)
        if join_the_roots:
            root = tables.nodes.add_row(flags=0, time=2.0)
            for child in (left, right):
                tables.edges.add_row(left=0, right=10.0, parent=root,
                                     child=child)
        site = tables.sites.add_row(position=5.0, ancestral_state="A")
        tables.mutations.add_row(site=site, node=left, derived_state="T")
        tables.sort()
        return tables.tree_sequence()

    def test_several_roots_fall_back_to_the_ancestral_state(self):
        """An ingroup resolving to no node asked for the MRCA of an empty set,
        which indexed the first of no nodes and raised IndexError."""
        from ancestree.posterior import Grade

        ts = self._forest(join_the_roots=False)
        assert ts.first().num_roots == 2
        truth = Grade.truth_at_focal(
            ts, "panel_root", ingroup_samples=["absent0", "absent1"])
        assert truth == Grade.truth_mapping(ts) == {5: "A"}

    def test_one_root_reads_at_that_root(self):
        from ancestree.posterior import Grade

        ts = self._forest(join_the_roots=True)
        assert ts.first().num_roots == 1
        truth = Grade.truth_at_focal(
            ts, "panel_root", ingroup_samples=["absent0", "absent1"])
        assert truth == {5: "A"}


def test_the_rank_shortcut_matches_the_pairwise_fold_across_roots():
    """The two ``_mrca`` paths agree on a tree whose samples span several
    roots, where the extremes of the post-order ranks lie in different roots."""
    import msprime

    from ancestree.focal import _mrca

    ts = msprime.sim_ancestry(6, sequence_length=1e4, recombination_rate=1e-7,
                              population_size=1e4, end_time=20, random_seed=1)
    n_multi = 0
    for tree in ts.trees():
        n_multi += tree.num_roots > 1
        ranks = {int(n): i for i, n in enumerate(tree.postorder())}
        nodes = sorted(ranks)
        for size in (2, 3, 5):
            for start in range(0, len(nodes) - size + 1, 3):
                subset = nodes[start:start + size]
                assert _mrca(tree, subset, ranks) == _mrca(tree, subset, None)
    assert n_multi > 0


class TestThePostOrderRefusesIndicesItCannotBoundsCheck:
    """The njit core indexes without checking, so the wrapper must.

    Each of these inputs either aborted the interpreter or returned a
    truncated walk over a buffer it had overrun.
    """

    @staticmethod
    def _valid():
        return np.array([2, 2, -1], dtype=np.int32), 2, 3

    def test_a_root_outside_the_node_range_is_refused(self):
        parents, _root, n = self._valid()
        for root in (-1, n, n + 5):
            with pytest.raises(ValueError, match="root_dense outside"):
                jk.postorder_csr_from_parents(parents, root, n)

    def test_a_parent_outside_the_node_range_is_refused(self):
        with pytest.raises(ValueError, match="outside"):
            jk.postorder_csr_from_parents(
                np.array([2, 99, -1], dtype=np.int32), 2, 3)

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="length must equal n_nodes"):
            jk.postorder_csr_from_parents(
                np.array([1, -1], dtype=np.int32), 1, 5)

    def test_a_root_that_is_not_the_root_is_refused(self):
        """Two shapes: a node with a parent, and a cycle with no -1 at all."""
        with pytest.raises(ValueError, match="not the root"):
            jk.postorder_csr_from_parents(
                np.array([0, -1, 1], dtype=np.int32), 0, 3)
        with pytest.raises(ValueError, match="exactly one -1 root"):
            jk.postorder_csr_from_parents(
                np.array([1, 0], dtype=np.int32), 0, 2)

    def test_the_valid_walk_is_unchanged(self):
        parents, root, n = self._valid()
        offsets, children, order = jk.postorder_csr_from_parents(parents, root, n)
        np.testing.assert_array_equal(order, [2])
        assert len(offsets) == n + 1 and len(children) == n - 1
