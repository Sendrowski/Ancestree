"""Tests for `Likelihood`: hand-computed Felsenstein values, polytomy
equivalence, multi-allelic correctness, missing-data marginalisation,
batched-vs-loop agreement, and underflow at wide polytomies.

The engine is tested against a `_ManualTree` defined here, which needs no
tskit fixture. The wide-polytomy tests build star trees with tskit.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest
import tskit
import ancestree as anc

from ancestree.likelihood import Likelihood, _normalise
from ancestree.models import JC69, K2
from ancestree.sites import Site
from ancestree.trees import Tree, TskitLocalTree
from testing._helpers import panel_site, panel_ts


# --------------------------------------------------------------------- helpers


@dataclass
class _ManualTree(Tree):
    """A hand-built tree for kernel tests.

    Provide `children_map: {node: [child1, child2, ...]}`, `branch_lengths: {node: length}`,
    `root_id: int`, `tip_to_sample: {tip_node: sample_id}`.
    """

    children_map: dict[int, list[int]]
    branch_lengths: dict[int, float]
    root_id: int
    tip_to_sample: dict[int, str]
    _po: tuple[int, ...] = field(init=False)
    _sample_to_tip: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        # Compute post-order via DFS from root.
        order: list[int] = []
        def visit(n: int) -> None:
            for c in self.children_map.get(n, []):
                visit(c)
            order.append(n)
        visit(self.root_id)
        self._po = tuple(order)
        self._sample_to_tip = {v: k for k, v in self.tip_to_sample.items()}

    @property
    def n_nodes(self) -> int:
        return len(self._po)

    @property
    def root(self) -> int:
        return self.root_id

    def children(self, node: int) -> Sequence[int]:
        return self.children_map.get(node, [])

    def branch_length(self, node: int) -> float:
        return self.branch_lengths.get(node, 0.0)

    def postorder(self) -> Sequence[int]:
        return self._po

    def tip_for_sample(self, sample_id: str) -> int | None:
        return self._sample_to_tip.get(sample_id)

    def n_tips(self) -> int:
        return sum(1 for n in self._po if not self.children_map.get(n))


def _brute_force_log_likelihoods(tree: _ManualTree, site: Site, model) -> np.ndarray:
    """Sum-over-all-internal-states baseline. Exponential in #internals, only
    use for tiny trees in tests."""
    S = model.n_states
    state_index = {s: i for i, s in enumerate(model.states)}
    P_by_branch = {n: model.transition_probs(tree.branch_length(n)) for n in tree.postorder()}

    # Tip observations as state indices (or None for missing/unknown).
    tip_obs: dict[int, int | None] = {}
    for n in tree.postorder():
        if not tree.children_map.get(n):
            sample = tree.tip_to_sample.get(n)
            allele = site.tip_alleles.get(sample) if sample else None
            tip_obs[n] = state_index.get(allele) if allele is not None else None

    out = np.zeros(S)
    # Enumerate all (root_state, internal_states) combinations.
    for root_state in range(S):
        # Recursively compute likelihood with this root assignment.
        def subtree_likelihood(node: int, parent_state: int | None) -> float:
            if not tree.children_map.get(node):
                # Tip
                obs = tip_obs[node]
                if obs is None:
                    return 1.0
                # P(observed obs | parent_state, branch_length)
                P = P_by_branch[node]
                return float(P[parent_state, obs])
            # Internal: sum over this node's state.
            total = 0.0
            if node == tree.root:
                # Root: fix state = root_state (called from outer loop), branch_length 0.
                state = root_state
                prod = 1.0
                for c in tree.children_map[node]:
                    prod *= subtree_likelihood(c, state)
                return prod
            P = P_by_branch[node]
            for state in range(S):
                trans = float(P[parent_state, state])
                prod = trans
                for c in tree.children_map[node]:
                    prod *= subtree_likelihood(c, state)
                total += prod
            return total

        out[root_state] = subtree_likelihood(tree.root, None)

    with np.errstate(divide="ignore"):
        return np.log(out)


# ------------------------------------------------------------------- fixtures


def _three_tip_tree(branch_lengths=(0.1, 0.1, 0.05, 0.15)) -> _ManualTree:
    """((A,B),C);, node ids: A=0, B=1, internal=3, C=2, root=4."""
    bl_a, bl_b, bl_internal, bl_c = branch_lengths
    return _ManualTree(
        children_map={4: [3, 2], 3: [0, 1]},
        branch_lengths={0: bl_a, 1: bl_b, 3: bl_internal, 2: bl_c, 4: 0.0},
        root_id=4,
        tip_to_sample={0: "A", 1: "B", 2: "C"},
    )


def _polytomy_tree(branch_lengths=(0.1, 0.1, 0.1)) -> _ManualTree:
    """(A,B,C);, three tips as a star polytomy under the root."""
    bl_a, bl_b, bl_c = branch_lengths
    return _ManualTree(
        children_map={3: [0, 1, 2]},
        branch_lengths={0: bl_a, 1: bl_b, 2: bl_c, 3: 0.0},
        root_id=3,
        tip_to_sample={0: "A", 1: "B", 2: "C"},
    )


def _zero_length_binary_tree(branch_lengths=(0.1, 0.1, 0.1)) -> _ManualTree:
    """((A,B):0,C). Equivalent to a 3-way polytomy when the internal edge is 0."""
    bl_a, bl_b, bl_c = branch_lengths
    return _ManualTree(
        children_map={4: [3, 2], 3: [0, 1]},
        branch_lengths={0: bl_a, 1: bl_b, 3: 0.0, 2: bl_c, 4: 0.0},
        root_id=4,
        tip_to_sample={0: "A", 1: "B", 2: "C"},
    )


# ----------------------------------------------------------------------- tests


class TestThreeTipJC:
    def test_matches_brute_force(self):
        tree = _three_tip_tree()
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"A": "A", "B": "A", "C": "C"},
        )
        model = JC69()
        engine = Likelihood(model)
        log_L = engine.log_likelihoods(tree, [site])[0]
        expected = _brute_force_log_likelihoods(tree, site, model)
        np.testing.assert_allclose(log_L, expected, atol=1e-12)

    def test_matches_brute_force_k2p(self):
        tree = _three_tip_tree()
        site = Site(
            chrom="1", pos=1, alleles=("A", "G"),
            tip_alleles={"A": "A", "B": "G", "C": "G"},
        )
        model = K2(kappa=2.5)
        engine = Likelihood(model)
        log_L = engine.log_likelihoods(tree, [site])[0]
        expected = _brute_force_log_likelihoods(tree, site, model)
        np.testing.assert_allclose(log_L, expected, atol=1e-12)


class TestPolytomyEquivalence:
    def test_polytomy_equals_zero_length_bifurcation(self):
        site = Site(
            chrom="1", pos=1, alleles=("A", "C", "G"),
            tip_alleles={"A": "A", "B": "C", "C": "G"},
        )
        model = JC69()
        engine = Likelihood(model)
        L_poly = engine.log_likelihoods(_polytomy_tree(), [site])[0]
        L_bin = engine.log_likelihoods(_zero_length_binary_tree(), [site])[0]
        np.testing.assert_allclose(L_poly, L_bin, atol=1e-12)


class TestMultiAllelic:
    def test_three_alleles_engine_matches_brute_force(self):
        tree = _three_tip_tree()
        site = Site(
            chrom="1", pos=1, alleles=("A", "C", "G"),
            tip_alleles={"A": "A", "B": "C", "C": "G"},
        )
        model = JC69()
        engine = Likelihood(model)
        log_L = engine.log_likelihoods(tree, [site])[0]
        expected = _brute_force_log_likelihoods(tree, site, model)
        np.testing.assert_allclose(log_L, expected, atol=1e-12)


class TestMissingData:
    def test_missing_tip_equals_marginalized_likelihood(self):
        """A site with one missing tip should equal the sum over that tip's
        possible states of the same site with that tip observed."""
        tree = _three_tip_tree()
        model = JC69()
        engine = Likelihood(model)

        site_missing = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"A": "A", "B": None, "C": "C"},
        )
        L_missing = np.exp(engine.log_likelihoods(tree, [site_missing])[0])

        L_summed = np.zeros(model.n_states)
        for allele in model.states:
            site_obs = Site(
                chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"A": "A", "B": allele, "C": "C"},
            )
            L_summed += np.exp(engine.log_likelihoods(tree, [site_obs])[0])

        np.testing.assert_allclose(L_missing, L_summed, atol=1e-12)

    def test_unknown_allele_treated_as_missing(self):
        """An allele char outside the model alphabet (e.g. 'N') marginalises out."""
        tree = _three_tip_tree()
        model = JC69()
        engine = Likelihood(model)
        site_n = Site(
            chrom="1", pos=1, alleles=("A", "C", "N"),
            tip_alleles={"A": "A", "B": "N", "C": "C"},
        )
        site_none = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"A": "A", "B": None, "C": "C"},
        )
        L_n = engine.log_likelihoods(tree, [site_n])[0]
        L_none = engine.log_likelihoods(tree, [site_none])[0]
        np.testing.assert_allclose(L_n, L_none, atol=1e-12)

    def test_multi_char_indel_allele_treated_as_missing(self):
        """A multi-character (indel-like) tip allele should not crash the kernel
        and must marginalise identically to ``None`` (all-ones partial). This
        is the contract the :class:`~ancestree.sources.TskitSource` path
        relies on, it does not filter out indel sites upstream, so the
        likelihood kernel must handle them transparently."""
        tree = _three_tip_tree()
        model = JC69()
        engine = Likelihood(model)
        site_indel = Site(
            chrom="1", pos=1, alleles=("A", "AT"),
            tip_alleles={"A": "AT", "B": "A", "C": "A"},
        )
        site_none = Site(
            chrom="1", pos=1, alleles=("A",),
            tip_alleles={"A": None, "B": "A", "C": "A"},
        )
        L_indel = engine.log_likelihoods(tree, [site_indel])[0]
        L_none = engine.log_likelihoods(tree, [site_none])[0]
        assert np.all(np.isfinite(L_indel))
        np.testing.assert_allclose(L_indel, L_none, atol=1e-12)


class TestBatching:
    def test_batched_call_equals_loop(self):
        tree = _three_tip_tree()
        model = JC69()
        engine = Likelihood(model)

        sites = [
            Site("1", 1, ("A", "C"), {"A": "A", "B": "A", "C": "C"}),
            Site("1", 2, ("A", "G"), {"A": "A", "B": "G", "C": "G"}),
            Site("1", 3, ("C", "T"), {"A": "C", "B": "T", "C": "T"}),
            Site("1", 4, ("A",),     {"A": "A", "B": "A", "C": "A"}),  # invariant
        ]
        L_batch = engine.log_likelihoods(tree, sites)
        L_loop = np.empty_like(L_batch)
        for i, s in enumerate(sites):
            L_loop[i] = engine.log_likelihoods(tree, [s])[0]
        np.testing.assert_allclose(L_batch, L_loop, atol=1e-12)

    def test_empty_batch_returns_empty_array(self):
        tree = _three_tip_tree()
        engine = Likelihood(JC69())
        out = engine.log_likelihoods(tree, [])
        assert out.shape == (0, 4)


class TestInfiniteSitesParsimonyLimit:
    def test_map_root_matches_parsimony_pick(self):
        """With JC69 + uniform prior, the MAP root state on a
        short-branch tree should be the parsimony pick: the allele observed at
        the most tips (modulo tree shape, but on a star-polytomy with all
        branches equal, just majority)."""
        tree = _polytomy_tree(branch_lengths=(0.01, 0.01, 0.01))
        site = Site(
            chrom="1", pos=1, alleles=("A", "C"),
            tip_alleles={"A": "A", "B": "A", "C": "C"},
        )
        model = JC69()
        engine = Likelihood(model)
        log_L = engine.log_likelihoods(tree, [site])[0]
        # Uniform prior cancels; MAP is argmax of log_L.
        map_state = model.states[int(np.argmax(log_L))]
        assert map_state == "A", f"Expected A (parsimony majority), got {map_state}"


def _star(n_tips: int, time: float) -> tskit.TreeSequence:
    """A single star tree: ``n_tips`` samples all joined to one root."""
    tables = tskit.TableCollection(sequence_length=10.0)
    for _ in range(n_tips):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(flags=0, time=time)
    for tip in range(n_tips):
        tables.edges.add_row(left=0, right=10.0, parent=root, child=tip)
    tables.sort()
    return tables.tree_sequence()


def _star_site(n_tips: int):
    """Half the tips ``A``, half ``G``, so the two states are exchangeable."""
    return anc.Site(
        chrom="1", pos=1, alleles=("A", "G"),
        tip_alleles={f"t{i}": ("A" if i < n_tips // 2 else "G")
                     for i in range(n_tips)},
    )


def _exact_star_log_likelihoods(n_tips, time, model):
    """log P(tips | root=s) summed directly over the star's independent tips."""
    P = model.transition_probs(time, pi=model.stationary())
    index = {a: i for i, a in enumerate(anc.STATES)}
    return np.array([
        sum(np.log(P[s, index["A" if i < n_tips // 2 else "G"]])
            for i in range(n_tips))
        for s in range(4)
    ])


class TestWidePolytomyUnderflow:
    """The child-message product must not underflow at a wide polytomy."""

    @pytest.mark.parametrize("n_tips", [200, 300, 500, 2000])
    def test_supported_states_match_the_exact_likelihood(self, n_tips):
        model = JC69()
        tree = anc.TskitLocalTree(
            _star(n_tips, 0.01), position=1.0,
            sample_map={f"t{i}": i for i in range(n_tips)},
        )
        got = np.asarray(
            anc.Likelihood(model).log_likelihoods(tree, [_star_site(n_tips)])
        ).ravel()
        want = _exact_star_log_likelihoods(n_tips, 0.01, model)
        # A and G are the observed states; C and T sit ~1e-620 below the row
        # max and are legitimately unrepresentable once rescaled.
        assert got[0] == pytest.approx(want[0], rel=1e-9)
        assert got[2] == pytest.approx(want[2], rel=1e-9)

    def test_posterior_is_not_uniform(self):
        n_tips = 500
        tree = anc.TskitLocalTree(
            _star(n_tips, 0.01), position=1.0,
            sample_map={f"t{i}": i for i in range(n_tips)},
        )
        posterior = anc.Likelihood(JC69()).posterior(tree, _star_site(n_tips))
        assert posterior["A"] == pytest.approx(0.5, abs=1e-6)
        assert posterior["G"] == pytest.approx(0.5, abs=1e-6)

    def test_row_max_underflow_at_long_branches(self):
        """All four states drift down together. The row scale must absorb it."""
        n_tips, time = 600, 5.0
        model = JC69()
        tree = anc.TskitLocalTree(
            _star(n_tips, time), position=1.0,
            sample_map={f"t{i}": i for i in range(n_tips)},
        )
        got = np.asarray(
            anc.Likelihood(model).log_likelihoods(tree, [_star_site(n_tips)])
        ).ravel()
        want = _exact_star_log_likelihoods(n_tips, time, model)
        assert got == pytest.approx(want, rel=1e-9)


class TestInternalNodeSample:
    """A sample that is also an internal node must contribute its own state."""

    def test_internal_sample_state_is_not_dropped(self):
        tables = tskit.TableCollection(sequence_length=10.0)
        leaf_a = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        leaf_b = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        middle = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=1.0)
        root = tables.nodes.add_row(flags=0, time=2.0)
        for child in (leaf_a, leaf_b):
            tables.edges.add_row(left=0, right=10.0, parent=middle, child=child)
        tables.edges.add_row(left=0, right=10.0, parent=root, child=middle)
        tables.sort()
        tree = anc.TskitLocalTree(
            tables.tree_sequence(), position=1.0,
            sample_map={"a": leaf_a, "b": leaf_b, "m": middle},
        )
        likelihood = anc.Likelihood(JC69())
        base = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                       tip_alleles={"a": "A", "b": "A"})
        with_middle = anc.Site(chrom="1", pos=1, alleles=("A", "G"),
                              tip_alleles={"a": "A", "b": "A", "m": "G"})
        assert not np.allclose(
            np.asarray(likelihood.log_likelihoods(tree, [base])).ravel(),
            np.asarray(likelihood.log_likelihoods(tree, [with_middle])).ravel(),
        )


class TestNormaliseRejectsNaN:

    def test_nan_row_raises_instead_of_going_uniform(self):
        with pytest.raises(ValueError, match="NaN"):
            _normalise(np.array([[np.nan, 0.0, 0.0, 0.0]]), 4)

    def test_all_negative_infinity_row_still_falls_back_to_uniform(self):
        probabilities, n_bad = _normalise(np.full((1, 4), -np.inf), 4)
        assert n_bad == 1
        assert probabilities == pytest.approx(np.full((1, 4), 0.25))


class TestLikelihoodGuards:
    def test_repr_names_the_model(self):
        assert repr(Likelihood(JC69())) == "Likelihood(model=JC69)"

    def test_a_seed_of_the_wrong_shape_is_refused(self):
        ts, ids = panel_ts()
        tree = TskitLocalTree(ts, position=0.0)
        with pytest.raises(ValueError, match=r"expected \(1, 4\)"):
            Likelihood(JC69()).log_likelihoods(
                tree, [panel_site(ids)],
                node_seeds={ids["mrca"]: np.ones((2, 4))})

    def test_the_native_kernel_needs_a_positive_time_scale(self):
        ts, ids = panel_ts()
        tip_states = np.zeros((1, ts.num_samples), dtype=np.int8)
        with pytest.raises(ValueError, match="time_scale must be positive"):
            Likelihood(JC69()).log_likelihoods_tskit_native(
                ts.first(), tip_states, ts.samples(), ts.tables.nodes.time,
                time_scale=0.0)

    def test_a_split_past_the_end_of_its_edge_is_refused(self):
        engine = Likelihood(JC69())
        parent = np.array([-1, 0], dtype=np.int32)
        with pytest.raises(ValueError, match="exceeds the branch"):
            engine._split_edge(2.0, parent, np.zeros((2, 4, 4)), 2, 0,
                               path_child=1, edge_length=1.0)
