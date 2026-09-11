r"""Felsenstein's pruning algorithm, batched across sites.

:meth:`Likelihood.log_likelihoods() <ancestree.likelihood.Likelihood.log_likelihoods>` returns a
``(n_sites, n_states)`` array of
:math:`\log P(\text{observed tip alleles} \mid \text{root} = s)`
for every site and every candidate root state :math:`s`.

Polytomies, multi-allelic sites and missing tip data need no special
handling: any number of child messages multiply at an internal node, the
alphabet is the substitution model's full state set, and an unobserved tip
contributes an all-ones partial vector that marginalises out its state.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING


import numpy as np
from scipy.special import logsumexp


from ancestree._jit_kernel import (
    felsenstein_postorder_dense,
    postorder_csr_from_parents,
)
from ancestree.models import SubstitutionModel
from ancestree.sites import BaseComposition, Site
from ancestree.trees import Tree
from ancestree._repr import ReprMixin

if TYPE_CHECKING:
    from ancestree.focal import ResolvedFocal
    from ancestree.posterior import Posterior
    from ancestree.priors import StationaryPrior



def _normalise(log_post: np.ndarray, n_states: int) -> tuple[np.ndarray, int]:
    """Log-softmax rows, falling back to uniform where a row is degenerate.

    A row that is all ``-inf`` (a prior excluding every observed allele, or a
    likelihood that underflowed to zero everywhere) has ``logsumexp`` of
    ``-inf``, so the subtraction would give NaN. Those rows are made uniform.

    :param log_post: ``(N, S)`` un-normalised log-posterior.
    :param n_states: ``S``, the alphabet size used for the uniform fallback.
    :return: ``(N, S)`` posteriors summing to 1 per row, and how many rows fell
        back to uniform.
    :raises ValueError: If any row contains a NaN.
    """
    if np.isnan(log_post).any():
        rows = np.flatnonzero(np.isnan(log_post).any(axis=1))
        raise ValueError(
            f"log-posterior contains NaN at {rows.size} row(s), first at index "
            f"{int(rows[0])}; the prior, the base composition or a branch "
            f"length is not finite"
        )
    ls = logsumexp(log_post, axis=1, keepdims=True)
    bad = ~np.isfinite(ls).squeeze(axis=1)
    n_bad = int(bad.sum())
    if n_bad:
        log_post = log_post.copy()
        log_post[bad] = -np.log(n_states)
        ls[bad] = 0.0
    return np.exp(log_post - ls), n_bad


class Likelihood(ReprMixin):
    """Vectorised Felsenstein pruning over a fixed substitution model.

    :param model: The substitution model whose ``Q`` matrix drives per-branch
        transition probabilities.
    :param base_composition: Optional :class:`~ancestree.sites.BaseComposition`
        whose :attr:`BaseComposition.pi <ancestree.sites.BaseComposition.pi>` is forwarded to
        ``model.Q(pi=...)`` / ``model.transition_probs(t, pi=...)``. Read by
        the non-symmetric models (:class:`~ancestree.models.F81`,
        :class:`~ancestree.models.HKY`, :class:`~ancestree.models.GTR`) and
        ignored by the symmetric ones (:class:`~ancestree.models.JC69`,
        :class:`~ancestree.models.K2`). ``None`` (default) leaves every model
        on its own uniform-``π`` default.

    .. rubric:: The algorithm in three stages

    1. Build tip partials. For each tip, set the partial likelihood vector
       to a one-hot at the observed allele, or all-ones if the tip is missing
       or its allele is outside the model alphabet (marginalises the unknown
       state). Shape per tip is ``(B, S)`` where ``B = len(sites)``,
       ``S = n_states``.

    2. Propagate post-order. For every internal node ``v`` with children
       ``(c_i)`` and child branch lengths ``(t_i)``, set

       .. math::

          \\text{partial}[v][s] = \\prod_i \\Big( \\sum_{s'} P(t_i)[s, s'] \\cdot \\text{partial}[c_i][s'] \\Big)

       which factors per child as ``partial[c_i] @ P(t_i).T``, so a polytomy contributes
       one further factor. Then row-max-rescale to keep numbers in
       :math:`[0, 1]` and accumulate the log-scale.

    3. Finalise. ``log P(data | root = s) = log partial[root][s] + accumulated_log_scale``.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"model": type(self.model)}

    def __init__(
        self,
        model: SubstitutionModel,
        *,
        base_composition: "BaseComposition | None" = None,
    ) -> None:
        """Store the model for later :meth:`log_likelihoods`."""
        self.model = model
        self.base_composition = base_composition
        self._pi: np.ndarray | None = (
            base_composition.pi if base_composition is not None else None
        )
        # Allele → state-index lookup, derived from the model's alphabet.
        self._state_index: dict[str, int] = {s: i for i, s in enumerate(model.states)}

    # ----------------------------------------------------------------- public

    def _clean_seeds(
        self, node_seeds: "Mapping[int, np.ndarray] | None", B: int, S: int,
    ) -> "dict[int, np.ndarray]":
        """Validate ``node_seeds`` and neutralise rows that carry no mass.

        A row that is all-zero, or holds a non-finite entry, cannot be
        rescaled and would propagate NaN through the postorder. Such a row
        is replaced by all-ones, the same value a tip with a missing allele
        carries, which marginalises the node's state out.

        :param node_seeds: ``{node: (B, S)}`` as passed by the caller, or
            ``None``.
        :param B: Batch size the seeds must match.
        :param S: Alphabet size the seeds must match.
        :return: ``{node: (B, S)}`` with every row finite and non-degenerate.
        :raises ValueError: If a seed has the wrong shape.
        """
        cleaned: dict[int, np.ndarray] = {}
        n_blank = 0
        for node, matrix in (node_seeds or {}).items():
            arr = np.array(matrix, dtype=float)
            if arr.shape != (B, S):
                raise ValueError(
                    f"node_seeds[{node}] has shape {arr.shape}, expected "
                    f"{(B, S)}"
                )
            bad = ~np.isfinite(arr).all(axis=1) | (arr.sum(axis=1) <= 0.0)
            if bad.any():
                n_blank += int(bad.sum())
                arr[bad, :] = 1.0
            cleaned[node] = arr
        if n_blank:
            self._log.warning(
                "%d seeded node row(s) carried no probability mass and were "
                "marginalised out; check the prior or the clade's alleles",
                n_blank,
            )
        return cleaned

    def log_likelihoods(
        self,
        tree: Tree,
        sites: Sequence[Site],
        node_seeds: "Mapping[int, np.ndarray] | None" = None,
    ) -> np.ndarray:
        """Return ``(n_sites, n_states)`` of ``log P(tip data | root = s)``.

        All sites must be evaluated against the same :class:`~ancestree.trees.Tree`. Group sites
        by their local tree upstream (see :class:`~ancestree.inference.ARGBasedInference`) and call
        this once per group. The per-site work is fully vectorised along
        the batch axis.

        :param tree: The :class:`~ancestree.trees.Tree` to evaluate.
        :param sites: Sequence of :class:`~ancestree.sites.Site` records sharing ``tree``.
        :param node_seeds: Optional ``{node: (len(sites), n_states)}`` giving a
            likelihood vector to seed at a node that is not a tip, replacing
            its all-ones row. Used to enter a whole
            collapsed clade as a single observation at its ancestor.
        :return: Array of shape ``(len(sites), model.n_states)`` with
            ``log P(observed tips | root = s)``. Empty ``sites`` returns
            a ``(0, n_states)`` array.
        """
        if not sites:
            return np.empty((0, self.model.n_states), dtype=float)

        return self._log_likelihoods_jit(tree, sites, node_seeds)

    def posterior(
        self,
        tree: "Tree",
        site: "Site",
        prior: "StationaryPrior | None" = None,
        node_seeds: "Mapping[int, np.ndarray] | None" = None,
    ) -> "Posterior":
        """Score one site against one tree and normalise to a posterior.

        Runs the kernel, adds the root-state log-prior and normalises, which is
        the whole path from a tree and a site to a distribution over candidate
        ancestral alleles. Use it to evaluate a hand-built tree without
        constructing an :class:`~ancestree.inference.Inference`. The inference
        classes do the same thing per site over a whole dataset.

        On a tree whose ingroup is collapsed to a single node with no tips
        beneath it, such as an
        :class:`~ancestree.trees.OutgroupLadderTree`, the ingroup's alleles
        reach the kernel only through ``node_seeds``. Omitting them there
        scores the outgroups alone.

        :param tree: Any :class:`~ancestree.trees.Tree` whose tip ids match the
            keys of :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
        :param site: The :class:`~ancestree.sites.Site` to score.
        :param prior: Root-state prior. ``None`` (default) uses a
            :class:`~ancestree.priors.StationaryPrior` over this instance's
            model and base composition.
        :param node_seeds: Partial likelihoods to inject at internal nodes, as
            ``{node_id: length-n_states array}``. For a ladder tree this is
            ``{tree.ingroup_mrca: exp(weight.log_probs([site]))}`` for an
            :class:`~ancestree.priors.StationaryPrior`.
        :return: A :class:`~ancestree.posterior.Posterior` summing to 1.
        """
        from ancestree.posterior import Posterior
        from ancestree.priors import StationaryPrior

        if prior is None:
            prior = StationaryPrior(self.model, self.base_composition)
        log_L = self.log_likelihoods(tree, [site], node_seeds=node_seeds)[0]
        log_post = log_L + prior.log_probs([site])[0]
        values, _ = _normalise(log_post[None, :], self.model.n_states)
        return Posterior(alleles=self.model.states, values=values[0])

    def log_likelihoods_tskit_native(
        self,
        tskit_tree,
        tip_states: np.ndarray,
        sample_nodes: np.ndarray,
        node_times: np.ndarray,
        time_scale: float,
        *,
        id_map_scratch: "np.ndarray | None" = None,
        focal: "ResolvedFocal | None" = None,
    ) -> np.ndarray:
        """ARG-mode fast path: pack dense arrays directly from a :class:`tskit.Tree`.

        Reads tskit's numpy buffers (post-order, parent array, node times) and
        the per-site tip states as int8 indices, then dispatches to the
        compiled ``felsenstein_postorder_dense`` kernel the generic path uses.

        Dense slots are numbered by tskit node id, so the order in which the
        child messages multiply and the per-node log-scales accumulate is a
        function of the local tree alone. The result is bit-identical however
        the walk reached the tree, and therefore across worker counts.

        :param tskit_tree: A :class:`tskit.Tree` positioned at the desired
            interval.
        :param tip_states: ``(B, n_samples) int8`` state index per batch row
            and sample, in ``{0, 1, 2, 3}`` (A, C, G, T) or ``-1`` for missing
            data. Column order matches ``sample_nodes``.
        :param sample_nodes: ``(n_samples,) int32`` tskit node ids of the
            samples (``ts.samples()``).
        :param node_times: ``(n_total_nodes,) float64`` node times,
            ``ts.tables.nodes.time``.
        :param time_scale: Multiplier applied to ``parent_time - child_time``
            to obtain the kernel's branch length (typically ``mu``).
        :param focal: Optional node to report at, from
            :meth:`FocalNode.resolve() <ancestree.focal.FocalNode.resolve>`.
            ``None`` (default) reads at the tree's own root. A focal node with
            ``tau > 0`` splits the edge above it, which adds one dense slot.
        :param id_map_scratch: Optional reusable ``int32`` buffer of length
            ``tskit_tree.parent_array.shape[0]`` filled with ``-1``, holding
            the original-node-id to dense-index lookup. Only the active slice
            is written, and it is reset to ``-1`` before returning. Allocated
            internally when ``None``.
        :return: ``(B, n_states)`` of ``log P(observed tips | root = s)``.
        :raises ValueError: If the tree has more than one root, if
            ``time_scale`` is not positive, or if ``focal`` names a node
            this tree does not contain.
        """
        S = self.model.n_states
        B = int(tip_states.shape[0])

        if tskit_tree.num_roots != 1:
            raise ValueError(
                "log_likelihoods_tskit_native requires a single-root tree; got "
                f"num_roots={tskit_tree.num_roots}. Marginalise multi-root "
                "segments per-root before calling the native kernel."
            )
        if not float(time_scale) > 0.0:
            raise ValueError(
                f"time_scale must be positive, got {time_scale}"
            )

        # Active nodes, ascending node id. The dense numbering, and with it
        # the CSR child order and the traversal order below, are then a
        # function of the tree alone and not of how it was reached.
        nodes = np.sort(np.asarray(tskit_tree.postorder(), dtype=np.int32))
        n_active = nodes.shape[0]

        # Map original node ids to dense [0, n_active). Reused across trees,
        # sized by the ARG's node count.
        if id_map_scratch is None:
            id_map = np.full(int(tskit_tree.parent_array.shape[0]), -1, dtype=np.int32)
        else:
            id_map = id_map_scratch
        id_map[nodes] = np.arange(n_active, dtype=np.int32)

        root_dense = int(id_map[int(tskit_tree.root)])

        # Per-node parent (dense) and branch length (scaled).
        parent_arr = tskit_tree.parent_array
        parent_orig = parent_arr[nodes]  # (n_active,) original ids. Root has tskit.NULL (-1)
        has_parent = parent_orig >= 0
        parent_dense = np.where(has_parent, id_map[parent_orig], -1).astype(np.int32)
        branch_t = np.where(
            has_parent,
            (node_times[parent_orig] - node_times[nodes]) * time_scale,
            0.0,
        )

        # Per-edge transition matrices in one vectorised call.
        P_per_node = np.ascontiguousarray(
            self.model.transition_probs(
                branch_t, pi=self._pi,
            ),
            dtype=np.float64,
        )

        # Re-root at the focal node (see _reroot_dense).
        if focal is not None and int(focal.node) != int(tskit_tree.root):
            parent_dense, P_per_node, n_active, root_dense = self._reroot_dense(
                tskit_tree, focal, id_map, parent_dense, P_per_node,
                branch_t, n_active, time_scale,
            )
        elif focal is not None and focal.tau > 0.0:
            parent_dense, P_per_node, n_active, root_dense = self._split_edge(
                float(focal.tau) * time_scale, parent_dense, P_per_node,
                n_active, root_dense, path_child=None,
            )

        # CSR children and the traversal that consumes them, from one walk of
        # the parent array.
        c_offsets, children_flat, postorder_internal = postorder_csr_from_parents(
            parent_dense, root_dense, n_active,
        )

        # Tip partials: (n_active, B, S). Initialise to all-ones so
        # internal nodes and samples-not-in-this-tree marginalise out.
        partials = np.ones((n_active, B, S), dtype=np.float64)
        sample_dense = id_map[sample_nodes]  # (n_samples,), -1 if not in tree
        # Vectorised one-hot population: pick (b, s) pairs with observed
        # state and a tip in this tree, zero their rows, set the observed
        # column to 1.
        valid_mask = tip_states >= 0
        b_idx, s_idx = np.where(valid_mask)
        if b_idx.size:
            d_idx = sample_dense[s_idx]
            ok = d_idx >= 0
            if ok.any():
                b_idx, s_idx, d_idx = b_idx[ok], s_idx[ok], d_idx[ok]
                states = tip_states[b_idx, s_idx].astype(np.intp)
                partials[d_idx, b_idx, :] = 0.0
                partials[d_idx, b_idx, states] = 1.0

        root_partial, scale_log = felsenstein_postorder_dense(
            B, S, root_dense,
            postorder_internal, c_offsets, children_flat,
            P_per_node, partials,
        )
        # Reset the written slice so the scratch buffer is all -1 again.
        id_map[nodes] = -1
        with np.errstate(divide="ignore"):
            return np.log(root_partial) + scale_log[:, None]

    def _log_likelihoods_jit(
        self,
        tree: Tree,
        sites: Sequence[Site],
        node_seeds: "Mapping[int, np.ndarray] | None" = None,
    ) -> np.ndarray:
        """Numba-JIT path for :meth:`log_likelihoods`.

        Builds dense flat arrays from the tree topology and the per-site
        tip data, then dispatches to the compiled
        ``felsenstein_postorder_dense``.

        The dense-array packing is per-tree-only, not per-site, so the
        cost amortises naturally across batched sites. The only
        per-site work is filling in the tip partials.
        """
        S = self.model.n_states
        B = len(sites)
        tip_alleles_by_site = [s.tip_alleles for s in sites]
        node_to_sample = self._build_node_to_sample(tree, tip_alleles_by_site)

        # Map original tree node ids → dense [0..n_nodes) in ascending node id,
        # so the CSR child order and the traversal order are a function of the
        # tree alone.
        nodes_orig = sorted(tree.postorder())
        n_nodes = len(nodes_orig)
        id_map: dict[int, int] = {orig: dense for dense, orig in enumerate(nodes_orig)}
        root_dense = id_map[tree.root]

        parent_dense = np.full(n_nodes, -1, dtype=np.int32)
        for dense, orig in enumerate(nodes_orig):
            for k in tree.children(orig):
                parent_dense[id_map[k]] = dense
        c_offsets, c_flat, postorder_internal = postorder_csr_from_parents(
            parent_dense, root_dense, n_nodes,
        )

        # Per-edge transition matrices: P_per_node[d] = P on the edge
        # entering node d. Root's row is unused. One batched
        # transition_probs call for all edges.
        ts_scale = tree.time_scale
        branch_t = np.zeros(n_nodes, dtype=float)
        for dense, orig in enumerate(nodes_orig):
            if dense == root_dense:
                continue
            branch_t[dense] = tree.branch_length(orig) * ts_scale
        P_all = self.model.transition_probs(
            branch_t, pi=self._pi,
        )  # (n_nodes, S, S)
        P_per_node = np.ascontiguousarray(P_all, dtype=np.float64)

        # Tip partials. Internal nodes start at all-ones and are
        # overwritten by the kernel in post-order.
        partials = np.ones((n_nodes, B, S), dtype=np.float64)
        state_idx = self._state_index
        for dense, orig in enumerate(nodes_orig):
            sample = node_to_sample.get(orig)
            if sample is None:
                # No observation here. An internal node is filled by the
                # traversal and a bare tip marginalises out via its all-ones row.
                continue
            for b in range(B):
                a = Site.canonical(tip_alleles_by_site[b].get(sample))
                if a is None:
                    continue  # missing → all-ones row
                # one-hot at the observed allele
                partials[dense, b, :] = 0.0
                partials[dense, b, state_idx[a]] = 1.0

        for node, matrix in self._clean_seeds(node_seeds, B, S).items():
            partials[id_map[node], :, :] = matrix

        root_partial, scale_log = felsenstein_postorder_dense(
            B, S, root_dense,
            postorder_internal, c_offsets, c_flat,
            P_per_node, partials,
        )
        with np.errstate(divide="ignore"):
            return np.log(root_partial) + scale_log[:, None]

    def _reroot_dense(
        self, tskit_tree, focal, id_map, parent_dense, P_per_node, branch_t,
        n_active: int, time_scale: float,
    ) -> "tuple[np.ndarray, np.ndarray, int, int]":
        """Reverse the parent pointers along the focal-to-root path.

        Each edge keeps its own transition matrix. Reversal only changes which
        node that matrix hangs under, so ``P_per_node`` is permuted, not
        recomputed. The old root is left in place as a unary node, which the
        post-order traversal carries through unchanged.

        :param tskit_tree: The tree being packed.
        :param focal: The resolved focal node.
        :param id_map: Original-to-dense node id map.
        :param parent_dense: Dense parent ids. Not modified in place.
        :param P_per_node: Per-node transition matrices.
        :param branch_t: Per-node branch lengths in substitution units.
        :param time_scale: Multiplier taking ``focal.tau`` from the tree's own
            branch units into substitution units.
        :param n_active: Number of dense nodes.
        :return: ``(parent_dense, P_per_node, n_active, root_dense)``.
        """
        import tskit

        path = [int(focal.node)]
        node = int(focal.node)
        while True:
            parent = int(tskit_tree.parent(node))
            if parent == tskit.NULL:
                break
            path.append(parent)
            node = parent
        dense_path = id_map[np.asarray(path, dtype=np.int32)]
        # A focal node absent from this tree maps to -1.
        if dense_path[0] < 0:
            raise ValueError(
                f"focal node {int(focal.node)} is not a node of this tree; "
                f"resolve the focal node against the tree being scored"
            )

        parent_dense = parent_dense.copy()
        P_per_node = P_per_node.copy()
        # Each q_i takes q_{i-1} as its parent, and the edge that ran into
        # q_{i-1} now runs into q_i, carrying its matrix with it.
        parent_dense[dense_path[1:]] = dense_path[:-1]
        parent_dense[dense_path[0]] = -1
        P_per_node[dense_path[1:]] = P_per_node[dense_path[:-1]]

        root_dense = int(dense_path[0])
        if focal.tau > 0.0:
            parent_dense, P_per_node, n_active, root_dense = self._split_edge(
                float(focal.tau) * time_scale, parent_dense, P_per_node,
                n_active, root_dense,
                # Always present: this method runs only for a focal node that
                # is not the root, so the path reaches at least its parent.
                path_child=int(dense_path[1]),
                edge_length=float(branch_t[dense_path[0]]),
            )
        return parent_dense, P_per_node, n_active, root_dense

    def _split_edge(
        self, tau: float, parent_dense, P_per_node, n_active: int,
        root_dense: int, *, path_child: int | None,
        edge_length: float | None = None,
    ) -> "tuple[np.ndarray, np.ndarray, int, int]":
        """Insert the point ``tau`` above the focal node as one extra dense node.

        A point part-way along an edge has to be a node for a rooted recursion
        to start there. The new node takes two children: the focal subtree at
        distance ``tau``, and whatever lay above the focal node at the
        remaining ``L - tau``. Above the tree's own root there is no second
        side, so the new node has one child and the posterior there is the root
        posterior smoothed by ``P(tau)``.

        The split lives in the packed arrays only. No tskit table and no
        :class:`~ancestree.trees.Tree` is rebuilt.

        :param tau: Distance above the focal node, in substitution units.
        :param parent_dense: Dense parent ids.
        :param P_per_node: Per-node transition matrices.
        :param n_active: Number of dense nodes before the split.
        :param root_dense: Dense id of the focal node, currently the root.
        :param path_child: Dense id of the node on the far side of the split
            edge, or ``None`` when extending above the tree's own root.
        :param edge_length: Length of the edge being split.
        :return: ``(parent_dense, P_per_node, n_active, root_dense)``.
        :raises ValueError: If ``tau`` runs past the end of that edge.
        """
        if edge_length is not None and tau > edge_length:
            raise ValueError(
                f"tau={tau} exceeds the branch above the focal node "
                f"({edge_length}); only above the tree's own root may a point "
                f"sit off an edge"
            )
        new = n_active
        parent_dense = np.append(parent_dense, -1).astype(np.int32)
        P_per_node = np.concatenate(
            [P_per_node, np.zeros((1,) + P_per_node.shape[1:])], axis=0,
        )

        # The focal subtree hangs below the new point at distance tau.
        parent_dense[root_dense] = new
        P_per_node[root_dense] = self.model.transition_probs(
            tau, pi=self._pi,
        )
        if path_child is not None:
            # The far side of the edge moves off the focal node onto the point.
            parent_dense[path_child] = new
            P_per_node[path_child] = self.model.transition_probs(
                max(float(edge_length or 0.0) - tau, 0.0),
                pi=self._pi,
            )
        return parent_dense, P_per_node, n_active + 1, new

    def _build_node_to_sample(
        self,
        tree: Tree,
        tip_alleles_by_site: Sequence[Mapping[str, str | None]],
    ) -> dict[int, str]:
        """Invert :meth:`Tree.tip_for_sample() <ancestree.trees.Tree.tip_for_sample>`
        over the samples in this batch.

        Only tips carrying tip-allele observations somewhere in this batch
        need an entry. Samples absent from every site's
        :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>` are
        skipped, and the tip is treated as missing.

        :param tree: The :class:`~ancestree.trees.Tree` being evaluated.
        :param tip_alleles_by_site: Per-site
            :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>` dicts.
        :return: Mapping from tip node id to caller-facing sample id.
        """
        all_samples: set[str] = set()
        for ta in tip_alleles_by_site:
            all_samples.update(ta.keys())

        node_to_sample: dict[int, str] = {}
        for sample in all_samples:
            tip = tree.tip_for_sample(sample)
            if tip is not None:
                node_to_sample[tip] = sample
        return node_to_sample
