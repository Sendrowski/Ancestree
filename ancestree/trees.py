r"""The trees the likelihood kernel runs on, and the node it reports at.

The :class:`~ancestree.trees.Tree` ABC is the surface the likelihood kernel
sees: children, branch lengths, post-order traversal, and a mapping from
sample names to tip nodes.

Backends:

- :class:`~ancestree.trees.TskitLocalTree`: a :class:`tskit.Tree` at a fixed
  genomic position (single-rooted, since multi-root segments raise).
- :class:`~ancestree.trees.OutgroupLadderTree`: the ingroup-polytomy plus
  outgroup-ladder topology, with :math:`2n_\mathrm{out} - 1` branch rates
  fitted by maximum likelihood in fixed-tree mode.
  :meth:`OutgroupLadderTree.at_focal` moves the readout anywhere along the
  backbone from the ingroup MRCA to the deepest join, and
  :meth:`OutgroupLadderTree.as_deep_rooted` is its far end.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from ancestree._repr import ReprMixin
from ancestree.sites import Site

if TYPE_CHECKING:
    import tskit
    from ancestree.focal import FocalNode
    from ancestree.posterior import Posterior


class Tree(ReprMixin, ABC):
    """A rooted tree with branch lengths and a sample-id ↔ tip-node mapping.

    Node ids are integers, not necessarily contiguous. The only invariants
    are that :attr:`root` is one of them, that :meth:`postorder` visits every
    node exactly once with children before parents, and that
    :meth:`children` of :attr:`root` returns the root's immediate children.

    Branch lengths are interpreted as substitution-model time after
    scaling by :attr:`time_scale`.
    """

    _time_scale: float = 1.0

    @property
    def time_scale(self) -> float:
        """Scalar multiplier applied to every branch length by the kernel.

        Default ``1.0`` (branch lengths in expected substitutions per site).
        ARG-mode inference sets this to the per-site, per-generation
        substitution rate ``μ``, so the kernel sees ``Q · (μ · branch_length)``
        in expected substitutions per site without rebuilding the tree.

        :return: Scaling factor. Defaults to ``1.0``.
        """
        return float(self._time_scale)

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        """Set :attr:`time_scale`. Must be strictly positive.

        :raises ValueError: If ``value`` is not strictly positive.
        """
        v = float(value)
        if v <= 0:
            raise ValueError(f"time_scale must be positive, got {v}")
        self._time_scale = v

    @property
    def n_nodes(self) -> int:
        """Total number of nodes (tips + internals + root).

        :return: Node count for this tree.
        """
        return len(self.postorder())

    @property
    @abstractmethod
    def root(self) -> int:
        """Node id of the tree's root.

        :return: Root node id.
        """

    @abstractmethod
    def children(self, node: int) -> Sequence[int]:
        """Immediate children of ``node``.

        :param node: Node id whose children to return.
        :return: Sequence of child node ids. Empty for tips.
        """

    @abstractmethod
    def branch_length(self, node: int) -> float:
        """Length of the branch from ``node`` to its parent. Zero at the root.

        :param node: Node id.
        :return: Branch length in substitution-model time units.
        """

    @abstractmethod
    def postorder(self) -> Sequence[int]:
        """Iterate nodes in post-order (every node after all its descendants).

        :return: Sequence of node ids in post-order.
        """

    @abstractmethod
    def tip_for_sample(self, sample_id: str) -> int | None:
        """Return the tip node id for a sample.

        :param sample_id: Caller-facing sample identifier.
        :return: Tip node id, or ``None`` if the sample is not in this tree.
        """

    @property
    def tskit_tree(self) -> "tskit.Tree | None":
        """The :class:`tskit.Tree` behind this one, if there is one.

        ``None`` for trees not backed by tskit.
        """
        return None

    def sample_for_tip(self, node: int) -> "str | None":
        """Inverse of :meth:`tip_for_sample`, for naming tips in output.

        Returns ``None``. Subclasses override this to name tips.

        :param node: A tip node id.
        :return: Caller-facing sample id, or ``None`` when the tip is unnamed.
        """
        return None

    @abstractmethod
    def n_tips(self) -> int:
        """Number of tip nodes (samples) in the tree.

        :return: Tip count.
        """

    def draw_text(
        self,
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        show_branch_lengths: bool = True,
    ) -> str:
        """Render the tree as ASCII.

        Subclasses with a backend-specific renderer (e.g. :class:`~ancestree.trees.TskitLocalTree`
        uses :meth:`tskit.Tree.draw_text() <tskit.Tree.draw_text>`) override
        this. The default implementation is an indented post-order printout.

        :param site: If given, tip labels are formatted as
            ``sample_id=allele`` from
            :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`. Tips
            with missing data show ``?``.
        :param posterior: If given, the root label is appended with the
            :class:`~ancestree.posterior.Posterior`'s MAP allele and its
            probability.
        :param show_branch_lengths: Append ``[bl=...]`` to every non-root
            label.
        :return: ASCII-rendered tree.
        """
        from ancestree._viz import TreeRenderer
        return TreeRenderer.draw_text(
            self, site=site, posterior=posterior,
            show_branch_lengths=show_branch_lengths,
        )

    @property
    def ingroup_mrca(self) -> "int | None":
        """Node the ingroup's likelihood vector is seeded on.

        ``None`` for trees that define no ingroup. Subclasses override this.
        """
        return None

    @property
    def ingroup_samples(self) -> tuple[str, ...]:
        """Sample ids forming the ingroup, empty where the tree defines none."""
        return ()

    @property
    def outgroup_samples(self) -> tuple[str, ...]:
        """Sample ids forming the outgroup, empty where the tree defines none."""
        return ()

    def at_focal(self, focal: "FocalNode | str | None" = None) -> "Tree | None":
        """A view of this tree read at ``focal``.

        Returns ``None``. Subclasses that can resolve a focal node override
        this; the caller falls back to a tskit view or refuses.

        :param focal: The :class:`~ancestree.focal.FocalNode` to read at, or
            ``None`` for the subclass's own default.
        :return: The re-rooted view, or ``None``.
        """
        return None

    def as_deep_rooted(self) -> "Tree | None":
        """A view of this tree read at its deepest join.

        Returns ``None``. Subclasses override this where a deep rooting is
        defined.

        :return: The re-rooted view, or ``None``.
        """
        return None

    def node_name(self, node: int) -> str | None:
        """Backend name for ``node``, shown by every renderer.

        :param node: Node id in this tree.
        :return: The name, or ``None`` to fall back to the tip's sample id and
            then to the node id.
        """
        return None

    def draw_svg(
        self,
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        **kwargs,
    ) -> str:
        """Render the tree as SVG.

        :param site: Optional :class:`~ancestree.sites.Site` for tip-allele overlay.
        :param posterior: Optional :class:`~ancestree.posterior.Posterior`
            for the root MAP overlay.
        :param kwargs: Passed through to
            :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`.
        :return: The SVG document as a string.
        """
        from ancestree._viz import TreeRenderer
        return TreeRenderer.draw_svg(
            self, site=site, posterior=posterior, **kwargs,
        )


class TskitLocalTree(Tree):
    """A :class:`~ancestree.trees.Tree` backed by a :class:`tskit.Tree` at a fixed genomic position.

    :param ts: The source tree sequence.
    :param position: A genomic position (in tskit coordinates) inside the
        segment whose local tree is wanted.
    :param sample_map: Mapping from caller-facing sample identifiers
        (strings, e.g. ``"n0"`` or VCF sample names split by ploidy) to
        tskit sample node ids (integers in ``range(ts.num_samples)``). If
        ``None``, the map is derived from the tree sequence's individual names
        (:meth:`TskitLocalTree.sample_map_from_individuals() <ancestree.trees.TskitLocalTree.sample_map_from_individuals>`),
        falling back to the string ``str(i)`` for each tskit sample id ``i``,
        convenient for pure-ARG workflows that do not go through VCF.
    :param root: Optional explicit root id for multi-rooted local trees
        (partial coalescence, common in ``tsinfer`` ARGs in
        low-diversity regions). When given, the wrapper is restricted to
        that root's subtree. Samples outside the subtree are reported as
        missing by :meth:`tip_for_sample` and excluded from the
        post-order traversal. ``None`` (default) requires a
        single-rooted local tree.
    :raises ValueError: If the local tree has more than one root and
        no explicit ``root`` was supplied (not all samples have
        coalesced within this segment).
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_tips": self.n_tips(), "root": self.root}

    def __init__(
        self,
        ts: "tskit.TreeSequence",
        position: float,
        sample_map: Mapping[str, int] | None = None,
        *,
        root: int | None = None,
    ) -> None:
        """Resolve the local tree at ``position`` via ``ts.at(position)``."""
        self._init(ts, ts.at(position), position, sample_map, root=root)

    @staticmethod
    def restrict(
        ts: "tskit.TreeSequence", sample_map: Mapping[str, int], *,
        keep_node_ids: bool = False,
    ) -> "tuple[tskit.TreeSequence, dict[str, int]]":
        """``ts`` simplified to the nodes of ``sample_map``, every site kept.

        :param ts: The tree sequence to restrict.
        :param sample_map: ``{name: node}`` for the samples to keep.
        :param keep_node_ids: Keep every node id, so the kept samples keep
            their names. Otherwise the kept nodes are renumbered in their
            original order.
        :return: The restricted tree sequence and ``sample_map`` in its
            numbering, both unchanged where ``sample_map`` covers every sample.
        """
        nodes = sorted(int(n) for n in sample_map.values())
        if len(nodes) >= ts.num_samples:
            return ts, dict(sample_map)
        restricted = ts.simplify(samples=nodes, filter_sites=False,
                                 filter_nodes=not keep_node_ids,
                                 record_provenance=False)
        if keep_node_ids:
            return restricted, dict(sample_map)
        rank = {n: i for i, n in enumerate(nodes)}
        return restricted, {s: rank[int(n)] for s, n in sample_map.items()}

    @staticmethod
    def default_sample_map(ts: "tskit.TreeSequence") -> dict[str, int]:
        """The individual-metadata map, or tskit node ids as their own names.

        :param ts: The source tree sequence.
        :return: ``{name: node_id}`` per haplotype.
        """
        return (TskitLocalTree.sample_map_from_individuals(ts)
                or {str(int(s)): int(s) for s in ts.samples()})

    @staticmethod
    def sample_map_from_individuals(
        ts: "tskit.TreeSequence",
    ) -> dict[str, int] | None:
        """Derive a ``{sample_name: tskit_node_id}`` map from individual metadata.

        Returns the mapping only when every sample's individual carries a
        ``name`` key in its metadata, else ``None``, leaving the caller
        free to fall back on the default ``str(node_id)`` naming.

        An individual owning more than one node (e.g. a diploid) gets one
        entry per sample node, named ``{name}_h<k>`` by the node's place
        among the individual's nodes, matching the haplotype naming the VCF /
        VCF-Zarr sources use, so the same diploid panel reads identically
        across all sources.

        :param ts: The source tree sequence.
        :return: ``{name: node_id}`` per haplotype, or ``None`` if any
            sample's individual has no ``name`` metadata.
        :raises ValueError: If two distinct individuals yield the same
            haplotype key, i.e. their ``name`` metadata is not unique.
        """
        out: dict[str, int] = {}
        for s in ts.samples():
            ind = ts.node(int(s)).individual
            if ind < 0:
                return None
            individual = ts.individual(ind)
            meta = individual.metadata or {}
            name = meta.get("name") if isinstance(meta, dict) else None
            if not name:
                return None
            nodes = list(individual.nodes)
            key = (str(name) if len(nodes) == 1
                   else f"{name}_h{nodes.index(int(s))}")
            if key in out:
                raise ValueError(
                    f"duplicate haplotype name {key!r} from distinct individuals "
                    f"in the tree sequence; individual `name` metadata must be "
                    f"unique for a sample map"
                )
            out[key] = int(s)
        return out

    @classmethod
    def from_tskit_tree(
        cls,
        tree: "tskit.Tree",
        sample_map: Mapping[str, int] | None = None,
        *,
        root: int | None = None,
    ) -> "TskitLocalTree":
        """Wrap an existing :class:`tskit.Tree`.

        Avoids the seek cost of ``ts.at(position)`` for a caller walking
        ``ts.trees()``.

        :param tree: An existing :class:`tskit.Tree` to wrap.
        :param sample_map: Same semantics as in ``__init__``.
        :param root: Optional explicit root for multi-rooted local trees.
            See ``__init__``.
        :return: A :class:`~ancestree.trees.TskitLocalTree` wrapping ``tree`` directly.
        """
        obj = cls.__new__(cls)
        obj._init(tree.tree_sequence, tree, tree.interval.left, sample_map, root=root)
        return obj

    @classmethod
    def from_newick(cls, newick_str: str) -> "TskitLocalTree":
        """Build a :class:`~ancestree.trees.TskitLocalTree` from a Newick string.

        Leaf names become caller-facing sample ids in the resulting
        ``sample_map``, and the wrapped tree sequence carries no sites or
        mutations. Intended for hand-built example trees, not production ARGs.

        :param newick_str: Newick string with branch lengths on every
            non-root edge. Polytomies are permitted. Leaves are placed at time
            0 and every internal node takes the depth of its deepest child
            path, so a shorter sibling's branch is lengthened to match and the
            resulting tree is ultrametric whatever the input.
        :return: A :class:`~ancestree.trees.TskitLocalTree` wrapping the
            single local tree of the constructed tree sequence.
        :raises ImportError: If the ``newick`` package is not installed.
        :raises ValueError: If ``newick_str`` parses to multiple roots, or a
            branch length leaves a parent no later than one of its children.
        """
        try:
            import newick as _newick
        except ImportError as exc:
            raise ImportError(
                "TskitLocalTree.from_newick requires the `newick` "
                "package. Install with `pip install newick`."
            ) from exc
        import tskit

        forest = _newick.loads(newick_str)
        if len(forest) != 1:
            raise ValueError(
                f"TskitLocalTree.from_newick expected a single rooted "
                f"tree; got {len(forest)} top-level entries."
            )
        root = forest[0]
        leaves = [n for n in root.walk() if n.is_leaf]
        internals = [n for n in root.walk() if not n.is_leaf]

        # Assign times: leaves at 0. Internals at max(child_time + child_branch_length).
        time_at: dict[int, float] = {}
        def _assign_times(n) -> float:
            """Node height above the leaves, recording it as it recurses."""
            if n.is_leaf:
                time_at[id(n)] = 0.0
                return 0.0
            t = max(_assign_times(c) + float(c.length or 0.0) for c in n.descendants)
            time_at[id(n)] = t
            return t
        _assign_times(root)

        # tskit requires a strictly greater time on every parent, which a
        # zero-length branch violates unless a sibling lifts the parent clear.
        for n in internals:
            for c in n.descendants:
                if time_at[id(n)] <= time_at[id(c)]:
                    raise ValueError(
                        f"TskitLocalTree.from_newick needs a positive branch "
                        f"length on every non-root edge; the edge into "
                        f"{c.name or '<unnamed>'} leaves its parent at the "
                        f"same time ({time_at[id(n)]})"
                    )

        tables = tskit.TableCollection(sequence_length=1.0)
        node_id: dict[int, int] = {}
        for leaf in leaves:
            node_id[id(leaf)] = tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE, time=0.0,
            )
        for n in internals:
            node_id[id(n)] = tables.nodes.add_row(flags=0, time=time_at[id(n)])
        def _add_edges(n) -> None:
            """Add one edge per parent-child pair, depth first."""
            for c in n.descendants:
                tables.edges.add_row(
                    left=0.0, right=1.0,
                    parent=node_id[id(n)], child=node_id[id(c)],
                )
                _add_edges(c)
        _add_edges(root)
        tables.sort()
        ts = tables.tree_sequence()
        return cls(
            ts, position=0.0,
            sample_map={leaf.name: node_id[id(leaf)] for leaf in leaves},
        )

    def _init(
        self,
        ts: "tskit.TreeSequence",
        tree: "tskit.Tree",
        position: float,
        sample_map: Mapping[str, int] | None,
        *,
        root: int | None = None,
    ) -> None:
        """Shared init used by both ``__init__`` and :meth:`from_tskit_tree`."""
        self._tree = tree

        if root is None:
            if tree.num_roots != 1:
                raise ValueError(
                    f"TskitLocalTree at position {position} has "
                    f"num_roots={tree.num_roots}; pass `root=<id>` to "
                    f"restrict the wrapper to a single root's subtree, "
                    f"or use ARGBasedInference which iterates roots "
                    f"automatically."
                )
            self._root: int = int(tree.root)
        else:
            if root not in tree.roots:
                raise ValueError(
                    f"root={root} is not a root of the local tree at "
                    f"position {position}; tree.roots = {list(tree.roots)}"
                )
            self._root = int(root)

        if sample_map is None:
            sample_map = (
                TskitLocalTree.default_sample_map(ts))
        # Restrict the caller-facing sample map to samples whose tip falls
        # under the configured root. Samples in sibling subtrees are
        # reported as missing.
        if root is not None:
            sample_map = {
                k: v for k, v in sample_map.items()
                # is_descendant(u, u) is True, so this also keeps the root.
                if tree.is_descendant(int(v), self._root)
            }
        self._sample_to_node: dict[str, int] = dict(sample_map)
        self._postorder: tuple[int, ...] = tuple(tree.postorder(self._root))

    @property
    def root(self) -> int:
        """Root node id of this view of the local tree.

        Equals the wrapped :attr:`tskit.Tree.root` for single-rooted trees,
        or the explicit ``root`` passed at construction for multi-rooted
        trees.
        """
        return self._root

    def children(self, node: int) -> Sequence[int]:
        """Immediate children of ``node``. Empty for tips."""
        return self._tree.children(node)

    def branch_length(self, node: int) -> float:
        """Branch length above ``node`` in the tree's native time units."""
        return float(self._tree.branch_length(node))

    def postorder(self) -> Sequence[int]:
        """Cached post-order traversal (tips before their parents)."""
        return self._postorder

    def tip_for_sample(self, sample_id: str) -> int | None:
        """Map a caller-facing sample id to its tip node id, or ``None``."""
        return self._sample_to_node.get(sample_id)

    def n_tips(self) -> int:
        """Number of tip (sample) nodes under this view's root."""
        # Counts only this view's subtree.
        return int(self._tree.num_samples(self._root))

    @property
    def tskit_tree(self) -> "tskit.Tree":
        """The wrapped :class:`tskit.Tree`.

        Intended for code paths that delegate to tskit-native operations (e.g.
        :meth:`tskit.Tree.draw_text() <tskit.Tree.draw_text>`
        / :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`).

        :return: The underlying :class:`tskit.Tree`.
        """
        return self._tree

    def sample_for_tip(self, node: int) -> str | None:
        """Inverse of :meth:`tip_for_sample`.

        :param node: A tip node id.
        :return: Caller-facing sample id, or ``None`` if the tip is not named
            (e.g. a sample in a sibling subtree of a multi-rooted local tree).
        """
        mapping: "dict[int, str] | None" = getattr(
            self, "_node_to_sample", None)
        if mapping is None:
            mapping = self._node_to_sample = {
                v: k for k, v in self._sample_to_node.items()
            }
        return mapping.get(node)

    def draw_text(
        self,
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        show_branch_lengths: bool = True,
        **tskit_kwargs,
    ) -> str:
        """ASCII render via :meth:`tskit.Tree.draw_text() <tskit.Tree.draw_text>` with label overlays.

        When ``show_branch_lengths`` (default), every non-root node's label
        gets a ``[bl=...]`` suffix, matching the base contract and the generic
        ``TreeRenderer`` renderer.
        """
        from ancestree._viz import TreeRenderer
        node_labels: dict[int, str] = {}
        for node in self._tree.nodes(self._root):
            is_root = node == self._root
            if self._tree.is_sample(node):
                sample_id = self.sample_for_tip(node)
                label = TreeRenderer.format_tip_label(sample_id, node, site)
            elif is_root and posterior is not None:
                label = TreeRenderer.format_root_label(posterior)
            elif show_branch_lengths and not is_root:
                # Base label for the branch-length suffix.
                label = f"node {node}"
            else:
                continue
            if show_branch_lengths and not is_root:
                label += f" [bl={self.branch_length(node):.4g}]"
            node_labels[node] = label
        kwargs = {"node_labels": node_labels, **tskit_kwargs}
        return self._tree.draw_text(**kwargs)

    def draw_svg(
        self,
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        **tskit_kwargs,
    ) -> str:
        """SVG render via :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>` with label overlays."""
        from ancestree._viz import TreeRenderer
        node_labels: dict[int, str] = {}
        for node in self._tree.nodes(self._root):
            if self._tree.is_sample(node):
                sample_id = self.sample_for_tip(node)
                node_labels[node] = TreeRenderer.format_tip_label(sample_id, node, site)
            elif node == self._root and posterior is not None:
                node_labels[node] = TreeRenderer.format_root_label(posterior)
        kwargs = {"node_labels": node_labels, **tskit_kwargs}
        return self._tree.draw_svg(**kwargs)




class RerootedTree(Tree):
    """Another :class:`~ancestree.trees.Tree` viewed as though rooted at one of its own nodes.

    The substitution models are time-reversible, so the posterior at an
    interior node equals the posterior at the root of the same tree re-rooted
    there. Re-rooting only reverses the edges on the path from that node to the
    old root. Every branch keeps its length, and the old root stays as a unary
    node.

    A focal point off a node is expressed by ``tau``: one extra node is
    introduced above ``focal``, splitting the edge above it into ``tau`` below
    and the remainder above. Above the topmost node there is no such edge, so
    the extra node takes ``focal`` as its only child and the posterior there is
    the root's pushed forward by ``P(tau)``.

    :param tree: The :class:`~ancestree.trees.Tree` to re-root.
    :param focal: Node of ``tree`` to root at.
    :param tau: Distance above ``focal`` at which to place the new root. ``0``
        roots at ``focal`` itself.
    :raises ValueError: If ``focal`` is not a node of ``tree``, or ``tau``
        exceeds the branch above ``focal``, which is unbounded only when
        ``focal`` is the topmost node.
    """

    def at_focal(self, focal: "FocalNode | str | None" = None) -> "Tree":
        """This view, which reports at the point it was rooted at.

        :param focal: Accepted for interface compatibility and unused.
        :return: ``self``.
        """
        return self

    @property
    def placement(self) -> tuple[int, float]:
        """Where the root of this view sits in the source tree.

        :return: ``(anchor_node, tau)``: the source-tree node the root sits
            above and the distance above it in the tree's time units. ``tau``
            is zero when the view is rooted at ``anchor_node`` itself.
        """
        source = self._source
        if isinstance(source, RerootedTree) and self._focal == source.root:
            anchor, tau = source.placement
            return anchor, tau + self._tau
        return self._focal, self._tau

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        anchor, tau = self.placement
        return {"anchor": anchor, "tau": tau}

    def node_name(self, node: int) -> str | None:
        """Delegate to the source tree. Re-rooting does not rename nodes."""
        return self._source.node_name(node)

    def __init__(self, tree: Tree, focal: int, tau: float = 0.0) -> None:
        """Build the re-rooted adjacency once, by depth-first walk."""
        self._source = tree
        self._focal = int(focal)
        self._tau = float(tau)

        order = list(tree.postorder())
        if self._focal not in order:
            raise ValueError(
                f"focal node {focal} is not in this tree; "
                f"it has {len(order)} nodes"
            )
        parent = {}
        self._source_edges: set[tuple[int, int]] = set()
        for node in order:
            for child in tree.children(node):
                parent[int(child)] = int(node)
                self._source_edges.add((int(node), int(child)))

        # Undirected adjacency, so the walk can leave `focal` in any direction.
        adjacent: dict[int, list[int]] = {int(n): [] for n in order}
        for child, par in parent.items():
            adjacent[child].append(par)
            adjacent[par].append(child)

        # `tau` inserts a node above `focal`, taking `focal` and whatever lay
        # above it as its two children.
        self._split: int | None = None
        above = parent.get(self._focal)
        if self._tau > 0.0:
            edge = (float("inf") if above is None
                    else float(tree.branch_length(self._focal)))
            if self._tau > edge:
                raise ValueError(
                    f"tau={tau} exceeds the branch above node {focal} ({edge})"
                )
            self._split = max(int(n) for n in order) + 1
            if above is None:
                adjacent[self._split] = [self._focal]
                adjacent[self._focal].append(self._split)
            else:
                adjacent[self._focal].remove(above)
                adjacent[above].remove(self._focal)
                adjacent[self._split] = [self._focal, above]
                adjacent[self._focal].append(self._split)
                adjacent[above].append(self._split)
            self._split_lengths = {self._focal: self._tau}
            if above is not None:
                self._split_lengths[above] = edge - self._tau

        self._root_node = self._split if self._split is not None else self._focal
        self._children: dict[int, tuple[int, ...]] = {}
        self._branch: dict[int, float] = {}
        self._postorder: list[int] = []
        self._walk(adjacent, tree)

    @property
    def time_scale(self) -> float:
        """The source tree's time scale, in substitutions per unit length."""
        return float(self._source.time_scale)

    @time_scale.setter
    def time_scale(self, v: float) -> None:
        """Set the source tree's time scale.

        :param v: Substitutions per unit branch length.
        """
        self._source.time_scale = v

    def _walk(self, adjacent, tree) -> None:
        """Depth-first from the new root, recording children and branch lengths."""
        stack: "list[tuple[int, int | None]]" = [(self._root_node, None)]
        visit_order = []
        while stack:
            node, came_from = stack.pop()
            kids = tuple(n for n in adjacent[node] if n != came_from)
            self._children[node] = kids
            visit_order.append(node)
            for kid in kids:
                # The branch between two nodes is the same length whichever
                # way it is traversed.
                if self._split is not None and node == self._split:
                    self._branch[kid] = self._split_lengths[kid]
                elif self._parent_in_source(kid, node):
                    # Edge kept its direction: it is the branch above `kid`.
                    self._branch[kid] = float(tree.branch_length(kid))
                else:
                    # Edge reversed: it is the branch that was above `node`.
                    self._branch[kid] = float(tree.branch_length(node))
                stack.append((int(kid), node))
        self._postorder = list(reversed(visit_order))

    def _parent_in_source(self, child: int, node: int) -> bool:
        """Whether ``node`` was ``child``'s parent in the original rooting."""
        return (int(node), int(child)) in self._source_edges

    @property
    def root(self) -> int:
        """The node this view is rooted at."""
        return self._root_node

    def children(self, node: int) -> "Sequence[int]":
        """Immediate children of ``node`` in the re-rooted orientation."""
        return self._children.get(int(node), ())

    def branch_length(self, node: int) -> float:
        """Length of the branch above ``node``. Zero at the new root."""
        return float(self._branch.get(int(node), 0.0))

    def postorder(self) -> "Sequence[int]":
        """Nodes with children before parents, from the new root."""
        return tuple(self._postorder)

    def tip_for_sample(self, sample_id: str) -> int | None:
        """Delegate to the source tree. Re-rooting does not move the tips."""
        return self._source.tip_for_sample(sample_id)

    def sample_for_tip(self, node: int) -> str | None:
        """Delegate to the source tree."""
        return self._source.sample_for_tip(node)

    @property
    def tskit_tree(self) -> "tskit.Tree | None":
        """Delegate to the source tree, whose genomic span re-rooting keeps."""
        return self._source.tskit_tree

    @property
    def ingroup_samples(self) -> tuple[str, ...]:
        """The source tree's ingroup."""
        return tuple(self._source.ingroup_samples)

    @property
    def outgroup_samples(self) -> tuple[str, ...]:
        """The source tree's outgroup."""
        return tuple(self._source.outgroup_samples)

    def as_deep_rooted(self) -> "Tree | None":
        """The source tree's deep rooting.

        :return: A re-rooted view of the source, or ``None`` where it defines
            no deep rooting.
        """
        return self._source.as_deep_rooted()

    @property
    def ingroup_mrca(self) -> "int | None":
        """Delegate to the source tree. Re-rooting does not renumber nodes.

        :return: Node id of the ingroup most recent common ancestor, or
            ``None`` where the source tree defines no ingroup.
        """
        return self._source.ingroup_mrca

    def n_tips(self) -> int:
        """Number of tips, counted from the source and not from the rooting.

        Re-rooting at a sample turns it into a unary root, which the source
        still counts as a tip.
        """
        return self._source.n_tips()


class OutgroupLadderTree(Tree):
    """Fixed species-tree topology for EST-SFS mode: ingroup MRCA as inference root, outgroup ladder above.

    The ingroup haplotypes are collapsed under their MRCA ``I``. Their whole
    contribution is the vector ``P(ingroup alleles | I = s)``, weighted by
    :class:`~ancestree.priors.KingmanIngroupWeight` and seeded on ``I``
    through
    :paramref:`node_seeds <ancestree.likelihood.Likelihood.log_likelihoods.node_seeds>`,
    so the ingroup has no node of its own. The tree carries the species-level
    topology, the ingroup MRCA ``I`` and ``n`` outgroup tips in a ladder of
    increasing divergence, and has exactly ``2n`` nodes.

    .. code-block:: text

                    I (default readout)
                     |
                    n_1           (or O_1 directly if n_outgroups == 1)
                   /   \\
                  O_1   n_2       (or O_2 if n_outgroups == 2)
                       /   \\
                     O_2   …
                           \\
                          n_{n-1}
                          /   \\
                    O_{n-1}   O_n

    Free branch-rate parameters (``2·n − 1`` for ``n`` outgroups), named as
    in fastDFE's ``MaximumLikelihoodAncestralAnnotation``:

    - ``K0``       : ``I → n_1`` (or ``I → O_1`` for ``n=1``)
    - ``K1``       : ``n_1 → O_1``
    - ``K2``       : ``n_1 → n_2``
    - ``K3``       : ``n_2 → O_2``
    - ...
    - ``K{2(n-1)-1}`` : branch above ``O_{n-1}``
    - ``K{2(n-1)}``   : branch above ``O_n``

    The path length from ``I`` to outgroup ``O_k`` is
    ``K0 + K2 + … + K{2(k-1)} + K{2k-1}`` for ``k < n`` and
    ``K0 + K2 + … + K{2(n-2)} + K{2(n-1)}`` for ``k = n``, matching
    fastDFE's ``get_outgroup_divergence`` convention. ``K0`` is identifiable
    only under a non-stationary root prior.

    :param ingroup_samples: Sample ids for the ingroup haplotypes, used for
        SFS-class lookup by :class:`~ancestree.inference.FixedTreeInference`.
        They resolve to no tip of this tree.
    :param outgroup_samples: Sample ids for the outgroup tips, closest first
        (``O_1`` first, ``O_n`` last).
    :param initial_rate: Starting value for all branch rates until
        :meth:`set_params` is called.
    :raises ValueError: If ``outgroup_samples`` is empty.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_ingroup": self.n_ingroup, "n_outgroups": self.n_outgroups}

    @staticmethod
    def order_outgroups_by_divergence(
        sites: "Sequence[Site]",
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
    ) -> list[str]:
        """Sort ``outgroup_samples`` closest-first by a Hartigan-style proxy.

        For each outgroup, compute the fraction of polymorphic sites at
        which its tip allele disagrees with the ingroup majority allele
        (ties in the majority are broken arbitrarily, first allele in
        insertion order). Outgroups are then sorted by ascending
        disagreement rate. Use this to obtain the inferred order without building a
        tree. :meth:`OutgroupLadderTree.with_outgroup_order_from_sites` returns
        both at once.

        :param sites: Polymorphic :class:`~ancestree.sites.Site` records carrying ingroup
            tip alleles.
        :param ingroup_samples: Ids of ingroup haplotypes, or of the
            individuals they belong to.
        :param outgroup_samples: Ids of outgroup haplotypes in any order.
        :return: ``outgroup_samples`` reordered closest-first.
        :raises ValueError: If no Site carries any ingroup tip allele,
            or if any outgroup has no jointly-observed sites with the
            ingroup.
        """
        majority_per_site: list[str | None] = []
        any_ingroup_seen = False
        for site in sites:
            counts = site.count_alleles(ingroup_samples)
            if counts:
                any_ingroup_seen = True
                majority_per_site.append(counts.most_common(1)[0][0])
            else:
                majority_per_site.append(None)
        if not any_ingroup_seen:
            raise ValueError(
                "order_outgroups_by_divergence: no Site carries ingroup tip "
                "alleles, so the ingroup-majority proxy cannot be computed. "
                "Pass Site records whose `tip_alleles` includes the ingroup "
                "samples, or construct OutgroupLadderTree directly with "
                "the desired order."
            )

        divergence: dict[str, float] = {}
        for og in outgroup_samples:
            n_diff = n_obs = 0
            for site, majority in zip(sites, majority_per_site):
                if majority is None:
                    continue
                og_allele = Site.canonical(site.tip_alleles.get(og))
                if og_allele is None:
                    continue
                n_obs += 1
                if og_allele != majority:
                    n_diff += 1
            if n_obs == 0:
                raise ValueError(
                    f"order_outgroups_by_divergence: outgroup {og!r} has "
                    f"no sites jointly observed with the ingroup, so its "
                    f"divergence cannot be estimated."
                )
            divergence[og] = n_diff / n_obs
        return sorted(outgroup_samples, key=lambda o: divergence[o])

    @classmethod
    def from_newick(
        cls,
        newick_str: str,
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
    ) -> "OutgroupLadderTree":
        """Build an :class:`~ancestree.trees.OutgroupLadderTree` with branch rates from a Newick string.

        Accepts a species-tree Newick rooted at the deepest ancestor, e.g.::

            ((((tsk_0,tsk_1,tsk_2):0.05,O_1:0.10):0.08,O_2:0.20):0.15,O_3:0.40);

        Leaves in neither sample list are pruned, and degree-2 nodes left by
        pruning are collapsed with their branch lengths summed. The ladder
        order is read from the topology, so the order of ``outgroup_samples``
        does not matter. After pruning, every supplied sample must be a leaf,
        the ingroup must be monophyletic, and each outgroup must branch off
        the path from the ingroup MRCA to the Newick root as a single leaf.
        The degree-2 Newick root folds into the deepest ladder node, so its
        two edges sum into ``K{2(n-1)}``. The branch rates are set from the
        Newick, so :class:`~ancestree.inference.FixedTreeInference` may be
        constructed with ``fit_required=False``.

        :param newick_str: Newick string with branch lengths on every edge.
        :param ingroup_samples: Ids of the ingroup haplotypes, matching leaf
            names in the Newick.
        :param outgroup_samples: Ids of the outgroup haplotypes, in any order.
        :return: A ladder whose branch rates are populated from the Newick.
        :raises ImportError: If the ``newick`` package is not installed.
        :raises ValueError: On missing samples, overlap between ingroup and
            outgroup, a non-monophyletic ingroup, or a ladder violation.
        """
        try:
            import newick as _newick
        except ImportError as e:
            raise ImportError(
                "OutgroupLadderTree.from_newick requires the `newick` "
                "package. Install with `pip install newick`."
            ) from e

        ingroup_list = list(ingroup_samples)
        outgroup_list = list(outgroup_samples)
        overlap = set(ingroup_list) & set(outgroup_list)
        if overlap:
            raise ValueError(
                f"OutgroupLadderTree.from_newick: same names appear in both "
                f"ingroup_samples and outgroup_samples: {sorted(overlap)}"
            )
        if not outgroup_list:
            raise ValueError(
                "OutgroupLadderTree.from_newick: outgroup_samples must be "
                "non-empty (the topology requires ≥1 outgroup)."
            )

        forest = _newick.loads(newick_str)
        if len(forest) != 1:
            raise ValueError(
                f"Newick string must contain exactly one tree; got {len(forest)}"
            )
        nw_root = forest[0]

        # Validate that every supplied sample appears as a leaf.
        leaf_names_in_newick = {
            n.name for n in nw_root.walk() if n.is_leaf and n.name
        }
        missing_ingroup = set(ingroup_list) - leaf_names_in_newick
        missing_outgroup = set(outgroup_list) - leaf_names_in_newick
        if missing_ingroup or missing_outgroup:
            parts = []
            if missing_ingroup:
                parts.append(f"ingroup: {sorted(missing_ingroup)}")
            if missing_outgroup:
                parts.append(f"outgroup: {sorted(missing_outgroup)}")
            raise ValueError(
                f"OutgroupLadderTree.from_newick: samples not found as "
                f"leaves in the Newick: {'; '.join(parts)}"
            )

        # Prune the Newick to keep only the listed samples. Collapse
        # any degree-2 internals produced by pruning (their branch
        # lengths sum into the surviving edges).
        keep = list(ingroup_list) + list(outgroup_list)
        nw_root.prune_by_names(keep, inverse=True)
        nw_root.remove_redundant_nodes()

        # Build a parent map: id(node) -> parent node.
        parents: dict[int, "_newick.Node | None"] = {}
        for n in nw_root.walk():
            parents[id(n)] = None
        def _set_parents(node):
            """Recursive parent-pointer assignment."""
            for c in node.descendants:
                parents[id(c)] = node
                _set_parents(c)
        _set_parents(nw_root)

        leaves: dict[str, "_newick.Node"] = {
            n.name: n for n in nw_root.walk() if n.is_leaf and n.name
        }
        ingroup_set = set(ingroup_list)

        # Find MRCA = the deepest internal node whose descendants include
        # every ingroup sample. Walking up from any ingroup leaf, the first
        # node containing all ingroup samples is the MRCA.
        if len(ingroup_list) == 1:
            mrca = leaves[ingroup_list[0]]
        else:
            ingroup_ids = {id(leaves[s]) for s in ingroup_list}
            cur = leaves[ingroup_list[0]]
            while not ingroup_ids.issubset({id(d) for d in cur.walk()}):
                cur = parents[id(cur)]
            mrca = cur

        # Check that the MRCA's descendants are exactly the ingroup samples
        # (no outgroup leaves nested inside the ingroup subtree).
        mrca_leaves = {d.name for d in mrca.walk() if d.is_leaf and d.name}
        extra = mrca_leaves - ingroup_set
        if extra:
            raise ValueError(
                f"OutgroupLadderTree.from_newick: non-ingroup leaves found "
                f"inside the ingroup MRCA subtree: {sorted(extra)}. Ingroup "
                f"must be monophyletic with no outgroups nested under it."
            )

        # Walk up from MRCA toward the Newick root, collecting K's and
        # outgroup names in closest-first order. At the LAST step (when
        # the parent IS the Newick root), the parent's two edges fold
        # into a single K{2(n-1)} = current.length + deepest_outgroup.length.
        K_values: list[float] = []
        outgroup_order: list[str] = []
        cur = mrca
        while True:
            parent = parents[id(cur)]
            if parent is None:
                raise ValueError(
                    "OutgroupLadderTree.from_newick: ingroup MRCA is the "
                    "Newick root; no outgroups present in the tree."
                )
            siblings = [s for s in parent.descendants if s is not cur]
            if len(siblings) != 1:
                raise ValueError(
                    f"OutgroupLadderTree.from_newick: expected exactly 1 "
                    f"sibling at each ladder step; got {len(siblings)}. "
                    f"Each non-ingroup outgroup must branch off as a single "
                    f"leaf at successive splits."
                )
            sibling = siblings[0]
            if not sibling.is_leaf:
                raise ValueError(
                    f"OutgroupLadderTree.from_newick: each outgroup sibling "
                    f"must be a leaf; got an internal node at this ladder step."
                )

            grandparent = parents[id(parent)]
            cur_len = float(cur.length) if cur.length is not None else 0.0
            sib_len = float(sibling.length) if sibling.length is not None else 0.0
            if grandparent is None:
                # Last step: fold the degree-2 Newick root into a single K.
                K_values.append(cur_len + sib_len)
                outgroup_order.append(sibling.name)
                break
            # Non-last step: append K{2i} (current's branch up) and K{2i+1}
            # (sibling outgroup's branch).
            K_values.append(cur_len)
            K_values.append(sib_len)
            outgroup_order.append(sibling.name)
            cur = parent

        n = len(outgroup_order)
        if len(K_values) != 2 * n - 1:
            raise ValueError(
                f"OutgroupLadderTree.from_newick: extracted {len(K_values)} K's "
                f"for {n} outgroups (expected {2 * n - 1}). Topology likely "
                f"violates the ladder constraint."
            )
        zero = [f"K{i}" for i, k in enumerate(K_values) if k <= 0.0]
        if zero:
            raise ValueError(
                f"OutgroupLadderTree.from_newick: the Newick must be dated, "
                f"but {', '.join(zero)} came out non-positive. Add branch "
                f"lengths, or omit the Newick and build the ladder from the "
                f"sample names so the rates are fitted."
            )
        tree = cls(ingroup_list, outgroup_order, initial_rate=0.0)
        tree.set_params(np.asarray(K_values, dtype=float))
        return tree

    @classmethod
    def with_outgroup_order_from_sites(
        cls,
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        sites: Sequence["Site"],
        *,
        initial_rate: float = 1.0,
    ) -> "OutgroupLadderTree":
        """Build an :class:`~ancestree.trees.OutgroupLadderTree` with outgroups auto-ordered by divergence.

        For ``n_outgroups >= 3`` a wrong closest-first ordering biases the
        fit, so the order is read off the data via
        :meth:`OutgroupLadderTree.order_outgroups_by_divergence()
        <ancestree.trees.OutgroupLadderTree.order_outgroups_by_divergence>`.

        :param ingroup_samples: Sample ids for the ingroup haplotypes.
        :param outgroup_samples: Sample ids for the outgroups in any
            order. The method reorders them.
        :param sites: Polymorphic :class:`~ancestree.sites.Site` records carrying both
            ingroup and outgroup tip alleles.
        :param initial_rate: Starting branch rate for the constructed
            tree (forwarded to ``__init__``).
        :return: A new :class:`~ancestree.trees.OutgroupLadderTree` with outgroups in inferred
            closest-first order.
        :raises ValueError: If no Site carries any ingroup tip alleles
            (the divergence proxy cannot be computed) or if any outgroup
            has no jointly-observed sites with the ingroup.
        """
        ordered = cls.order_outgroups_by_divergence(
            sites, ingroup_samples, outgroup_samples,
        )
        return cls(ingroup_samples, ordered, initial_rate=initial_rate)

    def __init__(
        self,
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        *,
        initial_rate: float = 1.0,
    ) -> None:
        """Build the ladder topology + param→branch map from the sample lists.

        :raises ValueError: If no outgroup is given, or if a sample is named
            as both ingroup and outgroup. An overlapping sample would serve
            as an outgroup tip in the kernel and as an ingroup haplotype in
            the ingroup weight at the same time.
        """
        if len(outgroup_samples) == 0:
            raise ValueError("OutgroupLadderTree requires at least one outgroup.")
        overlap = set(ingroup_samples) & set(outgroup_samples)
        if overlap:
            raise ValueError(
                f"OutgroupLadderTree: same names appear in both "
                f"ingroup_samples and outgroup_samples: {sorted(overlap)}")

        self._ingroup_samples: list[str] = list(ingroup_samples)
        self._outgroup_samples: list[str] = list(outgroup_samples)
        self._n_ingroup: int = len(self._ingroup_samples)
        n: int = len(self._outgroup_samples)
        self._n_outgroups: int = n

        # Node ids: 0 is I (root), 1..n the outgroup tips, n+1..2n-1 the joins.
        self._I: int = 0
        self._outgroup_tips: tuple[int, ...] = tuple(range(1, n + 1))
        self._ladder_nodes: tuple[int, ...] = tuple(range(n + 1, 2 * n))
        self._root: int = self._I

        # Only the outgroups are tips.
        self._sample_to_tip: dict[str, int] = {
            sid: node for sid, node in zip(self._outgroup_samples, self._outgroup_tips)
        }
        self._tip_to_sample: dict[int, str] = {
            node: sid for sid, node in self._sample_to_tip.items()
        }

        # Build the parent→children map.
        self._children_map: dict[int, list[int]] = {}
        if n == 1:
            self._children_map[self._I] = [self._outgroup_tips[0]]
        else:
            self._children_map[self._I] = [self._ladder_nodes[0]]
            for k in range(n - 2):
                self._children_map[self._ladder_nodes[k]] = [
                    self._outgroup_tips[k],
                    self._ladder_nodes[k + 1],
                ]
            self._children_map[self._ladder_nodes[n - 2]] = [
                self._outgroup_tips[n - 2],
                self._outgroup_tips[n - 1],
            ]

        # Parent lookup for divergence calculation.
        self._parent_map: dict[int, int] = {}
        for parent, kids in self._children_map.items():
            for c in kids:
                self._parent_map[c] = parent

        # Postorder traversal: tips before parents.
        po: list[int] = []
        def _visit(node: int) -> None:
            """Recursive postorder collector."""
            for c in self._children_map.get(node, []):
                _visit(c)
            po.append(node)
        _visit(self._I)
        self._postorder: tuple[int, ...] = tuple(po)

        # Internal branch map (length 2n-1): the full ladder edges the
        # kernel sees, in the natural EST-SFS order
        #   internal[0]:    K^{int}_0   = I → n_1   (or I → O_1 for n=1)
        #   internal[2k-1]: K^{int}_{2k-1} = n_k → O_k    (k=1..n-1)
        #   internal[2k]:   K^{int}_{2k}   = n_k → n_{k+1} (k=1..n-2)
        #   internal[2(n-1)]: K^{int}_{2(n-1)} = n_{n-1} → O_n
        internal_to_branch: list[int] = []
        if n == 1:
            internal_to_branch.append(self._outgroup_tips[0])
        else:
            internal_to_branch.append(self._ladder_nodes[0])  # K^int_0
            for k in range(1, n - 1):
                internal_to_branch.append(self._outgroup_tips[k - 1])  # K^int_{2k-1}
                internal_to_branch.append(self._ladder_nodes[k])  # K^int_{2k}
            internal_to_branch.append(self._outgroup_tips[n - 2])  # K^int_{2(n-1)-1}
            internal_to_branch.append(self._outgroup_tips[n - 1])  # K^int_{2(n-1)}
        self._internal_to_branch: tuple[int, ...] = tuple(internal_to_branch)

        self._n_params: int = 2 * n - 1
        self._param_names: tuple[str, ...] = tuple(
            f"K{j}" for j in range(2 * n - 1)
        )
        # The root has no branch length. ``set_params`` fills every K_i from
        # the initial-rate seed.
        self._branch_lengths: dict[int, float] = {self._I: 0.0}
        self.set_params(np.full(self._n_params, float(initial_rate)))

    # --------------------------------------------------------- Tree interface

    @property
    def root(self) -> int:
        """Inference root = ingroup MRCA (always node id 0)."""
        return self._root

    def children(self, node: int) -> Sequence[int]:
        """Immediate children of ``node`` in the ladder. Empty for tips."""
        return self._children_map.get(node, [])

    def branch_length(self, node: int) -> float:
        """Branch length above ``node`` (one of the fitted ``K_i``)."""
        return self._branch_lengths.get(node, 0.0)

    def postorder(self) -> Sequence[int]:
        """Cached post-order traversal (tips before parents)."""
        return self._postorder

    def tip_for_sample(self, sample_id: str) -> int | None:
        """Sample id to tip node id. Ingroup ids map to ``None``."""
        return self._sample_to_tip.get(sample_id)

    def sample_for_tip(self, node: int) -> str | None:
        """Tip node id to outgroup sample id, ``None`` for non-tip nodes."""
        return self._tip_to_sample.get(int(node))

    def n_tips(self) -> int:
        """Tip count: one per outgroup."""
        return self._n_outgroups

    # ---------------------------------------------------- EST-SFS parameters

    @property
    def n_outgroups(self) -> int:
        """Number of outgroup tips in the topology."""
        return self._n_outgroups

    @property
    def n_ingroup(self) -> int:
        """Number of ingroup haplotypes."""
        return self._n_ingroup

    @property
    def ingroup_samples(self) -> tuple[str, ...]:
        """Caller-facing ids of the ingroup haplotypes."""
        return tuple(self._ingroup_samples)

    @property
    def outgroup_samples(self) -> tuple[str, ...]:
        """Caller-facing ids of the outgroup tips, in tree order
        (closest first, most distant last)."""
        return tuple(self._outgroup_samples)

    @property
    def n_params(self) -> int:
        """Number of free branch-rate parameters (``2·n_outgroups − 1``)."""
        return self._n_params

    @property
    def param_names(self) -> tuple[str, ...]:
        """Human-readable names for :meth:`param_vector` entries
        (``K0, K1, …, K{2n-2}``)."""
        return self._param_names

    @property
    def param_vector(self) -> np.ndarray:
        """Current free branch rates as a length-``n_params`` ndarray.

        Parameter ``j`` is the internal ladder branch ``K^int_j`` for
        ``j = 0, …, 2n-2``.

        :return: Copy of the parameter vector.
        """
        out = np.empty(self._n_params, dtype=float)
        for j in range(self._n_params):
            out[j] = self._branch_lengths[self._internal_to_branch[j]]
        return out

    def set_params(self, x: Sequence[float] | np.ndarray) -> None:
        """Update the free branch rates from an optimizer's parameter vector.

        :param x: Length-``n_params`` sequence of new branch rates, ordered
            as in :attr:`param_names` (``[K0, K1, …, K{2n-2}]``).
        :raises ValueError: If ``x`` does not have shape ``(n_params,)``.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self._n_params,):
            raise ValueError(
                f"set_params expected length-{self._n_params} vector, "
                f"got shape {tuple(x.shape)}"
            )
        for j in range(self._n_params):
            self._branch_lengths[self._internal_to_branch[j]] = float(x[j])

    @property
    def _backbone(self) -> "tuple[list[int], list[float]]":
        """The I-to-deepest-join path and its branch lengths."""
        path, lengths = [self._I], []
        node = self._I
        while True:
            kids = [c for c in self._children_map.get(node, [])
                    if c in self._ladder_nodes]
            if not kids:
                break
            node = kids[0]
            lengths.append(float(self._branch_lengths[node]))
            path.append(node)
        # The backbone ends on the deepest outgroup's branch, not at the join
        # below it: that join is the ancestor of the panel minus its most
        # distant member, so the panel's own ancestor lies further along.
        deepest = self._outgroup_tips[-1]
        path.append(deepest)
        lengths.append(float(self._branch_lengths[deepest]))
        return path, lengths

    @property
    def _backbone_limit(self) -> float:
        """How far along the backbone the readout may be placed.

        The far end is the ancestor of the whole panel, which sits on the
        deepest outgroup's branch at a position the exact transition matrix
        cannot identify, since it satisfies ``P(a) P(b) == P(a + b)``. The
        ultrametric convention places it at half that outgroup's divergence,
        as :meth:`from_divergences` does.

        :return: Distance from the ingroup MRCA, in substitutions per site.
        """
        return 0.5 * float(self.outgroup_divergence()[-1])

    @classmethod
    def from_divergences(
        cls,
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        divergences: "Sequence[float]",
    ) -> "OutgroupLadderTree":
        """Build a ladder whose fitted divergences are the ones given.

        The ingroup-to-outgroup path lengths determine only the sum along each path,
        so the split points are placed at half the divergence, the ultrametric
        choice.

        :param ingroup_samples: Ids of the ingroup haplotypes.
        :param outgroup_samples: Outgroup ids, closest first.
        :param divergences: The I-to-O_k path lengths, in the same order.
        :return: A parameterised ladder.
        :raises ValueError: If the divergences are not increasing and positive.
        """
        div = [float(d) for d in divergences]
        if len(div) != len(outgroup_samples):
            raise ValueError(
                f"got {len(div)} divergences for "
                f"{len(outgroup_samples)} outgroups"
            )
        if div[0] <= 0 or any(b <= a for a, b in zip(div, div[1:])):
            raise ValueError(
                f"divergences must be positive and increasing; got {div}"
            )
        tree = cls(ingroup_samples, outgroup_samples)
        n = len(div)
        if n == 1:
            # One branch is the whole I -> O_1 path.
            tree.set_params([div[0]])
            return tree
        heights = [d / 2.0 for d in div]  # split point of each join
        params = [heights[0]]  # K0: I -> n_1
        for k in range(1, n - 1):
            params.append(div[k - 1] - heights[k - 1])  # K{2k-1}: n_k -> O_k
            params.append(heights[k] - heights[k - 1])  # K{2k}:   n_k -> n_{k+1}
        if n > 1:
            params.append(div[n - 2] - heights[n - 2])  # branch above O_{n-1}
            params.append(div[n - 1] - heights[n - 2])  # branch above O_n
        tree.set_params(params)
        return tree

    @property
    def ingroup_mrca(self) -> int:
        """The node the ingroup's likelihood vector is seeded on, and the shallow anchor.

        :return: Node id of the ingroup most recent common ancestor.
        """
        return self._I

    def as_deep_rooted(self) -> "Tree":
        """This ladder read at the deepest join.

        The deep end of :meth:`OutgroupLadderTree.at_focal`'s backbone, and the
        reading a site the ingroup has fixed needs.

        :return: A re-rooted view of this tree.
        """
        from ancestree.focal import FocalNode
        return self.at_focal(FocalNode("ingroup_mrca", fraction=1.0))

    @property
    def placement(self) -> tuple[int, float]:
        """``(ingroup_mrca, 0.0)``: this ladder reads at its own ingroup MRCA.

        A :class:`~ancestree.trees.RerootedTree` view returns the node and offset of its root
        under the same name.
        """
        return self.ingroup_mrca, 0.0

    def at_focal(self, focal: "FocalNode | str | None" = None) -> "Tree":
        """This ladder viewed from a point on, or above, its backbone.

        The readout point slides along the backbone of this one tree, so the
        two ends are the same topology read at different nodes. At the
        ingroup MRCA a site the ingroup has fixed reads back its
        own allele, which is why such sites are reported deeper.

        :param focal: A :class:`~ancestree.focal.FocalNode`, an anchor name, or
            ``None`` for the ingroup MRCA.
        :return: This tree when reading at its own root, otherwise a re-rooted
            view of it.
        """
        from ancestree.focal import FocalNode

        spec = FocalNode.parse(focal)
        path, lengths = self._backbone
        total = self._backbone_limit

        def view(distance: float) -> "Tree":
            """The ladder read ``distance`` above the ingroup MRCA."""
            if distance <= 0.0:
                return self
            remaining = distance
            for step, length in enumerate(lengths):
                if remaining <= length:
                    return RerootedTree(self, path[step + 1],
                                        length - remaining)
                remaining -= length
            raise AssertionError("distance runs past the backbone")

        def deep_end() -> "Tree":
            """The ancestor of the whole panel, on the deepest branch."""
            return view(total)

        if spec.anchor == "panel_root":
            return deep_end()
        if spec._placement is None:
            return self
        if spec.coalescences is not None:
            # The backbone ends on a branch, not at a join, so the joins are
            # the interior nodes. Asking past the last one lands on the panel's
            # own ancestor.
            joins = len(path) - 2
            step = int(spec.coalescences)
            if step > joins:
                return deep_end()
            return view(sum(lengths[:step]))
        if spec.depth is not None:
            want = float(spec.depth)
            if want <= total:
                return view(want)
            # Beyond the panel root the view extends one node above it,
            # carrying the remainder as its branch.
            deep = deep_end()
            return RerootedTree(deep, deep.root, want - total)
        return view(total * float(spec.fraction or 0.0))

    def node_name(self, node: int) -> str | None:
        """``"I"`` for the ingroup MRCA and ``"n_k"`` for the k-th ladder join.

        The ingroup MRCA carries no tip of its own, and a re-rooted view of
        this ladder turns it into one.

        :param node: Node id in this tree.
        :return: The name, or ``None`` for an outgroup tip.
        """
        if node == self._root:
            return "I"
        ladder_index = {nid: i + 1 for i, nid in enumerate(self._ladder_nodes)}
        if node in ladder_index:
            return f"n_{ladder_index[node]}"
        return None

    def outgroup_divergence(self) -> np.ndarray:
        """Path length from the ingroup MRCA ``I`` to each outgroup, in order.

        Matches fastDFE's ``MaximumLikelihoodAncestralAnnotation.get_outgroup_divergence``
        convention so simulations under known ``μ·t`` can be checked against
        a fitted :class:`~ancestree.trees.OutgroupLadderTree` directly.

        :return: Length-``n_outgroups`` array of ``I → O_k`` path lengths.
        """
        rates = np.zeros(self._n_outgroups, dtype=float)
        for k, tip in enumerate(self._outgroup_tips):
            total = 0.0
            node = tip
            while node != self._I:
                total += self._branch_lengths.get(node, 0.0)
                node = self._parent_map[node]
            rates[k] = total
        return rates


