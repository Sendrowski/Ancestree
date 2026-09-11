"""The node whose ancestral-state posterior a run reports.

Every mode evaluates Felsenstein's recursion toward a single node and reads the
posterior there. That node is a property of the question, not of
the mode: all modes report at the same one, by default the most recent common
ancestor of the ingroup, whose state is the ingroup's ancestral allele.

:class:`~ancestree.focal.FocalNode` names that node as a rule, not an id, because a run
spans many local trees and node ids are per tree. The rule resolves against each
tree in turn:

.. code-block:: python

    FocalNode()                                 # the ingroup's own MRCA
    FocalNode("panel_root")                     # the panel's deepest ancestor
    FocalNode("ingroup_mrca", fraction=0.5)     # halfway toward the panel root
    FocalNode("ingroup_mrca", coalescences=1)   # just above the first join
    FocalNode("ingroup_mrca", depth=1.5e6)      # a fixed depth, every tree

The substitution models are time-reversible, so the posterior at an interior
node equals the posterior at the root of the same tree re-rooted there, which
is how the kernel reads it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Named nodes a :class:`~ancestree.focal.FocalNode` can anchor on.
Anchor = Literal["ingroup_mrca", "panel_root"]

ANCHORS: tuple[str, ...] = ("ingroup_mrca", "panel_root")


@dataclass(frozen=True)
class ResolvedFocal:
    """A focal node resolved against one tree."""

    #: Node id in that tree.
    node: int
    #: Distance above :attr:`node`, along the edge into it, in the tree's own
    #: branch units. ``0.0`` means the node itself.
    tau: float = 0.0
    #: Whether the anchor node subtends the ingroup and nothing else. ``None``
    #: when no ingroup was supplied.
    ingroup_is_monophyletic: bool | None = None


@dataclass(frozen=True)
class FocalNode:
    """The node a run reports its posterior at.

    :param anchor: ``"ingroup_mrca"`` (the default) for the most recent common
        ancestor of the supplied ingroup samples, whose state is the ingroup's
        ancestral allele; ``"panel_root"`` for the most recent common ancestor
        of the whole panel, which is deeper whenever outgroups are present.
        With no outgroups declared the ingroup comprises the whole panel and
        the two anchors coincide.
    :param fraction: Position along the path from the anchor to the root, as a
        fraction in ``[0, 1]``. ``0`` is the anchor itself. Scale-free, so it
        means the same thing in every mode and across datasets.
    :param coalescences: Number of coalescences above the anchor, counting the
        join events on the path to the root. Discrete and topological, so it
        picks out the point just above the k-th outgroup joining.
    :param depth: Absolute distance from the sample plane, in the tree's own
        branch units: node time for a genealogy, expected substitutions per
        site above the ingroup MRCA for the fixed-tree ladder, whose collapsed
        ingroup tip sits at zero. Bounded below by the anchor, so a depth
        shallower than the ingroup MRCA reads at the MRCA. A depth beyond the
        root reads above it, where the posterior is the root's pushed forward
        and carries no further information.
    :raises ValueError: On an unknown anchor, on more than one placement, on
        a placement combined with ``anchor="panel_root"`` (the far end), on a fraction outside ``[0, 1]``, or on a negative
        depth or count.
    """

    #: The anchor node, ``"ingroup_mrca"`` or ``"panel_root"``.
    anchor: Anchor = "ingroup_mrca"
    #: Fraction of the anchor-to-root path, in ``[0, 1]``, or ``None``.
    fraction: float | None = None
    #: Number of coalescences above the anchor, or ``None``.
    coalescences: int | None = None
    #: Absolute depth above the sample plane in the tree's units, or ``None``.
    depth: float | None = None

    def __post_init__(self) -> None:
        """Validate the anchor and the two mutually exclusive offsets."""
        if self.anchor not in ANCHORS:
            raise ValueError(
                f"unknown focal anchor {self.anchor!r}; expected one of {ANCHORS}"
            )
        given = [n for n, v in (
            ("fraction", self.fraction), ("coalescences", self.coalescences),
            ("depth", self.depth),
        ) if v is not None]
        if len(given) > 1:
            raise ValueError(
                f"give at most one placement; got {', '.join(given)}"
            )
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"fraction must be in [0, 1]; got {self.fraction}")
        if self.depth is not None and self.depth < 0:
            raise ValueError(f"depth must be non-negative; got {self.depth}")
        if self.coalescences is not None and self.coalescences < 0:
            raise ValueError(
                f"coalescences must be non-negative; got {self.coalescences}"
            )
        if self.anchor == "panel_root" and self._placement is not None:
            raise ValueError(
                "a placement moves from the anchor toward the root, so it "
                'does nothing on anchor="panel_root"; anchor on "ingroup_mrca" '
                "instead"
            )

    @classmethod
    def parse(cls, spec: "FocalNode | str | None") -> "FocalNode":
        """Coerce a user-supplied spec to a :class:`~ancestree.focal.FocalNode`.

        :param spec: A :class:`~ancestree.focal.FocalNode`, an anchor name, or ``None`` for the
            default, the ingroup MRCA. With no ingroup or outgroup named the
            ingroup is the whole panel, so that node is the panel root.
        :return: The corresponding :class:`~ancestree.focal.FocalNode`.
        :raises TypeError: On any other type.
        """
        if spec is None:
            return cls()
        if isinstance(spec, FocalNode):
            return spec
        if isinstance(spec, str):
            return cls(anchor=spec)  # type: ignore[arg-type]
        raise TypeError(
            f"focal must be a FocalNode, an anchor name {ANCHORS}, or None; "
            f"got {type(spec).__name__}"
        )

    @property
    def _placement(self) -> "float | int | None":
        """Whichever of the three placements was given, if any.

        Callers test the result against ``None``: ``fraction=0.0`` and
        ``coalescences=0`` are placements naming the anchor itself.
        """
        for value in (self.fraction, self.coalescences, self.depth):
            if value is not None:
                return value
        return None

    @property
    def is_root(self) -> bool:
        """Whether this names the panel root, the tree's own topmost node."""
        return self.anchor == "panel_root" and self._placement is None

    def provenance(self) -> dict[str, object]:
        """This spec as provenance entries: the anchor and its placement.

        Every mode records the same keys, so an output says which node it was
        read at whichever mode produced it.

        :return: ``{"focal": anchor}`` plus at most one placement key.
        """
        entry: dict[str, object] = {"focal": self.anchor}
        if self.fraction is not None:
            entry["focal_fraction"] = float(self.fraction)
        if self.coalescences is not None:
            entry["focal_coalescences"] = int(self.coalescences)
        if self.depth is not None:
            entry["focal_depth"] = float(self.depth)
        return entry

    def describe(self) -> str:
        """The anchor and its placement as one log-ready phrase.

        :return: The anchor name, followed by ``(fraction=0.5)`` and the like
            when a placement is set.
        """
        placement = ", ".join(
            f"{key[len('focal_'):]}={value}"
            for key, value in self.provenance().items() if key != "focal"
        )
        return f"{self.anchor} ({placement})" if placement else self.anchor

    def resolve(
        self,
        tree,
        *,
        ingroup_nodes: "Sequence[int] | np.ndarray | None" = None,
        postorder_rank: "dict[int, int] | None" = None,
    ) -> ResolvedFocal:
        """Resolve against one :class:`tskit.Tree`.

        :param tree: The local tree to resolve against.
        :param ingroup_nodes: Sample node ids making up the ingroup. Required
            for the ``"ingroup_mrca"`` anchor.
        :param postorder_rank: Optional ``{node: postorder index}`` for this
            tree, which turns the MRCA into a single two-argument call.
        :return: The resolved node and offset.
        :raises ValueError: If the anchor needs an ingroup and none was given,
            or if the ingroup has no common ancestor in this tree.
        """
        import tskit

        if self.anchor == "panel_root":
            anchor_node = int(tree.root)
            monophyletic = None
        else:
            if ingroup_nodes is None or len(ingroup_nodes) == 0:
                raise ValueError(
                    'the "ingroup_mrca" anchor needs ingroup_nodes; pass '
                    "ingroup_samples= (or outgroup_samples=) to the inference"
                )
            anchor_node = _mrca(tree, ingroup_nodes, postorder_rank)
            if anchor_node == tskit.NULL:
                raise ValueError(
                    "the ingroup has no common ancestor in this tree (it spans "
                    "several roots)"
                )
            monophyletic = int(tree.num_samples(anchor_node)) == len(ingroup_nodes)

        if self._placement is None:
            return ResolvedFocal(anchor_node, 0.0, monophyletic)

        node, tau = FocalNode._walk_up(
            tree, anchor_node, self.fraction, self.depth, self.coalescences,
        )
        return ResolvedFocal(node, tau, monophyletic)

    @staticmethod
    def _walk_up(
        tree, anchor: int, fraction: float | None, depth: float | None,
        coalescences: int | None = None,
    ) -> tuple[int, float]:
        """Step from ``anchor`` toward the root by a fraction, a count or a depth.

        Returns the node immediately below the resulting point together with the
        remaining distance above it, so the caller is handed an edge and an offset
        along it.

        :param tree: The tree to walk.
        :param anchor: Node to start from.
        :param fraction: Fraction of the anchor-to-root path, or ``None``.
        :param depth: Absolute distance from the sample plane, or ``None``.
        :param coalescences: Number of join events above the anchor, or ``None``.
        :return: ``(node, tau)`` with ``tau`` measured above ``node``; ``tau`` may
            exceed the branch above ``node`` only when ``node`` is the root, which
            is the caller's cue to extend above it.
        """
        import tskit

        path, lengths = [anchor], []
        node = anchor
        while True:
            parent = tree.parent(node)
            if parent == tskit.NULL:
                break
            lengths.append(float(tree.branch_length(node)))
            path.append(int(parent))
            node = int(parent)

        if coalescences is not None:
            # Land on the k-th node up, so the point sits just above that join.
            step = min(int(coalescences), len(path) - 1)
            return path[step], 0.0

        if depth is not None:
            # Absolute, so measured from the sample plane, not the anchor,
            # and clamped below at the anchor: a depth inside the ingroup clade
            # asks about a node the ingroup has no single ancestor at.
            want_above = float(depth) - float(tree.time(anchor))
            if want_above <= 0.0:
                return anchor, 0.0
            for step, length in enumerate(lengths):
                if want_above <= length:
                    return path[step], want_above
                want_above -= length
            return path[-1], want_above  # above the root. The caller extends

        # Only the fraction placement is left. It is bounded by the path itself.
        total = sum(lengths)
        want = min(total * float(fraction or 0.0), total)

        for step, length in enumerate(lengths):
            if want <= length:
                return path[step], want
            want -= length
        return path[-1], 0.0


def _is_degenerate_focal(tree, resolved: "ResolvedFocal") -> bool:
    """Whether re-rooting at ``resolved`` would return a tip's own allele.

    The MRCA of a single ingroup sample is that sample's tip. Re-rooting there
    with no offset makes the tip both the query node and an observation, so the
    posterior collapses onto the allele it carries. A placement above the tip
    walks up to an internal node and is well defined.

    :param tree: The :class:`tskit.Tree` ``resolved`` was resolved against.
    :param resolved: The resolved focal node.
    :return: ``True`` when the focal node is a tip carrying no offset.
    """
    return bool(tree.is_leaf(int(resolved.node)) and resolved.tau <= 0.0)


def _mrca(tree, nodes, postorder_rank: "dict[int, int] | None") -> int:
    """MRCA of ``nodes``, in one call where the post-order ranks are known.

    A post-order traversal is a DFS, so each subtree occupies a contiguous run
    of ranks. The MRCA of a set is therefore the MRCA of its rank-extremes.
    Falls back to a pairwise fold when no ranks are supplied.

    :param tree: The tree to search.
    :param nodes: Node ids whose MRCA to find.
    :param postorder_rank: Optional ``{node: postorder index}``.
    :return: The MRCA node id, or ``tskit.NULL`` if there is none.
    """
    import tskit

    ids = [int(n) for n in nodes]
    if len(ids) == 1:
        return ids[0]
    if postorder_rank is not None:
        ranks = [postorder_rank[n] for n in ids]
        lo = ids[int(np.argmin(ranks))]
        hi = ids[int(np.argmax(ranks))]
        return int(tree.mrca(lo, hi))
    current = ids[0]
    for other in ids[1:]:
        current = int(tree.mrca(current, other))
        if current == tskit.NULL:
            return current
    return current
