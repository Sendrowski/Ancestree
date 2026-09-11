"""Tree rendering: ASCII layout and the shared tip/root label helpers.

The user-facing entry points are
:meth:`Tree.draw_text() <ancestree.trees.Tree.draw_text>` and
:meth:`Tree.draw_svg() <ancestree.trees.Tree.draw_svg>`.
``TreeRenderer`` provides the ASCII layout, the SVG
layout and the label-formatting helpers both share. SVG goes through
:meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>` for every backend: a
non-tskit tree is copied into a single-tree tree sequence whose node times
reproduce its branch lengths.
"""
from typing import TYPE_CHECKING

#: Colour the focal node is drawn in, shared by every renderer so the marker
#: reads the same whichever one produced the figure.
FOCAL_COLOUR = "#219EBC"

if TYPE_CHECKING:
    import tskit

    from ancestree.posterior import Posterior
    from ancestree.sites import Site
    from ancestree.trees import Tree


class TreeRenderer:
    """Label formatters and generic ASCII tree layout.

    All entry points are pure functions of their arguments.
    """

    @staticmethod
    def format_tip_label(
        sample_id: str | None,
        node: int,
        site: "Site | None",
    ) -> str:
        """Build a tip label, optionally annotated with the observed allele.

        :param sample_id: Caller-facing sample id, or ``None`` if not
            resolvable.
        :param node: Tree node id used as fallback when ``sample_id`` is
            ``None``.
        :param site: Optional :class:`~ancestree.sites.Site` providing the allele overlay.
        :return: Label string like ``"n0=A"``, ``"n0=?"``, or just ``"n0"``.
        """
        base = sample_id if sample_id is not None else str(node)
        if site is None or sample_id is None:
            return base
        allele = site.tip_alleles.get(sample_id)
        if allele is None:
            return f"{base}=?"
        return f"{base}={allele}"

    @staticmethod
    def format_root_label(
        posterior: "Posterior | None",
        base: str = "root",
    ) -> str:
        """Build a root label, optionally annotated with the MAP state.

        :param posterior: :class:`~ancestree.posterior.Posterior` over the
            root state, read via its
            :attr:`Posterior.map_allele <ancestree.posterior.Posterior.map_allele>`
            and
            :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
            properties. ``None`` leaves the label unannotated.
        :param base: Prefix string for the label (e.g. ``"root"`` or
            ``f"node {node}"``).
        :return: Label string like ``"root [MAP=A 0.8732]"`` or just ``base``.
        """
        if posterior is None:
            return base
        return f"{base} [MAP={posterior.map_allele} {posterior.max_prob:.4f}]"

    @classmethod
    def draw_text(
        cls,
        tree: "Tree",
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        show_branch_lengths: bool = True,
    ) -> str:
        """ASCII layout used when no backend-specific renderer overrides
        :meth:`Tree.draw_text() <ancestree.trees.Tree.draw_text>`.

        Tip labels resolve to ``sample_id=allele`` if a ``site`` is given
        and the tip's sample id is known via :meth:`Tree.tip_for_sample() <ancestree.trees.Tree.tip_for_sample>`.
        Otherwise to the node id.

        :param site: Optional :class:`~ancestree.sites.Site` for tip-allele overlay.
        :param posterior: Optional :class:`~ancestree.posterior.Posterior`
            appended to the root label as its MAP allele and probability.
        :param show_branch_lengths: Append ``[bl=...]`` to every non-root
            label.
        :return: ASCII tree as a single multi-line string (no trailing
            newline).
        """
        known_samples = list(site.tip_alleles.keys()) if site is not None else []
        lines: list[str] = []

        def label_internal(node: int, with_bl: bool) -> str:
            """Render an internal-node label (root gets the posterior MAP overlay)."""
            label = tree.node_name(node) or f"node {node}"
            if node == tree.root:
                label = cls.format_root_label(posterior, base=label)
            if with_bl and node != tree.root:
                label += f" [bl={tree.branch_length(node):.4g}]"
            return label

        def label_tip(node: int, with_bl: bool) -> str:
            """Render a tip label (sample id + allele overlay if ``site`` given)."""
            label = cls._tip_label(tree, node, site, known_samples)
            if with_bl:
                label += f" [bl={tree.branch_length(node):.4g}]"
            return label

        def visit(node: int, line_prefix: str, child_prefix: str) -> None:
            """Depth-first emit lines for ``node`` and its subtree."""
            children = list(tree.children(node))
            if children:
                lines.append(line_prefix + label_internal(node, show_branch_lengths))
                for i, c in enumerate(children):
                    is_last = i == len(children) - 1
                    connector = "└── " if is_last else "├── "
                    visit(
                        c,
                        line_prefix=child_prefix + connector,
                        child_prefix=child_prefix + ("    " if is_last else "│   "),
                    )
            else:
                lines.append(line_prefix + label_tip(node, show_branch_lengths))

        visit(tree.root, "", "")
        return "\n".join(lines)


    @classmethod
    def draw_svg(
        cls,
        tree: "Tree",
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        focal_at: "tuple[int, float] | None" = None,
        focal_label: str = "focal",
        **tskit_kwargs,
    ) -> str:
        """SVG layout for any :class:`~ancestree.trees.Tree` backend.

        :param tree: The tree to render.
        :param site: Optional :class:`~ancestree.sites.Site` for tip-allele overlay.
        :param posterior: Optional :class:`~ancestree.posterior.Posterior`
            appended to the root label as its MAP allele and probability.
        :param focal_at: ``(node, tau)`` marking where the posterior is read:
            a node of ``tree`` and the distance above it along the edge into
            it, in the tree's own branch units. A non-zero ``tau`` is drawn as
            a point on that branch.
        :param focal_label: Text placed beside the focal marker. ``""`` omits it.
        :param tskit_kwargs: Passed through to
            :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`.
        :return: The SVG document as a string.
        """
        ts_tree, labels, focal_id = cls._as_tskit_tree(
            tree, site=site, posterior=posterior, focal_at=focal_at,
        )
        if focal_id is not None:
            name = labels.get(focal_id, "")
            labels[focal_id] = (f"{name} ({focal_label})" if name and focal_label
                                else focal_label or name)
            # Scoped to this figure's root id, so several figures composed
            # into one document keep their own marker.
            scope = tskit_kwargs.get("root_svg_attributes", {}).get("id")
            selector = f".node.n{focal_id}"
            if scope:
                selector = f"#{scope} {selector}"
            style = (f"{selector} > .sym {{fill: {FOCAL_COLOUR}; "
                     f"stroke: white; r: 5px}}"
                     f" {selector} > .lab {{fill: {FOCAL_COLOUR}}}")
            tskit_kwargs["style"] = style + tskit_kwargs.get("style", "")
        return ts_tree.draw_svg(node_labels=labels, **tskit_kwargs)

    @classmethod
    def _as_tskit_tree(
        cls,
        tree: "Tree",
        *,
        site: "Site | None" = None,
        posterior: "Posterior | None" = None,
        focal_at: "tuple[int, float] | None" = None,
    ) -> "tuple[tskit.Tree, dict[int, str], int | None]":
        """Copy ``tree`` into a one-tree tree sequence, with its node labels.

        The copy is drawn ultrametric: every tip sits at height zero and an
        internal node at the longest path down to a tip below it, in the
        tree's own branch units. :mod:`tskit` requires a parent strictly above
        its children, so a zero-length branch is given a display floor of a
        thousandth of the tree height.

        :param tree: The tree to copy.
        :param site: Optional site supplying the tip-allele overlay.
        :param posterior: Optional posterior supplying the root overlay.
        :param focal_at: ``(node, tau)`` in the input tree's own ids and units.
            A non-zero ``tau`` splices a unary node at that height so the mark
            sits on the branch itself.
        :return: ``(tskit.Tree, {node: label}, focal node id or None)``, node
            ids being the copy's.
        """
        import tskit

        parent_of: dict[int, int] = {}
        order = [tree.root]
        stack = [tree.root]
        while stack:
            node = stack.pop()
            for kid in tree.children(node):
                parent_of[kid] = node
                order.append(kid)
                stack.append(kid)

        def heights(floor: float) -> dict[int, float]:
            """Tip-anchored node heights, children before parents."""
            out: dict[int, float] = {}
            for node in reversed(order):
                kids = list(tree.children(node))
                out[node] = max(
                    (out[k] + max(float(tree.branch_length(k)), floor)
                     for k in kids), default=0.0)
            return out

        floor = (heights(0.0)[tree.root] or 1.0) / 1000.0
        height_of = heights(floor)

        known_samples = list(site.tip_alleles.keys()) if site is not None else []
        tables = tskit.TableCollection(sequence_length=1.0)
        new_id: dict[int, int] = {}
        labels: dict[int, str] = {}
        focal_node = (int(focal_at[0]) if focal_at is not None
                      and float(focal_at[1]) <= 0.0 else None)
        focal_tip = (focal_node is not None
                     and not list(tree.children(focal_node)))
        for node in order:
            is_tip = not list(tree.children(node))
            # A focal tip is drawn as an interior node, a circle, not a
            # sample square.
            sample = is_tip and not (focal_tip and node == focal_node)
            new_id[node] = tables.nodes.add_row(
                flags=tskit.NODE_IS_SAMPLE if sample else 0,
                time=height_of[node],
            )
            if is_tip:
                labels[new_id[node]] = cls._tip_label(
                    tree, node, site, known_samples)
            elif node == tree.root and posterior is not None:
                labels[new_id[node]] = cls.format_root_label(posterior)
            else:
                labels[new_id[node]] = cls._svg_name(tree.node_name(node) or "")
        focal_id = None
        spliced = None
        if focal_at is not None:
            node, tau = int(focal_at[0]), float(focal_at[1])
            parent = parent_of.get(node)
            if tau <= 0.0 or parent is None:
                focal_id = new_id[node]
            else:
                # Strictly between the two, so tskit accepts both edges.
                lo, hi = height_of[node], height_of[parent]
                at = min(max(lo + tau, lo + floor / 2), hi - floor / 2)
                focal_id = tables.nodes.add_row(flags=0, time=at)
                labels[focal_id] = ""
                spliced = (node, parent, focal_id)

        for kid, parent in parent_of.items():
            if spliced is not None and kid == spliced[0]:
                continue
            tables.edges.add_row(left=0.0, right=1.0,
                                 parent=new_id[parent], child=new_id[kid])
        if spliced is not None:
            node, parent, mid = spliced
            tables.edges.add_row(left=0.0, right=1.0, parent=mid,
                                 child=new_id[node])
            tables.edges.add_row(left=0.0, right=1.0, parent=new_id[parent],
                                 child=mid)
        tables.sort()
        return tables.tree_sequence().first(), labels, focal_id

    @staticmethod
    def _svg_name(name: str) -> str:
        """``x_k`` rendered as ``x`` with ``k`` in subscript.

        :param name: A node name.
        :return: The SVG text, with a ``tspan`` for the subscript.
        """
        head, sep, sub = name.partition("_")
        if not sep:
            return name
        return (f'{head}<tspan baseline-shift="sub" font-size="75%">'
                f'{sub}</tspan>')

    @classmethod
    def _tip_label(
        cls,
        tree: "Tree",
        node: int,
        site: "Site | None",
        known_samples: list[str],
    ) -> str:
        """Name one tip: its backend name, else its sample id and allele.

        A tip the backend names carries no observed allele, such as the
        fixed-tree ladder's collapsed ingroup once a re-rooted view turns it
        into a tip, so the name stands alone.

        :param tree: The tree being rendered.
        :param node: Tip node id.
        :param site: Optional site supplying the allele overlay.
        :param known_samples: Candidate sample ids for the reverse lookup.
        :return: The label.
        """
        name = tree.node_name(node)
        if name is not None:
            return name
        sample_id = cls._sample_id_for_node(tree, node, known_samples)
        return cls.format_tip_label(sample_id, node, site)

    @staticmethod
    def _sample_id_for_node(
        tree: "Tree", node: int, known_samples: list[str],
    ) -> str | None:
        """Name the sample at a tip: the tree's own reverse map first, then
        ``known_samples`` searched linearly.

        :param tree: The tree whose tip is being named.
        :param node: Tip node id to name.
        :param known_samples: Candidate sample ids, typically the keys of
            :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
        :return: Sample id, or ``None`` if no candidate maps to ``node``.
        """
        named = tree.sample_for_tip(node)
        if named is not None:
            return named
        for s in known_samples:
            if tree.tip_for_sample(s) == node:
                return s
        return None
