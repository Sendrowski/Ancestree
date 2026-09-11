"""Figures for the node a run reports at.

Two views of the same choice. :class:`~ancestree.plotting.FocalTreePlot` draws a tree and marks the
focal node on it, so a figure states where its posterior was read.
:class:`~ancestree.plotting.FocalSweep` moves the reporting position from the ingroup MRCA to the
panel root and returns, or draws, the posterior along the way.

``matplotlib`` is not a dependency of the package, so it is imported when a
figure is actually drawn. :meth:`FocalSweep.posteriors` needs no plotting
backend and can be used on its own.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from ancestree.focal import FocalNode

from ancestree._viz import FOCAL_COLOUR, TreeRenderer

if TYPE_CHECKING:  # pragma: no cover
    from ancestree.models import SubstitutionModel
    from ancestree.sites import Site
    from ancestree.trees import OutgroupLadderTree, Tree

__all__ = ["FOCAL_COLOUR", "FocalTreePlot", "FocalSweep"]


def _require_matplotlib():
    """Import ``matplotlib.pyplot``, or explain how to install it.

    :return: The ``matplotlib.pyplot`` module.
    :raises ImportError: If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "drawing needs matplotlib, which is not a dependency of "
            "ancestree; install it with `pip install ancestree[plotting]` or "
            "`pip install matplotlib`"
        ) from exc
    return plt


class FocalTreePlot:
    """A tree drawn with its focal node marked.

    :param tree: The :class:`~ancestree.trees.Tree` to draw.
    :param focal: The node to mark, as a :class:`~ancestree.focal.FocalNode`,
        an anchor name, or ``None`` for the default anchor.
    :param ingroup_samples: Sample ids making up the ingroup, needed to resolve
        the ``"ingroup_mrca"`` anchor. ``None`` reads
        :attr:`OutgroupLadderTree.ingroup_samples
        <ancestree.trees.OutgroupLadderTree.ingroup_samples>` when the tree
        carries it.
    """

    def __init__(
        self,
        tree: "Tree",
        focal: "FocalNode | str | None" = None,
        *,
        ingroup_samples: Sequence[str] | None = None,
    ) -> None:
        """Lay the tree out once, then resolve the focal node within it."""
        self.tree = tree
        self.focal = FocalNode.parse(focal)
        if ingroup_samples is None:
            ingroup_samples = tree.ingroup_samples
        self.ingroup_samples = tuple(ingroup_samples)
        # One fixed canvas for every focal position, so a marker that moves is
        # the only difference between figures. The deepest node is its top.
        self.canvas = tree.as_deep_rooted() or tree
        self._position, self._depth = self._layout()
        self._parent = {
            kid: node for node in self.canvas.postorder()
            for kid in self.canvas.children(node)
        }

    def _layout(self) -> "tuple[dict[int, tuple[float, float]], dict[int, float]]":
        """Tip-ordered x positions and root-relative depths for every node.

        :return: ``({node: (x, y)}, {node: depth})`` with ``y`` increasing
            toward the root, so the deepest node sits at the top of the axes.
        """
        depth: dict[int, float] = {self.canvas.root: 0.0}
        order: list[int] = []
        stack = [self.canvas.root]
        while stack:
            node = stack.pop()
            order.append(node)
            for kid in self.canvas.children(node):
                depth[kid] = depth[node] + float(self.canvas.branch_length(kid))
                stack.append(kid)
        # Display floor for zero-length branches. Focal resolution uses the
        # true lengths.
        true_depth = dict(depth)
        floor = 0.08 * (max(depth.values()) or 1.0)
        parent_of = {kid: node for node in order
                     for kid in self.canvas.children(node)}
        for node in order[1:]:
            depth[node] = max(depth[node], depth[parent_of[node]] + floor)
        self._true_depth = true_depth

        next_x = [0.0]
        x: dict[int, float] = {}

        def assign(node: int) -> float:
            """Place tips left to right. Internal nodes above their children."""
            kids = list(self.canvas.children(node))
            if not kids:
                x[node] = next_x[0]
                next_x[0] += 1.0
                return x[node]
            spans = [assign(k) for k in kids]
            x[node] = 0.5 * (min(spans) + max(spans))
            return x[node]

        assign(self.canvas.root)
        height = max(depth.values()) or 1.0
        return {n: (x[n], height - depth[n]) for n in order}, depth

    def focal_point(self) -> "tuple[float, float]":
        """Coordinates of the focal position on the drawn tree.

        A position part-way along a branch is placed on that branch, so the
        marker sits exactly where the posterior is read.

        :return: ``(x, y)`` in axes data coordinates.
        """
        node, tau = self._resolve()
        px, py = self._position[node]
        if tau <= 0.0:
            return px, py
        parent = self._parent[node]
        qx, qy = self._position[parent]
        length = float(self.canvas.branch_length(node)) or 1.0
        step = min(tau / length, 1.0)
        return px + step * (qx - px), py + step * (qy - py)

    def _anchor(self) -> int:
        """The node the placement is measured from, in canvas coordinates."""
        if self.focal.anchor == "panel_root":
            return int(self.canvas.root)
        if not self.ingroup_samples:
            raise ValueError(
                'the "ingroup_mrca" anchor needs to know which samples are the '
                "ingroup; pass ingroup_samples=, or anchor on \"panel_root\""
            )
        tips = [self.canvas.tip_for_sample(s) for s in self.ingroup_samples]
        tips = [int(t) for t in tips if t is not None]
        if not tips:
            # A collapsed ingroup has no tips of its own. Its MRCA is a node
            # the tree names directly.
            mrca = self.canvas.ingroup_mrca
            return int(mrca) if mrca is not None else int(self.canvas.root)
        # The MRCA of the ingroup tips: the deepest node every one of them
        # descends from, found by intersecting their root-ward paths.
        paths = []
        for tip in tips:
            path, node = [], tip
            while node is not None:
                path.append(node)
                node = self._parent.get(node)
            paths.append(path)
        common = set(paths[0]).intersection(*(set(p) for p in paths[1:]))
        anchor = min(common, key=lambda n: self._position[n][1])
        # A ladder has no ingroup tip, so the anchor is the MRCA itself.
        if anchor in tips and self._parent.get(anchor) is not None:
            anchor = self._parent[anchor]
        return int(anchor)

    def _resolve(self) -> "tuple[int, float]":
        """The canvas node below the focal position, and the distance above it."""
        anchor = self._anchor()
        if self.focal._placement is None:
            return anchor, 0.0
        path, lengths = [anchor], []
        node = anchor
        while (parent := self._parent.get(node)) is not None:
            lengths.append(float(self.canvas.branch_length(node)))
            path.append(parent)
            node = parent
        if self.focal.coalescences is not None:
            return path[min(int(self.focal.coalescences), len(path) - 1)], 0.0
        total = sum(lengths)
        if self.focal.depth is not None:
            # A ladder measures depth from the ingroup MRCA, a genealogy from
            # the sample plane, both in the tree's own scale.
            limit = getattr(self.tree, "_backbone_limit", None)
            want = (float(self.focal.depth) if limit is not None
                    else float(self.focal.depth) - self._height(anchor))
        else:
            want = total * float(self.focal.fraction or 0.0)
        want = min(max(want, 0.0), total)
        for step, length in enumerate(lengths):
            if want <= length:
                return path[step], want
            want -= length
        return path[-1], 0.0

    def _height(self, node: int) -> float:
        """Distance from the sample plane up to ``node``, on the tree's scale.

        :param node: A canvas node.
        :return: Its height above the shallowest tip.
        """
        return max(self._true_depth.values()) - self._true_depth[node]

    def draw(self, ax=None, *, site: "Site | None" = None, label: str = "focal"):
        """Draw the tree and mark the focal node.

        :param ax: Axes to draw on. ``None`` creates a figure.
        :param site: Optional site whose tip alleles label the tips.
        :param label: Text placed beside the focal marker. ``""`` omits it.
        :return: The axes drawn on.
        """
        plt = _require_matplotlib()
        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 3.0))

        for node in self.canvas.postorder():
            x0, y0 = self._position[node]
            for kid in self.canvas.children(node):
                x1, y1 = self._position[kid]
                ax.plot([x0, x1], [y0, y1], "-", color="#444", lw=1.0,
                        zorder=1)

        known = list(site.tip_alleles) if site is not None else []
        for node in self.canvas.postorder():
            px, py = self._position[node]
            if self.canvas.children(node):
                ax.plot([px], [py], "o", color="#1f3a93", ms=5,
                        markeredgecolor="white", markeredgewidth=0.6, zorder=3)
                continue
            ax.plot([px], [py], "o", color="black", ms=4, zorder=3)
            text = TreeRenderer._tip_label(self.canvas, node, site, known)
            ax.annotate(text, xy=(px, py), xytext=(0, -9),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=8)

        fx, fy = self.focal_point()
        ax.plot([fx], [fy], "o", color=FOCAL_COLOUR, ms=10,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        if label:
            ax.annotate(label, xy=(fx, fy), xytext=(7, 5),
                        textcoords="offset points", fontsize=8,
                        color=FOCAL_COLOUR, ha="left", va="bottom")
        ax.set_axis_off()
        ax.margins(0.14)
        return ax

    def draw_svg(self, *, site: "Site | None" = None, label: str = "focal",
                 **tskit_kwargs) -> str:
        """Draw the tree as SVG and mark the focal node.

        The same figure as :meth:`draw`, rendered through
        :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`.

        :param site: Optional site whose tip alleles label the tips.
        :param label: Text placed beside the focal marker. ``""`` omits it.
        :param tskit_kwargs: Passed through to
            :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`.
        :return: The SVG document as a string.
        """
        return TreeRenderer.draw_svg(
            self.canvas, site=site, focal_at=self._resolve(),
            focal_label=label, **tskit_kwargs,
        )

    @classmethod
    def draw_svg_row(
        cls,
        tree: "Tree",
        positions: "Sequence[tuple[str, FocalNode | str | None]]",
        *,
        site: "Site | None" = None,
        size: tuple[int, int] = (300, 260),
        ingroup_samples: Sequence[str] | None = None,
        **tskit_kwargs,
    ) -> str:
        """Draw the tree once per focal position, side by side in one SVG.

        :param tree: The tree to draw.
        :param positions: ``(caption, focal)`` pairs, one panel each, with
            ``focal`` as the constructor takes it.
        :param site: Optional site whose tip alleles label the tips.
        :param size: ``(width, height)`` of each panel in pixels.
        :param ingroup_samples: As for the constructor.
        :param tskit_kwargs: Passed through to
            :meth:`tskit.Tree.draw_svg() <tskit.Tree.draw_svg>`.
        :return: The SVG document as a string.
        """
        import re
        from html import escape

        width, height = size
        caption_height = 24
        parts = []
        for k, (caption, focal) in enumerate(positions):
            plot = cls(tree, focal, ingroup_samples=ingroup_samples)
            attrs = {"id": f"focal-panel-{k}", "x": k * width,
                     "y": caption_height}
            panel = plot.draw_svg(site=site, size=size,
                                  root_svg_attributes=attrs, **tskit_kwargs)
            parts.append(panel)
            # Centred above the drawn root, whose group carries an absolute
            # offset within its panel.
            root = re.search(r'class="[^"]*\broot\b[^"]*" '
                             r'transform="translate\(([\d.]+)', panel)
            x = k * width + (float(root.group(1)) if root else width / 2)
            parts.append(
                f'<text x="{x}" y="{caption_height - 8}" '
                f'text-anchor="middle" font-size="14px" font-weight="bold">'
                f'{escape(caption)}</text>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width * len(positions)}" '
                f'height="{height + caption_height}">'
                + "".join(parts) + "</svg>")


class FocalSweep:
    """The posterior as the reporting position moves between the two anchors.

    The position runs from the ingroup MRCA at ``0`` to the panel root at
    ``1``, as :attr:`FocalNode.fraction <ancestree.focal.FocalNode.fraction>`
    parameterises it.

    :param tree: The :class:`~ancestree.trees.OutgroupLadderTree` to
        evaluate on. :meth:`FocalSweep.posteriors` reads its
        ``ingroup_samples``, ``ingroup_mrca`` and ``at_focal``, which the
        other trees do not carry.
    :param site: The :class:`~ancestree.sites.Site` to score.
    :param model: Substitution model.
    :param base_composition: Base composition supplying the stationary prior
        and any composition the model needs. ``None`` gives a uniform
        stationary prior, whichever model is in use. Pass the
        composition an inference ran with to reproduce its posteriors, since
        :class:`~ancestree.inference.FixedTreeInference` derives one from the
        data when none is given.
    :param fractions: Positions to evaluate. ``None`` uses eleven evenly
        spaced ones.
    """

    def __init__(
        self,
        tree: "OutgroupLadderTree",
        site: "Site",
        model: "SubstitutionModel",
        *,
        base_composition=None,
        fractions: "Sequence[float] | None" = None,
    ) -> None:
        """Record the positions to evaluate. Nothing is evaluated until asked for."""
        self.tree = tree
        self.site = site
        self.model = model
        self.base_composition = base_composition
        self.fractions = tuple(
            np.linspace(0.0, 1.0, 11) if fractions is None else fractions
        )

    def posteriors(self) -> np.ndarray:
        """Evaluate the posterior at every position.

        :return: ``(n_fractions, n_states)`` of posteriors, each summing to 1.
        """
        from ancestree.likelihood import Likelihood

        engine = Likelihood(self.model, base_composition=self.base_composition)
        out = np.empty((len(self.fractions), self.model.n_states))
        from ancestree.priors import KingmanIngroupWeight, StationaryPrior

        prior = np.exp(
            StationaryPrior(self.model, self.base_composition)
            .log_probs([self.site])[0])
        # The ingroup is seeded on its own MRCA, as in FixedTreeInference.
        ingroup = KingmanIngroupWeight(self.tree.ingroup_samples)
        seeds = {self.tree.ingroup_mrca: np.exp(ingroup.log_probs([self.site]))}
        for i, fraction in enumerate(self.fractions):
            view = self.tree.at_focal(
                FocalNode("ingroup_mrca", fraction=float(fraction))
            )
            log_l = np.asarray(
                engine.log_likelihoods(view, [self.site], node_seeds=seeds)
            )[0]
            weighted = np.exp(log_l - log_l.max()) * prior
            out[i] = weighted / weighted.sum()
        return out

    def draw(self, ax=None, *, alleles: "Sequence[str] | None" = None):
        """Draw the posterior against the reporting position.

        :param ax: Axes to draw on. ``None`` creates a figure.
        :param alleles: Alleles to show. ``None`` shows those the site observed.
        :return: The axes drawn on.
        """
        plt = _require_matplotlib()
        if ax is None:
            _, ax = plt.subplots(figsize=(5.4, 2.4))
        values = self.posteriors()
        states = list(self.model.states)
        shown = alleles if alleles is not None else [
            a for a in self.site.alleles if a in states
        ]
        for allele in shown:
            ax.plot(self.fractions, values[:, states.index(allele)],
                    marker="o", ms=3.5, lw=1.8, label=allele)
        ax.set_xlabel("reporting position, ingroup MRCA to panel root")
        ax.set_ylabel("posterior probability")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["ingroup\nMRCA", "0.25", "0.5", "0.75",
                            "panel\nroot"])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(loc="center right", fontsize=9, title="ancestral allele")
        return ax
