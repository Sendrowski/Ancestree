"""Showcase gallery: per-(tree, site, model) panel showing what the kernel
returns under a variety of configurations.

Each of the 8 panels constructs a small tree + Site, runs the Felsenstein
kernel directly, and renders (a) the tree with allele letters at the tips
and (b) the posterior over {A, C, G, T} at the inference root as a 4-bar
strip below. Selected to span the corner cases the library handles
natively: multi-allelic sites, missing outgroup tips, monoallelic-ingroup
caterpillar polarisation, recurrent homoplasy, root polytomies, K2 with
transition bias, and fixed-tree no-MLE.

Outputs ``results/reports/gallery.pdf``.

Run::

    python workflow/scripts/gallery.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.special import logsumexp

from ancestree import (
    JC69,
    K2,
    Likelihood,
    OutgroupLadderTree,
    STATES,
    Site,
    TskitLocalTree,
)
from ancestree.plotting import FOCAL_COLOUR
from ancestree.trees import RerootedTree


OUT_PDF = Path("results/reports/gallery.pdf")
STATE_COLORS = {"A": "#1f77b4", "C": "#2ca02c", "G": "#ff7f0e", "T": "#d62728"}


# ---------------------------------------------------------------- demo cases


def case_polymorphic_arg():
    nwk = "((i1:0.001,i2:0.001):0.002,o1:0.02);"
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"i1": "A", "i2": "C", "o1": "A"})
    return dict(title="(1) Polymorphic site, ARG mode",
                note="2 ingroup + 1 outgroup; read at the ingroup MRCA.",
                tree=tree, site=site, model=JC69(),
                ingroup_samples=("i1", "i2"))


def case_polyallelic():
    nwk = "((i1:0.002,i2:0.002):0.003,(o1:0.01,o2:0.01):0.005);"
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=("A", "C", "G"),
                tip_alleles={"i1": "A", "i2": "C", "o1": "G", "o2": "G"})
    return dict(title="(2) Polyallelic site (3 alleles)",
                note="Three distinct alleles at the tips; same kernel, no special case.",
                ingroup_samples=("i1", "i2"),
                tree=tree, site=site, model=JC69())


def case_missing_outgroup():
    nwk = "((i1:0.001,i2:0.001):0.002,o1:0.02);"
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"i1": "A", "i2": "A", "o1": None})
    return dict(title="(3) Missing outgroup tip",
                note="o1=None → all-ones partial; tip marginalises out.",
                tree=tree, site=site, model=JC69())


def case_caterpillar_monoallelic():
    tree = OutgroupLadderTree.from_divergences(
        ["i1"], ["o1", "o2", "o3"], [0.008, 0.024, 0.04],
    ).as_deep_rooted()
    site = Site(chrom="1", pos=1, alleles=("G", "A"),
                tip_alleles={"o1": "G", "o2": "A", "o3": "A"})
    return dict(title="(4) Monoallelic ingroup, deep-rooted ladder tree",
                note="Ingroup fixed for G; deep outgroups vote A. Focal node at the deepest ancestor.",
                tree=tree, site=site, model=JC69())


def case_recurrent_homoplastic():
    # Homoplastic: G appears in both subclades, requiring 2 mutations OR a
    # back-mutation. JC + full recurrence integrates over both histories.
    nwk = "((a:0.05,b:0.05):0.05,(c:0.05,d:0.05):0.05);"
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=("A", "G"),
                tip_alleles={"a": "G", "b": "A", "c": "G", "d": "A"})
    return dict(title="(5) Recurrent / homoplastic site",
                note="G at tips a and c — same allele on independent branches.",
                tree=tree, site=site, model=JC69())


def case_polytomy_root():
    nwk = "(i1:0.005,i2:0.005,i3:0.005,o1:0.03);"
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"i1": "A", "i2": "A", "i3": "C", "o1": "A"})
    return dict(title="(6) Root polytomy",
                note="4-way root polytomy; kernel runs over arbitrary degrees.",
                tree=tree, site=site, model=JC69())


def case_k2_kappa():
    nwk = "((i1:0.003,i2:0.003):0.005,o1:0.03);"
    tree = TskitLocalTree.from_newick(nwk)
    # A↔G is a transition. Under K2(κ=10) this is 10× more likely than A↔C/T.
    site = Site(chrom="1", pos=1, alleles=("A", "G"),
                tip_alleles={"i1": "A", "i2": "G", "o1": "A"})
    return dict(title="(7) K2 with κ = 10 (transition bias)",
                note="A↔G transitions weighted 10× over transversions.",
                ingroup_samples=("i1", "i2"),
                tree=tree, site=site, model=K2(kappa=10.0))


def case_outgroup_ladder_polymorphic():
    # OutgroupLadderTree with a polymorphic ingroup: the standard fixed-tree
    # workflow. Ingroup tips don't enter the kernel (collapsed into the root
    # by design) — they're shown as dashed ghosts. The kernel reads only the
    # outgroup tip alleles. All outgroups vote A, so the inference root sits
    # confidently at A despite the ingroup being split A/C.
    tree = OutgroupLadderTree(ingroup_samples=["i1", "i2"],
                              outgroup_samples=["o1", "o2", "o3"])
    # Ultrametric branch rates so all outgroup tips sit at the same depth.
    tree.set_params(np.array([0.010, 0.010, 0.005, 0.005, 0.005]))  # (K0..K4)
    site = Site(chrom="1", pos=1, alleles=("A", "C"),
                tip_alleles={"i1": "A", "i2": "C",
                             "o1": "A", "o2": "A", "o3": "A"})
    return dict(title="(8) Polymorphic ingroup, OutgroupLadderTree",
                note="Standard fixed-tree workflow; ingroup tips dashed (kernel uses outgroups only).",
                tree=tree, site=site, model=JC69())


CASES = [
    case_polymorphic_arg(),
    case_polyallelic(),
    case_missing_outgroup(),
    case_caterpillar_monoallelic(),
    case_recurrent_homoplastic(),
    case_polytomy_root(),
    case_k2_kappa(),
    case_outgroup_ladder_polymorphic(),
]


# -------------------------------------------------------------- tree drawing


TREE_HEIGHT = 2.0  # fixed visual height per panel. Each tree's deepest tip
                   # is rescaled to this depth so absolute branch lengths
                   # (which vary by orders of magnitude across cases) don't
                   # collapse the layout against the tip-label band.
LEAF_SPACING = 0.5  # horizontal spacing between adjacent leaves.


def _ingroup_mrca(tree, sample_ids) -> "int | None":
    """MRCA node of ``sample_ids``, or ``None`` when it is the tree's own root.

    The Tree protocol exposes children but not parents, so the parent map is
    built from one post-order pass.
    """
    tips = [t for t in (tree.tip_for_sample(s) for s in sample_ids)
            if t is not None]
    if len(tips) < 2:
        return None
    parent = {}
    for node in tree.postorder():
        for c in tree.children(node):
            parent[c] = node

    def ancestry(n):
        chain = [n]
        while n in parent:
            n = parent[n]
            chain.append(n)
        return chain

    common = set(ancestry(tips[0]))
    for t in tips[1:]:
        common &= set(ancestry(t))
    if not common:
        return None
    # The deepest shared ancestor is the one furthest from the root, i.e. with
    # the longest ancestry chain of its own.
    mrca = max(common, key=lambda n: len(ancestry(n)))
    return None if mrca == tree.root else mrca


def _case_focal(case) -> "int | None":
    """The node a case reports at, or ``None`` for the tree's own root.

    Panels whose ingroup is a proper subset of the panel read at the INGROUP
    MRCA, which is the package default and the node the manuscript reports at.
    Reading at the panel root there would show the divergence branch's answer,
    not the polymorphism's.
    """
    ids = case.get("ingroup_samples")
    if not ids:
        return None
    return _ingroup_mrca(case["tree"], ids)


def _sample_for_node(tree, node, sample_ids=()) -> str | None:
    """Tip id for a node, by inverting the public ``tip_for_sample``.

    Not by reaching for ``_tip_to_sample`` / ``_sample_to_node``: those are
    private to particular Tree subclasses and a re-rooted view (``RerootedTree``,
    which ``as_deep_rooted()`` returns) has neither, so every tip silently
    resolved to None and drew as "?".

    :param sample_ids: Candidate ids to invert over, normally the site's
        ``tip_alleles`` keys.
    """
    for sid in sample_ids:
        if tree.tip_for_sample(sid) == node:
            return sid
    # Fall back to the private maps for subclasses that carry them and tips
    # the caller did not name.
    if hasattr(tree, "_tip_to_sample"):
        sid = tree._tip_to_sample.get(node)
        if sid is not None:
            return sid
    for k, v in getattr(tree, "_sample_to_node", {}).items():
        if v == node:
            return k
    return None


def _is_outgroup_leaf(tree, node) -> bool:
    """Convention: tip whose sample id starts with 'o' is an outgroup. Used
    only to enforce the "outgroups on the right" tree-layout invariant."""
    if tree.children(node):
        return False
    sid = _sample_for_node(tree, node)
    return bool(sid and sid.lower().startswith("o"))


def _layout(tree) -> tuple[dict[int, tuple[float, float]], float]:
    """Compute (x, y) coordinates for each node and return the visual→real
    scale factor used. y = depth from root via branch lengths, rescaled to
    ``TREE_HEIGHT`` so each panel uses the full vertical space. x is
    assigned in a left-to-right traversal that puts outgroup-heavier
    subtrees on the right.

    Returns ``(pos, scale)`` where ``scale = TREE_HEIGHT / max_raw_depth``,
    so ``visual_units / scale`` recovers substitutions per site for the
    per-panel scale bar.
    """
    # Depths: cumulative branch length down from the root, then rescaled
    # so the deepest tip sits at TREE_HEIGHT.
    depth: dict[int, float] = {tree.root: 0.0}
    for node in reversed(list(tree.postorder())):
        for c in tree.children(node):
            depth[c] = depth[node] + tree.branch_length(c)
    max_d = max(depth.values()) or 1.0
    scale = TREE_HEIGHT / max_d
    depth = {n: d * scale for n, d in depth.items()}

    # Memoised counts under each subtree: outgroup leaves and ingroup
    # leaves. Used as sort keys so any subtree containing an ingroup leaf
    # ends up on the LEFT and pure-outgroup subtrees on the RIGHT.
    out_count: dict[int, int] = {}
    ing_count: dict[int, int] = {}
    def _count(n: int) -> None:
        kids = tree.children(n)
        if not kids:
            is_out = _is_outgroup_leaf(tree, n)
            sid = _sample_for_node(tree, n)
            is_ing = (not is_out) and sid is not None
            out_count[n] = 1 if is_out else 0
            ing_count[n] = 1 if is_ing else 0
        else:
            for c in kids:
                _count(c)
            out_count[n] = sum(out_count[c] for c in kids)
            ing_count[n] = sum(ing_count[c] for c in kids)
    _count(tree.root)

    # Left-to-right walk: at each internal node, visit children sorted so
    # subtrees containing ANY ingroup leaf come first (left). Among ties,
    # smaller outgroup-count first.
    next_x = [0.0]
    pos: dict[int, tuple[float, float]] = {}
    def _walk(n: int) -> None:
        kids = sorted(
            tree.children(n),
            key=lambda c: (ing_count[c] == 0, out_count[c], c),
        )
        if not kids:
            x = next_x[0]
            next_x[0] += LEAF_SPACING
            pos[n] = (x, depth[n])
            return
        for c in kids:
            _walk(c)
        x = sum(pos[c][0] for c in kids) / len(kids)
        pos[n] = (x, depth[n])
    _walk(tree.root)
    return pos, scale


def _nice_scale_length(real: float) -> float:
    """Round ``real`` down to the largest {1, 2, 5} × 10^k that fits."""
    if real <= 0:
        return 0.0
    exp = math.floor(math.log10(real))
    base = 10 ** exp
    for mult in (5, 2, 1):
        v = mult * base
        if v <= real:
            return v
    return base


def _ladder_tip(ax, x, allele, sid, ghost):
    """Draw one tip (allele circle + sample id) in the gallery's tip style."""
    ec = "#888" if ghost else STATE_COLORS.get(allele, "black")
    if allele:
        ax.text(x, -0.35, allele, ha="center", va="center", fontsize=10,
                fontweight="bold",
                color=("#888" if ghost else STATE_COLORS.get(allele, "black")),
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=ec,
                          lw=(1.0 if ghost else 1.4),
                          linestyle=("--" if ghost else "-")))
    else:
        ax.text(x, -0.35, "?", ha="center", va="center", fontsize=9,
                color="#888",
                bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec="#bbb",
                          lw=1.0, linestyle="--"))
    ax.text(x, -0.85, sid, ha="center", va="top", fontsize=7,
            color=("#999" if ghost else "#666"))


def _draw_outgroup_ladder(ax, tree, site):
    """Schematic of an :class:`OutgroupLadderTree` as a nested ladder.

    The ingroup collapses to a dashed cherry on the left (the kernel reads
    only the outgroups). The outgroups are drawn to its right and join the
    ingroup lineage one at a time --- the closest, :math:`o_1`, most recently,
    the most distant, :math:`o_n`, at the root. This is the EST-SFS
    outgroup-ladder topology of the main-text figure: the outgroups do not
    coalesce into their own clade before meeting the ingroup.
    """
    outs = list(getattr(tree, "outgroup_samples", ()) or ())  # o1..on
    ings = list(getattr(tree, "ingroup_samples", ()) or ())  # i1..
    n_out = len(outs)
    sp = 0.8
    H = TREE_HEIGHT

    # Tip x positions, left to right: the ingroup cherry first, then the
    # outgroups out to the right --- o_1 (closest) nearest the ingroup,
    # o_n (deepest) rightmost.
    tip_x: dict[str, float] = {}
    x = 0.0
    for ig in ings:
        tip_x[ig] = x
        x += sp
    for o in outs:  # o_1 nearest ingroup, o_n deepest rightmost
        tip_x[o] = x
        x += sp
    ing_center = sum(tip_x[ig] for ig in ings) / max(len(ings), 1)

    # Join heights: o1 shallowest, o_n deepest (= root). Ingroup MRCA just
    # below o1's join. Backbone x drifts right as it deepens.
    join_y = {o: H * (k + 1) / n_out for k, o in enumerate(outs)}
    I_y = join_y[outs[0]] * 0.5
    join_x: dict[str, float] = {}
    prev_x = ing_center
    for o in outs:  # o1..on, inward to outward
        join_x[o] = (tip_x[o] + prev_x) / 2.0
        prev_x = join_x[o]

    line = dict(color="#444", linewidth=1.0)
    # Dashed ingroup ghost branches from the ingroup MRCA to each ingroup tip.
    for ig in ings:
        ax.plot([ing_center, tip_x[ig]], [I_y, 0.0], linestyle="--",
                color="#888", linewidth=0.9)
    # Backbone: each outgroup joins the running lineage at its own node.
    prev = (ing_center, I_y)
    for o in outs:
        jx, jy = join_x[o], join_y[o]
        ax.plot([jx, prev[0]], [jy, prev[1]], **line)
        ax.plot([jx, tip_x[o]], [jy, 0.0], **line)
        prev = (jx, jy)

    # Tips.
    for o in outs:
        _ladder_tip(ax, tip_x[o], site.tip_alleles.get(o), o, ghost=False)
    for ig in ings:
        _ladder_tip(ax, tip_x[ig], site.tip_alleles.get(ig), ig, ghost=True)

    # Internal-node markers (ingroup MRCA + each outgroup join). The ingroup
    # MRCA is the focal node here, so it carries the focal marker.
    for px, py in [(join_x[o], join_y[o]) for o in outs]:
        ax.plot([px], [py], marker="o", color="#1f3a93", markersize=7,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.plot([ing_center], [I_y], marker="o", color=FOCAL_COLOUR, markersize=10,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4)

    ax.set_xlim(min(tip_x.values()) - 0.6, max(tip_x.values()) + 0.6)
    ax.set_ylim(-1.2, H + 0.5)
    ax.axis("off")


def _draw_tree(ax, tree, site, focal_override=None):
    """Draw the tree downward (root at top), allele letters at tips."""
    # OutgroupLadderTree gets a dedicated schematic (ingroup cherry on the
    # right, outgroups peeling off to its left and joining one at a time)
    # rather than the generic root-at-top layout, which would otherwise show
    # the outgroups as a clade hanging below the inference root.
    if isinstance(tree, OutgroupLadderTree):
        _draw_outgroup_ladder(ax, tree, site)
        return
    ghost_node = None
    pos, scale = _layout(tree)
    # Flip y so the root is at the top.
    max_y = max(y for _, y in pos.values())
    pos = {n: (x, max_y - y) for n, (x, y) in pos.items()}

    # Branches: straight diagonals from parent → child (cladogram / phylogram
    # style, matching the manuscript's outgroup-ladder figure).
    for node in tree.postorder():
        for c in tree.children(node):
            x0, y0 = pos[node]
            x1, y1 = pos[c]
            ax.plot([x0, x1], [y0, y1], "-", color="#444", linewidth=1.0)

    # Tips: circle with allele letter.
    for node in tree.postorder():
        if tree.children(node) or node == ghost_node:
            continue
        sid = _sample_for_node(tree, node, site.tip_alleles.keys())
        allele = site.tip_alleles.get(sid) if sid else None
        display_sid = sid
        x, y = pos[node]
        if allele is None:
            ax.text(x, y - 0.35, "?", ha="center", va="center", fontsize=9,
                    color="gray", bbox=dict(boxstyle="circle,pad=0.18",
                    fc="white", ec="lightgray"))
        else:
            ax.text(x, y - 0.35, allele, ha="center", va="center", fontsize=10,
                    fontweight="bold", color=STATE_COLORS.get(allele, "black"),
                    bbox=dict(boxstyle="circle,pad=0.18",
                              fc="white", ec=STATE_COLORS.get(allele, "black"),
                              lw=1.4))
        # Sample id below the allele.
        if display_sid:
            ax.text(x, y - 0.85, display_sid, ha="center", va="top",
                    fontsize=7, color="#666")

    # Filled circle at every internal node (root + ladder/coalescent
    # internals). Tips are drawn separately with allele letters. The ingroup
    # MRCA (ghost_node, a leaf in the re-rooted ladder view) is marked too,
    # since the dashed ingroup polytomy hangs from it. The focal node, whose
    # posterior the bar panel shows, is drawn larger and in its own colour.
    focal_node = (focal_override if focal_override is not None
                  else (ghost_node if ghost_node is not None else tree.root))
    for node in tree.postorder():
        if tree.children(node) or node == ghost_node:
            x, y = pos[node]
            is_focal = node == focal_node
            ax.plot([x], [y], marker="o",
                    color=FOCAL_COLOUR if is_focal else "#1f3a93",
                    markersize=10 if is_focal else 7,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)

    # OutgroupLadderTree convention: the ingroup is collapsed into the
    # inference root and has no tips on this tree. In the re-rooted ladder
    # view the ingroup MRCA is the shallowest leaf (``ghost_node``). Hang a
    # dashed ghost polytomy from it so the ingroup is visible (the kernel
    # itself ignores these ghost tips).
    ingroup_samples = getattr(tree, "ingroup_samples", ()) or ()
    # Track any x positions added beyond the real-node layout (ghost polytomy
    # tips) so the xlim calculation at the end can include them.
    extra_xs: list[float] = []

    if ingroup_samples and ghost_node is not None:
        rx, ry = pos[ghost_node]
        n_ing = len(ingroup_samples)
        leaf_y = ry - 0.6
        spread = (n_ing - 1) * LEAF_SPACING
        # Centre the ghost polytomy beneath the ingroup-MRCA node.
        left_x = rx - spread / 2.0
        extra_xs.extend([left_x, left_x + spread])
        # Dashed lines from the ingroup MRCA directly to each ghost tip.
        for i, sid in enumerate(ingroup_samples):
            tx = left_x + i * LEAF_SPACING
            ax.plot([rx, tx], [ry, leaf_y], linestyle="--",
                    color="#888", linewidth=0.9)
            allele = site.tip_alleles.get(sid)
            if allele:
                ax.text(tx, leaf_y - 0.35, allele, ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color=STATE_COLORS.get(allele, "#888"),
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                                  ec="#888", lw=1.0, linestyle="--"))
            else:
                ax.text(tx, leaf_y - 0.35, "?", ha="center", va="center",
                        fontsize=9, color="#888",
                        bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                                  ec="#bbb", lw=1.0, linestyle="--"))
            ax.text(tx, leaf_y - 0.85, sid, ha="center", va="top",
                    fontsize=7, color="#999")

    # Per-panel scale bar in the top-right corner: a horizontal line of
    # visual length ``nice_visual`` corresponding to ``nice_real`` subs/site.
    half_real = TREE_HEIGHT / scale / 2  # subs/site spanning half the panel
    nice_real = _nice_scale_length(half_real)
    nice_visual = nice_real * scale
    xs_layout = [x for x, _ in pos.values()]
    x_right = max(xs_layout) + 0.9
    y_bar = TREE_HEIGHT + 0.3
    ax.plot([x_right - nice_visual, x_right], [y_bar, y_bar],
            color="black", linewidth=1.4)
    # Tick caps.
    for xt in (x_right - nice_visual, x_right):
        ax.plot([xt, xt], [y_bar - 0.06, y_bar + 0.06],
                color="black", linewidth=1.4)
    ax.text(x_right - nice_visual / 2, y_bar + 0.12,
            f"{nice_real:g} subs/site", ha="center", va="bottom",
            fontsize=7, color="black")

    # Tighten axes. Include any ghost-polytomy tips in the lower x bound.
    x_min = min([*xs_layout, *extra_xs]) - 0.5
    ax.set_xlim(x_min, x_right + 0.3)
    # The re-rooted ladder hangs its ghost polytomy below the bottom tip band,
    # so it needs a little more room underneath.
    y_low = -1.9 if ghost_node is not None else -1.2
    ax.set_ylim(y_low, TREE_HEIGHT + 0.85)
    ax.axis("off")


def _fmt_prob(p: float) -> str:
    """Format a posterior probability without lying near the extremes.

    Default to 2 decimals. If that would round a sub-unity value up to
    ``1.00``, escalate precision (``0.99990`` renders as ``0.9999``, not
    ``1.00``). Values below ``0.005`` get no label (visually negligible).
    """
    if p < 0.005:
        return ""
    if p >= 1.0:
        return "1"
    for prec in (2, 3, 4, 5):
        s = f"{p:.{prec}f}"
        # Reject formats that round a sub-unity value up to 1.
        if not s.startswith("1."):
            return s
    return f"{p:.5f}"


def _draw_posterior(ax, posterior: np.ndarray):
    """4-bar chart of the posterior at the focal node over STATES."""
    bars = ax.bar(range(4), posterior, color=[STATE_COLORS[s] for s in STATES],
                  edgecolor="white")
    for b, p in zip(bars, posterior):
        label = _fmt_prob(p)
        if label:
            ax.text(b.get_x() + b.get_width() / 2, max(p, 0) * 1.25 + 0.004,
                    label,
                    ha="center", va="bottom", fontsize=7, color="black")
    ax.set_xticks(range(4))
    ax.set_xticklabels(STATES, fontsize=9)
    # symlog, linear below 0.05: on a linear axis a state holding a few percent
    # of the posterior is a bar a pixel high and reads as zero, which is the
    # opposite of what these panels are for -- several of them turn on a state
    # retaining a small but non-negligible mass. The linear region keeps 0 a
    # real value and the bars proportional where they are large.
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.35)
    ax.set_ylim(0, 1.6)
    ax.set_yticks([0, 1.0])
    ax.set_yticklabels(["0", "1"], fontsize=7)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --------------------------------------------------------- compute + render


def _posterior(case) -> np.ndarray:
    """Uniform prior + Felsenstein (default ``recurrence='full'``) →
    normalised posterior."""
    tree = case["tree"]
    focal = _case_focal(case)
    if focal is not None:
        # Time-reversible models: the posterior at an interior node is the
        # posterior at the root of the same tree re-rooted there.
        tree = RerootedTree(tree, focal, 0.0)
    log_L = Likelihood(case["model"]).log_likelihoods(
        tree, [case["site"]],
    )[0]
    log_post = log_L  # uniform prior is a constant. Cancels under logsumexp
    return np.exp(log_post - logsumexp(log_post))


def render(cases, out_pdf: Path) -> None:
    fig = plt.figure(figsize=(11, 10.6))
    outer = fig.add_gridspec(4, 2, hspace=0.42, wspace=0.18)
    for i, case in enumerate(cases):
        post = _posterior(case)
        sub = outer[i // 2, i % 2].subgridspec(
            2, 1, hspace=0.30, height_ratios=[3, 1],
        )
        ax_tree = fig.add_subplot(sub[0])
        ax_post = fig.add_subplot(sub[1])
        _draw_tree(ax_tree, case["tree"], case["site"], _case_focal(case))
        _draw_posterior(ax_post, post)
        ax_tree.set_title(case["title"], fontsize=10, loc="left", pad=4)
        # Description sits just below the bar chart. Anchor to the posterior
        # axis with clip_on=False so it isn't sliced at the frame.
        ax_post.text(0.0, -0.55, case["note"], transform=ax_post.transAxes,
                     fontsize=7.5, color="#555", va="top", clip_on=False)

    # One legend for the whole figure rather than a "focal" label beside every
    # node: the marker means the same thing in all eight panels.
    handles = [
        Line2D([], [], marker="o", color="none", markerfacecolor=FOCAL_COLOUR,
               markeredgecolor="white", markersize=10,
               label="focal node (posterior reported here)"),
        Line2D([], [], marker="o", color="none", markerfacecolor="#1f3a93",
               markeredgecolor="white", markersize=7, label="internal node"),
        Line2D([], [], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="lightgray", markersize=9,
               label="tip with no observed allele"),
        Line2D([], [], ls="--", color="#999", label="ghost ingroup tip"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.037))

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.3)
    print(f"Wrote {out_pdf}")
    plt.close(fig)

    # Also dump the per-case posteriors so the manuscript table can pull
    # the numbers if desired.
    summary = []
    for case in cases:
        post = _posterior(case)
        summary.append({
            "title": case["title"],
            "note": case["note"],
            "posterior": dict(zip(STATES, post.tolist())),
            "map": STATES[int(np.argmax(post))],
            "max_prob": float(np.max(post)),
        })
    json_path = out_pdf.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    render(CASES, OUT_PDF)
