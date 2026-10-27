"""Simulated ancestral states at the node each site class is graded at.

An ingroup-polymorphic site is graded at the ingroup MRCA ``I``: its truth is
the state at ``I`` and its derived-allele count is the number of ingroup
copies differing from that state. A site the ingroup has fixed is graded at
the root of the full outgroup panel, and is fixed ancestral or fixed derived
against the state there. A site is ingroup-polymorphic when the ingroup
carries at least two non-missing alleles.

``site.ancestral_state`` is the state at the root of the whole simulated tree
sequence. It agrees with the state at ``I`` unless a mutation lies on the path
between the two nodes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import tskit


def truth_at_node(site, tree, nodes) -> str:
    """The simulated state at the MRCA of ``nodes``.

    :param site: The :class:`tskit.Site` carrying the mutations.
    :param tree: The local tree covering it.
    :param nodes: Node ids whose MRCA the state is read at.
    :return: The true state at that node.
    """
    return _state_at(site, tree, tree.mrca(*[int(n) for n in nodes]))


def _state_at(site, tree, node: int) -> str:
    """The simulated state at ``node``: the most recent mutation on or above it.

    :param site: The :class:`tskit.Site` carrying the mutations.
    :param tree: The local tree covering it.
    :param node: Node id, or ``tskit.NULL`` for the site's ancestral state.
    :return: The true state at that node.
    """
    if node == tskit.NULL:
        return site.ancestral_state
    state, best = site.ancestral_state, float("inf")
    for mutation in site.mutations:
        m_node = int(mutation.node)
        if m_node != node and not tree.is_descendant(node, m_node):
            continue
        # A mutation's own time places it on the branch above its node. Two
        # mutations on one branch share that node and are ordered only by it.
        time = float(mutation.time)
        if np.isnan(time):
            time = tree.time(m_node)
        if time < best:
            best, state = time, mutation.derived_state
    return state


@dataclass(frozen=True)
class SiteTruth:
    """The simulated states one site is graded against.

    :param polymorphic: Whether the ingroup carries at least two non-missing
        alleles.
    :param at_ingroup_mrca: State at the ingroup MRCA ``I``.
    :param at_panel_root: State at the root of the full panel, or the state at
        ``I`` when no panel was given.
    :param derived: Ingroup copies differing from :attr:`graded`, missing
        genotypes excluded.
    """
    polymorphic: bool
    at_ingroup_mrca: str
    at_panel_root: str
    derived: int

    @property
    def graded(self) -> str:
        """State at ``I`` for an ingroup-polymorphic site, else at the panel root."""
        return self.at_ingroup_mrca if self.polymorphic else self.at_panel_root

    @property
    def site_class(self) -> str:
        """``polymorphic``, or ``fixed_ancestral`` / ``fixed_derived`` against
        the state at the panel root."""
        if self.polymorphic:
            return "polymorphic"
        return "fixed_derived" if self.derived else "fixed_ancestral"


def site_truths(ts: tskit.TreeSequence, ingroup_nodes: Sequence[int],
                panel_nodes: Sequence[int] | None = None,
                ) -> dict[int, SiteTruth]:
    """Per-site truth and derived count at the node each site class is graded at.

    :param ts: The simulated tree sequence, before any simplification, so the
        mutations above a panel's root are still present.
    :param ingroup_nodes: Sample node ids of the ingroup.
    :param panel_nodes: Sample node ids of the full panel, ingroup included.
        ``None`` grades every site at ``I``.
    :return: Position to :class:`SiteTruth`, for every site of ``ts``.
    """
    ingroup = [int(n) for n in ingroup_nodes]
    panel = None if panel_nodes is None else [int(n) for n in panel_nodes]
    out: dict[int, SiteTruth] = {}
    tree = ts.first()
    mrca = root = None
    for v in ts.variants(samples=ingroup):
        site = v.site
        if mrca is None or not tree.interval.left <= site.position < tree.interval.right:
            tree.seek(site.position)
            mrca = tree.mrca(*ingroup)
            root = mrca if panel is None else tree.mrca(*panel)
        at_i = _state_at(site, tree, mrca)
        at_root = at_i if panel is None else _state_at(site, tree, root)
        alleles = [v.alleles[g] for g in v.genotypes if g != tskit.MISSING_DATA]
        polymorphic = len(set(alleles)) > 1
        graded = at_i if polymorphic else at_root
        out[int(site.position)] = SiteTruth(
            polymorphic=polymorphic, at_ingroup_mrca=at_i,
            at_panel_root=at_root,
            derived=sum(1 for a in alleles if a != graded))
    return out
