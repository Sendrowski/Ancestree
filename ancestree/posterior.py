"""Per-site posterior over candidate ancestral alleles.

:class:`~ancestree.posterior.Posterior` is the return type of every inference
orchestrator. :class:`~ancestree.posterior.Grade` and
:class:`~ancestree.posterior.InferenceSummary` are the aggregate records
returned by :meth:`Inference.grade() <ancestree.inference.Inference.grade>` and
:meth:`Inference.summary() <ancestree.inference.Inference.summary>`.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import tskit

    from ancestree.focal import FocalNode
    from ancestree.sites import PolymorphicSiteFilter


@dataclass(frozen=True, slots=True, eq=False)
class Posterior:
    """A posterior distribution over candidate ancestral alleles at one site.

    Supports dict-style access::

        posterior["A"]                   # 0.7321 (float)
        for allele in posterior: ...     # iterate allele names
        for a, p in posterior.items(): ...
        posterior.to_dict()              # {"A": 0.73, "C": 0.05, ...}

    Convenience properties::

        posterior.map_allele             # "A", allele with highest probability
        posterior.max_prob               # 0.7321

    Vectorised access::

        posterior.values                 # ndarray for numpy ops
    """

    alleles: tuple[str, ...]
    """Allele identifiers, parallel to :attr:`values`. ``alleles[i]`` is the
    allele whose posterior probability is ``values[i]``. Typically the
    model's full alphabet, ``("A", "C", "G", "T")``."""

    values: np.ndarray
    """Length-``len(alleles)`` ndarray of posterior probabilities.
    Non-negative and summing to 1."""

    def __eq__(self, other: object) -> bool:
        """Value equality over :attr:`alleles` and :attr:`values`."""
        if not isinstance(other, Posterior):
            return NotImplemented
        return self.alleles == other.alleles and np.array_equal(
            self.values, other.values
        )

    def __hash__(self) -> int:
        """Hash over :attr:`alleles` and the :attr:`values` entries as
        floats, so instances are usable as set members / dict keys."""
        return hash((self.alleles, tuple(map(float, self.values))))

    def __getitem__(self, allele: str) -> float:
        """Posterior probability of ``allele``.

        :param allele: Allele identifier (must appear in :attr:`alleles`).
        :return: Probability as a Python float.
        :raises KeyError: If ``allele`` is not in this posterior's alphabet.
        """
        try:
            idx = self.alleles.index(allele)
        except ValueError:
            raise KeyError(allele) from None
        return float(self.values[idx])

    def __iter__(self) -> Iterator[str]:
        """Iterate allele labels in :attr:`alleles` order."""
        return iter(self.alleles)

    def __len__(self) -> int:
        """Number of candidate alleles (=
        :attr:`SubstitutionModel.n_states <ancestree.models.SubstitutionModel.n_states>`)."""
        return len(self.alleles)

    def __contains__(self, allele: object) -> bool:
        """``True`` if ``allele`` is in the model's alphabet."""
        return allele in self.alleles

    def items(self) -> Iterator[tuple[str, float]]:
        """Iterate ``(allele, probability)`` pairs in :attr:`alleles` order."""
        return zip(self.alleles, (float(v) for v in self.values))

    def to_dict(self) -> dict[str, float]:
        """Plain ``{allele: probability}`` dict view of the posterior."""
        return dict(self.items())

    @property
    def map_allele(self) -> str:
        """Allele with the maximum a-posteriori probability. Ties resolve to
        the first such allele in :attr:`alleles` order."""
        return self.alleles[int(np.argmax(self.values))]

    @property
    def max_prob(self) -> float:
        """Probability of :attr:`map_allele`."""
        return float(self.values.max())

    def __repr__(self) -> str:
        """Compact ``Posterior({'A': 0.7000, 'C': 0.1000, ...})`` representation."""
        body = ", ".join(f"{a!r}: {p:.4f}" for a, p in self.items())
        return f"Posterior({{{body}}})"


class Grade:
    """How a set of per-site posteriors scored against a known truth.

    Every ``(site, posterior)`` whose ``site.pos`` is a key of the truth
    counts one MAP match, ``posterior.map_allele == truth[pos]``, and adds
    its Brier score against the one-hot truth. Sites absent from the truth
    are skipped. Unpacks positionally as ``(map_recovery, brier, n_sites)``.

    :param posteriors: ``(site, posterior)`` pairs, as
        :meth:`Inference.infer() <ancestree.inference.Inference.infer>`
        yields them. Only ``site.pos`` is read, and the site is handed to
        ``filter``.
    :param truth: ``{position: allele}``, an iterable of such pairs, or a
        :class:`tskit.TreeSequence` carrying the true mutations, read at
        ``focal`` (:meth:`truth_at_focal`).
    :param filter: Optional
        :class:`~ancestree.sites.PolymorphicSiteFilter`, or any callable
        taking a site and returning whether it is scored.
    :param focal: The node of a tree-sequence truth at which the true allele
        is read, as a :class:`~ancestree.focal.FocalNode` or an anchor name.
        Defaults to the ingroup's most recent common ancestor, whichever
        node the posteriors were reported at.
    :param ingroup_samples: Ingroup sample names resolving ``focal`` in a
        tree-sequence truth. Defaults to every sample not named as an
        outgroup.
    :param outgroup_samples: Outgroup sample names resolving ``focal`` in a
        tree-sequence truth.
    :param sample_map: ``{sample: node}`` mapping the sample names onto the
        nodes of a tree-sequence truth, interpreted in the truth tree
        sequence's own node space and not in that of any inference. ``None``
        reads the individual names of the tree sequence.
    :raises ValueError: If the named ingroup matches no sample of a
        tree-sequence truth (:meth:`truth_at_focal`).
    """

    __slots__ = ("map_recovery", "brier", "n_sites")

    #: Fraction of graded sites whose MAP allele is the true one, in
    #: ``[0, 1]``. Higher is better.
    map_recovery: float
    #: Mean Brier score of the per-site posterior against the one-hot truth,
    #: ``sum_a (p_a - 1[a = truth])^2`` over the alleles ``a`` of the model's
    #: alphabet, in ``[0, 2]``. Lower is better. A proper scoring rule, so it
    #: is minimised by reporting the posterior the data support and penalises
    #: a confident error quadratically.
    brier: float
    #: Number of sites graded.
    n_sites: int

    def __init__(
        self,
        posteriors: "Iterable[tuple[Any, Posterior]]",
        truth: "Mapping[int, str] | Iterable[tuple[int, str]] | tskit.TreeSequence",
        filter: "PolymorphicSiteFilter | None" = None,
        *,
        focal: "FocalNode | str" = "ingroup_mrca",
        ingroup_samples: "Sequence[str] | None" = None,
        outgroup_samples: "Sequence[str] | None" = None,
        sample_map: "Mapping[str, int] | None" = None,
    ) -> None:
        import tskit

        if isinstance(truth, tskit.TreeSequence):
            truth = self.truth_at_focal(
                truth, focal,
                ingroup_samples=ingroup_samples,
                outgroup_samples=outgroup_samples,
                sample_map=sample_map,
            )
        truth = self.truth_mapping(truth)
        accepts = getattr(filter, "accepts", filter)
        n = hits = 0
        brier = 0.0
        for site, post in posteriors:
            t = truth.get(int(site.pos))
            if t is None:
                continue
            if accepts is not None and not accepts(site):
                continue
            n += 1
            hits += int(post.map_allele == t)
            # A truth allele outside the model alphabet carries zero mass.
            p_true = float(post[t]) if t in post else 0.0
            brier += float((post.values ** 2).sum()) - 2.0 * p_true + 1.0
        self.map_recovery = hits / n if n else 0.0
        self.brier = brier / n if n else 0.0
        self.n_sites = n

    def __iter__(self) -> Iterator[float]:
        """``(map_recovery, brier, n_sites)``, for positional unpacking."""
        yield self.map_recovery
        yield self.brier
        yield self.n_sites

    def __eq__(self, other: object) -> bool:
        """Equal to another grade, or to a ``(map_recovery, brier, n_sites)`` tuple."""
        if isinstance(other, (Grade, tuple)):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(tuple(self))

    @staticmethod
    def truth_mapping(
        truth: "Mapping[int, str] | Iterable[tuple[int, str]] | tskit.TreeSequence",
    ) -> dict[int, str]:
        """``{position: allele}`` from any of the accepted truth forms.

        :param truth: A mapping, an iterable of ``(position, allele)`` pairs,
            or a :class:`tskit.TreeSequence`, whose site ancestral states are
            the alleles at the ARG root.
        :return: The mapping, positions as integers.
        """
        import tskit

        if isinstance(truth, tskit.TreeSequence):
            return {int(s.position): s.ancestral_state for s in truth.sites()}
        if isinstance(truth, Mapping):
            return {int(k): v for k, v in truth.items()}
        return {int(k): v for k, v in truth}

    @staticmethod
    def truth_at_focal(
        ts: "tskit.TreeSequence",
        focal: "FocalNode | str | None" = None,
        *,
        ingroup_samples: "Sequence[str] | None" = None,
        outgroup_samples: "Sequence[str] | None" = None,
        sample_map: "Mapping[str, int] | None" = None,
    ) -> dict[int, str]:
        """The true allele at a focal node, per site of a tree sequence.

        On each local tree the focal node is resolved as an inference
        resolves it, and the site's allele there is its ancestral state with
        every mutation on the path from the root down to that node applied.
        A placement part-way along a branch excludes the branch's mutations
        above it, where their times are known. Where the ingroup spans
        several roots, or the focal node falls on a tip, the ARG-root state
        is taken, as an inference then reports at the root.

        :param ts: The tree sequence carrying the true mutations.
        :param focal: The :class:`~ancestree.focal.FocalNode`, an anchor
            name, or ``None`` for the ingroup MRCA.
        :param ingroup_samples: Ingroup sample names. ``None`` takes every
            sample not named as an outgroup.
        :param outgroup_samples: Outgroup sample names.
        :param sample_map: ``{sample: node}``, interpreted in the node space
            of ``ts`` and not in that of any inference. ``None`` reads the
            individual names of ``ts``.
        :return: ``{position: allele}`` over every site of ``ts``.
        :raises ValueError: If ``ingroup_samples`` names samples and none of
            them resolves to a node of ``ts``.
        """
        import math

        import tskit

        from ancestree.focal import FocalNode, _is_degenerate_focal, _mrca
        from ancestree.trees import TskitLocalTree

        focal = FocalNode.parse(focal)
        if sample_map is None:
            sample_map = TskitLocalTree.default_sample_map(ts)
        outgroup = set(outgroup_samples or ())
        ingroup = (list(ingroup_samples) if ingroup_samples is not None
                   else [s for s in sample_map if s not in outgroup])
        nodes = tuple(int(sample_map[s]) for s in ingroup if s in sample_map)
        if ingroup and not nodes and focal.anchor != "panel_root":
            raise ValueError(
                f"no ingroup sample of the truth matches {list(ingroup)[:3]}, "
                f"which holds {list(sample_map)[:3]}. Pass sample_map= keyed to it."
            )

        def state_at(tree, site, node, tau):
            """The allele at ``node``, ``tau`` above it, for ``site``."""
            state = site.ancestral_state
            limit = tree.time(node) + tau
            for m in site.mutations:
                if m.node != node and not tree.is_descendant(node, m.node):
                    continue
                if (m.node == node and tau > 0.0 and not math.isnan(m.time)
                        and m.time > limit):
                    continue
                state = m.derived_state
            return state

        out: dict[int, str] = {}
        for tree in ts.trees():
            sites = list(tree.sites())
            if not sites:
                continue
            resolved = None
            if focal.anchor == "panel_root" or nodes:
                ranks = {int(n): i for i, n in enumerate(tree.postorder())}
                if tree.num_roots == 1 or (
                        nodes and _mrca(tree, nodes, ranks) != tskit.NULL):
                    if focal.anchor == "panel_root" and tree.num_roots > 1:
                        resolved = None
                    else:
                        resolved = focal.resolve(
                            tree, ingroup_nodes=nodes, postorder_rank=ranks)
                        if _is_degenerate_focal(tree, resolved):
                            resolved = None
            for site in sites:
                if resolved is None:
                    out[int(site.position)] = site.ancestral_state
                else:
                    out[int(site.position)] = state_at(
                        tree, site, int(resolved.node), float(resolved.tau))
        return out

    def __repr__(self) -> str:
        """Render the scores as a sentence.

        :return: e.g. ``224 sites: MAP recovery 82.6%; mean Brier = 0.284``.
        """
        return (f"{self.n_sites} sites: MAP recovery {self.map_recovery:.1%}; "
                f"mean Brier = {self.brier:.3f}")


@dataclass(frozen=True)
class InferenceSummary:
    """Aggregate diagnostics over a full inference pass.

    Returned by :meth:`Inference.summary() <ancestree.inference.Inference.summary>`. Purely
    descriptive: the site count, the MAP-allele histogram, the spread of MAP
    confidences, and the mean per-site posterior entropy. It draws no
    thresholds and prescribes no filtering.
    """

    #: Number of sites the pass emitted.
    n_sites: int
    #: MAP-allele histogram (``{allele: count}``), most frequent first.
    map_alleles: dict[str, int]
    #: Mean MAP probability over all sites.
    mean_max_prob: float
    #: ``{0.05, 0.25, 0.5, 0.75, 0.95}`` quantiles of the per-site MAP
    #: probability, describing the confidence spread.
    max_prob_quantiles: dict[float, float]
    #: Mean Shannon entropy (bits) of the per-site posterior, ``0`` at one-hot
    #: certainty and ``log2(n_states)`` at uniform.
    mean_entropy_bits: float

    def __str__(self) -> str:
        """Compact multi-line table view, for logging or a short print."""
        if self.n_sites == 0:
            return "InferenceSummary: 0 sites"
        alleles = "  ".join(f"{a}:{c}" for a, c in self.map_alleles.items())
        q = self.max_prob_quantiles
        qline = "  ".join(f"q{int(p * 100):02d} {q[p]:.3f}" for p in sorted(q))
        return (
            f"InferenceSummary: {self.n_sites} sites\n"
            f"  MAP alleles   {alleles}\n"
            f"  MAP prob      mean {self.mean_max_prob:.3f} | {qline}\n"
            f"  posterior H   mean {self.mean_entropy_bits:.3f} bits"
        )
