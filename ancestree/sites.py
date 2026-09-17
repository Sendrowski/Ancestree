"""Site records and monomorphic-site calibration.

A :class:`~ancestree.sites.Site` is the unit the likelihood engine consumes:
a position with observed alleles at named samples, and an optional handle to
the local tree the inference orchestrator resolves later.
"""
import os
from abc import ABC, abstractmethod
import re
from collections import Counter

#: Trailing per-haplotype suffix a VCF reader appends.
_HAP_SUFFIX = re.compile(r"_h\d+$")
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ancestree import MISSING_ALLELES, STATE_INDEX, STATES, TRANSITION_PAIRS
from ancestree._repr import ReprMixin


@dataclass(frozen=True, slots=True)
class Site:
    """A single polymorphic (or monomorphic) site with observed tip alleles."""

    @staticmethod
    def canonical(allele: "str | None") -> "str | None":
        """The allele as a model state, or ``None`` where it names none.

        One spelling of the case fold every scoring path shares: an allele is
        upper-cased and returned only when it is one of
        :data:`ancestree.STATES`.

        :param allele: The observed allele, in any case.
        :return: The upper-cased state, or ``None``.
        """
        if not isinstance(allele, str):
            return None
        upper = allele.upper()
        return upper if upper in STATE_INDEX else None

    @staticmethod
    def is_called(allele: "str | None") -> bool:
        """Whether an allele records an observation at all.

        :param allele: The observed allele, in any case.
        :return: ``False`` for anything in
            :data:`ancestree.MISSING_ALLELES`, compared upper-cased.
        """
        if not isinstance(allele, str):
            return False
        return allele.upper() not in MISSING_ALLELES

    @staticmethod
    def _is_unrepresentable(allele: "str | None") -> bool:
        """Whether an allele is neither a model state nor a no-call.

        :param allele: The observed allele, compared upper-cased.
        :return: ``True`` for an allele outside both :data:`ancestree.STATES` and
            :data:`ancestree.MISSING_ALLELES`, such as an indel or ``*``.
        """
        if allele is None:
            return False
        upper = allele.upper()
        return upper not in MISSING_ALLELES and upper not in STATE_INDEX


    chrom: str
    """Contig / chromosome identifier."""

    pos: int
    """Position on the contig, in the source's own coordinates: 1-based from
    VCF / VCF-Zarr sources, 0-based from tskit / ARG sources. Writers match
    posteriors to template records on this value, so it is consistent within a
    single source but not normalised across source types."""

    alleles: tuple[str, ...]
    """Observed alleles, REF first. Each entry is a single-character string
    drawn from :attr:`SubstitutionModel.states <ancestree.models.SubstitutionModel.states>` (default
    A/C/G/T). Multi-character alleles (indels) are not handled by the
    likelihood kernel."""

    tip_alleles: Mapping[str, str | None]
    """Mapping from sample identifier to the observed allele at this site
    (must appear in :attr:`alleles`), or ``None`` for missing data.
    Missing tips are marginalised out by the kernel (all-ones partial)."""

    local_tree_handle: Any | None = None
    """Source-specific opaque handle used by the inference orchestrator to
    construct the :class:`~ancestree.trees.Tree` for this site.
    :class:`~ancestree.sources.TskitSource` stores the site position here;
    ``None`` when the same fixed tree applies to every site."""

    info: Mapping[str, Any] = field(default_factory=dict)
    """Free-form auxiliary information a caller may attach to a site. The
    bundled sources leave it empty, and the kernel does not consume it."""

    unphased: "frozenset[str]" = frozenset()
    """The individuals whose call at this site the source read as unphased.
    Where such a call carries two alleles, its haplotype order is drawn at
    random and not the source's own."""

    @classmethod
    def monomorphic(
        cls,
        base: str,
        samples: Iterable[str],
        *,
        chrom: str = "",
        pos: int = 0,
        local_tree_handle: Any | None = None,
        info: Mapping[str, Any] | None = None,
    ) -> "Site":
        """Construct a monomorphic site (all tips at the same allele).

        Useful at the per-site interface (a :class:`~ancestree.sites.SiteSource`
        that yields a record per genome position, polymorphic and monomorphic
        alike). The aggregation path of EST-SFS-style ML fitting takes
        monomorphic counts through
        :meth:`BaseComposition.project_sites_to_configs` as deterministic
        boundary configurations.

        :param base: The single observed allele (must be in the substitution
            model's alphabet, e.g. one of A/C/G/T for the default).
        :param samples: Sample identifiers. Each gets ``base`` as its tip
            allele.
        :param chrom: Optional contig label.
        :param pos: Optional position, in the source's own coordinates
            (see :attr:`pos`).
        :param local_tree_handle: Optional opaque tree handle.
        :param info: Optional auxiliary info dict.
        :return: A :class:`~ancestree.sites.Site` with ``alleles=(base,)`` and every tip
            observed as ``base``.
        """
        return cls(
            chrom=chrom,
            pos=pos,
            alleles=(base,),
            tip_alleles={s: base for s in samples},
            local_tree_handle=local_tree_handle,
            info=info if info is not None else {},
        )

    @property
    def has_representable_allele(self) -> bool:
        """Whether any allele here is one the kernel can score.

        ``False`` where every allele is a no-call or lies outside
        :data:`ancestree.STATES`, so every tip is marginalised and the
        posterior is the prior alone.
        """
        return any(
            a is not None and a.upper() in STATE_INDEX for a in self.alleles
        )

    @property
    def has_unrepresentable_allele(self) -> bool:
        """Whether any allele here is neither a state nor a no-call.

        An indel, ``*``, ``<DEL>`` or any other spelling outside both
        :data:`ancestree.STATES` and :data:`ancestree.MISSING_ALLELES`.
        """
        return any(Site._is_unrepresentable(a) for a in self.alleles)

    def n_unrepresentable_tips(self) -> int:
        """Count tips observed at an allele outside the alphabet.

        :return: Number of :attr:`tip_alleles` entries holding an allele that
            is neither in :data:`ancestree.STATES` nor a no-call. The kernel
            marginalises each of them as it marginalises a missing tip.
        """
        return sum(1 for a in self.tip_alleles.values() if Site._is_unrepresentable(a))

    def count_alleles(self, samples: Sequence[str]) -> Counter:
        """Count canonical (A/C/G/T) tip alleles among the given samples.

        Skips samples with missing tip alleles or alleles outside the
        :data:`ancestree.STATES` alphabet (``N``, gaps, indels).
        Used by the ingroup weights to derive ingroup SFS counts, though it is not
        ingroup-specific: any sample subset may be passed.

        :param samples: Sample identifiers to count over.
        :return: ``Counter`` mapping allele → observed count.
        """
        counts: Counter = Counter()
        wanted = set(samples)
        for sid, a in self.tip_alleles.items():
            if not _named(sid, wanted):
                continue
            a = Site.canonical(a)
            if a is None:
                continue
            counts[a] += 1
        return counts

    def restricted_to(self, names: "Collection[str]") -> "Site":
        """This site with only the tips in ``names``.

        :param names: Tip ids to keep.
        :return: A copy, whose ``alleles`` stay the record's.
        """
        return replace(self, tip_alleles={
            n: a for n, a in self.tip_alleles.items() if n in names})


def _path_format(path: "str | os.PathLike") -> "str | None":
    """The format a path names by its suffix.

    A URL's query and fragment are ignored, so ``https://host/x.vcf.gz?raw=true``
    names a VCF.

    :param path: A file path or URL.
    :return: ``"trees"``, ``"vcz"`` or ``"vcf"``, or ``None`` for any other
        suffix.
    """
    from urllib.parse import urlsplit

    text = str(path)
    if "://" in text:
        text = urlsplit(text).path
    text = text.rstrip("/").lower()
    if text.endswith(".trees"):
        return "trees"
    if text.endswith((".vcz", ".zarr")):
        return "vcz"
    if text.endswith((".vcf", ".vcf.gz", ".vcf.bgz", ".bcf")):
        return "vcf"
    return None


def _individual_of(sample_id: str) -> str:
    """Strip a trailing ``_h<k>`` haplotype suffix from a tip id.

    :param sample_id: Per-haplotype tip id, e.g. ``"i0_h1"``.
    :return: The individual it belongs to, e.g. ``"i0"``, or the id unchanged
        when it carries no haplotype suffix.
    """
    return _HAP_SUFFIX.sub("", sample_id)


def _named(sample_id: str, names: "Collection[str]") -> bool:
    """Whether a haplotype id, or the individual it belongs to, is in ``names``.

    :param sample_id: Per-haplotype tip id, e.g. ``"o0_h1"``.
    :param names: Sample names, each a haplotype id or an individual.
    :return: ``True`` for ``"o0_h1"`` against ``{"o0"}`` or ``{"o0_h1"}``.
    """
    return sample_id in names or _individual_of(sample_id) in names


def _by_individual(ids: "Iterable[str]") -> "frozenset[str]":
    """``ids`` together with the individuals they belong to."""
    ids = tuple(ids)
    return frozenset(ids) | {_individual_of(s) for s in ids}


def _unlabelled(
    samples: "Iterable[str]", named: "Collection[str]", *,
    chosen_by: "str | None" = None,
) -> list[str]:
    """The samples in neither the ingroup nor the outgroups.

    :param samples: Sample ids, matched by id or by individual.
    :param named: The ingroup and outgroup ids together.
    :param chosen_by: The argument that chose ``samples``, named in the error.
    :return: The unlabelled samples, in order.
    :raises ValueError: If ``chosen_by`` is given and a sample is unlabelled.
    """
    unlabelled = [s for s in samples if not _named(s, named)]
    if unlabelled and chosen_by is not None:
        raise ValueError(
            f"{len(unlabelled)} sample(s) in {chosen_by} are in neither "
            f"ingroup_samples nor outgroup_samples: {unlabelled[:5]}. "
            f"Label them, or leave them out of {chosen_by}.")
    return unlabelled


def _refuse_overlap(ingroup: "Iterable[str]", outgroup: "Iterable[str]") -> None:
    """Refuse a sample named as both ingroup and outgroup.

    :param ingroup: Named ingroup ids.
    :param outgroup: Named outgroup ids.
    :raises ValueError: If a sample, or the individual it belongs to, is in
        both lists.
    """
    ins, outs = set(ingroup), set(outgroup)
    both = sorted({s for s in ins if _named(s, outs)}
                  | {s for s in outs if _named(s, ins)})
    if both:
        raise ValueError(
            f"{len(both)} sample(s) are in both ingroup_samples and "
            f"outgroup_samples: {both[:5]}")


def _resolve_panel(
    panel: "Iterable[str]", ingroup: "Iterable[str]",
    outgroup: "Iterable[str]", *, chosen_by: "str | None" = None,
) -> "tuple[list[str], list[str], list[str], list[str]]":
    """The panel, and the ingroup and outgroups within it.

    With both lists named the panel is their union. With one named, the other
    is the rest of the panel, except that the other haplotypes of an outgroup
    individual are dropped. Names match a haplotype id or its individual.

    :param panel: Sample ids, in panel order.
    :param ingroup: Named ingroup ids, possibly none.
    :param outgroup: Named outgroup ids, possibly none.
    :param chosen_by: The argument that chose ``panel``. When given, a sample
        in neither list raises, and is otherwise dropped.
    :return: ``(panel, ingroup, outgroups, dropped)``, each in panel order.
    :raises ValueError: If a named sample is not in the panel or in both
        lists, or if ``chosen_by`` is given and a sample is in neither.
    """
    panel = list(panel)
    ins, outs = set(ingroup), set(outgroup)
    present = _by_individual(panel)
    for label, named in (("ingroup", ins), ("outgroup", outs)):
        absent = sorted(s for s in named if s not in present)
        if absent:
            raise ValueError(
                f"{label} sample(s) not in the panel: {absent[:5]}; the panel "
                f"holds {len(panel)} identifiers starting {panel[:3]}")
    _refuse_overlap(ins, outs)
    dropped: list[str] = []
    if ins and outs:
        dropped = _unlabelled(panel, ins | outs, chosen_by=chosen_by)
    elif outs:
        split = {_individual_of(s) for s in panel if _named(s, outs)}
        dropped = _unlabelled(
            (s for s in panel if _individual_of(s) in split), outs,
            chosen_by=chosen_by)
    gone = set(dropped)
    panel = [s for s in panel if s not in gone]
    if ins:
        members = [s for s in panel if _named(s, ins)]
    else:
        members = [s for s in panel if not _named(s, outs)]
    kept = set(members)
    return panel, members, [s for s in panel if s not in kept], dropped


#: How one site's contribution is keyed while the config histogram is built:
#: its sub-sampled allele-frequency spectrum paired with the outgroup alleles
#: observed at it. The spectrum's length varies with the projection, so the
#: arity is open.
ConfigKey = tuple[tuple[int, ...], tuple["str | None", ...]]


@dataclass(frozen=True, slots=True)
class BaseComposition:
    """Per-data base composition statistics for the inference layer.

    Carries composition, Ts/Tv and per-base count information only, over the
    fixed A/C/G/T alphabet ordered as :data:`ancestree.STATES`. The modelling parameters
    (the mutation rate ``mu``, the target-region length ``n_target_sites``) are set on
    the :class:`~ancestree.inference.Inference` constructors, so one instance serves
    both orchestrators on the same
    dataset. The monomorphic-site counts it supplies are what identify the fixed-tree
    branch-rate MLE. Without them the optimiser inflates the rates.

    .. code-block:: python

        import ancestree as anc

        bc = anc.BaseComposition.from_polymorphic_sites(sites)
        fixed = anc.FixedTreeInference(sites, model, bc,
                                      tree=tree, n_target_sites=L)
        arg = anc.ARGBasedInference(ts, model,
                                   base_composition=bc, mu=1.25e-8)
    """

    counts: Mapping[str, int]
    """Per-nucleotide non-negative counts over the whole target region,
    polymorphic sites included. All four :data:`~ancestree.STATES` keys must
    be present (use ``0`` for unobserved bases). The weights the
    :class:`~ancestree.inference.FixedTreeInference` monomorphic-site term
    multiplies the diagonal transition probability by come from
    :meth:`monomorphic_counts`, which subtracts the polymorphic sites.

    Constructors that do not know the monomorphic counts (notably
    :meth:`from_polymorphic_sites` and :meth:`no_counts`) leave this at
    all-zeros. The inference layer then reconstructs the per-base counts as
    ``pi × (n_target_sites − n_polymorphic)`` from :attr:`pi` and its own
    ``n_target_sites`` argument."""

    n_ts: int = 0
    """Optional: polymorphic-site transition count (A↔G or C↔T) cached at
    :meth:`from_polymorphic_sites` time so :attr:`kappa_estimate` is free."""

    n_tv: int = 0
    """Optional: polymorphic-site transversion count cached at
    :meth:`from_polymorphic_sites` time so :attr:`kappa_estimate` is free."""

    _pi_cache: Any = field(default=None, repr=False, compare=False)
    """Empirical ``π`` precomputed from polymorphic site data (set by
    :meth:`from_polymorphic_sites`). When present it takes precedence over the
    :attr:`counts`-derived ``π``, so the empirical composition is available
    even when :attr:`counts` is left at all-zeros.

    :meta private:
    """

    _pi_is_ascertained: bool = field(default=False, repr=False, compare=False)
    """Whether :attr:`pi` was tallied over variant sites alone, where a base's
    share is weighted by how often it mutates.

    :meta private:
    """

    def __post_init__(self) -> None:
        """Validate ``counts`` has all four A/C/G/T keys with non-negative ints."""
        missing = set(STATES) - set(self.counts)
        if missing:
            raise ValueError(
                f"BaseComposition.counts must contain all four states "
                f"{STATES}; missing: {sorted(missing)}"
            )
        extra = set(self.counts) - set(STATES)
        if extra:
            raise ValueError(
                f"BaseComposition.counts may only contain {STATES}; "
                f"got unexpected keys: {sorted(extra)}"
            )
        for b in STATES:
            v = self.counts[b]
            if not isinstance(v, (int, np.integer)) or v < 0:
                raise ValueError(
                    f"BaseComposition.counts[{b!r}] must be a non-negative "
                    f"int; got {v!r}"
                )

    @classmethod
    def from_counts(
        cls, *, A: int = 0, C: int = 0, G: int = 0, T: int = 0,
        n_ts: int = 0, n_tv: int = 0,
    ) -> "BaseComposition":
        """Construct from explicit per-base counts.

        :param A: Count of A sites over the whole region,
            polymorphic sites included.
        :param C: Count of C sites over the whole region,
            polymorphic sites included.
        :param G: Count of G sites over the whole region,
            polymorphic sites included.
        :param T: Count of T sites over the whole region,
            polymorphic sites included.
        :param n_ts: Optional polymorphic-site transition count
            (used by :attr:`kappa_estimate`).
        :param n_tv: Optional polymorphic-site transversion count
            (used by :attr:`kappa_estimate`).
        :return: A :class:`~ancestree.sites.BaseComposition`.
        """
        return cls(counts={"A": A, "C": C, "G": G, "T": T}, n_ts=n_ts, n_tv=n_tv)

    @classmethod
    def no_counts(cls) -> "BaseComposition":
        """Placeholder composition (all zeros).

        Use when monomorphic sites are iterated as
        :class:`~ancestree.sites.Site` records, or when the inference
        is constructed with an explicit ``n_target_sites`` that supplies
        the monomorphic-site weight. The composition carries no
        monomorphic-site evidence on its own, and in fitting mode
        :class:`~ancestree.inference.FixedTreeInference` raises if
        neither ``n_target_sites=`` nor non-zero counts are supplied.

        :return: A :class:`~ancestree.sites.BaseComposition` with all
            four counts at zero.
        """
        return cls(counts={"A": 0, "C": 0, "G": 0, "T": 0})

    @classmethod
    def from_n_target_sites(
        cls,
        n_total: int,
        *,
        stationary: np.ndarray | None = None,
    ) -> "BaseComposition":
        """Synthesise from a single total + a base distribution.

        Matches fastDFE's ``n_target_sites`` convention: uniform 1/4
        default. Counts are apportioned by the largest-remainder (Hamilton)
        method, so they are always non-negative and sum to exactly
        :attr:`n_total` regardless of how skewed ``stationary`` is.

        :param n_total: Total number of sites to distribute.
        :param stationary: Optional length-4 array over A/C/G/T summing to
            1. Defaults to uniform ``[0.25, 0.25, 0.25, 0.25]``.
        :return: A :class:`~ancestree.sites.BaseComposition` whose
            :attr:`counts` sum to ``n_total``.
        :raises ValueError: If ``n_total`` is negative, or ``stationary``
            is not length-4 or does not sum to 1.
        """
        if n_total < 0:
            raise ValueError(f"n_total must be non-negative; got {n_total}")
        if stationary is None:
            arr = np.full(4, 0.25)
        else:
            arr = np.asarray(stationary, dtype=float)
            if arr.shape != (4,):
                raise ValueError(
                    f"stationary must be length-4; got shape {arr.shape}"
                )
            if not np.isclose(arr.sum(), 1.0):
                raise ValueError(
                    f"stationary must sum to 1; got {arr.sum()}"
                )
        n = cls._largest_remainder(arr, n_total)
        return cls(counts=dict(zip(STATES, [int(x) for x in n])))

    @staticmethod
    def _largest_remainder(weights: np.ndarray, n_total: int) -> np.ndarray:
        """Apportion ``n_total`` integer counts across ``weights`` (summing to
        1) by the largest-remainder (Hamilton) method.

        Floors each ``weights * n_total`` then hands the leftover units to the
        largest fractional parts. Every count is non-negative and the total is
        exactly ``n_total`` for any weight vector.

        :param weights: Non-negative weights summing to ~1 (length-``k``).
        :param n_total: Integer total to distribute.
        :return: Integer count array of the same length as ``weights``.
        """
        w = np.asarray(weights, dtype=float)
        # Normalise so the target sums to exactly n_total.
        w = w / w.sum()
        target = w * n_total
        counts = np.floor(target).astype(int)
        remainder = int(n_total - counts.sum())
        if remainder > 0:
            # Hand the leftover units to the largest fractional parts.
            order = np.argsort(target - counts)[::-1]
            counts[order[:remainder]] += 1
        return counts

    @classmethod
    def from_fasta(
        cls,
        path: str | os.PathLike,
        *,
        contig: str | None = None,
    ) -> "BaseComposition":
        """Count A/C/G/T occurrences in a FASTA file.

        Counting is case-insensitive (lowercase soft-masked bases count as
        their base). Non-ACGT characters (``N``, gaps, IUPAC ambiguity
        codes) are silently skipped.

        :param path: Path to a plain FASTA file.
        :param contig: If given, only count this single contig. Otherwise
            count across all contigs in the file.
        :return: A :class:`~ancestree.sites.BaseComposition`.
        :raises FileNotFoundError: If ``path`` does not exist.
        :raises ValueError: If ``contig`` is given but absent from the file.
        """
        counts = {"A": 0, "C": 0, "G": 0, "T": 0}
        current_contig: str | None = None
        seen_target = contig is None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    parts = line[1:].split(None, 1)
                    current_contig = parts[0] if parts else ""
                    if contig is not None and current_contig == contig:
                        seen_target = True
                    continue
                if contig is not None and current_contig != contig:
                    continue
                for ch in line.upper():
                    if ch in counts:
                        counts[ch] += 1
        if not seen_target:
            raise ValueError(
                f"FASTA file {path} does not contain contig {contig!r}"
            )
        return cls(counts=counts)

    @property
    def n_total(self) -> int:
        """Sum of all four per-base counts (the region's total site count)."""
        return sum(self.counts.values())

    def monomorphic_counts(self, n_polymorphic: int) -> dict[str, int]:
        """Per-base counts of the region's monomorphic positions.

        :attr:`counts` spans the whole target region, so the monomorphic
        weight is that total less the polymorphic sites, re-apportioned across
        A/C/G/T by :attr:`pi`. A composition carrying no counts stays at zero.

        :param n_polymorphic: Number of polymorphic sites in the region.
        :return: Per-base monomorphic-site counts keyed by
            :data:`ancestree.STATES`.
        :raises ValueError: If the region holds fewer sites than
            ``n_polymorphic``.
        """
        if self.n_total == 0:
            return dict(self.counts)
        n_mono = self.n_total - int(n_polymorphic)
        if n_mono < 0:
            raise ValueError(
                f"the composition covers {self.n_total} sites, below the "
                f"{int(n_polymorphic)} polymorphic sites supplied, so no "
                f"monomorphic site is left to calibrate the branch rates and "
                f"they would be inflated by roughly n_total/n_polymorphic. It "
                f"is the length of the target region in bp, not the number of "
                f"variants."
            )
        arr = self._largest_remainder(self.pi, n_mono)
        return dict(zip(STATES, (int(x) for x in arr)))

    @property
    def pi(self) -> np.ndarray:
        """Empirical base frequencies as a length-4 simplex vector.

        The tally cached by :meth:`from_polymorphic_sites` takes precedence,
        then the frequencies derived from :attr:`counts`, then uniform when
        neither carries any signal. Zero entries are floored to ``1e-6`` and
        the vector renormalised so no prior evaluates ``log(0)``.

        :return: Length-4 array over A/C/G/T summing to 1.
        """
        if self._pi_cache is not None:
            return np.asarray(self._pi_cache, dtype=float)
        arr = np.array(
            [float(self.counts[b]) for b in STATES], dtype=float,
        )
        total = arr.sum()
        if total == 0:
            return np.full(4, 0.25)
        pi = arr / total
        pi = np.maximum(pi, 1e-6)
        pi /= pi.sum()
        return pi

    @property
    def kappa_estimate(self) -> float:
        """Empirical κ from :attr:`n_ts` / :attr:`n_tv` (low-divergence MLE).

        Returns :math:`2 \\cdot N_{\\text{Ts}} / N_{\\text{Tv}}`, the
        low-divergence MLE under HKY (or K2) with uniform stationary:
        each starting state has one transition target (rate :math:`\\kappa\\beta`)
        and two transversion targets (rate :math:`\\beta` each), so
        observed transitions per polymorphic site :math:`\\approx \\alpha t`
        and observed transversions :math:`\\approx 2\\beta t`.

        Cached at :meth:`from_polymorphic_sites` time (no extra pass over the data).
        Returns ``2.0`` (a neutral default) when neither :attr:`n_ts` nor
        :attr:`n_tv` were populated (e.g. constructed via :meth:`no_counts`
        or :meth:`from_counts` without the optional kwargs). Otherwise the
        estimate is clamped into ``(0.1, 50.0)``, the box the fit searches,
        so it can be handed to a model as an initial value.

        :return: Estimated transition/transversion ratio κ̂.
        """
        from ancestree.models import _KAPPA_CLAMP

        if self.n_ts == 0 and self.n_tv == 0:
            return 2.0
        if self.n_tv == 0:
            return _KAPPA_CLAMP[1]
        return float(np.clip(2.0 * self.n_ts / self.n_tv, *_KAPPA_CLAMP))

    @classmethod
    def from_polymorphic_sites(
        cls,
        sites: "Iterable[Site]",
        *,
        max_sites: "int | None" = None,
    ) -> "BaseComposition":
        """Empirical π and Ts/Tv from one pass over a site stream.

        Tallies every canonical (A/C/G/T) tip allele across every site for
        :attr:`pi`, and classifies biallelic sites as A↔G / C↔T (transition)
        or otherwise (transversion) for :attr:`n_ts` and :attr:`n_tv`.
        Multi-allelic sites contribute to π only, and sites with no canonical
        tip allele are skipped. :attr:`counts` is left at all zeros.

        A tally over variant sites alone is not the stationary distribution,
        since a base's share is weighted by how readily it mutates. Pass a
        whole-region composition (:meth:`from_fasta`, :meth:`from_counts`)
        where the stationary vector of a base-frequency model is wanted.

        :param sites: :class:`~ancestree.sites.Site` records, possibly lazy.
        :param max_sites: Optional cap on the number of sites consumed.
            ``None`` (default) consumes every site.
        :return: A :class:`~ancestree.sites.BaseComposition` carrying the
            empirical π and the Ts/Tv tallies.
        """
        pi_counts = {b: 0 for b in STATES}
        n_ts = 0
        n_tv = 0
        import itertools
        bounded = (
            sites if max_sites is None
            else itertools.islice(sites, max_sites)
        )
        for site in bounded:
            # Per-tip base tally (used only to derive empirical pi).
            site_alleles: set[str] = set()
            for allele in site.tip_alleles.values():
                allele = Site.canonical(allele)
                if allele is None:
                    continue
                pi_counts[allele] += 1
                site_alleles.add(allele)
            # Polymorphic-site Ts/Tv classification: biallelic only.
            if len(site_alleles) == 2:
                a, b = tuple(site_alleles)
                if (a, b) in TRANSITION_PAIRS:
                    n_ts += 1
                else:
                    n_tv += 1
        arr = np.array([float(pi_counts[b]) for b in STATES], dtype=float)
        total = arr.sum()
        if total == 0:
            pi_cache: np.ndarray | None = None
        else:
            pi = arr / total
            pi = np.maximum(pi, 1e-6)
            pi /= pi.sum()
            pi_cache = pi
        return cls(
            counts={b: 0 for b in STATES},
            n_ts=n_ts, n_tv=n_tv,
            _pi_cache=pi_cache,
            _pi_is_ascertained=True,
        )

    @classmethod
    def require(
        cls,
        bc: "BaseComposition | None",
        *,
        context: str = "EST-SFS mode",
    ) -> "BaseComposition":
        """Guard at EST-SFS entry points: a composition is mandatory.

        If monomorphic sites are part of the :class:`~ancestree.sites.Site`
        stream, pass :meth:`no_counts` as a placeholder.

        :param bc: Caller-supplied composition, or ``None``.
        :param context: Inference-mode label used in the error message.
        :return: ``bc`` unchanged when non-``None``.
        :raises ValueError: When ``bc`` is ``None``.
        """
        if bc is None:
            raise ValueError(
                f"{context} requires a BaseComposition: use "
                ".from_polymorphic_sites(), .from_fasta(), "
                ".from_n_target_sites(), .from_counts(), or .no_counts() "
                "as a placeholder when monomorphic sites are iterated as "
                "Site records."
            )
        return bc

    def project_sites_to_configs(
        self,
        sites: Iterable[Site],
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        subsample_size: int,
    ) -> tuple[list[Site], np.ndarray]:
        """Project polymorphic + monomorphic site data onto a unified
        ``(ingroup_AFS, outgroup_pattern)`` configuration histogram.

        On an :class:`~ancestree.trees.OutgroupLadderTree` the likelihood
        factors through the ingroup MRCA partial, so two sites sharing an
        ``(ingroup_AFS, outgroup_pattern)`` tuple contribute identically to the
        ML fit: the cost is the number of unique tuples, not the number of
        sites. Polymorphic sites are soft-projected to ``subsample_size``,
        monomorphic counts enter as boundary configs, and sites with more than
        two ingroup alleles are skipped here and scored at apply time.

        :param sites: Iterable of full-n polymorphic :class:`~ancestree.sites.Site` records.
        :param ingroup_samples: Ingroup sample ids, in the canonical order
            the inference uses. Tip ids in the returned synthetic Sites are
            drawn from this list (first ``subsample_size`` entries).
        :param outgroup_samples: Outgroup sample ids in the canonical order
            the inference uses. Tip ids in the returned synthetic Sites use
            these directly.
        :param subsample_size: Target sub-sample size for the AFS projection.
            Must be at least 1. A size above the number of called haplotypes at
            a site projects that site onto the baseline, so no upper bound is
            enforced.
        :return: ``(synthetic_sites, weights)``: one :class:`~ancestree.sites.Site` per unique
            config, and a ``(n_configs,) float64`` array of multiplicities,
            summed hypergeometric weights for polymorphic-source configs and
            raw per-base counts for monomorphic-source configs.
        :raises ValueError: If ``subsample_size`` is below 1, or if the
            composition covers fewer sites than ``sites`` yields.
        """
        weights, n_polymorphic = self._project_polymorphic(
            sites, ingroup_samples, outgroup_samples, subsample_size,
        )
        return self._finalize_configs(
            weights, self.monomorphic_counts(n_polymorphic),
            ingroup_samples, outgroup_samples,
            subsample_size,
        )

    def _project_polymorphic(
        self,
        sites: Iterable[Site],
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        subsample_size: int,
    ) -> tuple[dict, int]:
        """Accumulate the polymorphic-site half of the config histogram.

        Runs the per-site hypergeometric projection loop and returns the raw
        (un-finalised) weight dict together with the number of sites the
        iterator yielded. The site count lets
        :class:`~ancestree.inference.FixedTreeInference` derive the
        monomorphic-site total in the same pass, so a streaming source is read
        once. ``_finalize_configs()`` merges the monomorphic counts and
        materialises the config Sites.

        :param sites: Iterable of full-``n`` polymorphic :class:`~ancestree.sites.Site` records.
        :param ingroup_samples: Ingroup sample ids in canonical order.
        :param outgroup_samples: Outgroup sample ids in canonical order.
        :param subsample_size: Target sub-sample size for the AFS projection.
        :return: ``(weights, n_sites_seen)`` where ``weights`` maps
            ``(sub_afs, og_pattern)`` to accumulated multiplicity (monomorphic
            counts not yet merged) and ``n_sites_seen`` counts every yielded
            site, including the multi-allelic ones that are skipped.
        :raises ValueError: If ``subsample_size`` is below 1. The upper bound
            is the ingroup haplotype count, which the caller enforces.
        """
        # Bounded above by the caller against the ingroup HAPLOTYPE count: an
        # id may name an individual, so the id count is not that bound. Sites
        # whose called count falls below the target project onto the baseline.
        if subsample_size < 1:
            raise ValueError(
                f"subsample_size must be >= 1; got {subsample_size}"
            )

        from math import comb
        ingroup_samples = list(ingroup_samples)
        outgroup_samples = list(outgroup_samples)
        weights: dict[ConfigKey, float] = {}

        n_sites_seen = 0
        for site in sites:
            n_sites_seen += 1
            afs_full = [0, 0, 0, 0]
            for allele, count in site.count_alleles(ingroup_samples).items():
                afs_full[STATE_INDEX[allele]] = count
            n_called = sum(afs_full)
            present = [i for i, c in enumerate(afs_full) if c > 0]
            if len(present) > 2:
                continue
            og_pattern = tuple(
                (a.upper() if a is not None else None)
                for a in (site.tip_alleles.get(s) for s in outgroup_samples))
            # No called ingroup info → group on outgroup_pattern alone with
            # a zero-AFS marker (the kernel marginalises ingroup tips when
            # tip_alleles omits them).
            if n_called == 0:
                key: ConfigKey = ((0, 0, 0, 0), og_pattern)
                weights[key] = weights.get(key, 0.0) + 1.0
                continue
            # Called count below sub-sample target → group on raw full-n AFS.
            if n_called < subsample_size:
                key = (tuple(afs_full), og_pattern)
                weights[key] = weights.get(key, 0.0) + 1.0
                continue
            denom = comb(n_called, subsample_size)
            if len(present) == 1:
                # Monomorphic over called ingroup tips → single boundary config.
                i_only = present[0]
                sub_afs = [0, 0, 0, 0]
                sub_afs[i_only] = subsample_size
                key = (tuple(sub_afs), og_pattern)
                weights[key] = weights.get(key, 0.0) + 1.0
            else:
                i_a, i_b = present
                n_a, n_b = afs_full[i_a], afs_full[i_b]
                for k_a in range(max(0, subsample_size - n_b), min(subsample_size, n_a) + 1):
                    k_b = subsample_size - k_a
                    w = comb(n_a, k_a) * comb(n_b, k_b) / denom
                    sub_afs = [0, 0, 0, 0]
                    sub_afs[i_a] = k_a
                    sub_afs[i_b] = k_b
                    key = (tuple(sub_afs), og_pattern)
                    weights[key] = weights.get(key, 0.0) + w

        return weights, n_sites_seen

    def _finalize_configs(
        self,
        weights: dict,
        mono_counts: Mapping[str, int],
        ingroup_samples: Sequence[str],
        outgroup_samples: Sequence[str],
        subsample_size: int,
    ) -> tuple[list[Site], np.ndarray]:
        """Merge monomorphic counts and materialise the config Sites.

        Adds the monomorphic-boundary configs from ``mono_counts`` to the
        polymorphic ``weights`` accumulated by ``_project_polymorphic()``,
        then materialises one synthetic :class:`~ancestree.sites.Site` per unique config.
        Consumes ``weights`` in place (mono mass is added to it).

        :param weights: The dict returned by ``_project_polymorphic()``.
        :param mono_counts: Per-base monomorphic-site counts (``self.counts``
            on the materialised path. The freshly derived counts on the
            streaming path).
        :param ingroup_samples: Ingroup sample ids in canonical order.
        :param outgroup_samples: Outgroup sample ids in canonical order.
        :param subsample_size: Target sub-sample size for the AFS projection.
        :return: ``(synthetic_sites, weights)`` as documented on
            :meth:`project_sites_to_configs`.
        """
        ingroup_samples = list(ingroup_samples)
        outgroup_samples = list(outgroup_samples)
        # One synthetic tip per sub-sampled haplotype. An id naming an
        # individual stands for its whole genotype, so the sub-sample can be
        # wider than the id list. The extra tips carry a haplotype suffix,
        # which Site.count_alleles resolves back to the same individual.
        sub_ingroup_ids = list(ingroup_samples[:subsample_size])
        n_ids = len(ingroup_samples)
        while n_ids and len(sub_ingroup_ids) < subsample_size:
            source = ingroup_samples[len(sub_ingroup_ids) % n_ids]
            sub_ingroup_ids.append(
                f"{source}_h{len(sub_ingroup_ids) // n_ids}")

        # Monomorphic counts enter as boundary configs with all-same-base
        # outgroup patterns, summed onto any polymorphic mass in the same cell.
        for i, base in enumerate(STATES):
            n_b = float(mono_counts.get(base, 0))
            if n_b <= 0:
                continue
            sub_afs = [0, 0, 0, 0]
            sub_afs[i] = subsample_size
            og_pattern = tuple([base] * len(outgroup_samples))
            key = (tuple(sub_afs), og_pattern)
            weights[key] = weights.get(key, 0.0) + n_b

        configs: list[Site] = []
        wts: list[float] = []
        for (sub_afs, og_pattern), w in weights.items():
            tip_alleles: dict[str, str | None] = {}
            cursor = 0
            for state_idx, count in enumerate(sub_afs):
                allele = STATES[state_idx]
                for _ in range(count):
                    tip_alleles[sub_ingroup_ids[cursor]] = allele
                    cursor += 1
            for og_id, og_state in zip(outgroup_samples, og_pattern):
                if og_state is not None:
                    tip_alleles[og_id] = og_state
            present_alleles = tuple(
                STATES[i] for i, c in enumerate(sub_afs) if c > 0
            ) or (STATES[0],)
            configs.append(Site(
                chrom="",
                pos=0,
                alleles=present_alleles,
                tip_alleles=tip_alleles,
            ))
            wts.append(w)

        return configs, np.asarray(wts, dtype=np.float64)


class SiteTable:
    """Columnar store of polymorphic sites, list-like over :class:`~ancestree.sites.Site`.

    Holds positions, contig ids, tree handles, unphased individuals and a ``(n_sites, n_hap) int8``
    genotype matrix indexed through the global ``STATE_INDEX``.

    Supports ``len``, iteration, indexing, slicing and truthiness, rebuilding
    a :class:`~ancestree.sites.Site` only when one is asked for. ``info`` is not stored.

    :param sample_names: Haplotype order the genotype columns follow.
    """

    __slots__ = ("sample_names", "pos", "genotypes", "alleles", "chrom_of",
                 "_chrom_names", "handle", "unphased")

    def __init__(self, sample_names, pos, genotypes, alleles, chrom_of,
                 chrom_names, handle, unphased):
        self.sample_names = tuple(sample_names)
        self.pos = pos
        self.genotypes = genotypes
        self.alleles = alleles
        self.chrom_of = chrom_of
        self._chrom_names = chrom_names
        self.handle = handle
        self.unphased = unphased

    @classmethod
    def from_sites(cls, sites, sample_names=None):
        """Build streaming from any iterable of :class:`~ancestree.sites.Site`.

        The input is consumed one site at a time and the genotype buffer grows
        by doubling, so a generator input is never materialised.

        :param sites: Iterable of :class:`~ancestree.sites.Site`.
        :param sample_names: Haplotype order. Taken from the first site's
            ``tip_alleles`` when omitted.
        :return: A :class:`~ancestree.sites.SiteTable`.
        """
        import numpy as np
        from ancestree import STATE_INDEX

        it = iter(sites)
        first = next(it, None)
        if first is None:
            names = tuple(sample_names or ())
            return cls(names, np.empty(0, np.int64),
                       np.empty((0, len(names)), np.int8), [],
                       np.empty(0, np.int32), [], np.empty(0, np.float64), [])
        names = tuple(sample_names) if sample_names is not None \
            else tuple(first.tip_alleles)
        n_hap = len(names)
        cap = 1024
        pos = np.empty(cap, np.int64)
        handle = np.full(cap, np.nan, np.float64)
        unphased: list = []
        chrom_of = np.empty(cap, np.int32)
        g = np.full((cap, n_hap), -1, np.int8)
        alleles, chrom_names, chrom_id = [], [], {}
        n = 0
        for site in _chain_one(first, it):
            if n == cap:  # double, so the build is linear
                cap *= 2
                pos = np.resize(pos, cap)
                handle = np.resize(handle, cap)
                chrom_of = np.resize(chrom_of, cap)
                grown = np.full((cap, n_hap), -1, np.int8)
                grown[:n] = g[:n]
                g = grown
            c = str(site.chrom)
            ci = chrom_id.get(c)
            if ci is None:
                ci = chrom_id[c] = len(chrom_names)
                chrom_names.append(c)
            chrom_of[n] = ci
            pos[n] = int(site.pos)
            h = site.local_tree_handle
            handle[n] = np.nan if h is None else float(h)
            unphased.append(site.unphased)
            alleles.append(tuple(
                (a.upper() if a is not None else a) for a in site.alleles))
            ta = site.tip_alleles
            row = g[n]
            row[:] = -1
            for hi, name in enumerate(names):
                a = ta.get(name)
                if a is not None:
                    row[hi] = STATE_INDEX.get(a.upper(), -1)
            n += 1
        return cls(names, pos[:n].copy(), g[:n].copy(), alleles,
                   chrom_of[:n].copy(), chrom_names, handle[:n].copy(),
                   unphased)

    def __len__(self):
        return len(self.pos)

    def __bool__(self):
        return len(self.pos) > 0

    def _site(self, i):
        """Rebuild site ``i`` as a :class:`~ancestree.sites.Site`."""
        from ancestree import STATES
        row = self.genotypes[i]
        tip = {name: (STATES[row[hi]] if row[hi] >= 0 else None)
               for hi, name in enumerate(self.sample_names)}
        h = self.handle[i]
        return Site(chrom=self._chrom_names[self.chrom_of[i]],
                    pos=int(self.pos[i]), alleles=self.alleles[i],
                    tip_alleles=tip,
                    local_tree_handle=None if h != h else float(h),
                    unphased=self.unphased[i])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return SiteTable(self.sample_names, self.pos[i], self.genotypes[i],
                             self.alleles[i], self.chrom_of[i],
                             self._chrom_names, self.handle[i],
                             self.unphased[i])
        if i < 0:
            i += len(self)
        return self._site(i)

    def __iter__(self):
        for i in range(len(self)):
            yield self._site(i)

    @property
    def chroms(self):
        """The distinct contig names present, without rebuilding any site."""
        return set(self._chrom_names)


def _chain_one(first, rest):
    """Yield ``first``, then every item of ``rest``."""
    yield first
    yield from rest


class SiteSource(ReprMixin, ABC, Iterable[Site]):
    """An iterable over :class:`~ancestree.sites.Site` records.

    Backends include :class:`~ancestree.sources.TskitSource`,
    :class:`~ancestree.sources.CyVCF2Source`, and
    :class:`~ancestree.sources.VcfZarrSource`. Subclasses are responsible
    for iterating exactly once per record and yielding stable sample
    identifiers via
    :meth:`SiteSource.samples() <ancestree.sites.SiteSource.samples>`.
    """

    #: Whether the unphased-genotype notice has been emitted.
    _noted_unphased: bool = False
    _phase_seed: int = 0
    #: Whether a record carrying more haplotypes than the resolved ploidy has
    #: been reported.
    _warned_ploidy_truncated: bool = False

    @staticmethod
    def _refuse_unsupported_filters(kind, sample_filter, chrom_filter, ploidy,
                                    phased=None, phase_seed=None):
        """Reject filter arguments a source cannot honour.

        :class:`~ancestree.sources.TskitSource` and a pre-built :class:`~ancestree.sites.SiteSource` take the panel
        and contigs as they are.

        :param kind: Description of the source, for the message.
        :raises ValueError: If any filter was supplied.
        """
        given = [n for n, v in (("sample_filter", sample_filter),
                                ("chrom_filter", chrom_filter),
                                ("ploidy", ploidy),
                                ("phased", phased),
                                ("phase_seed", phase_seed)) if v is not None]
        if given:
            raise ValueError(
                f"{', '.join(given)} cannot be applied to {kind}: it carries its "
                f"own panel and contigs. Subset the source before passing it, or "
                f"pass a VCF/VCZ path, which these arguments do apply to.")

    @classmethod
    def resolve(
        cls,
        source,
        *,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        ploidy: int | None = None,
        phased: bool | None = None,
        phase_seed: int | None = None,
    ) -> "SiteSource | Sequence[Site]":
        """Resolve a source to a re-iterable site source without materialising it.

        Dispatches on ``source`` type and returns a re-iterable object, so a
        caller can iterate it more than once. Each ``__iter__`` re-reads from
        the store, keeping peak memory bounded.

        - ``list`` / :class:`~ancestree.sites.SiteSource` → returned as-is.
        - ``str`` / ``os.PathLike`` ending in ``.trees`` →
          :class:`~ancestree.sources.TskitSource`, ``.vcz`` →
          :class:`~ancestree.sources.VcfZarrSource`. Else
          :class:`~ancestree.sources.CyVCF2Source`.
        - :class:`tskit.TreeSequence` → wrapped in
          :class:`~ancestree.sources.TskitSource`.

        :param source: A :class:`~ancestree.sites.SiteSource`, a path
            (VCF / BCF / VCZ / ``.trees``), a :class:`tskit.TreeSequence`, or a
            ``list[Site]``.
        :param sample_filter: Forwarded to the VCF / VCZ source.
        :param chrom_filter: Forwarded to the VCF / VCZ source.
        :param ploidy: Forwarded to :class:`~ancestree.sources.CyVCF2Source`.
        :return: A re-iterable site source.
        :raises TypeError: If ``source``'s type is not recognised.
        """
        if isinstance(source, (SiteSource, list)):
            cls._refuse_unsupported_filters(
                "a pre-built site source", sample_filter, chrom_filter,
                ploidy, phased, phase_seed)
            return source
        if isinstance(source, (str, os.PathLike)):
            fmt = _path_format(source)
            if fmt == "trees":
                import tskit
                from ancestree.sources import TskitSource
                cls._refuse_unsupported_filters(
                    "a tree sequence", sample_filter, chrom_filter,
                    ploidy, phased, phase_seed)
                return TskitSource(tskit.load(source))
            kwargs: dict = {}
            if sample_filter is not None:
                kwargs["sample_filter"] = sample_filter
            if chrom_filter is not None:
                kwargs["chrom_filter"] = chrom_filter
            if phased is not None:
                kwargs["phased"] = phased
            if phase_seed is not None:
                kwargs["phase_seed"] = phase_seed
            if fmt == "vcz":
                from ancestree.sources import VcfZarrSource
                if ploidy is not None:
                    raise ValueError(
                        "ploidy is read from the store's call_genotype array for "
                        "a VCF Zarr source, so it cannot be overridden; drop "
                        "ploidy= or supply the data as VCF / BCF."
                    )
                return VcfZarrSource(source, **kwargs)
            from ancestree.sources import CyVCF2Source
            if ploidy is not None:
                kwargs["ploidy"] = ploidy
            return CyVCF2Source(source, **kwargs)
        try:
            import tskit
            if isinstance(source, tskit.TreeSequence):
                from ancestree.sources import TskitSource
                cls._refuse_unsupported_filters(
                    "a tree sequence", sample_filter, chrom_filter,
                    ploidy, phased, phase_seed)
                return TskitSource(source)
        except ImportError:  # pragma: no cover, tskit is a hard dep
            pass
        raise TypeError(
            f"unsupported source type "
            f"{type(source).__name__!r}. Expected "
            f"a path str/PathLike, a SiteSource, a tskit.TreeSequence, "
            f"or a list[Site]."
        )

    def _note_unphased_once(self) -> None:
        """Report, once, that haplotype assignment is being drawn."""
        if self._noted_unphased:
            return
        self._noted_unphased = True
        self._log.info(
            "Unphased heterozygous calls are present. Their alleles are "
            "assigned to haplotypes at random per site (phase_seed=%d), which "
            "keeps the file's allele order from forming a spurious clade but "
            "leaves the linkage between sites unused.",
            self._phase_seed,
        )


    def _warn_ploidy_truncated(
        self, chrom: str, pos: int, written: int, read: int,
    ) -> None:
        """Report, once, that a record carries more haplotypes than are read.

        :param chrom: Contig of the first such record.
        :param pos: Position of the first such record.
        :param written: Haplotypes that record carries.
        :param read: Haplotypes the source reads.
        """
        if self._warned_ploidy_truncated:
            return
        self._warned_ploidy_truncated = True
        self._log.warning(
            "%s:%d carries %d haplotypes but the source reads %d, so the "
            "trailing calls are dropped and that sample's allele counts are "
            "taken from the leading %d. Ploidy is resolved from the first "
            "called record; pass ploidy=%d to read them all.",
            chrom, pos, written, read, read, written,
        )

    def _warn_split_multiallelic(self, chrom: str, pos: int) -> None:
        """Report a position carrying more than one record.

        A split multiallelic recodes the carriers of every other alternate as
        the reference allele, so each half reports an inflated reference
        frequency and the position is scored more than once. The kernel takes
        poly-allelic sites directly, so the joined form is the one to feed it.

        :param chrom: Contig of the repeated record.
        :param pos: Position of the repeated record.
        """
        self._log.warning(
            "%s:%s carries more than one record, so the input has its "
            "multiallelic sites split across lines. Splitting recodes the "
            "other alternates' carriers as the reference allele, inflating "
            "the reference frequency, and scores the position once per line. "
            "The kernel handles poly-allelic sites directly: rejoin with "
            "`bcftools norm -m+` before annotating.", chrom, pos,
        )

    @abstractmethod
    def __iter__(self) -> Iterator[Site]:
        """Iterate :class:`~ancestree.sites.Site` records in source-canonical order."""
        ...

    @abstractmethod
    def samples(self) -> list[str]:
        """Stable list of sample identifiers that may appear in
        :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.

        The order is the canonical order the source uses, and is the one
        consumers mapping sample ids to tree tips consult.

        :return: Sample identifiers in source-canonical order.
        """

    @staticmethod
    def _decode(x: object) -> str:
        """Coerce a zarr-stored string-or-bytes scalar to a Python ``str``.

        Decodes as UTF-8, matching how htslib / bio2zarr treat VCF text, so a
        non-ASCII sample or contig name round-trips.

        :param x: The stored scalar.
        :return: Its ``str`` form.
        """
        if isinstance(x, (bytes, bytearray)):
            return bytes(x).decode("utf-8")
        return str(x)

    @staticmethod
    def _check_sample_filter(
        wanted: "Sequence[str]", available: "Sequence[str]", path: str,
    ) -> None:
        """Raise if any requested sample is absent from the source or repeated.

        :param wanted: Sample names the caller asked to keep.
        :param available: Sample names the source actually carries.
        :param path: Source path, for the error message.
        :raises KeyError: If any of ``wanted`` is missing.
        :raises ValueError: If any of ``wanted`` appears more than once.
        """
        missing = [s for s in wanted if s not in set(available)]
        if missing:
            raise KeyError(
                f"sample_filter references samples not present in "
                f"{path}: {missing}"
            )
        duplicated = sorted(s for s, n in Counter(wanted).items() if n > 1)
        if duplicated:
            raise ValueError(
                f"sample_filter names the same sample more than once for "
                f"{path}: {duplicated}. Each sample may be kept at most once."
            )

    @staticmethod
    def _haplotype_sample_names(
        sample_names: Sequence[str], ploidy: int,
    ) -> list[str]:
        """Expand sample names by ploidy: ``S`` (haploid) or ``S_h0``/``S_h1`` (diploid+)."""
        if ploidy == 1:
            return list(sample_names)
        return [f"{s}_h{h}" for s in sample_names for h in range(ploidy)]


class _PanelSites(SiteSource):
    """The sites of a source, each restricted to a panel.

    :param source: The sites, re-iterable.
    :param panel: Tip ids to keep, in panel order.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_samples": len(self._panel)}

    def __init__(self, source: "Iterable[Site]", panel: "Sequence[str]") -> None:
        """Store the source and the panel."""
        self._source = source
        self._panel = list(panel)
        self._names = frozenset(panel)

    def __iter__(self) -> Iterator[Site]:
        """Yield each site of the source restricted to the panel."""
        for site in self._source:
            yield site.restricted_to(self._names)

    def samples(self) -> list[str]:
        """The panel, in panel order."""
        return list(self._panel)


class PolymorphicSiteFilter(SiteSource):
    """Filter a site stream down to positions segregating within a sample set.

    Wraps any iterable of :class:`~ancestree.sites.Site` and yields the subset
    whose canonical A/C/G/T alleles, counted over ``samples``, are polymorphic
    (or monomorphic, with ``keep="monomorphic"``). The same object serves as
    an inference's ``source`` and as the ``filter`` of
    :meth:`Inference.grade() <ancestree.inference.Inference.grade>`::

        keep = PolymorphicSiteFilter(samples=ingroup)
        inf.grade(truth, filter=keep)  # score the subset
        subset = keep(source)  # the same subset, as a site source

    Filtering the input is not equivalent to filtering the score.
    :class:`~ancestree.inference.FixedTreeInference` fits its branch rates from
    the ingroup-fixed sites, so a run fitted on the ingroup-polymorphic subset
    alone drives every rate to its lower bound.

    :param source: The stream to filter. Optional: omit it to build a bare
        predicate for grading, then call the instance on a stream to bind one.
    :param samples: Sample identifiers to count over. Defaults to every sample
        the wrapped source reports.
    :param keep: ``"polymorphic"`` (default) or ``"monomorphic"``.
    :raises ValueError: If ``keep`` is neither of those.
    """

    def __init__(
        self,
        source: "Iterable[Site] | None" = None,
        samples: "Sequence[str] | None" = None,
        *,
        keep: str = "polymorphic",
    ) -> None:
        if keep not in ("polymorphic", "monomorphic"):
            raise ValueError(
                f"keep must be 'polymorphic' or 'monomorphic', got {keep!r}"
            )
        # A one-shot iterable would yield an empty second pass, and callers
        # such as FixedTreeInference walk their source more than once.
        if source is not None and iter(source) is source:
            source = list(source)
        self._source = source
        self._samples = list(samples) if samples is not None else None
        self.keep = keep

    def __call__(self, source: "Iterable[Site]") -> "PolymorphicSiteFilter":
        """Bind a stream, returning a filter over it with the same criterion.

        :param source: The stream to filter.
        :return: A new filter over ``source``.
        """
        return PolymorphicSiteFilter(source, self._samples, keep=self.keep)

    def __invert__(self) -> "PolymorphicSiteFilter":
        """The complementary filter over the same samples and source.

        :return: A filter keeping the sites this one rejects.
        """
        other = "monomorphic" if self.keep == "polymorphic" else "polymorphic"
        return PolymorphicSiteFilter(self._source, self._samples, keep=other)

    def accepts(self, site: Site) -> bool:
        """Whether ``site`` belongs to the kept subset.

        :param site: The site to test.
        :return: ``True`` if ``site`` passes the criterion.
        :raises KeyError: If a requested sample is absent from the site, so a
            mistyped identifier cannot classify every site as monomorphic.
        """
        counted = self._samples
        if counted is None:
            counted = list(site.tip_alleles)
        else:
            # An id matches a tip id or the individual a tip belongs to, the
            # same resolution Site.count_alleles applies.
            present = set(site.tip_alleles)
            present |= {_individual_of(s) for s in site.tip_alleles}
            missing = [s for s in counted if s not in present]
            if missing:
                raise KeyError(
                    f"samples absent from the site at {site.chrom}:{site.pos}: "
                    f"{missing[:5]}"
                )
        observed = sum(1 for c in site.count_alleles(counted).values() if c)
        return (observed > 1) if self.keep == "polymorphic" else (observed <= 1)

    def __iter__(self) -> "Iterator[Site]":
        """Iterate the kept subset of the wrapped stream.

        :raises ValueError: If no source was bound.
        """
        if self._source is None:
            raise ValueError(
                "no source bound; construct with one, or call this filter on a "
                "stream, to iterate"
            )
        for site in self._source:
            if self.accepts(site):
                yield site

    def samples(self) -> list[str]:
        """Sample identifiers of the wrapped source, in its canonical order.

        Reports the source's whole panel, not the subset the criterion counts
        over: consumers build their tip panel from this, so narrowing it here
        would drop the outgroups from every tree.

        :return: The wrapped source's sample identifiers.
        :raises ValueError: If no source was bound, or it reports none.
        """
        if self._source is None:
            raise ValueError("no source bound; construct with one to list samples")
        inner = getattr(self._source, "samples", None)
        if callable(inner):
            return list(inner())
        seen: dict[str, None] = {}
        for site in self._source:
            for name in site.tip_alleles:
                seen.setdefault(name, None)
        if seen:
            return list(seen)
        raise ValueError(
            f"{type(self._source).__name__} reports no samples; pass samples="
        )
