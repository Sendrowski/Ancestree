"""Per-site ingroup weights and the prior at the reporting node.

* :class:`~ancestree.priors.StationaryPrior`: the per-data base composition
  (:attr:`BaseComposition.pi <ancestree.sites.BaseComposition.pi>`), or
  uniform :math:`1/S` when none is supplied.
  Site-invariant, with zero free parameters. Goes in ``prior=``. The only
  one ARG and local-tree mode accept, and their default.

* :class:`~ancestree.priors.KingmanIngroupWeight`: closed-form
  count-proportional ingroup weight, the likelihood of the ingroup alleles
  given the state at the ingroup MRCA, not a prior. Algebraically identical
  to the Kingman SFS formula for a biallelic ingroup and extended to
  multi-allelic sites by count weighting. Zero free parameters. Fixed-tree
  mode only, where it is the default.

* :class:`~ancestree.priors.AdaptiveIngroupWeight`: per-bin
  probabilities fit from the data on top of the Kingman baseline, for when
  the SFS departs from neutrality. Converges to Kingman in the large-data
  neutral limit. Fixed-tree mode only.

The two ingroup weights go in ``ingroup_weight=`` and assume the single
shared ingroup-MRCA polytomy that only fixed-tree mode has, so ARG and
local-tree mode reject them.
"""
import logging
import os
from abc import ABC
from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import hypergeom

from ancestree import STATE_INDEX
from ancestree.settings import Settings
from ancestree.sites import BaseComposition, Site
from ancestree._repr import ReprMixin

if TYPE_CHECKING:
    from ancestree.models import SubstitutionModel




def _fit_pi_bin(
    data: list[tuple[float, float, float]],
    n_runs: int,
    seed: int,
) -> float:
    """L-BFGS-B fit of a single ``π_i`` with multistart from ``seed``.

    Module-level helper (picklable for :class:`~concurrent.futures.ProcessPoolExecutor`).
    ``data`` is the list of ``(p_major, p_minor, weight)`` per-site
    likelihoods for the bin. The weight is the hypergeometric
    probability that the site lands in this bin under the configured
    ``subsample_size`` (1.0 when no subsampling is in effect).
    ``n_runs`` independent L-BFGS-B starts from initial values drawn
    uniformly on ``[0.05, 0.95]`` pick the best by objective.
    """
    if not data:
        return 0.5  # no data → keep the symmetric default
    rng = np.random.default_rng(seed)
    p_major_arr = np.array([d[0] for d in data])
    p_minor_arr = np.array([d[1] for d in data])
    weight_arr = np.array([d[2] for d in data])

    def neg_log_L(pi_vec: np.ndarray) -> float:
        """Negative weighted log-mixture likelihood at this ``π``."""
        pi = float(pi_vec[0])
        mix = pi * p_major_arr + (1 - pi) * p_minor_arr
        mix = np.maximum(mix, 1e-300)  # guard against log(0)
        return -float((weight_arr * np.log(mix)).sum())

    best, moved = None, False
    for _ in range(n_runs):
        x0 = rng.uniform(0.05, 0.95, size=1)
        result = minimize(
            neg_log_L, x0, bounds=[(1e-9, 1 - 1e-9)], method="L-BFGS-B",
        )
        if abs(float(result.x[0]) - float(x0[0])) > 1e-8:
            moved = True
        if best is None or result.fun < best.fun:
            best = result
    if not moved:
        logging.getLogger("ancestree.AdaptiveIngroupWeight").warning(
            "per-bin weight is unidentified (the objective is flat in pi); "
            "using the symmetric 0.5 instead of a random start"
        )
        return 0.5
    return float(best.x[0])


class IngroupWeight(ReprMixin, ABC):
    """Per-site log vector of ``P(ingroup alleles | ingroup MRCA = s)``.

    The collapsed ingroup contributes this one vector, seeded on its own MRCA
    through
    :paramref:`node_seeds <ancestree.likelihood.Likelihood.log_likelihoods.node_seeds>`,
    so it enters inside Felsenstein's recursion and propagates to whatever node
    the run reports at. It is a likelihood, not a prior over ancestral states:
    the prior at the reporting node is
    :class:`~ancestree.priors.StationaryPrior`.

    Subclasses must implement :meth:`log_probs`. The optional :meth:`fit`
    defaults to a no-op: :class:`~ancestree.priors.KingmanIngroupWeight` is
    closed-form, while :class:`~ancestree.priors.AdaptiveIngroupWeight`
    overrides it to fit one probability per spectrum class.
    """

    def _log_prior_one(self, site: Site) -> np.ndarray:
        """Per-site log weight vector. The unit :meth:`log_probs` maps over.

        :param site: The site to score.
        :return: Length-4 ``log P(ingroup alleles | I = s)`` over states ``s``,
            for the ingroup MRCA ``I``, in :data:`ancestree.STATES` order.
        """
        raise NotImplementedError

    def log_probs(self, sites: Sequence[Site]) -> np.ndarray:
        """Per-site log weight for the collapsed ingroup.

        :param sites: Polymorphic :class:`~ancestree.sites.Site` records.
        :return: ``(len(sites), 4)`` array of ``log P(ingroup alleles | I = s)``
            for the ingroup MRCA ``I`` and state ``s``, in
            :data:`ancestree.STATES` order. Unobserved alleles get ``-inf``.
        """
        out = np.empty((len(sites), 4), dtype=float)
        for idx, site in enumerate(sites):
            out[idx] = self._log_prior_one(site)
        return out

    def fit(
        self,
        sites: Sequence[Site],
        log_L_per_site: np.ndarray,
    ) -> None:
        """Optionally fit free parameters from data (default no-op).

        The all-at-once wrapper around the incremental
        :meth:`AdaptiveIngroupWeight.begin_fit() <ancestree.priors.AdaptiveIngroupWeight.begin_fit>`
        / :meth:`AdaptiveIngroupWeight.accumulate() <ancestree.priors.AdaptiveIngroupWeight.accumulate>`
        / :meth:`AdaptiveIngroupWeight.end_fit() <ancestree.priors.AdaptiveIngroupWeight.end_fit>`
        protocol that :class:`~ancestree.inference.FixedTreeInference` drives.

        :param sites: Same sequence passed to :meth:`log_probs`.
        :param log_L_per_site: ``(N, 4)`` log-likelihoods at the fitted
            tree.
        """


class NoIngroupWeight(IngroupWeight):
    """Flat ingroup term: the outgroups alone carry the ancestral signal.

    Every state scores ``log 1 = 0``, so the collapsed ingroup contributes a
    uniform factor and the posterior at the reporting node is the prior times
    the outgroup likelihood. Pass as ``ingroup_weight=`` where the ingroup
    allele frequencies should not inform the call, for instance to isolate how
    much the outgroups contribute on their own.
    """

    def _log_prior_one(self, site: Site) -> np.ndarray:
        """Flat log vector, independent of the site.

        :param site: The site to score, unused.
        :return: Length-4 zero vector in :data:`ancestree.STATES` order.
        """
        return np.zeros(4)


class StationaryPrior(ReprMixin):
    """Per-site prior equal to the per-data base composition.

    :math:`\\pi(s) = \\pi_s` for every site, where :math:`\\pi` is the
    :class:`~ancestree.sites.BaseComposition`-supplied base composition,
    or uniform :math:`1/S` when none is supplied. The model itself never
    enters the prior, only its alphabet size, so the prior is non-uniform
    whenever the supplied composition is, including under JC69 and K2.

    Zero free parameters. The per-site log-prior is the same length-``S``
    vector for every site (computed once at construction and broadcast).

    :param model: The substitution model, kept on the prior so callers
        can resolve the alphabet (``model.n_states``) without a separate
        argument.
    :param base_composition: Optional :class:`~ancestree.sites.BaseComposition`
        whose :attr:`BaseComposition.pi <ancestree.sites.BaseComposition.pi>` provides the
        per-state log-prior. ``None`` falls back to uniform.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"model": type(self.model)}

    model: "SubstitutionModel"
    """Substitution model, used for the alphabet (``n_states``) only."""

    def __init__(
        self,
        model: "SubstitutionModel",
        base_composition: BaseComposition | None = None,
    ) -> None:
        """Cache :math:`\\log \\pi` from ``base_composition`` (uniform default)."""
        from ancestree.models import SubstitutionModel  # local: import cycle
        if not isinstance(model, SubstitutionModel):
            raise TypeError(
                f"StationaryPrior expected a SubstitutionModel, got "
                f"{type(model).__name__}"
            )
        self.model = model
        if base_composition is not None:
            pi = base_composition.pi
        else:
            pi = np.full(model.n_states, 1.0 / model.n_states)
        with np.errstate(divide="ignore"):
            self._log_prior_per_state: np.ndarray = np.log(pi)

    def log_probs(self, sites: Sequence[Site]) -> np.ndarray:
        """``(N, S)`` log-prior. Identical row for every site."""
        return np.broadcast_to(
            self._log_prior_per_state, (len(sites), self.model.n_states)
        ).copy()


class KingmanIngroupWeight(IngroupWeight):
    """Count-proportional per-site ingroup weight.

    The likelihood of the observed ingroup alleles given the state at the
    ingroup MRCA, not a prior over ancestral states. For a site with present
    ingroup alleles ``{a₁: n₁, …, a_k: n_k}`` summing to ``n_obs``, the
    weight is ``P(ingroup alleles | MRCA = a_j) ∝ n_j / n_obs``.
    For a biallelic site this collapses to Kingman's ``(n - i)/n`` and
    ``i/n`` weights. For 3+ segregating ingroup alleles count-proportional
    weighting is the simplest extension that reduces exactly to that
    biallelic case. Unobserved alleles get zero mass. Zero free parameters.
    A mono-allelic ingroup gives a delta on the allele it carries, and an
    unobserved ingroup gives uniform.

    :param ingroup_samples: Sample ids for the ingroup haplotypes,
        must match the keys used in :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
    :raises ValueError: If ``ingroup_samples`` is empty.
    """

    @staticmethod
    @lru_cache(maxsize=8192)
    def _hypergeom_pmf(n_obs: int, n_minor: int, n_sub: int) -> np.ndarray:
        """Hypergeometric mass over subsample bins ``0..n_sub``, cached: it depends
        only on ``(n_obs, n_minor, n_sub)``, not on the site's likelihoods, and is
        otherwise recomputed once per site."""
        m = hypergeom.pmf(np.arange(n_sub + 1), M=n_obs, n=n_minor, N=n_sub)
        m.setflags(write=False)
        return m

    @staticmethod
    def _count_proportional_log_prior(counts, n_obs: int) -> np.ndarray:
        """Kingman's count-proportional per-site log weight over the alphabet.

        ``P(ingroup alleles | I = a) ∝ n_a / n_obs``, for the ingroup MRCA ``I``,
        the count ``n_a`` of allele ``a`` and the total ``n_obs``. Unobserved
        alleles get ``-inf``.

        :param counts: Allele → observed ingroup count.
        :param n_obs: Total observed ingroup count.
        :return: Length-4 log weight in :data:`ancestree.STATES` order.
        """
        out = np.full(4, -np.inf)
        with np.errstate(divide="ignore"):
            for allele, c in counts.items():
                out[STATE_INDEX[allele]] = np.log(c / n_obs)
        return out


    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_ingroup": self.n_ingroup}

    ingroup_samples: list[str]
    """Sample ids for the ingroup haplotypes, must match the keys used in
    :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`."""

    n_ingroup: int
    """Number of ingroup haplotypes (``len(ingroup_samples)``)."""

    def __init__(self, ingroup_samples: Sequence[str]) -> None:
        """Store the ingroup sample ids the weight is computed over."""
        self.ingroup_samples = list(ingroup_samples)
        self.n_ingroup = len(self.ingroup_samples)
        if self.n_ingroup == 0:
            raise ValueError(
                "KingmanIngroupWeight requires at least one ingroup sample."
            )

    def _log_prior_one(self, site: Site) -> np.ndarray:
        """Per-site log vector over A/C/G/T of ``P(ingroup alleles | I = s)``.

        A mono-allelic ingroup gives a delta on the observed allele, an
        unobserved one gives uniform, and otherwise the weights are
        count-proportional.

        :param site: The site to weight.
        :return: Length-4 log vector.
        """
        return self._log_prior_from_counts(
            site.count_alleles(self.ingroup_samples))

    def _log_prior_from_counts(self, counts: Counter) -> np.ndarray:
        """The same log vector from a tally the caller supplies.

        :param counts: Canonical ingroup allele counts at one site, as
            :meth:`Site.count_alleles() <ancestree.sites.Site.count_alleles>`
            returns them over :attr:`ingroup_samples`.
        :return: Length-4 log vector.
        """
        folded: Counter = Counter()
        for allele, n in counts.items():
            state = Site.canonical(allele)
            if state is not None:
                folded[state] += n
        counts = folded
        n_obs = sum(counts.values())
        if n_obs == 0:
            return np.full(4, -np.log(4))
        if len(counts) == 1:
            out = np.full(4, -np.inf)
            (observed,) = counts
            out[STATE_INDEX[observed]] = 0.0
            return out
        return KingmanIngroupWeight._count_proportional_log_prior(counts, n_obs)

    def log_probs_from_counts(
        self, counts: Sequence[Counter],
    ) -> np.ndarray:
        """Per-site log weights from tallies the caller supplies.

        :param counts: One count tally per site, in site order, each over
            :attr:`ingroup_samples`.
        :return: ``(len(counts), 4)`` array, as :meth:`KingmanIngroupWeight.log_probs() <ancestree.priors.KingmanIngroupWeight.log_probs>`
            returns it.
        """
        out = np.empty((len(counts), 4), dtype=float)
        for idx, c in enumerate(counts):
            out[idx] = self._log_prior_from_counts(c)
        return out


class AdaptiveIngroupWeight(IngroupWeight):
    """Per-bin ancestral-state probabilities fit by L-BFGS-B from data.

    Generalises :class:`~ancestree.priors.KingmanIngroupWeight`: in place of
    the closed-form ``(n - i)/n`` weights for biallelic sites, one free
    parameter ``π_j`` per SFS-class bin ``j`` is fitted in a canonical
    subsampled space (folded, ``π_{n_sub-j} = 1 - π_j`` by symmetry). Each
    site contributes hypergeometric mass across the reachable bins (see
    ``subsample_size``), so sites with different missing-data patterns share
    one bin coordinate. The per-bin mixture likelihood

    .. math::

        L(\\pi_i) = \\sum_{\\text{sites in bin } i}
            \\log \\bigl[ \\pi_i \\, p_{\\text{major}}(\\text{site})
                  + (1 - \\pi_i) \\, p_{\\text{minor}}(\\text{site}) \\bigr]

    is maximised independently for each bin, where ``p_major`` and ``p_minor``
    are the Felsenstein likelihoods of the site given the major and the minor
    allele at the ingroup MRCA, at the fitted tree rates, supplied through
    :meth:`accumulate`.

    Multi-allelic sites take the count-proportional Kingman baseline, a
    monomorphic ingroup a delta on its allele, and an unobserved ingroup
    uniform, as in :class:`~ancestree.priors.KingmanIngroupWeight`. Until a
    fit completes the ``π_j`` sit at their Kingman values ``(n_sub - j)/n_sub``,
    which are also the neutral-coalescent truth, so the fitted weight converges
    to Kingman on large neutral data.

    :param ingroup_samples: Sample ids for the ingroup haplotypes, matching
        the keys used in :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
    :param n_runs: Independent L-BFGS-B starts per bin. Default ``4``.
    :param seed: Seed for the multistart sampling. Default ``0``.
    :param min_bin_n_sites: Weighted sites per fittable bin below which
        :meth:`end_fit` warns about MLE instability. Default ``20``. The
        warning also fires for a bin pegged at a parameter bound.
    :param parallelize: Dispatch the per-bin fits across worker processes.
        Default ``False``.
    :param subsample_size: Canonical SFS-bin size each site is projected
        onto by hypergeometric subsampling. ``None`` (default) is
        ``min(n_ingroup, 11)``. Sites with ``n_called < subsample_size`` take
        the count-proportional Kingman weight.
    :raises ValueError: If ``ingroup_samples`` is empty, or
        ``subsample_size`` is < 2 or > ``n_ingroup``.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_ingroup": self.n_ingroup, "subsample_size": self.subsample_size}

    #: Sample ids for the ingroup haplotypes, matching the keys of
    #: :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
    ingroup_samples: list[str]

    n_ingroup: int
    """Number of ingroup haplotypes (``len(ingroup_samples)``)."""

    #: Number of ingroup haplotypes each site is projected down to before
    #: binning by minor-allele count.
    subsample_size: int

    #: Number of random starts per per-bin fit.
    n_runs: int

    #: Weighted site count per bin below which :meth:`end_fit` warns about
    #: an unstable per-bin estimate.
    min_bin_n_sites: int

    #: Whether the per-bin fits run in a process pool.
    parallelize: bool

    #: Seed for the multistart fits.
    seed: int

    fitted: bool
    """``True`` once a fit has completed, ``False`` before."""

    def __init__(
        self,
        ingroup_samples: Sequence[str],
        *,
        n_runs: int = 4,
        seed: int = 0,
        min_bin_n_sites: int = 20,
        parallelize: bool = False,
        subsample_size: int | None = None,
    ) -> None:
        """Store ingroup samples + multistart config. Default π_i to Kingman."""
        self.ingroup_samples = list(ingroup_samples)
        self.n_ingroup = len(self.ingroup_samples)
        if self.n_ingroup == 0:
            raise ValueError(
                "AdaptiveIngroupWeight requires at least one ingroup sample."
            )
        self.n_runs = int(n_runs)
        self.seed = int(seed)
        self.min_bin_n_sites = int(min_bin_n_sites)
        self.parallelize = bool(parallelize)

        if subsample_size is None:
            # A one-haplotype ingroup resolves to 1 and never fits a bin.
            self.subsample_size = min(self.n_ingroup, 11)
            self._subsample_size_was_default = True
        else:
            ss = int(subsample_size)
            if ss < 2 or ss > self.n_ingroup:
                raise ValueError(
                    f"subsample_size must be in [2, n_ingroup={self.n_ingroup}]; "
                    f"got {ss}"
                )
            self.subsample_size = ss
            self._subsample_size_was_default = False

        # Per-bin π_j = P(major ancestral | minor count = j, sample size = subsample_size).
        # Default to Kingman until a fit runs.
        self._pi: dict[int, float] = self._kingman_default()
        self._n_sites_per_bin: dict[int, float] = {}
        self._n_sites_fallback_kingman: int = 0
        self.fitted: bool = False

    def _kingman_default(self) -> dict[int, float]:
        """Per-bin Kingman closed form ``(n_sub - j) / n_sub`` as the pre-fit baseline."""
        n = self.subsample_size
        return {j: (n - j) / n for j in range(n + 1)}

    @property
    def pi(self) -> dict[int, float]:
        """Per-bin ancestral-state probabilities ``{j: π_j}`` in the subsampled
        space (``j ∈ [0, subsample_size]``). Read-only view."""
        return dict(self._pi)

    @property
    def n_sites_per_bin(self) -> dict[int, float]:
        """Per-bin (fractional) hypergeometric site mass, keyed by raw
        subsample bin ``j`` including the deterministic endpoints. Empty until
        a fit completes. Under hypergeometric subsampling each site contributes a
        distribution across bins, so values are fractional sums of
        hypergeometric probabilities, not integers."""
        return dict(self._n_sites_per_bin)

    @property
    def n_sites_fallback_kingman(self) -> int:
        """Polymorphic sites that fell below ``subsample_size`` and were
        excluded from the per-bin fit (handled via Kingman fallback at apply
        time). Diagnostic for how permissive the fit is on partial-ingroup
        data."""
        return int(self._n_sites_fallback_kingman)

    def fit(
        self,
        sites: Sequence[Site],
        log_L_per_site: np.ndarray,
    ) -> None:
        """Fit per-bin ``π_j`` from biallelic sites, projecting to ``subsample_size``.

        Each biallelic site with ``n_called >= subsample_size`` contributes
        hypergeometric mass to every reachable subsampled bin ``j``:

        .. math::

            m_j(\\text{site})
                = \\mathrm{hypergeom.pmf}(j;\\, M=n_{\\text{called}},\\,
                                          n=k_{\\text{minor}},\\,
                                          N=n_{\\text{sub}}),

        and the per-bin weighted log-mixture
        :math:`\\sum_j m_j \\, \\log(\\pi_j \\, p_{\\text{major}} +
        (1-\\pi_j) \\, p_{\\text{minor}})` is maximised independently per
        fittable bin :math:`j \\in [1, (n_{\\text{sub}}+1)//2)` (folded.
        The upper half mirrors via :math:`\\pi_{n_{\\text{sub}}-j} = 1 -
        \\pi_j`). Sites with ``n_called < subsample_size`` are
        excluded from the fit and counted in
        :attr:`n_sites_fallback_kingman`. They re-enter at apply time
        via the count-proportional Kingman fallback.

        Monomorphic-ingroup sites and multi-allelic sites are handled
        directly by :meth:`log_probs` via the Kingman baseline (delta /
        count-proportional respectively) and do not enter :meth:`fit`.

        :param sites: Polymorphic :class:`~ancestree.sites.Site` records.
        :param log_L_per_site: ``(N, 4)`` log-likelihood at the fitted
            tree, one row per site.
        """
        if self._subsample_size_was_default:
            self._log.info(
                "The subsample size defaulted to "
                "min(n_ingroup=%d, 11) = %d. Pass subsample_size=N "
                "explicitly to override (or =n_ingroup to disable "
                "projection on full-observation sites).",
                self.n_ingroup, self.subsample_size,
            )

        self.begin_fit()
        self.accumulate(sites, log_L_per_site)
        self.end_fit()

    def begin_fit(self) -> None:
        """Reset the per-bin accumulators to start an incremental fit.

        Pair with :meth:`accumulate` (one or more batches), then
        :meth:`end_fit`. Each bin keeps ``{(p_major, p_minor): summed
        weight}``, so the accumulator is bounded by the number of distinct
        outgroup patterns and a batched fit equals an all-at-once one.
        """
        self._acc_bins: dict[int, dict[tuple[float, float], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._acc_bin_weight_totals: dict[int, float] = defaultdict(float)
        self._acc_n_fallback = 0

    def accumulate(
        self, sites: Sequence[Site], log_L_per_site: np.ndarray,
    ) -> None:
        """Fold one batch of sites + their per-site log-likelihoods into the
        running fit (between :meth:`begin_fit` and :meth:`end_fit`)."""
        n_sub = self.subsample_size
        for site_idx, site in enumerate(sites):
            counts = site.count_alleles(self.ingroup_samples)
            n_obs = sum(counts.values())
            if len(counts) != 2:
                continue  # monomorphic / multi-allelic: not for the fit
            (major, _), (minor, n_minor) = counts.most_common(2)
            if n_obs < n_sub:
                # Too few called alleles to project up to the canonical
                # bin space. Site re-enters at apply via Kingman fallback.
                self._acc_n_fallback += 1
                continue

            p_major = float(np.exp(log_L_per_site[site_idx, STATE_INDEX[major]]))
            p_minor = float(np.exp(log_L_per_site[site_idx, STATE_INDEX[minor]]))

            # Hypergeometric mass per subsampled bin j ∈ [0, n_sub].
            m = KingmanIngroupWeight._hypergeom_pmf(int(n_obs), int(n_minor), int(n_sub))

            for j in range(n_sub + 1):
                w = float(m[j])
                if w == 0.0:
                    continue
                self._acc_bin_weight_totals[j] += w
                # j=0 and j=n_sub are deterministic endpoints (monomorphic
                # in the subsample). They do not enter the L-BFGS-B fit. The
                # middle bin of even n_sub is its own fold mirror, fixed at 0.5.
                if j == 0 or j == n_sub:
                    continue
                if n_sub % 2 == 0 and j == n_sub // 2:
                    continue
                # Fold onto the lower mirror: pi_j*p_major + (1-pi_j)*p_minor
                # equals pi_mirror*p_minor + (1-pi_mirror)*p_major.
                bin_lo = min(j, n_sub - j)
                if j == bin_lo:
                    self._acc_bins[bin_lo][(p_major, p_minor)] += w
                else:
                    self._acc_bins[bin_lo][(p_minor, p_major)] += w

    def end_fit(self) -> None:
        """Finalise an incremental fit started with :meth:`begin_fit`."""
        bins = {
            j: [(pm, pn, w) for (pm, pn), w in d.items()]
            for j, d in self._acc_bins.items()
        }
        self._fit_bins(
            bins, dict(self._acc_bin_weight_totals), self._acc_n_fallback,
        )
        del self._acc_bins, self._acc_bin_weight_totals, self._acc_n_fallback

    def _fit_bins(
        self,
        bins: "dict[int, list[tuple[float, float, float]]]",
        bin_weight_totals: "dict[int, float]",
        n_fallback: int,
    ) -> None:
        """Run the per-bin folded L-BFGS-B from accumulated
        ``(p_major, p_minor, weight)`` contributions.

        :param bins: ``{j: [(p_major, p_minor, weight), …]}`` per fittable bin.
        :param bin_weight_totals: ``{j: total weight}`` over all bins.
        :param n_fallback: Number of inputs that fell back to Kingman.
        """
        n_sub = self.subsample_size
        # Fitted bins are j ∈ [1, (n_sub+1)//2), the lower mirrors accumulate
        # folds onto. The upper half is π_{n_sub-j} = 1 - π_j. Per-bin seeds
        # derived from self.seed give identical results in sequential and
        # parallel modes.
        fittable = list(bins)
        bin_seeds = {j: int(self.seed) * 10_000 + int(j) for j in fittable}

        n_workers = min(
            len(fittable), Settings._resolve_n_workers(os.cpu_count() or 1),
        )
        parallel = (
            n_workers > 1
            and Settings._use_parallel(self.parallelize)
            and len(fittable) > 1
        )
        if parallel and not Settings._fork_pool_ok(
                self._log, "spectrum bins", len(fittable)):
            parallel = False
        if parallel:
            with ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=Settings._cap_worker_threads,
                    initargs=(n_workers,)) as exec_:
                results = list(exec_.map(
                    _fit_pi_bin,
                    [bins[j] for j in fittable],
                    [self.n_runs] * len(fittable),
                    [bin_seeds[j] for j in fittable],
                ))
            for j, fitted_pi in zip(fittable, results):
                self._pi[j] = fitted_pi
                self._pi[n_sub - j] = 1.0 - fitted_pi
        else:
            for j in fittable:
                self._pi[j] = _fit_pi_bin(bins[j], self.n_runs, bin_seeds[j])
                self._pi[n_sub - j] = 1.0 - self._pi[j]

        self._n_sites_per_bin = dict(bin_weight_totals)
        self._n_sites_fallback_kingman = n_fallback
        self.fitted = True
        self._warn_on_unstable_pi()

    def _warn_on_unstable_pi(self) -> None:
        """Log a warning if per-bin (weighted) sample sizes are low or fits
        hit the parameter bounds. Both signal noisy MLE estimates.

        With few sites per bin (or short divergences making p_major ≈
        p_minor), the per-bin L-BFGS-B has little to constrain π_j and
        often pegs at 0 or 1. The overall posteriors may still be
        robust if the Felsenstein likelihood dominates (typical with
        ≥2 informative outgroups), but the fitted π_j themselves are
        unreliable and should not be reported as-is.
        """
        n_sub = self.subsample_size
        fittable_bins = list(range(1, (n_sub + 1) // 2))
        if not fittable_bins:
            return
        # Each fittable bin j is fit from its own mass pooled with its fold
        # mirror n_sub - j, so the effective sample size is the sum of both.
        low_n_bins = [
            j for j in fittable_bins
            if (self._n_sites_per_bin.get(j, 0.0)
                + self._n_sites_per_bin.get(n_sub - j, 0.0)) < self.min_bin_n_sites
        ]
        pegged_bins = [
            j for j in fittable_bins
            if self._pi[j] <= 1e-8 or self._pi[j] >= 1.0 - 1e-8
        ]
        if not (low_n_bins or pegged_bins):
            return
        msg_parts = []
        if low_n_bins:
            msg_parts.append(
                f"{len(low_n_bins)}/{len(fittable_bins)} bins below "
                f"min_bin_n_sites={self.min_bin_n_sites}: "
                f"{sorted(low_n_bins)}"
            )
        if pegged_bins:
            msg_parts.append(
                f"{len(pegged_bins)}/{len(fittable_bins)} bins pegged "
                f"at parameter bound: {sorted(pegged_bins)}"
            )
        self._log.warning(
            "AdaptiveIngroupWeight fitted π_i are noisy on this "
            "dataset: " + "; ".join(msg_parts) + ". The per-bin MLE is "
            "unreliable; consider KingmanIngroupWeight, or supply "
            "more polymorphic sites / fewer ingroup haplotypes."
        )

    def _log_prior_one(self, site: Site) -> np.ndarray:
        """Per-site log vector over A/C/G/T of ``P(ingroup alleles | I = s)``.

        Biallelic with ``n_called >= subsample_size``: hypergeometric
        projection over subsampled bins, mixed by fitted ``π_j``.
        Biallelic with ``n_called < subsample_size``: count-proportional
        Kingman fallback. Monomorphic: delta. Multi-allelic:
        count-proportional. All-missing: uniform.
        """
        counts = site.count_alleles(self.ingroup_samples)
        n_obs = sum(counts.values())
        if n_obs == 0:
            return np.full(4, -np.log(4))
        n_segregating = len(counts)
        if n_segregating == 1:
            out = np.full(4, -np.inf)
            (allele,) = counts
            out[STATE_INDEX[allele]] = 0.0
            return out
        if n_segregating == 2 and n_obs >= self.subsample_size:
            (major, _), (minor, n_minor) = counts.most_common(2)
            return self._log_prior_biallelic_subsampled(major, minor, n_minor, n_obs)
        # Biallelic with partial ingroup observation (n_obs <
        # subsample_size), or multi-allelic: count-proportional
        # Kingman fallback (subsample-invariant, well-defined for any
        # n_obs ≥ 1).
        return KingmanIngroupWeight._count_proportional_log_prior(counts, n_obs)

    def _log_prior_biallelic_subsampled(
        self, major: str, minor: str, n_minor: int, n_obs: int,
    ) -> np.ndarray:
        """Hypergeometric mixture of subsampled bin priors for a biallelic site.

        For each reachable subsampled bin ``j`` (mass ``m_j > 0``):

        - ``P(major was ancestral | bin j)`` is ``1`` at ``j=0``
          (all-major subsample), ``0`` at ``j=n_sub`` (all-minor),
          and ``π_j`` otherwise.
        - ``P(minor was ancestral | bin j) = 1 - the above``.

        The site log weight for major is
        ``logsumexp_j[log(m_j) + log P(major | j)]`` and likewise for
        minor. Other alleles get ``-inf``.
        """
        n_sub = self.subsample_size
        j_values = np.arange(n_sub + 1)
        m = KingmanIngroupWeight._hypergeom_pmf(int(n_obs), int(n_minor), int(n_sub))
        # Per-j conditional ancestral probabilities for major and minor.
        p_major_given_j = np.array([self._pi[j] for j in j_values], dtype=float)
        p_minor_given_j = 1.0 - p_major_given_j

        out = np.full(4, -np.inf)
        with np.errstate(divide="ignore"):
            log_m = np.log(np.where(m > 0, m, 1.0))  # mask zeros; -inf below
            log_pmaj = np.log(np.where(p_major_given_j > 0, p_major_given_j, 1.0))
            log_pmin = np.log(np.where(p_minor_given_j > 0, p_minor_given_j, 1.0))
            # Zero out impossible (mass) and -inf out impossible (conditional).
            log_m = np.where(m > 0, log_m, -np.inf)
            log_pmaj = np.where(p_major_given_j > 0, log_pmaj, -np.inf)
            log_pmin = np.where(p_minor_given_j > 0, log_pmin, -np.inf)
            out[STATE_INDEX[major]] = float(logsumexp(log_m + log_pmaj))
            out[STATE_INDEX[minor]] = float(logsumexp(log_m + log_pmin))
        return out


