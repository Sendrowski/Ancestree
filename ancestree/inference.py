"""The inference classes, differing only in the tree the likelihood is evaluated on.

:class:`~ancestree.inference.ARGBasedInference` reads a local tree per site
from a supplied :class:`tskit.TreeSequence` and scores the root state on it.

:class:`~ancestree.inference.FixedTreeInference` assumes one
:class:`~ancestree.trees.OutgroupLadderTree` for every site, fits its branch
rates by maximum likelihood, and scores the ingroup MRCA state.

:class:`~ancestree.local_tree_inference.LocalTreeInference` infers a dated
local tree per window from genotypes alone, then scores it with the ARG-mode
kernel.

All three implement :meth:`Inference.infer() <ancestree.inference.Inference.infer>`, which yields
``(Site, Posterior)`` pairs.
"""
import datetime
import logging
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from tqdm import tqdm  # not tqdm.auto: avoids ipywidgets dep and per-line output frames in notebooks



from ancestree import DEFAULT_MU, STATE_INDEX, STATES
from ancestree._repr import ReprMixin
from ancestree.sites import (
    _by_individual, _individual_of, _named, _path_format, _refuse_overlap,
    _resolve_panel, _unlabelled,
)
from ancestree.readers import Provenance
from ancestree.focal import FocalNode
from ancestree.likelihood import Likelihood, _normalise
from ancestree.models import JC69, SubstitutionModel
from ancestree.posterior import Grade, Posterior
from ancestree.settings import Settings
from ancestree.sources import _site_from_tskit_variant
from ancestree.sites import (
    BaseComposition,
    PolymorphicSiteFilter,
    Site,
    SiteSource,
)
from ancestree.trees import (
    OutgroupLadderTree, Tree, TskitLocalTree,
)

if TYPE_CHECKING:
    import msprime
    import tskit
    from ancestree.focal import ResolvedFocal

    from ancestree.local_tree_inference import LocalTreeInference
    from ancestree.posterior import InferenceSummary
    from ancestree.priors import (
            IngroupWeight,
        StationaryPrior,
    )


class Inference(ReprMixin, ABC):
    """Abstract base for ancestral-allele inference orchestrators.

    Concrete subclasses (e.g. :class:`~ancestree.inference.ARGBasedInference`,
    :class:`~ancestree.inference.FixedTreeInference`) iterate variants from their respective
    data source, run the Felsenstein kernel on the appropriate tree
    structure, apply the configured prior, and emit per-site
    :class:`~ancestree.posterior.Posterior` records.

    The single required method is :meth:`infer`, which returns an
    iterator of ``(Site, Posterior)`` pairs in the backend's natural
    variant order.
    """

    #: Collector for unnormalised per-site log-likelihoods, installed by
    #: ``_infer_marginalised()`` while it combines
    #: draws. ``None`` disables the hook.
    _log_L_sink: "dict | None" = None
    #: Sites seen this pass that carry one ingroup allele, counted only when
    #: the focal node is the ingroup MRCA.
    _n_ingroup_monomorphic: int = 0
    #: Local trees where no lineage had coalesced, so the tree constrained no
    #: ancestral node and the posterior is the prior alone.
    _n_uncoalesced_segments: int = 0
    #: Sites whose log-posterior was degenerate and took a uniform fallback.
    _n_uniform_fallback: int = 0
    #: Local trees whose ingroup is not monophyletic, so its MRCA subtends
    #: outgroup tips. Only a per-tree walk can raise this.
    _n_ingroup_non_monophyletic: int = 0
    #: Segments with no ingroup MRCA, reported at their own root instead.
    _n_focal_multiroot_fallback: int = 0
    #: Sites carrying an allele that is neither a model state nor a no-call,
    #: such as an indel, ``*`` or a symbolic allele.
    _n_unrepresentable_sites: int = 0
    #: Tips marginalised for holding such an allele.
    _n_unrepresentable_tips: int = 0
    #: Set once a walk has finished, so the counters above are final.
    #: Provenance written ahead of the walk omits them.
    _focal_counts_complete: bool = False
    #: The last completed walk's focal diagnostics, kept for the provenance
    #: record because the live counters are reset when the walk ends.
    _focal_totals: dict = {}

    #: The substitution model every site is scored under.
    model: "SubstitutionModel"
    #: ``None`` where the mode applies the model's stationary vector directly.
    prior: "StationaryPrior | None" = None
    #: Per-site rate scaling branch lengths measured in generations. Absent
    #: from the modes whose trees are in substitutions per site.
    mu: "float | msprime.RateMap | None" = None
    #: The local VCF Zarr store this inference was constructed from, where it
    #: was one. The default template for :meth:`Inference.to_zarr`, and through
    #: its records for :meth:`Inference.to_vcf`.
    _input_store_path: "str | None" = None
    #: The VCF this inference was constructed from, where it was one. The
    #: default template for :meth:`Inference.to_vcf`.
    _input_vcf_path: "str | None" = None
    #: Set where the caller's correctness depends on scoring in-process,
    #: so the global parallelism switch cannot promote a worker pool.
    _force_serial: bool = False
    _ingroup_samples: "tuple[str, ...]" = ()
    _outgroup_samples: "tuple[str, ...]" = ()

    #: Reporting node. The rule-based baselines read at the panel root and
    #: every model-based mode replaces this in its constructor.
    focal: "FocalNode" = FocalNode("panel_root")
    #: Chunks the ARG fork pool builds per worker. The parent receives a
    #: chunk whole, so this bounds how much of the ARG it holds at once
    #: against the per-chunk overhead of re-seeking the tree sequence.
    FORK_CHUNKS_PER_WORKER: int = 8

    #: Set by a parent inference that does its own user-facing logging,
    #: so an inner one it drives stays silent.
    _quiet: bool = False

    #: Contig a VCF / VCZ read was restricted to, or ``None`` for every contig.
    _chrom_filter: "str | None" = None
    #: Sample names a VCF / VCZ read was restricted to, or ``None`` for the
    #: whole panel.
    _sample_filter: "Sequence[str] | None" = None
    #: Whether the empty-source warning has already been emitted.
    _warned_no_sites: bool = False

    #: Inference-mode tag recorded in :meth:`provenance`. The model-based
    #: modes override it (``"arg"`` / ``"fixed-tree"`` / ``"local-tree"``);
    #: the rule-based baselines keep ``"unknown"``.
    _MODE: str = "unknown"

    @classmethod
    def from_arg(
        cls,
        source: "tskit.TreeSequence | str | os.PathLike",
        model: "SubstitutionModel | None" = None,
        *,
        mu: "float | msprime.RateMap | None" = None,
        base_composition: "BaseComposition | None" = None,
        prior: "StationaryPrior | None" = None,
        chrom: str = "1",
        sample_map: Mapping[str, int] | None = None,
        progress: bool = True,
        n_workers: int = 1,
        outgroup_samples: Sequence[str] | None = None,
        ingroup_samples: Sequence[str] | None = None,
        focal: "FocalNode | str | None" = None,
        baseline_check: bool = False,
        mu_matches_time_units: bool = False,
    ) -> "ARGBasedInference":
        """Build an :class:`~ancestree.inference.ARGBasedInference` from a :class:`tskit.TreeSequence`.

        A discoverable entry point alongside :meth:`from_fixed_tree` and
        :meth:`from_local_tree`. See the constructor for the parameters.

        :return: A configured :class:`~ancestree.inference.ARGBasedInference` instance.
        """
        return ARGBasedInference(
            source,
            model if model is not None else JC69(),
            mu=mu,
            base_composition=base_composition,
            prior=prior,
            chrom=chrom,
            sample_map=sample_map,
            progress=progress,
            n_workers=n_workers,
            outgroup_samples=outgroup_samples,
            ingroup_samples=ingroup_samples,
            focal=focal,
            baseline_check=baseline_check,
            mu_matches_time_units=mu_matches_time_units,
        )

    @classmethod
    def from_fixed_tree(
        cls,
        source: "str | os.PathLike | SiteSource | tskit.TreeSequence | Sequence[Site]",
        model: "SubstitutionModel | None" = None,
        base_composition: "BaseComposition | None" = None,
        *,
        tree: "OutgroupLadderTree | None" = None,
        ingroup_samples: Sequence[str] | None = None,
        outgroup_samples: Sequence[str] | None = None,
        n_target_sites: int | None = None,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        ploidy: int | None = None,
        initial_rates: "np.ndarray | None" = None,
        bounds: tuple[float, float] = (1e-9, 10.0),
        progress: bool = True,
        outgroup_similarity_threshold: float = 0.01,
        fixed_params: "Mapping[str, float] | None" = None,
        n_starts: int = 10,
        parallelize: bool = False,
        n_workers: int | None = None,
        seed: int = 42,
        ingroup_weight: "IngroupWeight | None" = None,
        prior: "StationaryPrior | None" = None,
        focal: "FocalNode | str | None" = None,
        fit_required: bool = True,
        baseline_check: bool = True,
        subsample_size: int | None = None,
        stream: bool | None = None,
    ) -> "FixedTreeInference":
        """Build a :class:`~ancestree.inference.FixedTreeInference` (fixed-tree / outgroup-ladder mode).

        A discoverable entry point alongside :meth:`from_arg`. Builds the
        outgroup-ladder topology from the supplied names. See the constructor
        for the parameters.

        :return: A configured :class:`~ancestree.inference.FixedTreeInference` instance.
        """
        return FixedTreeInference(
            source,
            model if model is not None else JC69(),
            base_composition,
            tree=tree,
            ingroup_samples=ingroup_samples,
            outgroup_samples=outgroup_samples,
            n_target_sites=n_target_sites,
            sample_filter=sample_filter,
            chrom_filter=chrom_filter,
            ploidy=ploidy,
            initial_rates=initial_rates,
            bounds=bounds,
            progress=progress,
            outgroup_similarity_threshold=outgroup_similarity_threshold,
            focal=focal,
            fixed_params=fixed_params,
            n_starts=n_starts,
            parallelize=parallelize,
            n_workers=n_workers,
            seed=seed,
            ingroup_weight=ingroup_weight,
            prior=prior,
            fit_required=fit_required,
            baseline_check=baseline_check,
            subsample_size=subsample_size,
            stream=stream,
        )

    @classmethod
    def from_local_tree(
        cls,
        source,
        model: "SubstitutionModel | None" = None,
        *,
        mu: float | None = None,
        rec_rate: float | None = None,
        sample_names: Sequence[str] | None = None,
        sequence_length: float | None = None,
        window: "int | str" = "8snp",
        block_size: "int | str | None" = None,
        n_time_bins: int = 32,
        prior: "StationaryPrior | None" = None,
        base_composition: "BaseComposition | None" = None,
        progress: bool = True,
        n_workers: int = 1,
        chunk_size: "int | str | None" = "10mb",
        halo: "int | str" = "auto",
        recombination_map=None,
        accessibility: "Sequence[tuple[float, float]] | None" = None,
        mutation_map=None,
        outgroup_samples: Sequence[str] | None = None,
        ingroup_samples: Sequence[str] | None = None,
        focal: "FocalNode | str | None" = None,
        baseline_check: bool = False,
        n_ensemble: "int | None" = 64,
        ensemble_seed: int = 0,
        member_chunk: int = 8,
        mu_matches_time_units: bool = False,
        time_grid: "np.ndarray | None" = None,
    ) -> "LocalTreeInference":
        """Build a :class:`~ancestree.local_tree_inference.LocalTreeInference`.

        A discoverable entry point alongside :meth:`from_arg` and
        :meth:`from_fixed_tree` for the VCF-only inferred-local-tree mode. See
        the constructor for the parameters.

        :return: A configured
            :class:`~ancestree.local_tree_inference.LocalTreeInference` instance.
        """
        from ancestree.local_tree_inference import LocalTreeInference
        return LocalTreeInference(
            source,
            model if model is not None else JC69(),
            mu=mu,
            rec_rate=rec_rate,
            sample_names=sample_names,
            sequence_length=sequence_length,
            window=window,
            block_size=block_size,
            n_time_bins=n_time_bins,
            prior=prior,
            base_composition=base_composition,
            progress=progress,
            n_workers=n_workers,
            chunk_size=chunk_size,
            halo=halo,
            recombination_map=recombination_map,
            accessibility=accessibility,
            mutation_map=mutation_map,
            outgroup_samples=outgroup_samples,
            ingroup_samples=ingroup_samples,
            focal=focal,
            baseline_check=baseline_check,
            n_ensemble=n_ensemble,
            ensemble_seed=ensemble_seed,
            member_chunk=member_chunk,
            mu_matches_time_units=mu_matches_time_units,
            time_grid=time_grid,
        )

    def _log_uniform_fallback_summary(self) -> None:
        """Emit one warning summarising uniform-posterior fallbacks, if any.

        ``_normalise_log_post`` accumulates a per-instance count of sites
        whose log-posterior was degenerate (all ``-inf``). This reports the
        total once, at the end of :meth:`infer`.
        """
        n = self._n_uniform_fallback
        if n:
            self._log.warning(
                "%d site(s) had a degenerate log-posterior (all -inf) and "
                "were assigned a uniform fallback. Check whether the prior or "
                "model assigns zero mass to all observed alleles there.", n,
            )
            self._n_uniform_fallback = 0
        n_mono = self._n_ingroup_monomorphic
        if n_mono:
            self._log.warning(
                "%d site(s) are monomorphic within the ingroup and were "
                "reported at the ingroup MRCA, where the ingroup's own allele "
                "is the answer.", n_mono,
            )
            self._n_ingroup_monomorphic = 0
        n_uncoal = self._n_uncoalesced_segments
        if n_uncoal:
            self._log.warning(
                "%d local tree(s) carried no coalescence at all, so they "
                "constrained no ancestral node and their sites were reported "
                "at the prior alone.", n_uncoal,
            )
            self._n_uncoalesced_segments = 0
        n_unrep = self._n_unrepresentable_sites
        if n_unrep:
            self._log.warning(
                "%d site(s) carry an allele outside the A/C/G/T alphabet "
                "(an indel, '*' or a symbolic allele) and %d tip(s) were "
                "marginalised for holding one. A site with no A/C/G/T allele "
                "at all is written unannotated.",
                n_unrep, self._n_unrepresentable_tips,
            )
            self._n_unrepresentable_sites = 0
            self._n_unrepresentable_tips = 0
        summary = self._focal_summary()
        if summary:
            self._log.info("Focal node: %s", summary)
        # Counters are per walk. The totals go to the provenance record.
        self._publish_focal_totals()
        self._n_ingroup_non_monophyletic = 0
        self._n_focal_multiroot_fallback = 0

    def _publish_focal_totals(self) -> None:
        """Record the finished walk's focal diagnostics for the provenance.

        Every path that completes a walk calls this before its counters are reset.
        """
        self._focal_totals = {
            "n_trees_ingroup_non_monophyletic":
                int(self._n_ingroup_non_monophyletic),
            "n_segments_without_ingroup_mrca":
                int(self._n_focal_multiroot_fallback),
        }
        self._focal_counts_complete = True

    def _focal_summary(self) -> str | None:
        """One line on how the focal node behaved over the walk.

        :return: The message, or ``None`` when nothing is worth saying.
        """
        if self.focal.is_root:
            return None
        parts = [f"reported at the {self.focal.describe()}"]
        if self._n_ingroup_non_monophyletic:
            parts.append(
                f"{self._n_ingroup_non_monophyletic} tree(s) where the ingroup "
                f"is not monophyletic, so its MRCA subtends outgroup tips"
            )
        if self._n_focal_multiroot_fallback:
            parts.append(
                f"{self._n_focal_multiroot_fallback} multi-root segment(s) with "
                f"no ingroup MRCA, reported at their own root instead"
            )
        return "; ".join(parts)

    @property
    def _focal_anchored_at_ingroup_mrca(self) -> bool:
        """Whether the focal node is anchored at the ingroup MRCA.

        The anchor alone, whatever placement is measured from it. Callers
        positioning a readout point relative to the anchor want this;
        callers asking where the run actually reports want
        :attr:`Inference._focal_is_ingroup_mrca`.
        """
        return self.focal.anchor == "ingroup_mrca"

    @property
    def _focal_is_ingroup_mrca(self) -> bool:
        """Whether this pass reports at the ingroup's own ancestor.

        A placement above the anchor moves the readout point to a deeper node,
        where ingroup-fixed sites are called against the outgroups. A zero
        placement names the anchor itself, so it still reports there.
        """
        return (self._focal_anchored_at_ingroup_mrca
                and not self.focal._placement)

    @abstractmethod
    def infer(self) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(Site, Posterior)`` for every variant.

        :return: Iterator of per-site posteriors over candidate
            ancestral alleles. Each :class:`~ancestree.posterior.Posterior` is a
            length-:attr:`model.n_states <ancestree.models.SubstitutionModel.n_states>`
            distribution summing to 1.
        """


    def _ensure_fitted(self) -> None:
        """Hook: run any required ML fit before a full inference pass.

        No-op for modes that need no fitting (ARG / local-tree / majority).
        :class:`~ancestree.inference.FixedTreeInference` overrides this to fit branch rates on
        demand, so :meth:`summary` / :meth:`grade` work on an unfitted
        instance.
        """

    def summary(self) -> "InferenceSummary":
        """Run a full inference pass and reduce it to aggregate diagnostics.

        Consumes :meth:`Inference.infer() <ancestree.inference.Inference.infer>`
        (so it triggers any required fitting and, like that method, is a
        one-shot pass over the data) and collapses the per-site
        posteriors into a small
        :class:`~ancestree.posterior.InferenceSummary`: the site count, the
        MAP-allele histogram, the spread of MAP confidences, and the mean per-site
        posterior entropy. Descriptive only: it sets no thresholds and does not filter.

        :return: An :class:`~ancestree.posterior.InferenceSummary`.
        """
        from ancestree.posterior import InferenceSummary

        self._ensure_fitted()
        n = 0
        map_counts: dict[str, int] = {}
        max_probs: list[float] = []
        entropy_sum = 0.0
        for _site, post in self.infer():
            n += 1
            allele = post.map_allele
            map_counts[allele] = map_counts.get(allele, 0) + 1
            max_probs.append(post.max_prob)
            p = post.values
            nz = p[p > 0]  # 0·log0 ≡ 0
            entropy_sum += float(-(nz * np.log2(nz)).sum())
        if n == 0:
            return InferenceSummary(0, {}, float("nan"), {}, float("nan"))
        mp = np.asarray(max_probs)
        quantiles = {q: float(np.quantile(mp, q))
                     for q in (0.05, 0.25, 0.5, 0.75, 0.95)}
        ordered = dict(sorted(map_counts.items(), key=lambda kv: (-kv[1], kv[0])))
        return InferenceSummary(
            n_sites=n,
            map_alleles=ordered,
            mean_max_prob=float(mp.mean()),
            max_prob_quantiles=quantiles,
            mean_entropy_bits=entropy_sum / n,
        )

    def provenance(self) -> dict[str, object]:
        """Structured provenance record describing this inference run.

        Written by :meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>`,
        :meth:`Inference.to_zarr() <ancestree.inference.Inference.to_zarr>` and
        :meth:`Inference.to_arg() <ancestree.inference.Inference.to_arg>`. The
        JSON-serialisable record carries ``software``, ``version``, ``mode``
        (``"arg"`` / ``"fixed-tree"`` / ``"local-tree"``), ``parameters`` (the
        run parameters, any ML-fitted values, and the focal diagnostics once a
        walk has finished) and a UTC ``timestamp``.

        :return: A :class:`~ancestree.readers.Provenance` record.
        """
        from ancestree import __version__

        parameters = self._provenance_parameters()
        panel = self._panel_samples()
        if panel:
            parameters = {**parameters, "panel_samples": list(panel)}
        # The same merge the writers apply once the walk has finished, so the
        # record returned here and the record written agree.
        if self._focal_counts_complete and not self.focal.is_root:
            parameters = {**parameters, **self._focal_totals}
        return Provenance(
            software="ancestree",
            version=__version__,
            mode=self._MODE,
            parameters=parameters,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def _provenance_parameters(self) -> dict[str, object]:
        """Mode-specific run parameters for :meth:`provenance`.

        Overridden by each concrete inference to expose the constructor
        arguments (and ML-fitted values) that determined the calls. The base
        implementation returns an empty mapping.

        :return: A JSON-serialisable ``dict`` of run parameters.
        """
        return {}

    def _model_provenance(self) -> dict[str, object]:
        """The model's own parameters and the composition the kernel read.

        Naming the model class alone does not identify the run: two runs
        differing only in a free parameter (HKY or K2 ``kappa``, GTR's
        exchangeabilities) or in the base composition supplying pi make
        different calls at the same sites.

        :return: JSON-serialisable model parameters and, where one is
            resolvable, the stationary distribution as a four-element list.
        """
        import numpy as np

        entry: dict[str, object] = {}
        for key, value in (getattr(self.model, "_repr_params", None) or {}).items():
            if isinstance(value, (int, float)):
                entry[f"model_{key}"] = float(value)
            elif isinstance(value, np.ndarray):
                entry[f"model_{key}"] = [float(v) for v in value.ravel()]
            else:
                entry[f"model_{key}"] = value
        try:
            pi = self._pi()
        except Exception:
            pi = None
        if pi is not None:
            entry["base_composition_pi"] = [float(v) for v in np.asarray(pi).ravel()]
        return entry

    @staticmethod
    def _require_stationary_prior(prior) -> None:
        """Refuse an ingroup weight offered as the root prior.

        An :class:`~ancestree.priors.IngroupWeight` conditions on a single
        shared ingroup-MRCA polytomy. A mode that reads a local tree per site
        has no such node, so the weight has nothing to condition on.

        :param prior: The ``prior`` argument, which may be ``None``.
        :raises TypeError: If it is anything other than a
            :class:`~ancestree.priors.StationaryPrior` or ``None``.
        """
        from ancestree.priors import StationaryPrior

        if prior is not None and not isinstance(prior, StationaryPrior):
            raise TypeError(
                "prior must be a StationaryPrior or None; got "
                f"{type(prior).__name__}. An IngroupWeight conditions on a "
                "single shared ingroup-MRCA polytomy, which this mode does "
                "not have: every site carries its own local tree with the "
                "ingroup samples as real tips."
            )

    def _count_ingroup_monomorphic(self, sites, counts=None) -> None:
        """Tally sites carrying at most one allele across the ingroup.

        A site monomorphic within the ingroup says nothing about the ingroup
        MRCA on its own, so a run reporting there answers from the outgroups.
        An unfolded spectrum built from those calls has no derived-fixed
        entry, which is what the summary warns about.

        :param sites: The sites just scored.
        :param counts: Ingroup allele counts tallied over
            :meth:`Inference._baseline_ingroup_samples`, one per site in site
            order, or ``None`` to tally them here.
        """
        if not self._focal_is_ingroup_mrca:
            return
        ingroup = self._baseline_ingroup_samples()
        if not ingroup:
            return
        if counts is None:
            counts = (site.count_alleles(ingroup) for site in sites)
        for tally in counts:
            if sum(1 for c in tally.values() if c) <= 1:
                self._n_ingroup_monomorphic += 1

    def _count_unrepresentable(self, sites) -> None:
        """Tally sites and tips whose alleles lie outside the alphabet.

        A site counts once when its allele list carries such an allele or one
        of its tips is observed at one. The tips are counted individually,
        each of them marginalised by the kernel.

        :param sites: The sites just scored.
        """
        for site in sites:
            n_tips = site.n_unrepresentable_tips()
            self._n_unrepresentable_tips += n_tips
            if n_tips or site.has_unrepresentable_allele:
                self._n_unrepresentable_sites += 1

    @staticmethod
    def _template_individual_names(
        ts, sample_map, nodes: "Collection[int] | None" = None,
    ) -> "list[str] | None":
        """Names of the VCF sample columns of a template written from ``ts``,
        in the source reader's naming convention.

        Each individual is one column, named from ``sample_map`` by
        individual (``i0_h0`` and ``i0_h1`` give ``i0``), or ``tsk_<j>``
        where ``sample_map`` names none of its haplotypes.

        :param ts: The tree sequence the columns are numbered in.
        :param sample_map: ``{name: node}`` for the named haplotypes.
        :param nodes: Nodes whose columns are named, in ``ts``'s numbering.
            ``None`` names every column.
        :return: One name per column, or ``None`` to keep tskit's defaults
            throughout where two columns would share a name.
        """
        by_node = {int(n): str(s) for s, n in sample_map.items()}
        model = ts.map_to_vcf_model()
        names = []
        for default, row in zip(model.individuals_name,
                                model.individuals_nodes):
            row_nodes = [int(n) for n in row if n >= 0]
            if nodes is not None and not any(n in nodes for n in row_nodes):
                continue
            known = {by_node[n] for n in row_nodes if n in by_node}
            if len(row_nodes) > 1:
                known = {_individual_of(s) for s in known}
            names.append(known.pop() if len(known) == 1 else str(default))
        return names if len(set(names)) == len(names) else None

    def _set_input_paths(self, source) -> None:
        """Take the file a source was read from as the default template.

        A local directory is a VCF Zarr store, the template of
        :meth:`Inference.to_zarr() <ancestree.inference.Inference.to_zarr>`.
        Any other path naming neither a store nor a tree sequence is a VCF,
        the template of
        :meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>`.

        :param source: A path, or a source carrying the path it read.
        """
        path = (str(source) if isinstance(source, (str, os.PathLike))
                else getattr(source, "_path", None))
        fmt = _path_format(path) if path is not None else None
        store = path is not None and os.path.isdir(path)
        self._input_store_path = path if store else None
        self._input_vcf_path = (
            path if path is not None and not store
            and fmt not in ("vcz", "trees") else None)

    def _panel_samples(self) -> tuple[str, ...]:
        """The resolved ingroup and outgroup sample ids."""
        return (*self._baseline_ingroup_samples(),
                *self._baseline_outgroup_samples())

    def _used_samples(self) -> "frozenset[str]":
        """The sample columns a restricted output keeps, the panel samples
        and their individuals."""
        return _by_individual(self._panel_samples())

    def _note_unnamed_ingroup(self) -> None:
        """Report that an unnamed ingroup is the whole panel.

        The focal anchor then resolves to the panel root and not to a
        clade's own ancestor, which is a different node and a different
        answer.
        """
        if self.focal.is_root:
            return
        if self._ingroup_samples or self._outgroup_samples:
            return
        self._log.info(
            "No ingroup or outgroup samples were named with focal=%r, so the "
            "ingroup is the whole panel and its MRCA is the panel root. Pass "
            "ingroup_samples= or outgroup_samples= to report at the "
            "ingroup's own ancestor.", self.focal.anchor,
        )

    def _pi(self):
        """Equilibrium frequencies from the run's base composition, if any.

        :return: The composition's ``pi``, or ``None`` when the run supplied
            no composition and the model uses its own default.
        """
        bc = getattr(self, "base_composition", None)
        return None if bc is None else bc.pi

    @staticmethod
    def _model_name(model: object) -> str:
        """Short class name of a substitution model, for provenance."""
        return type(model).__name__

    @staticmethod
    def _summarise_baseline_agreement(
        real: "Iterable[tuple[Site, Posterior]]",
        baseline: "Iterable[tuple[Site, Posterior]]",
        *,
        ingroup_samples: Sequence[str] | None,
    ) -> str | None:
        """Build the one-line baseline-agreement INFO log message.

        Walks two parallel ``(Site, Posterior)`` streams (the real
        inference's posteriors and the baseline's) and computes:

        - the overall MAP-agreement fraction,
        - the per-folded-SFS-bin agreement breakdown (when
          ``ingroup_samples`` is available),
        - the two bins with the lowest agreement (for the trailing
          "largest disagreement at SFS bin X (0.78)" hint).

        :param real: Iterator of ``(Site, Posterior)`` from the real
            inference.
        :param baseline: Iterator of ``(Site, Posterior)`` from the
            baseline.
        :param ingroup_samples: Ingroup ids for folded-SFS bin
            stratification. When ``None`` or empty, the trailing
            per-bin breakdown is omitted.
        :return: A formatted log line, or ``None`` if no comparable
            sites were observed.
        """
        n_total = n_agree = 0
        per_bin_total: dict[int, int] = {}
        per_bin_agree: dict[int, int] = {}
        ingroup = list(ingroup_samples or [])
        for (site_r, post_r), (site_b, post_b) in zip(real, baseline):
            if site_r is not site_b and (
                site_r.chrom != site_b.chrom or int(site_r.pos) != int(site_b.pos)
            ):
                return None
            n_total += 1
            agree = int(post_r.map_allele == post_b.map_allele)
            n_agree += agree
            if ingroup:
                counts = site_r.count_alleles(ingroup)
                n_obs = sum(counts.values())
                if n_obs >= 2:
                    minor = min(counts.values()) if len(counts) > 1 else 0
                    bin_j = min(minor, n_obs - minor)
                    per_bin_total[bin_j] = per_bin_total.get(bin_j, 0) + 1
                    per_bin_agree[bin_j] = per_bin_agree.get(bin_j, 0) + agree
        if n_total == 0:
            return None
        overall = n_agree / n_total
        msg = (
            f"baseline (MajorityOutgroupInference) MAP agreement: "
            f"{overall:.3f} on {n_total} polymorphic sites"
        )
        if per_bin_total:
            rates = sorted(
                (
                    (bin_j, per_bin_agree[bin_j] / per_bin_total[bin_j])
                    for bin_j in per_bin_total
                ),
                key=lambda pair: pair[1],
            )
            tail = "; ".join(
                f"SFS bin {bin_j} ({rate:.3f})" for bin_j, rate in rates[:2]
            )
            msg += f"; largest disagreement at {tail}"
        return msg

    #: When set, :meth:`infer` runs a
    #: :class:`~ancestree.inference.MajorityOutgroupInference` comparison
    #: over the same sites and INFO-logs the MAP agreement as a consistency check.
    baseline_check: bool = False

    #: Cap on sites buffered for the baseline comparison so the check stays
    #: bounded in memory under streaming inference over whole genomes. Beyond
    #: it the agreement is reported over the leading sample.
    _BASELINE_CHECK_MAX_SITES: int = 100_000

    def _baseline_outgroup_samples(self) -> tuple[str, ...]:
        """Outgroup ids for the majority-allele comparator, or ``()`` if the
        mode has no designated outgroups (then the check is skipped)."""
        return ()

    def _baseline_ingroup_samples(self) -> tuple[str, ...]:
        """Ingroup ids used to stratify the comparison by folded-SFS bin.
        ``()`` omits the per-bin breakdown."""
        return ()

    def _check_time_units(self, ts,
                          mu_matches_time_units: bool = False) -> None:
        """Refuse an uncalibrated ARG and flag one that never declared units.

        ``mu`` is a rate per unit of the tree sequence's own time. Times
        declared ``uncalibrated`` admit no such rate and are refused. Times
        declared ``unknown`` are assumed to be generations, with a warning:
        coalescent-unit times need ``mu`` per that unit, ``4 N_e mu`` for
        units of ``4 N_e`` generations. ``mu_matches_time_units`` asserts that
        ``mu`` was derived against this tree sequence's own time axis, as a
        rate fitted to the tree itself is.

        :param ts: The tree sequence about to be scored.
        :param mu_matches_time_units: Whether ``mu`` is expressed per
            unit of this tree sequence's node times.
        :raises ValueError: If the tree sequence declares uncalibrated times
            and ``mu`` was not declared to match them.
        """
        import tskit

        if mu_matches_time_units:
            return
        units = getattr(ts, "time_units", tskit.TIME_UNITS_UNKNOWN)
        if units == tskit.TIME_UNITS_UNCALIBRATED:
            raise ValueError(
                "the tree sequence declares uncalibrated node times, so no "
                "substitution rate scales them to expected substitutions. "
                "Date it first, set time_units to the unit its times are in "
                "and pass mu per that unit, or pass "
                "mu_matches_time_units=True if mu was fitted against this "
                "tree sequence's own time axis."
            )
        if units == tskit.TIME_UNITS_UNKNOWN:
            self._log.warning(
                "The tree sequence declares no time units, so its node times "
                "are assumed to be in generations and mu is applied per "
                "generation. Times in coalescent units need mu given per that "
                "unit instead (4*Ne*mu for times in units of 4*Ne "
                "generations), or the calls sit near the mu -> 0 limit."
            )

    def _refuse_empty_ingroup(
        self, n_panel: int, *, regardless: bool = False,
    ) -> None:
        """Refuse an ingroup left empty by the outgroup list.

        :param n_panel: Panel size, quoted in the message.
        :param regardless: Refuse even when the run reports at the panel root,
            for callers whose own arithmetic needs ingroup members.
        :raises ValueError: If every panel sample was named as an outgroup.
        """
        if not regardless and self.focal.is_root:
            return
        where = ("there is no ingroup to calibrate the fit on" if regardless
                 else f"focal={self.focal.anchor!r} has no node to resolve")
        raise ValueError(
            f"the ingroup is empty, so {where}; every one of the {n_panel} "
            f"panel samples was named as an outgroup"
        )

    def _resolve_panel(
        self, panel: "Iterable[str]", *, explicit: bool,
    ) -> tuple[str, ...]:
        """Resolve the panel, and the ingroup and outgroups within it.

        Sets ``_resolved_ingroup`` and ``_resolved_outgroups``.

        :param panel: The mode's own sample identifiers, in panel order.
        :param explicit: Whether the caller chose the panel. A haplotype in
            neither list then raises, and is otherwise dropped.
        :return: The panel, in panel order.
        :raises ValueError: If a named sample is not in the panel or in both
            lists, or if an explicit panel holds a haplotype in neither list.
        """
        panel, ingroup, outgroups, dropped = _resolve_panel(
            panel, self._ingroup_samples, self._outgroup_samples,
            chosen_by="the panel" if explicit else None)
        if dropped:
            self._log.info(
                "Ignoring %d sample(s) in neither ingroup_samples nor "
                "outgroup_samples", len(dropped))
        self._resolved_ingroup: tuple[str, ...] = tuple(ingroup)
        self._resolved_outgroups: tuple[str, ...] = tuple(outgroups)
        return tuple(panel)

    def _with_baseline_check(
        self, inner: "Iterator[tuple[Site, Posterior]]",
    ) -> "Iterator[tuple[Site, Posterior]]":
        """Pass ``inner`` through, then run the baseline consistency check on the
        collected posteriors once the stream is exhausted.

        Buffers at most :attr:`_BASELINE_CHECK_MAX_SITES` pairs (so streaming
        inference stays bounded in memory), and only buffers at all when the
        check will actually run: it is enabled, INFO logging is on, and
        outgroups are designated. With no outgroups it just logs that the
        comparison was skipped without retaining the stream.

        :param inner: The mode's own per-site posterior generator.
        :return: The same ``(Site, Posterior)`` stream, unchanged.
        """
        want = (
            self.baseline_check
            and not self._quiet
            and self._log.isEnabledFor(logging.INFO)
        )
        has_outgroups = want and bool(self._baseline_outgroup_samples())
        seen: list[tuple[Site, Posterior]] = []
        truncated = False
        n_yielded = 0
        for pair in inner:
            n_yielded += 1
            if has_outgroups and len(seen) < self._BASELINE_CHECK_MAX_SITES:
                seen.append(pair)
            elif has_outgroups:
                truncated = True
            yield pair
        self._warn_if_no_sites(n_yielded)
        if want:
            self._emit_baseline_check(seen, truncated=truncated)

    def _warn_if_no_sites(self, n_sites: int) -> None:
        """Warn once when a pass read no sites at all.

        :param n_sites: Sites the pass yielded.
        """
        if n_sites or self._warned_no_sites or self._quiet:
            return
        self._warned_no_sites = True
        names = None if self._sample_filter is None else list(self._sample_filter)
        self._log.warning(
            "%s read no sites, so nothing is annotated. Active source "
            "filters: chrom_filter=%s, sample_filter=%s. A contig label that "
            "does not match the data, such as 'chr1' against '1', is the "
            "usual cause.",
            type(self).__name__,
            "None (every contig)" if self._chrom_filter is None
            else repr(self._chrom_filter),
            "None (every sample)" if names is None
            else f"{len(names)} sample(s): {', '.join(names[:5])}")

    def _emit_baseline_check(
        self, real_pairs: "Sequence[tuple[Site, Posterior]]",
        *, truncated: bool = False,
    ) -> None:
        """Run :class:`~ancestree.inference.MajorityOutgroupInference` on the same sites and
        INFO-log the MAP agreement with the model output.

        :param real_pairs: The model's ``(Site, Posterior)`` results (capped
            to the buffer limit by :meth:`_with_baseline_check`).
        :param truncated: Whether ``real_pairs`` is a leading sample of a
            longer stream (noted in the log line).
        """
        outgroups = self._baseline_outgroup_samples()
        if not outgroups:
            self._log.info(
                "Baseline (MajorityOutgroupInference) skipped: no outgroups "
                "designated for the comparator rule."
            )
            return
        sites_only = [pair[0] for pair in real_pairs]
        baseline = MajorityOutgroupInference(
            sites_only, outgroups, model=self.model, for_comparison_only=True,
        )
        msg = self._summarise_baseline_agreement(
            real_pairs, baseline.infer(),
            ingroup_samples=self._baseline_ingroup_samples(),
        )
        if msg is not None:
            if truncated:
                msg += f" (leading {len(sites_only):,}-site sample)"
            self._log.info(msg)

    def _stash_log_L(self, sites, log_L) -> None:
        """Record unnormalised per-site log-likelihoods when collecting.

        ``_infer_marginalised()`` combines draws in likelihood space, so it
        needs what the per-draw normalisation discards.

        :param sites: Sites the rows of ``log_L`` correspond to.
        :param log_L: ``(len(sites), S)`` log-likelihoods, without the prior.
        """
        if self._log_L_sink is None:
            return
        for row, site in zip(np.asarray(log_L, dtype=float), sites):
            key = (site.local_tree_handle
                   if site.local_tree_handle is not None else site.pos)
            self._log_L_sink[key] = row

    def _normalise_log_post(
        self,
        log_post: np.ndarray,
        *,
        n_states: int,
    ) -> np.ndarray:
        """Log-softmax with a uniform fallback for degenerate rows.

        An all-``-inf`` row takes the uniform distribution and is counted for
        the summary warning at the end of :meth:`infer`.

        :param log_post: ``(N, S)`` un-normalised log-posterior.
        :param n_states: ``S``, alphabet size for the uniform fallback.
        :return: ``(N, S)`` normalised posterior probabilities, each row
            sums to 1.
        """
        values, n_bad = _normalise(log_post, n_states)
        if n_bad:
            self._n_uniform_fallback += n_bad
        return values

    def grade(
        self,
        truth: "Mapping[int, str] | tskit.TreeSequence",
        filter: "PolymorphicSiteFilter | None" = None,
        focal: "FocalNode | str" = "ingroup_mrca",
        *,
        sample_map: Mapping[str, int] | None = None,
    ) -> "Grade":
        """Run :meth:`infer` and grade the per-site posteriors against ``truth``.

        Like :meth:`summary`, consumes :meth:`infer` and so triggers any
        required fitting first.

        :param truth: ``{int(site_position): true_allele}``, or a
            :class:`tskit.TreeSequence` carrying the true mutations, whose
            allele at ``focal`` is then the truth
            (:meth:`Grade.truth_at_focal() <ancestree.posterior.Grade.truth_at_focal>`).
        :param filter: Optional
            :class:`~ancestree.sites.PolymorphicSiteFilter` restricting which sites
            are scored. Every site is still inferred, so this decomposes one
            run and does not change it. Pass the same filter as the
            inference's source to score exactly what was inferred.
        :param focal: The node of a tree-sequence truth at which the true
            allele is read, resolved over this run's panel, ingroup and outgroup
            samples. Defaults to the ingroup's most recent common ancestor,
            whichever node this run reports at.
        :param sample_map: ``{sample: node}`` mapping this run's sample names
            onto the nodes of a tree-sequence truth, interpreted in the truth
            tree sequence's own node space and not in that of any inference.
            ``None`` reads the individual names of the truth, falling back to
            node ids as strings.
        :return: A :class:`~ancestree.posterior.Grade`.
        :raises ValueError: If one of this run's samples matches no sample of a
            tree-sequence truth
            (:meth:`Grade.truth_at_focal() <ancestree.posterior.Grade.truth_at_focal>`).
        """
        self._ensure_fitted()
        return Grade(
            self.infer(), truth, filter, focal=focal,
            ingroup_samples=self._baseline_ingroup_samples(),
            outgroup_samples=self._baseline_outgroup_samples(),
            panel_samples=self._panel_samples(),
            sample_map=sample_map,
        )

    def _scale_for(self, tree: "Tree") -> "float | None":
        """The per-site rate scaling ``tree``'s branch lengths, or ``None``.

        Branch lengths measured in generations, as an ARG local tree and the
        local-tree mode's own genealogies are, need the per-site rate applied
        ahead of the kernel, or every transition saturates. A
        fixed-tree ladder's are in expected substitutions per site and
        are left alone.

        :param tree: The tree about to be scored.
        :return: The rate to assign to ``tree.time_scale``, or ``None`` to
            leave the branch lengths as they are.
        """
        mu = self.mu
        ts_tree = tree.tskit_tree
        if mu is None or ts_tree is None:
            return None
        if isinstance(mu, float):
            return float(mu)
        interval = ts_tree.interval
        return self._mu_for_interval(
            float(interval.left), float(interval.right))

    def _mu_for_interval(self, left: float, right: float) -> float:
        """Per-site rate over a genomic interval ``[left, right)``.

        A scalar ``mu`` is returned directly. A rate-map ``mu`` is reduced to
        its interval-weighted mean,
        ``(cumulative_mass(right) - cumulative_mass(left)) / (right - left)``.

        :param left: Interval start (bp).
        :param right: Interval end (bp).
        :return: The rate scaling branch lengths over the interval.
        """
        if isinstance(self.mu, float):
            return self.mu
        mass = (
            float(self.mu.get_cumulative_mass(right))
            - float(self.mu.get_cumulative_mass(left))
        )
        rate = mass / (right - left)
        if rate <= 0.0:
            # A zero-rate interval is floored, with one warning per run.
            if not getattr(self, "_warned_zero_rate", False):
                self._log.warning(
                    "Rate-map mu is zero over [%g, %g); flooring to %g so the "
                    "interval's sites are scored as mutationally uninformative "
                    "(prior-dominated) rather than crashing.",
                    left, right, _ZERO_RATE_FLOOR,
                )
                self._warned_zero_rate = True
            return _ZERO_RATE_FLOOR
        return rate

    def _focal_view(self, tree: "Tree") -> "Tree":
        """``tree`` viewed from the node this run reports at.

        :param tree: The tree supplied to :meth:`Inference.infer_site`.
        :return: ``tree`` itself when the run reports at the tree's own root,
            otherwise a re-rooted view reporting at the focal node.
        :raises ValueError: If a non-root focal node is configured and ``tree``
            offers no way to resolve it.
        """
        focal = self.focal
        if focal.is_root:
            return tree
        view = tree.at_focal(focal)
        if view is not None:
            return view
        ts_tree = tree.tskit_tree
        if ts_tree is not None:
            return self._tskit_focal_view(tree, ts_tree, focal)
        raise ValueError(
            f"focal={focal.anchor!r} is configured, but the tree supplied "
            f"({type(tree).__name__}) offers no way to resolve it, so the "
            f"posterior would describe the tree's own root instead. Pass a "
            f"tree that resolves the focal node, or configure "
            f"focal='panel_root' to report at the tree's own root."
        )

    def _tskit_focal_view(self, tree: "Tree", ts_tree, focal) -> "Tree":
        """``tree`` re-rooted at ``focal``, resolved on its own tskit tree.

        The ingroup is located through ``tree``, not through this run's
        panel, so a caller-supplied tree is read with its own node numbering.

        :param tree: The wrapper whose branch lengths the kernel will read.
        :param ts_tree: The :class:`tskit.Tree` backing it.
        :param focal: The configured :class:`~ancestree.focal.FocalNode`.
        :return: ``tree`` itself when the focal node is its root, else a
            :class:`~ancestree.trees.RerootedTree` at the focal node.
        :raises ValueError: If none of the ingroup samples are tips of ``tree``.
        """
        wanted = self._baseline_ingroup_samples()
        nodes = tuple(
            int(n) for n in (tree.tip_for_sample(s) for s in wanted)
            if n is not None
        )
        if not nodes:
            raise ValueError(
                f"focal={focal.anchor!r} is configured, but none of the "
                f"ingroup samples are tips of the tree supplied, so there is "
                f"no node to resolve it against."
            )
        resolved = focal.locate(ts_tree, nodes)
        if resolved is None or (int(resolved.node) == int(tree.root)
                                and resolved.tau <= 0.0):
            return tree
        from ancestree.trees import RerootedTree
        return RerootedTree(tree, int(resolved.node), resolved.tau)

    def infer_site(self, tree: "Tree", site: "Site") -> "Posterior":
        """Run inference on one site against one tree: single-call posterior.

        The single-site counterpart to :meth:`infer`'s tree-sequence walk:
        views ``tree`` from the configured focal node, runs the Felsenstein
        kernel on it, seeds the configured ingroup weight on its ingroup MRCA
        where one is set, applies the prior, normalises, and returns the
        :class:`~ancestree.posterior.Posterior` over candidate ancestral
        alleles. Intended for evaluating the kernel on a single, externally-supplied
        tree, such as one built from a Newick string or a single local tree taken from
        an ARG.

        :param tree: Any :class:`~ancestree.trees.Tree` whose tip ids
            match the keys of
            :attr:`Site.tip_alleles <ancestree.sites.Site.tip_alleles>`.
        :param site: The :class:`~ancestree.sites.Site` to infer the ancestral state of.
        :return: A length-:attr:`model.n_states <ancestree.models.SubstitutionModel.n_states>` :class:`~ancestree.posterior.Posterior`
            summing to 1.
        """
        from ancestree.priors import StationaryPrior
        bc = getattr(self, "base_composition", None)
        engine = Likelihood(self.model, base_composition=bc)
        scale = self._scale_for(tree)
        if scale is not None:
            tree.time_scale = scale
        tree = self._focal_view(tree)
        # Seed the ingroup weight as infer() does.
        seeds = None
        weight = getattr(self, "ingroup_weight", None)
        if weight is not None:
            anchor = tree.ingroup_mrca
            if anchor is None:
                raise ValueError(
                    "infer_site: an ingroup weight is configured but the tree "
                    f"supplied ({type(tree).__name__}) has no ingroup MRCA to "
                    "seed it on. Pass the ladder (or a focal view of it), or "
                    "use a mode without an ingroup weight."
                )
            with np.errstate(under="ignore"):
                seeds = {anchor: np.exp(weight.log_probs([site]))}
        log_L = engine.log_likelihoods(tree, [site], node_seeds=seeds)[0]
        if self.prior is not None:
            log_prior = self.prior.log_probs([site])[0]
        else:
            log_prior = StationaryPrior(self.model, bc).log_probs([site])[0]
        log_post = log_L + log_prior
        values = self._normalise_log_post(
            log_post[None, :], n_states=self.model.n_states,
        )[0]
        return Posterior(alleles=self.model.states, values=values)


    def _resolve_provenance(
        self,
        provenance: "Mapping[str, object] | None",
        min_confidence: float | None,
    ) -> "Provenance | Mapping[str, object]":
        """Default to :meth:`provenance`, recording the blanking cut-off.

        :param provenance: Caller-supplied record, or ``None`` to build one.
        :param min_confidence: Cut-off below which a call is blanked.
        :return: The record to write.
        """
        if provenance is None:
            provenance = self.provenance()
        # An empty record asks for no provenance, so nothing is merged into it.
        if not provenance:
            return provenance
        params = provenance.get("parameters")
        if isinstance(params, dict):
            provenance = Provenance({**provenance, "parameters": dict(params)})
        # A blanked site is only interpretable next to the cut-off that blanked it.
        if min_confidence is not None:
            provenance = Provenance({
                **provenance,
                "parameters": {**provenance.get("parameters", {}),
                               "min_confidence": float(min_confidence)},
            })
        return provenance

    def _completing(self, stream, *records):
        """``stream``, refreshing every one of ``records`` once the walk ends.

        The focal counters are published at the end of the walk, while the
        writers resolve their provenance before draining, so each record is
        updated in place as the last item passes.

        :param stream: The ``(Site, Posterior)`` pairs to forward.
        :param records: The provenance mappings to refresh: the one the writer
            emits, and any the caller supplied and still holds.
        :return: Generator over ``stream``.
        """
        yield from stream
        if not (self._focal_counts_complete and not self.focal.is_root):
            return
        for record in records:
            params = record.get("parameters") if record else None
            if isinstance(params, dict):
                params.update(self._focal_totals)

    def to_zarr(
        self,
        output_zarr: str | os.PathLike,
        *,
        input_zarr: str | os.PathLike | None = None,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
        min_confidence: float | None = None,
        store_posterior: bool = True,
        posteriors: "Iterable[tuple[Site, Posterior]] | None" = None,
        restrict_samples: bool = False,
    ) -> int:
        """Write an annotated VCF Zarr (VCZ) store with the ``variant_AA*`` arrays.

        Feeds the posteriors to
        :class:`~ancestree.writers.ZarrWriter`, which copies a template VCZ
        store and adds the ``variant_AA`` / ``variant_AA_prob`` / ``variant_AA_post``
        arrays plus the provenance record (see that writer for the on-disk
        layout). The template resolves in this order: an explicit
        ``input_zarr``, then the local VCZ store this inference was
        constructed from, and otherwise a template built with ``bio2zarr`` from
        the mode's source (the source VCF, or the tree sequence dumped to a
        temporary VCF).

        :param output_zarr: Destination VCZ store path, which must differ from
            the template.
        :param input_zarr: Template VCZ store. Defaults to the local VCZ store
            the inference was constructed from, or a ``bio2zarr``-built template
            from the mode's source otherwise.
        :param info: Optional run-level constants stored in the store's root
            ``attrs`` under ``ancestree_info``.
        :param provenance: Structured provenance stored under
            ``attrs['ancestree_provenance']``. Defaults to :meth:`provenance`.
            Pass ``{}`` to suppress.
        :param store_posterior: When ``True`` (default), also record the
            whole per-state posterior in the ``variant_AA_post`` array.
        :param min_confidence: Sites with
            :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
            below this get ``variant_AA = "."``. ``None`` (default) disables the check.
        :param posteriors: Stored ``(Site, Posterior)`` pairs to write, as
            :meth:`infer` yields them. ``None`` (default) runs :meth:`infer`.
        :param restrict_samples: Write only the samples the inference used,
            the ingroup and outgroups. ``False`` (default) writes every sample
            of the template.
        :return: Number of variants annotated.
        :raises ImportError: If a template must be built but ``bio2zarr`` is
            not installed.
        :raises ValueError: If no template is available and the mode has no
            source to build one from.
        """
        from ancestree.writers import ZarrWriter
        supplied, provenance = provenance, self._resolve_provenance(
            provenance, min_confidence)
        template = str(input_zarr) if input_zarr is not None else None
        if template is None:
            template = self._input_store_path
        cleanup_dir: str | None = None
        samples = self._used_samples() if restrict_samples else None
        if template is None:
            template, cleanup_dir, samples = self._build_default_template_vcz(
                restrict_samples)
        try:
            return ZarrWriter(
                template, output_zarr, min_confidence=min_confidence,
                samples=samples,
            ).write(
                self._completing(self._posteriors_for_writing(posteriors),
                                 provenance, supplied),
                store_posterior=store_posterior,
                info=info, provenance=provenance,
            )
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def _build_default_template_vcz(
        self, restrict_samples: bool = False,
    ) -> "tuple[str, str, frozenset[str] | None]":
        """Build a VCF Zarr template from the mode's default template VCF.

        Reuses :meth:`_default_template_vcf` (the source VCF, or the tree
        sequence dumped to a temporary VCF) and converts it to a VCZ store with
        ``bio2zarr``. The template's variants match the sites the posteriors
        carry, so the annotation aligns by ``(chrom, pos)``.

        :param restrict_samples: Whether the output holds only the samples
            the inference used.
        :return: ``(template_store_path, cleanup_dir, samples)``, ``samples``
            being the sample columns to write or ``None`` for all. The caller
            removes ``cleanup_dir`` once the store has been consumed.
        :raises ImportError: If ``bio2zarr`` is not installed.
        :raises ValueError: If the mode has no source to build a template from.
        """
        # Resolved ahead of the bio2zarr import, so a mode with no source fails
        # with a ValueError.
        vcf_template, owns_vcf, samples = self._default_template_vcf(
            None, restrict_samples)
        if "://" in vcf_template:
            raise ValueError(
                f"to_zarr cannot build a template from the remote VCF "
                f"{vcf_template!r}, which bio2zarr reads only from local "
                f"files. Pass input_zarr=<store>.")

        def _drop_temp() -> None:
            """Remove the VCF template this call created, if it made one."""
            if owns_vcf:
                try:
                    os.unlink(vcf_template)
                except OSError:
                    pass

        try:
            import bio2zarr.vcf as bio2zarr_vcf
        except ImportError as e:
            _drop_temp()
            raise ImportError(
                "to_zarr without an explicit input_zarr template builds one "
                "from the source with bio2zarr, which is not installed. Install "
                "it with `pip install ancestree[zarr]`, or pass "
                "input_zarr=<.vcz store>."
            ) from e
        cleanup_dir = tempfile.mkdtemp(prefix="ancestree_vcz_")
        vcz_path = os.path.join(cleanup_dir, "template.vcz")
        try:
            # The template is unindexed, so bio2zarr cannot count its records
            # and would log a progress notice that has no bearing here.
            log = logging.getLogger("bio2zarr")
            level = log.level
            log.setLevel(logging.ERROR)
            try:
                bio2zarr_vcf.convert([vcf_template], vcz_path,
                                     show_progress=False)
            finally:
                log.setLevel(level)
        except BaseException:
            # Remove the half-built template directory.
            shutil.rmtree(cleanup_dir, ignore_errors=True)
            raise
        finally:
            _drop_temp()
        return vcz_path, cleanup_dir, samples

    def to_vcf(
        self,
        output_vcf: str | os.PathLike,
        *,
        input_vcf: str | os.PathLike | None = None,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
        contig_id: str | None = None,
        min_confidence: float | None = None,
        store_posterior: bool = True,
        posteriors: "Iterable[tuple[Site, Posterior]] | None" = None,
        restrict_samples: bool = False,
    ) -> int:
        """Write an annotated VCF with ``AA`` / ``AA_prob`` / ``AA_post`` ``INFO`` fields.

        Feeds the posteriors to
        :class:`~ancestree.writers.VCFWriter`, which copies headers and
        variant records from a template VCF. When ``input_vcf`` is ``None``
        the template is the VCF the inference was built from, the records of
        its VCF Zarr store, or else the source or inferred tree sequence dumped
        to a temporary VCF.

        :param output_vcf: Where to write the annotated output.
        :param input_vcf: Optional template VCF path. Defaults to the
            template described above.
        :param info: Optional run-level constants threaded into the VCF
            header / per-record ``AA_<key>`` fields (see
            :class:`~ancestree.writers.VCFWriter`).
        :param provenance: Structured provenance written to the VCF header
            (``##source`` + ``##ancestree_provenance``). Defaults to
            :meth:`provenance`. Pass an explicit dict to override, or ``{}``
            to suppress.
        :param contig_id: Contig label used when auto-writing a template
            from a tree sequence. ``None`` (default) resolves to the
            inference's own contig
            (:paramref:`ARGBasedInference.chrom <ancestree.inference.ARGBasedInference.chrom>`), so the
            template's ``CHROM`` matches the chrom the posteriors carry and
            the annotation actually lands.
        :param store_posterior: When ``True`` (default), also record the
            whole per-state posterior in the ``AA_post`` ``INFO`` field.
        :param posteriors: Stored ``(Site, Posterior)`` pairs to write, as
            :meth:`infer` yields them. ``None`` (default) runs :meth:`infer`.
        :param min_confidence: Sites with
            :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
            below this get ``AA = "."``. ``None`` (default) disables the check.
        :param restrict_samples: Write only the samples the inference used,
            the ingroup and outgroups. ``False`` (default) writes every sample
            of the template.
        :return: Number of records annotated.
        :raises ValueError: If no template VCF is available.
        """
        from ancestree.writers import VCFWriter
        supplied, provenance = provenance, self._resolve_provenance(
            provenance, min_confidence)
        if input_vcf is not None:
            template, owns_temp = str(input_vcf), False
            samples = self._used_samples() if restrict_samples else None
        else:
            template, owns_temp, samples = self._default_template_vcf(
                contig_id, restrict_samples)
        try:
            return VCFWriter(
                template, output_vcf, min_confidence=min_confidence,
                samples=samples,
            ).write(
                self._completing(self._posteriors_for_writing(posteriors),
                                 provenance, supplied),
                store_posterior=store_posterior,
                info=info, provenance=provenance,
            )
        finally:
            if owns_temp:
                try:
                    os.unlink(template)
                except OSError:
                    pass

    def _default_template_vcf(
        self, contig_id: str | None, restrict_samples: bool = False,
    ) -> "tuple[str, bool, frozenset[str] | None]":
        """The VCF this inference was constructed from, the records of its VCF
        Zarr store where every array carries dimension names, or else a
        temporary VCF written from the mode's trees.

        :param contig_id: Contig label for a written template, or ``None`` for
            the mode's own.
        :param restrict_samples: Whether the output holds only the used samples.
        :return: ``(path, owns_temp, samples)``, ``samples`` being the columns
            to keep or ``None`` for all.
        :raises ValueError: There is no source to build a template from.
        :raises ImportError: If a store must be read but ``vcztools`` is not
            installed.
        """
        samples = self._used_samples() if restrict_samples else None
        if self._input_vcf_path is not None:
            return self._input_vcf_path, False, samples
        from ancestree.writers import ZarrWriter

        if (self._input_store_path is not None
                and ZarrWriter._is_tagged(self._input_store_path)):
            return self._vcf_from_store(self._input_store_path), True, samples
        return self._dump_template_vcf(contig_id, restrict_samples), True, None

    @staticmethod
    def _vcf_from_store(store: str) -> str:
        """Write the records of a VCF Zarr store to a temporary VCF.

        :param store: Local store path.
        :return: Path of the temporary file, which the caller unlinks.
        :raises ImportError: If ``vcztools`` is not installed.
        """
        try:
            import zarr
            from vcztools.retrieval import VczReader
            from vcztools.vcf_writer import write_vcf
        except ImportError as e:
            raise ImportError(
                "to_vcf from a VCF Zarr store writes its template with "
                "vcztools, which is not installed. Install it with `pip "
                "install ancestree-popgen[zarr]`, or pass input_vcf=<path>."
            ) from e
        return Inference._temporary_vcf(
            lambda fh: write_vcf(VczReader(zarr.open(store, mode="r")), fh))

    def _dump_template_vcf(
        self, contig_id: str | None, restrict_samples: bool = False,
    ) -> str:
        """Write a temporary template VCF from the mode's own trees.

        :param contig_id: Contig label for the template, or ``None`` to let
            the mode pick its own contig.
        :param restrict_samples: Whether the template holds only the samples
            the inference used.
        :return: Path of the temporary file, which the caller unlinks.
        :raises ValueError: There is no source to build a template from.
        """
        raise ValueError(
            f"{type(self).__name__}: no template VCF available. Pass "
            "input_vcf=<path> to to_vcf() or input_zarr=<path> to to_zarr(), "
            "or construct the inference from a VCF path."
        )

    @staticmethod
    def _write_template_vcf(ts, names, contig: str) -> str:
        """Write ``ts`` to a temporary template VCF.

        Positions are truncated to ``int(site.pos)``, the position the sites
        are keyed on.

        :param ts: The tree sequence to write.
        :param names: Its sample column names.
        :param contig: Contig label of every record.
        :return: Path of the temporary file, which the caller unlinks.
        """
        return Inference._temporary_vcf(lambda fh: ts.write_vcf(
            fh, contig_id=contig, individual_names=names,
            position_transform=lambda p: np.floor(np.asarray(p)).astype(int)))

    @staticmethod
    def _temporary_vcf(write) -> str:
        """A temporary VCF filled by ``write``, removed again if it fails.

        :param write: Callable taking the open text file.
        :return: Path of the file, which the caller unlinks.
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".vcf", delete=False, mode="w")
        try:
            write(tmp)
        except BaseException:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        tmp.close()
        return tmp.name

    def to_arg(
        self,
        output_path: str | os.PathLike,
        *,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
        store_posterior: bool = True,
        min_confidence: float | None = None,
        posteriors: "Iterable[tuple[Site, Posterior]] | None" = None,
        restrict_samples: bool = False,
    ) -> int:
        """Write an annotated ``.trees`` file with ``site.ancestral_state`` populated.

        Feeds the posteriors to
        :class:`~ancestree.writers.TskitWriter`. The MAP allele is
        written as each site's ``ancestral_state``. The full posterior
        is recorded in ``site.metadata['ancestree']`` when
        ``store_posterior=True``. The tree sequence whose sites are
        annotated is resolved per mode by ``_source_tree_sequence`` (the
        source ARG for :class:`~ancestree.inference.ARGBasedInference`, the inferred local-tree
        sequence for :class:`~ancestree.local_tree_inference.LocalTreeInference`).

        :param output_path: Destination ``.trees`` file path.
        :param info: Optional run-level constants stored once under
            ``site.metadata['ancestree']['inference']`` per annotated site.
        :param provenance: Structured provenance appended to the tskit
            provenance table. Defaults to :meth:`provenance` (version + mode
            + parameters). Pass an explicit dict to override, or ``{}`` to
            suppress.
        :param store_posterior: If ``True`` (default), the full posterior
            is stored under ``metadata['ancestree']['posterior']``.
        :param posteriors: Stored ``(Site, Posterior)`` pairs to write, as
            :meth:`infer` yields them. ``None`` (default) runs :meth:`infer`.
        :param min_confidence: Sites with
            :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
            below this get an empty ``ancestral_state``, the tskit convention
            for an unannotated site. ``None`` (default) disables the check.
        :param restrict_samples: Write only the samples the inference used,
            the ingroup and outgroups. ``False`` (default) writes every sample
            of the tree sequence.
        :return: Number of sites annotated.
        :raises ValueError: If this mode has no tree sequence to annotate
            (e.g. fixed-tree mode, where a VCF is written with :meth:`to_vcf`).
        """
        from ancestree.writers import TskitWriter
        supplied, provenance = provenance, self._resolve_provenance(
            provenance, min_confidence)
        return TskitWriter(
            self._source_tree_sequence(restrict_samples), output_path,
            min_confidence=min_confidence,
        ).write(
            self._completing(self._posteriors_for_writing(posteriors),
                             provenance, supplied), info=info,
            provenance=provenance, store_posterior=store_posterior,
        )

    def _posteriors_for_writing(self, posteriors):
        """``posteriors`` as given, or a fresh :meth:`infer` stream."""
        return posteriors if posteriors is not None else self.infer()

    def _source_tree_sequence(
        self, restrict_samples: bool = False,
    ) -> "tskit.TreeSequence":
        """Resolve the tree sequence whose sites :meth:`to_arg` annotates.

        Overridden by the modes that have one. The base implementation
        raises, making :meth:`to_arg` unavailable.

        :param restrict_samples: Whether to hold only the samples the
            inference used.
        :return: The tree sequence to annotate.
        :raises ValueError: There is no tree sequence to annotate.
        """
        raise ValueError(
            f"{type(self).__name__}.to_arg: this mode has no tree sequence to "
            "annotate; write a VCF with to_vcf() instead."
        )

# The ARGBasedInference whose ``_infer_range`` runs in fork-pool workers. The
# parent sets it before creating the pool and clears it after. Children
# inherit it by copy-on-write fork and receive only ``(tree_start,
# tree_stop)``, so the ARG is never pickled.
_ARG_INFERENCE_FOR_FORK: "ARGBasedInference | None" = None


def _arg_infer_chunk_worker(
    chunk: tuple[int, int],
) -> tuple[list[tuple[Site, np.ndarray]], int, int, int, int, int, int, int]:
    """Worker entrypoint for :meth:`ARGBasedInference._infer_fork_pool`.

    :param chunk: ``(tree_start, tree_stop)`` tree-index range.
    :return: The chunk's ``(Site, values)`` list and its seven diagnostic
        counts, for the parent to sum.
    """
    inference = _ARG_INFERENCE_FOR_FORK
    if inference is None:
        raise RuntimeError(
            "_arg_infer_chunk_worker invoked without an inference object "
            "set in the parent; fork did not propagate module globals."
        )
    start, stop = chunk
    # Per-chunk counters.
    inference._n_uniform_fallback = 0
    inference._n_ingroup_non_monophyletic = 0
    inference._n_focal_multiroot_fallback = 0
    inference._n_ingroup_monomorphic = 0
    inference._n_uncoalesced_segments = 0
    inference._n_unrepresentable_sites = 0
    inference._n_unrepresentable_tips = 0
    results = [
        (site, np.asarray(post.values))
        for site, post in inference._infer_range(start, stop, show_progress=False)
    ]
    return (
        results,
        inference._n_uniform_fallback,
        inference._n_ingroup_non_monophyletic,
        inference._n_focal_multiroot_fallback,
        inference._n_ingroup_monomorphic,
        inference._n_uncoalesced_segments,
        inference._n_unrepresentable_sites,
        inference._n_unrepresentable_tips,
    )


# Floor for a rate-map mu that integrates to zero over a local tree's span:
# branch lengths stay near zero and the posterior is prior-dominated.
_ZERO_RATE_FLOOR: float = 1e-12


class ARGBasedInference(Inference):
    """Infer the ancestral allele at every variant in a :class:`tskit.TreeSequence`, using its per-site local tree.

    :meth:`infer` walks the tree sequence and yields ``(Site, Posterior)``
    pairs in tskit's natural variant order. Given an iterable of tree
    sequences, the posterior is marginalised over them. With
    :class:`~ancestree.models.JC69` and the default
    :class:`~ancestree.priors.StationaryPrior` on a short-branch tree,
    :attr:`Posterior.map_allele <ancestree.posterior.Posterior.map_allele>`
    reproduces the :meth:`tskit.Tree.map_mutations` parsimony pick.

    .. code-block:: python

        import ancestree as anc

        inf = anc.ARGBasedInference("snps.trees", anc.JC69(), mu=1e-8)
        inf.to_arg("snps.annotated.trees")

    :param source: Source tree sequence, a path to a ``.trees`` file, or an
        iterable of either (a posterior sample of ARGs). Branch lengths are
        read from the tree and scaled by ``mu`` via
        :attr:`Tree.time_scale <ancestree.trees.Tree.time_scale>`.
    :param model: Substitution model. Defaults to :class:`~ancestree.models.JC69`.
    :param mu: Per-site substitution rate per unit of the tree sequence's
        ``time_units``, scaling branch lengths into expected substitutions
        per site. Omitting it falls back to ``1e-8`` with a warning. Pass
        ``mu=1.0`` when the branch lengths are in substitutions per site. A
        scalar ``float``, or an object exposing ``get_cumulative_mass(position)``
        such as :class:`msprime.RateMap`, whose per-tree rate is the
        interval-weighted mean over each local tree's span.
    :param base_composition: Optional :class:`~ancestree.sites.BaseComposition`
        of empirical base frequencies. Required by the non-symmetric models,
        ignored by :class:`~ancestree.models.JC69` and
        :class:`~ancestree.models.K2`. ``None`` is uniform.
    :param prior: Optional root :class:`~ancestree.priors.StationaryPrior`
        applied after the kernel. ``None`` (default) builds one from the base
        composition. An :class:`~ancestree.priors.IngroupWeight` is rejected
        here and belongs to :class:`~ancestree.inference.FixedTreeInference`.
    :param chrom: Contig name embedded in each emitted :class:`~ancestree.sites.Site`.
    :param sample_map: Optional ``{sample_name: tskit_node_id}`` whose keys
        are the panel. ``None`` derives one from the tree sequence's
        individual names, falling back to ``{str(i): i}`` over the sample
        nodes.
    :param progress: Show a tqdm progress bar over the walk (default
        ``True``). No bar is shown when ``n_workers > 1``.
    :param n_workers: Fork-pool workers for the walk. Default ``1``. On a
        platform without ``fork`` it warns and runs a single worker. Sites are
        emitted in tskit-natural order regardless.
    :param outgroup_samples: Sample ids treated as outgroups, used by the
        ``baseline_check`` comparison. Defaults to the panel samples outside
        ``ingroup_samples``. With both lists named, samples in neither are
        dropped from the tree sequence, or refused where ``sample_map`` names
        them.
    :param ingroup_samples: Sample ids making up the ingroup. Stratifies the
        baseline comparison by folded-SFS bin, and defines the ingroup whose
        MRCA ``focal="ingroup_mrca"`` reports at. Defaults to all non-outgroup
        panel samples.
    :param focal: Node to report the posterior at, as a
        :class:`~ancestree.focal.FocalNode` or an anchor name. ``None``
        (default) is the ingroup MRCA, and ``"panel_root"`` the deepest
        ancestor of the whole panel, deeper than the ingroup MRCA whenever
        outgroups are in the panel.
    :param baseline_check: INFO-log MAP agreement with the majority-outgroup
        rule (:class:`~ancestree.inference.MajorityOutgroupInference`) as a
        consistency check. Off by default, since ARG mode requires no outgroup designation.
        Set ``True`` together with ``outgroup_samples`` or ``ingroup_samples``
        to enable.
    :param mu_matches_time_units: Assert that ``mu`` is expressed per unit of
        this tree sequence's own node times. An undated ARG declares
        ``uncalibrated`` times and is otherwise refused, since no rate scales
        them. Set this when ``mu`` was fitted against the tree sequence itself,
        as a segregating-sites-over-total-branch-length rate is. It also
        silences the units warning on a tree sequence that declares none.

    :raises TypeError: If an :class:`~ancestree.priors.IngroupWeight` is
        passed as ``prior``, which this mode does not fit.
    :raises ValueError: If a named ingroup or outgroup sample is absent from
        the panel or named in both lists, or if ``sample_map`` names a sample
        in neither list while both are named.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"model": type(self.model), "mu": self.mu}

    def __init__(
        self,
        source: "tskit.TreeSequence | str | os.PathLike",
        model: SubstitutionModel | None = None,
        *,
        mu: "float | msprime.RateMap | None" = None,
        base_composition: BaseComposition | None = None,
        prior: "StationaryPrior | None" = None,
        chrom: str = "1",
        sample_map: Mapping[str, int] | None = None,
        progress: bool = True,
        n_workers: int = 1,
        outgroup_samples: Sequence[str] | None = None,
        ingroup_samples: Sequence[str] | None = None,
        baseline_check: bool = False,
        focal: "FocalNode | str | None" = None,
        mu_matches_time_units: bool = False,
    ) -> None:
        """Store the configuration. Defer all compute to :meth:`infer`."""
        from ancestree.priors import (
            StationaryPrior,
        )
        import tskit
        # An iterable of tree sequences is a posterior sample of ARGs. The
        # first draw sets up the geometry and the rest stay lazy.
        draws_rest = None
        draws_source = None
        ts = source
        if not isinstance(ts, (tskit.TreeSequence, str, os.PathLike)) \
                and hasattr(ts, "__iter__"):
            # A re-walkable source (list, tuple) is kept for repeated passes.
            draws_source = ts if iter(ts) is not ts else None
            it = iter(ts)
            first = next(it, None)
            if first is None:
                raise ValueError(
                    "source is an empty iterable; ARGBasedInference needs at "
                    "least one tree sequence to score")
            ts, draws_rest = first, it
        if isinstance(ts, (str, os.PathLike)):
            ts = tskit.load(ts)
        if mu is None:
            mu = DEFAULT_MU
            self._log.warning(
                "ARGBasedInference: no mu given, falling back to %g per site "
                "per generation. That is a human / great-ape figure and scales "
                "branch lengths into expected substitutions, so pass mu= for "
                "your species.", DEFAULT_MU,
            )
        if isinstance(mu, (int, float)):
            mu_value = float(mu)
            if not np.isfinite(mu_value):
                raise ValueError(
                    f"mu must be positive and finite, got {mu_value}")
            if mu_value <= 0:
                raise ValueError(f"mu must be positive, got {mu_value}")
            self.mu: "float | object" = mu_value
        elif hasattr(mu, "get_cumulative_mass"):
            self.mu = mu
        else:
            raise TypeError(
                f"mu must be a positive float or an object with a "
                f"`get_cumulative_mass(position)` method (e.g. "
                f"msprime.RateMap); got {type(mu).__name__}"
            )
        self._check_time_units(ts, mu_matches_time_units)
        #: Remaining draws of the ARG posterior, or None for a single ARG.
        self._draws_rest = draws_rest
        self._draws_source = draws_source
        self._draws_spent = False
        # A fresh model per inference: models carry mutable state.
        model = model if model is not None else JC69()
        self.model = model
        self.base_composition = base_composition
        self._require_stationary_prior(prior)
        self.prior: StationaryPrior = (
            prior if prior is not None
            else StationaryPrior(model, base_composition)
        )
        self.chrom = chrom
        self.progress = progress

        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1; got {n_workers}")
        self.n_workers = int(n_workers)

        self.baseline_check = bool(baseline_check)
        self._outgroup_samples: tuple[str, ...] = tuple(outgroup_samples or ())
        self._ingroup_samples: tuple[str, ...] = tuple(ingroup_samples or ())

        from ancestree.focal import FocalNode
        self.focal = FocalNode.parse(focal)
        self._note_unnamed_ingroup()

        explicit_map = sample_map is not None
        if sample_map is None:
            sample_map = TskitLocalTree.default_sample_map(ts)
        panel = self._resolve_panel(sample_map, explicit=explicit_map)
        # Scoring runs on the tree sequence restricted to the panel. The
        # supplied one is kept for output that holds every sample.
        self._input_ts = ts
        self._input_sample_map = dict(sample_map)
        self._panel_map = {s: int(sample_map[s]) for s in panel}
        self.ts, sample_map = TskitLocalTree.restrict(ts, self._panel_map)
        self.sample_map: dict[str, int] = sample_map
        self._node_to_sample: dict[int, str] = {v: k for k, v in self.sample_map.items()}

        self._ingroup_nodes: tuple[int, ...] = tuple(
            self.sample_map[s] for s in self._resolved_ingroup
        )
        if not self._ingroup_nodes:
            self._refuse_empty_ingroup(len(self.sample_map))
        # Counted over the walk and reported once.
        self._n_ingroup_non_monophyletic = 0
        self._n_focal_multiroot_fallback = 0
        self._focal_counts_complete = False

    def _focal_provenance(self) -> dict[str, object]:
        """The focal node and how often it was cleanly available.

        :return: Provenance entries. Empty when reading at the tree's own root.
        """
        entry = self.focal.provenance()
        if self.focal.is_root:
            return entry
        entry["n_ingroup"] = len(self._ingroup_nodes)
        if self._ingroup_samples:
            entry["ingroup_samples"] = list(self._ingroup_samples)
        if self._outgroup_samples:
            entry["outgroup_samples"] = list(self._outgroup_samples)
        if self._focal_counts_complete:
            entry.update(self._focal_totals)
        return entry

    def _baseline_outgroup_samples(self) -> tuple[str, ...]:
        return self._resolved_outgroups

    def _baseline_ingroup_samples(self) -> tuple[str, ...]:
        return self._resolved_ingroup

    def _infer_marginalised(self):
        """Per-site posteriors marginalised over an ARG posterior sample.

        Each draw ``G_m`` of the ``M`` genealogies is scored on the single-ARG
        path and the likelihoods are averaged, with the prior applied once,

        .. math::

            P(x \\mid s) \\approx \\frac{1}{M} \\sum_{m=1}^{M} P(x \\mid s, G_m),
            \\qquad P(s \\mid x) \\propto \\pi(s)\\, P(x \\mid s),

        where ``x`` is the site's tip alleles, ``s`` the state at the focal
        node and ``pi`` the root prior. One scorer per draw is walked
        concurrently and the streams are merged on site position, so a site is
        finished as soon as every draw has passed it and the accumulator holds
        only the positional skew between the walks. A site missing from some
        draw is averaged over the draws that carry it.
        """
        import heapq
        import itertools
        import operator

        scorers: list = []

        def keyed(draw):
            """One draw's ``(key, site, alleles, log_L)`` stream, ascending in key.

            :param draw: The draw's tree sequence.
            :return: Generator of one tuple per site of that draw.
            """
            scorer = self._clone_for(draw)
            # Scored in-process, so the likelihoods reach the sink.
            scorer.n_workers = 1
            scorer._force_serial = True
            scorer._log_L_sink = sink = {}
            scorers.append(scorer)
            for site, post in scorer.infer():
                # Keyed on the exact position: a continuous-coordinate ARG puts
                # several sites on one integer.
                key = (site.local_tree_handle
                       if site.local_tree_handle is not None else site.pos)
                # Positions are unique within a tree sequence, so taking the
                # row out bounds the sink to the local tree being walked.
                yield key, site, post.alleles, sink.pop(key)

        # Every draw is opened here, so a one-shot source is resolved up front.
        streams = [keyed(draw) for draw in self._chain_draws()]
        if not streams:
            return
        # heapq.merge needs each stream ascending in key: ``_infer_range``
        # walks local trees in index order and each tree's variants in
        # position order, which gives exactly that.
        by_key = operator.itemgetter(0)
        merged = heapq.merge(*streams, key=by_key)
        for _, group in itertools.groupby(merged, key=by_key):
            _, site, alleles, first = next(group)
            ref = float(np.max(first))
            acc = np.exp(first - ref)
            seen = 1
            for _, _, _, log_L in group:
                top = float(np.max(log_L))
                new_ref = max(ref, top)
                acc *= np.exp(ref - new_ref)
                acc += np.exp(log_L - new_ref)
                ref = new_ref
                seen += 1
            # pi applied once to the averaged likelihood, then normalised.
            log_mean = np.log(np.clip(acc / seen, 1e-300, None)) + ref
            log_post = log_mean + self.prior.log_probs([site])[0]
            values = self._normalise_log_post(
                log_post[None, :], n_states=self.model.n_states)[0]
            self._count_ingroup_monomorphic([site])
            self._count_unrepresentable([site])
            yield site, Posterior(alleles=alleles, values=values)
        # Genealogy-level events happen once per draw and add up.
        self._n_ingroup_non_monophyletic += sum(
            s._n_ingroup_non_monophyletic for s in scorers)
        self._n_focal_multiroot_fallback += sum(
            s._n_focal_multiroot_fallback for s in scorers)
        self._n_uncoalesced_segments += sum(
            s._n_uncoalesced_segments for s in scorers)
        self._publish_focal_totals()

    def _chain_draws(self):
        """Every draw in the posterior sample, resolving paths as they come.

        A re-walkable source (list, tuple) is iterated afresh on each call. A
        one-shot generator cannot be rewound, so a second pass raises.
        """
        import tskit

        def resolve(d):
            """Load a draw given as a path, restricted to the panel."""
            ts = tskit.load(d) if isinstance(d, (str, os.PathLike)) else d
            return TskitLocalTree.restrict(ts, self._panel_map)[0]

        if self._draws_source is not None:
            for draw in self._draws_source:
                yield resolve(draw)
            return
        if self._draws_spent:
            raise ValueError(
                "this ARG posterior sample was a one-shot iterator and has "
                "already been consumed; pass a list of tree sequences (or "
                "paths) if the inference has to be run more than once")
        self._draws_spent = True
        yield self.ts
        for draw in self._draws_rest:
            yield resolve(draw)

    def _clone_for(self, ts):
        """A silenced single-ARG scorer over ``ts`` with this instance's
        settings. Logging and the baseline check belong to the marginal."""
        import copy
        clone = copy.copy(self)
        clone.ts = ts
        clone._draws_rest = None
        clone.progress = False
        clone._quiet = True
        clone.baseline_check = False
        # Fresh counters per draw. The parent sums the genealogy-level ones
        # and tallies the per-site ones over the sites the merge emits.
        clone._n_ingroup_non_monophyletic = 0
        clone._n_focal_multiroot_fallback = 0
        clone._n_uncoalesced_segments = 0
        clone._n_uniform_fallback = 0
        clone._n_ingroup_monomorphic = 0
        clone._n_unrepresentable_sites = 0
        clone._n_unrepresentable_tips = 0
        clone._focal_counts_complete = False
        return clone

    def infer(self) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(Site, Posterior)`` for every variant in the tree sequence.

        Given an iterable of tree sequences, the posterior is marginalised over
        them (see ``_infer_marginalised()``), with every draw walked
        concurrently and each site finished once they have all passed it.
        Sites are emitted in tskit's natural variant order. In a multi-root local segment (partial coalescence, common in
        ``tsinfer`` ARGs) the kernel runs on each root's subtree and the
        per-root posteriors are averaged under a uniform prior over the roots,
        which broadens the distribution. Filter by local-tree ``num_roots``
        upstream to exclude such segments. With ``n_workers > 1`` the trees
        are chunked by site count across a fork pool and the stream is
        identical to the single-worker walk.

        :return: Iterator of ``(Site, Posterior)`` pairs.
        """
        if self._draws_rest is not None:
            if not self._quiet:
                self._log.info(
                    "Inferring the ancestral allele, marginalising over an ARG "
                    "posterior sample")
            yield from self._with_baseline_check(self._infer_marginalised())
            if not self._quiet:
                self._log_uniform_fallback_summary()
            return
        quiet = self._quiet
        if not quiet:
            self._log.info(
                "Inferring the ancestral allele at %s sites over %s local trees from the ARG",
                f"{self.ts.num_sites:,}", f"{self.ts.num_trees:,}",
            )
        self._check_alphabet()
        yield from self._with_baseline_check(self._infer_all(quiet))

    def _check_alphabet(self) -> None:
        """Refuse a tree sequence carrying no allele the model can read.

        Every tip would take the all-ones partial and every site would return
        the prior unchanged, which is finite and so trips no underflow or
        fallback counter.

        :raises ValueError: If no allele in the tree sequence maps to a model
            state.
        """
        states = {s.upper() for s in self.model.states}
        alleles: set[str] = set()
        for site in self.ts.sites():
            if site.ancestral_state:
                alleles.add(str(site.ancestral_state).upper())
            for mut in site.mutations:
                if mut.derived_state:
                    alleles.add(str(mut.derived_state).upper())
        if not alleles or alleles & states:
            return
        raise ValueError(
            f"no tip allele in this tree sequence maps to a model state, "
            f"so every site would return the prior unchanged. The alleles "
            f"present are {sorted(alleles)[:8]}; the model's states are "
            f"{sorted(states)}. A tree sequence built by tsinfer, SINGER or "
            f"msprime's binary mutation model carries a '0'/'1' alphabet and "
            f"must be re-encoded to nucleotides first.")

    def _infer_all(
        self, quiet: bool,
    ) -> Iterator[tuple[Site, Posterior]]:
        """Per-site posterior stream over the whole tree sequence."""
        n_workers = Settings._resolve_n_workers(
            self.n_workers, force_serial=self._force_serial)
        parallel = n_workers > 1
        if parallel and not Settings._fork_pool_ok(
                self._log, "tree chunks", n_workers):
            parallel = False
        if parallel:
            yield from self._infer_fork_pool()
        else:
            yield from self._infer_range(0, self.ts.num_trees, show_progress=self.progress)
        if not quiet:
            self._log_uniform_fallback_summary()

    def _infer_range(
        self,
        tree_start: int,
        tree_stop: int,
        *,
        show_progress: bool = False,
    ) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(Site, Posterior)`` for trees in ``[tree_start, tree_stop)``.

        Shared by the single-worker walk and each fork-pool worker.
        """
        engine = Likelihood(
            self.model,
            base_composition=self.base_composition,
        )
        ts = self.ts
        sample_nodes = ts.samples()

        # Per-walk arrays for the native kernel. ``bound_mask`` marks the
        # sample columns with a sample-map binding. The rest read as missing.
        node_times = np.asarray(ts.tables.nodes.time, dtype=np.float64)
        id_map_scratch = np.full(int(ts.num_nodes), -1, dtype=np.int32)
        state_index = STATE_INDEX
        bound_mask = np.array(
            [int(node) in self._node_to_sample for node in sample_nodes],
            dtype=bool,
        )

        # The variants of the genomic window the tree walk covers.
        if tree_stop <= tree_start:
            return
        left = ts.at_index(tree_start).interval.left
        right = ts.at_index(tree_stop - 1).interval.right
        variants_iter = ts.variants(left=left, right=right)

        # ts.at_index + tree.next() walks the slice without the prefix cost
        # of ts.trees().
        tree = ts.at_index(tree_start)

        if show_progress:
            pbar = tqdm(
                total=tree_stop - tree_start,
                desc="ARGBasedInference",
                unit=" trees",
                disable=Settings.disable_pbar,
            )
        else:
            pbar = None

        try:
            current_index = tree_start
            while current_index < tree_stop:
                if tree.num_sites:
                    variants_in_tree = [
                        next(variants_iter) for _ in range(tree.num_sites)
                    ]
                    sites_in_tree = [
                        self._parse_variant(v, sample_nodes)
                        for v in variants_in_tree
                    ]
                    tree_mu = self._mu_for_interval(
                        float(tree.interval.left), float(tree.interval.right),
                    )
                    focal = self._resolve_focal(tree)
                    if tree.num_roots == 1:
                        # Native path: int8 tip states straight into the kernel.
                        tip_states = self._pack_tip_states(
                            variants_in_tree, state_index, bound_mask,
                        )
                        log_L = engine.log_likelihoods_tskit_native(
                            tree, tip_states, sample_nodes, node_times,
                            time_scale=tree_mu, id_map_scratch=id_map_scratch,
                            focal=focal,
                        )
                        self._stash_log_L(sites_in_tree, log_L)
                        log_post = log_L + self.prior.log_probs(sites_in_tree)
                        posteriors = self._normalise_log_post(
                            log_post, n_states=self.model.n_states,
                        )
                    elif focal is not None:
                        # One root holds the ingroup. The others are
                        # disconnected and cancel on normalisation, so
                        # averaging them in would only dilute the answer.
                        holder = self._root_holding_focal(tree, focal)
                        local_tree = TskitLocalTree.from_tskit_tree(
                            tree, sample_map=self.sample_map, root=holder,
                        )
                        local_tree.time_scale = tree_mu
                        posteriors = self._compute_posteriors(
                            engine, local_tree, sites_in_tree, focal=focal,
                        )
                        # The row just stashed covers the holder's tips only.
                        others = [int(r) for r in tree.roots
                                  if int(r) != int(holder)]
                        if self._log_L_sink is not None and others:
                            offset = self._log_root_evidence(
                                engine, tree, others, sites_in_tree, tree_mu,
                                self.prior.log_probs(sites_in_tree),
                            )
                            for i, site in enumerate(sites_in_tree):
                                key = (site.local_tree_handle
                                       if site.local_tree_handle is not None
                                       else site.pos)
                                self._log_L_sink[key] = (
                                    self._log_L_sink[key] + offset[i])
                    else:
                        # Multi-root: marginalise over the roots that subtend
                        # a coalescence. tskit reports an uncoalesced lineage
                        # as a root of its own, and such a root is its own
                        # ancestor, so its "ancestral" allele is the tip's
                        # observed one and including it would echo the data
                        # back as part of the call.
                        # A root that is itself a sample tip is an
                        # uncoalesced lineage, its own ancestor, so its
                        # posterior is a point mass on the observed allele. A
                        # unary internal root is not that: it sits above its
                        # sample at positive time and its posterior is
                        # informative, so it stays in the mixture.
                        roots = [int(r) for r in tree.roots
                                 if not (tree.is_sample(int(r))
                                         and tree.num_children(int(r)) == 0)]
                        if not roots:
                            # No lineage has coalesced anywhere in this tree,
                            # so it constrains no ancestral node and the prior
                            # is the whole answer.
                            self._n_uncoalesced_segments += 1
                            log_prior = self.prior.log_probs(sites_in_tree)
                            if self._log_L_sink is not None:
                                offset = self._log_root_evidence(
                                    engine, tree, tree.roots, sites_in_tree,
                                    tree_mu, log_prior,
                                )
                                self._stash_log_L(
                                    sites_in_tree,
                                    np.zeros_like(log_prior) + offset[:, None])
                            posteriors = self._normalise_log_post(
                                log_prior, n_states=self.model.n_states,
                            )
                            self._count_ingroup_monomorphic(sites_in_tree)
                            self._count_unrepresentable(sites_in_tree)
                            for site, p in zip(sites_in_tree, posteriors):
                                yield site, Posterior(
                                    alleles=self.model.states, values=p)
                            if pbar is not None:
                                pbar.update(1)
                            current_index += 1
                            if current_index < tree_stop:
                                tree.next()
                            continue
                        # Marginalise under a uniform prior over which root is
                        # meant. Which one a caller is asking about is a choice
                        # the sequence data carry no information about, so that
                        # prior is not updated and the mixture is an
                        # equal-weight mean.
                        per_root_trees = [
                            TskitLocalTree.from_tskit_tree(
                                tree, sample_map=self.sample_map, root=r,
                            )
                            for r in roots
                        ]
                        for lt in per_root_trees:
                            lt.time_scale = tree_mu
                        log_prior = self.prior.log_probs(sites_in_tree)
                        per_root_log_L = [
                            engine.log_likelihoods(lt, sites_in_tree)
                            for lt in per_root_trees
                        ]
                        per_root = [
                            self._normalise_log_post(
                                log_L + log_prior,
                                n_states=self.model.n_states)
                            for log_L in per_root_log_L
                        ]
                        posteriors = np.mean(per_root, axis=0)
                        # The draws path mixes in likelihood space, so the row
                        # records the mixture with every root's evidence in it.
                        if self._log_L_sink is not None:
                            offset = self._log_root_evidence(
                                engine, tree, tree.roots, sites_in_tree,
                                tree_mu, log_prior,
                                known=dict(zip((int(r) for r in roots),
                                               per_root_log_L)),
                            )
                            self._stash_log_L(
                                sites_in_tree,
                                np.log(np.clip(posteriors, 1e-300, None))
                                - log_prior + offset[:, None],
                            )
                    self._count_ingroup_monomorphic(sites_in_tree)
                    self._count_unrepresentable(sites_in_tree)
                    for site, p in zip(sites_in_tree, posteriors):
                        yield site, Posterior(alleles=self.model.states, values=p)
                if pbar is not None:
                    pbar.update(1)
                current_index += 1
                if current_index < tree_stop:
                    tree.next()
        finally:
            if pbar is not None:
                pbar.close()

    def _log_root_evidence(
        self, engine, tree, roots, sites, tree_mu, log_pi, known=None,
    ) -> np.ndarray:
        """Per-site ``sum_r log Z_r`` over ``roots``.

        ``Z_r = sum_s pi_s L_r(s)`` is the marginal likelihood of the tips
        under root ``r``, with ``s`` the state at ``r``, ``pi_s`` the root
        prior and ``L_r`` the Felsenstein likelihood. A root that is a bare
        sample tip contributes ``pi_a`` for its observed allele ``a``, and 1
        when that allele is absent or outside the alphabet.

        The local trees of one draw are scored independently, so each carries
        only the evidence of its own tips. Summing the others' log evidence
        puts every draw's stashed row on the one scale
        :meth:`Inference._infer_marginalised` mixes in.

        :param engine: The :class:`~ancestree.likelihood.Likelihood` in use.
        :param tree: The tskit tree the roots belong to.
        :param roots: Node ids to accumulate evidence over.
        :param sites: The :class:`~ancestree.sites.Site` records scored.
        :param tree_mu: Time scale for this local tree.
        :param log_pi: ``(len(sites), S)`` log root prior.
        :param known: Optional ``{root: log_L}`` already computed, to avoid
            re-scoring a subtree.
        :return: ``(len(sites),)`` array of summed log evidence.
        """
        from ancestree.trees import TskitLocalTree
        out = np.zeros(len(sites), dtype=float)
        known = known or {}
        state_index = {a: i for i, a in enumerate(self.model.states)}
        for r in roots:
            r = int(r)
            if tree.is_sample(r) and tree.num_children(r) == 0:
                name = self._node_to_sample.get(r)
                for i, site in enumerate(sites):
                    allele = site.tip_alleles.get(name)
                    idx = state_index.get((allele or "").upper())
                    if idx is not None:
                        out[i] += log_pi[i, idx]
                continue
            log_L = known.get(r)
            if log_L is None:
                local_tree = TskitLocalTree.from_tskit_tree(
                    tree, sample_map=self.sample_map, root=r,
                )
                local_tree.time_scale = tree_mu
                log_L = engine.log_likelihoods(local_tree, sites)
            out += logsumexp(log_L + log_pi, axis=1)
        return out

    def _build_tree_chunks(self, n_chunks: int) -> list[tuple[int, int]]:
        """Group local trees into ``n_chunks`` roughly equal-site-count windows.

        Site density varies along the ARG, so chunks are cut by site count,
        not tree count.

        :return: Half-open ``(tree_start, tree_stop)`` pairs covering
            ``[0, num_trees)`` contiguously, empty chunks dropped.
        """
        ts = self.ts
        num_trees = int(ts.num_trees)
        if num_trees == 0:
            return []
        # At most one chunk per tree.
        n_chunks = min(n_chunks, num_trees)

        total_sites = int(ts.num_sites)
        if total_sites == 0:
            return [(0, num_trees)]
        breakpoints = ts.breakpoints(as_array=True)
        # A site on a breakpoint lands in the tree to its right.
        tree_idx = np.searchsorted(
            breakpoints, ts.sites_position, side="right",
        ) - 1
        tree_idx = np.clip(tree_idx, 0, num_trees - 1)
        per_tree_sites = np.bincount(tree_idx, minlength=num_trees).astype(np.int64)

        # Boundary k lands at the first tree whose cumulative site count
        # exceeds k * target.
        target = max(1, total_sites // n_chunks)
        cum = np.cumsum(per_tree_sites)
        thresholds = np.arange(1, n_chunks) * target
        boundaries = np.searchsorted(cum, thresholds, side="right")
        boundaries = np.clip(boundaries, 1, num_trees)
        # At least one tree per chunk.
        for k in range(boundaries.size):
            min_stop = k + 1
            max_stop = num_trees - (n_chunks - 1 - k)
            boundaries[k] = max(min_stop, min(boundaries[k], max_stop))
        # Strictly increasing, since a flat run in cum can repeat an index.
        for k in range(1, boundaries.size):
            if boundaries[k] <= boundaries[k - 1]:
                boundaries[k] = boundaries[k - 1] + 1
        boundaries = np.concatenate([[0], boundaries, [num_trees]])
        chunks: list[tuple[int, int]] = [
            (int(boundaries[k]), int(boundaries[k + 1]))
            for k in range(boundaries.size - 1)
            if int(boundaries[k]) < int(boundaries[k + 1])
        ]
        return chunks

    def _infer_fork_pool(self) -> Iterator[tuple[Site, Posterior]]:
        """Fork-pool driver for :meth:`infer`. Falls back to single-worker
        on non-fork platforms with a warning.

        The parent partitions trees into chunks (by site count), the
        fork-pool runs each chunk's ``_infer_range`` to completion
        in a child, and the parent yields results in chunk order.
        """
        import multiprocessing as mp

        if "fork" not in mp.get_all_start_methods():
            self._log.warning(
                "ARGBasedInference(n_workers=%s): the 'fork' start method is "
                "unavailable on this platform (likely Windows). Falling back "
                "to single-worker. Pickling the ARG to spawn workers would "
                "dominate the runtime and defeat the purpose of "
                "parallelisation.", self.n_workers,
            )
            yield from self._infer_range(0, self.ts.num_trees, show_progress=self.progress)
            return

        n_workers = Settings._resolve_n_workers(
            self.n_workers, force_serial=self._force_serial)
        # Several chunks per worker bound what the parent holds in flight.
        chunks = self._build_tree_chunks(n_workers * self.FORK_CHUNKS_PER_WORKER)
        if len(chunks) <= 1:
            yield from self._infer_range(0, self.ts.num_trees, show_progress=self.progress)
            return

        # The workers read self from the module global.
        global _ARG_INFERENCE_FOR_FORK
        _ARG_INFERENCE_FOR_FORK = self

        import collections

        ctx = mp.get_context("fork")
        # Workers return raw arrays and per-chunk counts for the parent to rewrap and sum.
        states = self.model.states
        try:
            with ctx.Pool(
                    processes=n_workers,
                    initializer=Settings._cap_worker_threads,
                    initargs=(n_workers,)) as pool:
                # A sliding window of submissions bounds the results queue.
                window = n_workers + 2
                remaining = iter(chunks)
                in_flight: collections.deque = collections.deque()
                for _ in range(window):
                    nxt = next(remaining, None)
                    if nxt is None:
                        break
                    in_flight.append(
                        pool.apply_async(_arg_infer_chunk_worker, (nxt,)))
                while in_flight:
                    (chunk_results, n_fallback, n_non_mono, n_multi,
                     n_mono, n_uncoal, n_unrep_sites,
                     n_unrep_tips) = in_flight.popleft().get()
                    nxt = next(remaining, None)
                    if nxt is not None:
                        in_flight.append(
                            pool.apply_async(_arg_infer_chunk_worker, (nxt,)))
                    self._n_ingroup_non_monophyletic += int(n_non_mono)
                    self._n_focal_multiroot_fallback += int(n_multi)
                    self._n_uniform_fallback += int(n_fallback)
                    self._n_ingroup_monomorphic += int(n_mono)
                    self._n_uncoalesced_segments += int(n_uncoal)
                    self._n_unrepresentable_sites += int(n_unrep_sites)
                    self._n_unrepresentable_tips += int(n_unrep_tips)
                    for site, values in chunk_results:
                        yield site, Posterior(alleles=states, values=values)
                    del chunk_results
        finally:
            _ARG_INFERENCE_FOR_FORK = None

    def _parse_variant(
        self,
        variant: "tskit.Variant",
        sample_nodes: np.ndarray,
    ) -> Site:
        """Convert a tskit ``Variant`` into a :class:`~ancestree.sites.Site`.

        A missing genotype (``-1``) and the empty-string placeholder allele
        both map to ``tip_alleles[s] = None``.

        :param variant: A :class:`tskit.Variant` from ``ts.variants()``.
        :param sample_nodes: The ``ts.samples()`` array.
        :return: A :class:`~ancestree.sites.Site` whose ``local_tree_handle``
            is the variant's genomic position.
        """
        return _site_from_tskit_variant(
            variant, sample_nodes, self._node_to_sample, self.chrom,
        )

    def _pack_tip_states(
        self,
        variants: list["tskit.Variant"],
        state_index: dict[str, int],
        bound_mask: np.ndarray,
    ) -> np.ndarray:
        """Pack a batch of tskit variants into an ``int8`` tip-state matrix.

        Entry ``[b, i]`` is the model state index of sample column ``i`` at
        variant ``b``, or ``-1`` for missing: a missing genotype, an allele
        outside the model alphabet, or a column with no sample-map binding.
        A ``-1`` yields the same all-ones tip partial as an absent
        ``tip_alleles`` entry on the generic path.

        :param variants: The :class:`tskit.Variant` records for one local
            tree, in the order of the sites built by :meth:`_parse_variant`.
        :param state_index: ``{allele_letter: state_index}`` in model order.
        :param bound_mask: ``(n_samples,) bool``. False columns are missing.
        :return: ``(len(variants), n_samples) int8`` tip-state matrix.
        """
        n_samples = bound_mask.shape[0]
        tip = np.full((len(variants), n_samples), -1, dtype=np.int8)
        for b, v in enumerate(variants):
            # Allele index to state index, with a trailing slot for g < 0.
            alleles = v.alleles
            lut = np.fromiter(
                (state_index.get(a.upper(), -1) if a else -1 for a in alleles),
                dtype=np.int8, count=len(alleles),
            )
            lut = np.append(lut, np.int8(-1))
            g = np.asarray(v.genotypes)
            g_safe = np.where(g < 0, len(alleles), g)
            tip[b] = lut[g_safe]
        tip[:, ~bound_mask] = -1
        return tip

    def _compute_posteriors(
        self,
        engine: Likelihood,
        local_tree: TskitLocalTree,
        sites: list[Site],
        *,
        focal: "ResolvedFocal | None" = None,
    ) -> np.ndarray:
        """Per-site posteriors ``P(root = s | data, model, prior)`` for a
        batch of sites on one local tree, prior applied and normalised.

        :param engine: A :class:`~ancestree.likelihood.Likelihood` for the
            inference's model.
        :param local_tree: The :class:`~ancestree.trees.TskitLocalTree` covering ``sites``.
        :param sites: The :class:`~ancestree.sites.Site` records to evaluate.
        :param focal: Optional node to report at. ``None`` reads at the tree's
            own root.
        :return: ``(len(sites), model.n_states)`` array whose rows sum to 1.
        """
        if focal is not None and (
            focal.node != local_tree.root or focal.tau > 0.0
        ):
            from ancestree.trees import RerootedTree
            local_tree = RerootedTree(local_tree, focal.node, focal.tau)
        log_L = engine.log_likelihoods(local_tree, sites)  # (B, S)
        self._stash_log_L(sites, log_L)
        log_post = log_L + self.prior.log_probs(sites)  # (B, S)
        return self._normalise_log_post(log_post, n_states=self.model.n_states)

    def _resolve_focal(self, tree) -> "ResolvedFocal | None":
        """Resolve the configured focal node against one local tree.

        :param tree: The :class:`tskit.Tree` covering the current segment.
        :return: The resolved focal node, or ``None`` when the run reads at the
            tree's own root (the default) or when the ingroup spans several
            roots and no MRCA exists.
        """
        from ancestree.focal import FocalNode

        if self.focal.is_root:
            return None
        if FocalNode.spans_roots(tree, self._ingroup_nodes):
            self._n_focal_multiroot_fallback += 1
            return None
        resolved = self.focal.locate(tree, self._ingroup_nodes)
        if resolved is not None and resolved.ingroup_is_monophyletic is False:
            self._n_ingroup_non_monophyletic += 1
        return resolved

    @staticmethod
    def _root_holding_focal(tree, focal: "ResolvedFocal") -> int:
        """The root of the multi-root segment whose subtree holds ``focal``.

        :param tree: The multi-root :class:`tskit.Tree`.
        :param focal: The resolved focal node.
        :return: That root's node id.
        """
        for root in tree.roots:
            if tree.is_descendant(int(focal.node), int(root)):
                return int(root)
        return int(tree.roots[0])

    # ────────────────────────────────────────────────────────────── output

    _MODE = "arg"

    def _provenance_parameters(self) -> dict[str, object]:
        """ARG-mode run parameters: model, prior, μ, workers.

        ``n_draws`` records how many genealogies the posterior was marginalised
        over, so an annotated output distinguishes a single-ARG run from one
        averaged across a chain. It is omitted where the sample is a one-shot
        iterator of unknown length.
        """
        mu = self.mu
        params: dict[str, object] = {
            "model": self._model_name(self.model),
            "prior": type(self.prior).__name__,
            "mu": float(mu) if isinstance(mu, float) else "variable",
            "chrom": self.chrom,
            "n_workers": int(self.n_workers),
            **self._model_provenance(),
            **self._focal_provenance(),
        }
        n_draws = self._n_draws()
        if n_draws is not None:
            params["n_draws"] = n_draws
        return params

    def _n_draws(self) -> "int | None":
        """Size of the posterior sample, or ``None`` if it cannot be counted.

        :return: Number of genealogies, or ``None`` for a one-shot iterator.
        """
        if self._draws_source is not None:
            try:
                return len(self._draws_source)
            except TypeError:
                return None
        if self._draws_rest is None:
            return 1
        try:
            return 1 + len(self._draws_rest)
        except TypeError:
            return None

    def _dump_template_vcf(
        self, contig_id: str | None, restrict_samples: bool = False,
    ) -> str:
        """Dump the supplied tree sequence, or the one restricted to the
        panel, to a temporary template VCF.

        :param contig_id: Contig label for the template. ``None`` resolves to
            :paramref:`ARGBasedInference.chrom <ancestree.inference.ARGBasedInference.chrom>`.
        :param restrict_samples: Whether to write the restricted tree sequence.
        :return: Path of the temporary file, which the caller unlinks.
        """
        contig = contig_id if contig_id is not None else self.chrom
        # Columns keep the names they carry in the unrestricted output.
        nodes = set(self._panel_map.values()) if restrict_samples else None
        names = self._template_individual_names(
            self._input_ts, self._input_sample_map, nodes)
        ts = self.ts if restrict_samples else self._input_ts
        return self._write_template_vcf(ts, names, contig)

    def _source_tree_sequence(
        self, restrict_samples: bool = False,
    ) -> "tskit.TreeSequence":
        """The source ARG. Its sites receive the annotated ancestral states.

        :param restrict_samples: Whether to hold only the panel.
        :return: The supplied tree sequence, or the one restricted to the panel
            with its node ids kept, so its samples keep their names.
        """
        if not restrict_samples:
            return self._input_ts
        return TskitLocalTree.restrict(self._input_ts, self._panel_map,
                                       keep_node_ids=True)[0]


class FixedTreeInference(Inference):
    """ML-fit branch rates on an :class:`~ancestree.trees.OutgroupLadderTree` + per-site posteriors.

    The free branch rates of the :class:`~ancestree.trees.OutgroupLadderTree`
    are fitted by maximum likelihood against the polymorphic-site outgroup
    evidence plus a monomorphic-site contribution from a
    :class:`~ancestree.sites.BaseComposition`. :meth:`infer` then yields
    per-site posteriors over the ingroup MRCA state under the fitted tree.
    Model-internal parameters (``K2(fit_kappa=True)``) are fitted jointly.

    .. code-block:: python

        import ancestree as anc

        inf = anc.FixedTreeInference(
            "snps.vcf.gz", anc.JC69(),
            ingroup_samples=["i1", "i2"], outgroup_samples=["o1", "o2"],
            n_target_sites=1_000_000)
        inf.fit()
        inf.to_vcf("snps.annotated.vcf.gz")

    Under a bare stationary root prior, detailed balance removes ``K0`` (the
    branch above the ingroup MRCA) from the marginal likelihood and the
    one-outgroup case is degenerate. The default
    :class:`~ancestree.priors.KingmanIngroupWeight` resolves both by
    conditioning on the ingroup frequencies. Monomorphic-site evidence is
    likewise required, as ``n_target_sites=L`` or a composition carrying
    per-base counts, or the fitted branch rates inflate.

    :param source: The site data: a VCF / BCF / VCZ path, a ``.trees`` path,
        a :class:`~ancestree.sites.SiteSource`, a :class:`tskit.TreeSequence`,
        or a ``list[Site]``. Ingroup alleles enter through the ingroup weight,
        outgroup alleles through the Felsenstein kernel.
    :param tree: Optional pre-built :class:`~ancestree.trees.OutgroupLadderTree`,
        fitted in place, whose sample lists are read from it. ``None``
        (default) builds the ladder from the sample names.
    :param ingroup_samples: Ingroup sample names. Required without ``tree``.
    :param outgroup_samples: Outgroup sample names, closest first. Required
        without ``tree``. ``[]`` is the no-outgroup mode, where the prior
        alone is the posterior.
    :param model: Substitution model. Defaults to :class:`~ancestree.models.JC69`.
        Branch rates are in expected substitutions per site, so there is no
        ``mu`` argument.
    :param base_composition: Monomorphic-site composition. With all-zero
        ``counts`` (:meth:`BaseComposition.no_counts() <ancestree.sites.BaseComposition.no_counts>`),
        ``n_target_sites`` is required and the per-base monomorphic weights
        are ``round(pi × (n_target_sites − n_polymorphic))``.
    :param n_target_sites: Total target-region length, from which the
        monomorphic weights are derived. Required when
        ``base_composition.counts`` is all-zero and ``fit_required=True``.
    :param sample_filter: Restrict a VCF / VCZ read to these sample names,
        each of which must be in ``ingroup_samples`` or ``outgroup_samples``.
        ``None`` reads every sample and ignores those in neither list.
    :param chrom_filter: Restrict a VCF / VCZ read to this contig.
    :param ploidy: Sample ploidy for the VCF reader. ``None`` infers it.
    :param initial_rates: Length-:attr:`OutgroupLadderTree.n_params <ancestree.trees.OutgroupLadderTree.n_params>`
        L-BFGS-B starting vector. Defaults to
        :attr:`OutgroupLadderTree.param_vector <ancestree.trees.OutgroupLadderTree.param_vector>`.
    :param bounds: ``(lower, upper)`` for every branch rate. Default
        ``(1e-9, 10.0)``.
    :param progress: Show a tqdm bar over the per-site pass. A streamed
        source gives the bar no total.
    :param fixed_params: Optional ``{param_name: value}`` held constant
        during the fit, for instance a published divergence or ``K0``.
    :param n_starts: Independent L-BFGS-B runs. Default ``10``. The first
        start is ``initial_rates``, the rest are log-uniform on ``bounds``,
        and the MLE is the highest-log-likelihood result.
    :param parallelize: Run the starts in parallel. Default ``False``.
    :param n_workers: Worker count when ``parallelize=True``. ``None``
        (default) is ``min(n_starts, os.cpu_count())``.
    :param seed: Seed for the multistart sampling. Default ``42``.
    :param ingroup_weight: Per-site :class:`~ancestree.priors.IngroupWeight`
        seeded on the ingroup MRCA, which identifies ``K0``.
        :class:`~ancestree.priors.KingmanIngroupWeight` is the default;
        :class:`~ancestree.priors.AdaptiveIngroupWeight` fits its per-bin
        parameters after the tree-rate MLE.
    :param prior: Optional root :class:`~ancestree.priors.StationaryPrior`.
        An :class:`~ancestree.priors.IngroupWeight` here raises
        :class:`TypeError`. ``None`` (default) applies the model's stationary
        vector, or the empirical π when the composition carries counts.
    :param outgroup_similarity_threshold: Warn when an outgroup pair differs
        at fewer than this fraction of jointly-observed polymorphic sites,
        since the fit then collapses their branch rates. Default ``0.01``.
        ``0`` disables the check.
    :param focal: Node to report the posterior at, as in the genealogy modes.
        ``None`` (default) or ``"ingroup_mrca"`` reads at the ladder's own
        root, where the ingroup weight applies. A
        :class:`~ancestree.focal.FocalNode` placed above it reads deeper,
        under the stationary prior, where a site the ingroup has fixed is
        answered from the outgroups.
    :param fit_required: Require an ML branch-rate fit before :meth:`infer`.
        ``True`` (default) makes :meth:`infer` raise until
        :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>`
        has run, and ``False`` uses the tree's current rates and relaxes the
        monomorphic-calibration check.
    :param baseline_check: INFO-log MAP agreement with the majority-outgroup
        rule (:class:`~ancestree.inference.MajorityOutgroupInference`) as a
        consistency check on the fitted model. Default ``True``.
    :param subsample_size: Ingroup AFS dimension the fit's configs are
        hypergeometrically projected onto, shared with an AFS-aware weight.
        ``None`` (default) inherits the weight's value, else
        ``min(n_ingroup, 11)``.
    :param stream: Stream sites in bounded memory. ``None`` (default) streams
        a path, a :class:`~ancestree.sites.SiteSource` and a
        :class:`tskit.TreeSequence`, and materialises a ``list[Site]`` and the
        no-outgroup mode. Results are identical either way.

    :raises ValueError: On a tree with no outgroups, an ingroup and
        outgroup set that overlap, a ``sample_filter`` entry in neither list,
        or an unresolvable focal placement.
    :raises NotImplementedError: On a combination the fixed-tree kernel does
        not implement, such as streaming with no outgroups named.
    """

    #: AFS projection dimension the fit's configs are projected onto, in
    #: haplotypes. ``None`` in the no-outgroup mode, which runs no fit.
    subsample_size: int | None = None

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"model": type(self.model),
                "n_outgroups": self.tree.n_outgroups if self.tree is not None else 0,
                "fitted": self.params_mle is not None}

    def __init__(
        self,
        source: "str | os.PathLike | SiteSource | tskit.TreeSequence | Sequence[Site]",
        model: "SubstitutionModel | None" = None,
        base_composition: BaseComposition | None = None,
        *,
        tree: "OutgroupLadderTree | None" = None,
        ingroup_samples: Sequence[str] | None = None,
        outgroup_samples: Sequence[str] | None = None,
        n_target_sites: int | None = None,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        ploidy: int | None = None,
        initial_rates: np.ndarray | None = None,
        bounds: tuple[float, float] = (1e-9, 10.0),
        progress: bool = True,
        outgroup_similarity_threshold: float = 0.01,
        focal: "FocalNode | str | None" = None,
        fixed_params: Mapping[str, float] | None = None,
        n_starts: int = 10,
        parallelize: bool = False,
        n_workers: int | None = None,
        seed: int = 42,
        ingroup_weight: "IngroupWeight | None" = None,
        prior: "StationaryPrior | None" = None,
        fit_required: bool = True,
        baseline_check: bool = True,
        subsample_size: int | None = None,
        stream: bool | None = None,
    ) -> None:
        """Resolve the source, tree and prior. Defer compute to
        :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>` / :meth:`infer`."""
        from ancestree.sources import SiteSource

        if isinstance(source, Tree):
            raise TypeError(
                "FixedTreeInference: `source` is the site data (path / VCF / "
                "VCZ / TreeSequence / SiteSource / list[Site]); pass a pre-built "
                "OutgroupLadderTree via tree= instead."
            )

        # An explicit ladder carries its own sample lists.
        if tree is not None:
            if ingroup_samples is None:
                ingroup_samples = list(tree.ingroup_samples) or None
            if outgroup_samples is None:
                outgroup_samples = list(tree.outgroup_samples) or None

        # No-outgroup mode: the posterior is the normalised prior on the ingroup.
        self._no_outgroup_mode = (
            outgroup_samples is not None and len(outgroup_samples) == 0
        )

        if stream is None:
            import tskit
            stream = (
                isinstance(source, (str, os.PathLike, SiteSource, tskit.TreeSequence))
                and not self._no_outgroup_mode
            )
        self._streaming = bool(stream)
        self._stream_source = None
        if self._streaming and self._no_outgroup_mode:
            raise NotImplementedError(
                "FixedTreeInference(stream=True) is not supported in the "
                "no-outgroup mode."
            )

        self._set_input_paths(source)
        self._chrom_filter = chrom_filter
        self._sample_filter = list(sample_filter) if sample_filter else None
        if self._streaming:
            self._stream_source = SiteSource.resolve(
                source, sample_filter=sample_filter,
                chrom_filter=chrom_filter, ploidy=ploidy,
            )
            actual_sites: list[Site] = []  # streamed twice, never materialised
        else:
            actual_sites = list(SiteSource.resolve(
                source, sample_filter=sample_filter,
                chrom_filter=chrom_filter, ploidy=ploidy,
            ))
        model = model if model is not None else JC69()
        actual_bc = (
            base_composition if base_composition is not None
            else BaseComposition.no_counts()
        )

        # Resolve the topology: no-outgroup (prior is the posterior), an
        # explicit ladder, or one built from the ingroup / outgroup names.
        if self._no_outgroup_mode:
            if ingroup_samples is None:
                raise ValueError(
                    "FixedTreeInference: ingroup_samples is required even "
                    "in the no-outgroup mode (outgroup_samples=[])."
                )
            actual_tree = None  # no ladder. The prior is the posterior
            self._ingroup_samples_no_out: list[str] = list(ingroup_samples)
        elif tree is not None:
            actual_tree = tree
        else:
            if ingroup_samples is None or outgroup_samples is None:
                raise ValueError(
                    "FixedTreeInference: ingroup_samples and outgroup_samples "
                    "are required to build the OutgroupLadderTree (or pass a "
                    "pre-built one via tree=)."
                )
            actual_tree = OutgroupLadderTree(ingroup_samples, outgroup_samples)

        self.tree = actual_tree
        self.sites: list[Site] = list(actual_sites)
        self._check_samples_present(outgroup_samples, actual_sites,
                                    self._stream_source,
                                    ingroup_samples=ingroup_samples)
        _refuse_overlap(ingroup_samples or (), outgroup_samples or ())
        self._check_filter_labelled(sample_filter, ingroup_samples,
                                    outgroup_samples)
        self.model = model
        actual_bc = BaseComposition.require(
            actual_bc, context="FixedTreeInference",
        )
        # One pass over the source projects the polymorphic sites onto the
        # config histogram. The no-outgroup path returns before any fit.
        resolved_subsample_size: int | None = None
        proj_weights: dict | None = None
        n_poly_projected: int | None = None
        if not self._no_outgroup_mode:
            resolved_subsample_size = self._resolve_subsample_size(
                ingroup_weight, subsample_size,
            )
            self.subsample_size = resolved_subsample_size
            proj_source = (
                self._stream_source if self._streaming else (self.sites or [])
            )
            proj_weights, n_poly_projected = actual_bc._project_polymorphic(
                proj_source,
                self.tree.ingroup_samples,
                self.tree.outgroup_samples,
                resolved_subsample_size,
            )
        # A composition without per-base counts takes them from n_target_sites
        # as round(pi x n_target), keeping its empirical pi.
        n_poly_for_counts = (
            n_poly_projected if n_poly_projected is not None
            else len(self.sites or [])
        )
        # Read before the region fill: the prior follows caller-supplied counts.
        supplied_counts = actual_bc.n_total > 0
        if actual_bc.n_total == 0 and n_target_sites is not None:
            arr = BaseComposition._largest_remainder(
                actual_bc.pi, int(n_target_sites),
            )
            actual_bc = BaseComposition(
                counts=dict(zip(STATES, [int(x) for x in arr])),
                n_ts=actual_bc.n_ts,
                n_tv=actual_bc.n_tv,
                _pi_cache=actual_bc._pi_cache,
                _pi_is_ascertained=actual_bc._pi_is_ascertained,
            )
        self.base_composition = actual_bc
        mono_counts = actual_bc.monomorphic_counts(n_poly_for_counts)
        # A fit needs monomorphic-site calibration: non-zero counts or
        # n_target_sites.
        if (
            fit_required
            and not self._no_outgroup_mode
            and self.base_composition.n_total == 0
            and n_target_sites is None
        ):
            raise ValueError(
                "FixedTreeInference requires monomorphic-site calibration "
                "for the branch-rate fit. Supply ``n_target_sites=L`` "
                "(the genome length, so the inference can derive the "
                "monomorphic-site count from the polymorphic count), "
                "or pass a ``base_composition`` whose ``counts`` are "
                "non-zero (e.g. via "
                "``BaseComposition.from_n_target_sites(L)`` / "
                "``BaseComposition.from_fasta(path)`` / "
                "``BaseComposition.from_counts(...)``). Use "
                "``fit_required=False`` if you intentionally want to "
                "skip the fit and use the tree's current branch rates."
            )
        tree = self.tree
        self.bounds = bounds
        self.progress = progress
        self.n_starts = int(n_starts)
        self.parallelize = bool(parallelize)
        self.n_workers = n_workers
        self.seed = int(seed)
        self.fit_required = bool(fit_required)
        self.baseline_check = bool(baseline_check)

        if self._no_outgroup_mode:
            # No ladder, parameters or fit: infer() emits the normalised prior.
            from ancestree.priors import KingmanIngroupWeight
            self.ingroup_weight = (
                ingroup_weight if ingroup_weight is not None
                else KingmanIngroupWeight(self._ingroup_samples_no_out)
            )
            self.prior = prior
            self._log.warning(
                "FixedTreeInference: no outgroups supplied (outgroup_samples=[]). "
                "Using the prior alone: no branch-rate fit and no Felsenstein "
                "evaluation. Per-site posteriors are the normalised prior over "
                "the ingroup; under the default KingmanIngroupWeight, "
                "mono-allelic ingroup sites fall back to uniform but "
                "polyallelic sites cannot place mass on unobserved alleles. "
                "Supply outgroup_samples for a proper fit, or pass a smoother "
                "IngroupWeight explicitly."
            )
            # provenance() and every annotated write read the focal node.
            self.focal = FocalNode.parse(focal)
            self._params_mle = None
            self._log_likelihood_mle = None
            self._model_params_mle = None
            return

        if self.n_starts < 1:
            raise ValueError(f"n_starts must be >= 1; got {n_starts}")

        # ---- fixed-param plumbing
        fixed_params = dict(fixed_params or {})
        unknown = set(fixed_params) - set(tree.param_names)
        if unknown:
            raise ValueError(
                f"fixed_params keys not in tree.param_names: "
                f"{sorted(unknown)}; expected subset of {tree.param_names}"
            )
        self._fixed_by_index: dict[int, float] = {
            tree.param_names.index(k): float(v)
            for k, v in fixed_params.items()
        }
        self._free_indices: list[int] = [
            i for i in range(tree.n_params) if i not in self._fixed_by_index
        ]
        if not self._free_indices:
            raise ValueError(
                "All parameters are fixed; nothing to optimise. "
                "Loosen fixed_params or remove it entirely."
            )

        # ---- initial full-vector x_0 (with fixed entries enforced)
        # np.array, so the in-place fixed-param writes never touch a caller's array.
        x0 = (
            tree.param_vector if initial_rates is None
            else np.array(initial_rates, dtype=float)
        )
        if x0.shape != (tree.n_params,):
            raise ValueError(
                f"initial_rates must be length-{tree.n_params}; "
                f"got shape {tuple(x0.shape)}"
            )
        for i, v in self._fixed_by_index.items():
            x0[i] = v
        self._x0 = x0

        self._engine = Likelihood(
            model,
            base_composition=self.base_composition,
        )
        # With prior=None the root prior is the empirical pi where counts were
        # supplied, else the model's stationary vector.
        from ancestree.models import _PiModel

        if (isinstance(model, _PiModel)
                and getattr(self.base_composition, "_pi_is_ascertained", False)):
            self._log.warning(
                "The base composition was tallied over variant sites alone, "
                "where a base's share is weighted by how readily it mutates, "
                "so it is not the stationary distribution %s reads it as. "
                "Fitted divergences shift by roughly a tenth. Supply a "
                "whole-region composition (BaseComposition.from_fasta or "
                "from_counts) for a base-frequency model.",
                type(model).__name__,
            )
        from ancestree.priors import IngroupWeight, KingmanIngroupWeight
        if ingroup_weight is None and self.tree.ingroup_samples:
            ingroup_weight = KingmanIngroupWeight(self.tree.ingroup_samples)
        if ingroup_weight is not None and not isinstance(
                ingroup_weight, IngroupWeight):
            raise TypeError(
                f"ingroup_weight must be an IngroupWeight or None; got "
                f"{type(ingroup_weight).__name__}. A StationaryPrior belongs "
                f"in prior=, where it is applied once at the readout; for a "
                f"flat ingroup term pass NoIngroupWeight()."
            )

        self.ingroup_weight = ingroup_weight
        from ancestree.priors import StationaryPrior
        if prior is not None and not isinstance(prior, StationaryPrior):
            raise TypeError(
                f"prior must be a StationaryPrior or None; got "
                f"{type(prior).__name__}. An IngroupWeight belongs in "
                f"ingroup_weight=, where it is seeded once on the ingroup "
                f"MRCA; passing it as prior= applies it again at the readout."
            )
        self.prior = (
            prior if prior is not None
            else StationaryPrior(
                model, self.base_composition if supplied_counts else None)
        )

        # ---- model-internal free params (e.g. K2/HKY's kappa, GTR's rates)
        self._model_free_param_names: tuple[str, ...] = tuple(model.free_params)
        self._model_free_param_bounds: list[tuple[float, float]] = [
            model.free_params[name][1] for name in self._model_free_param_names
        ]
        self._model_free_param_initials: list[float] = [
            model.free_params[name][0] for name in self._model_free_param_names
        ]

        self._params_mle: np.ndarray | None = None
        self._model_params_mle: dict[str, float] | None = None
        self._log_likelihood_mle: float | None = None

        # ---- config-multiplicity ML fit: weighted Felsenstein over the unique
        # (projected AFS, outgroup pattern) configs, monomorphic sites included
        # as boundary configs.
        subsample_size = resolved_subsample_size
        # Merge the monomorphic counts into the projection and materialise
        # the config sites.
        if not self.tree.ingroup_samples:
            self._refuse_empty_ingroup(
                len(self.tree.ingroup_samples) + len(self.tree.outgroup_samples),
                regardless=True)
        self._fit_configs, self._fit_weights = (
            self.base_composition._finalize_configs(
                proj_weights,
                mono_counts,
                self.tree.ingroup_samples,
                self.tree.outgroup_samples,
                subsample_size,
            )
        )
        if self._fit_configs:
            if self.ingroup_weight is not None:
                self._fit_log_prior = self.ingroup_weight.log_probs(
                    self._fit_configs)
            else:
                # The slot carries the ingroup term alone. The objective adds
                # the root prior itself.
                self._fit_log_prior = np.zeros(
                    (len(self._fit_configs), model.n_states),
                )
        else:
            self._fit_log_prior = None

        # The readout node. The fit itself maximises at the ingroup MRCA.
        self.focal = FocalNode.parse(focal)

        self._warn_on_redundant_outgroups(outgroup_similarity_threshold)

    def _ingroup_haplotype_count(self) -> int:
        """Panel tips the ingroup ids resolve to.

        An id names a tip or the individual a tip belongs to, the resolution
        :meth:`Site.count_alleles() <ancestree.sites.Site.count_alleles>`
        applies, so a diploid panel named per individual carries twice as many
        ingroup haplotypes as ids. Falls back to the id count where the panel
        is not reachable.

        :return: Number of ingroup haplotypes in the panel.
        """

        names = list(self.tree.ingroup_samples or [])
        if not names:
            return 0
        source = self._stream_source
        panel: "list[str] | None" = None
        if source is not None and hasattr(source, "samples"):
            panel = list(source.samples())
        else:
            sites = self.sites or (
                source if isinstance(source, (list, tuple)) else None)
            if sites:
                panel = list(sites[0].tip_alleles)
        if not panel:
            return len(names)
        wanted = set(names)
        return sum(
            1 for sid in panel
            if _named(sid, wanted)
        )

    def _resolve_subsample_size(
        self, ingroup_weight, subsample_size: "int | None",
    ) -> int:
        """Resolve the canonical AFS ``subsample_size`` for the config fit.

        Explicit user argument wins. Otherwise inherit from a supplied
        AFS-aware ingroup weight. Otherwise default to ``min(n_ingroup, 11)``.
        Raises on a sub-2 explicit value (a subsample of 1 carries no
        frequency information) or a mismatch against the supplied weight.

        :param ingroup_weight: The user-supplied weight (or ``None``).
        :param subsample_size: The user's explicit ``subsample_size`` (or
            ``None`` to auto-resolve).
        :return: The resolved sub-sample size.
        """
        n_in_resolve = self._ingroup_haplotype_count()
        if subsample_size is not None:
            resolved = int(subsample_size)
            if resolved < 2:
                raise ValueError(
                    "subsample_size must be >= 2 (a subsample of 1 carries no "
                    f"frequency information); got {resolved}"
                )
            if n_in_resolve and resolved > n_in_resolve:
                raise ValueError(
                    f"subsample_size must be in [2, {n_in_resolve}] for an "
                    f"ingroup of {n_in_resolve} haplotype(s); got {resolved}"
                )
        else:
            weight_ss = getattr(ingroup_weight, "subsample_size", None)
            if weight_ss is not None:
                resolved = int(weight_ss)
            else:
                resolved = min(max(n_in_resolve, 1), 11)
                if resolved < 2:
                    self._log.warning(
                        "The subsample size resolved to 1 from an ingroup of %s "
                        "haplotype(s): the frequency weight carries no "
                        "information at this size and reduces to its "
                        "monomorphic case.", n_in_resolve,
                    )
        if (
            ingroup_weight is not None
            and getattr(ingroup_weight, "subsample_size", None) is not None
            and ingroup_weight.subsample_size != resolved
        ):
            raise ValueError(
                f"subsample_size mismatch: FixedTreeInference resolved to "
                f"{resolved} but the supplied {type(ingroup_weight).__name__} "
                f"has subsample_size={ingroup_weight.subsample_size}. Pass a consistent "
                f"value (either drop subsample_size on the inference and let it "
                f"inherit from the weight, or construct the weight without "
                f"subsample_size so it inherits from the inference)."
            )
        return resolved

    # ---------------------------------------------------------- redundancy check

    _MIN_SITES_FOR_SIMILARITY_CHECK = 50
    # Stream prefix sampled for the redundant-outgroup check in streaming mode.
    _REDUNDANT_OUTGROUP_SAMPLE_CAP = 50_000

    def _warn_if_branch_rates_at_bounds(self) -> None:
        """Post-fit warning when a fitted branch rate is within 1% of a bound.

        A ``K_i`` at the lower bound means the outgroup tip carries no
        divergence in the fitted topology: an identifiability ridge, redundant
        outgroups, or too few polymorphic sites.
        """
        if self._params_mle is None or len(self._free_indices) == 0:
            return
        lo, hi = self.bounds
        # Multiplicative tolerance.
        lo_threshold = max(lo * 1.01, lo + 1e-9)
        hi_threshold = hi * 0.99
        for free_i, tree_i in enumerate(self._free_indices):
            name = self.tree.param_names[tree_i]
            value = float(self._params_mle[tree_i])
            at_lo = value <= lo_threshold
            at_hi = value >= hi_threshold
            if at_lo or at_hi:
                edge = "lower" if at_lo else "upper"
                self._log.warning(
                    "FixedTreeInference: fitted branch rate %r = %.3e is at "
                    "the %s bound (%.1e, %.1e). The fit may be at a degenerate "
                    "optimum (common causes: 2-outgroup identifiability ridge, "
                    "redundant / collinear outgroups, too few polymorphic "
                    "sites to anchor this branch). Inspect "
                    "``params_mle``/``outgroup_divergence_mle`` and "
                    "consider increasing ``n_starts``, adding more data, "
                    "or fixing this branch via ``fixed_params``.",
                    name, value, edge, lo, hi,
                )

    def _warn_on_redundant_outgroups(self, threshold: float) -> None:
        """Pre-fit warning for outgroup pairs with low pairwise divergence.

        Near-identical outgroups, typically both haplotypes of one diploid
        individual, drive their branch rates to the lower bound. In streaming
        mode the check runs on a bounded prefix of the stream.
        """
        if threshold <= 0:
            return
        if self.sites:
            sample: "Iterable[Site]" = self.sites
        elif self._streaming and self._stream_source is not None:
            import itertools
            sample = itertools.islice(
                self._stream_source, self._REDUNDANT_OUTGROUP_SAMPLE_CAP,
            )
        else:
            return
        outgroups = self.tree.outgroup_samples
        pairs = [(i, j) for i in range(len(outgroups))
                 for j in range(i + 1, len(outgroups))]
        if not pairs:
            return
        n_obs_per_pair = [0] * len(pairs)
        n_diff_per_pair = [0] * len(pairs)
        n_sites = 0
        for site in sample:
            n_sites += 1
            observed = [site.tip_alleles.get(o) for o in outgroups]
            for k, (i, j) in enumerate(pairs):
                ai, bj = observed[i], observed[j]
                if ai is None or bj is None:
                    continue
                n_obs_per_pair[k] += 1
                if ai != bj:
                    n_diff_per_pair[k] += 1
        if n_sites < self._MIN_SITES_FOR_SIMILARITY_CHECK:
            return
        for k, (i, j) in enumerate(pairs):
            a, b = outgroups[i], outgroups[j]
            n_obs, n_diff = n_obs_per_pair[k], n_diff_per_pair[k]
            if n_obs == 0:
                continue
            frac = n_diff / n_obs
            if frac < threshold:
                self._log.warning(
                    "Outgroups %r and %r differ at only %d/%d (%.2f%%) of "
                    "jointly-observed polymorphic sites, below the %.1f%% "
                    "outgroup_similarity_threshold. They carry "
                    "near-redundant information; the fit will "
                    "collapse their independent branch rates "
                    "(K_i pegged at the lower bound). Common cause: "
                    "passing both haplotypes of one diploid "
                    "individual as separate outgroup samples.",
                    a, b, n_diff, n_obs, 100 * frac, 100 * threshold,
                )

    # ------------------------------------------------------------------- fit

    def _ensure_fitted(self) -> None:
        """Fit branch rates on demand so :meth:`summary` / :meth:`grade` work
        on an unfitted instance (matching their documented behaviour)."""
        if self.fit_required and self._params_mle is None:
            self.fit()

    @staticmethod
    def _check_filter_labelled(sample_filter, ingroup_samples,
                               outgroup_samples) -> None:
        """Refuse a ``sample_filter`` sample in neither the ingroup nor the
        outgroups.

        :param sample_filter: The panel restriction, or ``None``.
        :param ingroup_samples: Named ingroup ids, or ``None``.
        :param outgroup_samples: Named outgroup ids, or ``None``.
        :raises ValueError: If a filtered sample is in neither list.
        """
        if not sample_filter:
            return
        # Filter entries name individuals, the lists may name haplotypes.
        named = _by_individual((*(ingroup_samples or ()),
                                *(outgroup_samples or ())))
        _unlabelled(sample_filter, named, chosen_by="sample_filter")

    @staticmethod
    def _check_samples_present(outgroup_samples, sites, source=None,
                               ingroup_samples=None) -> None:
        """Reject named ids that appear on no site.

        An outgroup matching no tip fits every branch rate to its lower bound
        and emits uniform posteriors. An ingroup id matching no tip drops out
        of the ingroup, which moves both the fitted branch rates and the MAP
        calls. An outgroup id names a ladder tip and matches a tip id exactly.
        An ingroup id enters through
        :meth:`Site.count_alleles() <ancestree.sites.Site.count_alleles>` and
        matches a tip id or the individual a tip belongs to. An ingroup
        absent in full is the outgroup-only mode, where the ladder carries no
        ingroup tip and the ingroup enters as a flat vector, so only a
        partially matched ingroup is refused.

        :param outgroup_samples: Named outgroup ids, or ``None``.
        :param sites: The materialised sites, empty when the source streams.
        :param source: The streaming source, whose panel stands in for the
            sites. The check is skipped when neither is available.
        :param ingroup_samples: Named ingroup ids, or ``None``.
        :raises ValueError: If a named sample is absent from every site.
        """
        out_named = list(outgroup_samples or [])
        in_named = list(ingroup_samples or [])
        if not out_named and not in_named:
            return
        want = set(out_named) | set(in_named)
        want_in = set(in_named)
        seen: set = set()
        present: set = set()
        if not sites:
            # Streaming: the panel comes from the source.
            panel = getattr(source, "samples", None)
            if not callable(panel):
                return
            present = set(panel())
            seen = (want & present) | (
                want_in & {_individual_of(s) for s in present})
            missing_out = [s for s in out_named if s not in seen]
            missing_in = FixedTreeInference._unmatched_ingroup(in_named, seen)
            if not missing_out and not missing_in:
                return
            raise ValueError(
                f"{len(missing_out) + len(missing_in)} named sample(s) are "
                f"absent from the source panel: "
                f"{FixedTreeInference._name_missing(missing_out, missing_in)}."
            )
        individuals: set = set()
        for site in sites:
            ids = set(site.tip_alleles)
            new_ids = ids - present
            if new_ids:
                present |= new_ids
                individuals |= {_individual_of(s) for s in new_ids}
            seen |= want & ids
            seen |= want_in & individuals
            if seen >= want:  # every named sample located. Stop early
                break
        missing_out = [s for s in out_named if s not in seen]
        missing_in = FixedTreeInference._unmatched_ingroup(in_named, seen)
        if missing_out or missing_in:
            example = sorted(present)[:4]
            raise ValueError(
                f"{len(missing_out) + len(missing_in)} named sample(s) appear "
                f"on no site: "
                f"{FixedTreeInference._name_missing(missing_out, missing_in)}. "
                f"Sites carry ids like {example}. A VCF source splits each "
                f"diploid into '<name>_h0' / '<name>_h1', so the individual's "
                f"own name matches no outgroup tip."
            )

    @staticmethod
    def _unmatched_ingroup(named: Sequence[str], seen: set) -> list[str]:
        """Ingroup ids the panel does not carry, where some of them it does.

        :param named: The named ingroup ids.
        :param seen: The ids located in the panel.
        :return: The unmatched ids, empty where the panel carries none of
            them, which is the outgroup-only mode.
        """
        missing = [s for s in named if s not in seen]
        return [] if len(missing) == len(named) else missing

    @staticmethod
    def _name_missing(missing_out: Sequence[str],
                      missing_in: Sequence[str]) -> str:
        """Render the unmatched ids of both lists for the guard's message.

        :param missing_out: Unmatched outgroup ids.
        :param missing_in: Unmatched ingroup ids.
        :return: One clause per non-empty list, each capped at six ids.
        """
        parts = []
        for kind, missing in (("outgroup", missing_out), ("ingroup", missing_in)):
            if missing:
                parts.append(
                    f"{kind} {list(missing[:6])}"
                    f"{' ...' if len(missing) > 6 else ''}")
        return "; ".join(parts)

    def fit(self) -> dict[str, float]:
        r"""Run L-BFGS-B to fit the tree's branch rates by ML.

        No-op in the no-outgroup mode (``outgroup_samples=[]``): the
        outgroup-ladder tree has no parameters and the per-site posterior
        is the normalised prior. Returns an empty dict in that case.

        Mutates ``self.tree`` in place to the MLE rates. The objective sums
        over the unique (ingroup sub-AFS, outgroup-pattern) configs ``c``,

        .. math::

            \log L(\boldsymbol K) = \sum_c w_c \log\!\Big[
                \sum_s \pi_{c,s}\,
                P(\text{config}_c \mid \text{root}=s; \boldsymbol K) \Big],

        where ``w_c`` is the config's multiplicity (hypergeometric mass
        accumulated from the polymorphic sites plus the per-base
        monomorphic counts of
        :paramref:`base_composition <ancestree.inference.FixedTreeInference.base_composition>`)
        and ``π_{c,s}`` the
        per-config root prior: the configured prior
        (:class:`~ancestree.priors.KingmanIngroupWeight` by default, or
        the stationary vector when none is set), overridden by a delta at
        the observed allele on monomorphic-sub-AFS configs.

        :return: Mapping from :attr:`OutgroupLadderTree.param_names <ancestree.trees.OutgroupLadderTree.param_names>` entry to the
            MLE rate. Empty dict in the no-outgroup mode.
        :raises RuntimeError: If every start fails to converge. A partial
            failure is tolerated and the best converged start wins.
        """
        if self._no_outgroup_mode:
            return {}
        if self._streaming:
            self._log.info(
                "Streaming fit: pass 1 (config histogram) complete; fitting "
                "%d branch rate(s) by ML, then a second streaming pass emits "
                "posteriors.", len(self.tree.param_names),
            )
        starts = self._generate_starts()
        results = (
            self._run_parallel(starts)
            if Settings._use_parallel(self.parallelize) and len(starts) > 1
            else [self._run_one(x0) for x0 in starts]
        )
        converged = [(i, r) for i, r in enumerate(results) if r.success]
        if not converged:
            msgs = "; ".join(f"start {i}: {r.message!r}" for i, r in enumerate(results))
            raise RuntimeError(
                f"All {len(results)} L-BFGS-B runs failed to converge: {msgs}"
            )
        best_idx, best = min(converged, key=lambda pair: pair[1].fun)

        full, model_values = self._apply_free_vector(best.x)
        self._params_mle = full.copy()
        if model_values:
            self._model_params_mle = model_values
            self.model.warn_if_bounds_hit()
        self._warn_if_branch_rates_at_bounds()
        self._log_likelihood_mle = float(-best.fun)

        # Stage 2: an adaptive weight fits its bins at the MLE tree, in batches.
        from ancestree.priors import AdaptiveIngroupWeight
        if isinstance(self.ingroup_weight, AdaptiveIngroupWeight):
            self.ingroup_weight.begin_fit()
            for batch in self._iter_site_batches():
                self.ingroup_weight.accumulate(
                    batch, self._engine.log_likelihoods(self.tree, batch),
                )
            self.ingroup_weight.end_fit()

        self._warn_if_outgroup_divergences_non_monotone()
        self._log_fit_summary(best.nit, results=results, best_idx=best_idx)
        return self.params_mle  # type: ignore[return-value]

    def _warn_if_outgroup_divergences_non_monotone(self) -> None:
        """Warn when the fitted ``I → O_k`` divergences do not increase along
        the supplied outgroup order. For ``n_outgroups >= 3`` the ladder is
        asymmetric and a mis-ordering biases the fit."""
        n = self.tree.n_outgroups
        if n < 2:
            return
        divs = self.tree.outgroup_divergence()
        names = self.tree.outgroup_samples
        # A sub-permille inversion is optimiser noise, not a mis-ordering.
        out_of_order = [
            (names[i], float(divs[i]), names[i + 1], float(divs[i + 1]))
            for i in range(n - 1)
            if divs[i] > divs[i + 1] and not np.isclose(
                divs[i], divs[i + 1], rtol=1e-3, atol=0.0,
            )
        ]
        if not out_of_order:
            return
        pair_str = ", ".join(
            f"{a}({da:.3e}) > {b}({db:.3e})" for a, da, b, db in out_of_order
        )
        self._log.warning(
            "Fitted outgroup divergences are not monotonically increasing "
            "along the supplied order; out-of-order pairs: %s. "
            "For n_outgroups >= 3 this biases the fit (the ladder is "
            "asymmetric). To get the closest-first order from the data, "
            "use OutgroupLadderTree.with_outgroup_order_from_sites(ingroup, "
            "outgroup, sites) for a pre-ordered tree, or "
            "OutgroupLadderTree.order_outgroups_by_divergence(sites, "
            "ingroup, outgroup) for just the sorted list.",
            pair_str,
        )

    # ---------------------------------------------------------- fit internals

    def _run_one(self, x0_free: np.ndarray):
        """Run a single L-BFGS-B optimisation from ``x0_free``."""
        bounds = [self.bounds] * len(self._free_indices) + self._model_free_param_bounds
        return minimize(
            self._neg_log_likelihood,
            x0_free,
            method="L-BFGS-B",
            bounds=bounds,
        )

    def _run_parallel(self, starts: list[np.ndarray]) -> list:
        """Dispatch independent starts across worker processes.

        Uses the ``fork`` start method so an unguarded module-level script does
        not re-import its ``__main__``, falling back to a serial fit where
        ``fork`` is unavailable (Windows) or where the resolved numba threading
        layer would not survive a fork.
        """
        import multiprocessing as mp

        n_workers = Settings._resolve_n_workers(
            self.n_workers or min(len(starts), os.cpu_count() or 1)
        )
        if n_workers <= 1:
            return [self._run_one(x0) for x0 in starts]
        if not Settings._fork_pool_ok(self._log, "starts", len(starts)):
            return [self._run_one(x0) for x0 in starts]
        if "fork" not in mp.get_all_start_methods():
            self._log.warning(
                "FixedTreeInference(parallelize=True): the 'fork' start method "
                "is unavailable on this platform (likely Windows); running the "
                "%d starts serially. Spawn would re-pickle the whole inference "
                "per start and re-import the caller's __main__.", len(starts),
            )
            return [self._run_one(x0) for x0 in starts]
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
                initializer=Settings._cap_worker_threads,
                initargs=(n_workers,)) as exec_:
            return list(exec_.map(self._run_one, starts))

    def _generate_starts(self) -> list[np.ndarray]:
        """Initial points for the L-BFGS-B variable vector (free params only).

        First start is always the user's ``initial_rates`` projection onto
        the free axes (so a deterministic single-start fit is reproducible
        as the ``n_starts=1`` case). Additional starts are log-uniformly
        sampled from ``bounds`` using ``self.seed``.
        """
        x0_free = np.array(
            [self._x0[i] for i in self._free_indices]
            + list(self._model_free_param_initials)
        )
        if self.n_starts == 1:
            return [x0_free]
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds
        log_lo, log_hi = np.log(max(lo, 1e-12)), np.log(hi)
        starts = [x0_free]
        for _ in range(self.n_starts - 1):
            log_x = rng.uniform(log_lo, log_hi, size=len(self._free_indices))
            x_new = np.exp(log_x).tolist()
            for m_lo, m_hi in self._model_free_param_bounds:
                m_log = rng.uniform(np.log(m_lo), np.log(m_hi))
                x_new.append(float(np.exp(m_log)))
            starts.append(np.array(x_new))
        return starts

    def _log_fit_summary(
        self, nit: int, *, results: list, best_idx: int,
    ) -> None:
        """Emit MLE branch rates + derived divergences at INFO level.

        Visible only when the caller configures the ``ancestree.inference``
        logger (e.g., ``logging.basicConfig(level=logging.INFO)``). No
        side effects on the silent default.
        """
        if not self._log.isEnabledFor(logging.INFO):
            return
        if len(results) == 1:
            self._log.info(
                "L-BFGS-B converged in %d iterations "
                "(marginal log-likelihood %.4f)",
                nit, self._log_likelihood_mle,
            )
        else:
            log_Ls = [-r.fun for r in results if r.success]
            self._log.info(
                "%d/%d L-BFGS-B starts converged; "
                "best is start %d (log L %.4f, range %.4f); winning fit "
                "took %d iterations",
                len(log_Ls), len(results), best_idx,
                self._log_likelihood_mle,
                max(log_Ls) - min(log_Ls) if len(log_Ls) > 1 else 0.0,
                nit,
            )
        rates_str = ", ".join(
            f"{name}={rate:.4e}{'*' if i in self._fixed_by_index else ''}"
            for i, (name, rate) in enumerate(
                zip(self.tree.param_names, self._params_mle),
            )
        )
        suffix = (
            " (* = fixed)" if self._fixed_by_index else ""
        )
        self._log.info("MLE branch rates: %s%s", rates_str, suffix)
        if self._model_params_mle:
            mle_str = ", ".join(
                f"{n}={v:.4e}" for n, v in self._model_params_mle.items()
            )
            self._log.info("MLE model params: %s", mle_str)
        divs = self.tree.outgroup_divergence()
        divs_str = ", ".join(
            f"{og}={d:.4e}"
            for og, d in zip(self.tree.outgroup_samples, divs)
        )
        self._log.info("Ingroup MRCA → outgroup divergences: %s", divs_str)

    def _apply_free_vector(
        self, x_free: np.ndarray,
    ) -> "tuple[np.ndarray, dict[str, float]]":
        """Write the L-BFGS-B vector into the tree and the model.

        :param x_free: The free branch rates followed by any model-internal
            free parameters. Fixed branch entries come from ``self._x0``.
        :return: The full branch-rate vector set on the tree, and the
            model-internal parameter values set on the model.
        """
        full = self._x0.copy()
        n_tree_free = len(self._free_indices)
        for free_i, tree_i in enumerate(self._free_indices):
            full[tree_i] = x_free[free_i]
        self.tree.set_params(full)
        model_values = {
            name: float(x_free[n_tree_free + j])
            for j, name in enumerate(self._model_free_param_names)
        }
        if model_values:
            self.model.set_free_params(model_values)
        return full, model_values

    def _neg_log_likelihood(self, x_free: np.ndarray) -> float:
        """Set tree rates from the L-BFGS-B vector and return ``-log L``.

        ``x_free`` holds the free branch rates and any trailing model
        parameters; fixed entries come from ``self._x0``. The objective is the
        multiplicity-weighted sum over the config histogram of
        ``logsumexp(log_L + log_prior, axis=1)``, with the per-config ingroup
        term precomputed at construction.
        """
        self._apply_free_vector(x_free)
        if not self._fit_configs:
            return 0.0
        log_L = self._engine.log_likelihoods(self.tree, self._fit_configs)  # (C, S)
        # The root prior of _infer_site_batch: site-independent, so one row.
        log_prior_root = self.prior.log_probs(self._fit_configs[:1])[0]
        log_post = log_L + self._fit_log_prior + log_prior_root
        per_config_marginal = logsumexp(log_post, axis=1)  # (C,)
        total = float((self._fit_weights * per_config_marginal).sum())
        return -total

    # ----------------------------------------------------------------- infer

    def infer(self) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(site, Posterior)`` for each polymorphic site.

        Posteriors use whatever branch rates the tree carries at call time
        and the configured root prior
        (:class:`~ancestree.priors.KingmanIngroupWeight` by default).
        Run :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>`
        first. Otherwise the tree's branch rates are whatever
        :paramref:`initial_rates <ancestree.inference.FixedTreeInference.initial_rates>`
        was at construction.

        :return: Iterator of ``(Site, Posterior)`` pairs in input order.
        :raises RuntimeError: If
            :paramref:`fit_required <ancestree.inference.FixedTreeInference.fit_required>`
            is set and
            :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>`
            was never called.
        """
        yield from self._with_baseline_check(self._infer_inner())
        self._log_uniform_fallback_summary()

    def _baseline_outgroup_samples(self) -> tuple[str, ...]:
        if not self._no_outgroup_mode and self.tree is not None:
            return tuple(self.tree.outgroup_samples)
        return ()

    def _baseline_ingroup_samples(self) -> tuple[str, ...]:
        if not self._no_outgroup_mode and self.tree is not None:
            return tuple(self.tree.ingroup_samples)
        return tuple(getattr(self, "_ingroup_samples_no_out", ()))

    def _infer_inner(self) -> Iterator[tuple[Site, Posterior]]:
        """Core per-site posterior generator (no baseline-check side-effects)."""
        if not self._quiet:
            if self._streaming:
                self._log.info(
                    "Inferring the ancestral allele on the fitted outgroup "
                    "ladder, streaming the sites")
            else:
                self._log.info(
                    "Inferring the ancestral allele at %s site(s) on the "
                    "fitted outgroup ladder", f"{len(self.sites):,}")
        if self._no_outgroup_mode:
            # The posterior is the normalised ingroup weight. A monoallelic
            # ingroup takes all mass on its observed allele.
            if not self.sites:
                return
            self._count_ingroup_monomorphic(self.sites)
            self._count_unrepresentable(self.sites)
            log_pr = self.ingroup_weight.log_probs(self.sites)
            post = self._normalise_log_post(
                log_pr, n_states=self.model.n_states)
            states = self.model.states
            state_index = STATE_INDEX
            ingroup = self._ingroup_samples_no_out
            for k, site in enumerate(self.sites):
                alleles = {site.tip_alleles.get(s) for s in ingroup}
                alleles.discard(None)
                if len(alleles) == 1:
                    j = state_index.get(next(iter(alleles)))
                    if j is not None:
                        post[k] = 0.0
                        post[k, j] = 1.0
                yield site, Posterior(alleles=states, values=post[k])
            return
        if self._params_mle is None and self.fit_required:
            raise RuntimeError(
                "FixedTreeInference.infer() called before fit(); "
                "branch rates have not been ML-fitted. Call .fit() first, "
                "or construct with fit_required=False to use the tree's "
                "current branch rates (e.g. supplied via "
                "OutgroupLadderTree.from_newick or tree.set_params)."
            )
        gen = self._infer_stream()
        if self.progress:
            gen = tqdm(
                gen, total=None if self._streaming else len(self.sites),
                desc="FixedTreeInference", unit=" sites",
                disable=Settings.disable_pbar,
            )
        yield from gen

    def _iter_site_batches(
        self, batch_size: int = 4096,
    ) -> "Iterator[list[Site]]":
        """Yield batches of sites: one batch of the materialised list, or the
        streamed source in ``batch_size`` chunks. Shared by the streamed
        inference pass and the adaptive prior's stage-2 fit, so both modes keep
        peak memory flat in genome length when streaming."""
        if self._streaming:
            batch: list[Site] = []
            for site in self._stream_source:
                batch.append(site)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        elif self.sites:
            yield list(self.sites)

    def _infer_stream(self) -> Iterator[tuple[Site, Posterior]]:
        """Score the sites one batch at a time.

        Streaming re-iterates the source so peak memory stays flat in genome
        length. Materialised yields the held site list as a single batch.
        """
        for batch in self._iter_site_batches():
            yield from self._infer_site_batch(batch)

    def _infer_site_batch(
        self, batch: "Sequence[Site]",
    ) -> Iterator[tuple[Site, Posterior]]:
        """Score one in-memory batch: Felsenstein on the focal view of the
        ladder, plus the prior, normalised to posteriors."""
        # The ingroup enters the kernel as one likelihood vector on its MRCA,
        # so it propagates to whatever node the run reports at. The prior
        # applies at that node.
        seeds = None
        counts = self._shared_ingroup_counts(batch)
        if self.ingroup_weight is not None:
            log_w = (
                self.ingroup_weight.log_probs(batch) if counts is None
                else self.ingroup_weight.log_probs_from_counts(counts)
            )
            with np.errstate(under="ignore"):
                seeds = {self.tree.ingroup_mrca: np.exp(log_w)}
        self._count_ingroup_monomorphic(batch, counts=counts)
        self._count_unrepresentable(batch)
        log_prior = self.prior.log_probs(batch)
        log_L = self._engine.log_likelihoods(
            self._focal_tree, batch, node_seeds=seeds,
        )
        posteriors = self._normalise_log_post(
            log_L + log_prior, n_states=self.model.n_states,
        )
        for site, p in zip(batch, posteriors):
            yield site, Posterior(alleles=self.model.states, values=p)

    def _shared_ingroup_counts(
        self, batch: "Sequence[Site]",
    ) -> "list[Counter] | None":
        """Ingroup allele counts for one batch, tallied once for the weight
        and the monomorphic diagnostic.

        :param batch: The sites about to be scored.
        :return: One tally per site in batch order, or ``None`` where the
            weight does not take pre-tallied counts or counts over a different
            sample set, so each consumer tallies its own.
        """
        from ancestree.priors import KingmanIngroupWeight

        weight = self.ingroup_weight
        if not isinstance(weight, KingmanIngroupWeight):
            return None
        ingroup = list(weight.ingroup_samples)
        if tuple(ingroup) != self._baseline_ingroup_samples():
            return None
        return [site.count_alleles(ingroup) for site in batch]

    @property
    def _focal_tree(self) -> "Tree | None":
        """The ladder as seen from the focal node, built on demand since
        :meth:`fit` sets the branch rates in place.

        :return: The ladder itself when reading at its own root, else a
            re-rooted view of it.
        """
        if self.tree is None:
            return None
        return self.tree.at_focal(self.focal)

    _MODE = "fixed-tree"

    def _base_composition_kind(self) -> str:
        """How the root prior's base composition was supplied.

        ``n_target_sites`` yields a composition that is uniform over the
        states, which is not the empirical composition of any sequence.

        :return: ``"empirical"``, ``"uniform"`` or ``"none"``.
        """
        import numpy as np

        bc = self.base_composition
        if bc is None or (bc.n_total <= 0 and not bc._pi_is_ascertained):
            return "none"
        pi = np.asarray(bc.pi, dtype=float)
        return "uniform" if np.allclose(pi, 1.0 / pi.size) else "empirical"

    def _provenance_parameters(self) -> dict[str, object]:
        """Fixed-tree run parameters: model, prior, outgroups,
        and (after :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>`) the ML-fitted branch rates + model params."""
        params: dict[str, object] = {
            "model": self._model_name(self.model),
            "ingroup_weight": (type(self.ingroup_weight).__name__
                               if self.ingroup_weight is not None else None),
            "prior": type(self.prior).__name__ if self.prior is not None else None,
            "ingroup_samples": ([] if self.tree is None
                                else list(self.tree.ingroup_samples)),
            "outgroup_samples": ([] if self.tree is None
                                 else list(self.tree.outgroup_samples)),
            "branch_rates_fitted": self._params_mle is not None,
            "n_outgroups": (
                0 if self._no_outgroup_mode or self.tree is None
                else int(self.tree.n_outgroups)
            ),
            "n_target_sites": (
                int(self.base_composition.n_total)
                if self.base_composition is not None else None),
            "base_composition": self._base_composition_kind(),
            "n_starts": int(self.n_starts),
            "seed": int(self.seed),
            "subsample_size": (int(self.subsample_size)
                               if self.subsample_size is not None
                               else None),
            **self.focal.provenance(),
        }
        fitted = self.params_mle
        if fitted is not None:
            params["fitted"] = {k: float(v) for k, v in fitted.items()}
        return params

    # ---------------------------------------------------- post-fit accessors

    @property
    def params_mle(self) -> dict[str, float] | None:
        """MLE parameters as ``{name: value}``, or ``None`` before fit.

        Includes the tree's branch-rate params plus any model-internal
        free params (e.g. K2's ``kappa`` when ``K2(fit_kappa=True)``).

        :return: Mapping or ``None``.
        """
        if self._params_mle is None:
            return None
        out = dict(zip(self.tree.param_names, self._params_mle.tolist()))
        if self._model_params_mle:
            out.update(self._model_params_mle)
        return out

    @property
    def model_params_mle(self) -> dict[str, float] | None:
        """MLE values for the model's free parameters, or ``None`` before fit
        (an empty dict if the model has no free params).

        :return: ``{name: value}`` from :meth:`SubstitutionModel.set_free_params() <ancestree.models.SubstitutionModel.set_free_params>`,
            or ``None``.
        """
        return self._model_params_mle

    @property
    def log_likelihood_mle(self) -> float | None:
        """Marginal log-likelihood at the MLE, or ``None`` before fit.

        :return: Scalar log-likelihood or ``None``.
        """
        return self._log_likelihood_mle

    @property
    def outgroup_divergence_mle(self) -> np.ndarray | None:
        """``I → O_k`` path lengths at the MLE, or ``None`` before fit.

        Matches fastDFE's
        ``MaximumLikelihoodAncestralAnnotation.get_outgroup_divergence``
        convention so msprime simulations with known ``μ·t`` can be
        compared directly.

        :return: Length-:attr:`OutgroupLadderTree.n_outgroups <ancestree.trees.OutgroupLadderTree.n_outgroups>`
            array, or ``None``.
        """
        if self._params_mle is None:
            return None
        return self.tree.outgroup_divergence()



class MajorityOutgroupInference(Inference):
    """Rule-based baseline: the ancestral allele is the majority allele among the outgroup tips.

    .. warning::

       A comparator only: an unweighted, fixed-confidence vote with no model
       and no calibrated uncertainty. The model-based classes invoke it under
       ``baseline_check=True``, and persistent disagreement with their
       posterior indicates model misspecification.

    For each site, the canonical (A/C/G/T) tip alleles at
    :paramref:`outgroup_samples <ancestree.inference.MajorityOutgroupInference.outgroup_samples>`
    are counted and the most frequent is the MAP allele, ties going to the
    allele carried by the earliest outgroup in the supplied list. The
    posterior places :paramref:`confidence <ancestree.inference.MajorityOutgroupInference.confidence>`
    on it and spreads the rest uniformly over the other alleles. With no
    outgroup data the posterior is uniform. With no outgroups at all and
    ``ingroup_samples`` given, the most frequent ingroup allele is taken as
    ancestral, ties spreading a uniform posterior over the tied alleles.

    :param sites: Polymorphic :class:`~ancestree.sites.Site` records.
    :param outgroup_samples: Outgroup sample ids, closest first. The order
        matters only for tie-breaking.
    :param ingroup_samples: Optional ingroup sample ids, the panel of the
        no-outgroup fallback. Ignored when an outgroup is present.
    :param model: Optional :class:`~ancestree.models.SubstitutionModel`
        fixing the state alphabet. Defaults to :class:`~ancestree.models.JC69`.
    :param confidence: Probability placed on the MAP allele. Default ``0.95``.
    :param for_comparison_only: Suppress the standalone-use warning, as the
        model-based classes do when they invoke this baseline.

    :raises ValueError: If ``confidence`` is outside ``(0, 1]``.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"confidence": self.confidence}

    def __init__(
        self,
        sites: "Sequence[Site]",
        outgroup_samples: Sequence[str],
        *,
        ingroup_samples: Sequence[str] | None = None,
        model: "SubstitutionModel | None" = None,
        confidence: float = 0.95,
        for_comparison_only: bool = False,
    ) -> None:
        if not 0.0 < confidence <= 1.0:
            raise ValueError(
                f"confidence must be in (0, 1]; got {confidence}"
            )
        if not for_comparison_only:
            self._log.warning(
                "MajorityOutgroupInference is a simple "
                "rule-based baseline (majority allele among outgroup "
                "tips) intended as a post-hoc consistency check on the real "
                "model-based inferences (ARGBasedInference / "
                "FixedTreeInference). Used standalone it discards the "
                "tree-topology and substitution-model information that "
                "Ancestree is built to exploit. Pass "
                "for_comparison_only=True if you are intentionally "
                "invoking it as a comparator (this suppresses the "
                "warning); otherwise use ARGBasedInference / "
                "FixedTreeInference for production inference."
            )
        self.sites: list[Site] = list(sites)
        self.outgroup_samples: tuple[str, ...] = tuple(outgroup_samples)
        # Ingroup panel for the no-outgroup major-allele fallback only.
        self.ingroup_samples: tuple[str, ...] = (
            tuple(ingroup_samples) if ingroup_samples is not None else ()
        )
        self.model = model if model is not None else JC69()
        self.confidence = float(confidence)
        self.ingroup_weight = None
        self.prior = None

    def _used_samples(self) -> "frozenset[str]":
        """The sample columns a restricted output keeps, the named samples and
        their individuals."""
        return _by_individual((*self.ingroup_samples, *self.outgroup_samples))

    def infer(self) -> Iterator[tuple[Site, Posterior]]:
        """Yield ``(Site, Posterior)`` per input site under the majority rule.

        See the class docstring for the exact rule and tie-breaking.

        :return: Iterator of ``(Site, Posterior)`` in input order.
        """
        states = self.model.states
        n_states = len(states)
        state_index = STATE_INDEX
        low = (1.0 - self.confidence) / max(1, n_states - 1)
        uniform = np.full(n_states, 1.0 / n_states)
        no_outgroup_fallback = (
            not self.outgroup_samples and bool(self.ingroup_samples)
        )
        if no_outgroup_fallback:
            yield from self._infer_ingroup_major(
                states, n_states, state_index,
            )
            return
        for site in self.sites:
            counts = site.count_alleles(self.outgroup_samples)
            if not counts:
                yield site, Posterior(alleles=states, values=uniform.copy())
                continue
            # Ties go to the allele of the earliest outgroup in the list.
            max_count = max(counts.values())
            tied = {a for a, c in counts.items() if c == max_count}
            if len(tied) == 1:
                map_allele = next(iter(tied))
            else:
                map_allele = None
                for og in self.outgroup_samples:
                    a = next(((v or "").upper()
                              for sid, v in site.tip_alleles.items()
                              if (sid == og or _individual_of(sid) == og)
                              and (v or "").upper() in tied), None)
                    if a is not None:
                        map_allele = a
                        break
                # Falls back to the alphabetically first tied allele.
                if map_allele is None:  # pragma: no cover
                    map_allele = sorted(tied)[0]
            j = state_index[map_allele]
            values = np.full(n_states, low)
            values[j] = self.confidence
            yield site, Posterior(alleles=states, values=values)

    def _infer_ingroup_major(
        self,
        states: Sequence[str],
        n_states: int,
        state_index: dict[str, int],
    ) -> Iterator[tuple["Site", Posterior]]:
        """No-outgroup fallback: the major ingroup allele is ancestral.

        The most frequent canonical allele among :attr:`ingroup_samples` takes
        ``confidence``, and ``k`` tied alleles take ``1/k`` each.

        :param states: Ordered state alphabet of the emitted posteriors.
        :param n_states: Alphabet size ``S``.
        :param state_index: Map allele → index into ``states``.
        :return: Iterator of ``(Site, Posterior)`` in input order.
        """
        uniform = np.full(n_states, 1.0 / n_states)
        for site in self.sites:
            counts = site.count_alleles(self.ingroup_samples)
            if not counts:
                yield site, Posterior(alleles=states, values=uniform.copy())
                continue
            max_count = max(counts.values())
            tied = [a for a, c in counts.items() if c == max_count]
            if len(tied) == 1:
                # Unique major allele: full confidence on it.
                values = np.full(n_states, (1.0 - self.confidence)
                                 / max(1, n_states - 1))
                values[state_index[tied[0]]] = self.confidence
            else:
                # Tie: uniform mass over the tied alleles, zero elsewhere.
                values = np.zeros(n_states)
                for a in tied:
                    values[state_index[a]] = 1.0 / len(tied)
            yield site, Posterior(alleles=states, values=values)
