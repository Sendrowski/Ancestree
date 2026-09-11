"""Shared logic for the robustness heatmap (methods × n_out × scenarios).

Pure, side-effect-free module imported by the two entry-point scripts:

- ``score_robustness_cell.py`` — computes one (scenario, method, n_out)
  cell via :func:`compute_cell` and writes its JSON. One snakemake job
  per cell, so ``snakemake -jN`` runs them concurrently.
- ``make_robustness_heatmap.py`` — the report: loads the per-cell JSONs
  (:func:`load_cell_summaries`) and renders the two heatmap PDFs + the
  runtime side-table via :func:`build_report`.

The heatmap scripts never compute cells themselves — the per-cell compute
is done (in parallel, one job per cell) and cached by the snakemake per-cell
rules. The render step only loads those cached summaries and plots.

The heatmap has one row per method, four ``n_out`` columns per scenario,
four site-class columns (``ISM bi`` / ``≥2 mut`` / ``i=n−1`` / ``fixed``),
and four scenario columns (three msprime sims + one SLiM aggregation).
Each cell is mean P(true) or MAP accuracy over its site class.
"""
from __future__ import annotations

import bisect
import json
import logging
import os
import sys
import time
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tskit

# matplotlib / seaborn are plotting-only and imported lazily inside
# render_heatmap, so the cell-compute path (incl. tsinfer) can run in the
# lean ``tsinfer-ancestree`` env, which has tsinfer but no matplotlib.

from ancestree import (
    AdaptiveIngroupWeight,
    ARGBasedInference,
    FixedTreeInference,
    JC69,
    KingmanIngroupWeight,
    LocalTreeInference,
    MajorityOutgroupInference,
    NoIngroupWeight,
    STATES,
    Site,
)
from collections.abc import Sequence

from ancestree.focal import FocalNode
from ancestree.sites import SiteSource

logging.getLogger("ancestree").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# Repo root resolved from this file's location (workflow/scripts/ -> repo),
# so the scenario data/report paths work on any machine (laptop or cluster),
# not just one hardcoded checkout.
REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results" / "data"
REPORTS = REPO / "results" / "reports"

# Per-site record: (sfs_bin, klass, map_hit, p_true, n_mutations, sum_sq), with
# an optional 7th field carrying the posterior mass on the observed alleles
# (the fixed-tree and panel-ARG paths emit it, the scorers read the first six).
# ``sum_sq`` = Σ_i posterior_i² — together with p_true it gives the Brier
# score per site: Brier = (1 − p_true)² + Σ_{i≠t} p_i² = 1 − 2·p_true + sum_sq.
PerSite = list[tuple[int, str, int, float, int, float]]

# Focal node per site class. A site polymorphic within the ingroup is polarised
# about the ingroup MRCA. A site the ingroup has fixed carries no information
# about that node and is called against the panel root instead. Each cell
# therefore runs once per class and scores only its own sites.
#: Trees drawn from the pairwise HMM's posterior that the local-tree row
#: marginalises each window's call over. ``None`` is the plug-in point estimate.
LOCAL_TREE_ENSEMBLE_MEMBERS = int(
    os.environ.get("ANCESTREE_ENSEMBLE_MEMBERS", "128"))

#: Cache key for the only panel-root table there is (the full panel).
_FULL_PANEL = "__full__"

FOCAL_POLY = "ingroup_mrca"
FOCAL_MONO = "panel_root"

# Each class is reported at the node split_truth grades it against:
# ingroup-polymorphic sites at the ingroup MRCA, ingroup-fixed sites at the
# root of the full outgroup panel. Both nodes are fixed by construction.
FOCAL_PASSES = ((FOCAL_POLY, True), (FOCAL_MONO, False))

#: Report modules loaded by :func:`_report_module`, keyed by file stem.
_REPORT_MODULES: dict = {}


def _report_module(name: str):
    """Load a sibling report script for its definitions, without running its
    sweep.

    The focal scans are snakemake ``script:`` targets, so each ends in a bare
    ``main()`` call that computes a multi-hour sweep on import. The trailing
    call is dropped so the classes can be driven from here. The module then
    takes its ``except NameError`` branch and fills in defaults the caller
    overrides. Loaded modules are memoised, so the numba kernels compile once
    per process.

    :param name: File stem of the script under ``workflow/scripts``.
    :return: The executed module.
    """
    if name in _REPORT_MODULES:
        return _REPORT_MODULES[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    source = path.read_text()
    marker = "\nmain()\n"
    if not source.endswith(marker):
        raise ValueError(f"{path} does not end in a bare main() call")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    _REPORT_MODULES[name] = module
    exec(compile(source[:-len(marker)] + "\n", str(path), "exec"),
         module.__dict__)
    return module


def _sumsq(post) -> float:
    """Σ_i posterior_i² for a Posterior (the off-diagonal Brier term needs it)."""
    v = np.asarray(post.values, dtype=float)
    return float(np.dot(v, v))


def _date_tsinfer(its, mu_true: float):
    """Date a tsinfer ARG into generations with the real tsinfer→tsdate pipeline.

    tsinfer emits topology with uncalibrated node times, so
    ``tsdate.preprocess_ts`` + ``tsdate.date`` (variational gamma) run at the
    true mutation rate to put absolute times in generations, and the kernel then
    runs at ``mu_true``. A chunk too sparse for tsinfer to resolve any topology leaves
    every tree a star of unlinked roots, on which tsdate asserts "Use fewer
    rescaling intervals". That case falls back to a single-parameter ML-rate
    rescale (``mu_eff = #sites / total tree length``, kernel at ``mu_eff``) and
    logs the substitution, so the cell still computes. Any other tsdate failure
    propagates.

    :return: ``(ts, mu)`` --- the dated ts at ``mu_true`` on success, or the
        original ts at ``mu_eff`` on the fallback.
    :raises AssertionError: On any tsdate assertion other than the rescaling
        one, which would otherwise be silently rescaled away.
    """
    import tsdate
    try:
        dated = tsdate.date(tsdate.preprocess_ts(its), mutation_rate=mu_true)
        return dated, mu_true
    except AssertionError as exc:
        if "rescaling intervals" not in str(exc):
            raise
        tbl = sum(t.total_branch_length * t.span for t in its.trees())
        mu_eff = (its.num_sites / tbl) if tbl > 0 else mu_true
        logging.getLogger("ancestree.robustness").warning(
            "tsdate could not date this ARG (%s): %d sites over %d trees left "
            "every tree a star of %d unlinked roots, so the kernel runs at "
            "mu_eff=%.3g rather than mu=%.3g",
            exc, its.num_sites, its.num_trees,
            max(t.num_roots for t in its.trees()), mu_eff, mu_true)
        return its, mu_eff


def _evenly_spread_outgroups(names: list[str], n_out: int) -> list[str]:
    """Pick ``n_out`` outgroups evenly spread by index (hence split depth)
    across ``names``, preserving the depth span as ``n_out`` varies instead of
    dropping the deepest outgroups first. Shallowest and deepest are included
    for ``n_out >= 2``; ``n_out == 1`` uses the shallowest.
    """
    if n_out <= 0:
        return []
    if n_out >= len(names):
        return list(names)
    idx = sorted({int(i) for i in np.linspace(0, len(names) - 1, n_out).round()})
    return [names[i] for i in idx]


# --------------------------------------------------------- lazy site source
class _ScenarioSiteSource(SiteSource):
    """Re-iterable, lazy :class:`SiteSource` over a scenario's tree sequence.

    Replaces the eager ``list[Site]`` the fixed-tree / local-tree paths used
    to build — one ``tip_alleles`` dict per kept sample per polymorphic site.
    On the dense ``cpg_hypermut`` scenario that list is ~7M sites × tens of
    samples and OOM-killed before inference even started. Yielding one
    :class:`Site` at a time keeps peak memory flat in the site count, so
    ``FixedTreeInference(stream=True)`` and ``LocalTreeInference(chunk_size=)``
    can bound their footprint (config histogram / one segment, respectively).

    Re-iterable by contract: each ``__iter__`` re-walks ``ts.variants()`` and
    (for the unphased path) re-seeds the switch-error RNG and phase state, so
    the two streaming passes (fit then infer, or per-chunk builds) observe
    byte-identical sites — matching the old single-materialised-list behaviour.

    ``keep_mode`` selects which sites to emit, mirroring the filters in the
    former list builders: ``"all"`` (the fixed-tree path — every variant, as
    ``sfs_bin`` was populated for all of them) or ``"truth_aa"`` (the
    local-tree path — only sites whose ancestral state is a base in STATES).
    The filter is intrinsic (read off the variant) rather than a closure over
    a big position dict, so the source stays small and picklable for the
    parallel fit.
    """

    def __init__(self, ts, keep_names, sample_id_to_node, name_for_node,
                 *, keep_mode="all", dip_pairs=None, switch_rate=0.0, seed=12345):
        self._ts = ts
        self._keep_names = list(keep_names)
        self._keep_mode = keep_mode
        self._dip_pairs = list(dip_pairs or [])
        self._switch_rate = float(switch_rate)
        self._seed = int(seed)
        keep_node_set = {sample_id_to_node[s] for s in self._keep_names}
        # (genotype-column index, sample name) for the kept haplotypes only,
        # in ts.samples() order so the index aligns with v.genotypes.
        self._kept_cols = [
            (j, name_for_node[int(node)])
            for j, node in enumerate(ts.samples())
            if int(node) in keep_node_set
        ]

    def samples(self) -> list[str]:
        return list(self._keep_names)

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        orient = [0] * len(self._dip_pairs)
        scramble = self._switch_rate > 0.0 and bool(self._dip_pairs)
        truth_only = self._keep_mode == "truth_aa"
        for v in self._ts.variants():
            if truth_only and v.site.ancestral_state not in STATES:
                continue
            pos = int(v.site.position)
            ta = {name: v.alleles[v.genotypes[j]] for j, name in self._kept_cols}
            if scramble:
                # per-pair phase orientation carried along the genome. Flips
                # with prob switch_rate at each het site (a switch-error
                # process), leaving runs of correct phase between switches.
                for pi, (a, b) in enumerate(self._dip_pairs):
                    if ta.get(a) == ta.get(b):
                        continue  # homozygous: phase irrelevant
                    if rng.random() < self._switch_rate:
                        orient[pi] ^= 1
                    if orient[pi]:
                        ta[a], ta[b] = ta[b], ta[a]
            yield Site(chrom="1", pos=pos, alleles=tuple(v.alleles),
                       tip_alleles=ta)


def _truth_at_node(site, tree, nodes) -> str:
    """The simulated state at the MRCA of ``nodes``.

    A run that reports at some node must be graded there too, otherwise the
    estimate and the truth are two different quantities and the difference
    reads as inference error. Walks the mutations above that node and takes the
    most recent one.

    :param site: The :class:`tskit.Site` carrying the mutations.
    :param tree: The local tree covering it.
    :param nodes: Node ids whose MRCA the state is read at.
    :return: The true state at that node.
    """
    import tskit

    mrca = tree.mrca(*[int(n) for n in nodes])
    if mrca == tskit.NULL:
        return site.ancestral_state
    state, best = site.ancestral_state, float("inf")
    for mutation in site.mutations:
        node = int(mutation.node)
        if node != mrca and not tree.is_descendant(mrca, node):
            continue
        # A mutation's own time places it on the branch above its node. Two
        # mutations on one branch share that node and are ordered only by it.
        time = float(mutation.time)
        if np.isnan(time):
            time = tree.time(node)
        if time < best:  # most recent mutation on the path wins
            best, state = time, mutation.derived_state
    return state


# ----------------------------------------------------------------- scenario
@dataclass
class Scenario:
    """Per-scenario state: truth-side sim handle + derived per-site maps."""
    key: str
    label: str
    trees_path: Path
    meta_path: Path
    # Per-site mutation rate used by the ARG / local-tree kernels for this
    # scenario (fixed-tree fits its own branch rates, so it ignores this).
    # Elevated for the CpG-hypermutability scenario.
    mu: float = 1.25e-8
    #: Per-site recombination rate the ARG tools are run at. Must match the
    #: rate the scenario was simulated under: SINGER refuses a panel whose
    #: coalescent depth and recombination rate are mutually implausible.
    rec_rate: float = 1e-8

    # Populated by .load():
    ts: tskit.TreeSequence = None
    n_ingroup: int = 0
    ingroup_names: list[str] = field(default_factory=list)
    all_outgroup_names: list[str] = field(default_factory=list)
    sample_id_to_node: dict[str, int] = field(default_factory=dict)
    name_for_node: dict[int, str] = field(default_factory=dict)
    sfs_bin: dict[int, int] = field(default_factory=dict)
    truth_by_pos: dict[int, str] = field(default_factory=dict)
    n_alleles_per_pos: dict[int, int] = field(default_factory=dict)
    mut_count_by_pos: dict[int, int] = field(default_factory=dict)

    # Optional cross-method coverage mask (installed by the intersection-scoring
    # path via set_keep_intervals). When set, the scoring loops record only
    # sites whose position falls in one of these sorted, non-overlapping
    # ``[lo, hi)`` intervals; None (default) scores every site.
    keep_intervals: "list[tuple[int, int]] | None" = None
    _keep_lo: list[int] = field(default_factory=list)
    _keep_hi: list[int] = field(default_factory=list)

    def set_keep_intervals(self, intervals) -> None:
        """Install (or clear, with ``None``) the coverage mask consulted by
        :meth:`_pos_kept`. Intervals are merged into sorted, non-overlapping
        ``[lo, hi)`` form so membership is an O(log n) bisection."""
        if intervals is None:
            self.keep_intervals = None
            self._keep_lo, self._keep_hi = [], []
            return
        merged = _merge_intervals(intervals)
        self.keep_intervals = merged
        self._keep_lo = [lo for lo, _ in merged]
        self._keep_hi = [hi for _, hi in merged]

    def _pos_kept(self, pos: int) -> bool:
        """Whether ``pos`` passes the coverage mask (always True with no mask)."""
        if self.keep_intervals is None:
            return True
        i = bisect.bisect_right(self._keep_lo, pos) - 1
        return i >= 0 and pos < self._keep_hi[i]

    def _scored_span(self) -> int:
        """Length of the region actually scored --- the coverage mask when one
        is installed, else the whole chunk. The fixed-tree fit needs it as the
        denominator for the monomorphic sites it never sees."""
        if self.keep_intervals is None:
            return int(self.ts.sequence_length)
        return int(sum(hi - lo for lo, hi in self.keep_intervals))

    def _masked_ts(self, ts):
        """``ts`` cut down to the installed coverage mask.

        The methods that carry their own genealogy (true-ARG, local-tree) would
        otherwise walk the whole chunk and discard the out-of-mask sites at
        scoring time, so their wall time would cover a span the panel-ARG tools
        never inferred on. ``simplify=False`` keeps the mutationless sites the
        ingroup-monomorphic column is scored on, which the simplifying form
        drops.
        """
        if self.keep_intervals is None:
            return ts
        return ts.keep_intervals([[lo, hi] for lo, hi in self.keep_intervals],
                                 simplify=False)

    def load(self) -> None:
        meta = json.loads(self.meta_path.read_text())
        self.ingroup_names = list(meta["ingroup_names"])
        self.n_ingroup = len(self.ingroup_names)
        outgroups_by_pop = dict(meta["outgroups_by_pop"])
        ordered_pops = sorted(outgroups_by_pop.keys(),
                              key=lambda s: int(s.split("_")[1]))
        self.all_outgroup_names = [n for p in ordered_pops
                                   for n in outgroups_by_pop[p]]
        self.ts = tskit.load(str(self.trees_path))
        self.name_for_node = {int(ind.nodes[0]): f"tsk_{ind.id}"
                              for ind in self.ts.individuals()}
        self.sample_id_to_node = {f"tsk_{ind.id}": int(ind.nodes[0])
                                  for ind in self.ts.individuals()}
        self.truth_at_ingroup_mrca: dict[int, str] = {}
        self._panel_root_truth: dict[tuple[str, ...], dict[int, str]] = {}
        ingroup_nodes = [self.sample_id_to_node[s] for s in self.ingroup_names]
        for tree in self.ts.trees():
            for site in tree.sites():
                self.truth_at_ingroup_mrca[int(site.position)] = _truth_at_node(
                    site, tree, ingroup_nodes,
                )
        for v in self.ts.variants():
            anc = v.site.ancestral_state
            pos = int(v.site.position)
            derived = sum(1 for j in range(self.n_ingroup)
                          if v.alleles[v.genotypes[j]] != anc)
            self.sfs_bin[pos] = derived
            self.truth_by_pos[pos] = anc
            observed = {a for g, a in zip(v.genotypes, v.alleles)
                        if a not in ("", None)}
            self.n_alleles_per_pos[pos] = len(observed)
            self.mut_count_by_pos[pos] = len(v.site.mutations)

    def ingroup_is_polymorphic(self, pos: int) -> bool:
        """Whether more than one allele segregates within the ingroup.

        Decides which of the two runs scores the site: the ingroup MRCA is the
        node an ingroup polymorphism is polarised about, while a site the
        ingroup has fixed is only informative about the deeper panel root.
        """
        b = self.sfs_bin.get(pos)
        return b is not None and 0 < b < self.n_ingroup

    def truth_table(self, focal: str) -> dict[int, str]:
        """Simulated states at whichever node ``focal`` names.

        Neither node moves with the outgroup selection. :data:`FOCAL_POLY` is
        the ingroup MRCA, built once by :meth:`Scenario.load`; :data:`FOCAL_MONO`
        is the root of the FULL outgroup panel, not of the subset a column
        happens to score.

        There is no ``panel`` argument: every row is graded against the same
        node, so a caller cannot select its own.

        :param focal: Either :data:`FOCAL_POLY` or :data:`FOCAL_MONO`.
        :return: Position to true state at that node.
        """
        if focal == FOCAL_POLY:
            return self.truth_at_ingroup_mrca
        cached = self._panel_root_truth.get(_FULL_PANEL)
        if cached is None:
            nodes = [self.sample_id_to_node[s]
                     for s in self.ingroup_names + list(self.all_outgroup_names)]
            cached = {
                int(site.position): _truth_at_node(site, tree, nodes)
                for tree in self.ts.trees() for site in tree.sites()
            }
            self._panel_root_truth[_FULL_PANEL] = cached
        return cached

    def split_truth(self, panel: "Sequence[str]") -> dict[int, str]:
        """One truth table graded at the node each site class is asked about.

        Every method is scored against the same question per site, whether or
        not it can report at that node. A method confined to the panel root
        therefore pays for the divergence branch on ingroup polymorphism, which
        is a property of the method and not of the grading.

        BOTH classes are graded at a node that does not move with ``n_out``.
        Polymorphic sites are asked about the ingroup MRCA, which is fixed by
        construction. Non-polymorphic (ingroup-fixed) sites are asked about the
        root of the FULL outgroup panel, not the root of whatever subset this
        column happens to use --- otherwise the question deepens with every
        outgroup added and the ``n_out`` axis conflates how much the outgroups
        buy with how hard the question became. At ``n_out = 0`` the moving
        version was degenerate: the panel root collapsed onto the ingroup MRCA
        and the truth was just the ingroup's own fixed allele, so every method
        scored perfectly on a question with no content.

        A column with few outgroups is therefore genuinely penalised for
        knowing less about a deep node. That is the method's property, exactly
        as the paragraph above argues for the polymorphic class.

        :param panel: Sample ids in the panel being scored. Used only for the
            positions it covers. The grading node is the full-panel root.
        :return: Position to true state at the node that site class names.
        """
        poly = self.truth_at_ingroup_mrca
        root = self.truth_table(FOCAL_MONO)
        return {pos: (poly.get(pos) if self.ingroup_is_polymorphic(pos)
                      else state)
                for pos, state in root.items()}

    def classify(self, pos: int) -> str:
        n_a = self.n_alleles_per_pos.get(pos, 0)
        b = self.sfs_bin.get(pos)
        if n_a > 2:
            return "polyallelic"
        if b is None or b == 0 or b == self.n_ingroup:
            return "biallelic_mono"
        return "biallelic_poly"

    def _select_outgroups(self, n_out: int) -> list[str]:
        """Outgroups evenly spread across split depths
        (see :func:`_evenly_spread_outgroups`)."""
        return _evenly_spread_outgroups(self.all_outgroup_names, n_out)

    # ---------- live inference helpers (per method × n_out) ---------------
    def run_arg(self, n_out: int, *, mu: float | None = None) -> tuple[PerSite, float]:
        """ARGBasedInference (Felsenstein kernel) on the true local trees.

        :param mu: Per-site mutation rate override. ``None`` uses the
            scenario's ``self.mu``. A tiny value (e.g. ``1e-12``) drives the
            kernel into its maximum-parsimony / minimum-change limit.
        """
        sel = self._select_outgroups(n_out)
        keep_names = self.ingroup_names + sel
        keep_node_ids = [self.sample_id_to_node[s] for s in keep_names]
        ts_sub = self._masked_ts(
            self.ts.simplify(samples=keep_node_ids, filter_sites=False))
        sample_map = {s: i for i, s in enumerate(keep_names)}
        out: PerSite = []
        elapsed = 0.0
        for focal, polymorphic in FOCAL_PASSES:
            inf = ARGBasedInference(
                ts_sub, JC69(), mu=(self.mu if mu is None else mu),
                sample_map=sample_map, focal=focal,
                ingroup_samples=self.ingroup_names, outgroup_samples=sel,
                progress=False,
            )
            truth = self.truth_table(focal)
            t0 = time.perf_counter()
            for site, post in inf.infer():
                pos = int(site.pos)
                if not self._pos_kept(pos):
                    continue
                if (polymorphic is not None
                        and self.ingroup_is_polymorphic(pos) is not polymorphic):
                    continue
                t = truth.get(pos)
                if t is None or t not in STATES:
                    continue
                b = self.sfs_bin.get(pos, -1)
                out.append((b, self.classify(pos),
                            int(post.map_allele == t),
                            float(post[t]),
                            self.mut_count_by_pos.get(pos, 0),
                            _sumsq(post)))
            elapsed += time.perf_counter() - t0
        return out, elapsed

    # ---- appendix baselines (parsimony limit + hard / trivial rules) -----
    def run_arg_mu0(self, n_out: int) -> tuple[PerSite, float]:
        """Maximum-parsimony limit of the ARG kernel: ARGBasedInference on
        the true local trees with ``mu -> 0`` (a tiny ``1e-12``).

        As ``mu -> 0`` the Felsenstein integral over branch lengths is
        dominated by the minimum-change reconstruction, so the posterior
        collapses onto the parsimony (minimum-mutation) root state(s) — the
        empirical demonstration of the Methods claim that the kernel
        reduces to maximum parsimony. Brier-proper 6-tuples (same record
        shape as :meth:`run_arg`).
        """
        return self.run_arg(n_out, mu=1e-12)

    def run_majority_outgroup(self, n_out: int) -> tuple[PerSite, float]:
        """``MajorityOutgroupInference``: majority allele among the first
        ``n_out`` outgroup tips. At ``n_out=0`` it falls back to the major
        (most frequent) *ingroup* allele, so every n_out gets a score.
        Brier-proper 6-tuples via :meth:`run_majority`.
        """
        return self.run_majority(n_out)

    def run_local_tree_unphased(self, n_out: int) -> tuple[PerSite, float]:
        """Local-tree mode on the simulated phase, as every heatmap row is.

        The phasing switch rate is varied on its own in the local-tree window
        benchmark (:mod:`infer_local_tree_window_cell`), which is where that
        sensitivity is reported; the heatmaps hold phase fixed so their rows
        differ only in the genealogy each method infers.
        """
        return self._run_local_tree(n_out)

    def _run_local_tree(
        self, n_out: int, *, switch_rate: float = 0.0, seed: int = 12345,
    ) -> tuple[PerSite, float]:
        """Shared local-tree path. Mirrors run_arg's panel (ingroup + first
        n_out outgroups), consuming only observed genotypes (no truth AA).

        ``switch_rate`` injects phasing error: ingroup haplotypes are paired
        into pseudo-diploids and a per-pair orientation flips with that
        probability at each het site (a switch-error process — 0 = perfect
        phase, 0.5 = fully random). Outgroups (single haplotypes) untouched.
        """
        sel = self._select_outgroups(n_out)
        keep_names = self.ingroup_names + sel
        # consecutive ingroup haplotype pairs → pseudo-diploids
        ing = self.ingroup_names
        dip_pairs = [(ing[2 * k], ing[2 * k + 1]) for k in range(len(ing) // 2)]
        # Lazy, re-iterable source instead of a materialised list of every
        # truth-AA site (OOM on dense scenarios). The switch-error scramble is
        # applied per-site inside the source and re-seeded each __iter__, so it
        # is deterministic across the chunked builder's repeated passes. Keying
        # tip_alleles by sample name (not post-simplify tsk id) keeps a
        # non-contiguous outgroup selection (e.g. even-spread [0,4,9]) from
        # reading as all-missing and collapsing those outgroups' TMRCA.
        source = _ScenarioSiteSource(
            self._masked_ts(self.ts), keep_names, self.sample_id_to_node,
            self.name_for_node, keep_mode="truth_aa",
            dip_pairs=dip_pairs, switch_rate=switch_rate, seed=seed,
        )
        # chunk_size → the segmented build path: one contiguous span at a time,
        # so peak memory tracks the segment (here 2 Mb) not the whole genome.
        # The library's own window ("8snp") and recurrence ("full") defaults are
        # what this row wants, so they are not restated: one tree per ~8-SNP
        # window with the default ~4-SNP block, chosen by scanning 3 to 30 SNP
        # scored at the ingroup MRCA, where accuracy plateaus at 8 SNP and
        # below. Block_size moves the score by under 4% and not consistently.
        # member_chunk is left at the library default: it batches members
        # through the packing and transition-matrix build, so lowering it to
        # dodge the whole-number-of-chunks requirement would cost real time for
        # a case no configured ensemble size hits.
        out: PerSite = []
        elapsed = 0.0
        for focal, polymorphic in FOCAL_PASSES:
            inf = LocalTreeInference(
                source, JC69(), mu=self.mu, rec_rate=self.rec_rate,
                sample_names=keep_names, focal=focal,
                ingroup_samples=self.ingroup_names, outgroup_samples=sel,
                chunk_size="2mb", progress=False,
                n_ensemble=LOCAL_TREE_ENSEMBLE_MEMBERS,
            )
            truth = self.truth_table(focal)
            t0 = time.perf_counter()
            for site, post in inf.infer():
                pos = int(site.pos)
                if not self._pos_kept(pos):
                    continue
                if (polymorphic is not None
                        and self.ingroup_is_polymorphic(pos) is not polymorphic):
                    continue
                t = truth.get(pos)
                if t is None or t not in STATES:
                    continue
                b = self.sfs_bin.get(pos, -1)
                out.append((b, self.classify(pos),
                            int(post.map_allele == t),
                            float(post[t]),
                            self.mut_count_by_pos.get(pos, 0),
                            _sumsq(post)))
            elapsed += time.perf_counter() - t0
        return out, elapsed

    def _fixed_tree_map_by_pos(self, n_out: int, *,
                               window: "tuple[int, int] | None" = None,
                               ) -> dict[int, str]:
        """Per-site MAP ancestral allele from fixed-tree inference (ingroup +
        ``n_out`` outgroups spread by depth). Used to *orient* the panel-ARG
        tools' input with a realistic (inferred) ancestral allele rather than the
        simulator truth.

        :param window: Restrict the fit and the returned calls to this span.
            Only in-window calls are ever consumed, since every tool runs on the
            window, and the fit is insensitive to the restriction: on
            ``cpg_hypermut`` chunk 0 the windowed and whole-chunk fits agree on
            the MAP allele at all 17,182 shared sites.
        """
        cache = None
        key = getattr(self, "key", None)
        ci = getattr(self, "chunk_idx", None)
        if key is not None and ci is not None:
            cache = ftmap_cache_path(key, ci, n_out, window)
        if cache is not None and _cache_fresh(cache, key, ci):
            d = json.loads(cache.read_text())
            return {int(p): a for p, a in d.items()}
        sel = self._select_outgroups(n_out)
        sites = self._sites_with_alleles(sel, window)
        span = (int(self.ts.sequence_length) if window is None
                else int(window[1]) - int(window[0]))
        prior = KingmanIngroupWeight(ingroup_samples=self.ingroup_names)
        inf = FixedTreeInference(
            sites, ingroup_samples=self.ingroup_names, outgroup_samples=sel,
            model=JC69(), n_target_sites=span,
            ingroup_weight=prior,
            parallelize=False, progress=False,
        )
        if sel:
            inf.fit()
        # One pass per site class, at the node that class's ancestral allele
        # is defined at: the ladder root for ingroup-polymorphic sites, the
        # panel root for ingroup-fixed ones, matching :meth:`split_truth`,
        # :data:`FOCAL_PASSES` and the two passes in
        # :meth:`_run_ft_with_prior`. At an ingroup-fixed site the ladder root
        # carries the ingroup's own allele, which is the derived one.
        runs = [(inf, None)]
        if sel:
            deep = FixedTreeInference(
                sites, tree=inf.tree, model=JC69(),
                n_target_sites=span,
                ingroup_weight=prior, fit_required=False,
                focal=FocalNode("ingroup_mrca", fraction=1.0),
                parallelize=False, progress=False,
            )
            runs = [(inf, True), (deep, False)]
        result: dict[int, str] = {}
        for run, polymorphic in runs:
            for site, post in run.infer():
                pos = int(site.pos)
                if (polymorphic is not None
                        and self.ingroup_is_polymorphic(pos) is not polymorphic):
                    continue
                result[pos] = post.map_allele
        if cache is not None:
            # Opportunistic write so a later shard reuses this fit. Atomic
            # (tmp + os.replace) since sibling shards (SINGER chains / other
            # modes) may compute the same key concurrently --- a reader must
            # never see a half-written file.
            try:
                tmp = cache.with_suffix(cache.suffix + f".tmp{os.getpid()}")
                tmp.write_text(json.dumps({str(p): a for p, a in result.items()}))
                os.replace(tmp, cache)
            except OSError as exc:
                # A cache that cannot be written means every later shard
                # recomputes; say so rather than degrading silently.
                print(f"WARNING: could not cache {cache}: {exc}", flush=True)
        return result

    def tsinfer_infer_chunk(self, n_out: int, *,
                            window: "tuple[int, int] | None" = None):
        """Infer + date one tsinfer ARG on a window --- inference only.

        tsinfer needs an ancestral allele to orient each site. It is given the
        fixed-tree MAP call rather than the simulator truth (oracle), a realistic
        two-stage pipeline: the fixed-tree pass orients, tsinfer builds the ARG
        on those calls, and the kernel re-infers the ancestral state. Sites whose fixed-tree MAP
        allele is not one of the two observed alleles cannot be oriented and are
        omitted from tsinfer's input (the kernel still scores them on the
        inferred topology at scoring time). tsinfer leaves its trees undated, so
        they are dated with the real tsinfer->tsdate pipeline
        (:func:`_date_tsinfer`) before scoring --- not a single-rate rescale.

        Split from scoring so the dated ARG can be persisted and re-scored. The
        fitted rate travels with it (returned here, written to the ARG meta).

        :param window: The ``(start, end)`` span to infer on, in absolute
            coordinates. Every method runs on the same budget window
            (:func:`method_windows`), so the wall times compare directly.
        :return: ``(draws, offset, runtime, mu)`` --- a one-element draw list,
            the window's start, wall time, and the fitted per-site rate the
            kernel must score with."""
        import time
        import tsinfer
        if n_out > len(self.all_outgroup_names):
            return [], 0, 0.0, self.mu
        sel = self._select_outgroups(n_out)
        keep_names = self.ingroup_names + sel
        ts_sub, offset = self._windowed_panel(n_out, window)
        # The orientation is the shared first stage of every panel-ARG method
        # and is cached across them, so it sits outside the timer here exactly
        # as it does for SINGER and Relate: each reported wall time is the
        # tool's own work on the window.
        map_by_pos = {p - offset: a for p, a in
                      self._fixed_tree_map_by_pos(n_out, window=window).items()}
        t0 = time.perf_counter()
        sd = tsinfer.SampleData(sequence_length=float(ts_sub.sequence_length))
        with sd:
            for v in ts_sub.variants():
                al = v.alleles
                if len(al) != 2 or any(a not in STATES for a in al if a is not None):
                    continue
                if len(set(v.genotypes)) < 2:
                    continue
                anc = map_by_pos.get(int(v.site.position))
                if anc not in al:
                    continue  # fixed-tree call not orientable on this site
                sd.add_site(v.site.position, v.genotypes, al,
                            ancestral_allele=al.index(anc))
        its = tsinfer.infer(sd, recombination_rate=1e-8).simplify()
        its, mu_use = _date_tsinfer(its, self.mu)
        return [its], offset, time.perf_counter() - t0, mu_use

    def run_tsinfer_arg(self, n_out: int, *,
                        window: "tuple[int, int] | None" = None,
                        ) -> tuple[PerSite, float]:
        """tsinfer ARG + Felsenstein kernel: :meth:`tsinfer_infer_chunk` then
        :meth:`score_arg_shard`. The workflow persists the dated ARG between the
        two. This convenience composition keeps it in memory. The panel's full
        in-window site set is scored on tsinfer's topology (the multiallelic /
        hypermutable sites its biallelic input drops are re-laid on at scoring
        time), matching the true-ARG reference's site set."""
        draws, _off, rt, mu_use = self.tsinfer_infer_chunk(n_out, window=window)
        if not draws:
            return [], rt
        post, obs = self.score_arg_shard(draws, n_out, window=window,
                                         mu=mu_use,
                                         mu_matches_time_units=True)
        return self.score_singer_posteriors(post, obs, n_out=n_out), rt

    def _sites_with_alleles(self, sel_outgroups: list[str],
                            window: "tuple[int, int] | None" = None,
                            ) -> list[Site]:
        keep_names = set(self.ingroup_names) | set(sel_outgroups)
        keep_nodes = {self.sample_id_to_node[s] for s in keep_names}
        out: list[Site] = []
        lo, hi = window if window is not None else (None, None)
        for v in self.ts.variants():
            pos = int(v.site.position)
            if self.sfs_bin.get(pos) is None:
                continue
            if lo is not None and not (lo <= pos < hi):
                continue
            tip_alleles: dict[str, str | None] = {}
            for j, node in enumerate(self.ts.samples()):
                if int(node) in keep_nodes:
                    sid = self.name_for_node[int(node)]
                    tip_alleles[sid] = v.alleles[v.genotypes[j]]
            out.append(Site(
                chrom="1", pos=pos, alleles=tuple(v.alleles),
                tip_alleles=tip_alleles,
            ))
        return out

    def _run_ft_with_prior(self, n_out: int, prior) -> tuple[PerSite, float]:
        sel = self._select_outgroups(n_out)
        if sel:
            # Outgroup-ladder mode → stream: the ML branch-rate fit needs only
            # a bounded config histogram (one streaming pass) and inference is
            # a second pass, so peak memory is flat in genome length instead of
            # holding ~7M Site dicts (the cpg_hypermut OOM).
            source = _ScenarioSiteSource(
                self._masked_ts(self.ts), self.ingroup_names + sel,
                self.sample_id_to_node, self.name_for_node,
                keep_mode="all",
            )
            # parallelize=True runs the n_starts L-BFGS restarts concurrently
            # (fork pool). The fit's optimisation dominates wall time on the
            # dense scenario at high outgroup count (cpg_hypermut anc_ft n10:
            # ~13.6k s of 10-D L-BFGS over a 2^n_out config table), and the
            # restarts are independent, so this is a near-linear speedup with
            # no change to the result (same starts, same selected optimum).
            # The job is given matching cpus via the rule's threads directive.
            # parallelize via a fork pool (the comment above). On spawn-default
            # platforms (macOS) the workers would re-exec this snakemake script;
            # RC_FT_PARALLELIZE=0 forces the serial path there (cheap at low
            # n_out). Defaults to the parallel fork pool, so the cluster is
            # unchanged.
            parallelize = os.environ.get("RC_FT_PARALLELIZE", "1") not in ("0", "false")
            inf = FixedTreeInference(
                source, ingroup_samples=self.ingroup_names, outgroup_samples=sel,
                model=JC69(), n_target_sites=self._scored_span(),
                ingroup_weight=prior,
                parallelize=parallelize, progress=False, stream=True,
            )
            t0 = time.perf_counter()
            inf.fit()
            fitted_tree = inf.tree
        else:
            # No-outgroup mode: the package can't stream it (the posterior
            # collapses to the normalised prior, no ladder to fit), so the
            # materialised list is unavoidable here — but it is the cheap
            # column (no kernel/fit) and covered by the rule's memory grant.
            sites = self._sites_with_alleles(
                sel, self.keep_intervals[0] if self.keep_intervals else None)
            inf = FixedTreeInference(
                sites, ingroup_samples=self.ingroup_names, outgroup_samples=sel,
                model=JC69(), n_target_sites=self._scored_span(),
                ingroup_weight=prior,
                parallelize=False, progress=False,
            )
            t0 = time.perf_counter()
            fitted_tree = inf.tree
        # One pass per site class, as in the genealogy modes. The second run
        # reuses the fitted ladder, so only the cheap inference pass repeats.
        runs = [(inf, None)]
        if sel:
            deep = FixedTreeInference(
                source, tree=fitted_tree, model=JC69(),
                n_target_sites=int(self.ts.sequence_length),
                ingroup_weight=prior, fit_required=False,
                focal=FocalNode("ingroup_mrca", fraction=1.0),
                parallelize=False, progress=False, stream=True,
            )
            runs = [(inf, True), (deep, False)]
        truth = self.split_truth(self.ingroup_names + sel)
        out: PerSite = []
        for run, polymorphic in runs:
            for site, post in run.infer():
                if (polymorphic is not None
                        and self.ingroup_is_polymorphic(site.pos) is not polymorphic):
                    continue
                t = truth.get(site.pos)
                if t is None:
                    continue
                b = self.sfs_bin.get(site.pos, -1)
                # 7th field: posterior mass on the observed alleles. For the
                # posterior-weighted SFS we renormalise each site's posterior over
                # its observed alleles, so a prior that places mass on unobserved
                # nucleotides (e.g. the stationary prior) does not leak that mass
                # into the mirror bin. Equals 1 for the count-based priors (Kingman
                # / adaptive) and once an outgroup concentrates the likelihood, so
                # the renormalisation is a no-op except for the uninformative case.
                out.append((b, self.classify(site.pos),
                            int(post.map_allele == t),
                            float(post[t]),
                            self.mut_count_by_pos.get(site.pos, 0),
                            _sumsq(post),
                            sum(float(post[a]) for a in site.alleles)))
        return out, time.perf_counter() - t0

    def run_ft(self, n_out: int) -> tuple[PerSite, float]:
        return self._run_ft_with_prior(
            n_out, KingmanIngroupWeight(ingroup_samples=self.ingroup_names))

    def run_ft_adaptive(self, n_out: int) -> tuple[PerSite, float]:
        """Fixed-tree mode with the adaptive (data-fit per-bin) prior at the
        default subsample size."""
        return self._run_ft_with_prior(
            n_out, AdaptiveIngroupWeight(ingroup_samples=self.ingroup_names))

    def run_ft_uniform(self, n_out: int) -> tuple[PerSite, float]:
        """Fixed-tree mode with a flat ingroup term --- the uninformative
        reference: the ingroup frequencies carry no weight, so under JC69 the
        call rests on the outgroups alone. Without an outgroup it cannot break
        the orientation symmetry, so it folds the spectrum."""
        return self._run_ft_with_prior(n_out, NoIngroupWeight())

    def run_majority(self, n_out: int) -> tuple[PerSite, float]:
        sel = self._select_outgroups(n_out)
        sites = self._sites_with_alleles(sel)
        # n_out=0 → no outgroup vote. Fall back to the major ingroup allele.
        inf = MajorityOutgroupInference(
            sites, outgroup_samples=sel,
            ingroup_samples=self.ingroup_names,
            for_comparison_only=True,
        )
        truth = self.split_truth(self.ingroup_names + sel)
        t0 = time.perf_counter()
        out: PerSite = []
        for site, post in inf.infer():
            # Honour the coverage mask, as every other scored method does.
            if not self._pos_kept(site.pos):
                continue
            t = truth.get(site.pos)
            if t is None:
                continue
            b = self.sfs_bin.get(site.pos, -1)
            out.append((b, self.classify(site.pos),
                        int(post.map_allele == t),
                        float(post[t]),
                        self.mut_count_by_pos.get(site.pos, 0),
                        _sumsq(post)))
        return out, time.perf_counter() - t0

    # ---- inference / scoring split (persisted-ARG panel workflow) --------
    # The panel-ARG methods (SINGER, Relate) separate the expensive, tool-
    # specific *inference* step (which persists the ARG) from the method-
    # agnostic *scoring* step (score_arg_shard), so a persisted ARG can be
    # re-scored without re-inference --- essential for SINGER, whose inference
    # is arm64-blocked, and good practice generally.

    def _windowed_panel(self, n_out: int,
                        window: "tuple[int, int] | None", *,
                        keep_all_sites: bool = False):
        """The panel (ingroup + ``n_out`` outgroups) restricted to ``window``,
        keeping every site (``filter_sites=False``) so scoring covers the
        multiallelic / monomorphic sites too. The window is trimmed to a 0-based
        frame (tools abort on a non-zero start); ``offset`` shifts positions back
        to absolute coordinates afterwards.

        ``keep_all_sites`` is the *scoring* path (:meth:`score_arg_shard`).
        ``keep_intervals`` drops the mutationless (panel-monomorphic) sites in
        tskit >=1.0, which would grade the panel-ARG methods on a strictly
        smaller site set than the true-ARG / tsinfer reference lays on its trees
        --- exactly the monomorphic / fixed columns the Balanced-Brier figure
        needs, and the very sites :func:`_singer.score_arg_topology` documents it
        scores. Scoring therefore skips the interval trim and lets the draw's
        own ``sequence_length`` window the site set (that function's
        ``0 <= pos < L`` filter). This keeps every in-window site, monomorphic ones included. It
        is valid only for the 0-based ``(0, end)`` windows :func:`singer_windows`
        / :func:`method_windows` emit (asserted). The *inference* path keeps
        trimming, since the tool must run on the windowed VCF.

        :return: ``(ts_sub, offset)``."""
        keep_names = self.ingroup_names + self._select_outgroups(n_out)
        keep_node_ids = [self.sample_id_to_node[s] for s in keep_names]
        ts_sub = self.ts.simplify(samples=keep_node_ids, filter_sites=False)
        offset = 0
        if window is not None:
            offset = int(window[0])
            if keep_all_sites:
                assert offset == 0, "keep_all_sites requires a 0-based window"
            else:
                ts_sub = ts_sub.keep_intervals([[offset, int(window[1])]]).trim()
        return ts_sub, offset

    #: Least fraction of a persisted ARG's own sites that must be sites of the
    #: current panel before it is trusted. A tool infers from the panel's
    #: genotypes, so a matching ARG scores ~1.0 here. One inferred from another
    #: realisation of the same scenario scores ~0.05, the density-matched
    #: coincidence rate. Set well below 1 so that a tool's own site filtering
    #: (or a re-simulated sim's handful of moved sites) does not trip it.
    ARG_PANEL_MATCH_MIN = 0.5

    def _check_arg_matches_panel(self, draws, ts_sub, label: str,
                                 window: "tuple[int, int] | None" = None) -> None:
        """Raise if a persisted ARG was inferred from different genotypes.

        An ARG from another realisation loads, carries a plausible topology and
        scores without error --- it simply answers about other data, so nothing
        downstream can notice. mtime triggers are the only other guard, and they
        are lost whenever the files are copied between machines.
        """
        # A superset ARG scores every overlap test at 1.0, so the span is
        # checked first: an ARG inferred over more sequence than the budget
        # window is a different amount of data, and its runtime is not
        # comparable with the other methods'. The tolerance covers the drift
        # between the tools' windows, not a whole-chunk ARG.
        if window is not None:
            arg_span = float(draws[0].sequence_length)
            want = float(window[1] - window[0])
            if want > 0 and arg_span > want * 1.05:
                raise RuntimeError(
                    f"{label}: the persisted ARG spans {arg_span:,.0f} bp but "
                    f"the budget window is {want:,.0f} bp "
                    f"({arg_span / want:.1f}x). It was inferred over more "
                    f"sequence than the other methods, so its runtime and site "
                    f"set are not comparable; re-infer it.")
        arg_pos = {int(s.position) for s in draws[0].sites()}
        if not arg_pos:
            return
        panel_pos = {int(s.position) for s in ts_sub.sites()}
        frac = len(arg_pos & panel_pos) / len(arg_pos)
        if frac < self.ARG_PANEL_MATCH_MIN:
            raise RuntimeError(
                f"{label}: only {100 * frac:.1f}% of the persisted ARG's "
                f"{len(arg_pos):,} sites are sites of the current panel "
                f"(expected ~100%). The ARG was inferred from a different "
                f"realisation of this scenario and must be re-inferred; "
                f"scoring on it would silently grade the wrong data.")

    def _shard_orient(self, n_out: int, offset: int,
                      window: "tuple[int, int] | None" = None) -> dict:
        """The fixed-tree MAP orientation for the ``_ft`` variant, shifted into
        the window's 0-based frame. Affects only the tool's input (inference),
        not scoring."""
        return {p - offset: a for p, a in
                self._fixed_tree_map_by_pos(n_out, window=window).items()}

    def score_arg_shard(self, draws, n_out: int, *,
                        window: "tuple[int, int] | None" = None,
                        burn_in: int = 0, mu: "float | None" = None,
                        focal: "str | None" = None,
                        mu_matches_time_units: bool = False,
                        ) -> tuple[dict, dict]:
        """Score a list of (persisted) ARG ``draws`` against the panel's full
        in-window site set --- the method-agnostic scoring step shared by every
        panel-ARG method. A single-genealogy method (Relate / tsinfer) passes a
        one-element list; SINGER passes its thinned draws and a ``burn_in``.
        ``mu`` overrides the scenario rate for the branch-length scaling (tsinfer
        dates its trees, so it scores with the fitted rate, not ``self.mu``).

        :param focal: Node to read each local tree's posterior at, as
            :class:`~ancestree.inference.ARGBasedInference` takes it. The panel
            is renamed ``tsk_i`` for the tool's frame, so the ingroup and
            outgroup ids are rebuilt here in that naming.
        :param mu_matches_time_units: Whether ``mu`` is per unit of the draws'
            own node times, which an undated ARG scored at a rate fitted to
            itself requires (see :func:`_date_tsinfer`).
        :return: ``(post, obs)`` keyed by absolute position (posteriors as
            lists, observed-allele sets sorted)."""
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _singer
        ts_sub, offset = self._windowed_panel(n_out, window, keep_all_sites=True)
        if draws:
            self._check_arg_matches_panel(
                draws, ts_sub, window=window, label=f"{getattr(self, 'key', '?')} "
                f"chunk {getattr(self, 'chunk_idx', '?')} n_out={n_out}")
        names = [f"tsk_{i}" for i in range(ts_sub.num_samples)]
        sample_map = {n: i for i, n in enumerate(names)}
        n_ing = len(self.ingroup_names)
        post_by_pos, obs_by_pos = _singer.score_arg_draws(
            draws, ts_sub, model=JC69(), mu=(self.mu if mu is None else mu),
            sample_map=sample_map, burn_in=burn_in,
            focal=focal, ingroup_samples=names[:n_ing],
            outgroup_samples=names[n_ing:],
            mu_matches_time_units=mu_matches_time_units)
        post = {int(p) + offset: [float(x) for x in v]
                for p, v in post_by_pos.items()}
        obs = {int(p) + offset: sorted(s) for p, s in obs_by_pos.items()}
        return post, obs

    def score_arg_shard_split(self, draws, n_out: int, *,
                              window: "tuple[int, int] | None" = None,
                              burn_in: int = 0, mu: "float | None" = None,
                              mu_matches_time_units: bool = False,
                              ) -> tuple[dict, dict, float]:
        """Persisted-ARG posteriors read at the node each site class is graded
        at: the ingroup MRCA for ingroup-polymorphic sites, the panel root for
        the rest, matching :meth:`split_truth` and :data:`FOCAL_PASSES`. One
        kernel pass per focal node over the same draws --- no re-inference.

        :return: ``(post, obs, scoring_seconds)`` keyed by absolute position."""
        post: dict = {}
        obs: dict = {}
        t0 = time.perf_counter()
        for focal, polymorphic in FOCAL_PASSES:
            p, o = self.score_arg_shard(
                draws, n_out, window=window, burn_in=burn_in, mu=mu,
                focal=focal, mu_matches_time_units=mu_matches_time_units)
            obs.update(o)
            for pos, vals in p.items():
                if (polymorphic is None
                        or self.ingroup_is_polymorphic(pos) is polymorphic):
                    post[pos] = vals
        return post, obs, time.perf_counter() - t0

    def _ingroup_ne(self, ts_sub) -> float:
        """Watterson effective size of the ingroup alone, in diploids.

        Estimated over the whole panel it rises with ``n_out``, because the
        outgroups' divergence enters the diversity: on baseline chunk 0 it
        runs 30,913 at no outgroups to 253,908 at ten, against a simulated
        30,000. The coalescent prior of a tool given that number would then
        move along the very axis the comparison reads, and only for the tools
        that take one.
        """
        import numpy as np

        nodes = [int(s) for s in ts_sub.samples()][:len(self.ingroup_names)]
        pi = float(np.asarray(
            ts_sub.diversity(sample_sets=[nodes], mode="site")).ravel()[0])
        return max(pi / (4.0 * self.mu), 1.0)

    def singer_infer_shard(self, n_out: int, *, ft_polarize: bool = False,
                           window: "tuple[int, int] | None" = None,
                           seed: int = 0):
        """One SINGER *inference* shard: run SINGER (one MCMC seed) on the panel
        (ingroup + ``n_out`` outgroups, optionally windowed) and return its
        thinned posterior ARG draws --- no scoring. Feeding the outgroups into
        SINGER's panel is the genealogical channel (extra lineages to place the
        root). Both variants orient the input VCF with the fixed-tree MAP call
        (the same orientation tsinfer and Relate receive, and never the true
        ancestral allele); ``ft_polarize`` only raises SINGER's trust in that
        orientation from ``-polar 0.5`` (marginalise, reconsidering from the
        topology) to ``-polar 0.99`` (mostly trust it).

        :return: ``(draws, offset, runtime)``."""
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _singer
        ts_sub, offset = self._windowed_panel(n_out, window)
        orient = self._shard_orient(n_out, offset, window)
        polar = "0.99" if ft_polarize else "0.5"
        n_iters = int(os.environ.get("SINGER_N_ITERS", "300"))
        thin = int(os.environ.get("SINGER_THIN", "20"))
        draws, rt = _singer.singer_infer(
            ts_sub, mu=self.mu, rec_rate=self.rec_rate, polar=polar, orient_by_pos=orient,
            n_iters=n_iters, thin=thin, seed=seed,
            ne_diploid=self._ingroup_ne(ts_sub))
        return draws, offset, rt

    def singer_shard(self, n_out: int, *, ft_polarize: bool = False,
                     window: "tuple[int, int] | None" = None, seed: int = 0,
                     ) -> tuple[dict, dict, float]:
        """Whole-shard SINGER panel: :meth:`singer_infer_shard` then
        :meth:`score_arg_shard`. The workflow persists the draws between the two.
        This convenience composition keeps them in memory.

        :return: ``(post, obs, runtime)`` keyed by absolute position."""
        # Feeding outgroups into the panel enlarges the ARG, so SINGER needs a
        # short warm-up before its posterior draws are usable. Default to a few.
        import os
        burn_in = int(os.environ.get("SINGER_BURN_IN", "5"))
        draws, _offset, rt = self.singer_infer_shard(
            n_out, ft_polarize=ft_polarize, window=window, seed=seed)
        post, obs = self.score_arg_shard(
            draws, n_out, window=window, burn_in=burn_in)
        return post, obs, rt

    def score_singer_posteriors(self, post_by_pos: dict, obs_by_pos: dict,
                                *, n_out: "int | None" = None) -> PerSite:
        """Score already-averaged panel-ARG posteriors into per-site records
        (the scoring tail shared by the whole-chunk and sharded paths). Records
        carry the observed-allele mass (7th field) like the fixed-tree path.

        The posteriors cover every in-window site (see
        :func:`_singer.score_arg_topology`), so the fixed / monomorphic sites are
        scored on the inferred topology directly --- no no-call fill is needed,
        and every column is graded on the same site set as the true-ARG
        reference. ``n_out`` selects the panel being scored, not the node it is
        graded at: both site classes are graded at nodes fixed across the row
        (see :meth:`split_truth`).
        """
        truth = self.split_truth(
            self.ingroup_names + self._select_outgroups(n_out))
        out: PerSite = []
        for pos, vals in post_by_pos.items():
            pos = int(pos)
            if not self._pos_kept(pos):
                continue
            t = truth.get(pos)
            if t is None or t not in STATES:
                continue
            vals = np.asarray(vals, dtype=float)
            b = self.sfs_bin.get(pos, -1)
            ptrue = float(vals[STATES.index(t)])
            map_state = STATES[int(np.argmax(vals))]
            sumsq = float(np.dot(vals, vals))
            obs = set(obs_by_pos.get(pos, STATES))
            p_obs = float(sum(vals[STATES.index(a)] for a in obs if a in STATES))
            out.append((b, self.classify(pos), int(map_state == t), ptrue,
                        self.mut_count_by_pos.get(pos, 0), sumsq, p_obs))
        return out

    def run_singer_arg_panel(self, n_out: int, *, ft_polarize: bool = False) -> tuple[PerSite, float]:
        """Whole-chunk SINGER panel ARG + Felsenstein kernel (single MCMC chain),
        averaged over SINGER's posterior ARG samples (the Bayesian
        marginalisation over the ARG). Convenience wrapper composing
        :meth:`singer_shard` (whole chunk, one seed, ingroup + ``n_out``
        outgroups in the panel) with :meth:`score_singer_posteriors`. The SLURM
        path shards instead. Returns empty if SINGER is unavailable."""
        post, obs, rt = self.singer_shard(n_out, ft_polarize=ft_polarize)
        return self.score_singer_posteriors(post, obs, n_out=n_out), rt

    def run_singer_arg_panel_ft(self, n_out: int) -> tuple[PerSite, float]:
        """SINGER panel ARG + kernel, oriented by the fixed-tree MAP call."""
        return self.run_singer_arg_panel(n_out, ft_polarize=True)

    def relate_infer_shard(self, n_out: int, *, ft_polarize: bool = False,
                           window: "tuple[int, int] | None" = None,
                           seed: int = 1):
        """One Relate *inference* shard: infer the genealogy of the panel
        (ingroup + ``n_out`` outgroups, optionally windowed) with Relate and
        return it --- no scoring. Outgroups enter Relate's sample panel as extra
        lineages; ``ft_polarize`` orients the input by the fixed-tree MAP call
        (affecting only the inferred genealogy, since the kernel re-polarises
        from the topology). Relate is deterministic, so one seed and no MCMC
        averaging.

        :return: ``(arg, offset, runtime)`` --- a one-element draw list for
            :meth:`score_arg_shard`, the window offset, and the Relate wall time."""
        import os, sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import _relate
        ts_sub, offset = self._windowed_panel(n_out, window)
        orient = self._shard_orient(n_out, offset, window) if ft_polarize else None
        arg, rt = _relate.relate_infer(
            ts_sub, mu=self.mu, rec_rate=self.rec_rate, orient_by_pos=orient, seed=seed,
            ne_diploid=self._ingroup_ne(ts_sub))
        return [arg], offset, rt

    def relate_shard(self, n_out: int, *, ft_polarize: bool = False,
                     window: "tuple[int, int] | None" = None, seed: int = 1,
                     ) -> tuple[dict, dict, float]:
        """Whole-shard Relate panel: :meth:`relate_infer_shard` then
        :meth:`score_arg_shard`. The workflow persists the ARG between the two.
        This convenience composition keeps it in memory.

        :return: ``(post, obs, runtime)`` keyed by absolute position."""
        draws, _offset, rt = self.relate_infer_shard(
            n_out, ft_polarize=ft_polarize, window=window, seed=seed)
        post, obs = self.score_arg_shard(draws, n_out, window=window, burn_in=0)
        return post, obs, rt

    def run_relate_arg_panel(self, n_out: int, *, ft_polarize: bool = False) -> tuple[PerSite, float]:
        """Whole-chunk Relate panel genealogy + Felsenstein kernel. Convenience
        wrapper composing :meth:`relate_shard` (whole chunk, ingroup + ``n_out``
        outgroups) with :meth:`score_singer_posteriors` (the scoring tail is
        method-agnostic)."""
        post, obs, rt = self.relate_shard(n_out, ft_polarize=ft_polarize)
        return self.score_singer_posteriors(post, obs, n_out=n_out), rt

    def run_relate_arg_panel_ft(self, n_out: int) -> tuple[PerSite, float]:
        """Relate panel genealogy + kernel, oriented by the fixed-tree MAP call."""
        return self.run_relate_arg_panel(n_out, ft_polarize=True)

# --------------------------------------------------------- scenario configs
SCENARIOS = [
    Scenario(
        key="baseline",
        label="Baseline",
        # 10-rung outgroup ladder (first 3 depths match the B6 baseline,
        # then 7 deeper) so the heatmap can show an n_out=10 row.
        trees_path=DATA / "baseline10_sim.trees",
        meta_path=DATA / "baseline10_sim_meta.json",
    ),
    Scenario(
        key="cpg_hypermut",
        label="Hypermutation",
        trees_path=DATA / "cpg_hypermut_sim.trees",
        meta_path=DATA / "cpg_hypermut_sim_meta.json",
        mu=1.25e-7,  # ~10× baseline = realistic CpG transition rate
    ),
    Scenario(
        key="strong_ils",
        label="Strong ILS",
        trees_path=DATA / "strong_ils.trees",
        meta_path=DATA / "strong_ils_meta.json",
    ),
    Scenario(
        key="outgroup_clade",
        label="Outgroup clade",
        trees_path=DATA / "outgroup_clade_sim.trees",
        meta_path=DATA / "outgroup_clade_sim_meta.json",
    ),
    # SLiM column: aggregation across forward-sim chunks. The gamma-SMC
    # cells here stay NaN by default (40-cell run is expensive).
    Scenario(
        key="slim",
        label="Purifying",
        trees_path=DATA / "_slim_aggregate",  # sentinel. Not loaded as a single ts
        meta_path=DATA / "slim_outgroup_bias10_fdel1.0_chunk0_meta.json",  # placeholder
    ),
    # Demographic-distortion scenarios: the baseline outgroup ladder with a
    # recent exponential ingroup size change (growth = excess rare variants;
    # decline = enriched high-frequency tail). Used to test the adaptive
    # polarization prior against a Kingman prior under SFS distortion, and to
    # add a decline row to the fractional-SFS figure. Chunked like the other
    # msprime scenarios. The monolithic trees_path is an unused placeholder.
    Scenario(
        key="pop_constant",
        label="Baseline",
        trees_path=DATA / "pop_constant.trees",
        meta_path=DATA / "pop_constant_chunk0_meta.json",
    ),
    Scenario(
        key="pop_growth",
        label="Growth",
        trees_path=DATA / "pop_growth.trees",
        meta_path=DATA / "pop_growth_chunk0_meta.json",
    ),
    Scenario(
        key="pop_decline",
        label="Decline",
        trees_path=DATA / "pop_decline.trees",
        meta_path=DATA / "pop_decline_chunk0_meta.json",
    ),
]

# The demographic-distortion scenarios live in SCENARIOS so their chunks/cells
# are generated by the shared machinery (and _SCENARIO_BY_KEY can resolve them
# for the prior-under-demography appendix), but they are NOT columns of the
# robustness heatmaps --- those show only the five robustness scenarios.
# FIG_SCENARIOS is the column set every plotting / runtime-table helper below
# iterates over; SCENARIOS stays the full registry for compute and lookups.
DEMOG_SCENARIO_KEYS = {"pop_constant", "pop_growth", "pop_decline"}
FIG_SCENARIOS = [sc for sc in SCENARIOS if sc.key not in DEMOG_SCENARIO_KEYS]


# ---------------------- pre-computed SLiM aggregation (unchanged from prev)
SLIM_FDEL = "1.0"
SLIM_N_CHUNKS = 100  # matches Snakefile SLIM10_N_CHUNKS (robustness SLiM, 10× sites)


def _scenario_ils_parts(sc) -> "list[tuple[Path | None, Path]]":
    """``(trees, meta)`` paths whose ILS averages to one scenario's.

    One entry per chunk for the chunked scenarios, one for the whole-genome sim
    otherwise. A ``None`` trees path means the trees are recovered from the meta
    (the SLiM chunks, which are recapitated on load).
    """
    if sc.key == "slim":
        return [(None, DATA / f"slim_outgroup_bias10_fdel{SLIM_FDEL}"
                              f"_chunk{ch}_meta.json")
                for ch in range(SLIM_N_CHUNKS)]
    if sc.key in CHUNKED_MSP_SCENARIOS:
        stem = _MSP_STEM[sc.key]
        return [(DATA / f"{stem}_chunk{ch}.trees",
                 DATA / f"{stem}_chunk{ch}_meta.json")
                for ch in range(ROBUSTNESS_MSP_N_CHUNKS)]
    return [(sc.trees_path, sc.meta_path)]


def _slim_chunk_meta(chunk_idx: int) -> dict:
    p = DATA / f"slim_outgroup_bias10_fdel{SLIM_FDEL}_chunk{chunk_idx}_meta.json"
    return json.loads(p.read_text())


# ----------------------------------------- per-scenario ILS (figure header)
# Span-weighted fraction of the genome where the ingroup is non-monophyletic
# on the true tree (the ingroup MRCA's leaf set is not exactly the ingroup).
# A scenario-level property of the genealogy, identical across inference
# modes, so it annotates each scenario column of the heatmap. Cached on disk
# (the trees are large, we only need the topology spans).
_ILS_CACHE_PATH = DATA / "scenario_ils.json"
ils_by_scenario: dict[str, float] = {}


def _span_weighted_ils(ts: "tskit.TreeSequence", ingroup_nodes: set[int]) -> float:
    ing = sorted(ingroup_nodes)
    target = set(ing)
    total = 0.0
    non_mono = 0.0
    for tree in ts.trees():
        if tree.num_edges == 0:
            continue
        span = tree.interval.right - tree.interval.left
        mrca = tree.mrca(*ing) if len(ing) > 1 else ing[0]
        total += span
        if mrca == tskit.NULL or set(tree.leaves(mrca)) != target:
            non_mono += span
    return non_mono / total if total else float("nan")


def _scenario_ingroup_nodes(ts: "tskit.TreeSequence", meta: dict) -> set[int]:
    """Ingroup sample nodes: SLiM tags them by population ``p0``. The
    msprime scenarios carry per-individual ``tsk_<id>`` names in the meta."""
    pop_name = {p.id: p.metadata.get("name", "") for p in ts.populations()}
    p0_nodes = {int(s) for s in ts.samples()
                if pop_name.get(int(ts.node(int(s)).population), "") == "p0"}
    if p0_nodes:
        return p0_nodes
    ing_names = set(meta.get("ingroup_names", []))
    node_of = {f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()}
    return {node_of[n] for n in ing_names if n in node_of}


def compute_scenario_ils(*, use_cache: bool = True) -> dict[str, float]:
    """Populate and return :data:`ils_by_scenario` (scenario key -> ILS
    fraction). Reads a JSON cache when present. Otherwise scans each
    scenario's true tree(s) and writes the cache."""
    global ils_by_scenario
    if use_cache and _ILS_CACHE_PATH.exists():
        ils_by_scenario = json.loads(_ILS_CACHE_PATH.read_text())
        return ils_by_scenario
    out: dict[str, float] = {}
    for sc in SCENARIOS:
        # Average over the pieces the scenario is actually simulated as: every
        # chunked scenario (SLiM and msprime alike) averages its chunks, an
        # unchunked one has the single whole-genome sim. A missing piece is
        # skipped rather than fatal --- ILS is a figure annotation, and a
        # scenario whose sims are not on disk should not sink the report.
        vals = []
        for trees_path, meta_path in _scenario_ils_parts(sc):
            if not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text())
            if trees_path is None:
                ts = _slim_local_trees(meta)
            elif trees_path.exists():
                ts = tskit.load(str(trees_path))
            else:
                continue
            vals.append(_span_weighted_ils(
                ts, _scenario_ingroup_nodes(ts, meta)))
        out[sc.key] = float(np.mean(vals)) if vals else float("nan")
    ils_by_scenario = out
    try:
        _ILS_CACHE_PATH.write_text(json.dumps(out, indent=2))
    except OSError as exc:
        print(f"WARNING: could not write cache: {exc}", flush=True)
    return out


_SLIM_MUTCOUNT_CACHE: dict[int, dict[int, int]] = {}


def _slim_local_trees(meta: dict):
    """Load a SLiM chunk's tree sequence from the local DATA dir. ``trees_path``
    in the meta JSON is stored as the absolute path of the machine that ran the
    forward sim (e.g. a laptop), so resolve by basename under DATA --- the same
    convention the msprime path uses (see ``Scenario.__init__``)."""
    return tskit.load(DATA / os.path.basename(meta["trees_path"]))


def _slim_chunk_mutcount(chunk_idx: int) -> dict[int, int]:
    if chunk_idx in _SLIM_MUTCOUNT_CACHE:
        return _SLIM_MUTCOUNT_CACHE[chunk_idx]
    meta = _slim_chunk_meta(chunk_idx)
    ts = _slim_local_trees(meta)
    out = {int(s.position): len(s.mutations) for s in ts.sites()}
    _SLIM_MUTCOUNT_CACHE[chunk_idx] = out
    return out


def _slim_classify(bin_: int, n_alleles: int, n_ingroup: int) -> str:
    if n_alleles > 2:
        return "polyallelic"
    if bin_ == 0 or bin_ == n_ingroup:
        return "biallelic_mono"
    return "biallelic_poly"


def _split_truth_for_panel(ts, ingroup_nodes, panel_nodes) -> dict[int, str]:
    """Truth at the node each site class is asked about, keyed by position.

    Read from the unsimplified sequence: simplifying to the panel deletes the
    branches above its root along with any mutation on them, which is exactly
    where a site the ingroup has fixed carries its only mutation.

    :param ts: The full simulated tree sequence, before any simplification.
    :param ingroup_nodes: Sample node ids making up the ingroup.
    :param panel_nodes: Sample node ids in the panel, ingroup included.
    :return: Position to true state at the ingroup MRCA for sites polymorphic
        within the ingroup, and at the panel root for the rest.
    """
    ingroup = [int(n) for n in ingroup_nodes]
    panel = [int(n) for n in panel_nodes]
    index = {int(n): j for j, n in enumerate(ts.samples())}
    ingroup_index = [index[n] for n in ingroup]
    polymorphic = {}
    for v in ts.variants():
        alleles = {v.alleles[v.genotypes[j]] for j in ingroup_index}
        alleles -= {None, ""}
        polymorphic[int(v.site.position)] = len(alleles) > 1
    out: dict[int, str] = {}
    for tree in ts.trees():
        for site in tree.sites():
            pos = int(site.position)
            out[pos] = _truth_at_node(
                site, tree, ingroup if polymorphic.get(pos) else panel,
            )
    return out


def _slim_chunk_ts_and_samples(chunk_idx: int, n_out: int):
    meta = _slim_chunk_meta(chunk_idx)
    ingroup_names = list(meta["ingroup_names"])
    all_outgroup_names = list(meta["outgroup_names"])
    outgroup_names = _evenly_spread_outgroups(all_outgroup_names, n_out)
    n_ingroup = len(ingroup_names)
    ts_full = _slim_local_trees(meta)
    pop_name_by_id = {p.id: p.metadata.get("name", "")
                      for p in ts_full.populations()}
    # out_T{k} -> p{k}, derived from the meta so 3- and 10-outgroup SLiM
    # sims both work.
    out_to_pop = {name: "p" + name[len("out_T"):] for name in all_outgroup_names}
    ingroup_nodes: list[int] = []
    out_nodes_by_pop: dict[str, list[int]] = {p: [] for p in out_to_pop.values()}
    for sn in ts_full.samples():
        node = ts_full.node(int(sn))
        pname = pop_name_by_id.get(int(node.population), "")
        if pname == "p0":
            ingroup_nodes.append(int(sn))
        elif pname in out_nodes_by_pop:
            out_nodes_by_pop[pname].append(int(sn))
    selected_out_nodes = [
        out_nodes_by_pop[out_to_pop[nm]][0] for nm in outgroup_names
    ]
    keep_nodes = list(ingroup_nodes) + selected_out_nodes
    keep_names = list(ingroup_names) + list(outgroup_names)
    ts_sub = ts_full.simplify(samples=keep_nodes, filter_sites=True)
    # Graded at the FULL outgroup panel's root, not this column's selection, so
    # the question does not deepen as outgroups are added --- see
    # Scenario.split_truth, which does the same for the msprime scenarios. The
    # panel being SCORED is still keep_nodes. Only the grading node is fixed.
    full_out_nodes = [out_nodes_by_pop[out_to_pop[nm]][0]
                      for nm in all_outgroup_names]
    truth_by_pos = _split_truth_for_panel(
        ts_full, ingroup_nodes, list(ingroup_nodes) + full_out_nodes)
    return ts_sub, keep_names, n_ingroup, truth_by_pos


def _slim_chunk_majority(chunk_idx: int, n_out: int) -> PerSite:
    ts_sub, keep_names, n_ingroup, truth_by_pos = _slim_chunk_ts_and_samples(
        chunk_idx, n_out,
    )
    ingroup_names = keep_names[:n_ingroup]
    outgroup_names = keep_names[n_ingroup:]
    sites: list[Site] = []
    flags: list[tuple[int, int, str, int]] = []
    for v in ts_sub.variants():
        pos = int(v.site.position)
        t = truth_by_pos.get(pos)
        if t is None or t not in STATES:
            continue
        n_alleles = sum(1 for a in v.alleles if a not in ("", None))
        ingroup_derived = sum(
            1 for j in range(n_ingroup)
            if v.alleles[v.genotypes[j]] != v.site.ancestral_state
        )
        tip_alleles = {sid: v.alleles[v.genotypes[j]]
                       for j, sid in enumerate(keep_names)}
        sites.append(Site(
            chrom="1", pos=pos, alleles=tuple(v.alleles),
            tip_alleles=tip_alleles,
        ))
        flags.append((ingroup_derived, n_alleles, t,
                      _slim_chunk_mutcount(chunk_idx).get(pos, 0)))
    if not sites:
        return []
    # n_out=0 → no outgroup vote. Fall back to the major ingroup allele.
    inf = MajorityOutgroupInference(
        sites, outgroup_samples=outgroup_names,
        ingroup_samples=ingroup_names, for_comparison_only=True,
    )
    out: PerSite = []
    for (site, post), (bin_, n_alleles, t, n_mut) in zip(inf.infer(), flags):
        klass = _slim_classify(bin_, n_alleles, n_ingroup)
        out.append((bin_, klass, int(post.map_allele == t),
                    float(post[t]), n_mut, _sumsq(post)))
    return out


def _ingroup_polymorphic(ts, n_ingroup: int) -> dict[int, bool]:
    """Whether more than one allele segregates within the ingroup, per site.

    :param ts: The panel tree sequence, ingroup first among the samples.
    :param n_ingroup: Number of leading samples making up the ingroup.
    :return: Position to whether the ingroup is polymorphic there.
    """
    out: dict[int, bool] = {}
    for v in ts.variants():
        alleles = {v.alleles[g] for g in v.genotypes[:n_ingroup]}
        alleles -= {None, ""}
        out[int(v.site.position)] = len(alleles) > 1
    return out


def _slim_chunk_arg_live(chunk_idx: int, n_out: int,
                         *, mu: float | None = None,
                         keep_intervals=None) -> PerSite:
    ts_sub, keep_names, n_ingroup, truth_by_pos = _slim_chunk_ts_and_samples(
        chunk_idx, n_out,
    )
    meta = _slim_chunk_meta(chunk_idx)
    if mu is None:
        mu = float(meta.get("mu_slim", meta.get("mu_target", 1.25e-8)))
    sample_map = {s: i for i, s in enumerate(keep_names)}
    keep = _pos_in_intervals(keep_intervals)
    ingroup_names = keep_names[:n_ingroup]
    outgroup_names = keep_names[n_ingroup:]
    polymorphic = _ingroup_polymorphic(ts_sub, n_ingroup)
    out: PerSite = []
    for focal, want in FOCAL_PASSES:
        inf = ARGBasedInference(
            ts_sub, JC69(), mu=mu, sample_map=sample_map, focal=focal,
            ingroup_samples=ingroup_names, outgroup_samples=outgroup_names,
            progress=False,
        )
        for site, post in inf.infer():
          pos = int(site.pos)
          if not keep(pos):
              continue
          if want is not None and polymorphic.get(pos, False) is not want:
              continue
          t = truth_by_pos.get(pos)
          if t is None or t not in STATES:
              continue
          v = next(iter(ts_sub.variants(
              samples=ts_sub.samples(),
              left=site.pos, right=site.pos + 1,
          )))
          n_alleles = sum(1 for a in v.alleles if a not in ("", None))
          ingroup_derived = sum(
              1 for j in range(n_ingroup)
              if v.alleles[v.genotypes[j]] != v.site.ancestral_state
          )
          klass = _slim_classify(ingroup_derived, n_alleles, n_ingroup)
          out.append((ingroup_derived, klass,
                      int(post.map_allele == t), float(post[t]),
                      _slim_chunk_mutcount(chunk_idx).get(pos, 0), _sumsq(post)))
    return out


def _slim_chunk_ft_map(chunk_idx: int, n_out: int) -> dict[int, str]:
    """Per-site fixed-tree MAP ancestral allele for one SLiM chunk (used to
    orient tsinfer with the inferred, not the true, ancestral allele)."""
    ts_sub, keep_names, n_ingroup, truth_by_pos = _slim_chunk_ts_and_samples(
        chunk_idx, n_out,
    )
    ingroup_names = keep_names[:n_ingroup]
    outgroup_names = keep_names[n_ingroup:]
    sites: list[Site] = []
    for v in ts_sub.variants():
        pos = int(v.site.position)
        if truth_by_pos.get(pos) not in STATES:
            continue
        tip_alleles = {sid: v.alleles[v.genotypes[j]]
                       for j, sid in enumerate(keep_names)}
        sites.append(Site(chrom="1", pos=pos, alleles=tuple(v.alleles),
                          tip_alleles=tip_alleles))
    if not sites:
        return {}
    prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
    inf = FixedTreeInference(
        sites, ingroup_samples=ingroup_names, outgroup_samples=outgroup_names,
        model=JC69(), n_target_sites=int(ts_sub.sequence_length),
        ingroup_weight=prior, parallelize=False, progress=False,
    )
    if outgroup_names:
        inf.fit()
    return {int(s.pos): p.map_allele for s, p in inf.infer()}


def _slim_pooled_ft(n_out: int, *, adaptive: bool = False) -> PerSite:
    """Fixed-tree on the SLiM column with ONE joint fit pooling every chunk's
    Sites (then per-site posteriors), instead of a separate fit per 100-kb
    chunk. Valid because the SLiM chunks are independent replicates of the
    *same* demography, so the outgroup-ladder model (and its branch rates)
    is shared — pooling avoids the per-chunk small-sample overfit. Mirrors
    B14's joint ``fit_slim`` design. ARG/local-tree stay per-chunk (they
    need a consistent-sample genealogy), which is lossless for them.
    """
    all_sites: list[Site] = []
    flags: list[tuple[int, int, str, int, int, bool]] = []
    ingroup_names: list[str] = []
    outgroup_names: list[str] = []
    total_seq_len = 0
    for c in range(SLIM_N_CHUNKS):
        meta_p = DATA / f"slim_outgroup_bias10_fdel{SLIM_FDEL}_chunk{c}_meta.json"
        if not meta_p.exists():
            continue
        ts_sub, keep_names, n_ingroup, truth_by_pos = _slim_chunk_ts_and_samples(
            c, n_out,
        )
        ingroup_names = keep_names[:n_ingroup]
        outgroup_names = keep_names[n_ingroup:]
        seq_len = int(ts_sub.sequence_length)
        # offset positions per chunk so the pooled Sites stay globally unique
        off = c * (seq_len + 1)
        total_seq_len += seq_len
        for v in ts_sub.variants():
            pos = int(v.site.position)
            t = truth_by_pos.get(pos)
            if t is None or t not in STATES:
                continue
            n_alleles = sum(1 for a in v.alleles if a not in ("", None))
            ingroup_derived = sum(
                1 for j in range(n_ingroup)
                if v.alleles[v.genotypes[j]] != v.site.ancestral_state
            )
            tip_alleles = {sid: v.alleles[v.genotypes[j]]
                           for j, sid in enumerate(keep_names)}
            # Which pass a site belongs to is decided by whether the ingroup
            # carries more than one allele, not by how many differ from the
            # ancestral state: a whole ingroup split between two derived
            # alleles is polymorphic yet has every sample differing.
            polymorphic = len({v.alleles[v.genotypes[j]]
                               for j in range(n_ingroup)}) > 1
            all_sites.append(Site(chrom="1", pos=off + pos,
                                  alleles=tuple(v.alleles),
                                  tip_alleles=tip_alleles))
            flags.append((ingroup_derived, n_alleles, t,
                          _slim_chunk_mutcount(c).get(pos, 0), n_ingroup,
                          polymorphic))
    if not all_sites:
        return []
    prior = (AdaptiveIngroupWeight(ingroup_samples=ingroup_names)
             if adaptive else
             KingmanIngroupWeight(ingroup_samples=ingroup_names))
    inf = FixedTreeInference(
        all_sites, ingroup_samples=ingroup_names, outgroup_samples=outgroup_names,
        model=JC69(), n_target_sites=total_seq_len,
        ingroup_weight=prior,
        parallelize=False, progress=False,
    )
    if outgroup_names:
        inf.fit()
    # One pass per site class, as everywhere else: the ingroup MRCA for a
    # polymorphism, the ladder's deepest join for a site the ingroup has fixed.
    runs = [(inf, None)]
    if outgroup_names:
        deep = FixedTreeInference(
            all_sites, tree=inf.tree, model=JC69(),
            n_target_sites=total_seq_len, ingroup_weight=prior,
            fit_required=False,
            focal=FocalNode("ingroup_mrca", fraction=1.0),
            parallelize=False, progress=False,
        )
        runs = [(inf, True), (deep, False)]
    out: PerSite = []
    for run, want in runs:
        for (site, post), (bin_, n_alleles, t, n_mut, n_ing, poly) in zip(
            run.infer(), flags,
        ):
            if want is not None and poly is not want:
                continue
            klass = _slim_classify(bin_, n_alleles, n_ing)
            out.append((bin_, klass, int(post.map_allele == t),
                        float(post[t]), n_mut, _sumsq(post)))
    return out


def slim_aggregate_persite(method_key: str, n_out: int) -> tuple[PerSite, float]:
    """Aggregate per-chunk per-site lists. Return (records, seconds).

    Per-chunk timing is not collected for SLiM (the pre-computed JSONs from
    the workflow do not carry it), so the per-chunk loop is timed here as a
    proxy for the aggregation cost. The underlying inference cost lives in the
    pre-computed JSONs.
    """
    t0 = time.perf_counter()
    agg: PerSite = []
    if method_key == "anc_arg":
        # Compute live at every n_out so this row stays consistent with the
        # max-parsimony (mu->0) row, which is the same ARGBasedInference kernel.
        # The cached per-chunk arg JSONs under-counted the fixed/divergence
        # class (cached vs live diverged at n_out=3), inflating anc_arg's Brier.
        for c in range(SLIM_N_CHUNKS):
            agg.extend(_slim_chunk_arg_live(c, n_out))
    elif method_key == "anc_ft":
        # Fixed-tree: ONE joint fit pooling all chunks' Sites (correct,
        # avoids per-chunk small-sample overfit). Computed live since the
        # robustness SLiM pipeline doesn't emit the B14-style infer JSONs.
        agg.extend(_slim_pooled_ft(n_out))
    elif method_key == "anc_ft_adaptive":
        agg.extend(_slim_pooled_ft(n_out, adaptive=True))
    # ----- appendix baselines (Brier-proper 6-tuples) -----
    elif method_key == "parsimony_mu0":
        for c in range(SLIM_N_CHUNKS):
            agg.extend(_slim_chunk_arg_live(c, n_out, mu=1e-12))
    elif method_key == "majority_outgroup":
        for c in range(SLIM_N_CHUNKS):
            agg.extend(_slim_chunk_majority(c, n_out))
    # tsinfer_arg (and SINGER / Relate) are handled per-chunk through the
    # persisted-ARG shard rules (see _PERSISTED_ARG_METHODS), not here.
    return agg, time.perf_counter() - t0


# -------------------------------------------------------------- method list
# n_out=10 only resolves for the baseline (10-outgroup ladder). The other
# scenarios have 3 outgroups, so their n_out=10 cells stay blank (guarded
# in compute_cell). n_out=2 dropped (the 0/1/3 spread already shows the
# trend without the extra row).
N_OUTS = (0, 1, 3, 10)
# The three substantive inference modes: Ancestree on the true ARG (the
# achievable ceiling for a tree-based method), Ancestree on the inferred
# local tree (VCF-only), and Ancestree fixed-tree (outgroup-ladder).
# Parsimony + majority-outgroup baselines and the PolarBEAR/EST-SFS proxy
# rows were dropped — they don't inform the local-tree vs fixed-tree story.
METHODS = [
    ("true ARG (oracle)",           "anc_arg"),
    ("local-tree mode", "local_tree_unphased"),
    ("fixed-tree mode",             "anc_ft"),
]


#: Ancestree's own inference modes, as opposed to the external ARG pipelines.
_ANCESTREE_MODES = {"anc_arg", "local_tree_unphased", "anc_ft"}


# ---------------------------------------------------------- cell compute
# Module-global result stores. Populated by load_cell_summaries() /
# load_cell_summaries() reads the snakemake-cached per-cell summaries.
per_site: dict[tuple[str, str, int], PerSite] = {}
runtimes: dict[tuple[str, str, int], float] = {}

# ----------------------------------------------------- light cell summary
# The plots never need raw per-site records. They need cheap reductions of
# them (per-frequency-bin Brier/accuracy folding, per-class site counts, a
# confidence histogram). Storing those *sufficient statistics* per cell makes
# the intermediate ~a few KB instead of ~470 MB, so reports run in <100 MB RAM
# and the whole set is git-trackable. See summarize_persite().
#
# A summary is a JSON-able dict:
#   {scenario, method, n_out, n_ingroup, runtime,
#    stats: {"klass|bin|rec": [count, pv_n, map_sum, ptrue_sum, sumsq_sum]},
#    conf_hist: {bin: [N_CONF_BINS ints]}}   # conf_hist only for anc_arg
# where rec = int(n_mut > 1). All fields are *additive* across chunks, so a
# cell summary is the element-wise sum of its 10 chunk summaries.
#
# Set RC_DUMP_PERSITE=1 to ALSO write the full per-site list alongside the
# summary (gitignored debug artifact, off by default so reruns stay light).
N_CONF_BINS = 512  # confidence bins over [0.5, 1.0] (peakedness)
_CONF_LO = 0.5

cell_summaries: dict[tuple[str, str, int], dict] = {}


def _conf_bin(conf: float) -> int:
    x = (conf - _CONF_LO) / (1.0 - _CONF_LO)
    i = int(x * N_CONF_BINS)
    return 0 if i < 0 else (N_CONF_BINS - 1 if i >= N_CONF_BINS else i)


def dump_persite_enabled() -> bool:
    """Whether to also emit the heavy per-site list (RC_DUMP_PERSITE=1)."""
    return os.environ.get("RC_DUMP_PERSITE", "") not in ("", "0", "false")


def summarize_persite(records, *, scenario: str, method: str, n_out: int,
                      n_ingroup: int, runtime: float) -> dict:
    """Reduce a per-site record list to the light summary (sufficient stats).

    Bins records by (klass, sfs_bin, rec=n_mut>1) accumulating the sums the
    plots need; NaN-posterior guards match filtered_mean / per_bin_brier
    (map uses all records, p_true/Brier use only valid-posterior records via
    ``pv_n``). A per-bin confidence histogram is built for ``anc_arg`` only
    (the sole consumer is the SFS-confidence-filter figure).
    """
    stats: dict[str, list] = {}
    conf: dict[str, list] = {}
    # Optional per-bin sum of the observed-renormalised true weight (p_true /
    # p_obs), present only when the records carry the 7th field. Feeds the
    # posterior-weighted SFS so unobserved-allele mass is not folded into the
    # mirror bin. Absent records leave this empty and the SFS falls back to
    # p_true (correct when the posterior is supported on the observed alleles).
    sfs: dict[str, float] = {}
    want_conf = (method == "anc_arg")
    for rec_t in records:
        b, klass, mh, pt, nm, sq = rec_t[:6]
        p_obs = rec_t[6] if len(rec_t) > 6 else None
        b = int(b); rec = 1 if nm > 1 else 0
        key = f"{klass}|{b}|{rec}"
        g = stats.get(key)
        if g is None:
            g = [0, 0, 0.0, 0.0, 0.0]
            stats[key] = g
        g[0] += 1  # count (all, map never NaN)
        g[2] += float(mh)  # map_sum
        if pt == pt:  # valid posterior
            g[1] += 1  # pv_n
            g[3] += float(pt)  # ptrue_sum
            g[4] += float(sq)  # sumsq_sum
            if p_obs is not None and p_obs > 0:
                sfs[key] = sfs.get(key, 0.0) + float(pt) / float(p_obs)
            if want_conf and 1 <= b <= n_ingroup - 1:
                row = conf.get(str(b))
                if row is None:
                    row = [0] * N_CONF_BINS
                    conf[str(b)] = row
                row[_conf_bin(pt if pt >= 0.5 else 1.0 - pt)] += 1
    summary = {
        "scenario": scenario, "method": method, "n_out": int(n_out),
        "n_ingroup": int(n_ingroup), "runtime": float(runtime),
        "stats": stats, "conf_hist": conf,
    }
    if sfs:
        summary["sfs_stats"] = sfs
    return summary


def empty_summary(scenario: str, method: str, n_out: int,
                  n_ingroup: int) -> dict:
    return {"scenario": scenario, "method": method, "n_out": int(n_out),
            "n_ingroup": int(n_ingroup), "runtime": 0.0,
            "stats": {}, "conf_hist": {}}


def merge_summaries(summaries) -> dict:
    """Element-wise sum of chunk summaries into a cell summary (additive)."""
    out: dict | None = None
    for s in summaries:
        if s is None:
            continue
        if out is None:
            out = {"scenario": s["scenario"], "method": s["method"],
                   "n_out": int(s["n_out"]), "n_ingroup": int(s["n_ingroup"]),
                   "runtime": 0.0, "stats": {}, "conf_hist": {}}
        out["runtime"] += float(s.get("runtime", 0.0))
        for k, g in s["stats"].items():
            o = out["stats"].get(k)
            if o is None:
                out["stats"][k] = list(g)
            else:
                for i in range(5):
                    o[i] += g[i]
        for b, row in s.get("conf_hist", {}).items():
            o = out["conf_hist"].get(b)
            if o is None:
                out["conf_hist"][b] = list(row)
            else:
                for i in range(len(row)):
                    o[i] += row[i]
        for k, w in s.get("sfs_stats", {}).items():
            sfs = out.setdefault("sfs_stats", {})
            sfs[k] = sfs.get(k, 0.0) + float(w)
        for k, w in s.get("runtime_parts", {}).items():
            parts = out.setdefault("runtime_parts", {})
            parts[k] = parts.get(k, 0) + w
    return out if out is not None else {}


def _group_metric(g: list, metric: str) -> tuple[float, int]:
    """(metric_sum, count) for one (klass,bin,rec) group, matching
    _site_metric/filtered_mean semantics."""
    if metric == "map":
        return g[2], g[0]
    if metric == "ptrue":
        return g[3], g[1]
    if metric == "brier":  # Σ(1 − 2·p_true + Σp²) over valid sites
        return g[1] - 2.0 * g[3] + g[4], g[1]
    raise ValueError(f"unknown metric {metric!r}")

# method_key → Scenario.run_* method name (msprime scenarios).
_METHOD_DISPATCH = {
    "anc_arg": "run_arg",
    "local_tree_unphased": "run_local_tree_unphased",
    "anc_ft": "run_ft",
    "anc_ft_adaptive": "run_ft_adaptive",
    "anc_ft_uniform": "run_ft_uniform",
    # Appendix baselines (HEATMAP_FIGURES["robustness_baselines"]):
    "parsimony_mu0": "run_arg_mu0",  # mu->0 limit of the ARG kernel
    "majority_outgroup": "run_majority_outgroup",
    "tsinfer_arg": "run_tsinfer_arg",  # tsinfer ARG (correct AA) + kernel
    # SINGER panel: ingroup + n_out outgroups as ARG samples (-polar 0.5), and
    # the _ft variant additionally oriented by the fixed-tree MAP call.
    "singer_arg_panel": "run_singer_arg_panel",
    "singer_arg_panel_ft": "run_singer_arg_panel_ft",
    # Relate panel: ingroup + n_out outgroups as sample haplotypes, and the _ft
    # variant additionally oriented by the fixed-tree MAP call.
    "relate_arg_panel": "run_relate_arg_panel",
    "relate_arg_panel_ft": "run_relate_arg_panel_ft",
}

# Appendix-figure method rows, each scored on the SAME grid as the main
# heatmap (see HEATMAP_FIGURES below). The three substantive modes (ARG /
# local-tree / fixed-tree) are kept as reference rows alongside the three
# hard / trivial baselines, and reuse the main heatmap's cached cells.
BASELINE_METHODS = [
    # First three rows all run on the TRUE local trees (oracle genealogy):
    # the full-likelihood kernel and its two parsimony limits.
    ("true ARG (oracle)",           "anc_arg"),  # reference (full likelihood)
    (r"max-parsimony ($\mu\!\to\!0$)", "parsimony_mu0"),  # parsimony limit of ARG mode
    ("local-tree mode", "local_tree_unphased"),
    ("SINGER",                      "singer_arg_panel"),  # outgroups in the SINGER panel, polar 0.5
    ("SINGER (fixed-tree)",         "singer_arg_panel_ft"),  # outgroups in panel + fixed-tree orient
    ("tsinfer + tsdate",            "tsinfer_arg"),  # tsinfer genealogy + kernel
    ("Relate",                      "relate_arg_panel_ft"),  # outgroups in panel + fixed-tree orient (Relate needs an ancestral allele)
    ("fixed-tree mode",             "anc_ft"),
    ("fixed-tree (adaptive)",       "anc_ft_adaptive"),
    ("majority outgroup",           "majority_outgroup"),
]

# The main comparison: Ancestree's three inference modes against the three
# external ARG pipelines, the two groups separated by a rule in the figure.
# tsinfer requires an ancestral orientation, so SINGER and Relate get the same
# fixed-tree MAP orientation for a consistent, leak-free comparison (never the
# true ancestral allele via REF=alleles[0]). SINGER additionally marginalises
# over that orientation (``-polar 0.5``); Relate takes it as given. true-ARG and
# local-tree use no outgroup orientation.
ARG_METHODS = [
    ("true ARG (oracle)", "anc_arg"),
    ("local-tree mode",   "local_tree_unphased"),
    ("fixed-tree mode",   "anc_ft"),
    ("SINGER",            "singer_arg_panel"),
    ("tsinfer + tsdate",  "tsinfer_arg"),
    ("Relate",            "relate_arg_panel_ft"),
]

#: Method rows per heatmap figure, keyed by the ``{figure}`` wildcard of rule
#: ``report_robustness_heatmap``. The rows are the only thing that differs
#: between the figures, so one renderer (make_robustness_heatmap.py) draws all
#: three and every output name derives from the key: ``<figure>.pdf``,
#: ``<figure>_map.pdf``, ``<figure>_runtimes.{json,md}`` under results/reports,
#: and ``bench_<figure>.pdf`` in the manuscript figures dir.
HEATMAP_FIGURES = {
    "robustness_heatmap": ARG_METHODS,
    "robustness_baselines": BASELINE_METHODS,
}

_SCENARIO_BY_KEY = {sc.key: sc for sc in SCENARIOS}


# ----------------------------------------------- chunked msprime scenarios
# The 4 msprime scenarios are simulated as ROBUSTNESS_MSP_N_CHUNKS independent
# replicates (like the SLiM column), so per-chunk inference runs as small,
# parallel, bounded-memory jobs and the cell job is a cheap concat. The
# physical file stems differ from the logical scenario keys (the baseline cell
# "baseline" is simulated as "baseline10_*").
ROBUSTNESS_MSP_N_CHUNKS = 10
CHUNKED_MSP_SCENARIOS = {
    "baseline", "cpg_hypermut", "strong_ils", "outgroup_clade",
    "pop_constant", "pop_growth", "pop_decline",
}
_MSP_STEM = {
    "baseline": "baseline10",
    "cpg_hypermut": "cpg_hypermut",
    "strong_ils": "strong_ils",
    "outgroup_clade": "outgroup_clade",
    "pop_constant": "pop_constant",
    "pop_growth": "pop_growth",
    "pop_decline": "pop_decline",
}


class SlimScenario(Scenario):
    """Adapter exposing one SLiM forward-sim chunk through the msprime
    :class:`Scenario` API, so the SINGER (and every other) Scenario method runs
    on SLiM chunks unchanged. SLiM chunks are ordinary tskit sequences. Only the
    meta format and the population-based sample layout differ from the msprime
    scenarios, which :meth:`load` bridges. ``classify`` is identical to
    :func:`_slim_classify`, so per-site classes match the other SLiM rows."""

    def load(self) -> None:
        meta = _slim_chunk_meta(self.chunk_idx)
        self.mu = float(meta.get("mu_slim", meta.get("mu_target", 1.25e-8)))
        self.rec_rate = float(meta.get("rec_rate_slim", 1e-8))
        self.ingroup_names = list(meta["ingroup_names"])
        self.n_ingroup = len(self.ingroup_names)
        self.all_outgroup_names = list(meta["outgroup_names"])
        # meta["trees_path"] is stored absolute. Resolve by basename under DATA
        # so the SLiM chunk is portable (e.g. rsynced to a cluster).
        self.ts = tskit.load(DATA / os.path.basename(meta["trees_path"]))
        pop_name = {p.id: p.metadata.get("name", "")
                    for p in self.ts.populations()}
        out_to_pop = {nm: "p" + nm[len("out_T"):] for nm in self.all_outgroup_names}
        ingroup_nodes: list[int] = []
        out_nodes: dict[str, list[int]] = {p: [] for p in out_to_pop.values()}
        for sn in self.ts.samples():
            pn = pop_name.get(int(self.ts.node(int(sn)).population), "")
            if pn == "p0":
                ingroup_nodes.append(int(sn))
            elif pn in out_nodes:
                out_nodes[pn].append(int(sn))
        self.sample_id_to_node = {}
        for i, nm in enumerate(self.ingroup_names):
            self.sample_id_to_node[nm] = ingroup_nodes[i]
        for nm in self.all_outgroup_names:
            self.sample_id_to_node[nm] = out_nodes[out_to_pop[nm]][0]
        self.name_for_node = {v: k for k, v in self.sample_id_to_node.items()}
        ingroup_set = set(ingroup_nodes)
        ingroup_idx = [j for j, sn in enumerate(self.ts.samples())
                       if int(sn) in ingroup_set]
        self.truth_at_ingroup_mrca = {}
        self._panel_root_truth = {}
        for v in self.ts.variants():
            anc = v.site.ancestral_state
            pos = int(v.site.position)
            self.sfs_bin[pos] = sum(1 for j in ingroup_idx
                                    if v.alleles[v.genotypes[j]] != anc)
            self.truth_by_pos[pos] = anc
            self.n_alleles_per_pos[pos] = len(
                {a for a in v.alleles if a not in ("", None)})
            self.mut_count_by_pos[pos] = len(v.site.mutations)
        for tree in self.ts.trees():
            for site in tree.sites():
                self.truth_at_ingroup_mrca[int(site.position)] = _truth_at_node(
                    site, tree, ingroup_nodes,
                )


def _chunk_scenario(scenario_key: str, chunk_idx: int) -> Scenario:
    """Build + load a Scenario for one chunk of a chunked scenario.

    Each msprime chunk is an independent replicate sharing the same sample /
    population layout, so the existing ``Scenario.run_*`` methods work on it
    verbatim. The per-scenario ``mu`` (elevated for cpg_hypermut) is inherited
    from the logical scenario's SCENARIOS entry. ``slim`` is a SLiM forward-sim
    chunk, loaded through :class:`SlimScenario` (same API, different files).
    """
    if scenario_key == "slim":
        sc = SlimScenario(key="slim", label="Purifying",
                          trees_path=DATA, meta_path=DATA)
        sc.chunk_idx = chunk_idx
        sc.load()
        return sc
    stem = _MSP_STEM[scenario_key]
    base = _SCENARIO_BY_KEY[scenario_key]
    sc = Scenario(
        key=scenario_key, label=base.label,
        trees_path=DATA / f"{stem}_chunk{chunk_idx}.trees",
        meta_path=DATA / f"{stem}_chunk{chunk_idx}_meta.json",
        mu=base.mu,
    )
    sc.chunk_idx = chunk_idx  # keys the per-chunk fixed-tree MAP cache
    sc.load()
    return sc


def _chunk_sim_trees_path(scenario_key: str, chunk_idx: int) -> Path:
    """Ground-truth sim trees backing a chunk. Used to detect stale derived
    caches (ftmap / singerseg / intersectmask): a regenerated sim has a newer
    mtime than caches computed off the old realization."""
    if scenario_key == "slim":
        return DATA / f"slim_outgroup_bias10_fdel1.0_chunk{chunk_idx}.trees"
    return DATA / f"{_MSP_STEM[scenario_key]}_chunk{chunk_idx}.trees"


def _cache_fresh(cache_path: Path, scenario_key: str, chunk_idx: int,
                 *, also_newer_than=()) -> bool:
    """True iff ``cache_path`` is at least as new as everything it derives from.

    Guards every per-chunk sidecar cache so that regenerating a scenario
    invalidates the derived fixed-tree MAP, budget window and intersection mask
    instead of silently reusing a realization that no longer matches.

    :param also_newer_than: Extra paths the cache must post-date. The
        intersection mask is derived from the persisted ARGs, not from the sim
        alone, and checking only the sim let re-inferred ARGs leave a mask that
        was too WIDE: score_arg_topology then silently drops the positions past
        each tool's reach, so the methods are graded on different site sets --
        exactly what the mask exists to prevent. Measured on baseline chunk 8
        n_out=1, the cached right edge was 215174 against a recomputed 214848.

    :return: Whether the cache may be reused. If a path cannot be resolved,
        falls back to plain existence.
    """
    if not cache_path.exists():
        return False
    try:
        mt = cache_path.stat().st_mtime
        sim = _chunk_sim_trees_path(scenario_key, chunk_idx)
        if sim.exists() and mt < sim.stat().st_mtime:
            return False
        for dep in also_newer_than:
            dep = Path(dep)
            if dep.exists() and mt < dep.stat().st_mtime:
                return False
        return True
    except (KeyError, OSError):
        return True


# ------------------------------------------------ SINGER sharding (SLURM path)
# SINGER is a full Bayesian ARG MCMC, orders of magnitude slower than tsinfer:
# a whole dense chunk (millions of segregating sites) is infeasible, so each
# chunk is SUBSAMPLED to one genome window holding ~SINGER_PANEL_SITE_BUDGET
# ingroup-segregating sites (chunks are i.i.d. replicates, so a prefix window is
# a fair subsample), fanned out over SINGER_N_CHAINS independent MCMC chains
# (seeds averaged as extra posterior-ARG draws). Every other method is held to
# the same window (method_windows), so the runtime strip compares measured wall
# times on one span.
SINGER_PANEL_SITE_BUDGET = int(os.environ.get("SINGER_PANEL_BUDGET", "5000"))
# Read from _singer_seeds so the DAG's chooser and this merge cannot
# disagree about how many chains a chunk has.
from _singer_seeds import N_CHAINS as SINGER_N_CHAINS  # noqa: E402


def ftmap_cache_path(scenario_key: str, chunk_idx: int, n_out: int,
                     window=None) -> Path:
    """Sidecar JSON holding a chunk's fixed-tree MAP calls, shared by every
    ARG mode so the fit runs once per key rather than once per shard.

    The key does not include the window, so one path holds one fit: pass the
    method's budget window, since a windowed and a whole-chunk fit differ in
    both the site set and ``n_target_sites``.

    :param scenario_key: Scenario the chunk belongs to.
    :param chunk_idx: Chunk index.
    :param n_out: Outgroups in the panel.
    :param window: The caller's budget window, required for the reason above.
    :return: Path to the sidecar.
    """
    return DATA / f"ftmap_{scenario_key}_chunk{chunk_idx}_n{n_out}.json"


def singer_seg_cache_path(scenario_key: str, chunk_idx: int) -> Path:
    """Sidecar JSON caching a chunk's ingroup-segregating site count + length."""
    return DATA / f"singerseg_{scenario_key}_chunk{chunk_idx}.json"


def _write_json_atomic(path, payload) -> None:
    """Write ``payload`` as JSON through a temp file in the same directory.

    Sibling shards under the SLURM benchmark can compute the same key at once,
    so a reader must never see a half-written file. ``write_text`` truncates
    first, which leaves exactly that on a kill or a concurrent write.

    :param path: Destination path.
    :param payload: JSON-serialisable object.
    """
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def _chunk_seg_and_length(scenario_key: str, chunk_idx: int) -> tuple[int, float]:
    """(ingroup-segregating site count, sequence length) for a chunk, cached to
    a sidecar JSON. Computing the seg count reloads the whole tree sequence
    (~30 s), so it is only done on a cache miss. Works for the SLiM forward-sim
    chunks too (``_chunk_scenario`` resolves them via :class:`SlimScenario`),
    which have no msprime meta JSON."""
    cache = singer_seg_cache_path(scenario_key, chunk_idx)
    if _cache_fresh(cache, scenario_key, chunk_idx):
        d = json.loads(cache.read_text())
        if "length" in d:
            return int(d["n_ingroup_seg"]), float(d["length"])
    sc = _chunk_scenario(scenario_key, chunk_idx)
    keep = [sc.sample_id_to_node[s] for s in sc.ingroup_names]
    n = int(sc.ts.simplify(samples=keep, filter_sites=True).num_sites)
    L = float(sc.ts.sequence_length)
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cache, {"n_ingroup_seg": n, "length": L})
    return n, L


def singer_n_windows(scenario_key: str, chunk_idx: int) -> int:
    """One budget-subsample window per chunk (see the module comment). Kept as a
    function so the shard-merge loop and the DAG enumeration agree."""
    return 1


def singer_windows(scenario_key: str, chunk_idx: int,
                   budget: int | None = None) -> list[tuple[int, int]]:
    """The single budget-subsample window for a chunk: an absolute
    ``(0, end)`` prefix covering ~``SINGER_PANEL_SITE_BUDGET`` ingroup-
    segregating sites (the whole chunk when it has fewer). Sites are ~uniform
    along the sequence, so the prefix fraction ``budget / seg_sites`` holds
    ~that many SNPs."""
    budget = int(budget or SINGER_PANEL_SITE_BUDGET)
    n_seg, L = _chunk_seg_and_length(scenario_key, chunk_idx)
    frac = min(1.0, budget / n_seg) if n_seg else 1.0
    return [(0, int(round(L * frac)))]


def method_windows(scenario_key: str, method_key: str,
                   chunk_idx: int) -> list[tuple[int, int]]:
    """The inference/scoring window for a method's shards.

    Every method infers on the same ``singer_windows`` budget-subsample prefix,
    so the wall times in the runtime strip are measured on one span and are
    directly comparable without rescaling. SINGER's MCMC is what makes a whole
    chunk infeasible and therefore sets the budget. The cheaper tools are held
    to it so the comparison is like for like.
    """
    return singer_windows(scenario_key, chunk_idx)


# --------------------------------------------------------------------------
# Coverage mask (cross-method intersection figure).
#
# The scoring loops consult an optional per-Scenario keep-mask (set_keep_intervals
# / _pos_kept) so a method can be graded on a restricted genomic region. This
# backs the ARG-comparison intersection figure, where every method is scored on
# the region they all cover (see intersect_mask_for_chunk / the section comment
# there). Intervals are kept in merged, sorted, non-overlapping [lo, hi) form.

def _merge_intervals(ivals) -> list[tuple[int, int]]:
    """Sort and merge overlapping/adjacent ``[lo, hi)`` intervals."""
    xs = sorted((int(a), int(b)) for a, b in ivals if int(b) > int(a))
    out: list[list[int]] = []
    for lo, hi in xs:
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def _pos_in_intervals(intervals):
    """A fast ``pos -> bool`` membership predicate for a keep-mask (``None`` ->
    accept every position). Shared by the SLiM scoring functions, which build
    records module-side rather than through :meth:`Scenario._pos_kept`."""
    if intervals is None:
        return lambda pos: True
    merged = _merge_intervals(intervals)
    los = [lo for lo, _ in merged]
    his = [hi for _, hi in merged]

    def keep(pos: int) -> bool:
        i = bisect.bisect_right(los, pos) - 1
        return i >= 0 and pos < his[i]

    return keep


def arg_shard_dir(tool: str, scenario_key: str, method_key: str, n_out: int,
                  chunk_idx: int, window_idx: int, seed: int) -> Path:
    """Directory holding one inference shard's persisted ARG draws
    (``draw_{k}.trees`` + ``meta.json``) --- the inference step's output, which
    :mod:`score_arg_shard` re-scores without re-inference. ``tool`` is
    ``"relate"`` or ``"singer"``. A single-genealogy tool writes one draw."""
    n_out = singer_shard_n_out(method_key, n_out)
    return DATA / (f"robustness_arg_{tool}_{scenario_key}_{method_key}"
                   f"_n{n_out}_chunk{chunk_idx}_w{window_idx}_s{seed}")


def arg_shard_seeds(tool: str, scenario_key: str, method_key: str, n_out: int,
                    chunk_idx: int, window_idx: int) -> list[int]:
    """The MCMC chain seeds whose persisted ARG draws are on disk for one shard.

    Read from the filesystem rather than assumed to be ``range(n_chains)``,
    because a chain re-run after a failure carries the seed it was re-run
    under. A seed counts only when its directory holds ``meta.json``, which the
    inference writes after the last draw: an aborted or cancelled chain leaves
    the directory behind without it, and contributes nothing to the posterior
    average.
    """
    import glob
    import re
    stem = arg_shard_dir(tool, scenario_key, method_key, n_out, chunk_idx,
                         window_idx, 0)
    pattern = str(stem).rsplit("_s", 1)[0] + "_s*"
    seeds = []
    for p in glob.glob(pattern):
        m = re.search(r"_s(\d+)$", p)
        if m is not None and os.path.exists(os.path.join(p, "meta.json")):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def tsinfer_arg_dir(scenario_key: str, n_out: int, chunk_idx: int) -> Path:
    """Directory holding one chunk's persisted tsinfer ARG (``draw_0.tsz`` +
    ``meta.json``). tsinfer is deterministic and runs on the chunk's single
    budget window, so its ARG path is keyed only by
    ``(scenario, n_out, chunk)``."""
    return DATA / f"robustness_arg_tsinfer_{scenario_key}_n{n_out}_chunk{chunk_idx}"


def singer_shard_n_out(method_key: str, n_out: int) -> int:
    """The n_out a shard is keyed on. Both panel methods feed the outgroups into
    SINGER's sample panel, so the posterior genuinely depends on n_out --- every
    column keeps its own n_out (identity)."""
    return int(n_out)


def singer_shard_path(scenario_key: str, method_key: str, n_out: int,
                      chunk_idx: int, window_idx: int, seed: int) -> Path:
    """Per-shard posterior JSON path (one genome sub-window x one MCMC chain)."""
    n_out = singer_shard_n_out(method_key, n_out)
    return DATA / (f"robustness_singer_shard_{scenario_key}_{method_key}"
                   f"_n{n_out}_chunk{chunk_idx}_w{window_idx}_s{seed}.json")


def _shard_inference_runtime(tool: str, scenario_key: str, method_key: str,
                             n_out: int, chunk_idx: int, w: int,
                             seed: int) -> "float | None":
    """The tool's own wall time for one shard, from the posterior JSON the
    inference step wrote, falling back to the ARG directory's ``meta.json``.

    :return: Seconds, or ``None`` when neither sidecar records it, so a caller
        can count the shards whose cost is unknown instead of adding a zero."""
    path_of = singer_shard_path if tool == "singer" else relate_shard_path
    p = path_of(scenario_key, method_key, n_out, chunk_idx, w, seed)
    if p.exists():
        rt = json.loads(p.read_text()).get("runtime")
        if rt is not None:
            return float(rt)
    meta = arg_shard_dir(tool, scenario_key, method_key, n_out, chunk_idx,
                         w, seed) / "meta.json"
    if meta.exists():
        rt = json.loads(meta.read_text()).get("runtime")
        if rt is not None:
            return float(rt)
    return None


def _merge_arg_shards(tool: str, scenario_key: str, method_key: str,
                      n_out: int, chunk_idx: int, *, seeds, burn_in: int,
                      keep_intervals) -> dict:
    """Per-chunk summary for a persisted-ARG tool, scored at the node each site
    class is graded at.

    The topologies are re-scored from the persisted draws
    (:func:`arg_shard_dir`) rather than read back from the shard posteriors,
    which were computed at the panel root and cannot answer the ingroup-MRCA
    question. Posteriors are averaged across seeds (the marginalisation over the
    ARG posterior) and concatenated across windows.

    :param tool: ``"singer"`` or ``"relate"``.
    :param burn_in: MCMC draws to drop per shard; ``0`` for a deterministic tool.
    :param keep_intervals: Cross-method coverage mask, or ``None``.
    """
    import _singer
    acc: dict[int, np.ndarray] = {}
    cnt: dict[int, int] = {}
    obs_by_pos: dict[int, list] = {}
    infer_s, score_s, n_shards, n_rt_missing = 0.0, 0.0, 0, 0
    sc = _chunk_scenario(scenario_key, chunk_idx)
    sc.set_keep_intervals(keep_intervals)
    try:
        windows = method_windows(scenario_key, method_key, chunk_idx)
        for w, window in enumerate(windows):
            found = arg_shard_seeds(tool, scenario_key, method_key, n_out,
                                    chunk_idx, w)
            # Identity, not count. Counting alone passes when a chain aborted
            # and left an empty directory behind while some other seed happened
            # to run: the shard then scores chains the DAG never selected,
            # silently, and the cell is built from a different posterior than
            # the one it claims. Scoring the DAG's selection rather than
            # whatever is on disk makes that impossible. Seeds left over from
            # an earlier run are simply not read.
            missing = sorted(set(seeds) - set(found))
            if missing:
                raise FileNotFoundError(
                    f"{scenario_key} {method_key} n_out={n_out} "
                    f"chunk={chunk_idx} window={w}: the DAG selected {tool} "
                    f"chains {sorted(seeds)} but {missing} have no persisted "
                    f"draws (on disk: {found}). A shard averaged over fewer "
                    "chains carries a noisier posterior than the cells it is "
                    "compared against, so re-run the missing chains, or if "
                    "they abort, add them to workflow/singer_aborting_seeds.txt "
                    "so the DAG stops choosing them, rather than scoring this "
                    "shard short.")
            # Every chain's post-burn-in draws are pooled and scored in ONE
            # call, so the chains are combined the same way the draws within a
            # chain are: in likelihood space, prior applied once. Averaging the
            # per-chain posteriors here instead would reintroduce equal
            # weighting one level up. All chains of a shard share window w, so
            # they share the coordinate offset.
            pooled: list = []
            for seed in sorted(seeds):
                arg_dir = arg_shard_dir(tool, scenario_key, method_key, n_out,
                                        chunk_idx, w, seed)
                draws = _singer.load_args(str(arg_dir))
                n_shards += 1
                rt = _shard_inference_runtime(tool, scenario_key, method_key,
                                              n_out, chunk_idx, w, seed)
                if rt is None:
                    n_rt_missing += 1
                else:
                    infer_s += rt
                # burn_in is per chain, so it is applied before pooling.
                pooled.extend(draws[int(burn_in):])
            if pooled:
                post, obs, dt = sc.score_arg_shard_split(
                    pooled, n_out, window=window, burn_in=0)
                score_s += dt
                obs_by_pos.update(obs)
                for pos, vals in post.items():
                    # Accumulated, not assigned: windows are disjoint so a
                    # position is normally seen once, and this keeps the
                    # original behaviour if that ever stops holding.
                    v = np.asarray(vals, float)
                    acc[pos] = acc.get(pos, np.zeros(len(STATES))) + v
                    cnt[pos] = cnt.get(pos, 0) + 1
        avg = {pos: acc[pos] / cnt[pos] for pos in acc}
        records = sc.score_singer_posteriors(avg, obs_by_pos, n_out=n_out)
    finally:
        sc.set_keep_intervals(None)
    # Wall time to produce this method's calls: the tool's own inference plus
    # the kernel pass that reads the ancestral state off its topologies, so the
    # runtime strip compares like with like against the rows (true-ARG,
    # local-tree) whose only cost is that same kernel pass.
    summary = summarize_persite(
        records, scenario=scenario_key, method=method_key, n_out=n_out,
        n_ingroup=_cell_n_ingroup(scenario_key), runtime=infer_s + score_s,
    )
    summary["runtime_parts"] = {
        "inference_s": infer_s, "scoring_s": score_s,
        "n_shards": n_shards, "n_shards_runtime_missing": n_rt_missing,
    }
    return summary


def singer_merge_chunk_summary(scenario_key: str, method_key: str, n_out: int,
                               chunk_idx: int, *, seeds=None,
                               keep_intervals=None, burn_in=None) -> dict:
    """Merge SINGER shards for one chunk into a per-chunk summary in the SAME
    format as every other per-chunk summary (so the existing cell concat picks
    it up). See :func:`_merge_arg_shards`.

    :param keep_intervals: optional cross-method coverage mask (see
        :func:`intersect_mask_for_chunk`). Restricts scoring to the common
        covered region for the intersection figure."""
    # The same chooser the DAG names its inputs with. Taking
    # range(SINGER_N_CHAINS) here instead made the merge ask for chains the DAG
    # had never built as soon as any seed was recorded as aborting.
    from _singer_seeds import runnable_seeds
    seeds = (runnable_seeds(scenario_key, method_key, n_out, chunk_idx,
                            SINGER_N_CHAINS)
             if seeds is None else list(seeds))
    # Supplied by the rule, which is what keeps the shard, chunk and
    # intersection paths averaging over the same draws. A SLURM job does not
    # necessarily inherit SINGER_BURN_IN, so the environment is only the
    # standalone-debug fallback.
    if burn_in is None:
        burn_in = int(os.environ.get("SINGER_BURN_IN", "5"))
    return _merge_arg_shards("singer", scenario_key, method_key, n_out,
                             chunk_idx, seeds=seeds, burn_in=int(burn_in),
                             keep_intervals=keep_intervals)


# ------------------------------------------------ Relate sharding (SLURM path)
# Relate runs on the same budget window as every other method (method_windows),
# so the runtime strip compares measured wall times on one span. Relate is
# deterministic given a seed, so a single "chain" suffices (no posterior-ARG
# average), and the shard enumeration (singer_n_windows = 1) is shared.
RELATE_N_CHAINS = int(os.environ.get("RELATE_N_CHAINS", "1"))


def relate_shard_path(scenario_key: str, method_key: str, n_out: int,
                      chunk_idx: int, window_idx: int, seed: int) -> Path:
    """Per-shard posterior JSON path (one genome sub-window x one Relate run)."""
    n_out = singer_shard_n_out(method_key, n_out)
    return DATA / (f"robustness_relate_shard_{scenario_key}_{method_key}"
                   f"_n{n_out}_chunk{chunk_idx}_w{window_idx}_s{seed}.json")


def relate_merge_chunk_summary(scenario_key: str, method_key: str, n_out: int,
                               chunk_idx: int, *, seeds=None,
                               keep_intervals=None) -> dict:
    """Merge Relate shards for one chunk into a per-chunk summary in the SAME
    format as every other per-chunk summary. Relate is deterministic, so its
    seeds are not a posterior average but are kept for parity with the SINGER
    path. See :func:`_merge_arg_shards`.

    :param keep_intervals: optional cross-method coverage mask (see
        :func:`intersect_mask_for_chunk`)."""
    seeds = list(range(RELATE_N_CHAINS)) if seeds is None else list(seeds)
    return _merge_arg_shards("relate", scenario_key, method_key, n_out,
                             chunk_idx, seeds=seeds, burn_in=0,
                             keep_intervals=keep_intervals)


def tsinfer_chunk_summary(scenario_key: str, n_out: int, chunk_idx: int,
                          *, keep_intervals=None) -> dict:
    """Score one chunk's persisted tsinfer dated ARG into a per-chunk summary ---
    the reusable core of ``score_tsinfer_chunk.py``. Lays the panel's full site
    set on tsinfer's local trees at the fitted (dated) rate from meta.json.

    :param keep_intervals: optional cross-method coverage mask (see
        :func:`intersect_mask_for_chunk`)."""
    import _singer
    arg_dir = tsinfer_arg_dir(scenario_key, n_out, chunk_idx)
    draws = _singer.load_args(str(arg_dir))
    meta_p = arg_dir / "meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    runtime = float(meta.get("runtime", 0.0))
    mu = meta.get("mu")  # fitted (dated) tsinfer rate; None -> sc.mu
    window = method_windows(scenario_key, "tsinfer_arg", chunk_idx)[0]
    sc = _chunk_scenario(scenario_key, chunk_idx)
    sc.set_keep_intervals(keep_intervals)
    try:
        score_s = 0.0
        if not draws:
            records: PerSite = []
        else:
            post, obs, score_s = sc.score_arg_shard_split(
                draws, n_out, window=window, mu=mu,
                mu_matches_time_units=True)
            records = sc.score_singer_posteriors(post, obs, n_out=n_out)
    finally:
        sc.set_keep_intervals(None)
    summary = summarize_persite(
        records, scenario=scenario_key, method="tsinfer_arg", n_out=n_out,
        n_ingroup=_cell_n_ingroup(scenario_key), runtime=runtime + score_s)
    summary["runtime_parts"] = {
        "inference_s": runtime, "scoring_s": score_s,
        "n_shards": 1 if draws else 0,
        "n_shards_runtime_missing": 0 if meta.get("runtime") is not None else 1,
    }
    return summary


# ------------------------------------------- cross-method intersection figure
# Every figure method infers on the one budget window (method_windows), sized by
# what SINGER's MCMC can afford, and every method is graded on the identical
# in-window site set. So the comparison is like-for-like twice over: no method
# is scored on a wider or easier span, and no method's wall time covers a span
# another never saw, which is what lets the runtime strip report measured
# seconds rather than seconds rescaled to a notional whole genome.
def intersect_mask_cache_path(scenario_key: str, chunk_idx: int,
                              n_out: int) -> Path:
    """Sidecar JSON caching a chunk's cross-method coverage intersection so the
    persisted ARGs are loaded once per ``(scenario, chunk, n_out)`` rather than
    once per figure method."""
    return DATA / f"intersectmask_{scenario_key}_chunk{chunk_idx}_n{n_out}.json"


def _persisted_arg_right_edge(dirpath: "Path | str") -> "float | None":
    """Right edge of a persisted ARG's covered span (its ``sequence_length`` in
    the 0-based window frame the tools infer in), or ``None`` when no draw is on
    disk. Only the first draw is decompressed --- every draw of a shard shares
    the tool's coordinate frame."""
    import glob
    import tszip
    paths = glob.glob(os.path.join(str(dirpath), "draw_*.tsz"))
    if not paths:
        return None
    return float(tszip.decompress(paths[0]).sequence_length)


def intersect_mask_for_chunk(scenario_key: str, chunk_idx: int,
                             n_out: int) -> list[tuple[int, int]]:
    """The cross-method common covered region for a chunk.

    Every method infers on the budget window (:func:`method_windows`), so the
    region is that window narrowed to where each tool's persisted ARG actually
    reaches, and for SINGER to where every averaged chain reaches: Relate stops
    at its last panel-segregating site, and SINGER's chains stop at their own
    positions short of the window. Every method is then graded on the identical
    in-region site set (monomorphic columns included, via
    :meth:`score_arg_shard`'s ``keep_all_sites`` path). Cached to a sidecar so
    the ARGs are loaded once, not once per method."""
    cache = intersect_mask_cache_path(scenario_key, chunk_idx, n_out)
    # The mask is derived from the ARGs, so it must post-date them. Checking
    # only the sim let a re-inferred ARG leave a mask that was too wide.
    deps = [tsinfer_arg_dir(scenario_key, n_out, chunk_idx) / "meta.json",
            arg_shard_dir("relate", scenario_key, "relate_arg_panel_ft",
                          n_out, chunk_idx, 0, 0) / "meta.json"]
    deps += [arg_shard_dir("singer", scenario_key, "singer_arg_panel", n_out,
                           chunk_idx, 0, s) / "meta.json"
             for s in arg_shard_seeds("singer", scenario_key,
                                      "singer_arg_panel", n_out, chunk_idx, 0)]
    if _cache_fresh(cache, scenario_key, chunk_idx, also_newer_than=deps):
        return [tuple(iv) for iv in json.loads(cache.read_text())["mask"]]
    hi = float(singer_windows(scenario_key, chunk_idx)[0][1])
    for tool, method in (("relate", "relate_arg_panel_ft"),
                         ("tsinfer", "tsinfer_arg")):
        if tool == "tsinfer":
            arg_dir = tsinfer_arg_dir(scenario_key, n_out, chunk_idx)
        else:
            arg_dir = arg_shard_dir(tool, scenario_key, method, n_out,
                                    chunk_idx, 0, 0)
        edge = _persisted_arg_right_edge(arg_dir)
        if edge is None:
            raise FileNotFoundError(
                f"{method} has no persisted ARG at {arg_dir}, so the "
                f"cross-method common covered set for {scenario_key} chunk "
                f"{chunk_idx} n_out={n_out} cannot be formed; every other "
                f"method would be graded on a wider span than this one.")
        hi = min(hi, edge)
    # SINGER's MCMC stops short of its own budget window on some chunks, and
    # each chain stops in a different place, so the region every chain of the
    # posterior average covers is the minimum over the chains present.
    singer_edges = [
        e for s in arg_shard_seeds("singer", scenario_key, "singer_arg_panel",
                                   n_out, chunk_idx, 0)
        if (e := _persisted_arg_right_edge(
            arg_shard_dir("singer", scenario_key, "singer_arg_panel", n_out,
                          chunk_idx, 0, s))) is not None]
    if not singer_edges:
        # Silently leaving the window unnarrowed is the one outcome this whole
        # mechanism exists to prevent: every other method would then be graded
        # on a wider site set than SINGER's, which is not a comparison. The
        # ARGs are not declared inputs of the intersect cells, so a purge or a
        # fresh checkout reaches here with nothing on disk.
        raise FileNotFoundError(
            f"no persisted SINGER ARG for scenario={scenario_key!r} "
            f"n_out={n_out} chunk={chunk_idx}; the intersection mask is "
            f"defined by SINGER's coverage, so scoring without it would "
            f"silently grade every other method on a wider site set. Re-run "
            f"the singer_arg_panel shards for this chunk.")
    hi = min(hi, min(singer_edges))
    mask = [(0, int(hi))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cache, {"mask": [list(iv) for iv in mask]})
    return mask


def intersect_chunk_summary(scenario_key: str, method_key: str, n_out: int,
                            chunk_idx: int, *, burn_in=None) -> dict:
    """One figure method's per-chunk summary scored ONLY on the common covered
    region (:func:`intersect_mask_for_chunk`). Re-uses the persisted ARGs / shard
    posteriors --- no re-inference: SINGER/Relate re-score their persisted shard
    posteriors, tsinfer its persisted dated ARG, and the true-ARG / local-tree
    modes re-run their inference-free kernel scoring, all under the mask."""
    n_ingroup = _cell_n_ingroup(scenario_key)
    if n_out > _scenario_n_outgroups(scenario_key):
        return empty_summary(scenario_key, method_key, n_out, n_ingroup)
    mask = intersect_mask_for_chunk(scenario_key, chunk_idx, n_out)
    if method_key in ("singer_arg_panel", "singer_arg_panel_ft"):
        return singer_merge_chunk_summary(
            scenario_key, method_key, n_out, chunk_idx, keep_intervals=mask,
            burn_in=burn_in)
    if method_key == "relate_arg_panel_ft":
        return relate_merge_chunk_summary(
            scenario_key, method_key, n_out, chunk_idx, keep_intervals=mask)
    if method_key == "tsinfer_arg":
        return tsinfer_chunk_summary(
            scenario_key, n_out, chunk_idx, keep_intervals=mask)
    if method_key in _METHOD_DISPATCH and method_key not in _PERSISTED_ARG_METHODS:
        # Everything not persisted above is re-scored live from the true trees /
        # outgroup ladder under the mask. Route EVERY scenario (SLiM included)
        # through the same Scenario the panel-ARG methods use, so these rows are
        # graded on the identical site set and classification. The alternative
        # `_slim_chunk_arg_live` path classifies on the n_out-outgroup panel and
        # drops monomorphic sites (filter_sites), diverging from the panel-ARG
        # methods, which score every in-panel site and classify on
        # SlimScenario's full sample set. SlimScenario.run_arg /
        # .run_local_tree_unphased reproduce the panel-ARG site set + classes
        # exactly (verified), so the SLiM intersection cells match the msprime
        # ones.
        sc = _chunk_scenario(scenario_key, chunk_idx)
        if n_out > len(sc.all_outgroup_names):
            return empty_summary(scenario_key, method_key, n_out, n_ingroup)
        sc.set_keep_intervals(mask)
        try:
            recs, dt = getattr(sc, _METHOD_DISPATCH[method_key])(n_out)
        finally:
            sc.set_keep_intervals(None)
        return summarize_persite(recs, scenario=scenario_key, method=method_key,
                                 n_out=n_out, n_ingroup=n_ingroup, runtime=dt)
    raise ValueError(f"not an intersection figure method: {method_key}")


def intersect_chunk_summary_path(scenario_key: str, method_key: str,
                                 n_out: int, chunk_idx: int) -> Path:
    """Where one chunk's intersection summary is persisted."""
    return (DATA / f"robustness_chunk_intersect_{scenario_key}_{method_key}"
                   f"_n{n_out}_chunk{chunk_idx}_summary.json")


def intersect_cell_summary(scenario_key: str, method_key: str,
                           n_out: int, *, from_shards: bool = False) -> dict:
    """Cell summary on the cross-method common site set = element-wise sum of the
    per-chunk intersection summaries (additive, like every other cell).

    :param from_shards: Read the per-chunk summaries the sharded rule wrote
        rather than recomputing them here. The rule fans the chunks out, so the
        cell job only sums. Recomputing in one process is what made this a
        multi-hour single-core job.
    """
    if from_shards:
        parts = [json.loads(intersect_chunk_summary_path(
            scenario_key, method_key, n_out, c).read_text())
            for c in scored_chunks(scenario_key, method_key, n_out)]
    else:
        parts = [intersect_chunk_summary(scenario_key, method_key, n_out, c)
                 for c in scored_chunks(scenario_key, method_key, n_out)]
    merged = merge_summaries(parts)
    if not merged:
        return empty_summary(scenario_key, method_key, n_out,
                             _cell_n_ingroup(scenario_key))
    merged["scenario"] = scenario_key
    merged["method"] = method_key
    merged["n_out"] = int(n_out)
    merged["n_ingroup"] = _cell_n_ingroup(scenario_key)
    return merged


def intersect_cell_summary_path(scenario_key: str, method_key: str,
                                n_out: int) -> Path:
    return (DATA / f"robustness_cell_intersect_{scenario_key}_{method_key}"
                   f"_n{n_out}_summary.json")


def compute_scenario_chunk(
    scenario_key: str, method_key: str, n_out: int, chunk_idx: int,
) -> tuple[PerSite, float]:
    """One (scenario, method, n_out) result for a single msprime chunk.

    Used by the per-chunk inference jobs (score_robustness_chunk.py). Returns
    an empty list when n_out exceeds the chunk's available outgroups (so the
    aggregated cell is simply blank, matching the monolithic guard)."""
    sc = _chunk_scenario(scenario_key, chunk_idx)
    if n_out > len(sc.all_outgroup_names):
        return [], 0.0
    return getattr(sc, _METHOD_DISPATCH[method_key])(n_out)


def _get_scenario(scenario_key: str) -> Scenario:
    """Return the (lazily loaded) Scenario for ``scenario_key``."""
    sc = _SCENARIO_BY_KEY[scenario_key]
    if sc.key != "slim" and sc.ts is None:
        sc.load()
    return sc


def _chunked_msp_meta0(scenario_key: str) -> dict:
    """chunk-0 meta for a chunked msprime scenario (all chunks share the
    sample/population layout)."""
    p = DATA / f"{_MSP_STEM[scenario_key]}_chunk0_meta.json"
    return json.loads(p.read_text())


def _scenario_n_outgroups(scenario_key: str) -> int:
    """Available outgroups for a scenario, read cheaply from its meta JSON
    (no tree-sequence load)."""
    if scenario_key == "slim":
        return len(_slim_chunk_meta(0)["outgroup_names"])
    if scenario_key in CHUNKED_MSP_SCENARIOS:
        meta = _chunked_msp_meta0(scenario_key)
        return sum(len(v) for v in meta["outgroups_by_pop"].values())
    sc = _SCENARIO_BY_KEY[scenario_key]
    meta = json.loads(sc.meta_path.read_text())
    return sum(len(v) for v in meta["outgroups_by_pop"].values())


def _cell_n_ingroup(scenario_key: str) -> int:
    """Ingroup size from a scenario's meta JSON (no tree-sequence load)."""
    if scenario_key == "slim":
        return len(_slim_chunk_meta(0)["ingroup_names"])
    if scenario_key in CHUNKED_MSP_SCENARIOS:
        return len(_chunked_msp_meta0(scenario_key)["ingroup_names"])
    sc = _SCENARIO_BY_KEY[scenario_key]
    meta = json.loads(sc.meta_path.read_text())
    return len(meta["ingroup_names"])


def chunk_summary_path(scenario_key: str, method_key: str, n_out: int,
                       chunk_idx: int) -> Path:
    return (DATA / f"robustness_chunk_{scenario_key}_{method_key}"
                   f"_n{n_out}_chunk{chunk_idx}_summary.json")


def cell_summary_path(scenario_key: str, method_key: str, n_out: int) -> Path:
    return DATA / f"robustness_cell_{scenario_key}_{method_key}_n{n_out}_summary.json"


# Methods whose inferred ARG is persisted per chunk (SINGER, Relate, tsinfer):
# on SLiM these run through the same per-chunk shard/arg rules as the msprime
# scenarios (not the live slim_aggregate loop), so their cell is a concat of the
# per-chunk summaries. The non-inferred SLiM rows (true-ARG anc_arg, local-tree,
# parsimony, majority) and the pooled fixed-tree fit stay in slim_aggregate.
_PERSISTED_ARG_METHODS = {
    "singer_arg_panel", "singer_arg_panel_ft",
    "relate_arg_panel", "relate_arg_panel_ft",
    "tsinfer_arg",
}


def scored_chunks(scenario_key: str, method_key: str, n_out: int) -> list[int]:
    """Chunk indices a cell is summed over, dropping the uninferable ones.

    Delegates to ``workflow/scripts/_singer_seeds.py``, which the DAG also
    reads, so the rule's inputs and the merge agree on the chunk set.

    :return: Ascending chunk indices.
    """
    import _singer_seeds
    return _singer_seeds.scored_chunks(
        scenario_key, method_key, n_out, n_chunks=_n_chunks(scenario_key))


def _n_chunks(scenario_key: str) -> int:
    """Number of independent chunks for a scenario (SLiM has 10x the msprime
    count for smoother spectra)."""
    return SLIM_N_CHUNKS if scenario_key == "slim" else ROBUSTNESS_MSP_N_CHUNKS


def chunk_aggregate_summary(scenario_key: str, method_key: str,
                            n_out: int) -> dict:
    """Cell summary = element-wise sum of every per-chunk summary.

    All chunks are required. A cell summed over a subset is structurally valid
    and the heatmap plots it without complaint, at a site count nothing
    compares against, so a missing chunk is an error rather than a smaller
    cell. The intersection path asserts the same invariant.

    :raises FileNotFoundError: If any chunk summary is absent.
    """
    parts, missing = [], []
    for c in scored_chunks(scenario_key, method_key, n_out):
        p = chunk_summary_path(scenario_key, method_key, n_out, c)
        if p.exists():
            parts.append(json.loads(p.read_text()))
        else:
            missing.append(c)
    if missing:
        raise FileNotFoundError(
            f"{scenario_key} {method_key} n_out={n_out}: chunk summaries "
            f"missing for chunks {missing}; a cell summed over a subset is "
            f"not comparable with the others"
        )
    merged = merge_summaries(parts)
    if not merged:
        merged = empty_summary(scenario_key, method_key, n_out,
                               _cell_n_ingroup(scenario_key))
    else:
        merged["scenario"] = scenario_key
        merged["method"] = method_key
        merged["n_out"] = int(n_out)
        merged["n_ingroup"] = _cell_n_ingroup(scenario_key)
    return merged


def compute_cell_summary(scenario_key: str, method_key: str,
                         n_out: int) -> dict:
    """Light per-cell summary, computed without ever materialising the full
    per-site list on disk.

    - chunked msprime: merge the 10 per-chunk summaries (the chunk jobs already
      reduced their records);
    - slim: build the (sparse, RAM-fitting) per-site list via the SLiM
      aggregator, then reduce;
    - other (single-sim) scenarios: live-compute then reduce.
    """
    n_ingroup = _cell_n_ingroup(scenario_key)
    if n_out > _scenario_n_outgroups(scenario_key):
        return empty_summary(scenario_key, method_key, n_out, n_ingroup)
    # Chunked scenarios (msprime, and SLiM's inferred-ARG rows) concat their
    # per-chunk summaries; SLiM's other rows still aggregate live.
    if scenario_key in CHUNKED_MSP_SCENARIOS or (
        scenario_key == "slim" and method_key in _PERSISTED_ARG_METHODS
    ):
        return chunk_aggregate_summary(scenario_key, method_key, n_out)
    if scenario_key == "slim":
        recs, dt = slim_aggregate_persite(method_key, n_out)
    else:
        sc = _get_scenario(scenario_key)
        recs, dt = getattr(sc, _METHOD_DISPATCH[method_key])(n_out)
        n_ingroup = sc.n_ingroup or n_ingroup
    return summarize_persite(recs, scenario=scenario_key, method=method_key,
                             n_out=n_out, n_ingroup=n_ingroup, runtime=dt)


def load_cell_summaries(paths) -> None:
    """Report path: load light per-cell summary JSONs into the globals.

    Populates ``cell_summaries`` (consumed by build_matrix / col counts) plus
    ``runtimes`` and each scenario's ``n_ingroup`` (so the runtime table and
    labels work unchanged). Peak memory is a few MB regardless of site count.
    """
    for p in paths:
        d = json.loads(Path(p).read_text())
        key = (d["scenario"], d["method"], int(d["n_out"]))
        cell_summaries[key] = d
        runtimes[key] = float(d.get("runtime", float("nan")))
        sc = _SCENARIO_BY_KEY.get(d["scenario"])
        if sc is not None and not sc.n_ingroup:
            sc.n_ingroup = int(d.get("n_ingroup", 0))


def load_summary(path) -> dict:
    """Load a single cell summary JSON."""
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------- aggregation
def _site_metric(tup, metric: str) -> float:
    """Per-site value for the chosen ``metric``.

    - ``"map"``   → MAP-hit (0/1), field 2.
    - ``"ptrue"`` → posterior mass on the truth, field 3.
    - ``"brier"`` → Brier score ``(1 − p_true)² + Σ_{i≠t} p_i²`` =
      ``1 − 2·p_true + Σ_i p_i²`` (fields 3 and 5). Lower is better.
    """
    if metric == "map":
        return float(tup[2])
    if metric == "ptrue":
        return float(tup[3])
    if metric == "brier":
        return 1.0 - 2.0 * float(tup[3]) + float(tup[5])
    raise ValueError(f"unknown metric {metric!r}")


def filtered_mean(records: PerSite, predicate, *, metric: str) -> tuple[float, int]:
    """Mean per-site ``metric`` over records satisfying
    ``predicate(bin, klass, n_mut)``."""
    s, n = 0.0, 0
    for tup in records:
        if predicate(tup[0], tup[1], tup[4]):
            val = _site_metric(tup, metric)
            if val != val:  # NaN guard (e.g. missing posterior)
                continue
            s += val
            n += 1
    if n == 0:
        return (float("nan"), 0)
    return (s / n, n)


# ----------------------------------------------------------- plot helpers
N_INGROUP_DEFAULT = SCENARIOS[0].n_ingroup if SCENARIOS[0].n_ingroup else 20

CATS = [
    ("ISM bi", lambda n_ing: (
        lambda b, k, m: 1 <= b <= n_ing - 1 and k == "biallelic_poly" and m == 1
    )),
    # Ingroup-monomorphic bins are the "fixed" column's own question and carry
    # most of the sites, so admitting them here would make this column report
    # the same thing for the bulk of its mass.
    ("≥2 mut", lambda n_ing: (
        lambda b, k, m: m > 1 and 1 <= b <= n_ing - 1
    )),
    ("i=n−1", lambda n_ing: (
        lambda b, k, m: b == n_ing - 1
    )),
    ("fixed", lambda n_ing: (
        lambda b, k, m: b == n_ing
    )),
]


# When True, all four site-class columns use one folded, imbalance-robust
# construction: score each frequency bin (ingroup derived count) separately,
# fold partner bins (i and n−i) at equal weight, then average the folds — so
# no bin's difficulty is hidden behind an abundant, easy neighbour. The two
# single-bin tail columns ("i=n−1", "fixed") additionally pull in their folded
# mirror (i=n−1 with the singleton i=1, the divergence bin i=n with the
# fixed-ancestral i=0), which their predicate would otherwise miss. On by
# default. Set False for the flat site-count mean.
BALANCE_FREQ_BINS = True

# cat-label → folded-mirror predicate factory: the partner frequency bin a
# single-bin column needs pulled in to form its fold (the broad columns
# already span both sides of each fold, so they need no entry).
_BALANCED_MIRROR = {
    "i=n−1": lambda n_ing: (lambda b, k, m: b == 1),
    "fixed": lambda n_ing: (lambda b, k, m: b == 0),
}


def _iter_stats(summary):
    """Unpack a cell summary's stats table into per-group tuples.

    :param summary: A loaded cell summary.
    :return: Iterator of ``(bin, klass, n_mut, group)``, where ``n_mut`` is 2
        for the recurrent groups and 1 otherwise.
    """
    for key, g in summary["stats"].items():
        klass, b, rec = key.split("|")
        yield int(b), klass, (2 if rec == "1" else 1), g


def summary_cat_value(summary, cat_label, pred_factory, n_ing, *,
                      metric: str) -> float:
    pred = pred_factory(n_ing)
    if not BALANCE_FREQ_BINS:
        s, n = 0.0, 0
        for b, klass, m, g in _iter_stats(summary):
            if pred(b, klass, m):
                ms, cnt = _group_metric(g, metric)
                s += ms; n += cnt
        return s / n if n else float("nan")
    preds = [pred]
    if cat_label in _BALANCED_MIRROR:
        preds.append(_BALANCED_MIRROR[cat_label](n_ing))
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for b, klass, m, g in _iter_stats(summary):
        if any(p(b, klass, m) for p in preds):
            ms, cnt = _group_metric(g, metric)
            sums[b] = sums.get(b, 0.0) + ms
            counts[b] = counts.get(b, 0) + cnt
    counts = {b: c for b, c in counts.items() if c}
    if not counts:
        return float("nan")
    # Within a fold, bin b and its mirror n-b are averaged at equal weight.
    # They are complementary classes (ingroup fixed ancestral against ingroup
    # fixed derived, and so on outwards), so equal weight is a balanced
    # accuracy: a method that always reports the observed allele scores 1 on
    # one and 0 on the other, and must come out at 0.5 rather than at the
    # majority class's share.
    #
    # Across folds, weight by site count. Folds are the same question at
    # different allele frequencies, not complementary classes, so an equal
    # weight there lets a fold of tens of sites outvote one of tens of
    # thousands and inverts the ordering in the recurrent-mutation column.
    fold_means: dict[int, list[float]] = {}
    fold_counts: dict[int, int] = {}
    for b in counts:
        f = min(b, n_ing - b)
        fold_means.setdefault(f, []).append(sums[b] / counts[b])
        fold_counts[f] = fold_counts.get(f, 0) + counts[b]
    total = sum(fold_counts.values())
    if not total:
        return float("nan")
    return sum(sum(xs) / len(xs) * fold_counts[f]
               for f, xs in fold_means.items()) / total


def summary_col_count(summary, cat_label, pred_factory, n_ing) -> int:
    preds = [pred_factory(n_ing)]
    if BALANCE_FREQ_BINS and cat_label in _BALANCED_MIRROR:
        preds.append(_BALANCED_MIRROR[cat_label](n_ing))
    return sum(g[0] for b, klass, m, g in _iter_stats(summary)
               if any(p(b, klass, m) for p in preds))


def summary_per_bin_brier(summary, n_ing_max: int):
    """Per-bin (0..n) mean Brier, matching plot_brier_per_bin.per_bin_brier."""
    bsum = [0.0] * (n_ing_max + 1)
    bcnt = [0] * (n_ing_max + 1)
    for b, klass, m, g in _iter_stats(summary):
        if 0 <= b <= n_ing_max:
            ms, cnt = _group_metric(g, "brier")
            bsum[b] += ms; bcnt[b] += cnt
    return [bsum[i] / bcnt[i] if bcnt[i] else float("nan")
            for i in range(n_ing_max + 1)]


def summary_per_bin_count(summary, n_ing_max: int):
    """Per-bin total site count (all classes), for the scenario-SFS figure."""
    out = [0] * (n_ing_max + 1)
    for b, klass, m, g in _iter_stats(summary):
        if 0 <= b <= n_ing_max:
            out[b] += g[0]
    return out


def summary_conf_filter(summary, fracs, n_ing: int):
    """Reconstruct the confidence-quantile-filtered unfolded SFS from the
    per-bin confidence histogram, matching plot_sfs_confidence_filter.

    Returns ``{frac: (threshold, sfs_proportions over i=1..n-1)}``. Sites are
    retained most-confident-first. The boundary confidence bin is included
    fractionally (spread across its derived-count distribution), approximating
    the exact per-site quantile to the histogram resolution (N_CONF_BINS)."""
    ch = summary.get("conf_hist", {})
    bins = sorted(int(b) for b in ch)
    x = list(range(1, n_ing))
    total = sum(sum(ch[str(b)]) for b in bins)
    out: dict[float, tuple] = {}
    for frac in fracs:
        target = frac * total
        kept = {b: 0.0 for b in bins}
        acc = 0.0
        thr = _CONF_LO
        for cb in range(N_CONF_BINS - 1, -1, -1):
            layer = {b: ch[str(b)][cb] for b in bins}
            layer_total = sum(layer.values())
            if layer_total == 0:
                continue
            thr = _CONF_LO + (cb / N_CONF_BINS) * (1.0 - _CONF_LO)
            if acc + layer_total <= target:
                for b in bins:
                    kept[b] += layer[b]
                acc += layer_total
            else:
                f = (target - acc) / layer_total if layer_total else 0.0
                for b in bins:
                    kept[b] += layer[b] * f
                acc = target
                break
        sfs = [kept.get(i, 0.0) for i in x]
        s = sum(sfs)
        sfs = [v / s for v in sfs] if s else sfs
        out[frac] = (thr, sfs)
    return out


# ---------------------------------------- big matrix + sns.heatmap
N_OUT = len(N_OUTS)
N_METHODS = len(METHODS)
N_SCEN = len(FIG_SCENARIOS)
N_CAT = len(CATS)

n_rows = N_METHODS * N_OUT
n_cols = N_SCEN * N_CAT

row_labels: list[str] = []
for m, (method_label, _) in enumerate(METHODS):
    for o, n_out in enumerate(N_OUTS):
        row_labels.append(rf"$n_\mathrm{{out}}={n_out}$")

col_labels: list[str] = []
for sc in FIG_SCENARIOS:
    for cat_lbl, _ in CATS:
        col_labels.append(cat_lbl)


def set_methods(methods) -> None:
    """Point the figure helpers at one figure's method rows.

    :func:`build_matrix` and the render helpers read ``METHODS`` and its derived
    globals at call time, so a report that plots a different row set selects it
    here before rendering. The row set is the only thing that distinguishes the
    heatmap figures from one another --- they share scenarios, n_out sweep,
    site classes, colour scale and runtime strip."""
    global METHODS, N_METHODS, n_rows, row_labels
    METHODS = list(methods)
    N_METHODS = len(METHODS)
    n_rows = N_METHODS * N_OUT
    row_labels = [rf"$n_\mathrm{{out}}={n_out}$"
                  for _ in METHODS for n_out in N_OUTS]


def build_matrix(metric: str) -> np.ndarray:
    mat = np.full((n_rows, n_cols), np.nan)
    for m, (_, method_key) in enumerate(METHODS):
        for o, n_out in enumerate(N_OUTS):
            r = m * N_OUT + o
            for sc_i, sc in enumerate(FIG_SCENARIOS):
                summ = cell_summaries.get((sc.key, method_key, n_out))
                if summ is None:
                    continue
                n_ing = sc.n_ingroup or N_INGROUP_DEFAULT
                for c_i, (cat_lbl, pred_factory) in enumerate(CATS):
                    mat[r, sc_i * N_CAT + c_i] = summary_cat_value(
                        summ, cat_lbl, pred_factory, n_ing, metric=metric)
    return mat


# Annotation strips (per-row median runtime, per-column site counts) are
# populated by build_report() once per_site / runtimes are filled.
row_runtimes: list[float] = []
col_counts: list[int] = []


def _compute_row_runtimes() -> None:
    """Per-row total wall time summed across the (non-SLiM) scenarios.

    SLiM is excluded — its per-cell "runtime" is just the aggregation-loop
    proxy over pre-computed JSONs, not comparable inference wall time.
    """
    global row_runtimes
    row_runtimes = []
    for _, method_key in METHODS:
        for n_out in N_OUTS:
            per_scen = [
                runtimes.get((sc.key, method_key, n_out), float("nan"))
                for sc in FIG_SCENARIOS
                if sc.key != "slim"  # SLiM aggregation time isn't comparable
            ]
            per_scen = [x for x in per_scen if not np.isnan(x)]
            row_runtimes.append(
                float(np.sum(per_scen)) if per_scen else float("nan")
            )


def _compute_col_counts() -> None:
    """Per-column absolute site counts (same site sets for both metrics)."""
    global col_counts
    col_counts = []
    for sc in FIG_SCENARIOS:
        # Site counts are method-independent (same sim), so any loaded cell for
        # the scenario serves as the reference. Prefer n_out=3, else any n_out.
        # (Hardcoding a single method broke figures that don't load it.)
        canonical = next((c for (s, _, no), c in cell_summaries.items()
                          if s == sc.key and no == 3), None)
        if canonical is None:
            canonical = next((c for (s, _, _), c in cell_summaries.items()
                              if s == sc.key), None)
        n_ing = sc.n_ingroup or N_INGROUP_DEFAULT
        for cat_lbl, pred_factory in CATS:
            col_counts.append(
                summary_col_count(canonical, cat_lbl, pred_factory, n_ing)
                if canonical is not None else 0
            )


def assert_full_coverage(methods=None, *, allow_missing=()) -> None:
    """Raise if any ``(scenario, method, n_out)`` cell the figure is about to
    plot is absent from :data:`cell_summaries`. Guards against silently blank
    heatmap cells (e.g. a stale scenario key or a not-yet-computed cell).

    :param methods: iterable of ``(label, method_key)`` to check. Defaults to
        the active :data:`METHODS`.
    :param allow_missing: ``(scenario_key, method_key, n_out)`` tuples that are
        legitimately absent and should not raise.
    :raises RuntimeError: listing every missing cell.
    """
    methods = METHODS if methods is None else methods
    missing = [
        (sc.key, mk, n)
        for _, mk in methods
        for sc in FIG_SCENARIOS
        for n in N_OUTS
        if (sc.key, mk, n) not in cell_summaries
        and (sc.key, mk, n) not in set(allow_missing)
    ]
    if missing:
        rows = "\n".join(f"  - {sc}/{mk}/n_out={n}" for sc, mk, n in missing)
        raise RuntimeError(
            f"Heatmap coverage incomplete: {len(missing)} expected cell(s) "
            f"absent from the loaded per-cell summary cache:\n{rows}\n"
            "Cells are computed (in parallel, one job per cell) and cached by "
            "the snakemake per-cell rules; run the pipeline to (re)generate the "
            "missing summaries rather than plotting from a partial cache. "
            "A remaining mismatch usually means a stale internal 'scenario' key."
        )


def _fmt_count(n: int) -> str:
    # No decimals in the site-count strip (e.g. 1.1k -> 1k, 7.2M -> 7M).
    if n == 0:
        return "0"
    if n >= 1_000_000:
        return f"{n/1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}k"
    return str(n)


def _fmt_seconds(s: float) -> str:
    if not np.isfinite(s):
        return "—"
    if s < 1.0:
        return f"{s*1000:.0f}ms"
    if s < 60.0:
        return f"{s:.1f}s"
    if s < 3600.0:
        return f"{s/60:.1f}m"
    if s < 172800.0:  # < 48 h
        return f"{s/3600:.2f}h"
    return f"{s/86400:.1f}d"


def render_heatmap(data: np.ndarray, *, cbar_label: str, out_path: Path,
                   cmap: str = "RdYlGn", vmin: float = 0.0,
                   vmax: float = 1.0) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import Normalize
    sns.set_theme(style="white", context="paper")

    # A leading run of Ancestree's own modes is set off from the external
    # pipelines below it, the rule between the two groups sitting in the gap
    # rather than over the cells on either side. The gap is a fraction of a
    # cell, so the matrix is drawn cell by cell rather than through
    # sns.heatmap, which only ever spaces its rows uniformly.
    keys = [k for _, k in METHODS]
    split = 0
    while split < len(keys) and keys[split] in _ANCESTREE_MODES:
        split += 1
    if not (0 < split < len(keys)) or any(k in _ANCESTREE_MODES for k in keys[split:]):
        split = None
    gap_row = None if split is None else split * N_OUT
    GAP = 0.24  # in cell heights

    def _top(r: int) -> float:
        """Upper edge of matrix row ``r`` in data coordinates."""
        return r if gap_row is None or r < gap_row else r + GAP

    def _mid(r: int) -> float:
        return _top(r) + 0.5

    height = _top(n_rows)
    fig, ax = plt.subplots(
        figsize=(0.40 * n_cols + 4.0, 0.32 * height + 2.0),
    )
    y_edges = np.array([_top(r) for r in range(n_rows + 1)], dtype=float)
    if gap_row is not None:
        y_edges[gap_row] = gap_row  # keep the block above flush with its cells
    x_edges = np.arange(n_cols + 1, dtype=float)
    norm = Normalize(vmin=vmin, vmax=vmax)
    masked = np.ma.masked_invalid(data)
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, norm=norm,
                         edgecolors="0.85", linewidth=0.3)
    if gap_row is not None:
        # The row above the gap keeps unit height; the gap is empty space.
        mesh.remove()
        for r0, r1 in ((0, gap_row), (gap_row, n_rows)):
            sub = np.ma.masked_invalid(data[r0:r1])
            edges = np.array([_top(r) for r in range(r0, r1 + 1)], dtype=float)
            if r1 == gap_row:
                edges[-1] = gap_row
            mesh = ax.pcolormesh(x_edges, edges, sub, cmap=cmap, norm=norm,
                                 edgecolors="0.85", linewidth=0.3)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7)
    cbar.set_label(cbar_label)

    # Cell values, in white on the darker cells for legibility.
    rgba = plt.get_cmap(cmap)(norm(data))
    for r in range(n_rows):
        for c in range(n_cols):
            v = data[r, c]
            if not np.isfinite(v):
                continue
            lum = 0.299 * rgba[r, c, 0] + 0.587 * rgba[r, c, 1] + 0.114 * rgba[r, c, 2]
            ax.text(c + 0.5, _mid(r), f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if lum < 0.45 else "black")

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticks([_mid(r) for r in range(n_rows)])
    ax.set_yticklabels(row_labels, rotation=0, fontsize=8)
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    # Per-column site counts, as a horizontal header row just above the
    # matrix (kept off the bottom edge so they don't collide with the
    # rotated site-class tick labels).
    for c_i, count in enumerate(col_counts):
        ax.text(
            c_i + 0.5, -0.35,
            _fmt_count(count),
            ha="center", va="bottom", fontsize=6, color="0.55",
        )
    ax.text(
        -0.3, -0.35, "#sites",
        ha="right", va="bottom", fontsize=6, color="0.55", style="italic",
    )

    # Per-row total wall time (summed over the live-inference scenarios),
    # as a light-grey strip just right of the matrix.
    if len(row_runtimes) == n_rows:
        for r, secs in enumerate(row_runtimes):
            ax.text(
                n_cols + 0.25, _mid(r), _fmt_seconds(secs),
                ha="left", va="center", fontsize=6, color="0.55",
            )
        ax.text(
            n_cols + 0.25, -0.35, "runtime",
            ha="left", va="bottom", fontsize=6, color="0.55", style="italic",
        )

    # Separator lines between scenarios and method blocks, the group rule
    # running through the middle of the gap.
    for i in range(1, N_SCEN):
        if gap_row is None:
            ax.vlines(i * N_CAT, 0, height, color="black", lw=1.6)
        else:
            ax.vlines(i * N_CAT, 0, gap_row, color="black", lw=1.6)
            ax.vlines(i * N_CAT, gap_row + GAP, height, color="black", lw=1.6)
    for i in range(1, N_METHODS):
        if i == split:
            # A bar rather than a line, so it fills the gap exactly at any
            # figure scale, touching the cells above and below without
            # covering them.
            from matplotlib.patches import Rectangle
            # A hair beyond the gap at each end, so the grey cell strokes on
            # the boundary rows do not show alongside it.
            ax.add_patch(Rectangle((0, gap_row - 0.02), n_cols, GAP + 0.04,
                                   facecolor="black", edgecolor="none",
                                   zorder=4))
        else:
            ax.hlines(_top(i * N_OUT), 0, n_cols, color="black", lw=1.6)

    for sc_i, sc in enumerate(FIG_SCENARIOS):
        x_mid = sc_i * N_CAT + N_CAT / 2
        ax.text(
            x_mid, -1.5, sc.label,
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
        # Per-scenario ILS fraction (ingroup non-monophyly), a property of
        # the genealogy shared by all three modes, in light grey under the
        # scenario title.
        ils = ils_by_scenario.get(sc.key)
        if ils is not None and np.isfinite(ils):
            ax.text(
                x_mid, -1.05, f"({100 * ils:.0f}% ILS)",
                ha="center", va="bottom", fontsize=6.5, color="0.55",
            )
    for m, (method_label, _) in enumerate(METHODS):
        ax.text(
            -2.6, _top(m * N_OUT) + N_OUT / 2, method_label,
            ha="right", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.subplots_adjust(left=0.20, right=0.92, top=0.86, bottom=0.18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------- write runtime side-table
def _write_runtime_table(out_json: Path, out_md: Path) -> None:
    payload: dict[str, dict] = {
        "schema": "method × n_out → {scenario: seconds}",
        "scenarios": [sc.key for sc in FIG_SCENARIOS],
        "methods": [{"label": lbl, "key": key} for lbl, key in METHODS],
        "n_outs": list(N_OUTS),
        "cells": {},
    }
    for label, method_key in METHODS:
        for n_out in N_OUTS:
            row_key = f"{method_key}__n{n_out}"
            payload["cells"][row_key] = {
                sc.key: runtimes.get((sc.key, method_key, n_out), float("nan"))
                for sc in FIG_SCENARIOS
            }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    # MD table: rows = method×n_out, cols = scenario.
    md_lines: list[str] = [
        "# Robustness heatmap — per-cell wall time (seconds)",
        "",
        ("Times are in seconds for the in-process Ancestree / parsimony /"
         " majority methods (per-cell ``time.perf_counter`` around the"
         " inference call)."),
        "",
        "| Method | n_out | "
        + " | ".join(sc.key for sc in FIG_SCENARIOS) + " |",
        "| --- | --- | "
        + " | ".join("---" for _ in FIG_SCENARIOS) + " |",
    ]
    for label, method_key in METHODS:
        for n_out in N_OUTS:
            cells = [
                runtimes.get((sc.key, method_key, n_out), float("nan"))
                for sc in FIG_SCENARIOS
            ]
            cell_strs = [_fmt_seconds(c) for c in cells]
            md_lines.append(
                f"| {label} | {n_out} | " + " | ".join(cell_strs) + " |"
            )
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_json} and {out_md}")


def build_report(
    out_pdf: Path, out_pdf_map: Path,
    out_runtimes_json: Path, out_runtimes_md: Path,
) -> None:
    """Render both heatmap PDFs + the runtime side-table from the globals.

    Assumes ``per_site`` / ``runtimes`` are already populated (by
    :func:`load_cell_summaries`).
    """
    _compute_row_runtimes()
    _compute_col_counts()
    compute_scenario_ils()
    _write_runtime_table(Path(out_runtimes_json), Path(out_runtimes_md))
    # Headline metric: balanced Brier (strictly proper — doesn't reward
    # overconfidence. Folded per-frequency-bin so class imbalance can't hide
    # the hard tails, see BALANCE_FREQ_BINS). Lower is better, so reverse the
    # colormap. Scale [0, 1] (0 = confident+correct, 1 = the colour-scale
    # midpoint / uninformative no-call). The few cells that exceed 1
    # (confidently-wrong baselines) saturate at the top but keep their label.
    render_heatmap(
        build_matrix("brier"),
        cbar_label="Balanced Brier (lower = better)",
        out_path=Path(out_pdf),
        cmap="RdYlGn_r", vmin=0.0, vmax=1.0,
    )
    # MAP accuracy alongside, for the hard-call (single-AA) use case.
    render_heatmap(
        build_matrix("map"),
        cbar_label="MAP accuracy on the indicated site class",
        out_path=Path(out_pdf_map),
    )
