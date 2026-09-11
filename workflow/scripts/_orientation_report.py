"""Per-comparison numerics and report assembly for the orientation benchmark.

One comparison pairs a mis-orientation arm against the true-ancestral-allele
arm of the same ARG tool, built from identical genotypes with an identical
seed, so the difference between the two arms isolates the cost of the
mis-declared ancestral allele. Sites are binned two ways:

- by the genomic distance in bp from the site to the nearest mis-oriented site.
- by the number of mis-oriented sites falling inside the local tree interval of
  the same tool's true-ancestral-allele ARG that contains the site.

Every field named ``*_excess`` is signed so that a positive value means the
mis-orientation degraded the inferred genealogy relative to the
true-ancestral-allele arm.

Symbols used in the TMRCA statistics:

- ``t_true`` --- pairwise time to the most recent common ancestor in the
  simulated truth, in generations, at a ``(site, haplotype-pair)`` point.
- ``t_inferred`` --- the same quantity read off an inferred ARG.
- ``rho`` --- Spearman rank correlation between ``t_true`` and ``t_inferred``
  pooled over the points of one bin, dimensionless in ``[-1, 1]``.
- ``slope`` --- ordinary-least-squares regression coefficient of
  ``t_inferred`` on ``t_true``, in generations per generation, ideal value 1.
- ``log_bias`` --- mean of ``log t_inferred - log t_true`` in natural log
  generations, ideal value 0.
- ``n_boot`` --- replicates of the site-block bootstrap behind every standard
  error, resampling whole per-site blocks of haplotype pairs with replacement.
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    BIN_LABELS,
    BOOTSTRAP_SEED,
    INTERVAL_LABELS,
    N_BOOTSTRAP,
    distance_bins,
    distance_to_nearest,
    interval_bins,
    interval_flip_counts,
    load_orientation,
)


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ordinal rank of every element of ``x``, in ``0..x.size - 1``.

    :param x: Values to rank.
    :return: The rank of each element, ties ordered as the sort orders them.
    """
    r = np.empty(x.size, dtype=np.intp)
    r[np.argsort(x)] = np.arange(x.size)
    return r


def _spearman(t_true: np.ndarray, t_inf: np.ndarray) -> float:
    """Spearman rank correlation between true and inferred pairwise TMRCA."""
    rt = np.argsort(np.argsort(t_true))
    ri = np.argsort(np.argsort(t_inf))
    return float(np.corrcoef(rt, ri)[0, 1])


def _slope(t_true: np.ndarray, t_inf: np.ndarray) -> float:
    """OLS regression coefficient of inferred on true TMRCA (generations per
    generation). The ideal value is 1."""
    return float(np.polyfit(t_true, t_inf, 1)[0])


def _slope_fast(t_true: np.ndarray, t_inf: np.ndarray) -> float:
    """The same slope by a single centred inner product, for the bootstrap."""
    x = t_true - t_true.mean()
    return float((x * (t_inf - t_inf.mean())).sum() / (x * x).sum())


def _log_bias(t_true: np.ndarray, t_inf: np.ndarray) -> float:
    """Mean of ``log t_inferred - log t_true`` in natural log generations; the
    ideal value is 0."""
    return float(np.mean(np.log(t_inf) - np.log(t_true)))


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Per-site paired comparison of two arms on one metric.

    :param a: Per-site metric of the true-ancestral-allele arm.
    :param b: Per-site metric of the mis-oriented arm.
    :return: the two means, the mean paired difference ``b - a``, and the
        standard error of the mean of that difference.
    """
    m = np.isfinite(a) & np.isfinite(b)
    x, y = a[m], b[m]
    if x.size == 0:
        return dict(n=0, mean_true_aa=None, mean_scheme=None, mean_excess=None,
                    sem_paired_excess=None)
    dd = y - x
    return dict(
        n=int(x.size),
        mean_true_aa=float(x.mean()),
        mean_scheme=float(y.mean()),
        mean_excess=float(dd.mean()),
        sem_paired_excess=(float(dd.std(ddof=1) / np.sqrt(dd.size))
                           if dd.size > 1 else 0.0),
    )


def _tmrca_block(tt: np.ndarray, ta: np.ndarray, tb: np.ndarray,
                 sites: np.ndarray, boot_rng, n_boot: int,
                 n_threads: int = 1) -> dict:
    """Pooled TMRCA metrics for one bin, with a site-block bootstrap error.

    :param tt: True pairwise TMRCA in generations at the ``(site, pair)`` points.
    :param ta: The true-ancestral-allele arm's TMRCA at the same points.
    :param tb: The mis-oriented arm's TMRCA at the same points.
    :param sites: Site index of each point. Points of one site are contiguous,
        and a whole site is resampled as one block.
    :param boot_rng: Generator drawing the block resample.
    :param n_boot: Bootstrap replicates.
    :param n_threads: Replicates evaluated concurrently. The replicates are
        independent and their resample indices are drawn up front in stream
        order, so the value does not enter the numbers.
    :return: the three TMRCA statistics for both arms, their excesses and the
        bootstrap standard errors of those excesses.
    """
    if tt.size < 3:
        return dict(n_tmrca_points=int(tt.size), n_sites_with_points=0)
    rho_a, rho_b = _spearman(tt, ta), _spearman(tt, tb)
    sl_a, sl_b = _slope(tt, ta), _slope(tt, tb)
    lb_a, lb_b = _log_bias(tt, ta), _log_bias(tt, tb)

    starts = np.flatnonzero(np.r_[True, sites[1:] != sites[:-1]])
    ends = np.r_[starts[1:], sites.size]
    lens = ends - starts
    n_blocks = starts.size
    picks = [boot_rng.integers(0, n_blocks, size=n_blocks)
             for _ in range(max(0, n_boot))]

    def replicate(pick: np.ndarray) -> tuple[float, float, float]:
        idx = np.repeat(starts[pick] - np.cumsum(np.r_[0, lens[pick][:-1]]),
                        lens[pick]) + np.arange(lens[pick].sum())
        xt, xa, xb = tt[idx], ta[idx], tb[idx]
        rt = _ranks(xt)
        return (float(np.corrcoef(rt, _ranks(xa))[0, 1])
                - float(np.corrcoef(rt, _ranks(xb))[0, 1]),
                abs(1.0 - _slope_fast(xt, xb)) - abs(1.0 - _slope_fast(xt, xa)),
                abs(_log_bias(xt, xb)) - abs(_log_bias(xt, xa)))

    workers = max(1, min(int(n_threads), max(1, n_boot)))
    if not picks:
        drawn = []
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            drawn = list(pool.map(replicate, picks))
    else:
        drawn = [replicate(p) for p in picks]
    b_rho = [d[0] for d in drawn]
    b_sl = [d[1] for d in drawn]
    b_lb = [d[2] for d in drawn]

    def se(values: list) -> float | None:
        """Bootstrap standard error, or None when no replicates were drawn."""
        return float(np.std(values, ddof=1)) if len(values) > 1 else None

    return dict(
        n_tmrca_points=int(tt.size),
        n_sites_with_points=int(n_blocks),
        tmrca_spearman_true_aa=rho_a,
        tmrca_spearman_scheme=rho_b,
        tmrca_spearman_excess=rho_a - rho_b,
        tmrca_spearman_excess_se_site_bootstrap=se(b_rho),
        tmrca_slope_true_aa=sl_a,
        tmrca_slope_scheme=sl_b,
        tmrca_slope_excess=abs(1.0 - sl_b) - abs(1.0 - sl_a),
        tmrca_slope_excess_se_site_bootstrap=se(b_sl),
        tmrca_log_bias_true_aa=lb_a,
        tmrca_log_bias_scheme=lb_b,
        tmrca_log_bias_excess=abs(lb_b) - abs(lb_a),
        tmrca_log_bias_excess_se_site_bootstrap=se(b_lb),
    )


METRIC_DIRECTIONS = {
    "convention": (
        "every field named *_excess is signed so that a positive value means "
        "the scheme's mis-orientation degraded the inferred genealogy relative "
        "to the true-ancestral-allele orientation built from identical "
        "genotypes, identical seed and identical parameters. Zero means "
        "orientation had no effect and a negative value means the mis-oriented "
        "ARG was closer to the truth"),
    "field_naming": (
        "within a comparison, fields suffixed _true_aa carry the "
        "true-ancestral-allele arm and fields suffixed _scheme carry the "
        "mis-oriented arm named by the comparison's scheme key"),
    "rf_norm_excess": {
        "definition": "mean over sites of (RF(scheme, truth) - RF(true_aa, truth)), both divided by rf_max",
        "units": "dimensionless, normalised Robinson-Foulds on a 20-leaf unrooted topology, where 0 = identical topology and 1 = no shared split",
        "positive_excess_means": "mis-orientation made the local topology worse",
    },
    "kc_lambda1_excess": {
        "definition": "mean over sites of (KC_lambda1(scheme, truth) - KC_lambda1(true_aa, truth))",
        "units": "generations (Kendall-Colijn distance at lambda = 1 is the Euclidean norm of the vector of pairwise MRCA branch-length depths, so it inherits the time units of the tree sequence)",
        "positive_excess_means": "mis-orientation made the dated local tree worse",
    },
    "tmrca_spearman_excess": {
        "definition": "rho(true_aa) - rho(scheme), where rho is the Spearman rank correlation between inferred and true pairwise TMRCA pooled over all (site, haplotype-pair) points in the bin",
        "units": "dimensionless correlation in [-1, 1], where higher rho is better, hence the reversed subtraction",
        "positive_excess_means": "mis-orientation lowered the rank agreement between inferred and true TMRCA",
    },
    "tmrca_slope_excess": {
        "definition": "|1 - slope(scheme)| - |1 - slope(true_aa)|, slope = OLS regression coefficient of inferred TMRCA on true TMRCA within the bin",
        "units": "dimensionless (generations per generation). The ideal slope is 1, so the absolute deviation from 1 is the error",
        "positive_excess_means": "mis-orientation moved the TMRCA regression slope further from 1",
    },
    "tmrca_log_bias_excess": {
        "definition": "|log_bias(scheme)| - |log_bias(true_aa)|, log_bias = mean(log t_inferred - log t_true) over the (site, pair) points in the bin",
        "units": "natural log generations. The ideal log-bias is 0, so the absolute value is the error",
        "positive_excess_means": "mis-orientation moved the mean log TMRCA further from the truth",
    },
    "tmrca_point_filter": "a (site, pair) point enters only if the pairwise TMRCA is strictly positive in the truth and in both arms",
    "site_filter": "a site is dropped from a comparison if the truth or either arm has a multi-rooted local tree at its position",
    "sem_conventions": {
        "rf_norm_excess / kc_lambda1_excess": "sem_paired_excess is the standard error of the mean of the per-site paired difference (scheme - true_aa), not the difference of two independent standard errors",
        "tmrca_*_excess": "standard error from a nonparametric bootstrap over sites within the bin, resampling whole per-site blocks of haplotype pairs with replacement",
    },
    "binning_by_distance_bp": (
        "by_distance_bp bins a site by the genomic distance in bp to the "
        "nearest site the scheme mis-orients. Bin 0 holds the mis-oriented "
        "sites themselves"),
    "binning_by_interval_flip_count": (
        "by_interval_flip_count bins a site by the number of mis-oriented "
        "sites whose position falls inside the half-open local tree interval "
        "of the SAME tool's true-ancestral-allele ARG that contains the site. "
        "The covariate is therefore tool-specific and is not distance-based"),
}


class OrientationDamageReport:
    """Pair every mis-orientation arm against its true-ancestral-allele arm.

    :param sites: The shared site set, as written by ``orientation_sites.py``.
    :param maps: Scheme name to ``orientation_map_{scheme}.json`` path.
    :param persite: ``(tool, scheme)`` to ``orientation_persite_*.npz`` path.
    :param n_boot: Bootstrap replicates behind every TMRCA standard error.
    :param boot_seed: Seed of the site-block bootstrap, reset per binning so the
        two binnings of one comparison draw the same resamples.
    :param n_threads: Bootstrap replicates evaluated concurrently within a bin.
    """

    def __init__(self, sites: dict, maps: dict, persite: dict,
                 n_boot: int = N_BOOTSTRAP, boot_seed: int = BOOTSTRAP_SEED,
                 n_threads: int = 1):
        self.sites = sites
        self.maps = maps
        self.persite = persite
        self.n_boot = int(n_boot)
        self.boot_seed = int(boot_seed)
        self.n_threads = int(n_threads)
        self.orientation: dict[str, np.ndarray] = {}
        self.map_summary: dict[str, dict] = {}
        self.sequence_length = 0.0

    @property
    def n_sites(self) -> int:
        """Number of shared sites."""
        return int(self.sites["pos"].size)

    def _load_maps(self) -> None:
        for scheme, path in self.maps.items():
            idx, summary = load_orientation(path, self.n_sites)
            self.orientation[scheme] = idx
            self.map_summary[scheme] = summary

    def _rows(self, site_bin: np.ndarray, point_bin: np.ndarray, labels,
              *, dist, n_flips_site, rf_a, rf_b, kc_a, kc_b, tt, ta, tb,
              psite) -> list[dict]:
        """One record per bin, plus an overall record over all shared sites."""
        daf = self.sites["daf"]
        boot_rng = np.random.default_rng(self.boot_seed)
        rows = []
        for lab in list(range(len(labels))) + ["overall"]:
            if lab == "overall":
                m = np.ones(self.n_sites, dtype=bool)
                pm = np.ones(tt.size, dtype=bool)
                name = "overall (all shared sites)"
            else:
                m = site_bin == lab
                pm = point_bin == lab
                name = labels[lab]
            dv = dist[m]
            finite = np.isfinite(dv)
            rec = {
                "bin": name,
                "bin_index": (None if lab == "overall" else int(lab)),
                "n_sites": int(m.sum()),
                "median_bp_to_nearest_mis_oriented_site":
                    (float(np.median(dv[finite])) if finite.any() else None),
                "mean_mis_oriented_sites_in_true_aa_tree_interval":
                    (float(n_flips_site[m].mean()) if m.any() else None),
                "mean_derived_allele_count":
                    (float(daf[m].mean()) if m.any() else None),
                "rf_norm": _paired(rf_a[m], rf_b[m]),
                "kc_lambda1": _paired(kc_a[m], kc_b[m]),
                "tmrca": _tmrca_block(tt[pm], ta[pm], tb[pm], psite[pm],
                                      boot_rng, self.n_boot, self.n_threads),
            }
            rows.append(rec)
        return rows

    def _comparison(self, tool: str, scheme: str) -> dict:
        """Score one mis-orientation arm against the true-ancestral-allele arm."""
        pos = self.sites["pos"]
        flip = self.orientation[scheme] != 0
        dist = distance_to_nearest(pos, flip)
        bin_idx = distance_bins(dist)

        with np.load(self.persite[(tool, "true_aa")]) as a:
            rf_a, kc_a = a["rf_norm"], a["kc_lambda1"]
            tt_a, ta = a["tmrca_true"], a["tmrca_inferred"]
            breakpoints = a["breakpoints"]
            n_trees_a = int(a["n_trees"])
            n_skip_a = int(a["n_sites_skipped_multiroot"])
            n_pairs = int(a["pairs"].shape[0])
        with np.load(self.persite[(tool, scheme)]) as b:
            rf_b, kc_b = b["rf_norm"], b["kc_lambda1"]
            tt_b, tb = b["tmrca_true"], b["tmrca_inferred"]
            n_trees_b = int(b["n_trees"])
            n_skip_b = int(b["n_sites_skipped_multiroot"])

        n_flips_site, occupancy = interval_flip_counts(breakpoints, pos,
                                                       pos[flip])
        iv_bin = interval_bins(n_flips_site)

        point_mask = (tt_a > 0) & (tt_b > 0) & (ta > 0) & (tb > 0)
        psite = (np.arange(tt_a.size, dtype=np.int32) // n_pairs)[point_mask]
        tt = tt_a[point_mask]
        ta = ta[point_mask]
        tb = tb[point_mask]
        del tt_a, tt_b, point_mask

        common = dict(
            n_mis_oriented_sites=int(flip.sum()),
            pct_mis_oriented_sites=100.0 * float(flip.sum()) / self.n_sites,
            n_trees_true_aa_arg=n_trees_a,
            n_trees_scheme_arg=n_trees_b,
            n_sites_skipped_multiroot_true_aa_arg=n_skip_a,
            n_sites_skipped_multiroot_scheme_arg=n_skip_b,
            n_sites_scored=int((np.isfinite(rf_a) & np.isfinite(rf_b)).sum()),
            n_tmrca_points_total=int(tt.size),
            max_bp_to_nearest_mis_oriented_site=(
                float(dist[np.isfinite(dist)].max())
                if np.isfinite(dist).any() else None),
            mean_bp_between_mis_oriented_sites=(
                self.sequence_length / int(flip.sum()) if flip.any() else None),
            true_aa_arg_interval_occupancy=occupancy,
        )
        rows_bp = self._rows(
            bin_idx, bin_idx[psite], BIN_LABELS, dist=dist,
            n_flips_site=n_flips_site, rf_a=rf_a, rf_b=rf_b, kc_a=kc_a,
            kc_b=kc_b, tt=tt, ta=ta, tb=tb, psite=psite)
        rows_iv = self._rows(
            iv_bin, iv_bin[psite], INTERVAL_LABELS, dist=dist,
            n_flips_site=n_flips_site, rf_a=rf_a, rf_b=rf_b, kc_a=kc_a,
            kc_b=kc_b, tt=tt, ta=ta, tb=tb, psite=psite)
        return dict(
            common,
            by_distance_bp=dict(
                bin_labels=list(BIN_LABELS),
                bin_counts=[int((bin_idx == k).sum())
                            for k in range(len(BIN_LABELS))],
                bins=rows_bp),
            by_interval_flip_count=dict(
                bin_labels=list(INTERVAL_LABELS),
                bin_counts=[int((iv_bin == k).sum())
                            for k in range(len(INTERVAL_LABELS))],
                bins=rows_iv),
        )

    def comparison(self, tool: str, scheme: str, meta: dict) -> dict:
        """Score the ``(tool, scheme)`` arm pair.

        :param tool: ARG tool whose two arms are paired.
        :param scheme: Mis-orientation scheme of the non-reference arm.
        :param meta: The simulation metadata, read for the sequence length the
            mean spacing between mis-oriented sites is taken against.
        :return: The comparison block.
        """
        self._load_maps()
        self.sequence_length = float(meta["length"])
        return self._comparison(tool, scheme)

    def merge(self, meta: dict, comparisons: dict) -> dict:
        """Assemble the report payload around precomputed comparison blocks.

        :param meta: The simulation metadata, carried into the report so the
            parameters the arms were built under travel with the numbers.
        :param comparisons: ARG tool to scheme to comparison block, in the
            order the report lists them.
        :return: The report payload.
        """
        self._load_maps()
        self.sequence_length = float(meta["length"])
        tools = list(comparisons)
        schemes = list(comparisons[tools[0]]) if tools else []
        daf = self.sites["daf"]
        return {
            "design": (
                "orientation damage on an ingroup-only 20-haplotype ARG panel: "
                "identical genotypes, identical seeds, one simulation, and one "
                "declared ancestral allele per orientation scheme. Every "
                "mis-orientation arm is paired per site against the "
                "true-ancestral-allele arm of the same ARG tool"),
            "schemes": list(self.maps),
            "mis_orientation_schemes": schemes,
            "tools": tools,
            "n_sites_shared": self.n_sites,
            "sequence_length": float(meta["length"]),
            "mu": float(meta["mu"]),
            "rec_rate": float(meta["rec_rate"]),
            "sim_seed": meta["seed"],
            "outgroup_split_times_gen": meta["outgroup_split_times"],
            "n_ingroup_haplotypes": int(meta["n_ingroup"]),
            "rf_max": 2.0 * (int(meta["n_ingroup"]) - 2),
            "bootstrap_replicates": self.n_boot,
            "bootstrap_seed": self.boot_seed,
            "derived_allele_count_definition": (
                "number of ingroup haplotypes carrying the allele that is not "
                "the simulator-true ancestral state, in 1..n_ingroup - 1"),
            "mean_derived_allele_count_all_sites": float(daf.mean()),
            "orientation_maps": self.map_summary,
            "metric_directions": METRIC_DIRECTIONS,
            "comparisons": comparisons,
        }


def _fmt(value, spec: str) -> str:
    """Format a possibly missing number for the markdown tables."""
    return "n/a" if value is None else format(float(value), spec)


def _markdown(payload: dict) -> str:
    """Render the per-bin tables of every comparison."""
    lines = ["# Orientation damage to inferred genealogies", "",
             payload["design"], "",
             f"Shared sites: {payload['n_sites_shared']:,} over "
             f"{payload['sequence_length']:,.0f} bp, "
             f"{payload['n_ingroup_haplotypes']} ingroup haplotypes, "
             f"rf_max = {payload['rf_max']:.0f}.", "",
             "Every excess is (mis-oriented arm) minus (true-ancestral-allele "
             "arm), so a positive value means the mis-orientation made the "
             "inferred genealogy worse.", ""]

    lines += ["## Mis-orientation rate per scheme", "",
              "| scheme | mis-oriented sites | % of shared sites |",
              "| --- | ---: | ---: |"]
    tool0 = payload["tools"][0]
    for scheme in payload["mis_orientation_schemes"]:
        c = payload["comparisons"][tool0][scheme]
        lines.append(f"| {scheme} | {c['n_mis_oriented_sites']:,} | "
                     f"{c['pct_mis_oriented_sites']:.3f} |")
    lines.append("")

    for tool in payload["tools"]:
        for scheme in payload["mis_orientation_schemes"]:
            comp = payload["comparisons"][tool][scheme]
            for key, title in (("by_distance_bp",
                                "binned by bp to the nearest mis-oriented site"),
                               ("by_interval_flip_count",
                                "binned by mis-oriented sites in the local tree "
                                "interval")):
                lines += [
                    f"## {tool} / {scheme}, {title}", "",
                    "| bin | sites | mean normalised RF, true-AA arm | "
                    "mean normalised RF, mis-oriented arm | "
                    "excess normalised RF | SEM of excess | "
                    "excess Kendall-Colijn (generations) | "
                    "excess TMRCA rank correlation |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
                for row in comp[key]["bins"]:
                    rf = row["rf_norm"]
                    kc = row["kc_lambda1"]
                    tm = row["tmrca"]
                    lines.append(
                        f"| {row['bin']} | {row['n_sites']:,} | "
                        f"{_fmt(rf['mean_true_aa'], '.5f')} | "
                        f"{_fmt(rf['mean_scheme'], '.5f')} | "
                        f"{_fmt(rf['mean_excess'], '+.5f')} | "
                        f"{_fmt(rf['sem_paired_excess'], '.5f')} | "
                        f"{_fmt(kc['mean_excess'], '+.4g')} | "
                        f"{_fmt(tm.get('tmrca_spearman_excess'), '+.5f')} |")
                lines.append("")
    return "\n".join(lines) + "\n"
