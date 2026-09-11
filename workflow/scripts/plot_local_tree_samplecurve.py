"""Ground-truth recovery against ensemble size, one line per scenario.

Ensemble mode marginalises each window's call over ``n_ensemble`` genealogies
drawn from the pairwise-HMM posterior. This figure asks what that buys: the
Brier score against the simulated truth as the member count grows, for every
robustness scenario, so the point of diminishing returns is visible rather than
assumed. Points are the per-chunk mean with the chunk-to-chunk spread shaded.

Scored through the same path as the heatmap (intersection mask, focal node per
site class, ``summarize_persite``), so the y values are on the heatmap's scale.
"""
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402

try:
    in_jsons = list(snakemake.input.cells)  # noqa: F821
    out_pdf = snakemake.output.pdf  # noqa: F821
    out_fig = snakemake.output.fig_copy  # noqa: F821
except NameError:
    # Standalone debug defaults; the DAG supplies these through `script:`.
    import glob as _glob
    in_jsons = sorted(_glob.glob(
        "results/data/samplecurve_local_tree_*.json"))
    out_pdf = "results/reports/bench_local_tree_samplecurve.pdf"
    out_fig = ("reports/manuscripts/latex/figures/"
               "bench_local_tree_samplecurve.pdf")

# (scenario, k) -> [per-chunk Brier]
by: dict[tuple[str, int], list[float]] = defaultdict(list)
for path in in_jsons:
    with open(path) as fh:
        summ = json.load(fh)
    stem = os.path.basename(path).rsplit(".", 1)[0]
    k = int(stem.rsplit("_k", 1)[1])
    scen = summ["scenario"]
    n_ing = summ.get("n_ingroup") or rc.N_INGROUP_DEFAULT
    # Every site class pooled: this figure is about the ensemble, not about
    # which class benefits, and the heatmap already splits by class.
    vals = [rc.summary_cat_value(summ, lbl, pf, n_ing, metric="brier")
            for lbl, pf in rc.CATS]
    vals = [v for v in vals if np.isfinite(v)]
    if vals:
        by[(scen, k)].append(float(np.mean(vals)))

scens = [s.key for s in rc.FIG_SCENARIOS if any(k[0] == s.key for k in by)]
labels = {s.key: s.label for s in rc.FIG_SCENARIOS}
ks = sorted({k for _, k in by})

def t_crit(n):
    """Two-sided 95% Student-t critical value for ``n`` observations."""
    from scipy import stats
    return stats.t.ppf(0.975, np.maximum(n - 1, 1))


fig, ax = plt.subplots(figsize=(6.4, 3.4))
colors = plt.get_cmap("tab10").colors
for i, scen in enumerate(scens):
    xs = [k for k in ks if (scen, k) in by]
    mean = np.array([np.mean(by[(scen, k)]) for k in xs])
    # 95% CI of the mean, not the raw spread: the band is there to say how well
    # the mean is pinned down, and with three chunks a standard deviation reads
    # far tighter than the estimate actually is. Student-t, since n is small.
    n = np.array([len(by[(scen, k)]) for k in xs], dtype=float)
    sd = np.array([np.std(by[(scen, k)], ddof=1) if len(by[(scen, k)]) > 1
                   else 0.0 for k in xs])
    half = np.where(n > 1, t_crit(n) * sd / np.sqrt(np.maximum(n, 1)), 0.0)
    ax.plot(xs, mean, "o-", lw=1.7, ms=4, color=colors[i % len(colors)],
            label=labels.get(scen, scen), alpha=0.85)
    ax.fill_between(xs, mean - half, mean + half,
                    color=colors[i % len(colors)], alpha=0.15, lw=0)
ax.set_xscale("log", base=2)
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("ensemble size (genealogies marginalised over)")
ax.set_ylabel("mean Brier score\n(lower = better)")
ax.grid(True, alpha=0.25, lw=0.5)
ax.legend(frameon=True, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
import shutil  # noqa: E402
shutil.copyfile(out_pdf, out_fig)
print(f"wrote {out_pdf} ({len(scens)} scenarios, k={ks})")
