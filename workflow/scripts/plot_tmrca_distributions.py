"""Pairwise TMRCA under the truth, the plug-in estimate and the ensemble.

One row per demography. Left: the three distributions over the same windows
and the same pairs, as densities per unit log-TMRCA so the areas are
comparable, on the HMM's own time grid drawn as a rug below the axis. Right:
each estimate against the truth, as the median and the central 90 per cent of
the estimate within bands of the true TMRCA, read off the pooled joint
histogram.

Every row shares one x range, since the rows differ only in demography and the
point is to read the shift between them.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

try:
    in_npz = list(snakemake.input.npz)  # type: ignore[name-defined]
    in_meta = list(snakemake.input.meta)  # type: ignore[name-defined]
    scenarios = list(snakemake.params.scenarios)  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
except NameError:
    scenarios = ["baseline", "pop_growth", "pop_decline"]
    in_npz = [f"results/data/tmrca_distributions_{s}.npz" for s in scenarios]
    in_meta = [f"results/reports/tmrca_distributions_{s}_meta.json"
               for s in scenarios]
    out_pdf = "results/reports/tmrca_distributions.pdf"

#: Row headings, in the order the scenarios are given.
TITLES = {"baseline": "constant size", "pop_growth": "recent growth",
          "pop_decline": "recent decline"}
#: Histogram bins per step of the HMM's own time grid. At 1 the bins match the
#: grid and the sampled distribution reads as a density; higher resolves within
#: a bin and shows the draws as the comb of grid values they would be without
#: the within-bin draw.
BINS_PER_GRID_STEP = 12
#: Extra widening for the truth alone. Its fine roughness is the finite set of
#: node heights a local tree carries, not a feature of the distribution.
TRUTH_BIN_FACTOR = 4
#: Truth bands thinner than this are dropped from the calibration panel: the
#: 5th and 95th percentiles of a handful of counts are the counts themselves.
MIN_PER_BAND = 500

rows = []
for npz, meta in zip(in_npz, in_meta):
    rows.append((np.load(npz), json.load(open(meta))))

# One x range for every row. The grid is fixed across scenarios, so this is
# the grid's own span; taken as a union it also survives per-row calibration.
half = 10 ** (max(float(d["grid_step"]) for d, _ in rows) / 2)
XLO = min(float(d["t_rep"].min()) for d, _ in rows) / half
XHI = max(float(d["t_rep"].max()) for d, _ in rows) * half


def _count(n):
    """``n`` as a compact decimal count, e.g. 129.4M."""
    for cut, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= cut:
            return f"{n / cut:.1f}{suffix}"
    return str(int(n))


def _density(data, key, factor, extra=1):
    """Counts rebinned ``extra``-fold coarser, normalised over the view.

    :return: ``(centres, density per unit log10 TMRCA)``.
    """
    f = factor * extra
    ed = data["bin_edges"][::f]
    counts = data[key].astype(float).reshape(-1, f).sum(axis=1)
    mid = np.sqrt(ed[:-1] * ed[1:])
    counts = np.where((mid >= XLO) & (mid <= XHI), counts, 0.0)
    return mid, counts / counts.sum() / np.diff(np.log10(ed))


def _moments(data, key):
    """``(mean, sd)`` of log10 TMRCA from the shards' exact accumulators."""
    n, s, ss = data[key]
    mean = s / n
    return mean, np.sqrt(max(ss / n - mean ** 2, 0.0))


SERIES = [
    ("hist_draws", "moments_draws", "posterior draws", "#762a83", "-", 1),
    ("hist_truth", "moments_truth", "simulated truth", "#222222", "-", 2),
    ("hist_point", "moments_point", "posterior mean (plug-in)",
     "#2166ac", "-", 3),
]
ESTIMATES = [
    ("joint_point", "posterior mean (plug-in)", "#2166ac", "-"),
    ("joint_draw_mean", "posterior draws", "#762a83", "-"),
]

fig, axes = plt.subplots(len(rows), 2, figsize=(11.5, 1.76 * len(rows)),
                         squeeze=False, sharex="col")

true_his = []
for r, ((data, meta), (ax, ax2)) in enumerate(zip(rows, axes)):
    t_rep = data["t_rep"]
    fine = np.median(np.diff(np.log10(data["bin_edges"])))
    factor = max(1, int(round((float(data["grid_step"]) / fine)
                              / BINS_PER_GRID_STEP)))
    peak = 0.0
    for key, mkey, label, colour, style, layer in SERIES:
        extra = TRUTH_BIN_FACTOR if key == "hist_truth" else 1
        x, d = _density(data, key, factor, extra)
        peak = max(peak, float(d.max()))

        if key == "hist_truth":
            ax.fill_between(x, d, step="mid", color=colour, alpha=0.13, lw=0,
                            zorder=layer)
        ax.step(x, d, where="mid", color=colour, ls=style, lw=1.7,
                zorder=layer + 3, alpha=0.55 if key == "hist_truth" else 1.0,
                label=label)
    inside = (t_rep >= XLO) & (t_rep <= XHI)
    # Equal fractional headroom above and below, so every row is padded alike.
    ax.plot(t_rep[inside], np.full(int(inside.sum()), -0.045 * peak), "|",
            color="#999999", ms=7, mew=0.9, alpha=0.5, clip_on=False,
            label=f"time grid ({t_rep.size} bins)" if r == 0 else None)
    ax.set_ylim(-0.09 * peak, 1.09 * peak)
    ax.set_xlim(XLO, XHI)
    ax.set_xscale("log")
    ax.set_ylabel(TITLES.get(scenarios[r], scenarios[r]), fontsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(4))

    true_edges, est_edges = data["true_edges"], data["est_edges"]
    true_mid = np.sqrt(true_edges[:-1] * true_edges[1:])
    est_mid = np.sqrt(est_edges[:-1] * est_edges[1:])
    data_hi = true_hi = 0.0
    for key, label, colour, style in ESTIMATES:
        j = data[key].astype(float)
        total = j.sum(axis=1)
        keep = total >= MIN_PER_BAND
        cum = np.cumsum(j[keep], axis=1) / total[keep, None]
        q = np.stack([est_mid[np.argmax(cum >= p, axis=1)]
                      for p in (0.05, 0.5, 0.95)])
        data_hi = max(data_hi, float(q[2].max()))
        true_hi = max(true_hi, float(true_mid[keep].max()))
        ax2.fill_between(true_mid[keep], q[0], q[2], color=colour, alpha=0.13,
                         lw=0)
        ax2.plot(true_mid[keep], q[1], color=colour, ls=style, lw=1.7,
                 alpha=0.75, label=label)
    ax2.plot((XLO, true_hi), (XLO, true_hi), color="#222222", lw=1.0, ls=":",
             label="$y = x$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_locator(LogLocator(numticks=4))
    true_his.append(true_hi)
    ax2.set_ylim(XLO, data_hi)
    if r == len(rows) // 2:
        ax2.set_ylabel("estimated TMRCA (gen.)")

for pane in axes:
    pane[1].set_xlim(XLO, max(true_his))
fig.supylabel("density", fontsize=10, x=0.005)
axes[-1][0].set_xlabel("pairwise TMRCA (generations)")
axes[-1][1].set_xlabel("true pairwise TMRCA (generations)")
handles, labels = [], []
for pane in (axes[0][0], axes[0][1]):
    for handle, label in zip(*pane.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
for pane in axes[:-1]:
    for cell in pane:
        cell.tick_params(labelbottom=False)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.subplots_adjust(hspace=0.0)
# The legend spans both columns, from the left axis edge to the right one.
x0 = axes[-1][0].get_position().x0
x1 = axes[-1][1].get_position().x1
legend = fig.legend(handles, labels, ncol=len(labels), loc="lower center",
                    mode="expand", frameon=True, framealpha=0.92,
                    edgecolor="#bbbbbb", fontsize=10, handlelength=2.6,
                    borderaxespad=0.0,
                    bbox_to_anchor=(x0, 0.065, x1 - x0, 0.05))
legend.get_frame().set_linewidth(0.6)
fig.savefig(out_pdf)
print(f"wrote {out_pdf} ({len(rows)} scenarios)")
