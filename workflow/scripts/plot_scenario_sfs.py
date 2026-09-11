"""Appendix figure: per-scenario polymorphic site-frequency spectrum.

Counts of segregating sites by unfolded ingroup derived-allele count
``i in {1, ..., n-1}`` (monomorphic classes excluded), linear scale, one
barplot per scenario, stacked vertically. Provides the polymorphic-site
budget context for the per-class robustness results (Figure 4 / the
baselines appendix figure). The spectra come straight from the cached
robustness cells, so no extra inference is run.

Invoked by the ``scenario_sfs_figure`` rule (``shell: python …``) or run
directly. Either way it globs the cached per-cell JSONs (scenarios + cell
loader shared with :mod:`_robustness_common`) and renders.

Outputs: ``results/reports/scenario_sfs.pdf`` (+ a ~200-dpi PNG) and a
copy at ``reports/manuscripts/latex/figures/bench_scenario_sfs.pdf``.
"""
import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc

# The SFS is a property of the truth, so any method's records give the same
# per-bin counts. Use the full-coverage ARG cells at a single n_out.
CANON_METHOD = "anc_arg"
CANON_NOUT = 3

try:
    out_pdf = Path(snakemake.output.pdf)  # type: ignore[name-defined]
    rc.load_cell_summaries(list(snakemake.input.cells))  # type: ignore[name-defined]
except NameError:
    out_pdf = rc.REPORTS / "scenario_sfs.pdf"
    # Not a bare robustness_cell_* glob: that also matches the
    # robustness_cell_intersect_* cells, which carry the SAME internal
    # (scenario, method, n_out) key over SINGER's budget prefix rather than the
    # whole chunk. They would collide in load_cell_summaries and glob order
    # would decide which spectrum is plotted. This figure wants whole chunks,
    # which is what the rule declares as its inputs.
    rc.load_cell_summaries([
        f for f in glob.glob(
            str(rc.REPO / "results" / "data" / "robustness_cell_*_summary.json"))
        if "robustness_cell_intersect_" not in Path(f).name])


# FIG_SCENARIOS, not SCENARIOS: the latter gained the three demography
# scenarios (pop_constant / pop_growth / pop_decline), which this appendix
# figure does not show and which have no anc_arg cells. Iterating SCENARIOS
# silently grew the figure to eight panels, three of them empty.
fig, axes = plt.subplots(len(rc.FIG_SCENARIOS), 1, figsize=(7.0, 4.16),
                         sharex=True)
for ax, sc in zip(axes, rc.FIG_SCENARIOS):
    n = sc.n_ingroup or 20
    summ = rc.cell_summaries.get((sc.key, CANON_METHOD, CANON_NOUT))
    per_bin = rc.summary_per_bin_count(summ, n) if summ else [0] * (n + 1)
    bins = list(range(1, n))  # polymorphic only (drop i=0 and i=n)
    ax.bar(bins, [per_bin[b] for b in bins], width=0.85,
           color="#4C72B0", edgecolor="none")
    ax.set_ylabel("sites")
    ax.text(0.985, 0.88, sc.label.replace("\n", " "), transform=ax.transAxes,
            ha="right", va="top", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8",
                      alpha=0.85))
    ax.set_xlim(0.3, n - 0.3)
    ax.set_xticks(range(1, n))
    ax.tick_params(labelsize=8)
axes[-1].set_xlabel(r"ingroup derived-allele count  $i$  (polymorphic)")


def _kfmt(v, _pos):  # compact counts ("0", "120k") -> no 1e5 offset text
    if v <= 0:
        return "0"
    return f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"


# Flush the panels together (shared x-axis). Panels share no y-scale, so each
# carries its own ticks. Two things keep the labels from colliding across the
# seam while staying stacked: drop each upper panel's "0" tick (prune="lower",
# only the bottom panel keeps it) so no label sits at the boundary, and add top
# headroom so a panel's peak tick is pulled down inside it rather than poking up
# across the seam into the panel above.
for idx, ax in enumerate(axes):
    last = idx == len(axes) - 1
    ax.yaxis.set_major_locator(
        MaxNLocator(nbins=4, prune=None if last else "lower"))
    ax.yaxis.set_major_formatter(FuncFormatter(_kfmt))
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.18)
fig.tight_layout()
fig.subplots_adjust(hspace=0.0)

out_pdf.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {out_pdf}")

