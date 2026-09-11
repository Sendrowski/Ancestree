"""Per-bin Brier across the unfolded spectrum, stacked vertically over the
outgroup counts n_out in {0, 1, 3, 10}.

The Brier score of a site is (1 - p_true)^2 + sum over the other states of
p_a^2, where p_a is the posterior probability of ancestral state a and p_true
that of the simulated ancestral state. Lower is better. Each curve is the mean
over the sites in one unfolded ingroup derived-allele count bin i.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402

NING = 20
NOUTS = [0, 1, 3, 10]

SCEN = [
    ("baseline", "Baseline"),
    ("cpg_hypermut", "Hypermutation"),
    ("strong_ils", "Strong ILS"),
    ("outgroup_clade", "Outgroup clade"),
    ("slim", "Purifying"),
]
MODES = [
    ("anc_arg", "ARG mode", "-"),
    ("local_tree_unphased", "local-tree mode", (0, (5, 2))),
    ("anc_ft", "fixed-tree mode", (0, (1, 1.4))),
]
COLORS = plt.cm.tab10(np.arange(len(SCEN)))

try:
    CELLS = {Path(p).name: Path(p) for p in snakemake.input.cells}  # type: ignore[name-defined]
    OUT = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    CELLS = {
        rc.intersect_cell_summary_path(sk, mk, n).name:
            rc.intersect_cell_summary_path(sk, mk, n)
        for sk, _sl in SCEN for mk, *_rest in MODES for n in NOUTS
    }
    OUT = Path("reports/manuscripts/latex/figures/bench_brier_per_bin_stack.pdf")


def per_bin_brier(scenario, mode, nout):
    """Per-bin (0..NING) mean Brier from the light cell summary.

    :param scenario: scenario key, e.g. ``strong_ils``.
    :param mode: method key, e.g. ``anc_arg``.
    :param nout: number of outgroups in the cell.
    :return: mean Brier per unfolded derived-allele count bin.
    """
    name = f"robustness_cell_intersect_{scenario}_{mode}_n{nout}_summary.json"
    summ = rc.load_summary(CELLS[name])
    return np.array(rc.summary_per_bin_brier(summ, NING))


fig, axes = plt.subplots(len(NOUTS), 1, figsize=(7.0, 5.0), sharex=True,
                         gridspec_kw={"hspace": 0.0})
x = np.arange(NING + 1)
for ax, nout in zip(axes, NOUTS):
    ax.axvspan(-0.5, 0.5, color="0.92", zorder=0)
    ax.axvspan(NING - 0.5, NING + 0.5, color="0.92", zorder=0)
    for (sk, slab), col in zip(SCEN, COLORS):
        for mk, mlab, ls in MODES:
            y = per_bin_brier(sk, mk, nout)
            ax.plot(x, y, color=col, linestyle=ls, linewidth=1.7, alpha=0.5)
    ax.set_xlim(-0.5, NING + 0.5)
    # symlog, not log: a bin whose Brier is exactly 0 needs the linear region
    # through zero, the interior bins need the logarithmic decades above it.
    ax.set_yscale("symlog", linthresh=3e-2, linscale=0.35)
    # Headroom above the Brier maximum of 2 so the 10^0 tick sits inside the
    # panel rather than on the seam with the panel above.
    ax.set_ylim(0, 4)
    ax.set_xticks(range(0, NING + 1, 2))
    # No tick at 0 and none at the very top: with hspace=0 an edge tick label
    # collides with its neighbour panel across the seam. The linear region
    # still reaches 0, it is simply unlabelled.
    ax.set_yticks([1e-2, 1e-1, 1.0])
    ax.grid(True, alpha=0.25, linewidth=0.5)
    # Outgroup count as a boxed label in the top-left of each panel.
    ax.text(0.012, 0.9, rf"$n_\mathrm{{out}} = {nout}$", transform=ax.transAxes,
            ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8",
                      alpha=0.85))

axes[-1].set_xlabel(r"unfolded ingroup derived-allele count $i$  (n$_\mathrm{ingroup}=20$)")
fig.supylabel("Brier per bin", fontsize=9, x=0.045, y=0.655)

scen_h = [Line2D([0], [0], color=c, lw=2.4, label=l) for (k, l), c in zip(SCEN, COLORS)]
mode_h = [Line2D([0], [0], color="0.25", linestyle=ls, lw=1.6, label=l) for k, l, ls in MODES]
# Reserve a bottom band and put the two legend rows there (deterministic, no
# reliance on tight-bbox expansion).
fig.subplots_adjust(left=0.135, right=0.90, top=0.985, bottom=0.32, hspace=0.0)
fig.legend(handles=scen_h, title="Simulation scenario", ncol=5, loc="lower center",
           bbox_to_anchor=(0.5, 0.135), fontsize=8, title_fontsize=8, framealpha=0.9)
fig.legend(handles=mode_h, title="Inference mode", ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.045), fontsize=8, title_fontsize=8, framealpha=0.9)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
