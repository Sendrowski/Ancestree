"""Plot accuracy against the focal node's position between the two anchors.

One row per site class, one column per outgroup count, one line per inference
mode. The x axis is the fraction of the path from the ingroup MRCA to the
deepest node the panel provides, so the curves are comparable across scenarios
whose divergences differ.
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:  # snakemake execution
    IN_JSON = snakemake.input.json
    OUT_PDF = snakemake.output.pdf
    FIG_COPY = snakemake.output.fig_copy
except NameError:  # direct execution
    IN_JSON = "results/reports/focal_interpolation.json"
    OUT_PDF = "results/reports/focal_interpolation.pdf"
    FIG_COPY = "reports/manuscripts/latex/figures/focal_interpolation.pdf"

#: One colour per inference mode.
COLOURS = {"fixed_tree": "#fb8500", "arg": "#023047",
           "local_tree": "#219ebc", "local_tree_ens": "#7209b7"}

MODE_LABEL = {"fixed_tree": "fixed tree", "arg": "ARG",
              "local_tree": "local tree",
              "local_tree_ens": "local tree (ensemble)"}

#: One row per panel: the site classes averaged into it, and its y label. The
#: last row weights the two ingroup-monomorphic classes equally despite their
#: unequal site counts, so it is the class-balanced Brier over fixed sites.
ROWS = (
    (("polymorphic",), "polymorphic\nmean Brier"),
    (("fixed_derived",), "fixed, derived\nmean Brier"),
    (("fixed_ancestral",), "fixed, ancestral\nmean Brier"),
    (("fixed_derived", "fixed_ancestral"),
     "fixed, derived\n+ ancestral"),
)


def series(rows: list, n_out: int, site_classes: tuple, mode: str) -> list:
    """Mean Brier averaged over ``site_classes``, per reporting position.

    Each class enters with equal weight regardless of its site count, so a
    multi-class row is the class-balanced Brier score.

    :param rows: The scan records.
    :param n_out: Outgroup count selecting the column.
    :param site_classes: Site classes whose mean Brier scores are averaged.
    :param mode: Inference mode selecting the curve.
    :return: ``(fraction, value)`` pairs sorted by fraction, restricted to the
        positions at which every requested class is present.
    """
    per_class = []
    for site_class in site_classes:
        per_class.append({
            r["fraction"]: r["mean_brier"] for r in rows
            if r["n_out"] == n_out and r["site_class"] == site_class
            and r["mode"] == mode and r["mean_brier"] is not None
            and r.get("grading", "moving") == "moving"
        })
    shared = set.intersection(*(set(d) for d in per_class))
    return [(f, sum(d[f] for d in per_class) / len(per_class))
            for f in sorted(shared)]


def main() -> None:
    """Render the scan to a four-row figure."""
    with open(IN_JSON) as handle:
        payload = json.load(handle)
    rows = payload["rows"]

    n_outs = sorted({r["n_out"] for r in rows})
    modes = [m for m in COLOURS if any(r["mode"] == m for r in rows)]
    fig, axes = plt.subplots(
        len(ROWS), len(n_outs),
        figsize=(3.4 * len(n_outs), 1.4 * len(ROWS)),
        sharex=True, sharey="row", squeeze=False,
    )
    for row, (site_classes, ylabel) in enumerate(ROWS):
        for col, n_out in enumerate(n_outs):
            ax = axes[row][col]
            for mode in modes:
                points = series(rows, n_out, site_classes, mode)
                if not points:
                    continue
                ax.plot([f for f, _ in points], [v for _, v in points],
                        lw=1.6, alpha=0.75, color=COLOURS[mode])
            if row == 0:
                ax.set_title(
                    f"{n_out} outgroup" + ("s" if n_out != 1 else ""),
                    fontsize=12,
                )
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_xticklabels(["ingroup\nMRCA", "0.5", "panel\nroot"])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(alpha=0.25, lw=0.6)
            if col > 0:
                ax.tick_params(axis="y", length=0)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8)
    # Flush panels: pull the end labels inside their own panel, so they do
    # not straddle the spine they share with the neighbouring panel.
    fig.canvas.draw()
    for ax in axes[-1]:
        ax.get_xticklabels()[0].set_horizontalalignment("left")
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
    for ax_row in axes:
        ax = ax_row[0]
        lo, hi = ax.get_ylim()
        shown = [t for t, y in zip(ax.get_yticklabels(), ax.get_yticks())
                 if lo <= y <= hi]
        if shown:
            shown[0].set_verticalalignment("bottom")
            shown[-1].set_verticalalignment("top")
    handles = [Line2D([], [], color=COLOURS[m], lw=2.4, alpha=0.75)
               for m in modes]
    fig.legend(handles, [MODE_LABEL[m] for m in modes], loc="lower center",
               ncol=len(modes), frameon=True, fontsize=11)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.subplots_adjust(wspace=0, hspace=0)
    # The manuscript wants vector. The report page wants the raster.
    fig.savefig(OUT_PDF)
    shutil.copyfile(OUT_PDF, FIG_COPY)
    print(f"wrote {OUT_PDF} and {FIG_COPY}")


main()
