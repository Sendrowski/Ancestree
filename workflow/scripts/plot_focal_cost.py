"""What a misplaced reporting node costs, with the question held fixed.

One curve per site class: ingroup polymorphism graded at the ingroup MRCA
throughout, and sites the ingroup has fixed graded at the panel root
throughout. Only the reporting node moves, so each curve measures the cost of
answering at the wrong node rather than a change of question.
"""
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

try:  # snakemake execution
    IN_JSON = snakemake.input.json  # type: ignore[name-defined]
    OUT_PDF = snakemake.output.pdf  # type: ignore[name-defined]
    FIG_COPY = snakemake.output.fig_copy  # type: ignore[name-defined]
except NameError:  # direct execution
    IN_JSON = "results/reports/focal_interpolation.json"
    OUT_PDF = "results/reports/focal_cost.pdf"
    FIG_COPY = "reports/manuscripts/latex/figures/focal_cost.pdf"

COLOURS = {"fixed_tree": "#fb8500", "arg": "#023047", "local_tree": "#219ebc",
           "local_tree_ens": "#7209b7"}
MODE_LABEL = {"fixed_tree": "fixed tree", "arg": "ARG",
              "local_tree": "local tree",
              "local_tree_ens": "local tree (ensemble)"}

#: One row per panel: the (site class, grading node) pairs averaged into it,
#: and its y label. The last row weights the two ingroup-monomorphic classes
#: equally despite their unequal site counts, so it is the class-balanced
#: Brier over fixed sites.
ROWS = (
    ((("polymorphic", "at_mrca"),), "polymorphic\nmean Brier"),
    ((("fixed_derived", "at_root"),), "fixed, derived\nmean Brier"),
    ((("fixed_ancestral", "at_root"),), "fixed, ancestral\nmean Brier"),
    ((("fixed_derived", "at_root"), ("fixed_ancestral", "at_root")),
     "fixed, derived\n+ ancestral"),
)


def series(rows: list, n_out: int, curves: tuple, mode: str) -> list:
    """Mean Brier averaged over ``curves``, per reporting position.

    Each class enters with equal weight regardless of its site count, so a
    multi-class row is the class-balanced Brier score.

    :param rows: The scan records.
    :param n_out: Outgroup count selecting the column.
    :param curves: ``(site class, grading node)`` pairs to average.
    :param mode: Inference mode selecting the curve.
    :return: ``(fraction, value)`` pairs sorted by fraction, restricted to the
        positions at which every requested class is present.
    """
    per_class = []
    for site_class, grading in curves:
        per_class.append({
            r["fraction"]: r["mean_brier"] for r in rows
            if r["n_out"] == n_out and r["site_class"] == site_class
            and r["mode"] == mode and r.get("grading") == grading
            and r["mean_brier"] is not None
        })
    shared = set.intersection(*(set(d) for d in per_class))
    return [(f, sum(d[f] for d in per_class) / len(per_class))
            for f in sorted(shared)]


def main() -> None:
    """Render the cost curves."""
    with open(IN_JSON) as handle:
        rows = json.load(handle)["rows"]
    n_outs = sorted({r["n_out"] for r in rows})
    modes = [m for m in COLOURS if any(r["mode"] == m for r in rows)]

    fig, axes = plt.subplots(
        len(ROWS), len(n_outs),
        figsize=(3.4 * len(n_outs), 1.4 * len(ROWS)),
        sharex=True, sharey="row", squeeze=False,
    )
    for row, (curves, ylabel) in enumerate(ROWS):
        for col, n_out in enumerate(n_outs):
            ax = axes[row][col]
            for mode in modes:
                pts = series(rows, n_out, curves, mode)
                if not pts:
                    continue
                ax.plot([f for f, _ in pts], [v for _, v in pts],
                        lw=1.6, alpha=0.75, color=COLOURS[mode])
            if row == 0:
                ax.set_title(f"{n_out} outgroup" + ("s" if n_out != 1 else ""),
                             fontsize=12)
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
    fig.savefig(OUT_PDF)
    shutil.copyfile(OUT_PDF, FIG_COPY)
    print(f"wrote {OUT_PDF} and {FIG_COPY}")


main()
