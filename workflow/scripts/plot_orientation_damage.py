"""Appendix figure: genealogy recovery under mis-orientation, one panel per ARG
method. Solid is the oracle ancestral allele and dashed the mis-oriented arm,
with the shaded gap the cost. The three dimensionless metrics share the left
axis in natural units; Kendall-Colijn takes a twin right axis.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402

try:
    in_json = snakemake.input.report  # type: ignore[name-defined]
    out_pdf = snakemake.output.pdf  # type: ignore[name-defined]
    scheme = snakemake.params.scheme  # type: ignore[name-defined]
except NameError:
    in_json = "results/reports/orientation_damage.json"
    out_pdf = "results/reports/bench_orientation_damage.pdf"
    scheme = "freq_biased5"

KC_SCALE = 1e5
TOOL_LABELS = {"tsinfer": "tsinfer + tsdate", "relate": "Relate"}
SCHEME_LABELS = {
    "random5": "random 5% mis-orientation",
    "freq_biased5": "frequency-biased 5% mis-orientation",
    "major_allele": "major-allele orientation",
    "fixed_tree_n1": "fixed-tree, 1 outgroup",
    "fixed_tree_n3": "fixed-tree, 3 outgroups",
}
#: Left-axis metrics as (block, oracle field, scheme field, colour, marker, label).
LEFT = [
    ("rf_norm", "mean_true_aa", "mean_scheme", "#762a83", "s",
     "Robinson–Foulds, norm. ($\\downarrow$)"),
    ("tmrca", "tmrca_spearman_true_aa", "tmrca_spearman_scheme", "#1b7837",
     "o", r"TMRCA rank $\rho$ ($\uparrow$)"),
    ("tmrca", "tmrca_slope_true_aa", "tmrca_slope_scheme", "#2166ac", "^",
     r"TMRCA slope ($\uparrow$)"),
]


def _tick(label: str) -> str:
    """Shorten a bin label to its upper edge for the x axis.

    :param label: Bin label as written by the report.
    :return: The upper edge, with an open final bin kept as ``> edge``.
    """
    if label.startswith("0"):
        return "0"
    if label.startswith(">"):
        return "> " + label.lstrip("> ").strip()
    return label.rsplit(" ", 1)[-1].strip("(],")


payload = json.loads(Path(in_json).read_text())
tools = [t for t in payload["tools"] if t in TOOL_LABELS]

fig, axes = plt.subplots(1, len(tools), figsize=(9.8, 4.0), sharey=True)
for ax, tool in zip(np.atleast_1d(axes), tools):
    cov = payload["comparisons"][tool][scheme]["by_distance_bp"]
    labels = cov["bin_labels"]
    bins = cov["bins"][:len(labels)]
    x = np.arange(len(bins))
    axR = ax.twinx()
    for block, fa, fb, colour, marker, _ in LEFT:
        a = [b[block][fa] for b in bins]
        d = [b[block][fb] for b in bins]
        ax.plot(x, a, "-", marker=marker, color=colour, lw=1.7, ms=4.5)
        ax.plot(x, d, "--", marker=marker, color=colour, lw=1.5, ms=4.5,
                alpha=0.8, markerfacecolor="white")
        ax.fill_between(x, a, d, color=colour, alpha=0.10, lw=0)
    ka = [b["kc_lambda1"]["mean_true_aa"] / KC_SCALE for b in bins]
    kb = [b["kc_lambda1"]["mean_scheme"] / KC_SCALE for b in bins]
    axR.plot(x, ka, "-", marker="v", color="0.4", lw=1.7, ms=4.5)
    axR.plot(x, kb, "--", marker="v", color="0.4", lw=1.5, ms=4.5, alpha=0.8,
             markerfacecolor="white")
    axR.fill_between(x, ka, kb, color="0.4", alpha=0.10, lw=0)

    ax.set_xticks(x)
    ax.set_xticklabels([_tick(lab) for lab in labels], fontsize=8)
    ax.set_title(TOOL_LABELS[tool], fontsize=13.8, pad=16)
    ax.set_xlim(-0.3, len(bins) - 0.7)
    ax.grid(True, alpha=0.3)
    for cnt, xi in zip(cov["bin_counts"][:len(bins)], x):
        ax.annotate(f"{cnt:,}", (xi, 1.015), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=6.5, color="0.45")
    if tool == tools[-1]:
        axR.set_ylabel(r"Kendall–Colijn ($\times 10^5$ gen., $\downarrow$)")
    else:
        axR.set_yticklabels([])

handles = [mlines.Line2D([], [], color=c, lw=1.8, marker=m, ms=4.5, label=lab)
           for _, _, _, c, m, lab in LEFT]
handles.append(mlines.Line2D(
    [], [], color="0.4", lw=1.8, marker="v", ms=4.5,
    label="Kendall–Colijn ($\\times 10^5$, right; $\\downarrow$)"))
handles += [
    mlines.Line2D([], [], color="0.25", lw=1.8, label="oracle ancestral allele"),
    mlines.Line2D([], [], color="0.25", lw=1.6, ls="--",
                  label=SCHEME_LABELS.get(scheme, scheme)),
]
np.atleast_1d(axes)[0].set_ylabel("metric value")
fig.supxlabel("distance to nearest mis-oriented site (bp)", fontsize=10, y=0.185)
fig.subplots_adjust(left=0.075, right=0.915, top=0.83, bottom=0.31, wspace=0.07)
fig.legend(handles=handles, ncol=3, loc="lower center",
           bbox_to_anchor=(0.5, 0.035), fontsize=8, framealpha=0.9)
for path in (out_pdf,):
    fig.savefig(path, dpi=200)
