"""Render the two local-tree appendix figures that replace dense tables:

- ``bench_local_tree_window_accuracy.pdf`` -- LocalTreeInference accuracy
  (Brier / MAP) vs. window size at the well-specified recombination rate,
  with the true-ARG ceiling (replaces the window table).
- ``bench_local_tree_genealogy.pdf`` -- genealogy-recovery metrics
  (TMRCA rank rho, normalised RF, regression slope, TMRCA log-bias, and the
  Kendall-Colijn distance) vs. window size (replaces the genealogy table).

Both read the committed benchmark JSONs, so no inference is re-run.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _local_tree_scan import _win_int, draw_scan_panel  # noqa: E402

try:
    in_window = snakemake.input.window  # type: ignore[name-defined]
    in_genealogy = snakemake.input.genealogy  # type: ignore[name-defined]
    out_acc_pdf = snakemake.output.accuracy_pdf  # type: ignore[name-defined]
    out_gen_pdf = snakemake.output.genealogy_pdf  # type: ignore[name-defined]
except NameError:
    in_window = "results/reports/local_tree_window_bench.json"
    in_genealogy = "results/reports/local_tree_genealogy.json"
    out_acc_pdf = "results/reports/bench_local_tree_window_accuracy.pdf"
    out_gen_pdf = "results/reports/bench_local_tree_genealogy.pdf"


# ------------------------------------------------ window size + robustness
# One figure, two panels on a shared Brier axis: the window-size scan on the
# left, the three misspecification scans through the reference window on the
# right. Both carry the true-ARG ceiling, so the gap to it reads the same way
# in either panel.
wb = json.loads(Path(in_window).read_text())
windows = [_win_int(w) for w in wb["windows"]]
cells = wb["cells"]
# Well-specified recombination rate (r = r_true) column.
brier = [cells[f"{w}__r1.0"]["mean_brier"] for w in wb["windows"]]
mapc = [cells[f"{w}__r1.0"]["mean_map"] for w in wb["windows"]]
ceil = wb["true_arg_ceiling"]


# One axes carrying all three scans, each on its own colour-coded x-axis
# stacked around the plot (see :mod:`_local_tree_scan`).
fig, ax = plt.subplots(figsize=(6.6, 3.4))
fig.subplots_adjust(top=0.76, bottom=0.30)
_handles, _labels = draw_scan_panel(fig, ax, wb)

# The legend goes under the window-size axis, centred on the plotting area.
_mid = 0.5 * (ax.get_position().x0 + ax.get_position().x1)
fig.legend(_handles, _labels, loc="upper center", ncol=5, fontsize=7.5,
           frameon=True, framealpha=0.95, edgecolor="0.5",
           handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
           borderpad=0.45, labelspacing=0.0,
           bbox_to_anchor=(_mid, 0.155), bbox_transform=fig.transFigure)

fig.savefig(out_acc_pdf, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------- A6: genealogy
gen = json.loads(Path(in_genealogy).read_text())
gw = [r["window_snps"] for r in gen["windows"]]
rho = [r["tmrca_spearman"] for r in gen["windows"]]
rf = [r["rf_norm"] for r in gen["windows"]]
slope = [r["tmrca_slope"] for r in gen["windows"]]
bias = [r["tmrca_log_bias"] for r in gen["windows"]]
kc = [r["kc_lambda1"] / 1e5 for r in gen["windows"]]

fig, axL = plt.subplots(figsize=(5.6, 4.4))
axR = axL.twinx()

# Mean polarisation Brier at the well-specified rates, scored on the same
# genealogy sim and the same per-window ARGs as the recovery metrics, so the
# overlay is self-consistent and defined at every window --- including the
# 1- and 2-SNP windows the 100 Mb window bench cannot reach.
brier_gen = [r.get("mean_brier", float("nan")) for r in gen["windows"]]

h1, = axL.plot(gw, rho, "o-", color="#1b7837", label=r"TMRCA rank $\rho$ ($\uparrow$)")
h2, = axL.plot(gw, rf, "s-", color="#762a83",
               label="Robinson–Foulds, norm. ($\\downarrow$)")
h3, = axL.plot(gw, slope, "^-", color="#2166ac", label=r"TMRCA slope ($\uparrow$)")
h4, = axL.plot(gw, bias, "d-", color="#d6604d", label=r"TMRCA log-bias ($\downarrow$)")
h6, = axL.plot(gw, brier_gen, "P:", color="#b8860b", lw=2.0,
               label=r"mean Brier ($\downarrow$)")
h5, = axR.plot(gw, kc, "v--", color="0.4",
               label="Kendall–Colijn ($\\times 10^5$, right; $\\downarrow$)")

# tsinfer ARG (built from the same genotypes with the correct ancestral
# allele) as a window-independent reference. Its node times are dated with
# tsdate (see report_local_tree_genealogy.py), so every metric --- rank rho,
# RF, slope, log-bias, Brier (left), KC (right) --- is comparable. Dotted
# horizontals in the matching metric colours.
handles = [h1, h2, h3, h4, h6, h5]
ts_ref = gen.get("tsinfer")
if ts_ref:
    refs = [(ts_ref["tmrca_spearman"], "#1b7837"),
            (ts_ref["rf_norm"], "#762a83"),
            (ts_ref["tmrca_slope"], "#2166ac"),
            (ts_ref["tmrca_log_bias"], "#d6604d")]
    if "mean_brier" in ts_ref:
        refs.append((ts_ref["mean_brier"], "#b8860b"))
    for val, c in refs:
        axL.axhline(val, ls=(0, (1, 1)), lw=1.6, color=c, alpha=0.8)
    axR.axhline(ts_ref["kc_lambda1"] / 1e5, ls=(0, (1, 1)), lw=1.6,
                color="0.4", alpha=0.8)
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], ls=(0, (1, 1)), lw=1.6, color="0.3",
                                 label="tsinfer (correct AA)"))

axL.set_xscale("log")
axL.set_xticks(gw)
axL.set_xticklabels(gw, rotation=45, ha="right", fontsize=8)
axL.set_xlabel("window size (SNPs)")
axL.set_ylabel("metric value")
axR.set_ylabel(r"Kendall-Colijn ($\times 10^5$ gen.)")
# Legend below the plot, freeing the axes (the in-axes legend crowded the
# curves).
axL.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22),
           ncol=3, fontsize=7.5, framealpha=0.95, columnspacing=1.1,
           handlelength=1.6, handletextpad=0.5)
fig.tight_layout()
fig.savefig(out_gen_pdf, bbox_inches="tight")
plt.close(fig)

print("wrote bench_local_tree_window_accuracy.pdf + bench_local_tree_genealogy.pdf")
