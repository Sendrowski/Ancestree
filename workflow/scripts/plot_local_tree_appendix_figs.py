"""Render the local-tree robustness figure from the committed benchmark JSONs.

One figure, two panels side by side:

- (a) ancestral-allele recovery (mean Brier) against the window size, the
  assumed recombination and mutation rates, and the phasing switch-error rate,
  with the true-ARG ceiling;
- (b) genealogy recovery (TMRCA rank rho, normalised RF, regression slope,
  TMRCA log-bias, Kendall-Colijn distance and the downstream mean Brier)
  against the window size, with a tsinfer ARG as a window-independent
  reference.

No inference is re-run.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _local_tree_scan import draw_scan_panel  # noqa: E402

try:
    in_window = snakemake.input.window  # type: ignore[name-defined]
    in_genealogy = snakemake.input.genealogy  # type: ignore[name-defined]
    out_pdf = snakemake.output.pdf  # type: ignore[name-defined]
except NameError:
    in_window = "results/reports/local_tree_window_bench.json"
    in_genealogy = "results/reports/local_tree_genealogy.json"
    out_pdf = "results/reports/bench_local_tree_window_genealogy.pdf"

#: Font scale for panels drawn at half the text width.
FS = 1.25
LEGEND = dict(frameon=True, framealpha=0.95, edgecolor="0.5", fontsize=7.5 * FS,
              handlelength=1.8, handletextpad=0.5, columnspacing=1.0,
              borderpad=0.45, labelspacing=0.3)

wb = json.loads(Path(in_window).read_text())
gen = json.loads(Path(in_genealogy).read_text())

FIG_H = 4.6
fig = plt.figure(figsize=(11.0, FIG_H))
# The right panel spans the left one's data area and its two stacked
# rate/switch axes, so both blocks share their height and edges.
BOTTOM, TOP_A = 0.38, 0.80
top_block = TOP_A + (36 * FS / 72.0) / FIG_H
ax_a = fig.add_axes([0.065, BOTTOM, 0.38, TOP_A - BOTTOM])
ax_b = fig.add_axes([0.555, BOTTOM, 0.38, top_block - BOTTOM])

# ------------------------------------------------ (a) window size + robustness
handles_a, labels_a = draw_scan_panel(fig, ax_a, wb, fs=FS)
LEGEND_Y = 0.25
fig.legend(handles_a, labels_a, loc="upper center", ncol=2,
           bbox_to_anchor=(0.065 + 0.19, LEGEND_Y), **LEGEND)

# ---------------------------------------------------- (b) genealogy recovery
gw = [r["window_snps"] for r in gen["windows"]]
rho = [r["tmrca_spearman"] for r in gen["windows"]]
rf = [r["rf_norm"] for r in gen["windows"]]
slope = [r["tmrca_slope"] for r in gen["windows"]]
bias = [r["tmrca_log_bias"] for r in gen["windows"]]
kc = [r["kc_lambda1"] / 1e5 for r in gen["windows"]]
# Mean Brier at the well-specified rates, scored on the same genealogy sim and
# the same per-window ARGs as the recovery metrics.
brier_gen = [r.get("mean_brier", float("nan")) for r in gen["windows"]]

ax_r = ax_b.twinx()
h1, = ax_b.plot(gw, rho, "o-", color="#1b7837", label=r"TMRCA rank $\rho$ ($\uparrow$)")
h2, = ax_b.plot(gw, rf, "s-", color="#762a83", label="Robinson–Foulds, norm. ($\\downarrow$)")
h3, = ax_b.plot(gw, slope, "^-", color="#2166ac", label=r"TMRCA slope ($\uparrow$)")
h4, = ax_b.plot(gw, bias, "d-", color="#d6604d", label=r"TMRCA log-bias ($\downarrow$)")
h6, = ax_b.plot(gw, brier_gen, "P:", color="#b8860b", lw=2.0, label=r"mean Brier ($\downarrow$)")
h5, = ax_r.plot(gw, kc, "v--", color="0.4",
                label="Kendall–Colijn ($\\times 10^5$, right; $\\downarrow$)")
handles_b = [h1, h2, h3, h4, h6, h5]

# tsinfer ARG built from the same genotypes with the correct ancestral allele
# and dated with tsdate: dotted horizontals in the matching metric colours.
ts_ref = gen.get("tsinfer")
if ts_ref:
    refs = [(ts_ref["tmrca_spearman"], "#1b7837"), (ts_ref["rf_norm"], "#762a83"),
            (ts_ref["tmrca_slope"], "#2166ac"), (ts_ref["tmrca_log_bias"], "#d6604d")]
    if "mean_brier" in ts_ref:
        refs.append((ts_ref["mean_brier"], "#b8860b"))
    for val, c in refs:
        ax_b.axhline(val, ls=(0, (1, 1)), lw=1.6, color=c, alpha=0.8)
    ax_r.axhline(ts_ref["kc_lambda1"] / 1e5, ls=(0, (1, 1)), lw=1.6, color="0.4", alpha=0.8)
    handles_b.append(mlines.Line2D([], [], ls=(0, (1, 1)), lw=1.6, color="0.3",
                                   label="tsinfer (correct AA)"))

ax_b.set_xscale("log")
ax_b.set_xticks(gw)
ax_b.set_xticklabels(gw)
ax_b.minorticks_off()
ax_b.tick_params(axis="x", labelsize=8 * FS)
ax_b.tick_params(axis="y", labelsize=9 * FS)
ax_r.tick_params(axis="y", labelsize=9 * FS)
ax_b.set_xlabel("window size (SNPs)", fontsize=10 * FS)
ax_b.set_ylabel("metric value", fontsize=10 * FS)
ax_r.set_ylabel(r"Kendall–Colijn ($\times 10^5$ gen.)", fontsize=10 * FS)
fig.legend(handles=handles_b, loc="upper center", ncol=2,
           bbox_to_anchor=(0.555 + 0.19, LEGEND_Y), **LEGEND)

# The right panel is titled; the left one is labelled by its stacked axes.
ax_b.set_title("Genealogy recovery", fontsize=11 * FS, pad=10)

fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}")
