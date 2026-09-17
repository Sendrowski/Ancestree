"""One local-tree appendix comparison figure: the same benchmark plot drawn
side by side for n_out = 1 (left) and n_out = 3 (right) with a shared y-axis,
showing how a single outgroup already makes ancestral-allele recovery robust.

``which`` selects the plot, mirroring the n_out = 0 main-text figures:

- ``scans``      -- window size and the rate / phasing misspecification scans
                    on one axes per outgroup count, laid out as in the
                    ingroup-only main-text figure.
- ``accuracy``   -- Brier / MAP vs. window size.
- ``rate``       -- recombination / mutation / phasing misspecification.
- ``genealogy``  -- genealogy-recovery metrics vs. window size. The legend is
                    placed below the panels.

The Brier score is computed over the ingroup-polymorphic sites (the same
target as the n_out = 0 figures), so the panels are directly comparable.

Reads the per-n_out aggregation JSONs (``window_jsons`` and, for ``genealogy``,
``gen_jsons``), built by the report rules. Runs no inference.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _local_tree_scan import _win_int, draw_scan_panel  # noqa: E402

try:
    which = str(snakemake.params.which)  # type: ignore[name-defined]
    n_outs = [int(x) for x in snakemake.params.n_outs]  # type: ignore[name-defined]
    window_jsons = list(snakemake.input.window)  # type: ignore[name-defined]
    gen_jsons = list(getattr(snakemake.input, "genealogy", []))  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
except NameError:
    which = "accuracy"
    n_outs = [1, 3]
    window_jsons = ["results/data/local_tree_window_og1.json",
                    "results/data/local_tree_window_og3.json"]
    gen_jsons = ["results/data/local_tree_genealogy_og1.json",
                 "results/data/local_tree_genealogy_og3.json"]
    out_pdf = f"results/reports/bench_local_tree_{which}_outgroups.pdf"


wbs = [json.loads(Path(p).read_text()) for p in window_jsons]
gens = [json.loads(Path(p).read_text()) for p in gen_jsons] if gen_jsons else []


def _accuracy(axL, wb):
    wins = [_win_int(w) for w in wb["windows"]]
    cells = wb["cells"]
    brier = [cells[f"{w}__r1.0"]["mean_brier"] for w in wb["windows"]]
    mapc = [cells[f"{w}__r1.0"]["mean_map"] for w in wb["windows"]]
    ceil = wb["true_arg_ceiling"]
    axR = axL.twinx()
    lw = 1.8
    al = 0.75
    l1, = axL.plot(wins, mapc, "o-", lw=lw, color="#1b7837", alpha=al,
                   label=r"MAP accuracy ($\uparrow$)")
    l3, = axR.plot(wins, brier, "^-", lw=lw, color="#762a83", alpha=al,
                   label=r"Brier (right axis, $\downarrow$)")
    axL.axhline(ceil["mean_map"], ls="--", lw=lw, color="#1b7837", alpha=0.7)
    axR.axhline(ceil["mean_brier"], ls="--", lw=lw, color="#762a83", alpha=0.7)
    axL.set_xscale("log")
    axL.set_xticks(wins); axL.set_xticklabels(wins, rotation=45, ha="right", fontsize=8)
    axL.set_xlabel("window size (SNPs)")
    proxy = mlines.Line2D([], [], ls="--", lw=lw, color="0.4", label="true-ARG ceiling")
    return axR, [l1, l3, proxy]


def _rate(ax, wb):
    rec_specs = wb["rec_scan_specs"]; mu_specs = wb["mu_specs"]; sw_specs = wb["switch_specs"]
    dw = wb["default_window"]
    idx = list(range(len(rec_specs)))
    rec_y = [wb["rec_scan_cells"][f"r{r}"]["mean_brier"] for r in rec_specs]
    mu_y = [wb["mu_cells"][f"{dw}__m{m}"]["mean_brier"] for m in mu_specs]
    sw_y = [wb["switch_cells"][f"{dw}__s{s}"]["mean_brier"] for s in sw_specs]
    ceil = wb["true_arg_ceiling"]["mean_brier"]
    ax.plot(idx, rec_y, "o-", color="C0", lw=1.6, alpha=0.85, label="recombination rate")
    ax.plot(idx, mu_y, "s-", color="C3", lw=1.6, alpha=0.85, label="mutation rate")
    ax.plot(idx, sw_y, "^-", color="C2", lw=1.6, alpha=0.85, label="phasing switch error (top axis)")
    ax.axhline(ceil, ls="--", color="0.35", lw=1.4, label="true-ARG ceiling")
    ax.set_xticks(idx); ax.set_xticklabels([f"{float(r):g}$\\times$" for r in rec_specs])
    ax.set_xlabel(r"assumed rate ($\times\,r_\mathrm{true}$, $\times\,\mu_\mathrm{true}$)")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    # Switch-error rate on a top twin axis (its meaning is given in the
    # caption). The n_out title is placed on this twin so it clears the top
    # tick labels in both panels.
    axt = ax.twiny(); axt.set_xlim(ax.get_xlim()); axt.set_xticks(idx)
    axt.set_xticklabels([f"{float(s) * 100:g}%" for s in sw_specs])
    return axt


def _genealogy(axL, gen, wb):
    gw = [r["window_snps"] for r in gen["windows"]]
    rho = [r["tmrca_spearman"] for r in gen["windows"]]
    rf = [r["rf_norm"] for r in gen["windows"]]
    slope = [r["tmrca_slope"] for r in gen["windows"]]
    bias = [r["tmrca_log_bias"] for r in gen["windows"]]
    kc = [r["kc_lambda1"] / 1e5 for r in gen["windows"]]
    cells = wb["cells"]
    brier_by_w = {_win_int(w): cells[f"{w}__r1.0"]["mean_brier"] for w in wb["windows"]}
    brier_gen = [brier_by_w.get(w, float("nan")) for w in gw]
    axR = axL.twinx()
    h1, = axL.plot(gw, rho, "o-", color="#1b7837", label=r"TMRCA rank $\rho$ ($\uparrow$)")
    h2, = axL.plot(gw, rf, "s-", color="#762a83", label="Robinson–Foulds, norm. ($\\downarrow$)")
    h3, = axL.plot(gw, slope, "^-", color="#2166ac", label=r"TMRCA slope ($\uparrow$)")
    h4, = axL.plot(gw, bias, "d-", color="#d6604d", label=r"TMRCA log-bias ($\downarrow$)")
    h6, = axL.plot(gw, brier_gen, "P:", color="#b8860b", lw=2.0, label=r"mean Brier, correct rates ($\downarrow$)")
    h5, = axR.plot(gw, kc, "v--", color="0.4", label="Kendall–Colijn ($\\times 10^5$, right; $\\downarrow$)")
    axL.set_xscale("log")
    axL.set_xticks(gw); axL.set_xticklabels(gw, rotation=45, ha="right", fontsize=8)
    axL.set_xlabel("window size (SNPs)")
    handles = [h1, h2, h3, h4, h6, h5]
    # tsinfer (correct-AA) reference: dotted horizontals in the matching metric
    # colours (same convention as the n_out=0 figure).
    ts_ref = gen.get("tsinfer")
    if ts_ref:
        for val, c in [(ts_ref["tmrca_spearman"], "#1b7837"), (ts_ref["rf_norm"], "#762a83"),
                       (ts_ref["tmrca_slope"], "#2166ac"), (ts_ref["tmrca_log_bias"], "#d6604d")]:
            axL.axhline(val, ls=(0, (1, 1)), lw=1.6, color=c, alpha=0.8)
        mb = ts_ref.get("mean_brier")
        if mb is not None and mb == mb:
            axL.axhline(mb, ls=(0, (1, 1)), lw=1.6, color="#b8860b", alpha=0.8)
        axR.axhline(ts_ref["kc_lambda1"] / 1e5, ls=(0, (1, 1)), lw=1.6, color="0.4", alpha=0.8)
        handles.append(mlines.Line2D([], [], ls=(0, (1, 1)), lw=1.6, color="0.3",
                                     label="tsinfer (correct AA)"))
    return axR, handles


if which == "scans":
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.4), sharey=True)
    fig.subplots_adjust(top=0.76, bottom=0.30, wspace=0.10)
    handles = labels = None
    for col, (ax, wb, n_out) in enumerate(zip(axes, wbs, n_outs)):
        handles, labels = draw_scan_panel(fig, ax, wb, ylabel=(col == 0))
        ax.text(0.5, 0.93, f"$n_\\mathrm{{out}} = {n_out}$", transform=ax.transAxes,
                ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9))
    _mid = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.5,
               frameon=True, framealpha=0.95, edgecolor="0.5",
               handlelength=1.8, handletextpad=0.5, columnspacing=1.2,
               borderpad=0.45, labelspacing=0.0,
               bbox_to_anchor=(_mid, 0.155), bbox_transform=fig.transFigure)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"wrote {out_pdf}")
elif which == "rate":
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.7), sharey=True)
    handles = None
    for col, (ax, wb, n_out) in enumerate(zip(axes, wbs, n_outs)):
        axt = _rate(ax, wb)
        axt.set_title(f"$n_\\mathrm{{out}} = {n_out}$")
        if col == 0:
            ax.set_ylabel("mean Brier (lower = better)")
            handles = ax.get_legend_handles_labels()
    # y-axis fully automatic (no forced bottom)
    fig.legend(*handles, loc="lower center", ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.10, 1, 1))

elif which == "accuracy":
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    axRs, handles = [], None
    for col, (ax, wb, n_out) in enumerate(zip(axes, wbs, n_outs)):
        axR, hs = _accuracy(ax, wb)
        axRs.append(axR)
        ax.set_title(f"$n_\\mathrm{{out}} = {n_out}$")
        if col == 0:
            ax.set_ylabel("accuracy (higher = better)")
            handles = hs
        else:
            axR.set_ylabel("mean Brier score (lower = better)")
    axes[0].set_ylim(top=1.0)  # accuracy cannot exceed 1
    rlo = min(min(a.get_ylim()) for a in axRs)
    rhi = max(max(a.get_ylim()) for a in axRs)
    for a in axRs:
        a.set_ylim(rlo, rhi)
    axRs[0].set_yticklabels([])
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))

else:  # genealogy
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    axRs, handles = [], None
    for col, (ax, gen, wb, n_out) in enumerate(zip(axes, gens, wbs, n_outs)):
        axR, hs = _genealogy(ax, gen, wb)
        axRs.append(axR)
        ax.set_title(f"$n_\\mathrm{{out}} = {n_out}$")
        if col == 0:
            ax.set_ylabel("metric value")
            handles = hs
        else:
            axR.set_ylabel(r"Kendall-Colijn ($\times 10^5$ gen.)")
    rlo = min(min(a.get_ylim()) for a in axRs)
    rhi = max(max(a.get_ylim()) for a in axRs)
    for a in axRs:
        a.set_ylim(rlo, rhi)
    axRs[0].set_yticklabels([])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.11), columnspacing=1.1,
               handlelength=1.6, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}", flush=True)
