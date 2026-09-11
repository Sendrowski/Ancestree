"""Appendix figure: fractional (expected) vs MAP vs ground-truth unfolded SFS
across the robustness scenarios (rows) and outgroup counts (columns), in
ARG mode.

Built from the per-cell robustness summaries
(``robustness_cell_<scenario>_anc_arg_n<k>_summary.json``, the
``infer_robustness_cell`` outputs). For each biallelic-polymorphic
(``klass = biallelic_poly``) bin at true derived count ``b`` the summary stores
``[count, pv_n, map_sum, ptrue_sum, sumsq_sum]``; the spectra follow as

    true:     count at b
    MAP:      map_sum at b, the rest (count - map_sum) at the fold partner n-b
    expected: ptrue_sum at b, (pv_n - ptrue_sum) at n-b

so no inference is re-run here.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCEN = [("baseline", "Baseline"), ("cpg_hypermut", "Hypermutation"),
        ("strong_ils", "Strong ILS"), ("outgroup_clade", "Outgroup clade"),
        ("slim", "Purifying"), ("pop_decline", "Decline")]
N_OUTS = [0, 1, 3, 10]

try:
    CELLS = {Path(p).name: Path(p) for p in snakemake.input.cells}  # type: ignore[name-defined]
    FIG = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    DATA = Path("results/data")
    CELLS = {
        f"robustness_cell_{key}_anc_arg_n{n_out}_summary.json":
            DATA / f"robustness_cell_{key}_anc_arg_n{n_out}_summary.json"
        for key, _label in SCEN for n_out in N_OUTS
    }
    FIG = Path("reports/manuscripts/latex/figures/bench_scenario_nout_fracsfs.pdf")

#: Screen copy of the figure, for review outside a PDF viewer.
PNG = Path("results/reports/bench_scenario_nout_fracsfs.png")


def _spectra(summary):
    """True / MAP / expected unfolded SFS (proportions) from a cell summary.

    :param summary: one per-cell robustness summary, as loaded from JSON.
    :return: ``(x, true, map, expected)`` over derived counts ``1..n-1``.
    """
    n = summary["n_ingroup"]
    T = np.zeros(n + 1); M = np.zeros(n + 1); E = np.zeros(n + 1)
    for key, g in summary["stats"].items():
        klass, b, _rec = key.split("|")
        if klass != "biallelic_poly":
            continue
        b = int(b)
        if b <= 0 or b >= n:
            continue
        count, pv_n, map_sum, ptrue_sum = g[0], g[1], g[2], g[3]
        T[b] += count
        M[b] += map_sum; M[n - b] += count - map_sum
        E[b] += ptrue_sum; E[n - b] += pv_n - ptrue_sum
    x = np.arange(1, n)
    return x, *(a[1:n] / a[1:n].sum() if a[1:n].sum() else a[1:n] for a in (T, M, E))


fig, axes = plt.subplots(len(SCEN), len(N_OUTS), figsize=(9.4, 5.4),
                         sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0, "hspace": 0})
for r, (key, label) in enumerate(SCEN):
    for c, n_out in enumerate(N_OUTS):
        ax = axes[r][c]
        f = CELLS[f"robustness_cell_{key}_anc_arg_n{n_out}_summary.json"]
        x, Tn, Mn, En = _spectra(json.loads(Path(f).read_text()))
        ax.plot(x, Tn, "o-", ms=3, lw=1.5, color="0.15", alpha=0.7, label="true SFS")
        ax.plot(x, Mn, "s-", ms=3, lw=1.3, color="#d62728", alpha=0.6, label="MAP estimates")
        ax.plot(x, En, "^-", ms=3, lw=1.3, color="#1f77b4", alpha=0.6, label="posterior-weighted (expected)")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 0.4)
        ax.grid(True, which="both", alpha=0.2, lw=0.4)
        if r == 0:
            ax.set_title(rf"$n_\mathrm{{out}} = {n_out}$", fontsize=11)
        if c == 0:
            ax.set_ylabel(label, fontsize=8, rotation=0, ha="right",
                          va="center", labelpad=4)
        if r == len(SCEN) - 1:
            ax.set_xlabel(r"derived count $i$")
axes[0][0].legend(fontsize=7, framealpha=0.9, loc="lower left")
fig.supylabel("proportion of segregating sites", fontsize=11, x=0.01)
fig.subplots_adjust(left=0.18, right=0.99, top=0.95, bottom=0.13, wspace=0, hspace=0)
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG)
PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=150)
print(f"wrote {FIG} + {PNG}", flush=True)
