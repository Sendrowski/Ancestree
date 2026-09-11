"""Appendix figure: the root-state prior under demographic distortion, in
fixed-tree mode, across outgroup counts 0, 1 and 3 (one row each).

Three priors are compared: the Kingman prior (neutral coalescent expectation
P(root=a)=n_a/n), the adaptive prior (per-bin probabilities fit from the data),
and the stationary prior (uniform over ancestral states under JC69, the
uninformative reference). Each panel shows the fractionally reconstructed
(expected) unfolded SFS under each prior against the ground truth, for three
demographies (a constant-size baseline, a recent expansion, a recent
contraction).

With no outgroup the posterior is the normalised prior, so the priors diverge:
the uniform prior cannot break the orientation symmetry and folds the spectrum,
while the Kingman prior pulls it toward the neutral expectation. A single
outgroup fixes the orientation through the Felsenstein likelihood, so all three
priors converge to the truth.

Built from the per-cell robustness summaries
(``robustness_cell_<scenario>_<method>_n<k>_summary.json``); no inference is
re-run. For each biallelic-polymorphic bin at true derived count ``b`` the
summary stores ``[count, pv_n, map_sum, ptrue_sum, sumsq_sum]``; the true and
fractional/expected spectra follow as

    true:     count at b
    expected: ptrue_sum at b, (pv_n - ptrue_sum) at the fold partner n-b
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402

SCEN = [("pop_constant", "Baseline", "#222222"),
        ("pop_growth", "Growth", "#1b9e77"),
        ("pop_decline", "Decline", "#7570b3")]
# Each prior is drawn as a connected curve with its own dash + marker, so it
# can be followed at a glance; the truth is a thick solid line.
PRIORS = [("anc_ft", "Kingman", (0, (5, 2)), "o", dict(mfc="none", mew=1.3, ms=4.5)),
          ("anc_ft_adaptive", "adaptive", (0, (1, 1.2)), "s", dict(mfc="none", mew=1.3, ms=4.0)),
          ("anc_ft_uniform", "uniform", (0, (4, 1.5, 1, 1.5)), "v", dict(mfc="none", mew=1.3, ms=4.5))]
N_OUTS = [0, 1, 3]

try:
    CELLS = {Path(p).name: Path(p) for p in snakemake.input.cells}  # type: ignore[name-defined]
    FIG = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    DATA = Path("results/data")
    CELLS = {
        f"robustness_cell_{key}_{method}_n{n_out}_summary.json":
            DATA / f"robustness_cell_{key}_{method}_n{n_out}_summary.json"
        for key, _label, _color in SCEN
        for method, *_rest in PRIORS
        for n_out in N_OUTS
    }
    FIG = Path("reports/manuscripts/latex/figures/bench_prior_demog.pdf")

#: Screen copy of the figure, for review outside a PDF viewer.
PNG = Path("results/reports/bench_prior_demog.png")


def _cell(scenario: str, method: str, n_out: int) -> dict:
    """One per-cell robustness summary.

    :param scenario: scenario key, e.g. ``pop_growth``.
    :param method: method key, e.g. ``anc_ft_adaptive``.
    :param n_out: number of outgroups in the cell.
    :return: the summary as loaded from JSON.
    """
    name = f"robustness_cell_{scenario}_{method}_n{n_out}_summary.json"
    return json.loads(Path(CELLS[name]).read_text())


def _spectra(summary):
    """True and posterior-weighted (expected) unfolded SFS (proportions) from a
    cell. The expected spectrum renormalises each site's posterior over its
    observed alleles (the ``sfs_stats`` sum of p_true / p_obs); when that field
    is absent it falls back to the raw p_true (equal when the posterior is
    already supported on the observed alleles).

    :param summary: one per-cell robustness summary, as loaded from JSON.
    :return: ``(x, true, expected)`` over derived counts ``1..n-1``.
    """
    n = summary["n_ingroup"]
    T = np.zeros(n + 1); E = np.zeros(n + 1)
    sfs = summary.get("sfs_stats") or {}
    for key, g in summary["stats"].items():
        klass, b, _rec = key.split("|")
        if klass != "biallelic_poly":
            continue
        b = int(b)
        if 0 < b < n:
            count, pv_n, ptrue = g[0], g[1], g[3]
            T[b] += count
            w = sfs.get(key, ptrue)  # observed-renormalised true weight
            E[b] += w; E[n - b] += pv_n - w
    x = np.arange(1, n)
    return x, T[1:n] / T[1:n].sum(), E[1:n] / E[1:n].sum()


fig, axes = plt.subplots(len(N_OUTS), 1, figsize=(7.2, 3.9), sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0})
for ax, n_out in zip(axes, N_OUTS):
    l2 = {m[0]: [] for m in PRIORS}
    for key, label, color in SCEN:
        # true SFS (prior-independent): read from the Kingman cell
        x, T, _ = _spectra(_cell(key, "anc_ft", n_out))
        ax.plot(x, T, "-", color=color, lw=3.0, alpha=0.55, solid_capstyle="round", zorder=1)
        for method, _ml, ls, marker, kw in PRIORS:
            _, _, E = _spectra(_cell(key, method, n_out))
            ax.plot(x, E, ls=ls, marker=marker, color=color, lw=1.3, alpha=0.65,
                    markevery=2, zorder=3, **kw)
            l2[method].append(float(np.sum(np.abs(E - T))))
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.18, lw=0.4)
    # n_out as a right-side row label, outside the frame.
    ax.text(1.015, 0.5, rf"$n_\mathrm{{out}} = {n_out}$", transform=ax.transAxes,
            rotation=270, va="center", ha="left", fontsize=9.5)
    # Mean L1 (over scenarios) of each prior's reconstructed SFS to the truth,
    # one row centered near the top of each panel.
    l2txt = r"$L_1$ to true SFS:  " + ",  ".join(
        f"{ml} {np.mean(l2[m]):.3f}" for m, ml, *_ in PRIORS)
    ax.text(0.5, 0.93, l2txt, transform=ax.transAxes, ha="center", va="top",
            fontsize=6.8, bbox=dict(boxstyle="round", fc="white", ec="0.85", alpha=0.8))
axes[-1].set_xlabel(r"derived count $i$")
fig.supylabel("proportion of segregating sites", fontsize=11, x=0.01)

scen_handles = [mlines.Line2D([], [], color=c, lw=3.0, alpha=0.7, label=lab) for _k, lab, c in SCEN]
src_handles = [mlines.Line2D([], [], color="0.45", lw=3.0, alpha=0.55, label="true SFS")]
src_handles += [mlines.Line2D([], [], color="0.45", ls=ls, marker=mk, lw=1.3, label=ml, **kw)
                for _m, ml, ls, mk, kw in PRIORS]
fig.tight_layout()
# Both legends below the plot in a single row, side by side and centred on the
# plot frame (not the figure, whose centre is shifted by the y-axis label).
_pos = axes[-1].get_position()
_cx = 0.5 * (_pos.x0 + _pos.x1)
fig.legend(handles=scen_handles, loc="upper right", bbox_to_anchor=(_cx - 0.01, 0.0),
           ncol=3, fontsize=8, framealpha=0.92, title="scenario", title_fontsize=8.5)
fig.legend(handles=src_handles, loc="upper left", bbox_to_anchor=(_cx + 0.01, 0.0),
           ncol=4, fontsize=8, framealpha=0.92, title="prior", title_fontsize=8.5)
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, bbox_inches="tight")
PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=150, bbox_inches="tight")
print(f"wrote {FIG} + {PNG}", flush=True)
