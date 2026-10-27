"""Divergence counts from MAP estimates and from posterior weights against the
simulated truth, across the robustness scenarios, outgroup counts and the three
inference modes.

Built from the per-cell robustness summaries
(``robustness_cell_<scenario>_<method>_n<k>_summary.json``, the
``infer_robustness_cell`` outputs). For each ingroup-monomorphic bin at true
derived count ``b`` the summary stores
``[count, pv_n, map_sum, ptrue_sum, sumsq_sum]``, graded at the root of the
full outgroup panel. The number of sites the ingroup has fixed for a derived
allele follows as

    true:     count at b = n
    MAP:      map_sum at b = n, plus (count - map_sum) at b = 0
    expected: ptrue_sum at b = n, plus (pv_n - ptrue_sum) at b = 0

so no inference is re-run here. At ``b = n`` the posterior mass on a third
allele is not counted, which makes both estimates lower bounds there.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCEN = [("baseline", "Baseline"), ("cpg_hypermut", "Hypermutation"),
        ("strong_ils", "Strong ILS"), ("outgroup_clade", "Outgroup clade"),
        ("slim", "Purifying")]
MODES = [("anc_arg", "ARG mode (true ARG)"),
         ("local_tree_unphased", "local-tree mode"),
         ("anc_ft", "fixed-tree mode")]
N_OUTS = [1, 3, 10]
#: Outgroup counts drawn as figure rows.
N_OUTS_PLOT = [3, 10]

try:
    CELLS = {Path(p).name: Path(p) for p in snakemake.input.cells}  # type: ignore[name-defined]
    FIG = Path(snakemake.output.pdf)  # type: ignore[name-defined]
    OUT_JSON = Path(snakemake.output.json)  # type: ignore[name-defined]
except NameError:
    DATA = Path("results/data")
    CELLS = {
        f"robustness_cell_{key}_{mode}_n{n_out}_summary.json":
            DATA / f"robustness_cell_{key}_{mode}_n{n_out}_summary.json"
        for key, _ in SCEN for mode, _ in MODES for n_out in N_OUTS
    }
    FIG = Path("results/reports/bench_divergence_counts.pdf")
    OUT_JSON = Path("results/reports/bench_divergence_counts.json")


def _counts(summary):
    """True, MAP and posterior-weighted fixed-derived counts of one cell.

    :param summary: one per-cell robustness summary, as loaded from JSON.
    :return: ``(true, map, expected)`` site counts.
    """
    n = summary["n_ingroup"]
    true = map_ = expected = 0.0
    for key, g in summary["stats"].items():
        klass, b, _rec = key.split("|")
        b = int(b)
        if not klass.endswith("_mono"):  # ingroup-fixed sites only
            continue
        count, pv_n, map_sum, ptrue_sum = g[0], g[1], g[2], g[3]
        if b == n:
            true += count
            map_ += map_sum
            expected += ptrue_sum
        elif b == 0:
            map_ += count - map_sum
            expected += pv_n - ptrue_sum
    return true, map_, expected


table = {}
for key, _ in SCEN:
    for mode, _ in MODES:
        for n_out in N_OUTS:
            f = CELLS[f"robustness_cell_{key}_{mode}_n{n_out}_summary.json"]
            t, m, e = _counts(json.loads(Path(f).read_text(encoding="utf-8")))
            table[f"{key}|{mode}|{n_out}"] = dict(
                true=t, map=m, expected=e,
                map_rel_error=m / t - 1, expected_rel_error=e / t - 1,
            )
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(table, indent=1), encoding="utf-8")

fig, axes = plt.subplots(len(N_OUTS_PLOT), len(MODES), figsize=(9.4, 4.6),
                         sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0.05, "hspace": 0.1})
x = np.arange(len(SCEN))
w = 0.38
for r, n_out in enumerate(N_OUTS_PLOT):
    for c, (mode, mode_label) in enumerate(MODES):
        ax = axes[r][c]
        rows = [table[f"{key}|{mode}|{n_out}"] for key, _ in SCEN]
        ax.bar(x - w / 2, [100 * v["map_rel_error"] for v in rows], w,
               color="#d62728", alpha=0.75, label="MAP estimates")
        ax.bar(x + w / 2, [100 * v["expected_rel_error"] for v in rows], w,
               color="#1f77b4", alpha=0.75, label="posterior-weighted (expected)")
        ax.axhline(0, color="0.15", lw=0.8)
        ax.set_ylim(-100, 100)
        ax.grid(True, axis="y", alpha=0.2, lw=0.4)
        if r == 0:
            ax.set_title(mode_label, fontsize=10)
        if c == 0:
            ax.set_ylabel(rf"$n_\mathrm{{out}} = {n_out}$", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in SCEN], rotation=35,
                           ha="right", fontsize=8)
axes[0][0].legend(fontsize=7, framealpha=0.9, loc="upper left")
fig.supylabel("relative error of the divergence count (%)", fontsize=10, x=0.01)
fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.2)
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG)
print(f"wrote {FIG} + {OUT_JSON}", flush=True)
