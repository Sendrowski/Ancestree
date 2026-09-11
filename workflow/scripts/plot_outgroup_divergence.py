"""Accuracy against outgroup divergence depth, one line per spacing scheme.

Builds the appendix's B8 figure from ``outgroup_divergence.json``, so it is
reproducible from a clean checkout.

Two panels, ARG and fixed-tree, sharing a y axis: within a panel the lines are
directly comparable, and the top axis restates the depth as expected divergence
in substitutions per site, which is the scale the kernel actually sees.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    in_json = Path(snakemake.input.json)  # noqa: F821
    out_pdf = Path(snakemake.output.pdf)  # noqa: F821
except NameError:
    # Standalone debug defaults; the DAG supplies these through `script:`.
    in_json = Path("results/reports/outgroup_divergence.json")
    out_pdf = Path("results/reports/bench_outgroup_divergence.pdf")

d = json.loads(in_json.read_text())
depths = d["depths"]
spacings = d["spacings"]
mu = float(d["sim_config"]["mu"])
ne = float(d["sim_config"]["pop_size"])
tau = [float(d["sim_config"].get("depth_values", {}).get(lbl, 0)) or
       next(c["depth"] for c in d["arg_cells"] if c["depth_label"] == lbl)
       for lbl in depths]

STYLE = {"linear": ("o", "-", "#1f77b4"), "geometric": ("s", "-", "#ff7f0e"),
         "compact_near": ("^", "--", "#d62728"),
         "compact_far": ("D", "-", "#2ca02c")}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
for ax, (key, title) in zip(axes, (("arg_cells", "ARG mode"),
                                   ("vcf_cells", "Fixed-tree (VCF) mode"))):
    cells = d[key]
    for sp in spacings:
        xs, ys = [], []
        for lbl, t in zip(depths, tau):
            hit = [c for c in cells
                   if c["depth_label"] == lbl and c["spacing"] == sp]
            if hit:
                # Evenly spaced positions, not the raw tau: the depths span
                # 5..100 and a linear axis crushes the first four together.
                xs.append(tau.index(t))
                ys.append(hit[0]["accuracy_polymorphic"])
        m, ls, col = STYLE.get(sp, ("o", "-", None))
        ax.plot(xs, ys, marker=m, linestyle=ls, color=col, label=sp, alpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(tau)))
    ax.set_xticklabels([f"{t / (2.0 * ne):.0f}" for t in tau])
    ax.set_xlabel(r"Deepest-outgroup divergence  $\tau = T/(2N_e)$"
                  f"  [Ne = {ne:.0f}]")
    ax.grid(True, alpha=0.3)
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    top.set_xticks(range(len(tau)))
    top.set_xticklabels([f"{2.0 * mu * t:.3f}" for t in tau], fontsize=8)
    top.set_xlabel(f"Expected divergence (subs/site, $\\mu$ = {mu:g})",
                   fontsize=9)
axes[0].set_ylabel("Accuracy (ingroup-polymorphic sites)")
axes[0].legend(title="Spacing", fontsize=8, title_fontsize=8, loc="lower left")
fig.suptitle("B8: optimal outgroup placement — depth × spacing", fontsize=11)
fig.tight_layout()
out_pdf.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_pdf}")
