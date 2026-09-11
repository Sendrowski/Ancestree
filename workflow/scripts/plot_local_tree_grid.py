"""Heatmap of mean Brier over the local-tree window x block grid.

Reads the assembled ``local_tree_grid.json`` and renders one heatmap with
the requested window on the y-axis, the requested block on the x-axis and
the mean Brier score as the cell value (lower is better). Each tested cell
is annotated with its value; the untested upper triangle (block > window) is
left blank. The true-ARG ceiling is drawn as a reference line on the
colourbar and named in the title.

No inference is re-run here.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402
import numpy as np  # noqa: E402

try:
    in_json = str(snakemake.input.grid)  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
except NameError:
    in_json = "results/reports/local_tree_grid.json"
    out_pdf = "results/reports/bench_local_tree_grid.pdf"

report = json.loads(Path(in_json).read_text())
windows = report["windows"]
blocks = report["blocks"]
brier = np.array(
    [[np.nan if v is None else float(v) for v in row]
     for row in report["grid"]["mean_brier"]],
    dtype=float,
)
win_eff = report["grid"]["window_snp_effective"]
blk_eff = report["grid"]["block_snp_effective"]
ceiling = float(report["true_arg_ceiling"]["mean_brier"])

masked = np.ma.masked_invalid(brier)
cmap = plt.get_cmap("viridis_r").copy()
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(5.58, 5.76))
im = ax.imshow(masked, cmap=cmap, aspect="equal", origin="upper")

ax.set_xticks(range(len(blocks)))
ax.set_xticklabels([b.replace("snp", "") for b in blocks])
ax.set_yticks(range(len(windows)))
ax.set_yticklabels([w.replace("snp", "") for w in windows])
ax.set_xlabel("HMM emission block size (requested SNPs)")
ax.set_ylabel("Local-tree window (requested SNPs)")

# Cell annotation: the Brier value, and beneath it the widths the run
# actually used once the block spec was resolved and the window snapped up.
lo, hi = float(masked.min()), float(masked.max())
for i in range(len(windows)):
    for j in range(len(blocks)):
        if masked.mask[i, j]:
            continue
        v = float(brier[i, j])
        shade = (v - lo) / (hi - lo) if hi > lo else 0.0
        ax.text(j, i - 0.09, f"{v:.4f}", ha="center", va="center", fontsize=7,
                color="white" if shade > 0.55 else "black")
        ax.text(j, i + 0.14, f"{win_eff[i][j]:.1f}/{blk_eff[i][j]:.1f}",
                ha="center", va="center", fontsize=5.5,
                color="white" if shade > 0.55 else "0.3")

for edge in range(1, len(blocks)):
    ax.axvline(edge - 0.5, color="0.85", lw=0.5)
for edge in range(1, len(windows)):
    ax.axhline(edge - 0.5, color="0.85", lw=0.5)

# Tie the bar to the image axes so it takes the heatmap's own height.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.12)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("mean Brier score (lower is better)")
if lo <= ceiling <= hi:
    cbar.ax.axhline(ceiling, color="crimson", lw=1.4)
    cbar.ax.annotate("true-ARG ceiling", xy=(0.5, ceiling),
                     xycoords=("axes fraction", "data"),
                     xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=6, color="crimson")

fig.tight_layout()
Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {out_pdf}", flush=True)
