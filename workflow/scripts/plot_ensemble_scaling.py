"""Ensemble-mode runtime over the draw count and the worker count.

Wall-clock seconds for one pass over the panel, growing with the draws and
falling with the workers. Peak memory is not plotted: ``member_chunk`` bounds
the genealogies drawn and scored at once, so it does not track the ensemble
size, and the per-process high-water mark understates a pooled run anyway.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    in_cells = list(snakemake.input)  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
    out_json = str(snakemake.output.summary)  # type: ignore[name-defined]
except NameError:
    import glob
    in_cells = sorted(glob.glob("results/data/ensemble_scaling_m*_w*.json"))
    out_pdf = "results/reports/ensemble_scaling.pdf"
    out_json = "results/reports/ensemble_scaling.json"

cells = [json.load(open(p)) for p in in_cells]
if not cells:
    # Direct execution with the per-cell JSONs absent: the committed summary
    # holds the same records, so the figure still reproduces from the repo.
    cells = json.load(open(out_json))
cells.sort(key=lambda c: (c["n_workers"], c["n_ensemble"]))
workers = sorted({c["n_workers"] for c in cells})
draws = sorted({c["n_ensemble"] for c in cells})
by_key = {(c["n_ensemble"], c["n_workers"]): c for c in cells}


def _grid(field):
    """``(len(draws), len(workers))`` of ``field``, NaN where a cell is absent."""
    m = np.full((len(draws), len(workers)), np.nan)
    for i, d in enumerate(draws):
        for j, w in enumerate(workers):
            c = by_key.get((d, w))
            if c is not None:
                m[i, j] = c[field]
    return m


def _panel(ax, mat, title, cbar_label, fmt):
    im = ax.imshow(mat, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(workers)), [str(w) for w in workers])
    ax.set_yticks(range(len(draws)), [str(d) for d in draws])
    ax.set_xlabel("workers")
    ax.set_title(title, fontsize=10)
    lo, hi = np.nanmin(mat), np.nanmax(mat)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            # White on the dark (high) end, black on the light end.
            rel = (v - lo) / (hi - lo) if hi > lo else 0.0
            ax.text(j, i, format(v, fmt), ha="center", va="center",
                    fontsize=8, color="white" if rel > 0.55 else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=8)


meta = cells[0]
fig, ax = plt.subplots(figsize=(4.3, 3.1))
_panel(ax, _grid("wall_s"), "wall-clock runtime (s)", "seconds", ".1f")
ax.set_ylabel("ensemble draws $M$")
fig.tight_layout()
fig.savefig(out_pdf)
plt.close(fig)

import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
Path(out_json).write_text(json.dumps(cells, indent=2))
print(f"wrote {out_pdf} and {out_json}")
