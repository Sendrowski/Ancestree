"""Aggregate the local-tree window x block grid cells into a report.

Reads the per-cell JSONs written by :mod:`infer_local_tree_grid_cell` and
the true-ARG ceiling cell of the window benchmark (same simulation), and
writes:

- ``local_tree_grid.json`` — the assembled grid. ``cells`` holds each cell's
  full payload under the key ``"<window>__<block>"``; ``grid`` holds
  row-major matrices over ``windows`` x ``blocks`` with ``null`` in the
  untested upper triangle (block > window).
- ``local_tree_grid.md`` — the same matrices as readable tables.

Only cells with block <= window are enumerated, so the grid is lower
triangular. Because an explicit block width is not capped at the window and
the window is rounded up to a whole number of blocks, the effective widths
recorded per cell can exceed the requested ones; the effective matrices
report what was run.

The heatmap over this JSON is rendered by :mod:`plot_local_tree_grid`.

Run via snakemake::

    snakemake -j 1 results/reports/local_tree_grid.md
"""
import json
from pathlib import Path

try:
    in_cells = list(snakemake.input.cells)  # type: ignore[name-defined]
    in_ceiling = str(snakemake.input.ceiling)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    out_md = str(snakemake.output.md)  # type: ignore[name-defined]
    windows = list(snakemake.params.windows)  # type: ignore[name-defined]
    blocks = list(snakemake.params.blocks)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    REPORTS = "results/reports"
    windows = ["1snp", "2snp", "4snp", "8snp", "16snp", "32snp", "64snp"]
    blocks = ["1snp", "2snp", "4snp", "8snp", "16snp", "32snp"]
    in_cells = [
        f"{DATA}/local_tree_grid_w{w}_b{b}.json"
        for w in windows for b in blocks
        if int(b[:-3]) <= int(w[:-3])
    ]
    in_ceiling = f"{DATA}/local_tree_window_true_arg.json"
    in_meta = f"{DATA}/local_tree_window_sim_meta.json"
    out_json = f"{REPORTS}/local_tree_grid.json"
    out_md = f"{REPORTS}/local_tree_grid.md"


def _snp(spec: str) -> int:
    """SNP count of a ``"<N>snp"`` spec.

    :param spec: Spec string such as ``"16snp"``.
    :return: The integer N.
    """
    return int(str(spec).strip().lower()[:-3])


cells: dict[str, dict] = {}
for path in in_cells:
    payload = json.loads(Path(path).read_text())
    cells[f"{payload['window']}__{payload['block']}"] = payload

meta = json.loads(Path(in_meta).read_text())
ceiling = json.loads(Path(in_ceiling).read_text())

#: Matrices assembled here so the plotting rule reads a single JSON.
METRICS = [
    "mean_brier", "mean_map", "mean_ptrue", "n_sites", "runtime",
    "window_bp", "block_size", "window_snp_effective", "block_snp_effective",
    "blocks_per_window",
]
grid: dict[str, list[list[float | None]]] = {}
for metric in METRICS:
    rows: list[list[float | None]] = []
    for w in windows:
        row: list[float | None] = []
        for b in blocks:
            cell = cells.get(f"{w}__{b}")
            row.append(None if cell is None else cell[metric])
        rows.append(row)
    grid[metric] = rows

tested = [(w, b) for w in windows for b in blocks if _snp(b) <= _snp(w)]
missing = [f"{w}__{b}" for w, b in tested if f"{w}__{b}" not in cells]
best_key = min(cells, key=lambda k: cells[k]["mean_brier"]) if cells else None

report = {
    "windows": windows,
    "blocks": blocks,
    "window_snps": [_snp(w) for w in windows],
    "block_snps": [_snp(b) for b in blocks],
    "n_cells": len(tested),
    "missing_cells": missing,
    "cells": cells,
    "grid": grid,
    "best_cell": best_key,
    "best_mean_brier": None if best_key is None else cells[best_key]["mean_brier"],
    "true_arg_ceiling": {
        "mean_brier": ceiling["mean_brier"],
        "mean_map": ceiling["mean_map"],
        "mean_ptrue": ceiling["mean_ptrue"],
        "n_sites": ceiling["n_sites"],
    },
    "simulation": {
        "mu": meta["mu"],
        "rec_rate": meta["rec_rate"],
        "n_ingroup": len(meta["ingroup_names"]),
    },
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)


def _table(metric: str, fmt: str) -> list[str]:
    """Render one metric matrix as a markdown table.

    :param metric: Key into the assembled ``grid``.
    :param fmt: Per-value format spec, e.g. ``".4f"``.
    :return: Markdown lines, header row first.
    """
    lines = ["| requested window | " + " | ".join(f"block {b}" for b in blocks) + " |",
             "|---" * (len(blocks) + 1) + "|"]
    for w, row in zip(windows, grid[metric]):
        vals = ["" if v is None else format(v, fmt) for v in row]
        lines.append(f"| {w} | " + " | ".join(vals) + " |")
    return lines


md = [
    "# Local-tree window x block grid",
    "",
    f"One ingroup-only msprime ARG (n_ingroup="
    f"{report['simulation']['n_ingroup']}, mu={report['simulation']['mu']:g}, "
    f"rec_rate={report['simulation']['rec_rate']:g}), true rates given to the "
    "inference, perfectly phased genotypes.",
    "",
    f"Cells scored: {len(cells)} of {len(tested)} (block <= window only).",
    f"True-ARG ceiling mean Brier: "
    f"{report['true_arg_ceiling']['mean_brier']:.4f}.",
    "",
    "Requested widths label the rows and columns. An explicit block is not "
    "capped at the window and the window is rounded up to a whole number of "
    "blocks, so the effective-width tables below record what was run.",
    "",
    "## Mean Brier score (lower is better)",
    "",
]
md += _table("mean_brier", ".4f")
md += ["", "## Mean MAP accuracy (higher is better)", ""]
md += _table("mean_map", ".4f")
md += ["", "## Mean posterior probability of the true allele", ""]
md += _table("mean_ptrue", ".4f")
md += ["", "## Effective window (SNP equivalents at the panel's mean density)", ""]
md += _table("window_snp_effective", ".2f")
md += ["", "## Effective block (SNP equivalents at the panel's mean density)", ""]
md += _table("block_snp_effective", ".2f")
md += ["", "## Effective window width (bp)", ""]
md += _table("window_bp", "d")
md += ["", "## Effective block width (bp)", ""]
md += _table("block_size", "d")
md += ["", "## Blocks per window", ""]
md += _table("blocks_per_window", ".2f")
md += ["", "## Wall-clock runtime (s)", ""]
md += _table("runtime", ".1f")
if best_key is not None:
    w_best, b_best = best_key.split("__")
    md += ["", f"Lowest mean Brier: window={w_best}, block={b_best} "
               f"({report['best_mean_brier']:.4f}).", ""]
with open(out_md, "w") as f:
    f.write("\n".join(md) + "\n")

print(f"Wrote {out_json} and {out_md}: {len(cells)} cells", flush=True)
