"""Report entry: render one robustness heatmap + runtime side-table.

One renderer for every heatmap figure in the manuscript. The figures differ
only in which method rows they show --- scenarios, n_out sweep, site classes,
Brier colour scale and runtime strip are shared --- so the row set is a
parameter (``{figure}``, resolved through :data:`_robustness_common.
HEATMAP_FIGURES`) rather than a script of its own:

- ``robustness_heatmap``   — the main comparison: the three inference modes
  plus SINGER, tsinfer + tsdate and Relate.
- ``robustness_baselines`` — appendix replica adding the adaptive fixed-tree
  prior and the parsimony / majority-outgroup baselines.

Every figure reads the intersection-scored cells (``robustness_cell_intersect_*``,
see rule ``robustness_intersect_cell``), so all rows of all figures are graded
on one identical site set --- SINGER's budget prefix per chunk.

Two modes (the logic lives in :mod:`_robustness_common`):

- **snakemake** (``report_robustness_heatmap`` rule) — loads the per-cell result
  JSONs listed by the rule and renders.
- **standalone** (``python workflow/scripts/make_robustness_heatmap.py
  <figure>``) — loads the same per-cell summary cache from ``results/data``.
  It never recomputes cells. A missing cell is an error pointing back to the
  snakemake per-cell rules (the parallel compute path).

Outputs per figure: ``<stem>.pdf`` (mean P(true)), ``<stem>_map.pdf`` (MAP
accuracy), a ~200-dpi PNG of the Brier figure, and ``<stem>_runtimes.{json,md}``.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc


try:
    figure = str(snakemake.wildcards.figure)  # type: ignore[name-defined]
    out_pdf = Path(snakemake.output.heatmap)  # type: ignore[name-defined]
    out_pdf_map = Path(snakemake.output.heatmap_map)  # type: ignore[name-defined]
    out_rt_json = Path(snakemake.output.runtimes_json)  # type: ignore[name-defined]
    out_rt_md = Path(snakemake.output.runtimes_md)  # type: ignore[name-defined]
    fig_copy = Path(snakemake.output.fig_copy)  # type: ignore[name-defined]
    cell_paths = list(snakemake.input.cells)  # type: ignore[name-defined]
except NameError:
    figure = sys.argv[1] if len(sys.argv) > 1 else "robustness_heatmap"
    out_pdf = rc.REPORTS / f"{figure}.pdf"
    out_pdf_map = rc.REPORTS / f"{figure}_map.pdf"
    out_rt_json = rc.REPORTS / f"{figure}_runtimes.json"
    out_rt_md = rc.REPORTS / f"{figure}_runtimes.md"
    fig_copy = (rc.REPO / "reports" / "manuscripts" / "latex" / "figures"
                / f"bench_{figure}.pdf")
    # Load the per-cell summary cache (the same summaries the snakemake path
    # loads). Never recompute in-process — assert_full_coverage() below raises
    # with the missing cells if the cache is incomplete.
    cell_paths = [str(p)
                  for p in rc.DATA.glob("robustness_cell_intersect_*_summary.json")]

methods = rc.HEATMAP_FIGURES[figure]
rc.set_methods(methods)
rc.load_cell_summaries(cell_paths)
rc.assert_full_coverage(methods)
rc.build_report(out_pdf, out_pdf_map, out_rt_json, out_rt_md)

# Copy the PDF into the manuscript figures dir (a tracked output, so the
# cells -> figure -> manuscript edge stays in the snakemake DAG).
fig_copy.parent.mkdir(parents=True, exist_ok=True)
fig_copy.write_bytes(Path(out_pdf).read_bytes())
print(f"Copied {out_pdf} -> {fig_copy}")
