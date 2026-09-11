"""One intersection-figure cell: an ARG-inference method scored on the
cross-method common covered site set (SINGER's budget prefix per chunk), so
every method in the ARG-comparison heatmap is graded on one identical site set
rather than on whatever span it happens to cover.

Re-uses the persisted per-chunk ARGs / shard posteriors --- no re-inference (see
:func:`_robustness_common.intersect_chunk_summary`). Writes the same light
per-cell summary schema as the ordinary cells, so the heatmap loads it verbatim.

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    method = str(snakemake.wildcards.method)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    out = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out = "baseline", "relate_arg_panel_ft", 1
    out = str(rc.intersect_cell_summary_path(scenario, method, n_out))

summary = rc.intersect_cell_summary(scenario, method, n_out, from_shards=True)
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(summary, f)

n_sites = sum(g[0] for g in summary["stats"].values())
print(f"Wrote {out}: {n_sites} sites (common covered set), "
      f"runtime={summary['runtime']:.2f}s", flush=True)
