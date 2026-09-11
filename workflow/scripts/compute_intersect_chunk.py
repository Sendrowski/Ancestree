"""One chunk of an intersection-figure cell, scored on the cross-method common
covered site set (SINGER's budget prefix for that chunk).

The shard of :mod:`compute_intersect_cell`, which sums these. Chunks are
independent, so they fan out and the cell is an element-wise sum.

Re-uses the persisted per-chunk ARGs / shard posteriors: no re-inference (see
:func:`_robustness_common.intersect_chunk_summary`).

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``, ``{chunk_idx}``.
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
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    out = str(snakemake.output[0])  # type: ignore[name-defined]
    burn_in = int(snakemake.params.burn_in)  # type: ignore[name-defined]
    # The module constant is what the scoring path reads. A rule param that
    # only sat in params would be declared and ignored.
    rc.LOCAL_TREE_ENSEMBLE_MEMBERS = int(
        snakemake.params.ensemble_members)  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out, chunk_idx = "baseline", "relate_arg_panel_ft", 1, 0
    out = str(rc.intersect_chunk_summary_path(scenario, method, n_out, chunk_idx))
    burn_in = None

summary = rc.intersect_chunk_summary(scenario, method, n_out, chunk_idx,
                                     burn_in=burn_in)
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    json.dump(summary, f)

n_sites = sum(g[0] for g in summary["stats"].values())
print(f"Wrote {out}: {n_sites} sites (common covered set), "
      f"runtime={summary.get('runtime', 0.0):.2f}s", flush=True)
