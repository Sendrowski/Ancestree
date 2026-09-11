"""Compute one per-chunk robustness result (scenario × method × n_out × chunk).

The 4 msprime robustness scenarios are simulated as independent chunks, so
each (scenario, method, n_out) cell is the element-wise sum of this script's
per-chunk *summaries* — one small, parallel SLURM job per chunk instead of one
monolithic multi-million-site job. This script reduces the chunk's per-site
records to a light summary (sufficient statistics, a few KB) and writes ONLY
that, so no per-site list ever hits disk. The cell job (score_robustness_cell)
sums the 10 chunk summaries.

Set ``RC_DUMP_PERSITE=1`` to also dump the full per-site list alongside the
summary (a ``*_persite.json`` debug artifact, off by default).

Used by two rules: the general ``score_robustness_chunk`` (method via the
wildcard) and ``score_robustness_tsinfer_chunk`` (output fixes the method, so
fall back to ``tsinfer_arg`` when no method wildcard is present).

Wildcards: ``{scenario}``, [``{method}``], ``{n_out}``, ``{chunk_idx}``.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    method = str(getattr(snakemake.wildcards, "method", "tsinfer_arg"))  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out, chunk_idx = "strong_ils", "anc_arg", 3, 0
    out_json = str(rc.chunk_summary_path(scenario, method, n_out, chunk_idx))

records, runtime = rc.compute_scenario_chunk(scenario, method, n_out, chunk_idx)
summary = rc.summarize_persite(
    records, scenario=scenario, method=method, n_out=n_out,
    n_ingroup=rc._cell_n_ingroup(scenario), runtime=runtime,
)

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"chunk_idx": chunk_idx, **summary}, f)

if rc.dump_persite_enabled():
    persite_path = out_json.replace("_summary.json", "_persite.json")
    with open(persite_path, "w") as f:
        json.dump({"scenario": scenario, "method": method, "n_out": n_out,
                   "chunk_idx": chunk_idx, "runtime": runtime,
                   "per_site": [list(r) for r in records]}, f)

print(
    f"Wrote {out_json}: {len(records)} site records reduced, "
    f"runtime={runtime:.2f}s",
    flush=True,
)
