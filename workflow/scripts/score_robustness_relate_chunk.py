"""Merge Relate shards for one chunk into a per-chunk summary.

Reads all (window x seed) shards for (scenario, method, n_out, chunk),
concatenates the disjoint genome sub-windows, scores against the chunk truth,
and reduces to the SAME light summary format as every other per-chunk summary,
so the existing cell concat (score_robustness_cell.py) picks it up unchanged.

Runs under ``workflow/envs/bench.yml`` (scoring only, no Relate binary needed).

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``, ``{chunk_idx}``.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    method = str(snakemake.wildcards.method)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out, chunk_idx = "strong_ils", "relate_arg_panel", 1, 0
    out_json = str(rc.chunk_summary_path(scenario, method, n_out, chunk_idx))

summary = rc.relate_merge_chunk_summary(scenario, method, n_out, chunk_idx)

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"chunk_idx": chunk_idx, **summary}, f)

n_sites = sum(g[0] for g in summary["stats"].values())
print(
    f"Wrote {out_json}: {n_sites} sites merged from "
    f"{rc.singer_n_windows(scenario, chunk_idx)} windows x "
    f"{rc.RELATE_N_CHAINS} seed(s), runtime={summary['runtime']:.2f}s",
    flush=True,
)
