"""Score one persisted tsinfer ARG chunk into a per-chunk summary --- the
scoring half of the tsinfer_arg path, split from inference
(infer_tsinfer_arg_chunk.py).

Loads the chunk's dated tsinfer ARG, lays the panel's full site set onto its
local trees (the Felsenstein kernel, at the fitted rate from meta.json), and
writes the per-chunk summary the cell concat consumes. Scoring only (no tsinfer
import needed), so it runs in the ancestree stack (bench.yml) and is arm64-
portable regardless of where the ARG was inferred.

Wildcards: ``{scenario}``, ``{n_out}``, ``{chunk_idx}``.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, n_out, chunk_idx = "baseline", 1, 0
    out_json = str(rc.DATA / f"robustness_chunk_{scenario}_tsinfer_arg"
                   f"_n{n_out}_chunk{chunk_idx}_summary.json")

# The persisted dated ARG (rule input, kept for DAG ordering) is re-derived from
# the wildcards inside tsinfer_chunk_summary --- no need to read input[0] here.
summary = rc.tsinfer_chunk_summary(scenario, n_out, chunk_idx)
n_sites = sum(g[0] for g in summary["stats"].values())
with open(out_json, "w") as f:
    json.dump(summary, f)

print(f"Wrote {out_json}: {n_sites} sites, "
      f"runtime={summary['runtime']:.2f}s", flush=True)
