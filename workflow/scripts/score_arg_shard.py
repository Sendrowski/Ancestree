"""Record one persisted ARG shard's inference runtime.

Despite the name this computes no posteriors: the chunk merge scores the ARGs
itself, through ``score_arg_shard_split`` with both focal passes. What the
merge reads from here is ``runtime``, for the heatmap's runtime strip, copied
from the ARG directory's own meta.json. This step gives the DAG a per-shard
node carrying that number.

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``, ``{chunk_idx}``,
``{window_idx}``, ``{seed}``. Param: ``burn_in``, recorded as provenance.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc
import _singer


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    method = str(snakemake.wildcards.method)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    window_idx = int(snakemake.wildcards.window_idx)  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.seed)  # type: ignore[name-defined]
    # input[0] is the meta.json marker. The draws sit beside it.
    arg_dir = os.path.dirname(
        str(snakemake.input[0]))  # type: ignore[name-defined]
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
    burn_in = int(snakemake.params.burn_in)  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out, chunk_idx, window_idx, seed = \
        "strong_ils", "relate_arg_panel", 1, 0, 0, 0
    tool = "singer" if method.startswith("singer") else "relate"
    arg_dir = str(rc.arg_shard_dir(
        tool, scenario, method, n_out, chunk_idx, window_idx, seed))
    out_json = str((rc.singer_shard_path if tool == "singer"
                    else rc.relate_shard_path)(
        scenario, method, n_out, chunk_idx, window_idx, seed))
    burn_in = 0

sc = rc._chunk_scenario(scenario, chunk_idx)
draws = _singer.load_args(arg_dir)
meta_p = os.path.join(arg_dir, "meta.json")
# meta.json carries the real (inference) runtime for the strip, and the rate the
# kernel must score with (tsinfer dates its trees, so it differs from sc.mu).
meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
runtime = meta.get("runtime", 0.0)
mu = meta.get("mu")  # None -> score_arg_shard falls back to sc.mu

# The runtime read above is the only field the merge consumes.
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({
        "scenario": scenario, "method": method, "n_out": n_out,
        "chunk_idx": chunk_idx, "window_idx": window_idx, "seed": seed,
        "runtime": runtime, "n_draws": len(draws),
    }, f)

print(
    f"Wrote {out_json}: {len(draws)} draw(s), runtime={runtime:.2f}s",
    flush=True,
)
