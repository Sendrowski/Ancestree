"""Infer + persist one tsinfer ARG chunk: (scenario, n_out, chunk) -> a
directory with the dated tsinfer ARG (``draw_0.tsz``) plus ``meta.json``.

This is the inference-only half of the tsinfer_arg path, split from scoring
(score_tsinfer_chunk.py). tsinfer runs on the same budget window as the other
panel-ARG methods, so the tools are compared on one span. Its trees are dated
(tsinfer->tsdate), and the fitted rate is written to meta.json so scoring
uses it. Runs in the dedicated tsinfer env (tsinfer.yml).

Wildcards: ``{scenario}``, ``{n_out}``, ``{chunk_idx}``.
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
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    # output[0] is the meta.json marker. The draws sit beside it.
    out_dir = os.path.dirname(
        str(snakemake.output[0]))  # type: ignore[name-defined]
except NameError:
    scenario, n_out, chunk_idx = "baseline", 1, 0
    out_dir = str(rc.tsinfer_arg_dir(scenario, n_out, chunk_idx))

sc = rc._chunk_scenario(scenario, chunk_idx)
window = rc.method_windows(scenario, "tsinfer_arg", chunk_idx)[0]
Path(out_dir).mkdir(parents=True, exist_ok=True)
draws, offset, runtime, mu = sc.tsinfer_infer_chunk(n_out, window=window)
n = _singer.dump_args(draws, out_dir)
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump({
        "scenario": scenario, "method": "tsinfer_arg", "n_out": n_out,
        "chunk_idx": chunk_idx, "runtime": runtime, "offset": offset,
        "n_draws": n, "mu": mu,
    }, f)

print(f"Wrote {n} tsinfer ARG(s) to {out_dir}, mu={mu:.3e}, "
      f"runtime={runtime:.2f}s", flush=True)
