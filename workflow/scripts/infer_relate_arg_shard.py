"""Infer + persist one Relate ARG shard: (scenario, method, n_out, chunk,
window, seed) -> a directory of ``draw_{k}.trees`` (one draw --- Relate is
deterministic) plus ``meta.json``.

This is the inference-only half of the Relate panel path, split from scoring
(score_arg_shard.py) so the ARG can be persisted once and re-scored without
re-running Relate. Runs in the ancestree stack (bench.yml). Needs RELATE_DIR
pointing at a Relate install with ``bin/Relate``.

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``, ``{chunk_idx}``,
``{window_idx}``, ``{seed}``.
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
    # output[0] is the meta.json marker. The draws sit beside it.
    out_dir = os.path.dirname(
        str(snakemake.output[0]))  # type: ignore[name-defined]
    os.environ.setdefault("RELATE_DIR", str(snakemake.params.relate_dir))  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out, chunk_idx, window_idx, seed = \
        "strong_ils", "relate_arg_panel", 1, 0, 0, 0
    out_dir = str(rc.arg_shard_dir(
        "relate", scenario, method, n_out, chunk_idx, window_idx, seed))

ft = method.endswith("_ft")
sc = rc._chunk_scenario(scenario, chunk_idx)
Path(out_dir).mkdir(parents=True, exist_ok=True)

# n_out beyond the chunk's outgroups -> empty shard (matches the cell guard).
if n_out > len(sc.all_outgroup_names):
    draws, offset, runtime = [], 0, 0.0
else:
    window = rc.method_windows(scenario, method, chunk_idx)[window_idx]
    draws, offset, runtime = sc.relate_infer_shard(
        n_out, ft_polarize=ft, window=window, seed=seed)

n = _singer.dump_args(draws, out_dir)
with open(os.path.join(out_dir, "meta.json"), "w") as f:
    json.dump({
        "scenario": scenario, "method": method, "n_out": n_out,
        "chunk_idx": chunk_idx, "window_idx": window_idx, "seed": seed,
        "runtime": runtime, "offset": offset, "n_draws": n, "mu": sc.mu,
    }, f)

print(f"Wrote {n} ARG draw(s) to {out_dir}, runtime={runtime:.2f}s", flush=True)
