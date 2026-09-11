"""Materialize the whole-chunk fixed-tree MAP orientation once per
``(scenario, chunk, n_out)`` -> a cached JSON ``{pos: map_allele}``.

This factors the expensive ``FixedTreeInference.fit`` (over every site in the
chunk) out of the per-shard ARG-inference path: the fit is deterministic and
identical across every SINGER chain, the Relate shard and the tsinfer chunk, so
a single rule computes it and all ARG modes load the sidecar (see
``ftmap_cache_path`` / ``_fixed_tree_map_by_pos`` in ``_robustness_common``).

The fit is identical across modes because it depends only on the chunk's sites
and the ingroup + first ``n_out`` outgroups --- not on the ARG tool. Sites
beyond the chunk's available outgroups yield an empty map (orientation unused).

Wildcards: ``{scenario}``, ``{chunk_idx}``, ``{n_out}``.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc


try:
    scenario = str(snakemake.wildcards.scenario)  # type: ignore[name-defined]
    chunk_idx = int(snakemake.wildcards.chunk_idx)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    out_path = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, chunk_idx, n_out = "strong_ils", 0, 1
    out_path = None

sc = rc._chunk_scenario(scenario, chunk_idx)
# The SAME window the consumers pass. They call _fixed_tree_map_by_pos with the
# method's budget window, and the window is part of the fit (it sets both the
# site set and n_target_sites), so materialising the whole-chunk fit here would
# populate a cache nothing reads while the consumers each recomputed their own.
window = rc.method_windows(scenario, "tsinfer_arg", chunk_idx)[0]
if out_path is None:
    out_path = str(rc.ftmap_cache_path(scenario, chunk_idx, n_out, window))

# n_out beyond the chunk's outgroups -> orientation is never used (shard is
# empty). Write an empty map so the rule output exists and the DAG is satisfied.
if n_out > len(sc.all_outgroup_names):
    m = {}
else:
    m = sc._fixed_tree_map_by_pos(n_out, window=window)

Path(out_path).write_text(json.dumps({str(p): a for p, a in m.items()}))
print(f"Wrote {len(m)} fixed-tree MAP calls to {out_path}", flush=True)
