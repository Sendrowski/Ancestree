"""One tsinfer-ARG (fixed-tree-MAP AA) robustness-heatmap cell.

Separate from :mod:`score_robustness_cell` because tsinfer lives in its own
dedicated ``envs/tsinfer.yml`` conda env (base has no tsinfer), set via the
rule's conda directive.

Builds the tsinfer ARG from the scenario's genotypes, orienting each site by
the fixed-tree MAP allele, polarises it with the Felsenstein kernel, and writes
the same light per-cell *summary* schema as :mod:`score_robustness_cell` (for
chunked msprime scenarios this just sums the per-chunk tsinfer summaries).

"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402


try:
    scenario = snakemake.wildcards.scenario  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    out = snakemake.output[0]  # type: ignore[name-defined]
except NameError:
    scenario = "strong_ils"
    n_out = 3
    out = "results/data/robustness_cell_strong_ils_tsinfer_arg_n3_summary.json"

summary = rc.compute_cell_summary(scenario, "tsinfer_arg", n_out)
Path(out).parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(summary, f)
n_sites = sum(g[0] for g in summary["stats"].values())
print(
    f"Wrote {out}: {n_sites} sites reduced, "
    f"runtime={summary['runtime']:.2f}s, n_ingroup={summary['n_ingroup']}",
    flush=True,
)
