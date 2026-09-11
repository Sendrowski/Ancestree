"""Compute one robustness-heatmap cell (scenario × method × n_out).

Writes a light per-cell *summary* (sufficient statistics — a few KB, see
``_robustness_common.summarize_persite``) that the report stage aggregates,
so peak memory stays tiny regardless of site count. For the chunked msprime
scenarios this is the element-wise sum of the 10 per-chunk summaries (no
per-site list ever materialised). For SLiM the (sparse) per-site list is built
then reduced.

Set ``RC_DUMP_PERSITE=1`` to also dump the full per-site list (a
``*_persite.json`` debug artifact, off by default, only for the SLiM and
live-compute paths).

Wildcards: ``{scenario}``, ``{method}``, ``{n_out}``.
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
    out_json = str(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    scenario, method, n_out = "strong_ils", "anc_arg", 3
    out_json = str(rc.cell_summary_path(scenario, method, n_out))

summary = rc.compute_cell_summary(scenario, method, n_out)

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(summary, f)

# Optional heavy per-site dump (debug only). Not available for the chunked
# msprime path, which never materialises per-site records.
if rc.dump_persite_enabled() and scenario not in rc.CHUNKED_MSP_SCENARIOS:
    if scenario == "slim":
        recs, _ = rc.slim_aggregate_persite(method, n_out)
    else:
        sc = rc._get_scenario(scenario)
        recs, _ = getattr(sc, rc._METHOD_DISPATCH[method])(n_out)
    with open(out_json.replace("_summary.json", "_persite.json"), "w") as f:
        json.dump({"scenario": scenario, "method": method, "n_out": n_out,
                   "n_ingroup": summary["n_ingroup"], "runtime": summary["runtime"],
                   "per_site": [list(r) for r in recs]}, f)

n_sites = sum(g[0] for g in summary["stats"].values())
print(
    f"Wrote {out_json}: {n_sites} sites reduced, "
    f"runtime={summary['runtime']:.2f}s, n_ingroup={summary['n_ingroup']}",
    flush=True,
)
