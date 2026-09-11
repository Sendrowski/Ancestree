"""Merge the per-comparison shards into the orientation-damage report.

Every mis-orientation scheme is scored against the true-ancestral-allele arm of
the same ARG tool in its own shard. This step reads those blocks, wraps them in
the simulation parameters and the orientation-map summaries, and renders the
per-bin markdown tables.

Every field named ``*_excess`` is signed so that a positive value means the
mis-orientation degraded the inferred genealogy relative to the
true-ancestral-allele arm.

Run via snakemake::

    snakemake -j 1 results/reports/orientation_damage.md
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    BOOTSTRAP_SEED,
    MIS_ORIENTATION_SCHEMES,
    N_BOOTSTRAP,
    SCHEMES,
    TOOLS,
    load_sites,
)
from _orientation_report import (  # noqa: E402
    OrientationDamageReport,
    _markdown,
)

try:
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    in_maps = list(snakemake.input.maps)  # type: ignore[name-defined]
    in_comparisons = list(snakemake.input.comparisons)  # type: ignore[name-defined]
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    out_md = str(snakemake.output.md)  # type: ignore[name-defined]
    n_boot = int(snakemake.params.n_boot)  # type: ignore[name-defined]
    boot_seed = int(snakemake.params.boot_seed)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    REPORTS = "results/reports"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    in_maps = [f"{DATA}/orientation_map_{s}.json" for s in SCHEMES]
    in_comparisons = [f"{DATA}/orientation_comparison_{t}_{s}.json"
                      for t in TOOLS for s in MIS_ORIENTATION_SCHEMES]
    out_json = f"{REPORTS}/orientation_damage.json"
    out_md = f"{REPORTS}/orientation_damage.md"
    n_boot = N_BOOTSTRAP
    boot_seed = BOOTSTRAP_SEED


maps = {Path(p).stem.replace("orientation_map_", ""): p for p in in_maps}
comparisons: dict[str, dict] = {}
for p in in_comparisons:
    tool, scheme = (Path(p).stem.replace("orientation_comparison_", "")
                    .split("_", 1))
    comparisons.setdefault(tool, {})[scheme] = json.loads(Path(p).read_text())

report = OrientationDamageReport(load_sites(in_sites), maps, {},
                                 n_boot=n_boot, boot_seed=boot_seed)
payload = report.merge(json.loads(Path(in_meta).read_text()), comparisons)
Path(out_json).write_text(json.dumps(payload, indent=1))
Path(out_md).write_text(_markdown(payload))
print(f"wrote {out_json} and {out_md}", flush=True)
