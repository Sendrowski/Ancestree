"""Score one ``(ARG tool, mis-orientation scheme)`` arm pair.

The mis-orientation arm is paired per site against the true-ancestral-allele
arm of the same tool, built from identical genotypes with an identical seed.
Both binnings are computed here, so the report step performs no numerics.

Run via snakemake::

    snakemake -j 1 results/data/orientation_comparison_tsinfer_random5.json
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    BOOTSTRAP_SEED,
    BOOTSTRAP_THREADS,
    N_BOOTSTRAP,
    load_sites,
)
from _orientation_report import OrientationDamageReport  # noqa: E402

try:
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    in_orientation = str(snakemake.input.orientation)  # type: ignore[name-defined]
    in_persite_true_aa = str(snakemake.input.persite_true_aa)  # type: ignore[name-defined]
    in_persite_scheme = str(snakemake.input.persite_scheme)  # type: ignore[name-defined]
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    tool = str(snakemake.wildcards.tool)  # type: ignore[name-defined]
    scheme = str(snakemake.wildcards.scheme)  # type: ignore[name-defined]
    n_boot = int(snakemake.params.n_boot)  # type: ignore[name-defined]
    boot_seed = int(snakemake.params.boot_seed)  # type: ignore[name-defined]
    n_threads = int(snakemake.threads)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    tool = "tsinfer"
    scheme = "freq_biased5"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    in_orientation = f"{DATA}/orientation_map_{scheme}.json"
    in_persite_true_aa = f"{DATA}/orientation_persite_{tool}_true_aa.npz"
    in_persite_scheme = f"{DATA}/orientation_persite_{tool}_{scheme}.npz"
    out_json = f"{DATA}/orientation_comparison_{tool}_{scheme}.json"
    n_boot = N_BOOTSTRAP
    boot_seed = BOOTSTRAP_SEED
    n_threads = BOOTSTRAP_THREADS


report = OrientationDamageReport(
    load_sites(in_sites),
    {scheme: in_orientation},
    {(tool, "true_aa"): in_persite_true_aa, (tool, scheme): in_persite_scheme},
    n_boot=n_boot, boot_seed=boot_seed,
    n_threads=max(1, min(n_threads, BOOTSTRAP_THREADS)))
block = report.comparison(tool, scheme, json.loads(Path(in_meta).read_text()))
Path(out_json).write_text(json.dumps(block, indent=1))
print(f"wrote {out_json}: {block['n_tmrca_points_total']:,} TMRCA points "
      f"over {block['n_sites_scored']:,} scored sites", flush=True)
