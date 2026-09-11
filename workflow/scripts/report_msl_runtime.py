"""Collect the MSL chr1 coverage and serial runtime of every method.

Every row is the serial cost of annotating the panel: for a sharded run that
is the sum of the shards' own wall times, not the observed elapsed time, so
the tools are compared on one definition. A method whose meta is absent
reports a null for whichever of the two figures the meta carries.
"""
import json
import os

try:
    LOCAL_TREE = snakemake.input.local_tree  # type: ignore[name-defined]
    FIXED_TREE = snakemake.input.fixed_tree  # type: ignore[name-defined]
    ARG = snakemake.input.arg  # type: ignore[name-defined]
    POLARBEAR = snakemake.input.polarbear  # type: ignore[name-defined]
    AGREEMENT = snakemake.input.agreement  # type: ignore[name-defined]
    ESTSFS = snakemake.params.estsfs  # type: ignore[name-defined]
    OUT = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    LOCAL_TREE = f"{DATA}/msl_localtree_chr1_full_meta.json"
    FIXED_TREE = f"{DATA}/msl_vcf_chr1_full_stream_meta.json"
    ARG = f"{DATA}/msl_arg_chr1full_streaming_meta.json"
    POLARBEAR = f"{DATA}/rerun_polarbear_meta.json"
    ESTSFS = f"{DATA}/rerun_estsfs_meta.json"
    AGREEMENT = "results/reports/msl_agreement_matrix.json"
    OUT = "results/reports/msl_runtime.json"

#: Sites the published EST-SFS call set covers, from the archive the
#: comparison reads. It is biallelic-only and needs both outgroups called.
ESTSFS_SITES = 985_038

#: Wall-clock total of the archived EST-SFS re-run, in seconds, known to the
#: tenth of a minute the table reports. Its meta is not an input, so the
#: measured total is carried alongside the site count it belongs to.
ESTSFS_TOTAL_SECONDS = 42.8 * 60


def _load(path):
    """The meta at ``path``, or ``None`` when it was never produced."""
    if not path or not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


local_tree = _load(LOCAL_TREE)
fixed_tree = _load(FIXED_TREE)
arg = _load(ARG)
polarbear = _load(POLARBEAR)
estsfs = _load(ESTSFS)

#: Sites PolarBEAR's kernel emits a call for, as counted by the agreement
#: report over PolarBEAR's own per-site posterior output.
polarbear_sites = _load(AGREEMENT)["n_calls"]["PolarBEAR"]

rows = [
    {"method": "ancestree_fixed_tree",
     "sites": fixed_tree and fixed_tree["n_sites_written"],
     "seconds": fixed_tree and fixed_tree["wall_total_s"]},
    {"method": "ancestree_arg",
     "sites": arg and arg["total_sites_written"],
     "seconds": arg and arg["total_seconds"]},
    {"method": "ancestree_local_tree",
     "sites": local_tree and local_tree["n_written"],
     "seconds": local_tree and local_tree["wall_s_serial_equivalent"]},
    {"method": "polarbear",
     "sites": polarbear_sites,
     "seconds": polarbear and polarbear["timings_seconds"]["total"]},
    {"method": "estsfs",
     "sites": ESTSFS_SITES,
     "seconds": (estsfs["timings_seconds"]["total"] if estsfs
                 else ESTSFS_TOTAL_SECONDS)},
]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as handle:
    json.dump({"rows": rows}, handle, indent=2)
print(f"report_msl_runtime: wrote {OUT}")
