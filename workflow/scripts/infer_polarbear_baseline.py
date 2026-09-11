"""B2 PolarBEAR-side inference: runs in the ``envs/polarbear.yml`` conda env.

PolarBEAR is distributed as a script repo (not a pip-installable package) with a
strict ``tskit==0.6 + numba`` dependency stack that conflicts with the rest
of the pipeline, so it gets its own env (set via the rule's conda
directive). Driven by the snakemake-object header below, with a
positional-argv / defaults fallback for standalone use.

Vendored PolarBEAR sources are imported from ``params.polarbear_src``
(defaults to ``external/polarbear/code/polarize``). The script stages the
inputs into a temp dir (PolarBEAR expects matching .trees / .vcf
side-by-side), runs ``site_polarize_ts_ML``, parses the resulting
``prob_ancstate.txt``, and emits per-site posteriors as JSON.

Run directly (defaults to the B2 simulation paths)::

    python workflow/scripts/infer_polarbear_baseline.py

Or via snakemake::

    snakemake -j 1 results/data/polarbear_baseline.json
"""
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    polarbear_src = snakemake.params.polarbear_src  # type: ignore[name-defined]
except NameError:
    if len(sys.argv) > 1:
        in_trees, in_vcf, out_json, mu_s, polarbear_src = sys.argv[1:6]
        mu = float(mu_s)
    else:
        in_trees = "results/data/polarbear_sim.trees"
        in_vcf = "results/data/polarbear_sim.vcf"
        out_json = "results/data/polarbear_baseline.json"
        mu = 1e-8
        polarbear_src = "external/polarbear/code/polarize"


sys.path.insert(0, polarbear_src)
import tskit
import site_polarize  # type: ignore[import-not-found]


# Stage inputs into a single temp dir (PolarBEAR's file_prepare.get_files
# expects .trees and .vcf side by side under one directory).
with tempfile.TemporaryDirectory(prefix="polarbear_baseline_") as tmp:
    tmp_path = Path(tmp)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "polarbear_out"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(in_trees, input_dir / "sim.trees")
    shutil.copy(in_vcf, input_dir / "sim.vcf")

    result_file = output_dir / "prob_ancstate.txt"
    # Wall-clock for the PolarBEAR inference call only (post staging, pre
    # output parse). Reported in inference_meta for the manuscript runtime
    # column.
    _t_inference_start = time.perf_counter()
    site_polarize.site_polarize_ts_ML(
        tskit.load(str(input_dir / "sim.trees")),
        str(input_dir / "sim.vcf"),
        mu,
        str(result_file),
    )
    inference_seconds = float(time.perf_counter() - _t_inference_start)
    print(f"PolarBEAR wrote {result_file}", flush=True)

    # Per-site VCF allele list: PolarBEAR indexes states by genotype-call
    # index (0=REF, 1=ALT1, ...), not by ACGT canonical index, so we need
    # the per-site allele tuple to translate idx → allele char.
    ts = tskit.load(in_trees)
    site_alleles: dict[int, tuple[str, ...]] = {
        int(v.site.position): tuple(v.alleles) for v in ts.variants()
    }

    # Parse PolarBEAR's prob_ancstate.txt. Format (patched, see
    # external/polarbear/code/polarize/site_polarize.py):
    #   pos \t max_prob \t inferred_state_idx \t f1 \t f2 \t f3 \t p0 \t p1 \t p2 \t p3
    out: dict[int, dict] = {}
    with open(result_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            pos = int(parts[0])
            max_prob = float(parts[1])
            idx = int(parts[2])
            alleles = site_alleles.get(pos)
            map_allele = (
                alleles[idx]
                if (alleles is not None and 0 <= idx < len(alleles))
                else None
            )

            # Filter columns (3): non_informative, mutation_count, multibranch.
            non_informative = mutation_count = multibranch = None
            if len(parts) >= 6:
                try:
                    non_informative = int(parts[3]) if parts[3] != "NaN" else None
                    mutation_count = int(parts[4]) if parts[4] != "NaN" else None
                    multibranch = int(parts[5]) if parts[5] != "NaN" else None
                except ValueError:
                    pass

            # Per-allele posterior (patched output: 4 columns after filter cols).
            posterior_by_allele: dict[str, float] = {}
            if len(parts) >= 10 and alleles is not None:
                raw_post = [float(x) for x in parts[6:10]]
                for k, p in enumerate(raw_post):
                    if k < len(alleles):
                        posterior_by_allele[alleles[k]] = p

            out[pos] = {
                "max_prob": max_prob,
                "map_idx": idx,
                "map_allele": map_allele,
                "posterior_by_allele": posterior_by_allele,
                "polarbear_non_informative": non_informative,
                "polarbear_mutation_count": mutation_count,
                "polarbear_multibranch": multibranch,
            }

inference_meta = {
    "mu": mu,
    "n_sites": len(out),
    "inference_seconds": inference_seconds,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": out, "inference_meta": inference_meta}, f)
print(f"Wrote {out_json}: {len(out)} sites", flush=True)
