"""B13 reproducibility: re-run PolarBEAR's polarisation step on the MSL
chr1 archived tskit ARG and the cleaned VCF, recording wall-clock time.

Reuses the archived 18 GB gamma-SMC tskit ARG (skipping the multi-day
gamma-SMC re-inference), the comparison being of the orienting kernel, not of
the SMC step. Wraps PolarBEAR's two scripts:

1. ``code/polarize/PolarBear.py`` — runs the Felsenstein kernel on the
   ARG, produces ``prob_ancstate.txt`` (per-site posteriors + flags).
2. ``code/polarize/PolarBear_ancestral_state_output.py`` — applies the
   parsimony / informativeness filter and emits the final
   ``anc_state.txt`` (pos\\tallele_idx, the file compared against).

The two stages are timed separately, giving kernel and post-processing cost
apart.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    polarbear_root = snakemake.params.polarbear_root  # type: ignore[name-defined]
    theta = float(snakemake.params.theta)  # type: ignore[name-defined]
    out_anc = snakemake.output.anc  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    work_dir = snakemake.params.work_dir  # type: ignore[name-defined]
except NameError:
    polarbear_root = "external/polarbear"
    in_trees = (
        f"{polarbear_root}/data/real_data/ancestral_state/PolarBEAR_gammaSMC/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned_tskit.trees"
    )
    in_vcf = (
        f"{polarbear_root}/data/real_data/ancestral_state/PolarBEAR_gammaSMC/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.vcf.gz"
    )
    theta = 4 * 1e4 * 1.25e-8
    out_anc = "results/data/rerun_polarbear_anc_state.txt"
    out_meta = "results/data/rerun_polarbear_meta.json"
    work_dir = "results/data/rerun_polarbear_workdir"


polarbear_root = Path(polarbear_root).resolve()
work_dir = Path(work_dir).resolve()
work_dir.mkdir(parents=True, exist_ok=True)


# PolarBear.py expects an INPUT_PATH directory holding the .trees file
# (and a VCF for sample-name resolution). We stage symlinks so we don't
# touch the archived directory.
trees_link = work_dir / Path(in_trees).name
vcf_link = work_dir / Path(in_vcf).name
for src, link in [(Path(in_trees).resolve(), trees_link),
                  (Path(in_vcf).resolve(), vcf_link)]:
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src)


# Stage 1: PolarBear.py — the Felsenstein kernel run, produces prob_ancstate.txt
polarbear_py = polarbear_root / "code" / "polarize" / "PolarBear.py"
print(f"rerun_polarbear_msl: stage 1 — {polarbear_py}", flush=True)
t0 = time.perf_counter()
cmd_kernel = [
    sys.executable, str(polarbear_py),
    "-I", str(work_dir),
    "-O", str(work_dir),
    "--m", str(theta),
]
print(f"  $ {' '.join(cmd_kernel)}", flush=True)
result = subprocess.run(cmd_kernel, cwd=str(polarbear_root), check=True)
t_kernel = time.perf_counter() - t0
print(f"  stage 1 wall: {t_kernel:.1f}s", flush=True)


# Stage 2: PolarBear_ancestral_state_output.py — apply filter + emit final calls.
# `PolarBear.py` writes a 10-column prob_ancstate.txt (pos, confidence, best,
# info, n_mut, poly + per-state posteriors A C G T), but
# `PolarBear_ancestral_state_output.py` does `columns[2:]` with dtype=int and
# fails on the float posterior columns. Truncate to the 6 columns the
# post-processor expects.
prob_file = work_dir / "prob_ancstate.txt"
if not prob_file.exists():
    raise RuntimeError(f"expected {prob_file} after stage 1 but not found")
prob_full = work_dir / "prob_ancstate_full.txt"
shutil.move(prob_file, prob_full)
with open(prob_full) as fin, open(prob_file, "w") as fout:
    for line in fin:
        parts = line.rstrip("\n").split("\t")
        fout.write("\t".join(parts[:6]) + "\n")
filter_py = polarbear_root / "code" / "polarize" / "PolarBear_ancestral_state_output.py"
print(f"\nrerun_polarbear_msl: stage 2 — {filter_py}", flush=True)
t1 = time.perf_counter()
out_anc_abs = Path(out_anc).resolve()
out_anc_abs.parent.mkdir(parents=True, exist_ok=True)
cmd_filter = [
    sys.executable, str(filter_py),
    "--vcf", str(vcf_link),
    "--probs", str(prob_file),
    "--filter_condition", "3",
    "--output", str(out_anc_abs),
]
print(f"  $ {' '.join(cmd_filter)}", flush=True)
subprocess.run(cmd_filter, cwd=str(polarbear_root), check=True)
t_filter = time.perf_counter() - t1
print(f"  stage 2 wall: {t_filter:.1f}s", flush=True)


# Count emitted sites + write meta.
n_calls = sum(1 for _ in open(out_anc))
t_total = time.perf_counter() - t0
meta = {
    "in_trees": str(in_trees),
    "in_vcf": str(in_vcf),
    "theta": theta,
    "filter_condition": 3,
    "n_calls_written": n_calls,
    "timings_seconds": {
        "stage1_kernel": float(t_kernel),
        "stage2_filter": float(t_filter),
        "total": float(t_total),
    },
    "polarbear_root": str(polarbear_root),
    "work_dir": str(work_dir),
}
Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"\nrerun_polarbear_msl: wrote {n_calls:,} calls to {out_anc} "
    f"in {t_total:.1f}s total (kernel {t_kernel:.1f}s, filter {t_filter:.1f}s)",
    flush=True,
)
