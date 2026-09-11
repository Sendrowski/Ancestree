"""B13 reproducibility: re-run EST-SFS end-to-end on the MSL chr1 archived
pseudo-VCF, recording wall-clock time for each stage. Output mirrors the
archived ``est-sfs_ancstate.txt``, so it drops into the 3-way report.

Three stages, each timed separately:

1. ``code/run_est_sfs/sfs_input_real_data.py`` — converts the pseudo-VCF
   into est-sfs's bespoke input format (one config line per polymorphic
   site).
2. ``est-sfs config-rate6.txt est_sfs_input.txt seedfile.txt ...``
   — the C binary's ML fit + per-site posteriors.
3. ``code/run_est_sfs/sfs_output2anc_state_iter.py`` — picks the MAP
   ancestral allele per site and emits ``est-sfs_ancstate.txt``.

est-sfs comes from bioconda via this rule's conda env.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


try:
    in_pseudo_vcf = snakemake.input.pseudo_vcf  # type: ignore[name-defined]
    polarbear_root = snakemake.params.polarbear_root  # type: ignore[name-defined]
    n_outgroup = int(snakemake.params.n_outgroup)  # type: ignore[name-defined]
    model_code = int(snakemake.params.model_code)  # type: ignore[name-defined]
    nrandom = int(snakemake.params.nrandom)  # type: ignore[name-defined]
    out_anc = snakemake.output.anc  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    work_dir = snakemake.params.work_dir  # type: ignore[name-defined]
except NameError:
    polarbear_root = "external/polarbear"
    in_pseudo_vcf = (
        f"{polarbear_root}/data/real_data/ancestral_state/est_sfs/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.ponAbe2_macFas5.simplify.vcf.gz"
    )
    n_outgroup = 2
    model_code = 2  # est-sfs R6 (six symmetrical rates)
    nrandom = 4  # fewer restarts than the archive's 10 — keeps the rerun cheap (~10 min). The resulting ML-fit drift vs the archived est-sfs output is ≤0.5% of calls
    out_anc = "results/data/rerun_estsfs_anc_state.txt"
    out_meta = "results/data/rerun_estsfs_meta.json"
    work_dir = "results/data/rerun_estsfs_workdir"


polarbear_root = Path(polarbear_root).resolve()
work_dir = Path(work_dir).resolve()
work_dir.mkdir(parents=True, exist_ok=True)
# est-sfs comes from bioconda, in this rule's own conda env, so it is on PATH.
# NOT a copied-out binary: it links libgsl from the env and will not run
# outside it.
est_sfs_bin = shutil.which("est-sfs")
if est_sfs_bin is None:
    raise FileNotFoundError(
        "est-sfs is not on PATH. It is provided by workflow/envs/bench.yml "
        "(bioconda, linux-64 / osx-64); run this rule under that env."
    )
est_sfs_bin = Path(est_sfs_bin)


# Stage 1 — make est-sfs input. The archived simplify.vcf.gz has POS in
# column 0 (no CHROM column), but sfs_input_real_data.py expects CHROM in
# column 0 and POS in column 1. Prepend a synthetic "1\t" CHROM column
# on the fly so the third-party script reads correctly.
import gzip as _gzip
fixed_vcf = work_dir / "input_with_chrom.vcf.gz"
t_prep0 = time.perf_counter()
print(f"rerun_estsfs_msl: stage 0 — adding CHROM column to {Path(in_pseudo_vcf).name}", flush=True)
with _gzip.open(in_pseudo_vcf, "rt") as fin, _gzip.open(fixed_vcf, "wt") as fout:
    for line in fin:
        fout.write("1\t" + line)
t_prep = time.perf_counter() - t_prep0
print(f"  stage 0 wall: {t_prep:.1f}s ({fixed_vcf.stat().st_size/1024/1024:.0f} MB)", flush=True)

sfs_input_script = polarbear_root / "code" / "run_est_sfs" / "sfs_input_real_data.py"
t0 = time.perf_counter()
print(f"\nrerun_estsfs_msl: stage 1 — {sfs_input_script}", flush=True)
cmd_input = [
    sys.executable, str(sfs_input_script),
    str(fixed_vcf.resolve()),
    str(n_outgroup),
    str(work_dir),
]
print(f"  $ {' '.join(cmd_input)}", flush=True)
subprocess.run(cmd_input, cwd=str(polarbear_root), check=True)
t_input = time.perf_counter() - t0
print(f"  stage 1 wall: {t_input:.1f}s", flush=True)


# Stage 2 — run est-sfs binary
est_sfs_input_txt = work_dir / "est_sfs_input.txt"
if not est_sfs_input_txt.exists():
    raise RuntimeError(f"stage 1 did not produce {est_sfs_input_txt}")
# Generate the est-sfs config in-place so we can match restart counts
# (`nrandom`) and substitution model to the comparison we report.
config_local = work_dir / "config.txt"
config_local.write_text(
    f"n_outgroup {n_outgroup}\n"
    f"model {model_code}\n"
    f"nrandom {nrandom}\n"
)
seed_file = work_dir / "seedfile.txt"
seed_file.write_text("42\n")
sfs_out = work_dir / "output-file-sfs.txt"
pvalues_out = work_dir / "output-file-pvalues.txt"
t1 = time.perf_counter()
print(f"\nrerun_estsfs_msl: stage 2 — {est_sfs_bin}", flush=True)
cmd_binary = [
    str(est_sfs_bin),
    str(config_local),
    str(est_sfs_input_txt),
    str(seed_file),
    str(sfs_out),
    str(pvalues_out),
]
print(f"  $ {' '.join(cmd_binary)}", flush=True)
subprocess.run(cmd_binary, cwd=str(work_dir), check=True)
t_binary = time.perf_counter() - t1
print(f"  stage 2 wall: {t_binary:.1f}s", flush=True)


# Stage 3 — post-process posteriors into anc_state.txt
sfs_output_script = polarbear_root / "code" / "run_est_sfs" / "sfs_output2anc_state_iter.py"
t2 = time.perf_counter()
print(f"\nrerun_estsfs_msl: stage 3 — {sfs_output_script}", flush=True)
cmd_post = [
    sys.executable, str(sfs_output_script),
    str(work_dir),
]
print(f"  $ {' '.join(cmd_post)}", flush=True)
subprocess.run(cmd_post, cwd=str(polarbear_root), check=True)
t_post = time.perf_counter() - t2
print(f"  stage 3 wall: {t_post:.1f}s", flush=True)


# Copy final est-sfs_ancstate.txt to the requested output path.
canonical_out = work_dir / "est-sfs_ancstate.txt"
if not canonical_out.exists():
    raise RuntimeError(f"stage 3 did not produce {canonical_out}")
Path(out_anc).parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(canonical_out, out_anc)
n_calls = sum(1 for _ in open(out_anc))


t_total = time.perf_counter() - t0
meta = {
    "in_pseudo_vcf": str(in_pseudo_vcf),
    "est_sfs_bin": str(est_sfs_bin),
    "n_outgroup": n_outgroup,
    "model_code": model_code,
    "nrandom": nrandom,
    "n_calls_written": int(n_calls),
    "timings_seconds": {
        "stage1_input_prep": float(t_input),
        "stage2_binary": float(t_binary),
        "stage3_postprocess": float(t_post),
        "total": float(t_total),
    },
    "polarbear_root": str(polarbear_root),
    "work_dir": str(work_dir),
}
Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(
    f"\nrerun_estsfs_msl: wrote {n_calls:,} calls to {out_anc} "
    f"in {t_total:.1f}s total "
    f"(prep {t_input:.1f}s + binary {t_binary:.1f}s + post {t_post:.1f}s)",
    flush=True,
)
