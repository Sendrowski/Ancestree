"""SLiM-outgroup-bias benchmark: build a tsinfer-inferred ARG from the
ingroup VCF for one ``(chunk_idx, f_del)`` cell.

tsinfer needs biallelic SNPs and integer-encoded genotypes, so the input is
restricted to the ingroup haplotypes (outgroups are dropped before inference,
the tsinfer-ARG path being ingroup-only by design) and multi-allelic and
missing-data sites are skipped. ``recombination_rate=rho_slim``
sets the matching model tsinfer builds the topology under.

tsinfer emits uncalibrated node times, which carry no scale a
substitution rate applies to, so the ARG is dated with tsdate at the
SLiM rate before it is written and its times are then in generations.
The rate the kernel must use is recorded in the side meta, since a chunk
too sparse for tsdate falls back to a rate fitted against the ARG's own
time axis.

Output is a single ``.trees`` file holding the dated ts. The next
script (``infer_slim_outgroup_bias_tsinfer_arg.py``) runs Felsenstein
on it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cyvcf2
import numpy as np
import tsinfer


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI fallback when not driven by snakemake."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-vcf", required=True)
    p.add_argument("--in-meta", required=True)
    p.add_argument("--out-trees", required=True)
    p.add_argument("--rec-rate-slim", type=float, required=True)
    p.add_argument("--mu-slim", type=float, required=True)
    return p.parse_args(argv)


try:
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_trees = snakemake.output[0]  # type: ignore[name-defined]
    rec_rate_slim = float(snakemake.params.rec_rate_slim)  # type: ignore[name-defined]
    mu_slim = float(snakemake.params.mu_slim)  # type: ignore[name-defined]
except NameError:
    _args = _parse_args(sys.argv[1:])
    in_vcf = _args.in_vcf
    in_meta = _args.in_meta
    out_trees = _args.out_trees
    rec_rate_slim = _args.rec_rate_slim
    mu_slim = _args.mu_slim


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
seq_len: int = int(meta["seq_len"])

print(
    f"build_tsinfer_arg: {in_vcf} -> {out_trees}; "
    f"ingroup={len(ingroup_names)} haps, seq_len={seq_len}, "
    f"rho_slim={rec_rate_slim:g}",
    flush=True,
)

# Determine the column indices of the ingroup samples in the VCF. We
# subset genotype rows to these before feeding tsinfer.
vcf = cyvcf2.VCF(in_vcf)
sample_to_idx = {s: i for i, s in enumerate(vcf.samples)}
missing = [s for s in ingroup_names if s not in sample_to_idx]
if missing:
    raise ValueError(f"VCF missing ingroup samples: {missing[:5]}")
ingroup_cols = np.array([sample_to_idx[s] for s in ingroup_names], dtype=np.int64)

n_ing = len(ingroup_names)
STATES = set("ACGT")

_t0 = time.perf_counter()
n_skipped_multiallelic = 0
n_skipped_missing = 0
n_skipped_invariant = 0
n_skipped_bad_allele = 0
n_used = 0

# In-memory SampleData (no path=) so we don't depend on the LMDB on-disk
# format. The dataset is small (~hundreds-of-thousands of sites at most)
# so the in-memory build is fast and reload-free.
sd = tsinfer.SampleData(
    sequence_length=seq_len,
)
with sd:
    samples = sd
    for variant in vcf:
        pos = int(variant.POS) - 1  # cyvcf2 is 1-based. Tskit/tsinfer is 0-based
        if pos < 0 or pos >= seq_len:
            continue
        # variant.genotypes is a list of [a, b, phased] per individual.
        # SLiM VCF here is haploid: one allele per individual.
        ref = variant.REF.upper()
        alts = [a.upper() for a in (variant.ALT or [])]
        alleles = [ref] + alts
        if not all(a in STATES for a in alleles):
            n_skipped_bad_allele += 1
            continue
        if len(alleles) != 2:
            # tsinfer's standard infer path requires biallelic input.
            n_skipped_multiallelic += 1
            continue
        gts_struct = variant.genotypes
        # Haploid layout: first slot per individual carries the call.
        gts = np.array([int(g[0]) for g in gts_struct], dtype=np.int8)
        gts_ing = gts[ingroup_cols]
        if (gts_ing < 0).any():
            n_skipped_missing += 1
            continue
        if np.unique(gts_ing).size < 2:
            n_skipped_invariant += 1
            continue
        samples.add_site(pos, gts_ing.tolist(), alleles)
        n_used += 1

    vcf.close()
site_prep_seconds = float(time.perf_counter() - _t0)
print(
    f"  added {n_used} sites to tsinfer.SampleData "
    f"(skipped {n_skipped_multiallelic} multi-allelic, "
    f"{n_skipped_missing} with missing data, "
    f"{n_skipped_invariant} invariant-in-ingroup, "
    f"{n_skipped_bad_allele} with non-ACGT alleles); "
    f"prep={site_prep_seconds:.2f}s",
    flush=True,
)

if n_used == 0:
    raise RuntimeError(
        f"No usable biallelic SNPs from {in_vcf} for tsinfer (all sites "
        f"were filtered)."
    )

_t0 = time.perf_counter()
# mismatch_ratio=1 is tsinfer's default-ish. recombination_rate drives the
# ancestor-matching model, not the time scale.
ts_inferred = tsinfer.infer(
    sd,
    recombination_rate=rec_rate_slim,
    mismatch_ratio=1.0,
)
infer_seconds = float(time.perf_counter() - _t0)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _robustness_common import _date_tsinfer  # noqa: E402

_t0 = time.perf_counter()
ts_inferred, mu_used = _date_tsinfer(ts_inferred, mu_slim)
date_seconds = float(time.perf_counter() - _t0)
print(f"  tsdate: time_units={ts_inferred.time_units!r}, "
      f"mu={mu_used:.3e} ({date_seconds:.2f}s)", flush=True)

print(
    f"  tsinfer.infer: {ts_inferred.num_samples} samples, "
    f"{ts_inferred.num_sites} sites, {ts_inferred.num_trees} trees "
    f"({infer_seconds:.2f}s)",
    flush=True,
)

Path(out_trees).parent.mkdir(parents=True, exist_ok=True)
ts_inferred.dump(out_trees)
print(f"Wrote {out_trees}", flush=True)

# Side-meta with timings for the report stage.
side_meta = {
    "tsinfer_prep_seconds": site_prep_seconds,
    "tsinfer_infer_seconds": infer_seconds,
    "tsinfer_n_sites_used": n_used,
    "tsinfer_n_skipped_multiallelic": n_skipped_multiallelic,
    "tsinfer_n_skipped_missing": n_skipped_missing,
    "tsinfer_n_skipped_invariant": n_skipped_invariant,
    "tsinfer_n_skipped_bad_allele": n_skipped_bad_allele,
    "tsinfer_num_trees": int(ts_inferred.num_trees),
    "tsinfer_num_edges": int(ts_inferred.num_edges),
    "tsinfer_num_sites": int(ts_inferred.num_sites),
    "tsdate_seconds": date_seconds,
    # The rate the kernel must run at, and the axis it is expressed per.
    "mu": mu_used,
    "time_units": str(ts_inferred.time_units),
}
with open(str(out_trees) + "_meta.json", "w") as f:
    json.dump(side_meta, f, indent=2)
