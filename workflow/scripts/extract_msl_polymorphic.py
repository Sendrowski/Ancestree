"""B13 stage 1: Extract polymorphic sites from PolarBEAR's MSL pseudo-VCF
into a compact ``.npz`` intermediate that downstream stages can subsample
and load without OOMing.

Why a compact intermediate? The pseudo-VCF has 157M lines for the full
chr1 and downstream stages need polymorphic-only data. Each
:class:`ancestree.sites.Site` with a 176-entry ``tip_alleles`` dict is
≈18 KB; 1.1M sites materialise as ≈20 GB. The same data as int8 arrays
is ≈200 MB — a 100× reduction. The streaming cost is paid once per ``region``
wildcard, and subsampling / fitting / inference then run cheaply from the
cache.

Output schema (``.npz``):

- ``pos``       (N,) int32      — 1-based positions
- ``ref_idx``   (N,) int8       — REF allele index, A=0/C=1/G=2/T=3
- ``alt_idx``   (N, max_alts) int8 — ALT allele indices (−1 = absent)
- ``ingroup``   (N, 2·n_indiv) int8 — per-haplotype allele index, −1 = missing
- ``outgroup``  (N, n_outgroup) int8 — per-outgroup allele index, −1 = missing
- ``ingroup_names`` (2·n_indiv,) str — diploid expansion (``{id}_h0``, ``{id}_h1``)
- ``outgroup_names`` (n_outgroup,) str
- ``meta`` (0-d object): dict of source-file paths, region bounds, counts

Wildcards (read from ``snakemake`` if present, else module defaults):

- ``region``: stem like ``chr1_0_25M`` → positions 0..25_000_000
"""
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np


# -------------------------------------------------------- input config
try:
    in_simplify_vcf = snakemake.input.simplify_vcf  # type: ignore[name-defined]
    in_panel = snakemake.input.panel  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    region = str(snakemake.wildcards.region)  # type: ignore[name-defined]
    outgroup_samples = list(snakemake.params.outgroup_samples)  # type: ignore[name-defined]
except NameError:
    in_simplify_vcf = (
        "external/polarbear/data/real_data/ancestral_state/est_sfs/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.ponAbe2_macFas5.simplify.vcf.gz"
    )
    in_panel = "external/polarbear/data/real_data/ancestral_state/MSL.panel"
    region = "chr1_0_25M"
    out_npz = f"results/data/msl_polymorphic_{region}.npz"
    out_meta = f"results/data/msl_polymorphic_{region}_meta.json"
    outgroup_samples = ["ponAbe2", "macFas5"]


# Parse region wildcard. Supports K / M / G suffixes on either or both
# numeric components, so ``chr1_0_25M`` and ``chr1_25M_50M`` both work.
def _parse_bp(s: str) -> int:
    s = s.strip()
    if s and s[-1] in "KMG":
        mult = {"K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[s[-1]]
        return int(s[:-1]) * mult
    return int(s)


m = re.match(r"chr(\w+)_(\d+[KMG]?)_(\d+[KMG]?)$", region)
if not m:
    raise ValueError(f"region wildcard {region!r} does not match chrN_start_end (with optional K/M/G)")
chrom = m.group(1)
start = _parse_bp(m.group(2))
end = _parse_bp(m.group(3))


# -------------------------------------------------------- ingroup panel
ingroup_individuals: list[str] = []
with open(in_panel) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ingroup_individuals.append(line.split()[0])
ingroup_names: list[str] = []
for ind in ingroup_individuals:
    ingroup_names.append(f"{ind}_h0")
    ingroup_names.append(f"{ind}_h1")
n_ingroup_haps = len(ingroup_names)
n_outgroup = len(outgroup_samples)
print(
    f"extract_msl_polymorphic: region={region} ({chrom}:{start}-{end:,}), "
    f"{len(ingroup_individuals)} ingroup individuals ({n_ingroup_haps} haps), "
    f"{n_outgroup} outgroups",
    flush=True,
)


# -------------------------------------------------------- parse + collect
NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
MAX_ALTS = 3  # 4-state nucleotides → up to 3 ALTs after REF
expected_gt = n_ingroup_haps // 2 + n_outgroup  # ingroup is per-individual (a|b)


positions: list[int] = []
ref_idxs: list[int] = []
alt_idxs: list[list[int]] = []
ingroup_arr: list[list[int]] = []
outgroup_arr: list[list[int]] = []

n_lines_seen = 0
n_lines_in_range = 0
n_lines_polymorphic = 0
t0 = time.perf_counter()
report_every = 5_000_000

with gzip.open(in_simplify_vcf, "rt") as f:
    for line in f:
        n_lines_seen += 1
        if n_lines_seen % report_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [stream] {n_lines_seen:>11,} lines  ({elapsed:>5.1f}s)  "
                f"in-range={n_lines_in_range:>9,}  polymorphic={n_lines_polymorphic:>8,}",
                flush=True,
            )
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split()
        # est-sfs pseudo-VCF drops CHROM: POS REF ALT [ingroup GTs] [outgroup GTs]
        if len(parts) < 3 + expected_gt:
            continue
        pos = int(parts[0])
        if pos < start:
            continue
        if pos >= end:
            # File is position-sorted → stop here.
            break
        n_lines_in_range += 1

        ref_char = parts[1]
        alt_str = parts[2]
        if ref_char not in NUC:
            continue
        ref_i = NUC[ref_char]
        if alt_str == ".":
            alt_list: list[int] = []
        else:
            alt_chars = alt_str.split(",")
            if any(a not in NUC for a in alt_chars):
                continue  # non-canonical
            alt_list = [NUC[a] for a in alt_chars]
        if len(alt_list) > MAX_ALTS:
            continue  # >4 alleles segregating across REF+ALT — shouldn't happen

        gts = parts[3:]
        ingroup_gts_raw = gts[: n_ingroup_haps // 2]
        outgroup_gts_raw = gts[-n_outgroup:]
        # Allele indices: 0=REF, 1=ALT[0], 2=ALT[1], ...; -1=missing.
        allele_lookup: list[int] = [ref_i] + alt_list

        ingroup_row = np.full(n_ingroup_haps, -1, dtype=np.int8)
        for ind_idx, gt in enumerate(ingroup_gts_raw):
            if not gt or "." in gt:
                continue
            sep = "|" if "|" in gt else ("/" if "/" in gt else None)
            if sep is not None:
                a, b = gt.split(sep, 1)
            else:
                a = b = gt
            try:
                ai = int(a); bi = int(b)
            except ValueError:
                continue
            if 0 <= ai < len(allele_lookup):
                ingroup_row[2 * ind_idx] = allele_lookup[ai]
            if 0 <= bi < len(allele_lookup):
                ingroup_row[2 * ind_idx + 1] = allele_lookup[bi]

        outgroup_row = np.full(n_outgroup, -1, dtype=np.int8)
        for og_idx, gt in enumerate(outgroup_gts_raw):
            if not gt or "." in gt:
                continue
            try:
                ai = int(gt)
            except ValueError:
                continue
            if 0 <= ai < len(allele_lookup):
                outgroup_row[og_idx] = allele_lookup[ai]

        # Ingroup-polymorphic filter: at least two distinct observed
        # alleles among the 174 MSL haplotypes (real MSL SNPs only).
        # The pseudo-VCF also contains positions where MSL has no GT and
        # only the outgroup species carry data — those aren't MSL SNPs
        # and would inflate the comparison vs PolarBEAR / est-sfs, which
        # both restrict to ingroup-polymorphic sites.
        ingroup_observed = set(ingroup_row[ingroup_row >= 0].tolist())
        if len(ingroup_observed) < 2:
            continue
        n_lines_polymorphic += 1

        positions.append(pos)
        ref_idxs.append(ref_i)
        # Pad alt_list to MAX_ALTS with -1 sentinel
        padded = alt_list + [-1] * (MAX_ALTS - len(alt_list))
        alt_idxs.append(padded)
        ingroup_arr.append(ingroup_row.tolist())
        outgroup_arr.append(outgroup_row.tolist())


t_pass = time.perf_counter() - t0
print(
    f"extract_msl_polymorphic: streamed {n_lines_seen:,} lines in {t_pass:.1f}s; "
    f"{n_lines_in_range:,} in range; {n_lines_polymorphic:,} polymorphic kept",
    flush=True,
)


# -------------------------------------------------------- save
positions_arr = np.asarray(positions, dtype=np.int32)
ref_arr = np.asarray(ref_idxs, dtype=np.int8)
alt_arr = np.asarray(alt_idxs, dtype=np.int8)
ingroup_np = np.asarray(ingroup_arr, dtype=np.int8)
outgroup_np = np.asarray(outgroup_arr, dtype=np.int8)
ingroup_names_arr = np.asarray(ingroup_names)
outgroup_names_arr = np.asarray(outgroup_samples)

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_npz,
    pos=positions_arr,
    ref_idx=ref_arr,
    alt_idx=alt_arr,
    ingroup=ingroup_np,
    outgroup=outgroup_np,
    ingroup_names=ingroup_names_arr,
    outgroup_names=outgroup_names_arr,
)
size_mb = Path(out_npz).stat().st_size / 1024 / 1024
print(
    f"extract_msl_polymorphic: wrote {out_npz} "
    f"({size_mb:.1f} MB; arrays = {ingroup_np.nbytes / 1024 / 1024:.0f} MB raw)",
    flush=True,
)

meta = {
    "region": region,
    "chrom": chrom,
    "start": start,
    "end": end,
    "n_lines_seen": n_lines_seen,
    "n_lines_in_range": n_lines_in_range,
    "n_lines_polymorphic": n_lines_polymorphic,
    "n_ingroup_individuals": len(ingroup_individuals),
    "n_ingroup_haps": n_ingroup_haps,
    "n_outgroup": n_outgroup,
    "outgroup_samples": list(outgroup_samples),
    "elapsed_seconds": t_pass,
    "in_simplify_vcf": str(in_simplify_vcf),
    "in_panel": str(in_panel),
}
Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"extract_msl_polymorphic: wrote {out_meta}", flush=True)
