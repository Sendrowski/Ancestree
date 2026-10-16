"""Extract the ingroup-fixed variant sites of one MSL pseudo-VCF chunk.

Companion of ``extract_msl_polymorphic.py``, which keeps the sites that
segregate within the ingroup. This step keeps the sites at which every called
ingroup haplotype carries one allele and at least one called outgroup carries
another. Together the two outputs hold every variant site of the panel, which
is what the fixed-tree branch-rate fit needs as its source.

An ingroup row with no genotype at all is read as homozygous reference: the
pseudo-VCF lists such positions for the outgroup alleles alone.

Output schema (``.npz``):

- ``pos``            (N,) int32 — 1-based positions
- ``ingroup_allele`` (N,) int8  — the allele the ingroup has fixed, A=0/C=1/G=2/T=3
- ``outgroup``       (N, n_outgroup) int8 — per-outgroup allele index, −1 = missing
- ``ingroup_names``  (2·n_indiv,) str
- ``outgroup_names`` (n_outgroup,) str
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np

try:
    in_simplify_vcf = snakemake.input.simplify_vcf  # type: ignore[name-defined]
    in_panel = snakemake.input.panel  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    outgroup_samples = list(snakemake.params.outgroup_samples)  # type: ignore[name-defined]
except NameError:
    region = "chr1_143750000_150M"
    in_simplify_vcf = f"results/data/msl_pseudo_vcf_{region}.vcf.gz"
    in_panel = "external/polarbear/data/real_data/ancestral_state/MSL.panel"
    out_npz = f"results/data/msl_fixed_differences_{region}.npz"
    out_meta = f"results/data/msl_fixed_differences_{region}_meta.json"
    outgroup_samples = ["ponAbe2", "macFas5"]

NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
n_outgroup = len(outgroup_samples)

ingroup_names: list[str] = []
with open(in_panel, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            ind = line.split()[0]
            ingroup_names += [f"{ind}_h0", f"{ind}_h1"]

positions: list[int] = []
ingroup_alleles: list[int] = []
outgroup_rows: list[list[int]] = []
n_lines = n_polymorphic = n_invariant = 0
t0 = time.perf_counter()
with gzip.open(in_simplify_vcf, "rt", encoding="utf-8") as f:
    for line in f:
        head = line.split(None, 3)
        if len(head) < 4:
            continue
        ref = NUC.get(head[1], -1)
        if ref < 0:
            continue
        alts = [] if head[2] == "." else [NUC.get(a, -1) for a in head[2].split(",")]
        if -1 in alts:
            continue
        n_lines += 1
        lookup = [ref] + alts
        rest = head[3].rsplit(None, n_outgroup)
        genotypes = rest[0]
        if "|" not in genotypes and "/" not in genotypes:
            fixed = ref
        else:
            seen = set(genotypes.replace("|", " ").replace("/", " ").split())
            seen.discard(".")
            if len(seen) != 1:
                n_polymorphic += 1
                continue
            k = int(next(iter(seen)))
            if k >= len(lookup):
                continue
            fixed = lookup[k]
        row = [lookup[int(g)] if g.isdigit() and int(g) < len(lookup) else -1
               for g in rest[1:]]
        if all(a in (-1, fixed) for a in row):
            n_invariant += 1
            continue
        positions.append(int(head[0]))
        ingroup_alleles.append(fixed)
        outgroup_rows.append(row)
elapsed = time.perf_counter() - t0

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_npz,
    pos=np.asarray(positions, dtype=np.int32),
    ingroup_allele=np.asarray(ingroup_alleles, dtype=np.int8),
    outgroup=np.asarray(outgroup_rows, dtype=np.int8).reshape(-1, n_outgroup),
    ingroup_names=np.asarray(ingroup_names),
    outgroup_names=np.asarray(outgroup_samples),
)
meta = {
    "n_lines": n_lines,
    "n_ingroup_polymorphic": n_polymorphic,
    "n_invariant": n_invariant,
    "n_fixed_differences": len(positions),
    "outgroup_samples": list(outgroup_samples),
    "elapsed_seconds": elapsed,
    "in_simplify_vcf": str(in_simplify_vcf),
}
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"extract_msl_fixed_differences: {json.dumps(meta)}", flush=True)
