"""Per-position ingroup state of one MSL pseudo-VCF chunk.

For every position the chunk lists, records whether the MSL ingroup is fixed
for one allele or polymorphic. An ingroup row with no genotype at all is read
as homozygous reference.

Output schema (``.npz``):

- ``pos``   (N,) int32 — 1-based positions
- ``state`` (N,) int8  — the fixed allele, A=0/C=1/G=2/T=3, or −2 where the
  ingroup is polymorphic
"""
import gzip
from pathlib import Path

import numpy as np

try:
    in_simplify_vcf = snakemake.input.simplify_vcf  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    n_outgroup = len(snakemake.params.outgroup_samples)  # type: ignore[name-defined]
except NameError:
    region = "chr1_143750000_150M"
    in_simplify_vcf = f"results/data/msl_pseudo_vcf_{region}.vcf.gz"
    out_npz = f"results/data/msl_ingroup_state_{region}.npz"
    n_outgroup = 2

NUC = {"A": 0, "C": 1, "G": 2, "T": 3}
positions: list[int] = []
states: list[int] = []
with gzip.open(in_simplify_vcf, "rt") as f:
    for line in f:
        head = line.split(None, 3)
        if len(head) < 4:
            continue
        ref = NUC.get(head[1], -1)
        if ref < 0:
            continue
        genotypes = head[3].rsplit(None, n_outgroup)[0]
        if "|" not in genotypes and "/" not in genotypes:
            state = ref
        else:
            seen = set(genotypes.replace("|", " ").replace("/", " ").split())
            seen.discard(".")
            alts = [] if head[2] == "." else [NUC.get(a, -1) for a in head[2].split(",")]
            lookup = [ref] + alts
            if len(seen) == 1 and int(next(iter(seen))) < len(lookup):
                state = lookup[int(next(iter(seen)))]
                if state < 0:
                    continue
            else:
                state = -2
        positions.append(int(head[0]))
        states.append(state)

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, pos=np.asarray(positions, dtype=np.int32),
                    state=np.asarray(states, dtype=np.int8))
print(f"extract_msl_ingroup_state: {len(positions):,} positions -> {out_npz}", flush=True)
