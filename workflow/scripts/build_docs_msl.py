"""Human example data of the docs, cut from one EPO comparison panel.

Writes, over one region of chr1 in GRCh38 coordinates:

- ``msl.vcf.gz``: every position at which the phased MSL sample or one of the
  nine EPO outgroups carries a second base. The MSL individuals are diploid,
  each outgroup a haploid sample, missing where unaligned. ``INFO/AA`` holds
  the Ensembl EPO ancestral base, lower case where EPO is less confident.
- ``msl.bed``: the positions the panel admits.
- ``msl_mu.bedgraph.gz``: the Roulette germline mutation rate at each admitted
  position, scaled to a mean of ``mu`` over them, the mean filling the
  positions Roulette leaves out.
- ``msl_rec.txt``: the HapMap II recombination map over the region.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import SPECIES  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_polymorphic = snakemake.input.polymorphic  # type: ignore[name-defined]
    in_rec_map = snakemake.input.rec_map  # type: ignore[name-defined]
    in_roulette = snakemake.input.roulette  # type: ignore[name-defined]
    out_vcf = snakemake.output.vcf  # type: ignore[name-defined]
    out_access = snakemake.output.access  # type: ignore[name-defined]
    out_mu = snakemake.output.mu  # type: ignore[name-defined]
    out_rec = snakemake.output.rec  # type: ignore[name-defined]
    left, right = snakemake.params.region  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    chunk = "chr1_93750000_100M"
    in_npz = f"results/data/epo_panel_{chunk}.npz"
    in_meta = f"results/data/epo_panel_{chunk}_meta.json"
    in_polymorphic = f"results/data/msl_polymorphic_{chunk}.npz"
    in_rec_map = "external/genetic_maps/HapMapII_GRCh38/genetic_map_Hg38_chr1.txt"
    in_roulette = f"results/data/roulette_{chunk}.npz"
    out_vcf, out_access = "docs/_static/msl.vcf.gz", "docs/_static/msl.bed"
    out_mu, out_rec = "docs/_static/msl_mu.bedgraph.gz", "docs/_static/msl_rec.txt"
    left, right = 95_000_000, 100_000_000
    mu = 1.25e-8

#: Sample name of each EPO outgroup, in the order of ``SPECIES``.
NAMES = {
    "pan_troglodytes": "chimpanzee", "pan_paniscus": "bonobo",
    "gorilla_gorilla": "gorilla", "pongo_abelii": "orangutan",
    "nomascus_leucogenys": "gibbon", "macaca_mulatta": "rhesus",
    "macaca_fascicularis": "cynomolgus", "chlorocebus_sabaeus": "vervet",
    "microcebus_murinus": "mouse_lemur",
}
CHROM = "chr1"

panel = np.load(in_npz)
with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
start = int(meta["start"])
haplotypes = np.load(in_polymorphic, allow_pickle=True)["ingroup_names"].astype(str)
n_hap = int(meta["n_haplotypes"])
individuals = list(dict.fromkeys(h.rsplit("_", 1)[0] for h in haplotypes[:n_hap]))
assert len(individuals) * 2 == n_hap

# Panel offsets r are 1-based positions relative to ``start``.
pos = panel["pos"].astype(np.int64) + start
keep = (pos > left) & (pos <= right)


def to_bed(runs: np.ndarray) -> np.ndarray:
    """Half-open offset runs as 0-based BED intervals clipped to the region."""
    bed = np.clip(runs.astype(np.int64) + start - 1, left, right)
    return bed[bed[:, 1] > bed[:, 0]]


access = to_bed(panel["access"])
Path(out_access).parent.mkdir(parents=True, exist_ok=True)
with open(out_access, "w", encoding="utf-8") as f:
    f.writelines(f"{CHROM}\t{s}\t{e}\n" for s, e in access)

# Roulette index r holds 1-based position start + r, so BED base b is r = b + 1 - start.
roulette = np.load(in_roulette)["rate"].astype(np.float64)
admitted = np.concatenate([np.arange(s, e) for s, e in access]) + 1 - start
rate = roulette[admitted]
rate = np.where(np.isfinite(rate), rate, np.nanmean(rate))
rate = np.array([float(f"{x:.3g}") for x in rate * mu / rate.mean()])
base = admitted + start - 1
# One line per run of equal rate over adjacent positions.
breaks = np.flatnonzero((np.diff(base) != 1) | (np.diff(rate) != 0)) + 1
run_start, run_end = np.r_[0, breaks], np.r_[breaks, len(base)]
with gzip.open(out_mu, "wt", encoding="utf-8") as f:
    f.writelines(f"{CHROM}\t{base[a]}\t{base[b - 1] + 1}\t{rate[a]:.3g}\n"
                 for a, b in zip(run_start, run_end))

BASES = np.array(list("ACGT"))
EPO = np.array(list("ACGTacgt."))
sample, species, epo = panel["sample"][keep], panel["species"][keep], panel["epo"][keep]
with gzip.open(out_vcf, "wt", encoding="utf-8") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write(f"##contig=<ID={CHROM},length=248956422,assembly=GRCh38>\n")
    f.write('##INFO=<ID=AA,Number=1,Type=String,Description="Ensembl EPO ancestral '
            'base (release 112), lower case where less confident">\n')
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(individuals + [NAMES[s] for s in SPECIES]) + "\n")
    for p, hap, out, aa in zip(pos[keep], sample, species, epo):
        seen = np.unique(np.concatenate([hap, out[out >= 0]]))
        # The sample's first haplotype is REF, the other bases ALT in order.
        alleles = [hap[0]] + [b for b in seen if b != hap[0]]
        index = {b: k for k, b in enumerate(alleles)}
        gts = [f"{index[hap[2 * k]]}|{index[hap[2 * k + 1]]}" for k in range(len(individuals))]
        gts += [str(index[b]) if b >= 0 else "." for b in out]
        f.write(f"{CHROM}\t{p}\t.\t{BASES[alleles[0]]}\t{','.join(BASES[alleles[1:]])}"
                f"\t.\tPASS\tAA={EPO[aa]}\tGT\t" + "\t".join(gts) + "\n")

with open(in_rec_map, encoding="utf-8") as f:
    header = f.readline().split()
    rows = [line.split() for line in f]
position = np.array([int(r[1]) for r in rows])
rate = np.array([float(r[2]) for r in rows])
# The map points bracketing the region, so its rate holds from the left edge.
first = max(int(np.searchsorted(position, left, side="right")) - 1, 0)
last = int(np.searchsorted(position, right, side="left")) + 1
cm = np.concatenate([[0.0], np.cumsum(np.diff(position[first:last]) * rate[first:last - 1] / 1e6)])
with open(out_rec, "w", encoding="utf-8") as f:
    f.write("Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n")
    f.writelines(f"{CHROM}\t{p}\t{r:.6f}\t{c:.6f}\n"
                 for p, r, c in zip(position[first:last], rate[first:last], cm))
print(f"build_docs_msl: {int(keep.sum()):,} records, {len(access):,} admitted runs "
      f"({len(base):,} bp), {len(run_start):,} mutation-rate runs", flush=True)
