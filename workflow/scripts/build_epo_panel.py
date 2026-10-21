"""Assemble the site panel of one chr1 region for the EPO comparison.

Combines the MSL ingroup state and genotypes with the aligned base of the nine
EPO outgroups, the GRCh38 reference base, the Ensembl EPO ancestral call and
the CpG context. A position is eligible where the MSL panel lists it, every
sampled haplotype is called, EPO aligns the human sequence in one block and
reports an ancestral base. A row is a site at which the sample of
``n_haplotypes`` or one of the outgroups carries a second base.

Output schema (``.npz``), coordinates relative to the region start:

- ``pos``       (N,) int32 — position within the region, 1-based
- ``sample``    (N, n_haplotypes) int8 — base per sampled haplotype
- ``reference`` (N,) int8 — GRCh38 reference base
- ``species``   (N, 9) int8 — base per outgroup, −1 where unaligned
- ``fast``      (N,) bool — CpG context: a C followed by a G, or a G preceded
  by a C, in any of the ten species at adjacent positions
- ``epo``       (N,) int8 — EPO call, 0..3 upper case, 4..7 lower case
- ``access``    (M, 2) int64 — half-open runs of eligible positions
- ``fast_start`` / ``fast_end`` — half-open runs of CpG-context positions

The metadata JSON carries the eligible-position counts, the substitution model
estimated from the data, the EPO tree and its branch-length multipliers, fitted
per CpG context under that model.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import REFERENCE, SPECIES, epo_tree, fitted_scale, sites_from_rows  # noqa: E402

from ancestree import BaseComposition  # noqa: E402

try:
    in_state = snakemake.input.state  # type: ignore[name-defined]
    in_polymorphic = snakemake.input.polymorphic  # type: ignore[name-defined]
    in_region_meta = snakemake.input.region_meta  # type: ignore[name-defined]
    in_alignment = list(snakemake.input.alignment)  # type: ignore[name-defined]
    in_fasta = snakemake.input.fasta  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    n_hap = int(snakemake.params.n_haplotypes)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    region = "chr1_143750000_150M"
    in_state = f"results/data/msl_ingroup_state_{region}.npz"
    in_polymorphic = f"results/data/msl_polymorphic_{region}.npz"
    in_region_meta = f"results/data/msl_polymorphic_{region}_meta.json"
    in_alignment = [f"results/data/epo_alignment_chr1_{part}.npz" for part in range(1, 10)]
    in_fasta = "external/ensembl_epo/homo_sapiens_ancestor_GRCh38/homo_sapiens_ancestor_1.fa"
    out_npz = f"results/data/epo_panel_{region}.npz"
    out_meta = f"results/data/epo_panel_{region}_meta.json"
    n_hap, mu = 20, 1.25e-8

#: The Ortheus defaults EPO runs with: HKY at 40% GC with an expected
#: transition/transversion ratio of 2, which is kappa = 25/6 there.
EPO_KAPPA, EPO_PI = 25.0 / 6.0, [0.3, 0.2, 0.2, 0.3]

with open(in_region_meta, encoding="utf-8") as f:
    region_meta = json.load(f)
start, end = int(region_meta["start"]), int(region_meta["end"])
size = end - start + 1  # index r = position - start, r in 1..end-start

state_npz = np.load(in_state)
state = np.full(size, -1, np.int8)
keep = (state_npz["pos"] > start) & (state_npz["pos"] <= end)
state[state_npz["pos"][keep] - start] = state_npz["state"][keep]

species = np.full((size, len(SPECIES)), -1, np.int8)
reference = np.full(size, -1, np.int8)
n_blocks = np.zeros(size, np.int8)  # alignment files covering the position
for path in in_alignment:
    aln = np.load(path, allow_pickle=True)
    assert list(aln["species_names"].astype(str)) == SPECIES
    inside = (aln["pos"] > start) & (aln["pos"] <= end)
    r = aln["pos"][inside] - start
    species[r] = aln["species"][inside]
    reference[r] = aln["human"][inside]
    n_blocks[r] += 1

with open(in_fasta, encoding="utf-8") as f:
    f.readline()
    ancestor = np.frombuffer("".join(line.strip() for line in f).encode(),
                             dtype=np.uint8)[start:end]
CODE = np.full(256, 8, np.int8)
for i, base in enumerate("ACGT"):
    CODE[ord(base)] = i
    CODE[ord(base.lower())] = 4 + i
epo = np.full(size, 8, np.int8)
epo[1:1 + len(ancestor)] = CODE[ancestor]

poly = np.load(in_polymorphic, allow_pickle=True)
sample_all = np.repeat(np.where(state >= 0, state, -1)[:, None], n_hap, axis=1)
inside = (poly["pos"] > start) & (poly["pos"] <= end)
inside &= state[np.clip(poly["pos"] - start, 0, size - 1)] == -2
if inside.any():
    sample_all[poly["pos"][inside] - start] = poly["ingroup"][inside, :n_hap]

# The base the full MSL panel has fixed must be the reference base.
aligned = (reference >= 0) & (n_blocks == 1) & (state != -1) & (sample_all >= 0).all(1)
aligned &= (state == -2) | (state == reference)
eligible = aligned & (epo < 8)

fast = np.zeros(size, bool)
for seq in [reference] + [species[:, j] for j in range(species.shape[1])]:
    fast[:-1] |= (seq[:-1] == 1) & (seq[1:] == 2)
    fast[1:] |= (seq[1:] == 2) & (seq[:-1] == 1)

panel = np.concatenate([sample_all, species], axis=1)
first = sample_all[:, :1]
varies = ((panel >= 0) & (panel != first)).any(1)
rows = np.flatnonzero(eligible & varies)


def runs(mask: np.ndarray) -> np.ndarray:
    """Half-open ``(start, end)`` runs of the true entries of ``mask``."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.zeros((0, 2), np.int64)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    return np.column_stack([np.concatenate([[idx[0]], idx[breaks + 1]]),
                            np.concatenate([idx[breaks], [idx[-1]]]) + 1])


fast_runs = runs(fast)
n_eligible = int(eligible.sum())
n_eligible_fast = int((eligible & fast).sum())

# Base frequencies from the reference over the eligible positions. Kappa from
# the transition/transversion count ratio R at the variant sites, which HKY
# expects to be kappa (pi_A pi_G + pi_C pi_T) / (pi_R pi_Y).
base_counts = np.bincount(reference[eligible], minlength=4)[:4]
pi = (base_counts / max(base_counts.sum(), 1)).tolist() if n_eligible else EPO_PI
kappa = EPO_KAPPA
if len(rows):
    names = [f"h{i}" for i in range(n_hap)] + SPECIES
    variant_sites, _ = sites_from_rows(rows, panel[rows], names)
    tally = BaseComposition.from_polymorphic_sites(variant_sites)
    a, c, g, t = pi
    kappa = float(tally.n_ts / tally.n_tv * (a + g) * (c + t) / (a * g + c * t))

# One branch-length multiplier per CpG context, fitted to the reference and
# outgroup columns of the eligible positions under that model.
tips = np.column_stack([reference, species])
scale = {}
for label, mask in (("non-CpG", ~fast), ("CpG", fast)):
    columns, weights = np.unique(tips[eligible & mask], axis=0, return_counts=True)
    scale[label] = fitted_scale(columns, weights, epo_tree(REFERENCE), kappa, pi,
                                max(n_eligible, 1), guess=1.0)

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_npz, pos=rows.astype(np.int32), sample=sample_all[rows], reference=reference[rows],
    species=species[rows], fast=fast[rows], epo=epo[rows], access=runs(eligible),
    fast_start=fast_runs[:, 0], fast_end=fast_runs[:, 1])
meta = dict(
    start=start, sequence_length=size, n_haplotypes=n_hap, n_rows=int(len(rows)),
    n_eligible=n_eligible, n_eligible_fast=n_eligible_fast,
    n_without_epo_call=int((aligned & (epo == 8)).sum()),
    mu=mu, rate_ratio=float(scale["CpG"] / scale["non-CpG"]),
    kappa=kappa, pi=pi,
    tree=epo_tree(REFERENCE),
    fixed_tree_runs=[
        dict(name="epo_model_one_rate", kappa=EPO_KAPPA, pi=EPO_PI, scale_slow=1.0, scale_fast=1.0),
        dict(name="estimated_model_class_rate", kappa=kappa, pi=pi,
             scale_slow=float(scale["non-CpG"]), scale_fast=float(scale["CpG"])),
    ],
)
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"build_epo_panel: {json.dumps({k: meta[k] for k in ('n_rows', 'n_eligible', 'n_eligible_fast', 'kappa', 'rate_ratio')})}",
      flush=True)
