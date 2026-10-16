"""Tabulate the site patterns of one EPO alignment file over the positions
at which the MSL ingroup is fixed.

A pattern is the allele the ingroup has fixed, the aligned base of each of the
nine other primates, the Ensembl EPO ancestral call at the position and the
sequence context. Positions are kept where the EPO human base equals the
allele the ingroup has fixed.

Context classes: ``0`` non-CpG, ``1`` CpG in any of the ten species (a C
followed by a G, or a G preceded by a C, at adjacent human positions), ``2``
a neighbouring position is missing from the alignment.

EPO call codes: ``0..3`` upper-case A/C/G/T (high confidence), ``4..7``
lower-case a/c/g/t (low confidence), ``8`` N, ``9`` ``-``, ``10`` ``.``.

Output schema (``.npz``): ``human`` (P,), ``species`` (P, 9) with 4 = missing,
``epo`` (P,), ``context`` (P,), ``count`` (P,) int64, ``species_names``.
"""
import json
from pathlib import Path

import numpy as np

try:
    in_alignment = snakemake.input.alignment  # type: ignore[name-defined]
    in_states = list(snakemake.input.states)  # type: ignore[name-defined]
    in_fasta = snakemake.input.fasta  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
except NameError:
    import glob
    in_alignment = "results/data/epo_alignment_chr1_5.npz"
    in_states = sorted(glob.glob("results/data/msl_ingroup_state_chr1_*.npz"))
    in_fasta = ("external/ensembl_epo/homo_sapiens_ancestor_GRCh38/"
                "homo_sapiens_ancestor_1.fa")
    out_npz = "results/data/epo_patterns_chr1_5.npz"
    out_meta = "results/data/epo_patterns_chr1_5_meta.json"

#: Radix of each pattern field: human, nine species, EPO call, context.
N_EPO, N_CONTEXT = 11, 3

z = np.load(in_alignment, allow_pickle=True)
pos, human, species = z["pos"].astype(np.int64), z["human"], z["species"]
species_names = list(z["species_names"].astype(str))

with open(in_fasta, encoding="utf-8") as f:
    f.readline()
    ancestor = np.frombuffer("".join(line.strip() for line in f).encode(), dtype=np.uint8)
EPO_CODE = np.full(256, 8, dtype=np.int8)
for i, base in enumerate("ACGT"):
    EPO_CODE[ord(base)] = i
    EPO_CODE[ord(base.lower())] = 4 + i
EPO_CODE[ord("-")] = 9
EPO_CODE[ord(".")] = 10
epo = EPO_CODE[ancestor[pos - 1]]

# The ingroup state at every chr1 position: -1 where the panel lists nothing.
state = np.full(len(ancestor) + 1, -1, dtype=np.int8)
for path in in_states:
    s = np.load(path)
    state[s["pos"]] = s["state"]
fixed = state[pos]

# CpG context from every species' own sequence at adjacent human positions.
n = len(pos)
has_next = np.zeros(n, bool)
has_next[:-1] = pos[1:] == pos[:-1] + 1
has_prev = np.zeros(n, bool)
has_prev[1:] = has_next[:-1]
cpg = np.zeros(n, bool)
for seq in [human] + [species[:, j] for j in range(species.shape[1])]:
    nxt = np.full(n, -1, np.int8)
    prv = np.full(n, -1, np.int8)
    nxt[:-1] = seq[1:]
    prv[1:] = seq[:-1]
    cpg |= (seq == 1) & has_next & (nxt == 2)
    cpg |= (seq == 2) & has_prev & (prv == 1)
context = np.where(cpg, 1, np.where(has_next & has_prev, 0, 2)).astype(np.int64)

keep = (fixed >= 0) & (fixed == human)
coded = np.where(species >= 0, species, 4).astype(np.int64)
key = human.astype(np.int64)
for j in range(coded.shape[1]):
    key = key * 5 + coded[:, j]
key = (key * N_EPO + epo) * N_CONTEXT + context
unique, count = np.unique(key[keep], return_counts=True)

k = unique.copy()
out_context = (k % N_CONTEXT).astype(np.int8)
k //= N_CONTEXT
out_epo = (k % N_EPO).astype(np.int8)
k //= N_EPO
out_species = np.zeros((len(k), coded.shape[1]), dtype=np.int8)
for j in range(coded.shape[1] - 1, -1, -1):
    out_species[:, j] = k % 5
    k //= 5
Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, human=k.astype(np.int8), species=out_species,
                    epo=out_epo, context=out_context, count=count.astype(np.int64),
                    species_names=np.asarray(species_names))
meta = {
    "n_alignment_positions": int(n),
    "n_ingroup_fixed": int((fixed >= 0).sum()),
    "n_ingroup_polymorphic": int((fixed == -2).sum()),
    "n_not_listed_by_panel": int((fixed == -1).sum()),
    "n_fixed_allele_differs_from_epo_human": int(((fixed >= 0) & (fixed != human)).sum()),
    "n_kept": int(keep.sum()),
    "n_patterns": int(len(unique)),
}
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"tabulate_epo_patterns: {json.dumps(meta)}", flush=True)
