"""Project one file of the Ensembl 10-primate EPO alignment onto human
coordinates.

Each alignment block that holds exactly one human sequence contributes, for
every human position it covers, the aligned base of each requested species.
A species absent from the block, present more than once in it (paralogues),
or carrying a gap or ambiguity code at the column is recorded as missing.
Blocks with several human sequences are skipped, and a human position covered
by more than one block is dropped.

Output schema (``.npz``):

- ``pos``     (N,) int32 — 1-based human positions, sorted
- ``human``   (N,) int8  — human reference base, A=0/C=1/G=2/T=3
- ``species`` (N, n_species) int8 — aligned base per species, −1 = missing
- ``species_names`` (n_species,) str
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np

try:
    in_maf = snakemake.input.maf  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    species = list(snakemake.params.species)  # type: ignore[name-defined]
    chrom = str(snakemake.params.chrom)  # type: ignore[name-defined]
except NameError:
    in_maf = "external/ensembl_epo/10_primates_chr1/10_primates.epo.1_5.maf.gz"
    out_npz = "results/data/epo_alignment_chr1_5.npz"
    out_meta = "results/data/epo_alignment_chr1_5_meta.json"
    species = ["pan_troglodytes", "gorilla_gorilla", "pongo_abelii", "macaca_mulatta"]
    chrom = "1"

HUMAN = f"homo_sapiens.{chrom}"

# Byte to base index, upper and lower case alike. Everything else is missing.
CODE = np.full(256, -1, dtype=np.int8)
for i, base in enumerate("ACGT"):
    CODE[ord(base)] = i
    CODE[ord(base.lower())] = i
COMPLEMENT = np.array([3, 2, 1, 0], dtype=np.int8)

positions: list[np.ndarray] = []
humans: list[np.ndarray] = []
aligned: list[np.ndarray] = []
n_blocks = n_multi_human = n_no_human = 0


def flush(block: list[list[str]]) -> None:
    """Project one block's rows onto the human positions it covers."""
    global n_blocks, n_multi_human, n_no_human
    if not block:
        return
    n_blocks += 1
    human_rows = [r for r in block if r[1] == HUMAN]
    if len(human_rows) != 1:
        if human_rows:
            n_multi_human += 1
        else:
            n_no_human += 1
        return
    _, _, start, size, strand, src_size, text = human_rows[0]
    h = CODE[np.frombuffer(text.encode(), dtype=np.uint8)]
    column = np.flatnonzero(np.frombuffer(text.encode(), dtype=np.uint8) != ord("-"))
    offset = np.arange(len(column), dtype=np.int64)
    if strand == "+":
        pos = int(start) + 1 + offset
    else:
        pos = int(src_size) - int(start) - offset
    rows = np.full((len(column), len(species)), -1, dtype=np.int8)
    for j, name in enumerate(species):
        found = [r for r in block if r[1].split(".", 1)[0] == name]
        if len(found) == 1:
            rows[:, j] = CODE[np.frombuffer(found[0][6].encode(), dtype=np.uint8)][column]
    human = h[column]
    if strand == "-":
        human = np.where(human >= 0, COMPLEMENT[human], -1)
        rows = np.where(rows >= 0, COMPLEMENT[rows], -1)
    keep = human >= 0
    positions.append(pos[keep].astype(np.int32))
    humans.append(human[keep])
    aligned.append(rows[keep])


t0 = time.perf_counter()
block: list[list[str]] = []
with gzip.open(in_maf, "rt", encoding="utf-8") as f:
    for line in f:
        if line.startswith("s "):
            block.append(line.split())
        elif line.startswith("a"):
            flush(block)
            block = []
flush(block)

pos = np.concatenate(positions) if positions else np.zeros(0, np.int32)
human = np.concatenate(humans) if humans else np.zeros(0, np.int8)
rows = (np.concatenate(aligned) if aligned
        else np.zeros((0, len(species)), np.int8))
order = np.argsort(pos, kind="stable")
pos, human, rows = pos[order], human[order], rows[order]
unique, counts = np.unique(pos, return_counts=True)
single = np.isin(pos, unique[counts == 1])
n_overlapping = int((~single).sum())
pos, human, rows = pos[single], human[single], rows[single]

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, pos=pos, human=human, species=rows,
                    species_names=np.asarray(species))
meta = {
    "n_blocks": n_blocks,
    "n_blocks_several_human": n_multi_human,
    "n_blocks_no_human": n_no_human,
    "n_positions": int(len(pos)),
    "n_positions_dropped_overlap": n_overlapping,
    "n_called_per_species": {s: int((rows[:, j] >= 0).sum())
                             for j, s in enumerate(species)},
    "elapsed_seconds": time.perf_counter() - t0,
    "in_maf": str(in_maf),
}
with open(out_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print(f"extract_epo_alignment: {json.dumps(meta)}", flush=True)
