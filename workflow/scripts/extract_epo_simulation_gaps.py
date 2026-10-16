"""Outgroup gap mask of one simulated replicate, taken from an analysed chr1
chunk.

Walks the eligible positions of the chunk's panel in order and records for
each of the first ``length`` of them which of the nine outgroups is missing
there. Position ``i`` of the replicate then carries the gaps of the ``i``-th
eligible position of the chunk.

Output schema (``.npz``):

- ``gaps`` (length, 9) bool — outgroup missing, in the order of ``SPECIES``
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import SPECIES  # noqa: E402

try:
    in_region_meta = snakemake.input.region_meta  # type: ignore[name-defined]
    in_panel = snakemake.input.panel  # type: ignore[name-defined]
    in_alignment = list(snakemake.input.alignment)  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    length = int(snakemake.params.length)  # type: ignore[name-defined]
except NameError:
    region = "chr1_6250000_12500000"
    in_region_meta = f"results/data/msl_polymorphic_{region}_meta.json"
    in_panel = f"results/data/epo_panel_{region}.npz"
    in_alignment = [f"results/data/epo_alignment_chr1_{part}.npz" for part in range(1, 10)]
    out_npz = "results/data/epo_simulation_gaps_seed1.npz"
    length = 2_000_000

with open(in_region_meta, encoding="utf-8") as f:
    region_meta = json.load(f)
start, end = int(region_meta["start"]), int(region_meta["end"])
size = end - start + 1

species = np.full((size, len(SPECIES)), -1, np.int8)
for path in in_alignment:
    aln = np.load(path, allow_pickle=True)
    assert list(aln["species_names"].astype(str)) == SPECIES
    inside = (aln["pos"] > start) & (aln["pos"] <= end)
    species[aln["pos"][inside] - start] = aln["species"][inside]

eligible = np.concatenate([np.arange(a, b) for a, b in np.load(in_panel)["access"]])
if len(eligible) < length:
    raise ValueError(f"{len(eligible):,} eligible positions in {start}-{end}, {length:,} needed")
gaps = species[eligible[:length]] < 0

Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, gaps=gaps)
print(f"extract_epo_simulation_gaps: {gaps.mean():.4f} of outgroup entries missing "
      f"over {length:,} positions -> {out_npz}", flush=True)
