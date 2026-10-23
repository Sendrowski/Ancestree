"""Roulette germline mutation rates (Seplyarskiy et al. 2023) over one chr1
chunk of the EPO comparison.

The per-chromosome VCF is read remotely through its index, and each position
gets the sum of the ``MR`` estimates of its three possible substitutions.
Positions the model leaves out are NaN. Coordinates are those of the panel:
index ``r`` holds position ``start + r``.
"""
import json
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

try:
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_npz = snakemake.output.npz  # type: ignore[name-defined]
    url = snakemake.params.url  # type: ignore[name-defined]
except NameError:
    region = "chr1_93750000_100M"
    in_meta = f"results/data/epo_panel_{region}_meta.json"
    out_npz = f"results/data/roulette_{region}.npz"
    url = "http://genetics.bwh.harvard.edu/downloads/Vova/Roulette/1_rate_v5.2_TFBS_correction_all.vcf.bgz"

with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
start, size = int(meta["start"]), int(meta["sequence_length"])
rate = np.zeros(size, np.float64)
seen = np.zeros(size, bool)
for record in VCF(url)(f"1:{start + 1}-{start + size - 1}"):
    r = record.POS - start
    if 0 < r < size:
        rate[r] += record.INFO.get("MR")
        seen[r] = True
rate[~seen] = np.nan
Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out_npz, rate=rate.astype(np.float32))
print(f"fetch_roulette: {int(seen.sum()):,} of {size - 1:,} positions -> {out_npz}", flush=True)
