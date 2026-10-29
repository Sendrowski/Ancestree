"""HKY parameters of the human docs example over its own region.

Estimated as ``build_epo_panel.py`` does for a whole chunk: the base
frequencies from the GRCh38 reference over the admitted positions, kappa from
the transition/transversion count ratio at the panel rows.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _epo_common import SPECIES, sites_from_rows  # noqa: E402

from ancestree import BaseComposition  # noqa: E402

try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_alignment = list(snakemake.input.alignment)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    left, right = snakemake.params.region  # type: ignore[name-defined]
except NameError:
    chunk = "chr1_93750000_100M"
    in_npz = f"results/data/epo_panel_{chunk}.npz"
    in_meta = f"results/data/epo_panel_{chunk}_meta.json"
    in_alignment = [f"results/data/epo_alignment_chr1_{part}.npz" for part in range(1, 10)]
    out_json = "results/data/docs_msl_model.json"
    left, right = 95_000_000, 100_000_000

panel = np.load(in_npz)
with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
start, n_hap = int(meta["start"]), int(meta["n_haplotypes"])

# Admitted 1-based positions inside (left, right].
runs = panel["access"].astype(np.int64) + start
admitted = np.concatenate([np.arange(s, e) for s, e in runs])
admitted = admitted[(admitted > left) & (admitted <= right)]

counts = np.zeros(4, np.int64)
for path in in_alignment:
    aln = np.load(path, allow_pickle=True)
    hit = np.isin(aln["pos"], admitted)
    counts += np.bincount(aln["human"][hit][aln["human"][hit] >= 0], minlength=4)[:4]
pi = (counts / counts.sum()).tolist()

pos = panel["pos"].astype(np.int64) + start
keep = (pos > left) & (pos <= right)
alleles = np.concatenate([panel["sample"][keep], panel["species"][keep]], axis=1)
names = [f"h{i}" for i in range(n_hap)] + SPECIES
sites, _ = sites_from_rows(pos[keep], alleles, names)
tally = BaseComposition.from_polymorphic_sites(sites)
a, c, g, t = pi
kappa = float(tally.n_ts / tally.n_tv * (a + g) * (c + t) / (a * g + c * t))

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(dict(region=[left, right], n_admitted=int(len(admitted)),
                   n_reference_bases=int(counts.sum()), kappa=kappa, pi=pi), f, indent=2)
print(f"estimate_docs_msl_model: kappa={kappa:.4f} pi={np.round(pi, 4).tolist()} "
      f"over {len(admitted):,} admitted positions", flush=True)
