"""B13 (streaming variant): whole-chr1 MSL fixed-tree fit in a single pass.

The chunked :mod:`fit_infer_msl_vcf` script split chr1 into 40 positional
regions and fit per-branch rates on a 100k-site random subsample per chunk,
purely to dodge ``FixedTreeInference``'s OOM on the full 1.1M-site cell
(materialising 1.1M ``Site`` objects at once). ``FixedTreeInference(...,
stream=True)`` removes that ceiling: it walks a *re-iterable* source in
bounded memory (config-histogram pass for the fit, then a per-site posterior
pass), never holding the full site list.

This script therefore fits ONE global outgroup-ladder rate set + HKY kappa on
*every* polymorphic chr1 SNP at once and emits per-site MAP calls in the same
``pos<TAB>allele_idx`` format as the chunked script, so the 3-way report can
consume it unchanged.

Wildcards (read from ``snakemake``, else module defaults glob every chunk):

- ``npz_glob`` — glob over the per-region ``msl_polymorphic_*.npz`` chunks
"""
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

from ancestree import (
    BaseComposition,
    FixedTreeInference,
    HKY,
    KingmanIngroupWeight,
    STATES,
    STATE_INDEX,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_io import MslPolymorphicNpzSource  # noqa: E402


# -------------------------------------------------------- input config
try:
    npz_paths = list(snakemake.input.npz)  # type: ignore[name-defined]
    meta_paths = list(snakemake.input.meta)  # type: ignore[name-defined]
    out_anc = snakemake.output.anc  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    out_posteriors = snakemake.output.posteriors  # type: ignore[name-defined]
except NameError:
    npz_paths = sorted(
        p for p in glob.glob("results/data/msl_polymorphic_chr1_*.npz")
        if not p.endswith("_meta.json")
    )
    meta_paths = [p.replace(".npz", "_meta.json") for p in npz_paths]
    out_anc = "results/data/msl_vcf_chr1_full_stream_calls.txt"
    out_meta = "results/data/msl_vcf_chr1_full_stream_meta.json"
    out_posteriors = "results/data/msl_vcf_chr1_full_stream_posteriors.npz"


# Sort chunks by genomic start so the stream is position-ordered.
def _start(meta_path: str) -> int:
    with open(meta_path) as f:
        return int(json.load(f)["start"])


order = sorted(range(len(npz_paths)), key=lambda i: _start(meta_paths[i]))
npz_paths = [npz_paths[i] for i in order]
meta_paths = [meta_paths[i] for i in order]

source = MslPolymorphicNpzSource(npz_paths)
ingroup_names = source.ingroup_names
outgroup_names = source.outgroup_names
print(
    f"stream_msl_vcf: {len(npz_paths)} chunks, "
    f"{len(ingroup_names)} ingroup haps, outgroups={outgroup_names}",
    flush=True,
)


# -------------------------------------------------------- BC (streaming pass)
t_bc0 = time.perf_counter()
bc = BaseComposition.from_polymorphic_sites(source)
kappa_hat = bc.kappa_estimate
print(
    f"stream_msl_vcf: empirical pi={bc.pi.tolist()}, "
    f"n_ts={bc.n_ts}, n_tv={bc.n_tv}, kappa_hat={kappa_hat:.3f}  "
    f"({time.perf_counter() - t_bc0:.1f}s, streamed)",
    flush=True,
)

# n_target_sites: total callable span = sum of per-chunk region spans
# (full chr1). The polymorphic / monomorphic ratio is what the rate fit reads.
n_target_sites = 0
for mp in meta_paths:
    with open(mp) as f:
        m = json.load(f)
    n_target_sites += int(m["end"]) - int(m["start"])

prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
model = HKY(kappa=kappa_hat, fit_kappa=True)


# -------------------------------------------------------- fit (streaming)
# Streaming form: pass the re-iterable source first; FixedTreeInference builds
# the OutgroupLadderTree from ingroup/outgroup sample lists internally and
# walks the source twice (config-histogram fit, then per-site infer).
t_fit0 = time.perf_counter()
inference = FixedTreeInference(
    source,
    model=model,
    ingroup_samples=ingroup_names,
    outgroup_samples=outgroup_names,
    base_composition=bc,
    n_target_sites=n_target_sites,
    ingroup_weight=prior,
    n_starts=4,
    parallelize=False,
    progress=False,
    stream=True,
)
params = inference.fit()
t_fit = time.perf_counter() - t_fit0
print(
    f"stream_msl_vcf: stream fit() took {t_fit:.1f}s; params={params}",
    flush=True,
)


# -------------------------------------------------------- infer + write
t_infer0 = time.perf_counter()
Path(out_anc).parent.mkdir(parents=True, exist_ok=True)
n_written = n_skipped = 0
post_positions: list[int] = []
post_values: list[list[float]] = []
with open(out_anc, "w") as f:
    for site, posterior in inference.infer():
        allele = posterior.map_allele
        if allele not in STATE_INDEX:
            n_skipped += 1
            continue
        pos_int = int(site.pos)
        f.write(f"{pos_int}\t{STATE_INDEX[allele]}\n")
        post_positions.append(pos_int)
        post_values.append([float(v) for v in posterior.values])
        n_written += 1
t_infer = time.perf_counter() - t_infer0
print(
    f"stream_msl_vcf: wrote {n_written:,} per-site calls to {out_anc} "
    f"(skipped {n_skipped} non-canonical) in {t_infer:.1f}s",
    flush=True,
)

Path(out_posteriors).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    out_posteriors,
    pos=np.asarray(post_positions, dtype=np.int32),
    posterior=np.asarray(post_values, dtype=np.float32),
    alleles=np.asarray(list(STATES)),
)
print(f"stream_msl_vcf: wrote per-site posteriors to {out_posteriors}", flush=True)


# -------------------------------------------------------- meta
meta = {
    "mode": "fixed_tree_streaming_whole_chr1",
    "n_chunks": len(npz_paths),
    "n_sites_written": n_written,
    "n_sites_skipped_unknown_allele": n_skipped,
    "n_target_sites": int(n_target_sites),
    "model": "HKY",
    "kappa_initial": float(kappa_hat),
    "kappa_fitted": (
        inference._model_params_mle.get("kappa")
        if inference._model_params_mle else None
    ),
    "params_mle": {k: float(v) for k, v in params.items()},
    "wall_fit_s": t_fit,
    "wall_infer_s": t_infer,
    "wall_total_s": t_fit + t_infer,
}
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"stream_msl_vcf: wrote meta to {out_meta}", flush=True)
