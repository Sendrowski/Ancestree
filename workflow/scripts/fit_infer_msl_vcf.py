"""B13 stage 2: load the compact polymorphic-sites .npz from stage 1,
optionally subsample, run FixedTreeInference (fit + infer), emit per-site
MAP calls in PolarBEAR-comparable ``pos<TAB>allele_idx`` format.

The subsample wildcard avoids ``FixedTreeInference``'s OOM on the full
1.1M-site cell: the MLE only needs a representative subset to fit
per-branch divergence rates + HKY κ. Per-site posteriors can then be
emitted for either the subsample (fast) or all sites (slower, fresh pass).

Wildcards (read from ``snakemake``, else module defaults):

- ``region``       — matches a stage-1 ``msl_polymorphic_{region}.npz``
- ``n_subsample``  — number of sites to keep before fitting; ``full`` to skip subsampling
- ``seed``         — RNG seed for the random subsample
- ``infer_scope``  — ``subsample`` (default, fast) or ``all`` (posterior over every site)
"""
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
    OutgroupLadderTree,
    STATES,
    STATE_INDEX,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_io import sites_from_arrays  # noqa: E402


# -------------------------------------------------------- input config
try:
    in_npz = snakemake.input.npz  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_anc = snakemake.output.anc  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    out_posteriors = snakemake.output.posteriors  # type: ignore[name-defined]
    region = str(snakemake.wildcards.region)  # type: ignore[name-defined]
    n_subsample_w = str(snakemake.wildcards.n_subsample)  # type: ignore[name-defined]
    seed = int(snakemake.wildcards.seed)  # type: ignore[name-defined]
    infer_scope = str(getattr(snakemake.wildcards, "infer_scope", "subsample"))  # type: ignore[name-defined]
except NameError:
    region = "chr1_0_25M"
    in_npz = f"results/data/msl_polymorphic_{region}.npz"
    in_meta = f"results/data/msl_polymorphic_{region}_meta.json"
    n_subsample_w = "100k"
    seed = 0
    infer_scope = "subsample"
    out_anc = f"results/data/msl_vcf_{region}_n{n_subsample_w}_seed{seed}_{infer_scope}.txt"
    out_meta = f"results/data/msl_vcf_{region}_n{n_subsample_w}_seed{seed}_{infer_scope}_meta.json"
    out_posteriors = f"results/data/msl_vcf_{region}_n{n_subsample_w}_seed{seed}_{infer_scope}_posteriors.npz"


# Parse n_subsample (accepts ``full``, ``25000``, ``25k``, ``1M``, ``all``)
def _parse_count(s: str) -> int | None:
    s = s.strip().lower()
    if s in ("full", "all", "none", ""):
        return None
    suf = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    if s[-1] in suf:
        return int(float(s[:-1]) * suf[s[-1]])
    return int(s)


n_subsample = _parse_count(n_subsample_w)


# -------------------------------------------------------- load + subsample
t0 = time.perf_counter()
npz = np.load(in_npz, allow_pickle=True)
pos: np.ndarray = npz["pos"]
ref_idx: np.ndarray = npz["ref_idx"]
alt_idx: np.ndarray = npz["alt_idx"]
ingroup: np.ndarray = npz["ingroup"]
outgroup: np.ndarray = npz["outgroup"]
ingroup_names = list(npz["ingroup_names"].astype(str))
outgroup_names = list(npz["outgroup_names"].astype(str))
n_total = len(pos)
print(
    f"fit_infer_msl_vcf: loaded {in_npz}  "
    f"({n_total:,} polymorphic sites, {len(ingroup_names)} ingroup haps, "
    f"{len(outgroup_names)} outgroups, {time.perf_counter() - t0:.1f}s)",
    flush=True,
)


if n_subsample is not None and n_subsample < n_total:
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n_total, size=n_subsample, replace=False))
    pos = pos[keep]
    ref_idx = ref_idx[keep]
    alt_idx = alt_idx[keep]
    ingroup = ingroup[keep]
    outgroup = outgroup[keep]
    print(f"fit_infer_msl_vcf: subsampled to {len(pos):,} sites (seed {seed})", flush=True)
else:
    print(f"fit_infer_msl_vcf: using all {n_total:,} sites (no subsample)", flush=True)


# -------------------------------------------------------- Site builder
def build_sites(pos, ref_idx, alt_idx, ingroup, outgroup):
    """Materialise :class:`Site` records (ingroup + outgroup) from the arrays."""
    return sites_from_arrays(
        pos, ref_idx, alt_idx, ingroup, ingroup_names, outgroup, outgroup_names,
    )


t_build0 = time.perf_counter()
fit_sites = build_sites(pos, ref_idx, alt_idx, ingroup, outgroup)
t_build = time.perf_counter() - t_build0
print(f"fit_infer_msl_vcf: built {len(fit_sites):,} Site records in {t_build:.1f}s", flush=True)


# -------------------------------------------------------- BC + tree
t_bc0 = time.perf_counter()
bc = BaseComposition.from_polymorphic_sites(fit_sites)
kappa_hat = bc.kappa_estimate
print(
    f"fit_infer_msl_vcf: empirical pi={bc.pi.tolist()}, "
    f"n_ts={bc.n_ts}, n_tv={bc.n_tv}, kappa_hat={kappa_hat:.3f}  "
    f"({time.perf_counter() - t_bc0:.1f}s)",
    flush=True,
)

tree = OutgroupLadderTree(ingroup_names, outgroup_names)
prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
model = HKY(kappa=kappa_hat, fit_kappa=True)

# n_target_sites: use the region's positional span (a slight over-estimate
# of callable-mask length, but the rate fit is robust to this — only the
# polymorphic / monomorphic ratio matters).
with open(in_meta) as f:
    region_meta = json.load(f)
n_target_sites = region_meta["end"] - region_meta["start"]


# -------------------------------------------------------- fit
t_fit0 = time.perf_counter()
inference = FixedTreeInference(
    fit_sites, model, tree=tree,
    base_composition=bc,
    n_target_sites=n_target_sites,
    ingroup_weight=prior,
    n_starts=4,
    parallelize=False,
    progress=False,
)
params = inference.fit()
t_fit = time.perf_counter() - t_fit0
print(f"fit_infer_msl_vcf: FixedTreeInference.fit() took {t_fit:.1f}s; params={params}", flush=True)


# -------------------------------------------------------- infer + write
# If ``infer_scope=="all"`` and we subsampled above, swap ``self.sites``
# to the full array now so ``inference.infer()`` walks every polymorphic
# site under the already-fitted per-branch rates and prior. The fitted
# parameters stay fixed; ``infer()`` evaluates the prior lazily per batch
# from ``self.sites``, so swapping the site list is all that's needed.
if infer_scope == "all" and n_subsample is not None and n_subsample < n_total:
    t_full0 = time.perf_counter()
    npz_all = np.load(in_npz, allow_pickle=True)
    full_sites = build_sites(
        npz_all["pos"], npz_all["ref_idx"], npz_all["alt_idx"],
        npz_all["ingroup"], npz_all["outgroup"],
    )
    print(
        f"fit_infer_msl_vcf: built {len(full_sites):,} full-set Site records in {time.perf_counter() - t_full0:.1f}s; "
        f"applying fitted params for the posterior pass",
        flush=True,
    )
    inference.sites = full_sites
else:
    full_sites = fit_sites


t_infer0 = time.perf_counter()
Path(out_anc).parent.mkdir(parents=True, exist_ok=True)
n_written = n_skipped = 0
# Also accumulate the full per-state posterior for every called site so
# the 3-way report can compute soft agreement (e.g. mean P(reference's
# MAP allele)) in addition to hard MAP-vs-MAP rates.
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
        # Posterior.values is the length-4 array over the model's states
        # (A, C, G, T) — already normalised to sum to 1.
        post_values.append([float(v) for v in posterior.values])
        n_written += 1
t_infer = time.perf_counter() - t_infer0
print(
    f"fit_infer_msl_vcf: wrote {n_written:,} per-site calls to {out_anc} "
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
print(f"fit_infer_msl_vcf: wrote per-site posteriors to {out_posteriors}", flush=True)


# -------------------------------------------------------- meta
meta = {
    "region": region,
    "n_subsample": n_subsample_w,
    "n_subsample_resolved": n_subsample,
    "seed": seed,
    "infer_scope": infer_scope,
    "n_sites_polymorphic_total": int(n_total),
    "n_sites_in_fit": len(fit_sites),
    "n_sites_in_infer": len(full_sites),
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
    "empirical_pi": [float(x) for x in bc.pi],
    "n_ts": int(bc.n_ts),
    "n_tv": int(bc.n_tv),
    "outgroup_divergence_mle": (
        inference.outgroup_divergence_mle.tolist()
        if inference.outgroup_divergence_mle is not None else None
    ),
    "timings_seconds": {
        "load_npz": t_build0 - t0,
        "build_sites": t_build,
        "base_composition": time.perf_counter() - t_bc0 if False else None,  # captured above
        "fit": t_fit,
        "infer": t_infer,
        "total": time.perf_counter() - t0,
    },
    "in_npz": str(in_npz),
}
Path(out_meta).parent.mkdir(parents=True, exist_ok=True)
with open(out_meta, "w") as f:
    json.dump(meta, f, indent=2)
print(f"fit_infer_msl_vcf: wrote {out_meta} ({meta['timings_seconds']['total']:.1f}s total)", flush=True)
