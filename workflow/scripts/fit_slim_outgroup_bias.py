"""SLiM-outgroup-bias benchmark: joint FixedTreeInference fit over ALL
chunk VCFs for one (f_del, n_out) cell.

Pools the polymorphic-site stream from every chunk VCF in the (f_del)
cell and runs a SINGLE ``FixedTreeInference.fit()`` over the concatenated
site list. Persists the fitted parameters (kappa MLE for HKY, K vector,
outgroup-divergence MLE, log-likelihood, wall-time, full param dict)
to a JSON that the downstream per-chunk infer rule loads via
``fixed_params=`` so each chunk's inference layer reuses the joint MLE.

Per-chunk MLEs on ~5 k polymorphic sites were over-fitting per-chunk
sampling noise (e.g. K_1 sliding to the 1e-9 lower bound on some chunks
even though the population-level back-mutation rate is non-zero). With
10× more data feeding a single fit, the joint MLE is anchored to the
population-level rates and the per-chunk inference layer becomes a
deterministic posterior-only pass.

Wildcards:

- ``{f_del}`` — fraction of sites under selection in the upstream SLiM
  sim (``0.0``, ``0.5``, ``1.0``).
- ``{n_out}`` ∈ ``{1, 2, 3}`` — number of outgroups exposed to the
  Felsenstein kernel (closest first).

Run directly::

    python workflow/scripts/fit_slim_outgroup_bias.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

from ancestree import (
    CyVCF2Source,
    FixedTreeInference,
    JC69,
    KingmanIngroupWeight,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI fallback when not driven by snakemake."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-vcfs", nargs="+", required=True)
    p.add_argument("--in-metas", nargs="+", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--f-del", type=float, required=True)
    p.add_argument("--n-out", type=int, required=True)
    p.add_argument("--length", type=int, required=True,
                   help="Cumulative target sequence length across chunks.")
    return p.parse_args(argv)


try:
    in_vcfs: list[str] = list(snakemake.input.vcfs)  # type: ignore[name-defined]
    in_metas: list[str] = list(snakemake.input.metas)  # type: ignore[name-defined]
    out_json = snakemake.output.fit  # type: ignore[name-defined]
    f_del = float(snakemake.wildcards.f_del)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    length = int(snakemake.params.length)  # type: ignore[name-defined]
except NameError:
    _args = _parse_args(sys.argv[1:])
    in_vcfs = list(_args.in_vcfs)
    in_metas = list(_args.in_metas)
    out_json = _args.out_json
    f_del = _args.f_del
    n_out = _args.n_out
    length = _args.length


# ------------------------------------------------------------------ metadata
# All chunks share the same ingroup_names + outgroup_names (the upstream
# recap+mutate stage uses the same sample-selection seed for every chunk),
# but verify it explicitly so we fail fast if a chunk drifted.
ingroup_names: list[str] | None = None
all_outgroup_names: list[str] | None = None
for meta_path in in_metas:
    with open(meta_path) as f:
        meta = json.load(f)
    ing = list(meta["ingroup_names"])
    out = list(meta["outgroup_names"])
    if ingroup_names is None:
        ingroup_names = ing
        all_outgroup_names = out
    else:
        if ing != ingroup_names or out != all_outgroup_names:
            raise ValueError(
                f"Inconsistent ingroup/outgroup names across chunks: "
                f"{meta_path} disagrees with the first chunk."
            )
assert ingroup_names is not None and all_outgroup_names is not None

if n_out > len(all_outgroup_names):
    raise ValueError(
        f"n_out={n_out} exceeds available outgroups "
        f"({len(all_outgroup_names)})"
    )
outgroup_names = all_outgroup_names[:n_out]

print(
    f"SLiM-outgroup-bias joint fit: f_del={f_del}, n_out={n_out}, "
    f"chunks={len(in_vcfs)}, ingroup={len(ingroup_names)} haps, "
    f"outgroups={outgroup_names}, n_target_sites={length}",
    flush=True,
)

# ---------------------------------------------------------------- pooled sites
# Each chunk VCF carries a disjoint position range. Concatenating their
# Site streams gives the full-length polymorphic-site list. The downstream
# FixedTreeInference's monomorphic-site calibration uses the cumulative
# n_target_sites = sum of per-chunk seq_len_per_chunk = the full sim length.
pooled_sites = []
for vcf_path in in_vcfs:
    src = CyVCF2Source(vcf_path)
    pooled_sites.extend(list(src))
print(f"Pooled {len(pooled_sites)} polymorphic sites across chunks", flush=True)

# Same kernel setup as the original per-chunk script: JC69 + Kingman prior
# + full recurrence + 20 random restarts.
prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
model = JC69()

inference = FixedTreeInference(
    pooled_sites,
    ingroup_samples=ingroup_names,
    outgroup_samples=outgroup_names,
    model=model,
    ingroup_weight=prior,
    n_target_sites=int(length),
    n_starts=20,
    parallelize=False,
    progress=False,
)
_t0 = time.perf_counter()
params = inference.fit()
fit_seconds = float(time.perf_counter() - _t0)

result = {
    "f_del": f_del,
    "n_out": n_out,
    "ingroup_size": len(ingroup_names),
    "ingroup_names": ingroup_names,
    "outgroup_names_used": outgroup_names,
    "n_chunks_pooled": len(in_vcfs),
    "n_sites_pooled": len(pooled_sites),
    "n_target_sites": int(length),
    "params_mle": params,
    "outgroup_divergence_mle": (
        inference.outgroup_divergence_mle.tolist()
        if inference.outgroup_divergence_mle is not None else None
    ),
    "log_likelihood_mle": float(inference.log_likelihood_mle),
    "fit_seconds": fit_seconds,
    "n_starts": 20,
    "model": "JC69",
    "recurrence": "full",
    "prior": "KingmanIngroupWeight",
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: n_sites={len(pooled_sites)}, "
    f"fit={fit_seconds:.2f}s, "
    f"K_MLE={result['outgroup_divergence_mle']}",
    flush=True,
)
