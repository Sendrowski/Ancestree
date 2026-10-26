"""SLiM-outgroup-bias benchmark inference: per-chunk posterior pass using
parameters from a joint fit over ALL chunks.

Reads the recap+mutate stage's VCF + meta and a joint-fit JSON (produced
by :mod:`workflow.scripts.fit_slim_outgroup_bias`). Constructs
``FixedTreeInference`` with ``fit_required=False``, sets the joint MLE
through ``tree.set_params``, then runs the per-site posterior pass on a
single chunk's polymorphic-site stream.

Per-chunk MLEs on ~5 k polymorphic sites were over-fitting per-chunk
sampling noise (e.g. K_1 sliding to the 1e-9 lower bound on some chunks
even though the population-level back-mutation rate is non-zero). With
the joint fit anchoring the rates to population-level estimates, the
per-chunk inference layer is now a deterministic posterior-only pass —
``fit_seconds`` is effectively zero (the wall time was paid upstream
once per (f_del, n_out) cell).

Wildcards:

- ``{f_del}`` — fraction of sites under selection in the upstream SLiM
  sim (``0.0``, ``0.5``, ``1.0``).
- ``{chunk_idx}`` ∈ ``0..N_CHUNKS-1`` — which chunk's VCF + meta to consume.
- ``{n_out}`` ∈ ``{1, 2, 3}`` — number of outgroups exposed to the
  Felsenstein kernel (closest first).

Run directly::

    python workflow/scripts/infer_slim_outgroup_bias.py
"""
import json
import time
from pathlib import Path

from ancestree import (
    FixedTreeInference,
    JC69,
    KingmanIngroupWeight,
)


try:
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    in_fit = snakemake.input.fit  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    f_del = float(snakemake.wildcards.f_del)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    length = int(snakemake.params.length)  # type: ignore[name-defined]
except NameError:
    in_vcf = "results/data/slim_outgroup_bias_fdel0.0_chunk0.vcf"
    in_meta = "results/data/slim_outgroup_bias_fdel0.0_chunk0_meta.json"
    in_fit = "results/data/slim_outgroup_bias_fdel0.0_n1_fit.json"
    out_json = "results/data/slim_outgroup_bias_fdel0.0_chunk0_n1.json"
    f_del = 0.0
    n_out = 1
    length = 100_000


with open(in_meta, encoding="utf-8") as f:
    meta = json.load(f)
with open(in_fit, encoding="utf-8") as f:
    fit_payload = json.load(f)

ingroup_names: list[str] = list(meta["ingroup_names"])
all_outgroup_names: list[str] = list(meta["outgroup_names"])
if n_out > len(all_outgroup_names):
    raise ValueError(
        f"n_out={n_out} exceeds available outgroups "
        f"({len(all_outgroup_names)})"
    )
outgroup_names = all_outgroup_names[:n_out]
positions: list[int] = list(meta["positions"])  # 0-based positions
truth_states: list[str] = list(meta["truth_ancestral_states"])
truth_by_pos: dict[int, str] = {
    int(p) + 1: s for p, s in zip(positions, truth_states)  # VCF is 1-based
}

# Validate the upstream fit cell matches our (f_del, n_out).
if float(fit_payload["f_del"]) != f_del or int(fit_payload["n_out"]) != n_out:
    raise ValueError(
        f"Joint-fit JSON cell mismatch: fit has "
        f"(f_del={fit_payload['f_del']}, n_out={fit_payload['n_out']}), "
        f"infer expects (f_del={f_del}, n_out={n_out})."
    )
joint_params_mle: dict[str, float] = dict(fit_payload["params_mle"])
joint_fit_seconds = float(fit_payload.get("fit_seconds", 0.0))
joint_K_MLE = fit_payload.get("outgroup_divergence_mle")
joint_kappa_MLE = joint_params_mle.get("kappa")

print(
    f"SLiM-outgroup-bias inference (joint-fit-driven): f_del={f_del}, "
    f"chunk={meta.get('chunk_idx')}, n_out={n_out}, "
    f"ingroup={len(ingroup_names)} haps, outgroups={outgroup_names}, "
    f"joint K_MLE={joint_K_MLE}",
    flush=True,
)

prior = KingmanIngroupWeight(ingroup_samples=ingroup_names)
# JC69 stays — same kernel setup as the joint-fit script. The joint fit's
# kappa MLE (None under JC69) is ignored when reconstructing the model.
model = JC69()

# Construct in source mode with fit_required=False (no MLE fit), then
# overwrite the tree branch rates to the joint MLE. The order of
# tree.param_names matches the joint fit's construction (same
# ingroup/outgroup lists feed both scripts), so the joint params dict
# maps 1-to-1 onto tree.param_names. ``kappa`` (HKY) would be absent
# under JC69 and is not consumed below.
inference = FixedTreeInference(
    in_vcf,
    ingroup_samples=ingroup_names,
    outgroup_samples=outgroup_names,
    model=model,
    ingroup_weight=prior,
    n_target_sites=int(length),
    fit_required=False,
    parallelize=False,
    progress=False,
)
joint_vec = [float(joint_params_mle[name]) for name in inference.tree.param_names]
inference.tree.set_params(joint_vec)

# .fit() is a no-op here (fit_required=False). Per-site infer below uses
# the joint MLE rates loaded above.
fit_seconds = 0.0

per_site: list[dict] = []
_t0 = time.perf_counter()
for site, post in inference.infer():
    pos = int(site.pos)
    truth = truth_by_pos.get(pos)
    posterior_values = post.values.tolist()
    posterior_alleles = list(post.alleles)
    per_site.append({
        "pos": pos,
        "alleles": list(site.alleles),
        "map_allele": post.map_allele,
        "max_prob": float(post.max_prob),
        "posterior": posterior_values,
        "posterior_alleles": posterior_alleles,
        "truth": truth,
        "map_correct": truth is not None and post.map_allele == truth,
    })
infer_seconds = float(time.perf_counter() - _t0)

# Per-site ingroup major-count for downstream folded-bin reporting.
# Re-parse the VCF light-touch via CyVCF2Source so we get ingroup counts
# without re-running inference. Cheaper to recompute from posterior data —
# but the source stream is the canonical truth, so use it.
from ancestree import CyVCF2Source

src = CyVCF2Source(in_vcf, sample_filter=ingroup_names)
ingroup_haplotype_ids = src.samples()
pos_to_ing_counts: dict[int, dict[str, int]] = {}
for site in src:
    counts = site.count_alleles(ingroup_haplotype_ids)
    pos_to_ing_counts[int(site.pos)] = dict(counts)

n_ingroup = len(ingroup_names)
for entry in per_site:
    counts = pos_to_ing_counts.get(entry["pos"], {})
    if counts:
        major_count = max(counts.values())
    else:
        major_count = n_ingroup
    entry["ingroup_major_count"] = int(major_count)
    entry["ingroup_minor_count"] = int(n_ingroup - major_count)
    # Derived-count (relative to TRUTH): how many ingroup haps carry a
    # non-truth allele. The true uSFS bin = number of haps NOT equal to
    # the truth ancestral allele.
    truth = entry["truth"]
    if truth is not None and counts:
        entry["ingroup_derived_count"] = int(n_ingroup - counts.get(truth, 0))
    else:
        entry["ingroup_derived_count"] = None

result = {
    "f_del": f_del,
    "n_out": n_out,
    # The chunk_idx recap_and_mutate_slim_outgroup_bias wrote to the meta.
    "chunk_idx": meta.get("chunk_idx"),
    "ingroup_size": n_ingroup,
    "outgroup_names_used": outgroup_names,
    "params_mle": joint_params_mle,
    "outgroup_divergence_mle": joint_K_MLE,
    "log_likelihood_mle": float(fit_payload.get("log_likelihood_mle", 0.0)),
    # ``fit_seconds=0``: this script does not fit. The joint fit in the
    # upstream rule took ``joint_fit_seconds``. The downstream report reads
    # BOTH so per-chunk timing tables stay meaningful.
    "fit_seconds": fit_seconds,
    "joint_fit_seconds": joint_fit_seconds,
    "joint_fit_kappa_mle": joint_kappa_MLE,
    "joint_fit_K_mle": joint_K_MLE,
    "infer_seconds": infer_seconds,
    "n_sites": len(per_site),
    "per_site": per_site,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: {len(per_site)} sites, "
    f"fit=0.00s (joint fit took {joint_fit_seconds:.2f}s), "
    f"infer={infer_seconds:.2f}s",
    flush=True,
)
