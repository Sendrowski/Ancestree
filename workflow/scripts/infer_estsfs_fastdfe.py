"""B3 fastDFE-side inference: est-sfs annotation via fastDFE.

One cell per ``(model, prior)`` combo. Runs in its own ``envs/fastdfe.yml``
conda env (fastDFE 1.2.2, which supports numpy>=2) to isolate fastDFE's
dependency closure. Driven by the snakemake-object header below, with a
positional-argv / defaults fallback for standalone use.

Wildcards:

- ``{model}`` ∈ ``{JC, K2}``
- ``{prior}`` ∈ ``{kingman, adaptive}``

Run directly (standalone, defaults to JC+kingman)::

    python workflow/scripts/infer_estsfs_fastdfe.py

Or via snakemake::

    snakemake -j 1 results/data/estsfs_fastdfe_JC_kingman.json
"""
import json
import sys
from pathlib import Path


try:
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    in_fasta = snakemake.input.fasta  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    prior = snakemake.wildcards.prior  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    model_name = snakemake.wildcards.model  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
except NameError:
    if len(sys.argv) > 1:
        in_vcf, in_fasta, in_meta, out_json, prior, length_s, model_name, n_out_s = sys.argv[1:9]
        length = float(length_s)
        n_out = int(n_out_s)
    else:
        in_vcf = "results/data/estsfs_sim_jc.vcf"
        in_fasta = "results/data/estsfs_sim_jc.fasta"
        in_meta = "results/data/estsfs_sim_jc_meta.json"
        out_json = "results/data/estsfs_fastdfe_jc_JC_kingman_n3.json"
        prior = "kingman"
        length = 2e5
        model_name = "JC"
        n_out = 3


import cyvcf2
import fastdfe as fd

with open(in_meta) as f:
    meta = json.load(f)
# Subset outgroups closest-first (the simulator pre-orders).
outgroup_names = list(meta["outgroup_names"])[:n_out]
n_ingroup = int(meta["n_ingroup"])

if model_name == "JC":
    model = fd.JCSubstitutionModel()
elif model_name == "K2":
    model = fd.K2SubstitutionModel()
else:
    raise ValueError(f"Unknown model: {model_name!r}")

if prior == "kingman":
    # fastDFE's own API, which kept the pre-rename names. ancestree's
    # KingmanPolarizationPrior -> KingmanIngroupWeight rename was applied
    # across the tree and caught these fd.* calls, which are not ours.
    prior_obj = fd.KingmanPolarizationPrior()
elif prior == "adaptive":
    prior_obj = fd.AdaptivePolarizationPrior(parallelize=False)
else:
    raise ValueError(f"Unknown prior: {prior!r}")

ann = fd.MaximumLikelihoodAncestralAnnotation(
    outgroups=outgroup_names,
    n_ingroups=n_ingroup,
    model=model,
    prior=prior_obj,
    n_target_sites=int(length),
)
annotated_vcf = str(Path(out_json).with_suffix(".annotated.vcf"))
fd.Annotator(
    vcf=in_vcf, annotations=[ann], output=annotated_vcf, fasta=in_fasta,
).annotate()

results: list[dict] = []
_vcf_in = cyvcf2.VCF(annotated_vcf)
for variant in _vcf_in:
    try:
        aa = variant.INFO["AA"]
    except KeyError:
        continue
    try:
        aa_prob = float(variant.INFO["AA_prob"])
    except (KeyError, TypeError):
        aa_prob = float("nan")
    results.append({
        "chrom": variant.CHROM,
        "pos": int(variant.POS),
        "alleles": [variant.REF, *(variant.ALT or [])],
        "map_allele": aa,
        "max_prob": aa_prob,
    })

_vcf_in.close()
inference_meta = {
    "params_mle": {k: float(v) for k, v in (ann.params_mle or {}).items()},
    "outgroup_divergence": ann.get_outgroup_divergence().tolist()
        if ann.params_mle is not None else None,
    "outgroup_sample_names": outgroup_names,
    "n_target_sites_used": int(length),
    "model": model_name,
    "prior": prior,
    "n_out": n_out,
}
if prior == "adaptive":
    # fastDFE's p_polarization is a length-(n_ingroup+1) array of
    # P(major ancestral | bin i) for i in 0..n_ingroup. Convention:
    # i = number of MINOR alleles.
    p_pol = ann.p_polarization
    if p_pol is not None:
        inference_meta["adaptive_pi"] = {
            str(i): float(p_pol[i]) for i in range(len(p_pol))
        }

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
