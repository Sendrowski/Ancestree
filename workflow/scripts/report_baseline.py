"""Outgroup-effect benchmark report: posterior certainty vs n_outgroups.

Reads the per-(n_out) inference outputs and aggregates summary stats:

- ``mean_max_prob`` / ``median_max_prob`` — nominal confidence (posterior
  concentration).
- ``mean_entropy`` — alternative confidence measure, in bits.
- ``accuracy`` — fraction of sites whose MAP matches the simulator's
  truth ancestral state.
- ``mean_brier`` — mean Brier score of the per-site posterior against the
  one-hot simulator truth, lower is better.

Also stratifies by ingroup-SFS class, showing where outgroups help most (singletons, doubletons, ... carry less ingroup signal and benefit
more from outgroup likelihood).

Writes a JSON dump plus a markdown summary. Both committed under
``testing/benchmarks/reports/``.
"""
import json
import math
from pathlib import Path
from collections import defaultdict


try:
    inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    resimulated_json = snakemake.input.resimulated  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    inputs = sorted(
        p for p in Path("results/data").glob("baseline_n*.json")
        if "resimulated" not in p.name
    )
    resimulated_json = "results/data/baseline_resimulated.json"
    out_json = "results/reports/baseline.json"
    out_md = "results/reports/baseline.md"
    sim_config = {}


STATES = ("A", "C", "G", "T")


def _entropy_bits(probs: list[float]) -> float:
    """Shannon entropy in bits. Safe on zero-probability bins."""
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log2(p)
    return s


def _brier(site: dict) -> float:
    """Brier score of one site's posterior against the one-hot truth:
    ``sum_a (p_a - 1[a = truth])^2 = sum_a p_a^2 - 2 p_truth + 1``."""
    truth = site["truth"]
    if truth not in STATES:
        return float("nan")
    post = [float(v) for v in site["posterior"]]
    return sum(v * v for v in post) - 2.0 * post[STATES.index(truth)] + 1.0


cells: dict[int, dict] = {}
for path in inputs:
    with open(path) as f:
        payload = json.load(f)
    n_out = int(payload["inference_meta"]["n_out"])
    sites = payload["sites"]

    n_sites = len(sites)
    max_probs = [s["max_prob"] for s in sites.values()]
    entropies = [_entropy_bits(s["posterior"]) for s in sites.values()]
    with_truth = [s for s in sites.values() if s["truth"] is not None]
    accuracy = (
        sum(s["map_correct"] for s in with_truth) / len(with_truth)
        if with_truth else float("nan")
    )
    briers = [_brier(s) for s in with_truth]
    briers = [b for b in briers if not math.isnan(b)]
    mean_brier = sum(briers) / len(briers) if briers else float("nan")

    # B11: baseline-MAP agreement. Sites carrying a `baseline_map` field
    # come from inference scripts that ran MajorityOutgroupInference too;
    # for older / outgroup-less cells the field is absent and we report
    # n/a. Per-bin breakdown uses alleles-count as the coarse proxy
    # (consistent with the existing by_n_segregating split).
    with_baseline = [s for s in sites.values() if "baseline_map" in s]
    if with_baseline:
        n_baseline = len(with_baseline)
        n_baseline_agree = sum(
            int(s["map_allele"] == s["baseline_map"]) for s in with_baseline
        )
        baseline_agreement = n_baseline_agree / n_baseline if n_baseline else float("nan")
        # Per-bin agreement (by minor-allele count from the alleles field).
        # ARG JSON doesn't carry minor_count, so we group by len(alleles).
        per_n_alleles: defaultdict[int, list[dict]] = defaultdict(list)
        for s in with_baseline:
            per_n_alleles[len(s.get("alleles", []))].append(s)
        baseline_agreement_by_n_alleles = {
            k: sum(int(r["map_allele"] == r["baseline_map"]) for r in v) / len(v)
            for k, v in sorted(per_n_alleles.items())
        }
    else:
        n_baseline = 0
        baseline_agreement = float("nan")
        baseline_agreement_by_n_alleles = {}

    # Stratify by ingroup SFS class (minor allele count over the ingroup
    # tips). We don't carry per-tip ingroup data in the JSON, so reuse
    # the alleles[0] (REF) as a proxy major. For the report purposes
    # the trend across n_out at each class is what matters, not the
    # absolute class assignment.
    by_class: defaultdict[int, list[dict]] = defaultdict(list)
    for s in with_truth:
        # Minor-count proxy: posterior over candidate alleles already
        # encodes most info. We group by (number of segregating alleles)
        # for a coarse but well-defined stratification.
        by_class[len(s["alleles"])].append(s)
    class_accuracy = {
        k: {
            "n_sites": len(v),
            "accuracy": sum(s["map_correct"] for s in v) / len(v),
            "mean_brier": sum(_brier(s) for s in v) / len(v),
            "mean_max_prob": sum(s["max_prob"] for s in v) / len(v),
        }
        for k, v in sorted(by_class.items())
    }

    cells[n_out] = {
        "n_out": n_out,
        "n_sites": n_sites,
        "n_sites_with_truth": len(with_truth),
        "mean_max_prob": sum(max_probs) / n_sites if n_sites else float("nan"),
        "median_max_prob": sorted(max_probs)[n_sites // 2] if n_sites else float("nan"),
        "mean_entropy_bits": sum(entropies) / n_sites if n_sites else float("nan"),
        "accuracy": accuracy,
        "mean_brier": mean_brier,
        "by_n_segregating": class_accuracy,
        "outgroup_pops_used": payload["inference_meta"]["outgroup_pops_used"],
        "n_baseline_compared": n_baseline,
        "baseline_agreement": baseline_agreement,
        "baseline_agreement_by_n_alleles": baseline_agreement_by_n_alleles,
    }

out_payload = {"cells": cells, "sim_config": sim_config}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)

# ------------------------------- Markdown summary --------------------------

ordered_n_outs = sorted(cells.keys())
header = "# Outgroup-effect benchmark (ARGBasedInference)\n\n"
header += (
    "Posterior certainty and accuracy as a function of the number of\n"
    "outgroups available to the Felsenstein kernel. Same simulated ARG;\n"
    "each row simplifies the tree sequence to ingroup + first ``n_out``\n"
    "outgroups (closest-first) before inference. ``JC69 + uniform prior +\n"
    "full`` throughout.\n\n"
)

if sim_config:
    header += "**Simulator settings**\n\n"
    for k, v in sim_config.items():
        header += f"- ``{k}`` = {v}\n"
    header += "\n"

table = "## Summary\n\n"
table += (
    "| n_out | n_sites | mean max_prob | median max_prob "
    "| mean entropy (bits) | accuracy | mean Brier | baseline agreement |\n"
)
table += (
    "|------:|--------:|--------------:|----------------:"
    "|--------------------:|---------:|-----------:|-------------------:|\n"
)
for n in ordered_n_outs:
    c = cells[n]
    base = c.get("baseline_agreement")
    base_str = "n/a" if base is None or math.isnan(base) else f"{base:.4f}"
    table += (
        f"| {n} | {c['n_sites']} | {c['mean_max_prob']:.4f} | "
        f"{c['median_max_prob']:.4f} | {c['mean_entropy_bits']:.4f} | "
        f"{c['accuracy']:.4f} | {c['mean_brier']:.4f} | {base_str} |\n"
    )
table += "\n"

# B11: baseline-agreement closing paragraph.
baseline_para = (
    "## Baseline-MAP agreement (vs `MajorityOutgroupInference`)\n\n"
    "``baseline agreement`` is the fraction of sites at which Ancestree's\n"
    "MAP allele matches the simple outgroup-majority rule (the\n"
    ":class:`~ancestree.inference.MajorityOutgroupInference` baseline,\n"
    "deliberately ignorant of substitution model, branch lengths, and\n"
    "ingroup SFS class). High agreement = the polariser signal\n"
    "dominates and the substitution-model + prior layers are\n"
    "minimally biasing the call. Low agreement = the model is moving\n"
    "posteriors away from the bare outgroup vote, worth investigating\n"
    "whether the bias is correctly identifying recurrent / homoplastic\n"
    "sites or whether it reflects model misspecification.\n\n"
    "The ``n_out=0`` cell omits this column — the baseline rule needs\n"
    "at least one outgroup observation per site to vote.\n\n"
)

interp = (
    "## Reading the trend\n\n"
    "- **Adding outgroups *lowers* nominal confidence (mean max_prob)**\n"
    "  because the kernel sees more conflicting tip patterns: with zero\n"
    "  outgroups the local tree's root is the ingroup MRCA and the\n"
    "  ingroup-only topology often resolves a single MAP allele with\n"
    "  near-1.0 mass; outgroup tips inject deep-time variation that\n"
    "  the kernel correctly reflects as posterior uncertainty.\n"
    "- **Adding outgroups *raises* accuracy** because the inference\n"
    "  question — what allele was ancestral at the simulator's deep\n"
    "  ancestor — requires information beyond the ingroup MRCA. Without\n"
    "  outgroups the MAP allele is the *ingroup-MRCA* state, which is\n"
    "  only weakly correlated with the deep ancestor under finite Ne.\n"
    "- The combination — high confidence and low accuracy at n_out=0\n"
    "  shrinking toward calibrated confidence at larger n_out — is the\n"
    "  expected signature of identifiability improving as the kernel\n"
    "  gains a longer evolutionary baseline.\n\n"
)

per_class = "## Per-segregating-alleles breakdown\n\n"
per_class += "Stratified by the site's number of segregating alleles (after the source's "
per_class += "SNP filter; almost all sites are biallelic).\n\n"
per_class += "| n_out | n_segregating | n_sites | accuracy | mean Brier | mean max_prob |\n"
per_class += "|------:|--------------:|--------:|---------:|-----------:|--------------:|\n"
for n in ordered_n_outs:
    c = cells[n]
    for k, v in c["by_n_segregating"].items():
        per_class += (
            f"| {n} | {k} | {v['n_sites']} | "
            f"{v['accuracy']:.4f} | {v['mean_brier']:.4f} | "
            f"{v['mean_max_prob']:.4f} |\n"
        )
per_class += "\n"

# ------------------------------ Re-simulation variability ------------------

resim_table = ""
if Path(resimulated_json).exists():
    with open(resimulated_json) as f:
        resim = json.load(f)
    n_seeds = resim["config"]["seeds"]
    resim_table = (
        f"## Re-simulation variability ({len(n_seeds)} seeds, same demographic)\n\n"
        "Each row aggregates over independent ARG draws from the *coalescent\n"
        "prior* (different seed, same demographic). Per-site averaging\n"
        "isn't possible because re-simulated mutations land at different\n"
        "positions, so we report per-seed aggregate stats (mean ± std).\n"
        "This is an **upper bound** on what proper ARG-posterior sampling\n"
        "(ARGweaver / SINGER / Relate-MCMC) could shift these numbers,\n"
        "since the coalescent prior is *not* conditioned on observed\n"
        "mutations.\n\n"
        "| n_out | mean max_prob (mean ± std) | accuracy (mean ± std) | mean #sites |\n"
        "|------:|---------------------------:|----------------------:|------------:|\n"
    )
    for n in ordered_n_outs:
        s = resim["summary"][str(n)]
        resim_table += (
            f"| {n} | {s['mean_max_prob_mean']:.4f} ± {s['mean_max_prob_std']:.4f} | "
            f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} | "
            f"{int(s['n_sites_mean'])} |\n"
        )
    resim_table += "\n"

with open(out_md, "w") as f:
    f.write(header + table + interp + baseline_para + resim_table + per_class)
print(f"Wrote {out_md}", flush=True)
