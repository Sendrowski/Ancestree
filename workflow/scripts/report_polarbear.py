"""B2 report: aggregate Ancestree + PolarBEAR per-site posteriors → report.

Joins the two per-site JSONs from :mod:`infer_polarbear_ancestree` and
:mod:`infer_polarbear_baseline` against msprime ground truth (computed inline
from the source ``.trees``) and writes the comparison report.

Run directly::

    python workflow/scripts/report_polarbear.py

Or via snakemake::

    snakemake -j 1 results/reports/polarbear_agreement.md
"""
import json

import numpy as np
import tskit

from ancestree import STATE_INDEX, STATES


try:
    in_ancestree = snakemake.input.ancestree  # type: ignore[name-defined]
    in_polarbear = snakemake.input.polarbear  # type: ignore[name-defined]
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    in_ancestree = "results/data/polarbear_ancestree.json"
    in_polarbear = "results/data/polarbear_baseline.json"
    in_trees = "results/data/polarbear_sim.trees"
    out_json = "results/reports/polarbear_agreement.json"
    out_md = "results/reports/polarbear_agreement.md"
    config = {
        "samples": 20, "length": 1e5, "mu": 1e-8,
        "rec_rate": 1e-8, "pop_size": 1e4, "seed": 42,
    }


def _load_sites_and_meta(path: str) -> tuple[dict[int, dict], dict]:
    """Load a polarbear-style cell JSON.

    Accepts the wrapped layout (``{"sites": {...}, "inference_meta":
    {...}}``) and the flat ``{pos: result}`` one.
    """
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "sites" in raw and isinstance(raw["sites"], dict):
        sites = {int(k): v for k, v in raw["sites"].items()}
        meta = raw.get("inference_meta") or {}
    else:
        sites = {int(k): v for k, v in raw.items()}
        meta = {}
    return sites, meta


anc, anc_meta = _load_sites_and_meta(in_ancestree)
pb, pb_meta = _load_sites_and_meta(in_polarbear)

# Compute per-site truth + parsimony score from the original ts.
ts = tskit.load(in_trees)
par_info: dict[int, dict] = {}
variants_by_site = {v.site.id: v for v in ts.variants()}
for tree in ts.trees():
    if tree.num_roots != 1:
        continue
    for ts_site in tree.sites():
        variant = variants_by_site[ts_site.id]
        alleles = list(variant.alleles)
        gens = variant.genotypes
        if any(g < 0 for g in gens) or any(alleles[g] not in STATE_INDEX for g in gens):
            par_info[int(ts_site.position)] = {"missing": True}
            continue
        states = [STATE_INDEX[alleles[g]] for g in gens]
        _anc, muts = tree.map_mutations(states, alleles=STATES)
        par_info[int(ts_site.position)] = {
            "truth": ts_site.ancestral_state,
            "parsimony_score": len(muts),
            "missing": False,
        }

shared_positions = sorted(set(pb) & set(anc))
print(
    f"B2 aggregate: ancestree={len(anc)} sites, polarbear={len(pb)} sites, "
    f"shared={len(shared_positions)}",
    flush=True,
)

# ----------------------------------------------------------------- compare
n_nonhom = n_hom = 0
nonhom_map_agree = hom_map_agree = 0
max_prob_diff_nonhom: list[float] = []
max_prob_diff_hom: list[float] = []
per_allele_diff_nonhom: list[float] = []
per_allele_diff_hom: list[float] = []
truth_anc = truth_pb = 0
n_pb_non_informative = 0
n_pb_multibranch = 0
n_pb_homoplasy = 0
pol_max_prob_on_pb_informative: list[float] = []
pol_max_prob_on_pb_non_informative: list[float] = []
brier_anc: list[float] = []
brier_pb: list[float] = []
examples_disagree: list[dict] = []

for pos in shared_positions:
    a, b = anc[pos], pb[pos]
    info = par_info.get(pos, {})
    if info.get("missing"):
        continue
    score = info.get("parsimony_score", 0)
    truth = info.get("truth")

    diff = abs(b["max_prob"] - a["max_prob"])
    map_agree = b["map_allele"] is not None and a["map_allele"] == b["map_allele"]

    if truth is not None:
        if a["map_allele"] == truth: truth_anc += 1
        if b["map_allele"] == truth: truth_pb += 1
        if truth in STATE_INDEX:
            pa = np.asarray(a["posterior"], dtype=float)
            y = np.zeros_like(pa)
            y[STATE_INDEX[truth]] = 1.0
            brier_anc.append(float(((pa - y) ** 2).sum()))
            if b.get("posterior_by_allele"):
                q = np.zeros_like(pa)
                for allele, prob in b["posterior_by_allele"].items():
                    if allele in STATE_INDEX:
                        q[STATE_INDEX[allele]] = float(prob)
                brier_pb.append(float(((q - y) ** 2).sum()))

    if b.get("posterior_by_allele"):
        diffs_here = [abs(p - a["posterior"][STATE_INDEX[allele]])
                      for allele, p in b["posterior_by_allele"].items()
                      if allele in STATE_INDEX]
        per_allele_max = max(diffs_here) if diffs_here else None
    else:
        per_allele_max = None

    if score <= 1:
        n_nonhom += 1
        max_prob_diff_nonhom.append(diff)
        if per_allele_max is not None:
            per_allele_diff_nonhom.append(per_allele_max)
        if map_agree:
            nonhom_map_agree += 1
        elif len(examples_disagree) < 5:
            examples_disagree.append({
                "pos": pos, "truth": truth, "parsimony_score": score,
                "polarbear": b,
                "ancestree": {k: v for k, v in a.items() if k != "posterior"},
                "ancestree_posterior": a["posterior"],
            })
    else:
        n_hom += 1
        max_prob_diff_hom.append(diff)
        if per_allele_max is not None:
            per_allele_diff_hom.append(per_allele_max)
        if map_agree:
            hom_map_agree += 1

    if b.get("polarbear_non_informative") == 1:
        n_pb_non_informative += 1
        pol_max_prob_on_pb_non_informative.append(float(a["max_prob"]))
    elif b.get("polarbear_non_informative") == 0:
        pol_max_prob_on_pb_informative.append(float(a["max_prob"]))
    if b.get("polarbear_multibranch") == 1:
        n_pb_multibranch += 1
    if (b.get("polarbear_mutation_count") or 0) > 1:
        n_pb_homoplasy += 1


def _mean(xs): return float(np.mean(xs)) if xs else None
def _max(xs): return float(np.max(xs)) if xs else None
def _div(n, d): return n / d if d else None


report = {
    "config": config,
    "n_sites_polarbear": len(pb),
    "n_sites_ancestree": len(anc),
    "n_shared": len(shared_positions),
    "n_nonhomoplastic": n_nonhom,
    "n_homoplastic": n_hom,
    "nonhom_map_agreement_rate": _div(nonhom_map_agree, n_nonhom),
    "hom_map_agreement_rate": _div(hom_map_agree, n_hom),
    "nonhom_max_prob_mean_abs_diff": _mean(max_prob_diff_nonhom),
    "nonhom_max_prob_max_abs_diff": _max(max_prob_diff_nonhom),
    "hom_max_prob_mean_abs_diff": _mean(max_prob_diff_hom),
    "hom_max_prob_max_abs_diff": _max(max_prob_diff_hom),
    "nonhom_per_allele_mean_abs_diff": _mean(per_allele_diff_nonhom),
    "nonhom_per_allele_max_abs_diff": _max(per_allele_diff_nonhom),
    "hom_per_allele_mean_abs_diff": _mean(per_allele_diff_hom),
    "hom_per_allele_max_abs_diff": _max(per_allele_diff_hom),
    "truth_recovery_ancestree": _div(truth_anc, len(shared_positions)),
    "truth_recovery_polarbear": _div(truth_pb, len(shared_positions)),
    "mean_brier_ancestree": _mean(brier_anc),
    "mean_brier_polarbear": _mean(brier_pb),
    "n_polarbear_non_informative": n_pb_non_informative,
    "n_polarbear_multibranch": n_pb_multibranch,
    "n_polarbear_homoplasy_flag": n_pb_homoplasy,
    "ancestree_mean_max_prob_on_pb_informative": _mean(pol_max_prob_on_pb_informative),
    "ancestree_mean_max_prob_on_pb_non_informative": _mean(pol_max_prob_on_pb_non_informative),
    "examples_disagreement_on_nonhomoplastic": examples_disagree,
    "inference_seconds_ancestree": anc_meta.get("inference_seconds"),
    "inference_seconds_polarbear": pb_meta.get("inference_seconds"),
}

with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {out_json}", flush=True)


def _fmt_rate(x): return "—" if x is None else f"{x:.4f} ({100*x:.2f}%)"
def _fmt_diff(x): return "—" if x is None else f"{x:.3e}"
def _fmt_meanprob(x): return "—" if x is None else f"{x:.4f}"
def _fmt_secs(x): return "—" if x is None else f"{x:.3f}"


cfg = report["config"]
md = f"""# B2 — Ancestree vs PolarBEAR agreement

## Simulation
- samples (haploid): `{cfg['samples']*2}` (msprime `samples={cfg['samples']}`, default ploidy 2)
- sequence length: `{cfg['length']:g}`
- mutation rate: `{cfg['mu']:g}`
- recombination rate: `{cfg['rec_rate']:g}`
- population size: `{cfg['pop_size']:g}`
- seed: `{cfg['seed']}`

## Counts
- PolarBEAR sites (polymorphic only): {report['n_sites_polarbear']}
- Ancestree sites: {report['n_sites_ancestree']}
- shared positions compared: {report['n_shared']}
- non-homoplastic (parsimony score ≤ 1): {report['n_nonhomoplastic']}
- homoplastic (parsimony score > 1): {report['n_homoplastic']}

## MAP-allele agreement
| Subset | Agreement rate |
|---|---|
| non-homoplastic | {_fmt_rate(report['nonhom_map_agreement_rate'])} |
| homoplastic     | {_fmt_rate(report['hom_map_agreement_rate'])} |

## Posterior numerical agreement
### `max_prob` scalar (max over states under each tool)
| Subset | n | mean abs diff | max abs diff |
|---|---|---|---|
| non-homoplastic | {report['n_nonhomoplastic']} | {_fmt_diff(report['nonhom_max_prob_mean_abs_diff'])} | {_fmt_diff(report['nonhom_max_prob_max_abs_diff'])} |
| homoplastic     | {report['n_homoplastic']} | {_fmt_diff(report['hom_max_prob_mean_abs_diff'])} | {_fmt_diff(report['hom_max_prob_max_abs_diff'])} |

### Per-allele posterior (worst-allele `|Δp|` per site, requires patched PolarBEAR output)
| Subset | n | mean abs diff | max abs diff |
|---|---|---|---|
| non-homoplastic | {report['n_nonhomoplastic']} | {_fmt_diff(report['nonhom_per_allele_mean_abs_diff'])} | {_fmt_diff(report['nonhom_per_allele_max_abs_diff'])} |
| homoplastic     | {report['n_homoplastic']} | {_fmt_diff(report['hom_per_allele_mean_abs_diff'])} | {_fmt_diff(report['hom_per_allele_max_abs_diff'])} |

## Truth recovery (vs msprime's `Site.ancestral_state`)
| Tool | accuracy |
|---|---|
| Ancestree | {_fmt_rate(report['truth_recovery_ancestree'])} |
| PolarBEAR | {_fmt_rate(report['truth_recovery_polarbear'])} |

## Mean Brier score against the simulated truth
| Tool | mean Brier |
|---|---|
| Ancestree | {_fmt_diff(report['mean_brier_ancestree'])} |
| PolarBEAR | {_fmt_diff(report['mean_brier_polarbear'])} |

## Inference wall-clock runtime (seconds, inference step only, excluding I/O)
| Tool | Runtime (s) |
|---|---|
| Ancestree | {_fmt_secs(report['inference_seconds_ancestree'])} |
| PolarBEAR | {_fmt_secs(report['inference_seconds_polarbear'])} |

## PolarBEAR uninformativeness flags (from `filter_tree.filter_tree`)
| Flag | Count | Fraction |
|---|---|---|
| non-informative (multiple alleles tie for min parsimony) | {report['n_polarbear_non_informative']} | {report['n_polarbear_non_informative']/report['n_shared']:.3f} |
| multibranch (mutation on polytomous node) | {report['n_polarbear_multibranch']} | {report['n_polarbear_multibranch']/report['n_shared']:.3f} |
| homoplasy (parsimony score > 1) | {report['n_polarbear_homoplasy_flag']} | {report['n_polarbear_homoplasy_flag']/report['n_shared']:.3f} |

### Ancestree's own confidence on the PolarBEAR-flagged non-informative subset
| Subset | mean `max_prob` (Ancestree) |
|---|---|
| PolarBEAR labels informative | {_fmt_meanprob(report['ancestree_mean_max_prob_on_pb_informative'])} |
| PolarBEAR labels non-informative | {_fmt_meanprob(report['ancestree_mean_max_prob_on_pb_non_informative'])} |
"""

with open(out_md, "w") as f:
    f.write(md)
print(f"Wrote {out_md}", flush=True)
