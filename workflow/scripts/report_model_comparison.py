"""B9 model-comparison benchmark report: per-cell accuracy + per-SFS-bin breakdown.

Aggregates the 10 ``model_comparison_{model}_{mode}.json`` cells produced
by :mod:`infer_model_comparison` (5 substitution models × 2 inference
modes) into:

- A 5×2 headline accuracy table over **ingroup-polymorphic** sites
  (matches B8's reporting convention — ingroup-monomorphic sites carry
  no orienting signal, so including them dilutes the comparison).
- A per-(model, mode, minor-allele-count) breakdown, showing which SFS
  bins take most of the misspecification cost. Low-MAC
  bins (singletons, doubletons) are the SFS classes ancestral-allele
  errors hit the hardest in downstream DFE work, so they're the
  diagnostic bins for "did the model bias matter here".
- A short Markdown narrative stating what each substitution model
  assumes about base frequencies and about transition/transversion
  rates, and where the model enters each of the two inference modes.

Standalone fallback path (when run as
``python workflow/scripts/report_model_comparison.py``) globs the data
directory for whatever ``model_comparison_*_*.json`` cells exist —
useful for partial / dev re-runs without rebuilding the full matrix.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import msprime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msprime_substitution_units import substitutions_per_mutation  # noqa: E402


try:
    inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    inputs = sorted(
        p for p in Path("results/data").glob("model_comparison_*_*.json")
        # Exclude the sim-meta JSON. Only per-cell inference outputs go here.
        if not p.name.endswith("_meta.json")
    )
    in_meta = "results/data/model_comparison_sim_meta.json"
    out_json = "results/reports/model_comparison.json"
    out_md = "results/reports/model_comparison.md"
    sim_config = {}


MODEL_ORDER = ["jc69", "k2", "f81", "hky", "gtr"]
MODE_ORDER = ["arg", "vcf"]


def _entropy_bits(probs: list[float]) -> float:
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log2(p)
    return s


# {(model, mode): cell_summary}
cells: dict[tuple[str, str], dict] = {}
for path in inputs:
    with open(path) as f:
        payload = json.load(f)
    info = payload["inference_meta"]
    sites = payload["sites"]
    model_name = info["model"]
    mode = info["mode"]

    # Stratify on ingroup-polymorphic sites. Ingroup-monomorphic sites
    # carry no SFS bin, so they don't enter the per-bin breakdown and
    # only enter the "overall" line as a separate denominator.
    poly_sites = [s for s in sites.values() if s.get("ingroup_polymorphic")]
    poly_with_truth = [s for s in poly_sites if s.get("truth") is not None]
    all_with_truth = [s for s in sites.values() if s.get("truth") is not None]

    def _summary(records: list[dict]) -> dict:
        n = len(records)
        if not n:
            return {
                "n_sites": 0,
                "accuracy": float("nan"),
                "mean_max_prob": float("nan"),
                "mean_entropy_bits": float("nan"),
                "baseline_agreement": float("nan"),
                "n_baseline_compared": 0,
            }
        # B11: baseline_map carried per-site by infer scripts that wired
        # MajorityOutgroupInference in. Absent wherever a cell has no
        # outgroups, since the baseline needs one.
        with_baseline = [r for r in records if "baseline_map" in r and "map_allele" in r]
        baseline_agreement = (
            sum(int(r["map_allele"] == r["baseline_map"]) for r in with_baseline)
            / len(with_baseline)
        ) if with_baseline else float("nan")
        return {
            "n_sites": n,
            "accuracy": sum(r["map_correct"] for r in records) / n,
            "mean_max_prob": sum(r["max_prob"] for r in records) / n,
            "mean_entropy_bits": sum(_entropy_bits(r["posterior"]) for r in records) / n,
            "baseline_agreement": baseline_agreement,
            "n_baseline_compared": len(with_baseline),
        }

    # Per-minor-allele-count breakdown (ingroup-polymorphic sites only).
    by_minor: defaultdict[int, list[dict]] = defaultdict(list)
    for s in poly_with_truth:
        by_minor[int(s.get("minor_count", 0))].append(s)
    by_minor_summary = {
        int(k): _summary(v) for k, v in sorted(by_minor.items())
    }

    cells[(model_name, mode)] = {
        "model": model_name,
        "mode": mode,
        "empirical_params": info.get("empirical_params"),
        "fitted_kappa": info.get("fitted_kappa"),
        "fitted_outgroup_divergence": info.get("fitted_outgroup_divergence"),
        "sim_kappa": info.get("sim_kappa"),
        "sim_pi": info.get("sim_pi"),
        "all": _summary(all_with_truth),
        "ingroup_polymorphic": _summary(poly_with_truth),
        "by_minor_count": by_minor_summary,
    }


# ---------- Carry sim-truth into the report so the markdown can label it
sim_meta: dict = {}
if Path(in_meta).exists():
    with open(in_meta) as f:
        sim_meta = json.load(f)

out_payload = {
    "cells": {f"{m}_{md}": v for (m, md), v in cells.items()},
    "sim_config": sim_config,
    "sim_meta": {
        k: sim_meta.get(k)
        for k in ("kappa", "equilibrium_frequencies", "mutation_model",
                  "n_ingroup", "length", "mu", "seed", "outgroup_split_times")
    },
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


# ------------------------------- Markdown summary --------------------------

sim_kappa = sim_meta.get("kappa")
sim_pi = sim_meta.get("equilibrium_frequencies")

header = "# Substitution-model comparison (B9)\n\n"
header += (
    "Per-cell ancestral-allele recovery as a function of the Ancestree\n"
    "substitution model. The simulator mutates the ARG under ``msprime.HKY``\n"
    "with both a transition/transversion bias and a non-uniform AT-rich\n"
    "equilibrium distribution — every misspecified model is wrong along at\n"
    "least one axis the simulator cares about.\n\n"
    "All accuracy numbers below are on **ingroup-polymorphic sites only**\n"
    "(matches B8's convention): ingroup-monomorphic sites carry no\n"
    "polarisation signal and would dilute the headline.\n\n"
)

if sim_kappa is not None and sim_pi is not None:
    header += (
        f"**Simulator mutation model:** ``HKY(kappa={sim_kappa}, "
        f"pi={tuple(round(x, 3) for x in sim_pi)})`` over ``(A, C, G, T)``.\n\n"
    )
if sim_config:
    header += "**Simulator settings**\n\n"
    for k, v in sim_config.items():
        if k in ("models", "modes"):
            continue
        header += f"- ``{k}`` = {v}\n"
    header += "\n"


# Overall (ingroup-polymorphic) accuracy table, rows = models, cols = modes.
table = "## Accuracy on ingroup-polymorphic sites\n\n"
table += "| model | "
table += " | ".join(f"acc ({mode})" for mode in MODE_ORDER)
table += " | "
table += " | ".join(f"max_prob ({mode})" for mode in MODE_ORDER)
table += " | "
table += " | ".join(f"baseline agree ({mode})" for mode in MODE_ORDER)
table += " | n_sites |\n"
table += "|:---|"
table += "".join("---:|" for _ in MODE_ORDER) * 3
table += "---:|\n"
for m in MODEL_ORDER:
    if not any((m, md) in cells for md in MODE_ORDER):
        continue
    accs = []
    mps = []
    bases = []
    n_ref = None
    for md in MODE_ORDER:
        c = cells.get((m, md))
        if c is None:
            accs.append("—")
            mps.append("—")
            bases.append("—")
            continue
        s = c["ingroup_polymorphic"]
        accs.append(f"{s['accuracy']:.4f}" if not math.isnan(s["accuracy"]) else "—")
        mps.append(f"{s['mean_max_prob']:.4f}" if not math.isnan(s["mean_max_prob"]) else "—")
        b = s.get("baseline_agreement", float("nan"))
        bases.append(f"{b:.4f}" if not math.isnan(b) else "n/a")
        n_ref = s["n_sites"]
    table += (
        f"| {m} | " + " | ".join(accs) + " | "
        + " | ".join(mps) + " | " + " | ".join(bases)
        + f" | {n_ref if n_ref is not None else '—'} |\n"
    )
table += "\n"

baseline_para = (
    "**Baseline-MAP agreement** is the fraction of sites where Ancestree's\n"
    "MAP allele matches the simple outgroup-majority rule\n"
    "(:class:`~ancestree.inference.MajorityOutgroupInference`). On a well-\n"
    "specified model these should be high — the polariser signal dominates\n"
    "and the model + prior only nudge marginal calls. A model that drives\n"
    "agreement noticeably *lower* than the truth-matching ``hky`` row is\n"
    "shifting calls away from the bare outgroup vote; the value of that\n"
    "shift depends on whether it's correctly flagging recurrence /\n"
    "homoplasy or reflecting model misspecification.\n\n"
)


# ---------- Truth-vs-fitted outgroup divergences (subs/site).
#
# ``fitted_outgroup_divergence`` is the genetic divergence (in subs/site)
# between the ingroup and each outgroup, i.e. the sum of the two branches
# descending from their most-recent common ancestor. For outgroup ``k`` that
# ancestor is the split at time ``T_k`` (generations), along which each
# lineage accrues ``mu * T_k`` mutation events, of which a fraction ``rho``
# change the state, so the pairwise divergence is ``2 * mu * T_k * rho``
# substitutions per site — the units the fitted branch rates carry.

_split_times = sim_config.get("outgroup_split_times")
_mu_sim = sim_config.get("mu")
_kappa_sim_cfg = sim_config.get("kappa")
_pi_sim_cfg = sim_config.get("equilibrium_frequencies")
_rho_sim = None
_truth_divergence = None
if (
    _split_times is not None and _mu_sim is not None
    and _kappa_sim_cfg is not None and _pi_sim_cfg is not None
    and len(_split_times) >= 3
):
    T1, T2, T3 = (float(x) for x in _split_times[:3])
    mu = float(_mu_sim)
    _rho_sim = substitutions_per_mutation(msprime.HKY(
        kappa=float(_kappa_sim_cfg),
        equilibrium_frequencies=[float(x) for x in _pi_sim_cfg],
    ))
    _truth_divergence = (
        2 * T1 * mu * _rho_sim,  # I → O_1
        2 * T2 * mu * _rho_sim,  # I → O_2
        2 * T3 * mu * _rho_sim,  # I → O_3
    )


truth_table = ""
if _truth_divergence is not None:
    truth_table = "## Fitted vs truth outgroup divergences (subs/site)\n\n"
    truth_table += (
        "Truth is the ingroup-to-outgroup genetic divergence, the sum of the\n"
        "two branches from their split: ``2 * mu * T_k * rho`` subs/site,\n"
        "where ``rho`` is the expected number of substitutions per mutation\n"
        "event under the simulating model. ``msprime`` scales its transition\n"
        "matrix by the largest row sum, so a mutation event leaves the state\n"
        "unchanged with probability ``P_ii`` and ``rho <= 1``. For our sim\n"
        f"``(T_1, T_2, T_3) = ({T1:.3g}, {T2:.3g}, {T3:.3g})`` gen,\n"
        f"``μ = {mu:.3g}`` per gen and ``rho = {_rho_sim:.4f}`` → divergences\n"
        f"``[{_truth_divergence[0]:.4f}, {_truth_divergence[1]:.4f}, "
        f"{_truth_divergence[2]:.4f}]`` subs/site.\n\n"
    )
    truth_table += (
        "| model | mode | div₁ (truth=" f"{_truth_divergence[0]:.4f}) | "
        "div₂ (truth=" f"{_truth_divergence[1]:.4f}) | "
        "div₃ (truth=" f"{_truth_divergence[2]:.4f}) |\n"
    )
    truth_table += "|:---|:---|---:|---:|---:|\n"
    for m in MODEL_ORDER:
        for md in MODE_ORDER:
            c = cells.get((m, md))
            if c is None:
                continue
            div = c.get("fitted_outgroup_divergence")
            if div is None:
                continue
            cells_strs = []
            for i, d in enumerate(div):
                ratio = d / _truth_divergence[i]
                cells_strs.append(f"{d:.4f} ({ratio:.2f}×)")
            truth_table += f"| {m} | {md} | " + " | ".join(cells_strs) + " |\n"
    truth_table += "\n"


# Empirical / fitted-parameter snapshot per cell.
params_table = "## Model parameters per cell\n\n"
params_table += "| model | mode | empirical params | fitted κ (post-MLE) |\n"
params_table += "|:---|:---|:---|---:|\n"
for m in MODEL_ORDER:
    for md in MODE_ORDER:
        c = cells.get((m, md))
        if c is None:
            continue
        emp = c.get("empirical_params") or {}
        emp_str = ", ".join(
            f"{k}={(v if isinstance(v, (int, float)) else [round(x, 3) for x in v])}"
            for k, v in emp.items()
        ) or "—"
        fk = c.get("fitted_kappa")
        fk_str = f"{fk:.4f}" if isinstance(fk, (int, float)) else "—"
        params_table += f"| {m} | {md} | {emp_str} | {fk_str} |\n"
params_table += "\n"


# Per-minor-allele-count (SFS) breakdown — bin x (model, mode) wide layout
# would explode beyond markdown's comfort zone, so emit a long form table.
mac_table = "## Per-minor-allele-count breakdown (ingroup-polymorphic)\n\n"
mac_table += "| model | mode | minor_count | n_sites | accuracy | mean max_prob |\n"
mac_table += "|:---|:---|---:|---:|---:|---:|\n"
for m in MODEL_ORDER:
    for md in MODE_ORDER:
        c = cells.get((m, md))
        if c is None:
            continue
        for k, v in sorted(c["by_minor_count"].items()):
            mac_table += (
                f"| {m} | {md} | {k} | {v['n_sites']} | "
                f"{v['accuracy']:.4f} | {v['mean_max_prob']:.4f} |\n"
            )
mac_table += "\n"


interp = (
    "## Reading the result\n\n"
    "- JC69 assumes uniform equilibrium frequencies and one exchangeability\n"
    "  for every substitution type.\n"
    "- K2 assumes uniform equilibrium frequencies and separate transition\n"
    "  and transversion exchangeabilities.\n"
    "- F81 assumes the empirical equilibrium frequencies π̂ and one\n"
    "  exchangeability for every substitution type, so the rate into a\n"
    "  state is proportional to that state's frequency.\n"
    "- HKY assumes π̂ and separate transition and transversion\n"
    "  exchangeabilities, the parameterisation the simulator mutates under.\n"
    "- GTR assumes π̂ and one free exchangeability for each of the six\n"
    "  reversible substitution types.\n"
    "- ARG mode vs fixed-tree (VCF) mode: ARG inference reads the local\n"
    "  trees directly and uses the substitution model only to translate\n"
    "  branch lengths into per-branch transition probabilities. Fixed-tree\n"
    "  mode fits per-branch rates along the outgroup ladder under the same\n"
    "  model, so the model enters both the Stage 1 branch-rate MLE and the\n"
    "  per-site posterior.\n"
    "- Neither per-bin accuracy nor branch-rate recovery differentiates\n"
    "  the models on this setup. The three-outgroup signal dominates the\n"
    "  per-site posterior, so accuracy is flat across the grid. Every model\n"
    "  scales ``Q`` to one expected substitution per unit branch length, so\n"
    "  the fitted outgroup divergences are in the same units whatever ``π``\n"
    "  and ``κ`` the model assumes, and they agree with each other and with\n"
    "  truth. See the fitted-vs-truth table above.\n\n"
)

with open(out_md, "w") as f:
    f.write(header + table + baseline_para + params_table + truth_table + mac_table + interp)
print(f"Wrote {out_md}", flush=True)
