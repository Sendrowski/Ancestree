"""B12 missing-data sweep report: per-bin SFS accuracy vs missing_frac.

Reads the per-(missing_frac, seed) inference outputs and aggregates
SFS-stratified accuracy across seeds. Mirrors B7's structure: one row
per missing_frac, error bars over the 3 seeds, folded-SFS-bin columns.

Standalone use::

    python workflow/scripts/report_missing_data.py
"""
import json
import math
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report_common import _entropy_bits, _mean_std  # noqa: E402


try:
    inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    inputs = sorted(Path("results/data").glob("missing_data_f*_seed*.json"))
    out_json = "results/reports/missing_data.json"
    out_md = "results/reports/missing_data.md"
    sim_config = {}


# Per-cell aggregation: one entry per (missing_frac, seed).
per_cell: dict[tuple[float, int], dict] = {}
for path in inputs:
    with open(path) as f:
        payload = json.load(f)
    info = payload["inference_meta"]
    mf = float(info["missing_frac"])
    sd = int(info["seed"])
    sites = payload["sites"]
    n_ingroup = int(info["n_ingroup"])

    max_probs = [s["max_prob"] for s in sites.values()]
    entropies = [_entropy_bits(s["posterior"]) for s in sites.values()]
    with_truth = [s for s in sites.values() if s["truth"] is not None]
    n_correct = sum(s["map_correct"] for s in with_truth)
    accuracy = n_correct / len(with_truth) if with_truth else float("nan")

    # Per folded-SFS-bin accuracy (only ingroup-polymorphic sites with truth).
    # B12's per-site JSON carries `minor_count` over the ingroup. Folded
    # bin = minor_count directly (1 = singleton, 2 = doubleton, ...).
    by_bin: defaultdict[int, list[dict]] = defaultdict(list)
    for s in with_truth:
        mc = s.get("minor_count")
        if mc is None or mc < 1:
            continue
        by_bin[int(mc)].append(s)
    per_bin = {
        k: {
            "n_sites": len(v),
            "accuracy": sum(s["map_correct"] for s in v) / len(v),
            "mean_max_prob": sum(s["max_prob"] for s in v) / len(v),
        }
        for k, v in sorted(by_bin.items())
    }

    # Distribution of n_outgroup_observed across sites (sanity check
    # for the masking effect).
    n_obs_dist: defaultdict[int, int] = defaultdict(int)
    for s in sites.values():
        n_obs_dist[int(s.get("n_outgroup_observed", 0))] += 1

    per_cell[(mf, sd)] = {
        "missing_frac": mf,
        "seed": sd,
        "n_sites": len(sites),
        "n_sites_with_truth": len(with_truth),
        "mean_max_prob": sum(max_probs) / len(max_probs) if max_probs else float("nan"),
        "mean_entropy_bits": sum(entropies) / len(entropies) if entropies else float("nan"),
        "accuracy": accuracy,
        "empirical_mask_rate": float(info.get("empirical_mask_rate", 0.0)),
        "fitted_outgroup_divergence": info.get("fitted_outgroup_divergence"),
        "per_bin": per_bin,
        "n_outgroup_observed_dist": dict(n_obs_dist),
    }


# Aggregate across seeds, per missing_frac.
mfs: list[float] = sorted({mf for (mf, _) in per_cell.keys()})


summary: dict[float, dict] = {}
for mf in mfs:
    cells = [c for (k_mf, _), c in per_cell.items() if k_mf == mf]
    acc_mean, acc_std = _mean_std([c["accuracy"] for c in cells])
    mp_mean, mp_std = _mean_std([c["mean_max_prob"] for c in cells])
    ent_mean, ent_std = _mean_std([c["mean_entropy_bits"] for c in cells])
    n_sites_mean, _ = _mean_std([float(c["n_sites"]) for c in cells])
    emp_rate_mean, _ = _mean_std([c["empirical_mask_rate"] for c in cells])

    bin_acc: defaultdict[int, list[float]] = defaultdict(list)
    bin_mp: defaultdict[int, list[float]] = defaultdict(list)
    bin_n: defaultdict[int, list[int]] = defaultdict(list)
    for c in cells:
        for k, v in c["per_bin"].items():
            bin_acc[int(k)].append(v["accuracy"])
            bin_mp[int(k)].append(v["mean_max_prob"])
            bin_n[int(k)].append(v["n_sites"])
    per_bin_summary: dict[int, dict] = {}
    for k in sorted(bin_acc.keys()):
        a_m, a_s = _mean_std(bin_acc[k])
        p_m, p_s = _mean_std(bin_mp[k])
        per_bin_summary[k] = {
            "accuracy_mean": a_m, "accuracy_std": a_s,
            "mean_max_prob_mean": p_m, "mean_max_prob_std": p_s,
            "n_sites_mean": stats.mean(bin_n[k]) if bin_n[k] else 0,
        }

    summary[mf] = {
        "missing_frac": mf,
        "n_seeds": len(cells),
        "n_sites_mean": n_sites_mean,
        "empirical_mask_rate_mean": emp_rate_mean,
        "accuracy_mean": acc_mean, "accuracy_std": acc_std,
        "mean_max_prob_mean": mp_mean, "mean_max_prob_std": mp_std,
        "mean_entropy_bits_mean": ent_mean, "mean_entropy_bits_std": ent_std,
        "per_bin": per_bin_summary,
    }


# Inflection-point detection: the largest mf at which mean accuracy is
# still within `tolerance` of the baseline (mf=0). Used in the closing
# paragraph.
TOLERANCE = 0.01  # 1% absolute accuracy drop
inflection_mf: float | None = None
if mfs:
    base_acc = summary[mfs[0]]["accuracy_mean"]
    if not math.isnan(base_acc):
        last_stable = mfs[0]
        for mf in mfs:
            a = summary[mf]["accuracy_mean"]
            if math.isnan(a):
                continue
            if base_acc - a <= TOLERANCE:
                last_stable = mf
        inflection_mf = last_stable


out_payload = {
    "summary": summary,
    "per_cell": {
        f"f{mf}_seed{sd}": v for (mf, sd), v in per_cell.items()
    },
    "sim_config": sim_config,
    "tolerance": TOLERANCE,
    "inflection_missing_frac": inflection_mf,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


# ------------------------------- Markdown summary --------------------------

header = "# Missing-data robustness sweep (B12)\n\n"
header += (
    "Per-bin SFS accuracy as a function of outgroup-tip missingness.\n"
    "Single simulated ARG (B6-style demography: ingroup of 20 +\n"
    "3 outgroups at 3e5/9e5/1.5e6 gens); each cell randomly nulls out\n"
    "``missing_frac`` of the outgroup tip slots per site (independent\n"
    "Bernoulli per (site, outgroup)) with a per-cell RNG seed, then runs\n"
    "``FixedTreeInference`` (HKY + empirical π, Kingman prior,\n"
    "full). 3 seeds per ``missing_frac`` for error bars.\n\n"
    "**Folded SFS bin** = ingroup minor-allele count. Each row reports\n"
    "the per-bin accuracy mean ± std across the 3 seeds.\n\n"
)

if sim_config:
    header += "**Simulator settings**\n\n"
    for k, v in sim_config.items():
        header += f"- ``{k}`` = {v}\n"
    header += "\n"

# Overall summary table.
overall_table = "## Overall summary (averaged over 3 seeds)\n\n"
overall_table += (
    "| missing_frac | n_seeds | n_sites | empirical mask rate "
    "| accuracy (mean ± std) | mean max_prob (mean ± std) "
    "| mean entropy bits |\n"
)
overall_table += (
    "|-------------:|--------:|--------:|--------------------:"
    "|----------------------:|---------------------------:"
    "|------------------:|\n"
)
for mf in mfs:
    s = summary[mf]
    overall_table += (
        f"| {mf:.2f} | {s['n_seeds']} | {int(s['n_sites_mean'])} | "
        f"{s['empirical_mask_rate_mean']:.3f} | "
        f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} | "
        f"{s['mean_max_prob_mean']:.4f} ± {s['mean_max_prob_std']:.4f} | "
        f"{s['mean_entropy_bits_mean']:.4f} |\n"
    )
overall_table += "\n"

# Per-bin SFS accuracy table.
all_bins = sorted({
    b for s in summary.values() for b in s["per_bin"].keys()
})
bin_table = "## Per-bin SFS accuracy (mean ± std over 3 seeds)\n\n"
bin_table += "Rows: missing_frac. Columns: folded-SFS bin (minor-allele count).\n\n"
bin_table += "| missing_frac | " + " | ".join(f"mc={b}" for b in all_bins) + " |\n"
bin_table += "|-------------:|" + "|".join(["----:"] * len(all_bins)) + "|\n"
for mf in mfs:
    s = summary[mf]
    row = f"| {mf:.2f} |"
    for b in all_bins:
        per_bin = s["per_bin"]
        if b in per_bin:
            v = per_bin[b]
            row += f" {v['accuracy_mean']:.3f} ± {v['accuracy_std']:.3f} |"
        else:
            row += " -- |"
    bin_table += row + "\n"
bin_table += "\n"

# Site-count companion (B7 style — spot bins with tiny n).
count_table = "## Per-bin site counts (mean over seeds)\n\n"
count_table += (
    "Same layout as the accuracy table; helps spot bins with few sites "
    "where the accuracy estimate is noisy.\n\n"
)
count_table += "| missing_frac | " + " | ".join(f"mc={b}" for b in all_bins) + " |\n"
count_table += "|-------------:|" + "|".join(["----:"] * len(all_bins)) + "|\n"
for mf in mfs:
    s = summary[mf]
    row = f"| {mf:.2f} |"
    for b in all_bins:
        per_bin = s["per_bin"]
        if b in per_bin:
            row += f" {int(per_bin[b]['n_sites_mean'])} |"
        else:
            row += " -- |"
    count_table += row + "\n"
count_table += "\n"


# Closing paragraph — name the inflection point.
infl_text = ""
if inflection_mf is not None:
    if inflection_mf == mfs[-1]:
        infl_text = (
            f"accuracy is essentially flat across the full sweep "
            f"(within ±{TOLERANCE:.0%} of the no-masking baseline up to "
            f"missing_frac={inflection_mf:.0%}). HKY + Kingman prior + 3 "
            "outgroups is robust to heavy outgroup-tip dropout on this "
            "demography — the prior and the surviving outgroups carry "
            "enough signal to keep MAP recovery stable."
        )
    elif inflection_mf == mfs[0]:
        infl_text = (
            f"accuracy degrades immediately past missing_frac=0 — there "
            f"is no safe missing fraction on this demography under HKY + "
            "Kingman. Every dropped outgroup tip is costing accuracy."
        )
    else:
        next_step = next((m for m in mfs if m > inflection_mf), None)
        infl_text = (
            f"accuracy is flat up to ``missing_frac={inflection_mf:.0%}`` "
            f"(within ±{TOLERANCE:.0%} of the no-masking baseline) and "
            f"degrades sharply past that"
        )
        if next_step is not None:
            d = (
                summary[mfs[0]]["accuracy_mean"]
                - summary[next_step]["accuracy_mean"]
            )
            infl_text += (
                f" — by ``missing_frac={next_step:.0%}`` the accuracy "
                f"drop is {d:.4f} absolute"
            )
        infl_text += "."

interp = (
    "## Reading the trend\n\n"
    f"{infl_text}\n\n"
    "- Outgroup-tip masking erodes the polariser signal by removing\n"
    "  evidence about the deep-ancestor state, which is exactly the\n"
    "  information ``OutgroupLadderTree`` is built to exploit.\n"
    "- The Kingman prior partially absorbs the loss at high-minor-count\n"
    "  bins where the ingroup-SFS class is informative; the worst-hit\n"
    "  bins should be the singletons / doubletons where the prior is\n"
    "  near-symmetric and the outgroup vote was the deciding signal.\n"
    "- The fully-masked endpoint (``missing_frac=1.0``) would degenerate\n"
    "  to the no-outgroup mode (posterior = Kingman prior over the\n"
    "  ingroup) — not in this sweep, but it's the asymptote each cell\n"
    "  is moving toward as more outgroups get nulled out.\n\n"
)

with open(out_md, "w") as f:
    f.write(header + overall_table + bin_table + count_table + interp)
print(f"Wrote {out_md}", flush=True)
