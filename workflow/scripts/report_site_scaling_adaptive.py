"""B15: aggregate adaptive-prior site-scaling cells → pi_i recovery summary.

For each (n_sub, n_sites) cell the report computes per-bin absolute
deviation |pi_fit_j - pi_kingman_j| averaged across seeds, plus a
headline mean over all fittable bins (skipping the symmetric centre bin
when n_sub is even, since it's pinned to 0.5 by construction).
"""
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


try:
    cell_inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    cell_inputs = sorted(Path("results/data").glob("site_scaling_adaptive_*_seed*.json"))
    out_json = "results/reports/site_scaling_adaptive.json"
    out_md = "results/reports/site_scaling_adaptive.md"
    sim_config = {}


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return stats.mean(vals), stats.stdev(vals)


cells_by_group: dict[tuple[int, int], list[dict]] = defaultdict(list)
for path in cell_inputs:
    with open(path) as f:
        d = json.load(f)
    n_sub = int(d["n_sub"])
    n_sites = int(d["n_sites_kept"])
    cells_by_group[(n_sub, n_sites)].append(d)


summary: dict[str, dict] = {}
for (n_sub, n_sites), cells in sorted(cells_by_group.items()):
    fittable_bins = [
        j for j in range(1, n_sub)
        if not (n_sub % 2 == 0 and j == n_sub // 2)
    ]
    per_bin_abs: dict[int, list[float]] = {j: [] for j in fittable_bins}
    mean_abs_err_per_cell: list[float] = []
    for c in cells:
        pi_fit = {int(k): float(v) for k, v in c["pi_fit"].items()}
        pi_truth = {int(k): float(v) for k, v in c["pi_kingman"].items()}
        per_cell_errs: list[float] = []
        for j in fittable_bins:
            if j in pi_fit and j in pi_truth:
                err = abs(pi_fit[j] - pi_truth[j])
                per_bin_abs[j].append(err)
                per_cell_errs.append(err)
        if per_cell_errs:
            mean_abs_err_per_cell.append(sum(per_cell_errs) / len(per_cell_errs))
    bin_summary = {}
    for j in fittable_bins:
        m, s = _mean_std(per_bin_abs[j])
        bin_summary[str(j)] = {
            "mean_abs_err": m, "std": s,
            "ref_kingman": (n_sub - j) / n_sub,
            "n_cells": len(per_bin_abs[j]),
        }
    headline_mean, headline_std = _mean_std(mean_abs_err_per_cell)
    key = f"sub{n_sub}__s{n_sites}"
    summary[key] = {
        "n_sub": n_sub,
        "n_sites": n_sites,
        "n_seeds": len(cells),
        "fittable_bins": fittable_bins,
        "per_bin": bin_summary,
        "mean_abs_err": {"mean": headline_mean, "std": headline_std},
    }


out_payload = {"summary": summary, "sim_config": sim_config}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


lines = ["# Site-scaling benchmark (AdaptiveIngroupWeight)\n\n"]
lines.append(
    "Mean absolute deviation |pi_fit - pi_kingman| over fittable bins, averaged across seeds.\n"
    "Kingman closed form is the asymptotic truth (manuscript §3.6).\n\n"
)
lines.append("| n_sub | n_sites | n_seeds | mean |pi - kingman| | std |\n")
lines.append("|--:|--:|--:|--:|--:|\n")
for key in sorted(summary.keys()):
    s = summary[key]
    are = s["mean_abs_err"]
    lines.append(
        f"| {s['n_sub']} | {s['n_sites']} | {s['n_seeds']} | "
        f"{are['mean']:.4f} | {are['std']:.4f} |\n"
    )

Path(out_md).parent.mkdir(parents=True, exist_ok=True)
with open(out_md, "w") as f:
    f.writelines(lines)
print(f"Wrote {out_json} + {out_md}: {len(summary)} cells")
