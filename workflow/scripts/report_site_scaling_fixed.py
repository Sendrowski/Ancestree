"""B14: aggregate fixed-tree site-scaling cells → parameter-recovery summary.

Reads the per-(model, n_out, n_sites, seed) inference JSONs plus the
matching reference cells (n_sites = ``all``, no seed sweep) and emits a
single aggregated JSON the figure script reads. For each
(model, n_out, n_sites) the report computes per-parameter relative
error against the reference cell (the asymptotic fit on every
polymorphic site), averaged across seeds.

The aggregated JSON schema::

    {
        "summary": {
            (model, n_out, n_sites): {
                "K": {"K1": {"mean": ..., "std": ...}, ...},
                "kappa": {"mean": ..., "std": ...} | null,
                "abs_rel_err": {"mean": ..., "std": ...},
                "n_seeds": int,
            }
        },
        "reference": {(model, n_out): {"K": [...], "kappa": ... | None}},
        "sim_config": {...},
    }

Tuple keys are stringified as ``"<model>__n<n_out>__s<n_sites>"`` for
JSON friendliness.
"""
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path


try:
    cell_inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    ref_inputs = list(snakemake.input.reference)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    data_root = Path("results/data")
    cell_inputs = sorted(data_root.glob("site_scaling_fixed_*_seed*.json"))
    ref_inputs = sorted(data_root.glob("site_scaling_fixed_ref_*.json"))
    out_json = "results/reports/site_scaling_fixed.json"
    out_md = "results/reports/site_scaling_fixed.md"
    sim_config = {}


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return stats.mean(vals), stats.stdev(vals)


reference: dict[tuple[str, int], dict] = {}
for path in ref_inputs:
    with open(path) as f:
        d = json.load(f)
    key = (d["model"], int(d["n_out"]))
    reference[key] = {
        "K": list(d.get("outgroup_divergence") or []),
        "kappa": (d["params_mle"] or {}).get("kappa"),
        "log_likelihood_mle": d.get("log_likelihood_mle"),
        "n_sites_kept": int(d.get("n_sites_kept", 0)),
    }


cells_by_group: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
for path in cell_inputs:
    with open(path) as f:
        d = json.load(f)
    model = d["model"]
    n_out = int(d["n_out"])
    n_sites = int(d["n_sites_kept"])
    cells_by_group[(model, n_out, n_sites)].append(d)


summary: dict[str, dict] = {}
for (model, n_out, n_sites), cells in sorted(cells_by_group.items()):
    ref = reference.get((model, n_out))
    if ref is None:
        continue
    ref_K = ref["K"]
    ref_kappa = ref["kappa"]
    per_K: list[list[float]] = [[] for _ in range(len(ref_K))]
    kappa_vals: list[float] = []
    abs_rel_err: list[float] = []
    for c in cells:
        K = list(c.get("outgroup_divergence") or [])
        if len(K) != len(ref_K):
            continue
        for i, k_val in enumerate(K):
            per_K[i].append(k_val)
        cell_kappa = (c.get("params_mle") or {}).get("kappa")
        if ref_kappa is not None and cell_kappa is not None:
            kappa_vals.append(cell_kappa)
        rels: list[float] = []
        for i, k_val in enumerate(K):
            if ref_K[i] != 0:
                rels.append(abs(k_val - ref_K[i]) / ref_K[i])
        if ref_kappa is not None and cell_kappa is not None and ref_kappa != 0:
            rels.append(abs(cell_kappa - ref_kappa) / ref_kappa)
        if rels:
            abs_rel_err.append(sum(rels) / len(rels))
    K_summary = {}
    for i, vals in enumerate(per_K, start=1):
        m, s = _mean_std(vals)
        K_summary[f"K{i}"] = {
            "mean": m, "std": s,
            "ref": ref_K[i - 1] if i - 1 < len(ref_K) else None,
            "rel_err_mean": (
                abs(m - ref_K[i - 1]) / ref_K[i - 1]
                if i - 1 < len(ref_K) and ref_K[i - 1] != 0 else float("nan")
            ),
        }
    kappa_block = None
    if ref_kappa is not None:
        m, s = _mean_std(kappa_vals)
        kappa_block = {
            "mean": m, "std": s, "ref": ref_kappa,
            "rel_err_mean": (
                abs(m - ref_kappa) / ref_kappa if ref_kappa != 0 else float("nan")
            ),
        }
    are_mean, are_std = _mean_std(abs_rel_err)
    key = f"{model}__n{n_out}__s{n_sites}"
    summary[key] = {
        "model": model,
        "n_out": n_out,
        "n_sites": n_sites,
        "n_seeds": len(cells),
        "K": K_summary,
        "kappa": kappa_block,
        "abs_rel_err": {"mean": are_mean, "std": are_std},
    }


out_payload = {
    "summary": summary,
    "reference": {f"{m}__n{n}": v for (m, n), v in reference.items()},
    "sim_config": sim_config,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


lines = ["# Site-scaling benchmark (FixedTreeInference)\n"]
lines.append("Mean absolute relative error of fitted (K_i, kappa) vs. the full-data fit, averaged across seeds.\n")
lines.append("\n| model | n_out | n_sites | n_seeds | mean |K_fit - K_ref|/K_ref | std |\n")
lines.append("|:-|--:|--:|--:|--:|--:|\n")
for key in sorted(summary.keys()):
    s = summary[key]
    are = s["abs_rel_err"]
    lines.append(
        f"| {s['model']} | {s['n_out']} | {s['n_sites']} | {s['n_seeds']} | "
        f"{are['mean']:.4f} | {are['std']:.4f} |\n"
    )
lines.append("\n## Reference fits (n_sites = all)\n\n")
for key, ref in sorted(reference.items()):
    K_fmt = ", ".join(f"{v:.4e}" for v in ref["K"])
    kappa_str = f", kappa={ref['kappa']:.4f}" if ref["kappa"] is not None else ""
    lines.append(f"- {key[0]} n_out={key[1]} (n_sites_fit={ref['n_sites_kept']}): K=[{K_fmt}]{kappa_str}\n")

Path(out_md).parent.mkdir(parents=True, exist_ok=True)
with open(out_md, "w") as f:
    f.writelines(lines)
print(f"Wrote {out_json} + {out_md}: {len(summary)} cells")
