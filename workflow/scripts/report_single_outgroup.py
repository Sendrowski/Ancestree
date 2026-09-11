"""Single-outgroup depth sweep report: scoring across the outgroup depth grid.

Aggregates the ARG-mode and fixed-tree (VCF) inference cells, one pair per
outgroup split time, into a per-cell table scored on ingroup-polymorphic
sites only. Scoring matches the B8 report: mean Brier over the four
nucleotide states, plus MAP accuracy and the number of scored sites.

The report is standalone (not in ``rule all``). The snakemake rule is
``report_single_outgroup``.
"""
import json
import math
import re
from pathlib import Path


try:
    arg_inputs = list(snakemake.input.arg_cells)  # type: ignore[name-defined]
    vcf_inputs = list(snakemake.input.vcf_cells)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
    depths = list(snakemake.params.depths)  # type: ignore[name-defined]
except NameError:
    arg_inputs = sorted(Path("results/data").glob("single_outgroup_arg_d*.json"))
    vcf_inputs = sorted(Path("results/data").glob("single_outgroup_vcf_d*.json"))
    out_json = "results/reports/single_outgroup_depth.json"
    out_md = "results/reports/single_outgroup_depth.md"
    sim_config = {}
    depths = sorted({
        m.group(1) for p in arg_inputs + vcf_inputs
        if (m := re.search(r"_d([^_/]+)\.json$", str(p)))
    }, key=float)


#: Nucleotide order the per-site posteriors are stored in.
_STATES = "ACGT"


def _summarise_sites(per_site: list[dict]) -> dict:
    """Score a list of per-site records, restricted to ingroup-polymorphic sites.

    :param per_site: Per-site dicts carrying ``map_correct``, ``max_prob``,
        ``ingroup_polymorphic``, ``truth`` and the four-state ``posterior``.
    :return: Dict with the number of scored sites, the mean Brier score
        (lower is better), the MAP accuracy (higher is better) and the mean
        posterior mass on the MAP allele.
    """
    polymorphic = [s for s in per_site if s.get("ingroup_polymorphic")]
    n = len(polymorphic)
    if not n:
        return {
            "n_sites_polymorphic": 0,
            "brier_polymorphic": float("nan"),
            "accuracy_polymorphic": float("nan"),
            "mean_max_prob_polymorphic": float("nan"),
        }
    accuracy = sum(int(s["map_correct"]) for s in polymorphic) / n
    mean_max_prob = sum(s["max_prob"] for s in polymorphic) / n
    # Brier over the four nucleotide states:
    # (1 - p_true)^2 + sum_{i != true} p_i^2.
    scored = [s for s in polymorphic
              if s.get("posterior") and _STATES.find(str(s.get("truth", ""))) >= 0]
    if scored:
        total = 0.0
        for site in scored:
            post = site["posterior"]
            ti = _STATES.find(str(site["truth"]))
            total += sum(x * x for x in post) - 2.0 * post[ti] + 1.0
        brier = total / len(scored)
    else:
        brier = float("nan")
    return {
        "n_sites_polymorphic": n,
        "brier_polymorphic": brier,
        "accuracy_polymorphic": accuracy,
        "mean_max_prob_polymorphic": mean_max_prob,
    }


def _load_arg_cell(path: Path) -> dict:
    """Load one ARG-mode per-site JSON and score it.

    :param path: Path to a ``single_outgroup_arg_d{depth}.json`` file.
    :return: Flat cell dict keyed for the report tables.
    """
    with open(path) as f:
        payload = json.load(f)
    inference_meta = payload["inference_meta"]
    per_site = [
        {
            "map_correct": bool(s.get("map_correct", False)),
            "max_prob": float(s["max_prob"]),
            "ingroup_polymorphic": bool(s.get("ingroup_polymorphic", True)),
            "posterior": s.get("posterior"),
            "truth": s.get("truth"),
        }
        for s in payload["sites"].values()
    ]
    return {
        "depth_label": inference_meta["depth_label"],
        "depth": inference_meta["depth"],
        "outgroup_split_time": float(inference_meta["outgroup_split_times"][0]),
        "n_out": int(inference_meta["n_out"]),
        "n_sites_total": len(per_site),
        "mode": "arg",
        **_summarise_sites(per_site),
    }


def _load_vcf_cell(path: Path) -> dict:
    """Load one fixed-tree (VCF) per-site JSON and score it.

    :param path: Path to a ``single_outgroup_vcf_d{depth}.json`` file.
    :return: Flat cell dict keyed for the report tables.
    """
    with open(path) as f:
        payload = json.load(f)
    per_site = [
        {
            "map_correct": bool(s["map_correct"]),
            "max_prob": float(s["max_prob"]),
            "ingroup_polymorphic": bool(s.get("ingroup_polymorphic", True)),
            "posterior": s.get("posterior"),
            "truth": s.get("truth"),
        }
        for s in payload["per_site"]
    ]
    return {
        "depth_label": payload["depth_label"],
        "depth": payload["depth"],
        "outgroup_split_time": float(payload["outgroup_split_times"][0]),
        "n_out": int(payload["n_out"]),
        "n_sites_total": payload["n_sites"],
        "mode": "vcf",
        "fitted_outgroup_divergence": payload.get("fitted_outgroup_divergence"),
        **_summarise_sites(per_site),
    }


arg_cells: dict[str, dict] = {}
for p in arg_inputs:
    cell = _load_arg_cell(Path(p))
    arg_cells[cell["depth_label"]] = cell

vcf_cells: dict[str, dict] = {}
for p in vcf_inputs:
    cell = _load_vcf_cell(Path(p))
    vcf_cells[cell["depth_label"]] = cell

depths_ordered = sorted(
    set(arg_cells) | set(vcf_cells) | set(depths), key=float,
)

payload = {
    "sim_config": sim_config,
    "depths": depths_ordered,
    "arg_cells": [arg_cells[d] for d in depths_ordered if d in arg_cells],
    "vcf_cells": [vcf_cells[d] for d in depths_ordered if d in vcf_cells],
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)


def _fmt(x: float) -> str:
    """Four-decimal float, or ``n/a`` for a NaN.

    :param x: Value to render.
    :return: Formatted string.
    """
    return "n/a" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"


Ne = float(sim_config.get("pop_size", 3e4))

lines: list[str] = []
lines.append("# Single-outgroup depth sweep\n\n")
lines.append(
    "Ancestral-allele scoring for an ingroup plus exactly one outgroup, as a\n"
    "function of that outgroup's split time. Depth is reported both in\n"
    "generations and in coalescent units ``tau = T / (2 Ne)``, where ``T`` is\n"
    "the split time in generations and ``Ne`` the diploid effective\n"
    "population size of the ingroup. Two inference modes per depth:\n"
    "ARG-based (full tree sequence) and fixed-tree (VCF,\n"
    "``FixedTreeInference`` with a Kingman ingroup weight). Scoring is over\n"
    "ingroup-polymorphic sites only. The mean Brier score is over the four\n"
    "nucleotide states, so lower is better; MAP accuracy is a fraction of\n"
    "the scored sites, so higher is better.\n\n"
)

if sim_config:
    lines.append("**Simulator settings (shared)**\n\n")
    for k, v in sim_config.items():
        lines.append(f"- ``{k}`` = {v}\n")
    lines.append("\n")


def _table(cells: dict[str, dict], mode: str) -> str:
    """Render one mode's per-depth table.

    :param cells: Cells keyed by depth label.
    :param mode: ``"arg"`` or ``"vcf"``, used in the heading.
    :return: Markdown table, empty if the mode has no cells.
    """
    if not cells:
        return ""
    out = f"## {mode.upper()} mode — score by outgroup depth\n\n"
    out += (
        "| depth (generations) | tau = T / (2 Ne) | n_poly | mean Brier "
        "(lower better) | MAP accuracy (higher better) | mean max_prob |\n"
    )
    out += (
        "|--------------------:|-----------------:|-------:|"
        "------------------------:|-----------------------------:"
        "|--------------:|\n"
    )
    for d in depths_ordered:
        cell = cells.get(d)
        if cell is None:
            continue
        tau = cell["outgroup_split_time"] / (2 * Ne)
        out += (
            f"| {d} | {tau:.4g} | {cell['n_sites_polymorphic']} | "
            f"{_fmt(cell['brier_polymorphic'])} | "
            f"{_fmt(cell['accuracy_polymorphic'])} | "
            f"{_fmt(cell['mean_max_prob_polymorphic'])} |\n"
        )
    out += "\n"
    return out


lines.append(_table(arg_cells, "arg"))
lines.append(_table(vcf_cells, "vcf"))


def _best(cells: dict[str, dict], mode: str) -> str:
    """One-line statement of the lowest-Brier depth for one mode.

    :param cells: Cells keyed by depth label.
    :param mode: ``"arg"`` or ``"vcf"``.
    :return: Markdown paragraph, empty if no cell carries a finite score.
    """
    scored = [c for c in cells.values() if not math.isnan(c["brier_polymorphic"])]
    if not scored:
        return ""
    best = min(scored, key=lambda c: c["brier_polymorphic"])
    tau = best["outgroup_split_time"] / (2 * Ne)
    return (
        f"**{mode.upper()}**: lowest mean Brier at depth "
        f"{best['depth_label']} generations (tau = {tau:.4g}), "
        f"{_fmt(best['brier_polymorphic'])} over "
        f"{best['n_sites_polymorphic']} ingroup-polymorphic sites.\n\n"
    )


lines.append("## Take-aways\n\n")
lines.append(_best(arg_cells, "arg"))
lines.append(_best(vcf_cells, "vcf"))

with open(out_md, "w") as f:
    f.writelines(lines)

print(
    f"Wrote {out_md} and {out_json} "
    f"({len(arg_cells)} ARG cells + {len(vcf_cells)} VCF cells)",
    flush=True,
)
