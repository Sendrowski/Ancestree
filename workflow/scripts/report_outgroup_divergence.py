"""B8 outgroup-divergence sweep report: per-bin SFS accuracy across (depth, spacing, n_out).

Aggregates the inference cells (one per sim × outgroup count ×
{ARG, fixed-tree, local-tree} mode) into a per-cell + per-bin accuracy table, plus
an overall ranking on ingroup-polymorphic sites only. The take-away the
user is after is a recommendation for outgroup placement: how deep the
deepest outgroup should be, how the outgroups should be spaced along the
ladder, and how many of them are needed.

The report is standalone (not in ``rule all``). The snakemake rule is
``report_outgroup_divergence``.
"""
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

# Same script-dir prepend trick as the simulate / infer scripts: lets
# the underscore-prefixed common helper import from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _outgroup_divergence_common import (  # noqa: E402
    OUTGROUP_COUNTS,
    SPACING_STRATEGIES,
    compute_split_times,
)


try:
    arg_inputs = list(snakemake.input.arg_cells)  # type: ignore[name-defined]
    vcf_inputs = list(snakemake.input.vcf_cells)  # type: ignore[name-defined]
    local_tree_inputs = list(snakemake.input.local_tree_cells)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
    depths = list(snakemake.params.depths)  # type: ignore[name-defined]
    spacings = list(snakemake.params.spacings)  # type: ignore[name-defined]
    n_outs = [int(n) for n in snakemake.params.n_outs]  # type: ignore[name-defined]
except NameError:
    arg_inputs = sorted(
        Path("results/data").glob("outgroup_divergence_arg_d*_s*_n*.json"))
    vcf_inputs = sorted(
        Path("results/data").glob("outgroup_divergence_vcf_d*_s*_n*.json"))
    local_tree_inputs = sorted(
        Path("results/data").glob("outgroup_divergence_local_tree_d*_s*_n*.json"))
    out_json = "results/reports/outgroup_divergence.json"
    out_md = "results/reports/outgroup_divergence.md"
    sim_config = {}
    depths = sorted({
        re.search(r"_d([^_]+)_s", str(p)).group(1)
        for p in arg_inputs + vcf_inputs + local_tree_inputs
        if re.search(r"_d([^_]+)_s", str(p))
    }, key=float)
    spacings = list(SPACING_STRATEGIES)
    n_outs = list(OUTGROUP_COUNTS)


# ----------------------------------------------------- SFS-bin helpers


#: Nucleotide order the per-site posteriors are stored in.
_STATES = "ACGT"


def _summarise_sites(per_site: list[dict], n_ingroup: int) -> dict:
    """Compute overall + per-SFS-bin accuracy stats on a list of per-site
    records. Filters to ``ingroup_polymorphic`` sites (matches B6).

    :param per_site: List of dicts each carrying at minimum
        ``map_correct``, ``max_prob``, ``ingroup_polymorphic``, and
        optionally ``minor_count`` (precomputed minor-allele count over
        the ingroup tips) and ``baseline_map`` / ``map_allele`` for the
        B11 baseline-agreement column.
    :param n_ingroup: Number of ingroup haplotypes (defines the SFS axis).
    """
    polymorphic = [s for s in per_site if s.get("ingroup_polymorphic")]
    n = len(polymorphic)
    if not n:
        return {
            "n_sites_polymorphic": 0,
            "brier_polymorphic": float("nan"),
            "accuracy_polymorphic": float("nan"),
            "mean_max_prob_polymorphic": float("nan"),
            "baseline_agreement_polymorphic": float("nan"),
            "n_baseline_compared": 0,
            "by_minor_count": {},
        }
    accuracy = sum(int(s["map_correct"]) for s in polymorphic) / n
    mean_max_prob = sum(s["max_prob"] for s in polymorphic) / n
    # B11: baseline-MAP agreement. `baseline_map` is the
    # MajorityOutgroupInference MAP allele, absent wherever a cell has no
    # outgroups, since the baseline needs one.
    with_baseline = [
        s for s in polymorphic
        if "baseline_map" in s and "map_allele" in s
    ]
    if with_baseline:
        baseline_agreement = sum(
            int(s["map_allele"] == s["baseline_map"]) for s in with_baseline
        ) / len(with_baseline)
    else:
        baseline_agreement = float("nan")
    by_bin: defaultdict[int, list[dict]] = defaultdict(list)
    for s in polymorphic:
        mc = s.get("minor_count")
        if mc is None or mc < 1:
            continue
        by_bin[int(mc)].append(s)
    by_minor_count = {}
    for k, v in sorted(by_bin.items()):
        with_b = [s for s in v if "baseline_map" in s and "map_allele" in s]
        b_rate = (
            sum(int(s["map_allele"] == s["baseline_map"]) for s in with_b)
            / len(with_b)
        ) if with_b else float("nan")
        by_minor_count[str(k)] = {
            "n_sites": len(v),
            "accuracy": sum(int(s["map_correct"]) for s in v) / len(v),
            "mean_max_prob": sum(s["max_prob"] for s in v) / len(v),
            "baseline_agreement": b_rate,
            "n_baseline_compared": len(with_b),
        }
    # Brier over the four nucleotide states, the score the rest of the
    # benchmarks report: (1 - p_true)^2 + sum_{i != true} p_i^2.
    scored = [s for s in polymorphic
              if s.get("posterior") and _STATES.find(str(s.get("truth", ""))) >= 0]
    if scored:
        total = 0.0
        for site in scored:
            post = site["posterior"]
            # The posterior is over the four nucleotides in ACGT order, not
            # over the site's own allele list, which holds only what is seen.
            ti = _STATES.find(str(site.get("truth", "")))
            if ti < 0:
                continue
            total += sum(x * x for x in post) - 2.0 * post[ti] + 1.0
        brier = total / len(scored)
    else:
        brier = float("nan")
    return {
        "n_sites_polymorphic": n,
        "brier_polymorphic": brier,
        "accuracy_polymorphic": accuracy,
        "mean_max_prob_polymorphic": mean_max_prob,
        "baseline_agreement_polymorphic": baseline_agreement,
        "n_baseline_compared": len(with_baseline),
        "by_minor_count": by_minor_count,
    }


# --------------------------------------- ARG / local-tree cell loader


def _load_sites_cell(path: Path) -> dict:
    """Load one tree-based (ARG or local-tree) per-site JSON, derive a flat list of per-site
    records compatible with :func:`_summarise_sites`, and run the
    summary. The ARG JSON carries ``alleles`` and ``posterior`` per site,
    but not per-tip alleles, so the minor-allele count is approximated from
    the posterior+truth and the ingroup polymorphism flag.

    Since the ARG inference does not emit per-tip ingroup genotypes, per-bin SFS
    stratification is skipped for the ARG cells, which carry only the overall
    accuracy plus the ingroup-polymorphic flag.
    """
    with open(path) as f:
        payload = json.load(f)
    sites = payload["sites"]
    inference_meta = payload["inference_meta"]
    per_site: list[dict] = []
    for pos, s in sites.items():
        per_site.append({
            "pos": int(pos),
            "map_correct": bool(s.get("map_correct", False)),
            "max_prob": float(s["max_prob"]),
            "ingroup_polymorphic": bool(s.get("ingroup_polymorphic", True)),
            # No minor_count in ARG JSON → per-bin will be empty for ARG.
            "minor_count": s.get("minor_count"),
            # B11: baseline_map carried through for the agreement column.
            "map_allele": s.get("map_allele"),
            # Carried for the Brier score, which needs the whole posterior.
            "posterior": s.get("posterior"),
            "alleles": s.get("alleles"),
            "truth": s.get("truth"),
            **({"baseline_map": s["baseline_map"]} if "baseline_map" in s else {}),
        })
    summary = _summarise_sites(per_site, n_ingroup=inference_meta["n_ingroup"])
    return {
        "depth_label": inference_meta["depth_label"],
        "depth": inference_meta["depth"],
        "spacing": inference_meta["spacing"],
        "n_out": int(inference_meta["n_out"]),
        "outgroup_split_times": inference_meta["outgroup_split_times"],
        "outgroup_names_used": inference_meta.get("outgroup_names_used"),
        "n_sites_total": len(per_site),
        "mode": inference_meta.get("mode", "arg"),
        **summary,
    }


# ---------------------------------------------------- VCF cell loader


def _load_vcf_cell(path: Path) -> dict:
    """Load one VCF-mode per-site JSON. The B8 VCF script carries
    ``ingroup_polymorphic`` per site. The minor-allele count recovered by
    counting unique ingroup tip alleles is *not* in the per-site dump, so the
    SFS bin is approximated from the (truth, map_allele) alone, which is only
    good enough for an overall report.
    """
    with open(path) as f:
        payload = json.load(f)
    per_site_in = payload["per_site"]
    per_site: list[dict] = []
    for s in per_site_in:
        per_site.append({
            "pos": s["pos"],
            "map_correct": bool(s["map_correct"]),
            "max_prob": float(s["max_prob"]),
            "p_true": float(s["p_true"]),
            "ingroup_polymorphic": bool(s.get("ingroup_polymorphic", True)),
            "minor_count": s.get("minor_count"),
            # B11: baseline_map carried through for the agreement column.
            "map_allele": s.get("map_allele"),
            # Carried for the Brier score, which needs the whole posterior.
            "posterior": s.get("posterior"),
            "alleles": s.get("alleles"),
            "truth": s.get("truth"),
            **({"baseline_map": s["baseline_map"]} if "baseline_map" in s else {}),
        })
    summary = _summarise_sites(per_site, n_ingroup=0)
    return {
        "depth_label": payload["depth_label"],
        "depth": payload["depth"],
        "spacing": payload["spacing"],
        "n_out": int(payload["n_out"]),
        "outgroup_split_times": payload.get("outgroup_split_times"),
        "outgroup_names_used": payload.get("outgroup_names_used"),
        "n_sites_total": payload["n_sites"],
        "mode": "vcf",
        "fitted_outgroup_divergence": payload.get("fitted_outgroup_divergence"),
        **summary,
    }


# ---------------------------------------------------- aggregate cells


arg_cells: dict[tuple[str, str, int], dict] = {}
for p in arg_inputs:
    cell = _load_sites_cell(Path(p))
    arg_cells[(cell["depth_label"], cell["spacing"], cell["n_out"])] = cell

vcf_cells: dict[tuple[str, str, int], dict] = {}
for p in vcf_inputs:
    cell = _load_vcf_cell(Path(p))
    vcf_cells[(cell["depth_label"], cell["spacing"], cell["n_out"])] = cell

local_tree_cells: dict[tuple[str, str, int], dict] = {}
for p in local_tree_inputs:
    cell = _load_sites_cell(Path(p))
    local_tree_cells[(cell["depth_label"], cell["spacing"], cell["n_out"])] = cell

_all_keys = (list(arg_cells.keys()) + list(vcf_cells.keys())
             + list(local_tree_cells.keys()))

# Sort canonical (depth, spacing, n_out) order from the params.
depths_ordered = sorted({k[0] for k in _all_keys}, key=lambda s: float(s))
spacings_ordered = list(SPACING_STRATEGIES)
n_outs_ordered = sorted({k[2] for k in _all_keys}) or list(n_outs)


# ---------------------------------------------------- ranking helpers


def _rank_cells(cells: dict[tuple[str, str, int], dict]) -> list[dict]:
    """Return cells sorted by overall ingroup-polymorphic accuracy
    descending. NaN accuracies sink to the bottom.
    """
    def _key(cell: dict) -> float:
        acc = cell["accuracy_polymorphic"]
        return -acc if not math.isnan(acc) else float("inf")

    return sorted(cells.values(), key=_key)


arg_ranked = _rank_cells(arg_cells)
vcf_ranked = _rank_cells(vcf_cells)
local_tree_ranked = _rank_cells(local_tree_cells)


# ----------------------------------------------------------- best picks


def _best_pick(ranked: list[dict]) -> dict | None:
    """First (highest-accuracy) cell, or ``None`` if the list is empty."""
    return ranked[0] if ranked else None


arg_best = _best_pick(arg_ranked)
vcf_best = _best_pick(vcf_ranked)
local_tree_best = _best_pick(local_tree_ranked)


# ------------------------------------------------------------- write JSON


payload = {
    "sim_config": sim_config,
    "depths": depths_ordered,
    "spacings": spacings_ordered,
    "n_outs": n_outs_ordered,
    "arg_cells": [
        cell for (d, sp, no) in [
            (d, sp, no) for d in depths_ordered for sp in spacings_ordered
            for no in n_outs_ordered
        ]
        if (cell := arg_cells.get((d, sp, no))) is not None
    ],
    "vcf_cells": [
        cell for (d, sp, no) in [
            (d, sp, no) for d in depths_ordered for sp in spacings_ordered
            for no in n_outs_ordered
        ]
        if (cell := vcf_cells.get((d, sp, no))) is not None
    ],
    "local_tree_cells": [
        cell for (d, sp, no) in [
            (d, sp, no) for d in depths_ordered for sp in spacings_ordered
            for no in n_outs_ordered
        ]
        if (cell := local_tree_cells.get((d, sp, no))) is not None
    ],
    "arg_ranking": arg_ranked,
    "vcf_ranking": vcf_ranked,
    "local_tree_ranking": local_tree_ranked,
    "arg_best": arg_best,
    "vcf_best": vcf_best,
    "local_tree_best": local_tree_best,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)


# -------------------------------------------------------- write Markdown


def _fmt_acc(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.4f}"


def _fmt_splits(splits: list[float]) -> str:
    return "[" + ", ".join(f"{t:.3g}" for t in splits) + "]"


lines: list[str] = []
lines.append("# B8: Outgroup-divergence sweep\n\n")
lines.append(
    "Per-cell ancestral-allele accuracy as a function of the deepest\n"
    "outgroup's divergence depth, the relative spacing of the three\n"
    "outgroups on the ladder, and how many of those outgroups the\n"
    "inference sees. Same demographic skeleton as B6 (ingroup + three\n"
    "outgroups) — only the three outgroup-split times move, and the\n"
    "``n_out`` axis subsets the same simulated tree sequence: 3 keeps all\n"
    "three outgroups, 2 keeps the shallowest and the deepest, 1 keeps the\n"
    "middle one. Three inference modes per cell: ARG-based (full ts),\n"
    "fixed-tree (VCF, ``FixedTreeInference`` + Kingman prior) and\n"
    "local-tree (``LocalTreeInference``, genealogies inferred from the\n"
    "genotypes alone). Accuracy is over **ingroup-polymorphic sites\n"
    "only**, matching B6's convention.\n\n"
)

if sim_config:
    lines.append("**Simulator settings (shared)**\n\n")
    for k, v in sim_config.items():
        lines.append(f"- ``{k}`` = {v}\n")
    lines.append("\n")

lines.append("**Spacing strategies (deepest split = ``d``)**\n\n")
example = 1.5e6
for sp in spacings_ordered:
    times = compute_split_times(example, sp)
    fractions = [t / example for t in times]
    frac_str = ", ".join(f"{f:.3g}·d" for f in fractions)
    lines.append(f"- ``{sp}``: ({frac_str})\n")
lines.append("\n")


def _table(cells_by_key: dict[tuple[str, str, int], dict], mode: str) -> str:
    out = f"## {mode.upper()} mode — accuracy by (depth, spacing, n_out)\n\n"
    out += "Accuracy on ingroup-polymorphic sites; ``n_poly`` is the number "
    out += "of such sites used. ``n_out`` is the number of outgroup "
    out += "populations exposed to the inference, and ``splits`` lists only "
    out += "the retained ones. ``baseline agree`` is the fraction of sites "
    out += "where Ancestree's MAP matches ``MajorityOutgroupInference``.\n\n"
    out += (
        "| depth | spacing | n_out | splits | n_poly | accuracy | mean max_prob "
        "| baseline agree |\n"
    )
    out += (
        "|------:|:--------|------:|:-------|-------:|---------:|--------------:"
        "|---------------:|\n"
    )
    for d in depths_ordered:
        for sp in spacings_ordered:
            for no in n_outs_ordered:
                cell = cells_by_key.get((d, sp, no))
                if cell is None:
                    continue
                out += (
                    f"| {d} | {sp} | {no} | "
                    f"{_fmt_splits(cell['outgroup_split_times'])} | "
                    f"{cell['n_sites_polymorphic']} | "
                    f"{_fmt_acc(cell['accuracy_polymorphic'])} | "
                    f"{_fmt_acc(cell['mean_max_prob_polymorphic'])} | "
                    f"{_fmt_acc(cell.get('baseline_agreement_polymorphic', float('nan')))} |\n"
                )
    out += "\n"
    return out


lines.append(_table(arg_cells, "arg"))
lines.append(_table(vcf_cells, "vcf"))
lines.append(_table(local_tree_cells, "local_tree"))


def _ranking_table(ranked: list[dict], mode: str) -> str:
    out = f"## {mode.upper()} mode — ranked by accuracy (top configs first)\n\n"
    out += "| rank | depth | spacing | n_out | n_poly | accuracy | mean max_prob |\n"
    out += "|-----:|------:|:--------|------:|-------:|---------:|--------------:|\n"
    for i, cell in enumerate(ranked, 1):
        out += (
            f"| {i} | {cell['depth_label']} | {cell['spacing']} | "
            f"{cell['n_out']} | "
            f"{cell['n_sites_polymorphic']} | "
            f"{_fmt_acc(cell['accuracy_polymorphic'])} | "
            f"{_fmt_acc(cell['mean_max_prob_polymorphic'])} |\n"
        )
    out += "\n"
    return out


lines.append(_ranking_table(arg_ranked, "arg"))
lines.append(_ranking_table(vcf_ranked, "vcf"))
lines.append(_ranking_table(local_tree_ranked, "local_tree"))


# ----------------------------------------------------- take-away paragraph


def _take_away(ranked: list[dict], mode: str) -> str:
    if not ranked:
        return f"_No {mode.upper()} cells available._\n\n"
    best = ranked[0]
    worst = ranked[-1]
    # Best-per-depth and best-per-spacing pivots.
    best_by_spacing: dict[str, dict] = {}
    for cell in ranked:
        sp = cell["spacing"]
        if sp not in best_by_spacing:
            best_by_spacing[sp] = cell
    best_by_depth: dict[str, dict] = {}
    for cell in ranked:
        d = cell["depth_label"]
        if d not in best_by_depth:
            best_by_depth[d] = cell
    best_depth_for_top_spacing = best_by_spacing[best["spacing"]]["depth_label"]
    top_spacing_overall = best["spacing"]
    out = f"**{mode.upper()} take-aways.** "
    out += (
        f"Top cell: depth={best['depth_label']}, spacing={top_spacing_overall}, "
        f"n_out={best['n_out']} "
        f"(accuracy={_fmt_acc(best['accuracy_polymorphic'])}); worst: "
        f"depth={worst['depth_label']}, spacing={worst['spacing']}, "
        f"n_out={worst['n_out']} "
        f"(accuracy={_fmt_acc(worst['accuracy_polymorphic'])}). "
    )
    out += (
        f"Holding spacing fixed at the winner ({top_spacing_overall}), "
        f"the best depth is {best_depth_for_top_spacing}. "
    )
    by_d_acc = sorted(
        (
            (d, max(
                (c["accuracy_polymorphic"] for c in ranked if c["depth_label"] == d),
                default=float("nan"),
            ))
            for d in sorted({c["depth_label"] for c in ranked}, key=float)
        ),
        key=lambda x: float(x[0]),
    )
    by_d_pretty = ", ".join(
        f"d={d}: max_acc={_fmt_acc(a)}" for d, a in by_d_acc
    )
    out += f"Per-depth ceiling (best spacing at each depth): {by_d_pretty}.\n\n"
    return out


lines.append("## Take-aways\n\n")
lines.append(_take_away(arg_ranked, "arg"))
lines.append(_take_away(vcf_ranked, "vcf"))
lines.append(_take_away(local_tree_ranked, "local_tree"))
# ---------------------------------------------------- per-bin SFS tables


def _bin_table(cells_by_key: dict[tuple[str, str, int], dict], mode: str) -> str:
    """Per-cell × per-(minor_count) accuracy table. Only emits bins that
    have at least one site somewhere in the sweep.
    """
    all_bins: set[int] = set()
    for cell in cells_by_key.values():
        for k in cell.get("by_minor_count", {}):
            all_bins.add(int(k))
    if not all_bins:
        return ""
    bins_sorted = sorted(all_bins)
    out = f"## {mode.upper()} mode — per-bin SFS accuracy (minor-allele count)\n\n"
    out += "Columns are ingroup minor-allele counts (1 = singleton). "
    out += "Each cell shows accuracy / n_sites for that (depth, spacing, bin).\n\n"
    out += ("| depth | spacing | n_out | "
            + " | ".join(f"mc={b}" for b in bins_sorted) + " |\n")
    out += ("|------:|:--------|------:|"
            + "|".join([":-----:"] * len(bins_sorted)) + "|\n")
    for d in depths_ordered:
        for sp in spacings_ordered:
            for no in n_outs_ordered:
                cell = cells_by_key.get((d, sp, no))
                if cell is None:
                    continue
                row = f"| {d} | {sp} | {no} |"
                for b in bins_sorted:
                    entry = cell.get("by_minor_count", {}).get(str(b))
                    if entry is None or not entry["n_sites"]:
                        row += " — |"
                    else:
                        row += f" {entry['accuracy']:.3f} / {entry['n_sites']} |"
                out += row + "\n"
    out += "\n"
    return out


lines.append(_bin_table(arg_cells, "arg"))
lines.append(_bin_table(vcf_cells, "vcf"))
lines.append(_bin_table(local_tree_cells, "local_tree"))

lines.append(
    "## Baseline-MAP agreement (`MajorityOutgroupInference`)\n\n"
    "The ``baseline agree`` column above is the fraction of\n"
    "ingroup-polymorphic sites where Ancestree's MAP allele matches the\n"
    "outgroup-majority rule (the\n"
    ":class:`~ancestree.inference.MajorityOutgroupInference` baseline). High\n"
    "agreement = the polariser signal dominates and the model + Kingman\n"
    "prior layers are minimally biasing the call. Low agreement = the\n"
    "model is moving posteriors away from the simple-outgroup-vote\n"
    "signal — under a well-specified likelihood low agreement is a flag\n"
    "for model misspecification (or for the prior shifting low-MAC bins\n"
    "where the outgroup vote is ambiguous).\n\n"
)


with open(out_md, "w") as f:
    f.writelines(lines)

print(
    f"Wrote {out_md} and {out_json} "
    f"({len(arg_cells)} ARG cells + {len(vcf_cells)} VCF cells "
    f"+ {len(local_tree_cells)} local-tree cells)",
    flush=True,
)
