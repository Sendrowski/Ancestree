"""B10 caterpillar benchmark report: ladder vs caterpillar on monoallelic-ingroup sites.

Reads the two per-cell JSONs produced by :mod:`infer_caterpillar` (one
for ``tree_type=ladder``, one for ``tree_type=caterpillar``) and emits
per-truth-ancestral-state accuracy + mean-max-prob tables. The whole
point of B10 is to make the "ladder fails on monoallelic-ingroup
fixed-derived sites" failure mode and its caterpillar fix visible in
the report.

Standalone use::

    python workflow/scripts/report_caterpillar.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report_common import _load  # noqa: E402


try:
    in_ladder = snakemake.input.ladder  # type: ignore[name-defined]
    in_caterpillar = snakemake.input.caterpillar  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    in_ladder = "results/data/caterpillar_ladder.json"
    in_caterpillar = "results/data/caterpillar_caterpillar.json"
    out_json = "results/reports/caterpillar.json"
    out_md = "results/reports/caterpillar.md"
    sim_config = {}


ladder = _load(in_ladder)
caterpillar = _load(in_caterpillar)


def _summary(per_site: list[dict]) -> dict:
    """Overall accuracy / mean max_prob over the per-site records."""
    n = len(per_site)
    if not n:
        return {
            "n_sites": 0,
            "accuracy": float("nan"),
            "mean_max_prob": float("nan"),
            "mean_p_true": float("nan"),
        }
    return {
        "n_sites": n,
        "accuracy": sum(int(s["map_correct"]) for s in per_site) / n,
        "mean_max_prob": sum(s["max_prob"] for s in per_site) / n,
        "mean_p_true": sum(
            s["p_true"] for s in per_site if not math.isnan(s["p_true"])
        ) / n,
    }


def _by_truth(per_site: list[dict]) -> dict:
    """Per-(truth-ancestral-state) summary table."""
    by_t: defaultdict[str, list[dict]] = defaultdict(list)
    for s in per_site:
        by_t[s["truth"]].append(s)
    return {k: _summary(v) for k, v in sorted(by_t.items())}


def _by_fixed_state(per_site: list[dict]) -> dict:
    """Per-(ingroup-fixed-derived flag) summary — the diagnostic split.

    Sites where the ingroup is fixed for the **derived** allele are the
    ones the ladder gets wrong by construction. Sites where the ingroup
    is fixed for the **ancestral** allele are the ones the ladder happens
    to get right (its MAP = the ingroup's allele = the ancestral allele).
    """
    ancestral_fixed = [s for s in per_site if not s["ingroup_fixed_is_derived"]]
    derived_fixed = [s for s in per_site if s["ingroup_fixed_is_derived"]]
    return {
        "ingroup_fixed_ancestral": _summary(ancestral_fixed),
        "ingroup_fixed_derived": _summary(derived_fixed),
    }


ladder_sites = ladder["per_site"]
cat_sites = caterpillar["per_site"]

ladder_summary = _summary(ladder_sites)
cat_summary = _summary(cat_sites)

ladder_by_truth = _by_truth(ladder_sites)
cat_by_truth = _by_truth(cat_sites)

ladder_by_fixed = _by_fixed_state(ladder_sites)
cat_by_fixed = _by_fixed_state(cat_sites)

# Cross-cell agreement (same site positions, same monoallelic filter ⇒
# the two cells should have identical site sets).
ladder_by_pos = {int(s["pos"]): s for s in ladder_sites}
cat_by_pos = {int(s["pos"]): s for s in cat_sites}
shared_positions = sorted(set(ladder_by_pos) & set(cat_by_pos))
n_agree = sum(
    1 for p in shared_positions
    if ladder_by_pos[p]["map_allele"] == cat_by_pos[p]["map_allele"]
)
agreement_rate = n_agree / len(shared_positions) if shared_positions else float("nan")


out_payload = {
    "sim_config": sim_config,
    "ladder": {
        "summary": ladder_summary,
        "by_truth": ladder_by_truth,
        "by_ingroup_fixed_state": ladder_by_fixed,
        "inference_meta": ladder.get("inference_meta", {}),
    },
    "caterpillar": {
        "summary": cat_summary,
        "by_truth": cat_by_truth,
        "by_ingroup_fixed_state": cat_by_fixed,
        "inference_meta": caterpillar.get("inference_meta", {}),
    },
    "cross_cell": {
        "n_shared_sites": len(shared_positions),
        "map_agreement_rate": agreement_rate,
    },
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


# ------------------------------ Markdown summary --------------------------

def _fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.4f}"


header = "# Monoallelic-ingroup caterpillar polarisation benchmark (B10)\n\n"
header += (
    "Compares the two ways :class:`~ancestree.inference.FixedTreeInference`\n"
    "can polarise a **monoallelic-ingroup** site: the bare\n"
    ":class:`~ancestree.trees.OutgroupLadderTree` rooted at the ingroup\n"
    "MRCA (which roots on the *derived* side of the only informative\n"
    "substitution and so reports the ingroup's own fixed allele) versus the\n"
    "deep-rooted caterpillar rooted at the\n"
    "deepest outgroup ancestor (full Felsenstein over all outgroups recovers\n"
    "the ancestral state). Both readings are the same ladder read at a\n"
    "different focal node, selected per site class by the caller;\n"
    "``FixedTreeInference`` takes that node as an argument.\n\n"
    "Both modes use ``JC69 + BaseComposition.no_counts() + uniform root prior``;\n"
    "branch lengths come from the simulator's known split times scaled by\n"
    "``mu`` (no MLE fit — the comparison isolates the *tree-choice* axis).\n\n"
)

if sim_config:
    header += "**Simulator settings**\n\n"
    for k, v in sim_config.items():
        header += f"- ``{k}`` = {v}\n"
    header += "\n"

overall = (
    "## Overall accuracy on monoallelic-ingroup sites\n\n"
    "| tree_type | n_sites | accuracy | mean max_prob | mean p_true |\n"
    "|:----------|--------:|---------:|--------------:|------------:|\n"
    f"| ladder      | {ladder_summary['n_sites']} | {_fmt(ladder_summary['accuracy'])} | "
    f"{_fmt(ladder_summary['mean_max_prob'])} | {_fmt(ladder_summary['mean_p_true'])} |\n"
    f"| caterpillar | {cat_summary['n_sites']} | {_fmt(cat_summary['accuracy'])} | "
    f"{_fmt(cat_summary['mean_max_prob'])} | {_fmt(cat_summary['mean_p_true'])} |\n\n"
)

# Per-truth-ancestral-state table.
all_truth_states = sorted(set(ladder_by_truth) | set(cat_by_truth))
truth_table = "## Accuracy stratified by truth ancestral state\n\n"
truth_table += "| tree_type | truth | n_sites | accuracy | mean max_prob |\n"
truth_table += "|:----------|:-----:|--------:|---------:|--------------:|\n"
for label, table in [("ladder", ladder_by_truth), ("caterpillar", cat_by_truth)]:
    for t in all_truth_states:
        v = table.get(t)
        if v is None:
            continue
        truth_table += (
            f"| {label} | {t} | {v['n_sites']} | "
            f"{_fmt(v['accuracy'])} | {_fmt(v['mean_max_prob'])} |\n"
        )
truth_table += "\n"

# The diagnostic split: by whether the ingroup-fixed allele is ancestral
# or derived. This is where the ladder vs caterpillar contrast lives.
fixed_table = (
    "## Accuracy stratified by ingroup-fixed allele state\n\n"
    "``ingroup_fixed_ancestral`` rows: the single allele segregating in\n"
    "the ingroup happens to be the simulator's ancestral allele. The\n"
    "ladder trivially gets these right (its MAP = ingroup allele = truth);\n"
    "the caterpillar should agree.\n\n"
    "``ingroup_fixed_derived`` rows: the ingroup is fixed for the\n"
    "*derived* allele (a fixed substitution swept on the ingroup lineage).\n"
    "The ladder reports the ingroup's allele as MAP — *wrong*. The\n"
    "caterpillar, which roots at the deep outgroup ancestor, can recover\n"
    "the ancestral state via the outgroup tips.\n\n"
    "| tree_type | ingroup_fixed | n_sites | accuracy | mean max_prob |\n"
    "|:----------|:--------------|--------:|---------:|--------------:|\n"
)
for label, table in [("ladder", ladder_by_fixed), ("caterpillar", cat_by_fixed)]:
    for k in ("ingroup_fixed_ancestral", "ingroup_fixed_derived"):
        v = table[k]
        fixed_table += (
            f"| {label} | {k.replace('ingroup_fixed_', '')} | "
            f"{v['n_sites']} | {_fmt(v['accuracy'])} | {_fmt(v['mean_max_prob'])} |\n"
        )
fixed_table += "\n"

# Cross-cell agreement.
agree_table = (
    "## Cross-cell MAP agreement\n\n"
    f"- shared sites: {len(shared_positions)}\n"
    f"- MAP agreement rate: {_fmt(agreement_rate)}\n\n"
    "Agreement is high on ``ingroup_fixed_ancestral`` sites (both trees\n"
    "land on the ingroup-allele MAP, which happens to be ancestral) and\n"
    "low on ``ingroup_fixed_derived`` sites (the ladder picks the\n"
    "ingroup's derived allele; the caterpillar picks the outgroup-implied\n"
    "ancestor). The cross-cell mismatch is the exact size of the\n"
    "``ingroup_fixed_derived`` failure rate of the ladder.\n\n"
)

interp = (
    "## Why a fixed site needs the deeper node\n\n"
    "A monoallelic-ingroup site carries no within-ingroup polarisation\n"
    "signal — every ingroup tip reads the same allele, so the SFS-prior\n"
    "is uninformative and the reading at the **ingroup MRCA** sees a\n"
    "single observation. There the MAP is just the ingroup's own allele:\n"
    "the readout sits on the *derived* side of the single substitution\n"
    "that separates the ingroup from the outgroups, so it restates the\n"
    "input whenever the fixed allele is derived. That is the right answer\n"
    "to the question asked, and the wrong question to ask.\n\n"
    "Moving the focal node to the ladder's deepest join asks the question\n"
    "the site actually poses. The collapsed ingroup tip then enters as an\n"
    "observation rather than as the readout, and full Felsenstein over it\n"
    "and the outgroup tips recovers the deep state from the\n"
    "outgroup-majority signal. The accuracy delta in the\n"
    "``ingroup_fixed_derived`` row above is the size of that difference.\n\n"
)

with open(out_md, "w") as f:
    f.write(header + overall + truth_table + fixed_table + agree_table + interp)
print(f"Wrote {out_md}", flush=True)
