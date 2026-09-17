"""B5 report: aggregate per-threshold infer outputs → polytomy.{json,md}.

Reads one ``polytomy_thresh{T}.json`` per threshold (produced by
:mod:`infer_polytomy`), computes per-threshold accuracy stats vs msprime
ground truth, and writes the combined JSON + Markdown report.

Run directly with the standalone-default threshold list::

    python workflow/scripts/report_polytomy.py

Or via snakemake (the rule's ``input`` expand()s over the threshold list)::

    snakemake -j 1 results/reports/polytomy.md
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report_common import _load  # noqa: E402


try:
    in_jsons = list(snakemake.input.per_threshold)  # type: ignore[name-defined]
    thresholds = list(snakemake.params.thresholds)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    thresholds = [0, 100, 500, 1000, 5000, 10000, 50000]
    in_jsons = [f"results/data/polytomy_thresh{t}.json" for t in thresholds]
    out_json = "results/reports/polytomy.json"
    out_md = "results/reports/polytomy.md"
    config = {
        "samples": 30, "length": 2e5, "mu": 1e-7,
        "rec_rate": 1e-8, "pop_size": 1e4, "seed": 42,
        "thresholds": thresholds,
    }


# Load per-threshold infer outputs. Sort by threshold for the table order.
loaded = [_load(p) for p in in_jsons]
loaded.sort(key=lambda d: d["stats"]["threshold"])
baseline = loaded[0]  # threshold = lowest = bifurcating reference

# Build the comparison table.
sweep: list[dict] = []
for d in loaded:
    s = d["stats"]
    per = d["per_site"]
    n_eval = 0
    n_correct = 0
    n_agree_baseline = 0
    max_prob_sum = 0.0
    for pos, r in per.items():
        if r["truth"] is None:
            continue
        n_eval += 1
        max_prob_sum += r["max_prob"]
        if r["map_allele"] == r["truth"]:
            n_correct += 1
        base = baseline["per_site"].get(pos)
        if base is not None and base["map_allele"] == r["map_allele"]:
            n_agree_baseline += 1
    sweep.append({
        "threshold": s["threshold"],
        "n_eval": n_eval,
        "truth_recovery": n_correct / n_eval if n_eval else None,
        "map_agree_with_baseline": n_agree_baseline / n_eval if n_eval else None,
        "mean_max_prob": max_prob_sum / n_eval if n_eval else None,
        "mean_collapsed_per_site": s["mean_collapsed_per_site"],
        "mean_polytomic_internals_per_site": s["mean_polytomic_internals_per_site"],
        "max_polytomy_arity": s["max_polytomy_arity"],
    })

report = {
    "config": config,
    "n_truth_sites": sweep[0]["n_eval"] if sweep else 0,
    "sweep": sweep,
}

with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {out_json}", flush=True)


def _fmt(x):
    return "—" if x is None else f"{x:.4f}"


rows = []
for s in sweep:
    rows.append(
        f"| {s['threshold']:.0f} | {s['n_eval']} | "
        f"{s['mean_collapsed_per_site']:.2f} | "
        f"{s['mean_polytomic_internals_per_site']:.2f} | "
        f"{s['max_polytomy_arity']} | "
        f"{_fmt(s['truth_recovery'])} | "
        f"{_fmt(s['map_agree_with_baseline'])} | "
        f"{_fmt(s['mean_max_prob'])} |"
    )
table = "\n".join(rows)

md = f"""# B5 — Polytomy correctness / information-loss sweep

## Simulation (held fixed across thresholds)
- samples (haploid): `{config['samples']*2}` (msprime `samples={config['samples']}`, ploidy 2)
- sequence length: `{config['length']:g}`
- recombination rate: `{config['rec_rate']:g}`
- population size: `{config['pop_size']:g}`
- mutation rate: `{config['mu']:g}`
- thresholds (generations): `{thresholds}`
- seed: `{config['seed']}`
- sites with truth + parsimony info: {report['n_truth_sites']}

## Collapse rule
Every internal node whose **parent branch** is shorter than the threshold is
removed. Its children attach directly to its grandparent, **keeping their
original child-side branch lengths**. This is the realistic tsinfer-style
polytomy: the topology forgets the internal split, branch lengths don't
compensate.

## Sweep

| threshold | n | coll/site | poly/site | max arity | truth | agree-baseline | mean max_prob |
|---|---|---|---|---|---|---|---|
{table}
"""

with open(out_md, "w") as f:
    f.write(md)
print(f"Wrote {out_md}", flush=True)
