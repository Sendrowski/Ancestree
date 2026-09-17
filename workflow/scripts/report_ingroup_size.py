"""Ingroup-size sweep benchmark report: per-bin SFS accuracy vs n_ingroup.

Reads the 18 per-(n_ingroup, seed) inference outputs and aggregates
SFS-stratified accuracy. The whole point of B7 is to check whether
adding more ingroup haplotypes helps ARG ingroup-only-mode recover the
deep-ancestor allele at *each* per-bin SFS slice — the B6 picture is
that the ingroup-only mode looks unconvincing, but B6 used only n=20.

For each (n_ingroup, derived_count) bin it reports ``accuracy`` (fraction
of MAP calls matching the simulator's truth ancestral allele) averaged
across the three subsampling seeds, plus a coarser overall ``accuracy``
and ``mean_max_prob`` per n_ingroup. Standard deviations across seeds
provide noise bars.

Bins use the **folded** SFS for cross-n_ingroup comparability: bin
index = ``min(derived_count, n_ingroup - derived_count)``. Each
n_ingroup contributes bins 1..floor(n_ingroup/2). The report keeps
them as separate columns and reports NaN where missing.

Standalone use::

    python workflow/scripts/report_ingroup_size.py
"""
import json
import math
import statistics as stats
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _report_common import _entropy_bits, _mean_std  # noqa: E402

# Hypergeometric down-projection: subsample n_ingroup → REFERENCE_N to make
# per-bin accuracy comparable across different n_ingroup values. Only applies
# when n_ingroup ≥ REFERENCE_N. For smaller cells the projection field is None.
REFERENCE_N = 20


def _projected_folded_weights(d: int, n: int, k: int = REFERENCE_N) -> list[float]:
    """Hypergeometric weights mapping a site at derived_count ``d`` in the
    full ``n``-sample to the folded SFS bins of a random ``k``-subsample.

    Returns a length-``k//2`` list where index ``i-1`` is the probability the
    subsample has folded minor count ``i`` (= ``i`` or ``k - i`` raw derived
    count). Index 0 (i=1) ... index k//2 - 1 (i=k//2). The monomorphic mass
    (raw subsample count 0 or k) is dropped — that site contributes no SFS
    bin in the projected view.
    """
    # P(X = j) = C(d, j) * C(n - d, k - j) / C(n, k), defined when both
    # binomials are valid.
    weights = [0.0] * (k + 1)
    total = math.comb(n, k)
    for j in range(max(0, k - (n - d)), min(d, k) + 1):
        weights[j] = math.comb(d, j) * math.comb(n - d, k - j) / total
    # Fold: bin i corresponds to raw counts i and k - i (when i != k - i).
    folded = []
    for i in range(1, k // 2 + 1):
        if i == k - i:
            folded.append(weights[i])
        else:
            folded.append(weights[i] + weights[k - i])
    return folded


def _projected_unfolded_weights(d: int, n: int, k: int = REFERENCE_N) -> list[float]:
    """Hypergeometric weights mapping a site at derived_count ``d`` in the
    full ``n``-sample to the *unfolded* SFS bins (0..k) of a random
    ``k``-subsample. Returns a length-``k+1`` list. Index ``j`` is the
    probability that the subsample contains ``j`` derived alleles.
    Includes the monomorphic tails ``j=0`` (fixed-ancestral) and ``j=k``
    (fixed-derived).
    """
    weights = [0.0] * (k + 1)
    total = math.comb(n, k)
    for j in range(max(0, k - (n - d)), min(d, k) + 1):
        weights[j] = math.comb(d, j) * math.comb(n - d, k - j) / total
    return weights


try:
    inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    inputs = sorted(Path("results/data").glob("ingroup_size_n*_seed*.json"))
    out_json = "results/reports/ingroup_size.json"
    out_md = "results/reports/ingroup_size.md"
    sim_config = {}


# Per-cell aggregation: one entry per (n_ingroup, seed).
per_cell: dict[tuple[int, int], dict] = {}
for path in inputs:
    with open(path) as f:
        payload = json.load(f)
    meta = payload["inference_meta"]
    n_ingroup = int(meta["n_ingroup"])
    seed = int(meta["seed"])
    sites = payload["sites"]

    STATES_ORDER = ("A", "C", "G", "T")
    max_probs = [s["max_prob"] for s in sites.values()]
    entropies = [_entropy_bits(s["posterior"]) for s in sites.values()]
    with_truth = [s for s in sites.values() if s["truth"] is not None]
    # Per-site Brier score against the one-hot truth:
    # B = sum_a (p_a - 1[a = truth])^2 = sum_a p_a^2 - 2 p_true + 1.
    for s in with_truth:
        t = s["truth"]
        if t in STATES_ORDER:
            post = [float(v) for v in s["posterior"]]
            p_true = post[STATES_ORDER.index(t)]
            s["brier"] = sum(v * v for v in post) - 2.0 * p_true + 1.0
        else:
            s["brier"] = float("nan")
    n_correct = sum(s["map_correct"] for s in with_truth)
    accuracy = n_correct / len(with_truth) if with_truth else float("nan")

    # Per folded-SFS-bin accuracy (only polymorphic sites with truth).
    # Folded bin keeps a single integer index that is meaningful across
    # different n_ingroup values: minor-allele count.
    by_bin: defaultdict[int, list[dict]] = defaultdict(list)
    for s in with_truth:
        dc = s.get("derived_count")
        if dc is None:
            continue
        if dc == 0 or dc == n_ingroup:
            continue  # monomorphic in this sub-sample. Uninformative for SFS
        folded = min(dc, n_ingroup - dc)
        by_bin[folded].append(s)
    per_bin = {
        k: {
            "n_sites": len(v),
            "accuracy": sum(s["map_correct"] for s in v) / len(v),
            "mean_max_prob": sum(s["max_prob"] for s in v) / len(v),
            "mean_brier": sum(s["brier"] for s in v) / len(v),
        }
        for k, v in sorted(by_bin.items())
    }

    # Projected per-bin accuracy: each polymorphic site is fractionally
    # assigned to the REFERENCE_N folded bins it could land in under a
    # hypergeometric subsample. Only meaningful when n_ingroup ≥ REFERENCE_N.
    projected_per_bin: dict[int, dict] | None = None
    if n_ingroup >= REFERENCE_N:
        sum_acc = [0.0] * (REFERENCE_N // 2)
        sum_w = [0.0] * (REFERENCE_N // 2)
        for s in with_truth:
            dc = s.get("derived_count")
            if dc is None or dc == 0 or dc == n_ingroup:
                continue
            w = _projected_folded_weights(dc, n_ingroup, REFERENCE_N)
            ok = 1.0 if s["map_correct"] else 0.0
            for i, wi in enumerate(w):
                sum_w[i] += wi
                sum_acc[i] += wi * ok
        projected_per_bin = {
            i + 1: {
                "n_eff_sites": sum_w[i],
                "accuracy": (sum_acc[i] / sum_w[i]) if sum_w[i] > 0 else float("nan"),
            }
            for i in range(REFERENCE_N // 2)
        }

    # Unfolded projection (matches Figure 4's x-axis). Interior bins
    # 1..REFERENCE_N-1 take the hypergeometric weight of every site. The two
    # tail bins are restricted to the site classes they name: bin 0 holds only
    # sites monomorphic in the subsample (derived count 0) and bin REFERENCE_N
    # only sites fixed for a non-ancestral allele (derived count n_ingroup).
    # A site polymorphic in the full sample carries subsampling weight into
    # both tails, and admitting it biases them: that mass is graded at the
    # ingroup MRCA with a resolved genealogy, scoring near the interior rather
    # than the fixed class, and it grows with n_ingroup until it dominates.
    # Dropping it leaves each tail's n_eff_sites an exact site count.
    projected_per_bin_unfolded: dict[int, dict] | None = None
    if n_ingroup >= REFERENCE_N:
        K = REFERENCE_N + 1
        sum_acc_u = [0.0] * K
        sum_br_u = [0.0] * K
        sum_w_u = [0.0] * K
        for s in with_truth:
            dc = s.get("derived_count")
            if dc is None:
                continue
            w = _projected_unfolded_weights(dc, n_ingroup, REFERENCE_N)
            if dc != 0:
                w[0] = 0.0
            if dc != n_ingroup:
                w[REFERENCE_N] = 0.0
            ok = 1.0 if s["map_correct"] else 0.0
            br = float(s.get("brier", float("nan")))
            for j, wj in enumerate(w):
                if wj <= 0:
                    continue
                sum_w_u[j] += wj
                sum_acc_u[j] += wj * ok
                if not math.isnan(br):
                    sum_br_u[j] += wj * br
        projected_per_bin_unfolded = {
            j: {
                "n_eff_sites": sum_w_u[j],
                "accuracy": (sum_acc_u[j] / sum_w_u[j]) if sum_w_u[j] > 0 else float("nan"),
                "mean_brier": (sum_br_u[j] / sum_w_u[j]) if sum_w_u[j] > 0 else float("nan"),
            }
            for j in range(K)
        }

    per_cell[(n_ingroup, seed)] = {
        "n_ingroup": n_ingroup,
        "seed": seed,
        "n_sites": len(sites),
        "n_sites_with_truth": len(with_truth),
        "mean_max_prob": sum(max_probs) / len(max_probs) if max_probs else float("nan"),
        "mean_entropy_bits": sum(entropies) / len(entropies) if entropies else float("nan"),
        "accuracy": accuracy,
        "per_bin": per_bin,
        "projected_per_bin": projected_per_bin,
        "projected_per_bin_unfolded": projected_per_bin_unfolded,
    }


# Aggregate across seeds, per n_ingroup. Mean ± std.
n_ingroups: list[int] = sorted({n for (n, _) in per_cell.keys()})


summary: dict[int, dict] = {}
for n in n_ingroups:
    cells = [c for (k_n, _), c in per_cell.items() if k_n == n]
    acc_mean, acc_std = _mean_std([c["accuracy"] for c in cells])
    mp_mean, mp_std = _mean_std([c["mean_max_prob"] for c in cells])
    ent_mean, ent_std = _mean_std([c["mean_entropy_bits"] for c in cells])
    n_sites_mean, _ = _mean_std([c["n_sites"] for c in cells])

    # Aggregate per-bin across seeds.
    bin_acc: defaultdict[int, list[float]] = defaultdict(list)
    bin_mp: defaultdict[int, list[float]] = defaultdict(list)
    bin_br: defaultdict[int, list[float]] = defaultdict(list)
    bin_n: defaultdict[int, list[int]] = defaultdict(list)
    for c in cells:
        for k, v in c["per_bin"].items():
            bin_acc[int(k)].append(v["accuracy"])
            bin_mp[int(k)].append(v["mean_max_prob"])
            bin_br[int(k)].append(v["mean_brier"])
            bin_n[int(k)].append(v["n_sites"])
    per_bin_summary: dict[int, dict] = {}
    for k in sorted(bin_acc.keys()):
        a_m, a_s = _mean_std(bin_acc[k])
        p_m, p_s = _mean_std(bin_mp[k])
        br_m, br_s = _mean_std(bin_br[k])
        per_bin_summary[k] = {
            "accuracy_mean": a_m, "accuracy_std": a_s,
            "mean_max_prob_mean": p_m, "mean_max_prob_std": p_s,
            "mean_brier_mean": br_m, "mean_brier_std": br_s,
            "n_sites_mean": stats.mean(bin_n[k]),
        }

    # Aggregate projected per-bin across seeds (only defined for n ≥ REFERENCE_N).
    projected_summary: dict[int, dict] | None = None
    cells_with_proj = [c for c in cells if c.get("projected_per_bin") is not None]
    if cells_with_proj:
        proj_acc: defaultdict[int, list[float]] = defaultdict(list)
        proj_n: defaultdict[int, list[float]] = defaultdict(list)
        for c in cells_with_proj:
            for k, v in c["projected_per_bin"].items():
                proj_acc[int(k)].append(v["accuracy"])
                proj_n[int(k)].append(v["n_eff_sites"])
        projected_summary = {}
        for k in sorted(proj_acc.keys()):
            a_m, a_s = _mean_std(proj_acc[k])
            projected_summary[k] = {
                "accuracy_mean": a_m, "accuracy_std": a_s,
                "n_eff_sites_mean": stats.mean(proj_n[k]),
            }

    # Aggregate unfolded projection across seeds (only defined for n ≥ REFERENCE_N).
    projected_unfolded_summary: dict[int, dict] | None = None
    cells_with_proj_u = [c for c in cells if c.get("projected_per_bin_unfolded") is not None]
    if cells_with_proj_u:
        pu_acc: defaultdict[int, list[float]] = defaultdict(list)
        pu_br: defaultdict[int, list[float]] = defaultdict(list)
        pu_n: defaultdict[int, list[float]] = defaultdict(list)
        for c in cells_with_proj_u:
            for k, v in c["projected_per_bin_unfolded"].items():
                pu_acc[int(k)].append(v["accuracy"])
                pu_br[int(k)].append(v["mean_brier"])
                pu_n[int(k)].append(v["n_eff_sites"])
        projected_unfolded_summary = {}
        for k in sorted(pu_acc.keys()):
            a_m, a_s = _mean_std(pu_acc[k])
            br_m, br_s = _mean_std(pu_br[k])
            projected_unfolded_summary[k] = {
                "accuracy_mean": a_m, "accuracy_std": a_s,
                "mean_brier_mean": br_m, "mean_brier_std": br_s,
                "n_eff_sites_mean": stats.mean(pu_n[k]),
            }

    summary[n] = {
        "n_ingroup": n,
        "n_seeds": len(cells),
        "n_sites_mean": n_sites_mean,
        "accuracy_mean": acc_mean, "accuracy_std": acc_std,
        "mean_max_prob_mean": mp_mean, "mean_max_prob_std": mp_std,
        "mean_entropy_bits_mean": ent_mean, "mean_entropy_bits_std": ent_std,
        "per_bin": per_bin_summary,
        "projected_per_bin": projected_summary,
        "projected_per_bin_unfolded": projected_unfolded_summary,
    }


out_payload = {
    "summary": summary,
    "per_cell": {f"n{n}_seed{s}": v for (n, s), v in per_cell.items()},
    "sim_config": sim_config,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(out_payload, f, indent=2)


# ------------------------------- Markdown summary --------------------------

header = "# Ingroup-size sweep benchmark (ARGBasedInference, n_out=0)\n\n"
header += (
    "Per-bin SFS accuracy and overall posterior confidence as a function\n"
    "of the number of ingroup haplotypes available to the Felsenstein\n"
    "kernel in **ARG ingroup-only mode** (n_out=0). One large simulated\n"
    "ARG; each cell randomly subsamples the ingroup down to ``n_ingroup``\n"
    "with a per-cell RNG seed for noise bars (3 seeds per n_ingroup).\n"
    "``JC69 + uniform prior + full`` throughout.\n\n"
    "**Folded SFS bin** = ``min(derived_count, n_ingroup - derived_count)``\n"
    "where ``derived_count`` is computed against the simulator's truth\n"
    "ancestral allele. Each n_ingroup contributes folded bins 1..⌊n/2⌋;\n"
    "missing bins show as ``--``.\n\n"
)

if sim_config:
    header += "**Simulator settings**\n\n"
    for k, v in sim_config.items():
        header += f"- ``{k}`` = {v}\n"
    header += "\n"

# Overall summary
table = "## Overall summary (averaged over 3 subsampling seeds)\n\n"
table += "| n_ingroup | n_seeds | n_sites | accuracy (mean ± std) | mean max_prob (mean ± std) | mean entropy bits |\n"
table += "|----------:|--------:|--------:|----------------------:|---------------------------:|------------------:|\n"
for n in n_ingroups:
    s = summary[n]
    table += (
        f"| {n} | {s['n_seeds']} | {int(s['n_sites_mean'])} | "
        f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f} | "
        f"{s['mean_max_prob_mean']:.4f} ± {s['mean_max_prob_std']:.4f} | "
        f"{s['mean_entropy_bits_mean']:.4f} |\n"
    )
table += "\n"

# Per-bin SFS accuracy table — rows = folded bins, cols = n_ingroup
all_bins = sorted({
    b for s in summary.values() for b in s["per_bin"].keys()
})
bin_table = "## Per-bin SFS accuracy (mean over seeds)\n\n"
bin_table += "Rows are folded-SFS bin (minor allele count over the subsampled ingroup);\n"
bin_table += "columns are n_ingroup. Cell shows accuracy mean (±std across seeds).\n\n"
bin_table += "| folded bin | " + " | ".join(f"n={n}" for n in n_ingroups) + " |\n"
bin_table += "|----------:|" + "|".join(["----:"] * len(n_ingroups)) + "|\n"
for b in all_bins:
    row = f"| {b} |"
    for n in n_ingroups:
        per_bin = summary[n]["per_bin"]
        if b in per_bin:
            v = per_bin[b]
            row += f" {v['accuracy_mean']:.3f} ± {v['accuracy_std']:.3f} |"
        else:
            row += " -- |"
    bin_table += row + "\n"
bin_table += "\n"

# Site-count companion (so the reader can spot bins with tiny n)
count_table = "## Per-bin site counts (mean over seeds)\n\n"
count_table += "Same layout as the accuracy table; helps spot bins with few sites where the accuracy estimate is noisy.\n\n"
count_table += "| folded bin | " + " | ".join(f"n={n}" for n in n_ingroups) + " |\n"
count_table += "|----------:|" + "|".join(["----:"] * len(n_ingroups)) + "|\n"
for b in all_bins:
    row = f"| {b} |"
    for n in n_ingroups:
        per_bin = summary[n]["per_bin"]
        if b in per_bin:
            row += f" {int(per_bin[b]['n_sites_mean'])} |"
        else:
            row += " -- |"
    count_table += row + "\n"
count_table += "\n"

baseline_note = (
    "## Baseline-MAP agreement (n/a)\n\n"
    "B7 is the ingroup-only sweep (n_out=0 by design). The B11\n"
    "``MajorityOutgroupInference`` baseline requires at least one\n"
    "outgroup observation per site to vote on the ancestral allele, so\n"
    "the agreement column is undefined for this benchmark. The cells'\n"
    "per-site JSON intentionally omits ``baseline_map``; the B6/B8/B9\n"
    "reports carry it where outgroups are available.\n\n"
)

interp = (
    "## Reading the trend\n\n"
    "- The headline question: does scaling up the ingroup beyond B6's\n"
    "  ``n=20`` cell actually help ARG ingroup-only mode polarise sites?\n"
    "  Compare the row at folded bin 1 (singletons) across the n_ingroup\n"
    "  columns — that bin carries the least ingroup signal and benefits\n"
    "  most from outgroups in B6.\n"
    "- If accuracy at each folded bin is roughly flat across n_ingroup,\n"
    "  the conclusion is that adding ingroup haplotypes mostly grows the\n"
    "  *catalog* of low-frequency sites rather than improving per-site\n"
    "  identifiability — i.e. ingroup-only mode is indeed information-\n"
    "  limited by the ingroup MRCA depth, not by sample size.\n"
    "- If accuracy at fixed bins climbs with n_ingroup, then larger\n"
    "  ingroups do shift the ingroup-MRCA toward the deep ancestor enough\n"
    "  for the kernel to recover more truth alleles — i.e. n=20 was just\n"
    "  too small in B6.\n\n"
)

with open(out_md, "w") as f:
    f.write(header + table + bin_table + count_table + baseline_note + interp)
print(f"Wrote {out_md}", flush=True)
