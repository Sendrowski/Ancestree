"""SLiM-outgroup-bias benchmark report: aggregate the (chunk × f_del × n_out)
grid into per-(f_del, n_out) summaries.

Per-cell aggregation across chunks:

- Per-site MAP accuracy versus the SLiM/msprime simulator's truth
  ancestral state. Counts are SUMMED across chunks before averaging.
- Per-bin uSFS soft estimate AND simulator-truth histogram are summed
  across chunks for each (f_del, n_out) cell, then the per-bin signed
  bias and L1 aggregate are computed on the chunk-summed counts.
- Fit + infer wall time, kappa MLE, and outgroup-divergence MLE are
  reported per-chunk (no aggregation — these vary between chunks).

Emits ``results/reports/slim_outgroup_bias.{md,json}``.
"""
import json
import math
from collections import defaultdict
from pathlib import Path


try:
    cell_inputs = list(snakemake.input.cells)  # type: ignore[name-defined]
    cell_inputs_arg = list(snakemake.input.cells_arg)  # type: ignore[name-defined]
    cell_inputs_tsinfer = list(snakemake.input.cells_tsinfer)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    f_dels = list(snakemake.params.f_dels)  # type: ignore[name-defined]
    n_outs = list(snakemake.params.n_outs)  # type: ignore[name-defined]
    chunk_idxs = list(snakemake.params.chunk_idxs)  # type: ignore[name-defined]
    sim_config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    data_dir = Path("results/data")
    # vcf-mode cells: original schema, filename ..._chunk{c}_n{n}.json
    # (exclude the *_arg_n*.json / *_tsinfer_arg.json variants).
    cell_inputs = sorted(
        p for p in data_dir.glob("slim_outgroup_bias_fdel*_chunk*_n*.json")
        if "_arg_n" not in p.name and "_tsinfer_arg" not in p.name
    )
    cell_inputs_arg = sorted(data_dir.glob(
        "slim_outgroup_bias_fdel*_chunk*_arg_n*.json"
    ))
    cell_inputs_tsinfer = sorted(data_dir.glob(
        "slim_outgroup_bias_fdel*_chunk*_tsinfer_arg.json"
    ))
    out_json = "results/reports/slim_outgroup_bias.json"
    out_md = "results/reports/slim_outgroup_bias.md"
    f_dels = [0.0, 0.5, 1.0]
    n_outs = [1, 2, 3]
    chunk_idxs = list(range(10))
    sim_config = {}


def _bias_summary(bin_truth: dict[int, float], bin_est: dict[int, float]) -> dict:
    """Per-bin signed bias and aggregate L1 / RMSE on chunk-summed counts."""
    bins = sorted(set(bin_truth) | set(bin_est))
    per_bin = {}
    abs_total = 0.0
    sq_total = 0.0
    for b in bins:
        t = bin_truth.get(b, 0.0)
        e = bin_est.get(b, 0.0)
        d = e - t
        per_bin[b] = {"truth": t, "est": e, "bias": d}
        abs_total += abs(d)
        sq_total += d * d
    n = len(bins)
    return {
        "per_bin": per_bin,
        "l1_total": abs_total,
        "rmse": math.sqrt(sq_total / n) if n else 0.0,
    }


def _aggregate_cell(chunk_payloads: list[dict]) -> dict:
    """Sum per-chunk accuracy + uSFS counts for one (mode, f_del, n_out) cell."""
    n_with_truth = 0
    n_correct = 0
    accuracy_by_folded_bin: dict[int, list[int]] = defaultdict(list)
    accuracy_by_unfolded_bin: dict[int, list[int]] = defaultdict(list)
    bin_est_unfolded: dict[int, float] = defaultdict(float)
    bin_truth_unfolded: dict[int, float] = defaultdict(float)
    bin_est_folded: dict[int, float] = defaultdict(float)
    bin_truth_folded: dict[int, float] = defaultdict(float)
    n_sites_total = 0
    n_ingroup_set = set()

    per_chunk_meta = []
    for payload in chunk_payloads:
        n_ingroup = int(payload["ingroup_size"])
        n_ingroup_set.add(n_ingroup)
        per_site = payload["per_site"]
        n_sites_total += int(payload["n_sites"])
        per_chunk_meta.append({
            "chunk_idx": payload.get("chunk_idx"),
            "fit_seconds": payload.get("fit_seconds"),
            # joint_fit_seconds (vcf-mode only) = wall time of the one
            # fit per (f_del, n_out) — identical across chunks but
            # surfaced per chunk so the report's per-chunk timing table
            # has a value to display.
            "joint_fit_seconds": payload.get("joint_fit_seconds"),
            "infer_seconds": payload.get("infer_seconds"),
            "kappa": (payload.get("params_mle") or {}).get("kappa"),
            "outgroup_divergence_mle": payload.get("outgroup_divergence_mle"),
            "n_sites": payload.get("n_sites"),
            "tsinfer_build_meta": payload.get("tsinfer_build_meta"),
        })

        for s in per_site:
            truth = s.get("truth")
            if truth is None:
                continue
            n_with_truth += 1
            n_correct += int(bool(s["map_correct"]))
            minor_count = int(s["ingroup_minor_count"])
            folded_bin = minor_count
            if folded_bin > n_ingroup // 2:
                folded_bin = n_ingroup - folded_bin
            accuracy_by_folded_bin[folded_bin].append(int(s["map_correct"]))
            unfolded = s.get("ingroup_derived_count")
            if unfolded is not None:
                accuracy_by_unfolded_bin[int(unfolded)].append(int(s["map_correct"]))

            alleles = s.get("alleles", [])
            post_alleles = s.get("posterior_alleles", [])
            post_vals = s.get("posterior", [])
            counts_by_allele: dict[str, float] = {}
            truth_count = n_ingroup - int(s["ingroup_derived_count"])
            counts_by_allele[truth] = truth_count
            other_alleles = [a for a in alleles if a != truth]
            if len(alleles) == 2 and other_alleles:
                counts_by_allele[other_alleles[0]] = n_ingroup - truth_count
            elif other_alleles:
                share = (n_ingroup - truth_count) / len(other_alleles)
                for a in other_alleles:
                    counts_by_allele[a] = share

            for p_allele, p_prob in zip(post_alleles, post_vals):
                if p_allele in counts_by_allele:
                    derived_under = n_ingroup - counts_by_allele[p_allele]
                    bin_est_unfolded[int(round(derived_under))] += float(p_prob)
                    folded_under = derived_under
                    if folded_under > n_ingroup // 2:
                        folded_under = n_ingroup - folded_under
                    bin_est_folded[int(round(folded_under))] += float(p_prob)

            bin_truth_unfolded[int(s["ingroup_derived_count"])] += 1.0
            bin_truth_folded[folded_bin] += 1.0

    if len(n_ingroup_set) != 1:
        raise ValueError(
            f"Inconsistent ingroup size across chunks: {n_ingroup_set}"
        )
    n_ingroup = next(iter(n_ingroup_set))

    accuracy = n_correct / n_with_truth if n_with_truth else float("nan")
    folded_acc = {b: sum(v) / len(v) for b, v in sorted(accuracy_by_folded_bin.items())}
    folded_n = {b: len(v) for b, v in sorted(accuracy_by_folded_bin.items())}
    unfolded_acc = {b: sum(v) / len(v) for b, v in sorted(accuracy_by_unfolded_bin.items())}
    unfolded_n = {b: len(v) for b, v in sorted(accuracy_by_unfolded_bin.items())}

    uSFS_unfolded_bias = _bias_summary(dict(bin_truth_unfolded), dict(bin_est_unfolded))
    uSFS_folded_bias = _bias_summary(dict(bin_truth_folded), dict(bin_est_folded))

    return {
        "n_chunks": len(chunk_payloads),
        "n_sites_total": n_sites_total,
        "ingroup_size": n_ingroup,
        "n_with_truth": n_with_truth,
        "accuracy": accuracy,
        "accuracy_by_folded_bin": folded_acc,
        "n_by_folded_bin": folded_n,
        "accuracy_by_unfolded_bin": unfolded_acc,
        "n_by_unfolded_bin": unfolded_n,
        "uSFS_unfolded_bias": uSFS_unfolded_bias,
        "uSFS_folded_bias": uSFS_folded_bias,
        "per_chunk_meta": per_chunk_meta,
    }


# Group per-chunk payloads by (mode, f_del, n_out) so we can sum across chunks.
# The tsinfer-ARG mode has no n_out (ingroup-only) — keyed by n_out=0.
per_cell_chunks: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
for path in cell_inputs:
    with open(path) as f:
        payload = json.load(f)
    f_del = float(payload["f_del"])
    n_out = int(payload["n_out"])
    per_cell_chunks[("vcf", f_del, n_out)].append(payload)
for path in cell_inputs_arg:
    with open(path) as f:
        payload = json.load(f)
    f_del = float(payload["f_del"])
    n_out = int(payload["n_out"])
    per_cell_chunks[("arg_true", f_del, n_out)].append(payload)
for path in cell_inputs_tsinfer:
    with open(path) as f:
        payload = json.load(f)
    f_del = float(payload["f_del"])
    # tsinfer-ARG is ingroup-only — collapse to a single sentinel n_out=0.
    per_cell_chunks[("arg_tsinfer", f_del, 0)].append(payload)


cells: dict[str, dict] = {}  # key = f"{mode}_{f_del}_n{n_out}"
for (mode, f_del, n_out), chunk_payloads in per_cell_chunks.items():
    try:
        agg = _aggregate_cell(chunk_payloads)
    except ValueError as e:
        raise ValueError(
            f"Inconsistent ingroup size across chunks for "
            f"mode={mode} f_del={f_del} n_out={n_out}: {e}"
        )
    agg["f_del"] = f_del
    agg["n_out"] = n_out
    agg["mode"] = mode
    cells[f"{mode}_{f_del}_n{n_out}"] = agg

# Back-compat alias: the original report code (and downstream notebooks)
# read the vcf cells under f"{f_del}_n{n_out}" with no mode prefix. Keep
# those keys pointing at the vcf cells so the existing report sections
# stay byte-compatible for that mode.
for (mode, f_del, n_out), payload_list in per_cell_chunks.items():
    if mode == "vcf":
        cells[f"{f_del}_n{n_out}"] = cells[f"{mode}_{f_del}_n{n_out}"]

# ---------------------------------------------------------- JSON output -----
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"cells": cells, "sim_config": sim_config}, f, indent=2)

# ---------------------------------------------------------- Markdown output -
md: list[str] = []
md.append("# SLiM outgroup-bias validation (Ancestree FixedTreeInference)\n")
md.append(
    "Full 4-population SLiM forward sim (ingroup p0 + outgroups p1/p2/p3 "
    "with nested splits) at rescaled $N_e = 1000$, joined back to the "
    "human-ape $N_e \\approx 10^4$ scale by $s = N_\\text{target}/N_\\text{slim} = 10$. "
    "Split times at the target scale: "
    "$T_1, T_2, T_3 = (3\\cdot 10^5, 6\\cdot 10^5, 9\\cdot 10^5)$ gens.\n"
)

if sim_config:
    md.append("**Simulator settings**\n")
    for k, v in sim_config.items():
        md.append(f"- `{k}` = {v}")
    md.append("")

md.append("## Headline per-(f_del, n_out) MAP accuracy + uSFS L1 bias\n")
md.append(
    "L1 totals are computed on the chunk-summed soft uSFS minus the "
    "chunk-summed truth uSFS — bias scales with site count.\n"
)
md.append("| f_del | n_out | chunks | sites total | accuracy | uSFS unfolded L1 | uSFS folded L1 |")
md.append("|------:|------:|-------:|------------:|---------:|-----------------:|---------------:|")
for f_del in f_dels:
    for n_out in n_outs:
        key = f"{f_del}_n{n_out}"
        c = cells.get(key)
        if c is None:
            md.append(f"| {f_del} | {n_out} | n/a | (missing) | n/a | n/a | n/a |")
            continue
        md.append(
            f"| {f_del} | {n_out} | {c['n_chunks']} | {c['n_sites_total']} | "
            f"{c['accuracy']:.4f} | "
            f"{c['uSFS_unfolded_bias']['l1_total']:.2f} | "
            f"{c['uSFS_folded_bias']['l1_total']:.2f} |"
        )
md.append("")

md.append("## Headline cross-mode accuracy + L1 bias (vcf / arg_true / arg_tsinfer)\n")
md.append(
    "Side-by-side comparison of the three inference paths. `vcf` uses "
    "FixedTreeInference on the ingroup VCF + outgroup ladder. `arg_true` "
    "runs ARGBasedInference on the simulator-truth ARG simplified to "
    "ingroup + n_out outgroups. `arg_tsinfer` runs ARGBasedInference on a "
    "tsinfer-built ingroup-only ARG (no outgroups; n_out is 'ingroup-only').\n"
)
md.append("| mode | f_del | n_out | chunks | sites total | accuracy | uSFS unfolded L1 | uSFS folded L1 |")
md.append("|:-----|------:|:------|-------:|------------:|---------:|-----------------:|---------------:|")
# Cross-mode rows: vcf + arg_true sweep n_out. Arg_tsinfer is ingroup-only.
_n_out_labels: dict[tuple[str, int], str] = {}
for f_del in f_dels:
    for mode in ("vcf", "arg_true"):
        for n_out in n_outs:
            c = cells.get(f"{mode}_{f_del}_n{n_out}")
            if c is None:
                md.append(f"| {mode} | {f_del} | {n_out} | n/a | (missing) | n/a | n/a | n/a |")
                continue
            md.append(
                f"| {mode} | {f_del} | {n_out} | {c['n_chunks']} | {c['n_sites_total']} | "
                f"{c['accuracy']:.4f} | "
                f"{c['uSFS_unfolded_bias']['l1_total']:.2f} | "
                f"{c['uSFS_folded_bias']['l1_total']:.2f} |"
            )
    c = cells.get(f"arg_tsinfer_{f_del}_n0")
    if c is None:
        md.append(f"| arg_tsinfer | {f_del} | ingroup-only | n/a | (missing) | n/a | n/a | n/a |")
    else:
        md.append(
            f"| arg_tsinfer | {f_del} | ingroup-only | {c['n_chunks']} | {c['n_sites_total']} | "
            f"{c['accuracy']:.4f} | "
            f"{c['uSFS_unfolded_bias']['l1_total']:.2f} | "
            f"{c['uSFS_folded_bias']['l1_total']:.2f} |"
        )
md.append("")

md.append("## tsinfer build statistics (per chunk)\n")
md.append(
    "Wall time + site-count loss from tsinfer's biallelic / no-missing "
    "filter, relative to the simulator-truth site count for the same chunk.\n"
)
md.append("| f_del | chunk | tsinfer prep (s) | tsinfer infer (s) | inferred sites | truth sites |")
md.append("|------:|------:|-----------------:|------------------:|---------------:|------------:|")
for f_del in f_dels:
    c = cells.get(f"arg_tsinfer_{f_del}_n0")
    if c is None:
        continue
    for chunk_meta in sorted(c["per_chunk_meta"], key=lambda x: x.get("chunk_idx") or 0):
        bm = chunk_meta.get("tsinfer_build_meta") or {}
        prep = bm.get("tsinfer_prep_seconds")
        ti = bm.get("tsinfer_infer_seconds")
        n_inf = bm.get("tsinfer_n_sites_used")
        # Truth-ts site count is whatever the vcf-mode (or arg-true) chunk
        # saw for the same (f_del, chunk_idx) — both read the same .trees.
        truth_n = None
        for ref_mode in ("arg_true", "vcf"):
            ref = cells.get(f"{ref_mode}_{f_del}_n3") or cells.get(f"{ref_mode}_{f_del}_n1")
            if ref is None:
                continue
            for ref_chunk in ref["per_chunk_meta"]:
                if ref_chunk.get("chunk_idx") == chunk_meta.get("chunk_idx"):
                    truth_n = ref_chunk.get("n_sites")
                    break
            if truth_n is not None:
                break
        prep_s = f"{prep:.2f}" if prep is not None else "n/a"
        ti_s = f"{ti:.2f}" if ti is not None else "n/a"
        n_inf_s = str(n_inf) if n_inf is not None else "n/a"
        truth_s = str(truth_n) if truth_n is not None else "n/a"
        md.append(
            f"| {f_del} | {chunk_meta.get('chunk_idx')} | {prep_s} | {ti_s} | "
            f"{n_inf_s} | {truth_s} |"
        )
md.append("")

md.append("## Per-`f_del` uSFS bias reduction with added outgroups\n")
md.append("Compares `n_out=1` (canonical EST-SFS minimum) to `n_out=3`.\n")
md.append("| f_del | L1 bias (n_out=1) | L1 bias (n_out=3) | bias reduction | accuracy n_out=1 | accuracy n_out=3 |")
md.append("|------:|------------------:|------------------:|---------------:|-----------------:|-----------------:|")
for f_del in f_dels:
    c1 = cells.get(f"{f_del}_n1")
    c3 = cells.get(f"{f_del}_n3")
    if c1 is None or c3 is None:
        md.append(f"| {f_del} | n/a | n/a | n/a | n/a | n/a |")
        continue
    l1_1 = c1["uSFS_unfolded_bias"]["l1_total"]
    l1_3 = c3["uSFS_unfolded_bias"]["l1_total"]
    reduction = l1_1 - l1_3
    md.append(
        f"| {f_del} | {l1_1:.2f} | {l1_3:.2f} | {reduction:+.2f} | "
        f"{c1['accuracy']:.4f} | {c3['accuracy']:.4f} |"
    )
md.append("")

md.append("## Top biased unfolded uSFS bins\n")
md.append(
    "Each cell's top-8 bins by absolute bias from the chunk-summed counts. "
    "Bin = number of derived ingroup haps (derived = non-truth allele).\n"
)
md.append("| f_del | n_out | bin | truth | est | bias |")
md.append("|------:|------:|----:|------:|----:|-----:|")
for f_del in f_dels:
    for n_out in n_outs:
        c = cells.get(f"{f_del}_n{n_out}")
        if c is None:
            continue
        per_bin = c["uSFS_unfolded_bias"]["per_bin"]
        ranked = sorted(per_bin.items(), key=lambda kv: -abs(kv[1]["bias"]))
        for b, v in ranked[:8]:
            md.append(
                f"| {f_del} | {n_out} | {b} | {v['truth']:.1f} | "
                f"{v['est']:.2f} | {v['bias']:+.2f} |"
            )
md.append("")

md.append("## Imperfect folded-bin MAP accuracy\n")
md.append(
    "Bins where MAP accuracy < 1.0 across chunks (the rest of the "
    "folded-SFS spectrum is at perfect accuracy). Full per-bin tables "
    "are in the companion JSON.\n"
)
md.append("| f_del | n_out | folded bin | n_sites | accuracy |")
md.append("|------:|------:|-----------:|--------:|---------:|")
for f_del in f_dels:
    for n_out in n_outs:
        c = cells.get(f"{f_del}_n{n_out}")
        if c is None:
            continue
        for b, acc in c["accuracy_by_folded_bin"].items():
            if acc >= 0.9999:
                continue
            n = c["n_by_folded_bin"][b]
            md.append(f"| {f_del} | {n_out} | {b} | {n} | {acc:.4f} |")
md.append("")

md.append("## Per-cell fit + infer wall time (seconds, by chunk)\n")
md.append(
    "vcf-mode chunks now share a single joint fit per ``(f_del, n_out)`` "
    "cell — kappa MLE and outgroup-divergence MLE are identical across "
    "chunks (the joint fit pooled all 10 chunks' polymorphic sites). "
    "``joint_fit_s`` is the wall time of that one fit (also identical "
    "across the cell's chunks); per-chunk ``fit_s`` is 0 by construction. "
    "The expected outgroup divergences at target scale are "
    "K_k = mu_target * T_k = "
    "[1.25e-8 * 4e6, 1.25e-8 * 8e6, 1.25e-8 * 12e6] = "
    "[5.0e-2, 1.0e-1, 1.5e-1].\n"
)
md.append("| f_del | n_out | chunk | fit_s | joint_fit_s | infer_s | kappa MLE | outgroup divergence MLE |")
md.append("|------:|------:|------:|------:|------------:|--------:|----------:|:------------------------|")
for f_del in f_dels:
    for n_out in n_outs:
        c = cells.get(f"{f_del}_n{n_out}")
        if c is None:
            continue
        for chunk_meta in sorted(c["per_chunk_meta"], key=lambda x: x.get("chunk_idx") or 0):
            og_div = chunk_meta.get("outgroup_divergence_mle")
            og_div_str = (
                "[" + ", ".join(f"{x:.4g}" for x in og_div) + "]"
                if og_div is not None else "n/a"
            )
            kappa = chunk_meta.get("kappa")
            kappa_str = f"{kappa:.3f}" if kappa is not None else "n/a"
            fit_s = chunk_meta.get("fit_seconds") or 0.0
            joint_fit_s = chunk_meta.get("joint_fit_seconds")
            joint_fit_str = f"{joint_fit_s:.2f}" if joint_fit_s is not None else "n/a"
            infer_s = chunk_meta.get("infer_seconds") or 0.0
            md.append(
                f"| {f_del} | {n_out} | {chunk_meta.get('chunk_idx')} | "
                f"{fit_s:.2f} | {joint_fit_str} | {infer_s:.2f} | "
                f"{kappa_str} | {og_div_str} |"
            )
md.append("")

Path(out_md).parent.mkdir(parents=True, exist_ok=True)
with open(out_md, "w") as f:
    f.write("\n".join(md) + "\n")
print(f"Wrote {out_md}", flush=True)
print(f"Wrote {out_json}", flush=True)
