"""B3 report: Ancestree vs fastDFE across (model, prior) cells.

Aggregates 2×2 cells of {JC, K2} × {Kingman, Adaptive} for both tools,
joins per-site posteriors per cell, and reports MAP-allele agreement,
posterior numerical diffs, truth recovery, and the fitted branch rates
+ outgroup divergences.

Run directly (defaults to JC+Kingman cell)::

    python workflow/scripts/report_estsfs.py

Or via snakemake::

    snakemake -j 1 results/reports/estsfs_agreement.md
"""
import json
import sys
from pathlib import Path

import msprime
import numpy as np
import tskit

from ancestree import GTR, STATE_INDEX, STATES

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msprime_substitution_units import substitutions_per_mutation  # noqa: E402


try:
    in_ancestree_paths = list(snakemake.input.ancestree)  # type: ignore[name-defined]
    in_fastdfe_paths = list(snakemake.input.fastdfe)  # type: ignore[name-defined]
    in_baseline_paths = list(snakemake.input.baseline)  # type: ignore[name-defined]
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    sims = list(snakemake.params.sims)  # type: ignore[name-defined]
    sim_specs = dict(snakemake.params.sim_specs)  # type: ignore[name-defined]
    models = list(snakemake.params.models)  # type: ignore[name-defined]
    priors = list(snakemake.params.priors)  # type: ignore[name-defined]
    n_outs = list(snakemake.params.n_outs)  # type: ignore[name-defined]
    config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    sims = ["jc", "hky2", "gtr"]
    sim_specs = {
        "jc": {"model": "HKY", "kappa": 1.0},
        "hky2": {"model": "HKY", "kappa": 2.0},
        "gtr": {
            "model": "GTR",
            "rates": [1.0, 4.0, 1.0, 1.0, 4.0, 1.0],
            "pi": [0.3, 0.2, 0.2, 0.3],
        },
    }
    models = ["JC", "K2"]
    priors = ["kingman", "adaptive"]
    n_outs = [2, 3]
    in_ancestree_paths = [
        f"results/data/estsfs_ancestree_{s}_{m}_{p}_n{n}.json"
        for s in sims for m in models for p in priors for n in n_outs
    ]
    in_fastdfe_paths = [
        f"results/data/estsfs_fastdfe_{s}_{m}_{p}_n{n}.json"
        for s in sims for m in models for p in priors for n in n_outs
    ]
    in_baseline_paths = [
        f"results/data/estsfs_baseline_{s}_{m}_n{n}.json"
        for s in sims for m in models for n in n_outs
    ]
    in_trees = "results/data/estsfs_sim_jc.trees"
    in_meta = "results/data/estsfs_sim_jc_meta.json"
    out_json = "results/reports/estsfs_agreement.json"
    out_md = "results/reports/estsfs_agreement.md"
    config = {
        "n_ingroup": 40, "length": 2e5, "mu": 1e-8,
        "rec_rate": 1e-8, "pop_size": 1e4, "outgroup_pop_size": 1e5,
        "outgroup_divergences": [5e4, 1e5, 1.5e5],
        "seed": 42, "model": "JC",
    }


# ----------------------------------------------------------- load all cells

def _load_cell(path: str) -> "dict | None":
    """Load a per-cell JSON, returning ``None`` if the file is missing.

    Missing cells are expected for the fastDFE × R6 combination (fastDFE's
    est-sfs wrapper doesn't implement R6). The report tolerates them by
    treating the fastDFE column as N/A for that row.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _truth_kappa(spec: dict) -> float | None:
    """Effective transition/transversion ratio implied by a sim spec.

    For HKY the literal ``kappa``. For GTR the ratio of (mean transition
    exchangeability) / (mean transversion exchangeability) — i.e. the κ
    a K2/HKY fit should converge to under symmetric-transversion GTR.
    Returns ``None`` when neither is well-defined.
    """
    if spec.get("model") == "HKY":
        return float(spec["kappa"])
    if spec.get("model") == "GTR":
        rates = list(spec["rates"])
        ts = (rates[1] + rates[4]) / 2.0  # r_AG + r_CT
        tv = (rates[0] + rates[2] + rates[3] + rates[5]) / 4.0
        return ts / tv if tv else None
    return None


def _sim_mutation_model(spec: dict) -> msprime.MatrixMutationModel:
    """Mutation model a sim spec drives ``msprime.sim_mutations`` with.

    :param spec: Simulation spec, either ``{"model": "HKY", "kappa": κ}``
        with κ the transition/transversion rate ratio and msprime's
        default uniform equilibrium frequencies, or ``{"model": "GTR",
        "rates": r, "pi": π}`` with ``r`` the six exchangeabilities in
        msprime's ``(AC, AG, AT, CG, CT, GT)`` order and ``π`` the four
        equilibrium frequencies over ``(A, C, G, T)``.
    :return: The msprime mutation model built from ``spec``.
    :raises ValueError: If ``spec`` names a model outside ``{HKY, GTR}``.
    """
    if spec.get("model") == "HKY":
        return msprime.HKY(kappa=float(spec["kappa"]))
    if spec.get("model") == "GTR":
        return msprime.GTR(
            relative_rates=[float(r) for r in spec["rates"]],
            equilibrium_frequencies=[float(x) for x in spec["pi"]],
        )
    raise ValueError(f"unknown sim model {spec.get('model')!r}")


def _path_for(prefix, paths, sim, model, prior, n):
    needle = f"{prefix}_{sim}_{model}_{prior}_n{n}.json"
    return next((p for p in paths if p.endswith(needle)), None)


def _baseline_path_for(paths, sim, model, n):
    """Baseline JSON paths omit the prior wildcard. Locate by (sim, model, n)."""
    needle = f"estsfs_baseline_{sim}_{model}_n{n}.json"
    return next((p for p in paths if p.endswith(needle)), None)


cells: dict[tuple[str, str, str, int], dict] = {}
for s in sims:
    for m in models:
        for p in priors:
            for n in n_outs:
                anc_path = _path_for("estsfs_ancestree", in_ancestree_paths, s, m, p, n)
                fd_path = _path_for("estsfs_fastdfe", in_fastdfe_paths, s, m, p, n)
                base_path = _baseline_path_for(in_baseline_paths, s, m, n)
                # fastDFE doesn't implement R6 — its cell may be absent
                # for that model. Carry None through. Downstream consumers
                # gracefully skip the column when fastdfe is missing.
                cell = {
                    "ancestree": _load_cell(anc_path),
                    "fastdfe": _load_cell(fd_path) if fd_path else None,
                }
                if base_path is not None:
                    cell["baseline"] = _load_cell(base_path)
                cells[(s, m, p, n)] = cell

with open(in_meta) as f:
    sim_meta = json.load(f)

# ------------------------------------------------------ per-site ground truth

ts = tskit.load(in_trees)
sample_nodes = list(ts.samples())
n_ingroup = int(sim_meta["n_ingroup"])
n_outgroup = int(sim_meta["n_outgroup"])
outgroup_divergences_sim = list(sim_meta["outgroup_divergences"])
outgroup_names_sim = list(sim_meta["outgroup_names"])

# Under ploidy=1 the simulator's first n_ingroup individuals are the ingroup
# samples. The next n_outgroup are the outgroups.
ingroup_node_ids: list[int] = []
outgroup_node_id_by_name: dict[str, int] = {}
for ind in ts.individuals():
    if ind.id < n_ingroup:
        ingroup_node_ids.extend(int(n) for n in ind.nodes)
    elif ind.id < n_ingroup + n_outgroup:
        # ploidy=1 → one node per individual
        outgroup_node_id_by_name[f"tsk_{ind.id}"] = int(ind.nodes[0])
outgroup_node_ids: set[int] = set(outgroup_node_id_by_name.values())
ingroup_indices = [
    i for i, n in enumerate(sample_nodes) if int(n) not in outgroup_node_ids
]

def _sum_branch_lengths(tree, u, v) -> float | None:
    """Sum of branch lengths along the path from u to v.

    tskit.Tree.path_length returns the **node count**, not the sum of
    branch lengths — hence this manual walk.
    """
    mrca = tree.mrca(u, v)
    if mrca == tskit.NULL:
        return None
    total = 0.0
    for start in (u, v):
        n = start
        while n != mrca and n != tskit.NULL:
            total += float(tree.branch_length(n))
            n = tree.parent(n)
    return total


# Empirical I → O_k path length: per local ARG segment, find the ingroup
# MRCA, sum branch lengths from I to each outgroup tip, weight by span.
empirical_path_gens: dict[str, float] = {}
mu = float(sim_meta["mu"])
total_span = 0.0
sum_per_og: dict[str, float] = {og: 0.0 for og in outgroup_node_id_by_name}
for tree in ts.trees():
    if tree.num_roots != 1:
        continue
    ingroup_mrca = tree.mrca(*ingroup_node_ids)
    if ingroup_mrca == tskit.NULL:
        continue
    span = float(tree.span)
    for og_name, og_node in outgroup_node_id_by_name.items():
        path_len = _sum_branch_lengths(tree, ingroup_mrca, og_node)
        if path_len is None:
            continue
        sum_per_og[og_name] += path_len * span
    total_span += span
for og_name in outgroup_node_id_by_name:
    empirical_path_gens[og_name] = (
        sum_per_og[og_name] / total_span if total_span else float("nan")
    )
# ``mu`` is a per-site per-generation rate of mutation EVENTS, while the
# fitted branch rates are expected substitutions per site. Each sim's
# path length is scaled by rho, the expected substitutions per mutation
# event under that sim's own mutation model.
sim_rho = {
    s: substitutions_per_mutation(_sim_mutation_model(sim_specs.get(s, {})))
    for s in sims
}
empirical_truth_div = {
    s: {
        og: empirical_path_gens[og] * mu * rho
        for og in outgroup_node_id_by_name
    }
    for s, rho in sim_rho.items()
}

par_info: dict[int, dict] = {}
variants_by_site = {v.site.id: v for v in ts.variants()}
for tree in ts.trees():
    if tree.num_roots != 1:
        continue
    for ts_site in tree.sites():
        variant = variants_by_site[ts_site.id]
        alleles = list(variant.alleles)
        gens = variant.genotypes
        if any(g < 0 for g in gens) or any(alleles[g] not in STATE_INDEX for g in gens):
            par_info[int(ts_site.position)] = {"missing": True}
            continue
        ingroup_gens = [int(gens[i]) for i in ingroup_indices]
        # Minor-allele count over the ingroup, taken across every segregating
        # allele, so a triallelic site counts as polymorphic whichever pair of
        # alleles the ingroup happens to carry.
        n_minor = len(ingroup_gens) - max(
            ingroup_gens.count(g) for g in set(ingroup_gens)
        )
        states = [STATE_INDEX[alleles[g]] for g in gens]
        _anc, muts = tree.map_mutations(states, alleles=STATES)
        par_info[int(ts_site.position)] = {
            "truth": ts_site.ancestral_state,
            "parsimony_score": len(muts),
            "ingroup_minor_count": n_minor,
            "missing": False,
        }


# ------------------------------------------------------- per-cell aggregation

def _truth_recovery(per_site: dict[int, dict]) -> tuple[float | None, int]:
    """Return (accuracy, n_evaluated) over INGROUP-POLYMORPHIC sites with
    a known truth. Ingroup-monomorphic sites are excluded — that's the
    convention used throughout the manuscript (\\S~\\ref{sec:benchmarks},
    "Setup compared to the baselines"): the monomorphic bulk is either
    trivially resolved (truth = observed ingroup allele) or
    unrecoverable from ingroup data alone (truth = the lone other
    allele), so including it inflates / distorts the comparison."""
    n_eval = correct = 0
    for pos, p in per_site.items():
        info = par_info.get(pos, {})
        if info.get("missing") or info.get("truth") is None:
            continue
        if int(info.get("ingroup_minor_count", 0)) == 0:
            continue  # ingroup-monomorphic — exclude per benchmark convention
        n_eval += 1
        if p["map_allele"] == info["truth"]:
            correct += 1
    return (correct / n_eval if n_eval else None, n_eval)


def _aggregate(anc_per_site: dict[int, dict], fd_per_site: dict[int, dict]) -> dict:
    shared = sorted(set(anc_per_site) & set(fd_per_site))
    n_nonhom = n_hom = 0
    nonhom_agree = hom_agree = 0
    post_diff: list[float] = []
    post_diff_when_agree: list[float] = []
    for pos in shared:
        a = anc_per_site[pos]
        b = fd_per_site[pos]
        info = par_info.get(pos, {})
        if info.get("missing"):
            continue
        score = info.get("parsimony_score", 0)
        agree = a["map_allele"] == b["map_allele"]
        if score <= 1:
            n_nonhom += 1
            nonhom_agree += int(agree)
        else:
            n_hom += 1
            hom_agree += int(agree)
        if b["map_allele"] in STATE_INDEX:
            anc_at_fd_map = a["posterior"][STATE_INDEX[b["map_allele"]]]
            fd_prob = float(b["max_prob"])
            # fastDFE can return ``nan`` max_prob for edge-case monomorphic
            # rows (e.g. a single allele observed across all samples). Skip
            # those rather than poisoning np.mean.
            if not np.isfinite(fd_prob) or not np.isfinite(anc_at_fd_map):
                continue
            d = abs(fd_prob - float(anc_at_fd_map))
            post_diff.append(d)
            if agree:
                post_diff_when_agree.append(d)
    return {
        "n_shared": len(shared),
        "n_nonhom": n_nonhom, "n_hom": n_hom,
        "nonhom_agree_rate": nonhom_agree / n_nonhom if n_nonhom else None,
        "hom_agree_rate": hom_agree / n_hom if n_hom else None,
        "post_mean_abs_diff": float(np.mean(post_diff)) if post_diff else None,
        "post_max_abs_diff": float(np.max(post_diff)) if post_diff else None,
        "post_mean_abs_diff_when_agree":
            float(np.mean(post_diff_when_agree)) if post_diff_when_agree else None,
        "post_max_abs_diff_when_agree":
            float(np.max(post_diff_when_agree)) if post_diff_when_agree else None,
    }


by_cell: dict[str, dict] = {}
for (s, m, p, n), data in cells.items():
    anc_sites = {int(k): v for k, v in data["ancestree"]["sites"].items()}
    fastdfe_data = data.get("fastdfe")
    if fastdfe_data is not None:
        fd_sites = {int(site["pos"]): site for site in fastdfe_data["sites"]}
        fd_acc, fd_n = _truth_recovery(fd_sites)
        fd_agree = _aggregate(anc_sites, fd_sites)
        fd_meta = fastdfe_data.get("inference_meta", {})
    else:
        fd_sites, fd_acc, fd_n, fd_agree, fd_meta = None, None, 0, None, {}
    anc_acc, anc_n = _truth_recovery(anc_sites)
    baseline_meta = (data.get("baseline") or {}).get("meta", {})
    by_cell[f"{s}|{m}|{p}|n{n}"] = {
        "agreement": fd_agree,
        "ancestree_truth_recovery": anc_acc,
        "ancestree_n_truth_eval": anc_n,
        "fastdfe_truth_recovery": fd_acc,
        "fastdfe_n_truth_eval": fd_n,
        "ancestree_meta": data["ancestree"].get("inference_meta", {}),
        "fastdfe_meta": fd_meta,
        "baseline_meta": baseline_meta,
        "inference_seconds_ancestree":
            data["ancestree"].get("inference_meta", {}).get("inference_seconds"),
        "inference_seconds_estsfs": baseline_meta.get("inference_seconds"),
        "inference_seconds_fastdfe": fd_meta.get("inference_seconds"),
    }

# Ground truth = tskit-computed empirical I→O_k path lengths.
# Analytic expectations (μ·t_split, with-vs-without ancestral coalescent
# corrections) are expectations over many replicates and don't matter
# for the single-replicate sanity check we're doing here.
truth_div_empirical = {
    s: [
        empirical_truth_div[s].get(og, float("nan"))
        for og in outgroup_names_sim
    ]
    for s in sims
}

report = {
    "config": config,
    "sims": sims,
    "sim_specs": sim_specs,
    "sim_kappas_effective": {s: _truth_kappa(sim_specs.get(s, {})) for s in sims},
    "models": models,
    "priors": priors,
    "n_outs": n_outs,
    "truth_outgroup_path_gens": [
        empirical_path_gens.get(og, float("nan")) for og in outgroup_names_sim
    ],
    "sim_substitutions_per_mutation": sim_rho,
    "truth_outgroup_divergences_empirical": truth_div_empirical,
    "cells": by_cell,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {out_json}", flush=True)


# ------------------------------------------------------------------- markdown

def _fmt_rate(x): return "—" if x is None else f"{x:.4f} ({100*x:.2f}%)"
def _fmt_diff(x): return "—" if x is None else f"{x:.3e}"
def _fmt_secs(x): return "—" if x is None else f"{x:.3f}"
def _fmt_div(div):
    if div is None:
        return "—"
    return ", ".join(f"{d:.3e}" for d in div)


cfg = config
agree_rows = []
post_rows = []
truth_rows = []
rates_rows = []
kappa_rows = []
runtime_rows = []
for s in sims:
    for n in n_outs:
        for m in models:
            for p in priors:
                c = by_cell[f"{s}|{m}|{p}|n{n}"]
                ag = c["agreement"]
                # Agreement is computed between Ancestree and fastDFE. For
                # cells where fastDFE didn't run (R6) it's None and the row
                # carries dashes rather than crashing the report.
                if ag is None:
                    agree_rows.append(
                        f"| {s} | {n} | {m} | {p} | — | — | — | — |"
                    )
                    post_rows.append(
                        f"| {s} | {n} | {m} | {p} | — | — | — | — |"
                    )
                else:
                    agree_rows.append(
                        f"| {s} | {n} | {m} | {p} | {ag['n_shared']} | "
                        f"{ag['n_nonhom']} / {ag['n_hom']} | "
                        f"{_fmt_rate(ag['nonhom_agree_rate'])} | "
                        f"{_fmt_rate(ag['hom_agree_rate'])} |"
                    )
                    post_rows.append(
                        f"| {s} | {n} | {m} | {p} | "
                        f"{_fmt_diff(ag['post_mean_abs_diff_when_agree'])} | "
                        f"{_fmt_diff(ag['post_max_abs_diff_when_agree'])} | "
                        f"{_fmt_diff(ag['post_mean_abs_diff'])} | "
                        f"{_fmt_diff(ag['post_max_abs_diff'])} |"
                    )
                truth_rows.append(
                    f"| {s} | {n} | {m} | {p} | "
                    f"{_fmt_rate(c['ancestree_truth_recovery'])} | "
                    f"{_fmt_rate(c['fastdfe_truth_recovery'])} |"
                )
                anc_div = c["ancestree_meta"].get("outgroup_divergence")
                fd_div = c["fastdfe_meta"].get("outgroup_divergence")
                # Closest-first subsetting for the n_out=n cells.
                truth_subset = truth_div_empirical[s][:n]
                rates_rows.append(
                    f"| {s} | {n} | {m} | {p} | {_fmt_div(truth_subset)} | "
                    f"{_fmt_div(anc_div)} | {_fmt_div(fd_div)} |"
                )
                if m == "K2":
                    anc_kappa = (
                        (c["ancestree_meta"].get("params_mle") or {}).get("kappa")
                    )
                    fd_kappa = (
                        (c["fastdfe_meta"].get("params_mle") or {}).get("k")
                    )
                    truth_kappa = _truth_kappa(sim_specs.get(s, {}))
                    kappa_rows.append(
                        f"| {s} | {n} | {p} | "
                        f"{'—' if truth_kappa is None else f'{truth_kappa:.4f}'} | "
                        f"{'—' if anc_kappa is None else f'{anc_kappa:.4f}'} | "
                        f"{'—' if fd_kappa is None else f'{fd_kappa:.4f}'} |"
                    )
                # Runtime is independent of prior on the baseline side
                # (est-sfs baseline has no prior wildcard), so emit only one
                # row per (sim, n_out, model) — pick the first prior for the
                # est-sfs column.
                if p == priors[0]:
                    runtime_rows.append(
                        f"| {s} | {n} | {m} | "
                        f"{_fmt_secs(c['inference_seconds_ancestree'])} | "
                        f"{_fmt_secs(c['inference_seconds_estsfs'])} | "
                        f"{_fmt_secs(c['inference_seconds_fastdfe'])} |"
                    )


per_k_rows = []
for s in sims:
    for m in models:
        for p in priors:
            for meta_key, tool_label in [
                ("ancestree_meta", "Ancestree"),
                ("fastdfe_meta", "fastDFE"),
            ]:
                params = (
                    by_cell[f"{s}|{m}|{p}|n3"][meta_key].get("params_mle") or {}
                )
                k_vals = " | ".join(
                    f"{params.get(f'K{i}', float('nan')):.3e}" for i in range(5)
                )
                per_k_rows.append(f"| {s} | {m} | {p} | {k_vals} | {tool_label} |")
per_k_block = "\n".join(per_k_rows)


# ---- R6 / GTR rate-recovery rows (Ancestree-only — fastDFE has no R6) ----
# Both truth and fitted are displayed in rate_AG = 1 normalisation (the
# reference scale Ancestree's R6 fit anchors to). HKY sims collapse to
# (1, κ, 1, 1, κ, 1) in GTR form → divide by κ to land on rate_AG=1:
# (1/κ, 1, 1/κ, 1/κ, 1, 1/κ).
gtr_rate_rows = []
for s in sims:
    spec = sim_specs.get(s, {})
    if spec.get("model") == "GTR":
        raw_truth = list(spec["rates"])
    elif spec.get("model") == "HKY":
        kappa = float(spec.get("kappa", 1.0))
        raw_truth = [1.0, kappa, 1.0, 1.0, kappa, 1.0]
    else:
        raw_truth = [float("nan")] * 6
    # Normalise to rate_AG = 1 (index 1).
    ag_truth = raw_truth[1] if raw_truth[1] not in (0, 0.0) else 1.0
    truth_rates = [t / ag_truth for t in raw_truth]
    for p in priors:
        for n in n_outs:
            cell = by_cell.get(f"{s}|R6|{p}|n{n}")
            if cell is None:
                continue
            params = (cell["ancestree_meta"].get("params_mle") or {})
            # rate_AG is the reference (held at 1.0 during the joint MLE)
            # and not reported by free_params. Inject it back at the
            # canonical position so the row keeps all 6 columns populated.
            fitted = [
                1.0 if name == "rate_AG"
                else params.get(name, float("nan"))
                for name in GTR.RATE_NAMES
            ]
            truth_str = " | ".join(f"{t:.3f}" for t in truth_rates)
            fit_str = " | ".join(f"{f:.3f}" for f in fitted)
            gtr_rate_rows.append(
                f"| {s} | {n} | {p} | {truth_str} | {fit_str} |"
            )
gtr_rate_block = "\n".join(gtr_rate_rows)


path_gens_str = ", ".join(
    f"{empirical_path_gens.get(og, float('nan')):.4g}" for og in outgroup_names_sim
)
truth_div_block = "\n".join(
    f"| {s} | {sim_rho[s]:.4f} | "
    + " | ".join(f"{d:.3e}" for d in truth_div_empirical[s])
    + " |"
    for s in sims
)


md = f"""# B3 — Ancestree (FixedTreeInference) vs fastDFE est-sfs

## Simulation matrix
- ploidy: `1` (haploid throughout)
- ingroup: `{cfg['n_ingroup']}` haps (Ne={cfg['pop_size']:g})
- outgroups: 3 nested splits at `{cfg['outgroup_divergences']}` gens
  (each outgroup contributes 1 haploid sample, Ne_out={cfg['outgroup_pop_size']:g})
- sequence length: `{cfg['length']:g}`
- mutation rate: `{cfg['mu']:g}` per-site per-gen
- recombination rate: `{cfg['rec_rate']:g}`
- sim sweeps: ``sim`` ∈ ``{sims}``. Per-sim Ts/Tv (effective κ): {report["sim_kappas_effective"]}.
  The ``gtr`` sim uses skewed equilibrium π=(0.3, 0.2, 0.2, 0.3) on top of
  transition-biased exchangeability rates.
- inference subsets to ``n_out`` ∈ ``{n_outs}`` closest outgroups
- seed: `{cfg['seed']}`

## Ground truth (`I → outgroup_k` divergence, substitutions per site)
Span-weighted average of the `I → O_k` branch-length sum over local
ARG segments, taken directly from the simulated tskit trees (so it
includes the actual realised coalescent depths in this replicate). The
span-weighted path lengths are `{path_gens_str}` generations for
(O_1, O_2, O_3), shared by every sim since all sims use the same
demography and seed.

Generations are converted to substitutions per site by `mu * rho`,
where `mu = {cfg['mu']:g}` is the per-site per-generation rate of
mutation events and `rho` is the expected number of substitutions per
mutation event under that sim's own mutation model. ``msprime``
normalises its transition matrix by the largest row sum, so an event
leaves the state unchanged with probability `P_ii` and
`rho = 1 - sum_i pi_i P_ii`, with `pi_i` the equilibrium frequency of
state `i`. The fitted branch rates carry the same units.

| sim | rho (subs per mutation event) | O_1 (closest) | O_2 | O_3 (farthest) |
|---|---|---|---|---|
{truth_div_block}


## MAP-allele agreement (Ancestree vs fastDFE)

| sim | n_out | model | prior | shared | non-hom / hom | agreement (non-hom) | agreement (hom) |
|---|---|---|---|---|---|---|---|
{chr(10).join(agree_rows)}

## Posterior numerical agreement

| sim | n_out | model | prior | mean diff (agree only) | max diff (agree only) | mean diff (all) | max diff (all) |
|---|---|---|---|---|---|---|---|
{chr(10).join(post_rows)}

## Truth recovery (msprime `Site.ancestral_state`)

| sim | n_out | model | prior | Ancestree | fastDFE |
|---|---|---|---|---|---|
{chr(10).join(truth_rows)}

## Branch rates vs ground truth (ingroup MRCA → each outgroup)

| sim | n_out | model | prior | empirical (tskit, subs/site) | Ancestree fitted | fastDFE fitted |
|---|---|---|---|---|---|---|
{chr(10).join(rates_rows)}

## Per-K branch rates (n_out=3 cells, both tools)

The optimiser variable vector `[K_0, K_1, K_2, K_3, K_4]` parameterises
the ladder branches in the est-sfs topology:
``I →(K_0) n_1 →(K_1) O_1`` and ``n_1 →(K_2) n_2 →(K_3,K_4) (O_2, O_3)``.
fastDFE uses the same naming.

| sim | model | prior | K_0 (I→n_1) | K_1 (n_1→O_1) | K_2 (n_1→n_2) | K_3 (n_2→O_2) | K_4 (n_2→O_3) | tool |
|---|---|---|---|---|---|---|---|---|
{per_k_block}

## Fitted κ vs simulation truth (K2 model cells)

Effective transition/transversion ratio. HKY sims display the literal κ.
The ``gtr`` sim displays the on-paper Ts/Tv ratio implied by its
exchangeability rates (transversion-symmetric construction).

| sim | n_out | prior | truth κ | Ancestree fitted κ | fastDFE fitted κ |
|---|---|---|---|---|---|
{chr(10).join(kappa_rows)}

## Fitted GTR exchangeability rates vs simulation truth (R6 cells, Ancestree)

Truth rates use the GTR parameterisation. For HKY sims the truth column
shows the collapsed-equivalent (transitions=κ, transversions=1). For the
``gtr`` sim it is the literal simulator rate vector. All rates are scaled
so ``rate_AG = 1`` (Ancestree's reference scale during the joint MLE).

| sim | n_out | prior | r_AC (t) | r_AG (t) | r_AT (t) | r_CG (t) | r_CT (t) | r_GT (t) | r_AC (f) | r_AG (f) | r_AT (f) | r_CG (f) | r_CT (f) | r_GT (f) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{gtr_rate_block}

## Inference wall-clock runtime (seconds, inference step only, excluding I/O)

Runtime is reported per (sim, n_out, model). The est-sfs baseline has no
prior wildcard, so we list one row per Ancestree prior at the first entry
in ``priors`` (other priors run at near-identical cost).

| sim | n_out | model | Ancestree (s) | est-sfs (s) | fastDFE (s) |
|---|---|---|---|---|---|
{chr(10).join(runtime_rows)}
"""

with open(out_md, "w") as f:
    f.write(md)
print(f"Wrote {out_md}", flush=True)
