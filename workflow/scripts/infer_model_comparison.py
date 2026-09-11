"""B9 model-comparison benchmark inference: one (model, mode) cell.

Loads the B9 simulation (``model_comparison_sim.trees`` — mutated under
HKY(κ=4, AT-rich π)) and runs ancestral-allele inference under one of
four Ancestree substitution models (``JC69``, ``K2``, ``F81``, ``HKY``)
in one of two modes (``arg`` / ``vcf``). Writes per-site posteriors plus
the simulator's truth ancestral state in a JSON schema matching B6 /
B8 (so :mod:`report_model_comparison` can stratify by SFS bin with the
same code path).

Wildcards:

- ``{model}`` ∈ ``{jc69, k2, f81, hky, gtr}`` — which Ancestree
  substitution model the kernel sees. ``jc69`` is the fully
  mis-specified baseline; ``hky`` matches the simulator and should
  recover near-truth accuracy; ``gtr`` over-parameterises the same truth
  with 6 free exchangeability rates.
- ``{mode}`` ∈ ``{arg, vcf}`` — ARG-mode runs
  :class:`~ancestree.inference.ARGBasedInference` on the full ts;
  VCF-mode dumps a temporary VCF and runs
  :class:`~ancestree.inference.FixedTreeInference` with the three
  outgroups in a closest-first ladder.

Empirical helpers (:meth:`~ancestree.sites.BaseComposition.from_polymorphic_sites`,
:meth:`~ancestree.models.K2.estimate_kappa_from_data`) preload model
parameters from the data. For ``K2``/``HKY`` in VCF mode ``fit_kappa=True``
additionally refines κ jointly with the branch rates inside
:class:`FixedTreeInference`'s L-BFGS-B fit. In ARG mode the kernel
consumes the model parameters as-is — no MLE pass.

Run directly (defaults to ``model=hky, mode=arg``)::

    python workflow/scripts/infer_model_comparison.py

Or via snakemake::

    snakemake -j 1 results/data/model_comparison_hky_arg.json
"""

import json
import tempfile
from pathlib import Path

import tskit

from ancestree import (
    ARGBasedInference,
    BaseComposition,
    F81,
    GTR,
    HKY,
    Inference,
    JC69,
    K2,
    KingmanIngroupWeight,
    MajorityOutgroupInference,
    TskitLocalTree,
    TskitSource,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    model_name = str(snakemake.wildcards.model)  # type: ignore[name-defined]
    mode = str(snakemake.wildcards.mode)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/model_comparison_sim.trees"
    in_meta = "results/data/model_comparison_sim_meta.json"
    model_name = "hky"
    mode = "arg"
    out_json = f"results/data/model_comparison_{model_name}_{mode}.json"
    mu = 1.25e-8


VALID_MODELS = {"jc69", "k2", "f81", "hky", "gtr"}
VALID_MODES = {"arg", "vcf"}
if model_name not in VALID_MODELS:
    raise ValueError(f"Unknown model {model_name!r}; expected one of {sorted(VALID_MODELS)}")
if mode not in VALID_MODES:
    raise ValueError(f"Unknown mode {mode!r}; expected one of {sorted(VALID_MODES)}")


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
ordered_outgroup_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
outgroup_names: list[str] = [
    name for pop in ordered_outgroup_pops for name in outgroups_by_pop[pop]
]

ts = tskit.load(in_trees)
sample_id_to_node: dict[str, int] = {
    f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()
}
keep_sample_names = ingroup_names + outgroup_names
keep_node_ids = [sample_id_to_node[s] for s in keep_sample_names]
ts_sub = ts.simplify(samples=keep_node_ids, filter_sites=False)
sample_map = {s: i for i, s in enumerate(keep_sample_names)}

ingroup_name_set = set(ingroup_names)


def _build_model(name: str, mode: str, ts_for_sites: tskit.TreeSequence):
    """Instantiate the named Ancestree substitution model + base composition.

    The model carries only structural parameters (``κ`` for K2/HKY). Per-data
    ``π`` lives on the :class:`~ancestree.sites.BaseComposition` built here from
    one pass over the sites.

    Empirical-parameter routes (``BaseComposition.from_polymorphic_sites``,
    ``K2.estimate_kappa_from_data``) preload the per-data parameters.
    In VCF mode ``fit_kappa=True`` is set on K2/HKY so
    :class:`FixedTreeInference`'s L-BFGS-B refines κ jointly with the
    branch rates. In ARG mode no parameters are fitted. The kernel
    consumes whatever was loaded.

    :param name: One of ``jc69 | k2 | f81 | hky | gtr``.
    :param mode: ``arg`` or ``vcf``.
    :param ts_for_sites: TreeSequence to pull sites from for the
        empirical pi / kappa estimation.
    :return: ``(model, base_composition, empirical_params)`` triple —
        the third is a small dict for the per-cell meta.
    """
    fit_kappa = (mode == "vcf")
    # One-pass empirical base composition + Ts/Tv. Capped at 10k sites
    # to match the previous ``max_sites=10_000`` calibration default.
    sites_for_cal = list(TskitSource(ts_for_sites, sample_map=sample_map))
    bc = BaseComposition.from_polymorphic_sites(sites_for_cal, max_sites=10_000)
    if name == "jc69":
        return JC69(), bc, {"kappa": 1.0}
    if name == "k2":
        # Empirical κ from majority-ingroup vs majority-outgroup Ts/Tv ratio
        # (the low-divergence MLE). Build the Site list from the same ts
        # the inference will consume so the κ̂ matches the sites the kernel sees.
        try:
            kappa_emp = K2.estimate_kappa_from_data(
                sites_for_cal, ingroup_names, outgroup_names,
            )
        except ValueError:
            kappa_emp = 2.0  # fallback: neutral default
        # Cap to the default upper bound so the MLE start point is finite.
        kappa_init = min(max(kappa_emp, 0.1), 50.0)
        return K2(kappa=kappa_init, fit_kappa=fit_kappa), bc, {"kappa": kappa_init}
    if name == "f81":
        return F81(), bc, {"pi": bc.pi.tolist()}
    if name == "hky":
        kappa_init = min(max(bc.kappa_estimate, 0.1), 50.0)
        return (
            HKY(kappa=kappa_init, fit_kappa=fit_kappa),
            bc,
            {"pi": bc.pi.tolist(), "kappa": kappa_init},
        )
    if name == "gtr":
        # Empirical 6-rate exchangeability estimator (analogue of HKY's
        # kappa_estimate). The joint MLE then refines under fit_rates=True.
        # ARG mode skips fitting and just consumes the empirical rates.
        try:
            gtr_emp = GTR.empirical_from_sites(
                sites_for_cal, bc, ingroup_names, fit_rates=fit_kappa,
            )
        except ValueError:
            # Fall back to F81-like uniform rates when a pair is unsampled.
            gtr_emp = GTR(
                rates=[1.0] * 6, fit_rates=fit_kappa,
            )
        return (
            gtr_emp,
            bc,
            {"pi": bc.pi.tolist(), "rates": gtr_emp.rates.tolist()},
        )
    raise AssertionError(f"unhandled model name {name!r}")


# No more mode-specific mu workaround — `mu` is owned by the Inference
# layer (ARG passes mu to ARGBasedInference; FixedTreeInference's K_i
# parameters are already in subs/site so mu doesn't apply there).
model, base_composition, empirical_params = _build_model(model_name, mode, ts_sub)

truth_by_pos: dict[int, str] = {
    int(site.position): site.ancestral_state for site in ts.sites()
}


def _ingroup_stats(site_tip_alleles: dict[str, str | None]) -> tuple[bool, int]:
    """Compute (ingroup_polymorphic, minor_count) — schema match B8."""
    counts: dict[str, int] = {}
    for s, a in site_tip_alleles.items():
        if s in ingroup_name_set and a is not None:
            counts[a] = counts.get(a, 0) + 1
    if len(counts) <= 1:
        return False, 0
    sorted_counts = sorted(counts.values(), reverse=True)
    return True, int(sorted_counts[1])


print(
    f"Model-comparison: model={model_name}, mode={mode}, "
    f"{ts_sub.num_samples} haplotypes, {ts_sub.num_sites} sites, "
    f"empirical_params={empirical_params}",
    flush=True,
)

results: dict[int, dict] = {}
sites_seen: list = []
fitted_outgroup_divergence = None
fitted_kappa = None

if mode == "arg":
    inference = ARGBasedInference(
        ts_sub, model,
        mu=mu,
        base_composition=base_composition,
        sample_map=sample_map,
        progress=False,
    )
    for site, posterior in inference.infer():
        pos = int(site.pos)
        truth = truth_by_pos.get(pos)
        ingroup_poly, minor_count = _ingroup_stats(site.tip_alleles)
        results[pos] = {
            "map_allele": posterior.map_allele,
            "max_prob": float(posterior.max_prob),
            "posterior": [float(p) for p in posterior.values],
            "alleles": list(site.alleles),
            "truth": truth,
            "map_correct": truth is not None and posterior.map_allele == truth,
            "ingroup_polymorphic": bool(ingroup_poly),
            "minor_count": minor_count,
        }
        sites_seen.append(site)
else:  # vcf
    nodes = TskitLocalTree.sample_map_from_individuals(ts) or {
        f"tsk_{int(s)}": int(s) for s in ts.samples()
    }
    ordered = ingroup_names + outgroup_names
    work = Path(tempfile.mkdtemp(prefix="model_comparison_vcf_"))
    in_vcf = work / "snps.vcf"
    with open(in_vcf, "w") as f:
        ts.write_vcf(
            f,
            contig_id="1",
            individual_names=ordered,
            individuals=[nodes[n] for n in ordered],
            allow_position_zero=True,
        )

    # Ingroup only. Base_composition is passed to the inference below.
    ingroup_weight = KingmanIngroupWeight(ingroup_samples=ingroup_names)
    vcf_inf = Inference.from_fixed_tree(
        in_vcf,
        ingroup_samples=ingroup_names,
        outgroup_samples=outgroup_names,
        model=model,
        base_composition=base_composition,
        n_target_sites=int(ts.sequence_length),
        ingroup_weight=ingroup_weight,
        parallelize=False,
        progress=False,
    )
    vcf_inf.fit()
    fitted_outgroup_divergence = (
        vcf_inf.outgroup_divergence_mle.tolist()
        if vcf_inf.outgroup_divergence_mle is not None else None
    )
    # If the model carries a fitted kappa, snapshot it after fit().
    if hasattr(model, "kappa"):
        fitted_kappa = float(model.kappa)

    for site, post_s in vcf_inf.infer():
        pos = int(site.pos)
        truth = truth_by_pos.get(pos)
        ingroup_poly, minor_count = _ingroup_stats(site.tip_alleles)
        results[pos] = {
            "map_allele": post_s.map_allele,
            "max_prob": float(post_s.max_prob),
            "posterior": [float(p) for p in post_s.values],
            "alleles": list(site.alleles),
            "truth": truth,
            "map_correct": truth is not None and post_s.map_allele == truth,
            "ingroup_polymorphic": bool(ingroup_poly),
            "minor_count": minor_count,
        }
        sites_seen.append(site)

# B11: optional baseline_map pass — MajorityOutgroupInference on the same
# sites. B9 always has 3 outgroups so the baseline is always applicable.
if outgroup_names and sites_seen:
    baseline = MajorityOutgroupInference(
        sites_seen, outgroup_names, for_comparison_only=True,
    )
    for site, post_b in baseline.infer():
        entry = results.get(int(site.pos))
        if entry is not None:
            entry["baseline_map"] = post_b.map_allele

inference_meta = {
    "model": model_name,
    "mode": mode,
    "mu": mu,
    "n_ingroup": len(ingroup_names),
    "n_outgroup_pops": len(ordered_outgroup_pops),
    "outgroup_pops_used": ordered_outgroup_pops,
    "outgroup_names_used": outgroup_names,
    "n_sites_inferred": len(results),
    "empirical_params": empirical_params,
    "fitted_kappa": fitted_kappa,
    "fitted_outgroup_divergence": fitted_outgroup_divergence,
    "sim_kappa": meta.get("kappa"),
    "sim_pi": meta.get("equilibrium_frequencies"),
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f, indent=2)

n_acc = sum(1 for r in results.values() if r["truth"] is not None)
n_hits = sum(1 for r in results.values() if r["map_correct"])
print(
    f"Wrote {out_json}: {len(results)} sites, "
    f"overall accuracy={n_hits / n_acc if n_acc else float('nan'):.4f}",
    flush=True,
)
