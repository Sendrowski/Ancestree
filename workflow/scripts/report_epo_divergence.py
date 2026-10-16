"""Human-branch divergence counts at the human-chimpanzee ancestor: the
Ensembl EPO ancestral calls against Ancestree on the same alignment and tree.

Over the chr1 positions at which the MSL ingroup is fixed, a site counts as
divergent where the state at the parent of the human sequence in the EPO
species tree differs from the allele the ingroup has fixed. EPO reports one
base per site with a confidence class. Ancestree is run on the ten aligned
sequences with the EPO species tree and its branch lengths as given, under
each model in ``MODELS``, and gives the count from its MAP estimates and from
its posterior weights. The branch lengths are used as given, multiplied by one
fitted scale, and multiplied by one fitted scale per CpG context class.
"""
import json
from pathlib import Path

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ancestree import (
    BaseComposition, FixedTree, FixedTreeInference, HKY, JC69, STATES, Site,
)
from ancestree.focal import FocalNode
from ancestree.likelihood import Likelihood

try:
    in_patterns = list(snakemake.input.patterns)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
except NameError:
    import glob
    in_patterns = sorted(glob.glob("results/data/epo_patterns_chr1_*.npz"))
    out_json = "results/reports/epo_divergence.json"
    out_md = "results/reports/epo_divergence.md"

#: The species tree of the Ensembl release-106 10-primate EPO alignment, in
#: substitutions per site, as printed in its README.
EPO_TREE = (
    "(((((((pan_troglodytes:0.00221173,pan_paniscus:0.00332827):0.00431379,"
    "homo_sapiens:0.00659621):0.00185825,gorilla_gorilla:0.00857049):0.00847976,"
    "pongo_abelii:0.016068):0.00267331,nomascus_leucogenys:0.0196815):0.0109481,"
    "((macaca_mulatta:0.00254698,macaca_fascicularis:0.00312302):0.0090058,"
    "chlorocebus_sabaeus:0.0117108):0.01846908):0.0454214526,"
    "microcebus_murinus:0.07593699);"
)
HUMAN = "homo_sapiens"
#: Stationary frequencies of A, C, G, T at 40% GC, the Ortheus default.
GC40 = np.array([0.3, 0.2, 0.2, 0.3])
#: Model name to (model, whether it reads the 40% GC composition). Ortheus
#: documents HKY with a transition/transversion ratio of 2 at 40% GC:
#: kappa = 2 reads the ratio as the rate parameter, kappa = 25/6 as the
#: expected ratio of the two counts.
MODELS = {
    "HKY, kappa = 2, 40% GC": (lambda: HKY(kappa=2.0), True),
    "HKY, kappa = 25/6, 40% GC": (lambda: HKY(kappa=25.0 / 6.0), True),
    "JC69": (lambda: JC69(), False),
}
CONTEXTS = {0: "non-CpG", 1: "CpG", 2: "context unknown"}

# ---------------------------------------------------------------- patterns
tables = [np.load(p, allow_pickle=True) for p in in_patterns]
species_names = list(tables[0]["species_names"].astype(str))
human = np.concatenate([t["human"] for t in tables])
species = np.concatenate([t["species"] for t in tables])
epo = np.concatenate([t["epo"] for t in tables])
context = np.concatenate([t["context"] for t in tables])
count = np.concatenate([t["count"] for t in tables]).astype(float)

# One inference per distinct ten-sequence column.
tips = np.column_stack([human, species])
unique_tips, inverse = np.unique(tips, axis=0, return_inverse=True)
inverse = inverse.ravel()
sites = []
for k, row in enumerate(unique_tips):
    tip_alleles = {HUMAN: STATES[int(row[0])]}
    for name, a in zip(species_names, row[1:]):
        tip_alleles[name] = STATES[int(a)] if a < 4 else None
    alleles = tuple(dict.fromkeys(a for a in tip_alleles.values() if a))
    sites.append(Site(chrom="1", pos=k + 1, alleles=alleles, tip_alleles=tip_alleles))

epo_base = np.where(epo < 8, epo % 4, -1)
epo_upper = epo < 4
epo_called = epo < 8
epo_divergent = epo_called & (epo_base != human)

result = {"n_sites": float(count.sum()), "n_patterns": int(len(count)),
          "n_distinct_columns": int(len(unique_tips)), "models": {}}


def tally(mask, p_derived, map_derived):
    """Counts over the sites of ``mask``."""
    c = count * mask
    return {
        "n_sites": float(c.sum()),
        "epo_high_confidence_calls": float((c * epo_upper).sum()),
        "epo_low_confidence_calls": float((c * (epo_called & ~epo_upper)).sum()),
        "epo_no_call": float((c * ~epo_called).sum()),
        "divergent_epo_high_confidence": float((c * epo_divergent * epo_upper).sum()),
        "divergent_epo_all_calls": float((c * epo_divergent).sum()),
        "divergent_ancestree_map": float((c * map_derived).sum()),
        "divergent_ancestree_posterior_weighted": float((c * p_derived).sum()),
        "divergent_ancestree_map_where_epo_calls": float((c * map_derived * epo_called).sum()),
        "divergent_ancestree_posterior_weighted_where_epo_calls":
            float((c * p_derived * epo_called).sum()),
        "divergent_ancestree_map_where_epo_high_confidence":
            float((c * map_derived * epo_upper).sum()),
        "divergent_ancestree_posterior_weighted_where_epo_high_confidence":
            float((c * p_derived * epo_upper).sum()),
    }


composition = BaseComposition.from_n_target_sites(
    int(count.sum()), stationary=GC40)
rows = np.arange(len(human))
is_cpg = context == 1


def column_weights(mask):
    """Site count of each distinct column over the pattern rows of ``mask``."""
    return np.bincount(inverse, weights=count * mask, minlength=len(unique_tips))


def fitted_scale(make_model, reads_composition, weights):
    """ML multiplier of the EPO branch lengths for the columns weighted."""
    engine = Likelihood(
        make_model(), base_composition=composition if reads_composition else None)
    log_pi = np.log(GC40 if reads_composition else np.full(4, 0.25))

    def negative_log_likelihood(log_scale):
        tree = FixedTree.from_newick(EPO_TREE, time_scale=float(np.exp(log_scale)))
        log_L = engine.log_likelihoods(tree, sites)
        return -float((weights * logsumexp(log_L + log_pi, axis=1)).sum())

    best = minimize_scalar(negative_log_likelihood, bounds=(np.log(0.05), np.log(50.0)),
                           method="bounded", options={"xatol": 1e-3})
    return float(np.exp(best.x))


def column_posteriors(make_model, reads_composition, scale):
    """Posterior at the human-chimpanzee ancestor of each distinct column."""
    kwargs = {"base_composition": composition} if reads_composition else {}
    inference = FixedTreeInference(
        sites, make_model(), tree=FixedTree.from_newick(EPO_TREE, time_scale=scale),
        ingroup_samples=[HUMAN],
        focal=FocalNode("ingroup_mrca", coalescences=1),
        parallelize=False, progress=False, **kwargs)
    return np.array([p.values for _, p in inference.infer()])


for name, (make_model, reads_composition) in MODELS.items():
    one = fitted_scale(make_model, reads_composition, column_weights(np.ones(len(count), bool)))
    non_cpg = fitted_scale(make_model, reads_composition, column_weights(~is_cpg))
    cpg = fitted_scale(make_model, reads_composition, column_weights(is_cpg))
    #: Rate treatment to the branch-length multiplier of (non-CpG, CpG) sites.
    treatments = {
        "EPO branch lengths": (1.0, 1.0),
        "one fitted scale": (one, one),
        "one fitted scale per context": (non_cpg, cpg),
    }
    cache = {}
    result["models"][name] = {}
    for treatment, scales in treatments.items():
        for scale in scales:
            if scale not in cache:
                cache[scale] = column_posteriors(make_model, reads_composition, scale)
        posterior = np.where(is_cpg[:, None], cache[scales[1]][inverse],
                             cache[scales[0]][inverse])
        p_derived = 1.0 - posterior[rows, human]
        map_allele = posterior.argmax(1)
        map_derived = map_allele != human
        p_on_epo = np.where(epo_called, posterior[rows, np.maximum(epo_base, 0)], np.nan)
        entry = {"scale": {"non-CpG": scales[0], "CpG": scales[1]},
                 "all": tally(np.ones(len(human), bool), p_derived, map_derived)}
        for code, label in CONTEXTS.items():
            entry[label] = tally(context == code, p_derived, map_derived)
        for label, mask in (("high confidence", epo_upper),
                            ("low confidence", epo_called & ~epo_upper)):
            c = count * mask
            entry[f"agreement with EPO, {label}"] = {
                "n_sites": float(c.sum()),
                "map_equals_epo_base": float((c * (map_allele == epo_base)).sum() / c.sum()),
                "mean_posterior_on_epo_base": float(np.nansum(c * p_on_epo) / c.sum()),
            }
        result["models"][name][treatment] = entry
    print(name, {t: e["scale"] for t, e in result["models"][name].items()}, flush=True)

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")

lines = ["# Human-branch divergence at the human-chimpanzee ancestor, MSL chr1", ""]
for name, by_treatment in result["models"].items():
    for treatment, entry in by_treatment.items():
        scale = entry["scale"]
        lines += [f"## {name}, {treatment} "
                  f"(non-CpG x{scale['non-CpG']:.3f}, CpG x{scale['CpG']:.3f})", "",
                  "| context | sites | EPO high-confidence | EPO all calls | "
                  "Ancestree MAP | Ancestree posterior-weighted |",
                  "|---|---|---|---|---|---|"]
        for label in ("all", *CONTEXTS.values()):
            t = entry[label]
            lines.append(
                f"| {label} | {t['n_sites']:,.0f} | {t['divergent_epo_high_confidence']:,.0f} | "
                f"{t['divergent_epo_all_calls']:,.0f} | {t['divergent_ancestree_map']:,.0f} | "
                f"{t['divergent_ancestree_posterior_weighted']:,.0f} |")
        lines.append("")
Path(out_md).write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out_json} + {out_md}", flush=True)
