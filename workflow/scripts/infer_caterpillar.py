"""B10 caterpillar benchmark inference: ladder vs caterpillar on monoallelic-ingroup sites.

Two kernel-level modes, selected by the ``{tree_type}`` wildcard:

- ``ladder``: build a plain :class:`~ancestree.trees.OutgroupLadderTree`
  rooted at the ingroup MRCA. For monoallelic-ingroup sites this roots
  *on the derived side* of the only informative substitution and so
  reports the ingroup's own allele as MAP — wrong whenever the fixed
  allele is actually derived.
- ``caterpillar``: build a deep-rooted caterpillar
  rooted at the deepest outgroup ancestor (the ``((((ING, O_1), O_2),
  O_3)`` shape) and infer the deep-root state. Full Felsenstein over
  all outgroup tips can recover the ancestral allele.

Both modes use ``BaseComposition.no_counts()`` for the substitution model
and a uniform prior over states. Branch rates / node depths come from
the simulator's known split times scaled by ``mu`` (no MLE fit —
the comparison isolates the polariser-tree choice, not branch-rate
inference).

Output is filtered to **monoallelic-ingroup sites only**, the contrast
between the two trees is invisible on polymorphic-ingroup sites.

Wildcards:

- ``{tree_type}`` ∈ ``{ladder, caterpillar}``.

Run directly (defaults to ``caterpillar``)::

    python workflow/scripts/infer_caterpillar.py
"""
import json
from pathlib import Path

import numpy as np
import tskit

from ancestree import (
    BaseComposition,
    JC69,
    Likelihood,
    OutgroupLadderTree,
    STATE_INDEX,
    STATES,
    Site,
)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    tree_type = str(snakemake.wildcards.tree_type)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/caterpillar_sim.trees"
    in_meta = "results/data/caterpillar_sim_meta.json"
    tree_type = "caterpillar"
    out_json = f"results/data/caterpillar_{tree_type}.json"
    mu = 1.25e-8


VALID_TREE_TYPES = {"ladder", "caterpillar"}
if tree_type not in VALID_TREE_TYPES:
    raise ValueError(
        f"Unknown tree_type {tree_type!r}; expected one of {sorted(VALID_TREE_TYPES)}"
    )


with open(in_meta) as f:
    meta = json.load(f)
ingroup_names: list[str] = list(meta["ingroup_names"])
outgroups_by_pop: dict[str, list[str]] = dict(meta["outgroups_by_pop"])
ordered_outgroup_pops = sorted(outgroups_by_pop.keys(), key=lambda s: int(s.split("_")[1]))
outgroup_names: list[str] = [
    name for pop in ordered_outgroup_pops for name in outgroups_by_pop[pop]
]
outgroup_split_times: list[float] = list(meta["outgroup_split_times"])

# Path lengths I → O_k in expected subs/site, under the simulator's known
# split times + mu. For an ultrametric outgroup ladder the I → O_k path
# length equals the depth of the MRCA(I, O_k), which is the k-th split
# time. With ingroup-MRCA-rooted inference (OutgroupLadderTree), the
# fastDFE param vector reduces to the closest-first divergences directly.
outgroup_divergences = [t * mu for t in outgroup_split_times]

ts = tskit.load(in_trees)
sample_id_to_node: dict[str, int] = {
    f"tsk_{ind.id}": int(ind.nodes[0]) for ind in ts.individuals()
}
ingroup_node_set = {sample_id_to_node[s] for s in ingroup_names}
outgroup_name_set = set(outgroup_names)

truth_by_pos: dict[int, str] = {
    int(site.position): site.ancestral_state for site in ts.sites()
}

# Build per-site Site records carrying every ingroup + outgroup tip allele.
# Filter to monoallelic-ingroup sites — the two trees give identical answers
# on polymorphic-ingroup sites, so they're not the comparison the benchmark
# is after.
sample_nodes = ts.samples()
sample_id_per_node: dict[int, str] = {
    int(node_id): name for name, node_id in sample_id_to_node.items()
}

mono_sites: list[Site] = []
mono_truth_ancestral: list[str] = []
mono_ingroup_fixed_allele: list[str] = []
for variant in ts.variants():
    alleles = tuple(variant.alleles)
    gens = variant.genotypes
    pos = int(variant.site.position)
    truth = truth_by_pos.get(pos)
    if truth is None or truth not in STATES:
        continue
    tip_alleles: dict[str, str | None] = {}
    ingroup_observed: set[str] = set()
    for sample_idx, node_id in enumerate(sample_nodes):
        sid = sample_id_per_node.get(int(node_id))
        if sid is None:
            continue
        g = int(gens[sample_idx])
        if g < 0:
            tip_alleles[sid] = None
            continue
        a = alleles[g]
        if a not in STATES:
            tip_alleles[sid] = None
            continue
        tip_alleles[sid] = a
        if int(node_id) in ingroup_node_set:
            ingroup_observed.add(a)
    if len(ingroup_observed) != 1:
        continue  # polymorphic or all-missing ingroup -> skip
    fixed = next(iter(ingroup_observed))
    mono_sites.append(Site(
        chrom="1", pos=pos,
        alleles=tuple(a for a in alleles if a in STATES),
        tip_alleles=tip_alleles,
    ))
    mono_truth_ancestral.append(truth)
    mono_ingroup_fixed_allele.append(fixed)


print(
    f"B10 caterpillar inference: tree_type={tree_type}, "
    f"{len(mono_sites)} monoallelic-ingroup sites, "
    f"{len(ingroup_names)} ingroup haps + {len(outgroup_names)} outgroups, "
    f"outgroup_divergences={outgroup_divergences}",
    flush=True,
)


# Uniform composition (JC69, symmetric) and uniform root prior so the two
# trees are compared on like-for-like footing: only the *tree* choice
# differs, not the prior or the rate matrix.
bc = BaseComposition.no_counts()
model = JC69()
engine = Likelihood(model, base_composition=bc)
n_states = model.n_states
log_uniform_prior = np.full(n_states, -np.log(n_states))


if tree_type == "ladder":
    # OutgroupLadderTree, branch lengths set so each ingroup MRCA → O_k
    # path equals the simulator's known split-time × mu. Built via Newick
    # (rather than poking the K vector) so the ladder geometry is the
    # same for every n_out — Newick branch lengths are explicit and the
    # external K parameterisation differs across n in (1, 2, >=3).
    n_out = len(outgroup_names)
    # Ingroup as a polytomy collapsed at depth 0 (no within-ingroup
    # branch lengths in the EST-SFS topology). Attached to the ladder
    # at the ingroup MRCA, which then walks out to each outgroup at the
    # known divergence depths.
    ingroup_clade = "(" + ",".join(f"{s}:0" for s in ingroup_names) + ")"
    depths_close_first = list(outgroup_divergences)
    # Innermost pair: (ingroup_MRCA, O_1) at depth d_1.
    node = f"({ingroup_clade}:{depths_close_first[0]},{outgroup_names[0]}:{depths_close_first[0]})"
    for k in range(1, n_out):
        seg = depths_close_first[k] - depths_close_first[k - 1]
        node = f"({node}:{seg},{outgroup_names[k]}:{depths_close_first[k]})"
    newick = node + ";"
    tree = OutgroupLadderTree.from_newick(
        newick, ingroup_samples=ingroup_names, outgroup_samples=outgroup_names,
    )

    log_L = engine.log_likelihoods(tree, mono_sites)  # (n_sites, n_states)
    log_post = log_L + log_uniform_prior
    # Normalise.
    m = log_post.max(axis=1, keepdims=True)
    log_post -= m + np.log(np.exp(log_post - m).sum(axis=1, keepdims=True))
    posteriors = np.exp(log_post)

elif tree_type == "caterpillar":
    # The same ladder read at its deepest join. Depths are the I -> O_k
    # path lengths, which from_divergences splits at half the divergence.
    ladder = OutgroupLadderTree.from_divergences(
        ingroup_names, outgroup_names, outgroup_divergences,
    )
    cat_tree = ladder.as_deep_rooted()
    # The ingroup is fixed at these sites, so its whole contribution is a
    # delta on the allele it carries, seeded on the ingroup MRCA.
    cat_sites: list[Site] = []
    seed = np.zeros((len(mono_sites), len(STATES)), dtype=float)
    for row, (site, fixed) in enumerate(zip(mono_sites, mono_ingroup_fixed_allele)):
        cat_sites.append(Site(
            chrom=site.chrom, pos=site.pos, alleles=site.alleles,
            tip_alleles={og: site.tip_alleles.get(og) for og in outgroup_names},
        ))
        if fixed in STATES:
            seed[row, STATES.index(fixed)] = 1.0
        else:
            seed[row, :] = 1.0
    log_L = engine.log_likelihoods(
        cat_tree, cat_sites,
        node_seeds={ladder.ingroup_mrca: seed},
    )
    log_post = log_L + log_uniform_prior
    m = log_post.max(axis=1, keepdims=True)
    log_post -= m + np.log(np.exp(log_post - m).sum(axis=1, keepdims=True))
    posteriors = np.exp(log_post)


per_site: list[dict] = []
for i, site in enumerate(mono_sites):
    post = posteriors[i]
    map_idx = int(np.argmax(post))
    map_allele = STATES[map_idx]
    truth = mono_truth_ancestral[i]
    fixed = mono_ingroup_fixed_allele[i]
    per_site.append({
        "pos": int(site.pos),
        "map_allele": map_allele,
        "max_prob": float(post[map_idx]),
        "posterior": [float(p) for p in post],
        "truth": truth,
        "ingroup_fixed_allele": fixed,
        "ingroup_fixed_is_derived": (fixed != truth),
        "map_correct": map_allele == truth,
        "p_true": float(post[STATE_INDEX[truth]]) if truth in STATE_INDEX else float("nan"),
    })


inference_meta = {
    "tree_type": tree_type,
    "mu": mu,
    "n_ingroup": len(ingroup_names),
    "n_outgroups": len(outgroup_names),
    "outgroup_names": outgroup_names,
    "outgroup_split_times": outgroup_split_times,
    "outgroup_divergences": outgroup_divergences,
    "n_monoallelic_ingroup_sites": len(per_site),
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"per_site": per_site, "inference_meta": inference_meta}, f, indent=2)

n_corr = sum(1 for s in per_site if s["map_correct"])
print(
    f"Wrote {out_json}: {len(per_site)} sites, "
    f"accuracy={n_corr / len(per_site) if per_site else float('nan'):.4f}",
    flush=True,
)
