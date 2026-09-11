"""SLiM-outgroup-bias benchmark: ARGBasedInference on the simulator-truth ARG.

Loads the recap+mutated tree sequence (ingroup p0 + 3 outgroups p1/p2/p3
at known divergence times), subsets to ingroup + the first ``n_out``
outgroups (closest first), simplifies, and runs
:class:`ARGBasedInference` with JC69 + full recurrence.

Branches in the recapitated ts are in SLiM-generation units (pyslim +
msprime preserve the SLiM time scale during recapitation + mutation
overlay). With ``time_scaling = Ne_target / Ne_slim = 10``,
``mu_slim = mu_target * 10 = 1.25e-7``. Passing ``mu_slim`` keeps
``K = mu * branch_length`` matched to the target scale since the
branch lengths themselves are the SLiM-scale ones.

Truth ancestral state at each polymorphic site comes from
``ts.site(site_id).ancestral_state`` (msprime overlaid the truth there).

Wildcards:

- ``{f_del}`` — fraction of sites under selection in the upstream SLiM
  sim (``0.0``, ``0.5``, ``1.0``).
- ``{n_out}`` ∈ ``{1, 2, 3}`` — number of outgroups included in the
  Felsenstein tree (closest first).

Run directly::

    python workflow/scripts/infer_slim_outgroup_bias_arg.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import tskit

from ancestree import ARGBasedInference, JC69

STATES = ("A", "C", "G", "T")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """CLI fallback when not driven by snakemake."""
    p = argparse.ArgumentParser()
    p.add_argument("--in-trees", required=True)
    p.add_argument("--in-meta", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--f-del", type=float, required=True)
    p.add_argument("--n-out", type=int, required=True)
    p.add_argument("--mu-slim", type=float, required=True)
    return p.parse_args(argv)


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    f_del = float(snakemake.wildcards.f_del)  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    mu_slim = float(snakemake.params.mu_slim)  # type: ignore[name-defined]
except NameError:
    _args = _parse_args(sys.argv[1:])
    in_trees = _args.in_trees
    in_meta = _args.in_meta
    out_json = _args.out_json
    f_del = _args.f_del
    n_out = _args.n_out
    mu_slim = _args.mu_slim


with open(in_meta) as f:
    meta = json.load(f)

ingroup_names: list[str] = list(meta["ingroup_names"])
all_outgroup_names: list[str] = list(meta["outgroup_names"])  # closest-first
if n_out > len(all_outgroup_names):
    raise ValueError(
        f"n_out={n_out} exceeds available outgroups "
        f"({len(all_outgroup_names)})"
    )
outgroup_names = all_outgroup_names[:n_out]
n_ingroup = len(ingroup_names)

# ----------------------------------------------------------- Load + simplify ts
ts_full = tskit.load(in_trees)
pop_name_by_id = {pop.id: pop.metadata.get("name", "") for pop in ts_full.populations()}
# Map outgroup ladder name ("out_T{k}") -> SLiM pop name ("p{k}"), derived
# from the meta so this handles any number of outgroups (3 or 10).
out_to_pop = {name: "p" + name[len("out_T"):] for name in all_outgroup_names}

ingroup_nodes: list[int] = []
out_nodes_by_pop: dict[str, list[int]] = {p: [] for p in out_to_pop.values()}
for sample_node in ts_full.samples():
    node = ts_full.node(int(sample_node))
    pname = pop_name_by_id.get(int(node.population), "")
    if pname == "p0":
        ingroup_nodes.append(int(sample_node))
    elif pname in out_nodes_by_pop:
        out_nodes_by_pop[pname].append(int(sample_node))

# The recap+mutate step already subsampled ingroup to n_ingroup_haps and
# picked exactly 1 haploid per outgroup pop. Preserve that exact ordering.
if len(ingroup_nodes) != n_ingroup:
    raise RuntimeError(
        f"ts has {len(ingroup_nodes)} ingroup samples but meta says {n_ingroup}"
    )

selected_out_nodes: list[int] = []
selected_out_names: list[str] = []
for out_name in outgroup_names:
    pop_name = out_to_pop[out_name]
    avail = out_nodes_by_pop[pop_name]
    if not avail:
        raise RuntimeError(f"No samples in outgroup pop {pop_name} for {out_name}")
    selected_out_nodes.append(int(avail[0]))
    selected_out_names.append(out_name)

keep_nodes = list(ingroup_nodes) + selected_out_nodes
keep_names = list(ingroup_names) + selected_out_names

# filter_sites=True drops sites that fall in regions where every sample is
# the ancestral state after the subset (i.e. the outgroup-only-private
# polymorphisms we don't want to score against).
ts_sub = ts_full.simplify(samples=keep_nodes, filter_sites=True)
# After simplify, the kept samples are renumbered 0..len(keep)-1 in the
# same order they were passed in.
sample_map = {s: i for i, s in enumerate(keep_names)}

print(
    f"ARG-true inference: f_del={f_del}, n_out={n_out}, "
    f"{ts_sub.num_samples} haplotypes, {ts_sub.num_sites} sites, "
    f"{ts_sub.num_trees} local trees, mu={mu_slim:g}",
    flush=True,
)

# Truth + per-site ingroup counts on the SUBSAMPLED ts (so positions are
# the ones the inference will yield). Truth state comes from the
# original ts via the same site_id (subsample doesn't change ancestral
# states, only filters out unreachable mutations).
truth_by_pos: dict[int, str] = {}
ingroup_indices_sub = list(range(n_ingroup))
ingroup_counts_by_pos: dict[int, dict[str, int]] = {}
for var in ts_sub.variants():
    site = var.site
    anc = site.ancestral_state
    pos = int(site.position)
    truth_by_pos[pos] = anc
    alleles = list(var.alleles)
    gens = var.genotypes
    counts: dict[str, int] = {}
    for i in ingroup_indices_sub:
        g = int(gens[i])
        if g < 0:
            continue
        a = alleles[g]
        if a is None or a not in STATES:
            continue
        counts[a] = counts.get(a, 0) + 1
    ingroup_counts_by_pos[pos] = counts


# ----------------------------------------------------------- Run inference
inference = ARGBasedInference(
    ts_sub, JC69(),
    mu=mu_slim,
    sample_map=sample_map,
    progress=False,
)

per_site: list[dict] = []
_t0 = time.perf_counter()
for site, post in inference.infer():
    pos = int(site.pos)
    truth = truth_by_pos.get(pos)
    if truth is None or truth not in STATES:
        continue
    counts = ingroup_counts_by_pos.get(pos, {})
    if counts:
        major_count = max(counts.values())
    else:
        major_count = n_ingroup
    minor_count = n_ingroup - major_count
    derived_count: int | None
    if truth is not None:
        # derived = number of ingroup haps NOT equal to the truth allele
        derived_count = int(n_ingroup - counts.get(truth, 0))
    else:
        derived_count = None
    posterior_values = [float(p) for p in post.values]
    posterior_alleles = list(post.alleles)
    per_site.append({
        "pos": pos,
        "alleles": list(site.alleles),
        "map_allele": post.map_allele,
        "max_prob": float(post.max_prob),
        "posterior": posterior_values,
        "posterior_alleles": posterior_alleles,
        "truth": truth,
        "map_correct": truth is not None and post.map_allele == truth,
        "ingroup_major_count": int(major_count),
        "ingroup_minor_count": int(minor_count),
        "ingroup_derived_count": derived_count,
    })
infer_seconds = float(time.perf_counter() - _t0)

result = {
    "f_del": f_del,
    "n_out": n_out,
    "chunk_idx": meta.get("chunk_idx"),
    "ingroup_size": n_ingroup,
    "outgroup_names_used": selected_out_names,
    "mode": "arg_true",
    "mu": mu_slim,
    "fit_seconds": 0.0,  # ARGBasedInference has no fit step
    "infer_seconds": infer_seconds,
    "n_sites": len(per_site),
    "per_site": per_site,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(
    f"Wrote {out_json}: {len(per_site)} sites, infer={infer_seconds:.2f}s",
    flush=True,
)
