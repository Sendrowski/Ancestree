"""B5 inference: polytomize the ARG at one threshold and run Ancestree.

Reads an msprime ``.trees`` file, builds a :class:`PolytomizedTree` view of
each local tree that collapses internal nodes whose parent-branch is
shorter than ``threshold``, runs the Ancestree kernel on every variant,
and writes per-site posteriors + per-site truth + polytomy-shape stats
to a single JSON file. The :mod:`report_polytomy` step aggregates one of
these per-threshold files across the configured threshold sweep.

Run directly for the default threshold (1000 generations)::

    python workflow/scripts/infer_polytomy.py

Or via snakemake with ``{threshold}`` wildcard::

    snakemake -j 1 results/data/polytomy_thresh1000.json
"""
import json
from collections.abc import Mapping, Sequence

import numpy as np
import tskit

from ancestree import JC69, STATE_INDEX, STATES, Site
from ancestree.likelihood import Likelihood
from ancestree.trees import Tree


try:
    in_trees = snakemake.input[0]  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    threshold = float(snakemake.wildcards.threshold)  # type: ignore[name-defined]
    mu = snakemake.params.mu  # type: ignore[name-defined]
except NameError:
    # Standalone defaults
    in_trees = "results/data/polytomy_sim.trees"
    out_json = "results/data/polytomy_thresh1000.json"
    threshold = 1000.0
    mu = 1e-7


# ----------------------------------------------------------- PolytomizedTree

class PolytomizedTree(Tree):
    """A ``Tree`` view that collapses internal nodes with parent-branch < threshold.

    Implements the :class:`ancestree.trees.Tree` ABC by lazily computing
    ``children`` and ``branch_length`` views that skip collapsed nodes. Tips
    are never collapsed (they have no children). Root is never collapsed
    (no parent). Children of a collapsed node attach to its grandparent
    while **keeping their original child-side branch lengths** (the internal
    segment is lost, not summed into the residual) — see B5's report for
    why this is the realistic tsinfer-style polytomy semantics.
    """

    def __init__(
        self,
        ts_tree: "tskit.Tree",
        sample_map: Mapping[str, int],
        threshold: float,
    ) -> None:
        self._ts_tree = ts_tree
        self._sample_to_node: dict[str, int] = dict(sample_map)
        self._node_to_sample: dict[int, str] = {v: k for k, v in sample_map.items()}
        self._threshold = float(threshold)
        if ts_tree.num_roots != 1:
            raise ValueError(
                f"PolytomizedTree needs a single-rooted tree; got {ts_tree.num_roots}"
            )

        # A node is collapsed iff it's internal, not the root, and its parent
        # branch is strictly shorter than `threshold`.
        self._collapsed: set[int] = set()
        for n in ts_tree.nodes():
            if n == ts_tree.root or ts_tree.is_sample(n):
                continue
            if float(ts_tree.branch_length(n)) < self._threshold:
                self._collapsed.add(n)

        self._children_map: dict[int, list[int]] = {}
        self._branch_length_map: dict[int, float] = {}
        kept = [n for n in ts_tree.nodes() if n not in self._collapsed]
        for n in kept:
            kids: list[int] = []
            for c in ts_tree.children(n):
                kids.extend(self._descend_through_collapsed(c))
            self._children_map[n] = kids
        for n in kept:
            self._branch_length_map[n] = (
                0.0 if n == ts_tree.root else float(ts_tree.branch_length(n))
            )

        po: list[int] = []
        seen: set[int] = set()

        def _visit(node: int) -> None:
            for c in self._children_map.get(node, []):
                if c not in seen:
                    _visit(c)
            seen.add(node)
            po.append(node)

        _visit(ts_tree.root)
        self._postorder = tuple(po)

    def _descend_through_collapsed(self, node: int) -> list[int]:
        if node not in self._collapsed:
            return [node]
        out: list[int] = []
        for c in self._ts_tree.children(node):
            out.extend(self._descend_through_collapsed(c))
        return out

    @property
    def n_nodes(self) -> int:
        return len(self._postorder)

    @property
    def root(self) -> int:
        return int(self._ts_tree.root)

    def children(self, node: int) -> Sequence[int]:
        return self._children_map.get(node, [])

    def branch_length(self, node: int) -> float:
        return self._branch_length_map.get(node, 0.0)

    def postorder(self) -> Sequence[int]:
        return self._postorder

    def tip_for_sample(self, sample_id: str) -> int | None:
        return self._sample_to_node.get(sample_id)

    def n_tips(self) -> int:
        return int(self._ts_tree.num_samples())


# ----------------------------------------------------------------- helpers

def _logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))


def _build_sites_for_tree(
    ts: tskit.TreeSequence,
    tree: tskit.Tree,
    sample_nodes: np.ndarray,
    node_to_sample: dict[int, str],
    chrom: str,
) -> list[Site]:
    sites: list[Site] = []
    for ts_site in tree.sites():
        variant = tskit.Variant(ts, samples=ts.samples())
        variant.decode(ts_site.id)
        alleles = tuple(variant.alleles)
        gens = variant.genotypes
        tip_alleles: dict[str, str | None] = {}
        for i, node in enumerate(sample_nodes):
            name = node_to_sample.get(int(node))
            if name is None:
                continue
            g = int(gens[i])
            tip_alleles[name] = None if g < 0 else (alleles[g] or None)
        sites.append(Site(
            chrom=chrom, pos=int(ts_site.position),
            alleles=alleles, tip_alleles=tip_alleles,
            local_tree_handle=float(ts_site.position),
        ))
    return sites


# --------------------------------------------------------------------- main

ts = tskit.load(in_trees)
print(
    f"B5 infer @ threshold={threshold}: loaded {ts.num_sites} sites, "
    f"{ts.num_trees} local trees from {in_trees}",
    flush=True,
)

engine = Likelihood(JC69())
sample_nodes = ts.samples()
sample_map = {str(int(s)): int(s) for s in sample_nodes}
node_to_sample = {v: k for k, v in sample_map.items()}
log_prior = np.full(4, -np.log(4))

per_site: dict[int, dict] = {}

# Polytomy-shape stats accumulated across trees (weighted by # sites in
# each tree, matching how the per-site accuracy comparison aggregates).
arities: list[int] = []
collapsed_weighted: list[int] = []
polytomic_weighted: list[int] = []

# Per-site truth + parsimony score on the *original* (un-polytomized) tree.
variants_by_site = {v.site.id: v for v in ts.variants()}

for tree in ts.trees():
    if tree.num_sites == 0 or tree.num_roots != 1:
        continue
    ptree = PolytomizedTree(tree, sample_map=sample_map, threshold=threshold)
    ptree.time_scale = mu  # branch lengths in generations → expected subs/site

    # Polytomy shape stats for this tree
    n_collapsed = len(ptree._collapsed)
    n_polytomic = 0
    for n in ptree.postorder():
        kids = ptree.children(n)
        if len(kids) > 2:
            n_polytomic += 1
            arities.append(len(kids))
    for _ in range(tree.num_sites):
        collapsed_weighted.append(n_collapsed)
        polytomic_weighted.append(n_polytomic)

    # Sites + Felsenstein on the polytomized view
    sites = _build_sites_for_tree(ts, tree, sample_nodes, node_to_sample, chrom="1")
    log_L = engine.log_likelihoods(ptree, sites)
    log_post = log_L + log_prior
    log_post -= _logsumexp(log_post)
    posterior = np.exp(log_post)

    for site, p in zip(sites, posterior):
        # Per-site truth + parsimony score on the original tree (cheap,
        # done here so each per-threshold file is self-contained).
        ts_site = next(s for s in tree.sites() if int(s.position) == site.pos)
        variant = variants_by_site[ts_site.id]
        alleles = list(variant.alleles)
        gens = variant.genotypes
        usable = (not any(g < 0 for g in gens)
                  and all(alleles[g] in STATE_INDEX for g in gens))
        if usable:
            states = [STATE_INDEX[alleles[g]] for g in gens]
            _anc, muts = tree.map_mutations(states, alleles=STATES)
            truth_allele: str | None = ts_site.ancestral_state
            parsimony_score = len(muts)
        else:
            truth_allele = None
            parsimony_score = None

        per_site[site.pos] = {
            "map_allele": STATES[int(np.argmax(p))],
            "max_prob": float(p.max()),
            "posterior": p.tolist(),
            "truth": truth_allele,
            "parsimony_score": parsimony_score,
        }

stats = {
    "threshold": threshold,
    "mean_collapsed_per_site": float(np.mean(collapsed_weighted)) if collapsed_weighted else 0.0,
    "mean_polytomic_internals_per_site": (
        float(np.mean(polytomic_weighted)) if polytomic_weighted else 0.0
    ),
    "max_polytomy_arity": int(max(arities)) if arities else 2,
    "n_sites_evaluated": len(per_site),
    "mu": mu,
}

with open(out_json, "w") as f:
    json.dump({"stats": stats, "per_site": per_site}, f, indent=2)
print(
    f"Wrote {out_json}: {len(per_site)} sites, "
    f"polytomic_internals/site={stats['mean_polytomic_internals_per_site']:.2f}, "
    f"max_arity={stats['max_polytomy_arity']}",
    flush=True,
)
