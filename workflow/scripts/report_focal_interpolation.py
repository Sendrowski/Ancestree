"""Accuracy as the focal node slides from the ingroup MRCA to the panel root.

The focal node is not a free parameter with an obvious setting: the ingroup
MRCA is the node an unfolded SFS is polarised about, the panel root is the node
a fixed difference is called against, and everything between is reachable. This
sweeps the interpolation so the choice can be read off rather than argued.

``fraction`` runs 0 (the ingroup MRCA) to 1 (the panel root) along the path
between them, per outgroup count and per site class. It is scale-free, so the
curves are comparable across scenarios whose divergences differ.

Each point is scored against the simulated truth read *at that same point*, so
the curve measures where the ancestral state is best recovered rather than how
far the estimate drifts from one fixed reference.
"""
import json

import numpy as np
import tskit

import ancestree as anc
from ancestree.focal import FocalNode
from ancestree.models import JC69

try:  # snakemake execution
    TREES = snakemake.input.trees
    OUT_JSON = snakemake.output.json
    FRACTIONS = snakemake.params.fractions
    N_OUTS = list(snakemake.params.n_outs)
    MODES = list(snakemake.params.modes)
    MU = snakemake.params.mu
except NameError:  # direct execution
    TREES = "results/data/baseline10_chunk0.trees"
    OUT_JSON = "results/reports/focal_interpolation.json"
    FRACTIONS = list(np.linspace(0.0, 1.0, 41))
    N_OUTS = [1, 3, 10]
    MODES = ["fixed_tree", "arg", "local_tree"]
    MU = 1.25e-8


def _panel(ts, n_out):
    """Ingroup nodes and an outgroup subset spread across split depths."""
    name = {p.id: (p.metadata or {}).get("name") for p in ts.populations()}
    ingroup = [int(n) for n in ts.samples()
               if name[ts.node(int(n)).population] == "ingroup"]
    outgroups = [int(n) for n in ts.samples()
                 if str(name[ts.node(int(n)).population]).startswith("outgroup")]
    outgroups.sort(key=lambda n: int(name[ts.node(n).population].split("_")[1]))
    if n_out >= len(outgroups):
        chosen = outgroups
    else:
        index = np.linspace(0, len(outgroups) - 1, n_out).round().astype(int)
        chosen = [outgroups[i] for i in dict.fromkeys(index)]
    return ingroup, chosen


def _truth_at(tree, site, node, height=0.0):
    """Simulated state at a point ``height`` above ``node``.

    A mutation sits at a time on the branch above its own node, so one on the
    very edge being split may fall either side of the point. Reading the state
    at the node below would quantise the truth to nodes while the estimate
    moves continuously, so compare times rather than topology alone.

    :param tree: The local tree covering the site.
    :param site: The :class:`tskit.Site` carrying the mutations.
    :param node: Node immediately below the point.
    :param height: Distance above ``node``, in the tree's own time units.
    :return: The simulated state at that point.
    """
    point_time = tree.time(node) + height
    state, best = site.ancestral_state, np.inf
    for mutation in site.mutations:
        on_path = (mutation.node == node
                   or tree.is_descendant(node, mutation.node))
        if not on_path:
            continue
        # Prefer the mutation's own time. Fall back to its node's when the
        # simulation left it unknown.
        time = float(mutation.time)
        if np.isnan(time):
            time = tree.time(mutation.node)
        if time <= point_time:
            continue  # below the point: the point predates it
        if time < best:
            best, state = time, mutation.derived_state
    return state


def _focal_point(tree, ingroup_nodes, fraction):
    """The node and offset ``fraction`` of the way from the ingroup MRCA up."""
    resolved = FocalNode(
        "ingroup_mrca", fraction=(fraction if fraction > 0 else None),
    ).resolve(tree, ingroup_nodes=ingroup_nodes)
    return resolved


class _GenotypeSource(anc.SiteSource):
    """Re-iterable genotype stream over a panel's variants.

    :param ts: The simplified panel tree sequence.
    :param names: ``{node: sample name}``.
    :param samples: Panel node ids, in ``ts.samples()`` order.
    """

    def __init__(self, ts, names, samples):
        self._ts = ts
        self._names = [names[int(n)] for n in samples]

    def samples(self) -> list:
        """:return: Sample names, in genotype-column order."""
        return list(self._names)

    def __iter__(self):
        """:return: One :class:`~ancestree.sites.Site` per variant."""
        for variant in self._ts.variants():
            yield anc.Site(
                chrom="1", pos=int(variant.site.position),
                alleles=tuple(a for a in variant.alleles if a),
                tip_alleles={name: variant.alleles[g]
                             for name, g in zip(self._names, variant.genotypes)},
            )


def _sites(ts, names, ing_nodes, samples):
    """Panel sites with tip alleles keyed by sample name.

    :param ts: The simplified panel tree sequence.
    :param names: ``{node: sample name}``.
    :param ing_nodes: Ingroup node ids.
    :param samples: All panel node ids, ingroup first.
    :return: List of :class:`~ancestree.sites.Site`.
    """
    out = []
    for variant in ts.variants():
        tip_alleles = {names[int(n)]: variant.alleles[g]
                       for n, g in zip(samples, variant.genotypes)}
        out.append(anc.Site(chrom="1", pos=int(variant.site.position),
                           alleles=tuple(a for a in variant.alleles if a),
                           tip_alleles=tip_alleles))
    return out


def _prepared(mode, ts, names, ing_nodes, samples):
    """Everything a mode needs that does not depend on the focal position.

    Built once per (mode, outgroup count) so the focal scan costs one kernel
    pass per grid point: ``local_tree`` runs its pairwise HMM here, and
    ``fixed_tree`` fits its ladder here.

    :param mode: One of ``fixed_tree``, ``arg``, ``local_tree``.
    :param ts: The simplified panel tree sequence.
    :param names: ``{node: sample name}``.
    :param ing_nodes: Ingroup node ids.
    :param samples: All panel node ids, ingroup first.
    :return: A fitted :class:`~ancestree.inference.FixedTreeInference` for
        ``fixed_tree``, else the tree sequence to score.
    """
    ingroup = [names[int(n)] for n in ing_nodes]
    outgroup = [names[int(n)] for n in samples[len(ing_nodes):]]
    if mode == "arg":
        return ts
    if mode == "local_tree":
        # Genotypes, not the simulated tree sequence: a tree sequence takes
        # LocalTreeInference's pre-built branch, which scores it as an ARG
        # without running the HMM.
        source = _GenotypeSource(ts, names, samples)
        return anc.LocalTreeInference(
            source, JC69(), mu=MU, rec_rate=1e-8,
            sample_names=ingroup + outgroup,
            ingroup_samples=ingroup, outgroup_samples=outgroup,
            window="8snp", chunk_size="2mb",
            progress=False,
        ).point_tree_sequence()
    if mode == "fixed_tree":
        inference = anc.FixedTreeInference(
            _sites(ts, names, ing_nodes, samples), JC69(),
            n_target_sites=int(ts.sequence_length),
            ingroup_samples=ingroup, outgroup_samples=outgroup,
            progress=False, baseline_check=False,
        )
        inference.fit()
        return inference
    raise ValueError(f"unknown mode {mode!r}")


def _calls(mode, prepared, names, ing_nodes, samples, focal):
    """Per-site posteriors for one mode at one focal position.

    :param mode: One of ``fixed_tree``, ``arg``, ``local_tree``.
    :param prepared: What :func:`_prepared` returned for ``mode``.
    :param names: ``{node: sample name}``.
    :param ing_nodes: Ingroup node ids.
    :param samples: All panel node ids, ingroup first.
    :param focal: Focal node specification to report at.
    :return: ``{position: Posterior}``.
    """
    if mode == "fixed_tree":
        prepared.focal = focal
        inference = prepared
    else:
        inference = anc.ARGBasedInference(
            prepared, JC69(), mu=MU,
            sample_map={v: k for k, v in names.items()}, progress=False,
            ingroup_samples=[names[int(n)] for n in ing_nodes],
            outgroup_samples=[names[int(n)] for n in samples[len(ing_nodes):]],
            focal=focal,
        )
    return {int(s.pos): p for s, p in inference.infer()}


def main() -> None:
    """Sweep the focal node and write per-class accuracy to JSON."""
    source = tskit.load(TREES)
    rows = []
    for n_out in N_OUTS:
        ingroup, outgroups = _panel(source, n_out)
        ts = source.simplify(samples=ingroup + outgroups, filter_sites=False)
        samples = list(ts.samples())
        names = {int(n): f"s{i}" for i, n in enumerate(samples)}
        ing_nodes = samples[:len(ingroup)]
        sample_map = {v: k for k, v in names.items()}

        # Three classes, not two. A monomorphic ingroup fixed for the
        # simulation's ancestral allele is answered correctly at the ingroup
        # MRCA by construction, so pooling it with the fixed-derived sites
        # turns the curve into a statement about class composition.
        site_class = {}
        ing_set = set(ing_nodes)
        for variant in ts.variants():
            alleles = {variant.alleles[g]
                       for node, g in zip(samples, variant.genotypes)
                       if int(node) in ing_set and variant.alleles[g]}
            pos = int(variant.site.position)
            if len(alleles) > 1:
                site_class[pos] = "polymorphic"
            elif alleles and next(iter(alleles)) == variant.site.ancestral_state:
                site_class[pos] = "fixed_ancestral"
            elif alleles:
                site_class[pos] = "fixed_derived"

        # Each mode's trees are built once and reused across the grid, so a
        # fine scan costs one kernel pass per point rather than a re-inference.
        for mode in MODES:
            prepared = _prepared(mode, ts, names, ing_nodes, samples)
            for fraction in FRACTIONS:
                focal = ("panel_root" if fraction >= 1.0 else
                         FocalNode("ingroup_mrca",
                                   fraction=(fraction if fraction > 0 else None)))
                calls = _calls(mode, prepared, names, ing_nodes,
                               samples, focal)

                # Three gradings per position. "moving" grades at the node
                # being reported at, so each point answers its own question.
                # "at_mrca" and "at_root" hold the question fixed and move only
                # the estimator, so those curves read as the cost of reporting
                # at the wrong node.
                truth = {}
                fixed = {"at_mrca": {}, "at_root": {}}
                for tree in ts.trees():
                    if not tree.num_sites:
                        continue
                    point = _focal_point(tree, ing_nodes, fraction)
                    shallow = _focal_point(tree, ing_nodes, 0.0)
                    deep = _focal_point(tree, ing_nodes, 1.0)
                    for site in tree.sites():
                        pos = int(site.position)
                        truth[pos] = _truth_at(tree, site, point.node, point.tau)
                        fixed["at_mrca"][pos] = _truth_at(
                            tree, site, shallow.node, shallow.tau)
                        fixed["at_root"][pos] = _truth_at(
                            tree, site, deep.node, deep.tau)

                for label in ("polymorphic", "fixed_ancestral",
                              "fixed_derived"):
                    for grading, table in (("moving", truth),
                                           ("at_mrca", fixed["at_mrca"]),
                                           ("at_root", fixed["at_root"])):
                        scores = []
                        for pos, posterior in calls.items():
                            if site_class.get(pos) != label:
                                continue
                            true = table.get(pos)
                            if true not in anc.STATES:
                                continue
                            probs = np.array([posterior[s] for s in anc.STATES])
                            hit = np.array([1.0 if s == true else 0.0
                                            for s in anc.STATES])
                            scores.append(float(((probs - hit) ** 2).sum()))
                        rows.append({
                            "mode": mode, "n_out": n_out, "fraction": fraction,
                            "site_class": label, "grading": grading,
                            "n_sites": len(scores),
                            "mean_brier":
                                float(np.mean(scores)) if scores else None,
                        })
                    print(f"{mode:11s} n_out={n_out:2d} "
                          f"fraction={fraction:.3f} {label:12s} "
                          f"n={rows[-1]['n_sites']:6d} "
                          f"brier={rows[-3]['mean_brier']}", flush=True)

    with open(OUT_JSON, "w") as handle:
        json.dump({"trees": TREES, "mu": MU, "rows": rows}, handle, indent=1)


main()
