"""One (window, block_size) cell of the local-tree window x block grid.

Builds the ingroup :class:`~ancestree.sites.Site` list from the simulated
ARG's genotypes (no truth allele consumed), runs
:class:`~ancestree.local_tree_inference.LocalTreeInference` at the cell's
window size and explicit HMM emission block width, both given as
``"<N>snp"`` specs (N SNPs' worth of span at the data's mean density), and
scores every polarised site against the msprime ground-truth ancestral
state. The true mutation and recombination rates are used and the genotypes
are perfectly phased, so the only thing varying across the grid is the pair
(window, block).

An explicit ``block_size`` is not capped at the window. The instance
resolves both specs to base pairs and then rounds the window up to a whole
number of blocks, so the width actually used can exceed the requested one.
The payload therefore records the requested specs, the resolved widths in
base pairs, and their SNP equivalents at the panel's mean density.

Reuses :func:`_robustness_common._sumsq` so the per-site Brier term is
computed identically to the other local-tree benchmarks.

Run directly with the standalone defaults::

    python workflow/scripts/infer_local_tree_grid_cell.py

Or via snakemake (one job per grid cell)::

    snakemake -j 4 results/data/local_tree_grid_w8snp_b4snp.json
"""
import json
import sys
import time
from pathlib import Path

import tskit

# Make the sibling _robustness_common importable when run as a snakemake
# script (the script is copied to a tempdir) or standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _robustness_common import (  # noqa: E402
    _sumsq, LOCAL_TREE_ENSEMBLE_MEMBERS)

from ancestree import JC69, STATES, Site  # noqa: E402
from ancestree import LocalTreeInference  # noqa: E402


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    window = str(snakemake.params.window)  # type: ignore[name-defined]
    block = str(snakemake.params.block)  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/local_tree_window_sim.trees"
    in_meta = "results/data/local_tree_window_sim_meta.json"
    out_json = "results/data/local_tree_grid_w8snp_b4snp.json"
    window = "8snp"
    block = "4snp"


meta = json.loads(Path(in_meta).read_text())
ingroup_names = list(meta["ingroup_names"])
mu_true = float(meta["mu"])
r_true = float(meta["rec_rate"])
ts = tskit.load(str(in_trees))

# Per-site truth + node->name map (single haplotype per individual).
name_for_node = {int(ind.nodes[0]): f"tsk_{ind.id}" for ind in ts.individuals()}
truth_by_pos = {int(s.position): s.ancestral_state for s in ts.sites()}
keep_set = set(ingroup_names)


def _score(infer_iter) -> list[tuple[int, float, float]]:
    """Score a polariser's ``(Site, Posterior)`` stream against truth.

    :param infer_iter: Iterable of ``(Site, Posterior)`` from an inference.
    :return: Per-site ``(map_hit, p_true, sum_sq)`` triples; ``sum_sq`` lets
        the report recover the Brier score.
    """
    recs: list[tuple[int, float, float]] = []
    for site, post in infer_iter:
        t = truth_by_pos.get(int(site.pos))
        if t is None or t not in STATES:
            continue
        recs.append((int(post.map_allele == t), float(post[t]), _sumsq(post)))
    return recs


sites: list[Site] = []
for v in ts.variants():
    pos = int(v.site.position)
    if truth_by_pos.get(pos) not in STATES:
        continue
    tip_alleles = {name_for_node[int(node)]: v.alleles[v.genotypes[j]]
                   for j, node in enumerate(ts.samples())
                   if name_for_node[int(node)] in keep_set}
    sites.append(Site(chrom="1", pos=pos,
                      alleles=tuple(v.alleles), tip_alleles=tip_alleles))

sequence_length = float(ts.sequence_length)
# Mean density the "<N>snp" specs are resolved against: the region under
# analysis divided by the sites handed to the inference.
bp_per_snp = sequence_length / max(1, len(sites))

t0 = time.perf_counter()
inf = LocalTreeInference(
    sites, JC69(), mu=mu_true, rec_rate=r_true,
    sample_names=ingroup_names,
    sequence_length=sequence_length,
    window=window, block_size=block, chunk_size="10mb", progress=False,
    ingroup_samples=ingroup_names,
    n_ensemble=LOCAL_TREE_ENSEMBLE_MEMBERS,
)
records = _score(inf.infer())
runtime = time.perf_counter() - t0

# Widths the instance ran at: the block spec resolved to bp, and the window
# spec snapped up to a whole number of those blocks.
inf._resolve_segmentation_params()
window_bp_used = int(inf._wbp)
block_bp_used = int(inf.block_size)

n = len(records)
mean_map = sum(r[0] for r in records) / n if n else float("nan")
mean_ptrue = sum(r[1] for r in records) / n if n else float("nan")
# Brier = 1 - 2*p_true + sum_sq, averaged over sites (lower is better).
mean_brier = (
    sum(1.0 - 2.0 * r[1] + r[2] for r in records) / n if n else float("nan")
)

payload = {
    "window": window,
    "block": block,
    "window_snp_requested": int(str(window)[:-3]),
    "block_snp_requested": int(str(block)[:-3]),
    "window_bp": window_bp_used,
    "block_size": block_bp_used,
    "window_snp_effective": window_bp_used / bp_per_snp,
    "block_snp_effective": block_bp_used / bp_per_snp,
    "blocks_per_window": window_bp_used / block_bp_used,
    "bp_per_snp": bp_per_snp,
    "sequence_length": sequence_length,
    "n_input_sites": len(sites),
    "mu_used": mu_true,
    "rec_rate_used": r_true,
    "n_ensemble": LOCAL_TREE_ENSEMBLE_MEMBERS,
    "n_sites": n,
    "mean_brier": mean_brier,
    "mean_map": mean_map,
    "mean_ptrue": mean_ptrue,
    "runtime": runtime,
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)
print(
    f"Wrote {out_json}: window={window} block={block} "
    f"window_bp={window_bp_used} block_bp={block_bp_used} "
    f"(~{payload['window_snp_effective']:.2f} / "
    f"{payload['block_snp_effective']:.2f} SNPs) "
    f"n={n} Brier={mean_brier:.4f} MAP={mean_map:.4f} "
    f"P(true)={mean_ptrue:.4f} ({runtime:.1f}s)",
    flush=True,
)
