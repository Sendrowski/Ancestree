"""One point of the local-tree accuracy-vs-ensemble-size curve.

One (chunk, k) per job, so the pairwise-HMM forward pass each point needs runs
concurrently on the cluster instead of serially in one process. Scored through
the same path the heatmap uses (intersection mask, focal node per site class,
summarize_persite), so the y values land on the heatmap's scale.
"""
import json
import os
import sys

# The sibling module, located relative to this file rather than by an absolute
# path: the latter only resolves on the checkout it was written on.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402

try:
    scen = snakemake.wildcards.scenario  # noqa: F821
    n_out = int(snakemake.wildcards.n_out)  # noqa: F821
    chunk = int(snakemake.wildcards.chunk)  # noqa: F821
    k = int(snakemake.wildcards.k)  # noqa: F821
    _out_path = snakemake.output[0]  # noqa: F821
except NameError:
    # Standalone debug defaults; the DAG supplies these through `script:`.
    scen, n_out, chunk, k = "baseline", 3, 0, 8
    _out_path = (f"results/data/samplecurve_local_tree_{scen}"
                 f"_n{n_out}_chunk{chunk}_k{k}.json")

rc.LOCAL_TREE_ENSEMBLE_MEMBERS = k
sc = rc._chunk_scenario(scen, chunk)
sc.set_keep_intervals(rc.intersect_mask_for_chunk(scen, chunk, n_out))
try:
    recs, runtime = sc.run_local_tree_unphased(n_out)
finally:
    sc.set_keep_intervals(None)

summary = rc.summarize_persite(
    recs, scenario=scen, method="local_tree_unphased", n_out=n_out,
    n_ingroup=rc._cell_n_ingroup(scen), runtime=runtime)

with open(_out_path, "w") as fh:
    json.dump(summary, fh)
