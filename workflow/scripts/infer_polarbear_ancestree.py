"""B2 Ancestree-side inference: ``JC69 + full + uniform`` on the simulated ARG.

Run directly::

    python workflow/scripts/infer_polarbear_ancestree.py

Or via snakemake::

    snakemake -j 1 results/data/polarbear_ancestree.json
"""
import json
import time

import tskit

from ancestree import ARGBasedInference, JC69, STATE_INDEX


try:
    in_trees = snakemake.input.trees  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    mu = snakemake.params.mu  # type: ignore[name-defined]
except NameError:
    in_trees = "results/data/polarbear_sim.trees"
    out_json = "results/data/polarbear_ancestree.json"
    mu = 1e-8


ts = tskit.load(in_trees)
print(
    f"B2 Ancestree on {in_trees}: {ts.num_sites} sites, "
    f"{ts.num_samples} haplotypes",
    flush=True,
)

inference = ARGBasedInference(
    ts, JC69(),
    mu=mu,
    progress=False,
)
# Wall-clock for the inference step only (post tskit.load, pre output write);
# reported in inference_meta for the manuscript runtime column.
_t_inference_start = time.perf_counter()
results: dict[int, dict] = {}
for site, posterior in inference.infer():
    results[site.pos] = {
        "map_idx": STATE_INDEX[posterior.map_allele],
        "map_allele": posterior.map_allele,
        "max_prob": posterior.max_prob,
        "posterior": posterior.values.tolist(),
    }
inference_seconds = float(time.perf_counter() - _t_inference_start)

inference_meta = {
    "mu": mu,
    "n_sites": len(results),
    "inference_seconds": inference_seconds,
}

with open(out_json, "w") as f:
    json.dump({"sites": results, "inference_meta": inference_meta}, f)
print(f"Wrote {out_json}: {len(results)} sites", flush=True)
