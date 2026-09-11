"""B6 resimulation variant: re-run the same demographic with N seeds.

For each of ``N`` seeds: simulate a fresh ARG with identical
demographic parameters, simplify down to ingroup + each ``n_out``
outgroups, run :class:`ARGBasedInference`, and record per-cell aggregate
stats (``mean_max_prob``, ``accuracy``, ``n_sites``).

Sites differ across re-simulations (different mutations), so per-site
averaging is not meaningful here and aggregation is at the per-(n_out, seed)
level instead. The output captures how much variability the
inference picks up across independent samples from the coalescent
prior, which is an upper bound on what proper ARG-posterior sampling
(ARGweaver, SINGER) could move the needle.

Run directly::

    python workflow/scripts/infer_baseline_resimulated.py

Or via snakemake::

    snakemake -j 1 results/data/baseline_resimulated.json
"""
import json
import statistics as stats
from pathlib import Path

import msprime

from ancestree import ARGBasedInference, JC69


try:
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    n_ingroup = int(snakemake.params.n_ingroup)  # type: ignore[name-defined]
    n_outgroup_pops = int(snakemake.params.n_outgroup_pops)  # type: ignore[name-defined]
    length = float(snakemake.params.length)  # type: ignore[name-defined]
    mu = float(snakemake.params.mu)  # type: ignore[name-defined]
    rec_rate = float(snakemake.params.rec_rate)  # type: ignore[name-defined]
    pop_size = float(snakemake.params.pop_size)  # type: ignore[name-defined]
    outgroup_split_times = list(snakemake.params.outgroup_split_times)  # type: ignore[name-defined]
    seeds = list(snakemake.params.seeds)  # type: ignore[name-defined]
    n_outs = list(snakemake.params.n_outs)  # type: ignore[name-defined]
except NameError:
    out_json = "results/data/baseline_resimulated.json"
    n_ingroup = 20
    n_outgroup_pops = 3
    length = 5e5
    mu = 1e-8
    rec_rate = 1e-8
    pop_size = 1e4
    outgroup_split_times = [5e4, 1.5e5, 4e5]
    seeds = list(range(100, 120))  # 20 independent re-simulations
    n_outs = [0, 1, 2, 3]


def _build_demography(pop_size: float, split_times: list[float]) -> "msprime.Demography":
    """Re-create the same ingroup + outgroup_1..N + ancestor chain demography."""
    demography = msprime.Demography()
    demography.add_population(name="ingroup", initial_size=pop_size)
    n_pops = len(split_times)
    for k in range(1, n_pops + 1):
        demography.add_population(name=f"outgroup_{k}", initial_size=pop_size)
    demography.add_population(name="anc_root", initial_size=pop_size)
    ancestor_chain: list[str] = ["ingroup"]
    for k in range(1, n_pops + 1):
        anc_name = "anc_root" if k == n_pops else f"anc_{k}"
        if anc_name != "anc_root":
            demography.add_population(name=anc_name, initial_size=pop_size)
        demography.add_population_split(
            time=split_times[k - 1],
            ancestral=anc_name,
            derived=[ancestor_chain[-1], f"outgroup_{k}"],
        )
        ancestor_chain.append(anc_name)
    return demography


def _simulate_one(seed: int):
    """Return (ts, ingroup_node_ids, outgroup_node_ids_by_pop, truth_by_pos)."""
    demography = _build_demography(pop_size, outgroup_split_times)
    samples = [msprime.SampleSet(n_ingroup, population="ingroup", ploidy=1)]
    for k in range(1, n_outgroup_pops + 1):
        samples.append(msprime.SampleSet(1, population=f"outgroup_{k}", ploidy=1))
    ts = msprime.sim_ancestry(
        samples=samples, demography=demography,
        sequence_length=length, recombination_rate=rec_rate,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
    pop_id_to_name = {p.id: p.metadata.get("name", f"pop_{p.id}") for p in ts.populations()}
    ingroup_nodes: list[int] = []
    outgroup_nodes: dict[str, list[int]] = {f"outgroup_{k}": [] for k in range(1, n_outgroup_pops + 1)}
    for ind in ts.individuals():
        node_id = int(ind.nodes[0])
        pop_name = pop_id_to_name[ts.node(node_id).population]
        if pop_name == "ingroup":
            ingroup_nodes.append(node_id)
        elif pop_name.startswith("outgroup_"):
            outgroup_nodes[pop_name].append(node_id)
    truth = {int(s.position): s.ancestral_state for s in ts.sites()}
    return ts, ingroup_nodes, outgroup_nodes, truth


def _run_cell(ts, ingroup_nodes, outgroup_nodes, truth, n_out):
    """Simplify to ingroup + n_out outgroups, run ARGBasedInference, aggregate stats."""
    ordered_pops = sorted(outgroup_nodes.keys(), key=lambda s: int(s.split("_")[1]))
    selected_nodes: list[int] = []
    for pop in ordered_pops[:n_out]:
        selected_nodes.extend(outgroup_nodes[pop])
    keep = ingroup_nodes + selected_nodes
    ts_sub = ts.simplify(samples=keep, filter_sites=False)
    sample_map = {f"n{i}": i for i in range(len(keep))}
    inference = ARGBasedInference(
        ts_sub, JC69(),
        mu=mu,
        sample_map=sample_map,
        progress=False,
    )
    max_probs: list[float] = []
    correct = 0
    n_with_truth = 0
    for site, posterior in inference.infer():
        max_probs.append(float(posterior.max_prob))
        t = truth.get(int(site.pos))
        if t is not None:
            n_with_truth += 1
            if posterior.map_allele == t:
                correct += 1
    return {
        "n_sites": len(max_probs),
        "mean_max_prob": stats.mean(max_probs) if max_probs else float("nan"),
        "accuracy": correct / n_with_truth if n_with_truth else float("nan"),
        "n_sites_with_truth": n_with_truth,
    }


per_seed: dict[int, dict[int, dict]] = {n: {} for n in n_outs}
print(f"Resimulation: {len(seeds)} seeds × {len(n_outs)} n_outs", flush=True)
for s_i, seed in enumerate(seeds, 1):
    ts, ingroup_nodes, outgroup_nodes, truth = _simulate_one(seed)
    for n_out in n_outs:
        per_seed[n_out][seed] = _run_cell(ts, ingroup_nodes, outgroup_nodes, truth, n_out)
    if s_i % max(1, len(seeds) // 10) == 0:
        print(f"  done seed {s_i}/{len(seeds)} (seed={seed}, sites={ts.num_sites})", flush=True)

# Aggregate across seeds, per n_out.
summary: dict[int, dict] = {}
for n_out in n_outs:
    cells = list(per_seed[n_out].values())
    mp = [c["mean_max_prob"] for c in cells]
    ac = [c["accuracy"] for c in cells]
    summary[n_out] = {
        "n_seeds": len(cells),
        "mean_max_prob_mean": stats.mean(mp),
        "mean_max_prob_std": stats.stdev(mp) if len(mp) > 1 else 0.0,
        "accuracy_mean": stats.mean(ac),
        "accuracy_std": stats.stdev(ac) if len(ac) > 1 else 0.0,
        "n_sites_mean": stats.mean(c["n_sites"] for c in cells),
    }

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({
        "summary": summary,
        "per_seed": {str(n): per_seed[n] for n in n_outs},
        "config": {
            "n_ingroup": n_ingroup, "n_outgroup_pops": n_outgroup_pops,
            "length": length, "mu": mu, "rec_rate": rec_rate,
            "pop_size": pop_size, "outgroup_split_times": outgroup_split_times,
            "seeds": seeds, "n_outs": n_outs,
        },
    }, f, indent=2)
print(f"Wrote {out_json}", flush=True)
