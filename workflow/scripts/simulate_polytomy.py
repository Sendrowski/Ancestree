"""B5 simulation: msprime ARG used by the polytomy stress test.

Run directly for the default parameters::

    python workflow/scripts/simulate_polytomy.py

Or via snakemake (params come from the rule definition)::

    snakemake -j 1 simulate_polytomy
"""
import msprime


try:
    out_trees = snakemake.output[0]  # type: ignore[name-defined]
    samples = snakemake.params.samples  # type: ignore[name-defined]
    length = snakemake.params.length  # type: ignore[name-defined]
    mu = snakemake.params.mu  # type: ignore[name-defined]
    rec_rate = snakemake.params.rec_rate  # type: ignore[name-defined]
    pop_size = snakemake.params.pop_size  # type: ignore[name-defined]
    seed = snakemake.params.seed  # type: ignore[name-defined]
except NameError:
    # Standalone defaults: same as the rule's params block in workflow/Snakefile.
    out_trees = "results/data/polytomy_sim.trees"
    samples = 30
    length = 2e5
    mu = 1e-7
    rec_rate = 1e-8
    pop_size = 1e4
    seed = 42


print(
    f"Simulating ARG: n_samples={samples} length={length:g} mu={mu:g} "
    f"rec_rate={rec_rate:g} Ne={pop_size:g} seed={seed}",
    flush=True,
)
ts = msprime.sim_ancestry(
    samples=samples,
    sequence_length=length,
    recombination_rate=rec_rate,
    population_size=pop_size,
    random_seed=seed,
)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed)
ts.dump(out_trees)
print(
    f"Wrote {out_trees}: {ts.num_sites} sites, {ts.num_samples} haplotypes, "
    f"{ts.num_trees} local trees",
    flush=True,
)
