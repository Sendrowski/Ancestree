"""B2 simulation: msprime ARG + matching VCF for the PolarBEAR comparison.

Writes both a ``.trees`` file (for Ancestree's ARG-mode kernel and PolarBEAR's
``tskit.load`` path) and a ``.vcf`` file (for PolarBEAR's cyvcf2 input).

Run directly::

    python workflow/scripts/simulate_polarbear.py

Or via snakemake::

    snakemake -j 1 simulate_polarbear
"""
import msprime


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_vcf = snakemake.output.vcf  # type: ignore[name-defined]
    samples = snakemake.params.samples  # type: ignore[name-defined]
    length = snakemake.params.length  # type: ignore[name-defined]
    mu = snakemake.params.mu  # type: ignore[name-defined]
    rec_rate = snakemake.params.rec_rate  # type: ignore[name-defined]
    pop_size = snakemake.params.pop_size  # type: ignore[name-defined]
    seed = snakemake.params.seed  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/polarbear_sim.trees"
    out_vcf = "results/data/polarbear_sim.vcf"
    samples = 20
    length = 1e5
    mu = 1e-8
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
with open(out_vcf, "w") as f:
    ts.write_vcf(f)
print(
    f"Wrote {out_trees} + {out_vcf}: {ts.num_sites} sites, "
    f"{ts.num_samples} haplotypes, {ts.num_trees} local trees",
    flush=True,
)
