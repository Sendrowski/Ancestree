"""B3 simulation: msprime ARG with haploid ingroup + N nested-split outgroups.

The outgroup populations are arranged in a phylogenetic ladder: the
closest outgroup shares its most-recent split with the ingroup, the
next-closest splits one level higher, and so on. Each outgroup
population contributes one haploid sample.

Writes:

- a ``.trees`` file (Ancestree's ARG-mode kernel reads this directly)
- a ``.vcf`` file (fastDFE consumes via cyvcf2)
- a ``.fasta`` file (fastDFE needs it for ``n_target_sites`` calibration —
  every position carries the simulated ancestral state at polymorphic
  positions and a uniformly random base at monomorphic positions)
- a ``meta.json`` carrying the ingroup + outgroup sample ids
  (outgroup_names is closest-first by ascending split-time)

Uses ``ploidy=1`` throughout so each msprime sample is one haplotype and
the VCF has one allele per (sample, site). Downstream est-sfs cells can
subset to fewer outgroups (closest-first) by taking the head of
``meta["outgroup_names"]``.

Run directly::

    python workflow/scripts/simulate_estsfs.py

Or via snakemake::

    snakemake -j 1 simulate_estsfs
"""
import json

import msprime
import numpy as np

from ancestree import OutgroupLadderTree, STATES, Site


try:
    out_trees = snakemake.output.trees  # type: ignore[name-defined]
    out_vcf = snakemake.output.vcf  # type: ignore[name-defined]
    out_fasta = snakemake.output.fasta  # type: ignore[name-defined]
    out_meta = snakemake.output.meta  # type: ignore[name-defined]
    n_ingroup = snakemake.params.n_ingroup  # type: ignore[name-defined]
    length = snakemake.params.length  # type: ignore[name-defined]
    mu = snakemake.params.mu  # type: ignore[name-defined]
    rec_rate = snakemake.params.rec_rate  # type: ignore[name-defined]
    pop_size = snakemake.params.pop_size  # type: ignore[name-defined]
    outgroup_pop_size = snakemake.params.outgroup_pop_size  # type: ignore[name-defined]
    outgroup_divergences = list(snakemake.params.outgroup_divergences)  # type: ignore[name-defined]
    seed = snakemake.params.seed  # type: ignore[name-defined]
    sim_spec = dict(snakemake.params.sim_spec)  # type: ignore[name-defined]
except NameError:
    out_trees = "results/data/estsfs_sim_jc.trees"
    out_vcf = "results/data/estsfs_sim_jc.vcf"
    out_fasta = "results/data/estsfs_sim_jc.fasta"
    out_meta = "results/data/estsfs_sim_jc_meta.json"
    n_ingroup = 40
    length = 2e5
    mu = 1e-8
    rec_rate = 1e-8
    pop_size = 1e4
    outgroup_pop_size = 1e5
    outgroup_divergences = [5e4, 1e5, 1.5e5]  # closest first
    seed = 42
    sim_spec = {"model": "HKY", "kappa": 1.0}


sim_model = sim_spec["model"]
if sim_model == "HKY":
    kappa_sim = float(sim_spec["kappa"])
    mutation_model = msprime.HKY(kappa=kappa_sim)
    sim_model_meta = {"model": "HKY", "kappa": kappa_sim}
elif sim_model == "GTR":
    rates_sim = [float(r) for r in sim_spec["rates"]]
    pi_sim = [float(p) for p in sim_spec["pi"]]
    if len(rates_sim) != 6:
        raise ValueError(f"GTR rates must have 6 entries, got {len(rates_sim)}")
    if len(pi_sim) != 4 or abs(sum(pi_sim) - 1.0) > 1e-9:
        raise ValueError(f"GTR pi must have 4 entries summing to 1, got {pi_sim}")
    mutation_model = msprime.GTR(
        relative_rates=rates_sim, equilibrium_frequencies=pi_sim,
    )
    sim_model_meta = {"model": "GTR", "rates": rates_sim, "pi": pi_sim}
else:
    raise ValueError(f"unknown sim model {sim_model!r}")


n_outgroup = len(outgroup_divergences)
demography = msprime.Demography()
demography.add_population(name="ingroup", initial_size=pop_size)
for k in range(n_outgroup):
    demography.add_population(name=f"outgroup_{k}", initial_size=outgroup_pop_size)
for k in range(n_outgroup):
    demography.add_population(name=f"anc_{k}", initial_size=pop_size)

# Nested splits, most-recent first. After each split the ingroup-side
# ancestor takes over as the "current" lineage for the next split.
prev = "ingroup"
for k, t in enumerate(outgroup_divergences):
    demography.add_population_split(
        time=float(t),
        derived=[prev, f"outgroup_{k}"],
        ancestral=f"anc_{k}",
    )
    prev = f"anc_{k}"

samples = {"ingroup": n_ingroup}
for k in range(n_outgroup):
    samples[f"outgroup_{k}"] = 1

print(
    f"Simulating (estsfs) ARG: ingroup={n_ingroup} haps (Ne={pop_size:g}), "
    f"outgroup_divergences={outgroup_divergences} (Ne_out={outgroup_pop_size:g}), "
    f"mu={mu:g}, seed={seed}",
    flush=True,
)
ts = msprime.sim_ancestry(
    samples=samples,
    sequence_length=length,
    recombination_rate=rec_rate,
    demography=demography,
    ploidy=1,
    random_seed=seed,
)
ts = msprime.sim_mutations(
    ts, rate=mu, random_seed=seed, model=mutation_model,
)

ingroup_names = [f"tsk_{i}" for i in range(n_ingroup)]
outgroup_names_raw = [f"tsk_{n_ingroup + k}" for k in range(n_outgroup)]

# Build Site records for the outgroup-order check (empirical closest-first).
sample_nodes = ts.samples()
sample_id_per_node = {
    int(node_id): f"tsk_{ind.id}"
    for ind in ts.individuals()
    for node_id in ind.nodes
}
sites_for_ordering: list[Site] = []
for variant in ts.variants():
    gens = variant.genotypes
    alleles = tuple(variant.alleles)
    tip_alleles: dict[str, str | None] = {}
    for sample_idx, node_id in enumerate(sample_nodes):
        sid = sample_id_per_node.get(int(node_id))
        g = int(gens[sample_idx])
        tip_alleles[sid] = (
            None if g < 0 or alleles[g] not in STATES else alleles[g]
        )
    sites_for_ordering.append(Site(
        chrom="1", pos=int(variant.site.position),
        alleles=tuple(a for a in alleles if a in STATES),
        tip_alleles=tip_alleles,
    ))
outgroup_names = OutgroupLadderTree.order_outgroups_by_divergence(
    sites_for_ordering, ingroup_names, outgroup_names_raw,
)

# ----------------------------------------------------------------- emit files

ts.dump(out_trees)
with open(out_vcf, "w") as f:
    ts.write_vcf(f)

rng = np.random.default_rng(seed)
seq = rng.choice(list("ACGT"), size=int(ts.sequence_length)).astype("U1")
for site in ts.sites():
    idx = int(site.position)
    if 0 <= idx < len(seq) and site.ancestral_state in STATES:
        seq[idx] = site.ancestral_state
with open(out_fasta, "w") as f:
    f.write(">1\n")
    seq_str = "".join(seq.tolist())
    for i in range(0, len(seq_str), 60):
        f.write(seq_str[i:i + 60] + "\n")

with open(out_meta, "w") as f:
    json.dump({
        "n_ingroup": n_ingroup,
        "n_outgroup": n_outgroup,
        "ingroup_names": ingroup_names,
        "outgroup_names": outgroup_names,
        "outgroup_divergences": outgroup_divergences,
        "mu": mu,
        "sim_model": sim_model_meta,
        "ploidy": 1,
    }, f, indent=2)

print(
    f"Wrote {out_trees} + {out_vcf} + {out_fasta} + {out_meta}: "
    f"{ts.num_sites} sites, {ts.num_samples} samples (ploidy=1), "
    f"outgroup_samples={outgroup_names}",
    flush=True,
)
