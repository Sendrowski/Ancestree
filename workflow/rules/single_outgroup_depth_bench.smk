# Appendix benchmark: ancestral-allele accuracy for an ingroup plus exactly
# one outgroup, as a function of that outgroup's split time. Independent of
# the outgroup-divergence depth x spacing grid: a single outgroup has no
# relative spacing, so depth is the only axis and it is resolved on a fine
# log-uniform grid.
#
# Modularized rule file, ``include:``-d from workflow/Snakefile after the
# outgroup-divergence block, whose ``OUTGROUP_DIVERGENCE_CONFIG`` sets the shared simulator
# settings (ingroup size, sequence length, mutation and recombination rates,
# population sizes, seed) so the two benchmarks differ only in the outgroup
# ladder. Each depth is its own simulation, its own ARG-mode cell and its own
# fixed-tree (VCF) cell, so the grid runs concurrently.
#
# Build with::
#
#     snakemake --profile workflow/profiles/slurm \
#         results/reports/bench_single_outgroup_depth.png

# ``DATA`` / ``REPORTS`` / ``OUTGROUP_DIVERGENCE_CONFIG`` / ``capped_runtime``
# come from the including Snakefile.

#: Outgroup split times in generations, 16 values log-uniform from 1e4 to 1e7
#: inclusive (a factor of 10**(1/5) per step), rounded to two significant
#: figures.
SINGLE_OUTGROUP_DEPTHS = [
    "1e4", "1.6e4", "2.5e4", "4e4",
    "6.3e4", "1e5", "1.6e5", "2.5e5",
    "4e5", "6.3e5", "1e6", "1.6e6",
    "2.5e6", "4e6", "6.3e6", "1e7",
]


rule simulate_single_outgroup:
    """msprime ARG with an ingroup plus one outgroup splitting at ``{depth}``."""
    output:
        trees = f"{DATA}/single_outgroup_sim_d{{depth}}.trees",
        meta = f"{DATA}/single_outgroup_sim_d{{depth}}_meta.json",
    params:
        **OUTGROUP_DIVERGENCE_CONFIG,
    wildcard_constraints:
        depth = r"[0-9.eE+-]+",
    resources:
        # The footprint is set by the simulated sequence length, not by any
        # input, so the profile's input-scaled default does not cover it.
        mem_mb = lambda wildcards, attempt: 8000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(60, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/simulate_single_outgroup.py"


rule infer_single_outgroup_arg:
    """ARGBasedInference on the single-outgroup sim at one ``{depth}``."""
    input:
        trees = rules.simulate_single_outgroup.output.trees,
        meta = rules.simulate_single_outgroup.output.meta,
    output:
        f"{DATA}/single_outgroup_arg_d{{depth}}.json",
    params:
        mu = OUTGROUP_DIVERGENCE_CONFIG["mu"],
    wildcard_constraints:
        depth = r"[0-9.eE+-]+",
    resources:
        # The per-site posterior table is held in memory until the dump and
        # reaches ~100 MB of JSON at the deepest cells.
        mem_mb = lambda wildcards, attempt: 24000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(240, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_single_outgroup_arg.py"


rule infer_single_outgroup_vcf:
    """FixedTreeInference on the single-outgroup sim at one ``{depth}``."""
    input:
        trees = rules.simulate_single_outgroup.output.trees,
        meta = rules.simulate_single_outgroup.output.meta,
    output:
        f"{DATA}/single_outgroup_vcf_d{{depth}}.json",
    params:
        mu = OUTGROUP_DIVERGENCE_CONFIG["mu"],
    wildcard_constraints:
        depth = r"[0-9.eE+-]+",
    resources:
        # Divergence MLE fit plus the full per-site record list, which reaches
        # ~95 MB of JSON at the deepest cells.
        mem_mb = lambda wildcards, attempt: 24000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(240, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_single_outgroup_vcf.py"


rule report_single_outgroup:
    """Aggregate the 16 depths x {arg, vcf} cells into the report pair."""
    input:
        arg_cells = expand(
            f"{DATA}/single_outgroup_arg_d{{depth}}.json",
            depth=SINGLE_OUTGROUP_DEPTHS,
        ),
        vcf_cells = expand(
            f"{DATA}/single_outgroup_vcf_d{{depth}}.json",
            depth=SINGLE_OUTGROUP_DEPTHS,
        ),
    output:
        json = f"{REPORTS}/single_outgroup_depth.json",
        md = f"{REPORTS}/single_outgroup_depth.md",
    params:
        sim_config = {
            **OUTGROUP_DIVERGENCE_CONFIG,
            "n_outgroup_pops": 1,
            "depths": SINGLE_OUTGROUP_DEPTHS,
        },
        depths = SINGLE_OUTGROUP_DEPTHS,
    resources:
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(60, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_single_outgroup.py"


rule plot_single_outgroup:
    """Mean Brier against the single outgroup's depth, ARG and fixed-tree modes."""
    input:
        report = rules.report_single_outgroup.output.json,
    output:
        pdf = f"{REPORTS}/bench_single_outgroup_depth.pdf",
        png = f"{REPORTS}/bench_single_outgroup_depth.png",
    resources:
        mem_mb = lambda wildcards, attempt: 4000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(30, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_single_outgroup.py"
