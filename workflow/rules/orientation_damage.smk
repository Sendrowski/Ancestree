# Appendix benchmark: how a mis-declared ancestral allele damages an inferred
# genealogy, and how far along the sequence that damage reaches.
#
# Modularized rule file, ``include:``-d from workflow/Snakefile. The 20 Mb
# ingroup + 3-outgroup-ladder ARG of the genealogy comparison is reused, so no
# simulation happens here. One shared site set is built once. Each orientation
# scheme then declares an ancestral allele at every one of those sites, and
# each (ARG tool, scheme) pair is inferred and scored in its own job. All
# binning lives in the per-comparison shards, so the report only collects them
# and the figure only selects and draws.
#
# Build with::
#
#     snakemake --profile workflow/profiles/slurm \
#         results/reports/bench_orientation_damage.pdf
#
# ``DATA`` / ``REPORTS`` / ``RELATE_DIR`` come from the including Snakefile.

# Ancestral-allele orientation schemes. ``true_aa`` is the reference arm each
# other arm is paired against. The remaining five are the mis-orientation arms.
# ``random5`` mis-orients a uniformly random 5% of the sites, a draw that is
# independent of the derived-allele frequency, so the distance to the nearest
# mis-oriented site carries no information about local high-frequency-variant
# density.
ORIENTATION_SCHEMES = ["true_aa", "major_allele", "random5", "freq_biased5",
                       "fixed_tree_n1",
                       "fixed_tree_n3"]
ORIENTATION_MIS_SCHEMES = [s for s in ORIENTATION_SCHEMES if s != "true_aa"]
ORIENTATION_TOOLS = ["tsinfer", "relate"]
# Outgroups in the panel the shared site set is read off. The outgroups never
# reach the ARG tools. They exist so the fixed-tree schemes have something to
# condition their ancestral-allele call on.
ORIENTATION_N_OUT_REFERENCE = 3
# Fraction of shared sites the ``random5`` scheme mis-orients, and the seed of
# that draw. Held as rule parameters so the draw is reproducible and so further
# rates can be added by giving the scheme a wildcard.
ORIENTATION_FLIP_FRACTION = 0.05
ORIENTATION_FLIP_SEED = 2025
# Ingroup haplotype pairs the pairwise-TMRCA metrics pool over, and the seed
# that draws them. Every shard scores the identical pair set.
ORIENTATION_N_PAIRS = 80
ORIENTATION_PAIR_SEED = 7
ORIENTATION_RELATE_SEED = 1
# Site-block bootstrap behind the TMRCA standard errors.
ORIENTATION_N_BOOTSTRAP = 0
ORIENTATION_BOOTSTRAP_SEED = 20250902
# Bootstrap replicates one comparison shard evaluates concurrently. The
# replicate resamples are drawn up front in stream order, so this sets only the
# wall time. Held small because the shards themselves run concurrently.
ORIENTATION_BOOTSTRAP_THREADS = 4
# Scheme the appendix figure draws.
ORIENTATION_FIGURE_SCHEME = "freq_biased5"


rule orientation_sites:
    """Shared site set: biallelic, ACGT, segregating among the 20 ingroup
    haplotypes of the reused 20 Mb outgroup-ladder simulation."""
    input:
        trees = rules.simulate_local_tree_genealogy_og.output.trees,
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
    output:
        npz = f"{DATA}/orientation_sites.npz",
    params:
        n_out_reference = ORIENTATION_N_OUT_REFERENCE,
    resources:
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: 120 * attempt,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/orientation_sites.py"


rule orientation_map:
    """Declared ancestral allele at every shared site under one scheme.

    One job per scheme so the two fixed-tree fits, which are the only expensive
    ones, run alongside the three closed-form schemes. Wildcard: ``{scheme}``.
    """
    input:
        trees = rules.simulate_local_tree_genealogy_og.output.trees,
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
    output:
        json = f"{DATA}/orientation_map_{{scheme}}.json",
    params:
        flip_fraction = ORIENTATION_FLIP_FRACTION,
        flip_seed = ORIENTATION_FLIP_SEED,
        n_out_reference = ORIENTATION_N_OUT_REFERENCE,
    wildcard_constraints:
        scheme = r"|".join(ORIENTATION_SCHEMES),
    resources:
        mem_mb = lambda wildcards, attempt: 24000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(360, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/orientation_map.py"


rule orientation_arg_tsinfer:
    """One tsinfer + tsdate ARG on the ingroup panel under one scheme.

    Runs in ``envs/tsinfer.yml``, the only env carrying tsinfer and tsdate.
    Wildcard: ``{scheme}``.
    """
    input:
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
        orientation = f"{DATA}/orientation_map_{{scheme}}.json",
    output:
        trees = f"{DATA}/orientation_arg_tsinfer_{{scheme}}.trees",
    wildcard_constraints:
        scheme = r"|".join(ORIENTATION_SCHEMES),
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(240, attempt),
    conda:
        "../envs/tsinfer.yml"
    script:
        "../scripts/orientation_arg_tsinfer.py"


rule orientation_arg_relate:
    """One Relate ARG on the ingroup panel under one scheme.

    The Relate binary lives outside the repo (RELATE_DIR, built by
    ``install_relate``), so it is declared as an input and its directory handed
    to the script as a parameter. Wildcard: ``{scheme}``.
    """
    input:
        trees = rules.simulate_local_tree_genealogy_og.output.trees,
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
        orientation = f"{DATA}/orientation_map_{{scheme}}.json",
        relate = f"{RELATE_DIR}/bin/Relate",
    output:
        trees = f"{DATA}/orientation_arg_relate_{{scheme}}.trees",
    params:
        relate_dir = RELATE_DIR,
        relate_seed = ORIENTATION_RELATE_SEED,
        n_out_reference = ORIENTATION_N_OUT_REFERENCE,
    wildcard_constraints:
        scheme = r"|".join(ORIENTATION_SCHEMES),
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(240, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/orientation_arg_relate.py"


rule orientation_score:
    """Per-site genealogy metrics of one ARG against the true local trees.

    Its own shard per (tool, scheme): the walk visits every shared site in both
    tree sequences and is the second expensive step of the benchmark. Writes
    per-site normalised Robinson-Foulds and Kendall-Colijn distances plus the
    per (site, haplotype-pair) TMRCA grids the report pairs across arms.
    Wildcards: ``{tool}``, ``{scheme}``.
    """
    input:
        trees = rules.simulate_local_tree_genealogy_og.output.trees,
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
        arg = f"{DATA}/orientation_arg_{{tool}}_{{scheme}}.trees",
    output:
        npz = f"{DATA}/orientation_persite_{{tool}}_{{scheme}}.npz",
    params:
        n_pairs = ORIENTATION_N_PAIRS,
        pair_seed = ORIENTATION_PAIR_SEED,
        n_out_reference = ORIENTATION_N_OUT_REFERENCE,
    wildcard_constraints:
        tool = r"|".join(ORIENTATION_TOOLS),
        scheme = r"|".join(ORIENTATION_SCHEMES),
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(360, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/orientation_score.py"


rule orientation_report_shard:
    """Pair one mis-orientation arm against the true-ancestral-allele arm of
    the same tool and bin the per-site excesses.

    Its own shard per (tool, scheme): both binnings, by bp to the nearest
    mis-oriented site and by the number of mis-oriented sites inside the local
    tree interval, and the site-block bootstrap behind the TMRCA standard
    errors are computed here. Wildcards: ``{tool}``, ``{scheme}``.
    """
    input:
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
        orientation = f"{DATA}/orientation_map_{{scheme}}.json",
        persite_true_aa = f"{DATA}/orientation_persite_{{tool}}_true_aa.npz",
        persite_scheme = f"{DATA}/orientation_persite_{{tool}}_{{scheme}}.npz",
    output:
        json = f"{DATA}/orientation_comparison_{{tool}}_{{scheme}}.json",
    params:
        n_boot = ORIENTATION_N_BOOTSTRAP,
        boot_seed = ORIENTATION_BOOTSTRAP_SEED,
    wildcard_constraints:
        tool = r"|".join(ORIENTATION_TOOLS),
        scheme = r"|".join(ORIENTATION_MIS_SCHEMES),
    threads: ORIENTATION_BOOTSTRAP_THREADS
    resources:
        mem_mb = lambda wildcards, attempt: 32000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(240, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_orientation_comparison.py"


rule orientation_report:
    """Collect the comparison shards into the report the figure reads.

    All binning lives in the shards, so this step only wraps their blocks in
    the simulation parameters and the orientation-map summaries and renders the
    markdown tables."""
    input:
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
        sites = rules.orientation_sites.output.npz,
        maps = expand(f"{DATA}/orientation_map_{{scheme}}.json",
                      scheme=ORIENTATION_SCHEMES),
        comparisons = expand(
            f"{DATA}/orientation_comparison_{{tool}}_{{scheme}}.json",
            tool=ORIENTATION_TOOLS, scheme=ORIENTATION_MIS_SCHEMES),
    output:
        json = f"{REPORTS}/orientation_damage.json",
        md = f"{REPORTS}/orientation_damage.md",
    params:
        n_boot = ORIENTATION_N_BOOTSTRAP,
        boot_seed = ORIENTATION_BOOTSTRAP_SEED,
    resources:
        mem_mb = lambda wildcards, attempt: 8000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(30, attempt),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_orientation_damage.py"


rule orientation_damage_figure:
    """Appendix figure: excess Robinson-Foulds error against distance to the
    nearest mis-oriented site and against local-tree mis-orientation load."""
    input:
        report = rules.orientation_report.output.json,
    output:
        pdf = f"{REPORTS}/bench_orientation_damage.pdf",
    params:
        scheme = ORIENTATION_FIGURE_SCHEME,
    resources:
        mem_mb = 4000,
        runtime = 30,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_orientation_damage.py"


rule copy_orientation_damage_figure:
    """Copy the orientation-damage figure PDF into the manuscript figures dir."""
    input:
        damage = rules.orientation_damage_figure.output.pdf,
    output:
        damage = "reports/manuscripts/latex/figures/bench_orientation_damage.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "cp {input.damage} {output.damage}"
