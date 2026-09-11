# Manuscript figure rules.
#
# Each manuscript figure and generated table is regenerated when the benchmark
# report or data it consumes changes. Build them all with::
#
#     snakemake -j 4 manuscript_figures
#
# ``DATA`` / ``REPORTS`` / ``LATEX`` come from the including Snakefile. The
# figures land in the manuscript repository cloned into ``reports/``, so these
# targets need that clone present. ``rule all`` does not.

FIGURES_DIR = f"{LATEX}/figures"


rule outgroup_divergence_figure:
    """Appendix: accuracy against outgroup divergence depth x spacing."""
    input:
        json = f"{REPORTS}/outgroup_divergence.json",
    output:
        pdf = f"{REPORTS}/bench_outgroup_divergence.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_outgroup_divergence.py"


rule msl_outgroup_count_figure:
    """Appendix: MSL recovery against the number of outgroups.

    A screen PNG is written beside the PDF, which the manuscript does not
    use, so only the PDF is declared."""
    input:
        npz = f"{DATA}/msl_polymorphic_chr1_143750000_150M.npz",
        meta = f"{DATA}/msl_polymorphic_chr1_143750000_150M_meta.json",
    output:
        f"{FIGURES_DIR}/bench_msl_outgroup_count.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/make_msl_outgroup_count_figure.py"


rule gallery_figure:
    """Kernel worked-example gallery (Figure A1): hand-built cases, no data deps."""
    input:
        maker = "workflow/scripts/gallery.py",
    output:
        f"{FIGURES_DIR}/gallery.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "python workflow/scripts/gallery.py && "
        "cp results/reports/gallery.pdf {output}"


rule root_examples_figure:
    """Root-state posteriors for example site patterns (appendix): one fixed
    genealogy, four tip configurations. Hand-built, no data deps."""
    input:
        maker = "workflow/scripts/gallery_root_examples.py",
    output:
        f"{FIGURES_DIR}/gallery_root_examples.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "python workflow/scripts/gallery_root_examples.py && "
        "cp results/reports/gallery_root_examples.pdf {output}"


rule expected_sfs_data:
    """ARG-mode posteriors on the baseline scenario ARG (n_out=0), pooled over
    the baseline chunks, into the true / MAP / fractional spectra plus the
    confidence histogram for the main-text SFS figure. Uses the same baseline
    sim as the robustness pipeline (one baseline sim)."""
    input:
        trees = expand(f"{DATA}/baseline10_chunk{{c}}.trees",
                       c=ROBUSTNESS_MSP_CHUNKS),
        metas = expand(f"{DATA}/baseline10_chunk{{c}}_meta.json",
                       c=ROBUSTNESS_MSP_CHUNKS),
    output:
        f"{DATA}/expected_sfs_baseline_n0.json",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/compute_expected_sfs.py"


rule sfs_fractional_figure:
    """MAP estimates distort the SFS and fractional attribution recovers it (main text)."""
    input:
        rules.expected_sfs_data.output,
    output:
        f"{FIGURES_DIR}/sfs_fractional_calls.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_sfs_fractional.py"


rule scenario_nout_fracsfs_figure:
    """Appendix: fractional vs MAP vs true SFS across scenarios x n_out (ARG mode),
    reconstructed from the per-cell robustness summaries (no inference re-run)."""
    input:
        cells = expand(
            f"{DATA}/robustness_cell_{{scenario}}_anc_arg_n{{n_out}}_summary.json",
            scenario=["baseline", "cpg_hypermut", "strong_ils",
                      "outgroup_clade", "slim", "pop_decline"],
            n_out=[0, 1, 3, 10],
        ),
    output:
        f"{FIGURES_DIR}/bench_scenario_nout_fracsfs.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_scenario_nout_fracsfs.py"


rule prior_demog_comparison_figure:
    """Appendix: the polarization prior (Kingman / adaptive / uniform) under
    demographic distortion (fixed-tree mode, no outgroup vs one outgroup), from
    the per-cell robustness summaries (no inference re-run)."""
    input:
        cells = expand(
            f"{DATA}/robustness_cell_{{scenario}}_{{method}}_n{{n_out}}_summary.json",
            scenario=["pop_constant", "pop_growth", "pop_decline"],
            method=["anc_ft", "anc_ft_adaptive", "anc_ft_uniform"],
            n_out=[0, 1, 3],
        ),
    output:
        f"{FIGURES_DIR}/bench_prior_demog.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_prior_demog_comparison.py"


rule ingroup_size_figure:
    """Ingroup-size saturation (Figure A2)."""
    input:
        f"{REPORTS}/ingroup_size.json",
        f"{REPORTS}/outgroup_divergence.json",
        f"{REPORTS}/single_outgroup_depth.json",
        maker = "workflow/scripts/plot_benchmarks.py",
    output:
        fig_copy = f"{FIGURES_DIR}/bench_ingroup_size.pdf",
        lines_copy = f"{FIGURES_DIR}/bench_outgroup_divergence_lines.pdf",
        ingroup_pdf = f"{REPORTS}/ingroup_size.pdf",
        divergence_pdf = f"{REPORTS}/outgroup_divergence.pdf",
        divergence_lines_pdf = f"{REPORTS}/outgroup_divergence_lines.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "python workflow/scripts/plot_benchmarks.py && "
        "cp {output.ingroup_pdf} {output.fig_copy} && "
        "cp {output.divergence_lines_pdf} {output.lines_copy}"


rule bench_sample_scaling:
    """Time the three modes over a ladder of ingroup sizes (Figure C16).

    Held apart from the figure: re-running this re-measures, and the timings
    are what the figure reports.
    """
    output:
        json = f"{REPORTS}/sample_scaling.json",
    params:
        sizes = [5, 10, 20, 40, 80],
    resources:
        mem_mb = 32000,
        runtime = 600,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/bench_sample_scaling.py"


rule sample_scaling_figure:
    input:
        json = rules.bench_sample_scaling.output.json,
    output:
        pdf = f"{REPORTS}/sample_scaling.pdf",
        fig_copy = f"{FIGURES_DIR}/sample_scaling.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_sample_scaling.py"


rule site_scaling_figure:
    """Site-count requirements for fixed-tree recovery (Figure A3)."""
    input:
        fixed = f"{REPORTS}/site_scaling_fixed.json",
        adaptive = f"{REPORTS}/site_scaling_adaptive.json",
    output:
        f"{FIGURES_DIR}/bench_site_scaling.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/make_site_scaling_figure.py"


# The three modes the per-bin figures draw, and the outgroup counts they draw
# them at. They read the intersection-scored cells, the same data the heatmaps
# use, so the rows are graded on one identical site set and inherit the
# ensemble size and focal convention rather than restating them.
BRIER_PER_BIN_MODES = ["anc_arg", "local_tree_unphased", "anc_ft"]


rule brier_per_bin_stack_figure:
    """Per-bin Brier across the unfolded SFS, one panel per n_out (main text)."""
    input:
        cells = expand(
            f"{DATA}/robustness_cell_intersect_{{scenario}}_{{method}}"
            f"_n{{n_out}}_summary.json",
            scenario=ROBUSTNESS_SCENARIOS, method=BRIER_PER_BIN_MODES,
            n_out=ROBUSTNESS_N_OUTS,
        ),
    output:
        f"{FIGURES_DIR}/bench_brier_per_bin_stack.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_brier_per_bin_stack.py"


rule scenario_sfs_figure:
    """Appendix: ingroup SFS per scenario (ARG mode, n_out=3), reconstructed
    from the per-cell robustness summaries."""
    input:
        cells = expand(
            f"{DATA}/robustness_cell_{{scenario}}_anc_arg_n3_summary.json",
            # The five FIG_SCENARIOS the figure plots.
            scenario=["baseline", "cpg_hypermut", "strong_ils",
                      "outgroup_clade", "slim"],
        ),
    output:
        pdf = f"{REPORTS}/scenario_sfs.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_scenario_sfs.py"




rule benchmark_tables:
    """The generated LaTeX tables, from the benchmark report JSONs.

    The EST-SFS tables also read the JC69 benchmark cells site by site, so
    those are declared alongside the report JSONs."""
    input:
        f"{REPORTS}/estsfs_agreement.json",
        f"{REPORTS}/msl_agreement_matrix.json",
        f"{REPORTS}/polarbear_agreement.json",
        f"{DATA}/estsfs_sim_jc.trees",
        f"{DATA}/estsfs_sim_jc_meta.json",
        expand(f"{DATA}/estsfs_ancestree_jc_JC_kingman_n{{n_out}}.json",
               n_out=ESTSFS_N_OUTS),
        expand(f"{DATA}/estsfs_baseline_jc_JC_n{{n_out}}.json",
               n_out=ESTSFS_N_OUTS),
        maker = "reports/scripts/make_benchmark_tables.py",
    output:
        expand(f"{LATEX}/tables/{{name}}.tex", name=MANUSCRIPT_TABLES),
    conda:
        "../envs/bench.yml"
    shell:
        "python reports/scripts/make_benchmark_tables.py"


rule manuscript_figures:
    """Aggregate: every figure the manuscript includes, and every table it inputs."""
    input:
        expand(f"{FIGURES_DIR}/{{name}}.pdf", name=MANUSCRIPT_FIGURES),
        rules.benchmark_tables.output,
        # The four tables outside MANUSCRIPT_TABLES, each from its own rule.
        # The three below are defined in the including Snakefile ahead of this
        # file's include, so ``rules`` resolves; msl_runtime_table is defined
        # further down here and is named by path.
        rules.kernel_bench_table.output,
        rules.edge_cases_table.output,
        rules.msl_informative_prior_table.output,
        f"{LATEX}/tables/msl_runtime.tex",


rule msl_runtime_report:
    """MSL chr1 coverage / serial runtime, from the run metas and the
    measured PolarBEAR call count."""
    input:
        local_tree = f"{DATA}/msl_localtree_chr1_full_meta.json",
        fixed_tree = f"{DATA}/msl_vcf_chr1_full_stream_meta.json",
        arg = f"{DATA}/msl_arg_chr1full_streaming_meta.json",
        polarbear = f"{DATA}/rerun_polarbear_meta.json",
        # The measured PolarBEAR call count the coverage column reports.
        agreement = f"{REPORTS}/msl_agreement_matrix.json",
    params:
        # Not an input: declaring it would pull the 43-minute est-sfs rerun
        # into the DAG. With the meta absent, the archived run's total is
        # carried by the script alongside the site count it belongs to.
        estsfs = f"{DATA}/rerun_estsfs_meta.json",
    output:
        json = f"{REPORTS}/msl_runtime.json",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_msl_runtime.py"


rule msl_runtime_table:
    """MSL chr1 coverage / serial runtime table, from the runtime report."""
    input:
        json = rules.msl_runtime_report.output.json,
        maker = "reports/scripts/make_msl_runtime_table.py",
    output:
        tex = "reports/manuscripts/latex/tables/msl_runtime.tex",
    conda:
        "../envs/bench.yml"
    shell:
        "python reports/scripts/make_msl_runtime_table.py"


rule scenario_sfs_figure_copy:
    """The scenario-SFS figure, in the manuscript's figure tree."""
    input:
        pdf = rules.scenario_sfs_figure.output.pdf,
    output:
        fig_copy = f"{FIGURES_DIR}/bench_scenario_sfs.pdf",
    shell:
        "cp {input.pdf} {output.fig_copy}"
