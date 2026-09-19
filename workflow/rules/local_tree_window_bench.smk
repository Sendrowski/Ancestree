# Appendix benchmark: LocalTreeInference accuracy as a function of the
# local-tree window size (in SNPs), a misspecified recombination rate, a
# misspecified mutation rate, and an injected phasing switch-error rate.
#
# Modularized rule file, ``include:``-d from workflow/Snakefile. One
# ingroup-only neutral msprime ARG (known JC69 ancestral truth) is shared
# across all four 1-D grids. Each cell builds local trees from genotypes
# alone, infers the ancestral allele with ARGBasedInference, and scores
# against truth. A separate true-ARG cell scores the genuine simulated
# genealogy as the achievable ceiling. The four grids are independent 1-D slices through
# the same origin (window=30snp, r=r_true, mu=mu_true, switch=0.0), not a
# full cross-product.
#
# Build with::
#
#     snakemake -j 4 --rerun-triggers mtime --nolock \
#         results/reports/local_tree_window_bench.md

# ``DATA`` / ``REPORTS`` / ``SCRIPTS`` come from the including Snakefile.

# Single shared simulation. Ingroup-only because this benchmark measures
# local-genealogy quality, not outgroup information. 20 haplotypes keeps
# the O(n^2) pairwise-coalescent HMM tractable across all 18 grid cells.
LT_WINDOW_CONFIG = dict(
    n_ingroup=20, length=1e8, mu=1.25e-8, rec_rate=1e-8,
    pop_size=3e4, seed=42,
)
# The genealogy-recovery benchmark shares the same 100 Mb ingroup-only ARG as
# the window/rec/mu/switch grids, so the whole analysis rests on one
# simulation.
LT_GENEALOGY_CONFIG = dict(LT_WINDOW_CONFIG, length=1e8)
# Primary grid: local-tree window sized to ~N SNPs.
# Block tracks the window (one tree per window's SNPs), so each point is a
# distinct local-tree resolution. 1-2 SNP/tree is dropped: a tree needs >2 SNPs
# to be defined, and at 100 Mb such a block count is not memory-feasible.
LT_WINDOW_WINDOWS = ["1snp", "2snp", "5snp", "10snp", "25snp", "50snp",
                     "100snp", "200snp", "400snp", "1000snp", "2000snp"]
# Secondary grid: assumed rec_rate as a multiple of the true rate (window
# held at the default, mu given correctly). "1.0" is the well-specified column.
LT_WINDOW_REC_SPECS = ["0.1", "1.0", "10.0"]
LT_WINDOW_DEFAULT_WINDOW = "8snp"
# The reference-window misspecification grids share a common 7-point ladder so
# they overlay on one axis. Rec and mu use the same multiplicative ladder
# (1.0 = well-specified, centred), and switch-error uses an additive ladder
# (0.0 = perfectly phased, at the origin).
LT_WINDOW_REC_SCAN_SPECS = ["0.01", "0.1", "1.0", "10.0", "100.0"]
LT_WINDOW_MU_SPECS = ["0.01", "0.1", "1.0", "10.0", "100.0"]
LT_WINDOW_SWITCH_SPECS = ["0.0", "0.05", "0.1", "0.2", "0.5"]

rule simulate_local_tree_window:
    """One neutral ingroup-only msprime ARG with JC69 ancestral truth."""
    output:
        trees = f"{DATA}/local_tree_window_sim.trees",
        meta = f"{DATA}/local_tree_window_sim_meta.json",
    params:
        **LT_WINDOW_CONFIG,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/simulate_local_tree_window.py"


rule simulate_local_tree_genealogy:
    """Dedicated 100 Mb ingroup-only ARG for the genealogy-recovery benchmark."""
    output:
        trees = f"{DATA}/local_tree_genealogy_sim_100mb.trees",
        meta = f"{DATA}/local_tree_genealogy_sim_100mb_meta.json",
    params:
        **LT_GENEALOGY_CONFIG,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/simulate_local_tree_window.py"


rule infer_local_tree_window_cell:
    """One (window, rec_rate) LocalTreeInference cell, scored vs truth.

    Window + rec_rate grid cells, with mu always given correctly. Wildcards:
    ``{window}`` (e.g. ``50snp``) and ``{rec}`` (rec_rate as a multiple of
    the true rate, e.g. ``0.1`` / ``1.0`` / ``10.0``).
    """
    input:
        trees = rules.simulate_local_tree_window.output.trees,
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        f"{DATA}/local_tree_window_w{{window}}_r{{rec}}.json",
    params:
        window = lambda wc: wc.window,
        rec_rate = lambda wc: wc.rec,
        mu = "1.0",  # mu given correctly in the window/rec grid
        switch = "0.0",  # perfectly phased in the window/rec grid
    wildcard_constraints:
        window = r"\d+snp",
        rec = r"[0-9.]+",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_mu_cell:
    """One mu-misspecification cell (window + rec_rate held at defaults).

    Wildcard ``{mu}`` is the assumed mutation rate as a multiple of the true
    simulation mu (e.g. ``0.25`` / ``1.0`` / ``4.0``). A misspecified mu
    rescales the inferred TMRCAs through both the emission rate and the
    time-grid calibration.
    """
    input:
        trees = rules.simulate_local_tree_window.output.trees,
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        f"{DATA}/local_tree_window_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m{{mu}}.json",
    params:
        window = LT_WINDOW_DEFAULT_WINDOW,
        rec_rate = "1.0",  # rec_rate given correctly in the mu grid
        mu = lambda wc: wc.mu,
        switch = "0.0",  # perfectly phased in the mu grid
    wildcard_constraints:
        mu = r"[0-9.]+",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_switch_cell:
    """One phasing-switch-error cell (window/rec/mu held at the origin).

    Wildcard ``{switch}`` is the per-het-site switch-error rate injected
    into the ingroup genotypes (e.g. ``0.0`` perfectly phased / ``0.01`` the
    1% used in the main-text robustness section / ``0.05``). Consecutive
    ingroup haplotypes are paired into pseudo-diploids and a per-pair
    orientation flips with this probability at each het site. The msprime
    ancestral truth is unchanged, only the phasing of the data fed to the
    inference varies.
    """
    input:
        trees = rules.simulate_local_tree_window.output.trees,
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        f"{DATA}/local_tree_window_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m1.0_s{{switch}}.json",
    params:
        window = LT_WINDOW_DEFAULT_WINDOW,
        rec_rate = "1.0",  # rec_rate given correctly in the switch grid
        mu = "1.0",  # mu given correctly in the switch grid
        switch = lambda wc: wc.switch,
    wildcard_constraints:
        switch = r"[0-9.]+",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_ceiling:
    """True-ARG ceiling: score the genuine simulated genealogy directly."""
    input:
        trees = rules.simulate_local_tree_window.output.trees,
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        f"{DATA}/local_tree_window_true_arg.json",
    params:
        window = "50snp",  # ignored in true_arg mode
        rec_rate = "true_arg",
        mu = "1.0",  # ignored in true_arg mode (always uses true mu)
        switch = "0.0",  # ignored in true_arg mode (scores genuine genealogy)
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule report_local_tree_window:
    """Aggregate the grid + ceiling into the table the appendix figure reads."""
    input:
        cells = expand(
            f"{DATA}/local_tree_window_w{{window}}_r{{rec}}.json",
            window=LT_WINDOW_WINDOWS, rec=LT_WINDOW_REC_SPECS,
        ),
        rec_scan_cells = expand(
            f"{DATA}/local_tree_window_w{LT_WINDOW_DEFAULT_WINDOW}_r{{rec}}.json",
            rec=LT_WINDOW_REC_SCAN_SPECS,
        ),
        mu_cells = expand(
            f"{DATA}/local_tree_window_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m{{mu}}.json",
            mu=LT_WINDOW_MU_SPECS,
        ),
        switch_cells = expand(
            f"{DATA}/local_tree_window_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m1.0_s{{switch}}.json",
            switch=LT_WINDOW_SWITCH_SPECS,
        ),
        ceiling = rules.infer_local_tree_window_ceiling.output[0],
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        json = f"{REPORTS}/local_tree_window_bench.json",
        md = f"{REPORTS}/local_tree_window_bench.md",
    params:
        windows = LT_WINDOW_WINDOWS,
        rec_specs = LT_WINDOW_REC_SPECS,
        rec_scan_specs = LT_WINDOW_REC_SCAN_SPECS,
        mu_specs = LT_WINDOW_MU_SPECS,
        switch_specs = LT_WINDOW_SWITCH_SPECS,
        default_window = LT_WINDOW_DEFAULT_WINDOW,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_local_tree_window.py"


LT_GEN_SHARDS = [str(w) for w in
                 [1, 2, 5, 10, 25, 50, 100, 200, 400, 1000, 2000]] + ["tsinfer"]


rule report_local_tree_genealogy_shard:
    """One window (or the tsinfer reference) of the genealogy bench.

    Sharded because the ensemble columns draw ENSEMBLE_MEMBERS members per
    window. Windows share no state, so each is an independent job.

    Runs under ``envs/tsinfer.yml``, which carries tsinfer."""
    input:
        trees = rules.simulate_local_tree_genealogy.output.trees,
        meta = rules.simulate_local_tree_genealogy.output.meta,
    output:
        json = f"{DATA}/local_tree_genealogy_shard_{{shard}}.json",
    params:
        shard = lambda wc: wc.shard,
    wildcard_constraints:
        shard = r"[0-9]+|tsinfer",
    threads: 4
    resources:
        runtime = 240,
        mem_mb = 16000,
    conda:
        "../envs/tsinfer.yml"
    script:
        "../scripts/report_local_tree_genealogy.py"


rule report_local_tree_genealogy:
    """Genealogy recovery: inferred vs. true local trees on the shared 100 Mb
    sim (TMRCA rank rho + calibration, normalised RF, Kendall-Colijn distance),
    varied over window size, plus a tsinfer (correct-AA) reference.

    Concatenates the per-window shards. No inference happens here."""
    input:
        trees = rules.simulate_local_tree_genealogy.output.trees,
        meta = rules.simulate_local_tree_genealogy.output.meta,
        shards = expand(f"{DATA}/local_tree_genealogy_shard_{{shard}}.json",
                        shard=LT_GEN_SHARDS),
    output:
        json = f"{REPORTS}/local_tree_genealogy.json",
        md = f"{REPORTS}/local_tree_genealogy.md",
    conda:
        "../envs/tsinfer.yml"
    script:
        "../scripts/report_local_tree_genealogy.py"


rule plot_local_tree_appendix_figs:
    """Render the two-panel local-tree robustness figure (window-size and
    misspecification scans, genealogy recovery) from the committed JSONs."""
    input:
        window = rules.report_local_tree_window.output.json,
        genealogy = rules.report_local_tree_genealogy.output.json,
    output:
        pdf = f"{REPORTS}/bench_local_tree_window_genealogy.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_local_tree_appendix_figs.py"


rule copy_local_tree_appendix_figures:
    """Copy the two-panel figure PDF into the manuscript figures directory."""
    input:
        rules.plot_local_tree_appendix_figs.output.pdf,
    output:
        "reports/manuscripts/latex/figures/bench_local_tree_window_genealogy.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "cp {input} {output}"


# =========================================================================
# Appendix outgroup comparison. The same three local-tree benchmarks, run with
# an outgroup ladder for n_out in {1, 3}, showing that a single outgroup already
# makes ancestral-allele recovery robust (ingroup-only, n_out=0, is the worst
# case shown in the main text). One shared deep 3-outgroup ladder sim is subset
# to n_out outgroups per cell (demography as in simulate_baseline), and the
# Brier is scored over the ingroup-polymorphic sites, matching the n_out=0
# figures. Data generation (per-n_out aggregation JSONs) is kept separate from
# the three side-by-side comparison plots.
LT_OG_NOUTS = ["1", "3"]
LT_OG_SPLIT_TIMES = [3e5, 9e5, 1.5e6]
# Pinned to 20 Mb: the appendix outgroup benchmark stays smaller than the
# 100 Mb main-text sim, since a deeper sim would be dominated by
# outgroup-private divergence sites.
LT_OG_CONFIG = dict(LT_WINDOW_CONFIG, length=2e7,
                    n_outgroup_pops=len(LT_OG_SPLIT_TIMES),
                    outgroup_split_times=LT_OG_SPLIT_TIMES)
LT_OG_GENEALOGY_CONFIG = dict(LT_WINDOW_CONFIG, length=2e7,
                              n_outgroup_pops=len(LT_OG_SPLIT_TIMES),
                              outgroup_split_times=LT_OG_SPLIT_TIMES)


rule simulate_local_tree_window_og:
    """20 Mb ingroup + 3-outgroup-ladder ARG for the appendix comparison."""
    output:
        trees = f"{DATA}/local_tree_window_og_sim.trees",
        meta = f"{DATA}/local_tree_window_og_sim_meta.json",
    params:
        **LT_OG_CONFIG,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/simulate_local_tree_window.py"


rule simulate_local_tree_genealogy_og:
    """20 Mb ingroup + 3-outgroup-ladder ARG for the genealogy comparison."""
    output:
        trees = f"{DATA}/local_tree_genealogy_og_sim_20mb.trees",
        meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json",
    params:
        **LT_OG_GENEALOGY_CONFIG,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/simulate_local_tree_window.py"


rule infer_local_tree_window_og_cell:
    """One (n_out, window, rec_rate) outgroup cell, scored vs truth."""
    input:
        trees = rules.simulate_local_tree_window_og.output.trees,
        meta = rules.simulate_local_tree_window_og.output.meta,
    output:
        f"{DATA}/local_tree_window_og{{n_out}}_w{{window}}_r{{rec}}.json",
    params:
        window = lambda wc: wc.window,
        rec_rate = lambda wc: wc.rec,
        mu = "1.0",
        switch = "0.0",
        n_out = lambda wc: wc.n_out,
    wildcard_constraints:
        n_out = r"[13]",
        window = r"\d+snp",
        rec = r"[0-9.]+",
    threads: 8
    resources:
        # The ensemble's scratch is bounded by _ensemble.MAX_SCRATCH_BYTES, so
        # the headroom here is for the HMM's own pairwise matrices.
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: 240 * attempt,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_og_mu_cell:
    """One (n_out, mu) outgroup cell (window + rec held at defaults)."""
    input:
        trees = rules.simulate_local_tree_window_og.output.trees,
        meta = rules.simulate_local_tree_window_og.output.meta,
    output:
        f"{DATA}/local_tree_window_og{{n_out}}_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m{{mu}}.json",
    params:
        window = LT_WINDOW_DEFAULT_WINDOW,
        rec_rate = "1.0",
        mu = lambda wc: wc.mu,
        switch = "0.0",
        n_out = lambda wc: wc.n_out,
    wildcard_constraints:
        n_out = r"[13]",
        mu = r"[0-9.]+",
    threads: 8
    resources:
        # The ensemble's scratch is bounded by _ensemble.MAX_SCRATCH_BYTES, so
        # the headroom here is for the HMM's own pairwise matrices.
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: 240 * attempt,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_og_switch_cell:
    """One (n_out, switch) outgroup cell (window/rec/mu held at the origin)."""
    input:
        trees = rules.simulate_local_tree_window_og.output.trees,
        meta = rules.simulate_local_tree_window_og.output.meta,
    output:
        f"{DATA}/local_tree_window_og{{n_out}}_w{LT_WINDOW_DEFAULT_WINDOW}_r1.0_m1.0_s{{switch}}.json",
    params:
        window = LT_WINDOW_DEFAULT_WINDOW,
        rec_rate = "1.0",
        mu = "1.0",
        switch = lambda wc: wc.switch,
        n_out = lambda wc: wc.n_out,
    wildcard_constraints:
        n_out = r"[13]",
        switch = r"[0-9.]+",
    threads: 8
    resources:
        # The ensemble's scratch is bounded by _ensemble.MAX_SCRATCH_BYTES, so
        # the headroom here is for the HMM's own pairwise matrices.
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: 240 * attempt,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule infer_local_tree_window_og_ceiling:
    """True-ARG ceiling for one n_out (genuine genealogy on the kept panel)."""
    input:
        trees = rules.simulate_local_tree_window_og.output.trees,
        meta = rules.simulate_local_tree_window_og.output.meta,
    output:
        f"{DATA}/local_tree_window_og{{n_out}}_true_arg.json",
    params:
        window = "50snp",
        rec_rate = "true_arg",
        mu = "1.0",
        switch = "0.0",
        n_out = lambda wc: wc.n_out,
    wildcard_constraints:
        n_out = r"[13]",
    threads: 8
    resources:
        # The ensemble's scratch is bounded by _ensemble.MAX_SCRATCH_BYTES, so
        # the headroom here is for the HMM's own pairwise matrices.
        mem_mb = lambda wildcards, attempt: 16000 * attempt,
        runtime = lambda wildcards, attempt: 240 * attempt,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_window_cell.py"


rule report_local_tree_window_og:
    """Aggregate the n_out outgroup cells into a data JSON (no figure: the
    side-by-side comparison plots are built by separate rules)."""
    input:
        cells = expand(
            DATA + "/local_tree_window_og{n_out}_w{window}_r{rec}.json",
            window=LT_WINDOW_WINDOWS, rec=LT_WINDOW_REC_SPECS, allow_missing=True,
        ),
        rec_scan_cells = expand(
            DATA + "/local_tree_window_og{n_out}_w" + LT_WINDOW_DEFAULT_WINDOW + "_r{rec}.json",
            rec=LT_WINDOW_REC_SCAN_SPECS, allow_missing=True,
        ),
        mu_cells = expand(
            DATA + "/local_tree_window_og{n_out}_w" + LT_WINDOW_DEFAULT_WINDOW + "_r1.0_m{mu}.json",
            mu=LT_WINDOW_MU_SPECS, allow_missing=True,
        ),
        switch_cells = expand(
            DATA + "/local_tree_window_og{n_out}_w" + LT_WINDOW_DEFAULT_WINDOW + "_r1.0_m1.0_s{switch}.json",
            switch=LT_WINDOW_SWITCH_SPECS, allow_missing=True,
        ),
        ceiling = DATA + "/local_tree_window_og{n_out}_true_arg.json",
        meta = rules.simulate_local_tree_window_og.output.meta,
    output:
        json = REPORTS + "/local_tree_window_og{n_out}.json",
        md = REPORTS + "/local_tree_window_og{n_out}.md",
    params:
        windows = LT_WINDOW_WINDOWS,
        rec_specs = LT_WINDOW_REC_SPECS,
        rec_scan_specs = LT_WINDOW_REC_SCAN_SPECS,
        mu_specs = LT_WINDOW_MU_SPECS,
        switch_specs = LT_WINDOW_SWITCH_SPECS,
        default_window = LT_WINDOW_DEFAULT_WINDOW,
    wildcard_constraints:
        n_out = r"[13]",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_local_tree_window.py"


rule report_local_tree_genealogy_og:
    """Genealogy recovery for one n_out on the 20 Mb outgroup sim, including
    the tsinfer (correct-AA) reference (tsinfer env)."""
    input:
        trees = rules.simulate_local_tree_genealogy_og.output.trees,
        meta = rules.simulate_local_tree_genealogy_og.output.meta,
    output:
        json = REPORTS + "/local_tree_genealogy_og{n_out}.json",
        md = REPORTS + "/local_tree_genealogy_og{n_out}.md",
    params:
        n_out = lambda wc: int(wc.n_out),
        run_tsinfer = True,
    wildcard_constraints:
        n_out = r"[13]",
    threads: 8
    resources:
        # Local-tree over a 20 Mb simulation. The ensemble scratch is bounded
        # by _ensemble.MAX_SCRATCH_BYTES, so this is near the plug-in size.
        mem_mb = lambda wildcards, attempt: 24000 * attempt,
        runtime = lambda wildcards, attempt: capped_runtime(600, attempt),
    conda:
        "../envs/tsinfer.yml"
    script:
        "../scripts/report_local_tree_genealogy.py"


rule plot_local_tree_scans_comparison:
    """Window-size and misspecification grids per outgroup count, laid out as
    the ingroup-only main-text figure."""
    input:
        window = expand(REPORTS + "/local_tree_window_og{n_out}.json", n_out=LT_OG_NOUTS),
    output:
        pdf = f"{REPORTS}/bench_local_tree_scans_outgroups.pdf",
    params:
        which = "scans",
        n_outs = LT_OG_NOUTS,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_local_tree_outgroup_comparison.py"


rule plot_local_tree_genealogy_comparison:
    """Fig: genealogy recovery, n_out=1 | n_out=3, shared y, legend below."""
    input:
        window = expand(REPORTS + "/local_tree_window_og{n_out}.json", n_out=LT_OG_NOUTS),
        genealogy = expand(REPORTS + "/local_tree_genealogy_og{n_out}.json", n_out=LT_OG_NOUTS),
    output:
        pdf = f"{REPORTS}/bench_local_tree_genealogy_outgroups.pdf",
    params:
        which = "genealogy",
        n_outs = LT_OG_NOUTS,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_local_tree_outgroup_comparison.py"


rule copy_local_tree_outgroup_comparison_figures:
    """Copy the outgroup-comparison figure PDFs into the manuscript."""
    input:
        scans = rules.plot_local_tree_scans_comparison.output.pdf,
        genealogy = rules.plot_local_tree_genealogy_comparison.output.pdf,
    output:
        scans = "reports/manuscripts/latex/figures/bench_local_tree_scans_outgroups.pdf",
        genealogy = "reports/manuscripts/latex/figures/bench_local_tree_genealogy_outgroups.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "cp {input.scans} {output.scans} && "
        "cp {input.genealogy} {output.genealogy}"
