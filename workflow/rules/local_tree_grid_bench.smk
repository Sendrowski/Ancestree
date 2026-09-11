# Appendix benchmark: LocalTreeInference accuracy over a 2-D grid of the
# local-tree WINDOW size and the HMM emission BLOCK size, both given in SNP
# equivalents, scored by mean Brier against the simulated ancestral truth.
#
# Modularized rule file, ``include:``-d from workflow/Snakefile after
# rules/local_tree_window_bench.smk, whose ingroup-only 100 Mb ARG and
# true-ARG ceiling cell this benchmark reuses. Each cell runs at the true
# mutation and recombination rates with perfectly phased genotypes, so the
# only thing varying is the (window, block) pair.
#
# An explicit block_size is not capped at the window: the two specs resolve
# to base pairs independently and the window is then rounded up to a whole
# number of blocks. Only block <= window gives distinct configurations, so
# just the lower triangle is enumerated, and every cell records the widths
# it actually ran at alongside the widths requested.
#
# Build with::
#
#     snakemake --profile workflow/profiles/slurm \
#         results/reports/bench_local_tree_grid.png

# ``DATA`` / ``REPORTS`` come from the including Snakefile.

#: Local-tree window widths, as SNP equivalents at the panel's mean density.
LT_GRID_WINDOWS = ["1snp", "2snp", "4snp", "8snp", "16snp", "32snp", "64snp"]
#: HMM emission block widths, same unit.
LT_GRID_BLOCKS = ["1snp", "2snp", "4snp", "8snp", "16snp", "32snp", "64snp"]


def _lt_grid_cells():
    """Enumerate the (window, block) pairs the grid tests.

    A block coarser than its window collapses to window == block, so only
    block <= window is distinct.

    :return: List of ``(window_spec, block_spec)`` pairs.
    """
    return [(w, b) for w in LT_GRID_WINDOWS for b in LT_GRID_BLOCKS
            if int(b[:-3]) <= int(w[:-3])]


LT_GRID_CELL_JSONS = [
    f"{DATA}/local_tree_grid_w{w}_b{b}.json" for w, b in _lt_grid_cells()
]


rule infer_local_tree_grid_cell:
    """One (window, block_size) LocalTreeInference cell, scored vs truth.

    Wildcards ``{window}`` and ``{block}`` are ``"<N>snp"`` specs. One job
    per cell so the grid runs concurrently. The fine end (1-SNP window over
    1-SNP blocks) is the expensive corner, hence the attempt-scaled
    resources.
    """
    input:
        trees = rules.simulate_local_tree_window.output.trees,
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        f"{DATA}/local_tree_grid_w{{window}}_b{{block}}.json",
    params:
        window = lambda wc: wc.window,
        block = lambda wc: wc.block,
    wildcard_constraints:
        window = r"\d+snp",
        block = r"\d+snp",
    threads: 8
    resources:
        # The ensemble's scratch is bounded by _ensemble.MAX_SCRATCH_BYTES, so
        # the footprint is set by the HMM's pairwise block matrices, which grow
        # as the block width shrinks.
        mem_mb = lambda wildcards, attempt: 24000 * attempt,
        runtime = lambda wildcards, attempt: min(360 * attempt, 720),
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/infer_local_tree_grid_cell.py"


rule report_local_tree_grid:
    """Assemble the lower-triangular grid of cells into the report pair."""
    input:
        cells = LT_GRID_CELL_JSONS,
        ceiling = rules.infer_local_tree_window_ceiling.output[0],
        meta = rules.simulate_local_tree_window.output.meta,
    output:
        json = f"{REPORTS}/local_tree_grid.json",
        md = f"{REPORTS}/local_tree_grid.md",
    params:
        windows = LT_GRID_WINDOWS,
        blocks = LT_GRID_BLOCKS,
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/report_local_tree_grid.py"


rule plot_local_tree_grid:
    """Heatmap of mean Brier over the window x block grid."""
    input:
        grid = rules.report_local_tree_grid.output.json,
    output:
        pdf = f"{REPORTS}/bench_local_tree_grid.pdf",
    conda:
        "../envs/bench.yml"
    script:
        "../scripts/plot_local_tree_grid.py"


rule copy_local_tree_grid_figure:
    """Copy the window-by-block heatmap into the manuscript figures dir."""
    input:
        pdf = rules.plot_local_tree_grid.output.pdf,
    output:
        pdf = "reports/manuscripts/latex/figures/bench_local_tree_grid.pdf",
    conda:
        "../envs/bench.yml"
    shell:
        "cp {input.pdf} {output.pdf}"
