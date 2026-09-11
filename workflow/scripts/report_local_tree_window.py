"""Aggregate the local-tree window benchmark cells into a report.

Reads the per-cell JSONs (window x rec_rate grid, the reference-window
recombination / mutation / switch-error misspecification scans, and the
true-ARG ceiling cell), and writes:

- ``local_tree_window_bench.json`` — the full grid of metrics.
- ``local_tree_window_bench.md`` — sub-tables (window x assumed rec_rate,
  the mu-misspecification scan, the phasing-switch-error scan), each cell
  = Brier / MAP / P(true), sharing the true-ARG ceiling row.

The appendix figure over these scans is rendered by
:mod:`plot_local_tree_appendix_figs`, which reads the JSON above.

Run via snakemake::

    snakemake -j 1 results/reports/local_tree_window_bench.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


try:
    in_cells = list(snakemake.input.cells)  # type: ignore[name-defined]
    in_rec_scan_cells = list(snakemake.input.rec_scan_cells)  # type: ignore[name-defined]
    in_mu_cells = list(snakemake.input.mu_cells)  # type: ignore[name-defined]
    in_switch_cells = list(snakemake.input.switch_cells)  # type: ignore[name-defined]
    in_ceiling = str(snakemake.input.ceiling)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    out_md = str(snakemake.output.md)  # type: ignore[name-defined]
    # The appendix outgroup-comparison aggregation requests only json/md (the
    # 2-panel figure is built by a separate plotting rule), so the overlay
    # figure below is skipped when no pdf/png output is wired in.
    out_pdf = getattr(snakemake.output, "pdf", None)  # type: ignore[name-defined]
    windows = list(snakemake.params.windows)  # type: ignore[name-defined]
    rec_specs = list(snakemake.params.rec_specs)  # type: ignore[name-defined]
    rec_scan_specs = list(snakemake.params.rec_scan_specs)  # type: ignore[name-defined]
    mu_specs = list(snakemake.params.mu_specs)  # type: ignore[name-defined]
    switch_specs = list(snakemake.params.switch_specs)  # type: ignore[name-defined]
    default_window = str(snakemake.params.default_window)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    REPORTS = "results/reports"
    windows = ["1snp", "2snp", "5snp", "10snp", "25snp", "50snp", "100snp",
               "200snp", "400snp", "1000snp", "2000snp"]
    rec_specs = ["0.1", "1.0", "10.0"]
    rec_scan_specs = ["0.01", "0.1", "1.0", "10.0", "100.0"]
    mu_specs = ["0.01", "0.1", "1.0", "10.0", "100.0"]
    switch_specs = ["0.0", "0.05", "0.1", "0.2", "0.5"]
    default_window = "30snp"
    in_cells = [
        f"{DATA}/local_tree_window_w{w}_r{r}.json"
        for w in windows for r in rec_specs
    ]
    in_rec_scan_cells = [
        f"{DATA}/local_tree_window_w{default_window}_r{r}.json"
        for r in rec_scan_specs
    ]
    in_mu_cells = [
        f"{DATA}/local_tree_window_w{default_window}_r1.0_m{m}.json"
        for m in mu_specs
    ]
    in_switch_cells = [
        f"{DATA}/local_tree_window_w{default_window}_r1.0_m1.0_s{s}.json"
        for s in switch_specs
    ]
    in_ceiling = f"{DATA}/local_tree_window_true_arg.json"
    in_meta = f"{DATA}/local_tree_window_sim_meta.json"
    out_json = f"{REPORTS}/local_tree_window_bench.json"
    out_md = f"{REPORTS}/local_tree_window_bench.md"
    out_pdf = None


meta = json.loads(Path(in_meta).read_text())

# Load every window/rec cell, keyed by (window, rec_spec).
cells: dict[tuple[str, str], dict] = {}
for p in in_cells:
    d = json.loads(Path(p).read_text())
    cells[(d["window"], d["rec_rate_spec"])] = d
# Load the mu-misspecification sweep, keyed by mu_spec.
mu_cells: dict[str, dict] = {}
for p in in_mu_cells:
    d = json.loads(Path(p).read_text())
    mu_cells[d["mu_spec"]] = d
# Load the phasing-switch-error sweep, keyed by switch_spec.
switch_cells: dict[str, dict] = {}
for p in in_switch_cells:
    d = json.loads(Path(p).read_text())
    switch_cells[d["switch_spec"]] = d
# Load the recombination-misspecification scan at the default window.
rec_scan_cells: dict[str, dict] = {}
for p in in_rec_scan_cells:
    d = json.loads(Path(p).read_text())
    rec_scan_cells[d["rec_rate_spec"]] = d
ceiling = json.loads(Path(in_ceiling).read_text())


def _win_snps(w: str) -> int:
    return int(w[:-3])


# ------------------------------------------------------------- JSON payload
payload = {
    "n_out": ceiling.get("n_out", 0),
    "simulation": {
        "n_ingroup": meta["n_ingroup"],
        "length": meta["length"],
        "mu": meta["mu"],
        "rec_rate_true": meta["rec_rate"],
        "pop_size": meta["pop_size"],
        "seed": meta["seed"],
        "n_sites": meta["n_sites"],
        "n_trees": meta["n_trees"],
    },
    "windows": windows,
    "rec_specs": rec_specs,
    "rec_scan_specs": rec_scan_specs,
    "mu_specs": mu_specs,
    "switch_specs": switch_specs,
    "default_window": default_window,
    "true_arg_ceiling": {
        "mean_brier": ceiling["mean_brier"],
        "mean_map": ceiling["mean_map"],
        "mean_ptrue": ceiling["mean_ptrue"],
        "n_sites": ceiling["n_sites"],
        "runtime": ceiling["runtime"],
    },
    "cells": {
        f"{w}__r{r}": cells[(w, r)] for w in windows for r in rec_specs
    },
    "mu_cells": {
        f"{default_window}__m{m}": mu_cells[m] for m in mu_specs
    },
    "switch_cells": {
        f"{default_window}__s{s}": switch_cells[s] for s in switch_specs
    },
    "rec_scan_cells": {
        f"r{r}": rec_scan_cells[r] for r in rec_scan_specs
    },
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(payload, f, indent=2)


# ----------------------------------------------------------------- md table
def _fmt_secs(s: float) -> str:
    if s < 60.0:
        return f"{s:.1f}s"
    return f"{s/60:.1f}m"


lines: list[str] = [
    "# Local-tree window-size + recombination-misspecification benchmark",
    "",
    (f"One neutral msprime ARG (ingroup-only, {meta['n_ingroup']} haplotypes, "
     f"L={meta['length']:,.0f} bp, mu={meta['mu']:g}, "
     f"true rec_rate={meta['rec_rate']:g}, Ne={meta['pop_size']:,.0f}, JC69, "
     f"seed={meta['seed']}): {meta['n_sites']:,} sites, "
     f"{meta['n_trees']:,} true local trees."),
    "",
    ("`LocalTreeInference` is run per cell and scored against the msprime "
     "ancestral truth. Each cell shows **Brier / MAP / P(true)** (Brier "
     "lower = better). The three sweeps are independent 1-D slices through "
     "the same origin (window=" + default_window + ", r=r_true, "
     "mu=mu_true); the true-ARG ceiling scores the genuine simulated "
     "genealogy through `ARGBasedInference`."),
    "",
    "## Window size x assumed recombination rate (mu well-specified)",
    "",
    "| window (SNPs) | "
    + " | ".join(f"{float(r):g}*r_true" if float(r) != 1.0 else "r_true"
                  for r in rec_specs)
    + " | runtime |",
    "| --- | " + " | ".join("---" for _ in rec_specs) + " | --- |",
]
for w in windows:
    row_cells: list[str] = []
    row_runtime = 0.0
    for r in rec_specs:
        c = cells[(w, r)]
        row_runtime += c["runtime"]
        row_cells.append(
            f"{c['mean_brier']:.3f} / {c['mean_map']:.3f} / {c['mean_ptrue']:.3f}"
        )
    lines.append(
        f"| {_win_snps(w)} | " + " | ".join(row_cells)
        + f" | {_fmt_secs(row_runtime)} |"
    )
# True-ARG ceiling row (single value, spans the rec_rate columns).
ceil_str = (
    f"{ceiling['mean_brier']:.3f} / {ceiling['mean_map']:.3f} / "
    f"{ceiling['mean_ptrue']:.3f}"
)
lines.append(
    "| **true ARG (ceiling)** | "
    + " | ".join(ceil_str for _ in rec_specs)
    + f" | {_fmt_secs(ceiling['runtime'])} |"
)
lines.append("")

# ------------------------------------------- mu-misspecification sub-table
lines += [
    f"## Assumed mutation rate (window={_win_snps(default_window)} SNPs, "
    "r=r_true)",
    "",
    ("The inference's assumed `mu` is swept as a multiple of the true "
     "simulation mu; the true mu (and the simulated data) are unchanged. "
     "A misspecified mu rescales the inferred TMRCAs through both the "
     "emission rate and the time-grid calibration."),
    "",
    "| assumed mu | "
    + " | ".join(f"{float(m):g}*mu_true" if float(m) != 1.0 else "mu_true"
                  for m in mu_specs)
    + " | runtime |",
    "| --- | " + " | ".join("---" for _ in mu_specs) + " | --- |",
]
mu_row: list[str] = []
mu_runtime = 0.0
for m in mu_specs:
    c = mu_cells[m]
    mu_runtime += c["runtime"]
    mu_row.append(
        f"{c['mean_brier']:.3f} / {c['mean_map']:.3f} / {c['mean_ptrue']:.3f}"
    )
lines.append(
    "| Brier / MAP / P(true) | " + " | ".join(mu_row)
    + f" | {_fmt_secs(mu_runtime)} |"
)
lines.append(
    "| **true ARG (ceiling)** | "
    + " | ".join(ceil_str for _ in mu_specs)
    + f" | {_fmt_secs(ceiling['runtime'])} |"
)
lines.append("")

# ----------------------------------------- switch-error sweep sub-table
lines += [
    f"## Phasing switch-error rate (window={_win_snps(default_window)} SNPs, "
    "r=r_true, mu=mu_true)",
    "",
    ("A switch-error process is injected into the ingroup genotypes fed to "
     "inference: consecutive haplotypes are paired into pseudo-diploids and "
     "a per-pair phase orientation flips with the given probability at each "
     "heterozygous site. The msprime ancestral truth (and ingroup allele "
     "frequencies) are unchanged. `0%` = perfectly phased; `1%` is the rate "
     "used in the main-text robustness section."),
    "",
    "| switch rate | "
    + " | ".join(f"{float(s) * 100:g}%" for s in switch_specs)
    + " | runtime |",
    "| --- | " + " | ".join("---" for _ in switch_specs) + " | --- |",
]
switch_row: list[str] = []
switch_runtime = 0.0
for s in switch_specs:
    c = switch_cells[s]
    switch_runtime += c["runtime"]
    switch_row.append(
        f"{c['mean_brier']:.3f} / {c['mean_map']:.3f} / {c['mean_ptrue']:.3f}"
    )
lines.append(
    "| Brier / MAP / P(true) | " + " | ".join(switch_row)
    + f" | {_fmt_secs(switch_runtime)} |"
)
lines.append(
    "| **true ARG (ceiling)** | "
    + " | ".join(ceil_str for _ in switch_specs)
    + f" | {_fmt_secs(ceiling['runtime'])} |"
)
lines.append("")
Path(out_md).write_text("\n".join(lines))


# ------------------------------------------------------------------- figure
# One panel overlaying the three reference-window misspecification scans on a
# shared 7-point step axis. Recombination and mutation rate share the bottom
# (multiple-of-true) axis, where 1x is well-specified and centred. The
# additive phasing switch-error rate is read off the top axis, where 0% is
# perfectly phased and sits at the origin.
if not out_pdf:
    print(f"Wrote {out_json}, {out_md} (data only, no figure)", flush=True)
else:
    idx = list(range(len(rec_scan_specs)))
    rec_y = [rec_scan_cells[r]["mean_brier"] for r in rec_scan_specs]
    mu_y = [mu_cells[m]["mean_brier"] for m in mu_specs]
    sw_y = [switch_cells[s]["mean_brier"] for s in switch_specs]

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(idx, rec_y, "o-", color="C0", lw=1.6, alpha=0.85,
            label="recombination rate")
    ax.plot(idx, mu_y, "s-", color="C3", lw=1.6, alpha=0.85,
            label="mutation rate")
    ax.plot(idx, sw_y, "^-", color="C2", lw=1.6, alpha=0.85,
            label="phasing switch error (top axis)")
    ax.axhline(ceiling["mean_brier"], ls="--", color="0.35", lw=2.8,
               label="true-ARG ceiling")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{float(r):g}$\\times$" for r in rec_scan_specs])
    ax.set_xlabel(r"assumed rate as a multiple of the truth "
                  r"($\times\,r_\mathrm{true}$, $\times\,\mu_\mathrm{true}$)")
    ax.set_ylabel("mean Brier score (lower = better)")
    ax.grid(True, which="major", ls=":", lw=0.5, alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(idx)
    ax_top.set_xticklabels([f"{float(s) * 100:g}%" for s in switch_specs])
    ax_top.set_xlabel("injected phasing switch-error rate")

    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_json}, {out_md}, {out_pdf}", flush=True)
