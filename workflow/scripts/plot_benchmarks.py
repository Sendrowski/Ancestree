"""Plots for the ingroup-size and outgroup-divergence grids.

Reads :file:`results/reports/ingroup_size.json` and
:file:`results/reports/outgroup_divergence.json` and writes two PDF
plots. Both inputs are produced by the snakemake reports of the
matching name. Rule ``ingroup_size_figure`` runs this script. It refits
nothing and only renders.

Run::

    python workflow/scripts/plot_benchmarks.py
"""
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPORTS = Path("results/reports")
B7_JSON = REPORTS / "ingroup_size.json"
B8_JSON = REPORTS / "outgroup_divergence.json"
SINGLE_JSON = REPORTS / "single_outgroup_depth.json"
B7_PDF = REPORTS / "ingroup_size.pdf"
B8_PDF = REPORTS / "outgroup_divergence.pdf"
B8_LINES_PDF = REPORTS / "outgroup_divergence_lines.pdf"


def _save_both(fig, pdf_path: Path) -> None:
    """Write the figure as both PDF and PNG (same stem) for inline preview."""
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {pdf_path}")


# ------------------------------------------------------ B7: ingroup-size

def plot_ingroup_size(payload: dict, out_pdf: Path) -> None:
    """Two-panel layout (top: MAP accuracy, bottom: mean Brier score) over the
    ingroup-polymorphic interior 1 ≤ i ≤ 19 of the *unfolded* SFS projected to
    n=20. One line per n_ingroup (≥ 20 only, smaller cells can't project up).
    Per-panel legends show each n's accuracy (top) / Brier (bottom) over the
    same interior.
    """
    summary = {int(k): v for k, v in payload["summary"].items()}
    ns_proj = [
        n for n in sorted(summary)
        if summary[n].get("projected_per_bin_unfolded")
    ]
    n_ref = 20  # REFERENCE_N from the report
    # The two tail bins name site classes that empty out as n_ingroup grows,
    # leaving too few sites to estimate, so the figure covers the interior.
    bins = list(range(1, n_ref))

    fig, (ax_acc, ax_br) = plt.subplots(2, 1, figsize=(8, 5.3), sharex=True)
    cmap = plt.cm.viridis
    for i, n in enumerate(ns_proj):
        proj = summary[n]["projected_per_bin_unfolded"]
        acc_vals: list[float] = []
        br_vals: list[float] = []
        for b in bins:
            cell = proj.get(str(b)) or proj.get(b)
            if cell is None:
                acc_vals.append(float("nan"))
                br_vals.append(float("nan"))
                continue
            acc_vals.append(cell["accuracy_mean"])
            br_vals.append(cell["mean_brier_mean"])

        # Polymorphic-only (interior, 1..n_ref-1) MAP accuracy and Brier
        poly_n = sum(
            (proj.get(str(b)) or proj.get(b) or {}).get("n_eff_sites_mean", 0.0)
            for b in range(1, n_ref)
        )
        poly_hits = sum(
            (proj.get(str(b)) or proj.get(b) or {}).get("n_eff_sites_mean", 0.0)
            * (proj.get(str(b)) or proj.get(b) or {}).get("accuracy_mean", 0.0)
            for b in range(1, n_ref)
        )
        poly_br = sum(
            (proj.get(str(b)) or proj.get(b) or {}).get("n_eff_sites_mean", 0.0)
            * (proj.get(str(b)) or proj.get(b) or {}).get("mean_brier_mean", 0.0)
            for b in range(1, n_ref)
        )
        acc_poly = poly_hits / poly_n if poly_n else float("nan")
        br_poly = poly_br / poly_n if poly_n else float("nan")

        color = cmap(i / max(1, len(ns_proj) - 1))
        ax_acc.plot(bins, acc_vals, "o-", color=color, lw=1.3, ms=4, alpha=0.65,
                    label=f"n = {n:<4d} (acc = {acc_poly:.3f})")
        ax_br.plot(bins, br_vals, "o-", color=color, lw=1.3, ms=4, alpha=0.65,
                   label=f"n = {n:<4d} ($\\overline{{B}}$ = {br_poly:.3f})")

    # Fixed-tree (no-outgroup) reference = the pure Kingman SFS prior. With
    # no outgroup the fixed-tree posterior reduces to the prior, which puts
    # P(allele of count a is ancestral) = a/n (verified against
    # KingmanIngroupWeight). At unfolded derived count i the true ancestral
    # allele has count n-i, so the prior puts (n-i)/n on it and i/n on the
    # derived allele; its Brier score is therefore (i/n)^2 + (i/n)^2 and its
    # MAP is right iff i < n/2 — the "no genealogy" floor that the SFS alone
    # reaches, with no per-site tree.
    ref_br = [2.0 * (i / n_ref) ** 2 for i in bins]
    ref_acc = [1.0 if i < n_ref / 2 else (0.5 if i == n_ref / 2 else 0.0)
               for i in bins]
    ax_acc.plot(bins, ref_acc, "k--", lw=1.6, zorder=5,
                label="fixed-tree mode (Kingman prior)")
    ax_br.plot(bins, ref_br, "k--", lw=1.6, zorder=5,
               label="fixed-tree mode (Kingman prior)")

    for ax in (ax_acc, ax_br):
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.2, n_ref - 0.2)
        ax.set_xticks(range(2, n_ref, 2))

    ax_acc.set_ylabel("MAP accuracy")
    ax_br.set_ylabel("mean Brier score (lower = better)")
    ax_br.set_xlabel(r"ingroup derived-allele count $i$ (unfolded, projected to $n=20$)")
    ax_acc.set_ylim(0.0, 1.04)
    ax_br.set_ylim(0.0, 1.88)
    ax_acc.legend(loc="lower left", fontsize=7, framealpha=0.9,
                  title="$n_\\mathrm{ingroup}$")
    ax_br.legend(loc="upper left", fontsize=7, framealpha=0.9,
                 title="$n_\\mathrm{ingroup}$")
    _save_both(fig, out_pdf)
    plt.close(fig)


# ----------------------------------------------- B8: outgroup-divergence

def plot_outgroup_divergence(payload: dict, out_pdf: Path) -> None:
    """Two side-by-side heatmaps (ARG and fixed-tree) of overall ingroup-polymorphic
    accuracy across the (depth, spacing) grid.
    """
    depths = list(payload["depths"])
    spacings = list(payload["spacings"])
    n_out_shown = max(payload.get("n_outs", [3]))

    def _grid(cells: list[dict]) -> np.ndarray:
        """Build a (n_depth, n_spacing) accuracy grid from the flat list, at
        the full outgroup count.
        """
        m = np.full((len(depths), len(spacings)), np.nan)
        for c in cells:
            if int(c.get("n_out", n_out_shown)) != n_out_shown:
                continue
            i = depths.index(c["depth_label"])
            j = spacings.index(c["spacing"])
            m[i, j] = c["accuracy_polymorphic"]
        return m

    arg_grid = _grid(payload["arg_cells"])
    vcf_grid = _grid(payload["vcf_cells"])

    vmin = float(np.nanmin(np.concatenate([arg_grid.ravel(), vcf_grid.ravel()])))
    vmax = float(np.nanmax(np.concatenate([arg_grid.ravel(), vcf_grid.ravel()])))
    # Pad the range a touch so the colourbar shows differentiation even
    # for near-flat panels.
    pad = max(0.005, 0.05 * (vmax - vmin))
    vmin, vmax = max(0.0, vmin - pad), min(1.0, vmax + pad)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    titles = ("ARG mode", "Fixed-tree mode")
    for ax, grid, title in zip(axes, (arg_grid, vcf_grid), titles):
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(spacings))); ax.set_xticklabels(spacings, rotation=30, ha="right")
        ax.set_yticks(range(len(depths))); ax.set_yticklabels(depths)
        ax.set_xlabel("Spacing")
        ax.set_ylabel("Deepest-outgroup divergence (generations)")
        ax.set_title(title)
        # Annotate each cell with its value.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not math.isnan(grid[i, j]):
                    txt_color = "white" if grid[i, j] < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                            color=txt_color, fontsize=8)
        fig.colorbar(im, ax=ax, label="Polymorphic-sites accuracy",
                     fraction=0.046, pad=0.04)

    fig.suptitle(
        "B8: Outgroup-divergence sweep — accuracy on ingroup-polymorphic sites",
        fontsize=12,
    )
    _save_both(fig, out_pdf)
    plt.close(fig)


def plot_outgroup_divergence_lines(payload: dict, out_pdf: Path,
                                   single: "dict | None" = None) -> None:
    """Line view of B8 at three outgroup populations: mean Brier against the
    deepest-outgroup depth, one line per spacing strategy, one panel per
    inference mode. Only cells with ``n_out == 3`` are drawn. X-axis is
    expressed in coalescent units tau = T / (2 Ne), where T is the deepest
    outgroup's split time in generations and Ne the diploid effective
    population size, so the recommendation transfers across species. Raw
    generations and expected substitutions per site are annotated on a
    secondary top axis. The three mode panels are stacked on one shared x
    axis, so both annotated scales are drawn once for the whole stack.

    :param payload: Parsed ``outgroup_divergence.json`` report.
    :param out_pdf: Destination path for the figure.
    :param single: Parsed ``single_outgroup_depth.json`` report, drawn as the
        one-outgroup reference on the modes it covers.
    """
    cfg = payload.get("sim_config", {})
    Ne = float(cfg.get("pop_size", 30000.0))
    mu = float(cfg.get("mu", 1.25e-8))

    n_out_shown = 3

    depths = list(payload["depths"])
    spacings = list(payload["spacings"])
    depth_vals = [float(d) for d in depths]
    tau_vals = [t / (2 * Ne) for t in depth_vals]

    spacing_styles = {
        "linear":       dict(color="#1f77b4", marker="o", linestyle="-"),
        "geometric":    dict(color="#ff7f0e", marker="s", linestyle="-"),
        "compact_near": dict(color="#d62728", marker="^", linestyle="-"),
        "compact_far":  dict(color="#2ca02c", marker="D", linestyle="-"),
    }

    def _series(cells: list[dict], spacing: str) -> list[float]:
        """Mean Brier per depth (in `depths` order) for one spacing strategy,
        over the cells carrying all three outgroup populations."""
        by_depth = {
            c["depth_label"]: c["brier_polymorphic"]
            for c in cells
            if c["spacing"] == spacing
            and int(c.get("n_out", n_out_shown)) == n_out_shown
        }
        return [by_depth.get(d, float("nan")) for d in depths]

    fig, axes = plt.subplots(3, 1, figsize=(6.29, 4.70), sharex=True,
                             sharey=True)
    titles = ("ARG mode", "Local-tree mode", "Fixed-tree mode")
    cell_lists = (
        payload["arg_cells"],
        payload.get("local_tree_cells", []),
        payload["vcf_cells"],
    )
    # The single-outgroup benchmark covers the ARG and fixed-tree modes only,
    # so the local-tree panel carries no reference line.
    single_keys = ("arg_cells", None, "vcf_cells")
    for ax, cells, title, skey in zip(axes, cell_lists, titles, single_keys):
        for sp in spacings:
            ys = _series(cells, sp)
            ax.plot(tau_vals, ys, label=sp.replace("_", " "), markersize=6, linewidth=1.5,
                    alpha=0.7, **spacing_styles[sp])
        if single is not None and skey is not None:
            s_depths = [float(d) for d in single["depths"]]
            s_tau = [d / (2 * Ne) for d in s_depths]
            by_depth = {c["depth_label"]: c["brier_polymorphic"]
                        for c in single[skey]}
            ys = [by_depth[d] for d in single["depths"]]
            ax.plot(s_tau, ys, label="1 outgroup", color="0.25", lw=1.4,
                    ls=(0, (4, 2)), marker="", alpha=0.7, zorder=1)
        ax.set_xscale("log")
        ax.text(0.99, 0.90, title, transform=ax.transAxes, ha="right",
                va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="#bbbbbb", linewidth=0.6, alpha=0.92))
        ax.grid(alpha=0.3)
        ax.minorticks_off()

    # Force bottom ticks at the actual sim depths so labels line up, with
    # the raw generation count under each.
    # Every other depth carries a tick: 16 labels collide otherwise.
    shown = list(zip(tau_vals, depth_vals))[::2]
    ax_bottom = axes[-1]
    ax_bottom.set_xticks([t for t, _ in shown])
    ax_bottom.set_xticklabels([f"{t:.1f}" for t, _ in shown])
    ax_bottom.minorticks_off()
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    ax_bottom.set_xlabel(r"Outgroup divergence $\tau$"
                         "\n(millions of generations below)",
                         labelpad=20)
    # The generation count sits under its tau in grey, as a separate
    # annotation: a tick label is one object and cannot be half-coloured.
    # Offset in points from the axis line, so the row keeps its place under
    # the tau labels whatever height the stacked panels end up with.
    for tau, gen in shown:
        ax_bottom.annotate(f"{gen / 1e6:g}", xy=(tau, 0.0),
                           xycoords=("data", "axes fraction"),
                           xytext=(0, -21), textcoords="offset points",
                           ha="center", va="top", fontsize=8, color="#666666",
                           annotation_clip=False)
    # Secondary top axis on the top panel: expected divergence (subs/site)
    # = 2·T·μ, with ticks pinned to the same depths as the bottom axis so the
    # columns line up one-to-one down the stack.
    sec = axes[0].secondary_xaxis(
        "top",
        functions=(lambda x: x * (2 * Ne) * 2 * mu,
                   lambda x: x / ((2 * Ne) * 2 * mu)),
    )
    sec_subs = [t * (2 * Ne) * 2 * mu for t, _ in shown]
    sec.set_xticks(sec_subs)
    sec.set_xticklabels([f"{v:.3f}" for v in sec_subs])
    sec.minorticks_off()
    sec.set_xlabel("Substitutions per site", fontsize=9)

    axes[1].set_ylabel("Mean Brier (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    # Its own row: the panels are laid out above it, so the frame clears the
    # axis labels rather than sitting on them.
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.subplots_adjust(hspace=0.0)
    # Centred on the axes box rather than the figure: the y-axis label sits
    # outside the box, so the two centres are not the same point.
    box = axes[-1].get_position()
    # Stretched to the axes box: same left and right edge as the panels.
    legend = fig.legend(handles, labels, loc="lower left", ncol=len(labels),
                        frameon=True, fontsize=8.42, mode="expand",
                        title="Outgroup spacing", columnspacing=1.9,
                        handlelength=1.85, borderpad=0.44,
                        bbox_to_anchor=(box.x0, 0.005, box.width, 0.07),
                        bbox_transform=fig.transFigure)
    legend.get_title().set_fontsize(8.42)
    _save_both(fig, out_pdf)
    plt.close(fig)


def main() -> None:
    """Render the ingroup-size and outgroup-divergence benchmark figures.

    Both inputs are required: a missing one is a pipeline error, not a reason
    to emit half the figures and exit successfully.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    missing = [p for p in (B7_JSON, B8_JSON) if not p.exists()]
    if missing:
        raise SystemExit(
            "plot_benchmarks: missing input(s): "
            + ", ".join(str(p) for p in missing)
        )
    with open(B7_JSON) as f:
        plot_ingroup_size(json.load(f), B7_PDF)
    single = None
    if SINGLE_JSON.exists():
        with open(SINGLE_JSON) as f:
            single = json.load(f)
    with open(B8_JSON) as f:
        b8 = json.load(f)
    plot_outgroup_divergence(b8, B8_PDF)
    plot_outgroup_divergence_lines(b8, B8_LINES_PDF, single)


if __name__ == "__main__":
    main()
