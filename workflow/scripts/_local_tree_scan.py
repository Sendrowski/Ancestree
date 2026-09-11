"""The local-tree scan panel: one axes carrying every input scan at once.

The window-size, recombination-rate, mutation-rate and phasing-switch scans
have different units and different numbers of points, so each is drawn against
a common normalised extent and given its own colour-coded x-axis outside the
frame. :func:`draw_scan_panel` renders one such panel from a
``local_tree_window_*`` report JSON, which lets the ingroup-only figure and the
per-outgroup panels share a layout.
"""
import matplotlib.lines as _mlines

WIN_C, RATE_C, SW_C, CEIL_C = "#1b6ca8", "#c7522a", "#2a7f62", "0.35"
LW = 1.9
AXIS_STEP = 36  # points between the stacked top axes


def _win_int(w):
    return int(str(w).replace("snp", ""))


def _even(n):
    return [i / (n - 1) for i in range(n)]


def draw_scan_panel(fig, ax, wb, *, ylabel=True):
    """Draw the four scans of report ``wb`` on ``ax``; return (handles, labels)."""
    windows = [_win_int(w) for w in wb["windows"]]
    cells = wb["cells"]
    brier = [cells[f"{w}__r1.0"]["mean_brier"] for w in wb["windows"]]
    ceil = wb["true_arg_ceiling"]
    dw = wb["default_window"]
    rec_specs, mu_specs = wb["rec_scan_specs"], wb["mu_specs"]
    switch_specs = wb["switch_specs"]
    rec_y = [wb["rec_scan_cells"][f"r{r}"]["mean_brier"] for r in rec_specs]
    mu_y = [wb["mu_cells"][f"{dw}__m{m}"]["mean_brier"] for m in mu_specs]
    sw_y = [wb["switch_cells"][f"{dw}__s{s}"]["mean_brier"] for s in switch_specs]

    xa, xb, xc = _even(len(windows)), _even(len(rec_specs)), _even(len(switch_specs))
    ax.plot(xa, brier, "^-", lw=LW, color=WIN_C, ms=5, label="window size")
    ax.plot(xb, rec_y, "o-", lw=LW, color=RATE_C, ms=5, alpha=0.85,
            label="recombination rate")
    ax.plot(_even(len(mu_specs)), mu_y, "s--", lw=LW, color=RATE_C, ms=5,
            alpha=0.85, label="mutation rate")
    ax.plot(xc, sw_y, "v-", lw=LW, color=SW_C, ms=5, label="phasing switch error")
    ax.axhline(ceil["mean_brier"], ls="--", lw=LW, color=CEIL_C, alpha=0.9,
               label="true-ARG ceiling")
    if ylabel:
        ax.set_ylabel(r"mean Brier score ($\downarrow$)", y=0.56)
    # Wide enough that the outermost tick labels (0.01x, 100x) clear the frame.
    ax.set_xlim(-0.06, 1.06)
    ax.set_xticks([])
    ax.grid(True, axis="y", ls=":", lw=0.5, alpha=0.6)
    ax.tick_params(axis="y", labelsize=9)
    handles, labels = ax.get_legend_handles_labels()

    def _extra_axis(offset, ticks, tick_labels, title, colour, side="top"):
        a = ax.twiny()
        a.set_xlim(ax.get_xlim())
        a.xaxis.set_ticks_position(side)
        a.xaxis.set_label_position(side)
        a.spines[side].set_position(("outward", offset))
        for other in ("top", "bottom", "left", "right"):
            if other != side:
                a.spines[other].set_visible(False)
        # Labels hug the spine on the stacked top axes; the bottom one keeps a
        # normal gap so it does not crowd the frame.
        a.tick_params(axis="x", labelsize=8, pad=1.0 if side == "top" else 3.5)
        a.set_xticks(ticks)
        a.set_xticklabels(tick_labels)
        a.set_xlabel(title, color=colour, fontsize=10)
        return a

    _extra_axis(0, xa, [str(w) for w in windows], "window size (SNPs)", WIN_C,
                side="bottom")
    _extra_axis(0, xb, [f"{float(r):g}×" for r in rec_specs],
                "assumed rate / true rate", RATE_C)
    _extra_axis(AXIS_STEP, xc, [f"{float(v) * 100:g}" for v in switch_specs],
                "switch-error rate (%)", SW_C)

    # Carry the plot frame up past the stacked axes so the whole block reads as
    # one box rather than a plot with loose rules above it.
    pos = ax.get_position()
    rise = (AXIS_STEP / 72.0) / fig.get_figheight()
    for x in (pos.x0, pos.x1):
        fig.add_artist(_mlines.Line2D(
            [x, x], [pos.y1, pos.y1 + rise], transform=fig.transFigure,
            color=ax.spines["left"].get_edgecolor(),
            lw=ax.spines["left"].get_linewidth(), zorder=0))
    return handles, labels
