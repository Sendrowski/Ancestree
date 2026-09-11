"""Mean Brier against the depth of a single outgroup, ARG and fixed-tree modes.

Reads the assembled ``single_outgroup_depth.json`` and renders one panel:
the mean Brier score over ingroup-polymorphic sites against the outgroup's
split time in coalescent units ``tau = T / (2 Ne)``, where ``T`` is the split
time in generations and ``Ne`` the diploid effective population size of the
ingroup. The raw split time in generations and the expected divergence in
substitutions per site, ``2 T mu`` with ``mu`` the per-site per-generation
mutation rate, are annotated on the bottom and on a secondary top axis.

No inference is re-run here.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    in_json = str(snakemake.input.report)  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
    out_png = str(snakemake.output.png)  # type: ignore[name-defined]
except NameError:
    in_json = "results/reports/single_outgroup_depth.json"
    out_pdf = "results/reports/bench_single_outgroup_depth.pdf"
    out_png = "results/reports/bench_single_outgroup_depth.png"

payload = json.loads(Path(in_json).read_text())
cfg = payload.get("sim_config", {})
Ne = float(cfg.get("pop_size", 3e4))
mu = float(cfg.get("mu", 1.25e-8))

depths = list(payload["depths"])
depth_vals = [float(d) for d in depths]
tau_vals = [t / (2 * Ne) for t in depth_vals]
subs_vals = [2 * t * mu for t in depth_vals]

mode_styles = {
    "ARG mode (lower is better)": dict(color="#1f77b4", marker="o", linestyle="-"),
    "Fixed-tree mode (lower is better)": dict(
        color="#d62728", marker="s", linestyle="--"),
}


def _series(cells: list[dict]) -> list[float]:
    """Mean Brier per depth, in the report's depth order.

    :param cells: One mode's cell list from the report payload.
    :return: Mean Brier per depth, NaN where a cell is absent.
    """
    by_depth = {c["depth_label"]: c["brier_polymorphic"] for c in cells}
    return [by_depth.get(d, float("nan")) for d in depths]


fig, ax = plt.subplots(figsize=(6.6, 4.1))
for label, cells in zip(
    mode_styles, (payload["arg_cells"], payload["vcf_cells"]),
):
    ax.plot(tau_vals, _series(cells), label=label, markersize=6,
            linewidth=1.6, alpha=0.9, **mode_styles[label])

ax.set_xscale("log")
ax.set_xlabel(r"Outgroup divergence $\tau = T / (2 N_e)$"
              "\n(millions of generations below)", labelpad=14)
ax.set_ylabel("Mean Brier (lower is better)")
ax.grid(alpha=0.3)
# Ticks at every other simulated depth, so the tau label, the generation
# count under it and the substitutions above it all sit at one position.
shown = list(zip(tau_vals, depth_vals))[::2]
ax.set_xticks([t for t, _ in shown])
ax.set_xticklabels([f"{t:.1f}" for t, _ in shown])
ax.minorticks_off()
# The generation count sits under its tau in grey, as a separate annotation:
# a tick label is one object and cannot be half-coloured.
for tau, gen in shown:
    ax.annotate(f"{gen / 1e6:g}", xy=(tau, -0.13),
                xycoords=("data", "axes fraction"), ha="center", va="top",
                fontsize=8, color="#666666", annotation_clip=False)

# Secondary top axis: expected divergence (subs/site) = 2·T·μ, pinned to the
# same depths as the bottom axis so the columns line up one-to-one.
sec = ax.secondary_xaxis(
    "top",
    functions=(lambda x: x * (2 * Ne) * 2 * mu,
               lambda x: x / ((2 * Ne) * 2 * mu)),
)
sec.set_xticks([t * (2 * Ne) * 2 * mu for t, _ in shown])
sec.set_xticklabels([f"{t * (2 * Ne) * 2 * mu:.3f}" for t, _ in shown])
sec.minorticks_off()
sec.set_xlabel("Substitutions per site", fontsize=9)

handles, labels = ax.get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.legend(handles, labels, loc="lower center", ncol=len(labels),
           frameon=True, fontsize=9, title="Inference mode",
           bbox_to_anchor=(0.5, 0.01))
for path in (out_pdf, out_png):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Wrote {out_pdf} and {out_png}", flush=True)
