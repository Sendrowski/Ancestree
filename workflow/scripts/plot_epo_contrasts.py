"""Unfolded spectrum of the MSL sample, or of the simulation matched to the EPO
comparison, under successive changes to EPO's reconstruction.

EPO is fixed-tree MAP on the reference sequence at the human--chimpanzee
ancestor under a single mutation rate. One curve each changes one setting of
it: the posterior-weighted spectrum, a separate mutation rate in the CpG
context, and local-tree mode. The ingroup sample and the ingroup MRCA are
chained from local-tree mode, since fixed-tree mode on EPO's tree cannot take
a sample, and the last curve applies every change. Curves give the percent
difference from the EPO spectrum on the data, where EPO is the zero line, or
from the simulated truth: at the ingroup MRCA for counts 1 to ``n - 1`` and at
the ingroup--Pan MRCA for counts 0 and ``n``. Configurations read at the
ingroup MRCA show the polymorphic counts alone.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    IN_JSON = Path(snakemake.input.json)  # type: ignore[name-defined]
    FIG = Path(snakemake.output.pdf)  # type: ignore[name-defined]
    SIMULATION = bool(snakemake.params.simulation)  # type: ignore[name-defined]
except NameError:
    IN_JSON = Path("results/reports/epo_spectra.json")
    FIG = Path("results/reports/epo_spectra.pdf")
    SIMULATION = False

#: EPO's reconstruction under its model on the simulation; on the data it is the zero line.
EPO = ("fixed_tree_reference_one_rate", "map")
EPO_ONE_RATE = "fixed_tree_reference_one_rate" if SIMULATION else "fixed_tree_reference_epo_model_one_rate"
EPO_CLASS_RATE = ("fixed_tree_reference_class_rate" if SIMULATION
                  else "fixed_tree_reference_epo_model_class_rate")
#: Configuration: method key, estimate, legend label, colour.
CONFIGURATIONS = [
    (EPO_ONE_RATE, "weighted", "posterior-weighted", "#4477AA"),
    (EPO_CLASS_RATE, "map", "separate CpG rate", "#228833"),
    ("local_tree_reference_node_one_rate_out9", "map", "local-tree", "#CCBB44"),
    ("local_tree_sample_node_one_rate_out9", "map", "local-tree, 20 haplotypes", "#EE6677"),
    ("local_tree_sample_mrca_one_rate_out9", "map", "local-tree, 20 haplotypes, ingroup MRCA",
     "#AA3377"),
    ("local_tree_sample_mrca_class_rate_out9" if SIMULATION
     else "local_tree_sample_mrca_roulette_rate_out9", "weighted", "all changes", "#66CCEE"),
]
if SIMULATION:
    CONFIGURATIONS.insert(0, (*EPO, "EPO model, MAP", "0.1"))
CONTEXTS = ([("all", "all sites"), ("hypermutable", "hypermutable sites")] if SIMULATION
            else [("all", "all sites"), ("CpG", "CpG sites")])

report = json.loads(IN_JSON.read_text(encoding="utf-8"))
methods = report["methods"]
n = report["n_haplotypes"]
x = np.arange(n + 1)


def baseline(context):
    """Spectrum the deviations are taken from."""
    if not SIMULATION:
        return np.asarray(methods["epo"][context]["spectrum_map"], dtype=float)
    truth = np.asarray(methods["truth_mrca"][context]["spectrum_weighted"], dtype=float)
    truth[[0, n]] = np.asarray(methods["truth_node"][context]["spectrum_weighted"], dtype=float)[[0, n]]
    return truth


def deviation(key, estimate, context):
    """Bins shown for a configuration and its percent difference from the baseline."""
    bins = x[1:n] if "mrca" in key else x
    spectrum = np.asarray(methods[key][context][f"spectrum_{estimate}"], dtype=float)
    return bins, 100 * (spectrum[bins] / baseline(context)[bins] - 1)


fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (context, title) in zip(axes, CONTEXTS):
    ax.axhline(0.0, color="0.1", lw=0.8, label="truth" if SIMULATION else "EPO")
    for key, estimate, label, colour in CONFIGURATIONS:
        ax.plot(*deviation(key, estimate, context), color=colour, marker="o", ms=2.5, lw=1.3,
                alpha=0.9 if colour == "0.1" else 0.55, label=label,
                zorder=3 if colour == "0.1" else 2)
    ax.set_title(title, fontsize=13.5)
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.set_xticks(x[::2])
    ax.set_xlabel("derived-allele count", fontsize=10)
axes[0].set_ylabel(f"difference from {'truth' if SIMULATION else 'EPO'} (%)", fontsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.36)
# Centred under the axes, not under the figure, whose left margin holds the y label.
x_centre = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
fig.legend(handles, labels, fontsize=7.5, ncol=2, framealpha=0.9, loc="lower center",
           bbox_to_anchor=(x_centre, 0.02), borderpad=0.8, columnspacing=2.5, labelspacing=0.5)
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG)
print(f"wrote {FIG}", flush=True)
