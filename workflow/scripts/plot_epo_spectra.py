"""Unfolded spectrum of the MSL sample under Ancestree relative to the one
under the Ensembl EPO ancestral base, over all sites and over the CpG context.

Count 0 holds the sites the sample has fixed for the ancestral base and count
``n`` those it has fixed for a derived one. EPO contributes one base per site,
Ancestree the posterior-weighted spectrum. A series read at the ingroup MRCA
has its polymorphic counts alone, since the sample's fixed base is the state
there. Colour separates the ingroup data and the node the state is read at,
and the line style the two modes.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    IN_JSON = Path(snakemake.input.json)  # type: ignore[name-defined]
    FIG = Path(snakemake.output.pdf)  # type: ignore[name-defined]
except NameError:
    IN_JSON = Path("results/reports/epo_spectra.json")
    FIG = Path("results/reports/epo_spectra.pdf")

#: Ingroup data and node: key suffix per mode, legend label, colour, polymorphic counts only.
CONDITIONS = [
    (("fixed_tree_reference_estimated_model_class_rate",
      "local_tree_reference_node_class_rate_out9"),
     "reference genome, at human\u2013chimpanzee ancestor", "#e69f00", False),
    (("fixed_tree_sample_node", "local_tree_sample_node_class_rate_out9"),
     "20 haplotypes, at human\u2013chimpanzee ancestor", "#009e73", False),
    (("fixed_tree_sample_mrca", "local_tree_sample_mrca_class_rate_out9"),
     "20 haplotypes, at their MRCA", "#cc3311", True),
]
#: Mode: legend label, line style, marker.
MODES = [("fixed-tree mode", "-", "s"), ("local-tree mode", "--", "o")]
COLUMNS = [("all", "all sites"), ("CpG", "CpG sites")]

report = json.loads(IN_JSON.read_text())
n = report["n_haplotypes"]
x = np.arange(n + 1)

fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.7), gridspec_kw={"wspace": 0.2})
for ax, (context, title) in zip(axes, COLUMNS):
    epo = np.asarray(report["methods"]["epo"][context]["spectrum_map"])
    ax.axhline(1.0, color="0.1", lw=0.8)
    for keys, _, colour, polymorphic_only in CONDITIONS:
        bins = x[1:n] if polymorphic_only else x
        for key, (_, style, marker) in zip(keys, MODES):
            spectrum = np.asarray(report["methods"][key][context]["spectrum_weighted"])
            ax.plot(bins, spectrum[bins] / epo[bins], color=colour, ls=style, marker=marker,
                    ms=2.5, lw=1.0, alpha=0.9)
    ax.set_title(title, fontsize=11.5)
    ax.grid(True, alpha=0.2, lw=0.4)
    ax.set_xticks(x[::2])
    ax.set_xlabel("derived-allele count", fontsize=10)
axes[0].set_ylabel("sites relative to EPO", fontsize=10)
handles = [Line2D([], [], color=colour, lw=1.5, label=label)
           for _, label, colour, _ in CONDITIONS]
handles += [Line2D([], [], color="0.3", ls=style, marker=marker, ms=3, lw=1.0, label=label)
            for label, style, marker in MODES]
fig.legend(handles=handles[:3], fontsize=7.5, ncol=3, frameon=True, framealpha=1.0,
           edgecolor="0.6", loc="lower center", bbox_to_anchor=(0.5, 0.09))
fig.legend(handles=handles[3:], fontsize=7.5, ncol=2, frameon=True, framealpha=1.0,
           edgecolor="0.6", loc="lower center", bbox_to_anchor=(0.5, 0.005))
fig.subplots_adjust(left=0.09, right=0.99, top=0.92, bottom=0.31)
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG)
print(f"wrote {FIG}", flush=True)
