"""Main-text figure: MAP ancestral calls distort the unfolded SFS, and both
fixes leave a trace.

Left --- filtering low-confidence calls before building the SFS is not
frequency-neutral: confidence falls with derived-allele frequency, so retaining
the most-confident fraction preferentially drops high-frequency-derived sites
and erodes the high-frequency tail.

Right --- committing each site to its MAP ancestral allele depletes the same
tail directly (ambiguous high-frequency sites are flipped to low frequency),
whereas attributing each site fractionally across ancestral states, an expected
unfolded SFS built from the posteriors, recovers the true shape.

Reads the pooled spectra written by :mod:`compute_expected_sfs`, which carry
both spectra and the per-bin confidence histogram from the same simulation.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _robustness_common as rc  # noqa: E402

try:
    SPECTRA = Path(snakemake.input[0])  # type: ignore[name-defined]
    OUT = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    SPECTRA = Path("results/data/expected_sfs_baseline_n0.json")
    OUT = Path("reports/manuscripts/latex/figures/sfs_fractional_calls.pdf")

#: Confidence quantiles retained in panel (a), most-confident sites first.
KEEP = [1.0, 0.9, 0.8, 0.7, 0.6]
COLORS = ["0.15", "#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]

d = json.loads(SPECTRA.read_text())
NING = int(d["n"])
x = np.arange(1, NING)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.6, 2.9), sharey=True)

# ---- Panel A: confidence-quantile filtering biases the SFS -----------------
res = rc.summary_conf_filter({"conf_hist": d["conf_hist"]}, KEEP, NING)
for frac, col in zip(KEEP, COLORS):
    thr, sfs = res[frac]
    lbl = "all sites (true SFS)" if frac == 1.0 else f"top {frac*100:.0f}%  ($p_{{\\max}} \\geq$ {thr:.3f})"
    axA.plot(x, sfs, marker="o", ms=3, lw=1.6, alpha=0.75, color=col, label=lbl)
# symlog, not log: the MAP spectrum reaches zero in the top bins, which a log
# axis cannot draw. The linear region below LINTHRESH keeps those points on
# the panel while the decades above it stay logarithmic.
LINTHRESH = 1e-3
axA.set_yscale("symlog", linthresh=LINTHRESH)
axA.set_xlabel(r"derived-allele count $i$")
axA.set_ylabel("proportion of segregating sites")
axA.set_xticks(range(0, NING + 1, 4))
axA.grid(True, which="both", alpha=0.25, lw=0.5)
axA.legend(title="SNPs kept (by confidence)", fontsize=7, title_fontsize=7,
           framealpha=0.9, loc="lower left")
axA.set_title("(a) Filtering by posterior confidence", fontsize=9.5)

# ---- Panel B: MAP vs fractional (expected) SFS vs truth ---------------------
T = np.array(d["true"])[1:NING]
M = np.array(d["map"])[1:NING]
E = np.array(d["expected"])[1:NING]
T, M, E = (a / a.sum() for a in (T, M, E))
axB.plot(x, T, marker="o", ms=4, lw=1.8, color="0.15", alpha=0.85, label="true SFS")
axB.plot(x, M, marker="s", ms=4, lw=1.6, color="#d62728", alpha=0.85,
         label="MAP estimates")
axB.plot(x, E, marker="^", ms=4, lw=1.6, color="#1f77b4", alpha=0.85,
         label="posterior-weighted (expected) SFS")
axB.set_yscale("symlog", linthresh=LINTHRESH)
axB.set_xlabel(r"derived-allele count $i$")
axB.set_xticks(range(0, NING + 1, 4))
axB.grid(True, which="both", alpha=0.25, lw=0.5)
axB.legend(fontsize=7.5, framealpha=0.9, loc="lower left")
axB.set_title("(b) MAP versus posterior-weighted calls", fontsize=9.5)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
