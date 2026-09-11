"""Runtime of the three inference modes against ingroup size.

Reads the timings measured by ``bench_sample_scaling.py``. Kept apart from the
measurement so the figure can be redrawn without re-timing, which would move
the very numbers it reports.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

try:
    in_json = str(snakemake.input.json)  # type: ignore[name-defined]
    out_pdf = str(snakemake.output.pdf)  # type: ignore[name-defined]
    out_copy = str(snakemake.output.fig_copy)  # type: ignore[name-defined]
except NameError:
    in_json = "results/reports/sample_scaling.json"
    out_pdf = "results/reports/sample_scaling.pdf"
    out_copy = "reports/manuscripts/latex/figures/sample_scaling.pdf"

rows = json.loads(Path(in_json).read_text())

fig, ax = plt.subplots(figsize=(5.6, 2.6))
xs = [r["n_ingroup"] for r in rows]
for key, label, colour in (("fixed_tree_s", "fixed tree", "#fb8500"),
                           ("arg_s", "ARG", "#023047"),
                           ("local_tree_s", "local tree", "#219ebc")):
    ax.plot(xs, [r[key] for r in rows], "o-", color=colour, lw=1.6, label=label)
ax.set_xscale("log")
# Log minor ticks would otherwise print their own labels over the majors.
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks(xs, [str(x) for x in xs])
ax.set_xlabel("ingroup haplotypes")
ax.set_ylabel("runtime (s)")
ax.grid(alpha=0.25, lw=0.6, which="both")
ax.legend(frameon=True, framealpha=0.92, edgecolor="#bbbbbb",
          fontsize=9).get_frame().set_linewidth(0.6)
fig.tight_layout()
fig.savefig(out_pdf)
plt.close(fig)
Path(out_copy).parent.mkdir(parents=True, exist_ok=True)
import shutil  # noqa: E402
shutil.copyfile(out_pdf, out_copy)
print(f"wrote {out_pdf} and {out_copy}")
