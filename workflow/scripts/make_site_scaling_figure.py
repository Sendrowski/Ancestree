"""Site-count requirements for parameter recovery, in one two-panel figure.

- Left panel (fixed-tree): parameter-recovery error against ``n_sites``, one
  line per (model, n_out). The y axis is the mean absolute relative deviation
  |K_fit - K_ref| / K_ref averaged across the ladder branches (and kappa for
  HKY), where K_i is the divergence of ladder branch i in substitutions per
  site and the reference cell is the full-data fit. Log-log scaling so the
  diminishing-returns slope is visible.

- Right panel (adaptive prior): mean |pi_fit - pi_kingman| averaged across
  fittable bins, where pi_i is the prior probability that the ancestral allele
  is the minor one in SFS bin i, against ``n_sites``, one line per
  ``subsample_size``. Same log-log style.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    FIXED_JSON = Path(snakemake.input.fixed)  # type: ignore[name-defined]
    ADAPTIVE_JSON = Path(snakemake.input.adaptive)  # type: ignore[name-defined]
    OUT = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    REPORTS = Path("results/reports")
    FIXED_JSON = REPORTS / "site_scaling_fixed.json"
    ADAPTIVE_JSON = REPORTS / "site_scaling_adaptive.json"
    OUT = Path("reports/manuscripts/latex/figures/bench_site_scaling.pdf")

fixed = json.loads(FIXED_JSON.read_text())
adaptive = json.loads(ADAPTIVE_JSON.read_text())


fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9))


# ----------------------------------------------- left: fixed-tree panel

ax = axes[0]
fixed_summary = fixed["summary"]
groups: dict[tuple[str, int], dict[int, dict]] = {}
for key, s in fixed_summary.items():
    g = (s["model"], int(s["n_out"]))
    groups.setdefault(g, {})[int(s["n_sites"])] = s

models = ("JC", "HKY")
n_outs = (1, 2, 3)
model_color = {"JC": "tab:blue", "HKY": "tab:orange"}
n_out_marker = {1: "o", 2: "s", 3: "^"}
n_out_alpha = {1: 0.45, 2: 0.7, 3: 1.0}

for model in models:
    for n_out in n_outs:
        cells = groups.get((model, n_out))
        if not cells:
            continue
        ns = sorted(cells.keys())
        means = [cells[n]["abs_rel_err"]["mean"] for n in ns]
        stds = [cells[n]["abs_rel_err"]["std"] for n in ns]
        ax.errorbar(
            ns, means, yerr=stds,
            color=model_color[model],
            marker=n_out_marker[n_out],
            alpha=n_out_alpha[n_out],
            lw=1.2, ms=5, capsize=2,
            label=rf"{model}, $n_\mathrm{{out}}={n_out}$",
        )

# 1/sqrt(L) reference decay: the diminishing-returns slope quoted in the text.
# Amplitude fitted to the data (median of mean * sqrt(L)) so it overlays the
# curves rather than floating arbitrarily.
all_L: list[float] = []
all_m: list[float] = []
for model in models:
    for n_out in n_outs:
        cells = groups.get((model, n_out))
        if not cells:
            continue
        for n, s in cells.items():
            all_L.append(float(n))
            all_m.append(float(s["abs_rel_err"]["mean"]))
if all_L:
    arr_L = np.array(all_L)
    arr_m = np.array(all_m)
    amp = float(np.median(arr_m * np.sqrt(arr_L)))
    Lref = np.array(sorted(set(all_L)))
    ax.plot(Lref, amp / np.sqrt(Lref), "k--", lw=1.3, alpha=0.7,
            label=r"$\propto 1/\sqrt{L_\mathrm{sites}}$")

ax.set_xscale("log")
ax.set_xlabel(r"polymorphic sites used in fit, $L_\mathrm{sites}$")
ax.set_ylabel(r"mean $|\hat{\theta} - \theta_\mathrm{ref}| / \theta_\mathrm{ref}$")
ax.set_title(r"Fixed-tree fit (branch rates $K_i$" + " + " + r"$\kappa$)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=7, ncol=2, frameon=True)


# ----------------------------------------------- right: adaptive panel

ax = axes[1]
adapt_summary = adaptive["summary"]
groups_a: dict[int, dict[int, dict]] = {}
for key, s in adapt_summary.items():
    groups_a.setdefault(int(s["n_sub"]), {})[int(s["n_sites"])] = s

n_subs = sorted(groups_a.keys())
sub_color = plt.cm.viridis(np.linspace(0.2, 0.85, len(n_subs)))

for color, n_sub in zip(sub_color, n_subs):
    cells = groups_a[n_sub]
    ns = sorted(cells.keys())
    means = [cells[n]["mean_abs_err"]["mean"] for n in ns]
    stds = [cells[n]["mean_abs_err"]["std"] for n in ns]
    ax.errorbar(
        ns, means, yerr=stds,
        color=color, marker="o", lw=1.4, ms=5, capsize=2,
        label=rf"$n_\mathrm{{sub}}={n_sub}$",
    )

ax.set_xscale("log")
ax.set_xlabel(r"polymorphic sites used in fit, $L_\mathrm{sites}$")
ax.set_ylabel(r"mean $|\hat{\pi}_i - \pi_i^\mathrm{Kingman}|$")
ax.set_title(r"Adaptive prior fit ($\pi_i$ per SFS bin)")
ax.set_ylim(bottom=0)
ax.legend(loc="upper right", fontsize=8, frameon=True)


fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"Wrote {OUT}")
