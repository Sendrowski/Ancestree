"""B3-pi report: AdaptiveIngroupWeight π_i side-by-side comparison.

Aggregates one Ancestree cell + one fastDFE cell (both run with the
adaptive prior on the larger pi-comparison sim, n=10 haps, ~2k
polymorphic sites) and shows per-bin polarization probabilities:

    Kingman closed form (n-i)/n   vs   Ancestree Adaptive π_i   vs   fastDFE Adaptive π_i

Useful as a sanity check that Adaptive recovers Kingman under
neutrality when per-bin sample sizes are large enough that the MLE is
stable (~hundreds of sites per bin).

Run directly::

    python workflow/scripts/report_pi_comparison.py

Or via snakemake::

    snakemake -j 1 results/reports/estsfs_pi_comparison.md
"""
import json
from pathlib import Path


try:
    in_ancestree = snakemake.input.ancestree  # type: ignore[name-defined]
    in_fastdfe = snakemake.input.fastdfe  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_md = snakemake.output.md  # type: ignore[name-defined]
    config = dict(snakemake.params.sim_config)  # type: ignore[name-defined]
except NameError:
    in_ancestree = "results/data/estsfs_pi_ancestree_jc_JC_adaptive_n3.json"
    in_fastdfe = "results/data/estsfs_pi_fastdfe_jc_JC_adaptive_n3.json"
    in_meta = "results/data/estsfs_pi_sim_meta.json"
    out_json = "results/reports/estsfs_pi_comparison.json"
    out_md = "results/reports/estsfs_pi_comparison.md"
    config = {
        "n_ingroup": 10, "length": 2_000_000, "mu": 1e-8,
        "rec_rate": 1e-8, "pop_size": 1e4, "outgroup_pop_size": 1e5,
        "outgroup_divergences": [5e4, 1e5, 1.5e5],
        "seed": 42, "model": "JC",
    }


with open(in_ancestree) as f:
    anc_meta = json.load(f).get("inference_meta", {})
with open(in_fastdfe) as f:
    fd_meta = json.load(f).get("inference_meta", {})
with open(in_meta) as f:
    sim_meta = json.load(f)


n = int(sim_meta["n_ingroup"])
anc_pi = {int(k): float(v) for k, v in anc_meta.get("adaptive_pi", {}).items()}
fd_pi = {int(k): float(v) for k, v in fd_meta.get("adaptive_pi", {}).items()}
anc_n_per_bin = {
    int(k): int(v) for k, v in anc_meta.get("adaptive_n_sites_per_bin", {}).items()
}


def kingman(i: int, n: int) -> float:
    """(n - i) / n — the closed-form Kingman polarization probability."""
    return (n - i) / n


# Compare per bin
rows = []
abs_diffs_anc_king = []
abs_diffs_fd_king = []
abs_diffs_anc_fd = []
for i in range(n + 1):
    k = kingman(i, n)
    a = anc_pi.get(i, float("nan"))
    f = fd_pi.get(i, float("nan"))
    n_sites = anc_n_per_bin.get(i, 0)
    rows.append({
        "i": i, "n_sites": n_sites,
        "kingman": k, "ancestree": a, "fastdfe": f,
        "anc_vs_king": a - k,
        "fd_vs_king":  f - k,
        "anc_vs_fd":   a - f,
    })
    # Track |diff| for non-boundary bins (boundaries are fixed)
    if 0 < i < n:
        abs_diffs_anc_king.append(abs(a - k))
        abs_diffs_fd_king.append(abs(f - k))
        abs_diffs_anc_fd.append(abs(a - f))


def _summarise(label: str, diffs: list[float]) -> dict:
    if not diffs:
        return {"label": label, "n": 0}
    arr = sorted(diffs)
    return {
        "label": label, "n": len(arr),
        "mean_abs_diff": sum(arr) / len(arr),
        "max_abs_diff": max(arr),
        "median_abs_diff": arr[len(arr) // 2],
    }


summary = [
    _summarise("Ancestree vs Kingman", abs_diffs_anc_king),
    _summarise("fastDFE   vs Kingman", abs_diffs_fd_king),
    _summarise("Ancestree vs fastDFE", abs_diffs_anc_fd),
]


report = {
    "config": config,
    "ancestree_inference_meta": anc_meta,
    "fastdfe_inference_meta": fd_meta,
    "per_bin": rows,
    "summary": summary,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(report, f, indent=2)
print(f"Wrote {out_json}", flush=True)


def _fmt(x):
    return "—" if x is None or (isinstance(x, float) and (x != x)) else f"{x:.4f}"


per_bin_lines = []
for r in rows:
    per_bin_lines.append(
        f"| {r['i']:>2} | {r['n_sites']:>5} | {_fmt(r['kingman'])} | "
        f"{_fmt(r['ancestree'])} | {_fmt(r['fastdfe'])} | "
        f"{_fmt(r['anc_vs_king']):>8} | {_fmt(r['fd_vs_king']):>8} | "
        f"{_fmt(r['anc_vs_fd']):>8} |"
    )

summary_lines = []
for s in summary:
    if s["n"] == 0:
        continue
    summary_lines.append(
        f"| {s['label']} | {s['n']} | {_fmt(s['mean_abs_diff'])} | "
        f"{_fmt(s['median_abs_diff'])} | {_fmt(s['max_abs_diff'])} |"
    )


cfg = config
md = f"""# B3-π — AdaptiveIngroupWeight π_i sanity comparison

Larger dataset designed so the per-bin Adaptive MLE has enough data
to actually converge (≥ 100 sites per fittable bin under Kingman SFS).
Should recover Kingman within sampling noise.

## Simulation
- ingroup: `{cfg['n_ingroup']}` haps (Ne={cfg['pop_size']:g})
- outgroups: 3 nested splits at `{cfg['outgroup_divergences']}` gens
- sequence length: `{cfg['length']:g}`
- mutation rate: `{cfg['mu']:g}` per-site per-gen
- sim model: `{cfg['model']}` (κ=1.0)
- seed: `{cfg['seed']}`

## Per-bin polarization probabilities π_i = P(major ancestral | minor count = i)

n_sites = sites that contributed to the Ancestree per-bin MLE (only
biallelic + fully-observed ingroup sites count). Bins ``i=0`` and
``i=n_ingroup`` are fixed by convention (no fit). Folded SFS:
``π_{{n-i}} = 1 - π_i``.

| i | n_sites | Kingman | Ancestree | fastDFE | Δ(anc-king) | Δ(fd-king) | Δ(anc-fd) |
|---|---|---|---|---|---|---|---|
{chr(10).join(per_bin_lines)}

## Summary (excluding boundary bins i=0 and i={n})

| comparison | n_bins | mean \\|Δ\\| | median \\|Δ\\| | max \\|Δ\\| |
|---|---|---|---|---|
{chr(10).join(summary_lines)}
"""

with open(out_md, "w") as f:
    f.write(md)
print(f"Wrote {out_md}", flush=True)
