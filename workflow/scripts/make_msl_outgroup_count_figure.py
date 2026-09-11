"""MSL chr1: 1-outgroup vs 2-outgroup comparison plots.

Runs Ancestree's FixedTreeInference on the MSL chr1 polymorphic-site
panel under two configurations:

- ``n_out = 1`` --- ponAbe2 (orangutan) only.
- ``n_out = 2`` --- ponAbe2 + macFas5 (rhesus).

Then emits a 2x2 panel:

- Top-left:  soft-uSFS shapes (1-og vs 2-og) across the folded
  ingroup minor-allele bin (no truth on real data, so the two estimates are
  compared against each other).
- Top-right: ratio (1-og soft uSFS) / (2-og soft uSFS) per bin ---
  values > 1 mean the 1-og run inflates that bin, < 1 deflates.
- Bottom-left:  per-bin MAP flip rate (fraction of sites where 1-og's
  MAP estimate differs from 2-og's).
- Bottom-right: per-bin total variation distance (mean |p_1 - p_2| / 2)
  between the 1-og and 2-og posteriors.

The headline question: at which folded-frequency bins does adding the
second outgroup actually change Ancestree's call?
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    NPZ = Path(snakemake.input.npz)  # type: ignore[name-defined]
    META = Path(snakemake.input.meta)  # type: ignore[name-defined]
    OUT_PDF = Path(snakemake.output[0])  # type: ignore[name-defined]
except NameError:
    DATA = Path("results/data")
    NPZ = DATA / "msl_polymorphic_chr1_143750000_150M.npz"
    META = DATA / "msl_polymorphic_chr1_143750000_150M_meta.json"
    OUT_PDF = Path("reports/manuscripts/latex/figures/"
                   "bench_msl_outgroup_count.pdf")

STATES = ("A", "C", "G", "T")


def _run(keep_idx, sites_cached, region_meta, ingroup_names):
    from ancestree import (
        BaseComposition, FixedTreeInference, HKY, KingmanIngroupWeight,
    )
    outgroup_names_all = sites_cached["outgroup_names_all"]
    outgroup_names = [outgroup_names_all[i] for i in keep_idx]
    sites = sites_cached["site_factory"](keep_idx, outgroup_names)
    bc = BaseComposition.from_polymorphic_sites(sites)
    model = HKY(kappa=bc.kappa_estimate, fit_kappa=True)
    ingroup_weight = KingmanIngroupWeight(ingroup_samples=ingroup_names)
    inf = FixedTreeInference(
        sites, ingroup_samples=ingroup_names, outgroup_samples=outgroup_names,
        model=model, n_target_sites=region_meta["end"] - region_meta["start"],
        ingroup_weight=ingroup_weight, base_composition=bc,
        recurrence="full", parallelize=False, progress=False,
    )
    inf.fit()
    posteriors_by_pos = {int(s.pos): p for s, p in inf.infer()}
    return posteriors_by_pos


def main() -> None:
    from ancestree.sites import Site

    npz = np.load(NPZ, allow_pickle=True)
    pos = npz["pos"]
    ref_idx = npz["ref_idx"]
    alt_idx = npz["alt_idx"]
    ingroup = npz["ingroup"]
    outgroup = npz["outgroup"]
    ingroup_names = list(npz["ingroup_names"].astype(str))
    outgroup_names_all = list(npz["outgroup_names"].astype(str))
    n_ingroup = len(ingroup_names)
    with open(META) as f:
        region_meta = json.load(f)

    print(
        f"MSL chr1 panel: {len(pos)} polymorphic sites, "
        f"{n_ingroup} ingroup haps, outgroups={outgroup_names_all}",
        flush=True,
    )

    # Folded ingroup minor-allele count per site (orientation-invariant).
    def folded_minor(k: int) -> int | None:
        counts = np.zeros(4, dtype=int)
        for i in range(n_ingroup):
            a = int(ingroup[k, i])
            if a >= 0:
                counts[a] += 1
        n_obs = int(counts.sum())
        if n_obs == 0:
            return None
        c_major = int(counts.max())
        return min(c_major, n_obs - c_major)

    i_minor_by_pos = {int(pos[k]): folded_minor(k) for k in range(len(pos))}

    def site_factory(keep_idx: list[int], outgroup_names: list[str]):
        out = []
        for k in range(len(pos)):
            ref = STATES[int(ref_idx[k])]
            alt_row = [STATES[int(a)] for a in alt_idx[k] if a >= 0]
            alleles = (ref, *alt_row)
            tip_alleles = {}
            for i, name in enumerate(ingroup_names):
                a = int(ingroup[k, i])
                tip_alleles[name] = STATES[a] if a >= 0 else None
            for j_out, j_orig in enumerate(keep_idx):
                a = int(outgroup[k, j_orig])
                tip_alleles[outgroup_names[j_out]] = STATES[a] if a >= 0 else None
            out.append(Site(chrom="1", pos=int(pos[k]),
                            alleles=alleles, tip_alleles=tip_alleles))
        return out

    cached = {"site_factory": site_factory, "outgroup_names_all": outgroup_names_all}

    print("running 1-outgroup (ponAbe2 only) ...", flush=True)
    p1 = _run([0], cached, region_meta, ingroup_names)
    print("running 2-outgroup (ponAbe2 + macFas5) ...", flush=True)
    p2 = _run([0, 1], cached, region_meta, ingroup_names)

    # Per-bin accumulators.
    shared = sorted(set(p1) & set(p2))
    by_bin = defaultdict(lambda: {
        "n": 0,
        "flip": 0,
        "tvd_sum": 0.0,
        "soft_sfs_1": np.zeros(n_ingroup + 1),
        "soft_sfs_2": np.zeros(n_ingroup + 1),
    })

    pos_idx = {int(pos[k]): k for k in range(len(pos))}

    for site_pos in shared:
        bin_i = i_minor_by_pos.get(site_pos)
        if bin_i is None:
            continue
        post1 = p1[site_pos]
        post2 = p2[site_pos]
        tvd = float(np.abs(post1.values - post2.values).sum() / 2.0)
        flip = int(post1.map_allele != post2.map_allele)
        by_bin[bin_i]["n"] += 1
        by_bin[bin_i]["flip"] += flip
        by_bin[bin_i]["tvd_sum"] += tvd

        # Soft uSFS: for each candidate ancestor allele, posterior * (i_derived under that allele).
        k = pos_idx[site_pos]
        counts = np.zeros(4, dtype=int)
        for i in range(n_ingroup):
            a = int(ingroup[k, i])
            if a >= 0:
                counts[a] += 1
        n_obs = int(counts.sum())
        # Per-site soft uSFS contribution: for each posterior allele a,
        # add posterior[a] to bin (n_obs - count(a)).
        for pst, soft_key in ((post1, "soft_sfs_1"), (post2, "soft_sfs_2")):
            for a_idx, a in enumerate(STATES):
                if pst.alleles and a not in pst.alleles:
                    continue
                p_ix = list(pst.alleles).index(a) if pst.alleles else None
                if p_ix is None:
                    continue
                p = float(pst.values[p_ix])
                if p <= 0:
                    continue
                derived_under = n_obs - int(counts[a_idx])
                if 0 <= derived_under <= n_ingroup:
                    by_bin[bin_i][soft_key][derived_under] += p

    # Aggregate full-spectrum soft uSFS (sum across bins).
    total_sfs_1 = np.zeros(n_ingroup + 1)
    total_sfs_2 = np.zeros(n_ingroup + 1)
    for v in by_bin.values():
        total_sfs_1 += v["soft_sfs_1"]
        total_sfs_2 += v["soft_sfs_2"]

    bins = sorted(by_bin)
    flip_rate = np.array([by_bin[b]["flip"] / by_bin[b]["n"] for b in bins])
    tvd_mean = np.array([by_bin[b]["tvd_sum"] / by_bin[b]["n"] for b in bins])

    # Plot.
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 5.8))

    # Top-left: full unfolded soft uSFS shapes (truth-agnostic).
    sfs_bins = np.arange(n_ingroup + 1)
    axes[0, 0].plot(sfs_bins, total_sfs_1, "o-", color="#3b528b", lw=1.3, ms=4,
                    alpha=0.8, label="1-outgroup (ponAbe2)")
    axes[0, 0].plot(sfs_bins, total_sfs_2, "s-", color="#21918c", lw=1.3, ms=4,
                    alpha=0.8, label="2-outgroup (+macFas5)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel(f"unfolded ingroup derived-allele count (0..{n_ingroup})")
    axes[0, 0].set_ylabel("estimated count of sites")
    axes[0, 0].set_title("MSL chr1: soft uSFS (1-og vs 2-og)")
    axes[0, 0].legend(fontsize=8, frameon=True)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: ratio of estimated counts per unfolded bin (1-og / 2-og).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(total_sfs_2 > 0, total_sfs_1 / total_sfs_2, np.nan)
    axes[0, 1].plot(sfs_bins, ratio, "o-", color="#5ec962", lw=1.3, ms=4, alpha=0.85)
    axes[0, 1].axhline(1.0, color="0.4", lw=0.6, linestyle="--")
    axes[0, 1].set_xlabel(f"unfolded ingroup derived-allele count (0..{n_ingroup})")
    axes[0, 1].set_ylabel("(1-og) / (2-og) soft uSFS count")
    axes[0, 1].set_title(r"MSL chr1: per-bin ratio of estimated uSFS")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: per-folded-bin MAP-flip rate (with sample-size shading).
    flip_overall = sum(by_bin[b]["flip"] for b in bins) / sum(by_bin[b]["n"] for b in bins)
    axes[1, 0].plot(bins, flip_rate, "o-", color="darkred", lw=1.1, ms=3, alpha=0.8,
                    label=f"flip rate (overall = {flip_overall:.3%})")
    axes[1, 0].set_xlabel(f"folded ingroup minor-allele count (1..{n_ingroup // 2})")
    axes[1, 0].set_ylabel("MAP-flip rate (1-og -> 2-og)")
    axes[1, 0].set_title(r"MSL chr1: per-bin MAP-flip rate")
    axes[1, 0].set_ylim(0, max(0.25, flip_rate.max() * 1.1))
    axes[1, 0].legend(fontsize=8, frameon=True, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: per-folded-bin posterior TVD.
    tvd_overall = sum(by_bin[b]["tvd_sum"] for b in bins) / sum(by_bin[b]["n"] for b in bins)
    axes[1, 1].plot(bins, tvd_mean, "o-", color="#7b3294", lw=1.1, ms=3, alpha=0.8,
                    label=f"mean TVD (overall = {tvd_overall:.3f})")
    axes[1, 1].set_xlabel(f"folded ingroup minor-allele count (1..{n_ingroup // 2})")
    axes[1, 1].set_ylabel(r"posterior TVD ($\frac{1}{2}\sum |p_1 - p_2|$)")
    axes[1, 1].set_title(r"MSL chr1: per-bin posterior disagreement")
    axes[1, 1].set_ylim(0, max(0.20, tvd_mean.max() * 1.1))
    axes[1, 1].legend(fontsize=8, frameon=True, loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")
    # Screen copy beside the vector figure, for review outside a PDF viewer.
    png_target = OUT_PDF.with_suffix(".png")
    fig.savefig(png_target, bbox_inches="tight", dpi=150)
    print(f"Wrote {png_target}")

    # Stdout summary: top-10 bins by flip rate, and top-10 by uSFS ratio departure.
    print("\n=== Top 10 folded bins by MAP-flip rate ===")
    order = np.argsort(-flip_rate)
    for j in order[:10]:
        b = bins[j]
        print(f"  bin={b:3d}  n={by_bin[b]['n']:4d}  "
              f"flip={flip_rate[j]:.3%}  TVD={tvd_mean[j]:.4f}")

    print("\n=== Top 10 unfolded bins by |log10(1-og / 2-og uSFS)| ===")
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.where(total_sfs_2 > 0, np.log10(ratio), 0.0)
    order = np.argsort(-np.abs(log_ratio))
    for j in order[:10]:
        if not np.isfinite(log_ratio[j]):
            continue
        print(f"  i={j:3d}  1-og={total_sfs_1[j]:8.1f}  2-og={total_sfs_2[j]:8.1f}  "
              f"ratio={ratio[j]:7.3f}  log10ratio={log_ratio[j]:+.3f}")


if __name__ == "__main__":
    main()
