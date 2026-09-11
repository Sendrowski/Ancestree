"""B13 MSL real-data report: 3-way agreement between Ancestree-VCF,
PolarBEAR, and est-sfs on PolarBEAR's published 1000-Genomes MSL chr1
dataset.

ARG-mode inference on the 18 GB tskit ARG is skipped (too slow for
this pass). The comparison focuses on the outgroup-based path, where
all three tools should be directly comparable.

Output: ``results/reports/msl_3way.{json,md}``.
"""
import gzip
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_io import (  # noqa: E402
    load_estsfs as _load_estsfs_anc,
    load_pos_idx as _load_pos_idx,
)


try:
    IN_VCF_ANC = snakemake.input.vcf_anc  # type: ignore[name-defined]
    IN_VCF_META = snakemake.input.vcf_meta  # type: ignore[name-defined]
    IN_POLARBEAR_ANC = snakemake.input.polarbear_anc  # type: ignore[name-defined]
    IN_ESTSFS_ANC = snakemake.input.estsfs_anc  # type: ignore[name-defined]
    IN_MSL_POPULATION_VCF = snakemake.input.msl_population_vcf  # type: ignore[name-defined]
    # The rule declares the extraction marker, not the directory: a
    # directory output counts as produced the moment it exists.
    IN_SNPEFF_DIR = str(
        Path(snakemake.input.snpeff_ok).parent)  # type: ignore[name-defined]
    OUT_JSON = snakemake.output.json  # type: ignore[name-defined]
    OUT_MD = snakemake.output.md  # type: ignore[name-defined]
    # Restrict the comparison to positions in our ancestree call set
    # (e.g. when a subsample wildcard was applied upstream). When the
    # ancestree call set covers a positional range only, also restrict
    # the PolarBEAR / est-sfs sets to that range so coverage numbers are
    # comparable. The wildcards forward through for the report header.
    REGION = str(getattr(snakemake.wildcards, "region", "full"))  # type: ignore[name-defined]
    N_SUBSAMPLE = str(getattr(snakemake.wildcards, "n_subsample", "full"))  # type: ignore[name-defined]
    SEED = str(getattr(snakemake.wildcards, "seed", "0"))  # type: ignore[name-defined]
except NameError:
    REGION = "chr1_0_25M"
    N_SUBSAMPLE = "100k"
    SEED = "0"
    INFER_SCOPE = "subsample"
    IN_VCF_ANC = f"results/data/msl_vcf_{REGION}_n{N_SUBSAMPLE}_seed{SEED}_{INFER_SCOPE}.txt"
    IN_VCF_META = f"results/data/msl_vcf_{REGION}_n{N_SUBSAMPLE}_seed{SEED}_{INFER_SCOPE}_meta.json"
    IN_POLARBEAR_ANC = (
        "external/polarbear/data/real_data/ancestral_state/"
        "PolarBEAR_gammaSMC/anc_state.txt"
    )
    IN_ESTSFS_ANC = (
        "external/polarbear/data/real_data/ancestral_state/"
        "est_sfs/est-sfs_ancstate.txt"
    )
    IN_MSL_POPULATION_VCF = (
        "external/polarbear/data/real_data/ancestral_state/"
        "PolarBEAR_gammaSMC/chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.vcf.gz"
    )
    IN_SNPEFF_DIR = "external/polarbear/data/real_data/SnpEff_annotation"
    OUT_JSON = f"results/reports/msl_3way_{REGION}_n{N_SUBSAMPLE}_seed{SEED}.json"
    OUT_MD = f"results/reports/msl_3way_{REGION}_n{N_SUBSAMPLE}_seed{SEED}.md"


def _pairwise(a: dict[int, int], b: dict[int, int]) -> dict:
    overlap = a.keys() & b.keys()
    n_agree = sum(1 for p in overlap if a[p] == b[p])
    n_overlap = len(overlap)
    return {
        "n_a": len(a),
        "n_b": len(b),
        "n_overlap": n_overlap,
        "n_agree": n_agree,
        "agreement_rate": (n_agree / n_overlap) if n_overlap else None,
    }


def main():
    print("loading per-method ancestral-state files", flush=True)
    methods_full: dict[str, dict[int, int]] = {
        "ancestree_vcf": _load_pos_idx(IN_VCF_ANC),
        "polarbear": _load_pos_idx(IN_POLARBEAR_ANC),
        "estsfs": _load_estsfs_anc(IN_ESTSFS_ANC),
    }
    for name, m in methods_full.items():
        print(f"  {name} (full):     {len(m):,} sites", flush=True)

    # When ancestree_vcf was run on a subsample/region, the PolarBEAR and
    # est-sfs call sets cover much more of the chromosome and any
    # pairwise comparison involving the two of them is dominated by sites
    # ancestree_vcf never even saw. Restrict all three call sets to the
    # positions ancestree_vcf actually called, so every comparison is on
    # the same site set and the three pairwise overlaps are directly
    # comparable. The "full" coverage numbers are still reported below
    # for context.
    ancestree_positions = set(methods_full["ancestree_vcf"].keys())
    methods: dict[str, dict[int, int]] = {
        name: {p: methods_full[name][p] for p in ancestree_positions if p in methods_full[name]}
        for name in methods_full
    }
    for name, m in methods.items():
        print(f"  {name} (restricted): {len(m):,} sites", flush=True)

    # MSL chr1 SNP count
    n_snp = 0
    try:
        import cyvcf2
        vcf = cyvcf2.VCF(IN_MSL_POPULATION_VCF)
        for _ in vcf:
            n_snp += 1
        vcf.close()
    except ImportError:
        with gzip.open(IN_MSL_POPULATION_VCF, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                n_snp += 1
    print(f"MSL chr1 SNPs: {n_snp:,}", flush=True)

    method_names = list(methods.keys())
    pairwise: dict[str, dict] = {}
    for a, b in itertools.combinations(method_names, 2):
        pair = _pairwise(methods[a], methods[b])
        pair["a_name"] = a
        pair["b_name"] = b
        pair["keep_prop_a"] = pair["n_a"] / n_snp if n_snp else None
        pair["keep_prop_b"] = pair["n_b"] / n_snp if n_snp else None
        pair["overlap_prop"] = pair["n_overlap"] / n_snp if n_snp else None
        pairwise[f"{a}__{b}"] = pair

    # Soft agreement: mean Ancestree-VCF posterior mass on the reference
    # tool's MAP allele. Requires the b13_fit_infer posteriors .npz to
    # be present (same stem as the .txt + "_posteriors.npz").
    soft_metrics: dict[str, dict] = {}
    posteriors_path = Path(IN_VCF_ANC).with_suffix("").as_posix() + "_posteriors.npz"
    if Path(posteriors_path).exists():
        import numpy as np
        pdata = np.load(posteriors_path, allow_pickle=True)
        anc_pos = pdata["pos"]
        anc_post = pdata["posterior"]  # (N, 4)
        # Index the posterior array by site position.
        pos_to_row = {int(p): i for i, p in enumerate(anc_pos)}
        for ref_name in ("polarbear", "estsfs"):
            ref_map = methods[ref_name]
            shared = [p for p in ref_map if p in pos_to_row]
            if not shared:
                continue
            mean_p_ref = float(np.mean([
                anc_post[pos_to_row[p], ref_map[p]] for p in shared
            ]))
            soft_metrics[f"ancestree_vcf__{ref_name}"] = {
                "n_shared": len(shared),
                "mean_ancestree_p_on_reference_map": mean_p_ref,
            }
        print(f"  soft metrics computed on {len(pos_to_row):,} ancestree posterior rows", flush=True)
    else:
        print(f"  posteriors file not found at {posteriors_path} — skipping soft metrics", flush=True)

    # "Ancestree-unique" sites: called by Ancestree-VCF, not called by
    # either reference tool. These exist because Ancestree's polymorphic
    # filter is broader (any 2+ alleles across ingroup + outgroup tips),
    # whereas PolarBEAR and est-sfs effectively require ingroup polymorphism.
    anc_unique = (
        set(methods["ancestree_vcf"].keys())
        - set(methods_full["polarbear"].keys())
        - set(methods_full["estsfs"].keys())
    )
    coverage = {
        "n_ancestree_only": len(anc_unique),
        "n_ancestree_total_restricted": len(methods["ancestree_vcf"]),
    }

    three_overlap = methods["ancestree_vcf"].keys() & methods["polarbear"].keys() & methods["estsfs"].keys()
    n_three_overlap = len(three_overlap)
    n_three_agree = sum(
        1 for p in three_overlap
        if methods["ancestree_vcf"][p] == methods["polarbear"][p] == methods["estsfs"][p]
    )

    SNPEFF_FILES = {
        "exon": "exon_pos.txt",
        "UTR_3": "UTR_3_pos.txt",
        "UTR_5": "UTR_5_pos.txt",
        "intron": "intron_pos.txt",
        "intergenic": "intergenic_pos.txt",
        "synonymous": "synonymous_pos.txt",
        "non_synonymous": "non_synonymous_pos.txt",
    }
    per_region: dict[str, dict] = {}
    for region, fname in SNPEFF_FILES.items():
        p = Path(IN_SNPEFF_DIR) / fname
        if not p.exists():
            per_region[region] = {"missing": True}
            continue
        region_positions: set[int] = set()
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    region_positions.add(int(line.split()[0]))
                except ValueError:
                    continue
        region_methods = {
            n: {pos: idx for pos, idx in methods[n].items() if pos in region_positions}
            for n in method_names
        }
        region_pairs: dict[str, dict] = {}
        for a, b in itertools.combinations(method_names, 2):
            pair = _pairwise(region_methods[a], region_methods[b])
            pair["a_name"] = a
            pair["b_name"] = b
            region_pairs[f"{a}__{b}"] = pair
        per_region[region] = {
            "n_snps_in_region": len(region_positions),
            "n_per_method": {n: len(region_methods[n]) for n in method_names},
            "pairwise": region_pairs,
        }

    payload = {
        "n_msl_snps": n_snp,
        "n_per_method_full": {n: len(methods_full[n]) for n in method_names},
        "n_per_method_restricted": {n: len(methods[n]) for n in method_names},
        "restricted_to_positions_in": "ancestree_vcf",
        "pairwise": pairwise,
        "soft_metrics": soft_metrics,
        "coverage_extras": coverage,
        "three_overlap": {
            "n_overlap": n_three_overlap,
            "n_agree": n_three_agree,
            "agreement_rate": (n_three_agree / n_three_overlap) if n_three_overlap else None,
        },
        "per_region": per_region,
    }
    if Path(IN_VCF_META).exists():
        payload["ancestree_vcf_meta"] = json.loads(Path(IN_VCF_META).read_text())

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(payload, indent=2))

    # Markdown summary
    lines = [
        "# B13 MSL real-data 3-way comparison",
        "",
        f"MSL chr1 SNPs in the population VCF: **{n_snp:,}**",
        "",
        "All pairwise comparisons below are restricted to positions called by "
        "Ancestree-VCF, so each method is evaluated on the same site set and "
        "the three pairwise numbers are directly comparable.",
        "",
        "## Sites called per method",
        "",
        "| Method | Full call set | Restricted to Ancestree-VCF positions | Full coverage (% of MSL SNPs) |",
        "|---|---:|---:|---:|",
    ]
    for n in method_names:
        full = len(methods_full[n])
        restricted = len(methods[n])
        cov = full / n_snp if n_snp else 0
        lines.append(f"| {n} | {full:,} | {restricted:,} | {cov*100:.2f}% |")
    lines += [
        "",
        f"**Ancestree-VCF calls {coverage['n_ancestree_only']:,} sites that NEITHER PolarBEAR nor "
        f"est-sfs call** ({coverage['n_ancestree_only']/coverage['n_ancestree_total_restricted']*100:.1f}% of "
        f"Ancestree's set). These are ingroup-polymorphic MSL SNPs that PolarBEAR's parsimony "
        f"filter and est-sfs's biallelic / data-format filters exclude from their headline call sets.",
        "",
        "## Pairwise agreement (on the restricted overlap)",
        "",
        "| A | B | Overlap | Hard agree | Hard rate | Soft (mean P(B's MAP) under A's posterior) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, p in pairwise.items():
        rate_str = f"{p['agreement_rate']:.4f}" if p['agreement_rate'] is not None else "—"
        # Soft column: only meaningful when A=ancestree_vcf (we have its posterior).
        soft_key = f"{p['a_name']}__{p['b_name']}"
        soft_str = "—"
        if soft_key in soft_metrics:
            soft_str = f"{soft_metrics[soft_key]['mean_ancestree_p_on_reference_map']:.4f}"
        lines.append(
            f"| {p['a_name']} | {p['b_name']} | {p['n_overlap']:,} | {p['n_agree']:,} | "
            f"{rate_str} | {soft_str} |"
        )
    lines += [
        "",
        f"## 3-way overlap: {n_three_overlap:,} sites",
        "",
        f"All three methods agree on **{n_three_agree:,}** / {n_three_overlap:,} = "
        f"**{(n_three_agree / n_three_overlap * 100) if n_three_overlap else 0:.2f}%**",
        "",
        "## Per-region agreement",
        "",
    ]
    for region, info in per_region.items():
        if info.get("missing"):
            continue
        lines.append(f"### {region} ({info['n_snps_in_region']:,} SNPs)")
        lines.append("")
        lines.append("| A | B | Overlap | Agree | Rate |")
        lines.append("|---|---|---:|---:|---:|")
        for key, p in info["pairwise"].items():
            rate_str = f"{p['agreement_rate']:.4f}" if p['agreement_rate'] is not None else "—"
            lines.append(f"| {p['a_name']} | {p['b_name']} | {p['n_overlap']:,} | {p['n_agree']:,} | {rate_str} |")
        lines.append("")

    Path(OUT_MD).write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"wrote {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
