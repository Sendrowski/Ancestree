"""B13 MSL real-data benchmark: verify PolarBEAR data archive layout.

The PolarBEAR paper's "Performance on Real Data" section uses a
real-data archive hosted on Edmond (MPG) — too big to commit. This
script checks the expected files under
``external/polarbear/data/real_data/`` and emits a small manifest JSON
listing what was found / missing. Downstream rules (:mod:`infer_msl_arg`,
:mod:`infer_msl_vcf`, :mod:`report_msl`) declare the same files as
``input:``, so snakemake will produce a clear "missing input" error if
the archive isn't populated.

Manual download (if the snakemake-level rule didn't fetch it for you):

1. Visit https://edmond.mpg.de/file.xhtml?fileId=330797&version=2.0
2. Click "Access File" → "Download"  (or
   ``curl -L -o /tmp/real_data.zip
   'https://edmond.mpg.de/api/access/datafile/330797'``).
3. Extract into ``external/polarbear/data/real_data/`` so the README's
   directory layout is preserved.

Run directly::

    python workflow/scripts/prepare_msl_realdata.py

Or via snakemake::

    snakemake -j 1 results/data/msl_manifest.json
"""
import json
from pathlib import Path


try:
    real_data_root = Path(snakemake.input.real_data_root)  # type: ignore[name-defined]
    out_manifest = snakemake.output.manifest  # type: ignore[name-defined]
except NameError:
    real_data_root = Path("external/polarbear/data/real_data")
    out_manifest = "results/data/msl_manifest.json"


# Canonical files we expect after extracting the PolarBEAR real-data
# archive. Paths are relative to ``real_data_root``. The keys are
# stable. Downstream scripts look them up by name.
EXPECTED: dict[str, str] = {
    # 1000G MSL chr1 cleaned (population-only, no outgroups). The same
    # VCF fed to gamma-SMC for ARG inference and to PolarBEAR's
    # ancestral-state output script.
    "msl_vcf_population": (
        "ancestral_state/PolarBEAR_gammaSMC/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.vcf.gz"
    ),
    # MSL chr1 + ponAbe2 + macFas5 outgroup VCF (pseudo-VCF format —
    # the bcftools query output described in the README).
    "msl_vcf_outgroups": (
        "ancestral_state/est_sfs/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.ponAbe2_macFas5.simplify.vcf.gz"
    ),
    # gamma-SMC tskit tree sequence over the MSL VCF.
    "gammaSMC_trees": (
        "ancestral_state/PolarBEAR_gammaSMC/"
        "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned_tskit.trees"
    ),
    # PolarBEAR pre-computed ancestral-state output (pos<TAB>allele_idx).
    "polarbear_anc_state": "ancestral_state/PolarBEAR_gammaSMC/anc_state.txt",
    # est-sfs pre-computed ancestral-state output (chr<TAB>pos<TAB>allele_char).
    "estsfs_anc_state": "ancestral_state/est_sfs/est-sfs_ancstate.txt",
    # SnpEff per-region position lists. Each is a one-per-line list of
    # 1-based chr1 positions for SNPs in that annotation class.
    "snpeff_exon": "SnpEff_annotation/exon_pos.txt",
    "snpeff_utr3": "SnpEff_annotation/UTR_3_pos.txt",
    "snpeff_utr5": "SnpEff_annotation/UTR_5_pos.txt",
    "snpeff_intron": "SnpEff_annotation/intron_pos.txt",
    "snpeff_intergenic": "SnpEff_annotation/intergenic_pos.txt",
    "snpeff_synonymous": "SnpEff_annotation/synonymous_pos.txt",
    "snpeff_non_synonymous": "SnpEff_annotation/non_synonymous_pos.txt",
    # MSL panel — list of sample ids used for the ancestral-state task.
    "msl_panel": "ancestral_state/MSL.panel",
    # 1000G mask. Used only to derive callable chr1 length for the
    # FixedTreeInference monomorphic-site weighting. Optional — we fall
    # back to GRCh38 chr1 length when absent.
    "chr1_mask": "chr1.mask.bed",
}


present: dict[str, dict] = {}
missing: list[str] = []
for key, rel in EXPECTED.items():
    p = real_data_root / rel
    if p.exists():
        present[key] = {
            "path": str(p),
            "size_bytes": p.stat().st_size,
        }
    else:
        missing.append(key)


manifest = {
    "real_data_root": str(real_data_root),
    "present": present,
    "missing": missing,
    "complete": not missing,
    "download_url": (
        "https://edmond.mpg.de/file.xhtml?fileId=330797&version=2.0"
    ),
    "instructions": (
        "Download the PolarBEAR real-data archive and extract under "
        "external/polarbear/data/real_data/ — see this script's "
        "docstring + the local README at "
        "external/polarbear/data/real_data/README.md."
    ),
}


Path(out_manifest).parent.mkdir(parents=True, exist_ok=True)
with open(out_manifest, "w") as f:
    json.dump(manifest, f, indent=2)

if missing:
    print(
        f"prepare_msl_realdata: MISSING {len(missing)}/{len(EXPECTED)} "
        f"file(s) under {real_data_root}: {missing}",
        flush=True,
    )
    print(
        "Manifest written; downstream rules will fail with explicit "
        "missing-input errors. Populate the data per the docstring "
        "and re-run.",
        flush=True,
    )
else:
    print(
        f"prepare_msl_realdata: all {len(EXPECTED)} expected files "
        f"present under {real_data_root}",
        flush=True,
    )
print(f"Wrote {out_manifest}", flush=True)
