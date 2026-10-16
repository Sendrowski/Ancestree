"""Chromosome-wide spectra of the EPO comparison, summed over the regions.

For every polarisation and CpG context, and for all sites together, the full
unfolded spectrum over derived counts ``0..n``: the polymorphic bins as
tabulated, bin ``n`` the divergence count, and bin ``0`` the remaining
eligible positions. Both the posterior-weighted and the MAP form are kept.
"""
import json
from pathlib import Path

import numpy as np

try:
    in_regions = list(snakemake.input.regions)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    import glob
    in_regions = sorted(glob.glob("results/data/epo_spectra_chr1_*.json"))
    out_json = "results/reports/epo_spectra.json"

regions = [json.loads(Path(p).read_text(encoding="utf-8")) for p in in_regions]
n = regions[0]["n_haplotypes"]
contexts = ["non-CpG", "CpG"]
n_eligible = {c: sum(r["n_eligible"][c] for r in regions) for c in contexts}

result = dict(n_regions=len(regions), n_haplotypes=n,
              n_eligible={**n_eligible, "all": sum(n_eligible.values())},
              n_without_epo_call=sum(r["n_without_epo_call"] for r in regions),
              kappa=[r["kappa"] for r in regions], methods={},
              seconds={k: sum(r["seconds"][k] for r in regions) for k in regions[0]["seconds"]})
for method in regions[0]["methods"]:
    by_context = {}
    for c in contexts:
        parts = [r["methods"][method][c] for r in regions]
        entry = {}
        for kind in ("weighted", "map"):
            spectrum = sum(np.asarray(p["spectrum_" + kind]) for p in parts)
            divergence = float(sum(p["divergence_" + kind] for p in parts))
            n_polymorphic = sum(p["n_polymorphic"] for p in parts)
            spectrum[n] = divergence
            spectrum[0] = n_eligible[c] - n_polymorphic - divergence
            entry["spectrum_" + kind] = spectrum.tolist()
        entry["n_polymorphic"] = int(sum(p["n_polymorphic"] for p in parts))
        entry["n_fixed_variant"] = int(sum(p["n_fixed"] for p in parts))
        by_context[c] = entry
    by_context["all"] = {
        key: (np.asarray(by_context["non-CpG"][key]) + np.asarray(by_context["CpG"][key])).tolist()
        for key in by_context["CpG"]}
    result["methods"][method] = by_context

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(f"report_epo_spectra: {len(regions)} regions, {result['n_eligible']['all']:,} positions "
      f"-> {out_json}", flush=True)
