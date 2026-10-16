"""Spectra of the simulation matched to the EPO comparison, summed over the
replicates.

For every polarisation and site class, the full unfolded spectrum over derived
counts ``0..n``, built as in ``report_epo_spectra.py``, in the
posterior-weighted and the MAP form. The classes restricted to discordant
local trees keep their polymorphic bins and divergence count alone. The mean
per-site Brier score of each run is carried through.
"""
import json
from pathlib import Path

import numpy as np

try:
    in_replicates = list(snakemake.input.replicates)  # type: ignore[name-defined]
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    import glob
    in_replicates = sorted(glob.glob("results/data/epo_simulation_spectra_seed*.json"))
    out_json = "results/reports/epo_simulation.json"

replicates = [json.loads(Path(p).read_text(encoding="utf-8")) for p in in_replicates]
n = replicates[0]["n_haplotypes"]
classes = ["normal", "hypermutable"]
n_eligible = {c: sum(r["n_eligible"][c] for r in replicates) for c in classes}
result = dict(n_replicates=len(replicates), n_haplotypes=n,
              n_eligible={**n_eligible, "all": sum(n_eligible.values())},
              discordant_fraction=float(np.mean([r["discordant_fraction"] for r in replicates])),
              methods={})
for method in replicates[0]["methods"]:
    by_class = {}
    for suffix in ("", ", discordant"):
        for c in classes:
            parts = [r["methods"][method][c + suffix] for r in replicates]
            entry = {}
            for kind in ("weighted", "map"):
                spectrum = sum(np.asarray(p["spectrum_" + kind]) for p in parts)
                spectrum[n] = float(sum(p["divergence_" + kind] for p in parts))
                if not suffix:
                    spectrum[0] = (n_eligible[c] - sum(p["n_polymorphic"] for p in parts)
                                   - spectrum[n])
                entry["spectrum_" + kind] = spectrum.tolist()
            by_class[c + suffix] = entry
        by_class["all" + suffix] = {
            key: (np.asarray(by_class["normal" + suffix][key])
                  + np.asarray(by_class["hypermutable" + suffix][key])).tolist()
            for key in by_class["normal" + suffix]}
    result["methods"][method] = by_class

# Mean per-site Brier score of each run against the truth at its own node.
result["brier"] = {}
for method in replicates[0]["brier"]:
    by_class = {}
    for c in list(replicates[0]["brier"][method]):
        total = sum(r["brier"][method][c]["total"] for r in replicates)
        count = sum(r["brier"][method][c]["n"] for r in replicates)
        by_class[c] = dict(mean=total / max(count, 1), n=count)
    result["brier"][method] = by_class

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=1)
print(f"report_epo_simulation: {len(replicates)} replicates -> {out_json}", flush=True)
