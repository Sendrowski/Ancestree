"""Appendix: on PolarBEAR's informative-genealogy sites, how closely do the
other call sets reproduce PolarBEAR's posterior / MAP call?

The posterior-level score is the Brier score of each mode's per-site
posterior against PolarBEAR's, ``sum_a (P_A - P_B)_a^2`` averaged over the
shared positions, so lower is better and zero means identical posteriors.

PolarBEAR drops the non-informative class (the local genealogy carries no
signal to orient the site). On the sites it *does* keep, the question is how
much the genealogy adds over a plain ingroup-frequency prior, answered by
comparing PolarBEAR against each ``Ancestree`` mode, including fixed-tree mode
with **no outgroups**, which reduces to the Kingman SFS prior alone
(count-proportional ``n_j / n_obs``, the majority allele as ancestral).

Output: ``results/reports/msl_informative_prior.json``.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from cyvcf2 import VCF

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ancestree import STATE_INDEX  # noqa: E402

PB_FULL = "results/data/rerun_polarbear_workdir/prob_ancstate_full.txt"
PB_VCF = ("results/data/rerun_polarbear_workdir/"
          "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.vcf.gz")

try:
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/msl_informative_prior.json"


# --- VCF: per-site alleles (ACGT idx) + ingroup nucleotide counts ----------
print("loading PolarBEAR VCF (alleles + ingroup counts) ...", flush=True)
alleles: dict[int, list[int]] = {}
ing_counts: dict[int, np.ndarray] = {}
for v in VCF(PB_VCF):
    al = [STATE_INDEX.get(a.upper(), -1) for a in (v.REF, *v.ALT)]
    alleles[v.POS] = al
    gt = v.genotype.array()[:, :2].ravel()
    gt = gt[gt >= 0]
    c = np.zeros(4)
    for a_idx, nuc in enumerate(al):
        if nuc >= 0:
            c[nuc] += np.count_nonzero(gt == a_idx)
    ing_counts[v.POS] = c


# --- PolarBEAR informative posteriors (ACGT) + the no-outgroup Kingman -------
print("reading PolarBEAR posteriors on informative sites ...", flush=True)
pb_post: dict[int, np.ndarray] = {}
kingman_post: dict[int, np.ndarray] = {}
for line in open(PB_FULL):
    p = line.rstrip("\n").split("\t")
    if len(p) < 10:
        continue
    pos = int(p[0])
    best, non_info = int(p[2]), int(p[3])
    if best < 0 or non_info != 0:  # informative genealogy only
        continue
    al = alleles.get(pos)
    c = ing_counts.get(pos)
    if al is None or c is None or c.sum() <= 0:
        continue
    lik = [float(x) for x in p[6:10]]  # posterior in VCF allele order
    vec = np.zeros(4)
    ok = True
    for a_idx, nuc in enumerate(al):
        if 0 <= nuc < 4:
            vec[nuc] += lik[a_idx]
        elif a_idx < len(lik) and lik[a_idx] > 0:
            ok = False
    if not ok or vec.sum() <= 0:
        continue
    pb_post[pos] = vec / vec.sum()
    # fixed-tree, no outgroups == Kingman prior alone: count-proportional.
    kingman_post[pos] = c / c.sum()

n_info = len(pb_post)
print(f"informative sites with usable posteriors: {n_info:,}", flush=True)


def _load_npz(paths):
    paths = [q for q in paths if q and Path(q).exists()]
    if not paths:
        return None
    pos = np.concatenate([np.load(q, allow_pickle=True)["pos"] for q in paths])
    post = np.concatenate([np.load(q, allow_pickle=True)["posterior"] for q in paths])
    return {int(x): i for i, x in enumerate(pos)}, post


# All comparators here use NO outgroups, matching PolarBEAR's setup: the
# question is what the ingroup genealogy (or, at the limit, the ingroup
# frequency prior) recovers on its own.
sources = {
    "ARG mode": _load_npz(sorted(glob.glob(
        "results/data/msl_arg_chr1_*_posteriors.npz"))),
    "local-tree mode": _load_npz(
        ["results/data/msl_localtree_chr1_full_posteriors.npz"]),
}


def _agreement(other_post: np.ndarray, idx: dict[int, int]) -> tuple[float, float, int]:
    """Brier score + MAP agreement of an npz source vs PolarBEAR over the
    shared informative positions."""
    ov = [pos for pos in pb_post if pos in idx]
    if not ov:
        return float("nan"), float("nan"), 0
    A = np.array([pb_post[pos] for pos in ov])
    B = other_post[np.fromiter((idx[pos] for pos in ov),
                               dtype=np.int64, count=len(ov))]
    B = B / B.sum(1, keepdims=True)
    brier = float(np.mean(((A - B) ** 2).sum(1)))
    mapa = float(np.mean(A.argmax(1) == B.argmax(1)))
    return brier, mapa, len(ov)


rows = []
# Kingman (no-outgroup fixed-tree), computed in-line above.
ov = list(pb_post)
A = np.array([pb_post[pos] for pos in ov])
B = np.array([kingman_post[pos] for pos in ov])
rows.append({
    "method": "fixed-tree mode (Kingman prior, no outgroups)",
    "brier": float(np.mean(((A - B) ** 2).sum(1))),
    "map_agreement": float(np.mean(A.argmax(1) == B.argmax(1))),
    "n": len(ov),
})
missing = [name for name, loaded in sources.items() if loaded is None]
if missing:
    raise FileNotFoundError(
        f"no posteriors found for {missing}, so the informative-prior table "
        f"would be written without those rows and the manuscript would build "
        f"around a silently truncated comparison. Build the chr1 ARG and "
        f"local-tree runs first, or drop the rows from the table definition.")
for name, loaded in sources.items():
    idx, post = loaded
    brier, mapa, n = _agreement(post, idx)
    rows.append({"method": name, "brier": brier,
                 "map_agreement": mapa, "n": n})

result = {"n_informative_sites": n_info, "reference": "PolarBEAR", "rows": rows}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
Path(out_json).write_text(json.dumps(result, indent=2))
print(f"wrote {out_json}", flush=True)
for r in rows:
    print(f"  {r['method']:46s} brier={r['brier']:.4f} "
          f"MAP={r['map_agreement']:.4f}  (N={r['n']:,})", flush=True)

