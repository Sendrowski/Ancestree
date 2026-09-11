"""MSL chr1 pairwise-agreement matrix + heatmap across all five call sets.

Replaces the hand-maintained pairwise-agreement table in the MSL section with
a single computed matrix and a heatmap figure. Methods compared:

- ``ancestree_vcf``       — whole-chr1 fixed-tree streaming fit (one go)
- ``ancestree_arg``       — gamma-SMC ARG mode
- ``ancestree_localtree`` — VCF-only inferred local trees
- ``polarbear``           — PolarBEAR published calls
- ``estsfs``              — EST-SFS published calls

Each off-diagonal cell is the hard-agreement rate (MAP_A == MAP_B) on that
pair's overlapping positions. Outputs:

- ``results/reports/msl_agreement_matrix.json`` — full pairwise stats + soft
- ``results/reports/msl_agreement_heatmap.pdf``
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _msl_io import load_estsfs, load_pos_idx  # noqa: E402


def concat(glob_pat: str) -> dict[int, int]:
    out: dict[int, int] = {}
    for f in glob.glob(glob_pat):
        out.update(load_pos_idx(f))
    return out


try:
    out_json = snakemake.output.json  # type: ignore[name-defined]
    out_pdf = snakemake.output.pdf  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/msl_agreement_matrix.json"
    out_pdf = "results/reports/msl_agreement_heatmap.pdf"

PB = ("external/polarbear/data/real_data/ancestral_state/"
      "PolarBEAR_gammaSMC/anc_state.txt")
ES = ("external/polarbear/data/real_data/ancestral_state/"
      "est_sfs/est-sfs_ancstate.txt")
# PolarBEAR's own published per-site posteriors, from the archive: one row per
# polymorphic site (1,140,085, against the 687,338 its headline anc_state.txt
# publishes) with the non-informative / polytomy / n_mut flags intact, which is
# what lets filter_condition be reapplied below. The rerun's
# prob_ancstate_full.txt differs only by four per-state posterior columns, so
# requiring it forced a PolarBEAR re-execution to obtain flags already shipped.
# Those four columns are still needed for the SOFT agreement, which therefore
# stays unavailable from the archive alone (guarded by the len < 10 below).
_PB_ROOT = ("external/polarbear/data/real_data/ancestral_state/"
            "PolarBEAR_gammaSMC")
# The rerun's wide file carries the four per-state posterior columns the
# archive omits, which the SOFT agreement needs. It is a declared input, so the
# DAG builds it before this rule runs.
try:
    PB_FULL = str(snakemake.input.prob_full)  # type: ignore[name-defined]  # noqa: F821
except NameError:
    PB_FULL = "results/data/rerun_polarbear_workdir/prob_ancstate_full.txt"
PB_VCF = (f"{_PB_ROOT}/"
          "chr1.snp_only.collapsed.bed_filtered.MSL.cleaned.vcf.gz")

# PolarBEAR exposes a `filter_condition` (0-3) that governs which sites its
# pipeline *publishes*. Its headline set uses the strictest level 3. But its
# kernel computes a posterior at every site regardless — the filter only
# flags (informative / parsimony) and drops sites afterwards. Ancestree
# likewise runs through every site whether informative or not, so the fair
# inter-tool comparison is each kernel's full output (its hard limit), not
# PolarBEAR's filtered publication set. We therefore use filter_condition 0:
# every site PolarBEAR's kernel emits a call for. (Coverage in the table
# below still reports PolarBEAR's published level-3 set.)
PB_FILTER_CONDITION = 0


def load_polarbear_calls(filter_condition: int = PB_FILTER_CONDITION) -> dict[int, int]:
    """PolarBEAR per-site MAP (nucleotide index, ACGT) from the full
    posterior file, keeping sites that pass ``filter_condition``.

    Columns of ``prob_ancstate_full.txt``: pos, max-posterior, then the four
    integer flags PolarBEAR's filter reads --- best-allele index (>=0 when
    inferred), non-informative (0=informative), mutation count, polytomy
    (0=non-polytomy) --- followed by the 4-state posterior.
    """
    from cyvcf2 import VCF
    from ancestree import STATE_INDEX
    alleles_by_pos = {v.POS: [v.REF, *v.ALT] for v in VCF(PB_VCF)}
    out: dict[int, int] = {}
    for line in open(PB_FULL):
        p = line.rstrip("\n").split("\t")
        # Six columns is the published layout: pos, confidence, best,
        # non_info, n_mut, polytomy --- every field this loop reads. The
        # old bound of 10 assumed the rerun's wider file and would skip
        # every row of the archive's.
        if len(p) < 6:
            continue
        pos = int(p[0])
        best, non_info, n_mut, polytomy = (int(p[2]), int(p[3]),
                                           int(p[4]), int(p[5]))
        al = alleles_by_pos.get(pos)
        if al is None or best < 0 or best >= len(al):
            continue
        if filter_condition >= 1 and non_info != 0:
            continue
        if filter_condition >= 2 and polytomy != 0:
            continue
        if filter_condition >= 3 and not (n_mut < len(al)):
            continue
        nuc = STATE_INDEX.get(al[best].upper(), -1)
        if nuc >= 0:
            out[pos] = nuc
    return out


methods = {
    "Ancestree-VCF": load_pos_idx("results/data/msl_vcf_chr1_full_stream_calls.txt"),
    "Ancestree-ARG": concat("results/data/msl_arg_chr1_*_calls.txt"),
    "Ancestree-LocalTree": load_pos_idx("results/data/msl_localtree_chr1_full_calls.txt"),
    "PolarBEAR": load_polarbear_calls(),
    "EST-SFS": load_estsfs(ES),
}
names = list(methods)
for nm in names:
    print(f"  {nm}: {len(methods[nm]):,} calls", flush=True)

K = len(names)
rate = np.full((K, K), np.nan)
overlap = np.zeros((K, K), dtype=np.int64)
agree = np.zeros((K, K), dtype=np.int64)
for a in range(K):
    for b in range(K):
        A, B = methods[names[a]], methods[names[b]]
        ov = A.keys() & B.keys()
        n_ov = len(ov)
        n_ag = sum(1 for p in ov if A[p] == B[p])
        overlap[a, b] = n_ov
        agree[a, b] = n_ag
        rate[a, b] = (n_ag / n_ov) if n_ov else np.nan

# Soft agreement: every tool exposes a per-site posterior over A,C,G,T, so for
# each ordered pair (a, b) we can measure the mean posterior mass method a
# places on method b's MAP allele, over the same overlap as the hard rate.
#   - Ancestree modes / EST-SFS: full ACGT posterior, indexed directly.
#   - PolarBEAR: prob_ancstate_full.txt (our patched run, which appends the full
#     normalised 4-state vector PolarBEAR's kernel computes but natively
#     discards) gives the posterior in VCF allele order (ref, alt1, ...). We
#     rotate it into ACGT with the per-site REF/ALT from the run's VCF. The
#     remapped MAP reproduces the published anc_state calls exactly.
arg_post = sorted(glob.glob("results/data/msl_arg_chr1_*_posteriors.npz"))
post_npz = {
    "Ancestree-VCF": ["results/data/msl_vcf_chr1_full_stream_posteriors.npz"],
    "Ancestree-LocalTree": ["results/data/msl_localtree_chr1_full_posteriors.npz"],
    "Ancestree-ARG": arg_post,
}
ES_IDXPOS = ("external/polarbear/data/real_data/ancestral_state/est_sfs/"
             "est_sfs_index_pos.txt")
ES_PVALS = ("external/polarbear/data/real_data/ancestral_state/est_sfs/"
            "output-file-pvalues.txt")

all_map_pos = set().union(*(methods[m].keys() for m in names))


def _load_post(paths):
    pos = np.concatenate([np.load(p, allow_pickle=True)["pos"] for p in paths])
    post = np.concatenate([np.load(p, allow_pickle=True)["posterior"] for p in paths])
    return {int(p): i for i, p in enumerate(pos)}, post


# ACGT posteriors keyed by source method (restricted to MAP positions).
ANC: dict[str, tuple] = {}
for mode, paths in post_npz.items():
    paths = [p for p in paths if p and Path(p).exists()]
    if paths:
        ANC[mode] = _load_post(paths)

ES_post: dict[int, np.ndarray] = {}
if Path(ES_PVALS).exists() and Path(ES_IDXPOS).exists():
    ip = [int(l.split()[2]) for l in open(ES_IDXPOS) if len(l.split()) >= 3]
    for line in open(ES_PVALS):
        p = line.split()
        if len(p) < 7:
            continue
        try:
            r = int(p[0])
        except ValueError:
            continue
        if r == 0:  # header lines
            continue
        pos = ip[r - 1]
        if pos in all_map_pos:
            ES_post[pos] = np.array([float(p[3]), float(p[4]), float(p[5]), float(p[6])])

PB_post: dict[int, np.ndarray] = {}
if Path(PB_FULL).exists() and Path(PB_VCF).exists():
    from cyvcf2 import VCF
    from ancestree import STATE_INDEX
    # pos -> nucleotide index (ACGT) of each VCF allele, in allele order.
    pb_alleles = {}
    for v in VCF(PB_VCF):
        if v.POS in all_map_pos:
            pb_alleles[v.POS] = [STATE_INDEX.get(a.upper(), -1) for a in (v.REF, *v.ALT)]
    for line in open(PB_FULL):
        p = line.split("\t")
        if len(p) < 10:
            continue
        pos = int(p[0])
        al = pb_alleles.get(pos)
        if al is None:
            continue
        lik = [float(x) for x in p[6:10]]  # posterior in VCF allele order
        vec = np.zeros(4)
        if all(0 <= nuc < 4 for nuc in al[:4]):
            for a_idx, nuc in enumerate(al[:4]):
                vec[nuc] += lik[a_idx]
            PB_post[pos] = vec


# Unify every method's posterior into (pos -> row index, (N, 4) ACGT matrix),
# so both the soft accuracy and the soft (posterior) agreement vectorise.
def _dict_to_mat(d: dict[int, np.ndarray]):
    keys = list(d)
    return {p: i for i, p in enumerate(keys)}, np.array([d[p] for p in keys])


POST: dict[str, tuple] = dict(ANC)
if ES_post:
    POST["EST-SFS"] = _dict_to_mat(ES_post)
if PB_post:
    POST["PolarBEAR"] = _dict_to_mat(PB_post)


def _shared_post(a: str, b: str):
    """Aligned posteriors of ``a`` and ``b`` over their shared MAP positions."""
    if a not in POST or b not in POST:
        return None
    rowA, postA = POST[a]
    rowB, postB = POST[b]
    ov = [p for p in (methods[a].keys() & methods[b].keys()) if p in rowA and p in rowB]
    if not ov:
        return None
    iA = np.fromiter((rowA[p] for p in ov), dtype=np.int64, count=len(ov))
    iB = np.fromiter((rowB[p] for p in ov), dtype=np.int64, count=len(ov))
    mb = np.fromiter((methods[b][p] for p in ov), dtype=np.int64, count=len(ov))
    return postA[iA], postB[iB], mb


# soft[a, b]   : mean posterior mass a places on b's MAP allele (directional).
# brier[a, b]  : mean per-site Brier score of a's posterior against b's,
#                sum_s (P_a - P_b)^2, symmetric, in [0, 2], zero on the
#                diagonal, and the standard Brier score wherever b is a
#                one-hot reference.
soft = np.full((K, K), np.nan)
brier = np.full((K, K), np.nan)
for a in range(K):
    for b in range(K):
        sp = _shared_post(names[a], names[b])
        if sp is None:
            continue
        PA, PB_, mb = sp
        soft[a, b] = float(np.mean(PA[np.arange(len(mb)), mb]))
        brier[a, b] = float(np.mean(((PA - PB_) ** 2).sum(1)))

# A method whose posteriors never loaded leaves its whole row and column NaN,
# which the main-text table then formats as the literal string "nan".
_absent = [nm for i, nm in enumerate(names)
           if not np.isfinite(brier[i]).any()]
if _absent:
    raise ValueError(
        f"no Brier score could be computed for {_absent}: their "
        f"per-site posteriors did not load, so every pairing is undefined. "
        f"Check that the posterior source carries the four per-state columns."
    )

result = {
    "methods": names,
    "n_calls": {nm: len(methods[nm]) for nm in names},
    "rate": rate.tolist(),
    "overlap": overlap.tolist(),
    "agree": agree.tolist(),
    "soft": soft.tolist(),
    "brier": brier.tolist(),
}
Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2)
print(f"wrote {out_json}", flush=True)

# -------------------------------------------------------- heatmap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

short = {
    "Ancestree-VCF": "Ancestree\nfixed-tree",
    "Ancestree-ARG": "Ancestree\nARG",
    "Ancestree-LocalTree": "Ancestree\nlocal-tree",
    "PolarBEAR": "PolarBEAR",
    "EST-SFS": "EST-SFS",
}
labels = [short[n] for n in names]
fig, ax = plt.subplots(figsize=(5.2, 4.4))
# Symmetric matrix → show only the upper triangle (incl. diagonal). Blank the
# redundant lower triangle.
_lower = np.tril(np.ones((K, K), dtype=bool), k=-1)
masked = np.ma.masked_array(brier, mask=np.isnan(brier) | _lower)
# Brier is a distance, so the scale starts at zero and runs to the worst
# pairing, rounded up, keeping the low-Brier structure legible.
_vmin = 0.0
_vmax = float(np.nanmax(masked)) if np.isfinite(masked).any() else 1.0
_vmax = max(_vmax, 1e-6)
# RdYlGn reversed (the robustness heatmap, Figure 4, reads green = good):
# low Brier = green = good, high = red.
cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad("0.85")
im = ax.imshow(masked, cmap=cmap, vmin=_vmin, vmax=_vmax, aspect="equal")
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xticks(np.arange(-0.5, K, 1), minor=True)
ax.set_yticks(np.arange(-0.5, K, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", length=0)
for i in range(K):
    for j in range(K):
        if j < i or np.isnan(brier[i, j]):  # upper triangle only
            continue
        val = brier[i, j]
        # Top line (and colour): Brier score of the two posteriors.
        # Bottom: MAP agreement fraction.
        lbl = f"{val:.3f}"
        if np.isfinite(rate[i, j]):
            lbl = f"{val:.3f}\n({rate[i, j]:.3f})"
        # Pick text colour by the cell's actual luminance so it stays readable
        # on the bright-yellow mid-range as well as the dark red/green ends.
        r, g, b, _ = cmap((np.clip(val, _vmin, _vmax) - _vmin) / (_vmax - _vmin))
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        ax.text(j, i, lbl,
                ha="center", va="center", fontsize=7,
                color="white" if lum < 0.5 else "black")
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r"Brier score $\sum_s (P_A - P_B)^2$", fontsize=9)
fig.tight_layout()
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(str(Path(out_pdf).with_suffix(".png")), dpi=160, bbox_inches="tight")
print(f"wrote {out_pdf}", flush=True)
