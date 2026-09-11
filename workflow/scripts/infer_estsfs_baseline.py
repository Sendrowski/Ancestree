"""B3 est-sfs baseline (Keightley & Jackson 2018) wrapper.

Independent C implementation of est-sfs invoked as an external binary, the
reference the fixed-tree fit is compared against. est-sfs's expected text input
is written from the same simulated VCF + meta that the fastDFE cell consumes
(see :mod:`infer_estsfs_fastdfe`), the binary is run, and the per-site
ancestral-state probabilities are parsed back into the JSON shape the report
consumes, the same one the fastDFE cell emits.

Binary lookup order:

1. ``$EST_SFS_BIN`` env var, if set.
2. ``est-sfs`` on ``$PATH`` (bioconda, via ``workflow/envs/bench.yml``).

Installed from bioconda (est-sfs 2.04, linux-64 and osx-64 builds exist,
there is no osx-arm64 build so Apple silicon resolves it under Rosetta).

Wildcards (Snakemake-driven):

- ``{sim}``    ∈ ``{jc, hky2}``
- ``{model}``  ∈ ``{JC, K2, R6}``  (R6 = est-sfs's six-parameter rate model)
- ``{n_out}``  ∈ ``{1, 2, 3}``

est-sfs's input format per site:

    n_A,n_C,n_G,n_T  o1_A,o1_C,o1_G,o1_T  o2_...  o3_...

where ``n_*`` are ingroup nucleotide counts and ``oN_*`` are one-hot
outgroup observations. The closest-first outgroup subset
(``meta["outgroup_names"][:n_out]``) is the one the other B3 cells use, so cells
are aligned site-for-site.

Run directly (defaults to JC + n_out=3 on the seeded jc sim)::

    python workflow/scripts/infer_estsfs_baseline.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ---------------------------------------------------------- snakemake glue

try:
    in_vcf = snakemake.input.vcf  # type: ignore[name-defined]
    in_meta = snakemake.input.meta  # type: ignore[name-defined]
    out_json = snakemake.output[0]  # type: ignore[name-defined]
    model_name = snakemake.wildcards.model  # type: ignore[name-defined]
    n_out = int(snakemake.wildcards.n_out)  # type: ignore[name-defined]
    sim = snakemake.wildcards.sim  # type: ignore[name-defined]
except NameError:
    in_vcf = "results/data/estsfs_sim_jc.vcf"
    in_meta = "results/data/estsfs_sim_jc_meta.json"
    out_json = "results/data/estsfs_baseline_jc_JC_n3.json"
    model_name = "JC"
    n_out = 3
    sim = "jc"


# ---------------------------------------------------------- binary lookup

# est-sfs comes from bioconda via this rule's conda env, so it is on PATH.
# EST_SFS_BIN still overrides, for running the script outside snakemake.
EST_SFS_BIN = os.environ.get("EST_SFS_BIN") or shutil.which("est-sfs")
if EST_SFS_BIN is None:
    raise FileNotFoundError(
        "est-sfs is not on PATH. It is provided by workflow/envs/bench.yml "
        "(bioconda, linux-64 / osx-64); run this rule under that env, or set "
        "EST_SFS_BIN=/path/to/est-sfs."
    )


# ---------------------------------------------------------- est-sfs config

MODEL_CODES = {"JC": 0, "K2": 1, "R6": 2}
if model_name not in MODEL_CODES:
    raise ValueError(f"Unknown est-sfs model {model_name!r}; pick from {sorted(MODEL_CODES)}")

NRANDOM = 10  # est-sfs's default random-restart count for the ML fit
STATES = ("A", "C", "G", "T")
STATE_IDX = {s: i for i, s in enumerate(STATES)}


# ---------------------------------------------------------- read inputs

import cyvcf2  # noqa: E402 — imported after the binary check to fail fast

with open(in_meta) as f:
    meta = json.load(f)
outgroup_names = list(meta["outgroup_names"])[:n_out]
ingroup_names = list(meta["ingroup_names"])

vcf = cyvcf2.VCF(in_vcf)
sample_to_idx = {s: i for i, s in enumerate(vcf.samples)}
ingroup_cols = [sample_to_idx[s] for s in ingroup_names]
outgroup_cols = [sample_to_idx[s] for s in outgroup_names]


# ---------------------------------------------------------- emit est-sfs input

def _counts_at(variant, cols: list[int]) -> tuple[int, int, int, int]:
    """Per-state counts at this variant for the given sample columns.

    Returns a length-4 tuple in STATES order.
    """
    counts = [0, 0, 0, 0]
    bases = variant.gt_bases
    for c in cols:
        b = bases[c]
        # cyvcf2 returns either "A" / "T" for haploid calls or
        # "A|T" / "./." for diploid. Our sims are haploid. Pad anyway.
        b = b.split("|")[0].split("/")[0]
        if b in STATE_IDX:
            counts[STATE_IDX[b]] += 1
    return tuple(counts)


tmpdir = Path(tempfile.mkdtemp(prefix="estsfs_"))
data_path = tmpdir / "data.txt"
config_path = tmpdir / "config.txt"
seed_path = tmpdir / "seedfile"
sfs_path = tmpdir / "sfs.txt"
panc_path = tmpdir / "p_anc.txt"

# Sequence length to pad with monomorphic sites — est-sfs calibrates K
# from the ratio of polymorphic to monomorphic configurations and only
# converges to the right rates when monomorphic sites are present. Under
# JC69 / K2 (symmetric) the choice of monomorphic base is irrelevant, so
# we just emit a uniform "all A" monomorphic line per non-polymorphic
# position.
trees_path = Path(in_vcf).with_suffix(".trees")
if trees_path.exists():
    import tskit
    sequence_length = int(tskit.load(str(trees_path)).sequence_length)
else:
    sequence_length = int(meta.get("length") or meta.get("sequence_length") or 0)
if sequence_length <= 0:
    raise ValueError(
        f"no sequence length for {in_vcf}: {trees_path} is absent and the "
        f"metadata carries neither 'length' nor 'sequence_length'. Without it "
        f"no monomorphic sites are emitted and est-sfs does not calibrate its "
        f"rates, so the baseline it returns would not be comparable.")
n_ingroup = len(ingroup_names)
mono_line = (
    f"{n_ingroup},0,0,0 " + " ".join("1,0,0,0" for _ in outgroup_cols)
)
poly_positions: set[int] = set()

site_positions: list[int] = []
site_alleles: list[tuple[str, ...]] = []
site_major: list[str] = []  # major (most-frequent) ingroup allele per site
site_minor: list[str | None] = []  # minor (second-most-frequent), or None if monomorphic
with data_path.open("w") as data_f:
    for variant in vcf:
        if not (variant.is_snp and not variant.is_indel):
            continue
        ref = (variant.REF or "").upper()
        alts = [a.upper() for a in (variant.ALT or []) if a]
        all_alleles = [ref, *alts]
        if not all(len(a) == 1 and a in STATE_IDX for a in all_alleles):
            continue
        ingroup_counts = _counts_at(variant, ingroup_cols)
        # est-sfs expects each outgroup as one fixed observation
        # (no within-outgroup polymorphism). Pull one allele per outgroup,
        # skip the site if any outgroup is missing.
        outgroup_one_hots: list[tuple[int, int, int, int]] = []
        skip = False
        for c in outgroup_cols:
            b = variant.gt_bases[c].split("|")[0].split("/")[0]
            if b not in STATE_IDX:
                skip = True
                break
            oh = [0, 0, 0, 0]
            oh[STATE_IDX[b]] = 1
            outgroup_one_hots.append(tuple(oh))
        if skip:
            continue
        fields = [",".join(str(c) for c in ingroup_counts)]
        fields.extend(",".join(str(c) for c in oh) for oh in outgroup_one_hots)
        data_f.write(" ".join(fields) + "\n")
        # Major / minor by ingroup count, ties broken by STATES order.
        ranked = sorted(range(4), key=lambda i: (-ingroup_counts[i], i))
        site_major.append(STATES[ranked[0]])
        site_minor.append(STATES[ranked[1]] if ingroup_counts[ranked[1]] > 0 else None)
        site_positions.append(int(variant.POS))
        site_alleles.append(tuple(all_alleles))
        poly_positions.add(int(variant.POS))

    vcf.close()
    # Pad with monomorphic lines so est-sfs has the right poly/mono ratio.
    n_mono = max(0, sequence_length - len(poly_positions))
    for _ in range(n_mono):
        data_f.write(mono_line + "\n")

with config_path.open("w") as cfg_f:
    cfg_f.write(f"n_outgroup {n_out}\n")
    cfg_f.write(f"model {MODEL_CODES[model_name]}\n")
    cfg_f.write(f"nrandom {NRANDOM}\n")

# est-sfs reads a seed from this file and writes a fresh one back. The
# exact bytes don't matter for reproducibility under nrandom>1.
seed_path.write_text("42\n")


# ---------------------------------------------------------- run binary

cmd = [
    EST_SFS_BIN,
    str(config_path), str(data_path), str(seed_path),
    str(sfs_path), str(panc_path),
]
print(f"$ {' '.join(cmd)}", file=sys.stderr)
# Wall-clock for the est-sfs binary call only (post data prep, pre output parse);
# reported in the meta block for the manuscript runtime column.
_t_inference_start = time.perf_counter()
proc = subprocess.run(cmd, capture_output=True, text=True)
inference_seconds = float(time.perf_counter() - _t_inference_start)
if proc.returncode != 0:
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise RuntimeError(
        f"est-sfs exited with status {proc.returncode}; see stderr above"
    )


# ---------------------------------------------------------- parse output

# p_anc.txt layout: lines starting with "0 " are header (tool version,
# site count, model, ML, ML-random-starts, per-branch rates, column
# legend). Data lines start with the 1-based site index, then:
#     <site_idx> <config_ind> <p_anc_major> <ptree_1> ... <ptree_n_tree>
# (n_tree = 4**(n_outgroup-1), these are per-internal-tree-state
# probabilities, NOT per-allele.) We use ``p_anc_major`` together with
# the major/minor ingroup alleles tracked above to derive a 4-state
# posterior comparable to Ancestree's: the major allele gets
# ``p_anc_major``, the minor gets ``1 - p_anc_major``, the two unobserved
# alleles get 0.
# Parse fitted branch rates from p_anc.txt's header (line starting with
# "0 Rates: k0 X k1 Y ..."). Derive ingroup-MRCA-to-outgroup divergences
# from the K_i exactly as Ancestree's OutgroupLadderTree does:
#   d(O_1) = K_0 + K_1
#   d(O_k) = K_0 + K_2 + K_4 + ... + K_{2(k-1)} + K_{2k-1}    (k > 1)
#   d(O_n) = K_0 + K_2 + ... + K_{2(n-2)} + K_{2(n-1)}        (deepest)
fitted_rates: list[float] = []
header_lines: list[str] = []
for line in panc_path.open():
    if not line.startswith("0 "):
        break
    header_lines.append(line)
    if line.startswith("0 Rates:"):
        toks = line.split()
        # tokens after "0 Rates:" alternate "kI" / "<value>"
        fitted_rates = [float(t) for t in toks[3::2]]


def _ladder_divergences(K: list[float], n: int) -> list[float]:
    """K_0..K_{2n-2} → ingroup-MRCA-to-outgroup-k divergences for k=1..n."""
    out: list[float] = []
    for k in range(1, n + 1):
        if k < n:
            d = K[0] + sum(K[2 * j] for j in range(1, k)) + K[2 * k - 1]
        else:
            d = K[0] + sum(K[2 * j] for j in range(1, k))
        out.append(d)
    return out


fitted_outgroup_divergence = (
    _ladder_divergences(fitted_rates, n_out) if len(fitted_rates) >= 2 * n_out - 1
    else None
)

results: list[dict] = []
# est-sfs writes one output line per input data line (including the
# monomorphic padding). We only care about the first len(site_positions)
# of them, which correspond to the polymorphic VCF sites we wrote first.
data_lines = (line for line in panc_path.open() if not line.startswith("0 "))
import itertools
for line, pos, alleles, major, minor in zip(
    itertools.islice(data_lines, len(site_positions)),
    site_positions, site_alleles, site_major, site_minor, strict=True,
):
    toks = line.split()
    # toks[0] is the 1-based site index, toks[1] is config_ind,
    # toks[2] is p_anc_major. The rest are tree-state probabilities.
    p_major_anc = float(toks[2])
    per_state = {s: 0.0 for s in STATES}
    per_state[major] = p_major_anc
    if minor is not None:
        per_state[minor] = 1.0 - p_major_anc
    map_allele = major if p_major_anc >= 0.5 else (minor or major)
    results.append({
        "chrom": "1",
        "pos": pos,
        "alleles": list(alleles),
        "map_allele": map_allele,
        "max_prob": max(per_state.values()),
        "per_state": per_state,
        "p_major_anc": p_major_anc,
        "major": major,
        "minor": minor,
    })

inference_meta = {
    "tool": "est-sfs",
    "binary": EST_SFS_BIN,
    "model": model_name,
    "n_out": n_out,
    "outgroup_sample_names": outgroup_names,
    "n_polymorphic_sites": len(results),
    "nrandom": NRANDOM,
    "fitted_rates": fitted_rates,
    "fitted_outgroup_divergence": fitted_outgroup_divergence,
    "inference_seconds": inference_seconds,
}

Path(out_json).parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w") as f:
    json.dump({"meta": inference_meta, "sites": results}, f, indent=2)

print(
    f"B3 est-sfs ({model_name}, n_out={n_out}, sim={sim}): "
    f"{len(results)} sites → {out_json}",
    file=sys.stderr,
)
