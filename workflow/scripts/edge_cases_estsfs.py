"""est-sfs 2.04 on the four edge-case classes with their three outgroups.

est-sfs fits its branch rates to the data it is given, so the four classes are
embedded in a background of sites drawn under JC69 on the same quartet-with-
ladder tree (:func:`_edge_cases.simulate_quartet_ladder`), and est-sfs is run
under JC69 (``model 0``). Its per-site output column 3, the posterior that the
ingroup-major allele is ancestral at the ingroup node, is recorded for the four
class rows. The major and minor alleles are the ones est-sfs.c assigns: the
first state of highest ingroup count in A, C, G, T order, and the first other
state of highest count. est-sfs writes 1 for a site the ingroup has fixed,
which is a default rather than a posterior.

Output: ``results/reports/edge_cases_estsfs.json``.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _edge_cases import CLASSES, simulate_quartet_ladder  # noqa: E402

STATES = "ACGT"

try:
    out_json = snakemake.output.json  # type: ignore[name-defined]
    n_background = int(snakemake.params.n_background)  # type: ignore[name-defined]
    seed = int(snakemake.params.seed)  # type: ignore[name-defined]
    nrandom = int(snakemake.params.nrandom)  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/edge_cases_estsfs.json"
    n_background = 100_000
    seed = 2025
    nrandom = 5

EST_SFS_BIN = (os.environ.get("EST_SFS_BIN") or shutil.which("est-sfs")
               or "external/est-sfs/est-sfs")
if not Path(EST_SFS_BIN).exists():
    raise FileNotFoundError(
        "est-sfs not found: run under workflow/envs/estsfs.yml or set "
        "EST_SFS_BIN=/path/to/est-sfs.")


def counts_line(ingroup: list[int], outgroup: list[int]) -> str:
    """One est-sfs input line: ingroup counts then one count vector per outgroup.

    :param ingroup: Ingroup state indices into ``ACGT``.
    :param outgroup: Outgroup state indices, closest first.
    :return: The line, without a newline.
    """
    ing = np.bincount(ingroup, minlength=4)
    fields = [",".join(map(str, ing))]
    for s in outgroup:
        one = np.zeros(4, dtype=int)
        one[s] = 1
        fields.append(",".join(map(str, one)))
    return " ".join(fields)


def major_minor(ingroup: list[int]) -> tuple[str, str]:
    """Major and minor ingroup alleles as est-sfs.c's assign_observed_type picks
    them.

    :param ingroup: Ingroup state indices into ``ACGT``.
    :return: ``(major, minor)`` nucleotides.
    """
    counts = np.bincount(ingroup, minlength=4)
    major = int(np.argmax(counts))
    rest = counts.copy()
    rest[major] = -1
    return STATES[major], STATES[int(np.argmax(rest))]


rows = []
for c in CLASSES:
    ing = [STATES.index(c["ingroup"][k]) for k in sorted(c["ingroup"])]
    out = [STATES.index(c["outgroup"][k]) for k in ("o1", "o2", "o3")]
    rows.append((c["name"], ing, out))
bg_ing, bg_out = simulate_quartet_ladder(n_background, seed)

work = Path(tempfile.mkdtemp(prefix="edge_estsfs_"))
data = work / "data.txt"
with data.open("w", encoding="utf-8") as f:
    for _, ing, out in rows:
        f.write(counts_line(ing, out) + "\n")
    for k in range(n_background):
        f.write(counts_line(list(bg_ing[k]), list(bg_out[k])) + "\n")
(work / "config.txt").write_text(f"n_outgroup 3\nmodel 0\nnrandom {nrandom}\n",
                                 encoding="utf-8")
(work / "seed.txt").write_text(f"{seed}\n", encoding="utf-8")
cmd = [EST_SFS_BIN, str(work / "config.txt"), str(data), str(work / "seed.txt"),
       str(work / "sfs.txt"), str(work / "panc.txt")]
proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
if proc.returncode != 0:
    sys.stderr.write(proc.stdout + proc.stderr)
    raise RuntimeError(f"est-sfs exited with status {proc.returncode}")

p_major = {}
for line in (work / "panc.txt").read_text(encoding="utf-8").splitlines():
    tok = line.split()
    if not tok or tok[0] == "0":
        continue
    site = int(tok[0])
    if site > len(rows):
        break
    p_major[site] = float(tok[2])
rates = next(line for line in (work / "panc.txt").read_text(encoding="utf-8").splitlines()
             if line.startswith("0 Rates:"))

report = {
    "est_sfs": str(EST_SFS_BIN), "model": "JC69", "nrandom": nrandom,
    "n_background_sites": n_background, "seed": seed, "rates": rates[2:],
    "classes": {},
}
for i, (name, ing, _) in enumerate(rows, start=1):
    major, minor = major_minor(ing)
    report["classes"][name] = {"p_major_ancestral": p_major[i],
                               "major": major, "minor": minor}
out = Path(out_json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote {out}: {json.dumps(report['classes'])}")
