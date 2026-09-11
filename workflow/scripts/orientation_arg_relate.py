"""Build one Relate ARG under a single orientation scheme.

Relate consumes a tree sequence, so the ingroup tree sequence is cut down to
the shared site set and its VCF is written with the scheme's declared ancestral
allele as REF. The reference arm passes no orientation at all, leaving REF
arbitrary, which is what Relate's own default does.

Relate's ``-N`` is the haploid effective size, twice the Watterson diploid
estimate ``pi / (4 mu)`` taken over the ingroup panel.

Wildcard: ``{scheme}``. Needs ``RELATE_DIR`` pointing at a Relate install with
``bin/Relate`` (see ``workflow/scripts/install_relate.sh``). Run via
snakemake::

    snakemake -j 1 results/data/orientation_arg_relate_random5.trees
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    N_OUT_REFERENCE,
    OrientationPanel,
    RELATE_SEED,
    load_orientation,
    load_sites,
)


try:
    in_trees = str(snakemake.input.trees)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    in_map = str(snakemake.input.orientation)  # type: ignore[name-defined]
    out_trees = str(snakemake.output.trees)  # type: ignore[name-defined]
    scheme = str(snakemake.wildcards.scheme)  # type: ignore[name-defined]
    relate_seed = int(snakemake.params.relate_seed)  # type: ignore[name-defined]
    n_out_reference = int(snakemake.params.n_out_reference)  # type: ignore[name-defined]
    os.environ.setdefault("RELATE_DIR", str(snakemake.params.relate_dir))  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    scheme = "random5"
    in_trees = f"{DATA}/local_tree_genealogy_og_sim_20mb.trees"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    in_map = f"{DATA}/orientation_map_{scheme}.json"
    out_trees = f"{DATA}/orientation_arg_relate_{scheme}.trees"
    relate_seed = RELATE_SEED
    n_out_reference = N_OUT_REFERENCE
    os.environ.setdefault("RELATE_DIR", str(Path.home() / "opt" / "relate"))

import _relate  # noqa: E402


sites = load_sites(in_sites)
pos = sites["pos"]
alleles = sites["alleles"]
anc_idx, _ = load_orientation(in_map, pos.size)

panel = OrientationPanel(in_trees, in_meta,
                         n_out_reference=n_out_reference).load()
ts_relate = panel.relate_panel(pos)
if ts_relate.num_sites != pos.size:
    raise ValueError(f"Relate panel carries {ts_relate.num_sites} sites, "
                     f"site set has {pos.size}")
ne_dip = panel.ne_diploid()

orient = None
if scheme != "true_aa":
    orient = {int(round(float(pos[k]))): str(alleles[k, anc_idx[k]])
              for k in range(pos.size)}

try:
    arg, runtime = _relate.relate_infer(
        ts_relate, mu=panel.mu, rec_rate=panel.rec_rate, orient_by_pos=orient,
        seed=relate_seed, start=0, end=int(ts_relate.sequence_length),
        ne_diploid=ne_dip)
except subprocess.CalledProcessError as e:
    print("COMMAND FAILED:", e.cmd, flush=True)
    print("STDOUT:", e.stdout, flush=True)
    print("STDERR:", e.stderr, flush=True)
    raise

arg.dump(out_trees)
print(f"relate/{scheme}: num_trees={arg.num_trees} "
      f"num_samples={arg.num_samples} ne_diploid={ne_dip:.1f} "
      f"runtime={runtime:.1f}s", flush=True)
