"""Kernel-level posteriors for the appendix edge-case comparison (Table A1).

Four hand-built site classes that fall outside the biallelic,
parsimony-resolvable, ingroup-polymorphic intersection the main benchmarks
restrict to, each evaluated WITHOUT outgroups (the \\texttt{PolarBEAR} /
ingroup-only regime) and then WITH three outgroups attached (the EST-SFS /
fixed-tree regime). Every (class, regime) is evaluated on the *same* tree by:

- \\texttt{Ancestree}: the Felsenstein sum-product marginal (full 4-state
  posterior over A,C,G,T), via :class:`~ancestree.inference.Likelihood`.
- \\texttt{PolarBEAR}: its own ML (max-product) kernel
  (``polarize_ML.calculate_node_info_tskit_tree``). PolarBEAR's *pipeline*
  builds ingroup-only trees, but its *kernel* runs on any tree it is handed,
  so it has a value in the outgroup regime too.

EST-SFS exposes only the scalar P^maj_anc (posterior that the ingroup-major
allele is ancestral) and operates only with outgroups. Those values come
from the est-sfs tool and are carried here as constants.

Output: ``results/reports/edge_cases.json``.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

try:
    out_json = snakemake.output.json  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/edge_cases.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ancestree import JC69, Likelihood, STATE_INDEX  # noqa: E402
from ancestree.sites import Site  # noqa: E402
from ancestree.trees import TskitLocalTree  # noqa: E402

# PolarBEAR's ML kernel (max-product) lives in the vendored source tree.
_PB = "external/polarbear/code/polarize"
sys.path.insert(0, _PB)
import polarize_ML  # type: ignore[import-not-found]  # noqa: E402

# Branch lengths are in substitutions/site; Ancestree reads them under JC69
# and PolarBEAR under theta=1 (so PolarBEAR's branch_len*theta equals the
# same expected number of substitutions). Both kernels therefore see one
# shared tree with one shared per-branch divergence.
PB_THETA = 1.0

# --- the four site classes ------------------------------------------------
# Ingroup is a balanced quartet ((i0,i1),(i2,i3)) except the polytomy class,
# which is a 4-way star. Outgroups attach as a nested ladder (o1 closest).
# Branches are short (subs/site) so the tree is unsaturated and the deeper
# outgroups stay informative for the root state.
IT = 0.015  # ingroup tip branch
IC = 0.015  # ingroup cherry -> ingroup-MRCA
RUNG = 0.01  # one ladder step between successive outgroup joins
S = IT + IC  # ingroup-MRCA depth (also the star tip branch)


def _ladder(ingroup_block: str) -> str:
    """Attach o1..o3 as a nested ladder above ``ingroup_block`` (whose MRCA
    sits at depth S). Ultrametric, all leaves at depth ``S + 3*RUNG``."""
    return (f"((({ingroup_block}:{RUNG},o1:{S+RUNG}):{RUNG},"
            f"o2:{S+2*RUNG}):{RUNG},o3:{S+3*RUNG});")


def _quartet(outgroups: bool) -> str:
    block = f"((i0:{IT},i1:{IT}):{IC},(i2:{IT},i3:{IT}):{IC})"
    return _ladder(block) if outgroups else block + ";"


def _star(outgroups: bool) -> str:
    block = f"(i0:{S},i1:{S},i2:{S},i3:{S})"
    return _ladder(block) if outgroups else block + ";"


CLASSES = [
    {"name": "Homoplastic biallelic", "topo": _quartet,
     "ingroup": {"i0": "A", "i1": "G", "i2": "A", "i3": "G"},
     "outgroup": {"o1": "A", "o2": "A", "o3": "A"},
     "estsfs": "0.00"},
    {"name": "Tri-allelic", "topo": _quartet,
     "ingroup": {"i0": "A", "i1": "A", "i2": "C", "i3": "G"},
     "outgroup": {"o1": "T", "o2": "T", "o3": "T"},
     "estsfs": "0.00"},
    {"name": "Ingroup-monomorphic", "topo": _quartet,
     "ingroup": {"i0": "A", "i1": "A", "i2": "A", "i3": "A"},
     "outgroup": {"o1": "G", "o2": "G", "o3": "G"},
     "estsfs": "1.00"},
    {"name": "4-way polytomy", "topo": _star,
     "ingroup": {"i0": "A", "i1": "A", "i2": "A", "i3": "G"},
     "outgroup": {"o1": "A", "o2": "A", "o3": "A"},
     "estsfs": "1.00"},
]


def _ancestree_post(nwk: str, tip_alleles: dict[str, str]) -> np.ndarray:
    tree = TskitLocalTree.from_newick(nwk)
    site = Site(chrom="1", pos=1, alleles=tuple(sorted(set(tip_alleles.values()))),
                tip_alleles=dict(tip_alleles))
    log_L = Likelihood(JC69()).log_likelihoods(tree, [site])[0]
    return np.exp(log_L - logsumexp(log_L))


def _polarbear_post(nwk: str, tip_alleles: dict[str, str]) -> np.ndarray:
    tree = TskitLocalTree.from_newick(nwk)
    tk = tree.tskit_tree
    n_leaves = tk.num_samples()
    geno = np.zeros(n_leaves, dtype=np.int64)
    for name, allele in tip_alleles.items():
        geno[tree._sample_to_node[name]] = STATE_INDEX[allele.upper()]
    root_info = polarize_ML.calculate_node_info_tskit_tree(
        tk, tk.root, lik_matrix=np.array([]),
        nuc_type_leafs=geno, theta=PB_THETA,
    )
    root_info = np.asarray(root_info, dtype=float)
    return np.exp(root_info - logsumexp(root_info))


rows = []
for c in CLASSES:
    ing = c["ingroup"]
    og = c["outgroup"]
    for outgroups in (False, True):
        nwk = c["topo"](outgroups)
        tips = dict(ing)
        if outgroups:
            tips.update(og)
        rows.append({
            "class": c["name"],
            "ingroup": dict(ing),
            "outgroups": dict(og) if outgroups else None,
            "ancestree": _ancestree_post(nwk, tips).tolist(),
            "polarbear": _polarbear_post(nwk, tips).tolist(),
            "estsfs": c["estsfs"] if outgroups else None,
        })

report = {"pb_theta": PB_THETA, "rows": rows}
out = Path(out_json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"wrote {out}")
for r in rows:
    print(f"  {r['class']:24s} outgroups={r['outgroups'] is not None}  "
          f"A={[round(x, 2) for x in r['ancestree']]}  "
          f"PB={[round(x, 2) for x in r['polarbear']]}")
