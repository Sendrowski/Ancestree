"""Kernel-level posteriors for the appendix edge-case comparison (Table A1).

Four hand-built site classes that fall outside the biallelic,
parsimony-resolvable, ingroup-polymorphic intersection the main benchmarks
restrict to, each evaluated WITHOUT outgroups (the \\texttt{PolarBEAR} /
ingroup-only regime) and then WITH three outgroups attached (the EST-SFS /
fixed-tree regime). Every (class, regime) is evaluated on the *same* tree by:

- \\texttt{Ancestree}: the Felsenstein sum-product marginal (full 4-state
  posterior over A,C,G,T), via :class:`~ancestree.inference.Likelihood`, read
  at the ingroup MRCA for an ingroup-polymorphic class and at the root for the
  ingroup-monomorphic one.
- \\texttt{PolarBEAR}: its own ML (max-product) kernel
  (``polarize_ML.calculate_node_info_tskit_tree``). PolarBEAR's *pipeline*
  builds ingroup-only trees, but its *kernel* runs on any tree it is handed,
  so it has a value in the outgroup regime too.

EST-SFS exposes only the scalar P^maj_anc (posterior that the ingroup-major
allele is ancestral) and operates only with outgroups. Those values are read
from the est-sfs 2.04 run of ``edge_cases_estsfs.py``.

Output: ``results/reports/edge_cases.json``.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

try:
    out_json = snakemake.output.json  # type: ignore[name-defined]
    in_estsfs = snakemake.input.estsfs  # type: ignore[name-defined]
except NameError:
    out_json = "results/reports/edge_cases.json"
    in_estsfs = "results/reports/edge_cases_estsfs.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ancestree import JC69, Likelihood, STATE_INDEX  # noqa: E402
from ancestree.sites import Site  # noqa: E402
from ancestree.trees import RerootedTree, TskitLocalTree  # noqa: E402
from _edge_cases import CLASSES, newick  # noqa: E402

# PolarBEAR's ML kernel (max-product) lives in the vendored source tree.
_PB = "external/polarbear/code/polarize"
sys.path.insert(0, _PB)
import polarize_ML  # type: ignore[import-not-found]  # noqa: E402

# Branch lengths are in substitutions/site; Ancestree reads them under JC69
# and PolarBEAR under theta=1 (so PolarBEAR's branch_len*theta equals the
# same expected number of substitutions). Both kernels therefore see one
# shared tree with one shared per-branch divergence.
PB_THETA = 1.0

def _ancestree_post(nwk: str, tip_alleles: dict[str, str],
                    at_ingroup_mrca: bool) -> np.ndarray:
    tree = TskitLocalTree.from_newick(nwk)
    if at_ingroup_mrca:
        ingroup = [tree.tip_for_sample(s) for s in tip_alleles if s.startswith("i")]
        tree = RerootedTree(tree, tree.tskit_tree.mrca(*ingroup), 0.0)
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


estsfs = json.loads(Path(in_estsfs).read_text(encoding="utf-8"))["classes"]
rows = []
for c in CLASSES:
    ing = c["ingroup"]
    og = c["outgroup"]
    for outgroups in (False, True):
        nwk = newick(c["star"], outgroups)
        tips = dict(ing)
        if outgroups:
            tips.update(og)
        p_major = estsfs[c["name"]]["p_major_ancestral"]
        rows.append({
            "class": c["name"],
            "ingroup": dict(ing),
            "outgroups": dict(og) if outgroups else None,
            "ancestree": _ancestree_post(
                nwk, tips, at_ingroup_mrca=c["polymorphic"]).tolist(),
            "polarbear": _polarbear_post(nwk, tips).tolist(),
            "estsfs": f"{p_major:.2f}" if outgroups else None,
        })

report = {"pb_theta": PB_THETA, "rows": rows}
out = Path(out_json)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"wrote {out}")
for r in rows:
    print(f"  {r['class']:24s} outgroups={r['outgroups'] is not None}  "
          f"A={[round(x, 2) for x in r['ancestree']]}  "
          f"PB={[round(x, 2) for x in r['polarbear']]}")
