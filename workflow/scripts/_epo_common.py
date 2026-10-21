"""Shared pieces of the Ensembl EPO comparison and its matched simulation.

- :data:`SPECIES`, :data:`PAN` and :func:`epo_tree`: the nine outgroups of
  the Ensembl release-106 10-primate EPO alignment and its species tree.
- :func:`summarise`: the unfolded spectrum of a sample polarised by a per-site
  distribution over the ancestral base, and the number of sites the sample has
  fixed for a derived base, from MAP estimates and from posterior weights.
- :func:`sites_from_rows`, :func:`local_tree_posteriors` and
  :func:`posterior_matrix`: one local-tree run as a row-aligned posterior
  matrix.
- :func:`parse_run` and :func:`interval_rate_map`: the run name and the
  per-class mutation map of a local-tree run.
- :func:`fitted_scale`: the ML multiplier of a fixed tree's branch lengths.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from ancestree import STATES, BaseComposition, FixedTree, HKY, Site
from ancestree.likelihood import Likelihood
from ancestree.local_tree_inference import LocalTreeInference

#: The outgroups of the 10-primate EPO alignment, in the order of the
#: extracted alignment columns.
SPECIES = [
    "pan_troglodytes", "pan_paniscus", "gorilla_gorilla", "pongo_abelii",
    "nomascus_leucogenys", "macaca_mulatta", "macaca_fascicularis",
    "chlorocebus_sabaeus", "microcebus_murinus",
]
#: The sister clade of human. Its common ancestor with human is the node EPO
#: reports for human.
PAN = ["pan_troglodytes", "pan_paniscus"]
#: Outgroup subsets a run may be restricted to, by size.
OUTGROUP_SUBSETS = {
    9: SPECIES,
    3: ["pan_troglodytes", "gorilla_gorilla", "pongo_abelii"],
    1: ["pan_troglodytes"],
}
#: One outgroup per nested lineage, nearest first, for the fitted outgroup ladder.
LADDER = [
    "pan_troglodytes", "gorilla_gorilla", "pongo_abelii", "nomascus_leucogenys",
    "macaca_mulatta", "microcebus_murinus",
]
#: Name of the human reference tip.
REFERENCE = "homo_sapiens"


def epo_tree(human: str) -> str:
    """The species tree of the 10-primate EPO alignment as printed in its
    README, in substitutions per site.

    :param human: Name of the human tip.
    :return: Newick string.
    """
    return (
        "(((((((pan_troglodytes:0.00221173,pan_paniscus:0.00332827):0.00431379,"
        f"{human}:0.00659621):0.00185825,gorilla_gorilla:0.00857049):0.00847976,"
        "pongo_abelii:0.016068):0.00267331,nomascus_leucogenys:0.0196815):0.0109481,"
        "((macaca_mulatta:0.00254698,macaca_fascicularis:0.00312302):0.0090058,"
        "chlorocebus_sabaeus:0.0117108):0.01846908):0.0454214526,"
        "microcebus_murinus:0.07593699);"
    )


def summarise(post: np.ndarray, counts: np.ndarray, classes: dict[str, np.ndarray],
              n: int, called: "np.ndarray | None" = None) -> dict:
    """Spectrum and divergence of a sample under one polarisation.

    A site polymorphic in the sample with exactly two bases adds, for each of
    them, its ancestral probability to the bin of the other's count. A site
    the sample has fixed counts as divergent by the probability that the
    ancestral base differs from the fixed one.

    :param post: ``(n_sites, 4)`` probability of each base being ancestral.
    :param counts: ``(n_sites, 4)`` copies of each base in the sample.
    :param classes: Site-class name to boolean row mask.
    :param n: Sample size.
    :param called: Rows that carry a polarisation, all by default.
    :return: Per class, the posterior-weighted and MAP spectra over derived
        counts ``0..n`` (polymorphic bins only) and the two divergence counts.
    """
    present = counts > 0
    n_alleles = present.sum(1)
    if called is None:
        called = np.ones(len(post), dtype=bool)
    best = post.argmax(1)
    rows = np.arange(len(post))
    fixed = (n_alleles == 1) & called
    fixed_base = present.argmax(1)
    polymorphic = (n_alleles == 2) & called
    out = {}
    for name, mask in classes.items():
        weighted = np.zeros(n + 1)
        committed = np.zeros(n + 1)
        m = polymorphic & mask
        for base in range(4):
            k = m & present[:, base]
            np.add.at(weighted, n - counts[k, base], post[k, base])
        k = m & present[rows, best]
        np.add.at(committed, n - counts[k, best[k]], 1.0)
        f = fixed & mask
        out[name] = dict(
            n_polymorphic=int(m.sum()), n_fixed=int(f.sum()),
            spectrum_weighted=weighted.tolist(), spectrum_map=committed.tolist(),
            divergence_weighted=float((1.0 - post[f, fixed_base[f]]).sum()),
            divergence_map=int((best[f] != fixed_base[f]).sum()),
        )
    return out


def sites_from_rows(positions: np.ndarray, alleles: np.ndarray, names: list[str],
                    ) -> tuple[list[Site], np.ndarray]:
    """Sites of a panel that vary within it.

    :param positions: ``(n_rows,)`` site positions.
    :param alleles: ``(n_rows, n_tips)`` base index per tip, ``-1`` missing.
    :param names: Tip names, one per column.
    :return: The sites, and the row index of each.
    """
    sites, rows = [], []
    for k in range(len(positions)):
        row = alleles[k]
        seen = np.unique(row[row >= 0])
        if seen.size < 2:
            continue
        sites.append(Site(
            chrom="1", pos=int(positions[k]), alleles=tuple(STATES[i] for i in seen),
            tip_alleles={n: (STATES[a] if a >= 0 else None) for n, a in zip(names, row)}))
        rows.append(k)
    return sites, np.asarray(rows, dtype=np.int64)


def fitted_scale(columns: np.ndarray, weights: np.ndarray, newick: str, kappa: float,
                 pi, n_sites: int, guess: float) -> float:
    """ML multiplier of the branch lengths of ``newick`` under HKY, for
    weighted columns of the human tip and the outgroups.

    :param columns: ``(K, 1 + len(SPECIES))`` base index of :data:`REFERENCE`
        and each outgroup, ``-1`` where unaligned.
    :param weights: ``(K,)`` site count of each column.
    :param newick: Tree whose branch lengths are scaled.
    :param kappa: HKY transition/transversion rate ratio.
    :param pi: Stationary frequencies of A, C, G, T, also the root prior.
    :param n_sites: Number of sites the composition describes.
    :param guess: Centre of the search interval, which spans a factor of 20
        either side.
    :return: The fitted multiplier.
    """
    names = [REFERENCE] + SPECIES
    sites = [Site(chrom="1", pos=i + 1,
                  alleles=tuple(dict.fromkeys(STATES[a] for a in col if a >= 0)),
                  tip_alleles={n: STATES[a] if a >= 0 else None for n, a in zip(names, col)})
             for i, col in enumerate(np.asarray(columns, int))]
    w = np.asarray(weights, float)
    engine = Likelihood(HKY(kappa=kappa), base_composition=BaseComposition.from_n_target_sites(
        n_sites, stationary=np.asarray(pi)))
    log_pi = np.log(pi)

    def negative_log_likelihood(log_scale):
        tree = FixedTree.from_newick(newick, time_scale=float(np.exp(log_scale)))
        return -float((w * logsumexp(engine.log_likelihoods(tree, sites) + log_pi, axis=1)).sum())

    best = minimize_scalar(negative_log_likelihood, bounds=(np.log(guess / 20), np.log(guess * 20)),
                           method="bounded", options={"xatol": 1e-4})
    return float(np.exp(best.x))


def posterior_matrix(n_rows: int, rows: np.ndarray, posteriors,
                     default: "np.ndarray | None" = None) -> np.ndarray:
    """Row-aligned posterior matrix of a run over a subset of the rows.

    :param n_rows: Number of rows in total.
    :param rows: Row index of each scored site.
    :param posteriors: The run's posteriors, one per scored site.
    :param default: ``(n_rows,)`` base that takes probability one at a row
        the run did not score, where its panel carries that base alone.
    :return: ``(n_rows, 4)`` matrix, all-zero at an unscored row without a
        default.
    """
    out = np.zeros((n_rows, 4))
    if default is not None:
        out[np.arange(n_rows), default] = 1.0
    if len(rows):
        out[rows] = np.array([[float(p[s]) for s in STATES] for p in posteriors])
    return out


def local_tree_posteriors(sites: list[Site], names: list[str], ingroup: list[str],
                          **kwargs) -> list:
    """Posteriors of one local-tree run at the ancestor of ``ingroup``.

    :param sites: The panel's sites.
    :param names: Panel tip names.
    :param ingroup: Tips whose common ancestor is reported.
    :param kwargs: Further :class:`~ancestree.local_tree_inference.LocalTreeInference`
        arguments.
    :return: One posterior per site, in the order of ``sites``.
    """
    inference = LocalTreeInference(
        sites, sample_names=names, ingroup_samples=ingroup,
        outgroup_samples=[n for n in names if n not in ingroup], progress=False,
        **kwargs)
    by_pos = {int(s.pos): p for s, p in inference.infer()}
    return [by_pos[int(s.pos)] for s in sites]



def parse_run(run: str) -> dict:
    """Decode a local-tree run name ``<panel>_<node>_<rate>_out<n>``.

    :param run: For example ``sample_mrca_class_rate_out9``.
    :return: ``panel`` (``"sample"`` or ``"reference"``), ``node`` (``"mrca"``
        for the ancestor of the panel's human tips, ``"node"`` for their
        ancestor with the Pan species present), ``class_rate`` and
        ``outgroups``.
    """
    panel, node, rate, _, out = run.split("_")
    return dict(panel=panel, node=node, class_rate=rate == "class",
                outgroups=OUTGROUP_SUBSETS[int(out.removeprefix("out"))])


def interval_rate_map(starts: np.ndarray, ends: np.ndarray, length: int,
                      inside: float, outside: float):
    """Rate map with one rate inside half-open intervals and another outside.

    :param starts: Interval starts, ascending and non-overlapping.
    :param ends: Interval ends.
    :param length: Sequence length the map covers.
    :param inside: Rate within the intervals.
    :param outside: Rate elsewhere.
    :return: :class:`msprime.RateMap`.
    """
    import msprime

    position = np.unique(np.concatenate([[0], starts, ends, [length]])).astype(float)
    mid = 0.5 * (position[:-1] + position[1:])
    # An interval boundary pair (start, end) brackets every midpoint inside it.
    within = np.searchsorted(starts, mid, side="right") > np.searchsorted(ends, mid, side="right")
    return msprime.RateMap(position=position, rate=np.where(within, inside, outside))
