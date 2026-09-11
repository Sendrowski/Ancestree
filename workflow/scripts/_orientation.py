"""Shared panel and site-set machinery for the orientation-damage experiment.

The experiment measures how a mis-declared ancestral allele degrades an
inferred genealogy. The ARG is built from the 20 ingroup haplotypes alone. The
outgroups enter only through the ancestral-allele call handed to the tools,
never through the panel. Genotypes are identical in every cell, so the only
thing that varies across orientation schemes is which of the two alleles is
declared ancestral.

Symbols used throughout:

- ``n_ing`` --- number of ingroup haplotypes in the scored panel (20 here).
- ``n_out_reference`` --- number of outgroups the 23-haplotype orientation
  panel carries, chosen by :func:`_robustness_common._evenly_spread_outgroups`
  so the split-depth span is preserved as the count varies.
- ``rf_max = 2 (n_ing - 2)`` --- the maximum Robinson-Foulds distance between
  two unrooted binary trees on ``n_ing`` leaves, the normaliser for every
  reported RF value (dimensionless, in ``[0, 1]`` after division).
- ``mu`` --- per-site per-generation mutation rate, read from the simulation
  metadata.
- ``rec_rate`` --- per-site per-generation recombination rate, read from the
  same place.
"""
import json
from pathlib import Path

import numpy as np
import tskit

from ancestree import STATES
from _robustness_common import _evenly_spread_outgroups

#: Orientation schemes, in the order they are reported. ``true_aa`` is the
#: reference arm every other arm is scored against.
SCHEMES = ("true_aa", "major_allele", "random5", "freq_biased5",
           "fixed_tree_n1", "fixed_tree_n3")

#: The mis-orientation arms, i.e. every scheme other than the reference.
MIS_ORIENTATION_SCHEMES = tuple(s for s in SCHEMES if s != "true_aa")

#: ARG tools scored, one shard per (tool, scheme).
TOOLS = ("tsinfer", "relate")

#: Upper edges, in bp, of the bins over the genomic distance from a site to the
#: nearest mis-oriented site. Bin 0 is the mis-oriented sites themselves
#: (distance exactly 0). Bin k covers ``(BIN_EDGES[k - 2], BIN_EDGES[k - 1]]``.
BIN_EDGES = (100.0, 300.0, 700.0, 1500.0, np.inf)

#: Human-readable name of every distance bin, aligned with ``BIN_EDGES``.
BIN_LABELS = ("0 (at a mis-oriented site)", "(0, 100]", "(100, 300]",
              "(300, 700]", "(700, 1500]", "> 1500")

#: Name of every bin over the number of mis-oriented sites falling inside the
#: local tree interval of the true-ancestral-allele ARG at the site.
INTERVAL_LABELS = ("0 mis-oriented sites in the interval", "exactly 1",
                   "exactly 2", "exactly 3", "4 to 6", "more than 6")

#: Number of ingroup haplotype pairs the pairwise-TMRCA metrics are pooled over.
N_PAIRS = 80

#: Seed of the generator that draws those pairs, so every shard scores the
#: identical pair set.
PAIR_SEED = 7

#: Seed handed to Relate, so the single genealogy it returns is reproducible.
RELATE_SEED = 1

#: Number of outgroups in the panel the shared site set is read off.
N_OUT_REFERENCE = 3

#: Fraction of shared sites the ``random5`` scheme mis-orients.
RANDOM_FLIP_FRACTION = 0.05

#: Seed of the ``random5`` draw.
RANDOM_FLIP_SEED = 2025

#: Bootstrap replicates behind every TMRCA standard error.
N_BOOTSTRAP = 0

#: Seed of the site-block bootstrap.
BOOTSTRAP_SEED = 20250902

#: Bootstrap replicates a comparison shard evaluates concurrently. Held small
#: because the shards themselves run concurrently.
BOOTSTRAP_THREADS = 4


def haplotype_pairs(n_ing: int, n_pairs: int = N_PAIRS,
                    seed: int = PAIR_SEED) -> list[tuple[int, int]]:
    """Draw the fixed set of ingroup haplotype pairs the TMRCA metrics pool over.

    :param n_ing: Number of ingroup haplotypes. Pairs are drawn from
        ``range(n_ing)`` and index the same haplotype in the truth and in every
        inferred tree sequence.
    :param n_pairs: Number of pairs to draw without replacement, capped at the
        ``n_ing (n_ing - 1) / 2`` available.
    :param seed: Seed of the ``numpy.random.default_rng`` behind the draw.
    :return: The drawn pairs as ``(i, j)`` with ``i < j``.
    """
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n_ing) for j in range(i + 1, n_ing)]
    idx = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)),
                     replace=False)
    return [all_pairs[k] for k in idx]


def distance_to_nearest(pos: np.ndarray, flip: np.ndarray) -> np.ndarray:
    """Genomic distance in bp from every site to the nearest mis-oriented site.

    :param pos: Site positions in bp, strictly ascending.
    :param flip: Boolean mask, ``True`` where the scheme declares the allele
        that is not the simulator-true ancestral state.
    :return: Distance in bp, exactly ``0`` at a mis-oriented site and ``inf``
        if no site is mis-oriented.
    """
    fpos = pos[flip]
    if fpos.size == 0:
        return np.full(pos.size, np.inf)
    ins = np.searchsorted(fpos, pos)
    left = np.where(ins > 0,
                    np.abs(pos - fpos[np.clip(ins - 1, 0, fpos.size - 1)]),
                    np.inf)
    right = np.where(ins < fpos.size,
                     np.abs(fpos[np.clip(ins, 0, fpos.size - 1)] - pos),
                     np.inf)
    return np.minimum(left, right)


def distance_bins(dist: np.ndarray) -> np.ndarray:
    """Assign every site to a distance bin index over :data:`BIN_LABELS`.

    :param dist: Distance in bp to the nearest mis-oriented site.
    :return: Integer bin index in ``[0, len(BIN_LABELS))``.
    """
    bin_idx = np.zeros(dist.size, dtype=int)
    for k in range(len(BIN_EDGES)):
        lo = 0.0 if k == 0 else BIN_EDGES[k - 1]
        bin_idx[(dist > lo) & (dist <= BIN_EDGES[k])] = k + 1
    bin_idx[dist == 0.0] = 0
    return bin_idx


def interval_flip_counts(breakpoints: np.ndarray, pos: np.ndarray,
                         flipped_pos: np.ndarray) -> tuple[np.ndarray, dict]:
    """Count the mis-oriented sites carried by each local tree interval.

    :param breakpoints: Tree-interval breakpoints of the reference ARG, in bp,
        as returned by ``TreeSequence.breakpoints(as_array=True)``.
    :param pos: Shared-site positions in bp, strictly ascending.
    :param flipped_pos: Positions of the mis-oriented sites, strictly ascending.
    :return: the per-site count of mis-oriented sites inside the half-open
        interval ``[left, right)`` of the tree containing the site, and the
        interval occupancy statistics of the whole ARG.
    """
    lo = np.searchsorted(flipped_pos, breakpoints[:-1], side="left")
    hi = np.searchsorted(flipped_pos, breakpoints[1:], side="left")
    per_interval = hi - lo
    iv = np.clip(np.searchsorted(breakpoints, pos, side="right") - 1,
                 0, breakpoints.size - 2)
    spans = np.diff(breakpoints)
    occupancy = {
        "n_intervals": int(per_interval.size),
        "mean_tree_span_bp": float(np.mean(spans)),
        "median_tree_span_bp": float(np.median(spans)),
        "n_intervals_with_0_mis_oriented_sites": int((per_interval == 0).sum()),
        "n_intervals_with_exactly_1": int((per_interval == 1).sum()),
        "n_intervals_with_more_than_1": int((per_interval > 1).sum()),
        "frac_intervals_with_0_mis_oriented_sites":
            float((per_interval == 0).mean()),
        "frac_intervals_with_exactly_1": float((per_interval == 1).mean()),
        "frac_intervals_with_more_than_1": float((per_interval > 1).mean()),
        "mean_mis_oriented_sites_per_interval": float(per_interval.mean()),
    }
    return per_interval[iv].astype(np.int32), occupancy


def interval_bins(n_flips: np.ndarray) -> np.ndarray:
    """Assign every site to a bin index over :data:`INTERVAL_LABELS`.

    :param n_flips: Per-site count of mis-oriented sites inside the local tree
        interval of the true-ancestral-allele ARG that contains the site.
    :return: Integer bin index in ``[0, len(INTERVAL_LABELS))``.
    """
    b = np.full(n_flips.size, 5, dtype=np.int8)
    b[n_flips <= 3] = n_flips[n_flips <= 3]
    b[(n_flips >= 4) & (n_flips <= 6)] = 4
    return b


class OrientationPanel:
    """The ingroup panel and shared site set the orientation arms are built on.

    The 23-haplotype panel (ingroup plus ``n_out_reference`` outgroups) defines
    which sites exist. The 20-haplotype ingroup simplification of it is both the
    scoring truth and the panel handed to the ARG tools, so a haplotype index
    means the same thing in the truth and in every inferred tree sequence.

    :param trees_path: Simulated tree sequence with the outgroup ladder.
    :param meta_path: Companion simulation metadata JSON.
    :param n_out_reference: Outgroups kept in the orientation panel.
    """

    def __init__(self, trees_path, meta_path,
                 n_out_reference: int = N_OUT_REFERENCE):
        self.trees_path = Path(trees_path)
        self.meta_path = Path(meta_path)
        self.n_out_reference = int(n_out_reference)
        self.meta: dict = {}
        self.ingroup_names: list[str] = []
        self.outgroup_names: list[str] = []
        self.truth_full: tskit.TreeSequence = None
        self.truth_ts: tskit.TreeSequence = None

    def load(self) -> "OrientationPanel":
        """Read the simulation, build the orientation panel and the ingroup truth.

        :return: This instance, so the call chains off the constructor.
        """
        self.meta = json.loads(self.meta_path.read_text())
        self.ingroup_names = list(self.meta["ingroup_names"])
        ts = tskit.load(str(self.trees_path))
        node_for_name = {f"tsk_{ind.id}": int(ind.nodes[0])
                         for ind in ts.individuals()}
        by_pop = self.meta["outgroups_by_pop"]
        ordered = sorted(by_pop.keys(), key=lambda s: int(s.split("_")[1]))
        all_og = [nm for p in ordered for nm in by_pop[p]]
        self.outgroup_names = _evenly_spread_outgroups(all_og,
                                                       self.n_out_reference)
        keep = self.ingroup_names + self.outgroup_names
        self.truth_full = ts.simplify(
            samples=[node_for_name[x] for x in keep], filter_sites=False)
        self.truth_ts = self.truth_full.simplify(
            samples=list(range(self.n_ingroup)), filter_sites=False)
        return self

    @property
    def n_ingroup(self) -> int:
        """Number of ingroup haplotypes in the scored panel."""
        return len(self.ingroup_names)

    @property
    def mu(self) -> float:
        """Per-site per-generation mutation rate of the simulation."""
        return float(self.meta["mu"])

    @property
    def rec_rate(self) -> float:
        """Per-site per-generation recombination rate of the simulation."""
        return float(self.meta["rec_rate"])

    @property
    def sequence_length(self) -> float:
        """Length of the simulated sequence in bp."""
        return float(self.truth_ts.sequence_length)

    @property
    def rf_max(self) -> float:
        """Maximum Robinson-Foulds distance on ``n_ingroup`` unrooted leaves."""
        return 2.0 * (self.n_ingroup - 2)

    def shared_sites(self) -> dict:
        """Collect the shared site set: biallelic, ACGT, ingroup-segregating.

        ``alleles[:, 0]`` is the simulator-true ancestral state, so the derived
        allele count is the number of ingroup haplotypes carrying
        ``alleles[:, 1]``.

        :return: ``pos`` (bp, ascending float), ``genotypes`` (``int8``, one
            allele index per ingroup haplotype), ``alleles`` (``<U1``, two
            nucleotides per site) and ``daf`` (``int16`` derived-allele count in
            ``1..n_ingroup - 1``).
        """
        n_ing = self.n_ingroup
        positions, genotypes, alleles, daf = [], [], [], []
        for v in self.truth_full.variants():
            al = v.alleles
            if len(al) != 2 or any(a not in STATES for a in al):
                continue
            g_ing = np.asarray(v.genotypes, dtype=np.int8)[:n_ing]
            if len(set(g_ing.tolist())) < 2:
                continue
            positions.append(float(v.site.position))
            genotypes.append(g_ing)
            alleles.append((al[0], al[1]))
            daf.append(int(np.bincount(g_ing, minlength=2)[1]))
        return {
            "pos": np.asarray(positions, dtype=float),
            "genotypes": np.asarray(genotypes, dtype=np.int8),
            "alleles": np.asarray(alleles, dtype="<U1"),
            "daf": np.asarray(daf, dtype=np.int16),
        }

    def relate_panel(self, pos: np.ndarray) -> tskit.TreeSequence:
        """Ingroup tree sequence restricted to the shared site set.

        Relate consumes a tree sequence and emits one VCF record per site, so
        the site table must hold exactly the shared sites and no others.

        :param pos: Shared-site positions in bp.
        :return: :attr:`truth_ts` with every other site deleted.
        """
        shared = {int(round(float(p))) for p in pos}
        drop = [s.id for s in self.truth_ts.sites()
                if int(round(s.position)) not in shared]
        return self.truth_ts.delete_sites(drop)

    def ne_diploid(self) -> float:
        """Watterson diploid effective size of the ingroup, ``pi / (4 mu)``.

        Relate takes the haploid effective size, i.e. twice this value.

        :return: Diploid effective population size in individuals, floored at 1.
        """
        pi = float(np.asarray(self.truth_ts.diversity(mode="site")).ravel()[0])
        return max(pi / (4.0 * self.mu), 1.0)


def load_sites(path) -> dict:
    """Read the shared site set written by ``orientation_sites.py``.

    :param path: Path to ``orientation_sites.npz``.
    :return: The ``pos`` / ``genotypes`` / ``alleles`` / ``daf`` arrays.
    """
    with np.load(path) as d:
        return {k: d[k] for k in ("pos", "genotypes", "alleles", "daf")}


def load_orientation(path, n_sites: int) -> tuple[np.ndarray, dict]:
    """Read an orientation map written by ``orientation_map.py``.

    :param path: Path to ``orientation_map_{scheme}.json``.
    :param n_sites: Number of shared sites the index is expected to cover.
    :return: an ``int8`` index into the site's allele pair naming the declared
        ancestral allele, aligned with the shared-site order, and the map's
        summary fields with the position map itself removed.
    """
    d = json.loads(Path(path).read_text())
    idx = np.asarray(d["ancestral_allele_index"], dtype=np.int8)
    if idx.size != n_sites:
        raise ValueError(
            f"{path} covers {idx.size} sites, site set has {n_sites}")
    return idx, {k: v for k, v in d.items()
                 if k not in ("ancestral_allele_index",
                              "ancestral_allele_by_position")}
