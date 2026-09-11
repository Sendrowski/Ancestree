"""Score one (tool, scheme) ARG against the simulator-true local trees, per site.

At every shared site the inferred local tree and the true local tree at that
position are compared and three quantities are recorded:

- ``rf_norm`` --- Robinson-Foulds distance between the two unrooted topologies
  divided by ``rf_max = 2 (n_ing - 2) = 36``. Dimensionless in ``[0, 1]``, where 0
  is an identical topology and 1 shares no split.
- ``kc_lambda1`` --- Kendall-Colijn distance at ``lambda = 1``, the Euclidean
  norm of the vector of pairwise MRCA branch-length depths, so it carries the
  tree sequence's time units (generations).
- ``tmrca_true`` / ``tmrca_inferred`` --- the pairwise time to the most recent
  common ancestor in generations, for each of the ``N_PAIRS = 80`` fixed
  ingroup haplotype pairs, laid out as ``(n_sites, n_pairs)`` and flattened.

A site whose inferred or true local tree has more than one root is skipped:
its ``rf_norm`` and ``kc_lambda1`` are ``nan`` and its whole TMRCA row is
``nan``. Pairing two arms and dropping non-positive TMRCA values is left to the
report, which sees both arms at once.

Wildcards: ``{tool}``, ``{scheme}``. Run via snakemake::

    snakemake -j 1 results/data/orientation_persite_relate_random5.npz
"""
import os
import sys
import time

import numpy as np
import tskit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    N_OUT_REFERENCE,
    N_PAIRS,
    OrientationPanel,
    PAIR_SEED,
    haplotype_pairs,
    load_sites,
)


class PerSiteScorer:
    """Walk the shared sites, comparing one inferred ARG with the truth.

    :param truth_ts: The simulator's tree sequence on the ingroup panel.
    :param arm_ts: The inferred ARG on the same panel and haplotype order.
    :param pos: Shared-site positions in bp, strictly ascending.
    :param pairs: Ingroup haplotype pairs the TMRCA vectors are taken over.
    :param rf_max: Normaliser for the Robinson-Foulds distance.
    :param log_every: Sites between progress lines.
    """

    def __init__(self, truth_ts, arm_ts, pos, pairs, rf_max: float,
                 log_every: int = 20000):
        self.truth_ts = truth_ts
        self.arm_ts = arm_ts
        self.pos = pos
        self.pairs = list(pairs)
        self.rf_max = float(rf_max)
        self.log_every = int(log_every)

    @property
    def n_sites(self) -> int:
        """Number of shared sites scored."""
        return int(self.pos.size)

    @property
    def n_pairs(self) -> int:
        """Number of haplotype pairs the TMRCA vectors span."""
        return len(self.pairs)

    def _tmrca_vec(self, tree) -> np.ndarray:
        """Pairwise TMRCA in generations over :attr:`pairs` for one tree."""
        return np.fromiter((tree.tmrca(i, j) for i, j in self.pairs),
                           dtype=float, count=self.n_pairs)

    def run(self) -> dict:
        """Score every shared site.

        :return: ``rf_norm``, ``kc_lambda1``, the flattened ``tmrca_true`` and
            ``tmrca_inferred`` grids, the ARG's tree ``breakpoints``, and the
            counts of trees and of sites skipped for a multi-rooted tree.
        """
        n_sites, n_pairs = self.n_sites, self.n_pairs
        rf = np.full(n_sites, np.nan)
        kc = np.full(n_sites, np.nan)
        tm_true = np.full((n_sites, n_pairs), np.nan)
        tm_inf = np.full((n_sites, n_pairs), np.nan)

        t_arm = tskit.Tree(self.arm_ts, sample_lists=True)
        t_true = tskit.Tree(self.truth_ts, sample_lists=True)
        t_arm.first()
        t_true.first()

        dist_key = None
        rf_val = kc_val = np.nan
        true_key = arm_key = None
        v_true = v_arm = None
        n_skip = 0
        t0 = time.perf_counter()
        for i in range(n_sites):
            p = self.pos[i]
            t_arm.seek(p)
            t_true.seek(p)
            if t_arm.num_roots != 1 or t_true.num_roots != 1:
                n_skip += 1
                continue
            ia, it = t_arm.index, t_true.index

            if (ia, it) != dist_key:
                dist_key = (ia, it)
                rf_val = t_arm.rf_distance(t_true) / self.rf_max
                kc_val = t_arm.kc_distance(t_true, 1.0)
            rf[i] = rf_val
            kc[i] = kc_val

            if it != true_key:
                true_key, v_true = it, self._tmrca_vec(t_true)
            if ia != arm_key:
                arm_key, v_arm = ia, self._tmrca_vec(t_arm)
            tm_true[i] = v_true
            tm_inf[i] = v_arm

            if self.log_every and (i + 1) % self.log_every == 0:
                print(f"  {i + 1:,}/{n_sites:,} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)

        print(f"  trees arm={self.arm_ts.num_trees:,} "
              f"truth={self.truth_ts.num_trees:,}; "
              f"skipped(num_roots!=1)={n_skip:,}; "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        return {
            "rf_norm": rf,
            "kc_lambda1": kc,
            "tmrca_true": tm_true.ravel(),
            "tmrca_inferred": tm_inf.ravel(),
            "breakpoints": self.arm_ts.breakpoints(as_array=True),
            "n_trees": np.int64(self.arm_ts.num_trees),
            "n_sites_skipped_multiroot": np.int64(n_skip),
        }


try:
    in_trees = str(snakemake.input.trees)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    in_arg = str(snakemake.input.arg)  # type: ignore[name-defined]
    out_npz = str(snakemake.output.npz)  # type: ignore[name-defined]
    tool = str(snakemake.wildcards.tool)  # type: ignore[name-defined]
    scheme = str(snakemake.wildcards.scheme)  # type: ignore[name-defined]
    n_pairs = int(snakemake.params.n_pairs)  # type: ignore[name-defined]
    pair_seed = int(snakemake.params.pair_seed)  # type: ignore[name-defined]
    n_out_reference = int(snakemake.params.n_out_reference)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    tool, scheme = "relate", "random5"
    in_trees = f"{DATA}/local_tree_genealogy_og_sim_20mb.trees"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    in_arg = f"{DATA}/orientation_arg_{tool}_{scheme}.trees"
    out_npz = f"{DATA}/orientation_persite_{tool}_{scheme}.npz"
    n_pairs = N_PAIRS
    pair_seed = PAIR_SEED
    n_out_reference = N_OUT_REFERENCE


panel = OrientationPanel(in_trees, in_meta,
                         n_out_reference=n_out_reference).load()
pos = load_sites(in_sites)["pos"]
pairs = haplotype_pairs(panel.n_ingroup, n_pairs=n_pairs, seed=pair_seed)

arm_ts = tskit.load(in_arg)
if arm_ts.num_samples != panel.n_ingroup:
    raise ValueError(f"{in_arg} carries {arm_ts.num_samples} samples, panel has "
                     f"{panel.n_ingroup}")

print(f"=== {tool} / {scheme} === sites={pos.size:,} pairs={len(pairs)} "
      f"rf_max={panel.rf_max}", flush=True)
result = PerSiteScorer(panel.truth_ts, arm_ts, pos, pairs,
                       panel.rf_max).run()
np.savez_compressed(
    out_npz, pos=pos, pairs=np.asarray(pairs, dtype=np.int32),
    rf_max=np.float64(panel.rf_max), **result)
print(f"wrote {out_npz}", flush=True)
