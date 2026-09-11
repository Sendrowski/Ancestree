"""Declare an ancestral allele at every shared site under one orientation scheme.

Schemes, all reading the same genotypes and differing only in which of a site's
two alleles they call ancestral:

- ``true_aa`` --- the simulator's true ancestral state, tskit ``alleles[0]``.
  This is the reference arm every other arm is scored against.
- ``major_allele`` --- the more frequent allele among the 20 ingroup
  haplotypes, with an exact 10/10 tie keeping the true ancestral state.
- ``random5`` --- a uniformly random fraction of the sites (0.05 by default)
  have the other allele declared ancestral. The draw is independent of the
  derived-allele frequency, so distance to the nearest mis-oriented site
  carries no information about local high-frequency-variant density.
- ``freq_biased5`` --- the same overall rate as ``random5``, but a site with
  ingroup derived-allele count ``i`` is mis-oriented with probability
  proportional to ``i / n``, the error profile of an orienter drawing from the
  Kingman site-frequency prior. Errors therefore concentrate on the
  high-frequency-derived sites that carry the deep-branch signal.
- ``fixed_tree_n1`` / ``fixed_tree_n3`` --- the MAP call of Ancestree's
  fixed-tree mode with one or three outgroups, which errs through incomplete
  lineage sorting and recurrent mutation.

Wildcard: ``{scheme}``. Run via snakemake::

    snakemake -j 1 results/data/orientation_map_random5.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _orientation import (  # noqa: E402
    N_OUT_REFERENCE,
    RANDOM_FLIP_FRACTION,
    RANDOM_FLIP_SEED,
    load_sites,
)
from _robustness_common import Scenario  # noqa: E402


class OrientationMapBuilder:
    """Resolve one orientation scheme into a per-site ancestral-allele index.

    :param sites: The shared site set, as written by ``orientation_sites.py``.
    :param scheme: Scheme name, one of the five documented above.
    :param trees_path: Simulated tree sequence, needed by the fixed-tree schemes.
    :param meta_path: Companion simulation metadata JSON.
    :param flip_fraction: Fraction of sites ``random5`` mis-orients, in ``[0, 1]``.
    :param flip_seed: Seed of the ``random5`` draw.
    """

    def __init__(self, sites: dict, scheme: str, *, trees_path, meta_path,
                 flip_fraction: float = RANDOM_FLIP_FRACTION,
                 flip_seed: int = RANDOM_FLIP_SEED):
        self.sites = sites
        self.scheme = str(scheme)
        self.trees_path = Path(trees_path)
        self.meta_path = Path(meta_path)
        self.flip_fraction = float(flip_fraction)
        self.flip_seed = int(flip_seed)

    @property
    def n_sites(self) -> int:
        """Number of shared sites the map covers."""
        return int(self.sites["pos"].size)

    def build(self) -> tuple[np.ndarray, dict]:
        """Compute the declared ancestral allele at every shared site.

        :return: an ``int8`` index into the site's allele pair (``0`` is the
            simulator-true ancestral state) and the scheme's parameters and
            diagnostics.
        :raises ValueError: If ``scheme`` is not a known scheme name.
        """
        if self.scheme == "true_aa":
            return np.zeros(self.n_sites, dtype=np.int8), {}
        if self.scheme == "major_allele":
            return self._major_allele()
        if self.scheme == "random5":
            return self._random_flip()
        if self.scheme == "freq_biased5":
            return self._frequency_biased_flip()
        if self.scheme.startswith("fixed_tree_n"):
            return self._fixed_tree(int(self.scheme.rsplit("_n", 1)[1]))
        raise ValueError(f"unknown orientation scheme {self.scheme!r}")

    def _major_allele(self) -> tuple[np.ndarray, dict]:
        """Most frequent ingroup allele, with an exact tie keeping the truth."""
        g = self.sites["genotypes"]
        n_derived = g.sum(axis=1, dtype=np.int32)
        n_ancestral = g.shape[1] - n_derived
        idx = (n_derived > n_ancestral).astype(np.int8)
        return idx, {"n_ties_kept_true_ancestral":
                     int((n_derived == n_ancestral).sum())}

    def _random_flip(self) -> tuple[np.ndarray, dict]:
        """Uniform draw of sites whose other allele is declared ancestral."""
        rng = np.random.default_rng(self.flip_seed)
        n_target = int(round(self.flip_fraction * self.n_sites))
        picked = rng.choice(self.n_sites, size=n_target, replace=False)
        idx = np.zeros(self.n_sites, dtype=np.int8)
        idx[picked] = 1
        return idx, {
            "flip_fraction_target": self.flip_fraction,
            "flip_seed": self.flip_seed,
            "flip_rng": (f"numpy.random.default_rng({self.flip_seed}).choice("
                         f"n_sites, size=round({self.flip_fraction} * n_sites),"
                         " replace=False)"),
        }

    def _frequency_biased_flip(self) -> tuple[np.ndarray, dict]:
        """Flip with probability proportional to derived-allele frequency.

        A site with derived count ``i`` out of ``n`` ingroup haplotypes is
        mis-oriented with probability ``c * i / n``, and ``c`` is solved so the
        realised rate matches ``flip_fraction``. The probability is capped at
        one, so a saturating class contributes its whole mass and ``c`` is
        raised until the target is met.
        """
        g = self.sites["genotypes"]
        n_hap = g.shape[1]
        derived = g.sum(axis=1, dtype=np.int32)
        weight = derived / float(n_hap)
        target = self.flip_fraction * self.n_sites
        lo, hi = 0.0, float(n_hap)
        for _ in range(200):  # bisect c against the expected flip count
            mid = 0.5 * (lo + hi)
            if np.minimum(mid * weight, 1.0).sum() < target:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        prob = np.minimum(scale * weight, 1.0)
        rng = np.random.default_rng(self.flip_seed)
        idx = (rng.random(self.n_sites) < prob).astype(np.int8)
        by_count = {int(k): float(prob[derived == k][0])
                    for k in np.unique(derived) if (derived == k).any()}
        return idx, {
            "flip_fraction_target": self.flip_fraction,
            "flip_seed": self.flip_seed,
            "flip_probability": "min(scale * derived_count / n_haplotypes, 1)",
            "scale": float(scale),
            "flip_probability_by_derived_count": by_count,
        }

    def _fixed_tree(self, n_out: int) -> tuple[np.ndarray, dict]:
        """Fixed-tree MAP call with ``n_out`` outgroups spread by split depth.

        A call that is neither of the site's two alleles cannot orient it. Such
        a site keeps the simulator-true ancestral state and is counted.
        """
        meta = json.loads(self.meta_path.read_text())
        scen = Scenario(key=f"orientation_ft_n{n_out}",
                        label=f"orientation fixed tree n_out={n_out}",
                        trees_path=self.trees_path, meta_path=self.meta_path,
                        mu=float(meta["mu"]), rec_rate=float(meta["rec_rate"]))
        scen.load()
        call_by_pos = scen._fixed_tree_map_by_pos(n_out)
        selected = scen._select_outgroups(n_out)
        del scen

        pos = self.sites["pos"]
        alleles = self.sites["alleles"]
        idx = np.zeros(self.n_sites, dtype=np.int8)
        n_unorientable = 0
        for k in range(self.n_sites):
            call = call_by_pos.get(int(pos[k]))
            if call == alleles[k, 1]:
                idx[k] = 1
            elif call != alleles[k, 0]:
                n_unorientable += 1
        return idx, {
            "n_outgroups": n_out,
            "outgroups_selected": selected,
            "n_calls_not_among_site_alleles": n_unorientable,
            "unorientable_site_handling": (
                "a MAP call that is neither of the site's two alleles keeps the "
                "simulator-true ancestral state"),
        }


try:
    in_trees = str(snakemake.input.trees)  # type: ignore[name-defined]
    in_meta = str(snakemake.input.meta)  # type: ignore[name-defined]
    in_sites = str(snakemake.input.sites)  # type: ignore[name-defined]
    out_json = str(snakemake.output.json)  # type: ignore[name-defined]
    scheme = str(snakemake.wildcards.scheme)  # type: ignore[name-defined]
    flip_fraction = float(snakemake.params.flip_fraction)  # type: ignore[name-defined]
    flip_seed = int(snakemake.params.flip_seed)  # type: ignore[name-defined]
    n_out_reference = int(snakemake.params.n_out_reference)  # type: ignore[name-defined]
except NameError:
    DATA = "results/data"
    scheme = "random5"
    in_trees = f"{DATA}/local_tree_genealogy_og_sim_20mb.trees"
    in_meta = f"{DATA}/local_tree_genealogy_og_sim_20mb_meta.json"
    in_sites = f"{DATA}/orientation_sites.npz"
    out_json = f"{DATA}/orientation_map_{scheme}.json"
    flip_fraction = RANDOM_FLIP_FRACTION
    flip_seed = RANDOM_FLIP_SEED
    n_out_reference = N_OUT_REFERENCE


sites = load_sites(in_sites)
builder = OrientationMapBuilder(sites, scheme, trees_path=in_trees,
                                meta_path=in_meta,
                                flip_fraction=flip_fraction,
                                flip_seed=flip_seed)
anc_idx, params = builder.build()

pos = sites["pos"]
alleles = sites["alleles"]
n_mis = int((anc_idx != 0).sum())
payload = {
    "scheme": scheme,
    "n_sites": int(pos.size),
    "n_out_reference": n_out_reference,
    "n_mis_oriented": n_mis,
    "pct_mis_oriented": 100.0 * n_mis / pos.size,
    "mean_derived_allele_count_all_sites": float(sites["daf"].mean()),
    "mean_derived_allele_count_mis_oriented": (
        float(sites["daf"][anc_idx != 0].mean()) if n_mis else None),
    "params": params,
    "ancestral_allele_index": [int(v) for v in anc_idx],
    "ancestral_allele_by_position": {
        str(int(pos[k])): str(alleles[k, anc_idx[k]]) for k in range(pos.size)},
}
Path(out_json).write_text(json.dumps(payload))
print(f"{scheme}: {n_mis:,} of {pos.size:,} sites mis-oriented "
      f"({payload['pct_mis_oriented']:.3f}%) {params}", flush=True)
