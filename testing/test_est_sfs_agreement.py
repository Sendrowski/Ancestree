"""Agreement with the original est-sfs (Keightley & Jackson 2018).

The fixtures in ``testing/fixtures/est_sfs`` were produced by evolving sites
down the outgroup ladder under JC69 and running est-sfs 2.04 on them, so both
tools see data their models can fit. There is one per outgroup count est-sfs
supports. The realised divergences are shared across them, so a count changes
only how many outgroups carry the signal. One outgroup is the case with no join
node, where the ladder's single branch is the whole divergence. See that
directory's README and ``generate.py``.

Two properties are pinned. The MAP ancestral allele must agree on every site
whose ingroup has an unambiguous major allele, and the posteriors must agree
to a tolerance set by the two documented model differences: est-sfs
approximates branch transition probabilities by a Poisson mutation count
truncated at three (``compute_poisson_vec``) where the kernel here uses the
exact ``P(t)``, and est-sfs fits one ancestral probability per frequency class
where :class:`~ancestree.priors.KingmanIngroupWeight` fixes it at
``(n - i) / n``.

Sites with an exact tie for the most frequent ingroup allele are excluded:
"major allele" is undefined there, and the two implementations break the tie
differently, which is a labelling difference rather than a disagreement.
"""
import pathlib

import numpy as np
import pytest

import ancestree as anc
from ancestree.models import JC69
from ancestree.priors import KingmanIngroupWeight
from ancestree.sites import Site

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "est_sfs"
STATES = ("A", "C", "G", "T")
INGROUP = [f"i{k}" for k in range(20)]
OUTGROUPS = ["o1", "o2", "o3"]

#: Outgroup counts est-sfs supports, each with its own fixture directory. The
#: realised divergences are shared across them, so a count changes only how many
#: outgroups carry the signal. One outgroup is the case with no join node, where
#: the ladder's single branch IS the whole divergence.
N_OUT = (1, 2, 3)


def _dir(n_out):
    return FIXTURE / f"n_out_{n_out}"


def _load(n_out=3):
    """Fixture sites, est-sfs's posteriors, and the rates it fitted.

    :return: ``(sites, ingroup_counts, p_est, rates)``.
    """
    lines = [l for l in (_dir(n_out) / "sites.txt").read_text().split("\n")
             if l.strip()]
    out = (_dir(n_out) / "p_anc.txt").read_text().splitlines()
    rates = [float(x) for x in
             next(l for l in out if l.startswith("0 Rates")).split()[3::2]]
    p_est = np.array([float(l.split()[2]) for l in out if not l.startswith("0 ")])
    sites, counts = [], []
    for idx, line in enumerate(lines):
        parts = line.split()
        c = [int(x) for x in parts[0].split(",")]
        tips, k = {}, 0
        for state_idx, n in enumerate(c):
            for _ in range(n):
                tips[INGROUP[k]] = STATES[state_idx]
                k += 1
        for oi, block in enumerate(parts[1:]):
            v = [int(x) for x in block.split(",")]
            tips[OUTGROUPS[oi]] = STATES[v.index(1)]
        counts.append(c)
        sites.append(Site(
            chrom="1", pos=idx + 1,
            alleles=tuple(sorted(set(tips.values()))), tip_alleles=tips,
        ))
    return sites, np.array(counts), p_est, rates


def _agreement(n_out):
    """Both tools' posteriors for the major ingroup allele, on comparable sites.

    :return: ``(ours, theirs, counts)`` over the sites whose major allele is
        unambiguous.
    """
    sites, counts, p_est, rates = _load(n_out)
    tree = anc.OutgroupLadderTree(INGROUP, OUTGROUPS[:n_out])
    tree.set_params(rates)
    inference = anc.FixedTreeInference(
        sites, JC69(), n_target_sites=2_000_000, tree=tree,
        ingroup_weight=KingmanIngroupWeight(ingroup_samples=INGROUP),
        fit_required=False, progress=False, baseline_check=False,
    )
    posteriors = [p for _, p in inference.infer()]
    ranked = np.sort(counts, axis=1)
    keep = np.where(ranked[:, -1] != ranked[:, -2])[0]
    major = [STATES[int(np.argmax(counts[i]))] for i in keep]
    ours = np.array([float(posteriors[i][m]) for i, m in zip(keep, major)])
    return ours, p_est[keep], counts[keep]


@pytest.fixture(scope="module", params=N_OUT, ids=lambda k: f"n_out={k}")
def comparison(request):
    """Ancestree and est-sfs posteriors on the unambiguous fixture sites."""
    return _agreement(request.param)


def test_every_map_call_matches_est_sfs(comparison):
    ours, theirs, _ = comparison
    assert len(ours) > 38_000
    assert np.array_equal(ours > 0.5, theirs > 0.5)


def test_posteriors_match_within_the_model_differences(comparison):
    ours, theirs, _ = comparison
    diff = np.abs(ours - theirs)
    # The mean is the informative bound: the two model differences are small
    # everywhere. The max is a handful of near-tie sites where est-sfs's fitted
    # class weight and the Kingman weight disagree most, and it sits at 2.0e-02
    # at every outgroup count.
    assert diff.mean() < 1e-4
    assert diff.max() < 3e-2


def test_the_largest_gap_is_at_the_near_tie_class():
    """The residual concentrates where the class weight matters most.

    Only asserted at three outgroups. The residual is a sum of the two
    documented components, and which one dominates depends on how much outgroup
    evidence there is. With three outgroups most sites are decided confidently
    by the outgroups, both tools saturate there, and what is left is the
    class-weight difference -- so the per-class mean residual falls with the
    major count (4.6e-05 at a near tie, 9.4e-06 at 17--19 copies) and the worst
    site is a near tie. With one or two outgroups the evidence is weaker, no
    class saturates, and the branch-length component is comparable throughout;
    the per-class mean then rises with the major count instead and the single
    worst site is not diagnostic of anything.
    """
    ours, theirs, counts = _agreement(3)
    n_major = counts.max(axis=1)
    diff = np.abs(ours - theirs)
    worst = int(np.argmax(diff))
    assert int(n_major[worst]) in (11, 12), (
        f"worst gap at n_major={n_major[worst]}, expected a near tie")
    means = [diff[(n_major >= lo) & (n_major <= hi)].mean()
             for lo, hi in ((11, 12), (13, 16), (17, 19))]
    assert means == sorted(means, reverse=True), (
        f"residual does not fall with the major count: {means}")


class TestInternalNodePosteriors:
    """The ``ptree`` block, marginalised onto each internal ladder node.

    est-sfs reports the joint over the two internal node states as the mean of
    the two ingroup-MRCA hypotheses weighted equally (``est-sfs.c:656``), so
    above ``b1`` it carries no frequency information. Feeding the kernel that
    same weighting must reproduce its marginals. Feeding the Kingman weight
    must not, and the gap is the information est-sfs discards there.
    """

    @staticmethod
    def _setup():
        """Fixture sites, est-sfs marginals at ``n_1`` and ``n_2``, and seeds."""
        sites, counts, _, rates = _load()
        out = (_dir(3) / "p_anc.txt").read_text().splitlines()
        ptree = np.array([[float(x) for x in l.split()[3:19]]
                          for l in out if not l.startswith("0 ")])
        # Index: n_1 = (i - 1) mod 4, n_2 = (i - 1) div 4.
        joint = ptree.reshape(-1, 4, 4)
        m1 = joint.sum(axis=1); m2 = joint.sum(axis=2)
        m1 /= m1.sum(1, keepdims=True); m2 /= m2.sum(1, keepdims=True)

        ranked = np.sort(counts, axis=1)
        keep = np.where(ranked[:, -1] != ranked[:, -2])[0]
        n = len(keep)
        major = np.array([int(np.argmax(counts[i])) for i in keep])
        minor = np.array([int(np.argsort(counts[i])[::-1][1]) for i in keep])
        n_major = counts[keep].max(axis=1)

        equal = np.zeros((n, 4))
        equal[np.arange(n), major] = 0.5
        equal[np.arange(n), minor] = 0.5
        kingman = np.zeros((n, 4))
        kingman[np.arange(n), major] = n_major / 20.0
        kingman[np.arange(n), minor] = 1.0 - n_major / 20.0
        return ([sites[i] for i in keep], rates,
                {1: m1[keep], 2: m2[keep]}, equal, kingman)

    @staticmethod
    def _readout(sites, rates, seed, coalescences):
        """The posterior at the node ``coalescences`` joins above the MRCA."""
        from ancestree.focal import FocalNode
        from ancestree.likelihood import Likelihood
        tree = anc.OutgroupLadderTree(INGROUP, OUTGROUPS)
        tree.set_params(rates)
        view = tree.at_focal(FocalNode("ingroup_mrca", coalescences=coalescences))
        log_l = np.asarray(Likelihood(JC69()).log_likelihoods(
            view, sites, node_seeds={tree.ingroup_mrca: seed}))
        w = np.exp(log_l - log_l.max(axis=1, keepdims=True)) * 0.25  # JC69 pi
        return w / w.sum(1, keepdims=True)

    @pytest.mark.parametrize("coalescences", [1, 2])
    def test_equal_weighting_reproduces_est_sfs(self, coalescences):
        sites, rates, marginals, equal, _ = self._setup()
        ours = self._readout(sites, rates, equal, coalescences)
        # est-sfs's marginals are reported to six decimals and its rates are
        # fitted, so the floor here is the fixture's own draw, not the kernel.
        assert np.abs(ours - marginals[coalescences]).max() < 2e-4

    def test_the_ingroup_frequency_reaches_the_internal_nodes(self):
        """The Kingman weight moves the readout above the MRCA.

        Pins what seeding buys over applying the weight at the readout: the
        posterior at ``n_1`` responds to the ingroup allele frequency, which
        est-sfs's equally-weighted average cannot express.
        """
        sites, rates, marginals, _, kingman = self._setup()
        ours = self._readout(sites, rates, kingman, 1)
        assert np.abs(ours - marginals[1]).max() > 0.1


class TestGroundTruth:
    """Score the readings against the states the fixture was generated from.

    The fixture records the simulated state at the ingroup MRCA and at both
    internal ladder nodes, so a reading at any of them can be graded directly
    rather than compared against another estimator.
    """

    @staticmethod
    def _truth():
        """Simulated states, keyed by coalescences above the ingroup MRCA."""
        h = np.load(_dir(3) / "hidden_states.npz")
        return {0: h["ingroup_mrca"], 1: h["n1"], 2: h["n2"]}

    def test_the_ingroup_weight_beats_a_flat_one_at_the_readout(self):
        """The frequency information is worth having where it applies."""
        sites, counts, _, rates = _load()
        ranked = np.sort(counts, axis=1)
        keep = np.where(ranked[:, -1] != ranked[:, -2])[0]
        truth = self._truth()[0][keep]
        n = len(keep)
        major = np.array([int(np.argmax(counts[i])) for i in keep])
        minor = np.array([int(np.argsort(counts[i])[::-1][1]) for i in keep])
        n_major = counts[keep].max(axis=1)

        flat = np.full((n, 4), 0.25)
        kingman = np.zeros((n, 4))
        kingman[np.arange(n), major] = n_major / 20.0
        kingman[np.arange(n), minor] = 1.0 - n_major / 20.0

        from ancestree.likelihood import Likelihood
        tree = anc.OutgroupLadderTree(INGROUP, OUTGROUPS)
        tree.set_params(rates)
        engine = Likelihood(JC69())
        sub = [sites[i] for i in keep]

        def brier(seed):
            log_l = np.asarray(engine.log_likelihoods(
                tree, sub, node_seeds={tree.ingroup_mrca: seed}))
            p = np.exp(log_l - log_l.max(axis=1, keepdims=True)) * 0.25
            p /= p.sum(1, keepdims=True)
            hit = np.zeros_like(p)
            hit[np.arange(n), truth] = 1.0
            return ((p - hit) ** 2).sum(1).mean()

        assert brier(kingman) < 0.6 * brier(flat)
