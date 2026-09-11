"""The paper's scientific claims, computed rather than read from a file.

``test_benchmarks.py`` asserts against JSON the pipeline wrote, so it cannot
fail on a defect in inference: the numbers only change when someone
regenerates the artifacts, and the floors are then compared against whatever
was just written. These run inference on a fixed seed and assert the same
claims against fresh output.

Marked slow because they infer rather than load. The external baselines that
genuinely need another tool keep the fixture pattern of
``test_est_sfs_agreement.py``: freeze the other tool's output, run Ancestree
live against it.
"""
import msprime
import pytest

import ancestree as anc

pytestmark = pytest.mark.slow

SEED = 42


@pytest.fixture(scope="module")
def panel():
    """The B2 configuration: 20 diploid samples over 1 Mb."""
    ts = msprime.sim_ancestry(
        samples=20, sequence_length=1e6, recombination_rate=1e-8,
        population_size=1e4, random_seed=SEED)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=SEED)
    assert ts.num_sites > 1000
    return ts


def _recovery(ts, **kw):
    """Fraction of sites whose MAP call is the simulated ancestral state."""
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=1e-8, progress=False,
                                **kw)
    truth = {int(s.position): s.ancestral_state for s in ts.sites()}
    hits = n = 0
    for site, post in inf.infer():
        t = truth.get(int(site.pos))
        if t is None:
            continue
        n += 1
        hits += int(post.map_allele == t)
    return hits / n, n


def test_truth_recovery_matches_its_reference(panel):
    """B2's headline claim, pinned rather than floored.

    A floor with slack tolerates a kernel bias large enough to move every
    posterior. The value is pinned to the seed instead, so any change to the
    kernel, the model or the focal node has to be deliberate.
    """
    rate, n = _recovery(panel)
    assert n == 1611, f"the panel changed: {n} sites scored, expected 1611"
    assert rate == pytest.approx(0.878336, abs=5e-4), (
        f"truth recovery moved to {rate:.6f} from 0.878336")



@pytest.fixture(scope="module")
def split_panel():
    """20 ingroup haplotypes and 2 outgroups, split at 2e5 generations."""
    demography = msprime.Demography()
    for name in ("ingroup", "outgroup", "anc"):
        demography.add_population(name=name, initial_size=1e4)
    demography.add_population_split(
        time=2e5, ancestral="anc", derived=["ingroup", "outgroup"])
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(20, population="ingroup", ploidy=1),
                 msprime.SampleSet(2, population="outgroup", ploidy=1)],
        demography=demography, sequence_length=1e6,
        recombination_rate=1e-8, random_seed=SEED)
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=SEED)
    assert ts.num_sites > 1000
    return ts


def _calls(ts, **kw):
    """``{position: was the MAP call the simulated ancestral state}``."""
    inference = anc.Inference.from_arg(
        ts, model=anc.JC69(), mu=1e-8, progress=False, **kw)
    truth = {int(s.position): s.ancestral_state for s in ts.sites()}
    out = {}
    for site, post in inference.infer():
        want = truth.get(int(site.pos))
        if want is not None:
            out[int(site.pos)] = post.map_allele == want
    return out


def test_outgroups_do_not_reduce_accuracy(split_panel):
    """B6: adding outgroups must not make the call worse.

    Both sides read at the ingroup MRCA and are scored on the sites they
    share. Comparing two focal nodes on an outgroup-free panel instead made
    both sides the same computation, since with nothing designated the ingroup
    is the whole panel and its MRCA is the panel root.
    """
    ingroup = [str(i) for i in range(20)]
    with_outgroups = _calls(
        split_panel, ingroup_samples=ingroup, outgroup_samples=["20", "21"])
    ingroup_only = _calls(split_panel.simplify(list(range(20))))

    shared = sorted(set(with_outgroups) & set(ingroup_only))
    assert len(shared) > 1000
    got = sum(with_outgroups[p] for p in shared) / len(shared)
    without = sum(ingroup_only[p] for p in shared) / len(shared)
    assert got >= without - 0.02, (
        f"with outgroups {got:.4f} against ingroup-only {without:.4f} "
        f"on {len(shared)} shared sites")
