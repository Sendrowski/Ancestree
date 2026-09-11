"""Ancestree against a frozen PolarBEAR run, inferring live.

The cross-tool claim in the paper is that ARG mode reproduces PolarBEAR's
per-site posteriors on the sites PolarBEAR calls. PolarBEAR needs its own
environment (a pinned tskit 0.6 + numba stack), so its output is committed
here in the same shape as ``testing/fixtures/est_sfs``: the other tool is a
fixture, Ancestree runs against it.

``test_benchmarks.py`` asserts the same claim against a report the pipeline
wrote, which cannot fail on a defect in the inference itself.
"""
import gzip
import json
from pathlib import Path

import msprime
import numpy as np
import pytest

import ancestree as anc

pytestmark = pytest.mark.slow

FIXTURE = Path(__file__).parent / "fixtures" / "polarbear" / "baseline.json.gz"

# POLARBEAR_CONFIG in the Snakefile. The panel the fixture was produced from.
SAMPLES, LENGTH, MU, REC, NE, SEED = 20, 2_000_000, 1e-8, 1e-8, 1e4, 42


@pytest.fixture(scope="module")
def baseline():
    with gzip.open(FIXTURE, "rt") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def ours():
    """ARG mode on the same simulated panel, inferred here."""
    ts = msprime.sim_ancestry(
        samples=SAMPLES, sequence_length=LENGTH, recombination_rate=REC,
        population_size=NE, random_seed=SEED)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=SEED)
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=MU,
                                 progress=False)
    return {int(s.pos): p for s, p in inf.infer()}


def test_the_fixture_describes_the_expected_panel(baseline):
    meta = baseline["inference_meta"]
    assert meta["n_sites"] == 3348
    assert meta["mu"] == MU


def test_every_polarbear_call_is_reproduced(baseline, ours):
    """The paper's claim: the MAP calls agree site for site."""
    theirs = baseline["sites"]
    shared = [int(k) for k in theirs if int(k) in ours]
    assert len(shared) > 3000, f"only {len(shared)} shared sites"
    disagree = [k for k in shared
                if ours[k].map_allele != theirs[str(k)]["map_allele"]]
    assert not disagree, (
        f"{len(disagree)} of {len(shared)} MAP calls differ from PolarBEAR")


def test_the_posteriors_agree_numerically(baseline, ours):
    """The MAP probabilities agree to the order of the series truncation.

    PolarBEAR keeps the zero- and one-mutation terms of the per-branch
    Poisson series, whereas the kernel evaluates the transition matrix
    exactly, so at this mutation rate the two differ at order 1e-3 in the
    MAP probability while the MAP alleles themselves agree site for site.
    """
    theirs = baseline["sites"]
    shared = [int(k) for k in theirs if int(k) in ours]
    diffs = [abs(float(np.max(ours[k].values)) - theirs[str(k)]["max_prob"])
             for k in shared]
    assert float(np.mean(diffs)) < 2e-3, (
        f"mean |diff| in the MAP probability is {np.mean(diffs):.3g}")
