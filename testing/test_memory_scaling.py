"""Peak memory of each mode must stay flat in the input length.

A mode streams its input in chunks, so doubling the length must leave its peak
resident set size about unchanged, while one that holds the whole input
roughly doubles. Each case runs on simulated inputs of length ``L`` and
``2 L`` in a fresh interpreter, which reports its own peak, and must also stay
below an absolute budget of about twice the peak measured when it was written.
"""
import multiprocessing as mp

import numpy as np
import pytest

from testing._helpers import peak_rss_mb

pytestmark = pytest.mark.slow

#: Haploid samples simulated, the last two the outgroups.
N_SAMPLES = 10
#: The shorter input length in bp, longer than an alignment's automatic chunk.
LENGTH = 2_000_000
#: Largest allowed ratio of the peak at ``2 L`` to the peak at ``L``.
MAX_RATIO = 1.5
#: Peak budget per case in MB, about twice the peak measured at ``2 L``.
BUDGET_MB = {
    "fixed-tree vcf": 400,
    "fixed-tree fasta": 600,
    "local-tree vcf": 500,
    "local-tree fasta": 4700,
    "arg trees": 400,
}

NAMES = [f"s{i}" for i in range(N_SAMPLES)]
INGROUP, OUTGROUP = NAMES[:-2], NAMES[-2:]


def _simulate(directory, length):
    """The ``.trees``, VCF and FASTA forms of one simulated panel, its
    samples named."""
    import msprime
    import tskit

    ts = msprime.sim_ancestry(N_SAMPLES, ploidy=1, population_size=1e4,
                              sequence_length=length, recombination_rate=1e-8,
                              random_seed=1)
    ts = msprime.sim_mutations(ts, rate=5e-8, model=msprime.JC69(),
                               random_seed=1)
    tables = ts.dump_tables()
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    rows = list(tables.individuals)
    tables.individuals.clear()
    for name, row in zip(NAMES, rows):
        tables.individuals.append(row.replace(metadata={"name": name}))
    ts = tables.tree_sequence()
    stem = directory / f"panel_{length}"
    ts.dump(f"{stem}.trees")
    with open(f"{stem}.vcf", "w") as f:
        ts.write_vcf(f, individual_names=NAMES)
    root = "".join(np.random.default_rng(1).choice(list("ACGT"), length))
    with open(f"{stem}.fa", "w") as f:
        for name, seq in zip(NAMES, ts.alignments(reference_sequence=root)):
            f.write(f">{name}\n{seq}\n")
    return stem


def _run(case, stem, length, queue):
    """Run one case in this child and report the sites scored and its peak,
    or the error it raised."""
    try:
        queue.put(_measure(case, stem, length))
    except Exception as exc:  # noqa: BLE001 - re-raised in the parent
        queue.put(exc)


def _measure(case, stem, length):
    """The sites ``case`` scores and this process's peak RSS in MB."""
    import logging

    import ancestree as anc

    logging.getLogger("ancestree").setLevel(logging.WARNING)

    mode, kind = case.split()
    path = f"{stem}.{'fa' if kind == 'fasta' else kind}"
    if mode == "fixed-tree":
        target = {"n_target_sites": length} if kind == "vcf" else {}
        inf = anc.FixedTreeInference(
            path, anc.JC69(), ingroup_samples=INGROUP,
            outgroup_samples=OUTGROUP, progress=False, n_starts=1, **target)
        inf.fit()
    elif mode == "local-tree":
        inf = anc.LocalTreeInference(
            path, mu=5e-8, rec_rate=1e-8, ingroup_samples=INGROUP,
            outgroup_samples=OUTGROUP, progress=False)
    else:
        import tskit
        inf = anc.ARGBasedInference(
            tskit.load(path), anc.JC69(), mu=5e-8, ingroup_samples=INGROUP,
            outgroup_samples=OUTGROUP, progress=False)
    return sum(1 for _ in inf.infer()), peak_rss_mb()


def _peak(case, stem, length):
    """Sites scored and peak RSS in MB of ``case`` run in a fresh interpreter."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_run, args=(case, stem, length, queue))
    proc.start()
    result = queue.get(timeout=1200)
    proc.join()
    if isinstance(result, Exception):
        raise result
    return result


@pytest.fixture(scope="module")
def panels(tmp_path_factory):
    directory = tmp_path_factory.mktemp("memory")
    return {length: _simulate(directory, length)
            for length in (LENGTH, 2 * LENGTH)}


@pytest.mark.parametrize("case", list(BUDGET_MB))
def test_peak_memory_is_flat_in_length(case, panels):
    n_short, short = _peak(case, panels[LENGTH], LENGTH)
    n_long, long = _peak(case, panels[2 * LENGTH], 2 * LENGTH)
    assert n_long > n_short > 0
    assert long < MAX_RATIO * short, (
        f"{case}: peak {short:.0f} MB at {LENGTH:,} bp, {long:.0f} MB at "
        f"{2 * LENGTH:,} bp")
    assert long < BUDGET_MB[case], f"{case}: peak {long:.0f} MB"
