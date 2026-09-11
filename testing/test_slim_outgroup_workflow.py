"""The SLiM-outgroup-bias workflow: the forward sim and the recap step.

The recap-and-mutate step runs from a committed tree sequence, so its outputs
are asserted wherever the suite runs. The forward sim needs the ``slim``
binary, an external tool rather than a Python package, and is covered by its
own test: that the ``.slim`` script still runs under the globals the workflow
injects, and still emits the demography the recap step reads.

The downstream :class:`~ancestree.inference.FixedTreeInference` step is
not exercised here, since the inference layer has its own tests.
"""
import collections
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import tskit


# pyslim is a declared test dependency, so it is imported outright: a guard
# here would turn a broken install into a silent pass.
import pyslim  # noqa: F401,E402

SLIM_BIN = shutil.which("slim")

REPO_ROOT = Path(__file__).resolve().parents[1]
SLIM_SCRIPT = REPO_ROOT / "workflow" / "scripts" / "simulate_slim_outgroup_bias.slim"
RECAP_SCRIPT = REPO_ROOT / "workflow" / "scripts" / "recap_and_mutate_slim_outgroup_bias.py"

#: The tree sequence :data:`SLIM_GLOBALS` produces, committed so the recap step
#: can be driven without the binary.
FIXTURE = Path(__file__).parent / "fixtures" / "slim" / "smoke_slim.trees"

#: Globals the workflow injects into the ``.slim`` script, sized for a forward
#: run of about a second: a short burn-in over small populations, with the
#: 4-pop demography (p0 plus the nested p1/p2/p3 splits) in one SLiM process.
SLIM_GLOBALS = [
    "SEED=123",
    "F_DEL=0.0",
    "SEQ_LEN=10000",
    "MU_SLIM=1.25e-7",
    "RHO_SLIM=1e-7",
    "N_E_SLIM=100",
    "N_E_OUT=5",
    "T1_SLIM=100",
    "T2_SLIM=200",
    "T3_SLIM=300",
]

#: Haploid samples per population: the p0 ingroup at ``2 * N_E_SLIM``, then the
#: three outgroups at ``2 * N_E_OUT``.
EXPECTED_SAMPLES_PER_POPULATION = {0: 200, 1: 10, 2: 10, 3: 10}

#: ``SEQ_LEN``, as the tree sequence reports it.
EXPECTED_SEQUENCE_LENGTH = 10_000.0

SKIP_REASON = (
    "needs the ``slim`` binary on PATH; the ``slim-ancestree`` mamba env "
    "provides it (see slim.yml)"
)


def _assert_demography(path: Path) -> None:
    """The populations and sample counts the recap step reads."""
    ts = tskit.load(str(path))
    counts = collections.Counter(ts.node(u).population for u in ts.samples())
    assert dict(counts) == EXPECTED_SAMPLES_PER_POPULATION
    assert ts.sequence_length == EXPECTED_SEQUENCE_LENGTH


@pytest.mark.skipif(SLIM_BIN is None, reason=SKIP_REASON)
def test_the_slim_script_runs(tmp_path: Path) -> None:
    """The forward sim, under the globals the workflow passes it.

    Renaming a ``-d`` global or altering the demography in the ``.slim`` script
    breaks the recap step, which reads the populations by index. This is the
    only test that reads the script itself, so it checks what comes out rather
    than only the exit status.
    """
    out_trees = tmp_path / "smoke_slim.trees"
    result = subprocess.run(
        [SLIM_BIN,
         *[arg for global_ in SLIM_GLOBALS for arg in ("-d", global_)],
         "-d", f"OUT_TREES='{out_trees}'",
         str(SLIM_SCRIPT)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"SLiM failed: stdout={result.stdout!r}, stderr={result.stderr!r}")
    assert out_trees.exists(), "SLiM did not write its tree sequence"
    _assert_demography(out_trees)


def test_the_fixture_is_what_the_slim_script_emits() -> None:
    """The committed input still describes the demography the globals give."""
    _assert_demography(FIXTURE)


def test_recap_and_mutate(tmp_path: Path) -> None:
    """The Python step: extend the deep root, overlay JC69, emit the VCF.

    The full 4-pop SLiM tree sequence is kept as it stands, with no
    ``tskit.union`` or grafting: the step extends the deep root, overlays JC69
    mutations, and picks one haploid per outgroup population.
    """
    out_trees = tmp_path / "smoke_final.trees"
    out_vcf = tmp_path / "smoke_final.vcf"
    out_meta = tmp_path / "smoke_meta.json"
    result = subprocess.run(
        [
            sys.executable, str(RECAP_SCRIPT),
            "--slim-trees", str(FIXTURE),
            "--out-trees", str(out_trees),
            "--out-vcf", str(out_vcf),
            "--out-meta", str(out_meta),
            "--f-del", "0.0",
            "--chunk-idx", "0",
            "--seed", "123",
            "--mu-slim", "1.25e-7",
            "--rec-rate-slim", "1e-7",
            "--n-e-slim", "100",
            "--mu-target", "1.25e-8",
            "--time-scaling", "10.0",
            "--outgroup-split-times-target", "1000", "2000", "3000",
            "--n-ingroup-haps", "100",
            "--seq-len", "10000",
        ],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    assert result.returncode == 0, (
        f"recap_and_mutate failed: stdout={result.stdout!r}, "
        f"stderr={result.stderr!r}")
    assert out_trees.exists()
    assert out_vcf.exists()
    assert out_meta.exists()

    meta = json.loads(out_meta.read_text())
    assert len(meta["outgroup_names"]) == 3
    assert all(name.startswith("out_T") for name in meta["outgroup_names"])
    assert len(meta["positions"]) == len(meta["truth_ancestral_states"])
    assert len(meta["positions"]) == meta["n_sites"]
    assert all(s in ("A", "C", "G", "T")
               for s in meta["truth_ancestral_states"])

    header_line = next(line for line in out_vcf.read_text().splitlines()
                       if line.startswith("#CHROM"))
    for outgroup in meta["outgroup_names"]:
        assert outgroup in header_line, (
            f"outgroup {outgroup} missing from VCF header")
