"""Which SINGER chain seeds a shard is built from.

The DAG names the seeds it wants as rule inputs, and the merge that scores
those shards has to agree exactly: a merge averaging a different set is a cell
built from a different posterior than the one it claims.

This module is the single answer, imported by both. It is deliberately
stdlib-only: the Snakefile evaluates input functions while building the DAG,
where importing the scoring stack (numpy, tskit, ancestree) would be slow and
is not available in every environment the DAG is built in.
"""
import functools
import os
from pathlib import Path

#: MCMC chains averaged per shard.
N_CHAINS = int(os.environ.get("SINGER_N_CHAINS", "13"))

#: Highest seed the search will consider before giving up. Seeds are taken in
#: ascending order and aborting ones skipped, so a chunk with several bad low
#: seeds needs headroom above ``N_CHAINS``.
MAX_SEED = int(os.environ.get("SINGER_MAX_SEED", "400"))

#: Recorded aborts, relative to this file (``workflow/``).
ABORTING_FILE = Path(__file__).resolve().parent.parent / "singer_aborting_seeds.txt"


#: ``(scenario, n_out, chunk)`` SINGER cannot infer at any seed. Dropped from
#: every method in that column, not only from SINGER's own cell: a cell summed
#: over a different chunk set than the row it is compared against is scored on
#: a different site set, which is the one invariant intersection scoring
#: exists to enforce. These slim chunks abort on every seed tried at the
#: production chain length while the same seeds run at 20 iterations, so the
#: instability is SINGER's chain length rather than the panel or its rates.
UNINFERABLE_COLUMN_CHUNKS = {
    ("slim", 0, 17), ("slim", 0, 70), ("slim", 0, 88), ("slim", 0, 93),
    ("slim", 1, 17), ("slim", 1, 52), ("slim", 1, 70), ("slim", 1, 90),
    ("slim", 1, 94),
    ("slim", 1, 0), ("slim", 1, 1), ("slim", 1, 2), ("slim", 1, 3),
    ("slim", 1, 4), ("slim", 1, 5), ("slim", 1, 6), ("slim", 1, 7),
    ("slim", 1, 9), ("slim", 1, 10), ("slim", 1, 11), ("slim", 1, 13),
    ("slim", 1, 14), ("slim", 1, 15), ("slim", 1, 18), ("slim", 1, 19),
    ("slim", 1, 21), ("slim", 1, 23), ("slim", 1, 24), ("slim", 1, 26),
    ("slim", 1, 28), ("slim", 1, 29), ("slim", 1, 32), ("slim", 1, 33),
    ("slim", 1, 34), ("slim", 1, 35), ("slim", 1, 36), ("slim", 1, 42),
    ("slim", 1, 44), ("slim", 1, 45), ("slim", 1, 47), ("slim", 1, 50),
    ("slim", 1, 51), ("slim", 1, 53), ("slim", 1, 54), ("slim", 1, 55),
    ("slim", 1, 57), ("slim", 1, 59), ("slim", 1, 62), ("slim", 1, 63),
    ("slim", 1, 65), ("slim", 1, 67), ("slim", 1, 68), ("slim", 1, 69),
    ("slim", 1, 72), ("slim", 1, 73), ("slim", 1, 74), ("slim", 1, 75),
    ("slim", 1, 77), ("slim", 1, 78), ("slim", 1, 81), ("slim", 1, 82),
    ("slim", 1, 83), ("slim", 1, 84), ("slim", 1, 85), ("slim", 1, 86),
    ("slim", 1, 88), ("slim", 1, 89), ("slim", 1, 91), ("slim", 1, 92),
    ("slim", 1, 93), ("slim", 1, 95), ("slim", 1, 96), ("slim", 1, 97),
    ("slim", 1, 98), ("slim", 1, 99),
}

#: Chunks per scenario, mirroring ``_robustness_common._n_chunks``.
N_CHUNKS = {"slim": 100}
DEFAULT_N_CHUNKS = 10


def scored_chunks(scenario, method, n_out, n_chunks=None):
    """Chunk indices a cell is summed over, dropping the uninferable ones.

    The drop is per column, so every method at one ``n_out`` is summed over the
    identical chunk set.

    :param n_chunks: Total chunks for the scenario. Defaults to :data:`N_CHUNKS`.
    :return: Ascending chunk indices.
    """
    total = N_CHUNKS.get(scenario, DEFAULT_N_CHUNKS) if n_chunks is None \
        else int(n_chunks)
    return [c for c in range(total)
            if (scenario, int(n_out), c) not in UNINFERABLE_COLUMN_CHUNKS]


@functools.cache
def aborting_seeds(path=None):
    """``(scenario, method, n_out, chunk, seed)`` tuples SINGER aborts on.

    SINGER dies with SIGABRT for particular seeds on particular chunks, the
    same way every time, so those seeds can never yield an ARG. Read from
    ``workflow/singer_aborting_seeds.txt`` rather than probed, because a failed
    job's output directory is removed and leaves nothing to detect it by.

    :param path: Override the file location. Defaults to
        :data:`ABORTING_FILE`.
    :return: Set of tuples.
    """
    out = set()
    for line in Path(path or ABORTING_FILE).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            scenario, method, n_out, chunk, seed = line.split()
            out.add((scenario, method, int(n_out), int(chunk), int(seed)))
    return out


def runnable_seeds(scenario, method, n_out, chunk_idx, n_seeds=None):
    """The first ``n_seeds`` seeds SINGER can actually run for one chunk.

    Taken in ascending order from ``range(MAX_SEED)``, skipping those in
    :func:`aborting_seeds`, so a chunk whose canonical seed cannot run advances
    onto one that can.

    :param n_seeds: How many are needed. Defaults to :data:`N_CHAINS`.
    :return: The chosen seeds, ascending.
    :raises ValueError: If fewer than ``n_seeds`` are runnable below
        :data:`MAX_SEED`.
    """
    want = N_CHAINS if n_seeds is None else int(n_seeds)
    aborting = aborting_seeds()
    chosen, seed = [], 0
    while len(chosen) < want and seed < MAX_SEED:
        if (scenario, method, int(n_out), int(chunk_idx), seed) not in aborting:
            chosen.append(seed)
        seed += 1
    if len(chosen) < want:
        raise ValueError(
            f"{scenario} {method} n_out={n_out} chunk={chunk_idx}: only "
            f"{len(chosen)} of {want} SINGER seeds below {MAX_SEED} are "
            f"runnable, the rest aborted. Raise SINGER_MAX_SEED or drop this "
            f"chunk from the scored set.")
    return chosen
