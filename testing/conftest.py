"""Shared pytest fixtures.

A small simulated tree sequence lives here so individual test modules do not
have to re-simulate. Non-fixture helpers live in :mod:`testing._helpers`.
"""
from __future__ import annotations

import os

# Give every xdist worker its own numba cache. The kernels are @njit(cache=True),
# and a shared cache directory means N workers compiling the same functions race
# on the same index/object files -- which crashes workers rather than merely
# slowing them, and looks like a mass test failure. Set before anything imports
# ancestree, since numba reads this when a cached function is first compiled.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker:
    _base = os.environ.get("NUMBA_CACHE_DIR") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "numba-pytest")
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(_base, _worker)
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import logging

import msprime
import pytest


@pytest.fixture(scope="session")
def small_ts():
    """A tiny msprime ARG: 10 diploid samples, 50kb, neutral mutation rate."""
    ts = msprime.sim_ancestry(
        samples=10,
        sequence_length=5e4,
        recombination_rate=1e-8,
        population_size=1e4,
        random_seed=1,
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
    return ts


@pytest.fixture(autouse=True)
def ancestree_log_level():
    """Hold the ``ancestree`` logger level constant across tests.

    The level is process-global, so a test that lowers or raises it, directly
    or through an import that configures logging, changes what every later
    test sees. Restored on failure as well as on success.
    """
    logger = logging.getLogger("ancestree")
    level = logger.level
    try:
        yield
    finally:
        logger.setLevel(level)
