"""The worker-pool helpers every parallel path in the package initialises with.

``Settings._cap_worker_threads`` is the initialiser of all four pools, so an
exception inside it leaves ``multiprocessing.Pool`` waiting on a child that
never reports ready: the run hangs rather than failing. Neither helper had a
test, so that failure mode was reachable without anything going red.
"""
import logging

import numba
import pytest

from ancestree.settings import Settings


class TestCapWorkerThreads:
    """The numba thread share a forked child is given."""

    @pytest.mark.parametrize("n_workers", [0, 1])
    def test_a_lone_worker_is_left_alone(self, n_workers):
        """One worker keeps the parent's whole pool."""
        before = numba.get_num_threads()
        Settings._cap_worker_threads(n_workers)
        assert numba.get_num_threads() == before

    def test_it_never_raises_and_never_asks_for_less_than_one_thread(self):
        """The initialiser must not raise: a raise inside it hangs the pool."""
        before = numba.get_num_threads()
        try:
            for n in (2, 3, 8, 1024):
                Settings._cap_worker_threads(n)
                assert numba.get_num_threads() >= 1
        finally:
            numba.set_num_threads(before)

    def test_the_share_shrinks_as_the_worker_count_grows(self):
        """More siblings means no more threads each."""
        before = numba.get_num_threads()
        try:
            Settings._cap_worker_threads(2)
            few = numba.get_num_threads()
            numba.set_num_threads(before)
            Settings._cap_worker_threads(64)
            many = numba.get_num_threads()
            assert many <= few
        finally:
            numba.set_num_threads(before)


class TestForkPoolOk:
    """The fork-safety gate, and the warning it owes the caller."""

    def test_it_returns_a_bool_and_warns_only_when_refusing(self, caplog):
        log = logging.getLogger("ancestree.test_settings_pools")
        with caplog.at_level(logging.WARNING, logger=log.name):
            ok = Settings._fork_pool_ok(log, "trees", 4)
        assert isinstance(ok, bool)
        warned = [r for r in caplog.records
                  if "Ignoring the parallelization request" in r.getMessage()]
        assert bool(warned) is (not ok)

    def test_the_refusal_names_the_units_it_would_have_spread(
            self, caplog, monkeypatch):
        """The message reads as '... workers on the <what>'.

        The refusal is unreachable where the threading layer is fork-safe,
        so the gate is forced rather than skipped.
        """
        monkeypatch.setattr(Settings, "_fork_is_safe",
                            classmethod(lambda cls: False))
        log = logging.getLogger("ancestree.test_settings_pools")
        with caplog.at_level(logging.WARNING, logger=log.name):
            assert Settings._fork_pool_ok(log, "local trees", 8) is False
        assert any("local trees" in r.getMessage() for r in caplog.records)
