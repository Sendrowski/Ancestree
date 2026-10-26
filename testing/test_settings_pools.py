"""The worker-pool helpers every parallel path in the package initialises with.

``Settings._cap_worker_threads`` runs in the initialiser of all four pools, so
an exception inside it stops every worker before it maps a single item.
Neither helper had a test, so that failure mode was reachable without anything
going red.
"""
import logging
import multiprocessing as mp
import sys

import numba
import pytest

from ancestree.settings import Settings, _ForkPool, _run_fork_task

FORK_AVAILABLE = "fork" in mp.get_all_start_methods()


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

    def test_capping_threads_without_numba_leaves_the_pool_alone(
            self, monkeypatch):
        before = numba.get_num_threads()
        monkeypatch.setitem(sys.modules, "numba", None)
        assert Settings._cap_worker_threads(4) is None
        assert numba.get_num_threads() == before


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


class TestForkPool:
    """The ordered fork-pool map every parallel inference path runs through."""

    log = logging.getLogger("ancestree.test_settings_pools")

    def test_the_worker_entrypoint_refuses_to_run_without_a_task(self):
        with pytest.raises(RuntimeError, match="no fork-pool task was set"):
            _run_fork_task(1)

    def test_one_worker_maps_in_process(self):
        seen = []
        pool = _ForkPool(1, self.log, "items")
        assert list(pool.map(lambda x: seen.append(x) or x * x, range(5))) == \
            [0, 1, 4, 9, 16]
        assert seen == [0, 1, 2, 3, 4]

    @pytest.mark.skipif(not FORK_AVAILABLE, reason="fork unavailable")
    def test_workers_inherit_an_unpicklable_function_and_keep_item_order(self):
        """A closure over a lock cannot be pickled, so only a forked worker
        can run it. More items than the in-flight window are returned in
        item order."""
        import threading

        lock = threading.Lock()

        def square(x):
            with lock:
                return x * x

        pool = _ForkPool(3, self.log, "items")
        assert pool.n_workers == 3
        assert list(pool.map(square, range(25))) == [x * x for x in range(25)]

    @pytest.mark.skipif(not FORK_AVAILABLE, reason="fork unavailable")
    @pytest.mark.timeout(60)
    def test_a_worker_killed_by_a_signal_fails_the_map_promptly(self):
        """A worker killed as the OOM killer does used to hang the map
        forever: the lost item's result never arrived, and the replacement
        worker forked without the task."""
        import os
        import signal
        import time

        def die_on_seven(x):
            if x == 7:
                os.kill(os.getpid(), signal.SIGKILL)
            return x

        pool = _ForkPool(3, self.log, "widgets")
        start = time.monotonic()
        with pytest.raises(RuntimeError, match=r"widgets died \(signal 9"):
            list(pool.map(die_on_seven, range(25)))
        assert time.monotonic() - start < 10

    def test_a_platform_without_fork_maps_in_process(self, monkeypatch, caplog):
        monkeypatch.setattr(mp, "get_all_start_methods", lambda: ["spawn"])
        with caplog.at_level(logging.WARNING, logger=self.log.name):
            pool = _ForkPool(4, self.log, "widgets")
        assert pool.n_workers == 1
        assert any("'fork' start method is unavailable" in r.getMessage()
                   and "widgets" in r.getMessage() for r in caplog.records)
