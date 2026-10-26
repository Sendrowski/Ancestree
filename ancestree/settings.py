"""Package-wide switches that override the equivalent per-call arguments."""
import itertools
import logging
import os
import signal
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator

#: The function a :class:`_ForkPool` worker maps, set by its initializer from
#: arguments inherited through fork, so neither it nor the object it is bound
#: to is pickled.
_FORK_TASK: "Callable | None" = None


class Settings:
    """Package-wide defaults, set directly on the class::

        import ancestree as anc

        anc.Settings.disable_pbar = True
        anc.Settings.parallelize = False

    Both switches are read in the parent process only. Workers are passed
    what they need explicitly.
    """

    #: Whether to suppress progress bars package-wide.
    disable_pbar: bool = False

    #: Package-wide parallelization switch: ``False`` disables it everywhere,
    #: ``True`` enables it everywhere, ``None`` defers to the caller.
    parallelize: bool | None = None

    @classmethod
    def _resolve_n_workers(
        cls, requested: int | None, *, force_serial: bool = False,
    ) -> int:
        """Resolve a worker count against the global switch.

        :param requested: The caller's worker count. ``None`` means unset.
        :param force_serial: Return ``1`` whatever the switch says, for callers
            whose correctness depends on running in-process.
        :return: The worker count to actually use, at least ``1``.
        """
        import multiprocessing as mp

        if force_serial or cls.parallelize is False:
            return 1
        # A daemonic worker may not have children, so a pool opened here
        # raises.
        if mp.current_process().daemon:
            return 1
        if cls.parallelize is True and (requested is None or requested <= 1):
            return os.cpu_count() or 1
        return max(1, requested) if requested is not None else 1

    @staticmethod
    def _cap_worker_threads(n_workers: int) -> None:
        """Give this worker process its share of the parent's numba threads,
        so ``n_workers`` siblings do not oversubscribe the machine.

        :param n_workers: Sibling worker processes sharing the machine.
        """
        if n_workers <= 1:
            return
        try:
            import numba
        except ImportError:
            return
        # set_num_threads raises above the pool numba was launched with.
        inherited = int(numba.get_num_threads())
        ceiling = int(getattr(numba.config, "NUMBA_NUM_THREADS", inherited))
        share = max(1, min(ceiling, inherited // n_workers))
        numba.set_num_threads(share)


def _init_fork_worker(fn: Callable, n_workers: int) -> None:
    """Initializer of a :class:`_ForkPool` worker.

    :param fn: The function the worker maps.
    :param n_workers: Sibling worker processes sharing the machine.
    """
    global _FORK_TASK
    _FORK_TASK = fn
    Settings._cap_worker_threads(n_workers)


def _run_fork_task(item):
    """Worker entrypoint of :class:`_ForkPool`: the inherited function on one
    item.

    :param item: The item, pickled to the worker.
    :return: The function's result, pickled back to the parent.
    :raises RuntimeError: If no worker initializer set the function.
    """
    if _FORK_TASK is None:
        raise RuntimeError(
            "no fork-pool task was set; the worker was not started by "
            "_ForkPool.")
    return _FORK_TASK(item)


class _ForkPool:
    """Ordered map over items in forked worker processes.

    At most ``n_workers + 2`` items are in flight. The map runs in-process
    with one worker, or where forking is unsafe or unavailable, with a warning.
    """

    def __init__(self, n_workers: int, log: logging.Logger, what: str) -> None:
        """
        :param n_workers: Resolved worker count.
        :param log: Logger the fallback warnings are emitted on.
        :param what: Plural noun for the items, as in "the tree chunks".
        """
        #: Plural noun for the items.
        self.what = what
        #: Worker processes the map uses, ``1`` for an in-process map.
        self.n_workers = (n_workers if n_workers > 1
                          and self._can_fork(log, what, n_workers) else 1)

    @staticmethod
    def _fork_is_safe() -> bool:
        """Whether forking a worker pool is safe with the active numba layer.

        :return: ``False`` only for the GNU OpenMP layer on Linux, where a
            forked child deadlocks.
        """
        try:
            import numba
            layer = numba.threading_layer()
        except Exception:  # noqa: BLE001  no kernel has run yet, or no numba
            return True
        return str(layer) != "omp" or not sys.platform.startswith("linux")

    @classmethod
    def _can_fork(cls, log: logging.Logger, what: str, n_workers: int) -> bool:
        """Whether a fork pool may be opened, warning when it may not.

        :param log: Logger the warning is emitted on.
        :param what: Plural noun for the items.
        :param n_workers: Workers the pool would open.
        :return: ``True`` when forking is safe and available.
        """
        import multiprocessing as mp

        if not cls._fork_is_safe():
            log.warning(
                "Ignoring the parallelization request: numba resolved the "
                "OpenMP threading layer, which is not fork-safe, and a forked "
                "worker would hang rather than fail. Unset "
                "NUMBA_THREADING_LAYER or set it to 'forksafe' to use up to %d "
                "parallel workers on the %s.", n_workers, what)
            return False
        if "fork" in mp.get_all_start_methods():
            return True
        log.warning(
            "The 'fork' start method is unavailable on this platform, so the "
            "%s run in-process. Spawned workers would pickle the inference.",
            what)
        return False

    def map(self, fn: Callable, items: Iterable) -> Iterator:
        """``fn`` over ``items``, results in item order.

        :param fn: The function, inherited by the workers.
        :param items: The items, each pickled to a worker.
        :return: Iterator of the results.
        :raises RuntimeError: If a worker process dies.
        """
        if self.n_workers <= 1:
            yield from map(fn, items)
            return
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        n = self.n_workers
        with ProcessPoolExecutor(
                n, mp_context=mp.get_context("fork"),
                initializer=_init_fork_worker, initargs=(fn, n)) as pool:
            pending = iter(items)
            in_flight = deque(pool.submit(_run_fork_task, item)
                              for item in itertools.islice(pending, n + 2))
            workers = list(pool._processes.values())
            try:
                while in_flight:
                    result = in_flight.popleft().result()
                    for item in itertools.islice(pending, 1):
                        in_flight.append(pool.submit(_run_fork_task, item))
                    yield result
            except BrokenProcessPool as e:
                pool.shutdown()
                died = ", ".join(
                    f"signal {-c}, {signal.strsignal(-c)}" if c < 0
                    else f"exit code {c}" for c in (
                        p.exitcode for p in workers
                        if p.exitcode != -signal.SIGTERM))
                raise RuntimeError(
                    f"A worker process on the {self.what} died ({died}), "
                    "possibly killed for running out of memory.") from e
            except BaseException:
                for p in workers:
                    p.terminate()
                raise
