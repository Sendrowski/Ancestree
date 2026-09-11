"""Package-wide switches that override the equivalent per-call arguments."""
import logging
import os
import sys


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
    def _use_parallel(cls, requested: bool) -> bool:
        """Resolve a caller's parallelization request against the global switch.

        :param requested: Whether the caller asked for parallel execution.
        :return: ``True`` when the work should actually run in parallel.
        """
        if cls.parallelize is None:
            return bool(requested)
        return bool(cls.parallelize)

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
        """Give this worker process its share of the parent's numba threads.

        Worker initialiser for every pool the package opens. A forked child
        inherits the parent's ``NUMBA_NUM_THREADS`` pool size, so ``n_workers``
        children each running a ``parallel=True`` kernel oversubscribe the
        machine by a factor of ``n_workers``.

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

    @staticmethod
    def _fork_is_safe() -> bool:
        """Whether forking a worker pool is safe with the active numba layer.

        GNU OpenMP, the ``omp`` layer on Linux, is fork-unsafe: a child forked
        after a parallel kernel has run deadlocks inside its first parallel
        region. Every other layer and platform is fork-safe.

        :return: ``False`` only for the OpenMP layer on Linux.
        """
        try:
            import numba
            layer = numba.threading_layer()
        except Exception:  # noqa: BLE001  no kernel has run yet, or no numba
            return True
        return str(layer) != "omp" or not sys.platform.startswith("linux")

    @classmethod
    def _fork_pool_ok(cls, log: logging.Logger, what: str, n: int) -> bool:
        """Whether a fork pool may be opened, warning the caller when it may not.

        :param log: Logger the warning is emitted on.
        :param what: Plural noun for the units the pool would spread, reading
            as "to use up to 8 parallel workers on the <what>".
        :param n: Upper bound on the workers that could run concurrently.
        :return: ``True`` when forking is safe, ``False`` once the caller has
            been warned to fall back to serial execution.
        """
        if cls._fork_is_safe():
            return True
        log.warning(
            "Ignoring the parallelization request: numba resolved the OpenMP "
            "threading layer, which is not fork-safe, and a forked worker "
            "would hang rather than fail. Unset NUMBA_THREADING_LAYER or set "
            "it to 'forksafe' to use up to %d parallel workers on the %s.",
            n, what)
        return False
