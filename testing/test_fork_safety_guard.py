"""Forking after a numba parallel kernel must not hang.

GNU OpenMP's thread pool is not fork-safe: a child forked once a
``parallel=True`` kernel has run dies inside its first parallel region, and
``multiprocessing.Pool`` has no worker-death detection, so the parent blocks
forever instead of raising. That applies to the ``omp`` layer on Linux alone,
which is also why numba offers ``omp`` under its own ``forksafe`` setting
everywhere else. Both fork sites degrade to serial rather than hanging.
"""
from ancestree.settings import Settings


def test_the_default_layer_is_fork_safe():
    assert Settings._fork_is_safe() is True


def test_openmp_on_linux_is_reported_unsafe(monkeypatch):
    import numba
    monkeypatch.setattr(numba, "threading_layer", lambda: "omp")
    monkeypatch.setattr("ancestree.settings.sys.platform", "linux")
    assert Settings._fork_is_safe() is False


def test_openmp_elsewhere_is_reported_safe(monkeypatch):
    """The macOS and Windows OpenMP runtimes fork cleanly, so the pool stands."""
    import numba
    monkeypatch.setattr(numba, "threading_layer", lambda: "omp")
    monkeypatch.setattr("ancestree.settings.sys.platform", "darwin")
    assert Settings._fork_is_safe() is True


def test_an_unavailable_layer_is_treated_as_safe(monkeypatch):
    """Before any kernel has run numba raises. That must not block forking."""
    import numba

    def _raise():
        raise ValueError("threading layer not initialized")

    monkeypatch.setattr(numba, "threading_layer", _raise)
    assert Settings._fork_is_safe() is True
