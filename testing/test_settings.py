"""Package-wide :class:`~ancestree.settings.Settings` switches."""
import multiprocessing
import os
import types

import msprime

import pytest
import ancestree as anc

from ancestree import Settings
from ancestree.inference import FixedTreeInference
from ancestree.models import JC69
from ancestree.sites import Site
from ancestree.trees import OutgroupLadderTree

from testing._helpers import no_counts


@pytest.fixture(autouse=True)
def restore_settings():
    """Reset both switches after every test so state never leaks."""
    yield
    Settings.disable_pbar = False
    Settings.parallelize = None


def _fittable(**overrides):
    ingroup, outgroup = ["i0", "i1"], ["o1", "o2"]
    sites = [
        Site(chrom="1", pos=i + 1, alleles=("A", "T"),
             tip_alleles={"i0": "T", "i1": "A", "o1": "A", "o2": "A"})
        for i in range(3)
    ]
    kwargs = dict(
        tree=OutgroupLadderTree(ingroup, outgroup), n_target_sites=3,
        progress=False, n_starts=2,
    )
    kwargs.update(overrides)
    return FixedTreeInference(sites, JC69(), no_counts(), **kwargs)


class TestDefaults:
    def test_defaults_defer_to_caller(self):
        assert Settings.disable_pbar is False
        assert Settings.parallelize is None


class TestUseParallel:
    @pytest.mark.parametrize("requested", [True, False])
    def test_none_defers_to_the_caller(self, requested):
        """``None`` leaves the caller's own argument in charge."""
        Settings.parallelize = None
        assert Settings._use_parallel(requested) is requested

    @pytest.mark.parametrize("requested", [True, False])
    def test_false_overrides_an_explicit_request(self, requested):
        """``False`` disables everywhere, even where parallelism was asked for."""
        Settings.parallelize = False
        assert Settings._use_parallel(requested) is False

    @pytest.mark.parametrize("requested", [True, False])
    def test_true_enables_even_unrequested(self, requested):
        """``True`` enables everywhere, including where it was not asked for."""
        Settings.parallelize = True
        assert Settings._use_parallel(requested) is True


class TestResolveNWorkers:
    @pytest.mark.parametrize("requested", [None, 1, 4])
    def test_false_forces_a_single_worker(self, requested):
        Settings.parallelize = False
        assert Settings._resolve_n_workers(requested) == 1

    @pytest.mark.parametrize("requested,expected", [(None, 1), (1, 1), (4, 4)])
    def test_none_passes_the_count_through(self, requested, expected):
        Settings.parallelize = None
        assert Settings._resolve_n_workers(requested) == expected

    @pytest.mark.parametrize("requested", [None, 1])
    def test_true_raises_an_unset_count_to_the_cpu_count(self, requested):
        """Forcing parallelism on must raise the count too, else the pool
        would still run with a single worker."""
        Settings.parallelize = True
        assert Settings._resolve_n_workers(requested) == (os.cpu_count() or 1)

    def test_true_respects_an_explicit_count(self):
        Settings.parallelize = True
        assert Settings._resolve_n_workers(4) == 4

    def test_a_daemonic_process_resolves_to_one_worker(self, monkeypatch):
        monkeypatch.setattr(Settings, "parallelize", None)
        monkeypatch.setattr(multiprocessing, "current_process",
                            lambda: types.SimpleNamespace(daemon=True))
        assert Settings._resolve_n_workers(8) == 1


class TestKillSwitchReachesTheFit:
    def test_forces_serial_multistart(self, monkeypatch):
        """``parallelize=False`` keeps ``fit()`` off the worker-pool path."""
        inf = _fittable(parallelize=True)
        monkeypatch.setattr(
            type(inf), "_run_parallel",
            lambda self, starts: pytest.fail("pool used despite the kill switch"),
        )
        Settings.parallelize = False
        inf.fit()

    def test_pool_used_when_switch_defers(self, monkeypatch):
        """Without the switch, an explicit request still reaches the pool."""
        inf = _fittable(parallelize=True)
        used = []
        monkeypatch.setattr(
            type(inf), "_run_parallel",
            lambda self, starts: used.append(len(starts)) or [
                type(inf)._run_one(inf, x0) for x0 in starts
            ],
        )
        inf.fit()
        assert used == [2]


SAMPLES = 8
INGROUP = tuple(f"s{i}" for i in range(6))
OUTGROUP = ("s6", "s7")
SAMPLE_MAP = {f"s{i}": i for i in range(SAMPLES)}


@pytest.fixture
def arg():
    ts = msprime.sim_ancestry(samples=SAMPLES, sequence_length=100_000, ploidy=1,
                              recombination_rate=1e-8, population_size=1e4,
                              random_seed=3)
    return msprime.sim_mutations(ts, rate=1e-7, random_seed=3)


class TestADaemonicWorkerDoesNotOpenItsOwnPool:
    """A segment worker is daemonic, and a daemon may not have children.

    The segment pool is opened on a plain infer(), and the inner ARG
    inference each worker builds must resolve to one worker even under
    Settings.parallelize.
    """

    def test_a_worker_resolves_to_one(self):
        assert Settings._resolve_n_workers(8, force_serial=True) == 1

    def test_local_tree_runs_under_the_global_switch(self, arg, tmp_path):
        vcf = tmp_path / "seg.vcf"
        with open(vcf, "w") as fh:
            arg.write_vcf(fh, individual_names=list(SAMPLE_MAP),
                          position_transform="legacy")
        Settings.parallelize = True
        inf = anc.LocalTreeInference(
            str(vcf), JC69(), mu=1e-7, rec_rate=1e-8, sequence_length=100_000,
            window=20_000, n_ensemble=None,
            ingroup_samples=INGROUP, outgroup_samples=OUTGROUP)
        assert sum(1 for _ in inf.infer()) > 0
