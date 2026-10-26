"""The pairwise-HMM kernel gives one result whatever the numba cache holds.

With ``fastmath`` the kernel's reductions were reassociated, and the
non-inlined ``_reset_row`` existed twice in a process that compiled the kernel:
its own library and the copy linked into the caller, which LLVM had reduced in
a different order. A process compiling afresh called the first, one loading the
caller from the on-disk cache called the second, so the posterior-mean TMRCAs
and the node times built from them differed in the last bits between a
cold-cache and a warm-cache run of identical code and input.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

_SCRIPT = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from ancestree._smc_kernel import pair_posterior_mean_tmrcas
rng = np.random.default_rng(1)
T, n = 32, 400
t_rep = np.geomspace(100, 1e6, T)
lam = 2 * 5e-8 * 100 * t_rep
out = pair_posterior_mean_tmrcas(
    rng.poisson(1.5, n)[None], t_rep, lam, np.log(lam), np.exp(-2e-6 * t_rep),
    rng.dirichlet(np.ones(T))[None], step=rng.uniform(0.5, 2, n - 1),
    emit_scale=rng.uniform(0.2, 1, n))[0]
sys.stdout.write(out.tobytes().hex())
"""


def test_a_cold_and_a_warm_numba_cache_give_identical_bits(tmp_path):
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path / "numba_cache")}
    runs = [subprocess.run([sys.executable, "-c", _SCRIPT, str(REPO)],
                           capture_output=True, text=True, env=env, check=True,
                           cwd=tmp_path).stdout
            for _ in range(2)]
    assert any((tmp_path / "numba_cache").rglob("*.nbi")), "nothing was cached"
    assert runs[0] == runs[1]

