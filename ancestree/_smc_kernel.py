"""Numba-JIT forward-backward for the pairwise coalescent HMM.

Backs :class:`~ancestree.local_tree_inference.PairwiseCoalescentHMM`,
which estimates the local TMRCA of a haplotype pair along the sequence
(a discretized PSMC′-style HMM).

The hidden state is a discretized coalescent time (``T`` bins). The per-block
emission is the number of pairwise differences in block ``k``, modelled as
``Poisson(s_k · λ_i)`` with ``λ_i = 2·μ·t_i·block`` the expected differences
over a fully exposed block of ``block`` base pairs at bin representative time
``t_i`` in generations, and ``s_k`` the block's exposure scale: the fraction of
its base pairs over which a difference could have been observed for this pair
(callable under the accessibility mask and called in both haplotypes) times its
local mutation rate over the reference ``μ_ref``. It is ``1`` for a fully
observed block at the reference rate. Being a count, not a binary
heterozygosity flag, it does not saturate at deep TMRCAs. The log-likelihood
for an observed count ``c`` is ``-s_k·λ_i + c·log λ_i`` up to terms constant
across bins (``c·log s_k`` and ``log c!``), which cancel in the per-block
normalisation.

The transition is the "reset on recombination"

    A[i, j] = r_i · δ_ij + (1 - r_i) · ν_j,    ν_j ∝ π_j · (1 - r_j)

where ``r_i`` is the per-block probability of no recombination given TMRCA bin
``i`` and ``π`` is the prior over bins. The reset row ``ν`` is the one that
makes ``π`` invariant under ``A``: because ``r_i`` is depth-dependent, resetting
to ``π`` itself would relax the chain toward ``π_j / (1 - r_j)`` and pull the
posterior mean TMRCA toward the present along the sequence, whereas the
coalescent marginal is the same at every position. Being diagonal plus rank one, it lets
each forward / backward step run in ``O(T)``, not ``O(T²)``, so a full
pass costs ``O(n_blocks · T)``.
"""
from __future__ import annotations


import numpy as np


from numba import njit  # noqa: F401

@njit(cache=True, fastmath=True)
def _emission(counts_k, lam, log_lam, s, e):
    """Fill ``e[i] ∝ Poisson(counts_k | s·λ_i)`` for all bins.

    ``s`` is the block's exposure scale, its observable base-pair fraction
    times its local mutation rate over the reference (``1`` for a fully
    observed block at the reference rate). Terms constant across bins are
    dropped and the maximum subtracted before exponentiating, so ``e`` is
    defined only up to a positive per-block factor, which the forward /
    backward scaling absorbs.
    """
    T = lam.shape[0]
    mx = -1.0e300
    for i in range(T):
        ll = -s * lam[i] + counts_k * log_lam[i]
        e[i] = ll
        if ll > mx:
            mx = ll
    for i in range(T):
        e[i] = np.exp(e[i] - mx)


@njit(cache=True, fastmath=True)
def _step_r(r, s, rk):
    """Fill ``rk[i] = r[i] ** s``, the no-recombination probability over a
    step of ``s`` nominal blocks (genetic distance in block units).

    ``s > 1`` (a masked gap or a wider rec-map step) lets more recombination
    accumulate, ``s < 1`` less.
    """
    T = r.shape[0]
    if s == 1.0:
        for i in range(T):
            rk[i] = r[i]
    else:
        for i in range(T):
            rk[i] = r[i] ** s


@njit(cache=True, fastmath=True)
def _reset_row(pi, rk, nu):
    """Fill ``nu[j] ∝ pi[j] · (1 - rk[j])``, the post-recombination row.

    The transition ``A[i, j] = rk[i]·δ_ij + (1 - rk[i])·nu[j]`` leaves ``pi``
    invariant exactly when ``nu`` carries this form, so the marginal over bins
    stays ``pi`` at every block, as the coalescent requires. ``rk`` is the
    no-recombination probability for the step being taken, so ``nu`` is
    recomputed whenever the step width changes.
    """
    T = pi.shape[0]
    tot = 0.0
    for j in range(T):
        v = pi[j] * (1.0 - rk[j])
        nu[j] = v
        tot += v
    if tot > 0.0:
        inv = 1.0 / tot
        for j in range(T):
            nu[j] *= inv
    else:
        for j in range(T):
            nu[j] = pi[j]


@njit(cache=True, inline="always")
def _grid_mean(t_rep, T):
    """Mean of the time grid, the least-committal TMRCA for a dead block."""
    tot = 0.0
    for i in range(T):
        tot += t_rep[i]
    return tot / T


def _pair_posterior_mean_impl(counts, t_rep, lam, log_lam, r, pi, step, mask,
                              escale):
    """Forward-backward posterior-mean TMRCA per block for one pair.

    :data:`_pair_posterior_mean_jit` is the compiled form.

    The transition between consecutive blocks ``k`` and ``k+1`` uses a per-step
    no-recombination probability ``r[i] ** step[k]``. With ``step`` all ones and
    ``mask`` all true this is the classic scalar-``r`` PSMC′ kernel. A block
    with ``mask[k] == False`` emits uniformly, contributing recombination
    distance but no Poisson information.

    :param counts: ``(n_blocks,)`` int, pairwise differences per block.
    :param t_rep: ``(T,)`` float64, representative time of each bin.
    :param lam: ``(T,)`` float64, Poisson mean ``2·μ·t_i·block`` per bin.
    :param log_lam: ``(T,)`` float64, ``log(lam)``, precomputed.
    :param r: ``(T,)`` float64, P(no recombination over one nominal
        block | bin). Raised to ``step[k]`` per transition.
    :param pi: ``(T,)`` float64, prior over bins (sums to 1).
    :param step: ``(n_blocks-1,)`` float64, genetic distance of each
        inter-block step in nominal-block units (``1.0`` = adjacent).
    :param mask: ``(n_blocks,)`` bool, emission active per block.
    :param escale: ``(n_blocks,)`` float64, per-block exposure scale applied
        to the Poisson mean: the block's observable base-pair fraction times
        its local mutation rate over the reference (``1.0`` = a fully observed
        block at the reference rate).
    :return: ``(n_blocks,)`` float64 posterior-mean TMRCA per block.
    """
    n = counts.shape[0]
    T = t_rep.shape[0]
    alpha = np.empty((n, T), dtype=np.float64)
    e = np.empty(T, dtype=np.float64)
    rk = np.empty(T, dtype=np.float64)
    nu = np.empty(T, dtype=np.float64)

    # ---- forward (scaled per block) ----
    if mask[0]:
        _emission(counts[0], lam, log_lam, escale[0], e)
    else:
        for i in range(T):
            e[i] = 1.0
    s0 = 0.0
    for i in range(T):
        a = pi[i] * e[i]
        alpha[0, i] = a
        s0 += a
    # An underflowed normaliser resets the row to the prior.
    if s0 > 0.0:
        inv = 1.0 / s0
        for i in range(T):
            alpha[0, i] *= inv
    else:
        for i in range(T):
            alpha[0, i] = pi[i]

    for k in range(1, n):
        if mask[k]:
            _emission(counts[k], lam, log_lam, escale[k], e)
        else:
            for i in range(T):
                e[i] = 1.0
        _step_r(r, step[k - 1], rk)
        _reset_row(pi, rk, nu)
        # S = sum_i alpha[k-1, i] * (1 - r_i)
        S = 0.0
        for i in range(T):
            S += alpha[k - 1, i] * (1.0 - rk[i])
        sk = 0.0
        for j in range(T):
            a = (alpha[k - 1, j] * rk[j] + nu[j] * S) * e[j]
            alpha[k, j] = a
            sk += a
        if sk > 0.0:
            inv = 1.0 / sk
            for j in range(T):
                alpha[k, j] *= inv
        else:
            for j in range(T):
                alpha[k, j] = pi[j]

    # ---- backward + posterior-mean accumulation ----
    pmean = np.empty(n, dtype=np.float64)
    beta = np.empty(T, dtype=np.float64)
    beta_next = np.empty(T, dtype=np.float64)
    for i in range(T):
        beta[i] = 1.0

    g_sum = 0.0
    pm = 0.0
    for i in range(T):
        g = alpha[n - 1, i] * beta[i]
        g_sum += g
        pm += g * t_rep[i]
    # block_tmrcas reserves 0.0 for "never written", so a dead block takes
    # the grid mean.
    pmean[n - 1] = pm / g_sum if g_sum > 0.0 else _grid_mean(t_rep, T)

    for k in range(n - 2, -1, -1):
        if mask[k + 1]:
            _emission(counts[k + 1], lam, log_lam, escale[k + 1], e)
        else:
            for i in range(T):
                e[i] = 1.0
        _step_r(r, step[k], rk)
        _reset_row(pi, rk, nu)
        # Q = sum_j nu_j * e_{k+1}(j) * beta_{k+1}(j)
        Q = 0.0
        for j in range(T):
            Q += nu[j] * e[j] * beta[j]
        bs = 0.0
        for i in range(T):
            b = rk[i] * e[i] * beta[i] + (1.0 - rk[i]) * Q
            beta_next[i] = b
            bs += b
        # An underflowed normaliser resets the row to a flat one.
        if bs > 0.0:
            inv = 1.0 / bs
            for i in range(T):
                beta[i] = beta_next[i] * inv
        else:
            for i in range(T):
                beta[i] = 1.0 / T

        g_sum = 0.0
        pm = 0.0
        for i in range(T):
            g = alpha[k, i] * beta[i]
            g_sum += g
            pm += g * t_rep[i]
        pmean[k] = pm / g_sum if g_sum > 0.0 else _grid_mean(t_rep, T)

    return pmean


_pair_posterior_mean_jit = njit(cache=True, fastmath=True)(
    _pair_posterior_mean_impl
)


def pair_posterior_mean_tmrca(
    counts: np.ndarray,
    t_rep: np.ndarray,
    lam: np.ndarray,
    log_lam: np.ndarray,
    r: np.ndarray,
    pi: np.ndarray,
    step: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    escale: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior-mean TMRCA per block for one haplotype pair.

    Thin dispatcher over the compiled kernel: coerces every array to the
    contiguous dtype the kernel is specialised on and fills the optional
    arguments with their neutral defaults.

    :param counts: ``(n_blocks,)`` number of pairwise differences per block.
    :param t_rep: ``(T,)`` representative bin times (generations).
    :param lam: ``(T,)`` Poisson mean ``2·μ·t_i·block`` per bin.
    :param log_lam: ``(T,)`` ``log(lam)``.
    :param r: ``(T,)`` P(no recombination over one nominal block | bin).
    :param pi: ``(T,)`` prior over bins.
    :param step: optional ``(n_blocks-1,)`` genetic distance of each
        inter-block step in nominal-block units. ``None`` is all ones
        (adjacent uniform blocks).
    :param mask: optional ``(n_blocks,)`` bool of callable blocks. ``None``
        is all callable. Non-callable blocks emit uniformly.
    :param escale: optional ``(n_blocks,)`` per-block exposure scale applied to
        the Poisson mean: the block's observable base-pair fraction times its
        local mutation rate over the reference. ``None`` is all ones (fully
        observed blocks at the reference rate).
    :return: ``(n_blocks,)`` posterior-mean TMRCA (generations).
    """
    counts = np.ascontiguousarray(counts, dtype=np.int64)
    t_rep = np.ascontiguousarray(t_rep, dtype=np.float64)
    lam = np.ascontiguousarray(lam, dtype=np.float64)
    log_lam = np.ascontiguousarray(log_lam, dtype=np.float64)
    r = np.ascontiguousarray(r, dtype=np.float64)
    pi = np.ascontiguousarray(pi, dtype=np.float64)
    n = counts.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if step is None:
        step = np.ones(n - 1, dtype=np.float64)
    else:
        step = np.ascontiguousarray(step, dtype=np.float64)
    if mask is None:
        mask = np.ones(n, dtype=np.bool_)
    else:
        mask = np.ascontiguousarray(mask, dtype=np.bool_)
    if escale is None:
        escale = np.ones(n, dtype=np.float64)
    else:
        escale = np.ascontiguousarray(escale, dtype=np.float64)
    return _pair_posterior_mean_jit(
        counts, t_rep, lam, log_lam, r, pi, step, mask, escale
    )
