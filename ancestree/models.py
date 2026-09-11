"""Substitution models: :class:`~ancestree.models.SubstitutionModel` ABC plus
:class:`~ancestree.models.JC69`, :class:`~ancestree.models.K2`,
:class:`~ancestree.models.F81`, :class:`~ancestree.models.HKY`, and
:class:`~ancestree.models.GTR`.

A model exposes a rate matrix
:meth:`SubstitutionModel.Q() <ancestree.models.SubstitutionModel.Q>` and a
transition-probability function
:meth:`SubstitutionModel.transition_probs() <ancestree.models.SubstitutionModel.transition_probs>`,
the exact ``P(t) = exp(Q * t)``, which admits any number of substitutions per
branch. Every model normalises ``Q`` to ``sum_i pi_i (-Q[i, i]) = 1``, where
``pi_i`` is the equilibrium frequency of state ``i`` and ``-Q[i, i]`` its escape
rate, so one unit of branch length is one expected substitution per site.

Scalar calls are cached per model instance. See
:meth:`SubstitutionModel.transition_probs() <ancestree.models.SubstitutionModel.transition_probs>`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import cast

import numpy as np
import scipy.linalg


from typing import TYPE_CHECKING

from ancestree import STATE_INDEX, STATES, TRANSITION_PAIRS
from ancestree._repr import ReprMixin

if TYPE_CHECKING:
    from ancestree.sites import BaseComposition, Site  # noqa: F401  (cross-refs + annotations)

_CACHE_MISS = object()  # sentinel distinguishing "absent" from a cached None

#: Box the ML fit searches for kappa, and the range an empirical estimate is
#: clamped into. A pinned kappa is not constrained by it.
_KAPPA_CLAMP: tuple[float, float] = (0.1, 50.0)

class SubstitutionModel(ReprMixin, ABC):
    """Continuous-time Markov substitution model on a fixed alphabet.

    Carries the structural substitution parameters (κ for K2/HKY,
    none for JC69/F81). The base composition :math:`\\pi` is passed to
    :meth:`Q` / :meth:`transition_probs` at call time. The per-generation
    rate :math:`\\mu` is applied by the inference layer, which scales
    branch lengths via :attr:`Tree.time_scale <ancestree.trees.Tree.time_scale>`.
    """

    #: The model alphabet, :data:`~ancestree.STATES`.
    states: tuple[str, ...] = STATES
    #: Number of states, ``len(states)``.
    n_states: int = 4

    TRANSITION_CACHE_MAXSIZE: int = 2048
    """LRU cap on :meth:`transition_probs` results, per instance."""

    def __init__(self) -> None:
        """Install the per-instance transition-probability cache."""
        self._install_cache()

    @staticmethod
    def _pi_to_key(pi: "np.ndarray | None") -> tuple[float, ...] | None:
        """Coerce a ``pi`` array to a hashable cache key (or ``None`` for default)."""
        if pi is None:
            return None
        return tuple(float(x) for x in pi)

    @staticmethod
    def _validate_pi(pi: "Sequence[float] | np.ndarray") -> np.ndarray:
        """Coerce ``pi`` to a length-4 simplex vector and validate it."""
        arr = np.asarray(pi, dtype=float)
        if arr.shape != (4,):
            raise ValueError(f"pi must be length-4 (A,C,G,T); got shape {arr.shape}")
        if np.any(arr <= 0):
            raise ValueError(f"pi entries must be strictly positive; got {arr.tolist()}")
        total = arr.sum()
        if not np.isclose(total, 1.0):
            raise ValueError(f"pi must sum to 1; got sum={total}")
        # Renormalise exactly.
        return arr / total

    def _install_cache(self) -> None:
        """(Re-)attach the lru_cache closure. Called from __init__ + after
        unpickling (closures are not picklable, so they are stripped and rebuilt)."""
        @lru_cache(maxsize=self.TRANSITION_CACHE_MAXSIZE)
        def _cached(
            t: float,
            pi_key: tuple[float, ...] | None,
        ) -> np.ndarray:
            """Per-(t, pi) cached transition matrix."""
            pi = np.asarray(pi_key, dtype=float) if pi_key is not None else None
            return self._transition_probs_full(t, pi=pi)

        self._cached_P = _cached
        # Eigendecomposition cache, keyed by Q's bytes.
        self._eig_cache: dict[bytes, "tuple | None"] = {}

    def __getstate__(self) -> dict:
        """Strip the unpicklable lru_cache closure for pickling."""
        state = self.__dict__.copy()
        state.pop("_cached_P", None)  # closure: not picklable, rebuilt on load
        state.pop("_eig_cache", None)  # rebuilt empty on load
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore attributes and rebuild the lru_cache closure."""
        self.__dict__.update(state)
        self._install_cache()

    def _exchangeability(self) -> np.ndarray:
        """Symmetric exchangeability matrix :math:`R` with a zero diagonal.

        :math:`R_{ij}` is the rate of the ``i ↔ j`` substitution relative to
        the other pairs; the overall scale is set by :meth:`_beta_for`, so only
        the ratios enter. The equal exchangeabilities returned here give
        :class:`~ancestree.models.JC69` at uniform :math:`\\pi` and
        :class:`~ancestree.models.F81` at any other :math:`\\pi`.

        :return: Array of shape ``(n_states, n_states)``.
        """
        R = np.ones((self.n_states, self.n_states))
        np.fill_diagonal(R, 0.0)
        return R

    def _beta_for(self, pi_arr: np.ndarray) -> float:
        """Scale making one unit of ``t`` one expected substitution per site.

        Under exchangeabilities :math:`R_{ij}` and equilibrium frequencies
        :math:`\\pi_j` (the probability of state ``j`` at stationarity), state
        ``i`` escapes at the unnormalised rate
        :math:`r_i = \\sum_{j \\neq i} R_{ij}\\,\\pi_j` and a site at
        equilibrium substitutes at :math:`\\sum_i \\pi_i r_i` per unit ``t``.
        Dividing by that mean rate gives
        :math:`\\sum_i \\pi_i\\,(-Q_{ii}) = 1`.

        :param pi_arr: Validated equilibrium frequencies over ``(A, C, G, T)``.
        :return: The scalar :math:`\\beta` multiplying the rate matrix.
        """
        R = self._exchangeability()
        return float(1.0 / float(pi_arr @ (R @ pi_arr)))

    def _build_Q(self, pi_arr: np.ndarray) -> np.ndarray:
        """Rate matrix ``Q[i,j] = β · R[i,j] · π_j`` for ``i ≠ j``.

        :param pi_arr: Validated equilibrium frequencies over ``(A, C, G, T)``.
        :return: Rate matrix of shape ``(n_states, n_states)``, rows summing
            to zero and one expected substitution per unit ``t``.
        """
        Q = self._beta_for(pi_arr) * self._exchangeability() * pi_arr[None, :]
        np.fill_diagonal(Q, -Q.sum(axis=1))
        return Q

    @abstractmethod
    def Q(self, *, pi: "Sequence[float] | np.ndarray | None" = None) -> np.ndarray:
        """Instantaneous rate matrix of shape ``(n_states, n_states)``.

        Off-diagonals are instantaneous substitution rates. Each row sums to
        zero. The diagonal entries ``Q[i,i]`` are negative and equal to
        ``-Σ_{j≠i} Q[i,j]``. The scale is fixed by
        ``Σ_i π_i (-Q[i,i]) = 1``, one expected substitution per unit ``t``.

        :param pi: Optional length-4 stationary base frequencies over
            ``(A, C, G, T)``. Used by the non-symmetric models
            (:class:`~ancestree.models.F81`, :class:`~ancestree.models.HKY`, :class:`~ancestree.models.GTR`). Ignored by the
            symmetric models (:class:`~ancestree.models.JC69`, :class:`~ancestree.models.K2`). Defaults to
            uniform when ``None``.
        :return: Rate matrix.
        """

    def stationary(
        self,
        *,
        pi: "Sequence[float] | np.ndarray | None" = None,
    ) -> np.ndarray:
        """Equilibrium distribution over states (rows of ``π Q = 0``).

        Uniform for the symmetric models :class:`~ancestree.models.JC69` and :class:`~ancestree.models.K2`, whose
        ``Q`` is symmetric whatever frequencies are supplied. :class:`~ancestree.models.F81`,
        :class:`~ancestree.models.HKY` and :class:`~ancestree.models.GTR` override this to return ``pi``.

        :param pi: Ignored here. Accepted only for signature uniformity with
            the overrides. See :meth:`Q`.
        :return: Length-``n_states`` uniform array summing to 1.
        """
        return np.full(self.n_states, 1.0 / self.n_states)

    def transition_probs(
        self,
        t: float | np.ndarray,
        *,
        pi: "Sequence[float] | np.ndarray | None" = None,
    ) -> np.ndarray:
        """Return ``P(t)`` with ``P(t)[i, j] = P(state j at child | state i at parent)``.

        Scalar ``t`` is looked up in an LRU cache keyed by ``(t, pi)``.
        A hit returns the cached matrix unchanged. Array ``t`` bypasses
        the cache (no vector-keyed cache).

        :param t: Branch length(s). May be a scalar or any-shape array. The
            returned array has shape ``t.shape + (n_states, n_states)``.
        :param pi: Optional length-4 base frequencies. See :meth:`Q`.
        :return: Transition probability matrix (or batch of matrices). Rows
            of every matrix sum to 1.
        :raises ValueError: If any branch length ``t`` is negative, which gives
            a non-stochastic ``exp(Qt)`` and signals an invalid tree.
        """
        t_arr = np.asarray(t, dtype=float)
        if not np.all(np.isfinite(t_arr)):
            raise ValueError(
                f"transition_probs requires finite t; got {t!r}"
            )
        if not np.all(t_arr >= 0):
            raise ValueError(
                f"transition_probs requires t >= 0 (and not NaN); got {t!r}"
            )
        if np.ndim(t) == 0:
            pi_key = self._pi_to_key(np.asarray(pi) if pi is not None else None)
            # A read-only view of the cached matrix: the cache is shared across
            # every call at this key, so a caller mutating it would poison them.
            cached = self._cached_P(float(t), pi_key)
            view = cached.view()
            view.flags.writeable = False
            return view
        pi_arr = np.asarray(pi, dtype=float) if pi is not None else None
        return self._transition_probs_full(t, pi=pi_arr)

    def clear_cache(self) -> None:
        """Drop all cached transition matrices on this model instance.

        Any in-place parameter change invalidates every cached ``P(t)``, so
        :meth:`K2.set_free_params`, :meth:`HKY.set_free_params` and
        :meth:`GTR.set_free_params` call this after writing a new value.
        Custom :class:`~ancestree.models.SubstitutionModel` subclasses that mutate state should
        do the same.
        """
        self._cached_P.cache_clear()

    def cache_info(self):
        """LRU cache statistics for this instance's transition-matrix cache.

        :return: ``functools.CacheInfo`` named tuple
            ``(hits, misses, maxsize, currsize)``.
        """
        return self._cached_P.cache_info()

    # ---------------------------------------------------- free-parameter protocol

    @property
    def free_params(self) -> dict[str, tuple[float, tuple[float, float]]]:
        """Model-internal free parameters as ``{name: (initial, (lo, hi))}``.

        Default: empty. Override in subclasses that have parameters the
        :class:`~ancestree.inference.FixedTreeInference` optimiser should fit jointly with the
        tree branch rates.

        :return: Mapping. Empty when no model-internal fitting requested.
        """
        return {}

    def set_free_params(self, values: dict[str, float]) -> None:
        """Update the free parameters from the optimiser's solution.

        Default: no-op. Override in subclasses with non-trivial
        :attr:`free_params`. Implementations should also call
        :meth:`clear_cache` if the change invalidates the transition
        matrix cache.

        :param values: Mapping from parameter name (must be a key in
            :attr:`free_params`) to the new value.
        """

    def warn_if_bounds_hit(self) -> None:
        """Log a warning (via the class logger) if any MLE-fit parameter has
        landed on its bounds, a common symptom of model misspecification or
        insufficient data. Called by
        :class:`~ancestree.inference.FixedTreeInference` after the fit
        completes. Default: no-op (no bounded params).
        """

    def _warn_bound(self, name: str, value: float, bounds: tuple[float, float], bounds_arg: str) -> None:
        """Warn when a fitted parameter landed within 2% of one of its bounds.

        :param name: Parameter name, as it appears in :attr:`free_params`.
        :param value: Fitted value.
        :param bounds: ``(lo, hi)`` box the fit searched within.
        :param bounds_arg: Constructor argument holding the box, named in the
            message as the one to widen.
        """
        lo, hi = bounds
        if value <= lo * 1.02:
            hit, bound = "lower bound", lo
        elif value >= hi * 0.98:
            hit, bound = "upper bound", hi
        else:
            return
        self._log.warning(
            "Fitted %s=%.4g hit the %s %g (within 2%%). Consider widening %s "
            "or investigating model misspecification.",
            name, value, hit, bound, bounds_arg,
        )

    # Above this eigenvector condition number the reconstruction is not
    # trusted and scipy's scaling-and-squaring is used.
    _EIG_COND_LIMIT: float = 1e8

    def _transition_probs_full(
        self,
        t: float | np.ndarray,
        *,
        pi: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Exact ``P(t) = exp(Q·t)``. Scalar or array ``t``.

        Delegates to :meth:`_expm_via_eig`, falling back to per-element
        :func:`scipy.linalg.expm` when that declines the reconstruction.
        """
        Q = self.Q(pi=pi)
        t_arr = np.asarray(t, dtype=float)
        P = self._expm_via_eig(Q, t_arr)
        if P is not None:
            return P
        if t_arr.ndim == 0:
            return scipy.linalg.expm(Q * float(t_arr))
        out = np.empty(t_arr.shape + (self.n_states, self.n_states), dtype=float)
        flat_out = out.reshape(-1, self.n_states, self.n_states)
        for k, tk in enumerate(t_arr.ravel()):
            flat_out[k] = scipy.linalg.expm(Q * float(tk))
        return out

    def _eig_of(self, Q: np.ndarray) -> "tuple | None":
        """Cached ``(λ, V, V⁻¹)`` with ``Q = V diag(λ) V⁻¹``.

        Keyed by ``Q``'s bytes: ``Q`` is invariant across the objective
        evaluations of a rate fit, so it is decomposed once. Returns ``None``
        when ``V`` is too ill-conditioned for the reconstruction to be trusted,
        and that verdict is cached too.

        :param Q: Rate matrix of shape ``(S, S)``.
        :return: ``(λ, V, Vinv)``, or ``None`` if ``Q`` is not usably
            diagonalisable.
        """
        key = Q.tobytes()
        cached = self._eig_cache.get(key, _CACHE_MISS)
        if cached is not _CACHE_MISS:
            return cast("tuple | None", cached)
        lam, V = np.linalg.eig(Q)
        # Q's rows sum to zero, so its largest eigenvalue is exactly 0 and any
        # positive real part is roundoff. exp(lam t) is then 1 at that mode for
        # every t, keeping P(t) stochastic at large t.
        lam = lam - lam.real.max()
        cond = np.linalg.cond(V)
        if not np.isfinite(cond) or cond > self._EIG_COND_LIMIT:
            self._eig_cache[key] = None
            return None
        Vinv = np.linalg.inv(V)
        if len(self._eig_cache) > 256:  # bound growth on a fit_kappa/rates run
            self._eig_cache.clear()
        self._eig_cache[key] = (lam, V, Vinv)
        return lam, V, Vinv

    def real_eig(
        self,
        *,
        pi: "Sequence[float] | np.ndarray | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray] | None":
        """Real ``(λ, V, V⁻¹)`` for ``exp(Q·t)``, for callers that exponentiate
        branch lengths themselves.

        For a reversible ``Q`` the symmetrised form
        ``diag(sqrt(pi)) Q diag(1/sqrt(pi))`` is decomposed with
        :func:`numpy.linalg.eigh`, which is real and orthonormal on the
        degenerate eigenspaces of JC69, K2 and F81. A non-reversible ``Q``
        yields ``None``, and the caller falls back to :meth:`transition_probs`.

        :param pi: Equilibrium frequencies, for models that take them.
        :return: ``(λ, V, Vinv)`` as ``float64``, or ``None``.
        """
        Q = self.Q(pi=np.asarray(pi, dtype=float) if pi is not None else None)
        # The stationary distribution is Q's left null vector, taken from an
        # SVD, which is stable when eigenvalues repeat.
        _, sv, vt = np.linalg.svd(Q.T)
        if sv[-1] > 1e-8 * max(sv[0], 1.0):
            return None  # no stationary distribution: not a usable rate matrix
        p_st = np.abs(vt[-1])
        tot = p_st.sum()
        if tot <= 0 or not np.isfinite(tot):
            return None
        p_st = p_st / tot
        if p_st.min() <= 0:
            return None
        # Detailed balance, pi_i Q_ij == pi_j Q_ji, makes Q similar to a
        # symmetric matrix. Every model here is reversible.
        flux = p_st[:, None] * Q
        if np.abs(flux - flux.T).max() > 1e-9 * max(np.abs(Q).max(), 1.0):
            return None  # non-reversible: no symmetric form to exploit
        # B = D Q D^-1 with D = diag(sqrt(pi)) is symmetric, so eigh gives real
        # eigenvalues and an orthonormal basis on degenerate eigenspaces.
        d = np.sqrt(p_st)
        B = (d[:, None] * Q) / d[None, :]
        lam, U = np.linalg.eigh(0.5 * (B + B.T))
        # Q's rows sum to zero, so its largest eigenvalue is exactly 0 and a
        # positive real part is roundoff, as in _eig_of.
        lam = lam - lam.max()
        V = U / d[:, None]
        Vinv = U.T * d[None, :]
        if not np.allclose(V @ (lam[:, None] * Vinv), Q, atol=1e-9, rtol=1e-7):
            return None  # reconstruction failed. Caller falls back
        return (np.ascontiguousarray(lam, dtype=np.float64),
                np.ascontiguousarray(V, dtype=np.float64),
                np.ascontiguousarray(Vinv, dtype=np.float64))

    def _expm_via_eig(
        self,
        Q: np.ndarray,
        t_arr: np.ndarray,
    ) -> "np.ndarray | None":
        """Vectorised ``exp(Q·t)`` over a branch-length array via the
        eigendecomposition of ``Q``.

        With ``Q = V diag(λ) V⁻¹`` from :meth:`_eig_of`, the whole array is
        exponentiated in one pass as ``P(t) = V diag(e^{λt}) V⁻¹``.

        :param Q: Rate matrix of shape ``(S, S)``.
        :param t_arr: Branch length(s), any shape (scalar ``ndim == 0`` allowed).
        :return: ``exp(Q·t)`` of shape ``t_arr.shape + (S, S)``, or ``None``
            when ``Q``'s eigenvector matrix is too ill-conditioned for the
            reconstruction to be trusted (caller falls back to scipy).
        """
        S = self.n_states
        eig = self._eig_of(Q)
        if eig is None:
            return None
        lam, V, Vinv = eig
        flat = t_arr.ravel()
        exp_lt = np.exp(np.outer(flat, lam))  # (N, S), complex iff λ is
        # P[n] = V diag(exp_lt[n]) Vinv, batched over branches.
        P = (V[None, :, :] * exp_lt[:, None, :]) @ Vinv
        # exp(Q·t) is non-negative. Clip reconstruction noise (e.g. ~1e-16
        # off-diagonals at t = 0) so rows stay valid probability vectors.
        P = np.maximum(P.real, 0.0).reshape(t_arr.shape + (S, S))
        return P


class _KappaMixin:
    """The transition / transversion ratio and its ML-fitting protocol.

    Mixed into the two models carrying a :math:`\\kappa` parameter
    (:class:`~ancestree.models.K2`, :class:`~ancestree.models.HKY`), which differ only in their ``Q``.

    :param kappa: Transition / transversion ratio. Must be positive.
    :param fit_kappa: Expose ``kappa`` through :attr:`free_params` so
        :meth:`FixedTreeInference.fit() <ancestree.inference.FixedTreeInference.fit>` estimates it.
    :param kappa_bounds: ``(lo, hi)`` box the ML fit searches within.
    """

    # Provided by the host SubstitutionModel.
    if TYPE_CHECKING:
        n_states: int

        def clear_cache(self) -> None: ...

        def _warn_bound(self, name: str, value: float, bounds: tuple[float, float], bounds_arg: str) -> None: ...

    kappa: float
    fit_kappa: bool
    kappa_bounds: tuple[float, float]

    #: Index pairs of the transitions A↔G and C↔T (A=0, C=1, G=2, T=3), the
    #: entries of ``Q`` scaled by κ.
    TRANSITION_INDEX_PAIRS: frozenset[tuple[int, int]] = frozenset(
        (STATE_INDEX[a], STATE_INDEX[b]) for a, b in TRANSITION_PAIRS
    )

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"kappa": self.kappa}

    def _exchangeability(self) -> np.ndarray:
        """Exchangeabilities of ``κ`` on the transitions and ``1`` elsewhere."""
        R = np.ones((self.n_states, self.n_states))
        for i, j in self.TRANSITION_INDEX_PAIRS:
            R[i, j] = self.kappa
        np.fill_diagonal(R, 0.0)
        return R

    def __init__(
        self,
        kappa: float = 2.0,
        *,
        fit_kappa: bool = False,
        kappa_bounds: tuple[float, float] = _KAPPA_CLAMP,
    ) -> None:
        """Validate ``kappa``. Store the free-parameter config."""
        if not np.isfinite(kappa) or kappa <= 0:
            raise ValueError(f"kappa must be positive and finite, got {kappa}")
        super().__init__()
        self.kappa = float(kappa)
        self.fit_kappa = bool(fit_kappa)
        lo, hi = (float(b) for b in kappa_bounds)
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 < lo < hi):
            raise ValueError(
                f"kappa_bounds must be (lo, hi) with 0 < lo < hi; got "
                f"{kappa_bounds!r}")
        self.kappa_bounds = (lo, hi)

    @property
    def free_params(self) -> dict[str, tuple[float, tuple[float, float]]]:
        """``{"kappa": (initial, bounds)}`` when ``fit_kappa=True``. Else ``{}``."""
        if self.fit_kappa:
            return {"kappa": (self.kappa, self.kappa_bounds)}
        return {}

    def set_free_params(self, values: dict[str, float]) -> None:
        """Update ``kappa`` (if provided) and invalidate the transition cache.

        :raises ValueError: If ``kappa`` is not strictly positive.
        """
        if "kappa" in values:
            new_kappa = float(values["kappa"])
            if new_kappa <= 0:
                raise ValueError(f"kappa must be positive, got {new_kappa}")
            self.kappa = new_kappa
            self.clear_cache()

    def warn_if_bounds_hit(self) -> None:
        """Warn if MLE-fit ``kappa`` landed within 2% of ``kappa_bounds``."""
        if not self.fit_kappa:
            return
        self._warn_bound("kappa", self.kappa, self.kappa_bounds, "kappa_bounds")


class _PiModel(SubstitutionModel):
    """Base for the models whose ``Q`` is driven by a base composition.

    Subclasses supply :meth:`_exchangeability`. Resolving ``π`` and
    assembling ``Q`` from it is shared.
    """

    def _resolve_pi(
        self, pi: "Sequence[float] | np.ndarray | None",
    ) -> tuple[np.ndarray, float]:
        """Validate ``pi`` (or use the uniform default) and recompute ``β``.

        :param pi: Base composition, or ``None`` for uniform.
        :return: ``(validated pi, β)``.
        """
        pi_arr = np.full(4, 0.25) if pi is None else self._validate_pi(pi)
        return pi_arr, self._beta_for(pi_arr)

    def Q(self, *, pi: "Sequence[float] | np.ndarray | None" = None) -> np.ndarray:
        """Rate matrix ``Q[i,j] = β · R[i,j] · π_j`` at the supplied composition.

        :param pi: Length-4 base frequencies over ``(A, C, G, T)``. Uniform
            when ``None``.
        :return: Rate matrix, with ``Σ_i π_i (-Q[i,i]) = 1``.
        """
        pi_arr, _ = self._resolve_pi(pi)
        return self._build_Q(pi_arr)

    def stationary(
        self, *, pi: "Sequence[float] | np.ndarray | None" = None,
    ) -> np.ndarray:
        """Return the supplied ``π`` (uniform default when ``pi`` is ``None``)."""
        pi_arr, _ = self._resolve_pi(pi)
        return pi_arr.copy()


class JC69(SubstitutionModel):
    """Jukes–Cantor 1969: all substitutions equiprobable, uniform stationary.

    No structural parameters. Branch lengths are interpreted directly as
    expected substitutions per site: every state escapes at
    :math:`-Q_{ii} = 1`, so :math:`\\sum_i \\pi_i(-Q_{ii}) = 1` at the uniform
    :math:`\\pi`. The per-generation rate :math:`\\mu`, if any, is supplied
    by the inference orchestrator via :attr:`Tree.time_scale <ancestree.trees.Tree.time_scale>`
    on the tree.
    """

    def Q(self, *, pi: "Sequence[float] | np.ndarray | None" = None) -> np.ndarray:
        """JC69 rate matrix: off-diagonals ``1/(S−1)``, diagonals ``−1``.

        The ``pi`` argument is accepted for interface uniformity but
        ignored (JC69's stationary distribution is uniform by definition).
        """
        return self._build_Q(self.stationary())


class K2(_KappaMixin, SubstitutionModel):
    """Kimura 1980 two-parameter model: separate transition / transversion rates.

    Transitions are within-group substitutions (A↔G between purines, C↔T
    between pyrimidines). Transversions are between-group (A↔C, A↔T, C↔G,
    G↔T). Every state escapes at :math:`-Q_{ii} = 1`, so branch
    lengths read as expected substitutions per site, just like
    :class:`~ancestree.models.JC69`.

    :param kappa: Ratio of transition rate to transversion rate.
        ``kappa = 1`` collapses to :class:`~ancestree.models.JC69`. Default ``2.0``.
    :param fit_kappa: When ``True``, expose ``kappa`` as a free parameter
        via the :attr:`free_params` protocol so :class:`~ancestree.inference.FixedTreeInference`
        fits it jointly with the tree branch rates. ``kappa`` then acts as
        the initial value. Default ``False``.
    :param kappa_bounds: ``(lo, hi)`` for L-BFGS-B when ``fit_kappa=True``.
        Default ``(0.1, 50.0)``.
    :raises ValueError: If ``kappa`` is non-positive.
    """

    @classmethod
    def estimate_kappa_from_data(
        cls,
        sites: "Sequence[Site]",
        samples_a: "Sequence[str]",
        samples_b: "Sequence[str]",
    ) -> float:
        """Empirical ``κ`` estimate from per-site transition/transversion ratio.

        For each site, compares the majority allele in ``samples_a`` against
        the majority allele in ``samples_b`` and classifies each difference
        as a transition (A↔G or C↔T) or a transversion, giving
        ``κ̂ = 2 · Ts / Tv``. Multi-hit corrections are ignored, so the
        estimate holds when the divergence between the two groups is small
        (``μ·t ≪ 1``).

        For an estimate at the whole-sample level prefer
        :attr:`BaseComposition.kappa_estimate <ancestree.sites.BaseComposition.kappa_estimate>`,
        which uses every biallelic site.

        :param sites: Polymorphic :class:`~ancestree.sites.Site` records carrying
            tip alleles for both groups.
        :param samples_a: First sample group (e.g. ingroup haplotype ids).
        :param samples_b: Second sample group (e.g. outgroup haplotype ids).
        :return: Empirical κ, clamped into ``(0.1, 50.0)``; the upper bound
            when no transversions are observed.
        :raises ValueError: When no pairwise differences are observed.
        """
        n_ts = n_tv = 0
        for site in sites:
            counts_a = site.count_alleles(samples_a)
            counts_b = site.count_alleles(samples_b)
            if not counts_a or not counts_b:
                continue
            maj_a = counts_a.most_common(1)[0][0]
            maj_b = counts_b.most_common(1)[0][0]
            if maj_a == maj_b:
                continue
            if (maj_a, maj_b) in TRANSITION_PAIRS:
                n_ts += 1
            else:
                n_tv += 1
        if n_ts + n_tv == 0:
            raise ValueError(
                "estimate_kappa_from_data: no pairwise differences "
                "observed between samples_a and samples_b; κ cannot "
                "be estimated."
            )
        if n_tv == 0:
            return _KAPPA_CLAMP[1]
        return float(np.clip(2.0 * n_ts / n_tv, *_KAPPA_CLAMP))

    def Q(self, *, pi: "Sequence[float] | np.ndarray | None" = None) -> np.ndarray:
        """K2 rate matrix: transition rate ``κ/(κ+2)``, transversion rate ``1/(κ+2)``.

        Each row has one transition target and two transversion targets, so
        the escape rate is ``1`` for every state. The ``pi`` argument is
        accepted for interface uniformity but ignored (K2's stationary
        distribution is uniform by definition).
        """
        return self._build_Q(self.stationary())


class F81(_PiModel):
    """Felsenstein 1981: rates proportional to target base frequencies.

    Off-diagonals are :math:`Q_{ij} = \\beta \\, \\pi_j` for ``i ≠ j``, where
    :math:`\\pi_j` is the equilibrium frequency of state ``j`` and
    :math:`\\beta = 1 / (1 - \\sum_k \\pi_k^2)` sets one unit of ``t`` to one
    expected substitution per site. Collapses to
    :class:`~ancestree.models.JC69` when ``pi`` is uniform.

    ``π`` is supplied at call time via :meth:`Q` / :meth:`transition_probs`
    (typically from a :class:`~ancestree.sites.BaseComposition` populated
    by :meth:`BaseComposition.from_polymorphic_sites() <ancestree.sites.BaseComposition.from_polymorphic_sites>`). See the
    package-level workflow in :class:`~ancestree.inference.FixedTreeInference`.
    A uniform default is used when ``pi`` is ``None`` (then F81 collapses
    to :class:`~ancestree.models.JC69`).

    ``P(t)`` has the closed form

    .. math::

        P_{ij}(t) = e^{-\\beta t}\\,\\delta_{ij} + (1 - e^{-\\beta t})\\,\\pi_j,

    so :meth:`transition_probs` skips :func:`scipy.linalg.expm` and evaluates
    this directly.
    """

    def _transition_probs_full(
        self,
        t: float | np.ndarray,
        *,
        pi: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """F81 closed form ``P(t) = e^{-βt} I + (1 - e^{-βt}) 1 π^T``."""
        S = self.n_states
        pi_arr, beta = self._resolve_pi(pi)
        t_arr = np.asarray(t, dtype=float)
        decay = np.exp(-beta * t_arr)  # (...,)
        # P[i,j] = decay·δ_ij + (1-decay)·π_j. Broadcast over t-shape.
        eye = np.eye(S)
        one_pi = np.broadcast_to(pi_arr, (S, S))  # (S, S)
        # Reshape decay for broadcast with (S, S) tail.
        decay_b = decay[..., None, None] if t_arr.ndim > 0 else decay
        off = -np.expm1(-beta * t_arr)
        off_b = off[..., None, None] if t_arr.ndim > 0 else off
        P = decay_b * eye + off_b * one_pi
        return P


class HKY(_KappaMixin, _PiModel):
    """Hasegawa–Kishino–Yano 1985: F81 with a transition/transversion bias.

    Off-diagonals are :math:`Q_{ij} = \\beta \\, \\kappa \\, \\pi_j` when
    ``i ↔ j`` is a transition (A↔G or C↔T), and :math:`Q_{ij} = \\beta\\,
    \\pi_j` for transversions, with :math:`\\pi_j` the equilibrium frequency of
    state ``j``. ``β`` sets one unit of ``t`` to one expected substitution
    per site:

    .. math::

        \\beta = \\frac{1}{\\sum_i \\pi_i r_i},
        \\quad r_i = (\\kappa - 1)\\,\\pi_{\\mathrm{partner}(i)}
                       + (1 - \\pi_i),

    where :math:`r_i` is the unnormalised escape rate at state ``i`` and
    ``partner(i)`` is its transition partner, so
    :math:`\\sum_i \\pi_i r_i = 1 - \\sum_k \\pi_k^2
    + 2(\\kappa - 1)(\\pi_A\\pi_G + \\pi_C\\pi_T)`.
    ``P(t)`` has no closed form in general, so
    :meth:`transition_probs` uses the inherited numerical default. HKY
    collapses to :class:`~ancestree.models.F81` at ``κ = 1`` and
    to :class:`~ancestree.models.JC69` additionally at uniform ``π``.

    ``π`` is supplied at call time (see :class:`~ancestree.models.F81`). Construct
    :class:`~ancestree.models.HKY` with just ``kappa``, and pass the empirical base
    composition from :class:`~ancestree.sites.BaseComposition` to
    :meth:`Q` / :meth:`transition_probs`.

    :param kappa: Transition/transversion rate ratio. ``κ = 1`` collapses
        to :class:`~ancestree.models.F81`. Default ``2.0``.
    :param fit_kappa: Expose ``kappa`` as a free parameter for joint
        MLE in :class:`~ancestree.inference.FixedTreeInference` (mirrors
        :class:`~ancestree.models.K2`). Default ``False``.
    :param kappa_bounds: ``(lo, hi)`` for L-BFGS-B when ``fit_kappa=True``.
        Default ``(0.1, 50.0)``.
    :raises ValueError: If ``kappa`` is non-positive.
    """


class GTR(_PiModel):
    """General Time-Reversible model with six free exchangeability rates.

    Off-diagonals are :math:`Q_{ij} = \\beta \\, r_{ij} \\, \\pi_j` for
    ``i ≠ j``, where ``r_ij = r_ji`` is the symmetric exchangeability rate
    between states ``i`` and ``j`` and ``π`` is the base composition supplied
    at call time (see :class:`~ancestree.models.F81`). The six rates, in canonical order, are
    :math:`(r_\\mathrm{AC}, r_\\mathrm{AG}, r_\\mathrm{AT}, r_\\mathrm{CG},
    r_\\mathrm{CT}, r_\\mathrm{GT})`. ``β`` sets one unit of ``t`` to one
    expected substitution per site, as in :class:`~ancestree.models.F81` and
    :class:`~ancestree.models.HKY`, so the overall rate scale is carried by
    the branch lengths. ``P(t)`` has no closed form, so
    :meth:`transition_probs` uses the inherited numerical default.

    GTR collapses to :class:`~ancestree.models.HKY` when ``r_AG = r_CT`` and the four
    transversion rates are equal, to :class:`~ancestree.models.F81` when all six rates are
    equal, and to :class:`~ancestree.models.JC69` when ``π`` is also uniform.

    :param rates: Length-6 vector of exchangeability rates in canonical order
        ``(AC, AG, AT, CG, CT, GT)``. Default all ones.
    :param fit_rates: Expose the rates as free parameters for joint MLE in
        :class:`~ancestree.inference.FixedTreeInference`. ``True`` holds
        ``rate_AG`` at its supplied value as the reference scale and fits the
        other five. A length-6 sequence of booleans selects the free rates
        explicitly. Default ``False``.
    :param fixed_rates: Partial mapping ``{rate_name: value}`` pinning those,
        with ``rate_AG`` held as the reference scale unless pinned explicitly,
        rates while the rest vary. Cannot be combined with a sequence-style
        ``fit_rates``. Default ``None``.
    :param rate_bounds: ``(lo, hi)`` for L-BFGS-B when any rate is free.
        Default ``(1e-3, 100.0)``.
    :raises ValueError: If ``rates`` is not a length-6 strictly positive
        vector, if ``fixed_rates`` is combined with a sequence-style
        ``fit_rates`` or names an unknown rate or pins one to a non-positive
        value, if a sequence ``fit_rates`` is not of length 6, or if
        ``rate_bounds`` is not ``(lo > 0, hi > lo)``.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"rates": self.rates}

    #: Canonical ordering of the six exchangeability rates over (A, C, G, T),
    #: each entry the (lo, hi) STATES-index pair of the upper triangle.
    RATE_PAIRS: tuple[tuple[int, int], ...] = (
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
    )
    #: Parameter names of the six rates, in :attr:`RATE_PAIRS` order.
    RATE_NAMES: tuple[str, ...] = (
        "rate_AC", "rate_AG", "rate_AT", "rate_CG", "rate_CT", "rate_GT",
    )

    #: The six exchangeability rates, in :attr:`RATE_NAMES` order.
    rates: np.ndarray

    fit_mask: tuple[bool, ...]
    """Per-rate fit flag in :attr:`RATE_NAMES` order, resolved at construction
    from ``fit_rates`` / ``fixed_rates``. ``True`` entries are exposed to the
    optimiser via :attr:`free_params`."""

    #: ``(lower, upper)`` bounds applied to every fitted rate.
    rate_bounds: tuple[float, float]

    @staticmethod
    def _validate_rates(rates: "Sequence[float] | np.ndarray") -> np.ndarray:
        """Coerce ``rates`` to a length-6 positive vector in GTR canonical order."""
        # Copy: the rates are written in place.
        arr = np.array(rates, dtype=float)
        if arr.shape != (6,):
            raise ValueError(
                f"rates must be length-6 (AC, AG, AT, CG, CT, GT); got shape {arr.shape}"
            )
        if np.any(arr <= 0):
            raise ValueError(
                f"rate entries must be strictly positive; got {arr.tolist()}"
            )
        return arr

    def __init__(
        self,
        rates: "Sequence[float] | np.ndarray | None" = None,
        *,
        fit_rates: "bool | Sequence[bool]" = False,
        fixed_rates: "Mapping[str, float] | None" = None,
        rate_bounds: tuple[float, float] = (1e-3, 100.0),
    ) -> None:
        """Validate the parameters and resolve the per-rate fit mask."""
        super().__init__()
        if rates is None:
            rates = np.ones(6)
        self.rates = self._validate_rates(rates)
        if fixed_rates is not None and not isinstance(fit_rates, bool):
            raise ValueError(
                "Cannot combine `fixed_rates=` with a Sequence-style "
                "`fit_rates=`; pick one. Use `fixed_rates={...}` for the "
                "subset-by-name shortcut, or `fit_rates=[...]` for the "
                "explicit length-6 mask."
            )
        if fixed_rates is not None:
            mask = [True] * 6
            for name, value in dict(fixed_rates).items():
                if name not in self.RATE_NAMES:
                    raise ValueError(
                        f"fixed_rates: unknown rate name {name!r}; "
                        f"expected one of {list(self.RATE_NAMES)}"
                    )
                v = float(value)
                if v <= 0:
                    raise ValueError(
                        f"fixed_rates[{name!r}] must be positive, got {v}"
                    )
                idx = self.RATE_NAMES.index(name)
                self.rates[idx] = v
                mask[idx] = False
            # rate_AG stays fixed as the reference scale whenever it was not
            # pinned explicitly and some other rate is free.
            if "rate_AG" not in fixed_rates and any(mask):
                mask[1] = False
            self.fit_mask = tuple(mask)
        elif isinstance(fit_rates, bool):
            # Fit every rate except the reference (rate_AG, index 1) so the
            # overall rate scale stays identified by the branch lengths.
            mask = [bool(fit_rates)] * 6
            if fit_rates:
                mask[1] = False
            self.fit_mask = tuple(mask)
        else:
            mask_arr = list(fit_rates)
            if len(mask_arr) != 6:
                raise ValueError(
                    f"fit_rates as a Sequence must be length 6; got {len(mask_arr)}"
                )
            self.fit_mask = tuple(bool(b) for b in mask_arr)
        if rate_bounds[0] <= 0 or rate_bounds[1] <= rate_bounds[0]:
            raise ValueError(
                f"rate_bounds must be (lo > 0, hi > lo); got {rate_bounds}"
            )
        self.rate_bounds = (float(rate_bounds[0]), float(rate_bounds[1]))

    def _exchangeability(self) -> np.ndarray:
        """The six exchangeability rates laid out symmetrically over the states."""
        R = np.zeros((self.n_states, self.n_states), dtype=float)
        for (a, b), r in zip(self.RATE_PAIRS, self.rates):
            R[a, b] = R[b, a] = r
        return R

    # ---------------------------------------------------- free-parameter protocol

    @classmethod
    def empirical_from_sites(
        cls,
        sites: "Sequence",
        base_composition: "BaseComposition",
        ingroup_samples: "Sequence[str]",
        *,
        fit_rates: "bool | Sequence[bool]" = False,
        rate_bounds: tuple[float, float] = (1e-3, 100.0),
    ) -> "GTR":
        """Construct a :class:`~ancestree.models.GTR` whose rates are estimated from segregating-pair counts.

        For each unordered allele pair :math:`\\{i, j\\}` the number of sites
        whose ingroup alleles are exactly :math:`\\{i, j\\}` is divided by
        :math:`\\pi_i \\pi_j`, where :math:`\\pi` is the base composition. The
        six quotients are proportional to the exchangeability rates and are
        normalised to :math:`r_\\mathrm{AG} = 1`, the reference convention of
        ``fit_rates=True``. This generalises the empirical κ of :class:`~ancestree.models.K2`
        and :class:`~ancestree.models.HKY` and serves as an initialiser for the joint fit.

        :param sites: Polymorphic :class:`~ancestree.sites.Site` records.
            Sites with other than two distinct ingroup alleles are ignored.
        :param base_composition: The :class:`~ancestree.sites.BaseComposition`
            supplying :math:`\\pi`, the same one the inference uses.
        :param ingroup_samples: Ingroup sample identifiers. Only these tips
            are counted, since an outgroup allele is divergence, not
            polymorphism.
        :param fit_rates: Forwarded to ``__init__``. Default ``False``.
        :param rate_bounds: Forwarded to ``__init__``.
        :return: A new :class:`~ancestree.models.GTR` with the empirical rate vector.
        :raises ValueError: If any pair has a zero count or a zero
            :math:`\\pi_i \\pi_j`.
        """
        from ancestree import STATE_INDEX
        # Map state-index pair to rate-pair index in canonical order.
        pair_to_idx = {pair: i for i, pair in enumerate(cls.RATE_PAIRS)}
        counts = np.zeros(6, dtype=float)
        for s in sites:
            observed_states = set(s.count_alleles(ingroup_samples))
            if len(observed_states) != 2:
                continue
            a, b = sorted(observed_states)
            pair = (STATE_INDEX[a], STATE_INDEX[b])
            counts[pair_to_idx[pair]] += 1
        pi = base_composition.pi
        if np.any(pi <= 0):
            raise ValueError(
                f"empirical_from_sites: π entries must be strictly positive; "
                f"got {pi.tolist()}"
            )
        pi_products = np.array(
            [pi[i] * pi[j] for (i, j) in cls.RATE_PAIRS], dtype=float,
        )
        if np.any(counts == 0):
            zeros = [
                cls.RATE_NAMES[i] for i in range(6) if counts[i] == 0
            ]
            raise ValueError(
                f"empirical_from_sites: no polymorphic sites observed for "
                f"pair(s) {zeros}; cannot estimate the corresponding rate "
                f"empirically. Supply more sites or fall back to a default "
                f"GTR(fit_rates=True)."
            )
        rates = counts / pi_products
        # Reference: rate_AG = 1 (index 1 in canonical order).
        rates = rates / rates[1]
        return cls(rates=rates, fit_rates=fit_rates, rate_bounds=rate_bounds)

    @property
    def free_params(self) -> dict[str, tuple[float, tuple[float, float]]]:
        """Free rates exposed to the joint MLE. Names from :attr:`GTR.RATE_NAMES`."""
        return {
            self.RATE_NAMES[i]: (float(self.rates[i]), self.rate_bounds)
            for i in range(6) if self.fit_mask[i]
        }

    def set_free_params(self, values: dict[str, float]) -> None:
        """Update rate entries from the optimiser. Invalidate the cache.

        Entries absent from ``values`` are left untouched, and the cache is
        cleared only when at least one rate actually changed.

        :raises ValueError: If any supplied rate is not strictly positive.
        """
        changed = False
        for i, name in enumerate(self.RATE_NAMES):
            if name in values:
                v = float(values[name])
                if v <= 0:
                    raise ValueError(f"{name} must be positive, got {v}")
                self.rates[i] = v
                changed = True
        if changed:
            self.clear_cache()

    def warn_if_bounds_hit(self) -> None:
        """Warn if any MLE-fit rate landed within 2% of ``rate_bounds``."""
        for i, name in enumerate(self.RATE_NAMES):
            if self.fit_mask[i]:
                self._warn_bound(
                    name, float(self.rates[i]), self.rate_bounds, "rate_bounds",
                )
