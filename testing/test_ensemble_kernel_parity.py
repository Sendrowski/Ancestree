"""The kernel's rescaling reproduces the unscaled log-space recursion.

``ancestree._jit_kernel.postorder_core`` rescales each node's partials to keep
the fast path in linear space, and drops to logs when a row underflows. Both
branches are checked here against an independent numpy post-order that carries
logs throughout and rescales nothing, so an error in the scaling bookkeeping
has somewhere to show up.

``score_ensemble`` calls ``postorder_core`` directly, addressing the transition
matrices as ``P[base + pidx[child]]``. That addressing is driven end to end by
``testing/test_focal.py``'s ``TestEnsembleReroot``, which compares
``score_ensemble`` against :class:`~ancestree.likelihood.Likelihood` with a
non-identity ``pidx``.
"""
import numpy as np
import pytest

from ancestree import F81, GTR, HKY, JC69, K2
from ancestree._jit_kernel import (postorder_core)

S = 4


def _balanced_tree(n_tips):
    """A caterpillar over ``n_tips`` tips as (root, post-order, CSR children).

    Tips are ``0..n_tips-1``. Internal nodes follow, each joining the previous
    accumulator to the next tip, so the topology is deep enough to exercise the
    rescaling path.
    """
    n_nodes = 2 * n_tips - 1
    children = {v: [] for v in range(n_nodes)}
    acc = 0
    for i in range(1, n_tips):
        parent = n_tips + i - 1
        children[parent] = [acc, i]
        acc = parent
    root = acc
    postorder = np.array(sorted(v for v in children if children[v]),
                         dtype=np.int32)
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    flat = []
    for v in range(n_nodes):
        offsets[v + 1] = offsets[v] + len(children[v])
        flat.extend(children[v])
    return root, postorder, offsets, np.array(flat, dtype=np.int32), n_nodes


def _inputs(n_tips, B, seed, *, underflow=False):
    rng = np.random.default_rng(seed)
    root, poi, cofs, cflat, n_nodes = _balanced_tree(n_tips)
    P = rng.random((n_nodes, S, S)) + 0.05
    P /= P.sum(axis=2, keepdims=True)
    partials = np.zeros((n_nodes, B, S), dtype=np.float64)
    for tip in range(n_tips):
        for b in range(B):
            partials[tip, b, rng.integers(S)] = 1.0
    for v in range(n_tips, n_nodes):
        partials[v] = 1.0
    if underflow:
        # Drive one state's message to exactly zero via a forbidden transition,
        # so the row has a zero but is not wholly zero: that is the reachable
        # way into the log-space branch. A tip admitting no state at all would
        # zero the whole row, which real data cannot produce.
        # Zero a ROW of P (state 0 unreachable across that edge), not a
        # column: the column form kills every state's message and collapses the
        # whole row, which is the out-of-contract case above.
        P[0, 0, :] = 0.0
    return root, poi, cofs, cflat, n_nodes, P, partials


def _reference_log_partials(root, cofs, cflat, P, partials, B):
    """Post-order in log space, no rescaling, as the arithmetic reference.

    ``msg[b, s] = sum_sp partial_child[b, sp] * P[child][s, sp]``, matching
    :func:`~ancestree._jit_kernel.postorder_core`'s inner combine.
    """
    def visit(v):
        with np.errstate(divide="ignore"):
            acc = np.log(np.asarray(partials[v], dtype=float))
        for child in cflat[cofs[v]:cofs[v + 1]]:
            child_log = visit(int(child))
            shift = child_log.max(axis=1, keepdims=True)
            shift = np.where(np.isfinite(shift), shift, 0.0)
            with np.errstate(divide="ignore"):
                acc = acc + np.log(
                    np.exp(child_log - shift) @ np.asarray(P[int(child)]).T
                ) + shift
        return acc

    return visit(int(root))


def _run_ensemble(root, poi, cofs, cflat, P, partials, B, n_nodes):
    part = partials.copy()
    scale_log = np.zeros(B, dtype=np.float64)
    # base=0 and pidx=identity make P[base + pidx[child]] == P_per_node[child],
    # which is the mapping score_ensemble sets up per (member, window).
    pidx = np.arange(n_nodes, dtype=np.int32)
    postorder_core(B, S, root, poi, cofs, cflat, P, pidx, 0, part,
                   scale_log, np.empty((B, S), dtype=np.float64),
                   np.empty(S, dtype=np.float64))
    return part[root].copy(), scale_log, part


def _assert_matches_reference(inputs, B):
    root, poi, cofs, cflat, n_nodes, P, partials = inputs
    root_partial, scale_log, _ = _run_ensemble(
        root, poi, cofs, cflat, P, partials, B, n_nodes)
    with np.errstate(divide="ignore"):
        got = np.log(np.asarray(root_partial, dtype=float)) \
            + np.asarray(scale_log, dtype=float)[:, None]
    want = _reference_log_partials(root, cofs, cflat, P, partials, B)
    np.testing.assert_allclose(got, want, rtol=1e-11, atol=0)


@pytest.mark.parametrize("n_tips,B,seed", [(4, 3, 0), (8, 5, 1), (12, 2, 2)])
def test_the_rescaled_pass_matches_log_space(n_tips, B, seed):
    _assert_matches_reference(_inputs(n_tips, B, seed), B)


def test_the_log_space_fallback_matches_too():
    """The underflow branch is the half most likely to drift. Pin it too."""
    _assert_matches_reference(_inputs(6, 3, 7, underflow=True), 3)

_PI = np.array([0.3, 0.2, 0.2, 0.3])


@pytest.mark.parametrize("model,pi", [
    (JC69(), None), (K2(kappa=2.0), None), (F81(), _PI),
    (HKY(kappa=2.5), _PI), (GTR(rates=[1.0, 2.0, 1.5, 0.8, 1.2, 1.0]), _PI),
])
def test_real_eig_reconstructs_transition_probs(model, pi):
    """The ensemble kernel rebuilds ``P(t)`` from ``real_eig`` per branch.

    It must agree with the batched :meth:`transition_probs` every other path
    uses, or ensemble mode would silently score against a different model.
    """
    eig = model.real_eig(pi=pi)
    assert eig is not None, "every shipped model is reversible"
    lam, V, Vinv = eig
    assert lam.dtype == V.dtype == Vinv.dtype == np.float64
    t = np.array([0.0, 1e-8, 1e-4, 0.03, 0.5, 3.0])
    ref = model.transition_probs(t, pi=pi)
    got = np.maximum((V[None] * np.exp(np.outer(t, lam))[:, None, :]) @ Vinv, 0.0)
    # Two routes to the same P: the model's own formula against the symmetric
    # eigendecomposition. Agreement is at rounding level, not bit-exact.
    assert np.allclose(ref, got, atol=1e-12, rtol=0)
    assert np.linalg.cond(V) < 10.0, "eigh must stay well conditioned"
