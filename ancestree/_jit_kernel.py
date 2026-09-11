"""Numba-JIT inner Felsenstein traversal for the
:class:`~ancestree.likelihood.Likelihood` engine.

The compiled kernel takes the tree topology as flat dense arrays
(post-order node list, CSR children, per-node transition matrices,
pre-initialised tip partials) so numba can specialise on numeric types
and elide Python-object overhead. The Python caller packs these arrays
once per tree and reuses them across batched sites.
"""
from __future__ import annotations

import numpy as np

from numba import njit


# ---------------------------------------------------------------- traversal
@njit(cache=True)
def postorder_internal_core(parent, root: int, nslot: int, cofs, cflat, offs,
                            stack, next_child, out):
    """Build the CSR children of a parent array and walk it in post-order.

    Every buffer is caller-owned, so this allocates nothing and may be called
    inside a ``prange``.

    :param parent: ``(nslot,)`` dense parent ids, ``-1`` where absent.
    :param root: Dense id of the node to walk from.
    :param nslot: Number of dense slots.
    :param cofs: ``(nslot + 1,)`` out: CSR offsets into ``cflat``.
    :param cflat: CSR children, ascending dense id, with room for one entry
        per node carrying a parent.
    :param offs: ``(nslot,)`` scratch fill cursor.
    :param stack: Traversal stack, with room for the tree's depth.
    :param next_child: Per-stack-frame cursor into ``cflat``, sized as
        ``stack``.
    :param out: Nodes with at least one child, children before parents, with
        room for every internal node.
    :return: Number of entries written to ``out``.
    """
    for d in range(nslot + 1):
        cofs[d] = 0
    for d in range(nslot):
        p = parent[d]
        if p >= 0:
            cofs[p + 1] += 1
    for d in range(nslot):
        cofs[d + 1] += cofs[d]
    for d in range(nslot):
        offs[d] = cofs[d]
    for d in range(nslot):
        p = parent[d]
        if p >= 0:
            cflat[offs[p]] = d
            offs[p] += 1

    n_out = 0
    top = 0
    stack[0] = root
    next_child[0] = cofs[root]
    while top >= 0:
        v = stack[top]
        if next_child[top] < cofs[v + 1]:
            c = cflat[next_child[top]]
            next_child[top] += 1
            if cofs[c + 1] > cofs[c]:  # internal: descend
                top += 1
                stack[top] = c
                next_child[top] = cofs[c]
        else:
            out[n_out] = v
            n_out += 1
            top -= 1
    return n_out


@njit(cache=True)
def postorder_csr_from_parents(parent_dense, root_dense: int, n_nodes: int):
    """CSR children and the internal-node post-order, from a parent array.

    Both come out of one walk, so each parent's children and the traversal
    that consumes them are in the same ascending dense id order.

    :param parent_dense: ``(n_nodes,)`` dense parent ids, ``-1`` at the root.
    :param root_dense: Dense id of the root to walk from.
    :param n_nodes: Number of dense nodes.
    :return: ``(children_offsets (n_nodes+1,) int32, children_flat
        (n_nodes-1,) int32, postorder_internal (n_internal,) int32)``.
    """
    if n_nodes == 0:
        return (np.zeros(1, dtype=np.int32), np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))
    # The core indexes on these without bounds checking, so an id out of range
    # writes outside its buffer rather than raising.
    if parent_dense.shape[0] != n_nodes:
        raise ValueError("parent_dense length must equal n_nodes")
    if root_dense < 0 or root_dense >= n_nodes:
        raise ValueError("root_dense outside [0, n_nodes)")
    n_roots = 0
    for i in range(n_nodes):
        p_i = parent_dense[i]
        if p_i < 0:
            n_roots += 1
        elif p_i >= n_nodes:
            raise ValueError("parent_dense holds an id outside [0, n_nodes)")
    if n_roots != 1:
        raise ValueError("parent_dense must hold exactly one -1 root")
    # The single -1 must be root_dense itself: a walk seeded anywhere else
    # revisits nodes and overruns the fixed-size stack.
    if parent_dense[root_dense] >= 0:
        raise ValueError("root_dense is not the root of parent_dense")
    offsets = np.empty(n_nodes + 1, dtype=np.int32)
    children = np.empty(max(n_nodes - 1, 0), dtype=np.int32)
    fill = np.empty(n_nodes, dtype=np.int32)
    out = np.empty(n_nodes, dtype=np.int32)
    stack = np.empty(n_nodes, dtype=np.int32)
    next_child = np.empty(n_nodes, dtype=np.int32)
    n_out = postorder_internal_core(parent_dense, root_dense, n_nodes, offsets,
                                    children, fill, stack, next_child, out)
    return offsets, children, out[:n_out]


# ---------------------------------------------------------------- kernel
# No fastmath here or in any kernel that inlines this one: its ninf flag folds
# the all-forbidden check (row_max_log == -inf) away and the row returns NaN.
@njit(cache=True)
def postorder_core(
    B: int,
    S: int,
    root: int,
    postorder_internal,  # (n_internal,) int32
    children_offsets,  # (n_nodes+1,) int32, CSR-style
    children_flat,  # (n_edges,) int32
    P,  # (n_rows, S, S) float64, transition matrices
    p_index,  # (n_nodes,) int32, row of P for the edge into each node
    base: int,  # added to p_index, so one P can hold many trees
    partials,  # (n_nodes, B, S) float64, tip partials seeded
    scale_log,  # (B,) float64 scratch, zeroed here
    combined,  # (B, S) float64 scratch
    log_row,  # (S,) float64 scratch
):
    """Run the post-order Felsenstein traversal on dense arrays.

    For each internal node ``v`` in post-order, multiplies each child's
    message ``partials[child] @ P[child].T`` element-wise into an accumulator
    seeded from ``partials[v]``, then rescales by the per-batch row max and
    accumulates the log-scaling factor. The seed carries the state of an
    internal node that is itself an observed sample. Tip partials and the
    per-edge ``P`` rows are supplied by the caller.

    :return: ``root_partial[B, S]``. ``scale_log`` is filled in place.
    """

    for b in range(B):
        scale_log[b] = 0.0
    for idx in range(postorder_internal.shape[0]):
        v = postorder_internal[idx]
        c_start = children_offsets[v]
        c_end = children_offsets[v + 1]

        # Start from the node's own seeded partial: all-ones for an ordinary
        # internal node, a one-hot where the node is itself an observed sample.
        for b in range(B):
            for s in range(S):
                combined[b, s] = partials[v, b, s]

        for ci in range(c_start, c_end):
            child = children_flat[ci]
            P_e = P[base + p_index[child]]  # (S, S)
            child_partial = partials[child]  # (B, S)
            # msg[b, s] = sum_{s'} child_partial[b, s'] * P[s, s']
            for b in range(B):
                for s in range(S):
                    acc = 0.0
                    for sp in range(S):
                        acc += child_partial[b, sp] * P_e[s, sp]
                    combined[b, s] *= acc

        for b in range(B):
            row_max = 0.0
            underflowed = False
            for s in range(S):
                v_bs = combined[b, s]
                if v_bs > row_max:
                    row_max = v_bs
                if v_bs <= 0.0:
                    underflowed = True
            # Below the smallest normal double, 1.0 / row_max overflows, so
            # the log-space path takes it.
            if row_max < 2.2250738585072014e-308:
                underflowed = True
            if not underflowed:
                inv = 1.0 / row_max
                for s in range(S):
                    combined[b, s] *= inv
                scale_log[b] += np.log(row_max)
                continue

            # A zero here is either a hard constraint or a state that left the
            # row's dynamic range mid-product. One per-row exponent cannot tell
            # them apart, so redo the row in log space, where both are exact.
            for s in range(S):
                seed = partials[v, b, s]
                if seed <= 0.0:
                    log_row[s] = -np.inf
                    continue
                acc_log = np.log(seed)
                for ci in range(c_start, c_end):
                    child = children_flat[ci]
                    msg = 0.0
                    for sp in range(S):
                        msg += partials[child][b, sp] * P[base + p_index[child]][s, sp]
                    if msg <= 0.0:
                        acc_log = -np.inf
                        break
                    acc_log += np.log(msg)
                log_row[s] = acc_log
            row_max_log = -np.inf
            for s in range(S):
                if log_row[s] > row_max_log:
                    row_max_log = log_row[s]
            if row_max_log == -np.inf:  # genuinely impossible for every state
                for s in range(S):
                    combined[b, s] = 0.0
                continue
            for s in range(S):
                combined[b, s] = np.exp(log_row[s] - row_max_log)
            scale_log[b] += row_max_log

        # Store back into partials.
        for b in range(B):
            for s in range(S):
                partials[v, b, s] = combined[b, s]

    root_partial = np.empty((B, S), dtype=np.float64)
    for b in range(B):
        for s in range(S):
            root_partial[b, s] = partials[root, b, s]
    return root_partial


@njit(cache=True)
def felsenstein_postorder_dense(
    B: int,
    S: int,
    root: int,
    postorder_internal,  # (n_internal,) int32
    children_offsets,  # (n_nodes+1,) int32, CSR-style
    children_flat,  # (n_edges,) int32
    P_per_node,  # (n_nodes, S, S) float64, P on the edge into each node
    partials,  # (n_nodes, B, S) float64, tip partials seeded
):
    """One tree's post-order traversal. See :func:`postorder_core`.

    Owns the scratch and addresses ``P_per_node`` one row per node, i.e. the
    identity index with no offset. The batched ensemble path calls
    ``postorder_core`` directly with its own scratch and a packed ``P``.

    :return: ``(root_partial[B, S], scale_log[B])``.
    """
    n_nodes = partials.shape[0]
    p_index = np.empty(n_nodes, dtype=np.int32)
    for v in range(n_nodes):
        p_index[v] = v
    scale_log = np.zeros(B, dtype=np.float64)
    combined = np.empty((B, S), dtype=np.float64)
    log_row = np.empty(S, dtype=np.float64)
    root_partial = postorder_core(
        B, S, root, postorder_internal, children_offsets, children_flat,
        P_per_node, p_index, 0, partials, scale_log, combined, log_row)
    return root_partial, scale_log
