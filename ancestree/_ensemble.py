r"""Ensemble sampling for local-tree mode.

Local-tree mode's plug-in estimate scores one genealogy per window: the per-pair
TMRCA posterior mean out of the pairwise HMM, agglomerated by average linkage.
Where that estimate mis-resolves a split inside the ingroup, the carrier set of a
derived allele stops being a clade and the Felsenstein kernel is confidently
wrong. Marginalising instead over :math:`B` genealogies drawn from the same HMM
posterior replaces the plug-in likelihood with

.. math::

    P(s \mid x) \;\propto\; \pi(s)\,
        \frac{1}{B}\sum_{b=1}^{B} P(x \mid s, \tau_b),

where :math:`s` is the ancestral state at the focal node (one of the four
nucleotides), :math:`x` the observed tip alleles at a site, :math:`\pi(s)` the
prior at that node, :math:`\tau_b` the :math:`b`-th sampled genealogy for the
window carrying the site, and :math:`B` the ensemble size. Each :math:`\tau_b`
is a joint draw of the per-block TMRCA path for every haplotype pair, taken by
forward-filtering backward-sampling from the HMM's own forward matrix, condensed
to per-window means and agglomerated by the same average-linkage rule the
plug-in estimate uses.

The kernels here are compiled and operate on dense arrays. The driver that feeds
them one segment at a time is
:class:`~ancestree.local_tree_inference.LocalTreeInference`.
"""
import llvmlite.ir as ir
import hashlib
import logging

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic

from ancestree._jit_kernel import postorder_core, postorder_internal_core

S_STATES = 4


# --------------------------------------------------------------- UPGMA
@intrinsic
def _fma(typingctx, a, b, c):
    """``a*b + c`` under a single rounding, matching the fused multiply-add the
    C compiler contracts scipy's average-linkage update into."""
    sig = types.float64(types.float64, types.float64, types.float64)

    def codegen(context, builder, signature, args):
        dbl = ir.DoubleType()
        fnty = ir.FunctionType(dbl, [dbl, dbl, dbl])
        fn = builder.module.declare_intrinsic("llvm.fma", [dbl], fnty)
        return builder.call(fn, args)
    return sig, codegen


@njit(cache=True, inline="always")
def _condensed_index(n, i, j):
    """Condensed index of the pair ``(i, j)`` with ``i < j``."""
    return n * i - (i * (i + 1)) // 2 + (j - i - 1)


@njit(cache=True)
def _nn_chain_average(D, n, Z):
    """Nearest-neighbour chain for average linkage; ``D`` is scratch."""
    size = np.ones(n, np.float64)
    chain = np.empty(n, np.int64)
    chain_len = 0
    for k in range(n - 1):
        if chain_len == 0:
            chain_len = 1
            for i in range(n):
                if size[i] > 0:
                    chain[0] = i
                    break
        while True:
            x = chain[chain_len - 1]
            if chain_len > 1:
                y = chain[chain_len - 2]
                current_min = D[_condensed_index(n, min(x, y), max(x, y))]
            else:
                y = -1
                current_min = np.inf
            for i in range(n):
                if size[i] == 0 or x == i:
                    continue
                dist = D[_condensed_index(n, min(x, i), max(x, i))]
                if dist < current_min:
                    current_min = dist
                    y = i
            if chain_len > 1 and y == chain[chain_len - 2]:
                break
            chain[chain_len] = y
            chain_len += 1
        chain_len -= 2
        if x > y:
            x, y = y, x
        nx = size[x]
        ny = size[y]
        Z[k, 0] = x
        Z[k, 1] = y
        Z[k, 2] = current_min
        Z[k, 3] = nx + ny
        size[x] = 0.0
        size[y] = nx + ny
        for i in range(n):
            if size[i] == 0 or i == y:
                continue
            dxi = D[_condensed_index(n, min(i, x), max(i, x))]
            dyi = D[_condensed_index(n, min(i, y), max(i, y))]
            D[_condensed_index(n, min(i, y), max(i, y))] = (
                _fma(nx, dxi, ny * dyi) / (nx + ny))


@njit(cache=True)
def _sort_and_label(Z, n, order, uf_parent, uf_size):
    """Stable sort of ``Z`` by height, then scipy's union-find relabelling."""
    m = n - 1
    for i in range(m):
        order[i] = i
    for i in range(1, m):
        key = order[i]
        kv = Z[key, 2]
        j = i - 1
        while j >= 0 and Z[order[j], 2] > kv:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = key
    Zs = np.empty((m, 4), np.float64)
    for i in range(m):
        for c in range(4):
            Zs[i, c] = Z[order[i], c]
    for i in range(2 * n - 1):
        uf_parent[i] = i
        uf_size[i] = 1
    next_label = n
    for i in range(m):
        x = int(Zs[i, 0])
        y = int(Zs[i, 1])
        p = x
        while uf_parent[p] != p:
            p = uf_parent[p]
        xr = p
        p = x
        while uf_parent[p] != p:
            nxt = uf_parent[p]
            uf_parent[p] = xr
            p = nxt
        p = y
        while uf_parent[p] != p:
            p = uf_parent[p]
        yr = p
        p = y
        while uf_parent[p] != p:
            nxt = uf_parent[p]
            uf_parent[p] = yr
            p = nxt
        if xr < yr:
            Zs[i, 0] = xr
            Zs[i, 1] = yr
        else:
            Zs[i, 0] = yr
            Zs[i, 1] = xr
        uf_parent[xr] = next_label
        uf_parent[yr] = next_label
        uf_size[next_label] = uf_size[xr] + uf_size[yr]
        Zs[i, 3] = uf_size[next_label]
        next_label += 1
    for i in range(m):
        for c in range(4):
            Z[i, c] = Zs[i, c]


@njit(cache=True, parallel=True)
def upgma_batch(condensed, n, Zout):
    """Average-linkage UPGMA over a stack of condensed distance matrices.

    :param condensed: ``(R, n(n-1)/2)`` float64 pairwise TMRCAs.
    :param n: Number of taxa.
    :param Zout: ``(R, n-1, 4)`` float64, filled with scipy-format linkages.
    """
    R = condensed.shape[0]
    npair = condensed.shape[1]
    for w in prange(R):
        D = np.empty(npair, np.float64)
        for i in range(npair):
            D[i] = condensed[w, i]
        Z = np.empty((n - 1, 4), np.float64)
        _nn_chain_average(D, n, Z)
        order = np.empty(n - 1, np.int64)
        ufp = np.empty(2 * n - 1, np.int64)
        ufs = np.empty(2 * n - 1, np.int64)
        _sort_and_label(Z, n, order, ufp, ufs)
        for i in range(n - 1):
            for c in range(4):
                Zout[w, i, c] = Z[i, c]


# ------------------------------------------------------- dense tree packing
@njit(cache=True)
def _pack_base(Z, n, is_ingroup, mu, branch_t, parent0, sample_dense,
               left, right, dist, tsid, time, po, dense, tmp_par, cnt,
               stack_c, stack_s):
    """Dense arrays for one window's linkage, before any re-rooting.

    Node ids are tskit's: a right-first post-order over the linkage, with the
    tie bump that keeps a parent strictly above both children. The dense index
    of a node is its position in tskit's own post-order, so the root is
    ``2n - 2`` and the samples occupy whatever slots the traversal gives them.

    :param Z: ``(n-1, 4)`` scipy-format linkage.
    :param n: Panel haplotypes.
    :param is_ingroup: ``(n,)`` bool, whether each linkage leaf is an
        ingroup haplotype. Membership, not a count: the ingroup is not
        required to occupy the leading columns.
    :param mu: Per-site per-generation mutation rate. Scales branch lengths
        into expected substitutions.
    :param branch_t: ``(2n+1,)`` out: scaled length of the edge above each
        dense node, ``0`` at the root. The two trailing slots are written by
        :func:`_reroot`.
    :param parent0: ``(2n-1,)`` out: dense parent, ``-1`` at the root.
    :param sample_dense: ``(n,)`` out: dense index of each sample.
    :return: Dense index of the ingroup MRCA.
    """
    n_all = 2 * n - 1
    for c in range(n_all):
        left[c] = -1
        right[c] = -1
        dist[c] = 0.0
        tsid[c] = -1
        time[c] = 0.0
    for k in range(n - 1):
        left[n + k] = int(Z[k, 0])
        right[n + k] = int(Z[k, 1])
        dist[n + k] = Z[k, 2]
    for i in range(n):
        tsid[i] = i

    nxt = n
    top = 0
    stack_c[0] = n_all - 1
    stack_s[0] = 0
    while top >= 0:
        c = stack_c[top]
        s = stack_s[top]
        if c < n:
            top -= 1
            continue
        if s == 0:
            stack_s[top] = 1
            top += 1
            stack_c[top] = left[c]
            stack_s[top] = 0
            top += 1
            stack_c[top] = right[c]
            stack_s[top] = 0
            continue
        top -= 1
        lt = time[left[c]]
        rt = time[right[c]]
        t = dist[c]
        floor = lt if lt > rt else rt
        if t <= floor:
            t = floor + (floor * 1e-6 if floor > 0.0 else 1e-6)
        time[c] = t
        tsid[c] = nxt
        nxt += 1

    n_act = 0
    top = 0
    stack_c[0] = n_all - 1
    stack_s[0] = 0
    while top >= 0:
        c = stack_c[top]
        s = stack_s[top]
        if c < n:
            top -= 1
            po[n_act] = c
            n_act += 1
            continue
        if s == 0:
            stack_s[top] = 1
            a = left[c]
            b = right[c]
            if tsid[a] < tsid[b]:
                first = a
                second = b
            else:
                first = b
                second = a
            top += 1
            stack_c[top] = second
            stack_s[top] = 0
            top += 1
            stack_c[top] = first
            stack_s[top] = 0
            continue
        top -= 1
        po[n_act] = c
        n_act += 1

    for d in range(n_act):
        dense[po[d]] = d
    for c in range(n_all):
        tmp_par[c] = -1
    for c in range(n, n_all):
        tmp_par[left[c]] = c
        tmp_par[right[c]] = c
    for d in range(n_act):
        p = tmp_par[po[d]]
        if p >= 0:
            parent0[d] = dense[p]
            branch_t[d] = (time[p] - time[po[d]]) * mu
        else:
            parent0[d] = -1
            branch_t[d] = 0.0
    for s in range(n):
        sample_dense[s] = dense[s]

    n_ing = 0
    for c in range(n_all):
        if c < n and is_ingroup[c]:
            cnt[c] = 1
            n_ing += 1
        else:
            cnt[c] = 0
    for c in range(n, n_all):
        cnt[c] = cnt[left[c]] + cnt[right[c]]
    # Descend while one child still holds the whole ingroup. A leaf is a valid
    # stopping point: the MRCA of a single haplotype is that haplotype.
    f = n_all - 1
    while f >= n:
        a = left[f]
        b = right[f]
        if cnt[a] == n_ing:
            f = a
        elif cnt[b] == n_ing:
            f = b
        else:
            break
    return dense[f]


@njit(cache=True, parallel=True)
def pack_base_all(Zs, n, is_ingroup, mu, branch_t, parent0, sample_dense, fmrca):
    """Run :func:`_pack_base` over a flat stack of ``(member, window)`` rows.

    :param Zs: ``(R, n-1, 4)`` linkages.
    :param branch_t: ``(R, 2n+1)`` out.
    :param parent0: ``(R, 2n-1)`` out.
    :param sample_dense: ``(R, n)`` out.
    :param fmrca: ``(R,)`` out: dense index of the ingroup MRCA.
    """
    R = Zs.shape[0]
    n_all = 2 * n - 1
    for w in prange(R):
        left = np.empty(n_all, np.int64)
        right = np.empty(n_all, np.int64)
        dist = np.empty(n_all, np.float64)
        tsid = np.empty(n_all, np.int64)
        time = np.empty(n_all, np.float64)
        po = np.empty(n_all, np.int64)
        dense = np.empty(n_all, np.int64)
        tmp_par = np.empty(n_all, np.int64)
        cnt = np.empty(n_all, np.int64)
        stack_c = np.empty(2 * n_all, np.int64)
        stack_s = np.empty(2 * n_all, np.int64)
        fmrca[w] = _pack_base(
            Zs[w], n, is_ingroup, mu, branch_t[w], parent0[w], sample_dense[w],
            left, right, dist, tsid, time, po, dense, tmp_par, cnt,
            stack_c, stack_s)


@njit(cache=True)
def _reroot(parent0, branch_t, fmrca, n, fraction, n_coal, pidx, cofs, cflat,
            poi, par, path, offs, stack_c, nxtc):
    """Re-root one window at the focal point and emit the traversal arrays.

    The focal point sits ``fraction`` of the way along the path from the
    ingroup MRCA to the root, measured in branch length. A point strictly
    inside an edge becomes one extra dense node with two children, the focal
    subtree at distance ``tau`` below it and the rest of the tree at the
    remaining ``L - tau``. The two lengths occupy the trailing slots of
    ``branch_t``, which ``pidx`` then points the affected nodes at. Reversal
    of the parent pointers alone handles the rest, since a time-reversible
    model lets every edge keep its transition matrix.

    A ``fraction`` of 1 or more reads at the tree's own root, which needs no
    re-rooting at all.

    :param parent0: ``(2n-1,)`` dense parents before re-rooting.
    :param branch_t: ``(2n+1,)`` scaled branch lengths. The trailing two slots
        are written here.
    :param fmrca: Dense index of the ingroup MRCA.
    :param n: Panel haplotypes.
    :param fraction: Position along the anchor-to-root path, in ``[0, 1]``.
    :param pidx: ``(2n+1,)`` out: which ``branch_t`` slot each node's edge
        matrix comes from.
    :param cofs: ``(2n+2,)`` out: CSR offsets into ``cflat``.
    :param cflat: ``(2n+1,)`` out: CSR children, ascending dense id.
    :param poi: ``(n,)`` out: internal nodes in post-order, padded with an
        unused slot when the tree needs no split node.
    :return: Dense index of the node the posterior is read at.
    """
    n_all = 2 * n - 1
    nslot = n_all + 2
    for d in range(nslot):
        pidx[d] = d
        par[d] = -1
    for d in range(n_all):
        par[d] = parent0[d]
    branch_t[n_all] = 0.0
    branch_t[n_all + 1] = 0.0
    root = n_all - 1

    if fraction < 1.0 or n_coal >= 0:
        plen = 0
        v = fmrca
        while v >= 0:
            path[plen] = v
            plen += 1
            v = parent0[v]
        step = plen - 1
        tau = 0.0
        if n_coal >= 0:
            # Land on the n_coal-th node up the path, or the root past the last join.
            step = n_coal if n_coal < plen - 1 else plen - 1
        else:
            total = 0.0
            for k in range(plen - 1):
                total += branch_t[path[k]]
            want = total * fraction
            if want > total:
                want = total
            for k in range(plen - 1):
                edge = branch_t[path[k]]
                if want <= edge:
                    step = k
                    tau = want
                    break
                want -= edge
        focal = path[step]
        for k in range(plen - 1, step, -1):
            par[path[k]] = path[k - 1]
            pidx[path[k]] = path[k - 1]
        par[focal] = -1
        root = focal
        if tau > 0.0:
            edge = branch_t[focal]
            branch_t[n_all] = tau
            branch_t[n_all + 1] = edge - tau if edge > tau else 0.0
            par[focal] = n_all
            pidx[focal] = n_all
            if step + 1 < plen:
                other = path[step + 1]
                par[other] = n_all
                pidx[other] = n_all + 1
            root = n_all

    n_int = postorder_internal_core(par, root, nslot, cofs, cflat, offs,
                                    stack_c, nxtc, poi)
    # The trailing slot is never a node and never a child, so visiting it
    # multiplies nothing and adds nothing to the running scale.
    for k in range(n_int, n):
        poi[k] = nslot - 1
    return root


@njit(cache=True, parallel=True)
def reroot_all(parent0, branch_t, fmrca, n, fraction, n_coal, pidx, cofs, cflat, poi,
               root):
    """Run :func:`_reroot` over a flat stack of ``(member, window)`` rows."""
    R = parent0.shape[0]
    n_all = 2 * n - 1
    nslot = n_all + 2
    for w in prange(R):
        par = np.empty(nslot, np.int64)
        path = np.empty(n_all, np.int64)
        offs = np.empty(nslot, np.int64)
        stack_c = np.empty(2 * nslot, np.int64)
        nxtc = np.empty(2 * nslot, np.int64)
        root[w] = _reroot(parent0[w], branch_t[w], fmrca[w], n, fraction, n_coal,
                          pidx[w], cofs[w], cflat[w], poi[w],
                          par, path, offs, stack_c, nxtc)


# ---------------------------------------------------------- Felsenstein
# No fastmath: inlines postorder_core, see the note in _jit_kernel.
@njit(cache=True, parallel=True)
def score_ensemble(branch_t, lam, V, Vinv, pidx, poi,
                   cofs, cflat, root, sample_dense, tip_states, row_off,
                   row_idx, nslot, S, out):
    """``log P(tips | focal = s)`` per member and site.

    The ensemble is a leading axis: the traversal arrays are ``(M, W, ...)``.
    The tip partials depend on the site alone, so they are seeded once per
    window and reused across the members.

    Each edge's ``P(t)`` is rebuilt here as ``V diag(e^{λt}) V⁻¹``, into a
    buffer private to the thread handling the window, so only windows carrying
    sites pay for the exponentials and no ``(M, W, nslot, S, S)`` array is
    formed.

    :param branch_t: ``(M, W, nslot)`` scaled branch lengths, post-rerooting.
    :param lam: ``(S,)`` real eigenvalues of ``Q``. See
        :meth:`SubstitutionModel.real_eig() <ancestree.models.SubstitutionModel.real_eig>`.
    :param V: ``(S, S)`` right eigenvectors.
    :param Vinv: ``(S, S)`` its inverse.
    :param row_off: ``(W+1,)`` CSR offsets of ``row_idx``.
    :param row_idx: Site rows, grouped by the window carrying them.
    :param nslot: Dense slots per window, ``2n + 1``.
    :param out: ``(M, n_sites, S)`` out.
    """
    M = pidx.shape[0]
    W = pidx.shape[1]
    n_samp = tip_states.shape[1]
    for w in prange(W):
        b0 = row_off[w]
        B = row_off[w + 1] - b0
        if B == 0:
            continue
        seed = np.ones((n_samp, B, S), np.float64)
        for b in range(B):
            row = row_idx[b0 + b]
            for s in range(n_samp):
                st = tip_states[row, s]
                if st >= 0:
                    for k in range(S):
                        seed[s, b, k] = 0.0
                    seed[s, b, st] = 1.0
        partials = np.empty((nslot, B, S), np.float64)
        scale_log = np.empty(B, np.float64)
        combined = np.empty((B, S), np.float64)
        log_row = np.empty(S, np.float64)
        Pw = np.empty((nslot, S, S), np.float64)  # this window's edges only
        ev = np.empty(S, np.float64)
        for m in range(M):
            for d in range(nslot):
                t = branch_t[m, w, d]
                for k in range(S):
                    ev[k] = np.exp(lam[k] * t)
                for i in range(S):
                    for j in range(S):
                        acc = 0.0
                        for k in range(S):
                            acc += V[i, k] * ev[k] * Vinv[k, j]
                        # exp(Q·t) is non-negative. Clip reconstruction
                        # noise (~1e-16 off-diagonals at t = 0) as the
                        # batched path does.
                        Pw[d, i, j] = acc if acc > 0.0 else 0.0
            for d in range(nslot):
                for b in range(B):
                    for s in range(S):
                        partials[d, b, s] = 1.0
            for s in range(n_samp):
                d = sample_dense[m, w, s]
                if d >= 0:
                    for b in range(B):
                        for k in range(S):
                            partials[d, b, k] = seed[s, b, k]
            r = root[m, w]
            postorder_core(B, S, r, poi[m, w], cofs[m, w], cflat[m, w],
                           Pw, pidx[m, w], 0, partials,
                           scale_log, combined, log_row)
            for b in range(B):
                row = row_idx[b0 + b]
                for s in range(S):
                    out[m, row, s] = np.log(partials[r, b, s]) + scale_log[b]


# ------------------------------------------------------------------ HMM
#: Ceiling on the retained HMM forward matrix, ``(n_pairs, n_ckpt, T)`` float64,
#: which sets the checkpoint stride.
MAX_HMM_ALPHA_BYTES = 256 * 1024 ** 2

#: Ceiling on a member chunk's sampled paths, ``(member_chunk, n_pairs,
#: n_blocks) int8``. Sized to keep the member chunk wide: see
#: :meth:`SegmentEnsemble._path_chunk`.
MAX_PATH_BYTES = 4 * 1024 ** 3


def _ckpt_stride(n_blocks: int, n_pairs: int, n_time_bins: int) -> int:
    """Blocks between retained forward distributions, to fit the budget.

    Every ``stride``-th block of the ``(P, K, T)`` forward matrix is stored and
    the segment between checkpoints is replayed as the backward walk reaches
    it. The replay totals ``K`` blocks for any stride, so the stride is set from
    :data:`MAX_HMM_ALPHA_BYTES` alone and is 1 when the full matrix fits.
    Checkpoints are float64, so the sampled paths do not depend on the stride.
    """
    full = n_pairs * max(n_blocks, 1) * n_time_bins * 8
    return max(1, int(np.ceil(full / MAX_HMM_ALPHA_BYTES)))


@njit(cache=True, inline="always")
def _forward_init(counts_p, lam, log_lam, pi_p, esc_p, a, e, T):
    """Filtered distribution at block 0, in place into ``a``.

    ``esc_p[k]`` is the fraction of block ``k`` over which this pair is called,
    so the Poisson mean is the exposure the count was actually generated from.
    """
    mx = -1e300
    for i in range(T):
        ll = -esc_p[0] * lam[0, i] + counts_p[0] * log_lam[0, i]
        e[i] = ll
        if ll > mx:
            mx = ll
    s0 = 0.0
    for i in range(T):
        e[i] = np.exp(e[i] - mx)
        a[i] = pi_p[i] * e[i]
        s0 += a[i]
    # An underflowed normaliser resets the row to the prior. Zeroing it is
    # absorbing: every later block's total is then zero too.
    if s0 > 0.0:
        inv = 1.0 / s0
        for i in range(T):
            a[i] *= inv
    else:
        for i in range(T):
            a[i] = pi_p[i]


@njit(cache=True, inline="always")
def _reset_row_k(pi_p, r, k, nu, T):
    """Fill ``nu[j] ∝ pi_p[j] · (1 - r[k, j])``, the row a recombination
    resets to when entering block ``k``.

    ``pi_p`` is invariant under ``A[i, j] = r[k, i]·δ_ij + (1 - r[k, i])·nu[j]``
    exactly for this ``nu``, so the marginal over time bins is the same at every
    block, as the coalescent requires.

    :return: The normaliser ``sum_j pi_p[j] · (1 - r[k, j])``.
    """
    tot = 0.0
    for j in range(T):
        v = pi_p[j] * (1.0 - r[k, j])
        nu[j] = v
        tot += v
    if tot > 0.0:
        inv = 1.0 / tot
        for j in range(T):
            nu[j] *= inv
    else:
        for j in range(T):
            nu[j] = pi_p[j]
    return tot


@njit(cache=True, fastmath=True, inline="always")
def _forward_step(counts_p, lam, log_lam, r, pi_p, esc_p, k, a_prev, a_cur,
                  e, nu, T):
    """One forward step into block ``k``."""
    _reset_row_k(pi_p, r, k, nu, T)
    mx = -1e300
    for i in range(T):
        ll = -esc_p[k] * lam[k, i] + counts_p[k] * log_lam[k, i]
        e[i] = ll
        if ll > mx:
            mx = ll
    for i in range(T):
        e[i] = np.exp(e[i] - mx)
    tot = 0.0
    for i in range(T):
        tot += a_prev[i] * (1.0 - r[k, i])
    sk = 0.0
    for j in range(T):
        v = (a_prev[j] * r[k, j] + nu[j] * tot) * e[j]
        a_cur[j] = v
        sk += v
    if sk > 0.0:
        inv = 1.0 / sk
        for j in range(T):
            a_cur[j] *= inv
    else:
        for j in range(T):
            a_cur[j] = pi_p[j]






@njit(cache=True, parallel=True)
def paths_to_condensed(path, edge_lo, edge_hi, t_bar, needed, blk_lo, blk_hi,
                       w0, pair_key, seed, member0, out):
    """Window-mean sampled TMRCAs for one block of windows, from the paths.

    Parallel over the windows in the block, each of which covers a contiguous
    run of blocks. A path names a bin of the discretised chain. The time is
    drawn from the pair's coalescent prior ``Exp(1/t_bar)`` truncated to the
    bin ``[a, b)`` by inverse CDF, through ``expm1`` and ``log1p`` so a bin far
    above ``t_bar`` still returns a time inside itself. The two extreme bins
    carry the prior's folded tails (see
    :meth:`PairwiseCoalescentHMM._coalescent_prior`). Consecutive blocks in
    the same state are one coalescence and share one draw, followed back only
    to the window's own first block. The stream is keyed on the pair, the block
    and the member's global index, so the draws do not depend on
    ``member_chunk`` or on the window blocking.

    :param path: ``(M, P, K) int8`` sampled bin per member, pair and block.
    :param edge_lo: ``(T,)`` lower edge of each bin, in generations.
    :param edge_hi: ``(T,)`` upper edge of each bin, in generations.
    :param t_bar: ``(P,)`` per-pair genome-average TMRCA, the prior mean.
    :param needed: Window indices to keep, over the whole segment.
    :param blk_lo: ``(n_windows,)`` first block of each window.
    :param blk_hi: ``(n_windows,)`` one past its last block.
    :param w0: Index into ``needed`` of this block's first window.
    :param pair_key: ``(P,) uint64`` per-pair stream key, as ``ffbs_paths``.
    :param seed: Base seed of the run.
    :param member0: Global index of ``path``'s first member.
    :param out: ``(M, n_w, P)`` float64 out for windows ``needed[w0:w0+n_w]``.
    """
    M, P, K = path.shape
    n_w = out.shape[1]
    for wi in prange(n_w):
        w = needed[w0 + wi]
        lo, hi = blk_lo[w], blk_hi[w]
        wt = 1.0 / (hi - lo)
        for b in range(M):
            mem = np.uint64(member0 + b)
            for pp in range(P):
                acc = 0.0
                for k in range(hi - 1, lo - 1, -1):
                    i = path[b, pp, k]
                    j = k
                    while j > lo and path[b, pp, j - 1] == i:
                        j -= 1
                    st = (mem * np.uint64(0x9E3779B97F4A7C15)
                          ^ pair_key[pp] * np.uint64(0xBF58476D1CE4E5B9)
                          ^ np.uint64(j) * np.uint64(0x94D049BB133111EB)
                          ^ np.uint64(seed))
                    st, _u0 = _split(st)
                    st, u = _split(st)
                    a = edge_lo[i]
                    tb = t_bar[pp]
                    d = (edge_hi[i] - a) / tb
                    acc += wt * (a - tb * np.log1p(-u * (-np.expm1(-d))))
                out[b, wi, pp] = acc


@njit(cache=True, parallel=True)
def hmm_forward_ckpt(counts, lam, log_lam, r, pi, esc, stride, alpha_ck):
    """Forward pass retaining only every ``stride``-th block's distribution.

    :param stride: Blocks between checkpoints. See :func:`_ckpt_stride`.
    :param alpha_ck: ``(P, ceil(K / stride), T) float64`` out.
    """
    P, K = counts.shape
    T = lam.shape[1]
    for p in prange(P):
        e = np.empty(T, np.float64)
        nu = np.empty(T, np.float64)
        a_prev = np.empty(T, np.float64)
        a_cur = np.empty(T, np.float64)
        _forward_init(counts[p], lam, log_lam, pi[p], esc[p], a_prev, e, T)
        for i in range(T):
            alpha_ck[p, 0, i] = a_prev[i]
        for k in range(1, K):
            _forward_step(counts[p], lam, log_lam, r, pi[p], esc[p], k,
                          a_prev, a_cur, e, nu, T)
            for j in range(T):
                a_prev[j] = a_cur[j]
            if k % stride == 0:
                c = k // stride
                for j in range(T):
                    alpha_ck[p, c, j] = a_prev[j]


@njit(cache=True, inline="always")
def _replay_segment(counts_p, lam, log_lam, r, pi_p, esc_p, alpha_ck_p, c,
                    stride, K, seg, e, nu, T):
    """Rebuild blocks ``[c*stride, min(c*stride+stride, K))`` into ``seg``.

    Seeded from the checkpoint at ``c*stride``, which is the exact filtered
    distribution there, so the rebuilt blocks match the full forward matrix
    exactly.

    :return: Number of blocks written to ``seg``.
    """
    lo = c * stride
    hi = lo + stride
    if hi > K:
        hi = K
    for j in range(T):
        seg[0, j] = alpha_ck_p[c, j]
    for k in range(lo + 1, hi):
        _forward_step(counts_p, lam, log_lam, r, pi_p, esc_p, k,
                      seg[k - lo - 1], seg[k - lo], e, nu, T)
    return hi - lo


def _pair_stream_keys(sample_names, pairs) -> np.ndarray:
    """A stable RNG stream key per haplotype pair.

    Keyed on the pair's sample names, not its position in the panel, so
    the same two haplotypes draw the same genealogies whichever columns they
    occupy. Falls back to the pair index when the panel is unnamed, which only
    the kernel-level tests do.

    :param sample_names: Panel names in genotype-column order, or ``None``.
    :param pairs: ``(i, j)`` column pairs, in the order the kernel walks them.
    :return: ``(len(pairs),) uint64`` keys.
    """
    if not sample_names:
        return np.arange(len(pairs), dtype=np.uint64)
    keys = np.empty(len(pairs), dtype=np.uint64)
    mask = (1 << 64) - 1
    for row, (i, j) in enumerate(pairs):
        a, b = sorted((str(sample_names[i]), str(sample_names[j])))
        digest = hashlib.blake2b(f"{a}\x00{b}".encode(), digest_size=8)
        keys[row] = np.uint64(int.from_bytes(digest.digest(), "big") & mask)
    return keys


@njit(cache=True, parallel=True)
def ffbs_paths(counts, lam, log_lam, alpha_ck, stride, K, r, pi, esc, seed0,
               pair_key,
               path):
    """Joint forward-filtering backward-sampling draws of the TMRCA path.

    Writes the sampled time-bin index per member, pair and block into ``path``
    as ``int8``. Window means are reduced from it by
    :func:`paths_to_condensed`. Each segment between checkpoints is replayed
    as the backward walk reaches it (see :func:`_ckpt_stride`).

    :param alpha_ck: ``(P, n_ckpt, T) float64`` retained forward distributions.
    :param stride: Blocks between checkpoints.
    :param K: Total blocks.
    :param r: ``(K, T)`` probability of not resetting the TMRCA on the step
        into each block.
    :param pi: ``(P, T)`` per-pair coalescent prior over time bins.
    :param seed0: First member's stream seed. Member ``b`` uses ``seed0 + b``.
    :param pair_key: ``(P,) uint64`` per-pair stream key. Derived from the
        pair's sample names, not its column position, so a panel
        given in a different column order draws the same genealogies.
    :param path: ``(M, P, K) int8`` out, the sampled time-bin index per member,
        pair and block.
    """
    P = alpha_ck.shape[0]
    T = alpha_ck.shape[2]
    n_ckpt = alpha_ck.shape[1]
    M = path.shape[0]
    for p in prange(P):
        q = np.empty(T, np.float64)
        cum = np.empty(T, np.float64)
        e = np.empty(T, np.float64)
        nu = np.empty(T, np.float64)
        seg = np.empty((stride, T), np.float64)
        j = np.empty(M, np.int64)
        st = np.empty(M, np.uint64)
        for b in range(M):
            st[b] = (np.uint64(seed0 + b) * np.uint64(0x2545F4914F6CDD1D)
                     + pair_key[p])
            st[b], _u = _split(st[b])
        for c in range(n_ckpt - 1, -1, -1):
            lo = c * stride
            n_seg = _replay_segment(counts[p], lam, log_lam, r, pi[p], esc[p],
                                    alpha_ck[p], c, stride, K, seg, e, nu, T)
            for k in range(lo + n_seg - 1, lo - 1, -1):
                a_k = seg[k - lo]
                if k == K - 1:
                    tot = 0.0
                    for i in range(T):
                        tot += a_k[i]
                        cum[i] = tot
                    for b in range(M):
                        st[b], u = _split(st[b])
                        jj = _search(cum, T, u * tot)
                        j[b] = jj
                        path[b, p, k] = jj
                    continue
                tot_q = 0.0
                for i in range(T):
                    q[i] = a_k[i] * (1.0 - r[k + 1, i])
                    tot_q += q[i]
                    cum[i] = tot_q
                _reset_row_k(pi[p], r, k + 1, nu, T)
                for b in range(M):
                    jj = j[b]
                    stay_num = a_k[jj] * r[k + 1, jj]
                    denom = stay_num + nu[jj] * tot_q
                    pstay = stay_num / denom if denom > 0.0 else 1.0
                    st[b], u = _split(st[b])
                    if u >= pstay and tot_q > 0.0:
                        st[b], v = _split(st[b])
                        jj = _search(cum, T, v * tot_q)
                    j[b] = jj
                    path[b, p, k] = jj



_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO53 = 1.0 / 9007199254740992.0


@njit(cache=True, inline="always")
def _split(state):
    """splitmix64: advance the state and return a uniform in ``[0, 1)``."""
    state = state + _GOLD
    z = state
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    z = z ^ (z >> np.uint64(31))
    return state, np.float64(z >> np.uint64(11)) * _TWO53


@njit(cache=True, inline="always")
def _search(cum, T, v):
    """First index ``i`` with ``v <= cum[i]``."""
    lo = 0
    hi = T - 1
    while lo < hi:
        mid = (lo + hi) >> 1
        if v <= cum[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ------------------------------------------------------------ scoring


#: Ceiling on the per-call scoring scratch, ``member_chunk x windows x ...``
#: packing arrays. Windows are scored independently, in blocks sized to fit.
#: The draws are not blocked: FFBS walks the whole block chain backwards.
MAX_SCRATCH_BYTES = 256 * 1024 ** 2


def _window_block(n_windows, n_members, n_hap):
    """How many windows to score at once, under :data:`MAX_SCRATCH_BYTES`.

    Floored at 256 so the ``prange`` over windows still saturates the cores.
    """
    nslot = 2 * n_hap + 1
    per_window = n_members * (
        nslot * 8  # branch_t
        + (2 * n_hap - 1) * 4 + (n_hap - 1) * 32 + 6 * nslot * 4)
    block = int(MAX_SCRATCH_BYTES // max(per_window, 1))
    # The floor never exceeds the windows that exist, and at least one window.
    return max(min(max(block, 256), n_windows), 1)


class SegmentEnsemble:
    """Posteriors for one segment, marginalised over sampled genealogies.

    Wraps the compiled kernels above around a resolved
    :class:`~ancestree.local_tree_inference.LocalTreeBuilder`: the pairwise HMM
    forward pass is run once, then genealogies are drawn from that posterior in
    chunks, scored, and folded into a running per-site sum.

    :param builder: A resolved builder for one segment.
    :param model: Substitution model supplying ``transition_probs``.
    :param n_hap: Haplotypes in the panel.
    :param ingroup: Ingroup haplotype indices into the panel, or ``None``
        for the whole panel. Resolved by identity, so the ingroup need
        not be the leading block of columns.
    :param pi: Equilibrium frequencies for ``Q``, or ``None`` for the model's
        own. The root prior is NOT applied here --- ``posterior`` returns a
        normalised likelihood, and the caller weights it by the prior.
    """

    #: Sites given a uniform posterior because their marginalised mass
    #: underflowed to zero, set by :meth:`SegmentEnsemble.posterior`.
    n_uniform_fallback: int = 0


    def __init__(self, builder, model, n_hap: int, ingroup,
                 pi=None) -> None:
        from itertools import combinations
        from ancestree.local_tree_inference import PairwiseCoalescentHMM
        self.model, self.n_hap = model, n_hap
        self.is_ingroup = np.zeros(n_hap, dtype=np.bool_)
        if ingroup is None:
            self.is_ingroup[:] = True
        else:
            for i in ingroup:
                self.is_ingroup[int(i)] = True
        if not self.is_ingroup.any():
            raise ValueError(
                "the ingroup is empty, so there is no ingroup MRCA to report "
                "at; name at least one ingroup haplotype, or leave the ingroup "
                "unset to use the whole panel")
        #: Equilibrium frequencies for Q, from the run's base composition.
        self._pi = None if pi is None else np.asarray(pi, dtype=float)
        self.mu = builder.mu
        self.builder = builder  # kept for iter_tree_sequences()'s window geometry

        genotypes, positions, block_of_site, n_blocks = builder._genotype_matrix()
        self.positions = positions.astype(np.int64)
        self.tip_states = genotypes
        self.n_sites = len(positions)
        intervals = builder._window_intervals()
        bounds = [builder._blocks_for_window(lo, hi, n_blocks)
                  for lo, hi in intervals]

        # A window's blocks are contiguous, so the range is kept directly.
        self.blk_lo = np.array([b0 for b0, _ in bounds], dtype=np.int64)
        self.blk_hi = np.array([b1 for _, b1 in bounds], dtype=np.int64)

        hmm = PairwiseCoalescentHMM(
            n_hap, mu=builder.mu, rec_rate=builder.rec_rate,
            block_size=builder.block_size, n_time_bins=builder.n_time_bins,
            time_grid=builder.time_grid)
        pairs = list(combinations(range(n_hap), 2))
        self.pair_key = _pair_stream_keys(
            builder.sample_names, pairs)
        # int16 per-block counts, widened to int32 only where the busiest
        # block holds more sites than int16 can count.
        max_per_block = int(np.bincount(block_of_site, minlength=n_blocks).max()) \
            if block_of_site.size else 0
        count_dtype = np.int16 if max_per_block <= np.iinfo(np.int16).max \
            else np.int32
        # ``esc`` is the fraction of each block over which the pair is called.
        # A difference is only observable where both haplotypes carry a call,
        # so this is the exposure the Poisson count was generated from.
        counts, esc = hmm._pair_block_counts(
            genotypes, block_of_site, n_blocks, dtype=count_dtype)
        step, block_mask, emit_scale = builder._block_geometry(n_blocks)
        edges, t_rep, _, _, _, t_bar = hmm.calibrate(
            counts, emit_scale=emit_scale, block_mask=block_mask,
            called_frac=esc)
        base_lam = 2.0 * hmm.mu * hmm.block_size * t_rep
        lam = np.repeat(base_lam[None, :], n_blocks, axis=0)
        if emit_scale is not None:
            lam *= emit_scale[:, None]
        if block_mask is not None:
            lam[~block_mask] = 0.0
        log_lam = np.log(np.clip(lam, 1e-300, None))
        base_r = 2.0 * hmm.rec_rate * hmm.block_size * t_rep
        r = np.exp(-np.repeat(base_r[None, :], n_blocks, axis=0)
                   * (1.0 if step is None else
                      np.concatenate(([1.0], np.asarray(step, float)))[:, None]))
        prior = np.stack([hmm._coalescent_prior(edges, t_bar[p])
                          for p in range(len(pairs))])
        # Checkpoint the forward matrix so only every stride-th block is
        # retained; the backward walk replays each segment as it arrives.
        self.ckpt_stride = _ckpt_stride(n_blocks, len(pairs), hmm.n_time_bins)
        n_ckpt = int(np.ceil(n_blocks / self.ckpt_stride))
        self.alpha_ck = np.empty((len(pairs), n_ckpt, hmm.n_time_bins),
                                 dtype=np.float64)
        hmm_forward_ckpt(counts, lam, log_lam, r, prior, esc,
                         self.ckpt_stride, self.alpha_ck)
        self.hmm_counts, self.hmm_lam, self.hmm_log_lam = counts, lam, log_lam
        self.hmm_esc = esc
        self.n_blocks = n_blocks
        self.hmm_r, self.hmm_pi, self.t_rep = r, prior, t_rep
        # Linear edges and the per-pair coalescent mean, so the within-bin draw
        # can follow the same Exp(1/t_bar) prior the chain was built on.
        self.edge_lo = edges[:-1].copy()
        self.edge_hi = edges[1:].copy()
        self.t_bar = np.asarray(t_bar, np.float64)

        # Sites are assigned by the same tiling _window_intervals() uses.
        edges = np.array([lo for lo, _ in intervals] + [intervals[-1][1]],
                         dtype=float)
        win_of_site = np.clip(
            np.searchsorted(edges, self.positions, side="right") - 1,
            0, len(intervals) - 1).astype(np.int64)
        self.needed = np.unique(win_of_site)
        order = np.argsort(win_of_site, kind="stable")
        self.row_idx = order.astype(np.int64)
        per_win = np.bincount(
            np.searchsorted(self.needed, win_of_site[order]),
            minlength=len(self.needed))
        self.row_off = np.zeros(len(self.needed) + 1, dtype=np.int64)
        self.row_off[1:] = np.cumsum(per_win)

    def _buffers(self, M, W):
        """Per-chunk packing and transition-matrix scratch."""
        n = self.n_hap
        n_all, nslot = 2 * n - 1, 2 * n + 1
        return (np.empty((M, W, n - 1, 4)), np.empty((M, W, nslot)),
                np.empty((M, W, n_all), np.int32), np.empty((M, W, n), np.int32),
                np.empty((M, W), np.int32), np.empty((M, W, nslot), np.int32),
                np.empty((M, W, nslot + 1), np.int32),
                np.empty((M, W, nslot), np.int32), np.empty((M, W, n), np.int32),
                np.empty((M, W), np.int64))

    def _score_chunk(self, path, fraction, n_coal, out, arrays, cond,
                     seed, member0):
        """Reduce and score one member chunk, a window block at a time.

        The window block is the pipeline's unit: its condensed matrices are
        built from the sampled paths, scored, and overwritten, so the full
        ``(member, window, pair)`` array is never formed. Segmentation is
        bounded by memory, not by contig boundaries.
        """
        W_all = len(self.needed)
        step = self._w_block
        for w0 in range(0, W_all, step):
            n_w = min(step, W_all - w0)
            blk = cond[:, :n_w]
            paths_to_condensed(path, self.edge_lo, self.edge_hi, self.t_bar,
                               self.needed, self.blk_lo, self.blk_hi, w0,
                               self.pair_key, seed, member0, blk)
            self._score_window_block(blk, fraction, n_coal, out, arrays, w0)
        return out

    def _score_window_block(self, condensed, fraction, n_coal, out, arrays, w0):
        """Score one block of windows starting at ``w0`` into ``out``."""
        n = self.n_hap
        n_all, nslot = 2 * n - 1, 2 * n + 1
        M, W = condensed.shape[0], condensed.shape[1]
        # A short final block gets its own contiguous buffers, so the reshapes
        # below are views.
        (Z, branch_t, parent0, sample_dense, fmrca, pidx, cofs, cflat, poi,
         root) = (arrays if W == self._w_block else self._buffers(M, W))
        condensed = np.ascontiguousarray(condensed)
        upgma_batch(condensed.reshape(M * W, -1), n, Z.reshape(M * W, n - 1, 4))
        pack_base_all(Z.reshape(M * W, n - 1, 4), n, self.is_ingroup, self.mu,
                      branch_t.reshape(M * W, nslot),
                      parent0.reshape(M * W, n_all),
                      sample_dense.reshape(M * W, n), fmrca.reshape(M * W))
        reroot_all(parent0.reshape(M * W, n_all),
                   branch_t.reshape(M * W, nslot), fmrca.reshape(M * W),
                   n, float(fraction), int(n_coal), pidx.reshape(M * W, nslot),
                   cofs.reshape(M * W, nslot + 1), cflat.reshape(M * W, nslot),
                   poi.reshape(M * W, n), root.reshape(M * W))
        score_ensemble(branch_t, self._lam, self._V, self._Vinv,
                       pidx, poi, cofs, cflat, root, sample_dense,
                       self.tip_states, self.row_off[w0:w0 + W + 1],
                       self.row_idx, nslot, S_STATES, out)

    def _path_chunk(self, member_chunk: int, n_members: int) -> int:
        """Members whose sampled paths, ``(n_pairs, n_blocks) int8`` each,
        fit :data:`MAX_PATH_BYTES`, reduced to a divisor of the ensemble size.
        Each member chunk pays one backward pass and its checkpoint replay.
        """
        per = max(self.alpha_ck.shape[0] * self.n_blocks, 1)
        cap = max(int(MAX_PATH_BYTES // per), 1)
        if member_chunk <= cap:
            return member_chunk
        c = min(cap, member_chunk)
        while n_members % c:
            c -= 1
        return c

    def _chunk_size(self, n_members: int, member_chunk: int) -> int:
        """Genealogies drawn at once.

        The largest divisor of ``n_members`` no greater than ``member_chunk``,
        so any ensemble size is accepted. Draws
        stream from ``seed + start``, so the chunking does not reach the
        result. It only bounds peak memory.

        :param n_members: Ensemble size.
        :param member_chunk: Upper bound on genealogies held at once.
        :return: Chunk size dividing ``n_members``.
        """
        chunk = min(member_chunk, n_members)
        while n_members % chunk:
            chunk -= 1
        if chunk < member_chunk:
            logging.getLogger(f"ancestree.{type(self).__name__}").info(
                "member_chunk=%d does not divide n_ensemble=%d; drawing %d at "
                "a time instead", member_chunk, n_members, chunk)
        return chunk

    def draw_chunks(self, n_members: int, *, member_chunk: int = 8,
                    seed: int = 0):
        """Yield the drawn genealogies, ``member_chunk`` at a time.

        Each yield is a ``(chunk, n_needed_windows, n_pairs)`` array of condensed
        pairwise-TMRCA matrices, the draws before UPGMA. The buffer is reused
        between yields, so a consumer that keeps a chunk must copy it. Member
        ``b`` draws from stream ``seed + b`` here and in :meth:`posterior`, so
        the materialised ensemble is the one the posterior was scored on.

        :param n_members: Ensemble size :math:`B`.
        :param member_chunk: Genealogies drawn at once.
        :param seed: Base seed.
        """
        chunk = self._path_chunk(
            self._chunk_size(n_members, member_chunk), n_members)
        path = self._draw_paths(chunk)
        draws = np.empty((chunk, len(self.needed), self.alpha_ck.shape[0]))
        for start in range(0, n_members, chunk):
            self._paths_into(path, chunk, start, seed)
            paths_to_condensed(path, self.edge_lo, self.edge_hi, self.t_bar,
                               self.needed, self.blk_lo, self.blk_hi, 0,
                               self.pair_key, seed, start, draws)
            yield draws

    def _draw_paths(self, chunk):
        """Scratch for one member chunk's sampled bin paths, ``(M, P, K)``."""
        return np.empty((chunk, self.alpha_ck.shape[0], self.n_blocks), np.int8)

    def _paths_into(self, path, chunk, start, seed):
        """Sample one member chunk's paths in place."""
        ffbs_paths(self.hmm_counts, self.hmm_lam, self.hmm_log_lam,
                   self.alpha_ck, self.ckpt_stride, self.n_blocks,
                   self.hmm_r, self.hmm_pi, self.hmm_esc, seed + start,
                   self.pair_key, path)

    def iter_tree_sequences(self, n_members: int, *, member_chunk: int = 8,
                            seed: int = 0):
        """Yield the ensemble's genealogies as tree sequences, one per member.

        Reifies exactly the draws :meth:`posterior` marginalises over, one at a
        time: peak memory holds a single member, not all :math:`B`. Only the
        windows carrying sites are drawn, so the spans between them hold no
        edges (an isolated-samples tree).

        The members are topology only --- no sites, unlike the genotype-baked
        plug-in tree from :meth:`LocalTreeBuilder.to_tree_sequence`. They are
        the sampled genealogies, not self-describing ARGs. Re-scoring one needs
        the genotypes supplied alongside it.

        :param n_members: Ensemble size :math:`B`.
        :param member_chunk: Genealogies drawn at once (memory bound).
        :param seed: Base seed. Member ``b`` draws from stream ``seed + b``.
        """
        from ancestree._upgma import build_windowed_tree_sequence
        b = self.builder
        all_intervals = b._window_intervals()
        intervals = [all_intervals[i] for i in self.needed]
        n = self.n_hap
        Z = np.empty((len(self.needed), n - 1, 4))
        for draws in self.draw_chunks(n_members, member_chunk=member_chunk,
                                      seed=seed):
            for member in draws:
                upgma_batch(member, n, Z)
                yield build_windowed_tree_sequence(
                    n, b.sequence_length, intervals, list(Z),
                    sample_names=b.sample_names)

    def posterior(self, fraction: float, n_members: int, *, n_coal: int = -1,
                  member_chunk: int = 8, seed: int = 0):
        """Per-site posterior marginalised over ``n_members`` genealogies.

        :param fraction: Reporting position along the ingroup-MRCA-to-root
            path; ``0.0`` is the ingroup MRCA and ``1.0`` the panel root.
        :param n_members: Ensemble size :math:`B`.
        :param n_coal: Coalescences to climb above the anchor when the focal
            node is placed by join count. ``-1`` means the placement is not a
            coalescence count, so ``fraction`` positions the readout instead.
        :param member_chunk: Genealogies drawn and scored at once. Bounds peak
            memory, which does not otherwise grow with ``n_members``.
        :param seed: Base seed. Member ``b`` draws from stream ``seed + b``.
        :return: ``(n_sites, 4)`` posterior, rows aligned with ``positions``.
        """
        chunk = self._path_chunk(self._chunk_size(n_members, member_chunk),
                                 n_members)
        W = len(self.needed)
        eig = self.model.real_eig(pi=self._pi)
        if eig is None:
            raise ValueError(
                f"{type(self.model).__name__} has no real eigendecomposition of "
                f"Q, which the ensemble's Felsenstein kernel needs to rebuild "
                f"exp(Q·t) per branch. Every substitution model provided with "
                f"ancestree is reversible and does; a custom non-reversible or "
                f"non-diagonalisable model cannot be used in ensemble mode.")
        self._lam, self._V, self._Vinv = eig
        self._w_block = _window_block(W, chunk, self.n_hap)
        acc = np.zeros((self.n_sites, S_STATES))
        ref = np.full(self.n_sites, -1e300)
        arrays = self._buffers(chunk, self._w_block)
        out = np.empty((chunk, self.n_sites, S_STATES))
        path = self._draw_paths(chunk)
        cond = np.empty((chunk, self._w_block, self.alpha_ck.shape[0]))
        for start in range(0, n_members, chunk):
            self._paths_into(path, chunk, start, seed)
            res = self._score_chunk(path, fraction, n_coal, out, arrays,
                                    cond, seed, start)
            top = res.max(axis=(0, 2))
            new = np.maximum(ref, top)
            acc *= np.exp(ref - new)[:, None]
            acc += np.exp(res - new[None, :, None]).sum(axis=0)
            ref = new
        total = acc.sum(axis=1, keepdims=True)
        good = total[:, 0] > 0
        self.n_uniform_fallback = int((~good).sum())
        post = np.full_like(acc, 1.0 / S_STATES)
        post[good] = acc[good] / total[good]
        return post
