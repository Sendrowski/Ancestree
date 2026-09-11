"""Reusable per-chunk ARG-mode inference helpers used by the streaming
production script and the kernel benchmark.

Two entry points:

- :func:`infer_chunk_batched` — load one ``.npz`` chunk, group SNPs by
  local-tree interval, and run the Felsenstein kernel once per tree on
  the batched site list. Returns ``(positions, calls, posteriors)``.
- :func:`infer_chunk_batched_fast` — same I/O but calls the kernel with a
  singleton site list per SNP, the unbatched baseline the streaming
  script used in v1. Kept so the benchmark can quantify the batching
  win directly.

The helper expects the parent process to have loaded the ``tskit`` ARG
already so the same ARG object can be shared across workers (under fork)
or across chunks (in the streaming run).
"""
from __future__ import annotations


import numpy as np
import tskit

from ancestree import JC69, STATE_INDEX, STATES
from ancestree.inference import ARGBasedInference
from ancestree.likelihood import Likelihood
from ancestree.posterior import Posterior
from ancestree.priors import StationaryPrior
from ancestree.sites import Site
from ancestree.trees import TskitLocalTree


NUC: tuple[str, str, str, str] = ("A", "C", "G", "T")


def build_inference(
    ts: tskit.TreeSequence, time_scale: float,
) -> tuple[ARGBasedInference, Likelihood, list[str]]:
    """Construct the ``ARGBasedInference`` + ``Likelihood`` engine pair
    used by both batched and per-site code paths.

    Returns the inference object (carries the sample_map), the engine,
    and the per-sample-node name list — all things both code paths need.
    """
    inference = ARGBasedInference(
        source=ts, model=JC69(), mu=time_scale,
        progress=False,
    )
    engine = Likelihood(JC69())
    sample_nodes = ts.samples()
    sample_names = [inference._node_to_sample[int(n)] for n in sample_nodes]
    return inference, engine, sample_names


def _row_to_site(
    pos: int,
    row: np.ndarray,
    sample_names: list[str],
) -> Site | None:
    """Convert one ``.npz`` ingroup row + position into a :class:`Site`.

    Returns ``None`` if the site has fewer than 2 distinct observed
    alleles (would be a no-op for inference, matches upstream filter).
    """
    obs_idx = sorted({int(a) for a in row if a >= 0})
    if len(obs_idx) < 2:
        return None
    alleles_chars = tuple(NUC[a] for a in obs_idx)
    tip_alleles: dict[str, str | None] = {
        name: (NUC[int(row[h])] if int(row[h]) >= 0 else None)
        for h, name in enumerate(sample_names)
    }
    return Site(
        chrom="1",
        pos=int(pos),
        alleles=alleles_chars,
        tip_alleles=tip_alleles,
        local_tree_handle=float(pos),
    )


def _posteriors_for_tree(
    tree: tskit.Tree,
    sites_in_tree: list[Site],
    inference: ARGBasedInference,
    engine: Likelihood,
    time_scale: float,
) -> np.ndarray:
    """Run the Felsenstein kernel on a batch of sites that share ``tree``.

    Handles the multi-root local-tree case by marginalising posteriors
    over the per-root subtrees — same behaviour as
    :meth:`ARGBasedInference.infer`.

    :return: ``(B, n_states)`` array of posteriors aligned with ``sites_in_tree``.
    """
    if tree.num_roots == 1:
        lt = TskitLocalTree.from_tskit_tree(
            tree, sample_map=inference.sample_map,
        )
        lt.time_scale = time_scale
        return inference._compute_posteriors(engine, lt, sites_in_tree)
    # A root that is an isolated sample tip is its own ancestor, so reading
    # there returns a point mass on the allele that tip carries. Dropping them
    # matches ARGBasedInference.infer, which the merge is compared against.
    roots = [int(r) for r in tree.roots
             if not (tree.is_sample(int(r)) and tree.num_children(int(r)) == 0)]
    if not roots:
        roots = [int(r) for r in tree.roots]
    per_root = []
    for r in roots:
        lt = TskitLocalTree.from_tskit_tree(
            tree, sample_map=inference.sample_map, root=int(r),
        )
        lt.time_scale = time_scale
        per_root.append(
            inference._compute_posteriors(engine, lt, sites_in_tree)
        )
    return np.mean(per_root, axis=0)


def _posteriors_for_tree_fast(
    tree: tskit.Tree,
    tip_states: np.ndarray,
    sample_nodes: np.ndarray,
    node_times: np.ndarray,
    engine: Likelihood,
    log_prior_per_state: np.ndarray,
    time_scale: float,
    id_map_scratch: np.ndarray | None = None,
):
    """Tskit-native fast path: build posteriors for SNPs sharing ``tree``.

    Bypasses :class:`~ancestree.trees.TskitLocalTree` + per-site
    :class:`~ancestree.sites.Site` materialisation. Calls
    :meth:`~ancestree.likelihood.Likelihood.log_likelihoods_tskit_native`
    directly on raw tskit numpy buffers + an ``(B, n_samples) int8``
    tip-state matrix. Applies a per-state log-prior (same for every
    site in ARG mode under :class:`StationaryPrior`) and softmax-
    normalises inline.

    Multi-root local trees are still routed through the generic Tree
    path for correctness (rare, not worth duplicating the
    marginalisation logic here). Returns ``None`` in that case so the
    caller falls back.

    :param log_prior_per_state: ``(n_states,)`` array of ``log π(state)``.
        For ``StationaryPrior + JC69`` this is ``log(1/4)`` per state.
    """
    if tree.num_roots != 1:
        return None
    log_L = engine.log_likelihoods_tskit_native(
        tree, tip_states, sample_nodes, node_times, time_scale,
        id_map_scratch=id_map_scratch,
    )
    log_post = log_L + log_prior_per_state[None, :]
    # Stable softmax.
    log_post = log_post - log_post.max(axis=1, keepdims=True)
    p = np.exp(log_post)
    s = p.sum(axis=1, keepdims=True)
    # Degenerate-row fallback to uniform (matches _normalise_log_post).
    bad = ~(s > 0).squeeze(axis=1)
    if bad.any():
        p[bad] = 1.0
        s[bad] = float(log_prior_per_state.size)
    p /= s
    return p


def infer_chunk_batched(
    ts: tskit.TreeSequence,
    positions: np.ndarray,
    ingroup: np.ndarray,
    inference: ARGBasedInference,
    engine: Likelihood,
    sample_names: list[str],
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """Group SNPs by local-tree interval. Run the kernel once per tree.

    Auto-dispatches to the tskit-native fast path
    (:func:`infer_chunk_batched_fast`) when the inference's prior is
    :class:`~ancestree.priors.StationaryPrior` (the ARG-mode default,
    yields ~1.4× over the generic path by skipping ``TskitLocalTree``
    + dict-based ``Site`` construction). Falls through to the generic
    Tree-based path for other priors.

    Walks ``positions`` forward (must be sorted), accumulating sites
    that fall in the current ``tree.interval``. When the next position
    crosses into a new tree, flushes the batch through the Felsenstein
    kernel (one call per tree, not per SNP).

    :return: ``(written_pos, map_alleles, posteriors, stats)`` where
        ``written_pos`` is a sorted ``int32`` array of called positions,
        ``map_alleles`` is the per-site integer state index (A=0…T=3),
        ``posteriors`` is a ``(N, 4) float32`` matrix, and ``stats``
        carries per-chunk counts.
    """
    # The fast path needs a stationary root prior. The kernel is always
    # compiled now (numba is a hard dependency and the interpreted path was
    # removed), so there is nothing else to gate on.
    if isinstance(inference.prior, StationaryPrior):
        return infer_chunk_batched_fast(
            ts, positions, ingroup, inference, engine, sample_names, time_scale,
        )

    n_snps = len(positions)
    # Jump straight to the first SNP's tree rather than walking from
    # ts.first() (chr1 has 2.26M trees, if our chunk starts at 118 Mb
    # we'd otherwise burn ~1 M tree.next() calls just to reach the data).
    tree = ts.at(int(positions[0])) if n_snps else ts.first()

    written_pos: list[int] = []
    map_idx: list[int] = []
    posts_acc: list[list[float]] = []
    n_skipped = n_multi_root = n_trees_used = 0

    i = 0
    while i < n_snps:
        pos = int(positions[i])
        while pos >= tree.interval.right:
            if not tree.next():
                break
        if pos < tree.interval.left or pos >= tree.interval.right:
            n_skipped += 1
            i += 1
            continue

        right = tree.interval.right
        sites_in_tree: list[Site] = []
        site_positions: list[int] = []
        while i < n_snps and int(positions[i]) < right:
            site = _row_to_site(int(positions[i]), ingroup[i], sample_names)
            if site is not None:
                sites_in_tree.append(site)
                site_positions.append(int(positions[i]))
            i += 1

        if not sites_in_tree:
            continue
        n_trees_used += 1
        if tree.num_roots > 1:
            n_multi_root += len(sites_in_tree)

        posteriors = _posteriors_for_tree(
            tree, sites_in_tree, inference, engine, time_scale,
        )
        for s_pos, post in zip(site_positions, posteriors):
            p = Posterior(alleles=STATES, values=post)
            map_allele = p.map_allele
            if map_allele not in STATE_INDEX:
                continue
            written_pos.append(s_pos)
            map_idx.append(STATE_INDEX[map_allele])
            posts_acc.append([float(v) for v in p.values])

    stats = {
        "n_sites_in_chunk": int(n_snps),
        "n_sites_written": int(len(written_pos)),
        "n_skipped_out_of_tree": int(n_skipped),
        "n_multi_root": int(n_multi_root),
        "n_trees_with_sites": int(n_trees_used),
    }
    return (
        np.asarray(written_pos, dtype=np.int32),
        np.asarray(map_idx, dtype=np.int8),
        np.asarray(posts_acc, dtype=np.float32),
        stats,
    )


def infer_chunk_batched_fast(
    ts: tskit.TreeSequence,
    positions: np.ndarray,
    ingroup: np.ndarray,
    inference: ARGBasedInference,
    engine: Likelihood,
    sample_names: list[str],
    time_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int | float]]:
    """Tskit-native batched inference — same I/O as :func:`infer_chunk_batched`.

    Bypasses the generic :class:`~ancestree.trees.Tree`/
    :class:`~ancestree.sites.Site` API in the hot loop by calling
    :func:`_posteriors_for_tree_fast` directly with raw tskit buffers +
    an ``(B, n_samples) int8`` tip-state matrix. Falls back to the
    generic path only for multi-root local trees (rare).

    Pre-conditions:
    - The inference's prior is :class:`StationaryPrior` (the ARG-mode
      default). Other priors require per-site evaluation and aren't
      supported by this fast path yet.
    """
    n_snps = len(positions)
    sample_nodes = np.asarray(ts.samples(), dtype=np.int32)
    # ``ts.tables.nodes.time`` materialises the full TableCollection
    # (a ~7 s dump of all 197 M nodes on the MSL chr1 ARG). The direct
    # ``ts.nodes_time`` accessor returns the underlying buffer in <1 ms.
    node_times = np.asarray(ts.nodes_time, dtype=np.float64)

    # Per-state log-prior — static under StationaryPrior + JC69 (uniform).
    log_prior_per_state = inference.prior._log_prior_per_state.copy()

    # Pre-allocate the dense node-id scratch (ts.num_nodes int32 ≈ 800 MB
    # on the MSL chr1 ARG with 197M nodes). Allocating it once amortises
    # over every per-tree call inside the chunk. The engine just writes
    # the n_active active entries and resets them on the way out.
    id_map_scratch = np.full(int(ts.num_nodes), -1, dtype=np.int32)

    tree = ts.at(int(positions[0])) if n_snps else ts.first()

    written_pos: list[int] = []
    map_idx: list[int] = []
    posts_acc: list[np.ndarray] = []
    n_skipped = n_multi_root = n_trees_used = 0

    NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

    i = 0
    while i < n_snps:
        pos = int(positions[i])
        while pos >= tree.interval.right:
            if not tree.next():
                break
        if pos < tree.interval.left or pos >= tree.interval.right:
            n_skipped += 1
            i += 1
            continue

        right = tree.interval.right
        site_rows: list[np.ndarray] = []
        site_positions: list[int] = []
        while i < n_snps and int(positions[i]) < right:
            row = ingroup[i]
            # Same filter as the generic path: keep only sites with ≥2
            # distinct observed ingroup alleles.
            obs_idx_set = {int(a) for a in row if a >= 0}
            if len(obs_idx_set) >= 2:
                site_rows.append(row)
                site_positions.append(int(positions[i]))
            i += 1

        if not site_rows:
            continue
        n_trees_used += 1
        tip_states = np.asarray(site_rows, dtype=np.int8)  # (B, n_samples)

        if tree.num_roots != 1:
            # Multi-root: fall back to the generic batched path.
            n_multi_root += tip_states.shape[0]
            # Build Site objects for the generic path
            local_sites = [
                _row_to_site(p_, r_, sample_names)
                for p_, r_ in zip(site_positions, site_rows)
            ]
            local_sites = [s for s in local_sites if s is not None]
            if not local_sites:
                continue
            posteriors = _posteriors_for_tree(
                tree, local_sites, inference, engine, time_scale,
            )
        else:
            posteriors = _posteriors_for_tree_fast(
                tree, tip_states, sample_nodes, node_times,
                engine, log_prior_per_state, time_scale,
                id_map_scratch=id_map_scratch,
            )

        for s_pos, post in zip(site_positions, posteriors):
            map_state = int(np.argmax(post))
            written_pos.append(s_pos)
            map_idx.append(map_state)
            posts_acc.append(post.astype(np.float32))

    stats = {
        "n_sites_in_chunk": int(n_snps),
        "n_sites_written": int(len(written_pos)),
        "n_skipped_out_of_tree": int(n_skipped),
        "n_multi_root": int(n_multi_root),
        "n_trees_with_sites": int(n_trees_used),
    }
    return (
        np.asarray(written_pos, dtype=np.int32),
        np.asarray(map_idx, dtype=np.int8),
        (np.stack(posts_acc) if posts_acc else np.zeros((0, 4), dtype=np.float32)),
        stats,
    )
