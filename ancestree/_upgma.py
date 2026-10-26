"""Assemble per-window UPGMA trees into a sites-free tskit TreeSequence.

The ``n`` panel haplotypes are sample nodes ``0..n-1`` shared by every window,
and each window adds its own internal nodes and edges over its
``[left, right)`` interval. Merge heights of
:func:`~ancestree._ensemble.upgma_batch` are the internal-node times in
generations, bumped by a tiny epsilon where a tie would break tskit's
strictly increasing times.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np


def build_windowed_tree_sequence(
    n_samples: int,
    sequence_length: float,
    window_intervals: "Sequence[tuple[float, float]]",
    window_linkages: "Sequence[np.ndarray]",
    sample_names: "Sequence[str] | None" = None,
):
    """Assemble per-window SciPy linkage matrices ``(n-1, 4)``, rows
    ``[child1, child2, height, size]``, into a sites-free tskit tree sequence.

    :param n_samples: Panel haplotypes, the shared leaves.
    :param sequence_length: Region length in tskit coordinates.
    :param window_intervals: One ``(left, right)`` per window, in order.
    :param window_linkages: One SciPy linkage matrix per window (same length /
        order as ``window_intervals``).
    :param sample_names: Optional per-leaf names attached as individual metadata.
    :return: A :class:`tskit.TreeSequence` with branch lengths in generations.
    """
    import tskit

    tables = tskit.TableCollection(sequence_length=sequence_length)
    if sample_names is not None:
        tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
        sample_individual = [
            tables.individuals.add_row(metadata={"name": str(name)})
            for name in sample_names
        ]
    else:
        sample_individual = [-1] * n_samples

    internal_times: list[float] = []
    e_left: list[float] = []
    e_right: list[float] = []
    e_parent: list[int] = []
    e_child: list[int] = []

    next_internal_id = n_samples  # global counter across windows

    for (left, right), z in zip(window_intervals, window_linkages):
        # Read the linkage rows directly. Row i merges clusters z[i, 0] and
        # z[i, 1] at height z[i, 2], forming cluster n_samples + i, and the
        # rows come in non-decreasing height order, so one forward pass emits
        # every node and edge. Cluster ids below n_samples are the shared
        # tskit sample nodes; each merge takes a fresh global id. A merge time
        # is bumped above both children where a tie violates tskit's
        # strictly-increasing-time requirement.
        zc = np.ascontiguousarray(z, dtype=np.float64)
        n_merge = zc.shape[0]
        cid_node = np.empty(n_samples + n_merge, dtype=np.int64)
        cid_time = np.zeros(n_samples + n_merge, dtype=np.float64)
        cid_node[:n_samples] = np.arange(n_samples, dtype=np.int64)
        # Right is popped before left, which sets the id order internal
        # nodes receive.
        stack = [(n_samples + n_merge - 1, False)]
        while stack:
            cid, children_done = stack.pop()
            if cid < n_samples:
                continue  # a sample node
            row = cid - n_samples
            if not children_done:
                stack.append((cid, True))
                stack.append((int(zc[row, 0]), False))
                stack.append((int(zc[row, 1]), False))
                continue
            lc = int(zc[row, 0])
            rc = int(zc[row, 1])
            t = float(zc[row, 2])
            floor = cid_time[lc] if cid_time[lc] > cid_time[rc] else cid_time[rc]
            if t <= floor:
                # tie / zero-length merge, nudge above the deeper child.
                t = floor + (floor * 1e-6 if floor > 0.0 else 1e-6)
            my_id = next_internal_id
            next_internal_id += 1
            cid_node[cid] = my_id
            cid_time[cid] = t
            internal_times.append(t)
            e_left.extend((left, left))
            e_right.extend((right, right))
            e_parent.extend((my_id, my_id))
            e_child.extend((int(cid_node[lc]), int(cid_node[rc])))

    n_internal = len(internal_times)
    flags = np.zeros(n_samples + n_internal, dtype=np.uint32)
    flags[:n_samples] = tskit.NODE_IS_SAMPLE
    times = np.concatenate((np.zeros(n_samples, dtype=np.float64),
                            np.asarray(internal_times, dtype=np.float64)))
    individual = np.full(n_samples + n_internal, -1, dtype=np.int32)
    individual[:n_samples] = np.asarray(sample_individual, dtype=np.int32)
    tables.nodes.set_columns(flags=flags, time=times, individual=individual)
    tables.edges.set_columns(
        left=np.asarray(e_left, dtype=np.float64),
        right=np.asarray(e_right, dtype=np.float64),
        parent=np.asarray(e_parent, dtype=np.int32),
        child=np.asarray(e_child, dtype=np.int32),
    )

    tables.sort()
    tables.time_units = "generations"
    return tables.tree_sequence()


class WindowedUpgmaTreeSequence:
    """Assemble per-window UPGMA trees into a sites-free :class:`tskit.TreeSequence`.

    :param n_samples: Panel haplotypes, the shared leaves.
    :param sequence_length: Region length in tskit coordinates.
    :param window_intervals: One ``(left, right)`` per window, tiling
        ``[0, sequence_length)`` in order.
    :param window_condensed_tmrcas: One condensed pairwise-TMRCA vector
        per window (scipy ``pdist`` order: ``(0,1),(0,2),…,(n-2,n-1)``),
        lengths ``n_samples·(n_samples-1)/2``, in generations.
    :param sample_names: Optional per-leaf names, stored as individual
        metadata ``{"name": ...}`` of each sample node.
    :param window_weights: Optional ``(n_windows, n(n-1)/2)`` exposure of each
        pair in each window, which weights the cluster means. ``None`` or zero
        rows is the plain average.
    :raises ValueError: If ``n_samples < 2``, ``sample_names`` length does
        not match ``n_samples``, ``window_intervals`` and
        ``window_condensed_tmrcas`` differ in length, or any condensed
        vector is not ``n_samples·(n_samples-1)/2`` long.
    """

    def __init__(
        self,
        n_samples: int,
        sequence_length: float,
        window_intervals: Sequence[tuple[float, float]],
        window_condensed_tmrcas: Sequence[np.ndarray],
        sample_names: Sequence[str] | None = None,
        window_weights: "np.ndarray | None" = None,
    ) -> None:
        if n_samples < 2:
            raise ValueError(
                f"need at least 2 samples to build local trees, got {n_samples}"
            )
        if sample_names is not None and len(sample_names) != n_samples:
            raise ValueError(
                f"sample_names length {len(sample_names)} != n_samples "
                f"{n_samples}"
            )
        if len(window_intervals) != len(window_condensed_tmrcas):
            raise ValueError(
                f"window_intervals ({len(window_intervals)}) and "
                f"window_condensed_tmrcas ({len(window_condensed_tmrcas)}) "
                "must have the same length"
            )
        expected = n_samples * (n_samples - 1) // 2
        for k, condensed in enumerate(window_condensed_tmrcas):
            if len(condensed) != expected:
                raise ValueError(
                    f"window {k}: condensed TMRCA vector has length "
                    f"{len(condensed)}, expected n*(n-1)/2 = {expected}"
                )
        self.n_samples = int(n_samples)
        self.sequence_length = float(sequence_length)
        self.window_intervals = window_intervals
        self.window_condensed_tmrcas = window_condensed_tmrcas
        self.sample_names = sample_names
        self.window_weights = window_weights

    @cached_property
    def linkages(self) -> np.ndarray:
        """``(n_windows, n_samples - 1, 4)`` SciPy-format average linkage of
        each window."""
        from ancestree._ensemble import upgma_batch

        n_pairs = self.n_samples * (self.n_samples - 1) // 2
        condensed = np.ascontiguousarray(
            self.window_condensed_tmrcas,
            dtype=np.float64).reshape(-1, n_pairs)
        weights = self.window_weights
        if weights is None or not len(weights):
            weights = np.empty((0, n_pairs))
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        linkages = np.empty((len(condensed), self.n_samples - 1, 4))
        upgma_batch(condensed, self.n_samples, linkages, weights)
        return linkages

    def build(self):
        """Build the sites-free per-window UPGMA :class:`tskit.TreeSequence`.

        :return: A :class:`tskit.TreeSequence` with ``n_samples`` sample nodes,
            one local tree per window, branch lengths in generations. The
            internal nodes of window ``w`` are ``n_samples + w·(n_samples - 1)``
            onwards.
        """
        return build_windowed_tree_sequence(
            self.n_samples, self.sequence_length, self.window_intervals,
            list(self.linkages), sample_names=self.sample_names,
        )

    def window_tree(self, w: int):
        """The tree of window ``w`` as the only tree of its own tree sequence.
        Its node ``n_samples + i`` is node ``n_samples + w·(n_samples - 1) + i``
        of :meth:`build`.

        :param w: Index of the window.
        :return: The :class:`tskit.Tree`.
        """
        return build_windowed_tree_sequence(
            self.n_samples, 1.0, [(0.0, 1.0)], [self.linkages[w]]).first()
