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
