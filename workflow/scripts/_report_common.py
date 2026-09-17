"""Helpers shared by the benchmark report scripts.

Summary statistics and Markdown cell formatters for the tabulating reports, and
the panel selection and truth lookup the focal-interpolation reports grade
against.
"""
import json
import math
import statistics as stats

import numpy as np


def _load(path: str) -> dict:
    """Read one JSON file.

    :param path: Path to the file.
    :return: The decoded object.
    """
    with open(path) as f:
        return json.load(f)


def _entropy_bits(probs: list[float]) -> float:
    """Shannon entropy in bits. Safe on zero-probability bins.

    :param probs: Probabilities of the states, summing to one.
    :return: ``-sum(p * log2(p))`` over the nonzero entries.
    """
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log2(p)
    return s


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, ignoring NaN entries.

    :param values: The values to summarise.
    :return: ``(mean, std)``. The standard deviation is ``0`` for a single
        value, and both are NaN when no value remains.
    """
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return stats.mean(vals), stats.stdev(vals)


def _fmt_rate(x):
    """Format a rate as a fraction and a percentage.

    :param x: The rate in ``[0, 1]``, or ``None``.
    :return: ``"0.1234 (12.34%)"``, or an em dash for ``None``.
    """
    return "—" if x is None else f"{x:.4f} ({100*x:.2f}%)"


def _fmt_diff(x):
    """Format a difference in scientific notation.

    :param x: The difference, or ``None``.
    :return: The value to three decimals, or an em dash for ``None``.
    """
    return "—" if x is None else f"{x:.3e}"


def _fmt_secs(x):
    """Format a duration in seconds.

    :param x: Seconds, or ``None``.
    :return: The value to three decimals, or an em dash for ``None``.
    """
    return "—" if x is None else f"{x:.3f}"


def _panel(ts, n_out):
    """Ingroup nodes and an outgroup subset spread across split depths.

    :param ts: The simulated tree sequence.
    :param n_out: Number of outgroups to retain.
    :return: ``(ingroup_nodes, outgroup_nodes)``.
    """
    name = {p.id: (p.metadata or {}).get("name") for p in ts.populations()}
    ingroup = [int(n) for n in ts.samples()
               if name[ts.node(int(n)).population] == "ingroup"]
    outgroups = [int(n) for n in ts.samples()
                 if str(name[ts.node(int(n)).population]).startswith("outgroup")]
    outgroups.sort(key=lambda n: int(name[ts.node(n).population].split("_")[1]))
    if n_out >= len(outgroups):
        chosen = outgroups
    else:
        index = np.linspace(0, len(outgroups) - 1, n_out).round().astype(int)
        chosen = [outgroups[i] for i in dict.fromkeys(index)]
    return ingroup, chosen


def _truth_at(tree, site, node, height=0.0):
    """Simulated state at a point ``height`` above ``node``.

    A mutation sits at a time on the branch above its own node, so one on the
    very edge being split may fall either side of the point. Reading the state
    at the node below would quantise the truth to nodes while the estimate
    moves continuously, so compare times rather than topology alone.

    :param tree: The local tree covering the site.
    :param site: The :class:`tskit.Site` carrying the mutations.
    :param node: Node immediately below the point.
    :param height: Distance above ``node``, in the tree's own time units.
    :return: The simulated state at that point.
    """
    point_time = tree.time(node) + height
    state, best = site.ancestral_state, np.inf
    for mutation in site.mutations:
        on_path = (mutation.node == node
                   or tree.is_descendant(node, mutation.node))
        if not on_path:
            continue
        # Prefer the mutation's own time. Fall back to its node's when the
        # simulation left it unknown.
        time = float(mutation.time)
        if np.isnan(time):
            time = tree.time(mutation.node)
        if time <= point_time:
            continue  # below the point: the point predates it
        if time < best:
            best, state = time, mutation.derived_state
    return state
