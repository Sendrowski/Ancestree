"""Conversion between msprime mutation events and substitutions per site.

``msprime`` mutation models count mutation EVENTS: the per-event
transition matrix ``P`` is the unnormalised rate matrix divided by its
largest row sum, with the remainder placed on the diagonal, so an event
leaves the state unchanged with probability ``P_ii``. Ancestree's
substitution models are scaled to one expected substitution per site per
unit branch length, so a simulated event rate has to be multiplied by the
fraction of events that change state before it can be compared with a
fitted branch rate.
"""
from __future__ import annotations

import numpy as np


def substitutions_per_mutation(model) -> float:
    """Expected substitutions per mutation event under an msprime model.

    Computes ``rho = 1 - sum_i pi_i P_ii``, where ``pi_i`` is the
    stationary probability of state ``i`` (``model.root_distribution``)
    and ``P_ii`` is the probability that a mutation event at a site in
    state ``i`` leaves it in state ``i`` (the diagonal of
    ``model.transition_matrix``). Multiplying a per-site per-generation
    mutation-event rate by ``rho`` gives the per-site per-generation
    substitution rate.

    :param model: An ``msprime.MatrixMutationModel`` instance, carrying
        ``root_distribution`` and ``transition_matrix`` over the same
        allele ordering.
    :return: Expected substitutions per mutation event, dimensionless, in
        ``(0, 1]``.
    """
    pi = np.asarray(model.root_distribution, dtype=float)
    p_diag = np.diag(np.asarray(model.transition_matrix, dtype=float))
    return 1.0 - float(np.sum(pi * p_diag))
