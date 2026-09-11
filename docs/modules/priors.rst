Ingroup weights and priors
==========================

An :class:`~ancestree.priors.IngroupWeight` is the likelihood of the
observed ingroup alleles given each candidate state at the ingroup MRCA,
seeded on that node inside Felsenstein's recursion. It is passed as
``ingroup_weight=`` and applies to fixed-tree mode only. The
:class:`~ancestree.priors.StationaryPrior` is the prior over states at the
reporting node, passed as ``prior=``, and is accepted by every mode.

.. automodule:: ancestree.priors
   :exclude-members: IngroupWeight, StationaryPrior, KingmanIngroupWeight,
      AdaptiveIngroupWeight, NoIngroupWeight

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   IngroupWeight
   StationaryPrior
   KingmanIngroupWeight
   AdaptiveIngroupWeight
   NoIngroupWeight

.. autoclass:: ancestree.priors.IngroupWeight

.. autoclass:: ancestree.priors.StationaryPrior

.. autoclass:: ancestree.priors.KingmanIngroupWeight

.. autoclass:: ancestree.priors.AdaptiveIngroupWeight

.. autoclass:: ancestree.priors.NoIngroupWeight
