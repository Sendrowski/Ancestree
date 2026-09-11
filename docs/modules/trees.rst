Trees
=====

The likelihood kernel is evaluated on a tree, and reports at one node of it.
Both are named here.

The :class:`~ancestree.trees.Tree` abstraction is the whole of what the kernel
sees: children, branch lengths, a post-order traversal, and a mapping from
sample names to tip nodes. :class:`~ancestree.trees.TskitLocalTree` wraps a
local tree taken from an ARG or inferred from genotypes, and
:class:`~ancestree.trees.OutgroupLadderTree` is the fixed ladder the
`est-sfs <https://sourceforge.net/projects/est-usfs/>`_ model assumes. Both present that same surface, which is why the inference modes
differ so little from one another.

A topology does not by itself say where the posterior is read.
:class:`~ancestree.focal.FocalNode` states that separately, and because a run
spans many local trees while node ids are per tree, it is a rule resolved
against each tree in turn into a :class:`~ancestree.focal.ResolvedFocal`. It
reaches positions part-way between the two named anchors as well as the anchors
themselves.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~ancestree.trees.Tree
   ~ancestree.trees.TskitLocalTree
   ~ancestree.trees.OutgroupLadderTree
   ~ancestree.trees.RerootedTree
   ~ancestree.focal.FocalNode
   ~ancestree.focal.ResolvedFocal
   ~ancestree.plotting.FocalTreePlot
   ~ancestree.plotting.FocalSweep

.. autoclass:: ancestree.trees.Tree

.. autoclass:: ancestree.trees.TskitLocalTree

.. autoclass:: ancestree.trees.OutgroupLadderTree

.. autoclass:: ancestree.trees.RerootedTree

.. autoclass:: ancestree.focal.FocalNode

.. autoclass:: ancestree.focal.ResolvedFocal

.. autoclass:: ancestree.plotting.FocalTreePlot

.. autoclass:: ancestree.plotting.FocalSweep
