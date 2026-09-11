.. _introduction:

Introduction
============
``ancestree`` is a Python package for ancestral-allele annotation:
given polymorphic sites and a genealogical context, it infers the
ancestral state at every site, emitting a full posterior over the four
nucleotide states rather than a single allele. One likelihood kernel
serves three modes that differ only in the tree it is evaluated on.
:class:`~ancestree.inference.FixedTreeInference` assumes a fixed tree
of ingroup and outgroups and fits its branch rates by maximum
likelihood [1]_.
:class:`~ancestree.inference.ARGBasedInference` uses the local tree of
each site in a supplied ARG [2]_.
:class:`~ancestree.local_tree_inference.LocalTreeInference` requires
only genotypes and infers the per-site genealogy of the sample from them
with a pairwise-coalescent HMM. Every mode is available from Python and from
the command line (:doc:`reference/CLI/usage`). ``ancestree`` is validated
against both reference methods on simulated and empirical data.

Motivation
----------
Knowing which allele is ancestral underpins many population-genetic
analyses: unfolded-SFS-based selection tests, distribution-of-fitness-effects
inference, allele-age estimation and variant-deleteriousness prediction all
require a per-site ancestral-allele assignment, and ARG-inference pipelines
such as |tsinfer| and |Relate| take one as input. Ancestral states are not
observed, however, and must be inferred, typically from outgroups that
diverged before the ingroup and are assumed to carry the ancestral
allele. That assumption fails at a fraction of sites, through incomplete
lineage sorting, recurrent mutation or a mutation on the outgroup's own
branch, and existing tools suit one setting only, either outgroup data on
a fixed topology or a fully resolved ARG. ``ancestree`` covers both within
one framework and degrades gracefully at the sites where the
outgroup-as-ancestral rule fails.

Features
--------

Three inference modes:

- :class:`~ancestree.inference.ARGBasedInference`: per-site ARG local trees.
- :class:`~ancestree.inference.FixedTreeInference`: outgroups on a fixed
  topology with ML-fitted branch rates.
- :class:`~ancestree.local_tree_inference.LocalTreeInference`: a VCF alone,
  with no input ARG.

``ancestree`` supports VCF (|cyvcf2|),
`VCF-Zarr <https://github.com/sgkit-dev/vcf-zarr-spec>`_ and ARG
(:mod:`tskit`) files, both for reading and for writing.

.. toctree::
   :caption: Python Reference
   :hidden:

   reference/Python/installation
   reference/Python/quickstart
   reference/Python/arg_based_inference
   reference/Python/local_tree_inference
   reference/Python/fixed_tree_inference
   reference/Python/focal_node
   reference/Python/substitution_models
   reference/Python/io

.. toctree::
   :caption: CLI Reference
   :hidden:

   reference/CLI/usage

.. toctree::
   :caption: API reference
   :maxdepth: 1
   :hidden:

   modules/inference
   modules/priors
   modules/likelihood
   modules/trees
   modules/models
   modules/io
   modules/settings

.. toctree::
   :caption: Project
   :maxdepth: 1
   :hidden:

   changelog

References
----------

.. [1] Keightley P.D. & Jackson B.C. (2018).
   Inferring the probability of the derived vs. the ancestral allelic
   state at a polymorphic site. Genetics 209: 897–906.

.. [2] Liang B. et al. (2026). Polarising SNPs without
   outgroup. Molecular Ecology Resources.
