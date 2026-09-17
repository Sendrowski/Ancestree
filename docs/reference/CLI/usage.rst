.. _reference.cli.usage:

CLI Usage
=========

``ancestree`` provides a command-line interface that wraps the three
inference modes, for annotating a VCF or a ``.trees`` file without writing
any Python. It is registered as the ``ancestree`` console script when the
package is installed.

Getting help
------------

.. code-block:: bash

   ancestree --version
   ancestree --help
   ancestree fixed-tree --help
   ancestree arg --help
   ancestree local-tree --help

Three subcommands are exposed, one per inference mode.

The examples below run against a demo panel of 4 diploid ingroup individuals
and a 3-outgroup ladder over 200 kb, 10,592 sites, so their output is
reproducible.

ARG mode
------------

.. code-block:: bash

   ancestree arg --trees demo.trees --mu 1.25e-8 --out annotated.trees

.. raw:: html
   :file: _output/arg.html

Flags
^^^^^

.. argparse::
   :module: ancestree.cli
   :func: build_parser
   :prog: ancestree
   :path: arg
   :nodefaultconst:

Local-tree mode
-------------------

.. code-block:: bash

   ancestree local-tree --vcf demo.vcf.gz --mu 1.25e-8 --rec-rate 1e-8 --sequence-length 200000 --out annotated.vcf.gz

.. raw:: html
   :file: _output/local_tree.html

Infers dated local genealogies per genomic window from the genotypes alone
(a pairwise-coalescent HMM followed by per-window UPGMA) and infers the
ancestral allele at every site with the same likelihood kernel the ``arg``
subcommand uses. By default each window's posterior is marginalised over an
ensemble of genealogies drawn from the HMM posterior rather than scored on a
single reconstructed tree: ``--ensemble-size`` sets the number drawn (64 by
default), ``--ensemble-seed`` fixes the draw, and ``--no-ensemble`` scores the
single posterior-mean tree instead. No outgroups and no input ARG are
required, only the panel genotypes, the region length, and rough
mutation / recombination rates. As with ``arg``, an annotated ``.trees``
file is written instead when ``--out`` ends in ``.trees``, and ``--out-trees``
additionally dumps the inferred (pre-annotation) local-tree sequence.

.. note::

   The pairwise-coalescent HMM is :math:`O(n^2)` in the number of
   haplotypes :math:`n`, and its TMRCA matrix grows with the region length.
   The subcommand therefore processes the genome in chunks internally:
   ``--chunk-size`` sets the chunk length (10 Mb by default, ``none``
   builds the whole input as one region), ``--halo`` sets the overlap
   discarded at chunk boundaries, and ``--n-workers`` fans whole chunks
   across a process pool. A single invocation covers a whole chromosome.

Flags
^^^^^

.. argparse::
   :module: ancestree.cli
   :func: build_parser
   :prog: ancestree
   :path: local-tree
   :nodefaultconst:

Fixed-tree mode
-------------------

.. code-block:: bash

   ancestree fixed-tree \
       --vcf demo.vcf.gz \
       --outgroups o1_h0,o2_h0,o3_h0 \
       --model K2 --fit-kappa \
       --ingroup-weight kingman \
       --n-target-sites 200000 \
       --out annotated.vcf.gz

.. raw:: html
   :file: _output/fixed_tree.html

ML-fits the branch rates of an
:class:`~ancestree.trees.OutgroupLadderTree` against the polymorphic
sites in ``--vcf``. The ladder is built from ``--outgroups``, read
closest outgroup first. ``--ingroup`` defaults to every haplotype whose
individual is not named in ``--outgroups``, so naming one haplotype of an
outgroup individual withholds that individual entirely. Pass it explicitly
to use a subset of the panel.

Passing ``--species-tree`` instead supplies the tree outright, topology
and branch lengths both, and no fit is performed. The Newick must be
dated. Extra taxa in it are pruned, and the ladder order follows its
topology rather than the order ``--outgroups`` was given in.

The fitted divergences increase along the ladder, as the assumed topology
requires. A rate pinned at its bound instead signals a degenerate fit.

Flags
^^^^^

.. argparse::
   :module: ancestree.cli
   :func: build_parser
   :prog: ancestree
   :path: fixed-tree
   :nodefaultconst:

Output and logging
------------------

The output format follows the ``--out`` extension: ``.vcf``, ``.vcf.gz``,
``.vcf.bgz`` or ``.bcf`` writes an annotated VCF, ``.vcz`` a VCF Zarr store and ``.trees`` a
tree sequence, the formats described in :doc:`../Python/io`. ``fixed-tree``
does not write ``.trees``. An output annotates the input when the input is a
VCF or a local VCF Zarr store of its format. Otherwise each site is written as
one record, holding the panel. VCF input and output need the ``[vcf]`` extra, VCF
Zarr the ``[zarr]`` extra and ``--species-tree`` the ``[newick]`` extra (see
:doc:`../Python/installation`).

``-v`` raises the logger to ``DEBUG`` and ``-q`` lowers it to ``WARNING``,
the two being mutually exclusive, and ``--no-progress`` hides the progress
bars. Those three are accepted on either side of the subcommand. ``--debug``
re-raises an unexpected error with its traceback, which is what to pass when
reporting a bug. It sits on the top-level parser and must therefore precede
the subcommand:

.. code-block:: bash

   ancestree --debug arg --trees demo.trees --mu 1.25e-8 --out annotated.trees
