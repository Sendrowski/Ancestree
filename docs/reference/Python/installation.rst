.. _reference.python.installation:

Installation
============

PyPI
^^^^

To install the ``ancestree`` package via pip:

.. code-block:: bash

   pip install ancestree

``ancestree`` is compatible with Python 3.10 through 3.13.

Optional backends are lazy-imported on first use, so only the backends
actually used need be installed. Select the matching extra:

.. code-block:: bash

   pip install "ancestree[vcf]"       # + cyvcf2 (read/write VCF / BCF)
   pip install "ancestree[zarr]"      # + zarr + bio2zarr (read/write VCF Zarr / VCZ stores)
   pip install "ancestree[newick]"    # + newick (parse species-tree Newicks)
   pip install "ancestree[maps]"      # + msprime (--recombination-map / --mutation-map)
   pip install "ancestree[plotting]"  # + matplotlib (ancestree.plotting)
   pip install "ancestree[all]"       # all of the above

Conda
^^^^^

``ancestree`` can also be installed via the ``conda-forge`` channel:

.. code-block:: bash

   mamba create -n ancestree -c conda-forge ancestree
   mamba activate ancestree

To enable VCF I/O via ``cyvcf2``, which is hosted on bioconda, add
the channel and request the package explicitly:

.. code-block:: bash

   mamba create -n ancestree -c conda-forge -c bioconda ancestree cyvcf2

Or use an environment file for reproducibility:

.. code-block:: yaml

   name: ancestree
   channels:
     - conda-forge
     - bioconda
   dependencies:
     - ancestree
     - cyvcf2          # [vcf]: CyVCF2Source and VCFWriter
     - zarr            # [zarr]: VcfZarrSource and ZarrWriter
     - newick          # [newick]: OutgroupLadderTree.from_newick
     - msprime         # [maps]: recombination and mutation maps
     - matplotlib      # [plotting]: ancestree.plotting
     - pip
     - pip:
         - bio2zarr    # [zarr]: VCZ template for ZarrWriter, PyPI only

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate ancestree
