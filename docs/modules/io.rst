Input and output
================

A run reads :class:`~ancestree.sites.Site` records and writes them back
annotated. Everything that crosses either boundary is collected here.

A site is a position together with the alleles observed at each named sample,
which is the whole of what the likelihood kernel asks of the data. A
:class:`~ancestree.sites.SiteSource` produces them, one implementation per store
format, each yielding sites in file order and reporting the panel it found:
:class:`~ancestree.sources.TskitSource`,
:class:`~ancestree.sources.CyVCF2Source` and
:class:`~ancestree.sources.VcfZarrSource`. Only the :mod:`tskit` path is needed
for a bare install, and the VCF and Zarr backends are imported on first use.
Every haplotype is a separate tip, so a diploid sample ``S`` appears as ``S_h0``
and ``S_h1``, and those are the names to give when naming ingroup and outgroup
samples.

A :class:`~ancestree.writers.Writer` takes the opposite direction.
:class:`~ancestree.writers.VCFWriter`, :class:`~ancestree.writers.TskitWriter`
and :class:`~ancestree.writers.ZarrWriter` each copy their input store and
record the MAP ancestral allele alongside its posterior, so the output holds the
same variants in the same order, together with a record of the run that produced
them. :class:`~ancestree.readers.Reader` inverts that, dispatching on the path's
suffix to recover both the per-site annotations, as
:class:`~ancestree.readers.Annotation` records, and the run's
:class:`~ancestree.readers.Provenance`.

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   ~ancestree.sites.Site
   ~ancestree.sites.SiteTable
   ~ancestree.sites.BaseComposition
   ~ancestree.sites.PolymorphicSiteFilter
   ~ancestree.sites.SiteSource
   ~ancestree.sources.TskitSource
   ~ancestree.sources.CyVCF2Source
   ~ancestree.sources.VcfZarrSource
   ~ancestree.writers.Writer
   ~ancestree.writers.VCFWriter
   ~ancestree.writers.TskitWriter
   ~ancestree.writers.ZarrWriter
   ~ancestree.readers.Reader
   ~ancestree.readers.Provenance
   ~ancestree.readers.Annotation
   ~ancestree.readers.Annotations

.. autoclass:: ancestree.sites.Site

.. autoclass:: ancestree.sites.SiteTable

.. autoclass:: ancestree.sites.BaseComposition

.. autoclass:: ancestree.sites.PolymorphicSiteFilter

.. autodata:: ancestree.STATES

.. autodata:: ancestree.STATE_INDEX

.. autodata:: ancestree.DEFAULT_MU

.. autodata:: ancestree.DEFAULT_REC_RATE

.. autodata:: ancestree.MISSING_ALLELES

.. autodata:: ancestree.TRANSITION_PAIRS

.. autoclass:: ancestree.sites.SiteSource

.. autoclass:: ancestree.sources.TskitSource

.. autoclass:: ancestree.sources.CyVCF2Source

.. autoclass:: ancestree.sources.VcfZarrSource

.. autoclass:: ancestree.writers.Writer

.. autoclass:: ancestree.writers.VCFWriter

.. autoclass:: ancestree.writers.TskitWriter

.. autoclass:: ancestree.writers.ZarrWriter

.. autoclass:: ancestree.readers.Reader

.. autoclass:: ancestree.readers.Provenance
   :no-inherited-members:

.. autoclass:: ancestree.readers.Annotation

.. autoclass:: ancestree.readers.Annotations
   :no-inherited-members:
