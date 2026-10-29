Changelog
=========

[0.3.0] - 2026-10-29
^^^^^^^^^^^^^^^^^^^^
- Added support for species trees through :class:`~ancestree.trees.FixedTree`, with ``--tree`` replacing ``--species-tree``.
- ARG and local-tree mode score each site at the mutation rate of its position, and local-tree mode handles missing calls per pair.
- Each contig takes its own sequence length, and BED and bedGraph maps are read onto 1-based VCF positions.
- Records mixing a SNP with an indel and haploid samples padded to diploid are accepted.
- ``Site.is_called`` and the transition-matrix cache of :class:`~ancestree.models.SubstitutionModel` are removed.
- Support Python 3.14.
- Parallel runs fail promptly when a worker process dies.
- Assorted fixes to I/O, documentation and error messages.

[0.2.0] - 2026-09-30
^^^^^^^^^^^^^^^^^^^^
- The root prior follows the base composition in every mode, and ``--base-composition`` is available on every subcommand.
- Sample handling is shared across modes: with both ingroup and outgroups named the panel is their union, an individual's name stands for all of its haplotypes, and ``restrict_samples`` (``--restrict-samples``) limits an annotated output to the samples used.
- An output in a format other than the input's is written from the sites, holding the panel genotypes. Writers refuse to overwrite their template or any path that is not a Zarr store.
- Local-tree mode reads a tree sequence as genotypes, takes the same reader arguments as the other modes and reads rate maps per contig.
- :meth:`LocalTreeInference.to_tree_sequence() <ancestree.local_tree_inference.LocalTreeInference.tree_sequences>` is renamed :meth:`LocalTreeInference.tree_sequences() <ancestree.local_tree_inference.LocalTreeInference.tree_sequences>`, and the ``contig_id`` argument of :meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>` and ``mu_matches_time_units`` of :class:`~ancestree.local_tree_inference.LocalTreeInference` are removed.
- Assorted fixes to I/O, documentation and error messages.

[0.1.1] - 2026-09-11
^^^^^^^^^^^^^^^^^^^^
- Initial release.
