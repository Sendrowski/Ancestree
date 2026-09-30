Changelog
=========

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
