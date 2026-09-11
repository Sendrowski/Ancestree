"""Shared test helpers.

Utilities that do not fit the pytest-fixture model, such as functions called
at class-statement scope rather than from inside a test method.
"""
from __future__ import annotations

from ancestree import BaseComposition
from ancestree.posterior import Posterior


def canonical_alleles(alleles) -> tuple[str, ...]:
    """The site's alleles in an order that does not encode the truth.

    tskit lists a variant's ancestral state first, so a fixture that passes
    that order through lets an estimator score perfectly by always calling
    ``alleles[0]``, without reading a genotype. Sorting keys the order to
    which alleles are present rather than to which one is ancestral.

    :param alleles: The variant's alleles, ancestral state first.
    :return: The called alleles, sorted, with placeholders dropped.
    """
    return tuple(sorted({a for a in alleles if a}))


def no_counts() -> BaseComposition:
    """Zero-count placeholder :class:`~ancestree.sites.BaseComposition` for paths that do not depend on the counts.

    :return: A zero-count :class:`~ancestree.sites.BaseComposition`.
    """
    return BaseComposition.no_counts()


def post(map_allele: str, p: float) -> Posterior:
    """A 4-state posterior peaked at ``map_allele`` with mass ``p``."""
    import numpy as np

    alleles = ("A", "C", "G", "T")
    rem = (1.0 - p) / 3.0
    values = np.array([p if a == map_allele else rem for a in alleles])
    return Posterior(alleles=alleles, values=values)


def write_vcz(
    path,
    *,
    positions,
    contigs_per_variant,
    contig_ids,
    alleles,
    sample_ids,
    genotypes,
    phased=None,
) -> None:
    """Write a minimal VCZ-format zarr store at ``path`` (bio2zarr layout).

    Mirrors what :class:`~ancestree.sources.VcfZarrSource` reads: the
    top-level arrays ``variant_position``, ``variant_contig``,
    ``variant_allele``, ``call_genotype``, ``sample_id``, ``contig_id``.
    Imports ``zarr`` / ``numpy`` lazily so importing this module never
    requires the optional zarr stack.

    :param path: Destination store path (``.vcz``).
    :param positions: Per-variant 1-based positions.
    :param contigs_per_variant: Per-variant contig index into ``contig_ids``.
    :param contig_ids: Contig labels.
    :param alleles: Per-variant allele lists (ref first).
    :param sample_ids: VCF sample ids (pre-haplotype-split).
    :param genotypes: ``(n_variants, n_samples, ploidy)`` int allele indices
        (``-1`` for missing).
    :param phased: ``True`` or ``False`` writes a ``call_genotype_phased``
        array carrying that value for every call, as ``bio2zarr`` does.
        ``None`` leaves the array out, giving a store that declares no phase.
    """
    import numpy as np
    import zarr

    def _create(root, name, data):
        if hasattr(root, "create_array"):  # zarr v3
            arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
            arr[:] = data
        else:  # zarr v2
            root.create_dataset(name, data=data, overwrite=True)

    n_variants = len(positions)
    max_alleles = max(len(a) for a in alleles)
    max_len = max((len(a) for row in alleles for a in row), default=1)
    allele_arr = np.full((n_variants, max_alleles), "", dtype=f"U{max_len}")
    for i, row in enumerate(alleles):
        for j, a in enumerate(row):
            allele_arr[i, j] = a

    root = zarr.open(str(path), mode="w")
    _create(root, "variant_position", np.asarray(positions, dtype=np.int64))
    _create(root, "variant_contig", np.asarray(contigs_per_variant, dtype=np.int32))
    _create(root, "variant_allele", allele_arr)
    gt = np.asarray(genotypes, dtype=np.int8)
    _create(root, "call_genotype", gt)
    if phased is not None:
        _create(root, "call_genotype_phased",
                np.full(gt.shape[:2], bool(phased), dtype=bool))
    _create(root, "sample_id", np.asarray(sample_ids, dtype="U"))
    _create(root, "contig_id", np.asarray(contig_ids, dtype="U"))


def ts_to_vcz(ts, path, *, contig: str = "1", sample_prefix: str = "n") -> list[str]:
    """Dump a (haploid) ``tskit.TreeSequence`` to a VCZ store via :func:`write_vcz`.

    Extracts per-site positions, alleles, and the genotype matrix directly,
    so no ``bio2zarr`` / VCF round-trip is needed. Samples are named
    ``{sample_prefix}{i}``.

    :param ts: A haploid tree sequence (one node per sample).
    :param path: Destination ``.vcz`` store path.
    :param contig: Single contig label written to the store.
    :param sample_prefix: Prefix for the generated sample ids.
    :return: The list of sample ids written.
    """
    positions, alleles, gts = [], [], []
    for v in ts.variants():
        positions.append(int(v.site.position))
        alleles.append(list(v.alleles))
        gts.append(list(v.genotypes))
    n_samples = int(ts.num_samples)
    sample_ids = [f"{sample_prefix}{i}" for i in range(n_samples)]
    genotypes = [[[g] for g in row] for row in gts]  # ploidy-1 axis
    write_vcz(
        path,
        positions=positions,
        contigs_per_variant=[0] * len(positions),
        contig_ids=[contig],
        alleles=alleles,
        sample_ids=sample_ids,
        genotypes=genotypes,
    )
    return sample_ids
