"""Reading sites from tskit, VCF and VCF Zarr.

Each source yields :class:`~ancestree.sites.Site` records in file order and
reports the sample names it found. Only the tskit path is needed for a bare
install. The VCF and Zarr backends are imported on first use, so their
dependencies are optional.

Sample names and ploidy: every haplotype is a separate tip, so a diploid
sample ``S`` in a VCF or VCZ store appears as two samples, ``S_h0`` and
``S_h1``. Either name may be given when specifying ingroup and outgroup
samples, the individual standing for all of its haplotypes. Haploid input
keeps the original name.
"""
from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ancestree.sites import Site, SiteSource
from ancestree.writers import (
    ZARR_ALLELE_FIELD,
    ZARR_CONTIG_FIELD,
    ZARR_POSITION_FIELD,
)
from ancestree._cyvcf2 import import_cyvcf2

if TYPE_CHECKING:
    import tskit

def _site_from_tskit_variant(
    variant: "tskit.Variant",
    sample_nodes: np.ndarray,
    node_to_sample: Mapping[int, str],
    chrom: str,
) -> Site:
    """Convert one tskit variant into a :class:`~ancestree.sites.Site`.

    Alleles are upper-cased, and those outside A/C/G/T pass through. A
    missing genotype (``-1``) and the empty-string placeholder allele both map
    to ``tip_alleles[s] = None``.

    :param variant: A variant from ``ts.variants()``.
    :param sample_nodes: The ``ts.samples()`` array, aligned with
        ``variant.genotypes``.
    :param node_to_sample: Mapping from tskit sample node id to caller-facing
        sample id. Nodes absent from it are left out of ``tip_alleles``.
    :param chrom: Contig name to embed in the emitted site.
    :return: The site, whose ``local_tree_handle`` is the variant's genomic
        position.
    """
    genotypes = variant.genotypes
    alleles = tuple((a.upper() if a else a) for a in variant.alleles)
    tip_alleles: dict[str, str | None] = {}
    for i, node in enumerate(sample_nodes):
        sample_name = node_to_sample.get(int(node))
        if sample_name is None:
            continue
        g = int(genotypes[i])
        a = alleles[g] if g >= 0 else None
        tip_alleles[sample_name] = a if a else None
    return Site(
        chrom=chrom,
        pos=Site.tskit_pos(variant.site.position,
                           variant.tree_sequence.sequence_length),
        alleles=alleles,
        tip_alleles=tip_alleles,
        local_tree_handle=float(variant.site.position),
    )


class TskitSource(SiteSource):
    """Iterate variants from a :class:`tskit.TreeSequence` as :class:`~ancestree.sites.Site` records.

    tskit's :attr:`Site.position <tskit.Site.position>` is a continuous
    float, and the emitted :attr:`Site.pos <ancestree.sites.Site.pos>`
    rounds it to the nearest integer on the tree sequence's axis
    (:meth:`Site.tskit_pos() <ancestree.sites.Site.tskit_pos>`).

    :param ts: The source tree sequence.
    :param chrom: Contig name to embed in every emitted :class:`~ancestree.sites.Site`.
        tskit has no contig concept, so this is purely for downstream
        identification (e.g. when writing back a VCF). Defaults to ``"1"``.
    :param sample_map: Mapping from caller-facing sample id (string) to
        tskit sample node id (integer in ``range(ts.num_samples)``). ``None``
        (default) derives one from the individual names
        (:meth:`TskitLocalTree.sample_map_from_individuals() <ancestree.trees.TskitLocalTree.sample_map_from_individuals>`),
        falling back to ``str(i)`` per sample when they carry none. Sample ids here must match those used in any
        :class:`~ancestree.trees.TskitLocalTree` consuming the sites. The cleanest pattern
        is to share one ``sample_map`` between the source and the trees.
    """

    format = "trees"

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"n_samples": len(self._samples_list), "chrom": self._chrom}

    def __init__(
        self,
        ts: "tskit.TreeSequence",
        chrom: str = "1",
        sample_map: Mapping[str, int] | None = None,
    ) -> None:
        """Wrap a :class:`tskit.TreeSequence` with caller-facing sample naming."""
        self._ts = ts
        self._chrom = chrom

        if sample_map is None:
            from ancestree.trees import TskitLocalTree
            derived = TskitLocalTree.sample_map_from_individuals(ts)
            if derived is not None:
                self._sample_map = derived
                self._samples_list = list(derived.keys())
            else:
                self._samples_list = [str(int(s)) for s in ts.samples()]
                self._sample_map = {name: int(s) for name, s in zip(self._samples_list, ts.samples())}
        else:
            self._sample_map = dict(sample_map)
            self._samples_list = list(self._sample_map.keys())

        # Reverse map: tskit sample node id → caller-facing sample id.
        self._node_to_sample: dict[int, str] = {v: k for k, v in self._sample_map.items()}
        self._log.info(
            "Reading variants from a tree sequence (%d sites, %d samples)",
            ts.num_sites, len(self._samples_list),
        )

    def samples(self) -> list[str]:
        """Caller-facing sample ids in the canonical order from the tskit ts."""
        return list(self._samples_list)

    def contig_lengths(self) -> dict[str, int]:
        """The tree sequence's length, as the length of its contig.

        :return: ``{chrom: length}``.
        """
        return {self._chrom: math.ceil(self._ts.sequence_length)}

    def sample_map(self) -> dict[str, int]:
        """The sample-id → tskit-sample-node map.

        :return: A fresh copy of the mapping, safe to mutate.
        """
        return dict(self._sample_map)

    def __iter__(self) -> Iterator[Site]:
        """Yield :class:`~ancestree.sites.Site` records, one per ``ts.variants()`` entry."""
        sample_nodes = self._ts.samples()
        for variant in self._ts.variants():
            yield _site_from_tskit_variant(
                variant, sample_nodes, self._node_to_sample, self._chrom,
            )


#: Records probed for a called genotype before ploidy falls back to diploid.
_PLOIDY_PROBE_RECORDS = 1000


class CyVCF2Source(SiteSource):
    """Stream :class:`~ancestree.sites.Site` records from a VCF / BCF via ``cyvcf2``.

    Per-haplotype tip naming: a diploid sample ``S`` is split into two
    haplotype-samples ``S_h0`` and ``S_h1`` (Ancestree's Felsenstein
    likelihood is per-haplotype). For haploid VCFs the original sample
    name is kept unchanged. Ploidy is auto-detected from the first called
    variant on ``chrom_filter``. Pass ``ploidy=`` explicitly to override.

    Non-SNP variants (indels, complex alleles, alleles outside the
    A/C/G/T alphabet) are silently skipped.

    The source is re-iterable: each ``iter(source)`` reopens the VCF.

    :param vcf_path: Path to a VCF / VCF.GZ / BCF readable by cyvcf2.
    :param sample_filter: Optional sequence of VCF sample names to keep
        (kept in the given order).
    :param phased: Whether the genotypes are phased, so haplotype ``k`` of an
        individual is the same lineage at every site. ``None`` (default) reads
        the per-record phase flag. Forcing ``True`` on unphased data builds a
        clade out of the arbitrary allele order. Forcing ``False`` randomises
        that order per site, which costs the real phase information where the
        data carry it.
    :param phase_seed: Seed for the per-site haplotype-order randomisation
        applied to unphased heterozygotes, so a run is reproducible.
    :param chrom_filter: Optional contig to restrict iteration to.
    :param ploidy: Override auto-detection. Use ``1`` for haploid VCFs
        that contain heterogeneous ploidy hints.
    :raises ImportError: If ``cyvcf2`` is not installed.
    :raises KeyError: If ``sample_filter`` names samples absent from the VCF.
    """

    position_base = 1
    format = "vcf"

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"path": str(self._path), "n_samples": len(self._samples_list)}

    def __init__(
        self,
        vcf_path: str | os.PathLike,
        *,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        ploidy: int | None = None,
        phased: bool | None = None,
        phase_seed: int = 0,
    ) -> None:
        """Probe the VCF for sample names and ploidy. Defer iteration."""
        self._path = str(vcf_path)
        self._sample_filter = list(sample_filter) if sample_filter is not None else None
        self._chrom_filter = chrom_filter

        vcf = self._open_vcf()
        try:
            file_order: list[str] = list(vcf.samples)
            if self._sample_filter is not None:
                self._check_sample_filter(
                    self._sample_filter, file_order, self._path,
                )
            self._vcf_samples: list[str] = (
                file_order if self._sample_filter is None
                else list(self._sample_filter))
            row_of = {name: i for i, name in enumerate(file_order)}
            self._genotype_rows = [row_of[s] for s in self._vcf_samples]
            # Ploidy is len(genotype) - 1. A bare "." reaches htslib as one
            # value, so records are probed until one carries a called
            # genotype. A sites-only VCF defaults to diploid. Records outside
            # chrom_filter are passed over without counting toward the probe.
            detected = None
            records = (r for r in vcf if chrom_filter is None
                       or r.CHROM == chrom_filter)
            for _ in range(_PLOIDY_PROBE_RECORDS):
                record = next(records, None)
                if record is None:
                    break
                gts = record.genotypes
                if not gts:
                    break
                if any(a >= 0 for g in gts for a in g[:-1]):
                    detected = max((len(g) - 1 for g in gts), default=1)
                    break
            if ploidy is None:
                ploidy = max(1, detected if detected is not None else 2)
            elif detected is not None and ploidy > detected:
                self._log.warning(
                    "ploidy=%d exceeds the %d haplotype(s) of the first called "
                    "record of %s. A call narrower than the ploidy leaves its "
                    "extra haplotypes missing.", ploidy, detected, self._path)
        finally:
            vcf.close()

        self._phased = phased
        self._phase_seed = int(phase_seed)
        self._ploidy = int(ploidy)
        if self._ploidy != 2:
            self._log.info(
                "Genotypes read as ploidy %d; pass ploidy= to override",
                self._ploidy)
        self._samples_list = self._haplotype_sample_names(self._vcf_samples, self._ploidy)
        self._log.info(
            "Reading variants from %s (%d haplotype samples)",
            self._path, len(self._samples_list),
        )

    @property
    def ploidy(self) -> int:
        """Ploidy that expands each per-sample genotype into per-haplotype tips."""
        return self._ploidy

    def samples(self) -> list[str]:
        """Per-haplotype tip ids in ``sample_filter`` order, else file order."""
        return list(self._samples_list)

    def contig_lengths(self) -> dict[str, int]:
        """The length of each contig the VCF header declares with one.

        :return: ``{contig: length}``.
        """
        import re

        cyvcf2 = import_cyvcf2()

        vcf = cyvcf2.VCF(self._path)
        try:
            header = vcf.raw_header
        finally:
            vcf.close()
        lengths = {}
        for fields in re.findall(r"^##contig=<(.*)>$", header, re.MULTILINE):
            entry = dict(f.split("=", 1) for f in fields.split(",") if "=" in f)
            if "ID" in entry and entry.get("length", "").isdigit():
                lengths[entry["ID"]] = int(entry["length"])
        return lengths

    def _open_vcf(self):
        """Open the VCF under the configured sample filter.

        :return: An open :class:`cyvcf2.VCF` handle.
        :raises ImportError: If cyvcf2 is not installed.
        """
        cyvcf2 = import_cyvcf2()

        with warnings.catch_warnings():
            # Absent samples raise in _check_sample_filter.
            warnings.filterwarnings(
                "ignore", message="not all requested samples found in VCF",
                category=UserWarning,
            )
            return cyvcf2.VCF(self._path, samples=self._sample_filter)

    def __iter__(self) -> Iterator[Site]:
        """Open the VCF and stream one :class:`~ancestree.sites.Site` per SNP record."""
        vcf = self._open_vcf()
        records = ((str(v.CHROM), int(v.POS), [v.REF, *(v.ALT or [])], v)
                   for v in vcf if self._chrom_filter is None
                   or v.CHROM == self._chrom_filter)
        try:
            yield from self._sites_from_records(
                records, self._vcf_samples, self._calls)
        finally:
            vcf.close()

    def _calls(self, variant) -> "tuple[list[list[int]], list[bool]]":
        """One record's allele indices and phase flags, in panel order.

        :param variant: The ``cyvcf2.Variant``.
        :return: ``(genotypes, phased)`` as
            :meth:`SiteSource._sites_from_records` takes them.
        """
        calls = variant.genotypes  # indices + a phase flag
        rows = [calls[r] for r in self._genotype_rows]
        return [row[:-1] for row in rows], [row[-1] for row in rows]


class VcfZarrSource(SiteSource):
    """Stream :class:`~ancestree.sites.Site` records from a VCF Zarr (VCZ) store via ``zarr-python``.

    Reads the VCZ layout written by ``bio2zarr`` (``vcf2zarr``): expects
    arrays ``variant_position``, ``variant_allele``, ``variant_contig``,
    ``contig_id``, ``call_genotype``, and ``sample_id`` at the store root.
    Genotypes are read in chunks to bound memory on large stores.

    Per-haplotype tip naming follows :class:`~ancestree.sources.CyVCF2Source`: a diploid
    sample ``S`` becomes ``S_h0`` / ``S_h1``. Non-SNP variants
    (multi-character alleles, alleles outside A/C/G/T) are skipped.

    :param zarr_path: Path (or any zarr-compatible URI) to the VCZ root.
    :param sample_filter: Optional sequence of sample names to keep
        (kept in the given order). Defaults to all samples in ``sample_id``.
    :param phased: Whether the genotypes are phased, so haplotype ``k`` of an
        individual is the same lineage at every site. ``None`` (default) reads
        the per-call flag from ``call_genotype_phased``, treating a store
        written without that array as unphased. Forcing ``True`` on unphased
        data builds a clade out of the arbitrary allele order. Forcing
        ``False`` randomises that order per site, which costs the real phase
        information where the data carry it.
    :param phase_seed: Seed for the per-site haplotype-order randomisation
        applied to unphased heterozygotes, so a run is reproducible.
    :param chrom_filter: Optional contig name to restrict iteration to.
    :param chunk_size: Number of variants to materialise per batch when
        streaming. Larger values cut zarr-read overhead but use more
        memory. Defaults to ``1000``.
    :raises ImportError: If ``zarr`` is not installed.
    :raises KeyError: If ``sample_filter`` names samples absent from the store.
    :raises ValueError: If ``call_genotype`` is not a 3-D
        ``(n_variants, n_samples, ploidy)`` array.
    """

    position_base = 1
    format = "vcz"

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"path": str(self._path), "n_samples": len(self._samples_list)}

    def __init__(
        self,
        zarr_path: str | os.PathLike,
        *,
        sample_filter: Sequence[str] | None = None,
        chrom_filter: str | None = None,
        chunk_size: int = 1000,
        phased: bool | None = None,
        phase_seed: int = 0,
    ) -> None:
        """Open the store, resolve sample indices and ploidy. Defer iteration."""
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "VcfZarrSource requires zarr. Install with "
                "`pip install zarr` or `conda install -c conda-forge zarr`."
            ) from e

        self._path = str(zarr_path)
        self._chrom_filter = chrom_filter
        self._chunk_size = int(chunk_size)
        self._phased = phased
        self._phase_seed = int(phase_seed)

        root = self._open_root()
        all_samples = [self._decode(s) for s in root["sample_id"][:]]
        if sample_filter is None:
            self._sample_indices = np.arange(len(all_samples), dtype=int)
            kept_names = list(all_samples)
        else:
            wanted = list(sample_filter)
            name_to_idx = {s: i for i, s in enumerate(all_samples)}
            self._check_sample_filter(wanted, all_samples, self._path)
            self._sample_indices = np.array(
                [name_to_idx[s] for s in wanted], dtype=int,
            )
            kept_names = wanted
        self._kept_sample_names = kept_names

        gt = root["call_genotype"]
        if gt.ndim != 3:
            raise ValueError(
                f"Expected call_genotype with shape (n_variants, n_samples, "
                f"ploidy); got ndim={gt.ndim} shape={gt.shape}"
            )
        self._ploidy = int(gt.shape[2])
        self._n_variants = int(gt.shape[0])
        self._contig_ids = [self._decode(c) for c in root["contig_id"][:]]
        self._samples_list = self._haplotype_sample_names(kept_names, self._ploidy)
        self._log.info(
            "Reading variants from %s (%d variants, %d haplotype samples)",
            self._path, self._n_variants, len(self._samples_list),
        )

    @property
    def ploidy(self) -> int:
        """Ploidy as recorded in the ``call_genotype`` array."""
        return self._ploidy

    def samples(self) -> list[str]:
        """Per-haplotype tip ids in ``sample_filter`` order, else store order."""
        return list(self._samples_list)

    def __iter__(self) -> Iterator[Site]:
        """Stream :class:`~ancestree.sites.Site` records, chunking the underlying zarr reads."""
        yield from self._sites_from_records(
            self._records(), self._kept_sample_names, self._calls)

    def _records(self):
        """The store's records on ``chrom_filter``, read a chunk at a time.

        :return: Generator of ``(chrom, pos, alleles, payload)`` as
            :meth:`SiteSource._sites_from_records` takes them.
        """
        root = self._open_root()
        pos_arr = root[ZARR_POSITION_FIELD]
        alleles_arr = root[ZARR_ALLELE_FIELD]
        contig_arr = root[ZARR_CONTIG_FIELD]
        gt_arr = root["call_genotype"]
        # Consulted only where the caller has not fixed the phase.
        phased_arr = (root.get("call_genotype_phased", None)
                      if self._phased is None else None)
        sample_idx = self._sample_indices
        for start in range(0, self._n_variants, self._chunk_size):
            end = min(start + self._chunk_size, self._n_variants)
            pos_batch = pos_arr[start:end]
            alleles_batch = alleles_arr[start:end]
            contig_batch = contig_arr[start:end]
            # Materialise as an ndarray before fancy-indexing the sample axis.
            gt_batch = np.asarray(gt_arr[start:end])[:, sample_idx, :]
            phased_batch = (
                np.asarray(phased_arr[start:end])[:, sample_idx]
                if phased_arr is not None else None)
            for i in range(end - start):
                chrom = self._contig_ids[int(contig_batch[i])]
                if self._chrom_filter is None or chrom == self._chrom_filter:
                    yield (chrom, int(pos_batch[i]),
                           [self._decode(x) for x in alleles_batch[i]],
                           (gt_batch[i], None if phased_batch is None
                            else phased_batch[i]))

    @staticmethod
    def _calls(payload) -> "tuple[list[list[int]], list[bool] | None]":
        """One record's allele indices and phase flags, in panel order.

        :param payload: The record's ``call_genotype`` and
            ``call_genotype_phased`` rows, the second ``None`` where unread.
        :return: ``(genotypes, phased)`` as
            :meth:`SiteSource._sites_from_records` takes them.
        """
        genotypes, phased = payload
        return genotypes.tolist(), None if phased is None else phased.tolist()

    def contig_lengths(self) -> dict[str, int]:
        """The length of each contig the store records one for, a
        non-positive ``contig_length`` meaning none.

        :return: ``{contig: length}``, empty without ``contig_length``.
        """
        root = self._open_root()
        if "contig_length" not in root:
            return {}
        return {SiteSource._decode(c): int(n) for c, n in
                zip(root["contig_id"][:], root["contig_length"][:]) if n > 0}

    def _open_root(self):
        """Open the VCZ root group."""
        import zarr
        return zarr.open(self._path, mode="r")
