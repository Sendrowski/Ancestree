"""Reading sites from tskit, VCF and VCF Zarr.

Each source yields :class:`~ancestree.sites.Site` records in file order and
reports the sample names it found. Only the tskit path is needed for a bare
install. The VCF and Zarr backends are imported on first use, so their
dependencies are optional.

Sample names and ploidy: every haplotype is a separate tip, so a diploid
sample ``S`` in a VCF or VCZ store appears as two samples, ``S_h0`` and
``S_h1``. Use those names when specifying ingroup and outgroup samples.
Haploid input keeps the original name.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

from ancestree import STATES
from ancestree.sites import Site, SiteSource
from ancestree.writers import (
    ZARR_ALLELE_FIELD,
    ZARR_CONTIG_FIELD,
    ZARR_POSITION_FIELD,
)

if TYPE_CHECKING:
    import tskit


def _site_from_tskit_variant(
    variant: "tskit.Variant",
    sample_nodes: np.ndarray,
    node_to_sample: Mapping[int, str],
    chrom: str,
) -> Site:
    """Convert one tskit variant into a :class:`~ancestree.sites.Site`.

    Alleles are upper-cased. Alleles outside A/C/G/T pass through and the
    kernel marginalises them at the tip. A missing genotype (``-1``) and the
    empty-string placeholder allele both map to ``tip_alleles[s] = None``.

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
        pos=int(variant.site.position),
        alleles=alleles,
        tip_alleles=tip_alleles,
        local_tree_handle=float(variant.site.position),
    )


class TskitSource(SiteSource):
    """Iterate variants from a :class:`tskit.TreeSequence` as :class:`~ancestree.sites.Site` records.

    tskit's :attr:`Site.position <tskit.Site.position>` is a continuous
    float, and the emitted :attr:`Site.pos <ancestree.sites.Site.pos>`
    truncates it.

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
    name is kept unchanged. Ploidy is auto-detected from the first
    variant. Pass ``ploidy=`` explicitly to override.

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

    @staticmethod
    def _phase_permutation(seed: int, pos: int, sample: str, ploidy: int) -> list[int]:
        """Draw a haplotype order for one unphased genotype.

        The 64-bit key is unranked into an ordering through the factorial
        number system, in integer arithmetic alone, so the draw is uniform
        over the ``ploidy!`` orderings to within the bias of reducing 64
        bits modulo ``ploidy!``, below ``1e-14`` for any ploidy up to eight.
        The key is derived from the record's position, the sample name and
        the ploidy, so the ordering is stable across passes and filters and
        reproducible across runs and platforms.

        :param seed: The source's ``phase_seed``.
        :param pos: The record's position.
        :param sample: Name of the sample in the panel.
        :param ploidy: Haplotypes carried by the genotype.
        :return: Source position of each haplotype, a permutation of
            ``range(ploidy)``.
        """
        digest = hashlib.blake2b(str(sample).encode(), digest_size=8)
        key = (int(seed) ^ (int(pos) * 0x9E3779B97F4A7C15)
               ^ (int(ploidy) * 0xD6E8FEB86659FD93)
               ^ (int.from_bytes(digest.digest(), "big")
                  * 0xBF58476D1CE4E5B9)) & 0xFFFFFFFFFFFFFFFF
        key = ((key ^ (key >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        key = ((key ^ (key >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        rank = (key ^ (key >> 31)) % math.factorial(ploidy)
        pool = list(range(ploidy))
        order: list[int] = []
        for k in range(ploidy, 0, -1):
            rank, j = divmod(rank, k)
            order.append(pool.pop(j))
        return order


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
            if ploidy is None:
                # Ploidy is len(genotype) - 1. A bare "." reaches htslib as one
                # value, so records are probed until one carries a called
                # genotype. A sites-only VCF defaults to diploid.
                for _ in range(_PLOIDY_PROBE_RECORDS):
                    record = next(iter(vcf), None)
                    if record is None:
                        break
                    gts = record.genotypes
                    if not gts:
                        break
                    if any(a >= 0 for g in gts for a in g[:-1]):
                        ploidy = max((len(g) - 1 for g in gts), default=1)
                        break
                if ploidy is None:
                    ploidy = 2
                ploidy = max(1, ploidy)
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

    def _open_vcf(self):
        """Open the VCF under the configured sample filter.

        :return: An open :class:`cyvcf2.VCF` handle.
        :raises ImportError: If cyvcf2 is not installed.
        """
        try:
            import cyvcf2
        except ImportError as e:
            raise ImportError(
                "CyVCF2Source requires cyvcf2. Install with "
                "`pip install cyvcf2` or `conda install -c bioconda cyvcf2`."
            ) from e

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
        prev_key = None
        warned_split = False
        try:
            for variant in vcf:
                if self._chrom_filter is not None and variant.CHROM != self._chrom_filter:
                    continue
                key = (variant.CHROM, variant.POS)
                if key == prev_key and not warned_split:
                    self._warn_split_multiallelic(variant.CHROM, variant.POS)
                    warned_split = True
                prev_key = key
                # Upper-case soft-masked alleles. An absent entry keeps its
                # slot, so the genotype indices stay aligned with the record.
                ref = variant.REF.upper() if variant.REF else None
                alts = [(a.upper() if a else None) for a in (variant.ALT or [])]
                all_alleles: list[str | None] = [ref, *alts]
                site_alleles = [a for a in all_alleles if a]
                if ref is None or not all(
                        len(a) == 1 and a in STATES for a in site_alleles):
                    continue

                genotypes = variant.genotypes  # indices + a phase flag
                tip_alleles: dict[str, str | None] = {}
                for row, sample in zip(self._genotype_rows, self._vcf_samples):
                    call = genotypes[row]
                    order = None
                    if self._ploidy > 1 and call is not None:
                        phased = self._phased if self._phased is not None \
                            else bool(call[-1])
                        if not phased:
                            called = [int(a) for a in call[:-1] if int(a) >= 0]
                            if len(set(called)) > 1:
                                self._note_unphased_once()
                                order = CyVCF2Source._phase_permutation(
                                    self._phase_seed, int(variant.POS), sample,
                                    self._ploidy)
                    if call is not None and len(call) - 1 > self._ploidy:
                        self._warn_ploidy_truncated(
                            str(variant.CHROM), int(variant.POS),
                            len(call) - 1, self._ploidy)
                    haps = self._haps_from_genotype(
                        call, self._ploidy, all_alleles, order=order)
                    if self._ploidy == 1:
                        tip_alleles[sample] = haps[0]
                    else:
                        for h, allele in enumerate(haps):
                            tip_alleles[f"{sample}_h{h}"] = allele

                yield Site(
                    chrom=str(variant.CHROM),
                    pos=int(variant.POS),
                    alleles=tuple(site_alleles),
                    tip_alleles=tip_alleles,
                )
        finally:
            vcf.close()

    @staticmethod
    def _haps_from_genotype(
        genotype, ploidy: int, alleles: "Sequence[str | None]",
        *, order: "Sequence[int] | None" = None,
    ) -> list[str | None]:
        """Decode one ``genotypes`` row into a length-``ploidy`` allele list.

        The row is per-haplotype allele indices followed by a phase flag. A
        negative or out-of-range index is missing, so its tip is
        marginalised, as is an index resolving to an absent allele. A row
        with fewer haplotypes than ``ploidy`` (a hemizygous record in a
        diploid-detected file) pads with ``None``, and a longer one is
        truncated.

        :param genotype: One row of ``cyvcf2.Variant.genotypes``.
        :param ploidy: Haplotypes to return.
        :param alleles: The record's alleles, REF first, upper-cased, with
            ``None`` in the slot of an allele the record does not carry.
        :param order: Source position of each haplotype, a permutation of
            ``range(ploidy)``. ``None`` keeps the written order.
        :return: Length-``ploidy`` list of alleles, ``None`` where missing.
        """
        if genotype is None:
            return [None] * ploidy
        idx = list(genotype[:-1])[:ploidy]
        n_alleles = len(alleles)
        out: list[str | None] = []
        for h in range(ploidy):
            src = int(order[h]) if order is not None else h
            j = int(idx[src]) if src < len(idx) else -1
            out.append(alleles[j] if 0 <= j < n_alleles else None)
        return out


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
        root = self._open_root()
        pos_arr = root[ZARR_POSITION_FIELD]
        alleles_arr = root[ZARR_ALLELE_FIELD]
        contig_arr = root[ZARR_CONTIG_FIELD]
        gt_arr = root["call_genotype"]
        # Consulted only where the caller has not fixed the phase. A store
        # written without the array carries unphased genotypes.
        phased_arr = (root.get("call_genotype_phased", None)
                      if self._phased is None else None)

        sample_idx = self._sample_indices
        prev_pos_key = None
        warned_split = False
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
                # Upper-case soft-masked alleles. An absent entry keeps its
                # slot, so the genotype indices stay aligned with the record.
                raw_alleles: tuple[str | None, ...] = tuple(
                    (a.upper() if a else None)
                    for a in (self._decode(x) for x in alleles_batch[i])
                )
                site_alleles = tuple(a for a in raw_alleles if a)
                chrom = self._contig_ids[int(contig_batch[i])]
                if self._chrom_filter is not None and chrom != self._chrom_filter:
                    continue
                # Tested before the allele filter, so a record skipped for
                # carrying a non-ACGT allele still marks the position it
                # shared with the record that follows it.
                pos_key = (chrom, int(pos_batch[i]))
                if pos_key == prev_pos_key and not warned_split:
                    self._warn_split_multiallelic(*pos_key)
                    warned_split = True
                prev_pos_key = pos_key
                # A record carrying no reference allele is not a SNP.
                if not site_alleles or raw_alleles[0] is None:
                    continue
                if not all(len(a) == 1 and a in STATES for a in site_alleles):
                    continue

                tip_alleles: dict[str, str | None] = {}
                gt_row = gt_batch[i]  # (n_kept_samples, ploidy)
                ph_row = None
                if self._ploidy > 1 and self._phased is not True:
                    ph_row = (phased_batch[i] if phased_batch is not None
                              else np.zeros(gt_row.shape[0], dtype=bool))
                for s_idx, name in enumerate(self._kept_sample_names):
                    order: Sequence[int] = range(self._ploidy)
                    if ph_row is not None and not bool(ph_row[s_idx]):
                        called = [int(a) for a in gt_row[s_idx] if int(a) >= 0]
                        if len(set(called)) > 1:
                            self._note_unphased_once()
                            order = CyVCF2Source._phase_permutation(
                                self._phase_seed, int(pos_batch[i]),
                                name, self._ploidy)
                    for h, src in enumerate(order):
                        allele_idx = int(gt_row[s_idx, src])
                        key = name if self._ploidy == 1 else f"{name}_h{h}"
                        tip_alleles[key] = (
                            raw_alleles[allele_idx]
                            if 0 <= allele_idx < len(raw_alleles) else None)

                yield Site(
                    chrom=str(chrom),
                    pos=int(pos_batch[i]),
                    alleles=site_alleles,
                    tip_alleles=tip_alleles,
                )

    def _open_root(self):
        """Open the VCZ root group."""
        import zarr
        return zarr.open(self._path, mode="r")
