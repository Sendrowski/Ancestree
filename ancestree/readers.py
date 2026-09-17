"""Reading annotated output back: the inverse of :mod:`ancestree.writers`.

:class:`~ancestree.readers.Reader` opens any of the three annotated formats a
run can write, dispatching on the path's suffix, and exposes what was recorded:

- :meth:`Reader.annotations() <ancestree.readers.Reader.annotations>` yields the
  per-site ancestral-allele assignments as
  :class:`~ancestree.readers.Annotation` records;
- :meth:`Reader.provenance() <ancestree.readers.Reader.provenance>` returns the
  :class:`~ancestree.readers.Provenance` record describing the run that made
  them.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import Iterable, Iterator, Mapping
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ancestree import STATES
from ancestree.sites import Site, SiteSource, _path_format
from ancestree.writers import (
    ZARR_ALLELE_FIELD,
    ZARR_CONTIG_FIELD,
    ZARR_POSITION_FIELD,
    AA_POST_FIELD,
    AA_PROB_FIELD,
    METADATA_KEY,
    PROVENANCE_HEADER,
    VCF_PROVENANCE_MARKER,
    ZARR_AA_FIELD,
    ZARR_AA_POST_FIELD,
    ZARR_AA_PROB_FIELD,
)
from ancestree._repr import ReprMixin

if TYPE_CHECKING:
    import tskit

    from ancestree.focal import FocalNode

    from ancestree.posterior import Grade


class Provenance(dict):
    """A run's provenance record: what produced a set of ancestral-allele assignments.

    A plain ``dict`` (so it serialises and indexes exactly as one) carrying
    ``software``, ``version``, ``mode``, ``parameters`` and ``timestamp``.
    Emitted by
    :meth:`Inference.provenance() <ancestree.inference.Inference.provenance>`
    and read back by
    :meth:`Reader.provenance() <ancestree.readers.Reader.provenance>`.
    """

    def __repr__(self) -> str:
        """Render the run header and its parameters, one per line.

        :return: A multi-line summary.
        """
        head = f"{self.get('software', 'ancestree')} {self.get('version', '')}".strip()
        if self.get("mode"):
            head += f" ({self['mode']} mode)"
        if self.get("timestamp"):
            head += f", {self['timestamp']}"
        params = self.get("parameters") or {}
        width = max((len(k) for k in params), default=0)
        lines = [f"  {k:<{width}}  {v}" for k, v in params.items()]
        return "\n".join([head, *lines])


@dataclasses.dataclass(frozen=True)
class Annotation:
    """One site's ancestral-allele assignment, as read back from an annotated output."""

    #: Position, in the source's own coordinates.
    pos: int
    #: MAP ancestral allele, or ``None`` where the site was left unannotated
    #: (below ``min_confidence``, or absent from the posteriors).
    aa: str | None
    #: Posterior probability of :attr:`aa`, or ``None``.
    aa_prob: float | None
    #: Contig label, or ``None`` for a tree sequence, which carries no contig.
    chrom: str | None = None
    #: The site's alleles: reference first for VCF and VCF Zarr, ancestral
    #: first for a tree sequence.
    alleles: tuple[str, ...] = ()
    #: Probability of each of the four states where the output records it,
    #: ``None`` where it was written without one.
    posterior: dict[str, float] | None = None

    def __repr__(self) -> str:
        """Render the site, its alleles and the assignment on one line.

        :return: e.g. ``1:556  G/C  AA=G  p=0.9993``.
        """
        parts = [f"{self.chrom}:{self.pos}" if self.chrom else str(self.pos)]
        if self.alleles:
            parts.append("/".join(self.alleles))
        parts.append(f"AA={self.aa if self.aa is not None else '.'}")
        if self.aa_prob is not None:
            parts.append(f"p={self.aa_prob:.4f}")
        return "  ".join(parts)


class Annotations(list):
    """A list of :class:`~ancestree.readers.Annotation` that prints as a table.

    Returned by :meth:`Reader.head() <ancestree.readers.Reader.head>`. A plain
    ``list`` in every other respect.
    """

    #: Columns rendered right-aligned. The rest are left-aligned.
    _NUMERIC = frozenset({"pos", "p", *STATES})

    @staticmethod
    def _probability(annotation: "Annotation", state: str) -> str:
        """Render one state's posterior probability, blank where absent."""
        value = (annotation.posterior or {}).get(state)
        return "" if value is None else f"{float(value):.4f}"

    def __repr__(self) -> str:
        """Render the records as a column-aligned table under a header row.

        Columns the data never fills are dropped: a tree sequence carries no
        ``chrom`` of its own, and an output written without the posterior has
        no state columns. Where the posterior is present it replaces the ``p``
        column, which is its value at the MAP allele.

        :return: The table, one record per line.
        """
        if not self:
            return ""
        states = [s for s in STATES if any((a.posterior or {}).get(s) is not None
                                           for a in self)]
        columns = [
            name for name, present in (
                ("chrom", any(a.chrom is not None for a in self)),
                ("pos", True),
                ("alleles", any(a.alleles for a in self)),
                ("AA", True),
                ("p", not states),
            ) if present
        ] + states
        rows = [
            [{
                "chrom": a.chrom or "",
                "pos": str(a.pos),
                "alleles": "/".join(a.alleles),
                "AA": a.aa if a.aa is not None else ".",
                "p": "" if a.aa_prob is None else f"{a.aa_prob:.4f}",
                **{s: self._probability(a, s) for s in states},
            }[name] for name in columns]
            for a in self
        ]
        widths = [
            max(len(name), *(len(row[i]) for row in rows))
            for i, name in enumerate(columns)
        ]
        def render(cells: list[str]) -> str:
            """One row, numeric columns right-aligned and the rest left."""
            return "  ".join(
                cell.rjust(width) if columns[i] in self._NUMERIC else cell.ljust(width)
                for i, (cell, width) in enumerate(zip(cells, widths))
            ).rstrip()

        return "\n".join([render(columns), *(render(row) for row in rows)])


class Reader(ReprMixin):
    """Read the assignments and provenance back out of an annotated output.

    The format is resolved from the path's suffix.

    .. code-block:: python

        import ancestree as anc

        reader = anc.Reader("annotated.vcf.gz")
        reader.head(3)
        reader.provenance()

    :param path: An output written by
        :meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>`,
        :meth:`Inference.to_zarr() <ancestree.inference.Inference.to_zarr>` or
        :meth:`Inference.to_arg() <ancestree.inference.Inference.to_arg>`.
    :raises FileNotFoundError: If ``path`` does not exist.
    """

    def __init__(self, path: str | os.PathLike):
        self._path = str(path)
        if not os.path.exists(self._path):
            raise FileNotFoundError(self._path)
        self.format = _path_format(self._path) or "vcf"

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"path": self._path, "format": self.format}

    def provenance(self) -> Provenance:
        """Read the provenance record the run wrote.

        Reads the last ``##ancestree_provenance`` header line of a VCF / BCF,
        the ``ancestree_provenance`` root attribute of a VCF Zarr store, or the
        most recent ``ancestree`` row of a tree sequence's provenance table.
        Re-annotating an output appends to the VCF header and the tree
        sequence's table, so each format is read newest first and reports the
        run that produced the assignments in the file.

        :return: The record, as
            :meth:`Inference.provenance() <ancestree.inference.Inference.provenance>`
            emitted it.
        :raises ValueError: If the output carries no ancestree provenance.
        """
        if self.format == "vcz":
            import zarr

            record = cast(
                "Any", zarr.open(self._path, mode="r")).attrs.get(
                    PROVENANCE_HEADER)
            if record is not None:
                return Provenance(record)
        elif self.format == "trees":
            import tskit

            for prov in reversed(list(tskit.load(self._path).provenances())):
                record = json.loads(prov.record)
                software = record.get("software")
                # tskit's schema makes software an object. The run record is
                # carried under parameters.
                if isinstance(software, dict):
                    if software.get("name") == "ancestree":
                        return Provenance(record.get("parameters") or {})
                elif software == "ancestree":
                    return Provenance(record)
        else:
            import cyvcf2

            marker = VCF_PROVENANCE_MARKER
            vcf = cyvcf2.VCF(self._path)
            try:
                header = vcf.raw_header.splitlines()
            finally:
                vcf.close()
            for line in reversed(header):
                if line.startswith(marker):
                    return Provenance(json.loads(line[len(marker):]))
        raise ValueError(f"{self._path} carries no ancestree provenance record")

    def head(self, n: int = 5) -> Annotations:
        """The first ``n`` assignments, for a brief look at what was written.

        :param n: How many sites to read. Default ``5``.
        :return: The records, printing one per line.
        """
        return Annotations(self.annotations(limit=n))

    def grade(
        self,
        truth: "Mapping[int, str] | Iterable[tuple[int, str]] | tskit.TreeSequence",
        filter: "Callable[[Site], bool] | None" = None,
        focal: "FocalNode | str" = "ingroup_mrca",
        *,
        sample_map: "Mapping[str, int] | None" = None,
    ) -> "Grade":
        """Grade the file's assignments against a truth.

        A record written without its posterior is scored as a point mass on
        its MAP allele.

        :param truth: ``{position: allele}``, an iterable of such pairs, or a
            :class:`tskit.TreeSequence` carrying the true mutations, whose
            allele at ``focal`` is then the truth.
        :param filter: Optional predicate over a
            :class:`~ancestree.sites.Site` restricting which sites are scored.
            An annotated file carries no genotypes, so a
            :class:`~ancestree.sites.PolymorphicSiteFilter`, which classifies a
            site by counting them, is refused: filter through
            :meth:`Inference.grade() <ancestree.inference.Inference.grade>` on
            the run itself instead.
        :param focal: The node of a tree-sequence truth at which the true
            allele is read, resolved over the panel, ingroup and outgroup
            samples recorded in the file's provenance. Defaults to the ingroup's
            most recent common ancestor.
        :param sample_map: ``{sample: node}`` mapping the provenance's sample
            names onto the nodes of a tree-sequence truth, read in that tree
            sequence's own node space. ``None`` reads the individual names of
            the truth, falling back to node ids as strings.
        :return: A :class:`~ancestree.posterior.Grade`.
        :raises ValueError: If a sample recorded in the provenance matches no
            sample of a tree-sequence truth, or if ``filter`` counts
            genotypes the file does not carry.
        """
        from ancestree.posterior import Grade, Posterior
        from ancestree.sites import Site

        if filter is not None and hasattr(filter, "accepts"):
            raise ValueError(
                "an annotated file carries no genotypes, so a "
                "PolymorphicSiteFilter cannot classify its sites and would "
                "silently grade none of them. Pass a predicate over a Site, "
                "or filter through Inference.grade() on the run itself.")

        import tskit

        # The sample lists resolve the focal node of a tree-sequence truth;
        # a position-to-allele truth needs no provenance record.
        params: dict[str, Any] = {}
        if isinstance(truth, tskit.TreeSequence):
            params = self.provenance().get("parameters", {})

        def pairs():
            for a in self.annotations():
                if a.aa is None:
                    continue
                if a.posterior is not None:
                    values = [float(a.posterior.get(s, 0.0)) for s in STATES]
                else:
                    values = [1.0 if s == a.aa else 0.0 for s in STATES]
                site = Site(chrom=a.chrom or "", pos=a.pos,
                            alleles=tuple(a.alleles), tip_alleles={})
                yield site, Posterior(tuple(STATES), np.asarray(values))

        return Grade(
            pairs(), truth, filter, focal=focal,
            ingroup_samples=params.get("ingroup_samples"),
            outgroup_samples=params.get("outgroup_samples"),
            panel_samples=params.get("panel_samples"),
            sample_map=sample_map,
        )

    def annotations(self, limit: int | None = None) -> Iterator[Annotation]:
        """Yield the per-site ancestral-allele assignments, in the file's own order.

        Reads the ``AA`` / ``AA_prob`` / ``AA_post`` ``INFO`` fields of a
        VCF / BCF, the ``variant_AA*`` arrays of a VCF Zarr store, or the
        ``site.metadata['ancestree']`` block of a tree sequence. Sites the
        writer left unannotated come back with
        :attr:`Annotation.aa <ancestree.readers.Annotation.aa>` at ``None``.

        :param limit: Stop after this many sites. ``None`` (default) reads all
            of them. The read is lazy either way, so a limit bounds the work on
            a genome-scale output.
        :return: Iterator of per-site assignments.
        """
        if self.format == "vcz":
            yield from self._zarr_annotations(limit)
        elif self.format == "trees":
            yield from self._tskit_annotations(limit)
        else:
            yield from self._vcf_annotations(limit)

    def _vcf_annotations(self, limit: int | None) -> Iterator[Annotation]:
        """Per-site assignments from the ``AA`` / ``AA_prob`` / ``AA_post`` ``INFO`` fields."""
        import cyvcf2

        vcf = cyvcf2.VCF(self._path)
        try:
            yield from self._annotations_from(vcf, limit)
        finally:
            vcf.close()

    def _annotations_from(self, vcf, limit: int | None) -> Iterator[Annotation]:
        """Per-site assignments from an open reader. The caller owns closing it.

        :param vcf: Open ``cyvcf2.VCF``.
        :param limit: Stop after this many records, ``None`` for all.
        """
        for n, variant in enumerate(vcf):
            if limit is not None and n >= limit:
                return
            allele = variant.INFO.get("AA")
            yield Annotation(
                pos=int(variant.POS),
                aa=None if allele in (None, ".") else str(allele),
                aa_prob=self._as_prob(variant.INFO.get(AA_PROB_FIELD)),
                chrom=str(variant.CHROM),
                alleles=(variant.REF, *variant.ALT),
                posterior=self._as_posterior(variant.INFO.get(AA_POST_FIELD)),
            )


    @staticmethod
    def _as_prob(raw: object) -> "float | None":
        """Parse an ``AA_prob`` value into a probability.

        :param raw: The field's value, ``None`` where the writer did not
            record one.
        :return: The probability, or ``None`` where absent or not a number.
        """
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(value) else value

    @staticmethod
    def _as_posterior(raw: object) -> "dict[str, float] | None":
        """Parse an ``AA_post`` value into ``{state: probability}``.

        :param raw: The field's value, ``None`` where the writer did not
            record one.
        :return: The per-state posterior, or ``None``.
        """
        if raw is None:
            return None
        values = raw if isinstance(raw, (tuple, list)) else str(raw).split(",")
        if len(values) != len(STATES):
            return None
        return dict(zip(STATES, (float(v) for v in values)))

    def _tskit_annotations(self, limit: int | None) -> Iterator[Annotation]:
        """Per-site assignments from each site's ``ancestree`` metadata block.

        A tree sequence has no contig of its own, so the label comes from the
        run's provenance record. The full posterior is present only when the
        output was written with ``store_posterior=True``.
        """
        import tskit

        ts = tskit.load(self._path)
        try:
            chrom = (self.provenance().get("parameters") or {}).get("chrom")
        except ValueError:
            chrom = None
        for n, site in enumerate(ts.sites()):
            if limit is not None and n >= limit:
                return
            metadata = site.metadata if isinstance(site.metadata, dict) else {}
            block = metadata.get(METADATA_KEY) or {}
            derived = dict.fromkeys(
                m.derived_state for m in site.mutations
                if m.derived_state != site.ancestral_state
            )
            called = site.ancestral_state not in ("", ".")
            yield Annotation(
                pos=int(site.position),
                aa=block.get("map_allele") if called else None,
                aa_prob=block.get("max_prob"),
                chrom=None if chrom is None else str(chrom),
                # A site blanked by min_confidence carries an empty
                # ancestral state, which is a marker, not an allele.
                alleles=tuple(
                    a for a in (site.ancestral_state, *derived) if a),
                posterior=block.get("posterior"),
            )

    def _posterior_or_none(self, values):
        """A stored posterior, or ``None`` where the row carries none.

        The float arrays use NaN as their missing marker.

        :param values: The row's stored probabilities, or ``None``.
        :return: A posterior mapping, or ``None``.
        """
        if values is None:
            return None
        row = [float(v) for v in values]
        if not row or any(math.isnan(v) for v in row):
            return None
        return self._as_posterior(row)

    def _zarr_annotations(self, limit: int | None) -> Iterator[Annotation]:
        """Per-site assignments from the store's ``variant_AA*`` arrays."""
        import zarr

        root = cast("Any", zarr.open(self._path, mode="r"))
        # Byte-typed stores (older bio2zarr, tsinfer-written) are decoded.
        _dec = SiteSource._decode
        if ZARR_AA_FIELD not in root:
            raise ValueError(f"{self._path} carries no ancestree annotations")
        stop = (root[ZARR_POSITION_FIELD].shape[0]
                if limit is None else limit)
        contigs = root["contig_id"][:]
        # A template without variant_allele is one ZarrWriter accepts, so the
        # rows read back with no allele list rather than failing.
        alleles = (root[ZARR_ALLELE_FIELD][:stop]
                   if ZARR_ALLELE_FIELD in root else [()] * int(stop))
        post = (root[ZARR_AA_POST_FIELD][:stop]
                if ZARR_AA_POST_FIELD in root else None)
        # A store carrying only the VCF specification's AA field has no
        # probability array, and reads back as None.
        probs = (root[ZARR_AA_PROB_FIELD][:stop]
                 if ZARR_AA_PROB_FIELD in root else [None] * int(stop))
        for i, (pos, contig, allele, prob) in enumerate(zip(
            root[ZARR_POSITION_FIELD][:stop],
            root[ZARR_CONTIG_FIELD][:stop],
            root[ZARR_AA_FIELD][:stop], probs,
        )):
            called = _dec(allele) not in (".", "")
            yield Annotation(
                pos=int(pos),
                aa=_dec(allele) if called else None,
                aa_prob=(None if prob is None or math.isnan(float(prob))
                         else float(prob)),
                chrom=_dec(contigs[int(contig)]),
                alleles=tuple(_dec(a) for a in alleles[i] if _dec(a)),
                posterior=self._posterior_or_none(
                    None if post is None else post[i]),
            )
