"""Output writers for annotated VCF, VCF Zarr and tskit ``.trees`` files.

All three consume the ``(Site, Posterior)`` pairs that
:meth:`Inference.infer() <ancestree.inference.Inference.infer>` yields, and record the MAP
ancestral allele together with its posterior probability:

- :class:`~ancestree.writers.VCFWriter` copies an input VCF and adds ``AA``,
  ``AA_prob`` and ``AA_post`` INFO fields. Variants with no matching posterior
  pass through unannotated.
- :class:`~ancestree.writers.TskitWriter` sets each site's
  ``ancestral_state`` to the MAP allele and stores the full posterior in the
  site metadata.
- :class:`~ancestree.writers.ZarrWriter` annotates a copy of a ``.vcz`` store
  with ``variant_AA``, ``variant_AA_prob`` and ``variant_AA_post`` arrays.

An optional ``info`` dict naming the prior, substitution model and fitted
parameters is recorded once per run. An optional ``provenance`` dict is
written to each format's own provenance location;
:meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>`,
:meth:`Inference.to_zarr() <ancestree.inference.Inference.to_zarr>` and
:meth:`Inference.to_arg() <ancestree.inference.Inference.to_arg>` populate it automatically, so
command-line and Python output carry the same record.
"""
from __future__ import annotations

import contextlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tskit

from ancestree.posterior import Posterior
from ancestree import STATE_INDEX, STATES
from ancestree.sites import Site, SiteSource
from ancestree._repr import ReprMixin


#: ``INFO`` / array names the writers emit and :mod:`ancestree.readers` reads.
AA_PROB_FIELD = "AA_prob"
AA_POST_FIELD = "AA_post"
PROVENANCE_HEADER = "ancestree_provenance"
#: The VCF header line the provenance record is written on.
VCF_PROVENANCE_MARKER = f"##{PROVENANCE_HEADER}="
METADATA_KEY = "ancestree"
#: The INFO fields ancestree writes, keyed by ID. The VCF writer emits them as
#: ``##INFO`` header lines, the Zarr writer as per-array ``description`` attributes.
AA_INFO_FIELDS: dict[str, dict[str, str]] = {
    "AA": {
        "Number": "1",
        "Type": "String",
        "Description": "Inferred ancestral allele (ancestree)",
    },
    "AA_prob": {
        "Number": "1",
        "Type": "Float",
        "Description": ("Posterior probability of the inferred ancestral "
                        "allele (ancestree)"),
    },
    "AA_post": {
        "Number": str(len(STATES)),
        "Type": "Float",
        "Description": "Posterior over " + ",".join(STATES) + " (ancestree)",
    },
}

#: The VCZ arrays the readers and writers use.
ZARR_POSITION_FIELD = "variant_position"
ZARR_CONTIG_FIELD = "variant_contig"
ZARR_ALLELE_FIELD = "variant_allele"

ZARR_AA_FIELD = "variant_AA"
ZARR_AA_PROB_FIELD = "variant_AA_prob"
ZARR_AA_POST_FIELD = "variant_AA_post"


class Writer(ReprMixin, ABC):
    """ABC for output writers that persist annotated ancestral alleles.

    Subclasses consume the ``(Site, Posterior)`` pairs of
    :meth:`Inference.infer() <ancestree.inference.Inference.infer>` and
    implement :meth:`write` for one output format.
    """

    @staticmethod
    def _row_locator(pos, contig, contig_ids, name_to_cidx):
        """Build a ``(chrom, pos) -> row`` lookup into a variant axis.

        On a ``(contig, pos)``-sorted axis each key is found by
        ``np.searchsorted`` over the contig's row block, using only the two
        integer arrays. An unsorted axis falls back to a ``(chrom, pos) -> row``
        dict. Either way the key maps to a single canonical row. Callers that must
        reach every row sharing a key (multiallelic-split records) expand it with
        :meth:`Writer._rows_for_site`.

        :param pos: Per-variant position as an integer array.
        :param contig: Per-variant contig index array.
        :param contig_ids: Contig labels indexed by contig index.
        :param name_to_cidx: Map from contig label to contig index.
        :return: A ``locate(chrom, pos) -> int`` function (``-1`` if no row).
        """
        import numpy as np

        n = int(pos.shape[0])
        # Canonical iff contig is non-decreasing and, within each contig run,
        # pos is non-decreasing, the ordering searchsorted relies on.
        if n == 0:
            return lambda chrom, p: -1
        dc = np.diff(contig)
        canonical = bool(np.all(dc >= 0)) and bool(np.all((dc > 0) | (np.diff(pos) >= 0)))

        if not canonical:
            row_of = {
                (contig_ids[int(contig[i])], int(pos[i])): i for i in range(n)
            }
            return lambda chrom, p: row_of.get((chrom, p), -1)

        # Precompute each contig's [lo, hi) row block once (contig is sorted).
        uniq, starts = np.unique(contig, return_index=True)
        bounds = [int(s) for s in starts] + [n]
        ranges = {
            int(cidx): (bounds[k], bounds[k + 1]) for k, cidx in enumerate(uniq)
        }

        def locate(chrom, p):
            """Row index of ``(chrom, p)`` in the template, or -1 if absent."""
            cidx = name_to_cidx.get(chrom)
            if cidx is None:
                return -1
            span = ranges.get(cidx)
            if span is None:
                return -1
            lo, hi = span
            # pos[lo:hi] is a view (basic slicing), so no per-call copy.
            j = lo + int(np.searchsorted(pos[lo:hi], p, side="left"))
            return j if j < hi and int(pos[j]) == p else -1

        return locate

    @staticmethod
    def _called_alleles(alleles) -> frozenset:
        """The called alleles of a record or site, as a set.

        Drops the missing-data placeholder a tree sequence appends, so a site read
        from a tree sequence compares equal to the template row it came from.

        :param alleles: Allele strings, possibly containing ``None``.
        :return: The non-empty allele strings.
        """
        return frozenset(a.upper() for a in alleles if a)

    @staticmethod
    def _rows_for_site(annotated, pos, contig, row: int, site, alleles=None):
        """Every row of a shared-key run that belongs to this site.

        Several records can share a position: a split multiallelic, or an indel
        beside a SNP. The sources yield a :class:`~ancestree.sites.Site` only for
        records whose alleles are all single canonical bases, so the posterior
        stream is a subset of the template rows and position alone does not
        identify which row a posterior belongs to. When the template's alleles are
        known the run is narrowed to the rows carrying this site's alleles.

        :param annotated: Per-row bool marking placed rows.
        :param pos: Per-row integer position.
        :param contig: Per-row contig index.
        :param row: The canonical row the locator returned.
        :param site: The site being placed.
        :param alleles: Per-row allele tuples, or None when the template does not
            expose them.
        :return: Row indices, empty when the run is exhausted.
        """
        n = int(pos.shape[0])
        lo = row
        while lo > 0 and pos[lo - 1] == pos[row] and contig[lo - 1] == contig[row]:
            lo -= 1
        hi = row
        while hi + 1 < n and pos[hi + 1] == pos[row] and contig[hi + 1] == contig[row]:
            hi += 1
        free = [j for j in range(lo, hi + 1) if not annotated[j]]
        if alleles is not None:
            # Compare as sets over the called alleles. A source may order REF and
            # ALT differently from the template, a template row may carry ALTs the
            # site does not use, and tskit appends ``None`` for missing data, none
            # of which changes which record the site describes.
            want = Writer._called_alleles(site.alleles)
            matched = [j for j in free if want <= Writer._called_alleles(alleles[j])]
            if matched:
                return matched[:1]
            # The site names an allele the record does not carry, so it describes
            # a different variant.
            return []
        # Every source emits one Site per record: a tree sequence with non-integer
        # positions puts several on one integer key, and a VCF-shaped stream yields
        # one per line of a split multiallelic. So a site claims a single row and
        # the next site at that key takes the following one.
        return free[:1]


    #: Written in place of the MAP allele when its probability is below
    #: ``min_confidence``. ``TskitWriter`` overrides it with tskit's own
    #: unannotated marker.
    AA_UNKNOWN = "."

    @staticmethod
    def _staged_path(output: str) -> str:
        """A sibling path to build ``output`` at before renaming it into place.

        The marker is a prefix, so the extension chain (``.vcf.gz`` and the
        like) is preserved. The leading dot keeps a partial build out of an
        unqualified glob.

        :param output: Final path.
        :return: Sibling path in the same directory.
        """
        d, name = os.path.split(output)
        return os.path.join(d, f".partial-{os.getpid()}-{name}")

    @staticmethod
    def _finish_staged(staged: str, output: str) -> None:
        """Rename a completed staged file onto ``output``.

        The staged file is removed if the rename fails, so a destination that
        cannot be replaced leaves nothing behind.

        :param staged: Path returned by :meth:`Writer._staged_path`.
        :param output: Final path.
        """
        try:
            os.replace(staged, output)
        except BaseException:
            if os.path.exists(staged):
                os.unlink(staged)
            raise

    def _call(self, site: Site, posterior: Posterior) -> tuple[str, float]:
        """The MAP allele to write, and its probability.

        A site carrying no A/C/G/T allele leaves every tip marginalised, so
        its posterior is the prior and its MAP allele is an artefact of the
        tie-break. Such a site is written as :attr:`AA_UNKNOWN`.

        :param site: The site the posterior was scored at.
        :param posterior: The site's :class:`~ancestree.posterior.Posterior`.
        :return: ``(allele, max_prob)``, the allele replaced by
            :attr:`AA_UNKNOWN` below ``min_confidence`` or where the site
            carries no allele the model can read.
        """
        max_prob = float(posterior.max_prob)
        if not site.has_representable_allele:
            return self.AA_UNKNOWN, max_prob
        if self._min_confidence is not None and max_prob < self._min_confidence:
            return self.AA_UNKNOWN, max_prob
        return str(posterior.map_allele), max_prob


    @staticmethod
    def _posterior_vector(posterior) -> list[float]:
        """The posterior as one probability per canonical state, in order.

        A posterior is over its model's alphabet, which need not be the full
        A/C/G/T set, so place each allele into its own slot and leave the
        states it does not cover at zero.

        :param posterior: The site's :class:`~ancestree.posterior.Posterior`.
        :return: One probability per entry of :data:`ancestree.STATES`.
        """
        values = [0.0] * len(STATES)
        for allele, probability in posterior.items():
            index = STATE_INDEX.get(allele)
            if index is not None:
                values[index] = float(probability)
        return values

    def _place_posteriors(
        self, posteriors, locate, pos, contig, row_alleles,
        annotated, aa, prob, post,
    ) -> tuple[int, int]:
        """Fill the pre-allocated per-row arrays from the posterior stream.

        Each posterior's ``(chrom, pos)`` key locates a canonical template row,
        which :meth:`Writer._rows_for_site` narrows to the rows carrying the site's
        alleles. Keys absent from the template and keys whose rows are already
        taken are counted and reported once each.

        :param posteriors: Iterable of ``(Site, Posterior)`` pairs.
        :param locate: The ``(chrom, pos) -> row`` lookup built by
            :meth:`Writer._row_locator`.
        :param pos: Per-row integer position.
        :param contig: Per-row contig index.
        :param row_alleles: Per-row allele tuples, or ``None`` when the
            template does not expose them.
        :param annotated: Per-row bool flags, set for every row filled.
        :param aa: Per-row MAP allele array.
        :param prob: Per-row MAP probability array.
        :param post: ``(n_rows, len(STATES))`` per-state posterior array, or
            ``None`` to skip the per-state posterior.
        :return: ``(n_seen, n_placed)``, the posteriors consumed and the rows
            filled.
        """
        store_post = post is not None
        n_seen = 0
        n_placed = 0
        n_absent = 0
        first_absent = None
        n_dropped = 0
        first_dropped = None
        for site, posterior in posteriors:
            n_seen += 1
            row = locate(str(site.chrom), int(site.pos))
            if row < 0:
                n_absent += 1
                if first_absent is None:
                    first_absent = (site.chrom, site.pos)
                continue
            rows = self._rows_for_site(annotated, pos, contig, row, site,
                                  row_alleles)
            if not rows:
                n_dropped += 1
                first_dropped = first_dropped or (site.chrom, site.pos)
                continue
            allele, max_prob = self._call(site, posterior)
            vector = self._posterior_vector(posterior) if store_post else None
            for r in rows:
                n_placed += 1
                annotated[r] = True
                aa[r] = allele
                prob[r] = max_prob
                if store_post:
                    post[r] = vector

        if first_absent is not None:
            chrom, position = first_absent
            self._log.warning(
                "%d posterior(s) name a record the template does not carry "
                "(first at %s:%s). A whole-run mismatch is refused, but a "
                "partial one leaves those assignments unwritten: check that the "
                "contig labels and the region covered agree.",
                n_absent, chrom, position,
            )
        if first_dropped is not None:
            chrom, position = first_dropped
            self._log.warning(
                "%d posterior(s) found no template row carrying their "
                "alleles (first at %s:%s). Either several sites share the "
                "record key (contig, integer position) and the template has "
                "fewer rows, or the row at that key carries a different "
                "allele set. Pass the source VCF / store as the template to "
                "annotate against the data the sites came from",
                n_dropped, chrom, position)

        return n_seen, n_placed

    def _warn_no_posteriors(self, n_records: int, target: str) -> None:
        """Report a non-empty template left wholly unannotated.

        :param n_records: Records the template carries.
        :param target: What the posteriors would have been matched against.
        """
        self._log.warning(
            "%s received no posteriors, so all %s record(s) of %s are written "
            "unannotated to %s.",
            type(self).__name__, f"{n_records:,}", target, self._output,
        )

    def _refuse_no_matches(
        self, n_posteriors: int, target: str,
        remedy: str = ("Check that the contig labels and positions of the "
                       "posteriors and of the template agree."),
    ) -> None:
        """Refuse an output in which nothing was annotated.

        :param n_posteriors: Posteriors offered.
        :param target: What the posteriors were matched against.
        :param remedy: Advice on what to reconcile, appended to the message.
        :raises ValueError: Always.
        """
        raise ValueError(
            f"{type(self).__name__} matched 0 of {n_posteriors:,} posteriors "
            f"against {target}, so nothing would be annotated. {remedy}"
        )

    @abstractmethod
    def write(
        self,
        posteriors: Iterable[tuple[Site, Posterior]],
        *,
        store_posterior: bool = True,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> int:
        """Persist the per-site posteriors. Returns the number of records annotated.

        :param posteriors: Iterable of ``(Site, Posterior)`` pairs.
            Typically fully consumed before writing. Keep the input
            stream and the posterior generator pointing at the same set
            of sites.
        :param store_posterior: Record the full per-state posterior alongside
            the MAP allele. ``False`` writes the allele and its probability only.
        :param info: Optional run-level metadata. Keys become
            destination-specific fields (``AA_<key>`` ``INFO`` in VCF;
            ``site.metadata['ancestree']['inference'][<key>]`` in tskit;
            ``attrs['ancestree_info'][<key>]`` in VCF Zarr).
        :param provenance: Optional structured run record (version, mode,
            parameters) written once to the format's canonical provenance
            location (VCF header, tskit provenance table, or VCF Zarr root
            ``attrs``), not per record.
        :return: Number of input records that received an annotation.
        """


    def _require_backend(self, module: str, install: str) -> None:
        """Fail at construction if the backend is missing, not mid-write.

        :param module: Import name to probe.
        :param install: Install hint quoted in the error.
        :raises ImportError: If ``module`` cannot be imported.
        """
        import importlib

        try:
            importlib.import_module(module)
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires {module}. Install with {install}."
            ) from e

    def _init_output(
        self, output: "str | os.PathLike", min_confidence: float | None,
        samples: "Iterable[str] | None" = None,
    ) -> None:
        """Record the destination, the reporting threshold and the samples.

        :param output: Destination path.
        :param min_confidence: Calls below this are written as missing.
        :param samples: Sample columns to write, or ``None`` for all of them.
        """
        self._output = str(output)
        self._min_confidence: float | None = (
            float(min_confidence) if min_confidence is not None else None
        )
        self._samples: "frozenset[str] | None" = (
            frozenset(samples) if samples is not None else None)

    def _refuse_template_overwrite(self) -> None:
        """Refuse an output that resolves to the template.

        :raises ValueError: If the output is the template.
        """
        if os.path.realpath(self._input) == os.path.realpath(self._output):
            raise ValueError(
                f"{type(self).__name__} cannot write over its template "
                f"({self._output}). Write to a different path.")

    def _kept_samples(self, names: "Iterable[str]") -> "list[str] | None":
        """The template's sample columns to write, in template order.

        :param names: The template's sample names.
        :return: The kept names, or ``None`` when every column is written.
        :raises ValueError: If no column is among the samples to write.
        """
        if self._samples is None:
            return None
        names = list(names)
        kept = [n for n in names if n in self._samples]
        if not kept:
            raise ValueError(
                f"none of the {len(names)} sample columns of {self._input!r} "
                f"is among the samples to write; the template holds "
                f"{names[:3]}")
        return kept


class VCFWriter(Writer):
    """Copy an input VCF to disk with ``AA`` / ``AA_prob`` / ``AA_post`` ``INFO``
    annotations.

    The input VCF is used as a template: its full header and every variant
    field other than the ``AA`` ``INFO`` fields, which each run rewrites,
    are passed through to the output. For each variant the writer
    looks up the posterior by ``(chrom, pos)`` and, if found, sets the
    ``INFO`` fields. Variants without a matching posterior are
    written unannotated.

    The output format follows the extension of ``output_vcf``,
    case-insensitively. A ``.gz`` or ``.bgz`` suffix writes bgzipped VCF,
    ``.bcf`` writes BCF and any other suffix writes plain VCF.

    :param input_vcf: Template VCF / VCF.GZ / BCF path.
    :param output_vcf: Destination path.
    :param min_confidence: Optional minimum
        :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
        required to commit to a MAP allele. When a site's MAP probability
        falls below this threshold, the writer emits ``AA=.`` (the VCF
        "ancestral allele unknown" sentinel) in place of the MAP allele.
        ``AA_prob`` still records the (sub-threshold) max probability.
        ``None`` (default) disables the check.
    :param samples: Sample columns to write, matched against the template's
        sample names. ``None`` (default) writes every column. ``INFO`` allele
        counts such as ``AC`` and ``AN`` are copied from the template.
    :raises ImportError: If ``cyvcf2`` is not installed.
    """

    AA_INFO_HEADER = {"ID": "AA", **AA_INFO_FIELDS["AA"]}
    AA_PROB_INFO_HEADER = {"ID": "AA_prob", **AA_INFO_FIELDS["AA_prob"]}
    AA_POST_INFO_HEADER = {"ID": "AA_post", **AA_INFO_FIELDS["AA_post"]}

    @staticmethod
    def _vcf_type_of(value: object) -> str:
        """Map a Python ``info`` value to a VCF 4.2 INFO ``Type`` token.

        Accepted: ``bool`` → ``Integer`` (0/1 for portability),
        ``int`` → ``Integer``, ``float`` → ``Float``, anything else
        stringifies as ``String``.
        """
        if isinstance(value, bool):
            return "Integer"
        if isinstance(value, int):
            return "Integer"
        if isinstance(value, float):
            return "Float"
        return "String"

    def _float_info(self, *values: float) -> "str | float | tuple[float, ...]":
        """One ``Type=Float`` ``INFO`` value, rendered for the output format.

        Text VCF receives a full-mantissa string, since htslib formats a float
        at six significant digits. BCF stores the number itself.

        :param values: The value, or the whole vector for a multi-valued field.
        :return: The values as cyvcf2 should receive them.
        """
        if self._output.lower().endswith(".bcf"):
            return values[0] if len(values) == 1 else tuple(values)
        return ",".join(f"{v:.17g}" for v in values)

    @staticmethod
    def _clear_owned_info(variant) -> None:
        """Drop the run-level ``AA_<key>`` fields a previous run left behind.

        Only the extra fields are cleared. ``AA``, ``AA_prob`` and ``AA_post``
        are rewritten unconditionally by the caller.

        :param variant: The template record being emitted.
        """
        for field_id in list(getattr(variant, "INFO", [])):
            key = field_id[0] if isinstance(field_id, tuple) else field_id
            if (str(key).startswith("AA_")
                    and str(key) not in ("AA_prob", "AA_post")):
                del variant.INFO[str(key)]

    @staticmethod
    def _coerce_info_value(value: object, vcf_type: str) -> object:
        """Coerce a Python info value to what ``cyvcf2.Variant.INFO`` accepts.

        A value that cannot be converted to the declared VCF type raises here.
        """
        if vcf_type == "Integer":
            return int(value)  # type: ignore[call-overload]
        if vcf_type == "Float":
            return float(value)  # type: ignore[arg-type]
        return str(value)

    def __init__(
        self,
        input_vcf: str | os.PathLike,
        output_vcf: str | os.PathLike,
        *,
        min_confidence: float | None = None,
        samples: "Iterable[str] | None" = None,
    ) -> None:
        """Validate that the backend is importable. Defer all I/O to :meth:`write`."""
        self._require_backend(
            "cyvcf2",
            "`pip install cyvcf2` or `conda install -c bioconda cyvcf2`")
        self._input = str(input_vcf)
        self._init_output(output_vcf, min_confidence, samples)

    def write(
        self,
        posteriors: Iterable[tuple[Site, Posterior]],
        *,
        store_posterior: bool = True,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> int:
        """Stream the template VCF to ``output_vcf`` with the ``AA`` annotations.

        :param posteriors: Iterable of ``(Site, Posterior)`` pairs. Fully
            consumed before writing. Keep the input VCF and the
            posterior generator pointing at the same set of sites.
        :param store_posterior: When ``True`` (default), also emit the
            ``AA_post`` ``INFO`` field holding the whole per-state posterior.
        :param info: Optional run-level constants (e.g.
            ``{"prior": "adaptive", "model": "K2", "kappa": 2.13}``).
            Each key ``K`` is declared as an ``INFO`` field ``AA_K`` in
            the VCF header (Type derived from the value type) and
            emitted on every annotated record. Values must be str, int,
            float, or bool.
        :param provenance: Optional structured run record (version, mode,
            parameters). Written once to the VCF header as a human-readable
            ``##source=ancestree <version> (<mode>)`` line plus a
            machine-readable ``##ancestree_provenance=<json>`` line. Not
            emitted per record.
        :return: Number of records annotated.
        :raises ValueError: If ``store_posterior`` is set for a ``.bcf``
            destination, which cannot carry a multi-valued ``Type=Float``
            field.
        """
        if store_posterior and self._output.lower().endswith(".bcf"):
            raise ValueError(
                f"store_posterior=True is not supported for BCF output "
                f"({self._output!r}): a multi-valued Type=Float INFO field "
                f"cannot be written through cyvcf2. Write .vcf or .vcf.gz to "
                f"keep AA_post, or pass store_posterior=False to write AA and "
                f"AA_prob only."
            )
        return self._write_streaming(
            posteriors, store_posterior=store_posterior,
            info=info, provenance=provenance,
        )

    def _add_headers(
        self,
        in_vcf,
        info: Mapping[str, object] | None,
        provenance: Mapping[str, object] | None,
        store_posterior: bool = True,
    ) -> list[tuple[str, str, object]]:
        """Add the ``AA`` / ``AA_prob`` / ``AA_post`` (and any ``info``) INFO headers
        + provenance.

        :param in_vcf: The open ``cyvcf2.VCF`` template whose header is edited.
        :param info: Optional run-level constants declared as ``AA_<key>``.
        :param provenance: Optional structured run record for the header.
        :return: ``[(field_id, vcf_type, value), ...]`` for the ``info`` fields,
            to be emitted per annotated record.
        """
        in_vcf.add_info_to_header(self.AA_INFO_HEADER)
        in_vcf.add_info_to_header(self.AA_PROB_INFO_HEADER)
        if store_posterior:
            in_vcf.add_info_to_header(self.AA_POST_INFO_HEADER)
        if provenance:
            version = str(provenance.get("version", ""))
            mode = str(provenance.get("mode", ""))
            in_vcf.add_to_header(
                f"##source=ancestree {version} ({mode})".rstrip()
            )
            in_vcf.add_to_header(
                VCF_PROVENANCE_MARKER
                + json.dumps(provenance, separators=(",", ":"), default=str)
            )

        info_fields: list[tuple[str, str, object]] = []  # (field_id, vcf_type, value)
        if info:
            for key, value in info.items():
                field_id = f"AA_{key}"
                if field_id in ("AA_prob", "AA_post"):
                    raise ValueError(
                        f"info key {key!r} maps to the reserved INFO field "
                        f"{field_id!r} (used for the ancestral-allele assignment); "
                        f"rename it"
                    )
                vcf_type = self._vcf_type_of(value)
                in_vcf.add_info_to_header({
                    "ID": field_id,
                    "Number": "1",
                    "Type": vcf_type,
                    "Description": f"Ancestree {key}",
                })
                info_fields.append((field_id, vcf_type, value))
        return info_fields

    def _write_streaming(
        self,
        posteriors: Iterable[tuple[Site, Posterior]],
        *,
        store_posterior: bool = True,
        info: Mapping[str, object] | None,
        provenance: Mapping[str, object] | None,
    ) -> int:
        """Two-pass placement: build key arrays, stream posteriors, then emit.

        The output must differ from the template: pass 2 opens it while pass 1
        is still reading the same file.

        Pass 1 reads the template once to build the ``(pos, contig)`` integer
        key arrays. Posteriors are streamed into their template rows (see
        ``_row_locator()``). Pass 2 re-reads the template and emits,
        annotating every record whose key locates to a placed row. Peak memory
        is the O(n_variants) key and flag arrays, not the posterior stream.
        """
        self._refuse_template_overwrite()
        import cyvcf2
        import numpy as np

        # Pass 1: integer key arrays in template file order.
        in_vcf = cyvcf2.VCF(self._input)
        kept = self._kept_samples(in_vcf.samples)
        positions: list[int] = []
        contigs: list[int] = []
        contig_ids: list[str] = []
        row_alleles: list[tuple[str, ...]] = []
        name_to_cidx: dict[str, int] = {}
        for variant in in_vcf:
            chrom = str(variant.CHROM)
            cidx = name_to_cidx.get(chrom)
            if cidx is None:
                cidx = len(contig_ids)
                name_to_cidx[chrom] = cidx
                contig_ids.append(chrom)
            positions.append(int(variant.POS))
            contigs.append(cidx)
            row_alleles.append(
                (str(variant.REF).upper(),
                 *(str(a).upper() for a in (variant.ALT or ()))))
        in_vcf.close()
        n = len(positions)
        pos = np.asarray(positions, dtype=np.int64)
        contig = np.asarray(contigs, dtype=np.int32)
        locate = self._row_locator(pos, contig, contig_ids, name_to_cidx)

        # Stream posteriors into their canonical rows.
        annotated = np.zeros(n, dtype=bool)
        aa = np.full(n, self.AA_UNKNOWN, dtype="U1")
        prob = np.empty(n, dtype=np.float64)
        post = (np.zeros((n, len(STATES)), dtype=np.float64)
                if store_posterior else None)
        n_posteriors, _ = self._place_posteriors(
            posteriors, locate, pos, contig, row_alleles,
            annotated, aa, prob, post,
        )
        if n_posteriors == 0 and n:
            self._warn_no_posteriors(n, f"the template {self._input}")

        # Pass 2: re-open and emit. Built at a sibling path and renamed onto
        # the destination once complete.
        in_vcf = cyvcf2.VCF(self._input, samples=kept)
        info_fields = self._add_headers(in_vcf, info, provenance, store_posterior)
        staged = self._staged_path(self._output)
        lower = self._output.lower()
        mode = ("wz" if lower.endswith((".gz", ".bgz"))
                else "wb" if lower.endswith(".bcf") else "w")
        writer = cyvcf2.Writer(staged, in_vcf, mode=mode)
        try:
            n_annotated = 0
            # Pass 2 walks the template in the same order pass 1 built the
            # arrays, so the row is the record index. Re-locating by key would
            # discard pass 1's choice among rows sharing a position.
            for row, variant in enumerate(in_vcf):
                if row < n and annotated[row]:
                    variant.INFO["AA"] = str(aa[row])
                    variant.INFO["AA_prob"] = self._float_info(float(prob[row]))
                    if post is not None:
                        variant.INFO["AA_post"] = self._float_info(
                            *(float(v) for v in post[row])
                        )
                    elif variant.INFO.get("AA_post") is not None:
                        del variant.INFO["AA_post"]
                    self._clear_owned_info(variant)
                    for field_id, vcf_type, value in info_fields:
                        variant.INFO[field_id] = self._coerce_info_value(value, vcf_type)
                    n_annotated += 1
                else:
                    # Unscored rows and AA_* fields from earlier runs are cleared.
                    for field_id in ("AA", "AA_prob", "AA_post"):
                        if variant.INFO.get(field_id) is not None:
                            del variant.INFO[field_id]
                    self._clear_owned_info(variant)
                writer.write_record(variant)
        except BaseException:
            writer.close()
            in_vcf.close()
            if os.path.exists(staged):
                os.unlink(staged)
            raise
        writer.close()
        in_vcf.close()
        if n_posteriors and n_annotated == 0:
            # Refused before the staged file reaches the destination.
            if os.path.exists(staged):
                os.unlink(staged)
            self._refuse_no_matches(
                n_posteriors, f"the template {self._input}")
        self._finish_staged(staged, self._output)
        self._log.info("Wrote %s annotated sites to %s",
                       f"{n_annotated:,}", self._output)
        return n_annotated


class TskitWriter(Writer):
    """Write a :class:`tskit.TreeSequence` with sites annotated by inferred ancestral alleles.

    Each site in the source tree sequence is rewritten with
    ``ancestral_state`` set to the MAP allele from the corresponding
    posterior. Where that differs from the input state, its mutations are
    re-derived against the new state, so every sample decodes to the same
    allele as in the input. The full posterior, MAP allele and MAP probability
    are stored under an ``"ancestree"`` key in ``site.metadata`` (JSON codec,
    set on the sites table if it has no schema).

    Sites without a matching posterior keep their original
    ``ancestral_state``, with any ``ancestree`` metadata block removed.

    :param input_ts: Source :class:`tskit.TreeSequence` (or a path to a
        ``.trees`` file, loaded eagerly).
    :param output_path: Where to write the annotated ``.trees`` file.
    :param min_confidence: Optional minimum
        :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
        required to commit to a MAP allele. When a site's MAP probability
        falls below this threshold, ``ancestral_state`` is written as the
        empty string ``""`` (the tskit convention for "not annotated")
        in place of the MAP allele. The posterior block (under
        ``site.metadata['ancestree']``) is still written with the full
        sub-threshold distribution. ``None`` (default) disables the check.
    :raises ImportError: If ``tskit`` is not installed.
    """

    @staticmethod
    def _tskit_provenance_row(provenance) -> dict:
        """The run record wrapped in tskit's provenance schema.

        :param provenance: The run record, as
            :meth:`Inference.provenance() <ancestree.inference.Inference.provenance>`
            emitted it.
        :return: A ``dict`` carrying ``schema_version``, ``software`` as an
            object, ``environment`` and the record under ``parameters``.
        """
        import platform

        import tskit

        return {
            "schema_version": "1.0.0",
            "software": {
                "name": str(provenance.get("software", "ancestree")),
                "version": str(provenance.get("version", "")),
            },
            "parameters": dict(provenance),
            "environment": {
                "os": {
                    "system": platform.system(),
                    "release": platform.release(),
                },
                "python": {"version": platform.python_version()},
                "libraries": {"tskit": {"version": tskit.__version__}},
            },
        }


    #: tskit's own "not annotated" marker, in place of the VCF ``.``.
    AA_UNKNOWN = ""

    def _repolarise(self, tables, repolarised: dict) -> None:
        """Rebuild the mutations of every site whose ancestral state changed.

        A parsimonious mutation set is re-derived against the new state from
        the genotypes the input tree sequence held, so every sample decodes to
        the allele it carried in the input. This includes the blank state of a
        sub-threshold site.

        :param tables: The table collection being written, with its sites
            rewritten.
        :param repolarised: ``{site_id: new_ancestral_state}`` for the sites
            whose state changed.
        """
        import tskit

        if not repolarised:
            return
        had_times = len(tables.mutations) > 0 and not bool(
            tskit.is_unknown_time(tables.mutations.time).all()
        )
        originals: dict[int, list] = {}
        for mutation in self._ts.mutations():
            originals.setdefault(mutation.site, []).append(mutation)

        variant = tskit.Variant(self._ts)
        tables.mutations.clear()
        for site_id in range(tables.sites.num_rows):
            if site_id not in repolarised:
                for mutation in originals.get(site_id, ()):
                    tables.mutations.add_row(
                        site=site_id, node=mutation.node,
                        derived_state=mutation.derived_state,
                        time=mutation.time, metadata=mutation.metadata,
                    )
                continue
            variant.decode(site_id)
            state = repolarised[site_id]
            alleles = tuple(variant.alleles)
            if state not in alleles:
                alleles += (state,)
            if not (variant.genotypes >= 0).any():
                # No observed genotype to place a mutation from. The site keeps
                # its new ancestral state and carries no mutations.
                continue
            position = tables.sites.position[site_id]
            tree = self._ts.at(position)
            _, mutations = tree.map_mutations(
                variant.genotypes, alleles, ancestral_state=state,
            )
            for mutation in mutations:
                tables.mutations.add_row(
                    site=site_id, node=mutation.node,
                    derived_state=mutation.derived_state,
                    time=tskit.UNKNOWN_TIME,
                )
        tables.sort()
        # tskit requires mutation times all known or all unknown, so every
        # mutation time is recomputed from the topology.
        tables.build_index()
        if had_times:
            tables.compute_mutation_times()
            tables.sort()
            tables.build_index()
        tables.compute_mutation_parents()

    @staticmethod
    def _coerce_metadata_to_dict(meta: object) -> dict:
        """Coerce arbitrary tskit ``site.metadata`` to a plain ``dict``.

        Empty / null / unparseable byte payloads collapse to ``{}``.
        Existing dicts are shallow-copied. Valid JSON bytes are parsed.
        Anything else (e.g. raw bytes from a non-JSON schema) collapses to
        ``{}``.
        """
        if isinstance(meta, dict):
            return dict(meta)
        if isinstance(meta, (bytes, bytearray)):
            if not meta:
                return {}
            try:
                decoded = json.loads(bytes(meta))
            except (TypeError, ValueError):
                return {}
            return decoded if isinstance(decoded, dict) else {}
        return {}

    def __init__(
        self,
        input_ts: "tskit.TreeSequence | str | os.PathLike",
        output_path: str | os.PathLike,
        *,
        min_confidence: float | None = None,
    ) -> None:
        """Resolve ``input_ts`` to a loaded :class:`tskit.TreeSequence`. Defer
        all output I/O to :meth:`write`."""
        import tskit
        if isinstance(input_ts, (str, os.PathLike)):
            self._ts = tskit.load(str(input_ts))
        else:
            self._ts = input_ts
        self._init_output(output_path, min_confidence)

    def write(
        self,
        posteriors: Iterable[tuple[Site, Posterior]],
        *,
        store_posterior: bool = True,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> int:
        """Rewrite the sites table with annotated ancestral states and dump.

        :param posteriors: Iterable of ``(Site, Posterior)`` pairs. Sites are
            matched to the template's sites by exact float position
            (:attr:`Site.local_tree_handle <ancestree.sites.Site.local_tree_handle>`,
            set by the ARG / local-tree sources),
            falling back to ``int(site.pos)`` when no handle is present. Exact
            float matching avoids collapsing distinct sites that share an
            integer position on a continuous genome.
        :param store_posterior: When ``True`` (default), store the full
            posterior dict in ``site.metadata`` under ``"ancestree"``.
            When ``False``, store only
            :attr:`Posterior.map_allele <ancestree.posterior.Posterior.map_allele>`
            and
            :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`.
        :param info: Optional run-level constants (e.g.
            ``{"prior": "adaptive", "model": "K2"}``) stored once per
            annotated site under ``site.metadata["ancestree"]["inference"]``.
        :param provenance: Optional structured run record (version, mode,
            parameters). Appended once as a row to the tskit ``provenances``
            table (the canonical tskit provenance location), not per site.
        :return: Number of sites annotated.
        """
        import tskit

        info_dict = dict(info) if info else None

        index_by_fpos: dict[float, tuple[Site, Posterior]] = {}
        index_by_ipos: dict[int, tuple[Site, Posterior]] = {}
        n_offered = 0
        for site, posterior in posteriors:
            n_offered += 1
            handle = site.local_tree_handle
            if handle is not None:
                index_by_fpos[float(handle)] = (site, posterior)
            else:
                ipos = int(site.pos)
                if ipos in index_by_ipos:
                    other = index_by_ipos[ipos][0]
                    remedy = ("Restrict the run to one contig"
                              if str(other.chrom) != str(site.chrom)
                              else "Supply posteriors carrying "
                                   "local_tree_handle")
                    raise ValueError(
                        f"two posteriors without a local-tree handle share the "
                        f"integer position {ipos} ({other.chrom}:{other.pos} "
                        f"and {site.chrom}:{site.pos}); a tskit site is matched "
                        f"on that integer, so one call would be dropped. "
                        f"{remedy}.")
                index_by_ipos[ipos] = (site, posterior)

        if n_offered == 0 and self._ts.num_sites:
            self._warn_no_posteriors(
                int(self._ts.num_sites), "the input tree sequence")

        tables = self._ts.dump_tables()
        sites_table = tables.sites

        if provenance:
            timestamp = str(provenance.get("timestamp", ""))
            tables.provenances.add_row(
                timestamp=timestamp,
                record=json.dumps(
                    self._tskit_provenance_row(provenance), default=str),
            )

        # Metadata must be dict-shaped for the ancestree block to merge in.
        if sites_table.metadata_schema.schema is None:
            sites_table.metadata_schema = tskit.MetadataSchema.permissive_json()

        new_rows: list[tuple[float, str, dict]] = []
        repolarised: dict[int, str] = {}
        n_annotated = 0
        for site_obj in self._ts.sites():
            scored = (
                index_by_fpos.get(float(site_obj.position))
                or index_by_ipos.get(int(site_obj.position)))
            base_meta = self._coerce_metadata_to_dict(site_obj.metadata)
            if scored is None:
                # An unscored site carries no ancestree block.
                base_meta.pop("ancestree", None)
                new_rows.append((site_obj.position, site_obj.ancestral_state, base_meta))
                continue

            site, site_posterior = scored
            state_to_write, max_prob = self._call(site, site_posterior)
            map_a = str(site_posterior.map_allele)
            extra: dict = {
                "map_allele": map_a,
                "max_prob": max_prob,
            }
            if store_posterior:
                extra["posterior"] = site_posterior.to_dict()
            if info_dict is not None:
                extra["inference"] = info_dict
            base_meta["ancestree"] = extra
            if state_to_write != site_obj.ancestral_state:
                repolarised[site_obj.id] = state_to_write
            new_rows.append((site_obj.position, state_to_write, base_meta))
            n_annotated += 1

        sites_table.clear()
        for pos, aa, meta in new_rows:
            sites_table.add_row(position=pos, ancestral_state=aa, metadata=meta)
        self._repolarise(tables, repolarised)

        new_ts = tables.tree_sequence()
        staged = self._staged_path(self._output)
        if n_annotated == 0 and n_offered:
            # An unannotated tree sequence still carries every site's original
            # ancestral_state, so it reads as a successful annotation to
            # anything that consumes that field.
            self._refuse_no_matches(
                n_offered, "the tree sequence's sites",
                "Check that the positions of the posteriors and of the tree "
                "sequence agree.")
        try:
            new_ts.dump(staged)
        except BaseException:
            if os.path.exists(staged):
                os.unlink(staged)
            raise
        self._finish_staged(staged, self._output)
        self._log.info("Wrote %s annotated sites to %s",
                       f"{n_annotated:,}", self._output)
        return n_annotated


class ZarrWriter(Writer):
    """Copy a VCF Zarr (VCZ) store and add the ``variant_AA*`` arrays.

    Follows the `VCF Zarr specification
    <https://github.com/sgkit-dev/vcf-zarr-spec>`_. The template store is
    copied to ``output_zarr`` and variant-indexed arrays are added at the root:
    ``variant_AA`` (MAP ancestral allele, ``"."`` where unannotated or below
    ``min_confidence``), ``variant_AA_prob`` (``float32`` MAP probability) and
    ``variant_AA_post`` (``(n_variants, 4) float32`` posterior over A, C, G, T).
    Unannotated rows carry the VCF-Zarr float-missing sentinel, which reads as
    ``NaN`` and exports as a missing ``INFO`` value.

    Each array carries ``_ARRAY_DIMENSIONS`` and a ``description`` attribute
    from which an exporter builds the ``##INFO`` line. ``provenance`` and
    ``info`` go in the root group's ``attrs`` under ``ancestree_provenance``
    and ``ancestree_info``. Only directory-backed stores are supported.

    :param input_zarr: Template VCZ store path.
    :param output_zarr: Destination store path, which must differ from
        ``input_zarr``.
    :param min_confidence: Optional minimum
        :attr:`Posterior.max_prob <ancestree.posterior.Posterior.max_prob>`
        to commit a MAP allele. Below it ``variant_AA`` gets ``"."``
        (``AA_prob`` still
        records the sub-threshold value). ``None`` (default) disables it.
    :param samples: Sample columns to write, matched against ``sample_id``.
        Every array with a ``samples`` dimension is subset, which requires
        every array to carry dimension names. ``None`` (default) writes every
        column. Allele counts such as ``variant_AC`` are copied from the
        template.
    :raises ImportError: If ``zarr`` is not installed.
    """

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields shown by :meth:`__repr__`."""
        return {"output": str(self._output)}

    AA_HEADER_LINES = tuple(
        f'##INFO=<ID={field},Number={spec["Number"]},Type={spec["Type"]},'
        f'Description="{spec["Description"]}">'
        for field, spec in AA_INFO_FIELDS.items()
        if field != "AA_post"
    )

    def __init__(
        self,
        input_zarr: str | os.PathLike,
        output_zarr: str | os.PathLike,
        *,
        min_confidence: float | None = None,
        samples: "Iterable[str] | None" = None,
    ) -> None:
        """Validate that the backend is importable. Defer all I/O to :meth:`write`."""
        self._require_backend(
            "zarr", "`pip install zarr` or `conda install -c conda-forge zarr`")
        self._input = str(input_zarr)
        self._init_output(output_zarr, min_confidence, samples)

    @staticmethod
    @contextlib.contextmanager
    def _string_dtypes_allowed():
        """Silence zarr v3's warning on the fixed-length string dtypes VCZ
        mandates."""
        import warnings

        with warnings.catch_warnings():
            try:
                from zarr.errors import UnstableSpecificationWarning
            except ImportError:  # zarr v2
                pass
            else:
                warnings.simplefilter("ignore", UnstableSpecificationWarning)
            yield

    @staticmethod
    def _create_array(root, name, shape, dtype, dims, chunks=None):
        """Create (overwriting) an empty root array tagged with its VCZ
        dimensions.

        Bridges the zarr v2 (``create_dataset``) and v3 (``create_array``)
        group APIs.

        :param dims: Dimension names, one per axis.
        :param chunks: Chunk shape, or ``None`` for zarr's own choice.
        :return: The new array.
        """
        if name in root:
            del root[name]
        if hasattr(root, "create_array"):  # zarr v3
            extra = ({"dimension_names": list(dims)}
                     if root.metadata.zarr_format == 3 else {})
            arr = root.create_array(name, shape=shape, dtype=dtype,
                                    chunks=chunks or "auto", **extra)
        else:  # zarr v2
            import numpy as np

            extra = {}
            if np.dtype(dtype) == object:
                from numcodecs import VLenUTF8
                extra["object_codec"] = VLenUTF8()
            arr = root.create_dataset(name, shape=shape, dtype=dtype,
                                      chunks=chunks, overwrite=True, **extra)
        arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)
        return arr

    @staticmethod
    def _create_variant_array(root, name, data, dims, description=None,
                              variant_chunk=None):
        """Create (overwriting) a root array holding ``data``.

        :param description: Text for the array's ``description`` attribute,
            which is what an exporter reads to build the field's ``##INFO``
            line on the way back out to VCF.
        :param variant_chunk: Chunk length along the variants axis, shared by
            every variant-indexed array in a VCZ store.
        """
        chunks = (int(variant_chunk), *data.shape[1:]) if variant_chunk else None
        with ZarrWriter._string_dtypes_allowed():
            arr = ZarrWriter._create_array(root, name, data.shape, data.dtype,
                                           dims, chunks)
            arr[:] = data
        if description is not None:
            arr.attrs["description"] = str(description)
        return arr

    def _subset_samples(self, store: str) -> None:
        """Keep only the sample columns to write, in every array with a
        ``samples`` dimension.

        :param store: Path of the staged store.
        """
        import numpy as np
        import zarr

        if self._samples is None:
            return
        root = zarr.open(store, mode="r+")
        names = [SiteSource._decode(s) for s in root["sample_id"][:]]
        kept = self._kept_samples(names)
        if len(kept) == len(names):
            return
        keep = set(kept)
        index = np.asarray([i for i, n in enumerate(names) if n in keep])
        dims_of = {
            name: list(root[name].attrs.get("_ARRAY_DIMENSIONS")
                       or getattr(getattr(root[name], "metadata", None),
                                  "dimension_names", None)
                       or ())
            for name in root.array_keys()}
        untagged = sorted(name for name, dims in dims_of.items() if not dims)
        if untagged:
            raise ValueError(
                f"cannot select sample columns in {self._input!r}: "
                f"{len(untagged)} array(s) carry no dimension names "
                f"(_ARRAY_DIMENSIONS or dimension_names), so their samples "
                f"axis is unknown: {untagged[:5]}")
        with self._string_dtypes_allowed():
            for name, dims in dims_of.items():
                if "samples" not in dims:
                    continue
                src = root[name]
                axis = dims.index("samples")
                shape = list(src.shape)
                shape[axis] = index.size
                chunks = list(src.chunks)
                chunks[axis] = min(chunks[axis], index.size)
                scratch = f"_subset_{name}"
                dst = self._create_array(root, scratch, tuple(shape),
                                         src.dtype, dims, tuple(chunks))
                dst.attrs.update({k: v for k, v in src.attrs.items()
                                  if k != "_ARRAY_DIMENSIONS"})
                step = max(int(src.chunks[0] if axis else src.shape[0]), 1)
                for lo in range(0, src.shape[0], step):
                    dst[lo:lo + step] = np.take(src[lo:lo + step], index,
                                                axis=axis)
                del root[name]
                os.rename(os.path.join(store, scratch),
                          os.path.join(store, name))
        header = root.attrs.get("vcf_header")
        if isinstance(header, str):
            lines = header.split("\n")
            for i, line in enumerate(lines):
                fields = line.split("\t")
                if line.startswith("#CHROM") and len(fields) > 9:
                    lines[i] = "\t".join(
                        fields[:9] + [f for f in fields[9:] if f in keep])
            root.attrs["vcf_header"] = "\n".join(lines)

    @staticmethod
    def _reconsolidate(store: str) -> None:
        """Rewrite consolidated metadata, where the store carries any.

        :param store: Path of the store.
        """
        import zarr

        v3_root = os.path.join(store, "zarr.json")
        consolidated = os.path.exists(os.path.join(store, ".zmetadata"))
        if not consolidated and os.path.exists(v3_root):
            with open(v3_root) as fh:
                consolidated = bool(json.load(fh).get("consolidated_metadata"))
        if consolidated:
            zarr.consolidate_metadata(store)

    def write(
        self,
        posteriors: Iterable[tuple[Site, Posterior]],
        *,
        store_posterior: bool = True,
        info: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
    ) -> int:
        """Copy the template store and add the ``AA`` / ``AA_prob`` arrays.

        :param posteriors: Iterable of ``(Site, Posterior)`` pairs. Streamed and
            placed into their template rows by ``(chrom, pos)`` without buffering
            (see ``_row_locator()``). Order-independent and subset-tolerant.
        :param store_posterior: When ``True`` (default), also write the
            ``variant_AA_post`` array holding the whole per-state posterior.
        :param info: Optional run-level constants stored in the root group's
            ``attrs`` under ``ancestree_info`` (run-level, not per variant).
        :param provenance: Optional structured run record stored in the root
            group's ``attrs`` under ``ancestree_provenance``.
        :return: Number of variants annotated.
        """
        import shutil
        import tempfile

        import zarr

        self._refuse_template_overwrite()
        # The build is staged in a sibling temp store and swapped into place
        # once every array is written and the empty-match refusal has passed.
        parent = os.path.dirname(os.path.abspath(self._output)) or "."
        staging = tempfile.mkdtemp(prefix=".ancestree_vcz_", dir=parent)
        work_dir = os.path.join(staging, "store")
        shutil.copytree(self._input, work_dir)

        try:
            self._subset_samples(work_dir)
            root = zarr.open(work_dir, mode="r+")
            n_annotated, n_seen = self._annotate_store(
                root, posteriors, info, provenance, store_posterior,
            )
            if n_annotated == 0 and n_seen:
                self._refuse_no_matches(
                    n_seen, f"the variants of {self._input!r}")
            self._reconsolidate(work_dir)
            self._swap_into_place(work_dir, self._output)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        shutil.rmtree(staging, ignore_errors=True)  # empty after the swap

        self._log.info("Wrote %s annotated sites to %s",
                       f"{n_annotated:,}", self._output)
        return n_annotated

    @staticmethod
    def _swap_into_place(src_dir: str, dst_dir: str) -> None:
        """Atomically rename ``src_dir`` onto ``dst_dir`` (same filesystem).

        An existing ``dst_dir`` is renamed aside first and only removed
        once the swap succeeds. A failed swap is rolled back so the original
        store is never lost.
        """
        import shutil

        if os.path.exists(dst_dir):
            backup = f"{dst_dir}.old-{os.getpid()}"
            os.rename(dst_dir, backup)
            try:
                os.rename(src_dir, dst_dir)
            except BaseException:
                os.rename(backup, dst_dir)  # roll back to the original
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.rename(src_dir, dst_dir)

    def _annotate_store(
        self,
        root,
        posteriors: Iterable[tuple[Site, Posterior]],
        info: Mapping[str, object] | None,
        provenance: Mapping[str, object] | None,
        store_posterior: bool = True,
    ) -> tuple[int, int]:
        """Write the ``variant_AA*`` arrays into ``root``.

        :return: ``(n_annotated, n_seen)``, the rows filled and the posteriors
            consumed.
        """
        import numpy as np

        pos = np.asarray(root[ZARR_POSITION_FIELD][:])
        contig = np.asarray(root[ZARR_CONTIG_FIELD][:])
        # Alleles identify which of several records sharing a position a
        # posterior describes. Absent from a store written without them, in
        # which case placement falls back to position order.
        row_alleles = None
        if ZARR_ALLELE_FIELD in root:
            row_alleles = [
                tuple(SiteSource._decode(a).upper() for a in row
                      if SiteSource._decode(a) not in ("", "."))
                for row in np.asarray(root[ZARR_ALLELE_FIELD][:])
            ]
        contig_ids = [SiteSource._decode(c) for c in root["contig_id"][:]]
        n_variants = int(pos.shape[0])
        name_to_cidx = {c: i for i, c in enumerate(contig_ids)}

        aa = np.full(n_variants, self.AA_UNKNOWN, dtype="U1")
        # VCF-Zarr float32 missing sentinel (htslib bcf_float_missing), which
        # exports as a missing INFO value and is a NaN bit pattern.
        aa_prob_missing = np.array([0x7F800001], dtype=np.uint32).view("f4")[0]
        prob = np.full(n_variants, aa_prob_missing, dtype="f4")
        post = (np.full((n_variants, len(STATES)), aa_prob_missing, dtype="f4")
                if store_posterior else None)

        # Place each posterior into its template row as the stream arrives.
        locate = self._row_locator(pos, contig, contig_ids, name_to_cidx)

        annotated = np.zeros(n_variants, dtype=bool)
        n_seen, n_annotated = self._place_posteriors(
            posteriors, locate, pos, contig, row_alleles,
            annotated, aa, prob, post,
        )
        if n_seen == 0 and n_variants:
            self._warn_no_posteriors(n_variants, f"the store {self._input!r}")

        # Follow the template's chunking along the variants axis.
        variant_chunk = getattr(
            root[ZARR_POSITION_FIELD], "chunks", (None,))[0]
        self._create_variant_array(
            root, ZARR_AA_FIELD, aa, ["variants"],
            description=AA_INFO_FIELDS["AA"]["Description"],
            variant_chunk=variant_chunk)
        self._create_variant_array(
            root, ZARR_AA_PROB_FIELD, prob, ["variants"],
            description=AA_INFO_FIELDS["AA_prob"]["Description"],
            variant_chunk=variant_chunk)
        # vcztools reads the VCF Number off the INFO_<key>_dim dimension.
        if post is not None:
            self._create_variant_array(
                root, ZARR_AA_POST_FIELD, post, ["variants", "INFO_AA_post_dim"],
                description=AA_INFO_FIELDS["AA_post"]["Description"],
                variant_chunk=variant_chunk,
            )
        elif ZARR_AA_POST_FIELD in root:
            del root[ZARR_AA_POST_FIELD]

        # A run passing no info clears an earlier run's block.
        if info:
            root.attrs["ancestree_info"] = json.loads(
                json.dumps(dict(info), default=str)
            )
        elif "ancestree_info" in root.attrs:
            del root.attrs["ancestree_info"]
        if provenance:
            root.attrs[PROVENANCE_HEADER] = json.loads(
                json.dumps(provenance, default=str)
            )
        # An exporter builds ##INFO lines from the arrays' description attributes.
        header = root.attrs.get("vcf_header")
        if isinstance(header, str) and "ID=AA," not in header:
            addition = "\n".join(self.AA_HEADER_LINES) + "\n"
            root.attrs["vcf_header"] = header.replace(
                "#CHROM", addition + "#CHROM", 1,
            ) if "#CHROM" in header else header + addition

        return n_annotated, n_seen
