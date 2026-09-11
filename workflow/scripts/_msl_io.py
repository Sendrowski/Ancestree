"""Shared IO helpers for the MSL chr1 benchmark scripts.

Shared by ``fit_infer_msl_vcf{,_streaming}.py``, ``infer_msl_local_tree.py``
and the ``report_msl_*`` scripts:

- :func:`sites_from_arrays` -- build :class:`~ancestree.sites.Site` records
  from the compact polymorphic-npz arrays (ingroup +/- outgroup, optional
  position offset);
- :class:`MslPolymorphicNpzSource` -- a re-iterable
  :class:`~ancestree.sites.SiteSource` over a list of per-region npz chunks
  (for the streaming whole-chromosome fit);
- :func:`load_pos_idx` / :func:`load_estsfs` -- parse the ``pos<TAB>idx``
  ancestral-call files the three Ancestree modes, PolarBEAR, and EST-SFS emit.

Imported via the same ``sys.path``-insert pattern the benchmark scripts use
for ``_msl_arg_kernel``.
"""
from __future__ import annotations

import numpy as np

from ancestree import STATES, STATE_INDEX, Site
from ancestree.sites import SiteSource


def sites_from_arrays(
    pos, ref_idx, alt_idx, ingroup, ingroup_names,
    outgroup=None, outgroup_names=None, *, chrom="1", pos_offset=0,
):
    """Build :class:`Site` records from the compact polymorphic-npz arrays.

    :param pos: ``(n_sites,)`` positions.
    :param ref_idx: ``(n_sites,)`` reference nucleotide indices into
        :data:`~ancestree.STATES`.
    :param alt_idx: ``(n_sites, n_alt)`` alt-allele indices (``-1`` padding).
    :param ingroup: ``(n_sites, n_ingroup)`` ingroup allele indices
        (``-1`` = missing).
    :param ingroup_names: Ingroup haplotype ids.
    :param outgroup: Optional ``(n_sites, n_out)`` outgroup allele indices.
        Omit (with ``outgroup_names``) for ingroup-only Sites.
    :param outgroup_names: Outgroup ids, paired with ``outgroup``.
    :param chrom: Chromosome label for the Sites.
    :param pos_offset: Added to every position (use ``-chunk_start`` for
        chunk-relative coordinates, as the local-tree windower needs).
    :return: ``list[Site]``.
    """
    nuc = STATES
    use_out = outgroup is not None and outgroup_names is not None
    out: list[Site] = []
    for k in range(len(pos)):
        ref = nuc[int(ref_idx[k])]
        alt_row = [nuc[int(a)] for a in alt_idx[k] if a >= 0]
        alleles = (ref, *alt_row)
        tip_alleles: dict[str, str | None] = {}
        ing_row = ingroup[k]
        for i, name in enumerate(ingroup_names):
            a = int(ing_row[i])
            tip_alleles[name] = nuc[a] if a >= 0 else None
        if use_out:
            og_row = outgroup[k]
            for j, name in enumerate(outgroup_names):
                a = int(og_row[j])
                tip_alleles[name] = nuc[a] if a >= 0 else None
        out.append(Site(
            chrom=chrom, pos=int(pos[k]) + pos_offset,
            alleles=alleles, tip_alleles=tip_alleles,
        ))
    return out


class MslPolymorphicNpzSource(SiteSource):
    """Re-iterable :class:`SiteSource` over the per-region polymorphic npz chunks.

    Each ``__iter__`` re-opens every chunk in genomic order and yields the
    polymorphic :class:`Site` records lazily (ingroup + outgroup), so a
    consuming streaming fit never holds more than one chunk's records at a
    time. Sample order is taken from the first chunk (identical across chunks
    by construction).

    :param npz_paths: Per-region ``msl_polymorphic_*.npz`` paths, genomic order.
    """

    def __init__(self, npz_paths) -> None:
        self._paths = list(npz_paths)
        if not self._paths:
            raise ValueError("no polymorphic npz chunks supplied")
        z0 = np.load(self._paths[0], allow_pickle=True)
        self.ingroup_names = list(z0["ingroup_names"].astype(str))
        self.outgroup_names = list(z0["outgroup_names"].astype(str))

    def samples(self) -> list[str]:
        return self.ingroup_names + self.outgroup_names

    def __iter__(self):
        for path in self._paths:
            z = np.load(path, allow_pickle=True)
            yield from sites_from_arrays(
                z["pos"], z["ref_idx"], z["alt_idx"],
                z["ingroup"], self.ingroup_names,
                z["outgroup"], self.outgroup_names,
            )


def load_pos_idx(path: str) -> dict[int, int]:
    """Parse a ``pos<TAB>allele_idx`` ancestral-call file into ``{pos: idx}``."""
    out: dict[int, int] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                pos, idx = int(p[0]), int(p[1])
            except ValueError:
                continue
            if idx < 0:
                continue
            out[pos] = idx
    return out


def load_estsfs(path: str) -> dict[int, int]:
    """Parse EST-SFS ``est-sfs_ancstate.txt`` (``chrom pos idx_or_letter``)."""
    out: dict[int, int] = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            try:
                pos = int(p[1])
            except ValueError:
                continue
            tok = p[2]
            try:
                idx = int(tok)
            except ValueError:
                idx = STATE_INDEX.get(tok.upper(), -1)
            if idx < 0:
                continue
            out[pos] = idx
    return out
