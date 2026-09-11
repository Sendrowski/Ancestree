"""Shared test helpers.

Utilities that do not fit the pytest-fixture model, such as functions called
at class-statement scope rather than from inside a test method, and the
fixtures more than one test module imports by name.
"""
from __future__ import annotations

import glob
import gzip
import os
from pathlib import Path

import pytest

from ancestree import BaseComposition
from ancestree.posterior import Posterior
from ancestree.sites import Site


#: Committed msprime panels shared by several test modules: the quickstart
#: and demo ARGs and the VCFs written from them.
PANELS = Path(__file__).resolve().parent / "fixtures" / "panels"
QUICKSTART_TREES = str(PANELS / "quickstart.trees")
QUICKSTART_VCF = str(PANELS / "quickstart.vcf.gz")
DEMO_TREES = str(PANELS / "demo.trees")
DEMO_VCF = str(PANELS / "demo.vcf.gz")


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


def write_skeleton_vcf(path, *, sample_ids, alleles_per_site, genotypes):
    """Write a plain-text VCF mirroring the same variant rows as a VCZ store.

    Just enough header + records to serve as a cyvcf2 template for
    :class:`~ancestree.writers.VCFWriter` (which iterates the template
    and annotates each record's ``INFO`` with ``AA``).

    :param path: Destination path.
    :param sample_ids: Sample column names.
    :param alleles_per_site: Per-site ``(ref, *alts)`` allele lists.
    :param genotypes: ``(n_sites, n_samples, 1)`` int allele indices.
    """
    n_variants, n_samples, _ = genotypes.shape
    lines = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1>",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_ids),
    ]
    for i in range(n_variants):
        ref, *alts = alleles_per_site[i]
        gts = "\t".join(str(int(genotypes[i, j, 0])) for j in range(n_samples))
        lines.append(
            f"chr1\t{i + 1}\t.\t{ref}\t{','.join(alts)}\t.\tPASS\t.\tGT\t{gts}"
        )
    path.write_text("\n".join(lines) + "\n")


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


_LADDER_PAIRS = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")]


@pytest.fixture(scope="module")
def ladder_panel(tmp_path_factory):
    """A four-haplotype panel with mixed transition and transversion sites.

    :return: ``(vcf_path, vcz_path, newick_path, alleles, genotypes)`` with
        samples ``i0, i1, o1, o2``.
    """
    import numpy as np

    rng = np.random.default_rng(5)
    n_sites = 300
    sample_ids = ["i0", "i1", "o1", "o2"]
    alleles = [_LADDER_PAIRS[k] for k in rng.integers(0, len(_LADDER_PAIRS), size=n_sites)]
    gt = rng.integers(0, 2, size=(n_sites, len(sample_ids)), dtype=np.int8)
    d = tmp_path_factory.mktemp("ladder_panel")
    vcf = d / "panel.vcf"
    write_skeleton_vcf(vcf, sample_ids=sample_ids, alleles_per_site=alleles,
                       genotypes=gt[:, :, None])
    vcz = d / "panel.vcz"
    write_vcz(
        vcz, positions=list(range(1, n_sites + 1)),
        contigs_per_variant=[0] * n_sites, contig_ids=["chr1"],
        alleles=[list(a) for a in alleles], sample_ids=sample_ids,
        genotypes=gt[:, :, None],
    )
    nwk = d / "species.nwk"
    nwk.write_text("(((i0:0.05,i1:0.05):0.05,o1:0.10):0.10,o2:0.20);")
    return vcf, vcz, nwk, alleles, gt


def panel_tables() -> "tuple[tskit.TableCollection, dict[str, int]]":
    """Six tips under one root: an ingroup of four over two clades, two outgroups.

    Node times are ``ab=1``, ``cd=1.5``, ``mrca=3``, ``omrca=2`` and
    ``root=6``, with every tip at ``0``. Deliberately non-ultrametric, and
    built as a table collection rather than through ``from_newick``, which
    would force ultrametricity and silently rewrite these branch lengths.

    :return: The tables and ``{name: node}`` for every node.
    """
    import tskit

    tables = tskit.TableCollection(sequence_length=10.0)
    ids: dict[str, int] = {}
    for name in ("a", "b", "c", "d", "o1", "o2"):
        ids[name] = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for name, time in (("ab", 1.0), ("cd", 1.5), ("mrca", 3.0),
                       ("omrca", 2.0), ("root", 6.0)):
        ids[name] = tables.nodes.add_row(flags=0, time=time)
    for parent, child in (
        ("ab", "a"), ("ab", "b"), ("cd", "c"), ("cd", "d"), ("mrca", "ab"),
        ("mrca", "cd"), ("omrca", "o1"), ("omrca", "o2"), ("root", "mrca"),
        ("root", "omrca"),
    ):
        tables.edges.add_row(left=0, right=10.0, parent=ids[parent],
                             child=ids[child])
    tables.sort()
    return tables, ids


def panel_ts() -> "tuple[tskit.TreeSequence, dict[str, int]]":
    """The :func:`panel_tables` genealogy as a tree sequence.

    :return: The tree sequence and ``{name: node}`` for every node.
    """
    tables, ids = panel_tables()
    return tables.tree_sequence(), ids


def panel_site(ids: dict[str, int], pos: int = 1):
    """A site over the panel, keyed by tskit node id, with one G among the ingroup.

    :param ids: The ``{name: node}`` map of :func:`panel_ts`.
    :param pos: The site position.
    :return: The site, both outgroups ``T``.
    """
    from ancestree.sites import Site

    observed = {"a": "A", "b": "A", "c": "G", "d": "A", "o1": "T", "o2": "T"}
    return Site(chrom="1", pos=pos, alleles=("A", "G", "T"),
                tip_alleles={str(ids[k]): v for k, v in observed.items()})


#: Per-generation mutation and recombination rates of the toy local-tree panels.
MU, REC = 1.25e-8, 1e-8


def toy_sites(positions, n_hap=4, seed=0, chrom="1"):
    """Deterministic biallelic sites for ``n_hap`` haplotypes.

    Every site is polymorphic: an all-reference or all-alternate draw has its
    first haplotype flipped.

    :param positions: Site positions, ascending.
    :param n_hap: Panel size.
    :param seed: Genotype seed.
    :param chrom: Contig label.
    :return: ``(sites, names)`` with names ``h0`` to ``h{n_hap - 1}``.
    """
    import numpy as np

    from ancestree.sites import Site

    rng = np.random.default_rng(seed)
    names = [f"h{i}" for i in range(n_hap)]
    out = []
    for p in positions:
        g = rng.integers(0, 2, n_hap)
        if g.sum() == 0:
            g[0] = 1
        elif g.sum() == n_hap:
            g[0] = 0
        out.append(Site(chrom=chrom, pos=int(p), alleles=("A", "C"),
                        tip_alleles={names[i]: ("C" if g[i] else "A")
                                     for i in range(n_hap)}))
    return out, names


def toy_builder(sites, names, length, **kw):
    """A small :class:`~ancestree.local_tree_inference.LocalTreeBuilder`.

    :param sites: The panel's sites.
    :param names: The haplotype names.
    :param length: The sequence length.
    :param kw: Constructor overrides, ``window`` defaulting to 200 bp and
        ``block_size`` to 50 bp.
    :return: The builder.
    """
    from ancestree import LocalTreeBuilder

    kw.setdefault("window", 200)
    kw.setdefault("block_size", 50)
    return LocalTreeBuilder(sites, mu=MU, rec_rate=REC, sample_names=names,
                            sequence_length=length, **kw)


def toy_inference(sites, names, **kw):
    """A small unchunked :class:`~ancestree.local_tree_inference.LocalTreeInference`.

    :param sites: The panel's sites.
    :param names: The haplotype names.
    :param kw: Constructor overrides, ``window`` defaulting to 200 bp,
        ``block_size`` to 50 bp and ``sequence_length`` to 2000 bp.
    :return: The inference under :class:`~ancestree.JC69`.
    """
    from ancestree import JC69, LocalTreeInference

    kw.setdefault("window", 200)
    kw.setdefault("block_size", 50)
    kw.setdefault("chunk_size", None)
    kw.setdefault("sequence_length", 2000.0)
    kw.setdefault("progress", False)
    return LocalTreeInference(sites, JC69(), mu=MU, rec_rate=REC,
                              sample_names=names, **kw)


def toy_chunked_inference(sites, names, **kw):
    """A small chunked :class:`~ancestree.local_tree_inference.LocalTreeInference`.

    :param sites: The panel's sites.
    :param names: The haplotype names.
    :param kw: Constructor overrides, ``window`` defaulting to 200 bp,
        ``block_size`` to 50 bp, ``chunk_size`` to 1000 bp and ``halo`` to
        500 bp.
    :return: The inference under :class:`~ancestree.JC69`.
    """
    from ancestree import JC69, LocalTreeInference

    kw.setdefault("window", 200)
    kw.setdefault("block_size", 50)
    kw.setdefault("chunk_size", 1000)
    kw.setdefault("halo", 500)
    kw.setdefault("progress", False)
    return LocalTreeInference(sites, JC69(), mu=MU, rec_rate=REC,
                              sample_names=names, **kw)


def write_vcf(path, rows, samples=("s1",), extra_header=()):
    """Write a small VCF with one ``GT`` column per sample.

    :param path: Destination, gzipped when the name ends in ``.gz``.
    :param rows: ``(chrom, pos, ref, alt, info, genotypes)`` per record.
    :param samples: Sample names of the header.
    :param extra_header: Further ``##`` lines, such as ``INFO`` declarations.
    :return: The path as a string.
    """
    header = [
        "##fileformat=VCFv4.2",
        "##contig=<ID=1,length=100000>",
        "##contig=<ID=2,length=100000>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        *extra_header,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples),
    ]
    body = [
        f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT\t" + "\t".join(gts)
        for chrom, pos, ref, alt, info, gts in rows
    ]
    text = "\n".join(header + body) + "\n"
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        with open(path, "w") as fh:
            fh.write(text)
    return str(path)


def site_pair(pos, allele="A", p=0.9, chrom="1", handle=None):
    """A ``(Site, Posterior)`` pair at ``chrom:pos`` peaked on ``allele``."""
    site = Site(chrom=chrom, pos=pos, alleles=("A", "C"),
                tip_alleles={"s1_h0": "A"}, local_tree_handle=handle)
    return site, post(allele, p)


def staged_files(directory):
    """The staged partial files left in ``directory``."""
    return glob.glob(os.path.join(str(directory), ".partial-*"))
