"""Tests for :class:`~ancestree.writers.ZarrWriter`.

Builds a small VCF Zarr (VCZ) store by hand (the ``bio2zarr`` layout that
:class:`~ancestree.sources.VcfZarrSource` reads), runs the writer over a
hand-made posterior stream, and checks the added ``variant_AA`` /
``variant_AA_prob`` arrays, the provenance / info root attrs, and the
embedded-header update.
"""
from __future__ import annotations

import numpy as np
import pytest

from ancestree.posterior import Posterior
from ancestree.sites import Site
from ancestree.writers import ZarrWriter
from testing._helpers import post

import zarr

# Fixed-length string arrays (the VCZ layout) draw a zarr-v3
# UnstableSpecificationWarning that the format mandates and so cannot act on;
# silence it for this module (the category resolves since zarr is imported).
pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.errors.UnstableSpecificationWarning"
)


def _create(root, name, data):
    """Create a root array, bridging the zarr v2 / v3 group APIs."""
    if hasattr(root, "create_array"):  # zarr v3
        arr = root.create_array(name, shape=data.shape, dtype=data.dtype)
        arr[:] = data
    else:  # zarr v2
        arr = root.create_dataset(name, data=data, overwrite=True)
    return arr


@pytest.fixture
def vcz_store(tmp_path):
    """A 3-variant, single-contig VCZ store with an embedded VCF header."""
    path = tmp_path / "in.vcz"
    root = zarr.open(str(path), mode="w")
    _create(root, "variant_position", np.array([100, 200, 300], dtype="i8"))
    _create(root, "variant_contig", np.array([0, 0, 0], dtype="i4"))
    _create(root, "contig_id", np.array(["1"], dtype="U1"))
    root.attrs["vcf_header"] = (
        "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    return str(path)


def _site(pos: int) -> Site:
    return Site(chrom="1", pos=pos, alleles=("A", "T"), tip_alleles={})


class TestZarrWriter:
    def test_adds_aa_arrays_aligned_to_variants(self, vcz_store, tmp_path):
        out = str(tmp_path / "out.vcz")
        pairs = [(_site(100), post("A", 0.9)), (_site(300), post("G", 0.8))]
        n = ZarrWriter(vcz_store, out).write(iter(pairs))
        assert n == 2
        r = zarr.open(out, mode="r")
        # pos 200 has no posterior → "." sentinel / NaN.
        assert [str(x) for x in r["variant_AA"][:]] == ["A", ".", "G"]
        prob = r["variant_AA_prob"][:]
        assert prob[0] == pytest.approx(0.9)
        assert np.isnan(prob[1])
        assert prob[2] == pytest.approx(0.8)

    def test_array_dimensions_attr(self, vcz_store, tmp_path):
        out = str(tmp_path / "out.vcz")
        ZarrWriter(vcz_store, out).write(iter([(_site(100), post("A", 0.9))]))
        r = zarr.open(out, mode="r")
        assert list(r["variant_AA"].attrs["_ARRAY_DIMENSIONS"]) == ["variants"]
        assert list(r["variant_AA_prob"].attrs["_ARRAY_DIMENSIONS"]) == ["variants"]

    def test_min_confidence_blanks_aa_keeps_prob(self, vcz_store, tmp_path):
        out = str(tmp_path / "out.vcz")
        pairs = [(_site(100), post("A", 0.4))]  # below threshold
        ZarrWriter(vcz_store, out, min_confidence=0.5).write(iter(pairs))
        r = zarr.open(out, mode="r")
        assert str(r["variant_AA"][0]) == "."
        assert r["variant_AA_prob"][0] == pytest.approx(0.4)

    def test_provenance_and_info_in_root_attrs(self, vcz_store, tmp_path):
        out = str(tmp_path / "out.vcz")
        prov = {"software": "ancestree", "version": "x", "mode": "fixed-tree"}
        ZarrWriter(vcz_store, out).write(
            iter([(_site(100), post("A", 0.9))]),
            info={"prior": "adaptive"}, provenance=prov,
        )
        r = zarr.open(out, mode="r")
        assert dict(r.attrs["ancestree_provenance"])["mode"] == "fixed-tree"
        assert dict(r.attrs["ancestree_info"]) == {"prior": "adaptive"}

    def test_embedded_header_gains_aa_info_lines(self, vcz_store, tmp_path):
        out = str(tmp_path / "out.vcz")
        ZarrWriter(vcz_store, out).write(iter([(_site(100), post("A", 0.9))]))
        r = zarr.open(out, mode="r")
        header = r.attrs["vcf_header"]
        assert "ID=AA," in header and "ID=AA_prob," in header
        # INFO lines precede the #CHROM column line.
        assert header.index("ID=AA,") < header.index("#CHROM")

    def test_in_place_annotation(self, vcz_store):
        # output == input annotates the store in place (no copy).
        n = ZarrWriter(vcz_store, vcz_store).write(
            iter([(_site(200), post("T", 0.7))])
        )
        assert n == 1
        r = zarr.open(vcz_store, mode="r")
        assert str(r["variant_AA"][1]) == "T"

    def test_multi_contig_alignment(self, tmp_path):
        # Same pos on different contigs must not cross-annotate. The searchsorted
        # locator keeps each contig's row block separate.
        path = str(tmp_path / "multi.vcz")
        root = zarr.open(path, mode="w")
        _create(root, "variant_position", np.array([100, 200, 100, 200], dtype="i8"))
        _create(root, "variant_contig", np.array([0, 0, 1, 1], dtype="i4"))
        _create(root, "contig_id", np.array(["1", "2"], dtype="U1"))
        out = str(tmp_path / "out.vcz")
        pairs = [
            (Site(chrom="2", pos=100, alleles=("A", "T"), tip_alleles={}), post("G", 0.9)),
            (Site(chrom="1", pos=200, alleles=("A", "T"), tip_alleles={}), post("A", 0.8)),
        ]
        n = ZarrWriter(path, out).write(iter(pairs))
        assert n == 2
        r = zarr.open(out, mode="r")
        # rows: [c1:100, c1:200, c2:100, c2:200] -> annotate c1:200 and c2:100 only.
        assert [str(x) for x in r["variant_AA"][:]] == [".", "A", "G", "."]

    def test_unsorted_store_uses_dict_fallback(self, tmp_path):
        # A store not sorted by (contig, pos) cannot be binary-searched. The
        # locator falls back to an exact (chrom, pos) -> row dict.
        path = str(tmp_path / "uns.vcz")
        root = zarr.open(path, mode="w")
        _create(root, "variant_position", np.array([300, 100, 200], dtype="i8"))
        _create(root, "variant_contig", np.array([0, 0, 0], dtype="i4"))
        _create(root, "contig_id", np.array(["1"], dtype="U1"))
        out = str(tmp_path / "out.vcz")
        pairs = [
            (_site(100), post("A", 0.9)),
            (_site(300), post("T", 0.8)),
        ]
        ZarrWriter(path, out).write(iter(pairs))
        r = zarr.open(out, mode="r")
        # rows follow the store order [300, 100, 200] -> [T, A, "."].
        assert [str(x) for x in r["variant_AA"][:]] == ["T", "A", "."]

    def test_info_numpy_value_sanitized(self, vcz_store, tmp_path):
        # A numpy-typed info value is JSON-sanitized, not left to crash zarr's
        # attr encoder mid-write.
        out = str(tmp_path / "out.vcz")
        ZarrWriter(vcz_store, out).write(
            iter([(_site(100), post("A", 0.9))]),
            info={"kappa": np.float32(2.13)},
        )
        r = zarr.open(out, mode="r")
        assert "kappa" in dict(r.attrs["ancestree_info"])


def test_annotation_arrays_carry_their_descriptions(tmp_path):
    """Each annotation array must describe itself for the trip back to VCF.

    An exporter builds a field's ``##INFO`` line from the array's own
    ``description`` attribute. The writer instead looked for whole header text
    under a key ``vcf2zarr`` does not write, so the branch never ran and the
    fields exported with empty descriptions, leaving a reader no way to tell
    what ``AA_prob`` or ``AA_post`` held.
    """
    import numpy as np
    import zarr

    from testing._helpers import write_vcz

    src = str(tmp_path / "src.vcz")
    out = str(tmp_path / "out.vcz")
    write_vcz(
        src, positions=[100, 200], contigs_per_variant=[0, 0], contig_ids=["1"],
        alleles=[["A", "C"], ["T", "G"]], sample_ids=["s1"],
        genotypes=np.zeros((2, 1, 2), dtype=np.int8),
    )
    pairs = [
        (Site(chrom="1", pos=pos, alleles=al, tip_alleles={"s1_h0": al[0]}),
         Posterior(alleles=("A", "C", "G", "T"),
                   values=np.array([0.9, 0.1, 0.0, 0.0])))
        for pos, al in ((100, ("A", "C")), (200, ("T", "G")))
    ]
    assert ZarrWriter(src, out).write(pairs) == 2

    root = zarr.open(out, mode="r")
    for name in ("variant_AA", "variant_AA_prob", "variant_AA_post"):
        description = root[name].attrs.get("description")
        assert description, f"{name} carries no description"
        assert "ancestree" in description


def test_added_arrays_follow_the_template_chunking(tmp_path):
    """Every variant-indexed array in a VCZ store must share its chunking.

    The arrays were created without a ``chunks=`` argument, so zarr chose one
    spanning every variant while the template kept its own. The VCF Zarr
    specification requires them to agree and ``vcztools`` refuses a store where
    they do not, so every output above the template's chunk length was
    unreadable by the reference tool for the format. Ancestree's own reader
    accepted it, and the interop fixture sits below the threshold where the
    two can differ, so nothing caught it.
    """
    import numpy as np
    import zarr

    from testing._helpers import write_vcz

    src, out = str(tmp_path / "src.vcz"), str(tmp_path / "out.vcz")
    positions = [100 * (i + 1) for i in range(6)]
    write_vcz(
        src, positions=positions, contigs_per_variant=[0] * 6, contig_ids=["1"],
        alleles=[["A", "C"]] * 6, sample_ids=["s1"],
        genotypes=np.zeros((6, 1, 2), dtype=np.int8),
    )
    # Re-chunk the template's position array so its chunk length is shorter
    # than the variant count, as bio2zarr's 1000-variant default is on any
    # real dataset.
    root = zarr.open(src, mode="r+")
    stored = root["variant_position"][:]
    del root["variant_position"]
    arr = root.create_array("variant_position", shape=stored.shape,
                            dtype=stored.dtype, chunks=(2,))
    arr[:] = stored
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    assert root["variant_position"].chunks == (2,)

    pairs = [
        (Site(chrom="1", pos=p, alleles=("A", "C"), tip_alleles={"s1_h0": "A"}),
         Posterior(alleles=("A", "C", "G", "T"),
                   values=np.array([0.9, 0.1, 0.0, 0.0])))
        for p in positions
    ]
    assert ZarrWriter(src, out).write(pairs) == 6

    written = zarr.open(out, mode="r")
    expected = written["variant_position"].chunks[0]
    for name in ("variant_AA", "variant_AA_prob", "variant_AA_post"):
        assert written[name].chunks[0] == expected, name


def test_a_store_without_probabilities_reads_back(tmp_path):
    """Alleles with no probability array must read, as the same data does in VCF.

    A store carrying the VCF specification's own ``AA`` field and nothing else
    has no probability array. Substituting ``None`` for the missing values left
    the conversion below to raise ``TypeError`` on them, so the earlier
    ``KeyError`` simply became a different exception.
    """
    import numpy as np
    import zarr

    from ancestree.readers import Reader
    from testing._helpers import write_vcz

    path = str(tmp_path / "aa_only.vcz")
    write_vcz(
        path, positions=[100, 200], contigs_per_variant=[0, 0], contig_ids=["1"],
        alleles=[["A", "C"], ["T", "G"]], sample_ids=["s1"],
        genotypes=np.zeros((2, 1, 2), dtype=np.int8),
    )
    root = zarr.open(path, mode="r+")
    arr = root.create_array("variant_AA", shape=(2,), dtype="<U1")
    arr[:] = np.asarray(["A", "T"], dtype="U1")
    arr.attrs["_ARRAY_DIMENSIONS"] = ["variants"]

    rows = list(Reader(path).annotations())
    assert [r.aa for r in rows] == ["A", "T"]
    assert [r.aa_prob for r in rows] == [None, None]


@pytest.fixture
def split_multiallelic_vcz(tmp_path):
    """A store whose rows 1 and 2 are one multiallelic record split in two."""
    path = tmp_path / "split.vcz"
    root = zarr.open(str(path), mode="w")
    _create(root, "variant_position", np.array([10, 20, 20, 30], dtype="i8"))
    _create(root, "variant_contig", np.array([0, 0, 0, 0], dtype="i4"))
    _create(root, "contig_id", np.array(["1"], dtype="U1"))
    return str(path)


class TestSplitMultiallelicRowsTakeTheirOwnCall:
    """Each row of a split-multiallelic record carries its own site's call.

    Every source emits one Site per record, so the rows sharing a position are
    distinct sites and one site's call must not be broadcast across the run.
    """

    def test_each_site_claims_one_row(self, split_multiallelic_vcz, tmp_path):
        out = str(tmp_path / "out.vcz")
        first = Site(chrom="1", pos=20, alleles=("A", "T"), tip_alleles={})
        second = Site(chrom="1", pos=20, alleles=("A", "G"), tip_alleles={})
        n = ZarrWriter(split_multiallelic_vcz, out).write(
            iter([(first, post("A", 0.7)), (second, post("G", 0.6))])
        )
        root = zarr.open(out, mode="r")
        assert list(root["variant_AA"][:]) == [".", "A", "G", "."]
        probabilities = root["variant_AA_prob"][:]
        assert probabilities[1] == pytest.approx(0.7)
        assert probabilities[2] == pytest.approx(0.6)
        assert n == 2

    def test_one_site_does_not_claim_the_whole_run(
        self, split_multiallelic_vcz, tmp_path,
    ):
        out = str(tmp_path / "out.vcz")
        site = Site(chrom="1", pos=20, alleles=("A", "T"), tip_alleles={})
        n = ZarrWriter(split_multiallelic_vcz, out).write(
            iter([(site, post("A", 0.7))])
        )
        root = zarr.open(out, mode="r")
        assert list(root["variant_AA"][:]) == [".", "A", ".", "."]
        assert n == 1

    def test_no_matching_row_refuses(self, split_multiallelic_vcz, tmp_path):
        """A store in which nothing matched is refused, not written.

        A complete, schema-valid store carrying no ancestral allele reads as a
        successful run to everything downstream, and a contig-label mismatch
        between template and posteriors is the usual way to produce one.
        """
        out = str(tmp_path / "out.vcz")
        site = Site(chrom="other", pos=999, alleles=("A", "T"), tip_alleles={})
        with pytest.raises(ValueError, match="matched 0 of"):
            ZarrWriter(split_multiallelic_vcz, out).write(
                iter([(site, post("A", 0.9))])
            )
