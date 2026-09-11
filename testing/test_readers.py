"""Reading annotated output back through :class:`ancestree.readers.Reader`."""
import json

import pytest
import ancestree as anc
from ancestree.readers import Provenance, Reader
from ancestree.sites import PolymorphicSiteFilter
from ancestree.writers import VCFWriter
from testing._helpers import site_pair, write_vcf


@pytest.fixture(scope="module")
def inference(small_ts):
    """An ARG-mode inference over the shared small tree sequence."""
    return anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)


class TestProvenance:
    """:meth:`ancestree.readers.Reader.provenance` per output format."""

    @staticmethod
    def _check(record, mode="arg"):
        """Assert a record carries the fields :meth:`provenance` writes."""
        assert record["software"] == "ancestree"
        assert record["version"] == anc.__version__
        assert record["mode"] == mode
        assert record["parameters"]["model"] == "JC69"
        assert record["timestamp"]

    def test_vcf(self, inference, tmp_path):
        """A VCF carries it on the ``##ancestree_provenance`` header line."""
        out = tmp_path / "annotated.vcf.gz"
        inference.to_vcf(out)
        self._check(anc.Reader(out).provenance())

    def test_trees(self, inference, tmp_path):
        """A tree sequence carries it as a row of the provenance table."""
        out = tmp_path / "annotated.trees"
        inference.to_arg(out)
        self._check(anc.Reader(out).provenance())

    def test_zarr(self, inference, tmp_path):
        """A VCZ store carries it in the root group's ``attrs``."""
        out = tmp_path / "annotated.vcz"
        inference.to_zarr(out)
        self._check(anc.Reader(out).provenance())

    def test_matches_the_written_record(self, inference, tmp_path):
        """What comes back equals what :meth:`provenance` emitted."""
        out = tmp_path / "annotated.vcf.gz"
        written = inference.provenance()
        inference.to_vcf(out, provenance=written)
        assert anc.Reader(out).provenance() == written

    def test_matches_a_record_captured_before_the_run(self, small_ts, tmp_path):
        """A record captured before inference runs still matches the file.

        The focal counters are only published at the end of the walk, so a
        record captured beforehand can miss the keys the file carries. The
        module-scoped fixture hides this, its earlier tests having already
        populated the counters.
        """
        inference = anc.Inference.from_arg(
            small_ts, anc.JC69(), mu=5e-8, progress=False)
        out = tmp_path / "annotated.vcf.gz"
        written = inference.provenance()
        inference.to_vcf(out, provenance=written)
        assert anc.Reader(out).provenance() == written

    def test_raises_without_a_record(self, tmp_path, small_ts):
        """An output written with provenance suppressed has none to read."""
        out = tmp_path / "plain.trees"
        small_ts.dump(out)
        with pytest.raises(ValueError, match="no ancestree provenance"):
            anc.Reader(out).provenance()


class TestProvenanceRepr:
    """The record prints as a readable block rather than a one-line dict."""

    def test_repr_lists_the_run_parameters(self, inference):
        """The header names the software, mode and time. One line per parameter."""
        text = repr(inference.provenance())
        head, *lines = text.split("\n")
        assert head.startswith(f"ancestree {anc.__version__} (arg mode), ")
        assert any(line.strip().startswith("model") for line in lines)
        assert "{" not in text

    def test_is_a_plain_dict(self, inference):
        """It stays a ``dict``, so the writers' ``json.dumps`` keeps working."""
        import json

        record = inference.provenance()
        assert isinstance(record, dict)
        assert json.loads(json.dumps(record, default=str))["mode"] == "arg"


class TestAnnotations:
    """:meth:`ancestree.readers.Reader.annotations` per output format."""

    @staticmethod
    def _first(path, n=3):
        """The first ``n`` annotations of an output."""
        return list(anc.Reader(path).annotations(limit=n))

    def test_formats_agree(self, inference, tmp_path):
        """All three formats report the same calls for the same sites."""
        vcf, trees, vcz = (tmp_path / f"a.{s}" for s in ("vcf.gz", "trees", "vcz"))
        inference.to_vcf(vcf)
        inference.to_arg(trees)
        inference.to_zarr(vcz)
        by_format = [self._first(p) for p in (vcf, trees, vcz)]
        assert all(len(records) == 3 for records in by_format)
        for records in zip(*by_format):
            assert len({r.pos for r in records}) == 1
            assert len({r.aa for r in records}) == 1
            assert max(r.aa_prob for r in records) - min(r.aa_prob for r in records) < 1e-6

    def test_limit_is_lazy(self, inference, tmp_path):
        """``limit`` bounds the read without materialising the whole file."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        assert len(self._first(out, 2)) == 2
        assert len(list(anc.Reader(out).annotations())) > 2

    def test_alleles_and_posterior_come_from_every_format(self, inference, tmp_path):
        """Both the alleles and the full posterior survive each format."""
        vcf, trees = tmp_path / "a.vcf.gz", tmp_path / "a.trees"
        inference.to_vcf(vcf)
        inference.to_arg(trees)
        first_vcf, first_trees = self._first(vcf, 1)[0], self._first(trees, 1)[0]
        assert len(first_vcf.alleles) >= 2 and len(first_trees.alleles) >= 2
        assert set(first_trees.alleles) == set(first_vcf.alleles)
        assert set(first_vcf.posterior) == set(anc.STATES)
        assert set(first_trees.posterior) == set(anc.STATES)

    def test_contig_is_reported_for_every_format(self, inference, tmp_path):
        """A tree sequence takes its contig from the run's provenance record."""
        for suffix in ("vcf.gz", "trees", "vcz"):
            out = tmp_path / f"a.{suffix}"
            getattr(inference, {"vcf.gz": "to_vcf", "trees": "to_arg",
                                "vcz": "to_zarr"}[suffix])(out)
            assert self._first(out, 1)[0].chrom == "1"

    def test_unannotated_sites_come_back_as_none(self, inference, tmp_path):
        """A site below ``min_confidence`` reads back with no call."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out, min_confidence=1.1)
        assert all(a.aa is None for a in self._first(out, 5))

    def test_repr_is_one_readable_line(self, inference, tmp_path):
        """The record prints as ``chrom:pos  alleles  AA=..  p=..``."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        text = repr(self._first(out, 1)[0])
        assert "\n" not in text
        assert "AA=" in text and "p=" in text


class TestReaderDispatch:
    """The format is resolved from the path, and a missing file fails early."""

    @staticmethod
    def _suffix_maps_to(tmp_path, name, expected):
        """Write ``name`` and assert its resolved format."""
        (tmp_path / name).write_text("")
        assert anc.Reader(tmp_path / name).format == expected

    def test_suffixes(self, tmp_path):
        """Each output suffix resolves to its own reader path."""
        self._suffix_maps_to(tmp_path, "a.vcf.gz", "vcf")
        self._suffix_maps_to(tmp_path, "a.bcf", "vcf")
        self._suffix_maps_to(tmp_path, "a.trees", "trees")
        self._suffix_maps_to(tmp_path, "a.vcz", "vcz")

    def test_missing_path_raises(self, tmp_path):
        """A path that does not exist fails at construction, not at read time."""
        with pytest.raises(FileNotFoundError):
            anc.Reader(tmp_path / "absent.vcf.gz")


def test_reader_repr_shows_the_path_and_the_format(tmp_path):
    path = write_vcf(tmp_path / "t.vcf", [])
    assert repr(Reader(path)) == f"Reader(path={path!r}, format='vcf')"


class TestHead:
    """:meth:`ancestree.readers.Reader.head` and its display form."""

    def test_returns_the_first_n(self, inference, tmp_path):
        """It reads exactly ``n`` records, in file order."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        head = anc.Reader(out).head(3)
        assert len(head) == 3
        assert [a.pos for a in head] == [a.pos for a in anc.Reader(out).annotations(limit=3)]

    def test_prints_as_a_table(self, inference, tmp_path):
        """The repr is a header row plus one line per record, not a list."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        header, *rows = repr(anc.Reader(out).head(3)).split("\n")
        assert header.split() == ["chrom", "pos", "alleles", "AA", *anc.STATES]
        assert len(rows) == 3
        assert not header.startswith("[")

    def test_columns_are_aligned(self, inference, tmp_path):
        """Every column starts at the same offset on every row."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        lines = repr(anc.Reader(out).head(5)).split("\n")
        offsets = {tuple(m.start() for m in __import__("re").finditer(r"\S+", line))
                   for line in lines}
        assert len(lines[0]) and all(len(o) == 4 + len(anc.STATES) for o in offsets)

    def test_is_a_plain_list(self, inference, tmp_path):
        """It stays a ``list``, so slicing and indexing work as usual."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        head = anc.Reader(out).head(3)
        assert isinstance(head, list)
        assert head[0].pos < head[-1].pos


class TestMissingData:
    """Reading outputs that carry partial annotations, or none at all."""

    def test_plain_vcf_has_no_calls(self, inference, tmp_path, small_ts):
        """A VCF never annotated reads back with every call empty."""
        out = tmp_path / "plain.vcf.gz"
        with open(tmp_path / "plain.vcf", "w") as fh:
            small_ts.write_vcf(fh)
        import gzip, shutil
        with open(tmp_path / "plain.vcf", "rb") as src, gzip.open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
        head = anc.Reader(out).head(3)
        assert [a.aa for a in head] == [None] * 3
        assert all(a.aa_prob is None for a in head)
        assert "AA" in repr(head)

    def test_plain_tree_sequence_has_no_calls(self, tmp_path, small_ts):
        """A tree sequence with no ancestree metadata reads back empty-called."""
        out = tmp_path / "plain.trees"
        small_ts.dump(out)
        head = anc.Reader(out).head(3)
        assert [a.aa for a in head] == [None] * 3
        assert all(a.posterior is None and a.chrom is None for a in head)

    def test_zarr_without_annotations_raises(self, tmp_path):
        """A VCZ store carrying no ``variant_AA`` is not an ancestree output."""
        import zarr

        path = tmp_path / "plain.vcz"
        root = zarr.open(str(path), mode="w")
        root.create_array("variant_position", shape=(2,), dtype="i8")[:] = [10, 20]
        with pytest.raises(ValueError, match="no ancestree annotations"):
            list(anc.Reader(path).annotations())

    def test_tree_sequence_without_stored_posterior(self, inference, tmp_path):
        """``store_posterior=False`` still yields calls, with no posterior column."""
        out = tmp_path / "a.trees"
        inference.to_arg(out, store_posterior=False)
        head = anc.Reader(out).head(3)
        assert all(a.aa is not None for a in head)
        assert all(a.posterior is None for a in head)
        assert repr(head).split("\n")[0].split() == ["chrom", "pos", "alleles", "AA", "p"]

    def test_head_beyond_the_end(self, inference, tmp_path):
        """Asking for more sites than exist returns what there is."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        total = len(list(anc.Reader(out).annotations()))
        assert len(anc.Reader(out).head(total + 100)) == total

    def test_empty_table_renders(self):
        """An empty result prints as nothing rather than raising."""
        from ancestree.readers import Annotations

        assert repr(Annotations()) == ""

    def test_table_survives_every_field_missing(self):
        """A record with no contig, alleles, call or posterior still renders."""
        from ancestree.readers import Annotation, Annotations

        text = repr(Annotations([Annotation(pos=7, aa=None, aa_prob=None)]))
        assert "7" in text and "." in text


class TestThresholdIsRecorded:
    """A blanked or dotted site is only interpretable next to its cut-off."""

    @pytest.mark.parametrize("suffix", [".vcf.gz", ".vcz", ".trees"])
    def test_min_confidence_reaches_provenance(self, inference, tmp_path, suffix):
        """Every format records the threshold that filtered its calls."""
        out = tmp_path / f"a{suffix}"
        {".vcf.gz": inference.to_vcf, ".vcz": inference.to_zarr,
         ".trees": inference.to_arg}[suffix](out, min_confidence=0.9)
        assert anc.Reader(out).provenance()["parameters"]["min_confidence"] == 0.9

    def test_absent_when_no_threshold_was_set(self, inference, tmp_path):
        """An unfiltered run records no threshold rather than a null one."""
        out = tmp_path / "a.vcf.gz"
        inference.to_vcf(out)
        assert "min_confidence" not in anc.Reader(out).provenance()["parameters"]


def test_an_uncovered_zarr_row_reads_back_without_a_posterior(tmp_path):
    """The float arrays mark absence with NaN, which is not a distribution.

    A row the posterior stream never covered would otherwise read back as
    four NaNs, where the VCF and tree-sequence readers report ``None``, so a
    consumer testing ``posterior is not None`` is misled on one format only.
    """
    import shutil

    import msprime

    import ancestree as anc

    ts = msprime.sim_ancestry(4, ploidy=1, sequence_length=30_000,
                              random_seed=5)
    ts = msprime.sim_mutations(ts, rate=3e-4, random_seed=5)
    inference = anc.ARGBasedInference(ts, mu=1e-8, chrom="1")
    pairs = list(inference.infer())
    assert len(pairs) > 10, "fixture produced too few sites"

    source = str(tmp_path / "src.vcz")
    out = str(tmp_path / "out.vcz")
    shutil.rmtree(source, ignore_errors=True)
    inference.to_zarr(source)
    anc.ZarrWriter(source, out).write(iter(pairs[:5]), store_posterior=True)

    rows = list(anc.Reader(out).annotations())
    uncovered = [r for r in rows if r.aa is None]
    assert uncovered, "no uncovered rows in the fixture"
    assert uncovered[0].posterior is None, (
        f"an uncovered row read back as {uncovered[0].posterior}; the NaN "
        f"missing-marker was taken for a distribution")


class TestProvenanceOfAReAnnotatedOutput:
    """A re-annotated output must report the run that produced its calls.

    Writing to a VCF appends a ``##ancestree_provenance`` header line without
    removing one the template already carries, and htslib keeps both.
    ``Reader.provenance`` must return the newest record from ``.vcf`` as it
    does from ``.trees`` and ``.vcz``.
    """

    @staticmethod
    def _two_runs(small_ts, tmp_path):
        """Annotate once, then re-annotate the result under a different model."""
        first = tmp_path / "first.vcf"
        second = tmp_path / "second.vcf"
        anc.Inference.from_arg(
            small_ts, anc.JC69(), mu=5e-8, progress=False).to_vcf(first)
        anc.Inference.from_arg(
            small_ts, anc.K2(kappa=4.0), mu=5e-8, progress=False
        ).to_vcf(second, input_vcf=first)
        return second

    def test_the_template_s_record_is_still_present(self, small_ts, tmp_path):
        """Guard the test below against passing because only one record exists."""
        second = self._two_runs(small_ts, tmp_path)
        marker = "##ancestree_provenance="
        lines = [ln for ln in second.read_text().splitlines()
                 if ln.startswith(marker)]
        assert len(lines) == 2

    def test_provenance_reports_the_run_that_wrote_the_calls(
            self, small_ts, tmp_path):
        second = self._two_runs(small_ts, tmp_path)
        record = anc.Reader(str(second)).provenance()
        assert record["parameters"]["model"] == "K2"


class TestReader:
    """Provenance forms, the grading of point-mass records and field parsing."""

    def test_a_provenance_row_whose_software_is_a_string_is_read(
            self, small_ts, tmp_path):
        record = {"software": "ancestree", "version": "0.0",
                  "mode": "arg", "parameters": {"chrom": "1"}}
        tables = small_ts.dump_tables()
        tables.provenances.add_row(timestamp="t", record=json.dumps(record))
        path = str(tmp_path / "p.trees")
        tables.tree_sequence().dump(path)
        read = Reader(path).provenance()
        assert isinstance(read, Provenance)
        assert read == record

    def test_grade_refuses_a_polymorphic_site_filter(self, tmp_path):
        path = write_vcf(tmp_path / "t.vcf", [])
        with pytest.raises(ValueError, match="PolymorphicSiteFilter"):
            Reader(path).grade({}, filter=PolymorphicSiteFilter(samples=["s1_h0"]))

    def test_grade_scores_a_point_mass_and_skips_unannotated_sites(self, tmp_path):
        """Without a stored posterior the MAP allele is a point mass.

        Site 100 is assigned ``A`` against a truth of ``A``, site 200 ``A``
        against ``C``, and site 300 is unannotated, so two sites are graded,
        one recovered, with Brier scores of 0 and 2. The file carries no
        provenance record: a position-to-allele truth needs none.
        """
        template = write_vcf(tmp_path / "t.vcf", [
            ("1", 100, "A", "C", ".", ["0/1"]),
            ("1", 200, "A", "C", ".", ["0/1"]),
            ("1", 300, "A", "C", ".", ["0/1"]),
        ])
        out = str(tmp_path / "out.vcf")
        VCFWriter(template, out).write(
            [site_pair(100), site_pair(200)], store_posterior=False)
        with pytest.raises(ValueError, match="no ancestree provenance"):
            Reader(out).provenance()
        grade = Reader(out).grade({100: "A", 200: "C", 300: "A"})
        assert grade.n_sites == 2
        assert grade.map_recovery == pytest.approx(0.5)
        assert grade.brier == pytest.approx(1.0)

    def test_a_non_numeric_probability_reads_as_none(self):
        assert Reader._as_prob("abc") is None
        assert Reader._as_prob("0.25") == pytest.approx(0.25)

    def test_a_posterior_of_the_wrong_length_reads_as_none(self):
        assert Reader._as_posterior("0.5,0.5") is None
        assert Reader._as_posterior("0.5,0.5,0,0") == {
            "A": 0.5, "C": 0.5, "G": 0.0, "T": 0.0}
