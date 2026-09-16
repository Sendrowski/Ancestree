"""Two ways VCFWriter lost or misplaced data without saying so.

Writing over the template truncated it: pass 2 opens the output while pass 1's
reader is still iterating the same path, so a file larger than htslib's first
read block was destroyed and then failed to parse. And template rows are keyed
on ``(contig, int(position))``, so two sites sharing an integer position -- a
tree sequence with ``discrete_genome=False``, or a split-multiallelic VCF --
collapse onto one row and the last call silently wins.

The staged-file discipline of every writer belongs here too: a failed
rename, swap or dump leaves neither a partial file nor a lost store, a run
that scores nothing is refused or reported, and a row the run does not
score carries nothing from an earlier run.
"""
import glob
import gzip
import os
import shutil

import numpy as np
import pytest

import ancestree as anc
from ancestree.posterior import Posterior
from ancestree.sites import Site
from ancestree.writers import TskitWriter, VCFWriter, Writer, ZarrWriter
from testing._helpers import post
from testing._helpers import DEMO_VCF, QUICKSTART_VCF
from testing._helpers import site_pair, staged_files, write_vcf

TEMPLATE = QUICKSTART_VCF


def test_writing_over_the_template_is_refused(tmp_path):
    panel = tmp_path / "panel.vcf.gz"
    shutil.copy(TEMPLATE, panel)
    before = os.path.getsize(panel)

    with pytest.raises(ValueError, match="cannot write over its template"):
        anc.VCFWriter(panel, panel).write([])

    assert os.path.getsize(panel) == before, "the template was modified"


def test_writing_over_the_zarr_template_is_refused(tmp_path):
    """The store must survive, with any ``variant_AA`` it already carried."""
    import zarr

    from testing._helpers import write_vcz

    path = str(tmp_path / "s.vcz")
    write_vcz(
        path, positions=[100, 200], contigs_per_variant=[0, 0],
        contig_ids=["1"], alleles=[["A", "C"], ["T", "G"]],
        sample_ids=["s1"], genotypes=np.zeros((2, 1, 2), dtype=np.int8),
    )
    root = zarr.open(path, mode="r+")
    ZarrWriter._create_variant_array(
        root, "variant_AA", np.asarray(["A", "T"], dtype="U1"), ["variants"])
    pair = site_pair(pos=100)

    with pytest.raises(ValueError, match="cannot write over its template"):
        ZarrWriter(path, path).write([pair])

    assert list(zarr.open(path, mode="r")["variant_AA"][:]) == ["A", "T"]


def test_a_different_output_path_is_allowed(tmp_path):
    """The guard must not block ordinary use."""
    panel = tmp_path / "panel.vcf.gz"
    shutil.copy(TEMPLATE, panel)
    out = tmp_path / "annotated.vcf.gz"
    assert anc.VCFWriter(panel, out).write([]) == 0
    assert out.exists()


def test_colliding_positions_are_reported(tmp_path, caplog):
    """Two sites on one template row must not collapse silently."""
    panel = tmp_path / "panel.vcf.gz"
    shutil.copy(TEMPLATE, panel)
    out = tmp_path / "annotated.vcf.gz"

    src = anc.CyVCF2Source(str(panel))
    first = next(iter(src))
    post = anc.Posterior(alleles=tuple("ACGT"), values=np.array([1.0, 0.0, 0.0, 0.0]))
    # The same integer position twice: the second must be reported, not
    # silently overwrite the first.
    pairs = [(first, post), (first, post)]

    with caplog.at_level("WARNING"):
        anc.VCFWriter(panel, out).write(pairs)

    assert any("no template row carrying their alleles" in r.message
               for r in caplog.records), (
        f"no collision warning; got {[r.message for r in caplog.records]}")


def test_two_sites_on_one_integer_position_get_their_own_rows(tmp_path):
    """Distinct sites sharing an integer position take separate rows.

    A site carrying a local_tree_handle comes from a source that emits one
    Site per genomic position, so it claims one row. A handle-less VCF site
    broadcasts to every row at its position, which a split multiallelic needs.
    """
    vcf = tmp_path / "split.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0\n"
        "1\t10\t.\tG\tT\t.\tPASS\t.\tGT\t0\n")
    out = tmp_path / "out.vcf"

    def _p(a):
        v = np.zeros(4)
        v["ACGT".index(a)] = 1.0
        return anc.Posterior(alleles=tuple("ACGT"), values=v)

    pairs = [
        (anc.Site(chrom="1", pos=10, alleles=("A", "C"),
                 tip_alleles={"s1": "A"}, local_tree_handle=10.2), _p("A")),
        (anc.Site(chrom="1", pos=10, alleles=("G", "T"),
                 tip_alleles={"s1": "G"}, local_tree_handle=10.7), _p("G")),
    ]
    assert anc.VCFWriter(vcf, out).write(pairs) == 2

    calls = [line.split("\t")[7] for line in out.read_text().splitlines()
             if not line.startswith("#")]
    assert "AA=A" in calls[0], f"row 1 got {calls[0]}"
    assert "AA=G" in calls[1], f"row 2 got {calls[1]}"


class TestRowLocator:
    """Keys that fall outside the template's variant axis resolve to no row."""

    def test_an_empty_axis_locates_nothing(self):
        locate = Writer._row_locator(
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32), [], {})
        assert locate("1", 100) == -1

    def test_a_contig_without_rows_locates_nothing(self):
        """A contig declared in the header but absent from the records."""
        pos = np.array([100, 200], dtype=np.int64)
        contig = np.array([0, 0], dtype=np.int32)
        locate = Writer._row_locator(pos, contig, ["1", "2"], {"1": 0, "2": 1})
        assert locate("1", 200) == 1
        assert locate("2", 200) == -1

    def test_a_run_is_walked_back_to_its_first_row(self):
        """A canonical row inside a shared-position run still claims the run's first free row."""
        annotated = np.zeros(2, dtype=bool)
        rows = Writer._rows_for_site(
            annotated, np.array([100, 100]), np.array([0, 0]), 1,
            site_pair(100)[0])
        assert rows == [0]


class TestAlleleTupleTolerance:
    """A record is identified by which alleles it carries, not their order.

    Narrowing a shared-position run to rows whose allele tuple matched
    exactly dropped the posterior whenever the source and the template
    disagreed on bookkeeping: a swapped REF and ALT, a template ALT the site
    does not use, or the missing-data placeholder a tree sequence appends.
    """

    @staticmethod
    def _site(alleles):
        from ancestree.sites import Site

        return Site(chrom="1", pos=100, alleles=tuple(alleles),
                    tip_alleles={"a": alleles[0]})

    @pytest.mark.parametrize("template,site_alleles", [
        (("A", "G"), ("A", "G")),                 # identical
        (("G", "A"), ("A", "G")),                 # REF and ALT swapped
        (("A", "G", "C"), ("A", "G")),            # template carries a spare ALT
        (("A", "G"), ("A", "G", None)),           # tskit's missing-data allele
    ])
    def test_a_record_carrying_the_site_alleles_is_matched(self, template,
                                                           site_alleles):
        from ancestree.writers import Writer

        annotated = np.zeros(1, dtype=bool)
        rows = Writer._rows_for_site(annotated, np.array([100]), np.array([0]), 0,
                              self._site(site_alleles), alleles=[template])
        assert rows == [0], (
            f"template {template} did not match site {site_alleles}")

    def test_a_record_missing_an_allele_is_not_matched(self):
        """A site naming an allele the record lacks describes another variant."""
        from ancestree.writers import Writer

        annotated = np.zeros(1, dtype=bool)
        rows = Writer._rows_for_site(annotated, np.array([100]), np.array([0]), 0,
                              self._site(("A", "T")), alleles=[("A", "G")])
        assert rows == []


def test_re_annotation_does_not_keep_a_stale_posterior(tmp_path):
    """A second run must not leave the first run's posterior beside its call.

    ``store_posterior=False`` writes no ``AA_post``, so a template already
    carrying one from an earlier run would keep a distribution that disagrees
    with the ``AA`` written next to it.
    """
    import cyvcf2
    from ancestree.posterior import Posterior
    from ancestree.sites import Site

    sites = [
        Site(chrom=v.CHROM, pos=int(v.POS), alleles=(v.REF, v.ALT[0]),
             tip_alleles={})
        for v in cyvcf2.VCF(TEMPLATE) if v.ALT
    ][:40]
    assert sites, "template yielded no biallelic records"

    def posteriors(top):
        for site in sites:
            yield site, Posterior(
                alleles=("A", "C", "G", "T"),
                values=np.array([top, 1.0 - top, 0.0, 0.0]))

    first = str(tmp_path / "first.vcf.gz")
    second = str(tmp_path / "second.vcf.gz")
    anc.VCFWriter(TEMPLATE, first).write(posteriors(0.9))
    anc.VCFWriter(first, second).write(posteriors(0.6), store_posterior=False)

    carried = sum(1 for v in cyvcf2.VCF(second)
                  if v.INFO.get("AA_post") is not None)
    assert carried == 0, (
        f"{carried} record(s) kept a posterior from the earlier run")


def test_a_tree_sequence_with_no_matches_is_refused():
    """A .trees in which nothing matched reads as successfully annotated.

    Every site keeps its original ``ancestral_state``, so unlike an empty VCF
    the output is indistinguishable from a real annotation downstream.
    """
    import msprime

    from ancestree.posterior import Posterior
    from ancestree.sites import Site
    from ancestree.writers import TskitWriter

    ts = msprime.sim_ancestry(3, ploidy=1, sequence_length=10_000,
                              random_seed=4)
    ts = msprime.sim_mutations(ts, rate=1e-5, random_seed=4)
    offered = [
        (Site(chrom="nope", pos=10 ** 9 + i, alleles=("A", "C"),
              tip_alleles={}),
         Posterior(alleles=("A", "C", "G", "T"),
                   values=np.array([0.9, 0.1, 0.0, 0.0])))
        for i in range(10)
    ]
    with pytest.raises(ValueError, match="matched 0 of"):
        TskitWriter(ts, "unused.trees").write(iter(offered))


def test_local_tree_templates_from_its_source_not_its_inference(tmp_path):
    """The auto-written template must describe the data being annotated.

    Templating from the inferred pseudo-ARG gives whichever alleles
    ``map_mutations`` retained, which need not be the input's, so the calls
    whose alleles it does not carry find no row and are dropped.
    """
    import ancestree as anc

    vcf = DEMO_VCF
    common = dict(mu=1.25e-8, rec_rate=1e-8, sequence_length=200_000.0,
                  n_ensemble=None, progress=False)
    scored = sum(1 for _ in anc.LocalTreeInference(vcf, **common).infer())
    written = anc.LocalTreeInference(vcf, **common).to_vcf(
        str(tmp_path / "annotated.vcf.gz"))
    assert written == scored, (
        f"{scored - written} of {scored} posteriors found no template row; "
        f"the template does not describe the source")


class TestReAnnotationCarriesNothingFromTheEarlierRun:
    """A second pass must not leave the first run's values in its output.

    Two ways it did. The run-level ``AA_<key>`` fields were cleared only on
    rows the new run did not score, so on scored rows a parameter the current
    model never had sat beside the new run's own. And the confidence was handed
    to cyvcf2 as a float, which renders at six significant digits, while the
    posterior beside it was written at seventeen, so the two disagreed within
    one record and a near-certain call read back as exactly 1.
    """

    @staticmethod
    def _template(tmp_path, n=6):
        path = tmp_path / "t.vcf"
        rows = "\n".join(f"1\t{100 * (i + 1)}\t.\tA\tC\t.\t.\t.\tGT\t0/1"
                         for i in range(n))
        path.write_text(
            "##fileformat=VCFv4.2\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            "##contig=<ID=1,length=100000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
            + rows + "\n")
        return str(path)

    @staticmethod
    def _pairs(n=6, top=0.9):
        rest = (1.0 - top) / 3.0
        return [
            (anc.Site(chrom="1", pos=100 * (i + 1), alleles=("A", "C"),
                     tip_alleles={"s1_h0": "A"}),
             Posterior(alleles=("A", "C", "G", "T"),
                       values=np.array([top, rest, rest, rest])))
            for i in range(n)
        ]

    def test_a_run_level_field_does_not_survive_into_the_next_run(self, tmp_path):
        first, second = str(tmp_path / "r1.vcf"), str(tmp_path / "r2.vcf")
        template = self._template(tmp_path)
        anc.VCFWriter(template, first).write(
            self._pairs(), info={"kappa": 2.5, "prior": "run1"})
        anc.VCFWriter(first, second).write(
            self._pairs(), info={"prior": "run2"})

        import cyvcf2

        record = next(iter(cyvcf2.VCF(second)))
        carried = {str(k) for k, _ in record.INFO
                   if str(k).startswith("AA_")
                   and str(k) not in ("AA_prob", "AA_post")}
        assert carried == {"AA_prior"}

    def test_the_confidence_and_the_posterior_agree_in_one_record(self, tmp_path):
        out = str(tmp_path / "a.vcf")
        # A mid-range value whose seventh significant digit survives the
        # single-precision round trip, so the two fields differ measurably
        # when only one of them is written at full precision.
        anc.VCFWriter(self._template(tmp_path), out).write(
            self._pairs(top=0.7123456789))

        import cyvcf2

        for record in cyvcf2.VCF(out):
            prob = float(record.INFO.get("AA_prob"))
            raw = record.INFO.get("AA_post")
            post = ([float(v) for v in raw] if isinstance(raw, (tuple, list))
                    else [float(v) for v in str(raw).split(",")])
            # A VCF Float INFO field is single precision, so a near-certain
            # call still reads back as 1.0 whatever precision is written. What
            # the writer controls, and what was wrong, is the two fields
            # disagreeing within one record.
            assert prob == pytest.approx(max(post), abs=1e-12)


class TestThePosteriorMatrixIsOnlyAllocatedWhenItIsStored:
    """The ``(n_records, 4)`` posterior matrix was built either way.

    Both ``VCFWriter._write_streaming`` and ``ZarrWriter._annotate_store``
    allocated and filled it before consulting ``store_posterior``, so a caller
    asking only for ``AA`` and ``AA_prob`` paid four floats per template
    record for a posterior that was then dropped.
    """

    @staticmethod
    def _watch(monkeypatch):
        """Collect the ``post`` argument of every ``_place_posteriors`` call."""
        from ancestree.writers import Writer

        seen: list = []
        original = Writer._place_posteriors

        def spy(self, *args, **kwargs):
            seen.append(args[-1])
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Writer, "_place_posteriors", spy)
        return seen

    @staticmethod
    def site_pair(pos):
        return (
            anc.Site(chrom="1", pos=pos, alleles=("A", "C"),
                     tip_alleles={"s1_h0": "A"}),
            Posterior(alleles=("A", "C", "G", "T"),
                      values=np.array([0.9, 0.1, 0.0, 0.0])),
        )

    @pytest.mark.parametrize("store_posterior", [True, False])
    def test_the_vcf_writer_allocates_it_only_when_asked(
            self, tmp_path, monkeypatch, store_posterior):
        template = tmp_path / "t.vcf"
        template.write_text(
            "##fileformat=VCFv4.2\n##contig=<ID=1,length=1000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t100\t.\tA\tC\t.\t.\t.\n"
            "1\t200\t.\tA\tC\t.\t.\t.\n"
        )
        out = str(tmp_path / "out.vcf")
        seen = self._watch(monkeypatch)

        n = anc.VCFWriter(template, out).write(
            [self.site_pair(100), self.site_pair(200)],
            store_posterior=store_posterior)

        assert n == 2
        assert [p is not None for p in seen] == [store_posterior]

        import cyvcf2

        for record in cyvcf2.VCF(out):
            assert record.INFO.get("AA") == "A"
            assert float(record.INFO.get("AA_prob")) == pytest.approx(0.9)
            assert (record.INFO.get("AA_post") is not None) is store_posterior

    @pytest.mark.parametrize("store_posterior", [True, False])
    def test_the_zarr_writer_allocates_it_only_when_asked(
            self, tmp_path, monkeypatch, store_posterior):
        import zarr

        from testing._helpers import write_vcz

        path = str(tmp_path / "in.vcz")
        write_vcz(
            path, positions=[100, 200], contigs_per_variant=[0, 0],
            contig_ids=["1"], alleles=[["A", "C"], ["A", "C"]],
            sample_ids=["s1"], genotypes=np.zeros((2, 1, 2), dtype=np.int8),
        )
        out = str(tmp_path / "out.vcz")
        seen = self._watch(monkeypatch)

        n = anc.ZarrWriter(path, out).write(
            [self.site_pair(100), self.site_pair(200)],
            store_posterior=store_posterior)

        assert n == 2
        assert [p is not None for p in seen] == [store_posterior]

        root = zarr.open(out, mode="r")
        assert [str(a) for a in root["variant_AA"][:]] == ["A", "A"]
        assert ("variant_AA_post" in root) is store_posterior


class TestReAnnotatingAStoreClearsTheEarlierPosterior:
    """A second run without a posterior must not leave the first run's behind.

    ``Reader.grade`` prefers a stored posterior to the MAP call, so a stale
    ``variant_AA_post`` makes the second run's Brier score a score of the
    first run's probabilities.
    """

    def test_the_posterior_array_is_gone_after_a_no_posterior_rerun(
            self, small_ts, tmp_path):
        import zarr

        import ancestree as anc
        from ancestree.writers import ZARR_AA_POST_FIELD

        out = tmp_path / "a.vcz"
        inference = anc.Inference.from_arg(
            small_ts, anc.JC69(), mu=5e-8, progress=False)
        inference.to_zarr(out, store_posterior=True)
        assert ZARR_AA_POST_FIELD in zarr.open_group(str(out), mode="r")

        rerun = tmp_path / "b.vcz"
        anc.Inference.from_arg(
            small_ts, anc.JC69(), mu=5e-8, progress=False,
        ).to_zarr(rerun, input_zarr=out, store_posterior=False)
        root = zarr.open_group(str(rerun), mode="r")
        assert ZARR_AA_POST_FIELD not in root, (
            "the earlier run's posterior survived, so grade() would score "
            "this run's calls against it")
        assert anc.Reader(rerun).head(1)[0].posterior is None


class TestStagedFiles:
    """A failed rename or swap leaves neither a partial file nor a lost store."""

    def test_a_failed_rename_removes_the_staged_file(self, tmp_path):
        staged = tmp_path / ".partial-x"
        staged.write_text("partial")
        output = tmp_path / "out"
        output.mkdir()
        (output / "keep").write_text("keep")
        with pytest.raises(OSError):
            Writer._finish_staged(str(staged), str(output))
        assert not staged.exists()
        assert (output / "keep").read_text() == "keep"

    def test_a_failed_swap_restores_the_original_store(self, tmp_path):
        dst = tmp_path / "store.vcz"
        dst.mkdir()
        (dst / "marker").write_text("original")
        with pytest.raises(FileNotFoundError):
            ZarrWriter._swap_into_place(str(tmp_path / "missing"), str(dst))
        assert (dst / "marker").read_text() == "original"
        assert not glob.glob(str(tmp_path / "store.vcz.old-*"))


class TestVCFWriter:
    """Rows the run does not score and failures during the second pass."""

    AA_HEADER = (
        '##INFO=<ID=AA,Number=1,Type=String,Description="x">',
        '##INFO=<ID=AA_prob,Number=1,Type=Float,Description="x">',
        '##INFO=<ID=AA_post,Number=4,Type=Float,Description="x">',
    )

    def test_an_unscored_row_loses_the_earlier_runs_fields(self, tmp_path, caplog):
        """A row that received no posterior carries no ``AA`` field from before."""
        import cyvcf2

        template = write_vcf(
            tmp_path / "t.vcf",
            [("1", 100, "A", "C", "AA=A;AA_prob=0.5;AA_post=1,0,0,0", ["0/1"]),
             ("1", 200, "A", "C", "AA=C;AA_prob=0.5;AA_post=0,1,0,0", ["0/1"])],
            extra_header=self.AA_HEADER)
        out = str(tmp_path / "out.vcf")
        with caplog.at_level("WARNING"):
            assert VCFWriter(template, out).write([]) == 0
        assert any("received no posteriors" in r.message for r in caplog.records)
        for record in cyvcf2.VCF(out):
            assert record.INFO.get("AA") is None
            assert record.INFO.get("AA_prob") is None
            assert record.INFO.get("AA_post") is None

    def test_a_failure_in_the_second_pass_leaves_no_partial_file(
            self, tmp_path, monkeypatch):
        template = write_vcf(tmp_path / "t.vcf",
                        [("1", 100, "A", "C", ".", ["0/1"])])
        out = tmp_path / "out.vcf"

        def _boom(value, vcf_type):
            raise RuntimeError("cannot coerce")

        monkeypatch.setattr(VCFWriter, "_coerce_info_value", staticmethod(_boom))
        with pytest.raises(RuntimeError, match="cannot coerce"):
            VCFWriter(template, out).write([site_pair(100)], info={"kappa": 2.0})
        assert not out.exists()
        assert staged_files(tmp_path) == []


class TestTskitWriter:
    """Handle-less collisions, an empty offer and a failed dump."""

    @pytest.mark.parametrize("chroms,remedy", [
        (("1", "2"), "Restrict the run to one contig"),
        (("1", "1"), "Supply posteriors carrying local_tree_handle"),
    ])
    def test_two_handle_less_posteriors_on_one_integer_are_refused(
            self, small_ts, tmp_path, chroms, remedy):
        pairs = [site_pair(100, chrom=chroms[0]), site_pair(100, chrom=chroms[1])]
        with pytest.raises(ValueError, match=remedy):
            TskitWriter(small_ts, tmp_path / "o.trees").write(pairs)

    def test_an_empty_offer_is_reported_and_written_unannotated(
            self, small_ts, tmp_path, caplog):
        import tskit

        out = tmp_path / "o.trees"
        with caplog.at_level("WARNING"):
            assert TskitWriter(small_ts, out).write([]) == 0
        assert any("received no posteriors" in r.message
                   and "input tree sequence" in r.message
                   for r in caplog.records)
        written = tskit.load(str(out))
        assert ([s.ancestral_state for s in written.sites()]
                == [s.ancestral_state for s in small_ts.sites()])

    def test_a_failed_dump_leaves_no_partial_file(
            self, small_ts, tmp_path, monkeypatch):
        import tskit

        def _dump(self, path, **kwargs):
            open(path, "wb").close()
            raise OSError("no space left on device")

        monkeypatch.setattr(tskit.TreeSequence, "dump", _dump)
        out = tmp_path / "o.trees"
        with pytest.raises(OSError, match="no space"):
            TskitWriter(small_ts, out).write([])
        assert not out.exists()
        assert staged_files(tmp_path) == []
