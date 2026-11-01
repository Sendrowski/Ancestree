"""FASTA alignments as input, and the FASTA and TSV outputs.

An alignment carries every site of its contig, monomorphic columns included,
so it must give the polymorphic sites the posteriors the same data give as a
VCF, without the monomorphic-site count a VCF needs.
"""
import gzip
import logging

import numpy as np
import pytest
import tskit

import ancestree as anc
from ancestree.cli import run
from testing._helpers import QUICKSTART_TREES

INGROUP = [f"i{i}" for i in range(6)]
OUTGROUP = ["o0", "o1"]
#: Three haploid samples and an outgroup on two contigs, the second filtered
#: out in the tests that read it.
VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=10>
##contig=<ID=2,length=8>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\tc\to
1\t3\t.\tA\tC\t.\tPASS\t.\tGT\t0\t0\t1\t1
1\t7\t.\tG\tT\t.\tPASS\t.\tGT\t1\t0\t0\t0
2\t2\t.\tT\tA\t.\tPASS\t.\tGT\t0\t1\t0\t0
"""
TREE = "((a:0.1,b:0.1):0.05,c:0.1,o:0.3);"


def _write_fasta(path, records, width=None):
    """Write ``{name: sequence}`` as FASTA, wrapped at ``width``."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt") as f:
        for name, seq in records.items():
            f.write(f">{name} description\n")
            step = width or len(seq)
            for i in range(0, len(seq), step):
                f.write(seq[i:i + step] + "\n")


@pytest.fixture(scope="module")
def quickstart():
    """The quickstart tree sequence and its alignment, one record per sample,
    the monomorphic columns taking a random root sequence."""
    ts = tskit.load(QUICKSTART_TREES)
    length = int(ts.sequence_length)
    root = "".join(np.random.default_rng(3).choice(list("ACGT"), length))
    names = [ts.individual(ts.node(u).individual).metadata["name"]
             for u in ts.samples()]
    return ts, dict(zip(names, ts.alignments(reference_sequence=root)))


@pytest.fixture
def quickstart_fasta(quickstart, tmp_path):
    path = tmp_path / "chr1.fa.gz"
    _write_fasta(path, quickstart[1], width=60)
    return str(path)


class TestAlignmentSource:
    def test_columns_become_sites(self, tmp_path):
        """Case is folded, gaps, N and ambiguity codes read as missing, and a
        column without a base is skipped while keeping its position."""
        path = tmp_path / "locus.fasta"
        _write_fasta(path, {"a": "AcG-N", "b": "aCT-R", "c": "ACG-?"})
        source = anc.AlignmentSource(path)
        sites = list(source)
        assert source.samples() == ["a", "b", "c"]
        assert source.contig_lengths() == {"locus": 5}
        assert [s.pos for s in sites] == [1, 2, 3]
        assert {s.chrom for s in sites} == {"locus"}
        assert sites[2].alleles == ("G", "T")
        assert dict(sites[1].tip_alleles) == {"a": "C", "b": "C", "c": "C"}
        assert source.all_sites and source.position_base == 1

    def test_reference_sets_the_positions_and_ref_allele(self, tmp_path):
        path = tmp_path / "locus.fa"
        _write_fasta(path, {"a": "AC-GT", "ref": "A-TGC"})
        source = anc.AlignmentSource(path, reference="ref")
        sites = list(source)
        assert source.contig_lengths() == {"locus": 4}
        assert [s.pos for s in sites] == [1, 2, 3, 4]
        # The column where the reference has a gap is skipped.
        assert [s.tip_alleles["a"] for s in sites] == ["A", None, "G", "T"]
        assert sites[3].alleles == ("C", "T")

    def test_several_files_are_contigs_with_the_union_of_samples(self, tmp_path):
        _write_fasta(tmp_path / "x.fa", {"a": "AC", "b": "AG"})
        _write_fasta(tmp_path / "y.fa", {"b": "T", "c": "T"})
        source = anc.AlignmentSource([tmp_path / "x.fa", tmp_path / "y.fa"])
        sites = list(source)
        assert source.samples() == ["a", "b", "c"]
        assert [(s.chrom, s.pos) for s in sites] == [("x", 1), ("x", 2), ("y", 1)]
        assert sites[2].tip_alleles["a"] is None
        only = anc.AlignmentSource([tmp_path / "x.fa", tmp_path / "y.fa"],
                                   chrom_filter="y", sample_filter=["c"])
        assert [dict(s.tip_alleles) for s in only] == [{"c": "T"}]

    @pytest.mark.parametrize("records, kwargs, match", [
        ({"a": "AC", "b": "A"}, {}, "not aligned"),
        ({"a": "AC"}, {"reference": "ref"}, "not a record"),
        ({}, {}, "no FASTA record"),
    ])
    def test_malformed_input_raises(self, tmp_path, records, kwargs, match):
        path = tmp_path / "x.fa"
        _write_fasta(path, records)
        with pytest.raises(ValueError, match=match):
            anc.AlignmentSource(path, **kwargs)

    @pytest.mark.parametrize("text, match", [
        ("AC\n>a\nAC\n", "before its first"),
        (">\nAC\n", "without a name"),
    ])
    def test_unheaded_sequence_raises(self, tmp_path, text, match):
        path = tmp_path / "x.fa"
        path.write_text(text)
        with pytest.raises(ValueError, match=match):
            anc.AlignmentSource(path)

    def test_spaces_within_a_sequence_line_are_dropped(self, tmp_path):
        path = tmp_path / "x"
        path.write_text(">a\nA C G\n>b\nATG\n")
        source = anc.AlignmentSource(path)
        assert source.contig_lengths() == {"x": 3}
        assert [s.tip_alleles["a"] for s in source] == ["A", "C", "G"]

    def test_chrom_filter_naming_no_file_raises(self, tmp_path):
        _write_fasta(tmp_path / "x.fa", {"a": "A"})
        with pytest.raises(ValueError, match="names no alignment"):
            anc.AlignmentSource(tmp_path / "x.fa", chrom_filter="y")

    def test_a_sample_named_twice_raises(self, tmp_path):
        path = tmp_path / "x.fa"
        path.write_text(">a\nA\n>a\nC\n")
        with pytest.raises(ValueError, match="twice"):
            anc.AlignmentSource(path)

    def test_two_files_naming_one_contig_raise(self, tmp_path):
        (tmp_path / "d").mkdir()
        _write_fasta(tmp_path / "x.fa", {"a": "A"})
        _write_fasta(tmp_path / "d" / "x.fasta", {"a": "A"})
        with pytest.raises(ValueError, match="both name contig"):
            anc.AlignmentSource([tmp_path / "x.fa", tmp_path / "d" / "x.fasta"])

    def test_resolve_dispatches_paths_and_refuses_ploidy(self, tmp_path):
        _write_fasta(tmp_path / "x.fa", {"a": "A"})
        _write_fasta(tmp_path / "y.fna.gz", {"a": "C"})
        assert isinstance(anc.SiteSource.resolve(str(tmp_path / "x.fa")),
                          anc.AlignmentSource)
        both = anc.SiteSource.resolve([str(tmp_path / "x.fa"),
                                       str(tmp_path / "y.fna.gz")])
        assert list(both.contig_lengths()) == ["x", "y"]
        with pytest.raises(ValueError, match="haploid"):
            anc.SiteSource.resolve(str(tmp_path / "x.fa"), ploidy=2)


class TestInference:
    def test_given_tree_matches_the_tree_sequence_sites(self, quickstart,
                                                        quickstart_fasta):
        """A polymorphic column scores as the same site read from the tree
        sequence, one position to the right."""
        ts = quickstart[0]
        tree = anc.FixedTree.from_tskit_tree(ts.first())
        kwargs = dict(tree=tree, ingroup_samples=INGROUP, progress=False)
        from_ts = {s.pos + 1: p.values for s, p in anc.FixedTreeInference(
            anc.TskitSource(ts), anc.JC69(), **kwargs).infer()}
        from_fasta = {s.pos: p.values for s, p in anc.FixedTreeInference(
            quickstart_fasta, anc.JC69(), **kwargs).infer()}
        assert len(from_fasta) == ts.sequence_length
        for pos, values in from_ts.items():
            np.testing.assert_allclose(from_fasta[pos], values, atol=1e-12)

    def test_ladder_fit_needs_no_target_site_count(self, quickstart,
                                                   quickstart_fasta):
        """The monomorphic columns are the calibration, so the fit equals that
        of the polymorphic sites with the alignment length as the target."""
        ts = quickstart[0]
        kwargs = dict(ingroup_samples=INGROUP, outgroup_samples=OUTGROUP,
                      progress=False, n_starts=2)
        aligned = anc.FixedTreeInference(quickstart_fasta, anc.JC69(), **kwargs)
        aligned.fit()
        sites = anc.FixedTreeInference(
            anc.TskitSource(ts), anc.JC69(),
            n_target_sites=int(ts.sequence_length), **kwargs)
        sites.fit()
        assert aligned.params_mle.keys() == sites.params_mle.keys()
        np.testing.assert_allclose(list(aligned.params_mle.values()),
                                   list(sites.params_mle.values()), rtol=1e-6)

    def test_local_tree_windows_count_polymorphic_sites_only(
            self, quickstart_fasta):
        """A SNP-sized window spans the SNPs, not the columns."""
        source = anc.AlignmentSource(quickstart_fasta)
        n_snps = sum(1 for _ in anc.PolymorphicSiteFilter(source))
        inf = anc.LocalTreeInference(
            source, mu=5e-8, rec_rate=1e-8, window="8snp",
            ingroup_samples=INGROUP, progress=False)
        width = 8 * source.contig_lengths()["chr1"] / n_snps
        assert inf.segmentation.window == pytest.approx(width, rel=0.25)

    def test_local_tree_ignores_monomorphic_columns(self, quickstart_fasta):
        """At fixed widths, the monomorphic columns between the first and last
        site leave the posteriors of the polymorphic sites unchanged."""
        kwargs = dict(mu=5e-8, rec_rate=1e-8, ingroup_samples=INGROUP,
                      outgroup_samples=OUTGROUP, n_ensemble=None,
                      window="2000", block_size="500", chunk_size=None,
                      # A list of sites is 0-based, so the axis must hold
                      # the alignment's last column, 50,000.
                      sequence_length=50_001, progress=False)
        sites = list(anc.AlignmentSource(quickstart_fasta))
        polymorphic = [s for s in sites[1:-1] if len(s.alleles) > 1]
        full = anc.LocalTreeInference(sites, sample_names=INGROUP + OUTGROUP,
                                      **kwargs)
        thin = anc.LocalTreeInference(
            [sites[0], *polymorphic, sites[-1]],
            sample_names=INGROUP + OUTGROUP, **kwargs)
        thin_post = {s.pos: p.values for s, p in thin.infer()}
        full_post = {s.pos: p.values for s, p in full.infer()
                     if s.pos in thin_post}
        assert full_post.keys() == thin_post.keys()
        for pos, values in thin_post.items():
            np.testing.assert_allclose(full_post[pos], values, atol=1e-10)

    def test_target_site_count_is_refused(self, quickstart_fasta):
        """The columns calibrate the fit, so a second calibration would
        rescale the branch rates."""
        with pytest.raises(ValueError, match="n_target_sites"):
            anc.FixedTreeInference(
                quickstart_fasta, anc.JC69(), ingroup_samples=INGROUP,
                outgroup_samples=OUTGROUP, n_target_sites=10)

    def test_given_tree_scores_its_tips_only(self, tmp_path):
        """A record off the tree informs no site, and the tips outside the
        ingroup are the outgroups."""
        path = tmp_path / "x.fa"
        _write_fasta(path, {"a": "GN", "b": "GN", "c": "AN", "o": "GN",
                            "extra": "GT"})
        inf = anc.FixedTreeInference(
            path, anc.JC69(), tree=anc.FixedTree.from_newick(TREE),
            ingroup_samples=["a", "b"], progress=False)
        assert inf._resolved_outgroups == ("c", "o")
        out = tmp_path / "anc.fa"
        inf.to_fasta(out)
        assert out.read_text() == ">x\nGN\n"

    def test_window_counts_sites_carrying_two_alleles(self, quickstart):
        """Monomorphic records, as an all-sites VCF holds, leave the
        SNP-sized window unchanged."""
        sites = list(anc.TskitSource(quickstart[0]))
        padded = [*sites, *(anc.Site(chrom=s.chrom, pos=s.pos + 1,
                                     alleles=("A",), tip_alleles={})
                            for s in sites if s.pos + 1 < 50_000)]
        padded.sort(key=lambda s: s.pos)
        kwargs = dict(mu=5e-8, rec_rate=1e-8, window="8snp",
                      sequence_length=50_000, ingroup_samples=INGROUP,
                      sample_names=INGROUP + OUTGROUP, progress=False)
        assert (anc.LocalTreeInference(padded, **kwargs).segmentation.window
                == anc.LocalTreeInference(sites, **kwargs).segmentation.window)

    def test_monomorphic_ingroup_sites_are_not_a_warning(
            self, quickstart_fasta, caplog):
        with caplog.at_level(logging.INFO, logger="ancestree"):
            list(anc.LocalTreeInference(
                quickstart_fasta, mu=5e-8, rec_rate=1e-8, n_ensemble=None,
                ingroup_samples=INGROUP, outgroup_samples=OUTGROUP,
                progress=False).infer())
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING
                    and "monomorphic within the ingroup" in r.getMessage()]


def _posteriors(pairs):
    """``(Site, Posterior)`` pairs from ``(chrom, pos, map, probability)``."""
    out = []
    for chrom, pos, allele, p in pairs:
        values = np.full(4, (1 - p) / 3)
        values[anc.STATE_INDEX[allele]] = p
        site = anc.Site(chrom=chrom, pos=pos, alleles=(allele,),
                        tip_alleles={"a": allele})
        out.append((site, anc.Posterior(anc.STATES, values)))
    return out


class TestFastaWriter:
    def test_case_marks_confidence_and_n_fills_the_rest(self, tmp_path):
        pairs = _posteriors([("x", 1, "A", 0.99), ("x", 3, "C", 0.6),
                             ("y", 2, "G", 0.95)])
        silent = anc.Site(chrom="y", pos=1, alleles=("T",),
                          tip_alleles={"a": None})
        pairs.insert(2, (silent, pairs[0][1]))
        out = tmp_path / "anc.fa"
        n = anc.FastaWriter(out, min_confidence=0.9,
                            contig_lengths={"x": 4, "z": 2}).write(pairs)
        assert n == 4
        assert out.read_text() == ">x\nANcN\n>y\nNG\n>z\nNN\n"
        annotations = list(anc.Reader(out).annotations())
        assert [(a.chrom, a.pos, a.aa) for a in annotations] == [
            ("x", 1, "A"), ("x", 3, None), ("y", 2, "G")]

    def test_wraps_and_compresses(self, tmp_path):
        pairs = _posteriors([("x", i, "T", 1.0) for i in range(1, 131)])
        out = tmp_path / "anc.fa.gz"
        anc.FastaWriter(out).write(pairs)
        lines = gzip.open(out, "rt").read().splitlines()
        assert [len(line) for line in lines] == [2, 60, 60, 10]

    def test_interrupted_contig_raises(self, tmp_path):
        pairs = _posteriors([("x", 1, "A", 1.0), ("y", 1, "A", 1.0),
                             ("x", 2, "A", 1.0)])
        with pytest.raises(ValueError, match="interrupted"):
            anc.FastaWriter(tmp_path / "anc.fa").write(pairs)

    def test_zero_based_sites_start_the_record(self, tmp_path):
        out = tmp_path / "anc.fa"
        anc.FastaWriter(out, position_base=0).write(
            _posteriors([("x", 0, "G", 1.0)]))
        assert out.read_text() == ">x\nG\n"
        with pytest.raises(ValueError, match="before the contig"):
            anc.FastaWriter(out).write(_posteriors([("x", 0, "G", 1.0)]))

    @pytest.mark.parametrize("chrom", ["", "chr 1"])
    def test_unwritable_contig_name_raises(self, tmp_path, chrom):
        with pytest.raises(ValueError, match="whitespace"):
            anc.FastaWriter(tmp_path / "anc.fa").write(
                _posteriors([(chrom, 1, "G", 1.0)]))

    def test_vcf_run_fills_its_contig_at_the_vcf_positions(self, tmp_path):
        """Only the filtered contig is written, over its declared length."""
        vcf = tmp_path / "x.vcf"
        vcf.write_text(VCF)
        inf = anc.FixedTreeInference(
            str(vcf), anc.JC69(), tree=anc.FixedTree.from_newick(TREE),
            ingroup_samples=["a", "b"], chrom_filter="1", progress=False)
        posteriors = list(inf.infer())
        out = tmp_path / "anc.fa"
        inf.to_fasta(out, posteriors=posteriors)
        lines = out.read_text().splitlines()
        assert lines[0] == ">1" and len(lines[1]) == 10
        assert [lines[1][s.pos - 1] for s, _ in posteriors] == [
            p.map_allele for _, p in posteriors]

    def test_tree_sequence_run_reads_back_one_to_the_right(self, quickstart,
                                                          tmp_path):
        ts = quickstart[0]
        inf = anc.FixedTreeInference(
            anc.TskitSource(ts), anc.JC69(),
            tree=anc.FixedTree.from_tskit_tree(ts.first()),
            ingroup_samples=INGROUP, progress=False)
        posteriors = list(inf.infer())
        out = tmp_path / "anc.fa"
        inf.to_fasta(out, posteriors=posteriors)
        read = {a.pos: a.aa for a in anc.Reader(out).annotations()}
        assert {s.pos + 1: p.map_allele for s, p in posteriors} == read

    def test_grading_counts_the_lower_case_bases_skipped(self, tmp_path,
                                                         caplog):
        out = tmp_path / "anc.fa"
        anc.FastaWriter(out, min_confidence=0.9).write(
            _posteriors([("x", 1, "A", 0.99), ("x", 2, "C", 0.6)]))
        with caplog.at_level(logging.WARNING, logger="ancestree"):
            grade = anc.Reader(out).grade({1: "A", 2: "C"})
        assert grade.n_sites == 1
        assert "1 of 2 records skipped" in caplog.text
        assert len(list(anc.Reader(out).annotations(limit=1))) == 1
        with pytest.raises(ValueError, match="provenance"):
            anc.Reader(out).provenance()


class TestTsv:
    def test_round_trip(self, tmp_path):
        pairs = _posteriors([("x", 1, "A", 0.99), ("x", 2, "C", 0.5)])
        out = tmp_path / "anc.tsv.gz"
        anc.TsvWriter(out, min_confidence=0.9).write(
            pairs, provenance={"mode": "test"}, info={"k": 1})
        reader = anc.Reader(out)
        assert reader.provenance()["mode"] == "test"
        rows = list(reader.annotations())
        assert [(a.chrom, a.pos, a.aa, a.aa_prob) for a in rows] == [
            ("x", 1, "A", 0.99), ("x", 2, None, 0.5)]
        np.testing.assert_allclose(
            [rows[1].posterior[s] for s in anc.STATES], pairs[1][1].values)

    def test_without_posterior_columns(self, tmp_path):
        out = tmp_path / "anc.tsv"
        anc.TsvWriter(out).write(_posteriors([("x", 1, "A", 0.7)]),
                                 store_posterior=False)
        assert out.read_text().splitlines() == [
            "chrom\tpos\tAA\tAA_prob", "x\t1\tA\t0.69999999999999996"]
        assert list(anc.Reader(out).annotations())[0].posterior is None

    def test_reads_an_iqtree_state_table(self, tmp_path):
        out = tmp_path / "run.state"
        out.write_text(
            "# comment\nNode\tSite\tState\tp_A\tp_C\tp_G\tp_T\n"
            "Node2\t1\tC\t0.10000\t0.80000\t0.05000\t0.05000\n")
        (row,) = anc.Reader(out).annotations()
        assert (row.pos, row.aa, row.aa_prob) == (1, "C", 0.8)
        grade = anc.Reader(out).grade({1: "C"})
        assert grade.n_sites == 1 and grade.map_recovery == 1.0
        with out.open("a") as f:
            f.write("\nNode1\t1\tA\t1\t0\t0\t0\n")
        assert len(list(anc.Reader(out).annotations(limit=1))) == 1
        with pytest.raises(ValueError, match="one node"):
            list(anc.Reader(out).annotations())

    def test_refuses_a_protein_state_table(self, tmp_path):
        out = tmp_path / "run.state"
        out.write_text("Node\tSite\tState\tp_A\tp_C\tp_D\tp_G\tp_T\n"
                       "Node2\t1\tD\t0\t0\t1\t0\t0\n")
        with pytest.raises(ValueError, match="p_D"):
            list(anc.Reader(out).annotations())

    def test_without_provenance_raises(self, tmp_path):
        out = tmp_path / "anc.tsv"
        out.write_text("chrom\tpos\tAA\tAA_prob\nx\t1\tA\t1\n")
        with pytest.raises(ValueError, match="provenance"):
            anc.Reader(out).provenance()


class TestCli:
    def test_fixed_tree_alignment_to_fasta_and_tsv(self, quickstart,
                                                    quickstart_fasta, tmp_path):
        ts = quickstart[0]
        newick = tmp_path / "tree.nwk"
        newick.write_text(ts.first().as_newick(
            node_labels={u: n for u, n in zip(ts.samples(), quickstart[1])}))
        fasta, tsv = tmp_path / "anc.fa", tmp_path / "anc.tsv"
        run(["fixed-tree", "--alignment", quickstart_fasta, "--tree",
             str(newick), "--ingroup", ",".join(INGROUP),
             "--out", str(fasta), str(tsv)])
        rows = list(anc.Reader(tsv).annotations())
        assert len(rows) == ts.sequence_length
        record = "".join(fasta.read_text().splitlines()[1:])
        assert record == "".join(r.aa for r in rows)
        assert anc.Reader(tsv).provenance()["mode"] == "fixed-tree"

    def test_local_tree_alignment_with_reference(self, quickstart_fasta,
                                                 tmp_path):
        out = tmp_path / "anc.tsv"
        run(["local-tree", "--alignment", quickstart_fasta, "--reference",
             "o0", "--mu", "5e-8", "--rec-rate", "1e-8", "--no-ensemble",
             "--ingroup", ",".join(INGROUP), "--out", str(out)])
        assert list(anc.Reader(out).annotations())[0].chrom == "chr1"

    def test_several_outputs_match_one_output_each(self, quickstart_fasta,
                                                   tmp_path):
        """One run writes every --out path, each as a run of its own would."""
        args = ["local-tree", "--alignment", quickstart_fasta, "--mu", "5e-8",
                "--rec-rate", "1e-8", "--no-ensemble",
                "--ingroup", ",".join(INGROUP)]
        outs = [tmp_path / name for name in ("all.vcf", "all.vcz", "all.trees",
                                             "all.tsv", "all.fa")]
        run([*args, "--out", *map(str, outs)])
        run([*args, "--out", str(tmp_path / "one.tsv")])
        expected = [(a.pos, a.aa) for a in
                    anc.Reader(tmp_path / "one.tsv").annotations()]
        for out in outs[:2] + outs[3:]:
            assert [(a.pos, a.aa) for a in
                    anc.Reader(out).annotations()] == expected, out
        assert tskit.load(outs[2]).num_sites

    @pytest.mark.parametrize("command, flags, match", [
        ("local-tree", ["--vcf", "x.vcf", "--reference", "a"],
         "requires --alignment"),
        ("fixed-tree", ["--vcf", "x.vcf", "--tree", "t.nwk",
                        "--reference", "a"],
         "requires --alignment"),
        ("local-tree", ["--alignment", "x.fa", "--ploidy", "2"],
         "--ploidy does not apply"),
        ("local-tree", ["--alignment", "x.fa", "--phased"],
         "--phased does not apply"),
        ("local-tree", ["--alignment", "x.fa", "--phase-seed", "3"],
         "--phase-seed does not apply"),
    ])
    def test_misplaced_flags_exit(self, command, flags, match, tmp_path):
        with pytest.raises(SystemExit, match=match):
            run([command, *flags, "--out", str(tmp_path / "o.vcf")])

    @pytest.mark.parametrize("outs, match", [
        (["o.tsv", "./o.tsv"], "twice"),
        (["o.tsv", "o.txt"], r"\.fna \(each optionally \.gz\)"),
    ])
    def test_out_paths_are_checked_before_the_run(self, outs, match,
                                                  tmp_path):
        with pytest.raises(SystemExit, match=match):
            run(["local-tree", "--alignment", "x.fa",
                 "--out", *(str(tmp_path / o) for o in outs)])
