"""Tests for :class:`VCFWriter` and :class:`TskitWriter`, and for the samples
and templates the inferences write through them.

Round-trip pattern: build a small site stream, manufacture matching
posteriors, write, and re-read the output (cyvcf2 / tskit) to confirm
``AA`` / ``AA_prob`` and ``site.ancestral_state`` / metadata land
where expected.
"""
from __future__ import annotations

from pathlib import Path

import cyvcf2
import msprime
import numpy as np
import pytest
import tskit

import ancestree as anc
from ancestree.local_tree_inference import LocalTreeInference
from ancestree.posterior import Posterior
from ancestree.readers import Reader
from ancestree.sites import Site
from ancestree.sources import TskitSource
from ancestree.trees import TskitLocalTree
from ancestree.writers import TskitWriter, VCFWriter
from testing._helpers import DEMO_TREES, QUICKSTART_TREES, toy_sites

#: The quickstart panel's ingroup and outgroup, leaving i4, i5 and o1 in neither.
ING = ["i0", "i1", "i2", "i3"]
OUT = ["o0"]


@pytest.fixture(scope="module")
def hap_ts():
    """Small haploid ARG used as the round-trip substrate."""
    ts = msprime.sim_ancestry(
        samples=6, ploidy=1,
        sequence_length=2e4, recombination_rate=1e-8,
        population_size=1e4, random_seed=7,
    )
    return msprime.sim_mutations(ts, rate=5e-8, random_seed=7)


def _fake_posteriors(sites: list[Site]) -> list[tuple[Site, Posterior]]:
    """Manufacture per-site posteriors over the site's own allele tuple.

    Sets a strong bias for the first allele of each site (0.85) so
    ``map_allele`` is unambiguous and easy to check.
    """
    pairs: list[tuple[Site, Posterior]] = []
    for site in sites:
        n = len(site.alleles)
        values = np.full(n, (1 - 0.85) / max(1, n - 1))
        values[0] = 0.85
        pairs.append((site, Posterior(alleles=tuple(site.alleles), values=values)))
    return pairs


def _post(map_allele: str, p: float) -> Posterior:
    """A 4-state posterior peaked at ``map_allele`` with mass ``p``."""
    alleles = ("A", "C", "G", "T")
    rest = (1.0 - p) / 3.0
    return Posterior(
        alleles=alleles,
        values=np.array([p if a == map_allele else rest for a in alleles]),
    )


class TestVCFWriter:
    """:class:`VCFWriter` should add AA + AA_prob to every matching record."""

    @pytest.fixture
    def in_vcf(self, hap_ts, tmp_path_factory):
        p = tmp_path_factory.mktemp("inv") / "in.vcf"
        with open(p, "w") as f:
            hap_ts.write_vcf(f, contig_id="1")
        return p

    def test_writes_aa_and_aa_prob(self, in_vcf, hap_ts, tmp_path):
        out_vcf = tmp_path / "out.vcf"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        n = VCFWriter(in_vcf, out_vcf).write(pairs)
        assert n == len(sites)

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        assert len(recovered) == len(sites)
        # The fixture puts 0.85 on each site's FIRST allele, so both the call
        # and its probability are known: a writer emitting REF everywhere, or
        # any allele in the alphabet, must fail here.
        for variant, site in zip(recovered, sites):
            assert variant.INFO.get("AA") == site.alleles[0], (
                variant.POS, variant.INFO.get("AA"), site.alleles)
            assert float(variant.INFO.get("AA_prob")) == pytest.approx(0.85)

    def test_unannotated_records_pass_through(self, in_vcf, hap_ts, tmp_path):
        """Sites missing from the posterior stream emit a record but no AA."""
        out_vcf = tmp_path / "partial.vcf"
        sites = list(TskitSource(hap_ts))
        partial = _fake_posteriors(sites[:1])  # only first site
        n = VCFWriter(in_vcf, out_vcf).write(partial)
        assert n == 1

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        assert len(recovered) == len(sites)
        first_aa = recovered[0].INFO.get("AA")
        assert first_aa is not None
        rest_aa = [v.INFO.get("AA") for v in recovered[1:]]
        assert all(a is None for a in rest_aa)

    def test_header_declares_aa_info_fields(self, in_vcf, tmp_path):
        out_vcf = tmp_path / "hdr.vcf"
        VCFWriter(in_vcf, out_vcf).write([])
        header_text = Path(out_vcf).read_text()
        assert "ID=AA," in header_text
        assert "ID=AA_prob," in header_text

    def test_info_kwarg_adds_per_record_and_header(self, in_vcf, hap_ts, tmp_path):
        """info={'prior':'adaptive', 'model':'K2', 'kappa': 2.1} should land in header + each record."""
        out_vcf = tmp_path / "with_info.vcf"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        VCFWriter(in_vcf, out_vcf).write(
            pairs,
            info={"prior": "adaptive", "model": "K2", "kappa": 2.13},
        )

        header_text = Path(out_vcf).read_text()
        assert "ID=AA_prior," in header_text
        assert "ID=AA_model," in header_text
        assert "ID=AA_kappa," in header_text
        # Float gets Type=Float. String gets Type=String.
        assert 'ID=AA_kappa,Number=1,Type=Float' in header_text
        assert 'ID=AA_prior,Number=1,Type=String' in header_text

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        for v in recovered:
            assert v.INFO.get("AA_prior") == "adaptive"
            assert v.INFO.get("AA_model") == "K2"
            assert pytest.approx(v.INFO.get("AA_kappa"), rel=1e-4) == 2.13

    @pytest.mark.parametrize("value,expected", [
        (True, "Integer"),  # bool before int
        (5, "Integer"),
        (1.5, "Float"),
        ("x", "String"),
        ([1, 2], "String"),  # anything else stringifies
    ])
    def test_vcf_type_token(self, value, expected):
        assert VCFWriter._vcf_type_of(value) == expected

    def test_coerce_info_integer(self):
        assert VCFWriter._coerce_info_value("5", "Integer") == 5

    def test_coerce_info_float(self):
        assert VCFWriter._coerce_info_value("1.5", "Float") == 1.5

    def test_coerce_info_string(self):
        assert VCFWriter._coerce_info_value(5, "String") == "5"


class TestReservedInfoKeys:
    """``info`` keys must not collide with the writer's own INFO fields."""

    @pytest.mark.parametrize("key", ["prob", "post"])
    def test_reserved_key_raises(self, tmp_path, key):
        template = tmp_path / "in.vcf"
        template.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1,length=1000>\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t5\t.\tA\tT\t.\t.\t.\n"
        )
        out = str(tmp_path / "out.vcf")
        site = Site(chrom="1", pos=5, alleles=("A", "T"), tip_alleles={})
        with pytest.raises(ValueError, match="reserved INFO field"):
            VCFWriter(str(template), out).write(
                iter([(site, _post("A", 0.9))]), info={key: 3.5},
            )


def test_a_missing_backend_names_the_writer_and_the_module(tmp_path):
    writer = VCFWriter(tmp_path / "in.vcf", tmp_path / "out.vcf")
    with pytest.raises(ImportError, match="VCFWriter requires ancestree_no_such"):
        writer._require_backend("ancestree_no_such_backend", "pip install x")


class TestTskitWriter:
    """:class:`TskitWriter` should set ``site.ancestral_state`` from MAP."""

    def test_ancestral_state_is_map_allele(self, hap_ts, tmp_path):
        out_ts = tmp_path / "annot.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        n = TskitWriter(hap_ts, out_ts).write(pairs)
        assert n == len(sites)

        import tskit
        new_ts = tskit.load(str(out_ts))
        expected_map = {int(site.pos): post.map_allele for site, post in pairs}
        for site in new_ts.sites():
            assert site.ancestral_state == expected_map[int(site.position)]

    def test_posterior_stored_in_metadata(self, hap_ts, tmp_path):
        out_ts = tmp_path / "with_meta.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        TskitWriter(hap_ts, out_ts).write(pairs, store_posterior=True)

        import tskit
        new_ts = tskit.load(str(out_ts))
        first = next(iter(new_ts.sites()))
        assert "ancestree" in first.metadata
        block = first.metadata["ancestree"]
        assert "map_allele" in block
        assert "max_prob" in block
        assert "posterior" in block
        assert pytest.approx(sum(block["posterior"].values()), rel=1e-6) == 1.0

    def test_store_posterior_false_omits_full_posterior(self, hap_ts, tmp_path):
        out_ts = tmp_path / "no_post.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        TskitWriter(hap_ts, out_ts).write(pairs, store_posterior=False)

        import tskit
        new_ts = tskit.load(str(out_ts))
        first = next(iter(new_ts.sites()))
        block = first.metadata["ancestree"]
        assert "posterior" not in block
        assert "map_allele" in block

    def test_unannotated_sites_keep_original(self, hap_ts, tmp_path):
        """Sites with no matching posterior keep their original ancestral_state."""
        out_ts = tmp_path / "partial.trees"
        sites = list(TskitSource(hap_ts))
        partial = _fake_posteriors(sites[:1])
        TskitWriter(hap_ts, out_ts).write(partial)

        import tskit
        new_ts = tskit.load(str(out_ts))
        original_states = {int(s.position): s.ancestral_state for s in hap_ts.sites()}
        first_pos = sites[0].pos
        for site in new_ts.sites():
            if int(site.position) == first_pos:
                # annotated → MAP allele (first allele in our fake construction)
                assert site.ancestral_state == sites[0].alleles[0]
            else:
                assert site.ancestral_state == original_states[int(site.position)]

    def test_accepts_path_input(self, hap_ts, tmp_path):
        """``input_ts`` may also be a path to a ``.trees`` file."""
        src_path = tmp_path / "src.trees"
        hap_ts.dump(str(src_path))
        out_path = tmp_path / "from_path.trees"
        pairs = _fake_posteriors(list(TskitSource(hap_ts)))
        n = TskitWriter(src_path, out_path).write(pairs)
        assert n == len(pairs)

    def test_info_kwarg_lands_in_metadata(self, hap_ts, tmp_path):
        """``info={...}`` should appear under ``metadata['ancestree']['inference']``."""
        out_ts = tmp_path / "with_info.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        TskitWriter(hap_ts, out_ts).write(
            pairs, info={"prior": "kingman", "model": "JC"},
        )

        import tskit
        new_ts = tskit.load(str(out_ts))
        first = next(iter(new_ts.sites()))
        assert first.metadata["ancestree"]["inference"] == {
            "prior": "kingman", "model": "JC",
        }


class TestTskitWriterMetadata:
    """Site metadata of any shape is coerced to a dict the block can merge into."""

    def test_a_dict_is_copied(self):
        meta = {"k": 1}
        out = TskitWriter._coerce_metadata_to_dict(meta)
        assert out == meta and out is not meta

    def test_json_bytes_are_parsed(self):
        assert TskitWriter._coerce_metadata_to_dict(b'{"k": 1}') == {"k": 1}

    @pytest.mark.parametrize("raw", [b"", b"not json", b"[1, 2]", "text", None])
    def test_anything_else_collapses_to_an_empty_dict(self, raw):
        assert TskitWriter._coerce_metadata_to_dict(raw) == {}


# -------------------------------------------------- min_confidence threshold


def _uniform_posteriors(sites: list[Site]) -> list[tuple[Site, Posterior]]:
    """Manufacture per-site posteriors that are exactly uniform.

    For any site with ``n`` alleles, every entry is ``1/n`` so
    ``max_prob = 1/n``, easy to position above or below an arbitrary
    confidence threshold.
    """
    pairs: list[tuple[Site, Posterior]] = []
    for site in sites:
        n = len(site.alleles)
        values = np.full(n, 1.0 / n)
        pairs.append((site, Posterior(alleles=tuple(site.alleles), values=values)))
    return pairs


def _two_site_continuous_ts():
    """Tiny 2-sample tree sequence with two sites whose float positions
    (``100.2``, ``100.8``) truncate to the *same* integer (``100``)."""
    import tskit

    tables = tskit.TableCollection(sequence_length=200.0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(flags=0, time=1.0)
    tables.edges.add_row(left=0, right=200.0, parent=root, child=0)
    tables.edges.add_row(left=0, right=200.0, parent=root, child=1)
    s0 = tables.sites.add_row(position=100.2, ancestral_state="A")
    s1 = tables.sites.add_row(position=100.8, ancestral_state="A")
    tables.mutations.add_row(site=s0, node=0, derived_state="T")
    tables.mutations.add_row(site=s1, node=1, derived_state="G")
    tables.sort()
    return tables.tree_sequence()


def _one_site_tree_sequence(*, with_edges: bool) -> tskit.TreeSequence:
    """A two-sample tree sequence carrying one site at position 5."""
    tables = tskit.TableCollection(sequence_length=10.0)
    a = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    b = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    if with_edges:
        root = tables.nodes.add_row(flags=0, time=1.0)
        for child in (a, b):
            tables.edges.add_row(left=0, right=10.0, parent=root, child=child)
    site = tables.sites.add_row(position=5.0, ancestral_state="A")
    if with_edges:
        tables.mutations.add_row(site=site, node=a, derived_state="T")
    tables.sort()
    return tables.tree_sequence()


class TestTskitWriterAllMissingSite:
    """An isolated site has no non-missing genotype for ``map_mutations``."""

    def test_all_missing_site_does_not_abort_the_write(self, tmp_path):
        out = str(tmp_path / "out.trees")
        # No edges, so every sample is isolated and decodes to missing.
        input_ts = _one_site_tree_sequence(with_edges=False)
        site = Site(chrom="1", pos=5, alleles=("A", "T"), tip_alleles={})
        TskitWriter(input_ts, out).write(iter([(site, _post("T", 0.95))]))
        written = tskit.load(out)
        assert written.site(0).ancestral_state == "T"
        assert written.num_mutations == 0


class TestBlankedTreesSiteReadsBackUncalled:
    """A sub-threshold ``.trees`` call must read back as uncalled, like VCF."""

    def test_sub_threshold_site_reads_back_as_none(self, tmp_path):
        out = str(tmp_path / "out.trees")
        input_ts = _one_site_tree_sequence(with_edges=True)
        site = Site(chrom="1", pos=5, alleles=("A", "T"), tip_alleles={})
        TskitWriter(input_ts, out, min_confidence=0.9).write(
            iter([(site, _post("A", 0.4))])
        )
        annotation = Reader(out).head(1)[0]
        assert annotation.aa is None
        assert annotation.aa_prob == pytest.approx(0.4)

    def test_confident_site_still_reads_back_called(self, tmp_path):
        out = str(tmp_path / "out.trees")
        input_ts = _one_site_tree_sequence(with_edges=True)
        site = Site(chrom="1", pos=5, alleles=("A", "T"), tip_alleles={})
        TskitWriter(input_ts, out, min_confidence=0.5).write(
            iter([(site, _post("A", 0.95))])
        )
        assert Reader(out).head(1)[0].aa == "A"


class TestTskitWriterPositionCollision:
    """Two sites sharing an integer position must not collapse to a
    single posterior (the writer keys on the exact float position)."""

    def test_distinct_float_positions_get_distinct_posteriors(self, tmp_path):
        import tskit

        ts = _two_site_continuous_ts()
        sites = list(TskitSource(ts))
        assert sites[0].pos == sites[1].pos == 100  # both truncate to 100
        assert sites[0].local_tree_handle != sites[1].local_tree_handle

        # Distinct MAP calls per site: 100.2 → T, 100.8 → G.
        p0 = Posterior(alleles=("A", "T"), values=np.array([0.1, 0.9]))
        p1 = Posterior(alleles=("A", "G"), values=np.array([0.1, 0.9]))
        out_ts = tmp_path / "collision.trees"
        n = TskitWriter(ts, out_ts).write([(sites[0], p0), (sites[1], p1)])
        assert n == 2

        new_ts = tskit.load(str(out_ts))
        by_pos = {s.position: s.ancestral_state for s in new_ts.sites()}
        assert by_pos[100.2] == "T"
        assert by_pos[100.8] == "G"


class TestVCFWriterStreamingParity:
    """The streaming two-pass writer must match the buffered path byte-for-byte.

    ``VCFWriter.write`` streams posteriors into template rows (bounded memory)
    for a re-openable file template, falling back to buffering only for a
    non-seekable stdin/pipe template. The two paths must produce identical
    output, including for multiallelic-split records that share ``(CHROM, POS)``.
    """

    def _write_multiallelic_vcf(self, path):
        # Two records at the same CHROM/POS (a multiallelic site split across
        # lines), plus two ordinary biallelic sites.
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=1>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1",
            "1\t10\t.\tA\tC\t.\tPASS\t.\tGT\t0\t1",
            "1\t20\t.\tG\tT\t.\tPASS\t.\tGT\t0\t1",
            "1\t20\t.\tG\tC\t.\tPASS\t.\tGT\t1\t0",
            "1\t30\t.\tT\tA\t.\tPASS\t.\tGT\t0\t1",
        ]
        path.write_text("\n".join(lines) + "\n")

    def _posteriors(self):
        return [
            (Site(chrom="1", pos=10, alleles=("A", "C"),
                  tip_alleles={"s0": "A", "s1": "C"}),
             Posterior(alleles=("A", "C"), values=np.array([0.9, 0.1]))),
            (Site(chrom="1", pos=20, alleles=("G", "T"),
                  tip_alleles={"s0": "G", "s1": "T"}),
             Posterior(alleles=("G", "T"), values=np.array([0.2, 0.8]))),
            (Site(chrom="1", pos=30, alleles=("T", "A"),
                  tip_alleles={"s0": "T", "s1": "A"}),
             Posterior(alleles=("T", "A"), values=np.array([0.7, 0.3]))),
        ]

    def test_a_split_record_takes_one_call_per_row(self, tmp_path):
        """One posterior claims one row of a split-multiallelic run.

        Broadcasting it across the run made the first site at a position stamp
        its allele onto every row there and discarded the sites behind it,
        which is what a VCF source yields for a split multiallelic: one Site
        per line. Downstream that flipped the derived count from k to n-k.
        """
        in_vcf = tmp_path / "in.vcf"
        self._write_multiallelic_vcf(in_vcf)
        out_vcf = tmp_path / "out.vcf"
        # write() picks the streaming path for a file template.
        n = VCFWriter(in_vcf, out_vcf).write(self._posteriors())

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        at_20 = [v for v in recovered if v.POS == 20]
        assert len(at_20) == 2, "expected both split records at POS 20"
        called = [v for v in at_20 if v.INFO.get("AA") is not None]
        assert len(called) == 1
        assert called[0].INFO.get("AA") == "T"
        # VCF stores AA_prob as a 4-byte Float, so compare at float32 tolerance.
        assert abs(float(called[0].INFO.get("AA_prob")) - 0.8) < 1e-6
        assert n == len(recovered) - 1

    def test_two_sites_at_one_position_keep_their_own_calls(self, tmp_path):
        in_vcf = tmp_path / "in.vcf"
        self._write_multiallelic_vcf(in_vcf)
        out_vcf = tmp_path / "out.vcf"
        pairs = list(self._posteriors())
        pairs.insert(2, (
            Site(chrom="1", pos=20, alleles=("G", "C"),
                 tip_alleles={"s0": "C", "s1": "G"}),
            Posterior(alleles=("G", "C"), values=np.array([0.35, 0.65])),
        ))
        n = VCFWriter(in_vcf, out_vcf).write(pairs)

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        at_20 = [v for v in recovered if v.POS == 20]
        assert [v.INFO.get("AA") for v in at_20] == ["T", "C"]
        assert n == len(recovered)


class TestVCFWriterContigAndWarning:
    """``ARGBasedInference.to_vcf`` writes the inference's own ``chrom``, and
    the writer refuses a run in which nothing matched."""

    def test_to_vcf_uses_inference_chrom(self, hap_ts, tmp_path):
        from ancestree import ARGBasedInference, JC69

        inf = ARGBasedInference(hap_ts, JC69(), mu=5e-8, chrom="chr20", progress=False)
        out_vcf = tmp_path / "auto.vcf"
        n = inf.to_vcf(out_vcf)
        assert n > 0
        recovered = list(cyvcf2.VCF(str(out_vcf)))
        assert recovered, "expected records in the auto-written VCF"
        assert all(v.CHROM == "chr20" for v in recovered)
        annotated = [v for v in recovered if v.INFO.get("AA") is not None]
        assert len(annotated) == n

    def test_writer_refuses_when_nothing_matches(self, hap_ts, tmp_path):
        """A wholly unannotated output must fail, not exit 0.

        A contig-label mismatch otherwise yields a complete, correctly-headed
        VCF carrying no AA at all, which is indistinguishable from success to
        a Snakemake DAG or any ``set -e`` pipeline.
        """
        # Template on contig "1". Posteriors carry a non-matching chrom.
        template = tmp_path / "tmpl.vcf"
        with open(template, "w") as f:
            hap_ts.write_vcf(f, contig_id="1")
        sites = [
            Site(chrom="nomatch", pos=s.pos, alleles=s.alleles,
                 tip_alleles=s.tip_alleles)
            for s in TskitSource(hap_ts)
        ]
        pairs = _fake_posteriors(sites)
        out_vcf = tmp_path / "out.vcf"
        with pytest.raises(ValueError, match="matched 0 of"):
            VCFWriter(template, out_vcf).write(pairs)
        # The destination must not be left holding the unannotated output.
        assert not out_vcf.exists()


class TestMinConfidence:
    """``min_confidence=`` blanks the MAP call when ``max_prob`` is below the threshold."""

    @pytest.fixture
    def in_vcf(self, hap_ts, tmp_path_factory):
        p = tmp_path_factory.mktemp("inv_thresh") / "in.vcf"
        with open(p, "w") as f:
            hap_ts.write_vcf(f, contig_id="1")
        return p

    def test_vcf_writes_dot_when_below_threshold(self, in_vcf, hap_ts, tmp_path):
        """Uniform posterior over ≥2 alleles → ``max_prob ≤ 0.5`` → ``AA=.``."""
        out_vcf = tmp_path / "low_conf.vcf"
        sites = list(TskitSource(hap_ts))
        pairs = _uniform_posteriors(sites)
        n = VCFWriter(in_vcf, out_vcf, min_confidence=0.95).write(pairs)
        assert n == len(sites)

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        # Every annotated record should carry the unknown sentinel.
        for variant in recovered:
            assert variant.INFO.get("AA") == "."
            # AA_prob still reflects the (sub-threshold) max probability.
            assert 0.0 <= float(variant.INFO.get("AA_prob")) <= 1.0

    def test_vcf_writes_map_when_above_threshold(self, in_vcf, hap_ts, tmp_path):
        """Strongly biased posterior (0.85) → above threshold → MAP allele emitted."""
        out_vcf = tmp_path / "high_conf.vcf"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)  # 0.85 mass on first allele
        VCFWriter(in_vcf, out_vcf, min_confidence=0.5).write(pairs)

        recovered = list(cyvcf2.VCF(str(out_vcf)))
        for variant in recovered:
            aa = variant.INFO.get("AA")
            assert aa in {"A", "C", "G", "T"}
            assert aa != "."

    def test_tskit_writes_empty_string_when_below_threshold(self, hap_ts, tmp_path):
        """Uniform posterior → ``ancestral_state == ""`` (tskit "not annotated")."""
        out_ts = tmp_path / "low_conf.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _uniform_posteriors(sites)
        n = TskitWriter(hap_ts, out_ts, min_confidence=0.95).write(pairs)
        assert n == len(sites)

        import tskit
        new_ts = tskit.load(str(out_ts))
        for site in new_ts.sites():
            assert site.ancestral_state == ""
            # Posterior block is still written.
            assert "ancestree" in site.metadata

    def test_tskit_writes_map_when_above_threshold(self, hap_ts, tmp_path):
        """Strongly biased posterior → MAP allele lands in ``ancestral_state``."""
        out_ts = tmp_path / "high_conf.trees"
        sites = list(TskitSource(hap_ts))
        pairs = _fake_posteriors(sites)
        TskitWriter(hap_ts, out_ts, min_confidence=0.5).write(pairs)

        import tskit
        new_ts = tskit.load(str(out_ts))
        expected_map = {int(site.pos): post.map_allele for site, post in pairs}
        for site in new_ts.sites():
            assert site.ancestral_state == expected_map[int(site.position)]
            assert site.ancestral_state != ""


class TestTskitRepolarisation:
    """Re-polarising a site must not change what its samples carry."""

    @staticmethod
    def _alleles(ts):
        """Per-site tuples of the allele each sample carries."""
        return [tuple(v.alleles[g] for g in v.genotypes) for v in ts.variants()]

    def test_genotypes_survive(self, hap_ts, tmp_path):
        """Every sample decodes to the same allele after annotation."""
        import tskit
        from ancestree.posterior import Posterior

        sites = list(TskitSource(hap_ts))
        # Call the allele no sample carries at the root, forcing every site to flip.
        pairs = [
            (site, Posterior(alleles=("A", "C", "G", "T"),
                             values=_flipped_values(site)))
            for site in sites
        ]
        out = tmp_path / "flipped.trees"
        TskitWriter(hap_ts, out).write(pairs)
        annotated = tskit.load(str(out))
        assert self._alleles(annotated) == self._alleles(hap_ts)

    def test_polymorphism_is_not_lost(self, hap_ts, tmp_path):
        """A flip must not collapse a segregating site to a single allele."""
        import tskit
        from ancestree.posterior import Posterior

        sites = list(TskitSource(hap_ts))
        pairs = [
            (site, Posterior(alleles=("A", "C", "G", "T"),
                             values=_flipped_values(site)))
            for site in sites
        ]
        out = tmp_path / "flipped.trees"
        TskitWriter(hap_ts, out).write(pairs)
        annotated = tskit.load(str(out))
        for before, after in zip(self._alleles(hap_ts), self._alleles(annotated)):
            assert len(set(after)) == len(set(before))

    def test_ancestral_state_is_the_map_allele(self, hap_ts, tmp_path):
        """The flip still lands in ``ancestral_state``."""
        import tskit
        from ancestree.posterior import Posterior

        sites = list(TskitSource(hap_ts))
        pairs = [
            (site, Posterior(alleles=("A", "C", "G", "T"),
                             values=_flipped_values(site)))
            for site in sites
        ]
        out = tmp_path / "flipped.trees"
        TskitWriter(hap_ts, out).write(pairs)
        annotated = tskit.load(str(out))
        for site in annotated.sites():
            assert site.ancestral_state == site.metadata["ancestree"]["map_allele"]


def _flipped_values(site):
    """A posterior peaked on an allele other than the site's first."""
    import numpy as np

    observed = [a for a in site.alleles if a in ("A", "C", "G", "T")]
    target = next(a for a in ("A", "C", "G", "T") if a in observed[1:]) \
        if len(observed) > 1 else observed[0]
    values = np.full(4, 0.01)
    values["ACGT".index(target)] = 0.97
    return values / values.sum()


class TestFullPosteriorInEveryFormat:
    """Every output format carries the whole posterior, not just the MAP."""

    @staticmethod
    def _written(inference, path):
        """Annotate ``path`` in the format its suffix implies."""
        {".gz": inference.to_vcf, ".vcz": inference.to_zarr,
         ".trees": inference.to_arg}[path.suffix](path)
        return path

    def test_formats_agree_on_the_posterior(self, small_ts, tmp_path):
        """The per-state probabilities round-trip identically from all three."""
        import ancestree as anc

        inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
        heads = [
            anc.Reader(self._written(inference, tmp_path / f"a{suffix}")).head(3)
            for suffix in (".vcf.gz", ".vcz", ".trees")
        ]
        for records in zip(*heads):
            assert all(r.posterior is not None for r in records)
            for state in anc.STATES:
                probs = [r.posterior[state] for r in records]
                assert max(probs) - min(probs) < 1e-6

    def test_vcf_declares_the_field(self, small_ts, tmp_path):
        """``AA_post`` is declared in the header with one value per state."""
        import cyvcf2
        import ancestree as anc

        inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
        out = self._written(inference, tmp_path / "a.vcf.gz")
        field = cyvcf2.VCF(str(out)).get_header_type("AA_post")
        assert field["Number"] == str(len(anc.STATES))
        assert field["Type"] == "Float"

    def test_posterior_sums_to_one(self, small_ts, tmp_path):
        """What a VCF stores is a distribution, not just the MAP mass."""
        import ancestree as anc

        inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
        out = self._written(inference, tmp_path / "a.vcf.gz")
        for record in anc.Reader(out).head(5):
            assert abs(sum(record.posterior.values()) - 1.0) < 1e-4
            assert record.posterior[record.aa] == max(record.posterior.values())


class TestStorePosteriorSymmetry:
    """``store_posterior`` behaves the same across all three output formats."""

    @staticmethod
    def _write(inference, path, store_posterior):
        """Annotate ``path`` in the format its suffix implies."""
        {".gz": inference.to_vcf, ".vcz": inference.to_zarr,
         ".trees": inference.to_arg}[path.suffix](
            path, store_posterior=store_posterior)
        return path

    @pytest.mark.parametrize("suffix", [".vcf.gz", ".vcz", ".trees"])
    @pytest.mark.parametrize("store", [True, False])
    def test_flag_is_honoured(self, small_ts, tmp_path, suffix, store):
        """The posterior is present exactly when the flag asks for it."""
        import ancestree as anc

        inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
        out = self._write(inference, tmp_path / f"a{suffix}", store)
        first = anc.Reader(out).head(1)[0]
        assert (first.posterior is not None) is store
        # The call itself survives either way.
        assert first.aa is not None and first.aa_prob is not None

    @pytest.mark.parametrize("suffix", [".vcf.gz", ".vcz", ".trees"])
    def test_the_call_does_not_depend_on_the_flag(self, small_ts, tmp_path,
                                                  suffix):
        """Every call and its probability match across the two flag settings.

        Protects against a writer that fabricates ``AA_prob`` when the
        posterior matrix is absent: presence assertions alone pass while the
        stored value is a constant.
        """
        import ancestree as anc

        def calls(store):
            inference = anc.Inference.from_arg(
                small_ts, anc.JC69(), mu=5e-8, progress=False)
            out = self._write(
                inference, tmp_path / f"{'on' if store else 'off'}{suffix}",
                store)
            return [(a.pos, a.aa, a.aa_prob)
                    for a in anc.Reader(out).annotations()]

        on, off = calls(True), calls(False)
        assert len(on) == len(off) and on
        assert [(p, a) for p, a, _ in on] == [(p, a) for p, a, _ in off]
        for (_, _, p_on), (_, _, p_off) in zip(on, off):
            assert p_on == pytest.approx(p_off, rel=1e-6)

    def test_vcf_header_follows_the_flag(self, small_ts, tmp_path):
        """``AA_post`` is declared only when it is written."""
        import gzip

        import ancestree as anc

        inference = anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)
        for store in (True, False):
            out = self._write(inference, tmp_path / f"h{store}.vcf.gz", store)
            declared = any(
                line.startswith("##INFO=<ID=AA_post")
                for line in gzip.open(out, "rt")
            )
            assert declared is store


class TestSubThresholdSitesKeepTheirGenotypes:
    """Blanking a site's ancestral state must not change what its samples carry."""

    @staticmethod
    def _alleles(ts):
        """Per-site tuples of the allele each sample carries."""
        return [tuple(v.alleles[g] for g in v.genotypes) for v in ts.variants()]

    def test_genotypes_survive_blanking(self, hap_ts, tmp_path):
        """Every sample decodes as before, though the site is marked uncalled.

        ``ancestral_state`` is allele 0, so blanking it without re-deriving the
        mutations left every sample that carried the ancestral allele decoding
        as the empty allele instead.
        """
        import tskit
        import ancestree as anc

        out = tmp_path / "blank.trees"
        anc.Inference.from_arg(hap_ts, anc.JC69(), mu=5e-8, progress=False).to_arg(
            out, min_confidence=1.1,  # above 1, so every site falls short
        )
        annotated = tskit.load(str(out))
        assert self._alleles(annotated) == self._alleles(hap_ts)

    def test_the_blank_marker_is_still_written(self, hap_ts, tmp_path):
        """The uncalled marker stays in the ancestral state, where readers expect it."""
        import tskit
        import ancestree as anc

        out = tmp_path / "blank.trees"
        anc.Inference.from_arg(hap_ts, anc.JC69(), mu=5e-8, progress=False).to_arg(
            out, min_confidence=1.1,
        )
        annotated = tskit.load(str(out))
        assert all(site.ancestral_state == "" for site in annotated.sites())
        assert all("ancestree" in site.metadata for site in annotated.sites())


def _unrepresentable_site_ts() -> tskit.TreeSequence:
    """Two-sample tree sequence with a readable site and an all-indel one.

    Position 2 carries ``A`` / ``T``. Position 6 carries ``AT`` / ``ATT``, so
    no allele there maps to a model state.
    """
    tables = tskit.TableCollection(sequence_length=10.0)
    a = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    b = tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    root = tables.nodes.add_row(flags=0, time=1.0)
    for child in (a, b):
        tables.edges.add_row(left=0, right=10.0, parent=root, child=child)
    readable = tables.sites.add_row(position=2.0, ancestral_state="A")
    tables.mutations.add_row(site=readable, node=a, derived_state="T")
    opaque = tables.sites.add_row(position=6.0, ancestral_state="AT")
    tables.mutations.add_row(site=opaque, node=a, derived_state="ATT")
    tables.sort()
    return tables.tree_sequence()


class TestSiteWithNoReadableAlleleIsNotCalled:
    """A site whose every allele is outside A/C/G/T must read back uncalled.

    Every tip there is marginalised, so the posterior is the prior, the
    four-way tie broke to ``"A"`` on ``argmax``, and each writer committed
    that ``"A"`` at 0.25 confidence at a site no sample carried ``A`` at.
    """

    @pytest.fixture
    def annotations(self, tmp_path, request):
        import ancestree as anc

        inference = anc.Inference.from_arg(
            _unrepresentable_site_ts(), anc.JC69(), mu=1e-8, progress=False)
        out = tmp_path / f"a{request.param}"
        {".gz": inference.to_vcf, ".vcz": inference.to_zarr,
         ".trees": inference.to_arg}[out.suffix](out)
        return {r.pos: r for r in Reader(out).head(5)}

    @pytest.mark.parametrize(
        "annotations", [".vcf.gz", ".vcz", ".trees"], indirect=True)
    def test_the_opaque_site_carries_no_allele_call(self, annotations):
        assert annotations[6].aa is None

    @pytest.mark.parametrize(
        "annotations", [".vcf.gz", ".vcz", ".trees"], indirect=True)
    def test_the_readable_site_is_still_called(self, annotations):
        assert annotations[2].aa == "T"

    def test_the_trees_output_keeps_no_fabricated_ancestral_state(self, tmp_path):
        """tskit's own ``ancestral_state`` field, not just what the reader shows."""
        import ancestree as anc

        out = tmp_path / "a.trees"
        anc.Inference.from_arg(
            _unrepresentable_site_ts(), anc.JC69(), mu=1e-8, progress=False,
        ).to_arg(out)
        states = [s.ancestral_state for s in tskit.load(str(out)).sites()]
        assert states[0] == "T"
        assert states[1] == ""


class TestRestrictSamples:
    """An annotated file keeps every sample unless ``restrict_samples`` is set.
    An output written from the sites holds the samples the inference used."""

    @staticmethod
    def _inference(**kw):
        return anc.Inference.from_arg(
            tskit.load(QUICKSTART_TREES), mu=5e-8, progress=False,
            **{"ingroup_samples": ING, "outgroup_samples": OUT, **kw})

    @pytest.mark.parametrize("restrict", [False, True])
    def test_vcf_from_the_sites_holds_the_panel(self, tmp_path, restrict):
        out = str(tmp_path / "out.vcf")
        self._inference().to_vcf(out, restrict_samples=restrict)
        assert cyvcf2.VCF(out).samples == ING + OUT

    @pytest.mark.parametrize("restrict, n", [(False, 8), (True, 5)])
    def test_arg(self, tmp_path, restrict, n):
        out = tmp_path / "out.trees"
        self._inference().to_arg(out, restrict_samples=restrict)
        assert tskit.load(out).num_samples == n

    def test_zarr_keeps_the_genotypes_of_the_kept_columns(self, tmp_path):
        import zarr

        full, sub = str(tmp_path / "full.vcz"), str(tmp_path / "sub.vcz")
        self._inference(ingroup_samples=None, outgroup_samples=None).to_zarr(full)
        self._inference().to_zarr(sub, input_zarr=full, restrict_samples=True)
        a, b = zarr.open(full, mode="r"), zarr.open(sub, mode="r")
        names = [str(s) for s in a["sample_id"][:]]
        keep = [names.index(s) for s in ING + OUT]
        assert [str(s) for s in b["sample_id"][:]] == ING + OUT
        np.testing.assert_array_equal(
            b["call_genotype"][:], a["call_genotype"][:][:, keep])
        assert len(a["sample_id"][:]) == 8


def test_an_explicit_partial_sample_map_names_the_columns(tmp_path):
    ts = tskit.load(QUICKSTART_TREES)
    smap = {s: n for s, n in TskitLocalTree.default_sample_map(ts).items()
            if s in ING + OUT}
    inf = anc.Inference.from_arg(ts, mu=5e-8, sample_map=smap,
                                 ingroup_samples=ING,
                                 outgroup_samples=OUT, progress=False)
    out = str(tmp_path / "out.vcf")
    inf.to_vcf(out)
    assert cyvcf2.VCF(out).samples == ING + OUT


class TestPrebuiltLocalTreeWritesTheInput:
    """A pre-built tree sequence is written as ARG mode writes it: a ``.trees``
    output annotates it, a VCF is written from the sites."""

    @staticmethod
    def _inference():
        return LocalTreeInference(tskit.load(QUICKSTART_TREES), mu=5e-8,
                                  ingroup_samples=ING, outgroup_samples=OUT,
                                  progress=False)

    @pytest.mark.parametrize("restrict, n", [(False, 8), (True, 5)])
    def test_arg(self, tmp_path, restrict, n):
        out = tmp_path / "out.trees"
        self._inference().to_arg(out, restrict_samples=restrict)
        assert tskit.load(out).num_samples == n

    def test_vcf(self, tmp_path):
        out = str(tmp_path / "out.vcf")
        self._inference().to_vcf(out)
        assert cyvcf2.VCF(out).samples == ING + OUT


def test_unnamed_samples_are_written_under_their_node_ids(tmp_path):
    """Samples of unnamed individuals are named by node id, so each is its own
    column under the name the sample lists use."""
    ts = msprime.sim_ancestry(4, sequence_length=1e4, population_size=1e4,
                              random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1e-7, model=msprime.JC69(),
                               random_seed=2)
    inf = anc.Inference.from_arg(ts, mu=5e-8, progress=False,
                                 ingroup_samples=["0", "1", "2", "3"],
                                 outgroup_samples=["6", "7"])
    out = str(tmp_path / "out.vcf")
    inf.to_vcf(out)
    assert cyvcf2.VCF(out).samples == ["0", "1", "2", "3", "6", "7"]


@pytest.mark.parametrize("suffix, magic", [
    (".vcf.bgz", b"\x1f\x8b"), (".BCF", b"BCF"), (".vcf", b"##fileformat"),
])
def test_the_output_format_follows_the_suffix(tmp_path, suffix, magic):
    import gzip

    out = tmp_path / f"out{suffix}"
    TestRestrictSamples._inference().to_vcf(str(out), store_posterior=False)
    head = out.read_bytes()
    if suffix == ".BCF":
        head = gzip.decompress(head)
    assert head.startswith(magic)


def _remote_inference(path):
    """A local-tree inference over toy sites read from a remote ``path``."""
    sites, names = toy_sites(range(0, 1000, 20))

    class Remote:
        _path = path

        def samples(self):
            return names

        def __iter__(self):
            return iter(sites)

    return LocalTreeInference(Remote(), mu=5e-8, rec_rate=1e-8, window=200,
                              block_size=50, sequence_length=1000.0,
                              chunk_size=None, n_ensemble=None, progress=False)


def test_a_restricted_arg_keeps_its_sample_names(tmp_path):
    """Samples named after their node ids keep those names in a restricted
    ``.trees`` output, matching the provenance and the VCF columns."""
    ts = msprime.sim_ancestry(8, ploidy=1, sequence_length=1e4,
                              population_size=1e4, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1e-7, model=msprime.JC69(),
                               random_seed=2)
    inf = anc.Inference.from_arg(ts, mu=5e-8, progress=False,
                                 ingroup_samples=["0", "1", "2", "3"],
                                 outgroup_samples=["6", "7"])
    out = tmp_path / "out.trees"
    inf.to_arg(out, restrict_samples=True)
    names = TskitLocalTree.default_sample_map(tskit.load(out))
    assert list(names) == ["0", "1", "2", "3", "6", "7"]


def test_a_restricted_arg_keeps_the_haplotype_names_of_a_split_individual(
        tmp_path):
    """A diploid of which one haplotype is kept read back under the bare
    individual name, which the run's own sample lists do not match."""
    inf = anc.Inference.from_arg(DEMO_TREES, mu=5e-8, progress=False,
                                 ingroup_samples=["i0", "i1_h0"],
                                 outgroup_samples=["o1_h1"])
    out = tmp_path / "out.trees"
    inf.to_arg(out, restrict_samples=True)
    names = TskitLocalTree.default_sample_map(tskit.load(out))
    assert list(names) == ["i0_h0", "i0_h1", "i1_h0", "o1_h1"]
    anc.Inference.from_arg(str(out), mu=5e-8, ingroup_samples=["i0", "i1_h0"],
                           outgroup_samples=["o1_h1"])


def test_a_store_without_a_suffix_is_a_store(tmp_path):
    import bio2zarr.vcf as bio2zarr_vcf

    from ancestree.sources import VcfZarrSource
    from testing._helpers import DEMO_VCF

    store = str(tmp_path / "panel_store")
    bio2zarr_vcf.convert([DEMO_VCF], store, show_progress=False)
    inf = LocalTreeInference(VcfZarrSource(store), mu=5e-8, rec_rate=1e-8,
                             sequence_length=2e5, chunk_size=None,
                             n_ensemble=None, progress=False)
    assert (inf._input_store_path, inf._input_vcf_path) == (store, None)


@pytest.mark.parametrize("chunk_size", [None, 25_000])
def test_a_store_to_a_vcf_is_written_from_the_sites(tmp_path, chunk_size):
    """Every site is a record, under the relabelled contig. A template
    exported from the panel's trees lost the records of sites monomorphic
    within the panel, whose alleles the trees no longer carry."""
    from testing._helpers import ts_to_vcz

    ts = msprime.sim_ancestry(6, ploidy=1, sequence_length=5e4,
                              population_size=1e4, recombination_rate=1e-8,
                              random_seed=2)
    ts = msprime.sim_mutations(ts, rate=1e-7, model=msprime.JC69(),
                               random_seed=2)
    store = str(tmp_path / "snps.vcz")
    ts_to_vcz(ts, store)
    inf = LocalTreeInference(store, mu=1.25e-8, rec_rate=1e-8,
                             sequence_length=ts.sequence_length,
                             ingroup_samples=["n0", "n1", "n2"],
                             outgroup_samples=["n3"], chrom="chrX",
                             chunk_size=chunk_size, n_ensemble=None,
                             progress=False)
    res = list(inf.infer())
    panel = {"n0", "n1", "n2", "n3"}
    assert any(len({s.tip_alleles[n] for n in panel}) == 1 for s, _ in res)
    out = str(tmp_path / "out.vcf")
    assert inf.to_vcf(out, posteriors=res) == len(res) == ts.num_sites
    records = list(cyvcf2.VCF(out))
    assert cyvcf2.VCF(out).samples == ["n0", "n1", "n2", "n3"]
    assert {r.CHROM for r in records} == {"chrX"}
    assert all(r.INFO.get("AA") is not None for r in records)


def test_a_remote_store_is_written_from_the_sites(tmp_path):
    """A store named by URL cannot be copied, so the store is written from
    the sites."""
    inf = _remote_inference("https://host/demo.vcz?raw=true")
    assert inf._output_template(None, "vcz", "out.vcz", False)[0] is None


@pytest.mark.parametrize("out, annotated", [("out.vcf", True),
                                            ("out.vcz", False)])
def test_a_remote_vcf_is_annotated_only_as_a_vcf(tmp_path, out, annotated):
    """cyvcf2 reads a remote VCF, so a VCF output annotates it. Any other
    output is written from the sites, as for a local VCF."""
    url = "https://host/demo.vcf.gz?raw=true"
    inf = _remote_inference(url)
    template, _ = inf._output_template(None, out[-3:], out, False)
    assert (template == url) is annotated
    if not annotated:
        assert inf.to_zarr(str(tmp_path / out)) == len(list(inf.infer()))


def test_a_restricted_site_is_written_to_its_own_record(tmp_path):
    """Restriction keeps the record's alleles, so a SNP whose ALT only a
    dropped sample carries is not mistaken for an indel at its position."""
    template = tmp_path / "t.vcf"
    template.write_text(
        "##fileformat=VCFv4.2\n##contig=<ID=chr1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n"
        "chr1\t5\tindel\tC\tCT\t.\tPASS\t.\tGT\t0\t0\n"
        "chr1\t5\tsnp\tC\tA\t.\tPASS\t.\tGT\t0\t1\n")
    site = Site(chrom="chr1", pos=5, alleles=("C", "A"),
                tip_alleles={"a": "C", "b": "A"}).restricted_to({"a"})
    out = str(tmp_path / "out.vcf")
    assert VCFWriter(str(template), out).write([(site, _post("C", 0.9))]) == 1
    calls = {r.ID: r.INFO.get("AA") for r in cyvcf2.VCF(out)}
    assert calls == {"indel": None, "snp": "C"}


class TestWritingFromTheSites:
    """An output without a file to annotate is written from the sites."""

    PANEL = frozenset({"i0", "i1", "o1"})

    @staticmethod
    def _sites():
        from ancestree.sources import CyVCF2Source
        from testing._helpers import DEMO_VCF

        return list(CyVCF2Source(DEMO_VCF))[:300]

    @pytest.mark.parametrize("suffix", [".vcf.gz", ".vcz"])
    def test_reading_the_output_gives_back_the_sites(self, tmp_path, suffix):
        from ancestree.sites import _named
        from ancestree.sources import CyVCF2Source, VcfZarrSource
        from ancestree.writers import ZarrWriter

        sites = self._sites()
        out = str(tmp_path / f"out{suffix}")
        writer = VCFWriter if suffix == ".vcf.gz" else ZarrWriter
        writer(None, out, samples=self.PANEL).write(_fake_posteriors(sites))
        source = (CyVCF2Source if suffix == ".vcf.gz" else VcfZarrSource)(out)
        kept = [s.restricted_to({n for n in s.tip_alleles
                                 if _named(n, self.PANEL)}) for s in sites]
        assert list(source) == kept
        assert source.samples() == list(kept[0].tip_alleles)

    def test_the_annotations_match_those_of_an_annotated_file(self, tmp_path):
        """Both strategies write the same record for a site."""
        from testing._helpers import DEMO_VCF

        pairs = _fake_posteriors(self._sites())
        annotated, written = str(tmp_path / "a.vcf"), str(tmp_path / "w.vcf")
        VCFWriter(DEMO_VCF, annotated, samples=self.PANEL).write(pairs)
        VCFWriter(None, written, samples=self.PANEL).write(pairs)

        def records(path):
            return [(r.CHROM, r.POS, r.REF, r.ALT, r.genotypes,
                     r.INFO.get("AA"), r.INFO.get("AA_prob"),
                     r.INFO.get("AA_post"))
                    for r in cyvcf2.VCF(path) if r.INFO.get("AA") is not None]

        assert cyvcf2.VCF(annotated).samples == cyvcf2.VCF(written).samples
        assert records(annotated) == records(written)

    @pytest.mark.parametrize("suffix", [".vcf", ".vcz"])
    def test_an_unphased_site_is_written_unphased(self, tmp_path, suffix):
        import zarr

        from ancestree.writers import ZarrWriter

        tips = {"a_h0": "A", "a_h1": "G", "b_h0": "G", "b_h1": "G"}
        sites = [Site(chrom="1", pos=p, alleles=("A", "G"), tip_alleles=tips,
                      phased=phased)
                 for p, phased in ((1, True), (2, False))]
        out = str(tmp_path / f"out{suffix}")
        writer = VCFWriter if suffix == ".vcf" else ZarrWriter
        writer(None, out).write(_fake_posteriors(sites))
        if suffix == ".vcf":
            phases = [[g[-1] for g in r.genotypes] for r in cyvcf2.VCF(out)]
        else:
            phases = zarr.open(out, mode="r")["call_genotype_phased"][:].tolist()
        assert phases == [[True, True], [False, False]]

    @pytest.mark.parametrize("suffix", [".vcf", ".vcz"])
    def test_a_failed_write_leaves_nothing_behind(self, tmp_path, suffix):
        from ancestree.writers import ZarrWriter

        pairs = _fake_posteriors(self._sites())

        def stream():
            yield pairs[0]
            raise RuntimeError("stream failed")

        writer = VCFWriter if suffix == ".vcf" else ZarrWriter
        with pytest.raises(RuntimeError, match="stream failed"):
            writer(None, str(tmp_path / f"out{suffix}")).write(stream())
        assert list(tmp_path.iterdir()) == []
