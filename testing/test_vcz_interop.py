"""VCF Zarr (VCZ) interoperability with the reference ``bio2zarr`` / ``vcztools`` tools.

Both directions of the round-trip, over a store ``bio2zarr`` actually wrote (built here
from a small VCF, so the on-disk layout is the reference one rather than a hand-made
approximation):

  * :class:`~ancestree.sources.VcfZarrSource` reads a ``bio2zarr`` store, recovering the
    sites, alleles, and per-haplotype tip alleles. And
  * :class:`~ancestree.writers.ZarrWriter` annotates that store with ``variant_AA`` /
    ``variant_AA_prob``, and ``vcztools`` reads the result back to the expected VCF, with
    the ``AA`` / ``AA_prob`` ``INFO`` fields derived from the added arrays.

``bio2zarr`` (Python API, no ``bgzip`` / ``tabix`` binaries needed) and the ``vcztools``
console script are pulled in by the ``zarr`` extra and the dev/CI environment. This module
uses them directly and is meant to run there, so it is not guarded against their absence.
"""
from __future__ import annotations

import subprocess

import msprime
import pytest
import tskit
import ancestree as anc

from ancestree import ARGBasedInference, FixedTreeInference, JC69, MajorityOutgroupInference
from ancestree.sites import Site
from ancestree.sources import VcfZarrSource
from ancestree.writers import ZarrWriter
from testing._helpers import post

import zarr
from testing._helpers import QUICKSTART_TREES

# ZarrWriter creates the fixed-length string ``variant_AA`` array VCZ mandates, which
# zarr v3 flags with an UnstableSpecificationWarning it has no V3 spec for.
pytestmark = pytest.mark.filterwarnings(
    "ignore::zarr.errors.UnstableSpecificationWarning"
)

_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##INFO=<ID=DP,Number=1,Type=Integer,Description="depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t10\t.\tA\tT\t.\tPASS\tDP=30\tGT\t0|1\t1|1
1\t20\t.\tC\tG\t.\tPASS\tDP=25\tGT\t0|0\t0|1
"""


def _bio2zarr_store(tmp_path) -> str:
    """Write the ``_VCF`` above and convert it to a VCZ store with ``bio2zarr``.

    Uses the ``bio2zarr`` Python API on a plain (unindexed) VCF, so no ``bgzip`` /
    ``tabix`` htslib binaries are needed. Returns the store path.
    """
    import bio2zarr.vcf as bio2zarr_vcf

    vcf = tmp_path / "in.vcf"
    vcf.write_text(_VCF)
    vcz = tmp_path / "in.vcz"
    bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)
    return str(vcz)


def _ts_vcf(ts, tmp_path) -> str:
    """Write ``ts`` to a plain VCF (contig ``"1"``) and return its path.

    No indexing: to_zarr's bio2zarr conversion and CyVCF2Source both read a plain VCF.
    """
    vcf = tmp_path / "ts.vcf"
    with open(vcf, "w") as f:
        ts.write_vcf(f, contig_id="1")
    return str(vcf)


def _site(pos: int, ref: str, alt: str) -> Site:
    return Site(chrom="1", pos=pos, alleles=(ref, alt), tip_alleles={})


# --- direction 1: we read a bio2zarr store --------------------------------------------

def test_reads_bio2zarr_store(tmp_path):
    """VcfZarrSource recovers sites, alleles, and per-haplotype tip alleles from a store
    written by bio2zarr."""
    sites = list(VcfZarrSource(_bio2zarr_store(tmp_path)))

    assert [(s.chrom, s.pos, s.alleles) for s in sites] == [
        ("1", 10, ("A", "T")),
        ("1", 20, ("C", "G")),
    ]
    # 0|1 and 1|1 at the first site → A,T on s1 and T,T on s2.
    assert dict(sites[0].tip_alleles) == {
        "s1_h0": "A", "s1_h1": "T", "s2_h0": "T", "s2_h1": "T",
    }


# --- direction 2: vcztools reads a store our writer produced ---------------------------

def test_vcztools_reads_writer_output(tmp_path):
    """vcztools reconstructs the expected VCF from a bio2zarr store the ZarrWriter has
    annotated: POS / REF / ALT / genotypes survive, and the added variant_AA /
    variant_AA_prob arrays surface as the AA / AA_prob INFO fields."""
    template = _bio2zarr_store(tmp_path)
    out = str(tmp_path / "out.vcz")
    pairs = [
        (_site(10, "A", "T"), post("A", 0.9)),
        (_site(20, "C", "G"), post("C", 0.8)),
    ]
    assert ZarrWriter(template, out).write(iter(pairs)) == 2

    result = subprocess.run(["vcztools", "view", out], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    assert len(body) == 2
    first = body[0].split("\t")
    assert first[1] == "10" and first[3] == "A" and first[4] == "T"
    assert first[9] == "0|1" and first[10] == "1|1"
    assert "AA=A" in first[7] and "AA_prob=0.9" in first[7]
    # the annotation carries alongside the template's own INFO, not in place of it.
    assert "DP=30" in first[7]
    assert "AA=C" in body[1].split("\t")[7]


def test_missing_aa_prob_exports_as_missing_not_nan(tmp_path):
    """An unannotated site's variant_AA_prob uses the VCF-Zarr float-missing sentinel, so
    vcztools exports it as a missing INFO value, not the literal token 'nan'."""
    template = _bio2zarr_store(tmp_path)
    out = str(tmp_path / "out.vcz")
    # annotate only pos 10. Pos 20 is left unannotated.
    ZarrWriter(template, out).write(iter([(_site(10, "A", "T"), post("A", 0.9))]))

    result = subprocess.run(["vcztools", "view", out], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    by_pos = {
        ln.split("\t")[1]: ln.split("\t")[7]
        for ln in result.stdout.splitlines()
        if ln and not ln.startswith("#")
    }
    assert "AA_prob=0.9" in by_pos["10"]
    # the unannotated site carries no AA / AA_prob and, crucially, no literal 'nan'.
    assert "nan" not in by_pos["20"].lower()
    assert "AA=" not in by_pos["20"]
    # the template's own INFO still survives on the unannotated site.
    assert "DP=25" in by_pos["20"]


def _all_aa_called(vcz_path) -> tuple[int, int]:
    """``(n_records, n_with_a_real_AA)`` from ``vcztools view`` over ``vcz_path``."""
    result = subprocess.run(["vcztools", "view", vcz_path], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines() if ln and not ln.startswith("#")]
    called = sum(
        1 for ln in body
        if "AA=" in ln.split("\t")[7]
        and not ln.split("\t")[7].split("AA=")[1].startswith(".")
    )
    return len(body), called


# --- default template: valid zarr output from non-zarr sources, no template needed ----
#
# to_zarr without an input_zarr builds a VCZ template from the mode's source (the source
# VCF, or the tree sequence dumped to VCF) via bio2zarr, then annotates it. These drive
# the two non-zarr sources (ARG and VCF) end to end with no template supplied, and confirm
# vcztools reads every site's AA back, i.e. the synthesized template aligned by (chrom, pos)
# and nothing was dropped.

def test_arg_source_default_template(tmp_path):
    """ARGBasedInference (tskit source), to_zarr with no template -> vcztools reads it."""
    ts = msprime.sim_ancestry(
        samples=6, sequence_length=2e4, recombination_rate=1e-8,
        population_size=1e4, random_seed=3,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
    assert ts.num_sites > 0

    out = str(tmp_path / "arg_out.vcz")
    n = ARGBasedInference(ts, JC69(), mu=1e-8, progress=False).to_zarr(out)
    assert n == ts.num_sites

    n_records, n_called = _all_aa_called(out)
    assert n_records == ts.num_sites
    assert n_called == ts.num_sites


def test_vcf_source_default_template(tmp_path):
    """FixedTreeInference (VCF source), to_zarr with no template -> vcztools reads it."""
    ts = msprime.sim_ancestry(
        samples=6, ploidy=1, sequence_length=2e4, recombination_rate=1e-8,
        population_size=1e4, random_seed=5,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=5)
    assert ts.num_sites > 0
    vcf = _ts_vcf(ts, tmp_path)

    inf = FixedTreeInference(
        vcf, JC69(),
        ingroup_samples=["tsk_0", "tsk_1", "tsk_2", "tsk_3"],
        outgroup_samples=["tsk_4", "tsk_5"],
        fit_required=False, progress=False, baseline_check=False,
    )
    out = str(tmp_path / "vcf_out.vcz")
    n = inf.to_zarr(out)
    assert n == ts.num_sites

    n_records, n_called = _all_aa_called(out)
    assert n_records == ts.num_sites
    assert n_called == ts.num_sites


class TestToZarrReusesAZarrSource:
    """A .vcz source is a template for to_zarr, never for to_vcf."""

    def test_a_store_source_needs_no_explicit_template(self, tmp_path):
        import bio2zarr.vcf as bio2zarr_vcf

        ts = tskit.load(QUICKSTART_TREES)
        names = [f"i{i}" for i in range(6)] + ["o0", "o1"]
        vcf = tmp_path / "src.vcf"
        with open(vcf, "w") as fh:
            ts.write_vcf(fh, individual_names=names, position_transform="legacy")
        store = str(tmp_path / "src.vcz")
        bio2zarr_vcf.convert([str(vcf)], store, show_progress=False)

        inf = anc.FixedTreeInference(
            store, JC69(), n_target_sites=100_000,
            ingroup_samples=names[:6], outgroup_samples=names[6:])
        inf.fit()
        assert inf.to_zarr(str(tmp_path / "out.vcz")) > 0


# ---------------------------------------------------- to_zarr template guard
def test_to_zarr_without_template_or_source_raises(tmp_path):
    # to_zarr builds a template from the mode's source when none is given, and
    # a source-less inference has nothing to build from.
    inf = MajorityOutgroupInference([], [], for_comparison_only=True)
    with pytest.raises(ValueError, match="no template VCF available"):
        inf.to_zarr(str(tmp_path / "out.vcz"))


def test_cli_vcf_in_vcz_out(tmp_path):
    """The local-tree CLI writes a vcztools-readable VCZ from a plain-VCF input with
    no template supplied."""
    from ancestree.cli import run

    ts = msprime.sim_ancestry(
        samples=8, ploidy=1, sequence_length=1e5, recombination_rate=1e-8,
        population_size=1e4, random_seed=7,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=7)
    assert ts.num_sites > 0
    vcf = _ts_vcf(ts, tmp_path)

    out = str(tmp_path / "cli_out.vcz")
    code = run([
        "local-tree", "--vcf", vcf, "--sequence-length", str(ts.sequence_length),
        "--mu", "1e-8", "--rec-rate", "1e-8", "--window", "20snp",
        "--block-size", "500", "--out", out,
    ])
    assert code == 0

    n_records, n_called = _all_aa_called(out)
    assert n_records > 0
    assert n_called > 0


def test_annotation_is_additive_for_vcf_and_zarr(tmp_path):
    """Ancestree annotates. It must not drop anything already in the input.

    Site.info is never populated by any reader and never read by any writer:
    VCF passthrough is the input file used as a template, and VCZ passthrough
    is a copy of the store. That is what lets sites be held columnar without a
    per-site mapping, so it is worth pinning against a custom INFO field rather
    than only the fields the writers happen to know about.
    """
    vcf = tmp_path / "in.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1,length=100000>\n"
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total depth">\n'
        '##INFO=<ID=MYTAG,Number=1,Type=String,Description="Custom">\n'
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2\ts3\to0\n"
        "1\t100\t.\tA\tC\t.\tPASS\tDP=44;MYTAG=keepme\tGT\t0\t0\t1\t1\t0\n"
        "1\t1500\t.\tG\tT\t.\tPASS\tDP=51;MYTAG=two\tGT\t0\t1\t1\t0\t0\n"
        "1\t2600\t.\tC\tA\t.\tPASS\tDP=39;MYTAG=three\tGT\t1\t1\t0\t0\t0\n"
        "1\t4100\t.\tT\tG\t.\tPASS\tDP=60;MYTAG=four\tGT\t0\t1\t0\t1\t0\n"
        "1\t5200\t.\tA\tG\t.\tPASS\tDP=12;MYTAG=five\tGT\t1\t0\t1\t0\t0\n")
    out = tmp_path / "out.vcf"
    from ancestree.cli import main
    with pytest.raises(SystemExit) as exc:  # main() exits 0 on success
        main(["local-tree", "--vcf", str(vcf), "--out", str(out),
              "--outgroups", "o0", "--mu", "2.5e-8", "--rec-rate", "1e-8",
              "--window", "2snp", "--sequence-length", "100000"])
    assert exc.value.code == 0
    body = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert len(body) == 5
    for line, tag in zip(body, ("keepme", "two", "three", "four", "five")):
        info = line.split("\t")[7]
        assert f"MYTAG={tag}" in info, info
        assert "DP=" in info and "AA=" in info, info


def test_vcztools_reads_an_output_whose_template_is_chunked(tmp_path):
    """``vcztools`` must read a store whose template chunks below its length.

    Every variant-indexed array in a VCZ store has to share the variants-axis
    chunk length, and ``vcztools`` refuses a store where they differ, so the
    added arrays must follow the template's chunking. A two-variant store
    cannot show this, since bio2zarr's chunking then degenerates to the
    variant count.
    """
    import bio2zarr.vcf as bio2zarr_vcf

    # vcztools accepts a chunk length that is any positive multiple of the
    # template's, so the count must not be a multiple of variants_chunk_size
    # below: at 40 the whole-store chunk zarr picks divides evenly and passes.
    n = 45
    rows = "\n".join(
        f"1\t{100 * (i + 1)}\t.\tA\tT\t.\t.\tDP=30\tGT\t0|1\t1|1"
        for i in range(n)
    )
    vcf = tmp_path / "many.vcf"
    vcf.write_text(_VCF.split("#CHROM", 1)[0]
                   + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tNA1\tNA2\n"
                   + rows + "\n")
    template = str(tmp_path / "many.vcz")
    # A chunk length below the variant count, as bio2zarr's 1000-variant
    # default is for any real dataset.
    bio2zarr_vcf.convert([str(vcf)], template, show_progress=False,
                         variants_chunk_size=10)
    assert zarr.open(template, mode="r")["variant_position"].chunks[0] < n

    out = str(tmp_path / "out.vcz")
    pairs = [(_site(100 * (i + 1), "A", "T"), post("A", 0.9)) for i in range(n)]
    assert ZarrWriter(template, out).write(iter(pairs)) == n

    result = subprocess.run(["vcztools", "view", out],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    body = [ln for ln in result.stdout.splitlines()
            if ln and not ln.startswith("#")]
    assert len(body) == n
    assert "AA=A" in body[0].split("\t")[7]
