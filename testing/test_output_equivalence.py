"""One dataset, every input and output format: the derived statistics agree.

The package reads a VCF or a VCF-Zarr store and writes a VCF, a VCZ store or a
tree sequence. Nothing pinned that the same data through different combinations
of those yields the same calls, so a backend could drift on allele order,
ploidy expansion, missing-value encoding or row placement and every existing
test would stay green. These compare the whole 2x2 grid on one small dataset,
and compare a downstream statistic rather than only the raw fields.

``bio2zarr`` is used through its Python API, so no htslib binaries are needed
and the fixtures stay fast.
"""
from __future__ import annotations

import msprime
import numpy as np
import pytest

from ancestree import FixedTreeInference, JC69, OutgroupLadderTree
from ancestree.readers import Reader
from ancestree.sites import BaseComposition, Site
from ancestree.sources import CyVCF2Source, VcfZarrSource

N_INGROUP = 6
INGROUP = [f"tsk_{i}" for i in range(N_INGROUP)]
OUTGROUP = [f"tsk_{N_INGROUP}"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """``(vcf_path, vcz_path)`` for one small haploid panel plus an outgroup."""
    tmp_path = tmp_path_factory.mktemp("equivalence")
    demography = msprime.Demography()
    demography.add_population(name="ingroup", initial_size=1e4)
    demography.add_population(name="outgroup", initial_size=1e4)
    demography.add_population(name="anc", initial_size=1e4)
    demography.add_population_split(
        time=2e5, ancestral="anc", derived=["ingroup", "outgroup"])
    ts = msprime.sim_ancestry(
        samples=[
            msprime.SampleSet(N_INGROUP, population="ingroup", ploidy=1),
            msprime.SampleSet(1, population="outgroup", ploidy=1),
        ],
        demography=demography, sequence_length=3e4,
        recombination_rate=1e-8, random_seed=11,
    )
    ts = msprime.sim_mutations(ts, rate=5e-7, random_seed=11)

    vcf = tmp_path / "panel.vcf"
    with open(vcf, "w") as handle:
        ts.write_vcf(handle, contig_id="1")

    import bio2zarr.vcf as bio2zarr_vcf

    vcz = tmp_path / "panel.vcz"
    bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)
    return str(vcf), str(vcz)


def _inference(source):
    """A pre-parameterised fixed-tree run, so no optimisation cost."""
    tree = OutgroupLadderTree(INGROUP, OUTGROUP)
    tree.set_params(np.array([0.02]))
    return FixedTreeInference(
        source, JC69(), BaseComposition.from_n_target_sites(30_000),
        tree=tree, fit_required=False, progress=False,
    )


def _calls(path):
    """``([aa], [prob])`` in file order, for annotated sites only.

    The allele is compared exactly. The probability only to the storage
    precision, since a VCZ store holds it as ``f4`` per the VCF-Zarr spec
    while a VCF carries a formatted decimal.
    """
    alleles, probabilities = [], []
    for record in Reader(path).annotations():
        if record.aa is None:
            continue
        alleles.append(record.aa)
        probabilities.append(float(record.aa_prob))
    return alleles, np.asarray(probabilities)


def _spectrum(path, source_path):
    """Unfolded derived-allele-count spectrum over the ingroup.

    Counts, per site, the ingroup haplotypes whose allele differs from the
    called ancestral allele, which is the statistic an unfolded SFS is built
    from and therefore the thing a backend difference would corrupt.
    """
    sites = {(s.chrom, s.pos): s for s in _open(source_path)}
    spectrum = [0] * (N_INGROUP + 1)
    for record in Reader(path).annotations():
        if record.aa is None:
            continue
        site = sites.get((str(record.chrom), int(record.pos)))
        if site is None:
            continue
        derived = sum(
            1 for h in INGROUP
            if site.tip_alleles.get(h) not in (record.aa, None)
        )
        spectrum[derived] += 1
    return spectrum


def _open(path):
    """The source matching ``path``'s format."""
    return VcfZarrSource(path) if str(path).endswith(".vcz") else CyVCF2Source(path)


class TestInputSourceEquivalence:
    """The two readers yield the same sites from the same data."""

    def test_site_streams_match(self, dataset):
        vcf, vcz = dataset
        from_vcf = list(CyVCF2Source(vcf))
        from_vcz = list(VcfZarrSource(vcz))
        assert len(from_vcf) == len(from_vcz) > 0
        for a, b in zip(from_vcf, from_vcz):
            assert (a.chrom, a.pos, a.alleles) == (b.chrom, b.pos, b.alleles)
            assert dict(a.tip_alleles) == dict(b.tip_alleles)

    def test_panels_match(self, dataset):
        vcf, vcz = dataset
        assert sorted(CyVCF2Source(vcf).samples()) == sorted(
            VcfZarrSource(vcz).samples())


@pytest.fixture(scope="module")
def written(dataset, tmp_path_factory):
    """``({source label: (vcf out, vcz out)}, source vcf)`` for the 2x2 grid."""
    vcf, vcz = dataset
    out = tmp_path_factory.mktemp("written")
    paths = {}
    for label, source_path in (("vcf", vcf), ("vcz", vcz)):
        inference = _inference(_open(source_path))
        as_vcf = str(out / f"from_{label}.vcf")
        as_vcz = str(out / f"from_{label}.vcz")
        inference.to_vcf(as_vcf, input_vcf=vcf)
        inference.to_zarr(as_vcz, input_zarr=vcz)
        paths[label] = (as_vcf, as_vcz)
    return paths, vcf


class TestOutputFormatEquivalence:
    """A run written to VCF and to VCZ reads back the same, and so does the
    spectrum derived from either."""

    def test_every_combination_gives_the_same_calls(self, written):
        paths, _ = written
        want_alleles, want_probs = _calls(paths["vcf"][0])
        assert want_alleles
        for label, outputs in paths.items():
            for kind, path in zip(("vcf", "vcz"), outputs):
                alleles, probabilities = _calls(path)
                assert alleles == want_alleles, f"{label} -> {kind}"
                assert np.allclose(
                    probabilities, want_probs, rtol=1e-6, atol=0,
                ), f"{label} -> {kind}"

    def test_every_combination_gives_the_same_spectrum(self, written):
        paths, source_vcf = written
        reference = _spectrum(paths["vcf"][0], source_vcf)
        assert sum(reference) > 0
        for label, (as_vcf, as_vcz) in paths.items():
            assert _spectrum(as_vcf, source_vcf) == reference, f"{label} -> vcf"
            assert _spectrum(as_vcz, source_vcf) == reference, f"{label} -> vcz"


_SPLIT_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2
1\t30\t.\tG\tA\t.\tPASS\t.\tGT\t0\t1\t0
1\t30\t.\tG\tT\t.\tPASS\t.\tGT\t0\t0\t1
"""

_JOINED_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2
1\t30\t.\tG\tA,T\t.\tPASS\t.\tGT\t0\t1\t2
"""


class TestSplitMultiallelicInputIsReported:
    """Split multiallelic input is announced, not consumed silently.

    Splitting recodes every other alternate's carriers as the reference
    allele, so each half reports an inflated reference frequency and the
    position is scored once per line. Both were silent, and the ingroup weight
    conditions on exactly those frequencies.
    """

    def test_split_input_warns(self, tmp_path, caplog):
        path = tmp_path / "split.vcf"
        path.write_text(_SPLIT_VCF)
        with caplog.at_level("WARNING"):
            sites = list(CyVCF2Source(str(path)))
        assert len(sites) == 2
        assert "norm -m+" in caplog.text

    def test_joined_input_is_quiet(self, tmp_path, caplog):
        path = tmp_path / "joined.vcf"
        path.write_text(_JOINED_VCF)
        with caplog.at_level("WARNING"):
            sites = list(CyVCF2Source(str(path)))
        assert len(sites) == 1
        assert sites[0].alleles == ("G", "A", "T")
        assert "norm -m+" not in caplog.text

    def test_a_filtered_record_still_marks_its_position(self, tmp_path, caplog):
        """A non-ACGT record shares its position with the SNP that follows.

        The Zarr path tested for a duplicate position after the allele
        filter, so a record skipped for carrying an indel never marked the
        position and the split went unreported on that backend.
        """
        import bio2zarr.vcf as bio2zarr_vcf

        vcf = tmp_path / "indel_split.vcf"
        vcf.write_text(_SPLIT_INDEL_VCF)
        vcz = tmp_path / "indel_split.vcz"
        bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)
        for source in (CyVCF2Source(str(vcf)), VcfZarrSource(str(vcz))):
            caplog.clear()
            with caplog.at_level("WARNING"):
                list(source)
            assert "norm -m+" in caplog.text, type(source).__name__

    def test_the_zarr_source_warns_too(self, tmp_path, caplog):
        import bio2zarr.vcf as bio2zarr_vcf

        vcf = tmp_path / "split.vcf"
        vcf.write_text(_SPLIT_VCF)
        vcz = tmp_path / "split.vcz"
        bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)
        with caplog.at_level("WARNING"):
            list(VcfZarrSource(str(vcz)))
        assert "norm -m+" in caplog.text


_SPLIT_INDEL_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2
1\t30\t.\tA\tAT\t.\tPASS\t.\tGT\t0\t1\t0
1\t30\t.\tA\tG\t.\tPASS\t.\tGT\t0\t0\t1
"""


_TRIPLOID_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t0|1|1\t1/1/0
1\t20\t.\tC\tG\t.\tPASS\t.\tGT\t0/0/1\t.
"""


def test_the_two_readers_agree_on_a_triploid_panel(tmp_path):
    """Arbitrary ploidy reads the same through either backend.

    cyvcf2's ``gt_bases`` refuses ploidy above 2, so the VCF path decoded
    genotype strings and could not read a polyploid panel at all while the
    Zarr path could. Reading allele indices instead covers any ploidy.
    """
    import bio2zarr.vcf as bio2zarr_vcf

    vcf = tmp_path / "triploid.vcf"
    vcf.write_text(_TRIPLOID_VCF)
    vcz = tmp_path / "triploid.vcz"
    bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)

    from_vcf = list(CyVCF2Source(str(vcf)))
    from_vcz = list(VcfZarrSource(str(vcz)))
    assert [s.ploidy for s in (CyVCF2Source(str(vcf)), VcfZarrSource(str(vcz)))] == [3, 3]
    assert len(from_vcf) == len(from_vcz) == 2
    for a, b in zip(from_vcf, from_vcz):
        assert (a.chrom, a.pos, a.alleles) == (b.chrom, b.pos, b.alleles)
        assert dict(a.tip_alleles) == dict(b.tip_alleles)


_LEADING_MISSING_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t.\t.
1\t20\t.\tC\tG\t.\tPASS\t.\tGT\t0|1\t1|1
"""


def test_a_leading_uncalled_record_does_not_halve_the_panel(tmp_path):
    """Ploidy is probed past records that carry no called genotype.

    htslib returns a fully uncalled genotype written as a bare "." with one
    value, indistinguishable from a haploid call, so taking ploidy from the
    first record read a diploid file as haploid and dropped one haplotype of
    every sample for the whole file.
    """
    path = tmp_path / "leading_missing.vcf"
    path.write_text(_LEADING_MISSING_VCF)
    source = CyVCF2Source(str(path))
    assert source.ploidy == 2
    assert sorted(source.samples()) == ["s1_h0", "s1_h1", "s2_h0", "s2_h1"]
    site = next(s for s in CyVCF2Source(str(path)) if s.pos == 20)
    # the heterozygote keeps both copies, so the derived allele survives
    assert dict(site.tip_alleles) == {
        "s1_h0": "C", "s1_h1": "G", "s2_h0": "G", "s2_h1": "G"}


def test_a_call_lands_on_the_record_its_alleles_describe(tmp_path):
    """A co-located indel must not receive the SNP's ancestral call.

    Rows were matched by (contig, position) alone while the sources skip
    every non-ACGT record, so the posterior stream was a strict subset of
    the template rows and each call was written one row early: the SNP's
    AA landed on the indel sharing its position and the SNP itself was
    left unannotated.
    """
    vcf = tmp_path / "indel_beside_snp.vcf"
    vcf.write_text(_SPLIT_INDEL_VCF)
    tree = OutgroupLadderTree(["s0", "s1"], ["s2"])
    tree.set_params(np.array([0.02]))
    inference = FixedTreeInference(
        CyVCF2Source(str(vcf)), JC69(),
        BaseComposition.from_n_target_sites(100),
        tree=tree, fit_required=False, progress=False,
    )
    out = tmp_path / "annotated.vcf"
    inference.to_vcf(str(out), input_vcf=str(vcf))

    rows = [ln.split("\t") for ln in out.read_text().splitlines()
            if not ln.startswith("#")]
    by_alt = {r[4]: r[7] for r in rows}
    assert by_alt["AT"] == ".", "the indel must carry no ancestral call"
    assert "AA=" in by_alt["G"], "the SNP must carry its own ancestral call"


def test_grade_brier_is_the_mean_brier_against_the_truth(dataset):
    """``brier`` is pinned to a value, not only to its ``[0, 2]`` range.

    Every existing caller asserted a range or compared the tuple to itself,
    so returning ``map_recovery`` in its place passed the whole suite.
    """
    vcf, _ = dataset
    inference = _inference(CyVCF2Source(vcf))
    posteriors = {int(s.pos): p for s, p in inference.infer()}
    # Truth that is neither constant nor equal to the MAP call everywhere, so
    # the Brier score and the hit rate cannot coincide by construction.
    truth = {pos: ("A" if pos % 2 else "C") for pos in posteriors}

    grade = inference.grade(truth)
    expected_brier = np.mean([
        float(((posteriors[pos].values - np.array(
            [1.0 if a == t else 0.0 for a in posteriors[pos].alleles]
        )) ** 2).sum())
        for pos, t in truth.items()
    ])
    expected_map = np.mean([
        float(posteriors[pos].map_allele == t) for pos, t in truth.items()
    ])
    assert grade.n_sites == len(truth)
    assert grade.brier == pytest.approx(expected_brier, abs=1e-12)
    assert grade.map_recovery == pytest.approx(expected_map, abs=1e-12)
    # the two must not be numerically identical, or the test proves nothing
    assert abs(grade.brier - grade.map_recovery) > 1e-6


def test_aa_post_survives_a_near_certain_call(tmp_path, dataset):
    """``AA_post`` keeps enough precision to recover ``1 - P``.

    Written at ``%g``'s six significant figures, a posterior of 0.99999987
    rounds to exactly 1, so a mis-assignment probability computed as
    ``1 - P(AA)`` came out zero instead of 1.3e-7.
    """
    vcf, _ = dataset
    inference = _inference(CyVCF2Source(vcf))
    out = tmp_path / "precise.vcf"
    inference.to_vcf(str(out), input_vcf=vcf)

    worst = 0.0
    n_seen = 0
    for line in out.read_text().splitlines():
        if line.startswith("#"):
            continue
        info = line.split("\t")[7]
        field = next((f for f in info.split(";") if f.startswith("AA_post=")), None)
        if field is None:
            continue
        n_seen += 1
        total = sum(float(v) for v in field.split("=", 1)[1].split(","))
        worst = max(worst, abs(total - 1.0))
    # The four written values still sum to one at double precision, which the
    # six-figure rendering could not guarantee.
    assert n_seen > 0
    assert worst < 1e-12, f"AA_post lost precision: worst |sum - 1| = {worst:g}"


_UNPHASED_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=100000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2
1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
1\t200\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
1\t400\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
1\t500\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
1\t600\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\t0/1
"""


class TestUnphasedGenotypes:
    """An unphased call carries no haplotype order, and is not read as one.

    A caller writes a heterozygote with its allele indices ascending, so
    reading position as haplotype makes the first haplotype the reference
    carrier at every heterozygous site of every sample simultaneously. The
    genealogy estimators read labelled haplotypes, so that reads as a clade:
    measured on a simulated diploid panel it halves the h0-h0 divergence and
    costs 16.6 points of MAP accuracy, against 3.9 for a per-site draw.
    """

    def test_the_written_order_is_not_taken_as_phase(self, tmp_path, caplog):
        path = tmp_path / "unphased.vcf"
        path.write_text(_UNPHASED_VCF)
        with caplog.at_level("INFO"):
            sites = list(CyVCF2Source(str(path)))
        first_hap = [s.tip_alleles["s1_h0"] for s in sites]
        assert len(set(first_hap)) > 1, (
            "every unphased heterozygote put the reference on h0")
        assert "phase_seed" in caplog.text

    def test_phased_true_takes_the_file_at_its_word(self, tmp_path):
        path = tmp_path / "unphased.vcf"
        path.write_text(_UNPHASED_VCF)
        sites = list(CyVCF2Source(str(path), phased=True))
        assert [s.tip_alleles["s1_h0"] for s in sites] == ["A"] * 6

    def test_the_assignment_is_stable_across_passes(self, tmp_path):
        """Fixed-tree mode reads the source twice, and the writers again."""
        path = tmp_path / "unphased.vcf"
        path.write_text(_UNPHASED_VCF)
        first = [s.tip_alleles["s1_h0"] for s in CyVCF2Source(str(path))]
        second = [s.tip_alleles["s1_h0"] for s in CyVCF2Source(str(path))]
        assert first == second

    def test_both_backends_draw_the_same_assignment(self, tmp_path):
        import bio2zarr.vcf as bio2zarr_vcf

        vcf = tmp_path / "unphased.vcf"
        vcf.write_text(_UNPHASED_VCF)
        vcz = tmp_path / "unphased.vcz"
        bio2zarr_vcf.convert([str(vcf)], str(vcz), show_progress=False)
        from_vcf = [s.tip_alleles["s1_h0"] for s in CyVCF2Source(str(vcf))]
        from_vcz = [s.tip_alleles["s1_h0"] for s in VcfZarrSource(str(vcz))]
        assert from_vcf == from_vcz


_INGROUP_FIXED_VCF = """##fileformat=VCFv4.2
##contig=<ID=1,length=1000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="gt">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2\ts3
1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t1\t1\t1\t0
1\t20\t.\tA\tG\t.\tPASS\t.\tGT\t1\t1\t1\t0
1\t30\t.\tC\tT\t.\tPASS\t.\tGT\t0\t1\t0\t0
"""


def test_ingroup_fixed_sites_are_reported_once_per_pass(tmp_path, caplog):
    """Sites the ingroup has fixed are announced, once, with a total.

    At the ingroup MRCA such a site returns the ingroup's own allele. The
    count is reported at the end of the pass rather than per site.
    """
    path = tmp_path / "ingroup_fixed.vcf"
    path.write_text(_INGROUP_FIXED_VCF)
    tree = OutgroupLadderTree(["s0", "s1", "s2"], ["s3"])
    tree.set_params(np.array([0.02]))
    inference = FixedTreeInference(
        CyVCF2Source(str(path)), JC69(),
        BaseComposition.from_n_target_sites(100),
        tree=tree, fit_required=False, progress=False)
    with caplog.at_level("WARNING"):
        list(inference.infer())
    hits = [r for r in caplog.records
            if "monomorphic within the ingroup" in r.message]
    assert len(hits) == 1, "expected exactly one report per pass"
    assert "2 site(s)" in hits[0].message


def test_an_ascertained_composition_is_reported_to_a_pi_model(caplog):
    """Variant sites alone do not estimate the stationary distribution.

    A site is variant in proportion to how readily its base mutates, so a
    tally over variant sites weights each base by its exit rate. Measured
    against a simulated HKY truth the estimate understates A and T by about
    a tenth and overstates C and G by about a sixth, and the error does not
    shrink with more sites. Only a model whose stationary vector is the
    supplied composition is affected.
    """
    from ancestree import HKY
    from ancestree.sites import BaseComposition

    names = ["a", "b", "c", "o"]
    sites = [
        Site(chrom="1", pos=pos, alleles=("A", "T"),
             tip_alleles={n: ("A" if pos % 3 else "T") for n in names})
        for pos in range(1, 200)
    ]
    composition = BaseComposition.from_polymorphic_sites(sites)
    assert composition._pi_is_ascertained

    def build(model):
        caplog.clear()
        with caplog.at_level("WARNING"):
            FixedTreeInference(
                sites, model, composition, ingroup_samples=names[:3],
                outgroup_samples=["o"], fit_required=False, progress=False)
        return any("variant sites alone" in r.message for r in caplog.records)

    assert build(HKY(kappa=2.0)), "a base-frequency model must be told"
    assert not build(JC69()), "a uniform stationary vector cannot be biased"
