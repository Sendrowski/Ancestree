"""Local-tree geometry on real-data-shaped inputs.

The suite's simulated panels hold sites spanning almost all of the declared
length on one unmasked contig. Real inputs are regional VCFs on a
whole-chromosome header, with masks and rate maps that cover the data only in
part, several contigs of different lengths and stretches without sites. These
fixtures place a small simulated panel far inside a much longer header, so
the block, window, chunk and halo geometry is exercised at every edge: a
halo-only core, a core whose segment holds no site, site-free stretches at
both ends of a contig, a site at ``POS == L``, and a contig of three sites.
"""
import msprime
import numpy as np
import pytest

from ancestree import JC69, LocalTreeBuilder, LocalTreeInference
from ancestree.sites import Site
from ancestree.trees import TskitLocalTree
from testing._helpers import write_vcz

MU = 3e-8
REC = 1e-8
#: Header length of contig 1, far longer than the data.
HEADER = 400_000
#: Genome position the simulated region starts at.
OFFSET = 250_000
SIM_LENGTH = 80_000
#: Stretch the gapped panel drops. Under CHUNK and HALO it leaves the core
#: [280 kb, 290 kb) with no site in its segment, beside halo-only cores.
GAP = (268_000, 301_000)
CHUNK = 10_000
HALO = 5_000
#: Contig 2: longer than contig 1, with three sites past contig 1's end, the
#: last at POS == L.
HEADER_2 = 600_000
#: Last position of a short contig 2, whose windows are wider than contig 1's.
SHORT_2 = 37_000
SITES_2 = (450_101, 451_377, HEADER_2)
#: Wider than either contig, so every segment holds its whole contig.
WIDE_HALO = 2_000_000
N_IND = 4
NAMES = [f"i{k}_h{h}" for k in range(N_IND) for h in (0, 1)]
N_ENSEMBLE = 4


def _simulated_rows():
    """Biallelic ``(pos, alleles, haplotype genotypes)`` on contig 1."""
    ts = msprime.sim_ancestry(
        N_IND, ploidy=2, sequence_length=SIM_LENGTH, recombination_rate=REC,
        population_size=1e4, random_seed=11)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=11)
    rows = []
    for variant in ts.variants():
        alleles = variant.alleles
        if len(alleles) != 2 or any(a not in "ACGT" for a in alleles):
            continue
        rows.append((OFFSET + int(variant.site.position),
                     tuple(alleles), np.asarray(variant.genotypes)))
    return rows


def _contig_2_rows():
    """The three polymorphic sites of contig 2."""
    rng = np.random.default_rng(3)
    rows = []
    for pos in SITES_2:
        g = rng.integers(0, 2, 2 * N_IND)
        g[0], g[1] = 0, 1
        rows.append((pos, ("G", "T"), g))
    return rows


def _write_vcf(path, records, contigs):
    """A phased diploid VCF. ``records`` holds ``(chrom, pos, alleles, g)``."""
    lines = ["##fileformat=VCFv4.2"]
    lines += [f"##contig=<ID={c},length={length}>" for c, length in contigs]
    lines += ['##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
              + "\t".join(f"i{k}" for k in range(N_IND))]
    for chrom, pos, alleles, g in records:
        calls = "\t".join(f"{g[2 * k]}|{g[2 * k + 1]}" for k in range(N_IND))
        lines.append(f"{chrom}\t{pos}\t.\t{alleles[0]}\t{alleles[1]}\t.\t"
                     f"PASS\t.\tGT\t{calls}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Paths of the regional, gapped and multi-contig VCFs and the VCZ."""
    root = tmp_path_factory.mktemp("geometry")
    rows = _simulated_rows()
    gapped = [r for r in rows if not GAP[0] <= r[0] < GAP[1]]
    multi = [("1", *r) for r in rows] + [("2", *r) for r in _contig_2_rows()]
    contigs = [("1", HEADER), ("2", HEADER_2)]
    store = root / "multi.vcz"
    write_vcz(
        store, positions=[r[1] for r in multi],
        contigs_per_variant=[int(r[0]) - 1 for r in multi],
        contig_ids=["1", "2"], alleles=[list(r[2]) for r in multi],
        sample_ids=[f"i{k}" for k in range(N_IND)],
        genotypes=[np.asarray(r[3]).reshape(N_IND, 2) for r in multi],
        phased=True, contig_lengths=[HEADER, HEADER_2])
    return {
        "regional": _write_vcf(root / "regional.vcf",
                               [("1", *r) for r in rows], [("1", HEADER)]),
        "gapped": _write_vcf(root / "gapped.vcf",
                             [("1", *r) for r in gapped], [("1", HEADER)]),
        "multi_vcf": _write_vcf(root / "multi.vcf", multi, contigs),
        "multi_vcz": str(store),
    }


def _maps(kind):
    """Rate maps and a mask on the 1-based axis of contig 1.

    ``"past"`` maps reach beyond both ends of the data, ``"short"`` maps start
    and stop inside it. Each leaves its first interval undefined.
    """
    if kind is None:
        return {}
    lo, mid, hi = ((240_000, 285_000, 345_000) if kind == "past"
                   else (262_000, 285_000, 318_000))
    return dict(
        recombination_map=msprime.RateMap(
            position=[0, lo, mid, hi], rate=[np.nan, 4 * REC, 0.25 * REC]),
        mutation_map=msprime.RateMap(
            position=[0, lo, mid, hi], rate=[np.nan, 3 * MU, 0.5 * MU]),
        accessibility=[(lo, mid - 2_000), (mid - 1_000, hi)],
    )


def _inference(path, *, chunk_size=None, halo=HALO, n_ensemble=None,
               maps=None, **kwargs):
    """A local-tree run on ``path`` at this module's rates."""
    return LocalTreeInference(
        path, JC69(), mu=MU, rec_rate=REC, progress=False,
        chunk_size=chunk_size, halo=halo, n_ensemble=n_ensemble,
        **_maps(maps), **kwargs)


def _resolved(inference):
    """``(block, window, halo)`` in bp as the run resolved them, the halo
    ``0`` unchunked."""
    segmentation = inference.segmentation
    return segmentation.block, segmentation.window, segmentation.halo


def _posteriors(inference):
    """``[(chrom, pos, values)]`` in the order :meth:`infer` yields them."""
    return [(str(site.chrom), int(site.pos), np.asarray(post.values))
            for site, post in inference.infer()]


def _records(path):
    """``(chrom, pos)`` of every record of the VCF at ``path``."""
    with open(path, encoding="utf-8") as fh:
        return [(f[0], int(f[1])) for f in
                (line.split("\t") for line in fh if not line.startswith("#"))]


def _positions(path):
    """Positions of the records of the single-contig VCF at ``path``."""
    return [pos for _chrom, pos in _records(path)]


def _covered(ts):
    """Merged ``[left, right)`` intervals under a tree with edges."""
    out = []
    for tree in ts.trees():
        if tree.num_edges == 0:
            continue
        left, right = tree.interval
        if out and out[-1][1] == left:
            out[-1][1] = right
        else:
            out.append([left, right])
    return [tuple(iv) for iv in out]


def _merged(intervals):
    """``intervals``, ascending and disjoint, with abutting ones merged."""
    out = []
    for left, right in sorted(intervals):
        if out and out[-1][1] == left:
            out[-1][1] = right
        else:
            out.append([left, right])
    return [tuple(iv) for iv in out]


def _groups(inference):
    """``[(interval, [member coverage])]`` over :meth:`tree_sequences`, with
    every member's sequence length."""
    out, lengths = [], set()
    for interval, members in inference.tree_sequences():
        members = list(members)
        lengths |= {m.sequence_length for m in members}
        out.append((interval, [_covered(m) for m in members]))
    return out, lengths


def _data_stretch(path, window_bp):
    """The windows from the first site's to the last site's, as one interval.

    The grid tiles contig 1 with ``n = round(HEADER / w)`` windows of width
    ``HEADER / n`` for a resolved window width ``w``, the last ending at the
    axis end ``HEADER + 1``.
    """
    positions = _positions(path)
    n = max(1, int(round(HEADER / window_bp)))
    step = HEADER / n
    first = min(int(positions[0] // step), n - 1)
    last = min(int(positions[-1] // step), n - 1)
    right = HEADER + 1.0 if last == n - 1 else (last + 1) * step
    return [(first * step, right)]


def _assert_same_intervals(got, expected):
    """Two interval lists equal to within float rounding of the grid."""
    assert len(got) == len(expected), f"{got} != {expected}"
    np.testing.assert_allclose(np.asarray(got, dtype=float),
                               np.asarray(expected, dtype=float),
                               rtol=0, atol=1e-6)


def _pair_tmrcas(ts, positions):
    """Pairwise sample TMRCAs of the tree over each position, one row each."""
    rows = []
    for x in positions:
        tree = ts.at(x)
        rows.append([tree.tmrca(a, b) for a in range(len(NAMES))
                     for b in range(a + 1, len(NAMES))])
    return np.asarray(rows)


def _assert_scored(rows):
    """Every posterior is finite, normalised and off the flat posterior a
    site outside every tree is given."""
    values = np.stack([r[2] for r in rows])
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)
    flat = np.all(np.abs(values - 1.0 / values.shape[1]) < 1e-9, axis=1)
    assert not flat.any(), [r[:2] for r, f in zip(rows, flat) if f]


def _assert_same_posteriors(left, right):
    """Two :func:`_posteriors` lists equal site for site, bit for bit."""
    assert [r[:2] for r in left] == [r[:2] for r in right]
    for (_c, pos, a), (_c2, _p, b) in zip(left, right):
        assert np.array_equal(a, b), f"posterior at {pos} differs"


#: ``(window, block_size)`` specs: the snp default, a coarse snp window, two
#: snp specs, two bp ints, and a bp window not a whole number of blocks.
WIDTH_SPECS = [("8snp", None), ("30snp", None), ("12snp", "3snp"),
               (5000, 1000), ("2500", "1kb")]


class TestResolvedWidths:
    """Block, window and halo widths, the SNP density taken over each
    contig's site extent and never over its header length."""

    #: ``dataset -> spec -> (block, window, halo)``, the halo at chunk_size
    #: CHUNK with halo HALO, raised to the widest window.
    EXPECTED = {
        "regional": {
            ("8snp", None): (1278, 2556, 5000),
            ("30snp", None): (1198, 9584, 9525),
            ("12snp", "3snp"): (958, 3832, 5000),
            (5000, 1000): (1000, 5000, 5001),
            ("2500", "1kb"): (1000, 3000, 5000),
        },
        "gapped": {
            ("8snp", None): (2092, 4184, 5000),
            ("30snp", None): (1962, 15696, 16001),
            ("12snp", "3snp"): (1569, 6276, 6251),
            (5000, 1000): (1000, 5000, 5001),
            ("2500", "1kb"): (1000, 3000, 5000),
        },
    }

    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    @pytest.mark.parametrize("spec", WIDTH_SPECS, ids=str)
    def test_widths_on_a_regional_vcf(self, inputs, dataset, spec):
        """Both paths resolve one block and window, a whole number of blocks
        per window, and the chunked halo clears the widest window."""
        window, block = spec
        plain = _resolved(_inference(inputs[dataset], window=window,
                                     block_size=block))
        chunked = _resolved(_inference(inputs[dataset], window=window,
                                       block_size=block, chunk_size=CHUNK))
        assert plain[:2] == chunked[:2]
        assert plain[1] % plain[0] == 0
        assert chunked == self.EXPECTED[dataset][spec]

    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_auto_halo_is_capped_at_half_the_chunk(self, inputs, dataset):
        inference = _inference(inputs[dataset], chunk_size=CHUNK, halo="auto")
        assert _resolved(inference)[2] == CHUNK // 2

    def test_auto_halo_reads_the_ingroup_diversity(self, inputs):
        """The auto halo is ``5 / (r · π / (2μ))`` for the ingroup's per-bp
        pairwise diversity ``π``, which the outgroups do not enter. ``π`` was
        taken over every tip, so divergent outgroups inflated it and shrank
        the halo below the span over which ingroup pairs decorrelate."""
        ingroup = [f"i{k}" for k in range(N_IND - 1)]
        rows = _simulated_rows()
        n = 2 * len(ingroup)
        pairdiff = 0.0
        for _pos, _alleles, g in rows:
            k = int(np.sum(g[:n]))
            pairdiff += 2.0 * k * (n - k) / (n * (n - 1))
        span = rows[-1][0] - rows[0][0] + 1
        expected = int(np.ceil(5.0 / (REC * pairdiff / span / (2.0 * MU))))
        inference = _inference(inputs["regional"], chunk_size=10 * HEADER,
                               halo="auto", ingroup_samples=ingroup,
                               outgroup_samples=[f"i{N_IND - 1}"])
        assert _resolved(inference)[2] == expected

    @pytest.mark.parametrize("store", ["multi_vcf", "multi_vcz"])
    def test_widths_over_several_contigs(self, inputs, store):
        """The density sums the site extents of both contigs."""
        inference = _inference(inputs[store], chunk_size=CHUNK)
        assert _resolved(inference) == (3690, 7380, 7409)

    def test_provenance_records_the_applied_widths(self, inputs):
        """The record holds the halo the run applied and the width of the
        windows tiling each contig. It held the halo as passed, before the
        floor raised it, and the nominal window, which no contig is tiled
        at."""
        inference = _inference(
            inputs["multi_vcf"], chunk_size=CHUNK, halo=100,
            sequence_length={"1": HEADER, "2": HEADER_2 + 3_000})
        params = inference._provenance_parameters()
        assert params["halo"] == _resolved(inference)[2] > 100
        assert params["window_bp"] == {
            c: inference.segmentation.contig(c).length
            / len(inference.segmentation.contig(c).windows) for c in "12"}
        assert params["window_bp"]["1"] != params["window_bp"]["2"]
        single = _inference(inputs["regional"])._provenance_parameters()
        assert single["window_bp"] == HEADER / round(HEADER / 2556)


    def test_a_contig_missing_from_the_header_sets_the_halo_floor(
            self, inputs, tmp_path):
        """A contig the header lists without a length ends at its last site,
        and its widest window raises a narrower explicit halo. The halo floor
        was taken over the header's lengths and the undeclared contigs a
        pre-pass saw, but bp widths and a fixed halo skip the pre-pass, so
        the contig missing from the header was left out and a halo narrower
        than its windows cut them at segment edges."""
        rows = [("1", *r) for r in _simulated_rows()]
        rows += [("2", pos - OFFSET, alleles, g) for _c, pos, alleles, g
                 in rows if pos - OFFSET <= SHORT_2]
        end_2 = rows[-1][1] + 1
        missing = _write_vcf(tmp_path / "missing.vcf", rows,
                             [("1", HEADER), ("2", "")])
        declared = _write_vcf(tmp_path / "declared.vcf", rows,
                              [("1", HEADER), ("2", end_2)])
        kwargs = dict(chunk_size=CHUNK, window=5000, block_size=1000,
                      halo=1000)
        n = round(end_2 / 5000)
        widest = int(np.ceil(end_2 / n + 1))
        assert widest > 5001
        assert _resolved(_inference(missing, **kwargs))[2] == widest
        assert _resolved(_inference(declared, **kwargs))[2] == widest


class TestEverySiteScoredOnce:
    """:meth:`infer` yields each source record once, in source order."""

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("maps", [None, "past", "short"])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_single_contig(self, inputs, dataset, maps, chunk_size,
                           n_ensemble):
        inference = _inference(inputs[dataset], chunk_size=chunk_size,
                               n_ensemble=n_ensemble, maps=maps)
        rows = _posteriors(inference)
        assert [r[:2] for r in rows] == _records(inputs[dataset])
        _assert_scored(rows)

    @pytest.mark.parametrize("spec", WIDTH_SPECS, ids=str)
    def test_every_width_spec(self, inputs, spec):
        window, block = spec
        inference = _inference(inputs["gapped"], chunk_size=CHUNK,
                               window=window, block_size=block)
        assert ([r[:2] for r in _posteriors(inference)]
                == _records(inputs["gapped"]))

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    def test_vcf_and_vcz_over_two_contigs_agree(self, inputs, n_ensemble):
        """Contig 2 holds three sites past contig 1's end, so each contig is
        laid on its own header length."""
        vcf = _posteriors(_inference(inputs["multi_vcf"], chunk_size=CHUNK,
                                     n_ensemble=n_ensemble))
        vcz = _posteriors(_inference(inputs["multi_vcz"], chunk_size=CHUNK,
                                     n_ensemble=n_ensemble))
        assert [r[:2] for r in vcf] == _records(inputs["multi_vcf"])
        assert [r[0] for r in vcf].count("2") == len(SITES_2)
        _assert_scored(vcf)
        _assert_same_posteriors(vcf, vcz)

    def test_a_contig_mapping_matches_the_header(self, inputs):
        header = _posteriors(_inference(inputs["multi_vcf"],
                                        chunk_size=CHUNK))
        mapped = _posteriors(_inference(
            inputs["multi_vcf"], chunk_size=CHUNK,
            sequence_length={"1": HEADER, "2": HEADER_2}))
        _assert_same_posteriors(header, mapped)

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    def test_a_contig_scores_as_it_does_alone(self, inputs, n_ensemble):
        """With bp widths and a fixed halo, contig 1 of the two-contig input
        is scored as the single-contig input is."""
        kwargs = dict(chunk_size=CHUNK, n_ensemble=n_ensemble, window=3000,
                      block_size=1000)
        both = _posteriors(_inference(inputs["multi_vcf"], **kwargs))
        alone = _posteriors(_inference(inputs["regional"], **kwargs))
        _assert_same_posteriors([r for r in both if r[0] == "1"], alone)


class TestChunkedMatchesUnchunked:
    """A halo wider than the contig gives every segment the whole contig, so
    the chunked run reproduces the unchunked one exactly."""

    @pytest.mark.parametrize("maps", [None, "past", "short"])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_plug_in(self, inputs, dataset, maps):
        plain = _inference(inputs[dataset], maps=maps)
        chunked = _inference(inputs[dataset], maps=maps, chunk_size=CHUNK,
                             halo=WIDE_HALO)
        expected = _posteriors(plain)
        _assert_same_posteriors(expected, _posteriors(chunked))
        positions = [r[1] for r in expected]
        assert np.array_equal(
            _pair_tmrcas(plain.point_tree_sequence(), positions),
            _pair_tmrcas(chunked.point_tree_sequence(), positions))
        left, right = plain.pairwise_tmrcas(), chunked.pairwise_tmrcas()
        assert np.array_equal(left.block_midpoints, right.block_midpoints)
        assert np.array_equal(left.tmrca, right.tmrca)

    @pytest.mark.parametrize("spec", WIDTH_SPECS, ids=str)
    def test_every_width_spec(self, inputs, spec):
        window, block = spec
        kwargs = dict(window=window, block_size=block, maps="past")
        _assert_same_posteriors(
            _posteriors(_inference(inputs["gapped"], **kwargs)),
            _posteriors(_inference(inputs["gapped"], chunk_size=CHUNK,
                                   halo=WIDE_HALO, **kwargs)))

    @pytest.mark.parametrize("maps", [None, "past"])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_ensemble_in_one_core(self, inputs, dataset, maps):
        """A chunk wider than the axis is one core keyed as the unchunked
        region, so the draws and their genealogies coincide."""
        plain = _inference(inputs[dataset], maps=maps, n_ensemble=N_ENSEMBLE)
        chunked = _inference(inputs[dataset], maps=maps,
                             n_ensemble=N_ENSEMBLE, chunk_size=10 * HEADER)
        expected = _posteriors(plain)
        _assert_same_posteriors(expected, _posteriors(chunked))
        positions = [r[1] for r in expected]
        (_, left), = plain.tree_sequences()
        (_, right), = chunked.tree_sequences()
        for a, b in zip(left, right, strict=True):
            assert np.array_equal(_pair_tmrcas(a, positions),
                                  _pair_tmrcas(b, positions))


class TestPinnedPosteriors:
    """A few literal posteriors, which no relational test constrains."""

    #: ``(dataset, chunk_size, n_ensemble, maps) -> {pos: posterior}``.
    PINNED = {
        ("regional", None, None, None): {
            251657: [0.000431881862028, 1.03338667135e-07, 1.03338667135e-07,
                     0.999567911461],
            292122: [0.000206904810477, 0.458051250601, 0.000206904810477,
                     0.541534939778],
            329591: [0.99999995226, 1.0090924165e-08, 2.75577389017e-08,
                     1.0090924165e-08],
        },
        ("regional", None, N_ENSEMBLE, "past"): {
            251657: [0.209758097566, 4.82901081625e-05, 4.82901081625e-05,
                     0.790145322217],
            292122: [0.000211820900561, 0.45499567319, 0.000211820900561,
                     0.544580685009],
            329591: [0.999999936368, 1.97166470596e-08, 2.41990882324e-08,
                     1.97166470596e-08],
        },
        ("gapped", CHUNK, None, "short"): {
            251657: [0.00016604069578, 5.91916857123e-08, 5.91916857123e-08,
                     0.999833840921],
            311098: [0.00020866723008, 0.00020866723008, 0.601412962398,
                     0.398169703142],
            329591: [0.999996021213, 1.32626236037e-06, 1.32626236049e-06,
                     1.32626236037e-06],
        },
        ("gapped", CHUNK, N_ENSEMBLE, None): {
            251657: [0.0829015582202, 2.11692205113e-05, 2.11692205113e-05,
                     0.917056103339],
            311098: [0.000197350364179, 0.000197350364179, 0.605784892428,
                     0.393820406843],
            329591: [0.999999853045, 4.00663153003e-08, 6.68226075125e-08,
                     4.00663153003e-08],
        },
    }

    @pytest.mark.parametrize("key", list(PINNED), ids=str)
    def test_pinned(self, inputs, key):
        dataset, chunk_size, n_ensemble, maps = key
        rows = {r[1]: r[2] for r in _posteriors(_inference(
            inputs[dataset], chunk_size=chunk_size, n_ensemble=n_ensemble,
            maps=maps))}
        for pos, expected in self.PINNED[key].items():
            np.testing.assert_allclose(rows[pos], expected, rtol=0,
                                       atol=1e-9)


class TestInferSiteMatchesInfer:
    """:meth:`infer_site` on the plug-in tree over a site reproduces
    :meth:`infer`, at the mutation map's rate where one is set."""

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("maps", [None, "past", "short"])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_every_site(self, inputs, dataset, maps, chunk_size):
        inference = _inference(inputs[dataset], maps=maps,
                               chunk_size=chunk_size)
        batch = [(site, np.asarray(post.values))
                 for site, post in inference.infer()]
        ts = inference.point_tree_sequence()
        sample_map = {name: i for i, name in enumerate(NAMES)}
        assert inference.sample_names == NAMES
        for site, expected in batch:
            tree = TskitLocalTree(ts, site.pos, sample_map=sample_map)
            np.testing.assert_allclose(
                inference.infer_site(tree, site).values, expected,
                rtol=0, atol=1e-12)

    def test_an_unknown_contig_is_named(self, inputs):
        """A site on a contig the source lacks raised a bare ``KeyError``
        from the contig lookup."""
        from dataclasses import replace
        inference = _inference(inputs["regional"])
        site = next(iter(inference._source))
        tree = TskitLocalTree(inference.point_tree_sequence(), site.pos,
                              sample_map={n: i for i, n in enumerate(NAMES)})
        with pytest.raises(ValueError, match=r"contig 'chr9'.*\['1'\]"):
            inference.infer_site(tree, replace(site, chrom="chr9"))


class TestTreeOutput:
    """Axis lengths, baked sites and member tiling of the tree output."""

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_point_tree_sequence(self, inputs, dataset, chunk_size):
        """The axis is the header length plus the VCF's position base, and
        every site is baked once, under a tree."""
        inference = _inference(inputs[dataset], chunk_size=chunk_size)
        ts = inference.point_tree_sequence()
        assert ts.sequence_length == HEADER + 1
        positions = _positions(inputs[dataset])
        assert [int(s.position) for s in ts.sites()] == positions
        assert all(ts.at(x).num_edges for x in positions)
        assert all(ts.at(x).num_roots == 1 for x in positions)

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_tree_sequences(self, inputs, dataset, chunk_size, n_ensemble):
        """Groups ascend without overlap, every member of a group covers the
        same stretch inside the group's interval, and together the groups
        cover what the point tree sequence covers, every site included."""
        inference = _inference(inputs[dataset], chunk_size=chunk_size,
                               n_ensemble=n_ensemble)
        groups, lengths = _groups(inference)
        assert lengths == {HEADER + 1.0}
        intervals = [interval for interval, _ in groups]
        assert intervals == sorted(intervals)
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
        covered = []
        for (lo, hi), members in groups:
            assert len(members) == (n_ensemble or 1)
            assert all(m == members[0] for m in members)
            assert all(lo <= left < right <= hi
                       for left, right in members[0])
            covered += members[0]
        covered = _merged(covered)
        point = _inference(inputs[dataset], chunk_size=chunk_size)
        assert covered == _covered(point.point_tree_sequence())
        for x in _positions(inputs[dataset]):
            assert any(left <= x < right for left, right in covered)

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_pairwise_tmrcas_cover_the_trees(self, inputs, dataset,
                                             chunk_size):
        """Every block a tree overlaps is reported. A block was reported only
        when its midpoint lay under a tree, so the first block of the data
        stretch, straddling the left edge of the first window, was left
        out."""
        inference = _inference(inputs[dataset], chunk_size=chunk_size)
        block = _resolved(inference)[0]
        (left, right), = _merged(_covered(inference.point_tree_sequence()))
        first, last = int(left // block), int(np.ceil(right / block))
        expected = [min((k + 0.5) * block, 0.5 * (k * block + HEADER + 1))
                    for k in range(first, last)]
        np.testing.assert_allclose(
            inference.pairwise_tmrcas().block_midpoints, expected,
            rtol=0, atol=1e-9)

    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_pairwise_tmrcas_on_the_block_grid(self, inputs, dataset):
        """Chunked, each block is reported once, on the unchunked grid."""
        plain = _inference(inputs[dataset]).pairwise_tmrcas()
        chunked = _inference(inputs[dataset],
                             chunk_size=CHUNK).pairwise_tmrcas()
        assert np.array_equal(plain.block_midpoints, chunked.block_midpoints)
        assert np.all(np.diff(chunked.block_midpoints) > 0)
        assert chunked.tmrca.shape == plain.tmrca.shape

    def test_a_block_past_the_data_stretch_is_reported_once(self, inputs):
        """The block straddling the right end of the data stretch reaches
        past it into the next core. Owned by the core holding its raw
        midpoint, it fell to a core with no tree and was reported by
        neither, so the chunked record held one block fewer than the
        unchunked one."""
        kwargs = dict(window="30snp", halo=WIDE_HALO)
        plain = _inference(inputs["gapped"], **kwargs).pairwise_tmrcas()
        chunked = _inference(inputs["gapped"], chunk_size=4_000,
                             **kwargs).pairwise_tmrcas()
        assert np.array_equal(plain.block_midpoints, chunked.block_midpoints)


class TestDataStretch:
    """Trees cover each contig's data stretch, the windows from the first
    site's to the last site's, interior gaps tiled, and nothing outside it."""

    def test_no_tree_past_the_window_of_the_last_site(self):
        """Windows of 1000/333 bp put an edge at 300.3, between the last site
        at 300 and one past it. The window from that edge holds no site, but
        it overlapped ``[first, last + 1)`` and got a tree."""
        rng = np.random.default_rng(1)
        names = ["a", "b", "c", "d"]
        sites = [Site(chrom="1", pos=int(pos), alleles=("A", "C"),
                      tip_alleles={n: "AC"[int(rng.random() < 0.4)]
                                   for n in names})
                 for pos in range(5, 301, 5)]
        builder = LocalTreeBuilder(sites, mu=MU, rec_rate=REC,
                                   sample_names=names, sequence_length=1000,
                                   window=3, block_size=1)
        (left, right), = _covered(builder.to_tree_sequence())
        step = 1000 / 333
        assert left == pytest.approx(step)
        assert right == pytest.approx(100 * step)
        assert right > 300

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_point_tree_sequence(self, inputs, dataset, chunk_size):
        inference = _inference(inputs[dataset], chunk_size=chunk_size)
        expected = _data_stretch(inputs[dataset], _resolved(inference)[1])
        _assert_same_intervals(_covered(inference.point_tree_sequence()),
                               expected)

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_every_member(self, inputs, dataset, chunk_size, n_ensemble):
        inference = _inference(inputs[dataset], chunk_size=chunk_size,
                               n_ensemble=n_ensemble)
        expected = _data_stretch(inputs[dataset], _resolved(inference)[1])
        groups, _ = _groups(inference)
        for k in range(n_ensemble or 1):
            covered = _merged(iv for _, members in groups
                              for iv in members[k])
            _assert_same_intervals(covered, expected)


class TestHmmBlocks:
    """The HMM runs over the blocks of the data stretch alone."""

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_no_block_outside_the_data_windows(self, inputs, dataset,
                                               chunk_size):
        """The HMM ran over every block of a segment's frame, so a regional
        VCF on a whole-chromosome header ran it over the whole header."""
        inference = _inference(inputs[dataset], chunk_size=chunk_size)
        block, window, _ = _resolved(inference)
        (left, right), = _data_stretch(inputs[dataset], window)
        for builder in inference.segments():
            geometry = builder.geometry
            edges = geometry.origin + geometry.block_edges
            assert edges[0] > left - block and edges[-1] < right + block
            inputs_ = builder.prepared_hmm()[2]
            assert inputs_.counts.shape[1] == geometry.n_blocks


class TestDataPlacement:
    """Posteriors and pairwise TMRCAs do not depend on where on the axis the
    data sit, given block, window and chunk grids aligned under the shift."""

    #: Shift onto a point of the block, window and chunk grids, moving the
    #: panel from 250 kb to 10 kb on the same header.
    SHIFT = 240_000
    WIDTHS = dict(window=4000, block_size=1000)

    @pytest.fixture(scope="class")
    @classmethod
    def shifted(cls, tmp_path_factory):
        """The regional panel moved down by :attr:`SHIFT`."""
        rows = [("1", pos - cls.SHIFT, alleles, g)
                for pos, alleles, g in _simulated_rows()]
        return _write_vcf(tmp_path_factory.mktemp("shifted") / "shifted.vcf",
                          rows, [("1", HEADER)])

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    def test_flanks_carry_no_weight(self, inputs, shifted, chunk_size):
        """Without a mask, the blocks before the first site and after the
        last entered the time-grid calibration, the per-pair prior mean and
        the HMM as zero-difference sequence at full exposure, so a panel far
        from 0 on a long header was pulled toward a TMRCA of 0 by its flanks,
        and its posteriors moved with its placement and with chunk_size."""
        runs = [_inference(path, chunk_size=chunk_size, **self.WIDTHS)
                for path in (inputs["regional"], shifted)]
        left, right = (_posteriors(run) for run in runs)
        assert [r[1] - self.SHIFT for r in left] == [r[1] for r in right]
        for a, b in zip(left, right):
            np.testing.assert_allclose(a[2], b[2], rtol=0, atol=1e-12)
        a, b = (run.pairwise_tmrcas() for run in runs)
        np.testing.assert_array_equal(a.block_midpoints - self.SHIFT,
                                      b.block_midpoints)
        np.testing.assert_allclose(a.tmrca, b.tmrca, rtol=1e-12)


class TestDeclaredLength:
    """A site past a contig's declared end is refused, naming where the
    length came from."""

    @pytest.fixture(scope="class")
    @classmethod
    def short_header(cls, tmp_path_factory):
        """The regional panel under a header shorter than its last site."""
        rows = [("1", *r) for r in _simulated_rows()]
        return _write_vcf(tmp_path_factory.mktemp("short") / "short.vcf",
                          rows, [("1", 20_000)])

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    def test_a_short_header_is_named(self, short_header, chunk_size):
        """A header length shorter than the last site was reported as a
        sequence_length that does not contain every site, though none was
        passed."""
        with pytest.raises(ValueError, match=r"the VCF header gives contig 1 "
                           r"length 20,000, but a site lies at 25\d,\d{3}; fix "
                           r"the header or pass sequence_length"):
            _posteriors(_inference(short_header, chunk_size=chunk_size))

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    def test_a_passed_length_keeps_its_message(self, short_header,
                                               chunk_size):
        with pytest.raises(ValueError, match="Pass a sequence_length"):
            _posteriors(_inference(short_header, chunk_size=chunk_size,
                                   sequence_length=20_000))


class TestSegments:
    """The sites and frames of the chunked work units."""

    @pytest.mark.parametrize("dataset", ["regional", "gapped"])
    def test_a_segment_holds_every_site_of_its_frame(self, inputs, dataset):
        """The frame origin is the left halo end snapped down to the block
        grid, but the sites were selected from the unsnapped halo end, so the
        block holding the origin was partly emptied of its sites while it
        emitted at full exposure."""
        inference = _inference(inputs[dataset], chunk_size=CHUNK)
        positions = np.asarray(_positions(inputs[dataset]))
        n_snapped = 0
        for sites, geometry in inference.segmentation.units():
            lo = geometry.origin
            hi = geometry.genome_core[1] + inference.segmentation.halo
            expected = positions[(positions >= lo) & (positions < hi)]
            assert [s.pos for s in sites] == expected.tolist(), lo
            n_snapped += bool(np.any(
                (expected < geometry.genome_core[0] - inference.segmentation.halo)))
        assert n_snapped


class TestInputChecks:
    """Malformed lengths and an input without a callable block."""

    def test_a_builder_needs_a_length(self):
        """Without a length or a geometry the builder failed on float(None)
        with a TypeError naming neither."""
        sites = [Site(chrom="1", pos=10, alleles=("A", "C"),
                      tip_alleles={"a": "A", "b": "C"})]
        with pytest.raises(ValueError, match="needs sequence_length"):
            LocalTreeBuilder(sites, mu=MU, rec_rate=REC,
                             sample_names=["a", "b"])

    @pytest.mark.parametrize("n_ensemble", [None, N_ENSEMBLE])
    def test_a_partial_base_is_a_whole_one(self, inputs, n_ensemble):
        """A non-integer length was truncated on the chunked path and kept on
        the unchunked one, so the two laid different axes and blocks."""
        kwargs = dict(sequence_length=HEADER - 0.5, n_ensemble=n_ensemble)
        plain = _inference(inputs["regional"], **kwargs)
        chunked = _inference(inputs["regional"], chunk_size=10 * HEADER,
                             **kwargs)
        _assert_same_posteriors(_posteriors(plain), _posteriors(chunked))
        whole = _inference(inputs["regional"], n_ensemble=n_ensemble)
        _assert_same_posteriors(_posteriors(plain), _posteriors(whole))
        assert (plain.point_tree_sequence().sequence_length
                == chunked.point_tree_sequence().sequence_length == HEADER + 1)

    @pytest.mark.parametrize("chunk_size", [None, CHUNK])
    def test_a_fully_masked_input_is_refused(self, inputs, chunk_size):
        """Chunked, a mask covering none of the data returned prior-only
        posteriors, where the unchunked build refuses."""
        with pytest.raises(ValueError, match="no block over this region is "
                                             "callable"):
            _posteriors(_inference(inputs["regional"], chunk_size=chunk_size,
                                   accessibility=[(1, 1000)]))
