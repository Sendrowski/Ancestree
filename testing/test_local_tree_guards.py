"""Constructor guards and I/O suffix checks in :mod:`ancestree.local_tree_inference`."""
from __future__ import annotations

import numpy as np
import pytest

from ancestree import (
    JC69,
    LocalTreeBuilder,
    LocalTreeInference,
    PairwiseCoalescentHMM,
    PairwiseTmrcas,
)
from ancestree.sites import Site


# ------------------------------------------------ PairwiseCoalescentHMM
def test_hmm_needs_two_haplotypes():
    """The HMM operates on haplotype pairs, so < 2 haplotypes is rejected."""
    with pytest.raises(ValueError, match=r"need >= 2 haplotypes, got 1"):
        PairwiseCoalescentHMM(1, mu=1.25e-8, rec_rate=1e-8)


# ------------------------------------------------ length specs
@pytest.mark.parametrize("spec", ["1gb", "banana", "8snps"])
def test_a_malformed_length_names_the_parameter_and_the_accepted_forms(spec):
    """A typo in a width must be readable, whichever surface it arrives on.

    ``--chunk-size`` was validated by the command line alone, so the same typo
    given to the constructor surfaced as ``invalid literal for int() with base
    10``, naming neither the parameter nor a single accepted form. On the
    chunked path that message arrives only after a full streaming pre-pass.
    """
    from ancestree.local_tree_inference import _parse_bp

    with pytest.raises(ValueError) as exc:
        _parse_bp(spec, name="chunk_size")
    assert "chunk_size must be" in str(exc.value)
    assert "'bp' / 'kb' / 'mb'" in str(exc.value)
    assert repr(spec) in str(exc.value)


@pytest.mark.parametrize("spec", ["0", "-1", 0, -1])
def test_a_non_positive_width_is_rejected_by_the_shared_parser(spec):
    """The API rejects what the command line rejects, from the same parser."""
    from ancestree.local_tree_inference import _parse_bp

    with pytest.raises(ValueError, match=r"chunk_size must be a positive width"):
        _parse_bp(spec, name="chunk_size")


@pytest.mark.parametrize("spec", ["8snps", "1gb"])
def test_a_malformed_window_lists_the_snp_form(spec):
    """``window`` and ``block_size`` take ``"<N>snp"``, so the error lists it."""
    from ancestree.local_tree_inference import (_raw_window_bp,
                                                _resolve_block_spec)

    with pytest.raises(ValueError, match=r"window must be.*'<N>snp'"):
        _raw_window_bp(spec, 60, 1e5)
    with pytest.raises(ValueError, match=r"block_size must be.*'<N>snp'"):
        _resolve_block_spec(spec, 60, 1e5)


def test_a_malformed_width_is_rejected_before_the_source_is_streamed():
    """The chunked path resolves its widths from a streaming pre-pass, so a
    typo reported from there costs a full pass over the source before the
    caller learns of it.
    """
    class _Counting(list):
        n = 0

        def __iter__(self):
            _Counting.n += 1
            return list.__iter__(self)

    sites = _Counting([
        _site("1", 100 * i + 1, "A", "C", {"a": "A", "b": "C"})
        for i in range(5)
    ])
    with pytest.raises(ValueError, match=r"window must be"):
        LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=["a", "b"],
            window="8snps", chunk_size="10mb", n_ensemble=None, progress=False,
        )
    assert _Counting.n == 0


# ------------------------------------------------ LocalTreeBuilder guards
def test_builder_rejects_nonpositive_mu():
    with pytest.raises(ValueError, match=r"mu must be positive, got 0"):
        LocalTreeBuilder(
            [], mu=0.0, rec_rate=1e-8,
            sample_names=["a", "b"], sequence_length=1000.0,
        )


@pytest.mark.parametrize("mu", [float("nan"), float("inf")])
def test_builder_rejects_non_finite_mu(mu):
    """``nan <= 0`` and ``inf <= 0`` are both False, so a positivity test alone
    passed them through and the run died deep in the time-grid calibration
    naming ``time_scale`` and ``t``, symbols the caller never set. A rate
    computed as ``theta / (4 * Ne)`` with ``Ne = 0`` reaches here.
    """
    with pytest.raises(ValueError, match=r"mu must be positive and finite"):
        LocalTreeBuilder(
            [], mu=mu, rec_rate=1e-8,
            sample_names=["a", "b"], sequence_length=1000.0,
        )


@pytest.mark.parametrize("mu", [float("nan"), float("inf")])
def test_hmm_rejects_non_finite_mu(mu):
    """The HMM scales its emissions by ``mu``, so it guards it as the builder does."""
    with pytest.raises(ValueError, match=r"mu must be positive and finite"):
        PairwiseCoalescentHMM(4, mu=mu, rec_rate=1e-8)


def test_hmm_rejects_nonpositive_mu():
    with pytest.raises(ValueError, match=r"mu must be positive, got 0"):
        PairwiseCoalescentHMM(4, mu=0.0, rec_rate=1e-8)


def test_builder_rejects_nonpositive_rec_rate():
    with pytest.raises(ValueError, match=r"rec_rate must be positive, got 0"):
        LocalTreeBuilder(
            [], mu=1.25e-8, rec_rate=0.0,
            sample_names=["a", "b"], sequence_length=1000.0,
        )


def test_builder_needs_two_samples():
    with pytest.raises(ValueError, match=r"need >= 2 samples, got 1"):
        LocalTreeBuilder(
            [], mu=1.25e-8, rec_rate=1e-8,
            sample_names=["only_one"], sequence_length=1000.0,
        )


# ------------------------------------------------ LocalTreeInference guards
def test_inference_genotype_path_requires_sample_names():
    """Building from genotypes without ``sample_names`` is rejected before any
    work happens. It names the caller's own haplotypes, so unlike the rates it
    has no defensible fallback."""
    with pytest.raises(
        ValueError, match=r"building from genotypes requires sample_names",
    ):
        LocalTreeInference([], JC69(), mu=1.25e-8, rec_rate=1e-8)


def test_inference_genotype_path_defaults_rec_rate_with_warning(caplog):
    """``rec_rate=None`` falls back to the package default and warns."""
    import logging

    from ancestree import DEFAULT_REC_RATE

    with caplog.at_level(logging.WARNING):
        inf = LocalTreeInference(
            [], JC69(), mu=1.25e-8, rec_rate=None, sample_names=["a", "b"],
            sequence_length=1000,
        )
    assert inf.rec_rate == pytest.approx(DEFAULT_REC_RATE)
    assert "no rec_rate given" in caplog.text


def test_inference_unchunked_requires_sequence_length():
    """The non-chunked genotype build path needs an explicit
    ``sequence_length`` (the chunked path derives it per segment)."""
    with pytest.raises(
        ValueError, match=r"building requires sequence_length",
    ):
        LocalTreeInference(
            [], JC69(), mu=1.25e-8, rec_rate=1e-8,
            sample_names=["a", "b"], sequence_length=None, chunk_size=None,
        )


class TestAChunkNarrowerThanTheBlockIsFlooredAtTheBlockWidth:
    """A chunk narrower than the resolved block must not shred the run.

    A segment shorter than one emission block gives the HMM a single truncated
    block, so the run degenerates into one HMM pass per block: ``chunk_size=1``
    over 73 sites in 100 kb produced no output in 260 s, against 0.6 s
    unchunked. ``block_size`` is resolved from the data's SNP density, so a
    caller cannot know in advance what width is too narrow.
    """

    @staticmethod
    def _sites(n=24, span=100_000):
        names = ["s0", "s1", "s2", "s3"]
        step = span // n
        return names, [
            _site("1", (i + 1) * step, "A", "C",
                  {"s0": "A", "s1": "C", "s2": "A" if i % 2 else "C", "s3": "C"})
            for i in range(n)
        ]

    def test_the_chunk_is_raised_to_the_block_width_with_a_warning(self, caplog):
        import logging

        names, sites = self._sites()
        inf = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=100_000, block_size=1000, chunk_size=100, halo=500,
            n_ensemble=None, progress=False,
        )
        with caplog.at_level(logging.WARNING, logger="ancestree.LocalTreeInference"):
            inf._resolve_segmentation_params()
        assert inf._chunk_size == inf.block_size == 1000
        assert any("narrower than the resolved block width" in r.message
                   and "chunk size" in r.message for r in caplog.records)

    def test_a_chunk_above_the_block_width_is_left_alone(self):
        """Small chunk sizes stay load-bearing: only a sub-block one moves."""
        names, sites = self._sites()
        inf = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=100_000, block_size=1000, chunk_size=4000, halo=500,
            n_ensemble=None, progress=False,
        )
        inf._resolve_segmentation_params()
        assert inf._chunk_size == 4000


# ------------------------------------------------ PairwiseTmrcas.write
def test_pairwise_tmrcas_write_unrecognised_suffix(tmp_path):
    """``PairwiseTmrcas.write`` selects a format by suffix, and an unknown one
    raises.
    """
    pt = PairwiseTmrcas(
        sample_names=("a", "b"),
        pairs=(("a", "b"),),
        block_midpoints=np.array([0.5]),
        tmrca=np.array([[1.0]]),
    )
    with pytest.raises(ValueError, match=r"unrecognised suffix"):
        pt.write(tmp_path / "out.foo")


def _site(chrom, pos, a0, a1, tip):
    return Site(chrom=chrom, pos=pos, alleles=(a0, a1), tip_alleles=tip)


class TestLocalTreeBuilderGuards:
    def test_multi_contig_rejected(self):
        """A single builder must reject sites spanning >1 contig."""
        names = ["s0", "s1", "s2"]
        sites = [
            _site("1", 100, "A", "C", {"s0": "A", "s1": "C", "s2": "A"}),
            _site("2", 300, "A", "C", {"s0": "C", "s1": "A", "s2": "C"}),
        ]
        with pytest.raises(ValueError, match="single contiguous coordinate axis"):
            LocalTreeBuilder(
                sites, mu=1e-8, rec_rate=1e-8, sample_names=names,
                sequence_length=1000,
            )

    def test_duplicate_positions_clear_error(self):
        """Duplicate site positions raise an actionable error, not a raw tskit one."""
        names = ["s0", "s1", "s2", "s3"]
        # Two records at pos 100 (a split multiallelic site).
        sites = [
            _site("1", 100, "A", "C", {"s0": "A", "s1": "C", "s2": "A", "s3": "C"}),
            _site("1", 100, "A", "G", {"s0": "A", "s1": "G", "s2": "G", "s3": "A"}),
            _site("1", 400, "A", "C", {"s0": "C", "s1": "A", "s2": "C", "s3": "A"}),
        ]
        b = LocalTreeBuilder(
            sites, mu=1e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=1000,
        )
        with pytest.raises(ValueError, match="duplicated"):
            b.to_tree_sequence()

    def test_out_of_range_position_warns(self, caplog):
        """A site at pos == sequence_length is clipped, with a warning."""
        import logging
        names = ["s0", "s1"]
        sites = [
            _site("1", 100, "A", "C", {"s0": "A", "s1": "C"}),
            _site("1", 1000, "A", "C", {"s0": "C", "s1": "A"}),  # == sequence_length
        ]
        b = LocalTreeBuilder(
            sites, mu=1e-8, rec_rate=1e-8, sample_names=names,
            sequence_length=1000,
        )
        with caplog.at_level(logging.WARNING, logger="ancestree.LocalTreeBuilder"):
            b._genotype_matrix()
        assert any("clipped onto the axis" in r.message for r in caplog.records)


def test_an_unmatched_sample_name_is_refused_as_a_value_error():
    """One exception type for a bad sample name across the three modes.

    An unknown outgroup id raised ``ValueError`` from
    :class:`FixedTreeInference` and ``KeyError`` here and in ARG mode, and an
    unknown ingroup id raised nothing at all in fixed-tree mode, so a caller
    guarding one constructor did not guard the other two.
    """
    sites = [_site("1", 100 * i + 1, "A", "C", {"a": "A", "b": "C"})
             for i in range(5)]
    for kwargs in ({"ingroup_samples": ["a", "typo"]},
                   {"outgroup_samples": ["typo"]}):
        with pytest.raises(ValueError, match="not in the panel"):
            LocalTreeInference(
                sites, JC69(), mu=1.25e-8, rec_rate=1e-8,
                sample_names=["a", "b"], n_ensemble=None, progress=False,
                **kwargs,
            )


class TestThePreBuiltPathIgnoresTheGenotypeArguments:
    """A supplied genealogy needs none of the local-tree building settings.

    Both of these reached the pre-built branch through checks meant for the
    genotype path: the depth guard ran before ``n_ensemble`` was cleared, and
    the ignored-argument report compared an array with ``!=``.
    """

    @staticmethod
    def _ts():
        import msprime

        ts = msprime.sim_ancestry(4, sequence_length=1000, random_seed=1,
                                  ploidy=1)
        return msprime.sim_mutations(ts, rate=1e-3, random_seed=2)

    def test_a_depth_focal_is_accepted_when_no_ensemble_is_drawn(self):
        from ancestree import FocalNode

        inference = LocalTreeInference(
            self._ts(), JC69(), mu=1e-8, focal=FocalNode(depth=0.5))
        assert inference.n_ensemble is None

    def test_an_array_time_grid_does_not_raise(self):
        LocalTreeInference(self._ts(), JC69(), mu=1e-8,
                           time_grid=np.array([0.0, 1.0, 2.0, 3.0]))

    def test_untouched_arguments_are_not_reported_as_ignored(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="ancestree"):
            LocalTreeInference(self._ts(), JC69(), mu=1e-8)
        assert not [r for r in caplog.records
                    if "Ignoring window" in r.getMessage()]


class TestTheSegmentBuilderCarriesTheMapAtItsOwnCoordinates:
    """A sliced rate map must be read at the segment's own origin.

    Slicing from 0 instead of the segment origin gives every segment the
    first segment's rates, which the end-to-end tests do not detect: they
    only assert that the run completes.
    """

    def test_each_segment_reads_the_global_rate_at_its_origin(self):
        import msprime

        from ancestree.sites import Site

        span = 6000
        sites = [Site(chrom="1", pos=p, alleles=("A", "C"),
                      tip_alleles={"a": "A", "b": "C"})
                 for p in range(100, span, 100)]
        rates = msprime.RateMap(position=[0, 1500, 3000, 4500, span],
                                rate=[1e-9, 4e-8, 1e-9, 4e-8])
        inference = LocalTreeInference(
            sites, JC69(), mu=1.25e-8, rec_rate=1e-8,
            sample_names=["a", "b"], window=200, block_size=50,
            sequence_length=span, chunk_size=1500, progress=False,
            recombination_map=rates)
        inference._resolve_segmentation_params()

        n_segments = 0
        for work_unit in inference._stream_segments():
            origin, seg_span = work_unit[3], work_unit[4]
            # The builder is what scores the segment, so assert on ITS map,
            # not on _slice_map's return value.
            builder = inference._segment_builder(work_unit)
            assert builder.recombination_map is not None
            n_segments += 1
            for local in (1.0, min(seg_span, span - origin) - 1.0):
                if local < 0:
                    continue
                got = builder.recombination_map.get_rate(local)
                want = rates.get_rate(origin + local)
                assert got == want, (
                    f"segment at {origin} reads {got} at local {local} where "
                    f"the global map holds {want}")
        assert n_segments > 1
