"""Tests for `TskitSource`: variant iteration, sample mapping, position handling,
missing-data exposure.
"""
from __future__ import annotations

import numpy as np
import tskit
import ancestree as anc

from ancestree.likelihood import Likelihood
from ancestree.models import JC69
from ancestree.sites import Site
from ancestree.sources import TskitSource
from ancestree.trees import TskitLocalTree


def test_iterates_all_variants(small_ts):
    src = TskitSource(small_ts)
    sites = list(src)
    assert len(sites) == small_ts.num_sites
    assert all(s.chrom == "1" for s in sites)


def test_chrom_override(small_ts):
    src = TskitSource(small_ts, chrom="chrX")
    s = next(iter(src))
    assert s.chrom == "chrX"


def test_positions_match_tskit_write_vcf(small_ts):
    """Ancestree emits `int(position)`, matching tskit's own write_vcf truncation."""
    src = TskitSource(small_ts)
    for site, ts_site in zip(src, small_ts.sites()):
        assert site.pos == int(ts_site.position)


def test_samples_list_matches_sample_map(small_ts):
    src = TskitSource(small_ts)
    samples = src.samples()
    assert len(samples) == small_ts.num_samples
    smap = src.sample_map()
    assert set(smap.keys()) == set(samples)


def test_tip_alleles_keys_are_caller_sample_ids(small_ts):
    mapping = {f"ind_{int(s)}": int(s) for s in small_ts.samples()}
    src = TskitSource(small_ts, sample_map=mapping)
    first = next(iter(src))
    assert set(first.tip_alleles.keys()) == set(mapping.keys())


def test_local_tree_handle_is_position(small_ts):
    src = TskitSource(small_ts)
    for site, ts_site in zip(src, small_ts.sites()):
        assert site.local_tree_handle == float(ts_site.position)


def test_named_diploid_individuals_yield_both_haplotypes():
    """A tree sequence with named diploid individuals exposes
    both haplotypes (``name_h0`` / ``name_h1``), not one node per
    individual."""
    import msprime
    ts = msprime.sim_ancestry(
        samples=3, ploidy=2, sequence_length=2e4,
        recombination_rate=1e-8, population_size=1e4, random_seed=5,
    )
    ts = msprime.sim_mutations(ts, rate=5e-8, random_seed=5)
    # Attach a "name" to each individual so the metadata-derived map is used.
    tables = ts.dump_tables()
    tables.individuals.clear()
    tables.individuals.metadata_schema = tskit.MetadataSchema.permissive_json()
    for i in range(ts.num_individuals):
        tables.individuals.add_row(metadata={"name": f"ind{i}"})
    ts = tables.tree_sequence()

    src = TskitSource(ts)
    names = src.samples()
    assert len(names) == ts.num_samples  # one entry per haplotype, none dropped
    assert names == [f"ind{i}_h{h}" for i in range(3) for h in range(2)]
    site = next(iter(src))
    assert set(site.tip_alleles) == set(names)


def test_soft_masked_alleles_are_read():
    """A lowercased reference must not read as an all-missing panel.

    Soft-masked input is ordinary in reference genomes, and without a case
    fold every tip would fall outside the A/C/G/T alphabet.
    """
    import msprime

    ts = msprime.sim_ancestry(2, ploidy=1, sequence_length=50_000,
                              random_seed=4)
    ts = msprime.sim_mutations(ts, rate=1e-5, random_seed=4)
    assert ts.num_sites > 0

    tables = ts.dump_tables()
    sites = tables.sites.copy()
    tables.sites.clear()
    for row in sites:
        tables.sites.add_row(position=row.position,
                             ancestral_state=row.ancestral_state.lower())
    mutations = tables.mutations.copy()
    tables.mutations.clear()
    for row in mutations:
        tables.mutations.add_row(site=row.site, node=row.node,
                                 parent=row.parent,
                                 derived_state=row.derived_state.lower())
    lowered = tables.tree_sequence()

    observed = [s for s in anc.TskitSource(lowered)]
    assert observed, "no sites read at all"
    called = sum(1 for s in observed for a in s.tip_alleles.values()
                 if a in "ACGT")
    assert called > 0, (
        "every tip of a soft-masked panel read as missing")


def _indel_ts():
    """Tiny ``tskit.TreeSequence`` carrying a multi-character derived allele."""
    tables = tskit.TableCollection(sequence_length=10)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=0, time=1)
    tables.edges.add_row(left=0, right=10, parent=2, child=0)
    tables.edges.add_row(left=0, right=10, parent=2, child=1)
    tables.sites.add_row(position=5, ancestral_state="A")
    tables.mutations.add_row(site=0, node=0, derived_state="AT")
    tables.sort()
    return tables.tree_sequence()


class TestNonSNPHandling:
    """Indels / multi-character alleles surface unchanged from
    :class:`TskitSource` (no upstream skip) and rely on the kernel's
    marginalisation contract."""

    def test_indel_site_is_emitted_with_multichar_tip(self):
        """TskitSource does not skip multi-character derived states. The multi-
        character allele appears verbatim in ``tip_alleles`` for downstream
        marginalisation."""
        ts = _indel_ts()
        sites = list(TskitSource(ts))
        assert len(sites) == 1
        assert "AT" in set(sites[0].tip_alleles.values())

    def test_indel_site_is_finite_under_likelihood_kernel(self):
        """The likelihood kernel must not crash on multi-character tip alleles
        from a TskitSource site, and the marginalised value must match
        a ``None`` tip on the same tree."""
        ts = _indel_ts()
        src = TskitSource(ts)
        sites = list(src)
        tree = TskitLocalTree.from_tskit_tree(ts.first(), sample_map=src.sample_map())
        engine = Likelihood(JC69())
        log_L = engine.log_likelihoods(tree, sites)
        assert np.all(np.isfinite(log_L))
        # Equivalent site with the multi-char tip rewritten to None.
        s = sites[0]
        none_tips = {k: (None if v == "AT" else v) for k, v in s.tip_alleles.items()}
        none_site = Site(
            chrom=s.chrom, pos=s.pos, alleles=s.alleles, tip_alleles=none_tips,
        )
        log_L_none = engine.log_likelihoods(tree, [none_site])
        np.testing.assert_allclose(log_L, log_L_none, atol=1e-12)


def _n_allele_ts():
    """Tiny ``tskit.TreeSequence`` whose site carries an ``N`` ancestral state."""
    tables = tskit.TableCollection(sequence_length=10)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(flags=0, time=1)
    tables.edges.add_row(left=0, right=10, parent=2, child=0)
    tables.edges.add_row(left=0, right=10, parent=2, child=1)
    tables.sites.add_row(position=5, ancestral_state="N")
    tables.mutations.add_row(site=0, node=0, derived_state="C")
    tables.sort()
    return tables.tree_sequence()


class TestNoCallAlleleSpellings:
    """``N`` is a no-call, scored exactly as an absent tip allele is.

    ``N`` shared one bucket with indels and symbolic alleles, so a site
    carrying it could not be told apart from one the kernel cannot read at
    all. Splitting the buckets must leave the likelihood untouched.
    """

    def test_an_n_tip_scores_as_a_none_tip(self):
        ts = _n_allele_ts()
        src = TskitSource(ts)
        sites = list(src)
        tree = TskitLocalTree.from_tskit_tree(ts.first(), sample_map=src.sample_map())
        engine = Likelihood(JC69())
        log_L = engine.log_likelihoods(tree, sites)
        assert np.all(np.isfinite(log_L))
        s = sites[0]
        none_tips = {k: (None if v == "N" else v) for k, v in s.tip_alleles.items()}
        none_site = Site(
            chrom=s.chrom, pos=s.pos, alleles=s.alleles, tip_alleles=none_tips,
        )
        log_L_none = engine.log_likelihoods(tree, [none_site])
        np.testing.assert_allclose(log_L, log_L_none, atol=1e-12)

    def test_an_n_tip_is_not_counted_as_unrepresentable(self):
        site = list(TskitSource(_n_allele_ts()))[0]
        assert not site.has_unrepresentable_allele
        assert site.n_unrepresentable_tips() == 0

    def test_an_n_ancestral_state_still_leaves_the_site_callable(self):
        """The derived ``C`` is readable, so the site is annotated as usual."""
        assert list(TskitSource(_n_allele_ts()))[0].has_representable_allele
