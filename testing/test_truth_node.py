"""The benchmark truth and derived-allele count are read at the grading node.

An ingroup-polymorphic site is graded at the ingroup MRCA, so its derived count
is taken against the state there, not against ``site.ancestral_state`` (the
state at the root of the whole simulation). The two disagree once a second
mutation sits on the path between those nodes.
"""
import sys
from pathlib import Path

import tskit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflow" / "scripts"))

from _truth import site_truths  # noqa: E402


def _ladder_ts(mutations, *, missing=()):
    """Three ingroup tips under one MRCA plus one outgroup, one site at 5.

    :param mutations: ``(node_key, derived_state, time)`` triples, with keys
        ``i0``, ``i1``, ``i2``, ``mrca`` and ``out``.
    :param missing: Ingroup keys whose genotype is made missing by detaching
        the tip from the tree.
    :return: ``(ts, ingroup_nodes, panel_nodes)``.
    """
    tables = tskit.TableCollection(sequence_length=10.0)
    tips = {k: tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            for k in ("i0", "i1", "i2", "out")}
    nodes = dict(tips)
    nodes["mrca"] = tables.nodes.add_row(flags=0, time=1.0)
    root = tables.nodes.add_row(flags=0, time=3.0)
    for k in ("i0", "i1", "i2"):
        if k not in missing:
            tables.edges.add_row(left=0, right=10.0, parent=nodes["mrca"],
                                 child=tips[k])
    tables.edges.add_row(left=0, right=10.0, parent=root, child=nodes["mrca"])
    tables.edges.add_row(left=0, right=10.0, parent=root, child=tips["out"])
    site = tables.sites.add_row(position=5.0, ancestral_state="A")
    for key, state, time in mutations:
        tables.mutations.add_row(site=site, node=nodes[key],
                                 derived_state=state, time=time)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    ingroup = [tips["i0"], tips["i1"], tips["i2"]]
    return tables.tree_sequence(), ingroup, ingroup + [tips["out"]]


def test_derived_count_is_taken_at_the_ingroup_mrca():
    """A -> T above the MRCA, then T -> A back on i0: the ingroup holds A, T, T.

    Against the deep state A two copies are derived; against the state T at
    the ingroup MRCA only i0 is.
    """
    ts, ingroup, panel = _ladder_ts([("mrca", "T", 2.0), ("i0", "A", 0.5)])
    t = site_truths(ts, ingroup, panel)[5]
    assert ts.site(0).ancestral_state == "A"
    assert t.polymorphic
    assert t.at_ingroup_mrca == "T"
    assert t.graded == "T"
    assert t.derived == 1


def test_an_ingroup_holding_only_derived_alleles_is_polymorphic():
    """Two different mutations below the MRCA leave no copy of its state.

    Polymorphism is read off the alleles present, so the site is graded at the
    MRCA with every copy derived rather than treated as ingroup-fixed.
    """
    ts, ingroup, panel = _ladder_ts([("i0", "C", 0.5), ("i1", "G", 0.5),
                                     ("i2", "G", 0.5)])
    t = site_truths(ts, ingroup, panel)[5]
    assert t.polymorphic
    assert t.graded == "A"
    assert t.derived == 3


def test_a_fixed_site_is_counted_at_the_panel_root():
    """A mutation above the MRCA fixes T in the ingroup. The panel root still
    carries A, so the site is fixed derived."""
    ts, ingroup, panel = _ladder_ts([("mrca", "T", 2.0)])
    t = site_truths(ts, ingroup, panel)[5]
    assert not t.polymorphic
    assert t.at_ingroup_mrca == "T"
    assert t.graded == "A"
    assert t.derived == 3
    assert t.site_class == "fixed_derived"
    # Without a panel every site is graded at the ingroup MRCA.
    assert site_truths(ts, ingroup)[5].derived == 0
    assert site_truths(ts, ingroup)[5].site_class == "fixed_ancestral"


def test_missing_genotypes_are_not_counted():
    ts, ingroup, panel = _ladder_ts([("i1", "T", 0.5)], missing=("i2",))
    t = site_truths(ts, ingroup, panel)[5]
    assert t.polymorphic
    assert t.derived == 1


class TestRobustnessBins:
    """The heatmap's fixed columns and end bins hold ingroup-fixed sites only."""

    @staticmethod
    def _rc():
        import _robustness_common as rc
        return rc

    def test_classes_follow_ingroup_polymorphism(self):
        rc = self._rc()
        assert rc._classify(True, 2) == "biallelic_poly"
        assert rc._classify(True, 3) == "polyallelic"
        assert rc._classify(False, 2) == "biallelic_mono"
        assert rc._classify(False, 3) == "polyallelic_mono"

    def test_a_polymorphic_site_with_every_copy_derived_is_not_fixed(self):
        rc = self._rc()
        n = 20
        summary = {"stats": {
            f"polyallelic|{n}|1": [5, 5, 1.0, 1.0, 5.0],
            f"polyallelic_mono|{n}|1": [7, 7, 2.0, 2.0, 7.0],
            f"biallelic_mono|0|0": [11, 11, 3.0, 3.0, 11.0],
            f"biallelic_poly|3|0": [13, 13, 4.0, 4.0, 13.0],
        }}
        fixed = dict(rc.CATS)["fixed"]
        assert rc.summary_col_count(summary, "fixed", fixed, n) == 7 + 11
        counts = rc.summary_per_bin_count(summary, n)
        assert counts[n] == 7
        assert counts[0] == 11
        assert counts[3] == 13
