"""A mistyped outgroup must be rejected on the streaming path too.

``_check_samples_present`` received the materialised site list, which is empty
whenever the source streams, and returned early on empty. Streaming is the
auto-resolved default for a path, a SiteSource or a TreeSequence, so the guard
never ran in production. Its own docstring names the failure: the ladder is
built from the names, no tip ever matches, every branch rate pegs at its lower
bound, and the fit reports convergence while emitting a uniform posterior.
"""
import pytest
import tskit

import ancestree as anc

TREES = "docs/_static/quickstart.trees"
ING = [f"i{i}" for i in range(6)]


def test_a_mistyped_outgroup_is_rejected_when_streaming():
    ts = tskit.load(TREES)
    with pytest.raises(ValueError, match="absent from the source panel"):
        anc.Inference.from_fixed_tree(
            ts, ingroup_samples=ING, outgroup_samples=["o0", "NOT_A_SAMPLE"],
            model=anc.JC69(), n_target_sites=50_000, progress=False)


def test_correct_outgroups_are_accepted_when_streaming():
    ts = tskit.load(TREES)
    inf = anc.Inference.from_fixed_tree(
        ts, ingroup_samples=ING, outgroup_samples=["o0", "o1"],
        model=anc.JC69(), n_target_sites=50_000, progress=False)
    assert inf is not None


def test_a_mistyped_ingroup_is_rejected_when_streaming():
    """A mistyped ingroup id was dropped from the ingroup without a word.

    The ingroup list drives the polymorphic-config projection, so a shortened
    ingroup moves the fitted branch rates and the MAP ancestral alleles while
    the run reports convergence and stays bit-identical to a genuinely
    shorter ingroup list.
    """
    ts = tskit.load(TREES)
    with pytest.raises(ValueError, match=r"ingroup \['i5_TYPO'\]"):
        anc.Inference.from_fixed_tree(
            ts, ingroup_samples=ING[:5] + ["i5_TYPO"],
            outgroup_samples=["o0", "o1"], model=anc.JC69(),
            n_target_sites=50_000, progress=False)


def test_correct_ingroup_names_are_accepted_when_streaming():
    ts = tskit.load(TREES)
    inf = anc.Inference.from_fixed_tree(
        ts, ingroup_samples=ING, outgroup_samples=["o0", "o1"],
        model=anc.JC69(), n_target_sites=50_000, progress=False)
    assert inf.tree.ingroup_samples == tuple(ING)


def _sites(tip_ids, n=60):
    """``n`` biallelic sites over ``tip_ids``, alternating which tip carries C."""
    return [
        anc.Site(chrom="1", pos=i + 1, alleles=("A", "C"),
                 tip_alleles={sid: ("C" if (i + j) % 3 == 0 else "A")
                              for j, sid in enumerate(tip_ids)})
        for i in range(n)
    ]


def test_a_mistyped_ingroup_is_rejected_from_a_site_list():
    tips = ["o0", "o1", "i0", "i1", "i2"]
    with pytest.raises(ValueError, match=r"ingroup \['i2_TYPO'\]"):
        anc.FixedTreeInference(
            _sites(tips), anc.JC69(), ingroup_samples=["i0", "i1", "i2_TYPO"],
            outgroup_samples=["o0", "o1"], n_target_sites=1000,
            fit_required=False, progress=False)


def test_an_ingroup_named_per_individual_is_accepted():
    """``i0`` designates ``i0_h0`` and ``i0_h1``, as ``count_alleles`` reads it."""
    tips = ["o0", "o1", "i0_h0", "i0_h1", "i1_h0", "i1_h1"]
    inf = anc.FixedTreeInference(
        _sites(tips), anc.JC69(), ingroup_samples=["i0", "i1"],
        outgroup_samples=["o0", "o1"], n_target_sites=1000,
        fit_required=False, progress=False)
    assert inf.tree.ingroup_samples == ("i0", "i1")


def test_an_ingroup_the_panel_never_carries_is_accepted():
    """No ingroup id in the panel is the outgroup-only mode.

    The ladder carries no ingroup tip there and the ingroup enters as a flat
    vector, which is how the simulated-outgroup fixtures are built.
    """
    inf = anc.FixedTreeInference(
        _sites(["o0", "o1"]), anc.JC69(), ingroup_samples=["i0", "i1"],
        outgroup_samples=["o0", "o1"], n_target_sites=1000,
        fit_required=False, progress=False)
    assert inf.tree.ingroup_samples == ("i0", "i1")
