"""``Likelihood.posterior``: kernel, prior and normalisation in one call."""
import numpy as np
import pytest
from ancestree.posterior import Grade
import tskit
from scipy.special import logsumexp

import ancestree as anc
from ancestree import ARGBasedInference, JC69
from ancestree.sites import Site
from testing._helpers import post
from testing._helpers import QUICKSTART_TREES


@pytest.fixture(scope="module")
def tree_and_site(small_ts):
    """The leftmost local tree of the shared ARG, and its first site."""
    sample_map = anc.TskitLocalTree.sample_map_from_individuals(small_ts)
    tree = anc.TskitLocalTree(small_ts, position=0.0, sample_map=sample_map)
    tree.time_scale = 1e-5
    site = next(iter(anc.TskitSource(small_ts, sample_map=sample_map)))
    return tree, site


class TestPosterior:
    """The single-call path agrees with doing it by hand."""

    @pytest.mark.parametrize("model", [anc.JC69(), anc.K2(kappa=4.0), anc.HKY(kappa=4.0)])
    def test_matches_the_explicit_computation(self, tree_and_site, model):
        """Identical to log_likelihoods + prior + softmax."""
        tree, site = tree_and_site
        bc = anc.BaseComposition.from_counts(A=300, C=200, G=200, T=300)
        kernel = anc.Likelihood(model, base_composition=bc)
        log_p = (kernel.log_likelihoods(tree, [site])[0]
                 + anc.StationaryPrior(model, bc).log_probs([site])[0])
        expected = np.exp(log_p - logsumexp(log_p))
        assert np.allclose(kernel.posterior(tree, site).values, expected)

    def test_sums_to_one(self, tree_and_site):
        """The returned posterior is normalised."""
        tree, site = tree_and_site
        post = anc.Likelihood(anc.JC69()).posterior(tree, site)
        assert post.values.sum() == pytest.approx(1.0)
        assert post.alleles == anc.STATES

    def test_honours_an_explicit_prior(self, tree_and_site):
        """A supplied prior replaces the stationary default."""
        tree, site = tree_and_site
        model = anc.JC69()
        skewed = anc.StationaryPrior(
            model, anc.BaseComposition.from_counts(A=970, C=10, G=10, T=10),
        )
        default = anc.Likelihood(model).posterior(tree, site)
        assert anc.Likelihood(model).posterior(tree, site, prior=skewed)["A"] > default["A"]


    def test_degenerate_row_falls_back_to_uniform(self, tree_and_site):
        """An all-``-inf`` log-posterior gives uniform rather than NaN."""
        from ancestree.likelihood import _normalise

        values, n_bad = _normalise(np.full((1, 4), -np.inf), 4)
        assert n_bad == 1
        assert np.allclose(values, 0.25)


class TestGrade:
    """:class:`ancestree.posterior.Grade`: the record ``grade`` returns."""

    @pytest.fixture
    def inference(self, small_ts):
        """An ARG-mode inference over the shared tree sequence."""
        return anc.Inference.from_arg(small_ts, anc.JC69(), mu=5e-8, progress=False)

    def test_a_grade_over_the_stream_is_grade_itself(self, inference, small_ts):
        truth = Grade.truth_mapping(small_ts)
        assert Grade(inference.infer(), truth) == inference.grade(truth)

    def test_truth_at_the_panel_root_is_the_ancestral_state(self):
        ts = tskit.load(QUICKSTART_TREES)
        assert Grade.truth_at_focal(ts, "panel_root") == Grade.truth_mapping(ts)

    def test_truth_at_the_ingroup_mrca_is_the_ingroup_allele_where_fixed(self):
        """At a site fixed within the ingroup, the state at its MRCA is that allele."""
        ts = tskit.load(QUICKSTART_TREES)
        ingroup = [f"i{i}" for i in range(6)]
        truth = Grade.truth_at_focal(
            ts, "ingroup_mrca", ingroup_samples=ingroup,
            outgroup_samples=["o0", "o1"])
        nodes = anc.TskitLocalTree.sample_map_from_individuals(ts)
        fixed = 0
        for v in ts.variants():
            alleles = {v.alleles[v.genotypes[nodes[s]]] for s in ingroup}
            if len(alleles) == 1:
                fixed += 1
                assert truth[int(v.site.position)] == next(iter(alleles))
        assert fixed > 0
        assert truth != Grade.truth_mapping(ts)

    def test_a_tree_sequence_truth_is_read_at_the_ingroup_mrca_by_default(self):
        ts = tskit.load(QUICKSTART_TREES)
        ingroup = [f"i{i}" for i in range(6)]
        inf = anc.Inference.from_arg(
            ts, anc.JC69(), mu=5e-8, progress=False,
            ingroup_samples=ingroup, outgroup_samples=["o0", "o1"])
        mrca = Grade.truth_at_focal(
            ts, "ingroup_mrca", ingroup_samples=ingroup, outgroup_samples=["o0", "o1"])
        assert inf.grade(ts) == inf.grade(mrca)
        assert inf.grade(ts).map_recovery > inf.grade(Grade.truth_mapping(ts)).map_recovery

    def test_the_truth_node_is_chosen_independently_of_the_reporting_node(self):
        """A run reporting at the ingroup MRCA graded at the panel root."""
        ts = tskit.load(QUICKSTART_TREES)
        ingroup = [f"i{i}" for i in range(6)]
        inf = anc.Inference.from_arg(
            ts, anc.JC69(), mu=5e-8, progress=False,
            ingroup_samples=ingroup, outgroup_samples=["o0", "o1"])
        root = inf.grade(ts, focal="panel_root")
        assert root == inf.grade(Grade.truth_mapping(ts))
        res = list(inf.infer())
        assert root == Grade(res, ts, focal="panel_root",
                             ingroup_samples=ingroup, outgroup_samples=["o0", "o1"])
        fixed = ~anc.PolymorphicSiteFilter(samples=ingroup)
        assert Grade(res, ts, filter=fixed, ingroup_samples=ingroup,
                     outgroup_samples=["o0", "o1"]).brier < 0.01
        assert Grade(res, ts, filter=fixed, focal="panel_root", ingroup_samples=ingroup,
                     outgroup_samples=["o0", "o1"]).brier > 0.1

    def test_stored_posteriors_write_the_same_file(self, tmp_path):
        """A writer given the stored stream matches one that reruns infer()."""
        ts = tskit.load(QUICKSTART_TREES)
        ingroup = [f"i{i}" for i in range(6)]
        inf = anc.Inference.from_arg(
            ts, anc.JC69(), mu=5e-8, progress=False,
            ingroup_samples=ingroup, outgroup_samples=["o0", "o1"])
        stored = list(inf.infer())
        inf.to_arg(tmp_path / "rerun.trees")
        inf.to_arg(tmp_path / "stored.trees", posteriors=stored)
        a = [(x.pos, x.aa, x.posterior) for x in anc.Reader(tmp_path / "rerun.trees").annotations()]
        b = [(x.pos, x.aa, x.posterior) for x in anc.Reader(tmp_path / "stored.trees").annotations()]
        assert a == b

    @pytest.mark.parametrize("sample_map", [False, True])
    def test_reader_grades_an_annotated_file_like_the_inference(
            self, tmp_path, sample_map):
        """The panel, named by both lists or by a partial ``sample_map``, is
        read back from the file's provenance."""
        ts = tskit.load(QUICKSTART_TREES)
        panel = [f"i{i}" for i in range(4)] + ["o0"]
        kw = dict(ingroup_samples=panel[:4], outgroup_samples=panel[4:])
        if sample_map:
            full = anc.TskitLocalTree.default_sample_map(ts)
            kw = dict(sample_map={s: full[s] for s in panel})
        inf = anc.Inference.from_arg(
            ts, anc.JC69(), mu=5e-8, progress=False, **kw)
        out = tmp_path / "annotated.trees"
        inf.to_arg(out)
        for focal in ("ingroup_mrca", "panel_root"):
            got, want = anc.Reader(out).grade(ts, focal=focal), inf.grade(ts, focal=focal)
            assert got.n_sites == want.n_sites
            assert got.map_recovery == want.map_recovery
            assert got.brier == pytest.approx(want.brier, abs=1e-6)

    def test_tree_sequence_matches_the_mapping(self, inference, small_ts):
        """Passing the tree sequence derives the same truth as doing it by hand."""
        truth = {int(s.position): s.ancestral_state for s in small_ts.sites()}
        assert inference.grade(small_ts) == inference.grade(truth)

    def test_unpacks_positionally(self, inference, small_ts):
        """It stays a tuple, so existing three-way unpacking keeps working."""
        graded = inference.grade(small_ts)
        map_recovery, brier, n = graded
        assert (map_recovery, brier, n) == (
            graded.map_recovery, graded.brier, graded.n_sites)

    def test_repr_reads_as_a_sentence(self, inference, small_ts):
        """The repr names the site count and both scores."""
        text = repr(inference.grade(small_ts))
        assert text.startswith(f"{inference.grade(small_ts).n_sites} sites: ")
        assert "MAP recovery" in text and "mean Brier" in text

    def test_empty_truth_grades_nothing(self, inference):
        """No overlapping sites gives zeroed scores rather than a division error."""
        graded = inference.grade({})
        assert graded == (0.0, 0.0, 0)

    def test_an_unmatched_ingroup_is_refused(self, small_ts):
        """An ingroup naming no sample of the truth must not be graded at the
        ARG root.

        The samples of a plain msprime tree sequence carry no name metadata,
        so they fall back to their node ids and names of any other convention
        match nothing. An ingroup that resolved no node made every site's
        truth its ancestral state, the allele at the ARG root rather than at
        the ingroup MRCA the run reports at, with no warning.
        """
        assert not anc.TskitLocalTree.sample_map_from_individuals(small_ts)
        names = [f"n{int(n)}" for n in small_ts.samples()[:4]]
        with pytest.raises(ValueError, match="no ingroup sample of the truth matches"):
            Grade.truth_at_focal(small_ts, "ingroup_mrca", ingroup_samples=names)
        inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
        with pytest.raises(ValueError, match="Pass sample_map="):
            Grade(inf.infer(), small_ts, ingroup_samples=names)

    # ----------------------------------------------------------------- grade()
    def test_grade_skips_sites_absent_from_truth(self, small_ts):
        inf = ARGBasedInference(small_ts, JC69(), mu=1e-8)
        variants = list(small_ts.variants())
        # Truth for only the first half of the sites, so the rest hit the
        # `t is None: continue` branch.
        truth = {int(v.site.position): v.alleles[0] for v in variants[: len(variants) // 2]}
        map_recovery, brier, n = inf.grade(truth)
        assert 0.0 <= map_recovery <= 1.0
        assert 0.0 <= brier <= 2.0
        assert n == len(truth)

    def test_grade_tolerates_out_of_alphabet_truth_allele(self, small_ts):
        """A truth allele outside the model alphabet (e.g. 'N')
        scores as a full miss rather than aborting grade() with a KeyError."""
        inf = ARGBasedInference(small_ts, JC69(), mu=1e-8, progress=False)
        variants = list(small_ts.variants())
        # Real truth for the first site, an out-of-alphabet 'N' for the second.
        truth = {int(variants[0].site.position): variants[0].alleles[0]}
        if len(variants) > 1:
            truth[int(variants[1].site.position)] = "N"
        map_recovery, brier, n = inf.grade(truth)  # must not raise
        assert 0.0 <= map_recovery <= 1.0
        assert 0.0 <= brier <= 2.0
        assert n == len(truth)

    @staticmethod
    def _grade() -> Grade:
        site = Site(chrom="1", pos=1, alleles=("A", "C"),
                    tip_alleles={"a": "A"})
        return Grade([(site, post("A", 0.9))], {1: "A"})

    def test_compares_as_its_tuple_and_hashes_like_it(self):
        grade = self._grade()
        assert grade == (1.0, grade.brier, 1)
        assert grade != "not a grade"
        assert hash(grade) == hash((1.0, grade.brier, 1))

    def test_truth_accepts_pairs(self):
        assert Grade.truth_mapping([(1, "A"), ("2", "C")]) == {1: "A", 2: "C"}


class TestGradeAgainstADifferentlyNamedTruth:
    """Grading a run whose panel is named as a VCF names it.

    A panel read from a VCF written by ``ts.write_vcf`` carries haplotype
    names, while the samples of the simulating tree sequence carry no name
    metadata and fall back to their node ids as strings. The ingroup then
    matches no sample of the truth, and the guard that refuses that case named
    a remedy, ``sample_map=``, that :meth:`Inference.grade` had no parameter
    for.
    """

    #: One outgroup individual, so the ingroup MRCA sits below the panel root.
    OUTGROUP = ("tsk_9_h0", "tsk_9_h1")

    @staticmethod
    def _vcf_names(ts):
        """``{tsk_<individual>_h<haplotype>: node}`` over the samples of ``ts``."""
        return {f"tsk_{int(n) // 2}_h{int(n) % 2}": int(n) for n in ts.samples()}

    @pytest.fixture
    def inference(self, small_ts):
        return anc.Inference.from_arg(
            small_ts, JC69(), mu=1e-8, progress=False,
            sample_map=self._vcf_names(small_ts),
            outgroup_samples=list(self.OUTGROUP))

    def test_without_a_sample_map_the_ingroup_matches_nothing(
            self, inference, small_ts):
        assert not anc.TskitLocalTree.sample_map_from_individuals(small_ts)
        with pytest.raises(ValueError, match="Pass sample_map="):
            inference.grade(small_ts)

    def test_a_sample_map_grades_against_the_tree_sequence(
            self, inference, small_ts):
        names = self._vcf_names(small_ts)
        graded = inference.grade(small_ts, sample_map=names)
        ingroup = [s for s in names if s not in self.OUTGROUP]
        assert graded.n_sites == small_ts.num_sites > 0
        assert graded == Grade(
            inference.infer(), small_ts, ingroup_samples=ingroup,
            outgroup_samples=list(self.OUTGROUP), sample_map=names)


class TestGradingReadsTheScoredPanel:
    """The truth is read on the tree sequence the inference scored."""

    ING = ["i0", "i1", "i2", "i3"]
    OUT = ["o0"]

    def test_samples_in_neither_list_are_dropped(self):
        ts = tskit.load(QUICKSTART_TREES)
        full = anc.TskitLocalTree.default_sample_map(ts)
        small, small_map = anc.TskitLocalTree.restrict(
            ts, {s: full[s] for s in self.ING + self.OUT})
        got = Grade.truth_at_focal(ts, "panel_root", ingroup_samples=self.ING,
                                   outgroup_samples=self.OUT)
        want = Grade.truth_at_focal(small, "panel_root",
                                    ingroup_samples=self.ING,
                                    outgroup_samples=self.OUT,
                                    sample_map=small_map)
        assert got == want

    def test_names_match_by_individual(self):
        from testing._helpers import DEMO_TREES

        ts = tskit.load(DEMO_TREES)
        ing = ["i0_h0", "i0_h1", "i1_h0", "i1_h1"]
        by_name = Grade.truth_at_focal(ts, "panel_root", ingroup_samples=ing,
                                       outgroup_samples=["o1"])
        by_haplotype = Grade.truth_at_focal(
            ts, "panel_root", ingroup_samples=ing,
            outgroup_samples=["o1_h0", "o1_h1"])
        assert by_name == by_haplotype

    def test_a_partial_sample_map_is_graded_on_its_panel(self):
        ts = tskit.load(QUICKSTART_TREES)
        full = anc.TskitLocalTree.default_sample_map(ts)
        sub = {s: full[s] for s in self.ING}
        inf = anc.Inference.from_arg(ts, mu=5e-8, focal="panel_root",
                                     sample_map=sub, progress=False)
        small, small_map = anc.TskitLocalTree.restrict(ts, sub)
        truth = Grade.truth_at_focal(small, "panel_root", sample_map=small_map)
        assert inf.grade(ts) == Grade(inf.infer(), truth)

