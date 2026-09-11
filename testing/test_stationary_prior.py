"""Tests for StationaryPrior."""
import numpy as np
import pytest

from ancestree import (
    F81,
    HKY,
    JC69,
    K2,
    BaseComposition,
    IngroupWeight,
    Site,
    StationaryPrior,
)


def _dummy_sites(n: int) -> list[Site]:
    """``n`` minimal Sites, StationaryPrior is site-invariant, contents do not matter."""
    return [
        Site(chrom="1", pos=i + 1, alleles=("A", "C"), tip_alleles={})
        for i in range(n)
    ]


class TestConstruction:
    def test_is_not_an_ingroup_weight(self):
        """It is the prior at the reporting node, not the ingroup's likelihood."""
        assert not isinstance(StationaryPrior(JC69()), IngroupWeight)

    def test_rejects_non_model(self):
        with pytest.raises(TypeError, match="SubstitutionModel"):
            StationaryPrior([0.25, 0.25, 0.25, 0.25])  # not a model

    def test_caches_log_prior_at_construction(self):
        sp = StationaryPrior(JC69())
        assert sp._log_prior_per_state.shape == (4,)
        assert np.allclose(sp._log_prior_per_state, np.log(0.25))


class TestLogProbs:
    def test_jc69_is_uniform(self):
        sp = StationaryPrior(JC69())
        log_p = sp.log_probs(_dummy_sites(10))
        assert log_p.shape == (10, 4)
        assert np.allclose(log_p, np.log(0.25))

    def test_k2_is_uniform(self):
        # K2 has uniform stationary by symmetry (even with kappa != 1).
        sp = StationaryPrior(K2(kappa=3.5))
        log_p = sp.log_probs(_dummy_sites(5))
        assert log_p.shape == (5, 4)
        assert np.allclose(log_p, np.log(0.25))

    def test_every_row_identical(self):
        sp = StationaryPrior(JC69())
        log_p = sp.log_probs(_dummy_sites(7))
        for row in log_p:
            assert np.allclose(row, log_p[0])

    def test_empty_sites_yields_empty_array(self):
        sp = StationaryPrior(JC69())
        log_p = sp.log_probs([])
        assert log_p.shape == (0, 4)


class TestNonUniformComposition:
    """The branch that matters: a supplied base composition must reach the row.

    Every other test in this file uses a symmetric model with no composition,
    so they all assert log(0.25) and would pass against a hardcoded uniform
    row. StationaryPrior is the only prior ARG and local-tree mode accept, so
    this is the load-bearing path.
    """

    @staticmethod
    def _skewed():
        """A composition far from uniform, with its expected pi."""
        bc = BaseComposition.from_counts(A=400, C=100, G=100, T=400)
        return bc, np.asarray(bc.pi, dtype=float)

    @pytest.mark.parametrize("model_cls", [F81, HKY])
    def test_the_row_equals_the_supplied_pi(self, model_cls):
        bc, pi = self._skewed()
        prior = StationaryPrior(model_cls(), base_composition=bc)
        rows = np.exp(prior.log_probs(_dummy_sites(3)))
        assert rows.shape == (3, 4)
        np.testing.assert_allclose(rows, np.tile(pi, (3, 1)), rtol=1e-12)

    def test_the_row_is_not_uniform(self):
        """Guards the guard: a uniform pi would make the check above vacuous."""
        _, pi = self._skewed()
        assert np.max(np.abs(pi - 0.25)) > 0.1, pi

    def test_the_composition_applies_under_a_symmetric_model_too(self):
        """The supplied composition is the prior, whatever the model.

        JC69's own stationary distribution is uniform, but the prior at the
        reporting node is a separate choice: this is what ``--prior
        composition`` selects against ``--prior uniform``. Pinned because the
        name invites the opposite assumption.
        """
        bc, pi = self._skewed()
        rows = np.exp(StationaryPrior(JC69(), base_composition=bc)
                      .log_probs(_dummy_sites(2)))
        np.testing.assert_allclose(rows, np.tile(pi, (2, 1)), rtol=1e-12)

    def test_no_composition_is_uniform(self):
        """Without one, every model's row is uniform."""
        rows = np.exp(StationaryPrior(JC69()).log_probs(_dummy_sites(2)))
        np.testing.assert_allclose(rows, 0.25, rtol=1e-12)
