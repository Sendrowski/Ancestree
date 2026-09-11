"""Slow benchmark assertions.

Reads the JSON reports under ``results/reports/`` and asserts that
key agreement / recovery metrics stay within hard thresholds. The
reports themselves are built by the Snakemake workflow (see
``workflow/Snakefile``). These tests just gate them.

Thresholds are calibrated from current runs with a small safety margin
so the suite catches a real drift but tolerates the usual ±1% of
seeded msprime jitter and scipy-MLE drift.

Run::

    make benchmark      # snakemake → reports → pytest -m slow
    pytest -m slow      # if reports already exist

Each test skips with a clear message when its report is missing so a
partial benchmark run still gives useful feedback.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


REPORTS = Path(__file__).resolve().parents[1] / "results" / "reports"


def _load_report(name: str) -> dict:
    """Load a JSON report under ``results/reports/``, skipping if missing.

    :param name: Filename (e.g. ``"polarbear_agreement.json"``).
    :return: Parsed JSON content.
    """
    path = REPORTS / name
    if not path.exists():
        pytest.skip(f"{path} not found — run `make benchmark` to generate it")
    return json.loads(path.read_text())


# ------------------------------------------------------------------ B2: PolarBEAR


@pytest.mark.slow
class TestB2PolarBEARAgreement:
    """Ancestree under ```` should reproduce
    PolarBEAR's MAP picks and posteriors site-for-site, and should
    recover the simulator's true ancestral allele at essentially the
    same rate."""

    @pytest.fixture(scope="class")
    def report(self):
        return _load_report("polarbear_agreement.json")

    def test_no_sites_dropped(self, report):
        assert report["n_sites_polarbear"] == report["n_sites_ancestree"]
        assert report["n_shared"] == report["n_sites_polarbear"]

    def test_map_agreement_on_nonhomoplastic(self, report):
        assert report["nonhom_map_agreement_rate"] >= 0.99

    def test_posterior_match_within_numerical_tolerance(self, report):
        assert report["nonhom_per_allele_max_abs_diff"] <= 1e-3
        assert report["nonhom_per_allele_mean_abs_diff"] <= 1e-4

    def test_truth_recovery_floor(self, report):
        assert report["truth_recovery_ancestree"] >= 0.80
        assert report["truth_recovery_polarbear"] >= 0.80

    def test_truth_recovery_parity(self, report):
        # Two tools agreeing on every site must also have the same
        # truth-recovery rate (modulo the rare informative-vs-not split).
        delta = abs(
            report["truth_recovery_ancestree"]
            - report["truth_recovery_polarbear"]
        )
        assert delta <= 0.02


# --------------------------------------------------------------------- B3: est-sfs


@pytest.mark.slow
class TestB3EstSFSAgreement:
    """Across the parameter grid (sim × model × prior × n_out), Ancestree's
    fixed-tree fit should agree with fastDFE's est-sfs implementation on
    the majority of polymorphic non-homoplastic sites, and recover the
    truth at a rate close to fastDFE's."""

    @pytest.fixture(scope="class")
    def report(self):
        return _load_report("estsfs_agreement.json")

    def test_all_cells_meet_agreement_floor(self, report):
        # 0.80 floor accommodates the n_out=3 cells where Ancestree's
        # per-site MAP beats fastdfe's by ~16 pp (90.6% vs 74.4%), both
        # tools are converging to different posteriors at the same fitted
        # K because fastdfe uses a uniform prior on monomorphic
        # ingroup sites.
        bad: list = []
        n_seen = 0
        for key, cell in report["cells"].items():
            rate = (cell.get("agreement") or {}).get("nonhom_agree_rate")
            if rate is None:
                continue  # fastdfe does not support every cell (e.g. R6 model)
            n_seen += 1
            if rate < 0.80:
                bad.append((key, rate))
        # Without this the test passes on zero comparisons, which is what a
        # renamed key or an unbuilt report would silently produce.
        assert n_seen >= 30, f"only {n_seen} cells carried nonhom_agree_rate"
        assert not bad, f"cells below 0.80 nonhom_agree_rate: {bad}"

    def test_all_cells_truth_recovery_floor(self, report):
        # 0.65 floor accommodates GTR-sim × n_out=1 cells where the
        # closest outgroup is the only signal AND the inference model
        # mismatches the sim, recovery floors at ~0.68 there.
        bad: list = []
        n_seen = 0
        for key, cell in report["cells"].items():
            r = cell.get("ancestree_truth_recovery")
            if r is None:
                continue
            n_seen += 1
            if r < 0.65:
                bad.append((key, r))
        assert n_seen >= 50, f"only {n_seen} cells carried a truth recovery"
        assert not bad, f"cells below 0.65 ancestree_truth_recovery: {bad}"

    def test_ancestree_not_more_than_10pp_below_fastdfe_on_truth(self, report):
        # One-sided: Ancestree must not lose more than 10 pp of truth
        # recovery vs fastdfe. Ancestree beating fastdfe is fine (and
        # expected at n>=2 after fixing the monomorphic-ingroup prior
        # delta: the n>=2 cells run ~15 pp above fastdfe / EST-SFS C
        # because the count-proportional Kingman prior + delta on
        # monomorphic-ingroup sites is sharper at high-count polymorphic
        # sites than EST-SFS's flatter prior).
        bad: list = []
        n_seen = 0
        for key, cell in report["cells"].items():
            a = cell.get("ancestree_truth_recovery")
            f = cell.get("fastdfe_truth_recovery")
            if a is None or f is None:
                continue  # fastdfe does not support every cell (e.g. R6 model)
            n_seen += 1
            if f - a > 0.10:
                bad.append((key, a, f))
        assert n_seen >= 30, f"only {n_seen} cells compared against fastdfe"
        assert not bad, (
            f"cells where Ancestree truth-recovery dropped >10pp below "
            f"fastdfe: {bad}"
        )

    def test_ancestree_K_matches_estsfs_baseline_within_20pct(self, report):
        """Per-cell, per-branch divergence MLE within 20% of the EST-SFS C
        baseline.

        Catches the bug class where Ancestree's likelihood subtly diverges
        from EST-SFS's at the MLE level (e.g. a monomorphic-ingroup-prior
        error driving the K-MLE to several× truth at n_out=1 while soft
        per-site recovery stays within bounds). Per-site agreement metrics
        alone do not reliably catch K-level discrepancies, so K is tested
        directly against the independent C implementation that EST-SFS
        publishes.

        20% tolerance accommodates sampling noise on small per-cell
        polymorphic-site counts. It is a tail bound, not the expected
        agreement: across the 108 per-branch comparisons the median is 0.4-1.0%
        depending on outgroup count, which is the number the manuscript quotes.
        A handful of cells at 15-18% is that tail, and the median is asserted
        below so a real drift is caught even though the bound is loose, if
        the fit degraded, the median would move, not just the extremes.
        """
        bad: list = []
        rels: list = []
        for key, cell in report["cells"].items():
            anc = (cell.get("ancestree_meta") or {}).get("outgroup_divergence")
            base = (cell.get("baseline_meta") or {}).get(
                "fitted_outgroup_divergence"
            )
            if anc is None or base is None or len(anc) != len(base):
                continue
            for k_idx, (a, b) in enumerate(zip(anc, base)):
                if b is None or b <= 0:
                    continue
                rel = abs(a - b) / b
                rels.append(rel)
                if rel > 0.20:
                    bad.append((key, k_idx + 1, float(a), float(b), float(rel)))
        assert not bad, (
            f"cells with >20% per-branch K mismatch vs EST-SFS baseline: {bad}"
        )
        # The bound above is a tail. This is the claim the manuscript makes.
        median = sorted(rels)[len(rels) // 2]
        assert median < 0.03, (
            f"median per-branch K mismatch is {median:.1%}, expected ~1%: the "
            f"fit has degraded rather than a few cells landing in the tail"
        )


# ----------------------------------------------------------- B3 adaptive π_i


@pytest.mark.slow
class TestB3AdaptivePiAgreement:
    """The fitted per-bin adaptive π_i should track fastDFE's est-sfs π_i
    on a long, well-powered simulation. The comparison is on the bins fastDFE
    reports, and only on bins with sufficient site counts, since bins below
    the noisy-fit threshold can drift more."""

    @pytest.fixture(scope="class")
    def report(self):
        return _load_report("estsfs_pi_comparison.json")

    def test_outgroup_divergence_within_5pct(self, report):
        a = report["ancestree_inference_meta"]["outgroup_divergence"]
        f = report["fastdfe_inference_meta"]["outgroup_divergence"]
        assert len(a) == len(f)
        for ai, fi in zip(a, f):
            rel = abs(ai - fi) / fi
            assert rel <= 0.05, f"ancestree={ai} fastdfe={fi} rel={rel}"





@pytest.mark.slow
class TestB5Polytomy:
    """Branch-collapsing the ARG into polytomies should not break MAP
    agreement at threshold=0 (no collapse), and should stay close to the
    no-collapse baseline as the threshold grows."""

    @pytest.fixture(scope="class")
    def report(self):
        return _load_report("polytomy.json")

    def test_no_collapse_matches_baseline_exactly(self, report):
        zero = next(s for s in report["sweep"] if s["threshold"] == 0)
        assert zero["map_agree_with_baseline"] == 1.0
        assert zero["truth_recovery"] >= 0.85

    def test_collapsing_degrades_gracefully(self, report):
        # Aggressive polytomy collapse legitimately reduces agreement
        # with the no-collapse baseline. The requirement is graceful
        # degradation, a broken collapse would crater to ~random.
        for s in report["sweep"]:
            assert s["map_agree_with_baseline"] >= 0.90, (
                f"threshold={s['threshold']}: agreement "
                f"{s['map_agree_with_baseline']:.3f} below 0.90"
            )


# ---------------------------------------------------------------- outgroup effect


@pytest.mark.slow
class TestOutgroupEffect:
    """Adding outgroups should not *decrease* MAP accuracy on the same
    truth set, more information should help (or at worst be neutral)."""

    @pytest.fixture(scope="class")
    def report(self):
        return _load_report("baseline.json")

    def test_accuracy_does_not_collapse_with_more_outgroups(self, report):
        cells = report["cells"]
        accs = {int(k): c["accuracy"] for k, c in cells.items()}
        # The n_out=0 case is the floor. Every n_out > 0 should be at
        # least as good (within a small jitter).
        base = accs[0]
        for n_out, acc in sorted(accs.items()):
            if n_out == 0:
                continue
            assert acc >= base - 0.02, (
                f"n_out={n_out} accuracy {acc:.3f} below n_out=0 baseline "
                f"{base:.3f}"
            )
