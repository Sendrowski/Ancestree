"""Tests for the :mod:`ancestree.cli` console interface.

Two layers of coverage:

- Parser-level: round-trip ``build_parser().parse_args(...)`` for
  the subcommands, plus the top-level ``--version`` and ``--help``
  short-circuits.
- End-to-end: synthesise a tiny tskit tree sequence with msprime
  and run ``ancestree arg`` against it via :func:`ancestree.cli.run`,
  asserting the output ``.trees`` / ``.vcf`` files exist and carry the
  expected ancestral-allele annotation.
"""
from __future__ import annotations
import argparse
import json
import logging


import msprime
import numpy as np
import pytest
import tskit
import ancestree as anc

from ancestree.cli import (
    _empirical_composition,
    _lib_default,
    _run_local_tree,
    build_parser,
    run,
)
from ancestree.inference import _individual_of
from ancestree.priors import NoIngroupWeight
from ancestree.settings import Settings
from ancestree.writers import PROVENANCE_HEADER, VCF_PROVENANCE_MARKER

from testing._helpers import ladder_panel, write_skeleton_vcf
from testing._helpers import DEMO_TREES, DEMO_VCF, QUICKSTART_TREES


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tiny_ts():
    """A miniature ARG with a handful of mutations, kept small so the
    end-to-end test stays fast."""
    ts = msprime.sim_ancestry(
        samples=4, ploidy=1,
        sequence_length=1e4, recombination_rate=1e-8,
        population_size=1e4, random_seed=11,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=11)
    assert not (ts.num_sites == 0), "simulation produced zero sites; bump --rate"
    return ts


@pytest.fixture(scope="module")
def tiny_trees_path(tiny_ts, tmp_path_factory):
    """``tiny_ts`` dumped to a ``.trees`` file for the CLI to consume."""
    p = tmp_path_factory.mktemp("cli_trees") / "tiny.trees"
    tiny_ts.dump(str(p))
    return p


def _vcf_provenance(path) -> dict:
    """The provenance record an annotated VCF carries in its header."""
    for line in path.read_text().splitlines():
        if line.startswith(VCF_PROVENANCE_MARKER):
            return json.loads(line[len(VCF_PROVENANCE_MARKER):])
    raise AssertionError(f"no provenance header in {path}")


def _aa_calls(path) -> list[str]:
    """The ``AA`` INFO values of every record in an annotated VCF."""
    calls = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        info = dict(kv.split("=", 1) for kv in line.split("\t")[7].split(";")
                    if "=" in kv)
        calls.append(info["AA"])
    return calls


# ---------------------------------------------------------------------------- parser-level


class TestParser:
    """Verify the argparse shape without invoking the inference engine."""

    def test_top_level_version_flag(self, capsys):
        """``--version`` exits with code 0 and prints ``ancestree <version>``."""
        from ancestree import __version__
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_no_subcommand_errors(self):
        """No subcommand → argparse error (SystemExit code 2)."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_fixed_tree_parses(self):
        parser = build_parser()
        ns = parser.parse_args([
            "fixed-tree",
            "--vcf", "in.vcf.gz",
            "--species-tree", "tree.nwk",
            "--ingroup", "a,b,c",
            "--outgroups", "o1,o2",
            "--model", "HKY", "--fit-kappa",
            "--ingroup-weight", "kingman",
            "--n-target-sites", "100000",
            "--out", "annot.vcf.gz",
        ])
        assert ns.command == "fixed-tree"
        assert ns.vcf == "in.vcf.gz"
        assert ns.species_tree == "tree.nwk"
        assert ns.ingroup == ["a", "b", "c"]
        assert ns.outgroups == ["o1", "o2"]
        assert ns.model == "HKY"
        assert ns.fit_kappa is True
        assert ns.ingroup_weight == "kingman"
        assert ns.prior == "composition"
        assert ns.n_target_sites == 100000
        assert ns.out == "annot.vcf.gz"
        # Defaults, which mirror FixedTreeInference's own: the fit is serial
        # unless --parallelize is given, and the worker count is then resolved
        # by the API as min(n_starts, CPU count).
        assert ns.parallelize is False
        assert ns.n_workers is None

    def test_fixed_tree_requires_vcf(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "fixed-tree",
                "--species-tree", "tree.nwk",
                "--ingroup", "a", "--outgroups", "o1",
                "--out", "annot.vcf",
            ])

    def test_arg_parses_defaults(self):
        parser = build_parser()
        ns = parser.parse_args([
            "arg", "--trees", "x.trees", "--out", "annot.vcf.gz",
        ])
        assert ns.command == "arg"
        # nargs="+": a posterior sample of genealogies is several paths.
        assert ns.trees == ["x.trees"]
        assert ns.out == "annot.vcf.gz"
        assert ns.model == "JC69"
        assert ns.prior == "composition"
        # Unset at the CLI layer: ARGBasedInference supplies the fallback rate
        # and warns, so the default lives in one place rather than two.
        assert ns.mu is None
        assert ns.n_workers == 1


    def test_global_verbose_quiet_mutually_exclusive(self):
        with pytest.raises(SystemExit) as exc:
            run([
                "-v", "--quiet", "arg", "--trees", "x.trees",
                "--out", "annot.vcf.gz",
            ])
        assert exc.value.code == 2

    def test_local_tree_parses_defaults(self):
        parser = build_parser()
        ns = parser.parse_args([
            "local-tree", "--vcf", "in.vcf.gz",
            "--sequence-length", "20000000", "--out", "annot.vcf.gz",
        ])
        assert ns.command == "local-tree"
        assert ns.vcf == "in.vcf.gz"
        assert ns.sequence_length == pytest.approx(2e7)
        assert ns.window == "8snp"
        assert ns.model == "JC69"
        assert ns.prior == "composition"
        # Both unset here; LocalTreeInference supplies the fallbacks and warns.
        assert ns.mu is None
        assert ns.rec_rate is None
        assert ns.block_size is None
        assert ns.n_time_bins == 32
        assert ns.handler.__name__ == "_run_local_tree"

    def test_local_tree_needs_neither_length_nor_chunk_size(self):
        """``local-tree`` runs with both omitted, chunking by default."""
        args = build_parser().parse_args([
            "local-tree", "--vcf", "in.vcf.gz", "--out", "annot.vcf.gz",
        ])
        assert args.sequence_length is None
        assert args.chunk_size is not None

    def test_verbose_accepted_after_subcommand(self):
        p = build_parser()
        args = p.parse_args(
            ["arg", "--trees", "x.trees", "--mu", "1e-8", "--out", "o.vcf", "-v"],
        )
        assert getattr(args, "verbose", 0) == 1

    def test_verbose_accepted_before_subcommand(self):
        p = build_parser()
        args = p.parse_args(
            ["-v", "arg", "--trees", "x.trees", "--mu", "1e-8", "--out", "o.vcf"],
        )
        assert getattr(args, "verbose", 0) == 1

    def test_no_verbosity_defaults(self):
        p = build_parser()
        args = p.parse_args(
            ["arg", "--trees", "x.trees", "--mu", "1e-8", "--out", "o.vcf"],
        )
        assert getattr(args, "verbose", 0) == 0
        assert getattr(args, "quiet", False) is False


# ---------------------------------------------------------------------------- end-to-end


class TestArgSubcommandE2E:
    """Drive :func:`ancestree.cli.run` against a real tiny tree sequence."""

    def test_arg_to_trees(self, tiny_trees_path, tmp_path):
        """``arg`` with a ``.trees`` output writes an annotated tree sequence."""
        out_path = tmp_path / "annot.trees"
        code = run([
            "arg",
            "--trees", str(tiny_trees_path),
            "--mu", "1e-8",
            "--out", str(out_path),
        ])
        assert code == 0
        assert out_path.exists()

        import tskit
        new_ts = tskit.load(str(out_path))
        n_with_meta = 0
        for site in new_ts.sites():
            assert site.ancestral_state in {"A", "C", "G", "T"}
            block = site.metadata.get("ancestree") if site.metadata else None
            if block is not None:
                n_with_meta += 1
                assert "map_allele" in block
                assert "max_prob" in block
                assert 0.0 <= block["max_prob"] <= 1.0
        assert n_with_meta > 0

    def test_arg_to_vcf(self, tiny_trees_path, tmp_path):
        """``arg`` with a ``.vcf`` output produces a VCF with ``AA`` INFO fields."""
        import cyvcf2
        out_path = tmp_path / "annot.vcf"
        code = run([
            "arg",
            "--trees", str(tiny_trees_path),
            "--mu", "1e-8",
            "--out", str(out_path),
        ])
        assert code == 0
        assert out_path.exists()

        rdr = cyvcf2.VCF(str(out_path))
        try:
            n_records = 0
            n_with_aa = 0
            n_blank = 0
            n_differs_from_ref = 0
            for record in rdr:
                n_records += 1
                aa = record.INFO.get("AA")
                if aa is not None:
                    n_with_aa += 1
                    assert aa in {"A", "C", "G", "T", "."}
                    if aa == ".":
                        n_blank += 1
                    elif aa != record.REF:
                        n_differs_from_ref += 1
                    prob = record.INFO.get("AA_prob")
                    assert prob is not None
                    assert 0.0 <= float(prob) <= 1.0
            assert n_records > 0
            assert n_with_aa > 0
            # A run emitting REF at every site, or the uncalled sentinel
            # everywhere, satisfies the alphabet check above. Neither is
            # inference. No threshold is set on the reference-differing
            # fraction, only that the call is not constant.
            assert n_blank == 0, f"{n_blank} sites came back uncalled"
            assert n_differs_from_ref > 0, "every call equals REF"

        finally:
            rdr.close()

    def test_arg_unknown_extension_rejected(self, tiny_trees_path, tmp_path):
        """An unsupported output suffix produces a clear error (SystemExit)."""
        out_path = tmp_path / "annot.bogus"
        with pytest.raises(SystemExit) as exc:
            run([
                "arg",
                "--trees", str(tiny_trees_path),
                "--mu", "1e-8",
                "--out", str(out_path),
            ])
        # SystemExit may carry a message (str) or numeric code. Either is fine.
        assert exc.value.code != 0

    def test_fixed_tree_rejects_trees_output(self, tmp_path):
        """``fixed-tree`` must refuse ``.trees`` output (only VCF-family valid)."""
        out_path = tmp_path / "annot.trees"
        # No valid VCF or Newick is needed, since the suffix check fires first.
        with pytest.raises(SystemExit):
            run([
                "fixed-tree",
                "--vcf", str(tmp_path / "missing.vcf"),
                "--species-tree", str(tmp_path / "missing.nwk"),
                "--ingroup", "a,b",
                "--outgroups", "o1",
                "--out", str(out_path),
            ])

    def test_fixed_tree_vcf_in_vcf_out(self, tmp_path):
        """End-to-end ``fixed-tree`` with a plain VCF input (no ``--template-vcf``):
        confirms the default-to-self template path runs ``.fit()`` and writes
        an annotated VCF with ``AA`` INFO fields. Covers the ML-fit chaining
        that the VCZ test, which routes through an explicit
        ``--template-vcf``, does not exercise.

        The CLI falls back to
        :meth:`~ancestree.sites.BaseComposition.no_counts` when no explicit
        ``base_composition`` is supplied, and ``--n-target-sites`` then
        derives the monomorphic-site calibration."""
        import numpy as np
        import cyvcf2

        ingroup = ["i0", "i1"]
        outgroup = ["o1", "o2"]
        sample_ids = ingroup + outgroup
        n_samples = len(sample_ids)

        rng = np.random.default_rng(13)
        n_sites = 300
        alleles = [["A", "G"]] * n_sites
        gt = rng.integers(0, 2, size=(n_sites, n_samples, 1), dtype=np.int8)

        vcf_in = tmp_path / "input.vcf"
        write_skeleton_vcf(vcf_in, sample_ids=sample_ids,
                            alleles_per_site=alleles, genotypes=gt)
        nwk = tmp_path / "species.nwk"
        nwk.write_text("(((i0:0.05,i1:0.05):0.05,o1:0.10):0.10,o2:0.20);")

        out_path = tmp_path / "annot.vcf"
        code = run([
            "fixed-tree",
            "--vcf", str(vcf_in),
            "--species-tree", str(nwk),
            "--ingroup", ",".join(ingroup),
            "--outgroups", ",".join(outgroup),
            "--prior", "uniform",
            "--n-target-sites", str(n_sites),
            "--out", str(out_path),
        ])
        assert code == 0
        assert out_path.exists()

        rdr = cyvcf2.VCF(str(out_path))
        try:
            n_records, n_with_aa = 0, 0
            n_blank = 0
            n_differs_from_ref = 0
            for record in rdr:
                n_records += 1
                aa = record.INFO.get("AA")
                if aa is not None:
                    n_with_aa += 1
                    assert aa in {"A", "C", "G", "T", "."}
                    if aa == ".":
                        n_blank += 1
                    elif aa != record.REF:
                        n_differs_from_ref += 1
            assert n_records > 0
            assert n_with_aa > 0
            # A constant call satisfies the alphabet check above but is not
            # inference: REF everywhere, or the uncalled sentinel everywhere.
            assert n_blank == 0, f"{n_blank} sites came back uncalled"
            assert n_differs_from_ref > 0, "every call equals REF"
        finally:
            rdr.close()


@pytest.mark.filterwarnings("ignore::Warning:zarr")
def test_arg_writes_a_vcz_store(tiny_ts, tiny_trees_path, tmp_path, caplog):
    """``arg --out x.vcz`` builds a template from the tree sequence and
    annotates every site."""
    import zarr

    out = tmp_path / "annot.vcz"
    with caplog.at_level(logging.INFO, logger="ancestree.cli"):
        code = run(["arg", "--trees", str(tiny_trees_path), "--mu", "1e-7",
                    "--out", str(out)])
    assert code == 0
    messages = [r.getMessage() for r in caplog.records]
    assert any(m == f"arg: wrote {tiny_ts.num_sites} annotated records to {out}"
               for m in messages)
    root = zarr.open(str(out), mode="r")
    aa = [str(a) for a in root["variant_AA"][:]]
    assert len(aa) == tiny_ts.num_sites
    assert set(aa) <= {"A", "C", "G", "T"}
    assert root.attrs[PROVENANCE_HEADER]["mode"] == "arg"


# ---------------------------------------------------------------------------- VCZ input


def _write_minimal_vcz(path, *, sample_ids, alleles_per_site, genotypes):
    """Write a minimal VCZ-format zarr store matching ``bio2zarr``'s layout.

    Mirrors the helper in ``test_vcf_zarr_source.py``, kept inline here to
    avoid cross-test-file imports and to keep the CLI VCZ smoke test
    self-contained. Tolerates both zarr v2 (``create_dataset``) and zarr v3
    (``create_array``) so the test is not pinned to one major version.

    :param path: Output store path (a directory).
    :param sample_ids: Per-haplotype sample ids (``ploidy=1`` here, so one id
        per chromosome copy).
    :param alleles_per_site: ``list[list[str]]`` of REF/ALT per site.
    :param genotypes: ``(n_sites, n_samples, 1)`` int8 allele indices.
    """
    import numpy as np
    import zarr

    n_variants = genotypes.shape[0]
    max_alleles = max(len(a) for a in alleles_per_site)

    allele_arr = np.full((n_variants, max_alleles), "", dtype="U1")
    for i, row in enumerate(alleles_per_site):
        for j, a in enumerate(row):
            allele_arr[i, j] = a

    root = zarr.open(str(path), mode="w")
    _mk = getattr(root, "create_array", None) or root.create_dataset

    def _put(name, data, dtype):
        _mk(name, shape=data.shape, dtype=dtype)
        root[name][:] = data

    _put("variant_position", np.arange(1, n_variants + 1, dtype=np.int64), "int64")
    _put("variant_contig", np.zeros(n_variants, dtype=np.int32), "int32")
    _put("variant_allele", allele_arr, "U1")
    _put("call_genotype", genotypes.astype(np.int8), "int8")
    sid_dtype = f"U{max(len(s) for s in sample_ids)}"
    _put("sample_id", np.asarray(sample_ids, dtype="U"), sid_dtype)
    _put("contig_id", np.asarray(["chr1"], dtype="U"), "U4")


class TestFixedTreeVczInput:
    """Confirm ``fixed-tree --vcf`` transparently accepts ``.vcz`` stores
    (the inference layer dispatches by extension. The CLI just passes the
    path through)."""

    @pytest.mark.filterwarnings(
        # zarr v3 emits UnstableSpecificationWarning for the U-dtypes the
        # bio2zarr layout uses. Not a real failure mode for the CLI.
        "ignore::Warning:zarr",
    )
    def test_fixed_tree_reads_vcz(self, tmp_path):
        """End-to-end: hand-craft a small ``.vcz`` store + Newick species
        tree + sibling VCF template, run ``ancestree fixed-tree --vcf
        store.vcz --template-vcf skel.vcf``, verify the annotated VCF
        lands on disk with ``AA`` INFO fields populated."""
        import numpy as np
        import cyvcf2

        ingroup = ["i0", "i1"]
        outgroup = ["o1", "o2"]
        sample_ids = ingroup + outgroup
        n_samples = len(sample_ids)

        rng = np.random.default_rng(7)
        n_sites = 300
        alleles = [["A", "G"]] * n_sites
        gt = rng.integers(0, 2, size=(n_sites, n_samples, 1), dtype=np.int8)

        store = tmp_path / "input.vcz"
        _write_minimal_vcz(store, sample_ids=sample_ids,
                           alleles_per_site=alleles, genotypes=gt)
        skel = tmp_path / "input.vcf"
        write_skeleton_vcf(skel, sample_ids=sample_ids,
                            alleles_per_site=alleles, genotypes=gt)

        nwk = tmp_path / "species.nwk"
        nwk.write_text("(((i0:0.05,i1:0.05):0.05,o1:0.10):0.10,o2:0.20);")

        out_path = tmp_path / "annot.vcf"
        code = run([
            "fixed-tree",
            "--vcf", str(store),
            "--template-vcf", str(skel),
            "--species-tree", str(nwk),
            "--ingroup", ",".join(ingroup),
            "--outgroups", ",".join(outgroup),
            "--prior", "uniform",
            "--n-target-sites", str(n_sites),
            "--out", str(out_path),
        ])
        assert code == 0, "CLI returned non-zero on .vcz input"
        assert out_path.exists()

        rdr = cyvcf2.VCF(str(out_path))
        try:
            n_records = 0
            n_with_aa = 0
            n_blank = 0
            n_differs_from_ref = 0
            for record in rdr:
                n_records += 1
                aa = record.INFO.get("AA")
                if aa is not None:
                    n_with_aa += 1
                    assert aa in {"A", "C", "G", "T", "."}
                    if aa == ".":
                        n_blank += 1
                    elif aa != record.REF:
                        n_differs_from_ref += 1
            assert n_records > 0
            assert n_with_aa > 0
            # A constant call satisfies the alphabet check above but is not
            # inference: REF everywhere, or the uncalled sentinel everywhere.
            assert n_blank == 0, f"{n_blank} sites came back uncalled"
            assert n_differs_from_ref > 0, "every call equals REF"
        finally:
            rdr.close()


@pytest.mark.filterwarnings("ignore::Warning:zarr")
class TestFixedTreeHandler:
    """The ``fixed-tree`` branches beyond the species-tree VCF-to-VCF run."""

    def test_adaptive_weight_with_species_tree_falls_back_to_kingman(
            self, ladder_panel, tmp_path, caplog):
        vcf, _vcz, nwk, _alleles, _gt = ladder_panel
        out = tmp_path / "annot.vcf"
        with caplog.at_level(logging.WARNING, logger="ancestree.cli"):
            code = run([
                "fixed-tree", "--vcf", str(vcf), "--species-tree", str(nwk),
                "--ingroup", "i0,i1", "--outgroups", "o1,o2",
                "--ingroup-weight", "adaptive", "--out", str(out),
            ])
        assert code == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("Kingman values are used instead" in m for m in messages)
        prov = _vcf_provenance(out)
        assert prov["parameters"]["branch_rates_fitted"] is False

    def test_ingroup_is_derived_from_the_panel(
            self, ladder_panel, tmp_path, caplog):
        vcf, _vcz, nwk, _alleles, _gt = ladder_panel
        out = tmp_path / "annot.vcf"
        with caplog.at_level(logging.INFO, logger="ancestree.cli"):
            code = run([
                "fixed-tree", "--vcf", str(vcf), "--species-tree", str(nwk),
                "--outgroups", "o1,o2", "--out", str(out),
            ])
        assert code == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("--ingroup not given" in m and "i0,i1" in m for m in messages)
        assert _vcf_provenance(out)["parameters"]["ingroup_samples"] == ["i0", "i1"]

    def test_outgroups_alone_build_the_ladder_and_fit(
            self, ladder_panel, tmp_path, caplog):
        """Without ``--species-tree`` the ladder is fitted, and a
        base-frequency model without ``--empirical-composition`` reports its
        uniform frequencies."""
        vcf, _vcz, _nwk, alleles, _gt = ladder_panel
        out = tmp_path / "annot.vcf"
        with caplog.at_level(logging.WARNING, logger="ancestree.cli"):
            code = run([
                "fixed-tree", "--vcf", str(vcf),
                "--ingroup", "i0,i1", "--outgroups", "o1,o2",
                "--model", "F81", "--prior", "uniform",
                "--n-target-sites", str(len(alleles)), "--n-starts", "1",
                "--seed", "3", "--out", str(out),
            ])
        assert code == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("Model F81 runs with uniform base frequencies" in m
                   for m in messages)
        params = _vcf_provenance(out)["parameters"]
        assert params["branch_rates_fitted"] is True
        assert params["outgroup_samples"] == ["o1", "o2"]
        assert set(params["fitted"]) >= {"K1", "K2"}
        assert all(np.isfinite(v) for v in params["fitted"].values())
        calls = _aa_calls(out)
        assert len(calls) == len(alleles)
        assert set(calls) <= {"A", "C", "G", "T"}

    @pytest.mark.parametrize("model, fit_kappa", [
        ("K2", False), ("K2", True), ("JC69", False),
    ])
    def test_empirical_composition_feeds_the_model(
            self, ladder_panel, tmp_path, caplog, monkeypatch, model, fit_kappa):
        """``--empirical-composition`` reports the tally, hands the composition
        to the inference, and seeds kappa for a two-parameter model whose
        kappa is not fitted."""
        import inspect

        from ancestree import inference as inference_module

        built = []

        class Recording(inference_module.FixedTreeInference):
            """The handler's inference, with its constructor arguments kept."""

            __signature__ = inspect.signature(inference_module.FixedTreeInference)

            def __init__(self, *args, **kwargs):
                built.append(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(inference_module, "FixedTreeInference", Recording)
        vcf, _vcz, nwk, alleles, gt = ladder_panel
        out = tmp_path / "annot.vcf"
        argv = [
            "fixed-tree", "--vcf", str(vcf), "--species-tree", str(nwk),
            "--ingroup", "i0,i1", "--outgroups", "o1,o2",
            "--model", model, "--empirical-composition",
            "--max-calibration-sites", "200", "--out", str(out),
        ]
        if fit_kappa:
            argv.append("--fit-kappa")
        with caplog.at_level(logging.INFO, logger="ancestree.cli"):
            code = run(argv)
        assert code == 0
        messages = [r.getMessage() for r in caplog.records]
        reported = [m for m in messages if m.startswith("Empirical base composition")]
        assert len(reported) == 1
        expected = _empirical_composition(str(vcf), 200)
        assert f"kappa = {expected.kappa_estimate:.3f}" in reported[0]
        (kwargs,) = built
        np.testing.assert_allclose(kwargs["base_composition"].pi, expected.pi)
        fitted_model = kwargs["model"]
        assert type(fitted_model).__name__ == model
        if model == "K2":
            seeded = not fit_kappa
            assert fitted_model.fit_kappa is fit_kappa
            assert fitted_model.kappa == pytest.approx(
                expected.kappa_estimate if seeded else _lib_default(type(fitted_model), "kappa"))
        assert len(_aa_calls(out)) == len(alleles)

    def test_vcz_output_ignores_the_template_vcf(
            self, ladder_panel, tmp_path, caplog):
        """A ``.vcz`` destination is built from the input, and a supplied
        ``--template-vcf`` is reported as ignored."""
        import zarr

        vcf, _vcz, nwk, alleles, _gt = ladder_panel
        out = tmp_path / "annot.vcz"
        with caplog.at_level(logging.INFO, logger="ancestree.cli"):
            code = run([
                "fixed-tree", "--vcf", str(vcf), "--species-tree", str(nwk),
                "--ingroup", "i0,i1", "--outgroups", "o1,o2",
                "--template-vcf", str(vcf), "--out", str(out),
            ])
        assert code == 0
        messages = [r.getMessage() for r in caplog.records]
        assert any("--template-vcf is ignored for a .vcz --out" in m
                   for m in messages)
        assert any(m == f"Wrote {len(alleles)} annotated variants to {out}"
                   for m in messages)
        root = zarr.open(str(out), mode="r")
        aa = [str(a) for a in root["variant_AA"][:]]
        assert len(aa) == len(alleles)
        assert set(aa) <= {"A", "C", "G", "T"}
        assert root.attrs[PROVENANCE_HEADER]["parameters"]["ingroup_samples"] == ["i0", "i1"]


# ---------------------------------------------------------------------------- local-tree


class TestLocalTreeE2E:
    """End-to-end ``local-tree``: build local trees from a VCF and annotate."""

    @pytest.fixture(scope="class")
    @classmethod
    def panel_vcf(cls, tmp_path_factory):
        """An 8-haplotype msprime panel dumped to a plain VCF."""
        ts = msprime.sim_ancestry(
            samples=8, ploidy=1, sequence_length=2e5,
            recombination_rate=1e-8, population_size=1e4, random_seed=3,
        )
        ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
        assert not (ts.num_sites == 0), "simulation produced zero sites; bump --rate"
        d = tmp_path_factory.mktemp("cli_localtree")
        vcf = d / "panel.vcf"
        with open(vcf, "w") as f:
            ts.write_vcf(f, contig_id="1")
        return vcf, float(ts.sequence_length)

    def test_local_tree_annotates_a_non_default_contig(self, tmp_path):
        """A source contig other than the ``--chrom`` default must still annotate.

        The output VCF is templated from the input, so its contig names carry
        over. Templating from an auto-written file labelled ``--chrom`` instead
        would match no posterior and write every record unannotated.
        """
        import cyvcf2
        ts = msprime.sim_ancestry(
            samples=6, ploidy=1, sequence_length=5e4,
            recombination_rate=1e-8, population_size=1e4, random_seed=7,
        )
        ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=7)
        assert not (ts.num_sites == 0), "simulation produced zero sites"
        src = tmp_path / "chr7.vcf"
        with open(src, "w") as f:
            ts.write_vcf(f, contig_id="chr7")  # deliberately not the --chrom default

        out = tmp_path / "annotated.vcf"
        assert run([
            "local-tree", "--vcf", str(src),
            "--mu", "1e-7", "--rec-rate", "1e-8",
            "--sequence-length", str(int(ts.sequence_length)),
            "--out", str(out),
        ]) == 0
        annotated = [r for r in cyvcf2.VCF(str(out)) if r.INFO.get("AA") is not None]
        assert annotated, "no records were annotated"
        assert annotated[0].CHROM == "chr7"

    def test_local_tree_vcf_in_vcf_out(self, panel_vcf, tmp_path):
        """VCF in → annotated VCF out with ``AA`` INFO fields, no outgroups."""
        import cyvcf2
        vcf, seq_len = panel_vcf
        out_path = tmp_path / "annot.vcf"
        code = run([
            "local-tree",
            "--vcf", str(vcf),
            "--sequence-length", str(seq_len),
            "--mu", "1e-8", "--rec-rate", "1e-8",
            "--window", "20snp", "--block-size", "500",
            "--out", str(out_path),
        ])
        assert code == 0
        assert out_path.exists()
        rdr = cyvcf2.VCF(str(out_path))
        try:
            n_records, n_with_aa = 0, 0
            n_blank = 0
            n_differs_from_ref = 0
            for record in rdr:
                n_records += 1
                aa = record.INFO.get("AA")
                if aa is not None:
                    n_with_aa += 1
                    assert aa in {"A", "C", "G", "T", "."}
                    if aa == ".":
                        n_blank += 1
                    elif aa != record.REF:
                        n_differs_from_ref += 1
            assert n_records > 0
            assert n_with_aa > 0
            # A constant call satisfies the alphabet check above but is not
            # inference: REF everywhere, or the uncalled sentinel everywhere.
            assert n_blank == 0, f"{n_blank} sites came back uncalled"
            assert n_differs_from_ref > 0, "every call equals REF"
        finally:
            rdr.close()

    def test_local_tree_to_trees(self, panel_vcf, tmp_path):
        """``.trees`` output writes an annotated tree sequence."""
        import tskit
        vcf, seq_len = panel_vcf
        out_path = tmp_path / "annot.trees"
        code = run([
            "local-tree",
            "--vcf", str(vcf),
            "--sequence-length", str(seq_len),
            "--window", "20snp", "--block-size", "500",
            "--out", str(out_path),
        ])
        assert code == 0
        ts = tskit.load(str(out_path))
        assert ts.num_sites > 0
        assert any(s.ancestral_state in {"A", "C", "G", "T"} for s in ts.sites())

    def test_local_tree_chunked_to_trees(self, panel_vcf, tmp_path):
        """``--chunk-size`` with ``.trees`` output stitches the per-segment
        builds into one genome-wide annotated tree sequence. ``--out-trees``
        also writes the un-annotated reconstruction."""
        import tskit
        vcf, _seq_len = panel_vcf
        out_path = tmp_path / "annot.trees"
        recon = tmp_path / "recon.trees"
        code = run([
            "local-tree",
            "--vcf", str(vcf),
            "--chunk-size", "50000",
            "--window", "20snp", "--block-size", "500",
            "--out", str(out_path),
            "--out-trees", str(recon),
        ])
        assert code == 0
        ts = tskit.load(str(out_path))
        assert ts.num_sites > 0
        assert any(s.ancestral_state in {"A", "C", "G", "T"} for s in ts.sites())
        recon_ts = tskit.load(str(recon))  # the side-artifact reconstruction
        assert recon_ts.num_samples == ts.num_samples

    def test_vcz_without_template_errors(self, tmp_path):
        """``.vcz`` input without ``--template-vcf`` should fail clearly
        (no silent fallback to cyvcf2 trying to read the zarr store)."""
        store = tmp_path / "input.vcz"
        store.mkdir()  # path exists but no template, CLI must bail at the check
        nwk = tmp_path / "species.nwk"
        nwk.write_text("((i0:0.1,i1:0.1):0.1,o1:0.2);")
        with pytest.raises(SystemExit) as exc:
            run([
                "fixed-tree",
                "--vcf", str(store),
                "--species-tree", str(nwk),
                "--ingroup", "i0,i1", "--outgroups", "o1",
                "--n-target-sites", "100",
                "--out", str(tmp_path / "annot.vcf"),
            ])
        msg = str(exc.value)
        assert "template-vcf" in msg or "VCZ" in msg or ".vcz" in msg


def test_recombination_map_reaches_the_inference(tmp_path):
    """A HapMap file drives the HMM and is recorded in the provenance."""
    ts = msprime.sim_ancestry(
        samples=6, ploidy=1, sequence_length=1e5, recombination_rate=1e-8,
        population_size=1e4, random_seed=3,
    )
    ts = msprime.sim_mutations(ts, rate=1e-7, random_seed=3)
    assert ts.num_sites > 0
    vcf = tmp_path / "panel.vcf"
    with open(vcf, "w") as f:
        ts.write_vcf(f, contig_id="1")
    hapmap = tmp_path / "map.txt"
    hapmap.write_text(
        "Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n"
        "chr1\t0\t1.0\t0.0\n"
        "chr1\t50000\t2.0\t0.05\n"
        "chr1\t100000\t0.0\t0.15\n"
    )
    out = tmp_path / "annot.vcf"
    code = run([
        "local-tree", "--vcf", str(vcf), "--sequence-length", "100000",
        "--mu", "1e-7", "--recombination-map", str(hapmap),
        "--window", "10snp", "--block-size", "1000", "--out", str(out),
    ])
    assert code == 0
    params = _vcf_provenance(out)["parameters"]
    assert params["recombination_map_path"] == str(hapmap)
    calls = _aa_calls(out)
    assert len(calls) == ts.num_sites
    assert set(calls) <= {"A", "C", "G", "T"}


class TestArgLocalTreeRejectKingmanPrior:
    """Kingman and adaptive priors are fixed-tree only.

    The arg and local-tree subparsers must not advertise ``--prior kingman``,
    and argparse rejects it as an invalid choice.
    """

    def test_arg_prior_kingman_rejected_by_argparse(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "arg", "--trees", "x.trees", "--ingroup-weight", "kingman",
                "--out", "o.vcf",
            ])

    def test_local_tree_prior_kingman_rejected_by_argparse(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "local-tree", "--vcf", "x.vcf", "--rec-rate", "1e-8",
                "--ingroup-weight", "kingman", "--out", "o.vcf",
            ])

    def test_arg_stationary_still_accepted(self):
        parser = build_parser()
        ns = parser.parse_args([
            "arg", "--trees", "x.trees", "--prior", "uniform", "--out", "o.vcf",
        ])
        assert ns.prior == "uniform"


class TestNoPosteriorFlag:
    """``--no-posterior`` drops the per-state posterior from any output."""

    @pytest.mark.parametrize("subcommand", ["arg", "fixed-tree", "local-tree"])
    def test_flag_is_offered(self, subcommand, capsys):
        """Every mode that writes output exposes the flag."""
        from ancestree.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([subcommand, "--help"])
        assert "--no-posterior" in capsys.readouterr().out

    def test_flag_suppresses_the_field(self, tmp_path):
        """Running with the flag leaves ``AA_post`` out of the VCF."""
        import gzip

        from ancestree.cli import run

        trees = DEMO_TREES
        for flag, expected in (([], True), (["--no-posterior"], False)):
            out = tmp_path / f"o{expected}.vcf.gz"
            assert run(["arg", "--trees", trees, "--mu", "1.25e-8",
                        "--out", str(out), *flag]) == 0
            declared = any(
                line.startswith("##INFO=<ID=AA_post")
                for line in gzip.open(out, "rt")
            )
            assert declared is expected


class TestDerivedIngroup:
    """``--ingroup`` defaults to the non-outgroup individuals' haplotypes."""

    @staticmethod
    def _demo() -> str:
        return DEMO_VCF

    def test_excludes_every_haplotype_of_a_named_outgroup_individual(self):
        from ancestree.cli import _derive_ingroup

        derived = _derive_ingroup(self._demo(), ["o1_h0", "o2_h0", "o3_h0"])
        # o1_h1 / o2_h1 / o3_h1 must not leak in: naming one haplotype of an
        # outgroup individual withholds the individual.
        assert derived == [
            "i0_h0", "i0_h1", "i1_h0", "i1_h1",
            "i2_h0", "i2_h1", "i3_h0", "i3_h1",
        ]

    def test_ids_without_a_haplotype_suffix_pass_through(self):
        from ancestree.cli import _individual_of

        assert _individual_of("o1_h0") == "o1"
        assert _individual_of("plain") == "plain"
        assert _individual_of("s_h12") == "s"

    def test_all_samples_excluded_is_an_error(self):
        from ancestree.cli import _derive_ingroup

        every = [f"{name}_h0" for name in ("i0", "i1", "i2", "i3", "o1", "o2", "o3")]
        with pytest.raises(SystemExit, match="no ingroup"):
            _derive_ingroup(self._demo(), every)

    def test_explicit_ingroup_still_parses(self):
        from ancestree.cli import build_parser

        args = build_parser().parse_args([
            "fixed-tree", "--vcf", "x.vcf.gz", "--ingroup", "a,b",
            "--outgroups", "o1", "--out", "y.vcf.gz",
        ])
        assert args.ingroup == ["a", "b"]

    def test_ingroup_is_optional(self):
        from ancestree.cli import build_parser

        args = build_parser().parse_args([
            "fixed-tree", "--vcf", "x.vcf.gz",
            "--outgroups", "o1", "--out", "y.vcf.gz",
        ])
        assert not args.ingroup


class TestDerivedIngroupFollowsThePanel:
    """``--samples`` restricts the panel, so it must restrict the ingroup.

    Without ``--ingroup`` the ingroup is derived from the VCF's own sample
    list. Reading that list unfiltered names haplotypes the run never loads.
    """

    @staticmethod
    def _vcf(tmp_path):
        import tskit

        ts = tskit.load(QUICKSTART_TREES)
        names = [f"i{i}" for i in range(6)] + ["o0", "o1"]
        p = tmp_path / "panel.vcf"
        with open(p, "w") as fh:
            ts.write_vcf(fh, individual_names=names,
                         position_transform="legacy")
        return str(p)

    def test_the_ingroup_is_confined_to_the_panel(self, tmp_path):
        from ancestree.cli import _derive_ingroup

        panel = ["i0", "i1", "o0", "o1"]
        ingroup = _derive_ingroup(self._vcf(tmp_path), ["o0", "o1"],
                                  sample_filter=panel)
        assert ingroup, "no ingroup derived"
        outside = [h for h in ingroup if h.split("_h")[0] not in panel]
        assert not outside, f"ingroup names haplotypes outside the panel: {outside}"

    def test_an_unfiltered_run_keeps_every_non_outgroup(self, tmp_path):
        from ancestree.cli import _derive_ingroup

        ingroup = _derive_ingroup(self._vcf(tmp_path), ["o0", "o1"])
        assert len(ingroup) == 6


class TestProvenanceNamesTheCompositionItUsed:
    """``n_target_sites`` yields a composition uniform over the states.

    Recording it as empirical claims the run measured a base composition from
    sequence when it measured none.
    """

    @staticmethod
    def _source():
        import tskit

        import ancestree as anc
        return list(anc.TskitSource(tskit.load(QUICKSTART_TREES)))

    def _kind(self, **kwargs):
        import ancestree as anc
        from ancestree.models import JC69

        inf = anc.FixedTreeInference(
            self._source(), JC69(),
            ingroup_samples=[f"i{i}" for i in range(6)],
            outgroup_samples=["o0", "o1"], **kwargs)
        inf.fit()
        return inf._provenance_parameters()["base_composition"]

    def test_a_target_site_count_is_not_an_empirical_composition(self):
        assert self._kind(n_target_sites=100_000) == "uniform"

    def test_a_measured_composition_is_reported_as_empirical(self):
        from ancestree.sites import BaseComposition

        bc = BaseComposition.from_counts(A=400, C=100, G=300, T=200)
        assert self._kind(base_composition=bc) == "empirical"


def _args(**kw):
    base = dict(out="out.vcf", vcf="in.vcf", chunk_size=None,
                sequence_length=1000.0, out_trees=False)
    base.update(kw)
    return argparse.Namespace(**base)


class TestLocalTreeDispatchGuards:
    def test_bad_output_extension(self):
        with pytest.raises(SystemExit, match="must end with"):
            _run_local_tree(_args(out="result.txt"))


TREES = QUICKSTART_TREES


ING = [f"i{i}" for i in range(6)]


OUT = ["o0", "o1"]


def _fixed(weight):
    ts = tskit.load(TREES)
    inf = anc.Inference.from_fixed_tree(
        ts, ingroup_samples=ING, outgroup_samples=OUT, model=anc.JC69(),
        n_target_sites=50_000, ingroup_weight=weight, progress=False)
    inf.fit()
    return np.array([p.values for _s, p in inf.infer()])


def test_no_ingroup_weight_is_not_the_default():
    """``--ingroup-weight none`` must differ from the Kingman default."""
    default = _fixed(None)
    none = _fixed(NoIngroupWeight())
    diff = float(np.max(np.abs(default - none)))
    assert diff > 0.1, (
        f"asking for no ingroup weight changed nothing (max |diff| {diff:.3g})")


@pytest.mark.parametrize("size", [1, 7, 10, 12, 16, 100])
def test_any_ensemble_size_is_accepted(size):
    """Any ensemble size is accepted, not only 1..8 and the multiples of 8."""
    ts = tskit.load(TREES)
    inf = anc.Inference.from_local_tree(
        list(anc.TskitSource(ts)), model=anc.JC69(), mu=5e-8, rec_rate=1e-8,
        sample_names=ING + OUT, sequence_length=float(ts.sequence_length),
        n_ensemble=size, progress=False)
    assert len(list(inf.infer())) > 0


def test_outgroups_are_excluded_per_individual():
    """A VCF source splits a diploid, so naming o0 must withhold both haps."""
    ts = tskit.load(TREES)
    inf = anc.Inference.from_arg(ts, model=anc.JC69(), mu=5e-8,
                                outgroup_samples=["o0"], progress=False)
    panel = ["i0_h0", "i0_h1", "o0_h0", "o0_h1"]
    got = inf._ingroup_from_panel(panel)
    assert "o0_h0" not in got and "o0_h1" not in got, (
        f"outgroup haplotypes leaked into the ingroup: {got}")
    assert set(got) == {"i0_h0", "i0_h1"}


def test_the_haplotype_suffix_stripper_is_conservative():
    assert _individual_of("i0_h1") == "i0"
    assert _individual_of("o0") == "o0"
    assert _individual_of("s_h_x") == "s_h_x"


@pytest.mark.parametrize("func_name, mode", [
    ("_run_arg", "arg"),
    ("_run_local_tree", "local-tree"),
])
def test_each_mode_reports_its_own_model_defaults(func_name, mode):
    """Each mode's handler warns about model defaults once, under its own name."""
    import ast
    import inspect
    import textwrap

    from ancestree import cli

    src = textwrap.dedent(inspect.getsource(getattr(cli, func_name)))
    modes = [
        node.args[0].value
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_warn_model_defaults"
        and node.args and isinstance(node.args[0], ast.Constant)
    ]
    assert modes == [mode]


@pytest.mark.parametrize("pre, post", [
    (["-v"], ["-q"]),
    (["-q"], ["-v"]),
    ([], ["-v", "-q"]),
])
def test_verbose_and_quiet_are_exclusive_across_the_subcommand(pre, post):
    """``-v`` and ``-q`` are exclusive even when split across the subcommand.

    The verbosity flags live on a parent parser shared by the top parser and
    every subparser, so the exclusivity must hold across parse passes.
    """
    from ancestree import cli

    argv = pre + ["fixed-tree"] + post + [
        "--vcf", "x.vcf", "--outgroups", "o", "--out", "y.vcf"]
    with pytest.raises(SystemExit) as exc:
        cli.run(argv)
    assert exc.value.code == 2


def test_no_progress_disables_the_progress_bar(tiny_trees_path, tmp_path, monkeypatch):
    """``--no-progress`` flips the package-wide progress-bar setting."""
    monkeypatch.setattr(Settings, "disable_pbar", False)
    out = tmp_path / "annot.trees"
    code = run(["--no-progress", "arg", "--trees", str(tiny_trees_path),
                "--mu", "1e-7", "--out", str(out)])
    assert code == 0
    assert Settings.disable_pbar is True
    assert out.exists()
