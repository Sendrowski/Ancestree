"""Command-line interface for :mod:`ancestree`.

Exposes three subcommands matching the three inference modes:

- ``ancestree fixed-tree``: :class:`~ancestree.inference.FixedTreeInference`
  on a VCF / VCZ + species-tree Newick. ML-fits branch rates on the
  outgroup-ladder topology and emits an annotated VCF.
- ``ancestree arg``: :class:`~ancestree.inference.ARGBasedInference`
  on a tskit ``.trees`` file. No outgroups required. The local ARG
  trees carry the genealogical signal. Emits either an annotated VCF
  or an annotated ``.trees`` file.
- ``ancestree local-tree``:
  :class:`~ancestree.local_tree_inference.LocalTreeInference` on a VCF / VCZ
  of genotypes alone. Infers a dated local tree per window (PSMC'-style
  pairwise-coalescent HMM → per-window UPGMA), then infers the ancestral
  allele on it with the ARG kernel. No outgroups and no input ARG required.
  Emits an annotated VCF, VCF-Zarr, or ``.trees`` file.

The output format is inferred from the ``--out`` extension. An output
annotates the input when the input has its format, and is otherwise written
from the sites with the samples the inference used:

- ``.vcf``, ``.vcf.gz``, ``.vcf.bgz``, ``.bcf`` route to
  :meth:`Inference.to_vcf() <ancestree.inference.Inference.to_vcf>`;
- ``.vcz`` routes to :meth:`Inference.to_zarr() <ancestree.inference.Inference.to_zarr>`;
- ``.trees`` (valid under ``arg`` and ``local-tree``) routes to the
  matching ``to_arg`` writer.

All three writers embed an ``ancestree`` provenance record (version, mode,
parameters): the VCF header, the tskit provenance table, or the VCZ root
``attrs`` respectively, so CLI and Python-API outputs are self-describing.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import sys
from typing import Sequence, cast

from ancestree import DEFAULT_MU, __version__
from ancestree.settings import Settings
from ancestree.focal import FocalNode
from ancestree.sites import _individual_of, _path_format

__all__ = ["main", "build_parser", "run"]

#: Logger the CLI's own diagnostics are emitted on.
_log = logging.getLogger("ancestree.cli")


# ---------------------------------------------------------------------------- helpers


def _lib_default(cls, param: str):
    """The constructor default a flag configures.

    A flag reads its default from the class it builds, so the two cannot state
    different values.

    :param cls: The class whose constructor the flag feeds.
    :param param: Constructor parameter name.
    :return: That parameter's default.
    :raises KeyError: If the constructor takes no such parameter.
    """
    parameters = inspect.signature(cls).parameters
    if param not in parameters:
        raise KeyError(f"{cls.__name__}() takes no parameter {param!r}")
    return parameters[param].default


def _chunk_size(value: str):
    """Parse ``--chunk-size``, with ``none`` opting out of chunking.

    The constructor takes ``None`` to mean "build the whole input as one
    region", which a command line has no other way to express once the flag
    carries a width by default. The width itself is validated by the
    constructor's own parser, whose message argparse reports only when it
    arrives as an :class:`argparse.ArgumentTypeError`.

    :param value: Raw flag value.
    :return: ``None`` for ``none`` / ``off``, else the value for the
        constructor to resolve.
    :raises argparse.ArgumentTypeError: On anything that is not a width.
    """
    if value.strip().lower() in ("none", "off"):
        return None
    from ancestree.local_tree_inference import _parse_bp

    try:
        _parse_bp(value, name="--chunk-size")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{exc}. Pass 'none' to build the whole input as one region."
        ) from exc
    return value


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated string into a list of non-empty tokens.

    Used by ``--ingroup`` / ``--outgroups`` so the user can pass either
    ``pop1,pop2`` (no spaces) or quote a list with spaces.

    :param value: Raw flag value from the command line.
    :return: List of trimmed, non-empty tokens.
    """
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _warn_ignored(args, pairs) -> None:
    """Report flags that this combination discards.

    Whether a flag was given is read off the parser's own record of which
    destinations appeared on the command line, so a flag left at a non-empty
    default is not reported and one passed at its default value still is.

    :param args: Parsed arguments.
    :param pairs: ``(flag_attr, condition, reason)`` triples. The flag is
        reported when it was given and the condition holds.
    """
    supplied: "frozenset[str]" = getattr(args, "_supplied", frozenset())
    for attr, ignored, reason in pairs:
        if attr in supplied and ignored:
            _log.warning("--%s is ignored %s",
                         attr.replace("_", "-"), reason)


def _derive_ingroup(vcf_path: str, outgroups: Sequence[str],
                    sample_filter: "Sequence[str] | None" = None) -> list[str]:
    """Every haplotype whose individual is not named in ``outgroups``.

    Exclusion is per individual: naming ``o1_h0`` as an outgroup also
    withholds ``o1_h1``, keeping outgroup material out of the ingroup.

    :param vcf_path: VCF / BCF / VCZ path whose sample list defines the panel.
    :param outgroups: Outgroup ids as given on the command line.
    :param sample_filter: The panel the run is restricted to, so the ingroup
        names only haplotypes the inference will read.
    :return: Ingroup haplotype ids, in the panel's own order.
    :raises SystemExit: If no ingroup haplotype remains.
    """
    from ancestree.sources import CyVCF2Source, VcfZarrSource

    source_cls = VcfZarrSource if _path_format(vcf_path) == "vcz" else CyVCF2Source
    excluded = {_individual_of(o) for o in outgroups}
    kept = ({_individual_of(s) for s in sample_filter}
            if sample_filter else None)
    ingroup = [
        s for s in source_cls(vcf_path).samples()
        if _individual_of(s) not in excluded
        and (kept is None or _individual_of(s) in kept)
    ]
    if not ingroup:
        raise SystemExit(
            f"fixed-tree: --outgroups accounts for every sample in "
            f"{vcf_path!r}, leaving no ingroup. Pass --ingroup explicitly."
        )
    return ingroup


def _build_model(name: str, *, fit_kappa: bool, fit_rates: bool, kappa: float | None = None):
    """Instantiate a :class:`~ancestree.models.SubstitutionModel` by short name.

    ``--fit-kappa`` is honoured for the two κ-aware models (``K2``, ``HKY``)
    and ``--fit-rates`` for ``GTR``. Both are silently ignored on models that
    do not expose the matching free parameter.

    :param name: One of ``JC69``, ``K2``, ``F81``, ``HKY``, ``GTR``.
    :param fit_kappa: Forwarded to :class:`~ancestree.models.K2`
        / :class:`~ancestree.models.HKY` constructors.
    :param fit_rates: Forwarded to :class:`~ancestree.models.GTR`.
    :param kappa: Starting transition / transversion ratio for ``K2`` / ``HKY``;
        ``None`` keeps each model's own default.
    :return: An instantiated substitution model.
    :raises ValueError: On an unknown model name.
    """
    from ancestree.models import JC69, K2, F81, HKY, GTR
    key = name.upper()
    if key == "JC69":
        return JC69()
    if key == "K2":
        return K2(fit_kappa=fit_kappa) if kappa is None else K2(kappa=kappa, fit_kappa=fit_kappa)
    if key == "F81":
        return F81()
    if key == "HKY":
        return HKY(fit_kappa=fit_kappa) if kappa is None else HKY(kappa=kappa, fit_kappa=fit_kappa)
    if key == "GTR":
        return GTR(fit_rates=fit_rates)
    raise ValueError(
        f"Unknown --model {name!r}; choose one of JC69, K2, F81, HKY, GTR."
    )


def _empirical_composition(vcf: str, max_sites: int | None):
    """Derive the base composition from the input variants.

    :param vcf: VCF / BCF / VCZ path to read.
    :param max_sites: Cap on the sites tallied. ``None`` reads all of them.
    :return: A :class:`~ancestree.sites.BaseComposition` carrying empirical
        ``pi`` and the Ts/Tv ratio behind
        :attr:`BaseComposition.kappa_estimate
        <ancestree.sites.BaseComposition.kappa_estimate>`.
    """
    from ancestree.sites import BaseComposition
    from ancestree.sources import CyVCF2Source, VcfZarrSource

    source = VcfZarrSource(vcf) if _path_format(vcf) == "vcz" else CyVCF2Source(vcf)
    return BaseComposition.from_polymorphic_sites(source, max_sites=max_sites)


def _build_ingroup_weight(
    name: str, *, ingroup_samples: Sequence[str] | None,
    parallelize: bool = False,
):
    """Instantiate an :class:`~ancestree.priors.IngroupWeight` by short name.

    - ``none`` → no weight. The ingroup contributes nothing and the outgroups
      alone decide.
    - ``kingman`` → :class:`~ancestree.priors.KingmanIngroupWeight`, the
      closed-form ``(n - i) / n``.
    - ``adaptive`` → :class:`~ancestree.priors.AdaptiveIngroupWeight`, one
      probability per spectrum class fitted once the branch-rate MLE is in
      hand.

    :param name: ``none`` / ``kingman`` / ``adaptive``.
    :param ingroup_samples: Required for ``kingman`` / ``adaptive``.
    :param parallelize: Whether ``adaptive`` dispatches its per-class fit
        across worker processes.
    :return: An :class:`~ancestree.priors.IngroupWeight`, or ``None``.
    :raises ValueError: On an unknown name or missing ingroup ids.
    """
    from ancestree.priors import AdaptiveIngroupWeight, KingmanIngroupWeight

    key = name.lower()
    if key == "none":
        from ancestree.priors import NoIngroupWeight
        return NoIngroupWeight()
    if not ingroup_samples:
        raise ValueError(
            f"--ingroup-weight {key} requires --ingroup to know which samples "
            "make up the ingroup."
        )
    if key == "kingman":
        return KingmanIngroupWeight(list(ingroup_samples))
    if key == "adaptive":
        return AdaptiveIngroupWeight(
            list(ingroup_samples), parallelize=parallelize,
        )
    raise ValueError(
        f"Unknown --ingroup-weight {name!r}; choose one of none, kingman, "
        "adaptive."
    )


def _build_prior(name: str, *, model):
    """Instantiate the prior on the state at the reporting node.

    :param name: ``composition`` defers to the base composition the inference
        derives from the data. ``uniform`` overrides it with a flat prior over
        the alphabet.
    :param model: Substitution model, for the alphabet size.
    :return: A :class:`~ancestree.priors.StationaryPrior`, or ``None`` to let
        the inference use its own base composition.
    :raises ValueError: On an unknown name.
    """
    from ancestree.priors import StationaryPrior

    key = name.lower()
    if key == "composition":
        return None
    if key == "uniform":
        return StationaryPrior(model)
    raise ValueError(
        f"Unknown --prior {name!r}; choose composition or uniform."
    )


def _build_focal(args: argparse.Namespace) -> FocalNode:
    """Build the readout anchor from the parsed focal flags.

    :param args: Parsed arguments carrying ``--focal`` and the mutually
        exclusive focal-position flags.
    :return: The node the per-site posterior is reported at.
    """
    return FocalNode(
        args.focal.replace("-", "_"),
        fraction=args.focal_fraction,
        coalescences=args.focal_coalescences,
        depth=args.focal_depth,
    )


def _configure_logging(verbosity: int, quiet: bool) -> None:
    """Set the ``ancestree`` logger level from ``-v`` / ``--quiet`` flags.

    Default is ``INFO`` (matching the package import-time default).
    ``-v`` lowers to ``DEBUG``. ``--quiet`` raises to ``WARNING``.
    The two are mutually exclusive, rejected by :func:`run`.

    :param verbosity: Count from ``-v`` (0 = INFO, ≥1 = DEBUG).
    :param quiet: ``True`` raises the level to WARNING.
    """
    logger = logging.getLogger("ancestree")
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbosity >= 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level :class:`argparse.ArgumentParser` for ``ancestree``.

    Public so the parser can be introspected without invoking :func:`main`.

    :return: The configured top-level argparse parser.
    """
    # Verbosity flags are accepted before and after the subcommand via a shared
    # parent parser. SUPPRESS defaults keep a parse pass that does not see the
    # flag from clobbering an earlier value. Exclusivity is enforced in run.
    verbosity_parent = argparse.ArgumentParser(add_help=False)
    verbosity_parent.add_argument(
        "-v", "--verbose", action="count", default=argparse.SUPPRESS,
        help="Log at DEBUG level (the default is INFO). Excludes -q.",
    )
    verbosity_parent.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="Log at WARNING level, silencing the INFO output. Excludes -v.",
    )
    verbosity_parent.add_argument(
        "--no-progress", action="store_true", default=argparse.SUPPRESS,
        help="Do not show progress bars.",
    )

    parser = argparse.ArgumentParser(
        prog="ancestree",
        parents=[verbosity_parent],
        description=(
            "Ancestral-allele annotation via Felsenstein likelihood on ARGs "
            "and fixed trees. Pick a subcommand: `fixed-tree` for "
            "outgroup-ladder (EST-SFS-style) inference, `arg` for "
            "tskit/ARG-based (PolarBEAR-style) inference, and `local-tree` "
            "for VCF-only inferred-local-tree inference."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"ancestree {__version__}",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Re-raise unexpected errors with their traceback. Without it "
             "a single line is reported. Argument errors are unaffected.",
    )

    sub = parser.add_subparsers(
        dest="command", required=True, metavar="{fixed-tree,arg,local-tree}",
    )

    output_parent = argparse.ArgumentParser(add_help=False)
    output_parent.add_argument(
        "--no-posterior", action="store_true",
        help="Omit the per-state posterior from the output (the AA_post INFO "
             "field, the variant_AA_post array, or the tree-sequence site "
             "metadata). The MAP allele and its probability are still written.",
    )
    output_parent.add_argument(
        "--min-confidence", type=float, default=None, metavar="P",
        help="Blank the ancestral-allele call at any site whose MAP posterior "
             "is below P, writing AA=. instead. The posterior itself is still "
             "written where it is requested. Default: report every site.",
    )
    output_parent.add_argument(
        "--restrict-samples", action="store_true",
        help="Keep only the ingroup and outgroups in an output annotating "
             "the input.",
    )

    _add_fixed_tree_parser(sub, parents=[verbosity_parent, output_parent])
    _add_arg_parser(sub, parents=[verbosity_parent, output_parent])
    _add_local_tree_parser(sub, parents=[verbosity_parent, output_parent])
    return parser


def _add_model_args(p: argparse.ArgumentParser, *, fit_flags: bool = True) -> None:
    """Add the shared ``--model`` flag (and, when ``fit_flags``, the
    ``--fit-kappa`` / ``--fit-rates`` toggles), identical across subcommands.

    :param p: Subcommand parser to extend.
    :param fit_flags: When ``True`` (fixed-tree only) also add the
        model-parameter fit toggles. ``False`` for ``arg`` and ``local-tree``
        mode, which take branch lengths from the ARG or the inferred local
        trees and never fit the rate matrix.
    """
    p.add_argument(
        "--model", default="JC69",
        choices=["JC69", "K2", "F81", "HKY", "GTR"],
        help="Substitution model. Default: JC69.",
    )
    if fit_flags:
        p.add_argument(
            "--fit-kappa", action="store_true",
            help="K2/HKY only: expose kappa as a free parameter for joint MLE.",
        )
        p.add_argument(
            "--fit-rates", action="store_true",
            help="GTR only: fit the six exchangeability rates jointly.",
        )


def _add_focal_args(p: argparse.ArgumentParser) -> None:
    """Add the panel-membership flags and the readout-anchor group.

    :param p: Subcommand parser to extend.
    """
    p.add_argument(
        "--ingroup", type=_split_csv,
        help=(
            "Comma-separated ingroup sample ids. Needed with "
            "--focal=ingroup-mrca; defaults to every sample not in --outgroups."
        ),
    )
    p.add_argument(
        "--outgroups", type=_split_csv,
        help=("Comma-separated outgroup sample ids. Defaults to every sample "
              "not in --ingroup. With both given, other samples are ignored."),
    )
    p.add_argument(
        "--focal", default="ingroup-mrca",
        choices=["ingroup-mrca", "panel-root"],
        help=(
            "Anchor to report the posterior at. 'ingroup-mrca' (default) is "
            "the ingroup's own MRCA; 'panel-root' is the MRCA of the whole "
            "panel, deeper whenever outgroups are in the panel."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--focal-fraction", type=float, metavar="F",
        help=(
            "Move the readout point F of the way from the anchor to the root, "
            "with F in [0, 1]. Scale-free."
        ),
    )
    g.add_argument(
        "--focal-coalescences", type=int, metavar="K",
        help="Move the readout point up past K coalescences above the anchor.",
    )
    g.add_argument(
        "--focal-depth", type=float, metavar="D",
        help=(
            "Place the readout point at absolute depth D, in the tree's own "
            "branch units, bounded below by the anchor."
        ),
    )


def _warn_model_defaults(mode: str, model_name: str) -> None:
    """Report the model parameters this subcommand cannot set.

    ``arg`` and ``local-tree`` take branch lengths from the genealogy and offer
    no ``--empirical-composition`` or fit toggles, so a model with free
    parameters runs at uniform base frequencies and the model's own default
    kappa.

    :param mode: Subcommand name, for the message.
    :param model_name: Value of ``--model``.
    """
    key = model_name.upper()
    if key in ("F81", "HKY", "GTR"):
        _log.warning(
            "%s: model %s runs with uniform base frequencies, as this subcommand "
            "has no --empirical-composition. Use the Python API to supply a "
            "base composition.", mode, model_name)
    if key in ("K2", "HKY"):
        _log.warning(
            "%s: model %s runs at its default kappa, as this subcommand "
            "has no --fit-kappa. Use the Python API to set it.",
            mode, model_name)



def _add_fixed_tree_parser(
    sub: argparse._SubParsersAction, *, parents: "list | None" = None,
) -> None:
    """Register the ``fixed-tree`` subcommand.

    .. note:: Defaults come from :class:`~ancestree.inference.FixedTreeInference`
       via :func:`_lib_default`.

    :param sub: The top-level subparsers action.
    :param parents: Parent parsers (e.g. the shared verbosity group) to
        inherit so the flags are accepted after the subcommand too.
    """
    from ancestree.inference import FixedTreeInference

    p = sub.add_parser(
        "fixed-tree",
        parents=parents or [],
        help="EST-SFS-style: ML-fit a fixed outgroup-ladder tree on a VCF.",
        description=(
            "ML-fit branch rates on an OutgroupLadderTree built from "
            "--species-tree, then emit per-site posteriors at the focal node, "
            "the ingroup MRCA by default, as an annotated VCF."
        ),
    )
    p.add_argument(
        "--samples", type=_split_csv, default=None,
        help="Comma-separated subset of VCF samples to read (default: all).",
    )
    p.add_argument(
        "--focal", default="ingroup-mrca",
        choices=["ingroup-mrca", "panel-root"],
        help=(
            "Anchor to report the posterior at. 'ingroup-mrca' (default) is "
            "the ingroup's own MRCA; 'panel-root' is the MRCA of the whole "
            "panel, deeper whenever outgroups are in the panel."
        ),
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--focal-fraction", type=float, metavar="F",
        help=(
            "Move the readout point F of the way from the ingroup MRCA to the "
            "ladder's deepest join, with F in [0, 1]. A site the ingroup has "
            "fixed reads back its own allele at F=0."
        ),
    )
    g.add_argument(
        "--focal-coalescences", type=int, metavar="K",
        help="Move the readout point up past K outgroup joins.",
    )
    g.add_argument(
        "--focal-depth", type=float, metavar="D",
        help=(
            "Place the readout point D expected substitutions per site above "
            "the ingroup MRCA."
        ),
    )
    p.add_argument(
        "--vcf", required=True,
        help="Input VCF / VCF.GZ / BCF / VCZ store with the polymorphic sites.",
    )
    p.add_argument(
        "--species-tree", default=None,
        help="Dated Newick to use as the tree, topology and branch lengths "
             "both. Skips the ML fit; extra taxa are pruned and the ladder "
             "order comes from the topology. Without it, the ladder is built "
             "from --outgroups, closest first, and its rates are fitted.",
    )
    p.add_argument(
        "--ingroup", type=_split_csv,
        help=(
            "Comma-separated ingroup sample ids "
            "(must match VCF sample columns and Newick leaves). Defaults to "
            "every haplotype whose individual is not named in --outgroups."
        ),
    )
    p.add_argument(
        "--outgroups", required=True, type=_split_csv,
        help=(
            "Comma-separated outgroup sample ids, closest-first "
            "(must match VCF and Newick names)."
        ),
    )
    _add_model_args(p)
    p.add_argument(
        "--baseline-check", action=argparse.BooleanOptionalAction,
        default=_lib_default(FixedTreeInference, "baseline_check"),
        help="Log MAP agreement with the majority-outgroup baseline, as a "
             "consistency check on the fit. Buffers sites; pass "
             "--no-baseline-check to skip it. Default: %(default)s.",
    )
    p.add_argument(
        "--ingroup-weight", default="kingman",
        choices=["none", "kingman", "adaptive"],
        help="How the ingroup allele frequencies enter, as a likelihood on "
             "the ingroup MRCA. Default: kingman.",
    )
    p.add_argument(
        "--prior", default="composition", choices=["composition", "uniform"],
        help="Prior on the state at the reporting node: the base composition "
             "of the data, or uniform over the alphabet. Default: "
             "composition.",
    )
    p.add_argument(
        "--n-starts", type=int, metavar="N",
        default=_lib_default(FixedTreeInference, "n_starts"),
        help="Independent optimiser starts for the branch-rate fit; more "
             "starts trade runtime against the chance of a local optimum. "
             "Recorded in the output provenance. Default: %(default)s.",
    )
    p.add_argument(
        "--seed", type=int, metavar="S",
        default=_lib_default(FixedTreeInference, "seed"),
        help="Seed for the multistart draw, so a fit is reproducible. "
             "Recorded in the output provenance. Default: %(default)s.",
    )
    p.add_argument(
        "--n-target-sites", type=int, default=None,
        help=(
            "Genome length L for monomorphic-site calibration, required for "
            "the branch-rate MLE. See the Python API to supply an explicit "
            "base composition instead."
        ),
    )
    p.add_argument(
        "--empirical-composition", action="store_true",
        help="Derive the base composition (pi) and the transition / transversion "
             "ratio (kappa) from the input variants in place of the uniform "
             "base frequencies. Relevant to --model F81 / HKY / GTR; the kappa "
             "estimate seeds K2 and HKY unless --fit-kappa is given.",
    )
    p.add_argument(
        "--max-calibration-sites", type=int, default=None,
        help="Cap on the sites tallied for --empirical-composition. Default: all.",
    )
    p.add_argument(
        "--parallelize", action="store_true",
        help="Dispatch the branch-rate multistart and, with --ingroup-weight adaptive, "
             "the per-bin prior fit across worker processes. Off by default.",
    )
    p.add_argument(
        "--n-workers", type=int,
        default=_lib_default(FixedTreeInference, "n_workers"),
        help="Worker processes used when --parallelize is given. Defaults to "
             "min(n_starts, CPU count), as in the Python API.",
    )
    p.add_argument(
        "--out", required=True,
        help="Output VCF (.vcf, .vcf.gz, .vcf.bgz, .bcf) or VCF Zarr store "
             "(.vcz).",
    )
    p.add_argument(
        "--template-vcf", default=None,
        help=(
            "VCF whose records the output copies. Defaults to --vcf when it "
            "is a VCF."
        ),
    )
    p.set_defaults(handler=_run_fixed_tree)


def _add_arg_parser(
    sub: argparse._SubParsersAction, *, parents: "list | None" = None,
) -> None:
    """Register the ``arg`` subcommand. Arguments as for
    :func:`_add_fixed_tree_parser`."""
    from ancestree.inference import ARGBasedInference

    p = sub.add_parser(
        "arg",
        parents=parents or [],
        help="PolarBEAR-style: per-site ARG-local-tree inference on a .trees file.",
        description=(
            "Walk a tskit tree sequence and emit per-site posteriors over "
            "the local ARG root state. Outgroups are optional; the local "
            "ARG topology supplies the genealogical signal."
        ),
    )
    p.add_argument(
        "--trees", required=True, nargs="+", metavar="TREES",
        help="Input tskit .trees file(s). Passing several marginalises the "
             "per-site posterior over them, which is how a posterior sample "
             "of genealogies (an MCMC chain) is consumed; the draw count is "
             "recorded in the output provenance.",
    )
    p.add_argument(
        "--mu", type=float, default=None,
        help=(
            "Per-site per-generation mutation rate, which scales branch "
            "lengths from generations into expected substitutions. Defaults "
            "to 1e-8 with a warning; that is a human / great-ape figure, so "
            "pass your species' rate."
        ),
    )
    p.add_argument(
        "--mu-matches-time-units", action="store_true",
        help=(
            "Treat --mu as being per unit of the tree sequence's own time "
            "axis and not per generation. Use this when the branch "
            "lengths are not in generations and --mu was fitted against that "
            "same axis, e.g. an undated tsinfer ARG scored with a rate fitted "
            "to itself. Without it an uncalibrated tree sequence is refused."
        ),
    )
    _add_model_args(p, fit_flags=False)
    p.add_argument(
        "--prior", default="composition", choices=["composition", "uniform"],
        help=(
            "Prior on the state at the reporting node: the base composition "
            "of the data, or uniform over the alphabet. The ingroup weights "
            "(kingman / adaptive) are fixed-tree only and are selected with "
            "--ingroup-weight. Default: composition."
        ),
    )
    p.add_argument(
        "--chrom", default=_lib_default(ARGBasedInference, "chrom"),
        help="Contig label of the annotated sites. Default: %(default)s.",
    )
    p.add_argument(
        "--n-workers", type=int,
        default=_lib_default(ARGBasedInference, "n_workers"),
        help="Worker processes for the tree walk (fork-pool). "
             "Default: %(default)s.",
    )
    _add_focal_args(p)
    p.add_argument(
        "--out", required=True,
        help=(
            "Output path. The format is inferred from the extension: "
            "'.vcf', '.vcf.gz', '.vcf.bgz' and '.bcf' write an annotated VCF, '.vcz' an "
            "annotated VCF Zarr store, and '.trees' an annotated tskit tree "
            "sequence."
        ),
    )
    p.set_defaults(handler=_run_arg)


def _add_local_tree_parser(
    sub: argparse._SubParsersAction, *, parents: "list | None" = None,
) -> None:
    """Register the ``local-tree`` subcommand. Arguments as for
    :func:`_add_fixed_tree_parser`."""
    from ancestree.local_tree_inference import LocalTreeInference

    p = sub.add_parser(
        "local-tree",
        parents=parents or [],
        help="VCF-only: infer per-window local trees from genotypes, then infer ancestral alleles.",
        description=(
            "Infer a dated local tree per genomic window from genotypes alone "
            "(pairwise-coalescent HMM + per-window UPGMA) and emit per-site "
            "posteriors over the local-tree root state. No outgroups and no "
            "input ARG are required."
        ),
    )
    p.add_argument(
        "--vcf", required=True,
        help="Input VCF / VCF.GZ / BCF / VCZ store of the panel genotypes.",
    )
    p.add_argument(
        "--ploidy", type=int, default=None,
        help=("Haplotypes per sample. Read from the first called genotype "
              "when omitted. VCF / BCF input only."),
    )
    p.add_argument(
        "--phased", dest="phased", action=argparse.BooleanOptionalAction,
        default=None,
        help=("Whether the genotypes are phased. Read from each record's own "
              "phase flag when omitted. Forcing --phased on unphased data "
              "builds a clade out of the arbitrary allele order; "
              "--no-phased randomises that order per site."),
    )
    p.add_argument(
        "--phase-seed", type=int, default=None,
        help=("Seed for the per-site haplotype-order randomisation applied to "
              "unphased heterozygotes. Default: %(default)s."),
    )
    p.add_argument(
        "--samples", type=_split_csv, default=None,
        help=(
            "Comma-separated subset of VCF samples to read (default: all)."
        ),
    )
    p.add_argument(
        "--sequence-length", type=float, default=None,
        help=("Region length in bp (the local-tree windows tile [0, L)). "
              "Give it whenever it is known: under --chunk-size it is what "
              "lays one window grid over the whole region and not one per "
              "segment."),
    )
    p.add_argument(
        "--chunk-size", type=_chunk_size,
        default=_lib_default(LocalTreeInference, "chunk_size"),
        help=("Process the genome in chunks of this size (bp int or '5mb') to "
              "bound memory and enable parallelism: contigs are independent and "
              "long contigs are sliced with a halo overlap. The input is "
              "streamed (must be position-sorted) and --sequence-length is "
              "then derived per segment. Pass 'none' to build the whole "
              "input as one region. Default: %(default)s."),
    )
    p.add_argument(
        "--halo", default=_lib_default(LocalTreeInference, "halo"),
        help=("Halo overlap (bp int / '50kb', or 'auto') added when a contig is "
              "sliced across chunks, then discarded so the posteriors do not "
              "depend on the chunking. 'auto' grounds it in the recombination correlation "
              "length. Only used with --chunk-size. Default: %(default)s."),
    )
    p.add_argument(
        "--n-workers", type=int,
        default=_lib_default(LocalTreeInference, "n_workers"),
        help=("Fork-pool workers. With --chunk-size, fans whole segments across "
              "the pool (peak memory ~ n_workers x chunk-size). "
              "Default: %(default)s."),
    )
    p.add_argument(
        "--mu", type=float, default=None,
        help="Per-site per-generation mutation rate. Defaults to 1e-8 with a "
             "warning; that is a human / great-ape figure, so pass your "
             "species' rate.",
    )
    p.add_argument(
        "--rec-rate", type=float, default=None,
        help=(
            "Per-site per-generation recombination rate driving the HMM's "
            "TMRCA-reset transitions. Defaults to 1e-8 with a warning; that "
            "is a human / great-ape figure, so pass your species' rate."
        ),
    )
    p.add_argument(
        "--recombination-map", default=None,
        help=(
            "Optional HapMap-format genetic map file (Chr, Position(bp), "
            "Rate(cM/Mb), Map(cM)) driving the HMM's TMRCA-reset rate in place "
            "of the constant --rec-rate. Read via msprime.RateMap.read_hapmap."
        ),
    )
    p.add_argument(
        "--accessibility", default=None,
        help=(
            "Optional BED file of callable regions, scaling each block's "
            "Poisson emission by its accessible base-pair fraction, with a "
            "block whose fraction does not exceed 0.05 emitting uniformly so "
            "inaccessible gaps are not read as recent-TMRCA stretches."
        ),
    )
    p.add_argument(
        "--mutation-map", default=None,
        help=(
            "Optional bedGraph file (chrom, start, end, rate) of the local "
            "per-site mutation rate, used in the HMM emission in place of the "
            "constant --mu; gaps default to --mu. A high-rate block's extra "
            "differences are then read as mutation, not a deeper coalescence."
        ),
    )
    p.add_argument(
        "--ensemble-size", type=int, metavar="B",
        default=_lib_default(LocalTreeInference, "n_ensemble"),
        help=(
            "Genealogies per window to marginalise over, drawn from the "
            "pairwise HMM's posterior. Cost is linear in B. Default: %(default)s."
        ),
    )
    p.add_argument(
        "--member-chunk", type=int, metavar="M",
        default=_lib_default(LocalTreeInference, "member_chunk"),
        help="Genealogies drawn and scored at once. Peak memory is set by "
             "this and not by --ensemble-size, so lower it on wide panels. "
             "--ensemble-size must be a whole number of chunks. Default: %(default)s.",
    )
    p.add_argument(
        "--ensemble-seed", type=int, metavar="SEED",
        default=_lib_default(LocalTreeInference, "ensemble_seed"),
        help="Base seed for the genealogy draws. Default: %(default)s.",
    )
    p.add_argument(
        "--no-ensemble", action="store_true",
        help=("Skip genealogy sampling and use the plug-in estimate: the "
              "posterior-mean TMRCA, one tree per window."),
    )
    p.add_argument(
        "--window", default=_lib_default(LocalTreeInference, "window"),
        help=(
            "Local-tree window: a SNP count ('8snp') or a bp width "
            "('20kb', '5000'). Default: %(default)s."
        ),
    )
    p.add_argument(
        "--block-size", default=_lib_default(LocalTreeInference, "block_size"),
        help=(
            "HMM emission block width: a SNP count ('4snp') or a bp width. "
            "Omit for a density-adaptive default (~4 SNPs per block)."
        ),
    )
    p.add_argument(
        "--n-time-bins", type=int,
        default=_lib_default(LocalTreeInference, "n_time_bins"),
        help="Number of discretized coalescent-time bins. Default: %(default)s.",
    )
    _add_model_args(p, fit_flags=False)
    p.add_argument(
        "--prior", default="composition", choices=["composition", "uniform"],
        help="Prior on the state at the reporting node: the base "
             "composition of the data, or uniform over the alphabet. "
             "Default: composition.",
    )
    p.add_argument(
        "--chrom", default=None,
        help="Contig label of the annotated sites (default: the source's).",
    )
    p.add_argument(
        "--out", required=True,
        help=(
            "Output path. The format is inferred from the extension: "
            "'.vcf', '.vcf.gz', '.vcf.bgz' and '.bcf' write an annotated VCF, '.vcz' an "
            "annotated VCF Zarr store and '.trees' an annotated tskit tree "
            "sequence."
        ),
    )
    p.add_argument(
        "--out-trees", default=None,
        help="Optional path to also dump the inferred (pre-annotation) "
             "local-tree sequence.",
    )
    _add_focal_args(p)
    p.set_defaults(handler=_run_local_tree)


# ---------------------------------------------------------------------------- handlers


def _run_fixed_tree(args: argparse.Namespace) -> int:
    """Execute the ``fixed-tree`` subcommand.

    Builds an :class:`~ancestree.inference.FixedTreeInference` from the
    parsed args, runs the branch-rate ML fit, and writes the annotated output:
    a VCF via :meth:`FixedTreeInference.to_vcf() <ancestree.inference.FixedTreeInference.to_vcf>`, or a VCF
    Zarr store via :meth:`FixedTreeInference.to_zarr() <ancestree.inference.FixedTreeInference.to_zarr>` when
    ``--out`` is a ``.vcz`` path.

    :param args: Parsed argparse namespace from :func:`build_parser`.
    :return: Process exit code (0 on success).
    :raises SystemExit: If ``--out`` has an unsupported extension.
    """
    st = args.species_tree is not None
    _warn_ignored(args, [
        ("fit_kappa", st, "with --species-tree: the branch-rate fit is skipped"),
        ("fit_rates", st, "with --species-tree: the branch-rate fit is skipped"),
        ("n_target_sites", st,
         "with --species-tree: only the fit consumes the monomorphic weights"),
        ("parallelize", st, "with --species-tree: no fit runs to parallelise"),
        ("max_calibration_sites", not args.empirical_composition,
         "without --empirical-composition"),
    ])
    if st and args.ingroup_weight == "adaptive":
        _log.warning(
            "--ingroup-weight adaptive with --species-tree: its per-bin fit "
            "runs inside fit(), which --species-tree skips, so the Kingman "
            "values are used instead")
    out_format = _path_format(args.out)
    out_is_zarr = out_format == "vcz"
    if out_format not in ("vcf", "vcz"):
        raise SystemExit(
            f"fixed-tree --out must end with .vcf, .vcf.gz, .vcf.bgz, .bcf, or .vcz "
            f"(got {args.out!r}). The .trees format is only valid under "
            f"the `arg` and `local-tree` subcommands."
        )
    from ancestree.inference import FixedTreeInference
    from ancestree.trees import OutgroupLadderTree

    ingroup = args.ingroup
    if not ingroup:
        ingroup = _derive_ingroup(args.vcf, args.outgroups,
                                  sample_filter=(args.samples or None))

    # Build the ladder first, so a topology error surfaces early. A supplied Newick is used verbatim. Otherwise --outgroups is
    # taken as closest-first and the rates are fitted.
    if args.species_tree is not None:
        with open(args.species_tree) as fh:
            newick_str = fh.read().strip()
        tree = OutgroupLadderTree.from_newick(
            newick_str, ingroup_samples=ingroup,
            outgroup_samples=args.outgroups,
        )
    else:
        tree = OutgroupLadderTree(ingroup, args.outgroups)

    template_vcf = args.template_vcf
    if out_is_zarr and template_vcf is not None:
        _log.warning(
            "fixed-tree: --template-vcf is ignored for a .vcz --out"
        )

    base_composition = None
    if args.empirical_composition:
        base_composition = _empirical_composition(args.vcf, args.max_calibration_sites)
        pi = ", ".join(f"{x:.3f}" for x in base_composition.pi)
        _log.info(
            "Empirical base composition pi = (%s), kappa = %.3f",
            pi, base_composition.kappa_estimate,
        )
        if args.model in ("K2", "HKY") and not args.fit_kappa:
            args_kappa = base_composition.kappa_estimate
        else:
            args_kappa = None
    else:
        args_kappa = None
        if args.model in ("F81", "HKY", "GTR"):
            _log.warning(
                "Model %s runs with uniform base frequencies. Pass "
                "--empirical-composition to derive pi and kappa from the data.",
                args.model,
            )
    model = _build_model(
        args.model, fit_kappa=args.fit_kappa, fit_rates=args.fit_rates,
        kappa=args_kappa,
    )
    ingroup_weight = _build_ingroup_weight(
        args.ingroup_weight, ingroup_samples=ingroup,
        parallelize=args.parallelize,
    )
    prior = _build_prior(args.prior, model=model)

    inference = FixedTreeInference(
        args.vcf,
        model=model,
        tree=tree,
        sample_filter=(args.samples or None),
        baseline_check=args.baseline_check,
        focal=_build_focal(args),
        fit_required=args.species_tree is None,
        n_target_sites=args.n_target_sites,
        n_starts=args.n_starts,
        seed=args.seed,
        base_composition=base_composition,
        ingroup_weight=ingroup_weight,
        prior=prior,
        parallelize=args.parallelize,
        n_workers=args.n_workers,
    )
    if args.species_tree is None:
        inference.fit()  # ML branch rates. Skipped when a dated tree is given

    if out_is_zarr:
        n = inference.to_zarr(args.out, store_posterior=not args.no_posterior,
                              min_confidence=args.min_confidence,
                              restrict_samples=args.restrict_samples)
        _log.info("Wrote %d annotated variants to %s", n, args.out)
    else:
        n = inference.to_vcf(args.out, input_vcf=template_vcf,
                             store_posterior=not args.no_posterior,
                             min_confidence=args.min_confidence,
                             restrict_samples=args.restrict_samples)
        _log.info("Wrote %d annotated records to %s", n, args.out)
    return 0


def _run_arg(args: argparse.Namespace) -> int:
    """Execute the ``arg`` subcommand.

    Routes to :meth:`ARGBasedInference.to_vcf() <ancestree.inference.ARGBasedInference.to_vcf>` or
    :meth:`ARGBasedInference.to_arg() <ancestree.inference.ARGBasedInference.to_arg>` based on the
    ``--out`` extension.

    :param args: Parsed argparse namespace from :func:`build_parser`.
    :return: Process exit code (0 on success).
    :raises SystemExit: If ``--out`` has an unsupported extension.
    """
    from ancestree.inference import ARGBasedInference

    model = _build_model(args.model, fit_kappa=False, fit_rates=False)
    _warn_model_defaults("arg", args.model)
    prior = _build_prior(args.prior, model=model)
    focal = _build_focal(args)

    inference = ARGBasedInference(
        args.trees[0] if len(args.trees) == 1 else args.trees,
        model=model,
        mu=args.mu,
        mu_matches_time_units=args.mu_matches_time_units,
        prior=prior,
        chrom=args.chrom,
        n_workers=args.n_workers,
        focal=focal,
        ingroup_samples=args.ingroup,
        outgroup_samples=args.outgroups,
    )

    out_format = _path_format(args.out)
    if out_format == "trees":
        n = inference.to_arg(args.out,
                             store_posterior=not args.no_posterior,
                             min_confidence=args.min_confidence,
                             restrict_samples=args.restrict_samples)
        _log.info(
            "arg: wrote %d annotated sites to %s", n, args.out,
        )
        return 0
    if out_format == "vcf":
        n = inference.to_vcf(args.out, store_posterior=not args.no_posterior,
                             min_confidence=args.min_confidence,
                             restrict_samples=args.restrict_samples)
        _log.info(
            "arg: wrote %d annotated records to %s", n, args.out,
        )
        return 0
    if out_format == "vcz":
        n = inference.to_zarr(args.out,
                              store_posterior=not args.no_posterior,
                              min_confidence=args.min_confidence,
                              restrict_samples=args.restrict_samples)
        _log.info(
            "arg: wrote %d annotated records to %s", n, args.out,
        )
        return 0
    raise SystemExit(
        f"arg --out must end with .vcf, .vcf.gz, .vcf.bgz, .bcf, .vcz, or .trees "
        f"(got {args.out!r})."
    )


def _read_bed_intervals(path: str) -> list[tuple[float, float]]:
    """Read half-open ``(start, end)`` bp intervals from a BED file.

    Columns beyond the first three (chrom, start, end) are ignored. ``track``,
    ``browser``, comment lines and blanks are skipped. Coordinates are taken
    verbatim (BED is 0-based half-open), pooled across contigs: the
    local-tree accessibility mask is applied per segment, so the contig column
    is not needed here.
    """
    intervals: list[tuple[float, float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if len(f) < 3:
                continue
            intervals.append((float(f[1]), float(f[2])))
    return intervals


def _import_msprime():
    """Import ``msprime`` (the ``maps`` extra) or exit with a clear message."""
    try:
        import msprime
    except ImportError as e:
        raise SystemExit(
            "the --recombination-map / --mutation-map options require msprime; "
            "install it with `pip install ancestree[maps]`."
        ) from e
    return msprime


def _read_ratemap_bedgraph(path: str, default_rate: float):
    """Build an :class:`msprime.RateMap` from a 4-column bedGraph (chrom, start,
    end, rate).

    Rows are sorted and assumed non-overlapping. Any gap (including a prefix
    before the first interval) is filled with ``default_rate``, so the map
    covers ``[0, last_end]`` contiguously. The contig column is ignored (the
    map is applied per segment by position, as for the recombination map).

    :raises SystemExit: If msprime is not installed, or the file yields no
        usable interval.
    """
    msprime = _import_msprime()

    intervals: list[tuple[float, float, float]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("#", "track", "browser")):
                continue
            f = line.split()
            if len(f) < 4:
                continue
            intervals.append((float(f[1]), float(f[2]), float(f[3])))
    intervals.sort()
    position: list[float] = [0.0]
    rate: list[float] = []
    for start, end, r in intervals:
        cur = position[-1]
        if start < cur:  # overlap: clip to current frontier
            start = cur
        if end <= start:
            continue
        if start > cur:  # gap before this interval
            position.append(start)
            rate.append(default_rate)
        position.append(end)
        rate.append(r)
    if len(position) < 2:
        raise SystemExit(f"--mutation-map: no usable intervals in {path}")
    return msprime.RateMap(position=position, rate=rate)


def _run_local_tree(args: argparse.Namespace) -> int:
    """Execute the ``local-tree`` subcommand.

    Builds a :class:`~ancestree.sites.SiteSource` from the genotype VCF,
    infers per-window local trees via
    :class:`~ancestree.local_tree_inference.LocalTreeInference`, and writes an
    annotated VCF, VCF-Zarr (``.vcz``), or ``.trees`` file based on the
    ``--out`` extension.

    :param args: Parsed argparse namespace from :func:`build_parser`.
    :return: Process exit code (0 on success).
    :raises SystemExit: If ``--out`` has an unsupported extension.
    """
    chunked = getattr(args, "chunk_size", None) is not None
    no_ens = bool(getattr(args, "no_ensemble", False))
    _warn_ignored(args, [
        ("halo", not chunked, "without --chunk-size"),
        ("ensemble_size", no_ens, "with --no-ensemble"),
        ("ensemble_seed", no_ens, "with --no-ensemble"),
        ("n_workers", not chunked and not no_ens,
         "on the default single-region ensemble path, which is threaded "
         "inside the kernel: pass --chunk-size or --no-ensemble to fan out"),
        ("member_chunk", no_ens, "with --no-ensemble"),
    ])
    out_format = _path_format(args.out)
    if out_format is None:
        raise SystemExit(
            f"local-tree --out must end with .vcf, .vcf.gz, .vcf.bgz, .bcf, .vcz, or "
            f".trees (got {args.out!r})."
        )
    from ancestree.local_tree_inference import LocalTreeInference

    from ancestree.sites import SiteSource
    source = cast(SiteSource, SiteSource.resolve(
        args.vcf, sample_filter=(args.samples or None),
        ploidy=args.ploidy, phased=args.phased, phase_seed=args.phase_seed,
    ))
    # Canonical (ploidy-expanded) haplotype ids. Left to the library without
    # --samples, so the panel then follows --ingroup and --outgroups.
    sample_names = source.samples() if args.samples else None

    model = _build_model(args.model, fit_kappa=False, fit_rates=False)
    _warn_model_defaults("local-tree", args.model)
    prior = _build_prior(args.prior, model=model)
    focal = _build_focal(args)

    recombination_map = None
    if args.recombination_map is not None:
        recombination_map = _import_msprime().RateMap.read_hapmap(args.recombination_map)
    accessibility = (
        _read_bed_intervals(args.accessibility)
        if args.accessibility is not None else None
    )
    mutation_map = (
        # Gaps are filled with the same rate the inference would fall back to.
        # args.mu is None unless given, and msprime turns a None rate into NaN.
        _read_ratemap_bedgraph(
            args.mutation_map,
            default_rate=args.mu if args.mu is not None else DEFAULT_MU,
        )
        if args.mutation_map is not None else None
    )

    inference = LocalTreeInference(
        source, model,
        mu=args.mu,
        rec_rate=args.rec_rate,
        sample_names=sample_names,
        sequence_length=args.sequence_length,
        window=args.window,
        n_ensemble=None if args.no_ensemble else args.ensemble_size,
        ensemble_seed=args.ensemble_seed,
        member_chunk=args.member_chunk,
        block_size=args.block_size,
        n_time_bins=args.n_time_bins,
        prior=prior,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
        halo=args.halo,
        recombination_map=recombination_map,
        accessibility=accessibility,
        mutation_map=mutation_map,
        focal=focal,
        ingroup_samples=args.ingroup,
        outgroup_samples=args.outgroups,
        chrom=args.chrom,
    )
    # The library receives RateMap objects, so the paths they came from are
    # recorded here or lost.
    inference._map_sources = {
        "recombination_map": args.recombination_map,
        "mutation_map": args.mutation_map,
        "accessibility": args.accessibility,
    }

    # Optional side artifact: the un-annotated reconstructed local-tree
    # sequence. On the chunked path this stitches the per-segment builds into
    # one genome-wide tree sequence (single contig).
    if args.out_trees:
        # The plug-in build, which an ensemble run does not score.
        if inference.n_ensemble is not None:
            _log.warning(
                "--out-trees writes the plug-in genealogy, which an ensemble "
                "run does not score: the annotations marginalise over %d "
                "drawn trees. Pass --no-ensemble to annotate from the tree "
                "this writes.", inference.n_ensemble)
        inference.point_tree_sequence().dump(args.out_trees)
        _log.info("Wrote the reconstructed local trees to %s",
                  args.out_trees)
    if out_format == "trees":
        n = inference.to_arg(args.out,
                             store_posterior=not args.no_posterior,
                             min_confidence=args.min_confidence,
                             restrict_samples=args.restrict_samples)
        _log.info("Wrote %d annotated sites to %s", n, args.out)
    elif out_format == "vcz":
        n = inference.to_zarr(args.out,
                              store_posterior=not args.no_posterior,
                              min_confidence=args.min_confidence,
                              restrict_samples=args.restrict_samples)
        _log.info("Wrote %d annotated variants to %s", n, args.out)
    else:
        n = inference.to_vcf(
            args.out,
            store_posterior=not args.no_posterior,
            min_confidence=args.min_confidence,
            restrict_samples=args.restrict_samples,
        )
        _log.info("Wrote %d annotated records to %s", n, args.out)
    return 0


# ---------------------------------------------------------------------------- entry point


def _supplied_dests(argv: "Sequence[str]") -> frozenset:
    """Destination names whose long option appears in ``argv``.

    Used to tell a flag left at its default from one passed at a value that
    happens to equal the default, which a comparison against the default
    cannot do.

    :param argv: Argument vector, without the program name.
    :return: Destination names, as argparse derives them from the long option.
    """
    out = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0]
        if name:
            out.add(name.replace("-", "_"))
    return frozenset(out)


def run(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch the matching subcommand handler.

    Returns the exit code without raising ``SystemExit``.

    :param argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    :return: Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args._supplied = _supplied_dests(
        sys.argv[1:] if argv is None else argv)
    verbosity = getattr(args, "verbose", 0)
    quiet = getattr(args, "quiet", False)
    if verbosity and quiet:
        parser.error(
            "argument -q/--quiet: not allowed with argument -v/--verbose")
    _configure_logging(verbosity, quiet)
    if getattr(args, "no_progress", False):
        Settings.disable_pbar = True
    return int(args.handler(args) or 0)


def _debug_requested(argv: "Sequence[str] | None") -> bool:
    """Whether ``--debug`` was asked for on this command line.

    Read back through the parser, so an argparse abbreviation such as
    ``--deb`` reaches the traceback path too.

    :param argv: Argument vector, or ``None`` for :data:`sys.argv`.
    :return: Whether the flag was given.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        known, _rest = build_parser().parse_known_args(tokens)
    except SystemExit:
        return any(t.startswith("--deb") for t in tokens)
    return bool(getattr(known, "debug", False))


def main(argv: Sequence[str] | None = None) -> None:
    """Console-script entry point. Calls :func:`run` and exits.

    Registered in ``pyproject.toml`` as ``ancestree = "ancestree.cli:main"``.

    :param argv: Optional argument vector (defaults to ``sys.argv[1:]``).
    """
    try:
        code = run(argv)
    except Exception as e:
        # Operational errors report one line. --debug re-raises so an
        # unexpected one carries its traceback.
        if _debug_requested(argv):
            raise
        print(f"ancestree: error: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code)


if __name__ == "__main__":
    main()
