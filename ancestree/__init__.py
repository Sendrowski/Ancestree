"""Ancestree: ancestral allele annotation via Felsenstein likelihood on ARGs and fixed trees."""
import logging as _logging
import os as _os
import sys as _sys

import numba as _numba

# A fork-safe threading layer, set before any kernel is imported since the layer
# is fixed when threads first launch. See Settings._fork_is_safe for the hazard.
if "NUMBA_THREADING_LAYER" not in _os.environ:
    _numba.config.THREADING_LAYER = "forksafe"

# Plain (non-notebook) tqdm, so log lines written through .write() consolidate
# into a single stderr output frame in Jupyter, not one frame per write.
from tqdm import tqdm as _tqdm

__version__ = "0.1.1"

#: Canonical nucleotide alphabet. Every per-allele array column is indexed by
#: position in this tuple.
STATES: tuple[str, ...] = ("A", "C", "G", "T")
#: Index of each state in :data:`STATES`.
STATE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATES)}
#: Allele spellings that record no observation: an absent entry, the VCF
#: missing marker, an empty placeholder, and the ``N`` no-call. Membership is
#: tested on the upper-cased allele, so any case of ``n`` is covered. A tip
#: holding one is marginalised by the kernel.
MISSING_ALLELES: frozenset[str | None] = frozenset({None, "", ".", "N"})
#: Ordered nucleotide pairs counted as transitions (purine<->purine,
#: pyrimidine<->pyrimidine). Every other substitution is a transversion.
TRANSITION_PAIRS: frozenset[tuple[str, str]] = frozenset(
    {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
)

#: Fallback per-site per-generation mutation rate (a great-ape figure). An
#: inference falling back to it logs a warning naming the value.
DEFAULT_MU: float = 1e-8
#: Fallback per-site per-generation recombination rate. Equal to
#: :data:`DEFAULT_MU`, so the fallback rho/theta ratio is 1.
DEFAULT_REC_RATE: float = 1e-8


class _TqdmLoggingHandler(_logging.StreamHandler):
    """A :class:`logging.StreamHandler` on stderr that re-emits records through
    :func:`tqdm.write`, so log lines and active progress bars co-exist cleanly.
    """

    def __init__(self):
        super().__init__(stream=_sys.stderr)

    def emit(self, record: _logging.LogRecord) -> None:
        """Format the record and emit via :func:`tqdm.write`."""
        try:
            msg = self.format(record)
            _tqdm.write(msg, file=_sys.stderr)
            self.flush()
        except Exception:  # pragma: no cover. Defensive: never let logging crash
            self.handleError(record)


class _ColoredFormatter(_logging.Formatter):
    """ANSI-colored log formatter with a per-level colour scheme.

    Colours are emitted unconditionally. Terminals that do not render ANSI
    escapes show them as harmless text.
    """

    _COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[31m",  # red
    }
    _RESET = "\033[0m"

    def format(self, record: _logging.LogRecord) -> str:
        """Wrap the standard formatter's output in the level-specific colour.

        :param record: The record to render.
        :return: The formatted line, bracketed by the colour and reset codes.
        """
        color = self._COLORS.get(record.levelname, self._RESET)
        return f"{color}{super().format(record)}{self._RESET}"


# INFO-level, coloured, tqdm-aware handler on the "ancestree" logger unless the
# caller has attached one. Silence with logging.getLogger("ancestree").setLevel(...).
_logger = _logging.getLogger("ancestree")
if not _logger.handlers:
    _handler = _TqdmLoggingHandler()
    _handler.setFormatter(_ColoredFormatter("%(levelname)s:%(name)s: %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.INFO)

from ancestree.settings import Settings
from ancestree.sites import (
    BaseComposition, PolymorphicSiteFilter, Site, SiteSource, SiteTable,
)
from ancestree.trees import (
    OutgroupLadderTree, RerootedTree, Tree, TskitLocalTree,
)
from ancestree.models import SubstitutionModel, JC69, K2, F81, HKY, GTR
from ancestree.likelihood import Likelihood
from ancestree.focal import FocalNode, ResolvedFocal
from ancestree.posterior import Grade, InferenceSummary, Posterior
from ancestree.priors import (
    AdaptiveIngroupWeight,
    KingmanIngroupWeight,
    IngroupWeight,
    NoIngroupWeight,
    StationaryPrior,
)
from ancestree.sources import CyVCF2Source, TskitSource, VcfZarrSource
from ancestree.inference import (
    ARGBasedInference,
    FixedTreeInference,
    Inference,
    MajorityOutgroupInference,
)
from ancestree.local_tree_inference import (
    LocalTreeBuilder,
    LocalTreeInference,
    PairwiseCoalescentHMM,
    PairwiseTmrcas,
)
from ancestree.plotting import FocalSweep, FocalTreePlot
from ancestree.readers import Annotation, Annotations, Provenance, Reader
from ancestree.writers import TskitWriter, VCFWriter, Writer, ZarrWriter

__all__ = [
    "__version__",
    "STATES",
    "STATE_INDEX",
    "MISSING_ALLELES",
    "TRANSITION_PAIRS",
    "DEFAULT_MU",
    "DEFAULT_REC_RATE",
    "Settings",
    "BaseComposition",
    "Site",
    "PolymorphicSiteFilter",
    "SiteSource",
    "SiteTable",
    "Tree",
    "TskitLocalTree",
    "OutgroupLadderTree",
    "RerootedTree",
    "SubstitutionModel",
    "JC69",
    "K2",
    "F81",
    "HKY",
    "GTR",
    "Likelihood",
    "Posterior",
    "InferenceSummary",
    "FocalNode",
    "ResolvedFocal",
    "Grade",
    "FocalTreePlot",
    "FocalSweep",
    "IngroupWeight",
    "StationaryPrior",
    "KingmanIngroupWeight",
    "AdaptiveIngroupWeight",
    "NoIngroupWeight",
    "TskitSource",
    "CyVCF2Source",
    "VcfZarrSource",
    "Inference",
    "ARGBasedInference",
    "FixedTreeInference",
    "LocalTreeBuilder",
    "LocalTreeInference",
    "PairwiseCoalescentHMM",
    "PairwiseTmrcas",
    "MajorityOutgroupInference",
    "Reader",
    "Provenance",
    "Annotation",
    "Annotations",
    "Writer",
    "VCFWriter",
    "TskitWriter",
    "ZarrWriter",
]
