"""Import of the optional cyvcf2 backend, which has no Windows build."""
from __future__ import annotations

import sys
from types import ModuleType

#: The error message for VCF input or output without cyvcf2.
_MISSING = (
    "VCF support in ancestree requires the optional cyvcf2 package, which is not available on Windows. "
    "Use WSL, or convert the VCF to a VCF-Zarr store on Linux or macOS and read the store instead."
    if sys.platform == "win32" else
    "VCF support in ancestree requires the optional cyvcf2 package. "
    "Install it with `pip install \"ancestree-popgen[vcf]\"` or `conda install -c bioconda cyvcf2`."
)


def import_cyvcf2() -> ModuleType:
    """
    Import cyvcf2.

    :return: The cyvcf2 module.
    :raises ImportError: If cyvcf2 is not installed.
    """
    try:
        import cyvcf2
    except ImportError as e:
        raise ImportError(_MISSING) from e

    return cyvcf2
