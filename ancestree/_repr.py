"""Shared ``__repr__`` and logger for the package's classes.

:class:`~ancestree._repr.ReprMixin` renders ``ClassName(field=value, ...)`` from the fields a
class declares in :attr:`ReprMixin._repr_params <ancestree._repr.ReprMixin._repr_params>`,
so every class prints its identifying configuration. It also carries
:attr:`ReprMixin._log <ancestree._repr.ReprMixin._log>`, the per-class logger
every subclass emits its diagnostics on.
"""
from __future__ import annotations

import logging

import numpy as np


def _format(value: object) -> str:
    """Compact display form for one repr field.

    :param value: The field value.
    :return: Its rendered form.
    """
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, np.ndarray):
        return "[" + ", ".join(f"{float(x):.4g}" for x in value) + "]"
    return repr(value)


class ReprMixin:
    """Renders ``ClassName(field=value, ...)`` from :attr:`_repr_params`."""

    @property
    def _log(self) -> logging.Logger:
        """Per-class logger, so records render as ``{LEVEL}:ancestree.{ClassName}: ...``.

        :return: The logger named after this instance's class.
        """
        return logging.getLogger(f"ancestree.{type(self).__name__}")

    @property
    def _repr_params(self) -> dict[str, object]:
        """Fields identifying this instance, in display order.

        :return: Mapping from field name to value. Empty by default, which
            renders as ``ClassName()``.
        """
        return {}

    def __repr__(self) -> str:
        """Render the class name and its :attr:`_repr_params`.

        :return: ``ClassName(field=value, ...)``.
        """
        fields = ", ".join(f"{k}={_format(v)}" for k, v in self._repr_params.items())
        return f"{type(self).__name__}({fields})"
