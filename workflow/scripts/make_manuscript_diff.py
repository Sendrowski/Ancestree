"""latexdiff the manuscript against a baseline, holding the bibliography out.

``main.tex`` carries its ``thebibliography`` inline. latexdiff marks up the
tokens inside it, and the markup lands between ``\\csname`` and ``\\endcsname``
in natbib's ``\\providecommand`` preamble, which is not valid TeX and halts the
engine. The bibliography is not something a revision diff should mark up, so it
is replaced by a sentinel in both inputs and the new one is restored afterwards.
"""
import re
import subprocess
import sys
from pathlib import Path

#: Marks where the bibliography was lifted out. A comment, so latexdiff carries
#: it through untouched whichever side it appears on.
SENTINEL = "%DIFBIBLIOGRAPHYPLACEHOLDER"

BIB = re.compile(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
                 re.DOTALL)


class ManuscriptDiff:
    """A marked-up copy of one manuscript against another.

    :param old: Baseline ``main.tex``.
    :param new: Current ``main.tex``.
    :param out: Where to write the marked-up document.
    """

    def __init__(self, old: Path, new: Path, out: Path) -> None:
        self.old = Path(old)
        self.new = Path(new)
        self.out = Path(out)

    def _strip_bibliography(self, path: Path) -> "tuple[Path, str | None]":
        """Write a copy of ``path`` with its bibliography replaced by the sentinel.

        The copy sits beside the original so that latexdiff's ``--flatten``
        resolves the ``sections/`` includes relative to the right directory.

        :param path: The document to copy.
        :return: ``(temporary path, the bibliography that was removed)``.
        """
        text = path.read_text()
        found = BIB.search(text)
        tmp = path.with_name(f"_nobib_{path.name}")
        tmp.write_text(BIB.sub(SENTINEL, text) if found else text)
        return tmp, found.group(0) if found else None

    def run(self) -> None:
        """Produce the marked-up document, bibliography restored."""
        old_tmp, _ = self._strip_bibliography(self.old)
        new_tmp, new_bib = self._strip_bibliography(self.new)
        try:
            diff = subprocess.run(
                ["latexdiff", "--flatten", "--graphics-markup=none",
                 str(old_tmp), str(new_tmp)],
                capture_output=True, text=True, check=True,
            ).stdout
        finally:
            old_tmp.unlink(missing_ok=True)
            new_tmp.unlink(missing_ok=True)
        if new_bib is not None:
            if SENTINEL not in diff:
                raise RuntimeError(
                    "the bibliography sentinel did not survive latexdiff, so "
                    "the marked-up document would carry no bibliography")
            diff = diff.replace(SENTINEL, new_bib)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.out.write_text(diff)


try:
    old_path = snakemake.input.old            # noqa: F821
    new_path = snakemake.input.new            # noqa: F821
    out_path = snakemake.output[0]            # noqa: F821
except NameError:
    old_path = "reports/manuscripts/latex_baseline/main.tex"
    new_path = "reports/manuscripts/latex/main.tex"
    out_path = "reports/manuscripts/latex_diff/main.tex"

if __name__ == "__main__" or "snakemake" in dir():
    ManuscriptDiff(Path(old_path), Path(new_path), Path(out_path)).run()
    print(f"wrote {out_path}", file=sys.stderr)
