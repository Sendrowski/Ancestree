"""Write the outputs an executed docs notebook displays to ``docs/outputs``.

The docs notebooks are executed in place, so a changed result shows up only in
the notebook JSON, which a git viewer does not render. This writes every output
``docs/reference/<Language>/<page>.ipynb`` displays to
``docs/outputs/<page>/<NN>-<section>/<language>-<n>.<ext>``, where it does.

Sections are numbered in page order from ``00``, the part under the page title,
and every subheading takes a number whether or not it has outputs. Within a
section the files are numbered in display order: figures as SVG, or PNG without
an SVG rendering, tables as HTML and all other output as text. Cells tagged
``remove-cell`` or ``remove-output`` display nothing. Values that differ between
executions of an unchanged notebook are replaced by fixed values: timestamps,
progress-bar timings and rates, the salted hash ids of matplotlib SVGs and the
paths of temporary files. ANSI colour codes and trailing spaces are stripped.

The ``extract_docs_outputs`` Snakemake rule runs this on one notebook once
``execute_docs_notebooks`` has executed it. Run directly::

    python docs/extract_outputs.py [<notebook> ...]

it does the same for the given notebooks, or for every notebook under
``docs/reference/Python`` when none is given.
"""
import base64
import json
import re
import sys
import tempfile
from pathlib import Path

NOTEBOOKS = sorted((Path(__file__).resolve().parent / "reference" / "Python").glob("*.ipynb"))

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

_FENCED_CODE = re.compile(r"^(`{3,}|~{3,}).*?^\1", re.MULTILINE | re.DOTALL)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: An ISO 8601 date and time, as provenance records and SVG metadata carry.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[+-]\d{2}:\d{2}|Z)?")

#: Elapsed time of a progress bar, and the remaining time where the total is known.
_PROGRESS_TIME = re.compile(r"(?<=\[)\d+(?::\d+)+(?:<(?:\d+(?::\d+)+|\?))?(?=,)")

#: Rate of a progress bar with the spaces it is padded with, as units per second or seconds per unit.
_PROGRESS_RATE = re.compile(r"(?<=,)[ \t]*[\d.?]+(?=[ \t]*[A-Za-z]+/s\]|s/[ \t]*[A-Za-z]+\])")

#: Temporary directory of the machine that executes the notebooks, and names of temporary files created in it.
_TEMP_DIR = re.compile(re.escape(tempfile.gettempdir()))
_TEMP_FILE = re.compile(r"(?<=<tmp>/)tmp[a-z0-9_]{8}")

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)

#: Id of a matplotlib SVG element: its type prefix and a content hash salted afresh on every rendering.
_SVG_ID = re.compile(r"(?<=[\"#])(C[0-9a-f]+_[0-9a-f]+_|[hmp]|image)[0-9a-f]{10}(?=[\")])")


def _tags(cell: dict) -> list:
    """Cell tags.

    :param cell: Notebook cell.
    :return: The tags in the cell metadata.
    """
    return cell.get("metadata", {}).get("tags", [])


def _displayed_outputs(cell: dict) -> list[dict]:
    """Outputs the docs display for a code cell, in display order.

    :param cell: Notebook code cell.
    :return: The data of each displayed output: a non-empty stream as plain
        text, a table as HTML, another value as plain text, and a figure with
        all its renderings.
    """
    if "remove-cell" in _tags(cell) or "remove-output" in _tags(cell):
        return []

    displayed = []

    for output in cell.get("outputs", []):
        if output["output_type"] == "stream":
            text = "".join(output["text"])
            if text.strip():
                displayed.append({"text/plain": text})

        elif output["output_type"] in ("display_data", "execute_result"):
            data = output["data"]
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                html = "".join(data.get("text/html", ""))
                data = {"text/html": html} if "<table" in html else {"text/plain": data["text/plain"]}
            displayed.append(data)

    return displayed


def _slug(title: str) -> str:
    """Directory name of a section heading.

    :param title: Markdown heading text.
    :return: Its words in lower case joined by hyphens, without MyST roles and literals.
    """
    title = re.sub(r"\{[^}]*\}`([^`]*)`", r"\1", title)

    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _output_file(data: dict) -> tuple[str, bytes]:
    """File extension and normalised content of a displayed output.

    :param data: Output data by MIME type.
    :return: The extension and content: the SVG of a figure, the PNG of a figure
        without an SVG rendering, the HTML of a table, or the text.
    """
    if "image/svg+xml" in data:
        ids: dict[str, str] = {}
        svg = _SVG_ID.sub(
            lambda m: m[1] + ids.setdefault(m[0], f"{len(ids):010x}"), "".join(data["image/svg+xml"])
        )
        return "svg", _TIMESTAMP.sub("1970-01-01T00:00:00", svg).encode()

    if "image/png" in data:
        return "png", base64.b64decode("".join(data["image/png"]))

    if "text/html" in data:
        return "html", "".join(data["text/html"]).encode()

    text = _TRAILING_SPACE.sub("", _ANSI.sub("", "".join(data["text/plain"])))
    text = _TEMP_FILE.sub("tmp--------", _TEMP_DIR.sub("<tmp>", text))
    text = _PROGRESS_TIME.sub(lambda m: re.sub(r"[\d?]+", "--", m[0]), text)
    text = _PROGRESS_RATE.sub(" --", text)

    text = _TIMESTAMP.sub("1970-01-01T00:00:00", text)

    # a stream ends with a line break or not depending on when it was flushed
    return "txt", (text.rstrip("\n") + "\n").encode()


def _sections(nb: dict, page: str) -> list[tuple[str, list[dict]]]:
    """Split the displayed outputs of a notebook by section.

    :param nb: Parsed notebook JSON.
    :param page: Page name, the title of section ``00`` when the notebook has no title heading.
    :return: The directory name and the displayed output data of each section, in page order.
    """
    titles, outputs = [page], [[]]

    for cell in nb["cells"]:
        if "remove-cell" in _tags(cell):
            continue

        if cell["cell_type"] == "markdown":
            for level, title in _HEADING.findall(_FENCED_CODE.sub("", "".join(cell["source"]))):
                if len(level) == 1 and len(titles) == 1:
                    titles[0] = title
                else:
                    titles.append(title)
                    outputs.append([])

        elif cell["cell_type"] == "code":
            outputs[-1] += _displayed_outputs(cell)

    return [(f"{number:02d}-{_slug(title)}", data) for number, (title, data) in enumerate(zip(titles, outputs))]


def extract(notebook: Path) -> None:
    """Write the displayed outputs of an executed notebook, replacing the files of its language.

    :param notebook: Notebook under ``docs/reference/<Language>/``, whose outputs
        go to ``docs/outputs/<page>/`` as ``<language>-<n>`` files.
    """
    language = notebook.parent.name.lower()
    directory = notebook.resolve().parents[2] / "outputs" / notebook.stem
    page_sections = _sections(json.loads(notebook.read_text()), notebook.stem)

    for path in directory.glob(f"*/{language}-*"):
        path.unlink()

    for name, outputs in page_sections:
        for n, data in enumerate(outputs, start=1):
            extension, content = _output_file(data)
            path = directory / name / f"{language}-{n}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    # directories of sections absent from the page that hold no file of any language
    current = {name for name, _ in page_sections}
    for section in directory.glob("*/"):
        if section.name not in current and not any(section.iterdir()):
            section.rmdir()


if __name__ == "__main__":
    try:
        notebooks = [Path(snakemake.input.notebook)]
    except NameError:
        notebooks = [Path(arg) for arg in sys.argv[1:]] or NOTEBOOKS

    for notebook in notebooks:
        extract(notebook)
