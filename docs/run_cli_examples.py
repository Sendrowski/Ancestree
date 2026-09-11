"""Run the CLI examples and capture their output for the docs.

Each invocation runs against the demo panel in ``docs/_static`` and its
combined stdout/stderr is written to
``docs/reference/CLI/_output/<name>.html`` as a colourised rendering, which
``usage.rst`` embeds with ``.. raw:: html``. Regenerate with::

    python docs/run_cli_examples.py

The ``cli_transcripts`` Snakemake rule runs it as part of the docs refresh.

ANSI colour codes are translated into inline CSS spans and tqdm
carriage-return frames collapsed to their final state, so a progress bar lands
as one line whether or not it carries a total. Commands run in a temporary
directory and are rewritten to their display form, so the captured text carries
no absolute paths.
"""
import html
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE / "_static"
OUT = HERE / "reference" / "CLI" / "_output"

EXAMPLES = {
    "arg": [
        "arg",
        "--trees", str(STATIC / "demo.trees"),
        "--mu", "1.25e-8",
        "--out", "annotated.trees",
    ],
    "local_tree": [
        "local-tree",
        "--vcf", str(STATIC / "demo.vcf.gz"),
        "--mu", "1.25e-8",
        "--rec-rate", "1e-8",
        "--sequence-length", "200000",
        "--out", "annotated.vcf.gz",
    ],
    "fixed_tree": [
        "fixed-tree",
        "--vcf", str(STATIC / "demo.vcf.gz"),
        "--outgroups", "o1_h0,o2_h0,o3_h0",
        "--model", "K2", "--fit-kappa",
        "--ingroup-weight", "kingman",
        "--n-target-sites", "200000",
        "--out", "annotated.vcf.gz",
    ],
}

_ANSI = re.compile(r"\x1b\[([0-9;]*)m")

#: SGR codes the package's log formatter emits, mapped to CSS colours that
#: stay legible on both the light and dark docs themes.
_SGR_CSS = {
    "31": "#d13438",  # red
    "32": "#2a9d3f",  # green
    "33": "#b8860b",  # yellow
    "36": "#0a7ea4",  # cyan
}


#: A tqdm frame, in both its determinate form (a percentage and the bar)
#: and its total-less form (a count, a unit and the elapsed / rate bracket).
_BAR = re.compile(r"\d+%\||\b\d+ \S+ \[\d{2}:\d{2}[<,]")


def _paths(text: str) -> str:
    """Collapse tqdm frames and rewrite absolute paths, keeping ANSI codes.

    :param text: Raw combined stdout/stderr.
    :return: The same text with stable paths and one frame per bar.
    """
    lines = [
        line.split("\r")[-1] if "\r" in line else line for line in text.split("\n")
    ]
    # The tqdm-aware log handler writes each frame on its own line, so keep
    # only the last of every consecutive run of them.
    kept: list[str] = []
    for line in lines:
        if _BAR.search(line) and kept and _BAR.search(kept[-1]):
            kept[-1] = line
        else:
            kept.append(line)
    # tqdm opens its bar with a newline, leaving a gap after the log line
    # that precedes it; drop blank lines that sit directly before a bar.
    trimmed: list[str] = []
    for line in kept:
        if not line.strip() and trimmed and not trimmed[-1].strip():
            continue
        trimmed.append(line)
    out: list[str] = []
    for i, line in enumerate(trimmed):
        nxt = trimmed[i + 1] if i + 1 < len(trimmed) else ""
        if not line.strip() and _BAR.search(nxt):
            continue
        out.append(line)
    return "\n".join(out).replace(str(STATIC) + "/", "")


def _to_html(text: str) -> str:
    """Render ANSI-coloured text as an HTML ``<pre>`` block.

    :param text: Cleaned text that still carries its SGR escapes.
    :return: A self-contained ``<pre>`` fragment.
    """
    out, open_span = [], False
    pos = 0
    for m in _ANSI.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        code = m.group(1)
        if open_span:
            out.append("</span>")
            open_span = False
        css = _SGR_CSS.get(code)
        if css:
            out.append(f'<span style="color: {css}">')
            open_span = True
        pos = m.end()
    out.append(html.escape(text[pos:]))
    if open_span:
        out.append("</span>")
    body = "".join(out).strip()
    return f'<pre class="cli-output">{body}</pre>\n'


def main() -> int:
    """Run every example and write its captured output.

    :return: Process exit status; non-zero if any invocation failed.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    status = 0
    for name, args in EXAMPLES.items():
        with tempfile.TemporaryDirectory() as work:
            proc = subprocess.run(
                [sys.executable, "-m", "ancestree.cli", *args],
                cwd=work, capture_output=True, text=True,
                # The Intel OpenMP runtime otherwise prints an ``OMP: Info``
                # deprecation line into the transcript.
                env={**os.environ, "KMP_WARNINGS": "0"},
            )
        raw = proc.stdout + proc.stderr
        (OUT / f"{name}.html").write_text(_to_html(_paths(raw)))
        print(f"{name}: {'ok' if proc.returncode == 0 else 'FAILED'}")
        status |= proc.returncode
    return status


if __name__ == "__main__":
    raise SystemExit(main())
