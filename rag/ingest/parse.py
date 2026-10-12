"""Phase 0 parser: **deliberately naive**, and that is the point.

This module extracts plain text and nothing else. It throws away tables, headings,
page boundaries, reading order, and every structural signal that
[docs/PARSING.md](../../docs/PARSING.md) shows to matter. On HTML it reuses
``strip_naive`` -- the function whose measured behaviour is that it recovers **10%**
of dollar amounts, separating the rest from their currency symbol.

**Why ship something this bad on purpose.**

Phase 2 spends 16 hours on parsing, and its gate is "``table_lookup`` recall@20
improves >= 20 points." Improves against *what*? A good parser here would leave Phase 2
with nothing to beat, and the 20-point claim would be unmeasurable. This is the
control group -- rule 2 in ``PLAN.md``: one corpus, all the way through, so every
later improvement is a number rather than a vibe.

So the schema fields ``page``, ``bbox``, ``section_path`` and ``parent_id`` stay
**empty** after Phase 0. That is expected, not a defect. Phase 2 and Phase 3 fill
them, and the delta is what those phases are worth.

Do not improve this file. Replace it in Phase 2, and keep this one runnable so the
comparison stays honest.
"""

from __future__ import annotations

from pathlib import Path

from rag.ingest.format_probe import strip_naive


class UnsupportedFormat(ValueError):
    pass


def parse_html(path: str | Path) -> str:
    """Extract text from an HTML filing by stripping every tag.

    Args:
        path: Path to an ``.htm``/``.html`` file.

    Returns:
        Plain text with all tags and entities collapsed to single spaces.

    Example:

        >>> import tempfile, os
        >>> p = os.path.join(tempfile.mkdtemp(), "x.htm")
        >>> _ = open(p, "w").write("<p>Total</p><td>$</td><td>1,234</td>")
        >>> parse_html(p).strip()
        'Total $ 1,234'

    Note the ``$`` separated from ``1,234`` -- they are in different table cells.
    That is the documented failure this parser is meant to exhibit.
    """
    return strip_naive(Path(path).read_text(encoding="utf-8", errors="ignore"))


def parse_pdf(path: str | Path) -> str:
    """Extract text from a PDF by concatenating each page's text layer.

    Pages with no text layer contribute an empty string **silently** -- Citizens
    Financial's FY2024 annual report is 86% such pages. Phase 2 adds the text-yield
    check and OCR routing; this function deliberately does not.

    Args:
        path: Path to a ``.pdf`` file.

    Returns:
        Concatenated page text, pages separated by a newline.
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def parse(path: str | Path) -> str:
    """Dispatch to the right parser on file extension.

    Args:
        path: Path to an ``.htm``, ``.html``, ``.pdf`` or ``.txt`` file.

    Returns:
        Plain text.

    Raises:
        UnsupportedFormat: For any other extension. Deliberately loud -- a corpus
            silently skipping a file type is exactly the failure mode
            ``docs/LESSONS.md`` L1 is about.
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".htm", ".html"}:
        return parse_html(path)
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    raise UnsupportedFormat(f"no Phase 0 parser for {suffix!r} ({path})")
