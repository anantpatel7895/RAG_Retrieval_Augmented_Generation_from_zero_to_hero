"""Format-level measurement: what HTML and PDF each give you, and what they destroy.

This is the logic behind [docs/PARSING.md](../../docs/PARSING.md) and
`notebooks/02_parsing_html_vs_pdf.ipynb`. It lives here rather than in notebook
cells so it can be tested, reused by the pipeline, and re-run when the corpus
changes -- the document's numbers are then regenerable instead of frozen.

Every probe returns a plain ``dict`` so results serialise to JSON and slow scans can
be cached (see :func:`cached`).

Typical use::

    from rag.ingest.format_probe import html_probe, pdf_probe, table_recovery

    h = html_probe("data/raw/edgar_jpmorgan_FY2025_10k_008131.htm")
    p = pdf_probe("data/raw/edgar_jpmorgan_FY2025_ars_023927.pdf")
    print(f"{table_recovery(h, p):.1%} of the HTML's tables survive PDF extraction")

Examples in the docstrings below are doctests -- run them with::

    python -m pytest --doctest-modules rag/ingest/format_probe.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
ENTITY_RE = re.compile(r"&#?\w+;")

# Block tags imply a visual break, so they must become whitespace. Inline tags do
# not -- collapsing them to a space is what separates "$" from its value.
BLOCK_RE = re.compile(
    r"</?(p|div|tr|td|th|table|li|ul|ol|h[1-6]|br|section|article)\b[^>]*>", re.I
)
INLINE_RE = re.compile(r"</?(span|ix:[\w-]+|a|b|i|em|strong|font|sup|sub|u)\b[^>]*>", re.I)

#: Regex fragment matching a monetary amount, e.g. ``4.4`` or ``794,433,813,496``.
#:
#: An earlier version used ``[\d,]{4,}``, which silently excluded every amount
#: containing a decimal point -- the regex then matched a *later* number instead and
#: produced a plausible but wrong result. See ``docs/PARSING.md`` section 9.
AMOUNT = r"[\d,]+(?:\.\d+)?"


def strip_naive(html: str) -> str:
    """Flatten HTML to text by replacing **every** tag with a single space.

    This is what quick-and-dirty extractors do, and it is the baseline the other
    strategies are measured against. Its flaw: an inline tag between two pieces of
    one phrase becomes whitespace, breaking their adjacency.

    Args:
        html: Raw HTML source.

    Returns:
        Text with all tags and HTML entities collapsed to single spaces.
        Leading/trailing whitespace is *not* trimmed.

    Example -- an inline tag splits a currency symbol from its value:

        >>> strip_naive("<span>$</span><ix:nonFraction>1,234</ix:nonFraction>")
        ' $ 1,234 '

    Note that runs of whitespace collapse to a single space, so adjacent tags do not
    accumulate padding:

        >>> strip_naive("<p>Total</p><p>net income</p>")
        ' Total net income '
    """
    return WS_RE.sub(" ", TAG_RE.sub(" ", ENTITY_RE.sub(" ", html)))


def strip_inline_aware(html: str) -> str:
    """Flatten HTML to text, preserving adjacency across **inline** tags.

    Block-level tags (``<p>``, ``<td>``, ``<tr>``, ...) become a space because they
    imply a visual break. Inline tags (``<span>``, ``<ix:*>``, ``<b>``, ...) are
    removed entirely, so text they wrap stays glued to its neighbours.

    This roughly doubles recovery of ``$1,234``-style amounts versus
    :func:`strip_naive` -- and still fails on the majority, because most currency
    symbols in a 10-K sit in their own *table cell*, which is a block boundary. No
    text-level strategy can fix that; see :func:`html_probe`.

    Args:
        html: Raw HTML source.

    Returns:
        Text with block tags as spaces, inline tags removed, entities as spaces.

    Example -- inline tags no longer break the amount, but a cell boundary still does:

        >>> strip_inline_aware("<span>$</span><ix:nonFraction>1,234</ix:nonFraction>")
        '$1,234'
        >>> strip_inline_aware("<tr><td>$</td><td>1,234</td></tr>")
        ' $ 1,234 '
        >>> strip_inline_aware("a&#160;b")
        'a b'
    """
    h = BLOCK_RE.sub(" ", html)
    h = INLINE_RE.sub("", h)
    h = ENTITY_RE.sub(" ", TAG_RE.sub(" ", h))
    return re.sub(r"[ \t]+", " ", h)


def cell_texts(html: str) -> list[str]:
    """Extract the visible text of every **paired** ``<td>...</td>`` cell.

    Self-closing ``<td/>`` spacer cells are deliberately excluded -- they carry no
    text, and in a real 10-K they outnumber paired cells (37,738 vs 22,610 in
    JPMorgan's FY2025 filing). :func:`html_probe` counts them separately.

    Args:
        html: Raw HTML source.

    Returns:
        One trimmed string per paired cell, in document order. Empty cells appear as
        ``""``, so positional adjacency between cells is preserved -- which is what
        makes the "``$`` cell followed by a number cell" check possible.

    Example:

        >>> cell_texts("<tr><td> $ </td><td><span>1,234</span></td></tr>")
        ['$', '1,234']
        >>> cell_texts("<tr><td/><td>x</td><td/></tr>")
        ['x']
        >>> cell_texts("<tr><td>a</td><td></td><td>b</td></tr>")
        ['a', '', 'b']
    """
    out = []
    for inner in re.findall(r"<td[^>]*>(.*?)</td>", html, re.S):
        t = ENTITY_RE.sub(" ", TAG_RE.sub("", inner))
        out.append(WS_RE.sub(" ", t).strip())
    return out


def html_probe(path: str | Path) -> dict[str, Any]:
    """Measure structure and extraction quality for one HTML filing.

    Reads the whole file into memory (a 10-K is ~13 MB, so this is fine).

    Args:
        path: Path to an ``.htm``/``.html`` file.

    Returns:
        A JSON-serialisable dict:

        ==========================  ====================================================
        Key                         Meaning
        ==========================  ====================================================
        ``path``                    The path passed in, as a string
        ``raw_chars``               Characters in the source file
        ``text_chars``           　 Characters remaining after tag-stripping
        ``text_ratio``              ``text_chars / raw_chars`` -- fraction that is content
        ``tables``                  Count of ``<table`` opening tags
        ``td_paired``               Paired ``<td>...</td>`` cells
        ``td_self_closing``         Self-closing ``<td/>`` spacer cells
        ``xbrl_tags``               Count of ``<ix:`` inline-XBRL tags
        ``dollar_chars``            Raw ``$`` characters in the source
        ``naive_attached``          Amounts where ``$`` stayed glued, naive strip
        ``naive_split``             Amounts where ``$`` got separated, naive strip
        ``aware_attached``          Same, inline-aware strip
        ``aware_split``             Same, inline-aware strip
        ``dollar_only_cells``       Cells whose entire text is ``"$"``
        ``dollar_cell_then_number`` Of those, how many are followed by a number-only cell
        ``number_only_cells``       Cells whose entire text is a bare amount
        ==========================  ====================================================

    Real output for JPMorgan's FY2025 10-K::

        {'path': 'data/raw/edgar_jpmorgan_FY2025_10k_008131.htm',
         'raw_chars': 12927325,
         'text_chars': 1513182,
         'text_ratio': 0.11705,          # only ~12% of the file is content
         'tables': 679,
         'td_paired': 22610,
         'td_self_closing': 37738,       # 62% of cells are print spacers
         'xbrl_tags': 8738,
         'dollar_chars': 3725,
         'naive_attached': 328,          # 10% survive naive extraction
         'naive_split': 3004,
         'aware_attached': 780,          # 23% survive inline-aware extraction
         'aware_split': 2552,
         'dollar_only_cells': 2922,
         'dollar_cell_then_number': 2552,   # <-- equals aware_split exactly
         'number_only_cells': 10579}

    The last two numbers are the point of this function. ``aware_split`` (2,552)
    matching ``dollar_cell_then_number`` (2,552) *exactly* proves the mechanism:
    every amount the inline-aware strategy fails to recover is one whose ``$`` lives
    in its own table cell. That is a structural separation, so grid-aware table
    parsing is required -- no text heuristic can close the gap.
    """
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    cells = cell_texts(raw)

    naive, aware = strip_naive(raw), strip_inline_aware(raw)

    def counts(text: str) -> tuple[int, int]:
        attached = len(re.findall(rf"\${AMOUNT}", text))
        split = len(re.findall(rf"\$\s+{AMOUNT}", text))
        return attached, split

    n_att, n_split = counts(naive)
    a_att, a_split = counts(aware)

    dollar_only = [i for i, t in enumerate(cells) if t == "$"]
    followed = sum(
        1 for i in dollar_only if i + 1 < len(cells) and re.fullmatch(AMOUNT, cells[i + 1] or "")
    )

    text_chars = len(WS_RE.sub(" ", TAG_RE.sub(" ", raw)))
    return {
        "path": str(path),
        "raw_chars": len(raw),
        "text_chars": text_chars,
        "text_ratio": text_chars / max(len(raw), 1),
        "tables": raw.lower().count("<table"),
        "td_paired": len(cells),
        "td_self_closing": len(re.findall(r"<td[^>]*/>", raw)),
        "xbrl_tags": raw.count("<ix:"),
        "dollar_chars": raw.count("$"),
        "naive_attached": n_att,
        "naive_split": n_split,
        "aware_attached": a_att,
        "aware_split": a_split,
        "dollar_only_cells": len(dollar_only),
        "dollar_cell_then_number": followed,
        "number_only_cells": sum(1 for t in cells if re.fullmatch(AMOUNT, t or "")),
    }


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #

#: Typographic ligatures that must be normalised before chunking, or they break
#: token matching (``ﬁnancial`` will not match a query for ``financial``).
LIGATURES = "ﬁﬂﬀﬃﬄ"


def pdf_probe(path: str | Path, table_sample: int = 25) -> dict[str, Any]:
    """Measure page-level extraction quality for one PDF.

    Requires ``pymupdf`` (imported lazily so importing this module stays cheap).

    ``empty_pages`` is the load-bearing number: a page with no text layer extracts
    as an empty string and raises **no error**, so a scanned document silently
    contributes nothing to the index. Citizens Financial's FY2024 annual report has
    167 of 194 pages empty -- 86% invisible, with no warning anywhere.

    Table detection is **sampled, not exhaustive** -- a full scan of a 350-page
    report takes minutes. ``pages_sampled`` and ``tables_sampled`` are both returned
    alongside ``tables_estimated`` so the extrapolation is visible rather than
    hidden inside one number.

    Args:
        path: Path to a ``.pdf`` file.
        table_sample: Approximate number of evenly-spaced pages to run table
            detection on. Higher is slower and more accurate. ``25`` is the default
            used for ``docs/PARSING.md``; ``12`` roughly halves the runtime.

    Returns:
        A JSON-serialisable dict:

        ====================== ==========================================================
        Key                    Meaning
        ====================== ==========================================================
        ``path``               The path passed in, as a string
        ``pages``              Total page count
        ``empty_pages``        Pages whose extracted text is blank (scanned/image-only)
        ``text_chars``         Characters extracted across all pages
        ``est_tokens``         ``text_chars // 4`` -- crude, for planning not billing
        ``ligatures``          Occurrences of ``ﬁ ﬂ ﬀ ﬃ ﬄ``; needs normalisation
        ``hyphen_breaks``      ``word-\\nword`` line-break hyphenations to rejoin
        ``images_first40``     Embedded images in the first 40 pages; high + empty
                               pages together indicate a scan
        ``pages_sampled``      Pages actually scanned for tables
        ``tables_sampled``     Tables found on those pages
        ``tables_estimated``   Extrapolated whole-document table count
        ====================== ==========================================================

    Real output for Citigroup's FY2025 annual report at ``table_sample=12``::

        {'path': 'data/raw/edgar_citigroup_FY2025_ars_000187.pdf',
         'pages': 334,
         'empty_pages': 1,
         'text_chars': 1230894,
         'est_tokens': 307723,
         'ligatures': 0,
         'hyphen_breaks': 2,
         'images_first40': 58,
         'pages_sampled': 13,
         'tables_sampled': 2,
         'tables_estimated': 51}

    Note how sampling density changes the estimate: the same document at
    ``table_sample=25`` yields ``tables_estimated`` around 190. **Do not compare
    estimates produced at different ``table_sample`` values** -- fix it once per
    study, which is why the notebook pins it to 25.
    """
    import pymupdf

    doc = pymupdf.open(str(path))
    pages = [p.get_text() for p in doc]
    text = "".join(pages)

    step = max(1, doc.page_count // max(table_sample, 1))
    idx = list(range(0, doc.page_count, step))
    tables_sampled = sum(len(doc[i].find_tables().tables) for i in idx)

    return {
        "path": str(path),
        "pages": doc.page_count,
        "empty_pages": sum(1 for p in pages if not p.strip()),
        "text_chars": len(text),
        "est_tokens": len(text) // 4,
        "ligatures": sum(text.count(c) for c in LIGATURES),
        "hyphen_breaks": len(re.findall(r"[a-z]-\n[a-z]", text)),
        "images_first40": sum(len(doc[i].get_images()) for i in range(min(doc.page_count, 40))),
        "pages_sampled": len(idx),
        "tables_sampled": tables_sampled,
        "tables_estimated": int(tables_sampled * doc.page_count / max(len(idx), 1)),
    }


def table_recovery(html: dict[str, Any], pdf: dict[str, Any]) -> float:
    """Fraction of the HTML's declared tables that PDF extraction finds.

    The HTML is ground truth because both documents report the *same fiscal year for
    the same company* -- the 10-K as inline XBRL, the annual report as a PDF. That
    pairing is what makes a parser bake-off scoreable without hand-labelling a
    single page (``docs/PARSING.md`` section 7).

    Args:
        html: Output of :func:`html_probe` -- only ``tables`` is read.
        pdf: Output of :func:`pdf_probe` -- only ``tables_estimated`` is read.

    Returns:
        Ratio in ``[0, ~1]``. Returns ``0.0`` when the HTML declares no tables,
        rather than raising -- a missing denominator means "no signal", not an error.
        Values can exceed 1.0 in principle if extrapolation overshoots.

    Example:

        >>> table_recovery({"tables": 400}, {"tables_estimated": 100})
        0.25
        >>> table_recovery({"tables": 0}, {"tables_estimated": 12})
        0.0

    Measured across 55 pairs: median **14.4%**, range **0% to 85.4%**. That spread --
    same parser, same document type, same industry -- is why Phase 2 must route
    parsers per document on a measured signal rather than pick one globally.
    """
    h = int(html["tables"] or 0)
    return (int(pdf["tables_estimated"] or 0) / h) if h else 0.0


# --------------------------------------------------------------------------- #
# Caching -- slow scans should not gate a notebook re-run
# --------------------------------------------------------------------------- #


def cached(path: str | Path, build: Callable[[], Any]) -> Any:
    """Read JSON from ``path``, computing and writing it via ``build()`` if absent.

    Exists so a notebook stays re-runnable. The 55-pair scan behind
    ``docs/PARSING.md`` takes several minutes; without a cache, every re-run of the
    notebook would pay that cost and people would stop re-running it -- which is how
    notebooks end up carrying stale outputs.

    There is **no invalidation**: the cache is keyed only on the filename. Delete the
    file to force a recompute. That is deliberate -- automatic invalidation would
    need to hash every input document, and an explicit delete is easier to reason
    about than a cache that sometimes decides to refresh itself.

    Args:
        path: Where to store the JSON. Parent directories are created. Use
            ``data/interim/`` -- it is gitignored, since this is regenerable.
        build: Zero-argument callable returning any JSON-serialisable value. Called
            only on a cache miss.

    Returns:
        The parsed JSON on a hit, or whatever ``build()`` returned on a miss.

    Example:

        >>> import tempfile, os
        >>> d = tempfile.mkdtemp()
        >>> p = os.path.join(d, "demo.json")
        >>> calls = []
        >>> def build():
        ...     calls.append(1)
        ...     return {"answer": 42}
        >>> cached(p, build)
        {'answer': 42}
        >>> cached(p, build)          # second call reads the file
        {'answer': 42}
        >>> len(calls)                # build() ran only once
        1
    """
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    result = build()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1))
    return result
