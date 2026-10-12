"""Phase 0 chunker: fixed-size windows. **Deliberately naive**, like the parser.

Splits on whitespace into fixed token-ish windows with overlap. It respects nothing:
not sentences, not paragraphs, not headings, not tables. A table that spans a window
boundary is cut in half; a row header ends up in one chunk and its value in the next.

**Why ship this.** Phase 3's gate is "the best chunker beats fixed-512 by >= 10 points
recall@20." That comparison needs a fixed-size baseline to exist and stay runnable.
See ``rag/ingest/parse.py`` for the same argument at more length.

Phase 3 replaces this with structure-aware, semantic, contextual and small-to-big
strategies -- and measures each against this one.

**On units -- measured, not assumed.** This chunker splits on **whitespace**, so
``size`` counts *words*, not tokens. On our corpus the ratio is:

    JPMorgan FY2025:  189,994 words  ->  378,295 est. tokens   =  1.99 tokens/word

Roughly **2 tokens per word** -- far above the ~1.3 typical of prose, because
financial text is dense with long numbers (``794,433,813,496``) and multi-syllable
terms that a real tokenizer splits several ways.

So ``DEFAULT_SIZE = 200`` words targets ~400 tokens. An earlier version used 400
words while *calling* it 400 tokens, which made every downstream chunk-count and
cost figure 2x wrong -- `docs/LESSONS.md` L7, committed in the same docstring that
warns about it. Phase 3 replaces this with a real tokenizer and the guessing stops.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from rag.schema import Chunk, DocMeta

#: Words per chunk. ~200 words x 1.99 tokens/word ~= 400 tokens on this corpus.
DEFAULT_SIZE = 200
#: 15% overlap. Overlap is not a substitute for structure -- see Phase 3.
DEFAULT_OVERLAP = 30


def chunk_text(
    text: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[tuple[int, str]]:
    """Split text into fixed whitespace-token windows.

    Args:
        text: Plain text from `rag.ingest.parse`.
        size: Words per window.
        overlap: Words repeated between consecutive windows. Must be < ``size``.

    Yields:
        ``(index, chunk_text)`` pairs, index starting at 0.

    Raises:
        ValueError: If ``overlap >= size``, which would loop forever.

    Example:

        >>> list(chunk_text("a b c d e", size=3, overlap=1))
        [(0, 'a b c'), (1, 'c d e')]
        >>> list(chunk_text("", size=3, overlap=1))
        []
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be < size ({size})")

    words = text.split()
    if not words:
        return

    step = size - overlap
    idx = 0
    for start in range(0, len(words), step):
        window = words[start : start + size]
        if not window:
            break
        yield idx, " ".join(window)
        idx += 1
        if start + size >= len(words):
            break


def chunk_document(
    doc: DocMeta,
    text: str,
    *,
    embed_model_version: str,
    ingest_run_id: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Turn one parsed document into `Chunk` objects carrying the filter metadata.

    Chunk IDs are **deterministic** -- ``sha256(doc_id:index)`` truncated -- so
    re-ingesting an unchanged document upserts over the same points instead of
    duplicating them. Idempotent re-ingest is the same property the fetcher's
    content hashing provides one layer up.

    Args:
        doc: Manifest row for the source document.
        text: Its parsed plain text.
        embed_model_version: Stamped on every chunk; Phase 11's blue/green reindex
            keys off this to know which chunks need re-embedding.
        ingest_run_id: Which ingest produced these chunks.
        size: Words per chunk.
        overlap: Words repeated between chunks.

    Returns:
        A list of `Chunk`. ``page``, ``bbox``, ``section_path`` and ``parent_id``
        are left empty -- the Phase 0 parser cannot populate them, and Phases 2-3
        exist to fix that.
    """
    out: list[Chunk] = []
    for idx, body in chunk_text(text, size=size, overlap=overlap):
        cid = hashlib.sha256(f"{doc.doc_id}:{idx}".encode()).hexdigest()[:32]
        out.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                text=body,
                doc_type=doc.doc_type,
                effective_date=doc.effective_date,
                version=doc.version,
                source_url=doc.source_url,
                sensitivity=doc.sensitivity,
                tenant_id=doc.tenant_id,
                checksum=doc.checksum,
                embed_model_version=embed_model_version,
                ingest_run_id=ingest_run_id,
            )
        )
    return out
