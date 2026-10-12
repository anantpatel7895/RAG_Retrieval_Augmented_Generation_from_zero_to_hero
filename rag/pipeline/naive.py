"""The Phase 0 pipeline: parse -> chunk -> embed -> index -> search.

**This is the permanent control group.** Every later phase is measured against it, so
it is frozen after Phase 1 records the baseline. Changes go in new modules; this one
stays runnable so the comparison stays honest (``PLAN.md`` rule 2).

Each stage is the dumbest thing that works:

===============  ==========================================  ===================
Stage            Implementation                              Replaced in
===============  ==========================================  ===================
parse            regex tag-strip / PDF text layer            Phase 2
chunk            fixed 200-word windows, 15% overlap         Phase 3
                 (~400 tokens: this corpus runs 1.99 tok/word)
embed            whatever ``RAG_EMBED_PROVIDER`` names       Phase 4
retrieve         dense cosine top-k, no rerank, no hybrid    Phases 5-6
===============  ==========================================  ===================

Deliberately absent: table handling, structure awareness, sparse/hybrid retrieval,
reranking, query rewriting. Those are later phases, and each one needs this baseline
to exist in order to prove it was worth building.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag.chunk.fixed import DEFAULT_OVERLAP, DEFAULT_SIZE, chunk_document
from rag.config import Settings, settings
from rag.embed.provider import get_provider, verify_contract
from rag.index.qdrant_store import QdrantStore
from rag.ingest.parse import UnsupportedFormat, parse
from rag.schema import Chunk, DocMeta


def load_manifest(cfg: Settings | None = None) -> list[DocMeta]:
    cfg = cfg or settings
    path = cfg.raw_dir / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path} — run `rag corpus fetch`")
    return [DocMeta(**json.loads(ln)) for ln in path.read_text().splitlines() if ln.strip()]


def iter_chunks(
    docs: list[DocMeta],
    *,
    embed_model_version: str,
    run_id: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Iterator[tuple[DocMeta, list[Chunk], str | None]]:
    """Parse and chunk each document.

    Yields:
        ``(doc, chunks, error)``. On a parse failure ``chunks`` is empty and
        ``error`` holds the message -- a document that fails must be **visible**,
        not silently absent (``docs/LESSONS.md`` L1).
    """
    for doc in docs:
        try:
            text = parse(doc.path)
        except (UnsupportedFormat, OSError, Exception) as e:  # noqa: BLE001
            yield doc, [], f"{type(e).__name__}: {e}"
            continue
        chunks = chunk_document(
            doc,
            text,
            embed_model_version=embed_model_version,
            ingest_run_id=run_id,
            size=size,
            overlap=overlap,
        )
        yield doc, chunks, None


def ingest(
    limit: int | None = None,
    *,
    batch: int = 256,
    recreate: bool = False,
    cfg: Settings | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    """Run the full pipeline over the corpus and report what happened.

    Args:
        limit: Only the first N documents. ``None`` means the whole corpus.
        batch: Chunks embedded and upserted per call.
        recreate: Drop and rebuild the collection first.
        cfg: Settings override.
        progress: Optional callable ``(message)`` for per-document output.

    Returns:
        Counts and timings, plus a ``failures`` list. **Read the failures** -- an
        ingest that reports 118 of 120 documents is not a success.
    """
    cfg = cfg or settings
    run_id = f"ingest-{int(time.time())}"

    provider = get_provider("ingest", cfg)
    verify_contract(provider)  # fails loudly on dim or normalization drift

    store = QdrantStore(cfg)
    created = store.create(dim=provider.spec.dim, recreate=recreate) # create collection

    docs = load_manifest(cfg)[: limit or None]
    t0 = time.monotonic()
    n_chunks = n_points = 0
    failures: list[dict[str, str]] = []

    pending: list[Chunk] = []

    def write(group: list[Chunk]) -> int:
        """Embed and upsert one batch. Explicit argument, no shared mutable state."""
        if not group:
            return 0
        vecs = provider.embed_documents([c.text for c in group])
        return store.upsert(group, vecs)

    for doc, chunks, err in iter_chunks(
        docs, embed_model_version=provider.model_version, run_id=run_id
    ):
        if err:
            failures.append({"doc_id": doc.doc_id, "error": err})
            if progress:
                progress(f"  ! {doc.doc_id}: {err}")
            continue
        n_chunks += len(chunks)
        pending.extend(chunks)
        while len(pending) >= batch:
            n_points += write(pending[:batch])
            pending = pending[batch:]
        if progress:
            progress(f"  + {doc.doc_id:44} {len(chunks):>5} chunks")

    n_points += write(pending)
    elapsed = time.monotonic() - t0

    return {
        "run_id": run_id,
        "collection_created": created,
        "documents": len(docs),
        "failed": len(failures),
        "failures": failures,
        "chunks": n_chunks,
        "points_written": n_points,
        "seconds": round(elapsed, 1),
        "chunks_per_sec": round(n_chunks / elapsed, 1) if elapsed else None,
        "embed_model_version": provider.model_version,
    }


def search(query: str, limit: int = 5, *, cfg: Settings | None = None) -> list[dict[str, Any]]:
    """Embed a query and return the top matches.

    Uses ``embed_queries`` -- the query side of the asymmetric interface. Calling
    ``embed_documents`` here would apply the wrong prefix and silently degrade
    recall (Step 0.4).
    """
    cfg = cfg or settings
    provider = get_provider("query", cfg)
    vec = provider.embed_queries([query])[0]
    return QdrantStore(cfg).search(vec, limit=limit)
