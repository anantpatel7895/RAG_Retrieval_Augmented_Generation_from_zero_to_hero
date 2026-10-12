"""Qdrant adapter.

Unlike the Phase 0 parser and chunker, nothing here is throwaway -- three decisions
are baked in now because they are expensive or impossible to change later.

**1. Vector configuration is immutable after collection creation.**
Qdrant fixes the *set of named vectors*, their dimension, and their distance metric
when the collection is created. Phase 5 adds sparse vectors for hybrid search. If we
create this collection dense-only, Phase 5 means: create a new collection, re-embed
71,248 chunks, re-upsert, swap. So we **declare the sparse vector now and leave it
unpopulated** -- an empty named vector costs nothing on disk.

**2. Payload indexes decide whether filtering works or quietly stops working.**
Qdrant will happily filter on an unindexed payload field -- by scanning. That passes
at our 120 documents and dies at 500k. Worse, `docs/PARSING.md` aside, an unindexed
field prevents *cardinality estimation*, which is how Qdrant chooses between brute
force over the filtered subset and filtered graph traversal. Get that wrong and
recall collapses silently (see `question_answer.md` Q5.6). Phase 5's metadata
pre-filtering and Phase 12's ACL enforcement are both this mechanism.

**3. Point IDs are deterministic**, derived from `chunk_id`, so re-ingesting an
unchanged document upserts over the same points rather than duplicating them.

**On scale, honestly.** 71,248 chunks brute-forces in about 7 ms
([NUMBERS.md](../../NUMBERS.md)). We do not need an ANN index at this size. We build
one because Phase 5 depends on operating `ef_search`, filtered traversal and the
recall/latency curve, and those cannot be learned by reading -- `docs/LESSONS.md` L9.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from qdrant_client import QdrantClient, models

from rag.config import Settings, settings
from rag.schema import Chunk

DENSE = "dense"
SPARSE = "sparse"

#: Payload fields that get an index at creation time.
#:
#: Each one is a filter dimension some later phase depends on:
#:   doc_type, effective_date  -> Phase 5 metadata pre-filtering, Phase 8 recency
#:   tenant_id, sensitivity    -> Phase 12 ACL-enforced retrieval
#:   doc_id                    -> Phase 11 incremental sync and deletion
#:   embed_model_version       -> Phase 11 blue/green reindex
INDEXED_PAYLOAD: dict[str, models.PayloadSchemaType] = {
    "doc_type": models.PayloadSchemaType.KEYWORD,
    "effective_date": models.PayloadSchemaType.DATETIME,
    "tenant_id": models.PayloadSchemaType.KEYWORD,
    "sensitivity": models.PayloadSchemaType.KEYWORD,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    "embed_model_version": models.PayloadSchemaType.KEYWORD,
}


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk.

    Qdrant point IDs must be an unsigned integer or a UUID, so a hex `chunk_id`
    cannot be used directly. A UUID5 over the chunk_id is stable across runs, which
    is what makes re-ingest an upsert instead of a duplicate.

    Example:

        >>> point_id("abc") == point_id("abc")
        True
        >>> point_id("abc") == point_id("abd")
        False
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    """Thin wrapper over `qdrant_client` for one collection."""

    def __init__(self, cfg: Settings | None = None, client: QdrantClient | None = None) -> None:
        self.cfg = cfg or settings
        self.client = client or QdrantClient(url=self.cfg.qdrant_url, timeout=60)
        self.collection = self.cfg.collection

    # -- lifecycle ---------------------------------------------------------- #

    def create(self, dim: int, *, recreate: bool = False) -> bool:
        """Create the collection, declaring both dense and sparse vectors.

        Args:
            dim: Dense vector dimension. Must match the embedding model's -- it is
                immutable afterwards, so `rag.embed.provider.verify_contract` should
                have run first.
            recreate: Drop an existing collection first. Destructive.

        Returns:
            True if a collection was created, False if it already existed and
            ``recreate`` was False.
        """
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return False
        if exists:
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            },
            # Declared now, populated in Phase 5. An unpopulated named vector costs
            # nothing; adding one later would require recreating the collection.
            sparse_vectors_config={
                SPARSE: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,  # BM25-style term weighting
                ),
            },
        )
        for field, schema in INDEXED_PAYLOAD.items():
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )
        return True

    def info(self) -> dict[str, Any]:
        """Collection stats plus which payload indexes actually exist."""
        c = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points": c.points_count,
            "status": str(c.status),
            "payload_indexes": sorted((c.payload_schema or {}).keys()),
        }

    def drop(self) -> None:
        self.client.delete_collection(self.collection)

    # -- write -------------------------------------------------------------- #

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        """Upsert chunks with their dense vectors.

        Args:
            chunks: Chunks to store; `Chunk.payload()` supplies the payload.
            vectors: One dense vector per chunk, same order.

        Returns:
            Number of points written.

        Raises:
            ValueError: If the counts disagree -- a silent zip truncation here would
                mis-pair vectors with payloads and corrupt the index invisibly.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if not chunks:
            return 0

        points = [
            models.PointStruct(
                id=point_id(ch.chunk_id),
                vector={DENSE: list(vec)},
                payload=ch.payload() | {"chunk_id": ch.chunk_id},
            )
            for ch, vec in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove every chunk of one document.

        Phase 11 needs this to make deletion propagate; Phase 12 needs it because a
        document that stays retrievable after deletion is a compliance breach, not
        untidiness.
        """
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
            wait=True,
        )

    # -- read --------------------------------------------------------------- #

    def search(
        self,
        vector: Sequence[float],
        limit: int = 10,
        *,
        flt: models.Filter | None = None,
        ef_search: int | None = None,
    ) -> list[dict[str, Any]]:
        """Dense similarity search, optionally filtered.

        Args:
            vector: Query embedding, from `embed_queries` -- **not**
                `embed_documents`. Using the wrong side is the asymmetric-prefix bug
                from Step 0.4 and it fails silently.
            limit: How many results.
            flt: Qdrant filter. Applied *inside* the search, not to its output --
                post-filtering would return fewer than `limit` results and is not an
                access control (Phase 12).
            ef_search: HNSW beam width for this query only. The one ANN parameter
                tunable at query time without a rebuild, which makes it the knob you
                reach for at 3am to trade recall for latency.

        Returns:
            One dict per hit: ``{score, chunk_id, doc_id, text, doc_type,
            effective_date}``.
        """
        params = models.SearchParams(hnsw_ef=ef_search) if ef_search else None
        res = self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            using=DENSE,
            limit=limit,
            query_filter=flt,
            search_params=params,
            with_payload=True,
        ).points

        return [
            {
                "score": p.score,
                "chunk_id": (p.payload or {}).get("chunk_id"),
                "doc_id": (p.payload or {}).get("doc_id"),
                "doc_type": (p.payload or {}).get("doc_type"),
                "effective_date": (p.payload or {}).get("effective_date"),
                "text": ((p.payload or {}).get("text") or "")[:300],
            }
            for p in res
        ]

    def scroll_ids(self, limit: int = 10_000) -> Iterable[str]:
        """Iterate stored chunk_ids. Used by the recall ground-truth sweep."""
        offset = None
        while True:
            pts, offset = self.client.scroll(
                collection_name=self.collection,
                limit=min(limit, 1000),
                with_payload=["chunk_id"],
                offset=offset,
            )
            for p in pts:
                yield (p.payload or {}).get("chunk_id", "")
            if offset is None:
                break
