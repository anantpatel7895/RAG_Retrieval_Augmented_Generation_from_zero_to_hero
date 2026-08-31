"""The chunk and document schema.

Freeze this early and change it reluctantly: PLAN.md Phase 3 warns that a schema
change after 500k chunks are indexed is a full reindex. Fields that are unused
until a later phase are declared now and left empty -- adding a payload field to
Qdrant later is cheap, but discovering you never captured `page` when you need
citation highlighting in Phase 8 means re-parsing the whole corpus.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Sensitivity = Literal["public", "internal", "confidential", "restricted"]


class DocMeta(BaseModel):
    """One row of data/raw/manifest.jsonl -- the corpus inventory."""

    doc_id: str
    path: str
    title: str
    doc_type: str = "unknown"  # annual_report | circular | manual | policy
    source_url: str | None = None
    effective_date: date | None = None
    version: str | None = None
    owner: str | None = None
    sensitivity: Sensitivity = "public"
    tenant_id: str = "default"
    checksum: str = ""
    n_pages: int | None = None
    bytes: int | None = None

    # Measured at fetch time, not estimated. The corpus chain in NUMBERS.md is
    # computed from these -- a planning estimate of 60k tokens/doc turned out to be
    # 6x wrong on a real bank 10-K, which made contextual-retrieval cost 36x wrong
    # (it scales as D^2/c). Never extrapolate corpus size from one sample.
    est_tokens: int | None = None
    text_ratio: float | None = None  # stripped text / raw bytes; ~0.12 for iXBRL 10-Ks


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str

    # --- provenance (Phase 2/8: citations need these) -----------------------
    parent_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page: int | None = None
    bbox: tuple[float, float, float, float] | None = None

    # --- filtering (Phase 5/12: pre-filters and ACLs key off these) ---------
    doc_type: str = "unknown"
    effective_date: date | None = None
    version: str | None = None
    source_url: str | None = None
    sensitivity: Sensitivity = "public"
    tenant_id: str = "default"

    # --- lifecycle (Phase 11: incremental sync and reindex key off these) ---
    checksum: str = ""
    embed_model_version: str = ""
    ingest_run_id: str = ""

    def payload(self) -> dict[str, object]:
        """Qdrant payload. `text` is stored so we can read chunks back without a join."""
        d = self.model_dump(mode="json")
        d.pop("chunk_id", None)  # lives as the point id
        return d


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk_bytes):
            h.update(block)
    return h.hexdigest()
