"""Single source of truth for configuration.

Everything tunable lives here so that an eval run can serialise the exact config
that produced it. From Phase 1 onward, every run artifact records `Settings` --
"which config produced run 42" must never be a guessing game.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

EmbedProvider = Literal["stub", "tei", "st"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_prefix="RAG_",
        extra="ignore",
    )

    # --- paths --------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    runs_dir: Path = REPO_ROOT / "runs"

    # --- generation ---------------------------------------------------------
    # Not RAG_-prefixed: the SDK reads ANTHROPIC_API_KEY directly, so keep one name.
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    answer_model: str = "claude-sonnet-5"
    judge_model: str = "claude-opus-5"
    cheap_model: str = "claude-haiku-4-5"

    # --- embeddings ---------------------------------------------------------
    # dim and prefixes are NOT configurable -- they come from rag/embed/registry.py,
    # which is the model contract. See the module docstring there for why.
    embed_provider: EmbedProvider = "stub"
    embed_model: str = "BAAI/bge-m3"
    embed_batch_size: int = 64

    # One URL in dev; two deployments in production. Ingest saturating the GPU must
    # not be able to blow the query latency SLO -- see rag/embed/provider.py.
    tei_url: str = "http://localhost:8080"
    tei_ingest_url: str | None = None  # falls back to tei_url

    # --- storage ------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag_chunks"
    postgres_dsn: str = "postgresql://rag:rag@localhost:5432/rag"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    def tei_url_for(self, role: str) -> str:
        if role == "ingest" and self.tei_ingest_url:
            return self.tei_ingest_url
        return self.tei_url


settings = Settings()
