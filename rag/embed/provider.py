"""Embedding providers behind one interface.

Two design decisions worth understanding before you read the code.

**1. `embed_documents` and `embed_queries` are separate methods.**
Not stylistic. The registry knows the prefix convention; splitting the methods means
the prefix is applied in exactly one place and cannot be forgotten at a call site.
A single `embed(texts, mode=...)` puts the burden back on every caller.

**2. Providers are role-scoped: "ingest" or "query".**
The same model serves two workloads with opposite characteristics:

                    ingest                  query
    shape           500k texts, batched     1 text
    bound by        GPU throughput          network RTT + latency SLO
    on failure      retry, resume           fail fast, degrade
    scale on        queue depth             p95 latency
    tolerance       minutes                 50 ms

Point both at one instance and a nightly reindex saturates the GPU while query p95
goes from 40 ms to seconds. Nothing is "down"; users just see timeouts. In dev both
roles resolve to the same URL. In production they are separate deployments, and the
role argument is the seam that makes that a config change rather than a refactor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag.config import Settings, settings
from rag.embed.registry import ModelSpec, get_spec

Role = Literal["ingest", "query"]


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_version(self) -> str: ...

    @property
    def spec(self) -> ModelSpec: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class _Base:
    def __init__(self, cfg: Settings, role: Role) -> None:
        self.cfg = cfg
        self.role = role
        self._spec = get_spec(cfg.embed_model)

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def model_version(self) -> str:
        """Stamped onto every chunk. Phase 11's blue/green reindex keys off this."""
        return f"{self.cfg.embed_provider}:{self.cfg.embed_model}:{self._spec.dim}"

    def _batched(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        n = self.cfg.embed_batch_size
        for i in range(0, len(texts), n):
            out.extend(self._encode(list(texts[i : i + n])))
        return out

    def _encode(self, batch: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._batched([self._spec.doc_prefix + t for t in texts])

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._batched([self._spec.query_prefix + t for t in texts])


class StubProvider(_Base):
    """Deterministic pseudo-vectors. No GPU, no network, no model download.

    Lets the pipeline run end-to-end before the GPU box exists, and lets CI (Phase 9)
    exercise ingestion for free. Same text always yields the same vector, so upserts
    are idempotent and tests are stable.

    Retrieval quality under the stub is meaningless by construction. A green smoke
    test here proves plumbing, never quality -- never quote a number measured on it.
    """

    def _encode(self, batch: list[str]) -> list[list[float]]:
        import hashlib
        import random

        vecs: list[list[float]] = []
        for text in batch:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = random.Random(seed)
            v = [rng.gauss(0.0, 1.0) for _ in range(self._spec.dim)]
            norm = sum(x * x for x in v) ** 0.5 or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class TEIProvider(_Base):
    """HuggingFace Text Embeddings Inference over HTTP -- the GPU box."""

    def __init__(self, cfg: Settings, role: Role) -> None:
        super().__init__(cfg, role)
        self.base_url = cfg.tei_url_for(role)
        # Ingest tolerates minutes; a query must fail fast so the caller can degrade
        # rather than hold a request open past its SLO.
        timeout = 300.0 if role == "ingest" else 10.0
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # Retry only what is plausibly transient. `ConnectError` is deliberately
    # EXCLUDED: a refused connection means nothing is listening, and the host will
    # not start listening within a 20-second backoff. Retrying it turned a
    # one-character URL mistake into 30 seconds of exponential backoff and a wall
    # of stack frames -- slow failure is worse than fast failure, because it hides
    # the cause. Timeouts, read errors and 5xx are the transient cases.
    @retry(
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError)
        )
        | retry_if_exception(
            lambda e: isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
        ),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _encode(self, batch: list[str]) -> list[list[float]]:
        r = self.client.post("/embed", json={"inputs": batch, "truncate": True})
        r.raise_for_status()
        vecs: list[list[float]] = r.json()
        return vecs

    def info(self) -> dict[str, object]:
        r = self.client.get("/info")
        r.raise_for_status()
        data: dict[str, object] = r.json()
        return data


class STProvider(_Base):
    """sentence-transformers in-process. GPU box only -- needs the [local] extra."""

    def __init__(self, cfg: Settings, role: Role) -> None:
        super().__init__(cfg, role)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg.embed_model)

    def _encode(self, batch: list[str]) -> list[list[float]]:
        arr = self.model.encode(batch, normalize_embeddings=self._spec.normalize)
        return [list(map(float, row)) for row in arr]


class ContractViolation(RuntimeError):
    pass


def verify_contract(provider: EmbeddingProvider) -> None:
    """Fail loudly at startup if the served model disagrees with the registry.

    This is contract verification, structurally the same as a schema registry
    rejecting an incompatible producer. It is checked empirically -- by embedding a
    probe and measuring the result -- rather than by trusting a metadata endpoint,
    because the thing we actually care about is what the vectors look like.

    Run this before any ingest run and at API startup. The cost is one embedding
    call; the alternative is discovering a corrupt index after 500k upserts.
    """
    spec = provider.spec
    probe = provider.embed_queries(["contract probe"])[0]

    if len(probe) != spec.dim:
        raise ContractViolation(
            f"registry declares dim={spec.dim} but the served model returned "
            f"{len(probe)}-d vectors. Fix RAG_EMBED_MODEL or the registry entry -- "
            f"a dim mismatch corrupts the index silently."
        )

    if spec.normalize:
        norm = sum(x * x for x in probe) ** 0.5
        if abs(norm - 1.0) > 1e-3:
            raise ContractViolation(
                f"registry declares normalize=True but probe vector has L2 norm {norm:.4f}. "
                f"Cosine similarity assumes unit vectors -- unnormalized vectors make "
                f"scores incomparable across chunks of different length."
            )

    if isinstance(provider, TEIProvider):
        try:
            served = provider.info().get("model_id")
        except httpx.HTTPError:
            return  # /info is a nicety; the empirical checks above are the real gate
        if served and served != provider.cfg.embed_model:
            raise ContractViolation(
                f"TEI at {provider.base_url} serves {served!r} but config says "
                f"{provider.cfg.embed_model!r}. Querying an index built by a different "
                f"model returns plausible-looking garbage."
            )


def get_provider(role: Role = "query", cfg: Settings | None = None) -> EmbeddingProvider:
    cfg = cfg or settings
    match cfg.embed_provider:
        case "stub":
            return StubProvider(cfg, role)
        case "tei":
            return TEIProvider(cfg, role)
        case "st":
            return STProvider(cfg, role)
    raise ValueError(f"unknown embed provider {cfg.embed_provider!r}")
