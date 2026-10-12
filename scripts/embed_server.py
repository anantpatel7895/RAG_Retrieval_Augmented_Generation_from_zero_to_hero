"""A TEI-compatible embedding server. Runs ON THE GPU BOX, not in RAG_VENV.

**Why this exists.** The GPU box has no `nvidia-container-toolkit`, so Docker cannot
pass GPUs through, and installing it requires `systemctl restart docker` -- which on a
shared machine would kill a colleague's 20 GB vLLM job. This needs no privileges and
no daemon restart.

**Why it works without changing a line of our pipeline.** `rag.embed.provider.TEIProvider`
does not depend on TEI. It depends on a *contract*:

    POST /embed   {"inputs": ["text", ...], "truncate": true}   ->  [[float, ...], ...]
    GET  /info                                                  ->  {"model_id": "...", ...}

Anything satisfying that contract is a drop-in. That is what the interface in Step 0.4
was for -- TEI, this server, or a hosted API are interchangeable, and the pipeline
cannot tell which one it is talking to. The response of `/embed` is a **bare JSON
array**, not an object; matching that exactly is what makes it drop-in.

--------------------------------------------------------------------------------
RUN
    pip install fastapi uvicorn sentence-transformers
    CUDA_VISIBLE_DEVICES=1 python embed_server.py

`CUDA_VISIBLE_DEVICES=1` is the process-level equivalent of `--gpus '"device=1"'`.
It is set **at launch, deliberately not in code**: on a shared box the device is an
operational decision, and burying it in a source file is how a job ends up on
someone else's GPU. Measured 2026-09-02, GPU 0 holds another team's vLLM engine
(20.5 GB of 23 GB) and GPU 1 is ours.

CHECK
    curl -s localhost:8080/info | python3 -m json.tool
    curl -s -X POST localhost:8080/embed -H 'Content-Type: application/json' \
         -d '{"inputs":["capital adequacy ratio"]}' | head -c 120
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import os
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
PORT = int(os.environ.get("EMBED_PORT", "8080"))

# BGE-M3 supports 8192 tokens, but our chunks are ~400. Attention is quadratic in
# sequence length, so capping this at 512 is a large speedup for zero quality loss
# on our data. Raise it only if the chunker starts producing longer chunks --
# `--auto-truncate` behaviour below would otherwise silently cut text.
MAX_SEQ_LEN = int(os.environ.get("EMBED_MAX_SEQ_LEN", "512"))

# Sentences per forward pass. Larger = better GPU utilisation and more VRAM.
# 64 is deliberately modest: we share GPU 1 with our own ~9 GB process.
BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "64"))

# float16 halves VRAM (~2.3 GB -> ~1.2 GB) and roughly doubles throughput on an A10,
# which has dedicated fp16 tensor cores. BGE-M3 is routinely served in fp16 with no
# measurable retrieval loss.
#
# This defaulted to fp32 in the first version -- `SentenceTransformer` does not infer
# it -- while gpu-compose.yml advertised `--dtype=float16`. The documentation claimed
# a setting the code never implemented, which is worse than not offering it: /info
# reported `torch.float32` and only a careful read of that field caught it. Verify
# `model_dtype` in /info after any change here.
DTYPE = os.environ.get("EMBED_DTYPE", "float16")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("embed_server")

app = FastAPI(title="TEI-compatible embedding server")
model: SentenceTransformer | None = None
_stats = {"requests": 0, "texts": 0, "seconds": 0.0}


class EmbedRequest(BaseModel):
    inputs: list[str] | str
    # TEI accepts and ignores/honours these; we accept them so the client needs no
    # special-casing. `truncate` matters: without it, an over-long input raises
    # instead of being cut, and a single bad chunk would fail a whole batch.
    truncate: bool = True
    normalize: bool = True


class RerankRequest(BaseModel):
    query: str
    texts: list[str]
    raw_scores: bool = Field(default=False)


@app.on_event("startup")
def load() -> None:
    global model
    t0 = time.monotonic()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # fp16 only makes sense on GPU; on CPU it is slower than fp32.
    kwargs: dict[str, object] = {}
    if device == "cuda" and DTYPE != "float32":
        kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, DTYPE)}

    model = SentenceTransformer(MODEL_ID, device=device, **kwargs)  # type: ignore[arg-type]
    model.max_seq_length = MAX_SEQ_LEN

    log.info(
        "model=%s device=%s dtype=%s max_seq_len=%s",
        MODEL_ID,
        device,
        next(model.parameters()).dtype,
        MAX_SEQ_LEN,
    )
    if device == "cuda":
        # CUDA_VISIBLE_DEVICES remaps indices, so torch always sees its allotted GPU
        # as device 0. Log the real name so it is obvious which card we actually got.
        log.info(
            "gpu=%s  visible=%s  vram_total=%.1f GB",
            torch.cuda.get_device_name(0),
            os.environ.get("CUDA_VISIBLE_DEVICES", "<all>"),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        log.warning("NO GPU VISIBLE — running on CPU. Did you set CUDA_VISIBLE_DEVICES?")
    log.info("loaded in %.1fs", time.monotonic() - t0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if model is not None else "loading"}


@app.get("/info")
def info() -> dict[str, object]:
    """Mirrors TEI's /info. `model_id` is what `verify_contract` checks against config."""
    if model is None:
        raise HTTPException(503, "model still loading")
    dim = model.get_sentence_embedding_dimension()
    thr = _stats["texts"] / _stats["seconds"] if _stats["seconds"] else None
    return {
        "model_id": MODEL_ID,
        "model_dtype": str(next(model.parameters()).dtype),
        "max_input_length": MAX_SEQ_LEN,
        "max_client_batch_size": BATCH_SIZE,
        "embedding_dimension": dim,
        "device": str(next(model.parameters()).device),
        # Throughput measured on real traffic. Phase 4 needs this number: reindex
        # wall time = total_chunks / texts_per_sec, and that is the DR objective.
        # NUMBERS.md currently carries a FLOP-derived *estimate* of ~310/s.
        "texts_per_sec_observed": round(thr, 1) if thr else None,
        "requests_served": _stats["requests"],
    }


@app.post("/embed")
def embed(req: EmbedRequest) -> list[list[float]]:
    """Embed one or more texts.

    Returns a **bare array of arrays**, matching TEI exactly. Wrapping it in an
    object would break `TEIProvider`, which does `r.json()` and expects a list.
    """
    if model is None:
        raise HTTPException(503, "model still loading")

    texts = [req.inputs] if isinstance(req.inputs, str) else req.inputs
    if not texts:
        return []

    t0 = time.monotonic()
    vecs = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        # Cosine similarity assumes unit vectors, and our Qdrant collection uses
        # COSINE distance. Normalising here means the store never has to guess.
        normalize_embeddings=req.normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    dt = time.monotonic() - t0

    _stats["requests"] += 1
    _stats["texts"] += len(texts)
    _stats["seconds"] += dt
    if _stats["requests"] % 20 == 0:
        log.info(
            "served %d requests / %d texts — %.1f texts/sec cumulative",
            _stats["requests"],
            _stats["texts"],
            _stats["texts"] / _stats["seconds"],
        )

    return [[float(x) for x in row] for row in vecs]


@app.post("/rerank")
def rerank(req: RerankRequest) -> list[dict[str, object]]:
    """Not implemented — Phase 6 territory, and it needs a *cross-encoder*.

    Fails loudly rather than falling back to bi-encoder similarity, which would look
    like a working reranker while delivering none of the benefit. A silent
    approximation here would be exactly the failure class in docs/LESSONS.md L1.
    """
    raise HTTPException(
        501,
        "reranking needs a cross-encoder (BAAI/bge-reranker-v2-m3), not this "
        "bi-encoder. Run a second server for it in Phase 6.",
    )


if __name__ == "__main__":
    # Single worker on purpose: multiple workers would each load a full copy of the
    # model into VRAM. Batching inside one process is the right lever here.
    uvicorn.run(app, host="0.0.0.0", port=PORT, workers=1, log_level="info")
