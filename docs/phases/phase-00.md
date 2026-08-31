# Phase 0 — Foundations & Environment

**~10 hrs** · Prerequisite: none · Next: [Phase 1](phase-01.md)

## Phase learning goals

By the end you can:
1. Explain what a vector store *is* — and why it is not "a database that stores vectors."
2. Walk through HNSW layer by layer and explain **why greedy search converges** and why it is still approximate.
3. Derive index memory and per-query compute from scratch, and name the corpus size at which each architectural decision flips.
4. Justify Qdrant against pgvector, Milvus, Elasticsearch and FAISS on named axes — and say when each is the right answer instead.
5. Name the two silent embedding failures and design so neither can happen quietly.

---

## Step 0.1 — Interpreter & reproducibility ✅

**Learning goal.** Spot environment-inheritance bugs before they corrupt a benchmark.

**Concept.** `pyenv` resolves `.python-version` by walking *up* the directory tree. A repo without its own file silently adopts a parent's interpreter. Here the repo sits under `~/Anant_Learning/`, which pins 3.11.7, while the intended env is `RAG_VENV` on 3.12.10.

**Why it matters more here than elsewhere:** your outputs are *measurements*. A different interpreter means a different resolved dependency set, which means different text extraction and possibly different numerics — and you'll attribute the score change to your retrieval work.

**Built.** [.python-version](../../.python-version), [pyproject.toml](../../pyproject.toml) with GPU deps quarantined in a `[local]` extra.

---

## Step 0.2 — Environment topology ✅

**Learning goal.** Argue why the embedder is a network dependency from day one.

**Concept.** Two workloads, opposite characteristics, same model:

| | Ingest | Query |
|---|---|---|
| Shape | 100k+ texts, batched | 1 text |
| Bound by | GPU throughput | Network RTT + SLO |
| On failure | Retry, resume, backpressure | Fail fast, degrade |
| Tolerance | Minutes | Tens of ms |

Point both at one instance and a nightly reindex saturates the GPU while query p95 goes from 40 ms to seconds. Nothing is "down"; users just see timeouts.

**Built.** [docker-compose.yml](../../docker-compose.yml) (Qdrant, pgvector, Langfuse behind a profile), [docker/gpu-compose.yml](../../docker/gpu-compose.yml) (TEI embedder + reranker).

---

## Step 0.3 — Config & schema ✅

**Learning goal.** Explain why chunk schema is a one-way door.

**Concept.** Fields you didn't capture can't be backfilled without re-parsing the corpus. Realising in Phase 8 that citation highlighting needs `page` and `bbox` — when you never stored them — means re-running ingestion over everything. Declaring an unused field costs nothing; omitting a needed one costs a full re-parse.

**Built.** [rag/config.py](../../rag/config.py), [rag/schema.py](../../rag/schema.py).

---

## Step 0.4 — The embedding model contract ✅

**Learning goal.** Name the two silent embedding failures; design them out.

**Concept.** Both produce *no error*: (1) **prefix asymmetry** — e5-family needs `query: ` / `passage: `, wrong prefixes cost 5–15 pts recall silently; (2) **dimension drift** — config says 1024, model emits 768, and if the collection was created from the same wrong config everything "works" on a junk index.

Defence: a version-controlled registry owning dim + prefixes, and a startup probe that verifies dimension and L2 norm *empirically* rather than trusting a metadata endpoint.

**Built.** [rag/embed/registry.py](../../rag/embed/registry.py), [rag/embed/provider.py](../../rag/embed/provider.py).

---

## Step 0.5 — The vector store ◀ **current**

**Learning goal.** Explain the ANN trade, walk HNSW layer by layer, derive the memory and compute numbers, and defend a store choice on named axes.

### Part 1 — Concept

#### The task and why the obvious answer dies

You hold *n* vectors. A query arrives. Return the *k* closest.

```
10M chunks × 1024 dims × 4 bytes = 41 GB just to hold them
10M × 1024                       = 10.2 billion multiply-adds per query
```

Single-threaded: ~1 second. Well-parallelized: ~100 ms — **per query**. Ten concurrent users saturate the machine.

#### The trade that defines the field

Give up exactness. Accept "probably the top 10, usually" for a ~1000× speedup. That is **Approximate Nearest Neighbour** search.

Internalize the difference from a B-tree: a B-tree returns *the* answer with a complexity guarantee. An ANN index returns a *probable* answer with a **recall characteristic you tune**. You cannot reason about it the way you reason about a database index — and "recall@k of the index" is a distinct thing from "recall@k of your retrieval," which is a distinct thing from "did the user get a good answer."

#### HNSW, mechanism first

Two ideas stacked.

**Idea 1 — navigable small world (NSW).** Build a graph where each node links to some near neighbours *and* a few far ones. Greedy routing works: long links cover distance fast, short links refine. Kleinberg's result is the theory — when long-range link probability falls off with distance at the right rate, greedy routing reaches its target in polylogarithmic hops.

**Idea 2 — hierarchy.** Assign each node a maximum layer drawn from an exponentially decaying distribution: layer 0 holds every node, layer 1 holds ~1/e of them, layer 2 ~1/e², and so on. The result is a **skip list in metric space** — sparse express lanes above, the full graph below.

**Search:**
1. Enter at the single top-layer entry point.
2. At each layer, **greedy descent**: repeatedly move to whichever neighbour is closer to the query. Stop when no neighbour is closer — a local minimum for that layer.
3. Drop to the next layer down, using that node as the entry point.
4. At layer 0, switch from pure greedy to **beam search**: maintain a candidate set of size `ef_search`, expand all of them, keep the best `ef`. Return the top *k*.

**Why it converges.** Each greedy step strictly decreases distance to the query. Distance is bounded below by zero and the graph is finite, so the walk must terminate — at a local minimum. That's the guarantee: **termination, not correctness.**

**Why the hierarchy matters.** Greedy descent on a flat graph reaches a local minimum that may be far from the true nearest neighbour. The upper layers make long jumps first — coarse positioning — so by the time you reach layer 0 you're already in the right neighbourhood and the local minimum is very likely the global one. The hierarchy buys **quality of the local minimum**, not just speed.

**Why it is still approximate.** Greedy can get trapped. `ef_search` is the mitigation: a wider beam explores more of the graph and escapes more traps. That's the entire recall/latency knob — `ef_search` up means more distance computations means higher recall.

**Construction, and the non-obvious part.** Inserting a node: find the `ef_construct` nearest at each layer, then connect to `M` of them. The subtle bit is that connecting to the **M nearest** is *wrong* — it clusters links in one direction and destroys navigability. HNSW uses a **diversity heuristic**: prefer a candidate if it is closer to the new node than to any already-selected neighbour. This keeps links pointing in varied directions, which is what makes greedy routing work at all.

**Three parameters, three tradeoffs:**

| Param | Controls | Up means |
|---|---|---|
| `M` | Links per node | More recall, more memory (`n × 2M × 4` bytes), slower build |
| `ef_construct` | Build-time beam | Better graph, slower build. Build-time only. |
| `ef_search` | Query-time beam | Higher recall, higher latency. **Tunable per query, live.** |

`ef_search` is the only one you can change without rebuilding. That's why it's the 3am knob.

#### The other family, and quantization

**IVF** partitions space by clustering; search opens only the `nprobe` nearest cells. Faster to build than HNSW, worse recall at equal speed, but far better at very large scale combined with compression.

**Quantization is orthogonal** to both — compress each vector (fp32 → int8 → binary), usually with a re-scoring pass on the top candidates using full-precision vectors. It cuts memory *and* speeds up search, because HNSW is memory-latency bound, not FLOP bound.

#### Why a vector DB isn't just an index

Postgres can store vectors. FAISS can do ANN. Neither is a vector database. What the database adds:

1. **Filtered search that preserves recall** — the genuinely hard problem, below.
2. **Live mutability** — insert/update/delete without rebuilding.
3. **Persistence, snapshots, replication, sharding.**
4. **Payload storage and indexing**, so metadata lives with the vector.

#### Filtered ANN — the hard problem, and why it decides our choice

"Nearest 10 chunks **where** `tenant_id = X` **and** `effective_date > 2024`."

- **Post-filter** — search, then drop non-matching. Ask for 10, get 2. Worse as the filter narrows.
- **Pre-filter then brute force** — correct, but if the filter matches 2M rows you're back to a linear scan.
- **Filter during traversal** — skip non-matching nodes while walking. This *breaks the graph*: the path to the answer may run through excluded nodes, so the walk stalls in a disconnected region and recall collapses silently.

Real systems solve it by **estimating filter cardinality** and switching strategy — brute force when the match set is small, filtered traversal when it's large — plus building extra links so filtered subgraphs stay connected.

> This mechanism is the reason we're choosing a vector DB at all. With no filters, `faiss` + a pickle file would genuinely be enough. Phase 5 (metadata pre-filtering) and Phase 12 (ACL-enforced retrieval) are both *entirely* this mechanism.

**Deletion is the other dirty secret.** HNSW has no clean delete — removing a node would strand the links pointing at it. Systems tombstone instead, and the graph degrades until you rebuild. Phase 11's "verify a deleted document can no longer be retrieved" gate exists because of this, and it's a compliance problem, not just a hygiene one.

### Part 2 — Numbers to derive

| Quantity | Formula | 60k (ours) | 100M (D2) |
|---|---|---|---|
| Vectors fp32 | `n × d × 4` | 246 MB | **410 GB** |
| Vectors binary | `n × d ÷ 8` | 7.7 MB | **12.8 GB** |
| HNSW links | `n × 2M × 4`, M=16 | 7.7 MB | 12.8 GB |
| Payload | `n × tok × 4` | 96 MB | **160 GB** |
| Exact search | `n × d` MAC | 61M ≈ 6 ms | 102B ≈ 10 s |
| HNSW, ef=128 | `ef × log₂n × d` MAC | 2.1M ≈ 0.2 ms | 3.5M ≈ 0.35 ms |

**Three insights that fall out:**
1. At 60k chunks **brute force is fine** (6 ms). We use ANN to learn it, not because we need it.
2. HNSW cost is nearly **independent of n** — only `log₂ n` grows. 10× the corpus, 14% more work.
3. fp32 at 100M = 410 GB → you must shard. Binary → ~26 GB with the graph → **one machine**. Quantization is the decision that determines whether you have a distributed system.

Full workings: [NUMBERS.md](../../NUMBERS.md).

### Part 3 — Landscape & choice

| Option | What it really is | Strong when | Weak when |
|---|---|---|---|
| **FAISS** | Library, not a DB | Research, learning internals | You need persistence, filtering, CRUD |
| **pgvector** | Postgres extension | Metadata already in PG; joins/transactions; < ~2–5M vectors | Memory-hungry builds; no native sparse; scaling = scaling PG |
| **Qdrant** | Purpose-built, Rust, single binary | Filtering central; native sparse + fusion; strong quantization | One more system to run |
| **Weaviate** | Same class, heavier, inline embedding modules | Batteries included | You want to own the pipeline — we do |
| **Milvus** | Distributed-first (etcd + object store + MQ) | 1B+ vectors | Heavy ops for anything smaller |
| **Elasticsearch** | Lexical engine that grew vectors | Already running it; lexical is primary | Weaker quantization and vector perf; JVM ops |
| **Pinecone** | SaaS | No ops team | Cost at scale, and it **hides internals** — disqualifying here |

**Why Qdrant for us**, tied to phases:
1. **Filtered HNSW** with cardinality-based strategy selection — Phase 5 and Phase 12 *are* this mechanism.
2. **Native sparse vectors + built-in fusion** — Phase 5 hybrid without a second system.
3. **Scalar / product / binary quantization** — makes the memory math hands-on.
4. **One container, inspectable dashboard** — low friction for a loop we'll run hundreds of times.

**When it's the wrong answer:** a few million chunks with metadata already in Postgres → **pgvector is very likely correct**, and adding a vector DB is unjustified complexity. That's why pgvector stays as the secondary store and Phase 5 makes you write the honest comparison.

### Part 4 — Design

- **Contract.** Collection name, vector config (named vectors, dim, distance), payload schema, payload indexes. **Vector config is immutable after creation** — this is a one-way door.
- **Failure domain.** Qdrant down → retrieval fails hard (acceptable, loud). Qdrant *reachable but serving a collection built by a different embedding model* → confident nonsense (unacceptable, silent). Mitigation: store `embed_model_version` on every point and assert at query time.
- **Scaling axis.** RAM for vectors + graph first; payload second. Sharding is forced by total index bytes ÷ per-node RAM.
- **Capacity.** ~350 MB all-in at our scale. Single node, no quantization needed, no sharding. Say so in the ADR — it justifies *not* building things.

### Part 5 — Build

1. Collection creation with **dense + sparse named vectors declared now** (an unpopulated named vector is free; adding one later means recreating the collection and re-upserting everything).
2. **Payload indexes** on `doc_type`, `effective_date`, `tenant_id`, `sensitivity`, `doc_id` — without these, filters degrade to scans that pass in dev and fail at 500k.
3. Upsert with deterministic point ids (idempotent re-ingest).
4. Search with optional filter.
5. **Optional (recommended, ~2 hrs):** implement brute-force k-NN and a toy HNSW; plot recall vs `ef_search` yourself. Phase 5's tuning stops being cargo-cult.

**Gate:** upsert 100 vectors, search, confirm filtered search returns correct results; `ef_search` sweep shows the recall/latency curve moving.

---

## Step 0.6 — Corpus manifest

**Learning goal.** Explain why ingestion must be idempotent and content-addressed before you have any content.

**Concept.** The manifest is the corpus **inventory**: one row per source document with `doc_id`, path, checksum, and the metadata that later becomes filter dimensions. Content hashing is what makes re-ingestion idempotent — re-running over an unchanged corpus must be a no-op, or every experiment costs a full re-embed.

**Numbers.** 400 docs → manifest is trivial. At D1 scale (10M docs, 5% daily change), the manifest *is* the change-detection system: 500k changed docs/day → 868 chunks/sec sustained. Diffing checksums is what keeps that from being 10M re-embeds.

**Build.** `rag/ingest/manifest.py` — scan `data/raw/`, hash, emit `manifest.jsonl`.

---

## Step 0.7 — Cost accounting from day one

**Learning goal.** Explain why cost instrumentation belongs in Phase 0 rather than Phase 11.

**Concept.** You cannot retrofit cost attribution. If token usage isn't recorded per stage per run from the first day, "which change tripled our spend" is unanswerable. Cheap now, impossible later.

**Numbers.** One judged eval run = $11.55; 100 runs = $1,155. That's a real budget line, and it's why the inner loop is retrieval-only eval (free) with judged evals nightly.

**Build.** `rag/obs/cost.py` — a `record()` that appends `{run_id, stage, model, in_tok, out_tok, cache_read, usd}` to JSONL, plus the pricing table.

---

## Step 0.8 — Wire it together

**Learning goal.** State precisely what a smoke test proves and what it does not.

**Concept.** With the stub embedder, a green smoke test proves **plumbing**: files parse, chunks form, vectors upsert, search returns. It proves **nothing about quality** — stub vectors are random. Conflating the two is how people report "RAG is working" on a system that retrieves noise.

**Build.** `rag/cli.py` (typer) + `Makefile`: `make up`, `make smoke`, `make lint`.

**Gate:** `make up && make smoke` ingests the 5 PDFs in [books/](../../books/) and returns search results.

---

## Phase gate

- [ ] `make up && make smoke` green end to end
- [ ] Recall/latency curve plotted from your own `ef_search` sweep
- [ ] ADR written: why Qdrant, and the corpus size at which you'd switch to pgvector
- [ ] `question_answer.md` has entries for every step
- [ ] You can whiteboard HNSW search and explain why it converges — without notes

## Traps

- Treating the smoke test as evidence of quality.
- Creating the collection without payload indexes — works in dev, dies at scale.
- Dense-only collection now, hybrid "later" — that's a full rebuild.
- Skipping the capacity math because the corpus is small. The point is knowing where the thresholds are.
