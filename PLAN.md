# RAG: Zero → Hero (Production + System Design Track)

**Audience:** 5 yrs AI/ML/GenAI. No Python basics, no "what is an embedding."
**Premise:** the deliverable of each step is *understanding*, verified by a working system. Code is the artifact, not the goal.
**Duration:** 14 phases, ~160 hrs including the system-design thread. ~14 weeks at 10 hrs/wk.

---

## How to use this

`PLAN.md` is the map. Each phase has its own working document under [docs/phases/](docs/phases/) — that's what we open when we work the phase.

| Document | Purpose |
|---|---|
| [PLAN.md](PLAN.md) | Rules, method, stack, phase index, design exercises |
| [docs/phases/phase-NN.md](docs/phases/) | The working document for one phase: concepts, numbers, builds |
| [docs/DATA.md](docs/DATA.md) | What the corpus **is** — units, what a 10-K contains, every size figure derived |
| [docs/PARSING.md](docs/PARSING.md) | HTML vs PDF — what each format destroys, problems → solutions → cost. Phase 2 foundation |
| [NUMBERS.md](NUMBERS.md) | Every formula in the project, with worked examples |
| [question_answer.md](question_answer.md) | Running Q&A log — the real test of whether a step landed |
| `docs/decisions/` | One ADR per significant choice |
| `docs/design/` | The five system-design exercise write-ups |

---

## Rules of engagement

1. **Eval before optimization.** Phase 1 builds the golden set and harness. After that, no change ships without a measured delta. If you can't measure it, you don't get to claim it.
2. **One corpus, all the way through.** Every phase improves the *same* system on the *same* data. That's how "hybrid search helped" becomes a number instead of a vibe.
3. **Every phase has a numeric gate.** Not "understand reranking" — "nDCG@10 +8 pts for <250 ms p95."
4. **Latency and cost are features.** Every phase logs p50/p95 and $/1k queries. A trick that adds 900 ms is a regression until proven otherwise.
5. **Break it on purpose.** Each phase ends adversarially: what does this now fail on? Into the eval set it goes.
6. **Design before code.** Four questions, answered in the ADR before writing anything: **contract** (what does it promise, how is it versioned), **failure domain** (what do downstream components see when it's slow, down, or silently wrong), **scaling axis** (what breaks first as corpus/QPS/tenants grow), **capacity math** (the numbers, before and after).
7. **Derive the numbers.** Every step computes its own capacity math — for our corpus *and* at the scale where the answer changes. The contrast is where the design insight lives.
8. **Write the answer down.** Every step appends to [question_answer.md](question_answer.md). If you can't write the answer without looking it up, the step isn't finished.

**Anti-goal:** no framework until Phase 11. Phases 1–8 are raw SDKs on purpose — know what LangChain/LlamaIndex hide before deciding whether to accept the abstraction.

---

## The step format

A phase is a theme; a **step** is the unit of work. Six parts, in order:

| Part | Question it answers |
|---|---|
| **0. Learning goal** | What can you *do* after this that you couldn't before? Stated as a capability, not a topic. |
| **1. Concept** | What is this, what problem does it exist to solve, how does the mechanism actually work? |
| **2. Numbers** | What must you be able to derive on a whiteboard? Our scale, and the scale where the answer flips. |
| **3. Landscape & choice** | What are the alternatives, on what axes do they differ, why this one for us, what changes the answer? |
| **4. Design** | Contract / failure domain / scaling axis (rule 6). → ADR |
| **5. Build & Q&A** | Working code + verification, then entries in `question_answer.md`. |

Concept before choice. Choice before design. Design before code.

---

## The corpus (locked)

**Financial + regulatory documents**, 200–500 PDFs:
- Annual reports / 10-Ks — dense tables, footnotes, multi-column
- Regulatory circulars (RBI / BIS / Basel) — long, cross-referencing, numbered clauses
- Product & API manuals — code blocks, screenshots, step lists
- Policy docs, plus some scanned pages needing OCR

Chosen because it has **tables that carry the answer**, **numbers that must not be hallucinated**, **cross-document references**, **near-duplicate versions** (2023 vs 2024 circular), and **natural access-control structure**. Nearly every hard production problem appears without contrivance.

**Hard requirement:** ≥30 documents where the answer lives in a table, ≥20 needing two documents.

**Working estimate** used throughout for capacity math — recompute once the real corpus lands:

```
400 docs × 120 pages × 500 tok/page = 24M tokens
24M tokens ÷ 400 tok/chunk          ≈ 60,000 chunks
```

---

## Stack (locked)

| Layer | Choice | Why |
|---|---|---|
| Runtime | Python 3.12.10 via **pyenv** (`RAG_VENV`) | Pinned by `.python-version`, auto-activates on `cd` |
| Packages | `pyproject.toml` + pip, pinned with **pip-tools** | Unpinned deps silently move eval scores |
| Generation | Claude via `anthropic` SDK | `claude-sonnet-5` serving, `claude-opus-5` judges, `claude-haiku-4-5` routing/rerank |
| Embeddings | **BGE-M3** on GPU via TEI; e5 / Qwen3 as bake-off candidates | Local-first; hosted model kept as control |
| Vector DB | **Qdrant** primary, **pgvector** secondary | Filtered HNSW + native sparse; pgvector to test "just use Postgres" honestly |
| Reranker | `bge-reranker-v2-m3` on GPU via TEI | Cross-encoder mechanics at real latency |
| Parsing | `pymupdf`, `unstructured`, `docling`, + VLM pass | Parsing is where most quality is won or lost |
| Serving | FastAPI + SSE | Streaming changes architecture, not just UI |
| Tracing | Langfuse (docker) | You cannot debug RAG from logs |
| Eval | Custom harness first, RAGAS second | Write your own scorers before trusting a library's |

---

## Repo layout

```
rag/
  ingest/     loaders, parsers, table extraction, OCR, normalization
  chunk/      chunkers + metadata
  embed/      providers behind one interface, registry, cache
  index/      vector store adapters, schema, migrations
  retrieve/   dense, sparse, fusion, filters, MMR
  rerank/     cross-encoder, LLM rerank, calibration
  query/      rewriting, decomposition, routing, HyDE
  generate/   prompt assembly, citations, abstention
  serve/      FastAPI, streaming, caching, auth, tenancy
  obs/        tracing, metrics, cost accounting
eval/
  datasets/   golden sets (versioned JSONL), adversarial sets
  scorers/    retrieval metrics, groundedness, judges
  runs/       run artifacts, leaderboard
  harness.py  one command → full report
docs/
  phases/     the working document per phase
  decisions/  ADRs
  design/     system design exercise write-ups
```

**Rule:** `notebooks/` never imports into `rag/`. Code graduates to a module or it dies.

---

## Phase index

Each links to its working document. **The learning goal is the deliverable.**

| # | Phase | Learning goal — after this you can… | Hrs |
|---|---|---|---|
| [0](docs/phases/phase-00.md) | Foundations & environment | Explain what a vector store *is*, derive index memory and query cost from scratch, and justify a store choice against alternatives | 10 |
| [1](docs/phases/phase-01.md) | Baseline + eval harness | Design an eval that can detect a 3-point regression, and calibrate an LLM judge against human labels | 16 |
| [2](docs/phases/phase-02.md) | Ingestion & parsing | Predict which documents a given parser will silently destroy, and compute the break-even for a VLM parser | 16 |
| [3](docs/phases/phase-03.md) | Chunking & metadata | Explain why optimal chunk size differs by question type, and cost a contextual-retrieval run before running it | 12 |
| [4](docs/phases/phase-04.md) | Embeddings | Choose an embedding model from *your* data, and plan a model migration across 100M vectors | 14 |
| [5](docs/phases/phase-05.md) | Retrieval, hybrid & ANN internals | Explain HNSW layer by layer, tune the recall/latency curve, and say why filtered ANN is hard | 16 |
| [6](docs/phases/phase-06.md) | Reranking & context assembly | Explain why a cross-encoder can't be a retriever, and calibrate an abstention threshold | 12 |
| [7](docs/phases/phase-07.md) | Query understanding & routing | Decompose a multi-hop question, and justify each 300 ms of added query latency | 12 |
| [8](docs/phases/phase-08.md) | Generation, citations, abstention | Verify a citation *supports* a claim rather than merely existing, and handle conflicting sources | 12 |
| [9](docs/phases/phase-09.md) | Observability & CI | Root-cause a bad answer from a trace in under 5 minutes | 10 |
| [10](docs/phases/phase-10.md) | Advanced architectures | Decide, with numbers, when agentic RAG / GraphRAG / ColBERT / long-context each earn their cost | 16 |
| [11](docs/phases/phase-11.md) | Production engineering | Execute a zero-downtime reindex, and build a defensible cost and latency model | 16 |
| [12](docs/phases/phase-12.md) | Security & governance | Break your own system by prompt injection and ACL bypass, then close both | 12 |
| [13](docs/phases/phase-13.md) | Capstone | Present the whole system, and design one 10× larger on a whiteboard | 16 |

---

## System design exercises

Rule 6 applies the design lens to what you're building. These apply it to systems **an order of magnitude beyond** what you've built — the only way to practise designing something you can't go measure.

Each becomes a design doc in `docs/design/`: requirements → capacity math → component diagram → data & control flow → failure modes → rejected alternatives → rollout. Timebox: 90 min designing, 60 min writing up. **Cold first, then look things up.**

| # | After | Problem | Forces you to reason about |
|---|---|---|---|
| D1 | Phase 3 | Ingest 10M documents, daily refresh, 5% change rate | Throughput math, queueing, idempotency, backpressure, poison messages, cost of full re-parse |
| D2 | Phase 5 | 100M chunks, 1k QPS, p95 < 200 ms | Memory math, sharding vs replication, ANN under load, cache modelling, hot-shard skew |
| D3 | Phase 8 | Multi-tenant API, 200 tenants, uneven sizes | Isolation models, noisy neighbours, per-tenant quality and cost attribution, cold tenants |
| D4 | Phase 11 | Migrate 100M vectors to a new embedding model, zero downtime | Dual-write, backfill throughput, cutover consistency, rollback, GPU-hours |
| D5 | Phase 12 | RAG for a regulated bank: audit, residency, ACLs, RTBF | Compliance as architectural driver, deletion propagation, provenance, blast radius |

**Every exercise ends with a number you'd defend in review and at least one explicitly rejected alternative.**

---

## Appendix A — Production problem playbook

Grows as you go. By Phase 13 it's the most valuable artifact here.

| Symptom | Likely cause | Where to look | Fix |
|---|---|---|---|
| Answer in corpus, never retrieved | Parsing loss or chunk boundary | Grep raw text; absent → parsing | Ph2 parser routing; Ph3 structure-aware chunking |
| Exact codes/IDs not found | Dense-only retrieval | Compare dense vs BM25 on that query | Ph5 hybrid + RRF |
| High recall, bad answers | Rerank or context assembly | Is gold chunk in final top-5? At what position? | Ph6 rerank + position handling |
| Confidently wrong on unanswerable | No abstention path | Score distribution vs threshold | Ph6 calibration + Ph8 refusal |
| Quality dropped after deploy | Untracked config change | Diff between run_ids | Ph9 CI gate |
| p95 spike, p50 flat | ANN params or tail dependency | Per-span latency histogram | Ph5 `ef_search`; Ph11 timeouts |
| Cost doubled, traffic flat | Cache invalidated | `cache_read_input_tokens` = 0? | Ph11 caching; hunt the volatile prefix |
| Stale answers after doc update | Broken incremental sync | Chunk `ingest_run_id` vs source mtime | Ph11 upsert + tombstones |
| Cross-tenant data visible | Post-hoc filtering or cache leak | Filter pre- or post-search? Cache key? | Ph12 filter-level ACL |
| 8 near-identical chunks in context | No dedup/MMR | Inspect final context | Ph6 MMR + dedup |
| Conflicting answers across versions | No recency preference | `effective_date` on retrieved chunks | Ph8 conflict handling |

---

## Appendix B — Metrics: what each actually measures

| Metric | Measures | Watch out |
|---|---|---|
| recall@k | Did retrieval get the answer anywhere in k? | The ceiling on everything downstream. Fix first, always. |
| precision@k | How much noise is in context | Matters at generation, not search |
| MRR | How high the *first* correct result ranked | Blind to the 2nd–nth correct doc |
| nDCG@k | Graded relevance, position-weighted | Needs graded labels to mean anything |
| Groundedness | Claims traceable to context | High score ≠ correct — context can be wrong |
| Answer correctness | Match to gold | Judge-dependent; calibrate the judge |
| Citation precision/recall | Citations that exist **and** support | Existence ≠ support |
| Abstention accuracy | Refuses when it should, answers when it should | Measure both directions or you over-tune one |
| Context relevance | Fraction of context that was useful | Direct proxy for wasted tokens = wasted money |

**Slice everything** by question_type, difficulty, doc_type. Aggregates hide the regressions that matter.

---

## Appendix C — Interview depth map

The PDFs in [books/](books/) are a **check after each phase**, never the curriculum.

| PDF | Read after |
|---|---|
| `RAG_and_vector_practical_guide.pdf` | Phase 1 |
| `RAG_Embed_and_vector_db_question.pdf` | Phase 4–5 |
| `vectordatabase_interview_question.pdf` | Phase 5 |
| `RAGAS_Evaluation.pdf` | Phase 1 + 9 |
| `RAG_topic_wise_interview_question.pdf` | Phase 13 sweep |

---

## Appendix D — Progress

| Phase | Hrs | Status | Gate result |
|---|---|---|---|
| 0 Foundations | 10 | ▶ in progress | |
| 1 Eval harness | 16 | ☐ | |
| 2 Parsing | 16 | ☐ | |
| 3 Chunking | 12 | ☐ | |
| 4 Embeddings | 14 | ☐ | |
| 5 Retrieval | 16 | ☐ | |
| 6 Reranking | 12 | ☐ | |
| 7 Query understanding | 12 | ☐ | |
| 8 Generation | 12 | ☐ | |
| 9 Observability | 10 | ☐ | |
| 10 Advanced | 16 | ☐ | |
| 11 Production | 16 | ☐ | |
| 12 Security | 12 | ☐ | |
| 13 Capstone | 16 | ☐ | |
