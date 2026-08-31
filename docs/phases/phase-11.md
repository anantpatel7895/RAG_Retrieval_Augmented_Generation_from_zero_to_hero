# Phase 11 — Production Engineering

**~16 hrs** · Prerequisite: [Phase 10](phase-10.md) · Next: [Phase 12](phase-12.md) · **Design exercise D4 lands here**

## Phase learning goals

1. Make a document deletion actually propagate — index, cache, and traces.
2. Execute a zero-downtime reindex with a rehearsed rollback.
3. Explain how a semantic cache serves a *wrong* answer, and bound the risk.
4. Build a defensible cost model and enforce a latency budget.
5. Decide on a framework with evidence, having built it raw.

---

## Step 11.1 — Incremental sync and deletion

**Learning goal.** Prove a deleted document is unreachable — by every path.

**Concept.** Change detection via checksum (Phase 0's manifest), upserts on deterministic chunk ids, and **deletes that actually propagate**.

Deletion is the hard one, for the reason from [Phase 0](phase-00.md): HNSW has no clean delete. Systems tombstone, and the graph carries the dead node until a rebuild. So "deleted" can mean:

- removed from the payload store but still reachable via the graph
- gone from the index but still in the **semantic cache**
- gone from both but still in **traces** and eval artifacts

**Test it properly:** delete a source document, then verify it cannot be retrieved, cannot be cited, does not appear in a cached answer, and is purged from traces per your retention policy. Most systems fail at least one of these, and it's a compliance problem, not hygiene.

---

## Step 11.2 — Zero-downtime reindex

**Learning goal.** Rehearse the migration from [Phase 4](phase-04.md) end to end.

**Concept.** Blue/green behind a collection alias:
1. Create the new collection with the new `embed_model_version`
2. **Backfill** — re-embed everything (the expensive step)
3. **Dual-write** new ingests to both during backfill, or you'll finish backfilling a stale corpus
4. **Shadow-read** — serve old, query new in parallel, compare metrics on live traffic
5. **Atomic cutover** — flip the alias
6. **Rollback** — keep old until confident

**Numbers.** `100M ÷ 310 chunks/sec ≈ 4 days` ([NUMBERS.md §3](../../NUMBERS.md)). Four days of dual-write and shadow-read is why steps 3 and 4 exist — a backfill that ignores concurrent writes ends with a new index missing four days of documents.

**Rehearse it with a real embedding-model change.** An unrehearsed runbook is fiction.

---

## Step 11.3 — Caching, and its risks

**Learning goal.** Name what each cache layer can get wrong.

| Layer | Keyed on | Risk |
|---|---|---|
| **Exact query cache** | Normalized query string + filters + tenant | Stale after corpus update — needs invalidation on ingest |
| **Semantic cache** | Query embedding, hit above similarity τ | **Serves a wrong answer.** "Tier 1 minimum for 2023" and "…for 2024" are highly similar and have different answers |
| **Prompt cache** | Stable prefix | Silent miss from a volatile prefix |

**The semantic cache deserves real fear.** Similarity is not equivalence — exactly the [Phase 4](phase-04.md) lesson. Two queries differing only in a year, an entity, or a negation embed almost identically and have completely different answers. Bound it: set τ high, exclude queries containing numbers/dates/entities, scope cache keys by tenant, and measure the wrong-hit rate explicitly on your golden set. If you can't measure it, don't ship it.

**The tenant scoping is a security control, not a performance detail** — see [Phase 12](phase-12.md).

---

## Step 11.4 — Latency budget, enforced

| Stage | p95 budget |
|---|---|
| Route | 150 ms |
| Rewrite (conditional) | 200 ms |
| Embed query | 50 ms |
| Hybrid retrieve | 150 ms |
| Rerank | 250 ms |
| Generate TTFT | 800 ms |
| **Total** | **1.5 s** |

Enforce with per-stage timeouts, and define what happens when each blows:

| Stage times out | Degradation |
|---|---|
| Rewrite | Use the raw query |
| Rerank | Return retrieval order |
| Embedder | Fall back to lexical-only |
| Vector DB | Honest error — do **not** answer ungrounded |

**That last row is the important one.** Under pressure, the tempting failure mode is to answer from parametric knowledge. That converts an outage into a hallucination, which is worse.

---

## Step 11.5 — Cost model

**Concept.** $/query by stage, then the levers in order — free wins before quality tradeoffs:

1. **Prompt caching** — free, ~$233k/day at D2 scale ([Phase 8](phase-08.md))
2. **Batch API −50%** for all offline work: evals, contextual chunks, metadata extraction, GraphRAG construction
3. **Context budgeting** — dynamic k from [Phase 6](phase-06.md)
4. **Model tiering** — Haiku for routing/rerank, Sonnet for serving, Opus for judges and hard queries
5. **Effort tuning** — the first real quality tradeoff, not the first lever

**Judge cost per completed task, not per request.** A cheaper model that needs a retry isn't cheaper.

---

## Step 11.6 — Load and chaos testing

**Build.** Concurrency ramp to find the knee. Then break things deliberately: vector DB down, embedder returning 429s, LLM timeout, one tenant sending 100× normal traffic.

Every dependency needs a **tested** degradation path. Untested fallback code is a second outage waiting inside the first.

---

## Step 11.7 — The framework decision

**Learning goal.** Evaluate LangChain / LlamaIndex / Haystack from a position of knowledge.

You've now built every component raw. Evaluate honestly: what would the framework have saved, what would it have hidden, and what would migrating cost now? Write the ADR either way. **"We rejected it, here's why"** is a complete and respectable answer — and so is adopting it.

---

## Design exercise D4

**Migrate 100M vectors to a new embedding model, zero downtime.** → `docs/design/D4-model-migration.md`

Must address: backfill throughput and GPU count (4 days on one A100 — how many do you buy?), dual-write consistency, shadow-read comparison methodology, cutover atomicity, rollback window, cost in GPU-hours, and how you detect mid-backfill that the new model is *worse*.

---

## Phase gate

- [ ] p95 TTFT < 1.5 s at 50 concurrent users
- [ ] Zero-downtime reindex **rehearsed**, including rollback
- [ ] Deletion verified through index, cache, and traces
- [ ] $/1k queries documented by stage
- [ ] Every dependency has a tested degradation path
- [ ] Semantic cache wrong-hit rate measured, or the cache isn't shipped
- [ ] Framework ADR written
- [ ] D4 written up

## Traps

- Backfilling without dual-write.
- Semantic caching without measuring wrong hits.
- Degrading to ungrounded answers.
- Cost models that ignore eval and ingestion spend.
