# Phase 7 — Query Understanding & Routing

**~12 hrs** · Prerequisite: [Phase 6](phase-06.md) · Next: [Phase 8](phase-08.md)

## Phase learning goals

1. Explain the query-document gap and which technique closes which part of it.
2. Explain *why* HyDE works and predict when it will hurt.
3. Decompose a multi-hop question into a retrieval plan.
4. Justify every millisecond a query-understanding technique adds — or delete it.
5. Design a router that isn't a single point of failure.

---

## Step 7.1 — The query-document gap

**Learning goal.** Name the four distinct gaps; match each to its fix.

### Concept

Users don't write queries that look like answers. Four distinct failures, four different fixes:

| Gap | Example | Fix |
|---|---|---|
| **Vocabulary** | "capital buffer rules" vs "…shall maintain a capital conservation buffer of…" | HyDE, query expansion |
| **Context** | "what about for 2024?" — meaningless standalone | Query rewriting |
| **Compositional** | "how did X change between the 2023 and 2024 circular?" — needs two retrievals | Decomposition |
| **Structural** | "policies effective after March 2024" — a *filter*, not a semantic query | Self-query filter extraction |

Applying the wrong fix costs latency and buys nothing. That's why this phase is a router, not a pipeline.

---

## Step 7.2 — Query rewriting

**Learning goal.** Justify 200 ms with a number.

**Concept.** Conversational → standalone. Mandatory the moment you have multi-turn: *"what about for 2024?"* embeds to nothing useful. Rewriting resolves pronouns, ellipsis, and implied entities against conversation history.

**Build.** Haiku, short output, ~200 ms. Build a **multi-turn eval slice** for this — you don't have one yet, and without it you cannot measure the technique that makes multi-turn work at all.

---

## Step 7.3 — HyDE

**Learning goal.** Explain the mechanism, then predict where it fails.

### Concept

**Hypothetical Document Embeddings**: ask the LLM to *write a fake answer* to the query, then embed **that** instead of the query.

Why it works: from [Phase 4](phase-04.md), embedding models place text near text that is *similar*, and a question is not similar to its answer. A hypothetical answer lives in **answer-space** — same register, same vocabulary, same shape as the real passage. You've replaced a cross-space comparison with a same-space one.

**When it hurts** — and it will, on some slices:
- The model hallucinates specifics (a fake regulation number) that pull retrieval to the wrong region.
- Queries that are already answer-shaped gain nothing and pay latency.
- Very short factual lookups where the query terms are exactly the right signal.

**Measure per slice.** HyDE is the clearest example in this project of a technique that is net-positive on average and net-negative on a slice you care about.

---

## Step 7.4 — Multi-query and decomposition

**Multi-query.** Generate 3 paraphrases, retrieve for each, fuse with RRF (Phase 5). Increases recall by covering more of the phrasing space. Costs 3× retrieval — parallelize it so wall-clock stays flat.

**Decomposition.** *"How did the Tier 1 requirement change between the 2023 and 2024 circular?"* is not one retrieval. Plan:
1. Retrieve Tier 1 requirement, 2023 circular
2. Retrieve Tier 1 requirement, 2024 circular
3. Compare in the generation step

This is where your `multi_hop` slice finally moves. Note this is also the boundary where "RAG pipeline" becomes "agent" — the difference is whether the plan is fixed or model-generated, which is Phase 10's question.

---

## Step 7.5 — Self-query: natural language → filter

**Concept.** *"policies effective after March 2024"* contains a **structured filter**, not a semantic query. Extract `effective_date > 2024-03-01` and pass it to the Phase 5 pre-filter, leaving "policies" as the semantic part.

Use **structured output** so the model cannot emit a malformed filter. Validate extracted fields against the schema — a hallucinated field name must fail loudly, never silently drop the filter (silently dropping it returns confident, unfiltered results).

**This connects to Phase 5's cardinality problem:** an extracted filter that matches 3 documents needs brute force; one matching 2M needs filtered traversal. Same mechanism.

---

## Step 7.6 — The router

**Learning goal.** Design a classifier that fails safe.

### Concept

Classify each query → `{direct answer, single retrieval, multi-hop, table lookup, clarify, refuse}`. Then apply only the techniques that class needs. Without a router you pay for every technique on every query.

**The router is a single point of failure.** A misroute is worse than no routing: send a multi-hop question down the single-retrieval path and it fails in a way the rest of the pipeline can't detect.

Three defences:
1. **Measure routing accuracy as its own metric**, with a labelled routing set.
2. **Fail safe** — when confidence is low, route to the *more capable* path, not the cheaper one.
3. **Keep classes few.** Four classes you can measure beat nine you can't.

### Numbers — the latency budget this phase spends

| Technique | Added p95 | Justified by |
|---|---|---|
| Router (Haiku) | ~150 ms | Everything below being conditional |
| Rewrite | ~200 ms | Multi-turn slice accuracy |
| HyDE | ~300 ms | Vocabulary-gap slices only |
| Multi-query ×3 | ~50 ms (parallel) | Recall on ambiguous queries |
| Decomposition | ~400 ms + extra retrieval | multi_hop slice only |

Naively stacked: **1.1 s before retrieval even starts**, against a 1.5 s TTFT budget ([NUMBERS.md §6](../../NUMBERS.md)). That's the whole argument for routing — techniques must be conditional, not cumulative.

Router cost: 86.4M queries/day × ~500 tokens on Haiku ≈ `500 × 86.4e6 × 1 ÷ 1e6` = **$43k/day** at D2 scale. Even the cheap model isn't free at scale.

---

## Phase gate

- [ ] `multi_hop` slice improves **≥ 25 pts**
- [ ] New multi-turn slice ≥ 0.85 answer accuracy
- [ ] Router accuracy ≥ 0.9, measured on a labelled routing set
- [ ] Every technique has a per-slice lift number and a latency cost; anything without both is deleted
- [ ] Total added latency fits the budget

## Traps

- Stacking every technique — each adds latency and a failure mode.
- Rewriting queries that were already fine.
- A router with 9 classes when 4 would do.
- Silently dropping a malformed extracted filter.
