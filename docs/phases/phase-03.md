# Phase 3 — Chunking & Metadata

**~12 hrs** · Prerequisite: [Phase 2](phase-02.md) · Next: [Phase 4](phase-04.md) · **Design exercise D1 lands here**

## Phase learning goals

1. Explain chunking as *choosing the unit of retrieval*, and the tension that makes it hard.
2. Explain why optimal chunk size differs by question type — and what that implies architecturally.
3. Cost a contextual-retrieval run **before** running it, with and without prompt caching.
4. Decouple "what makes retrieval accurate" from "what makes generation accurate."

---

## Step 3.1 — The chunking tension

**Learning goal.** State the tradeoff precisely enough to predict which strategy wins for which question type.

### Concept

A chunk is the **unit of retrieval**: the smallest thing that can be independently found. That creates a genuine tension:

| Smaller chunks | Larger chunks |
|---|---|
| Precise embeddings — vector is *about* one thing | Diluted embeddings — one vector averaging many topics |
| Higher precision@k | Higher recall (answer more likely inside) |
| Answer may be split across a boundary | Self-contained context for the model |
| More chunks → more index, more cost | Fewer, cheaper |

Two distinct failure modes sit at the ends:

**Boundary splitting** — the answer straddles two chunks, and neither is retrievable for the question.

**Context starvation** — the chunk is retrieved but unusable: *"as noted above, this rate applies to all Tier 2 instruments"* — retrieved perfectly, and meaningless. This is the failure people underestimate. The chunk is *relevant* and *useless*.

### Strategies

| Strategy | Mechanism | Wins on |
|---|---|---|
| Fixed-token | Split every N tokens | Nothing. The baseline. |
| Recursive character | Split on ¶ → sentence → word, respecting a max | Cheap general default |
| **Structure-aware** | Split on the heading hierarchy from Phase 2 | Well-structured docs — *our corpus*. Usually the winner. |
| Semantic | Embed sentences, cut where embedding distance spikes | Unstructured prose. Expensive; rarely beats structure-aware on structured docs |
| **Contextual** | Prepend an LLM-generated situating sentence before embedding | Context starvation — directly |
| Late chunking | Embed the long document, then pool per chunk | Preserves cross-chunk context; needs a long-context embedder (BGE-M3 qualifies) |

---

## Step 3.2 — Contextual retrieval, and its cost

**Learning goal.** Cost an LLM-per-chunk operation before running it, and explain how prompt caching changes it by an order of magnitude.

### Concept

For each chunk, ask an LLM: *given the whole document, write 1–2 sentences situating this chunk.* Prepend that before embedding. It attacks context starvation directly — *"as noted above, this rate…"* becomes *"In the section on Tier 2 capital instruments of the 2024 circular, describing the applicable rate…"*.

The naive implementation sends the full document with **every chunk**. That is the expensive part, and it is exactly what prompt caching is for: the document goes in the cached prefix, the chunk varies in the suffix.

### Numbers — do this before you run it

```
400 docs × 60k tokens/doc, 150 chunks/doc = 60,000 chunks
```

| Approach | Input tokens | Cost (Haiku, $1/MTok in) |
|---|---|---|
| Naive: full doc per chunk | 60,000 × 60k = **3.6B** | **$3,600** |
| Cached: write once, read at 0.1× | 400 × (60k×1.25 + 149×60k×0.1) = **388M** | **$388** |
| + output (60k × 80 tok @ $5/MTok) | | +$24 |
| + Batch API −50% | | **≈ $206** |

**$3,600 → $206, a 17× reduction, from one architectural decision.** Verify it worked by checking `cache_read_input_tokens` is non-zero — if it's zero, something in your prefix is varying (a timestamp, unsorted JSON, a per-chunk id placed too early) and you are paying full price silently.

This is the single most useful cost lesson in the project, and it generalizes to every "LLM over every item" operation.

---

## Step 3.3 — Small-to-big (parent document retrieval)

**Learning goal.** Explain why the retrieval unit and the generation unit should differ.

**Concept.** Embed **small, precise** chunks. On a hit, return the **parent section** to the LLM.

This dissolves the tension in 3.1 rather than compromising on it: small chunks give precise vectors and high precision@k; the parent gives the model enough context to actually answer. The `parent_id` field in [rag/schema.py](../../rag/schema.py) exists for exactly this — declared in Phase 0 so we don't have to re-index for it now.

---

## Step 3.4 — The sweep, and what it reveals

**Learning goal.** Show empirically that one chunk size cannot be optimal for all question types.

**Build.** Grid over chunk size × overlap × strategy, run the fast harness, plot recall@20 **per question_type**.

**What you'll find:** `factoid` peaks at small chunks (precision), `summarization` peaks at large (coverage), `table_lookup` doesn't care about size but cares enormously about the Phase 2 representation. There is no single optimum.

**The architectural implication:** either accept a compromise, or **route** — different chunk indexes per question type, selected by the Phase 7 router. Note the finding now; the decision lands in Phase 7.

Beware: with 60-question slices, moves under ~20 pts are noise ([NUMBERS.md §5](../../NUMBERS.md)). Report CIs on every cell of that grid.

---

## Step 3.5 — Freeze the metadata schema

**Learning goal.** Recognize which fields are one-way doors.

Already declared in [rag/schema.py](../../rag/schema.py). Confirm each is populated: `doc_id`, `chunk_id`, `parent_id`, `section_path`, `page`, `bbox`, `doc_type`, `effective_date`, `version`, `source_url`, `sensitivity`, `tenant_id`, `checksum`, `embed_model_version`, `ingest_run_id`.

An unpopulated field is as bad as a missing one — verify with a real query, not by reading the code.

---

## Design exercise D1

**Ingest 10M documents, refreshed daily, 5% change rate.** 90 min design, 60 min write-up → `docs/design/D1-ingestion-at-scale.md`.

Must address: throughput math (868 chunks/sec sustained), queue design and backpressure, idempotency, poison-message handling, partial failure and resume, the cost of a full re-parse vs incremental, and how you'd detect that 200k documents silently stopped ingesting.

**Ends with:** a defensible number and at least one explicitly rejected alternative.

---

## Phase gate

- [ ] Best chunker beats fixed-512 by **≥ 10 pts recall@20**, reported per slice with CIs
- [ ] Contextual retrieval measured with a real cost number and a verified cache hit rate
- [ ] Sweep plot showing different optima per question_type
- [ ] D1 written up
- [ ] Every schema field verified populated

## Traps

- Tuning chunk size on the aggregate score.
- Overlap used as a substitute for structure.
- Running contextual retrieval without checking the cache actually hit.
- Changing the schema after indexing — that's a full reindex.
