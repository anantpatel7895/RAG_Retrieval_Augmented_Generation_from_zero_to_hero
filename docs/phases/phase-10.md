# Phase 10 — Advanced Retrieval Architectures

**~16 hrs** · Prerequisite: [Phase 9](phase-09.md) · Next: [Phase 11](phase-11.md)

> The point of this phase is **judgment**, not adoption. Every technique here is genuinely good at something and genuinely not worth it for most systems. The gate requires **at least two kills** with evidence.

## Phase learning goals

1. Decide with numbers when agentic RAG, GraphRAG, late interaction, and long-context each earn their cost.
2. Compute the storage and token multipliers *before* building.
3. Run the long-context-vs-RAG experiment honestly, including what prompt caching does to it.
4. Write a defensible rejection.

---

## Step 10.1 — Agentic RAG

**Concept.** The model decides *when* and *what* to retrieve, in a loop, with retrieval as a tool. Contrast with everything so far, where retrieval was a fixed pipeline stage.

What it buys: multi-hop without pre-planned decomposition, self-directed follow-up, and the ability to notice that the first retrieval was useless and try different terms.

**Numbers.** 3–5 LLM turns instead of 1, each carrying the growing conversation:
```
Fixed pipeline:  4k context × 1 call        = $0.013
Agentic:         ~6k avg × 4 calls          ≈ $0.058     (4.5×)
Latency:         800 ms TTFT → 3-5 s
```
At D2 scale that's $1.1M/day → $5M/day. **Find the query class that justifies 4.5×** — usually complex multi-hop, never simple lookup. That's a routing decision (Phase 7), not an architecture decision.

---

## Step 10.2 — Self-correcting retrieval (CRAG)

**Concept.** Grade retrieved documents for relevance; on low confidence, re-query with different terms or fall back to another source. Cheap version of agentic — one conditional retry rather than an open loop.

**Measure:** lift on the slices where initial retrieval is weak, against ~1 extra round-trip. Often the best value in this phase because the retry is *conditional* — most queries never pay it.

---

## Step 10.3 — GraphRAG

**Concept.** Extract entities and relations from every chunk, build a graph, detect communities, summarize them. Answers questions vector search **structurally cannot**: *"what are the main themes across all 2024 circulars?"* has no single relevant chunk — the answer is a property of the whole corpus.

Also strong on multi-hop where the hops are *entity* relations rather than semantic similarity.

**Numbers — construction cost, which is where it's decided:**
```
Entity extraction: 60,000 chunks × (1k in + 500 out) on Haiku
  input:  60M tok × $1/MTok  = $60
  output: 30M tok × $5/MTok  = $150
  + community summarization  ≈ $50
  total ≈ $260  (≈$130 with Batch API)

At 100M chunks: ≈ $350,000, and it must be rebuilt as the corpus changes.
```

**The decision this forces:** at our scale, $130 is nothing — build it and measure. At D2 scale, $350k plus incremental-update complexity means it needs to be answering questions worth that. Most corpora don't have interesting enough structure. **This is a likely kill; kill it with the number.**

---

## Step 10.4 — Late interaction (ColBERT / ColPali)

**Concept.** Instead of one vector per chunk, one vector **per token**. Score with **MaxSim**: for each query token, take its best match among document tokens, then sum. It's a middle ground between bi-encoder (one dot product) and cross-encoder (full attention) — richer than the former, far cheaper than the latter.

**ColPali** takes it further for documents: embed **image patches** of the rendered page directly, skipping text extraction entirely. Given how much of [Phase 2](phase-02.md) was fighting PDF parsing, this is genuinely interesting for our table-heavy corpus.

**Numbers — storage is the decider:**
```
Dense:  1 vec × 1024 dim × 4 B                = 4 KB/chunk
ColBERT: 400 tokens × 128 dim × 2 B (fp16)    = 100 KB/chunk     → 25×

60k chunks:   246 MB → 6 GB
100M chunks:  410 GB → 10 TB
```

25× storage and a more expensive scoring step. Justifiable if it wins big on `table_lookup` — measure it there specifically, since that's the slice where our other approaches struggle most.

---

## Step 10.5 — Multimodal and RAPTOR

**Multimodal.** Charts and diagrams that contain the answer. Two approaches: embed page images (ColPali-style) or generate VLM descriptions at ingest and embed those. The second is cheaper and works with your existing index; the first is more faithful.

**RAPTOR.** Recursive summarization into a tree — cluster chunks, summarize, cluster the summaries, repeat. Retrieval hits any level. Good for whole-document questions; adds ingestion cost and a staleness problem (the tree must be rebuilt when documents change).

---

## Step 10.6 — Long context vs RAG

**Learning goal.** Run the experiment everyone argues about, honestly.

**Concept.** With 1M-token context windows, why retrieve at all? Stuff 200k tokens of relevant documents and let the model find it.

**Numbers:**
```
RAG:                 4k context   × $2/MTok = $0.008 in
Long context:        200k context × $2/MTok = $0.400 in     (50×)
Long context cached: 200k × 0.1×            = $0.040        (5×)
```

Latency: TTFT on 200k tokens is seconds, not the 800 ms budget.

**The honest conclusion is nuanced:** RAG still wins clearly on cost and latency. But caching narrows a 50× gap to 5×, and long-context wins on *quality* for questions requiring synthesis across a whole document, where chunking destroys the very relationships the question is about. The real answer is a **hybrid**: retrieve to select the relevant documents, then give the model whole documents rather than chunks — which is [Phase 3](phase-03.md)'s small-to-big taken to its limit.

Run this on your `summarization` and `multi_hop` slices specifically.

---

## Phase gate

- [ ] An ADR per technique with a **keep/kill** decision backed by numbers
- [ ] **At least two kills** — knowing what not to build is the skill
- [ ] Long-context experiment run on your own slices, with cost and latency recorded
- [ ] Anything kept is wired into the Phase 7 router as a *conditional* path, not an unconditional stage

## Traps

- Adopting agentic RAG because it's fashionable.
- GraphRAG on a corpus too small to have interesting structure.
- Comparing techniques without holding the eval constant.
- Measuring quality lift without measuring the token multiplier.
