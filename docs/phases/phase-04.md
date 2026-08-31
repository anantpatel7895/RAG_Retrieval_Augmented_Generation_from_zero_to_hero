# Phase 4 — Embeddings

**~14 hrs** · Prerequisite: [Phase 3](phase-03.md) · Next: [Phase 5](phase-05.md)

## Phase learning goals

1. Explain what an embedding model is *trained to do*, and why that predicts where it fails.
2. Explain why MTEB rank does not transfer to your corpus.
3. Derive the index-size / recall tradeoff for Matryoshka truncation and quantization.
4. Mine hard negatives and fine-tune a bi-encoder — then judge honestly whether it beat just adding a reranker.
5. Plan an embedding-model migration across 100M vectors with zero downtime.

---

## Step 4.1 — What the model is actually trained to do

**Learning goal.** Predict failure modes from the training objective.

### Concept

Embedding models are trained with **contrastive learning**: pull (query, relevant passage) pairs together in vector space, push (query, irrelevant passage) pairs apart. The dominant loss is **MultipleNegativesRankingLoss** — within a batch, each query's positive is its own, and every *other* item in the batch is a negative. It's cheap (negatives are free) and it scales.

Three consequences fall straight out of the objective:

**1. The model learns the notion of "relevant" from its training data.** Trained on web QA, "relevant" means what it means on the web. Your regulatory corpus has a different notion — a clause is relevant to a query about a different clause it cross-references. The model has never seen this.

**2. In-batch negatives are *easy* negatives.** Random other passages are trivially distinguishable. The model never learns to separate two passages about *nearly* the same thing — which is exactly what your corpus is full of (2023 vs 2024 circular). This is the entire motivation for **hard negative mining** in 4.4.

**3. Similarity is not relevance.** Cosine similarity measures "these texts are about similar things." A query and its answer are often *not* similar texts — the question *"what is the Tier 1 minimum?"* and the answer *"…shall maintain not less than 6%…"* share almost no vocabulary. This gap is why we need rerankers, HyDE, and hybrid search at all.

### Why MTEB doesn't transfer

MTEB averages over datasets with **their** notion of relevance, **their** domain vocabulary, and **their** query distribution. A model 2 points higher on MTEB average can be 10 points worse on financial regulatory text. **Benchmark on your own golden set. Always.**

---

## Step 4.2 — Bake-off on your data

**Build.** 4–5 models scored on your Phase 1 golden set: BGE-M3, an e5 variant, a Qwen3-embedding, one small/fast model, and one hosted model as a control.

Report per model: recall@20 (per slice), latency/1k chunks, index size, and $/1M tokens.

**Asymmetric prefixes are load-bearing here.** The registry from Phase 0 handles it — but *verify* per model, because getting it wrong makes a good model look bad and you'd draw the wrong conclusion from your own bake-off.

---

## Step 4.3 — Dimensionality, Matryoshka, quantization

**Learning goal.** Find the knee on the recall-vs-size curve rather than guessing.

### Concept

**Matryoshka (MRL)** — models trained so that the *first* k dimensions are themselves a usable embedding. Truncate 1024 → 512 → 256 and lose surprisingly little. Works only if the model was trained for it; truncating a non-MRL model destroys it.

**Quantization** compresses each component:

| Mode | Size | Typical recall cost |
|---|---|---|
| fp32 | `n×d×4` | — |
| int8 (scalar) | `n×d×1` | ~1–2 pts |
| binary | `n×d÷8` | ~5–10 pts, **recoverable by re-scoring** |

**Re-scoring** is the trick that makes binary viable: search the binary index for the top ~200, then re-rank those with full-precision vectors. You pay 32× less memory for the search and full precision only where it matters.

### Numbers

| Config | 60k | 100M |
|---|---|---|
| 1024-d fp32 | 246 MB | 410 GB |
| 512-d fp32 | 123 MB | 205 GB |
| 1024-d int8 | 61 MB | 102 GB |
| 1024-d binary | 7.7 MB | 12.8 GB |

**The threshold that matters:** at 100M, fp32 forces a cluster; binary + re-scoring fits one machine. Quantization is not a memory optimization — it determines whether you operate a distributed system.

---

## Step 4.4 — Domain adaptation (full treatment)

**Learning goal.** Fine-tune a bi-encoder properly, then judge honestly whether it was worth it.

### Concept — hard negative mining

From 4.1: in-batch negatives are too easy. **Hard negatives** are passages that *look* relevant and aren't. Mine them:

1. Run your current retriever over each training query.
2. Take top-k results that are **not** the gold passage.
3. Filter out false negatives — passages that are actually relevant but unlabelled. **Skipping this step is the most common way fine-tuning makes things worse**, because you're explicitly teaching the model to push away correct answers.
4. Train with MNRL, positives + mined hard negatives.

**Hold out a clean test split** from before mining, and check for **catastrophic forgetting** on general queries — a model tuned hard on regulatory text can lose the ability to handle a plain question.

### The honest assessment

Compare against simply adding a reranker (Phase 6). Fine-tuning gives you:
- A custom model you now own, version, monitor, and retrain when the corpus drifts
- A migration burden every time you change it (see 4.5)

If a reranker gets 80% of the lift with none of that, the reranker wins. **You need to have done this once to know** — which is why it's in the plan even though the likely verdict is "not worth it."

---

## Step 4.5 — Migration planning

**Learning goal.** Answer the interview question: *"your embedding provider deprecates your model with 60 days notice — walk me through it."*

### Concept

Changing the embedding model invalidates **every vector in the index**. Query vectors from model B are meaningless against passages embedded by model A — not degraded, *meaningless*, and it fails silently with plausible-looking results.

The migration:
1. New collection, new `embed_model_version`.
2. **Backfill** — re-embed the whole corpus into it. This is the expensive step.
3. **Shadow-read** — serve from old, query new in parallel, compare metrics on live traffic.
4. **Atomic cutover** via a collection alias.
5. **Rollback** — keep the old collection until confident.

### Numbers

```
100M chunks ÷ 310 chunks/sec ≈ 4 days on one A100
```

Four days of continuous GPU time. Against a 60-day notice that's feasible — but only if you *have* the blue/green machinery already, and only if the ingest embedder is separate from the query embedder (Phase 0, Step 0.2) so the backfill doesn't blow your query SLO for four days.

**This is why the `role` split existed from day one.** The architecture decision in Phase 0 is what makes this migration survivable.

---

## Phase gate

- [ ] Model comparison table on **your** golden set, per slice, with CIs
- [ ] `chunks/sec` measured on the GPU box — **replace the estimate in [NUMBERS.md](../../NUMBERS.md)**
- [ ] Quantization config holding recall within 2 pts at ≥ 4× memory reduction
- [ ] Fine-tune run with a written keep/kill verdict vs. reranking
- [ ] Migration runbook drafted

## Traps

- MTEB-driven selection.
- Mixed-up query/passage prefixes invalidating your own bake-off.
- Hard negatives mined without false-negative filtering.
- Fine-tuning evaluated only on the domain it was tuned for.
