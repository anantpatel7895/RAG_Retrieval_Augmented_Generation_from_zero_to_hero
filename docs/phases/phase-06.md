# Phase 6 — Reranking & Context Assembly

**~12 hrs** · Prerequisite: [Phase 5](phase-05.md) · Next: [Phase 7](phase-07.md)

## Phase learning goals

1. Explain why a cross-encoder cannot be a retriever — from the architecture, with the FLOP math.
2. Calibrate raw reranker scores into something you can threshold on.
3. Measure the lost-in-the-middle effect on your own system.
4. Build an abstention threshold with a defensible precision/recall tradeoff.

---

## Step 6.1 — Bi-encoder vs cross-encoder

**Learning goal.** Explain the architectural difference and derive why it forces a two-stage pipeline.

### Concept

**Bi-encoder** (your embedding model): encodes query and document **separately** into vectors, compares with cosine. Because documents are encoded independently of the query, you can embed them **once, offline**, and at query time do only vector math. That's what makes indexing possible.

The cost: query and document never see each other. All interaction is compressed into a single dot product between two fixed vectors.

**Cross-encoder** (the reranker): concatenates `[query, document]` into **one** input and runs a full transformer over the pair. Every query token attends to every document token. It can represent "this passage answers this question" in a way a dot product structurally cannot — which is worth 10+ nDCG points.

The cost: the computation **depends on the query**, so nothing can be precomputed. Scoring N documents means N full forward passes.

### The math that forces the architecture

```
Forward pass ≈ 2 × params × tokens
bge-reranker-v2-m3 (XLM-R large, 568M params), 512-token pairs:

per pair  = 2 × 568e6 × 512      = 5.7e11 FLOP  (573 GFLOP)
100 pairs = 5.7e13 FLOP          = 57 TFLOP
A100 @ ~140 TFLOP/s effective    ≈ 0.41 s
```

**Scoring 100 candidates takes ~400 ms on an A100.** Now extrapolate: scoring all 60,000 chunks would take ~4 minutes **per query**. At 100M chunks it's over a month. Per query.

That's the whole argument, quantitatively: a cross-encoder is ~10⁴× too expensive to be an index, and ~10× better than a bi-encoder at ranking. So you use the cheap one to get from 100M to 100, and the expensive one to get from 100 to 5. **Two stages is not a convention, it's arithmetic.**

Levers if 400 ms is too slow: fewer candidates (50 instead of 100 → 200 ms), shorter sequences (256 tokens → half), or a smaller reranker (base ~278M → half again).

---

## Step 6.2 — The retrieve-depth tradeoff

**Build.** Compare retrieve-100→rerank-5 vs retrieve-20→rerank-5. Hold the candidate generator constant so you're measuring rerank depth, not retrieval.

**The key insight:** reranking **cannot fix recall**. If the gold chunk isn't in the candidate set, no reranker recovers it. Deeper candidates raise the recall ceiling and cost linear latency. The right depth is where recall@N plateaus — find it on your curve from Phase 5, don't guess.

---

## Step 6.3 — Score calibration

**Learning goal.** Turn raw scores into a threshold you can defend.

### Concept

Reranker outputs are **logits**, not probabilities. A score of 3.2 means nothing absolute — not "76% relevant" — and the distribution shifts across query types. You cannot threshold on raw scores.

**Calibrate**: fit a mapping from score to P(relevant) on labelled data — logistic regression on the raw score (Platt scaling) is usually enough. Now "score > τ" means something stable.

**Why this matters far beyond reranking:** the calibrated score is what makes **abstention** possible. "If nothing exceeds τ, refuse" is your primary hallucination defence, and Phase 8 wires it into the answer path. An uncalibrated threshold silently drifts and either refuses everything or nothing.

**Build.** Sweep τ, plot precision vs recall on the Phase 1 unanswerable slice, pick the operating point. Both error directions cost something: too high and you refuse answerable questions; too low and you hallucinate. Write down which you're optimizing for and why.

---

## Step 6.4 — Context assembly as a real component

**Learning goal.** Treat "what goes in the prompt" as an engineered artifact.

**Concept.** Between rerank and generate, decisions that materially move quality:

- **Dedup** near-identical chunks (Phase 5's MMR helps; also dedup by content hash).
- **Merge adjacent** chunks from the same section — two consecutive chunks are usually better as one coherent passage.
- **Ordering** — relevance order vs document order. Test both.
- **Lost in the middle** — models attend more to the beginning and end of long contexts. **Measure it on your system**: place the gold chunk at position 1, 5, and 15 and record answer accuracy at each. If the drop is real, put the best chunk first or last, not in the middle.
- **Dynamic k** — a fixed top-5 makes easy questions pay for 8k tokens. Use the calibrated score distribution: include chunks until the score drops off a cliff.

### Numbers

At $0.013/query (4k context, Sonnet 5), halving context to 2k saves `4000×2/1e6 = $0.008` → $0.005/query. At D2's 86.4M queries/day that's **$690k/day**. Context budgeting is not micro-optimization.

---

## Step 6.5 — LLM reranking

**Concept.** Use `claude-haiku-4-5` to rerank, batched, with structured output. Compare against the cross-encoder on quality, latency, and cost.

Cross-encoder wins on latency and cost per query at steady volume; the LLM reranker wins on flexibility (you can express *criteria* — "prefer more recent circulars") and needs no GPU. Know which situation calls for which.

---

## Phase gate

- [ ] Reranking adds **≥ 10 pts nDCG@10** for **< 250 ms p95**
- [ ] Calibration curve fitted; τ chosen with a written precision/recall justification
- [ ] Lost-in-the-middle measured on your own system, with numbers
- [ ] Dynamic-k implemented and its token savings measured
- [ ] You can derive the cross-encoder FLOP argument from scratch

## Traps

- Reranking a candidate set that never contained the answer.
- Thresholding on uncalibrated scores.
- Fixed top-k forever.
- Comparing rerankers without holding the candidate set constant.
