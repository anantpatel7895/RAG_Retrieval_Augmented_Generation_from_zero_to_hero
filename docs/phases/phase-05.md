# Phase 5 — Retrieval, Hybrid Search & ANN Internals

**~16 hrs** · Prerequisite: [Phase 4](phase-04.md) · Next: [Phase 6](phase-06.md) · **Design exercise D2 lands here**

## Phase learning goals

1. Explain *why* dense retrieval fails on exact identifiers — from the training objective, not by assertion.
2. Write down BM25 and RRF and explain every term.
3. Tune the recall/latency curve yourself and know which knob to turn at 3am.
4. Explain why filtered ANN is hard and how a real system solves it.
5. Decide sharding vs replication from memory and QPS math.

---

## Step 5.1 — Why dense retrieval fails on exact terms

**Learning goal.** Predict which query classes dense search will miss, before testing.

### Concept

A dense embedding is a **lossy compression of meaning into 1024 floats**. Compression discards what the training objective didn't reward — and exact tokens are precisely what gets discarded:

- **Rare identifiers** (`RBI/2024-25/107`, error code `ERR_4032`, a SKU) appear too rarely in training for the model to have learned a distinct representation. They're absorbed into a generic "looks like an identifier" region of the space.
- **Subword fragmentation** — the tokenizer splits `ERR_4032` into pieces that individually mean nothing; pooling averages them into noise.
- **Averaging** — pooling over 400 tokens means one rare, decisive term contributes ~0.25% of the vector. The signal is mathematically drowned.

**Lexical search has the exact opposite profile.** BM25 doesn't know what words mean, so it can't be fooled about *which* word appeared. Rare terms get **high IDF** — the rarer the term, the more weight it carries. Dense and sparse fail on disjoint query classes, which is why fusing them beats either.

### BM25, term by term

```
score(D,Q) = Σ_i  IDF(qᵢ) · [ f(qᵢ,D) · (k₁+1) ] / [ f(qᵢ,D) + k₁·(1 − b + b·|D|/avgdl) ]

IDF(qᵢ) = ln( (N − n(qᵢ) + 0.5) / (n(qᵢ) + 0.5) + 1 )
```

- `f(qᵢ,D)` — term frequency. More occurrences, higher score.
- `k₁` (~1.2–2.0) — **saturation**. The 10th occurrence adds much less than the 2nd. Without it, keyword stuffing wins.
- `b` (~0.75) — **length normalization**. Without it, long documents win by containing everything.
- `IDF` — rare terms dominate. This is the whole reason BM25 finds identifiers.

---

## Step 5.2 — Fusion

**Learning goal.** Explain why RRF is robust and when weighted fusion beats it.

### Concept

Dense scores (cosine, ~0–1) and BM25 scores (unbounded, corpus-dependent) are **not comparable**. Adding them is meaningless; normalizing them is fragile because BM25's range shifts with the query.

**Reciprocal Rank Fusion** sidesteps this by discarding scores and using only **ranks**:

```
RRF(d) = Σ_systems  1 / (k + rank_s(d))        k ≈ 60
```

- Rank-based → immune to score-scale mismatch. Nothing to normalize, nothing to tune per corpus.
- `k` damps the top: with k=60, rank 1 scores 1/61 and rank 2 scores 1/62 — nearly equal. Small `k` makes it winner-take-all; large `k` flattens toward a plain vote.
- A document ranked well by **both** systems beats one ranked brilliantly by one — which is exactly the behaviour you want from an ensemble.

**Weighted score fusion** can beat RRF when you *know* one system is stronger and your scores are genuinely calibrated. Test both; RRF is the safe default and usually within a point.

---

## Step 5.3 — Filtered ANN: the hard problem

**Learning goal.** Explain the three strategies and why the naive ones fail.

### Concept

*"Nearest 10 where `tenant_id = X` and `effective_date > 2024`"*

| Strategy | Mechanism | Fails when |
|---|---|---|
| **Post-filter** | Search top-N, then drop non-matching | Filter is selective — ask for 10, get 2. Worse as the filter narrows. |
| **Pre-filter + brute force** | Get matching ids, scan them exactly | Match set is large — 2M matches is a linear scan |
| **Filter during traversal** | Skip non-matching nodes while walking the graph | **Breaks the graph.** The path to the answer may run through excluded nodes; the walk stalls in a disconnected region and recall collapses *silently* |

**How real systems solve it:** estimate the filter's **cardinality** first, then choose — brute force when the match set is small, filtered traversal when it's large — plus build additional links so common filtered subgraphs stay connected.

**Payload indexes are what make cardinality estimation possible.** Without an index on the field, the engine can neither estimate nor filter efficiently, and it degrades to a scan. This passes in dev at 60k and dies at 5M. It is the single most common vector-DB production surprise.

> Phase 12's ACL enforcement is this exact mechanism with a security consequence: a filter applied *after* search is not an access control.

---

## Step 5.4 — Tuning the curve

**Learning goal.** Produce the recall-vs-latency curve yourself and pick a defensible operating point.

**Build.** Sweep `ef_search` (and `m` / `ef_construct` at build time; `nprobe` if you also try IVF). For each: measure recall@20 against an **exact brute-force ground truth** you compute yourself, plus p50/p95 latency.

Plot recall on y, latency on x. **This curve is the most important artifact of the phase.** Pick the operating point and write down why.

Remember from Phase 0: only `ef_search` is tunable at query time without a rebuild. That makes it the 3am knob — and it means your incident runbook says "drop `ef_search` to shed latency at a known recall cost," which is a real, quantified degradation lever rather than a guess.

---

## Step 5.5 — Scale test & the pgvector comparison

**Build.**
1. Duplicate the corpus to 1M–5M vectors. Measure ingest time, index build time, memory, p50/p95, and behaviour under concurrent load. **Verify the params you tuned at 60k still hold** — they usually don't.
2. Same data and queries in **pgvector**. Measure honestly. Write the ADR: at what corpus size and query pattern is "just use Postgres" the right production answer? For many real systems it is, and the ability to say so is worth more than defending your tool choice.

---

## Step 5.6 — MMR and diversity

**Concept.** Retrieval returns 8 near-identical chunks from the same section; your context window is full of one fact repeated. **Maximal Marginal Relevance** trades relevance against novelty:

```
MMR = argmax_d [ λ · sim(d, q) − (1−λ) · max_{d'∈selected} sim(d, d') ]
```

λ=1 is pure relevance; λ=0 is pure diversity. Typically 0.5–0.7. Cheap, and it directly improves the precision of what reaches the model.

---

## Design exercise D2

**100M chunks, 1k QPS, p95 < 200 ms.** → `docs/design/D2-retrieval-at-scale.md`

Must address: memory math (410 GB fp32 vs 12.8 GB binary — and what that does to node count), sharding vs replication (sharding splits the index for memory; replication duplicates it for QPS — you likely need both), scatter-gather fan-out and tail latency, hot-shard skew from uneven tenants, cache hit modelling, and what degrades first under 2× load.

**Anchor:** at 1k QPS, per-query CPU budget across a 32-core node is `32 ÷ 1000 = 32 ms` of CPU per query. HNSW search costs ~1–3 ms. Where does the other 29 ms go?

---

## Phase gate

- [ ] Hybrid beats dense-only by **≥ 8 pts recall@20**
- [ ] Exact-identifier slice goes from near-0 to ≥ 0.9
- [ ] Published recall-vs-latency curve with a chosen operating point and written justification
- [ ] p95 < 150 ms at 1M vectors
- [ ] pgvector ADR written honestly
- [ ] D2 written up

## Traps

- Tuning `ef_search` without measuring recall against exact ground truth.
- Post-filtering.
- Assuming params tuned at 60k hold at 5M.
- Forgetting payload indexes.
