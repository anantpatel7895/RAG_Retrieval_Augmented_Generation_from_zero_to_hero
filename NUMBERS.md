# NUMBERS — the capacity math

Every formula this project depends on, with worked examples at **our scale** and at **the scale where the answer changes**. The contrast is the point: most architectural decisions in RAG are threshold decisions, and you can only see the threshold by computing both sides.

**Rule:** derive these, don't memorize them. In a design interview you will be given different inputs.

---

## 0. The corpus chain

Everything else hangs off this. Get it wrong and every downstream number is wrong.

```
docs × pages/doc × tokens/page          = total tokens
total tokens ÷ tokens/chunk × overlap   = total chunks
total tokens × 4                        ≈ total characters (English)
```

**Ours:**
```
400 docs × 120 pages × 500 tok/page  = 24,000,000 tokens
24M ÷ 400 tok/chunk                  = 60,000 chunks
× 1.15 (15% overlap)                 ≈ 69,000 chunks   → call it 60k for planning
24M × 4                              ≈ 96 MB of text
```

**Sanity anchors worth memorizing:** 1 token ≈ 4 chars ≈ 0.75 words. A dense A4 page ≈ 500 tokens. A 10-K ≈ 100–150k tokens. This whole corpus (24M tokens) is ~24× a 1M-token context window.

---

## 1. Storage & memory

| Quantity | Formula | 60k chunks | 100M chunks (D2) |
|---|---|---|---|
| Vectors, fp32 | `n × d × 4` | 246 MB | **410 GB** |
| Vectors, fp16 | `n × d × 2` | 123 MB | 205 GB |
| Vectors, int8 | `n × d × 1` | 61 MB | 102 GB |
| Vectors, binary | `n × d ÷ 8` | 7.7 MB | **12.8 GB** |
| HNSW links | `n × 2M × 4` (M=16 → 32 links, uint32 ids) | 7.7 MB | 12.8 GB |
| Payload (text) | `n × chunk_tokens × 4` | 96 MB | **160 GB** |

*(d = 1024 for BGE-M3.)*

### What falls out of this

**Quantization decides whether you have a distributed system.** 410 GB of fp32 vectors does not fit one machine — you shard, and now you own routing, fan-out, partial failure, and rebalancing. Binary quantization brings vectors + graph to ~26 GB, which fits in RAM on a single large node. That is not a memory optimization; it's the difference between one box and a cluster.

**Payload dominates at scale.** 160 GB of chunk text vs 12.8 GB of binary vectors. Keeping payload in RAM is what actually blows your memory budget — this is why production vector stores memory-map payload to disk and keep only vectors and graph hot.

**At our scale none of this matters.** 60k chunks is ~350 MB all-in. We are not using ANN because we need it; we're using it because Phase 5 needs us to understand it. Complexity must be earned.

---

## 2. Compute & query latency

```
Exact (brute force) search:  n × d  multiply-accumulates
HNSW search:                 ≈ ef_search × log₂(n) × d  MACs
1 MAC = 2 FLOP.  A modern core sustains ~10–30 GFLOP/s on SIMD dot products.
```

| Scale | Exact search | HNSW (ef=128) | Speedup |
|---|---|---|---|
| 60k | 61M MAC ≈ **6 ms** | 2.1M MAC ≈ 0.2 ms | 29× |
| 10M | 10.2B MAC ≈ **1 s** | 3.1M MAC ≈ 0.3 ms | ~3,300× |
| 100M | 102B MAC ≈ **10 s** | 3.5M MAC ≈ 0.35 ms | ~29,000× |

### What falls out of this

**HNSW cost is nearly independent of n.** Only the `log₂ n` term grows — 100M is 4× the corpus of 10M but only 14% more work. That single property is why graph indexes won.

**Real-world HNSW is memory-latency bound, not FLOP bound.** The arithmetic says 0.35 ms; you'll measure 1–3 ms. The graph walk is a pointer chase with poor cache locality, so you're paying for random DRAM access, not multiplies. Consequence: **quantization speeds up search**, because smaller vectors mean more of them per cache line.

**Brute force is genuinely fine below ~100k vectors.** If someone proposes a vector DB for 50k chunks, the correct answer is a numpy array.

---

## 3. Throughput & capacity planning

```
Reindex wall time  = total_chunks ÷ (chunks/sec measured on your GPU)
QPS capacity       = cores ÷ per_query_cpu_seconds
Shards needed      = total_index_bytes ÷ per_node_ram_bytes
Ingest rate needed = docs_changed_per_day ÷ seconds_available
```

### Deriving embedding throughput

Don't guess it — derive it, then measure it.

```
Forward-pass cost ≈ 2 × params × tokens   FLOP
BGE-M3 (XLM-R large) = 568M params, chunks ≈ 400 tokens

per chunk = 2 × 568e6 × 400          = 4.5e11 FLOP  (455 GFLOP)
A100 bf16, ~45% of 312 TFLOPS peak   = 140 TFLOP/s
throughput = 1.4e14 ÷ 4.5e11         ≈ 310 chunks/sec
```

Call it **300–800 chunks/sec** on one A100 depending on batching and sequence length. **Measure it in Phase 4 and replace this number.**

| Corpus | @ 310 chunks/sec |
|---|---|
| 60k (ours) | **3 minutes** |
| 10M | 9 hours |
| 100M | **~4 days** |

Four days on one GPU. That single number forces three architectural conclusions: Phase 11's migration **must** be a blue/green backfill rather than a stop-the-world rebuild; you need many GPUs or many days; and re-embedding is expensive enough that `embed_model_version` on every chunk (so you only re-embed what changed) is not optional.

**D1 check** — 10M docs, daily refresh, 5% change rate:
```
500k docs/day × 150 chunks/doc = 75M chunks/day
75M ÷ 86,400 s                 = 868 chunks/sec sustained, all day
868 ÷ 310                      ≈ 3 GPUs just to break even
```
Three GPUs with **zero headroom** for backfills, retries, or a burst. Realistically 5–6, or a queue that can absorb a day of backlog and a documented lag SLO. Note how different this conclusion is from a hand-waved "one GPU should do it" — the arithmetic is the design.

---

## 4. Cost

```
Query cost   = (in_tok × $in + out_tok × $out) ÷ 1e6
Daily cost   = query_cost × QPS × 86,400
```

| Model | $/MTok in | $/MTok out |
|---|---|---|
| `claude-opus-5` | $5 | $25 |
| `claude-sonnet-5` | $2 | $10 |
| `claude-haiku-4-5` | $1 | $5 |

Modifiers: **Batch API −50%** (offline work). **Prompt cache read ≈ 0.1×** input, **cache write ≈ 1.25×**.

**One RAG query** — 4,000 tokens of context, 500 out, on Sonnet 5:
```
(4000 × 2 + 500 × 10) ÷ 1e6 = $0.013
```

**Same query at D2 scale** (1k QPS = 86.4M queries/day):
```
86.4M × $0.013 = $1,123,200 / day
```

That number is the entire argument for context budgeting, model tiering, and caching — and it lands harder derived than asserted. Halve the context and route 70% to Haiku, and it drops by roughly an order of magnitude.

**Eval cost** — the one people forget:
```
Judge on Opus 5: (3000 in × $5 + 500 out × $25) ÷ 1e6 = $0.0275 per question
420 questions                                          = $11.55 per full run
100 runs across the project                            = $1,155
```
This is precisely why the inner loop is the **retrieval-only** eval (no LLM, seconds, free) and the judged eval runs nightly.

---

## 5. Statistics — can your eval even see the change?

The most-skipped math in RAG, and the reason people ship regressions.

```
Standard error of a proportion:  SE = √(p(1−p) ÷ n)
95% confidence interval:         ±1.96 × SE
```

At p = 0.8 (a typical recall@20):

| Question set | SE | 95% CI | Smallest trustworthy change |
|---|---|---|---|
| 420 questions | 2.0 pts | ±3.9 pts | ~8 pts |
| 120 questions | 3.7 pts | ±7.2 pts | ~14 pts |
| 60-question slice | 5.2 pts | ±10.1 pts | **~20 pts** |

### What falls out of this

**A 3-point aggregate "improvement" on 420 questions is noise.** If you celebrate it, you will ship it, and later be unable to explain why quality drifted.

**Slice-level results are far noisier than you want them to be.** A 60-question `table_lookup` slice cannot resolve anything under ~20 points. Either grow the important slices or state the uncertainty every time you quote them.

**Paired testing is the escape hatch.** Both systems run the *same* questions, so you compare per-question outcomes (McNemar) rather than two independent proportions. This cuts the required n dramatically because shared question difficulty cancels out. **Always evaluate paired, always on a frozen question set.** Phase 9's "block on >3 pt regression" CI gate is only meaningful because it's paired.

---

## 6. Latency budget

Every stage gets an allocation, and the sum is the SLO. Enforce it in Phase 11.

| Stage | p95 budget | Notes |
|---|---|---|
| Query rewrite | 200 ms | Haiku, short output; skip when the router says single-hop |
| Embed query | 50 ms | Network RTT dominates, not GPU |
| Hybrid retrieve | 150 ms | Dense + sparse in parallel, then fuse |
| Rerank | 250 ms | Cross-encoder over 50–100 candidates |
| Generate (TTFT) | 800 ms | First token, not full completion |
| **Total TTFT** | **1.5 s** | |

**The stages are not equal.** Rerank and rewrite are the negotiable ones — both are optional per query, and Phase 7's router exists partly to skip them when a query doesn't need them.

---

## 7. Quick reference

| Want | Formula |
|---|---|
| Chunks from a corpus | `docs × pages × tok/page ÷ tok/chunk` |
| Vector RAM | `n × d × bytes` (fp32=4, int8=1, binary=1/8) |
| HNSW graph RAM | `n × 2M × 4` |
| Exact search cost | `n × d` MACs |
| HNSW search cost | `ef × log₂n × d` MACs |
| Reindex time | `chunks ÷ chunks_per_sec` |
| Query $ | `(in×$in + out×$out) ÷ 1e6` |
| Detectable delta | `1.96 × √(p(1−p)/n)` |
