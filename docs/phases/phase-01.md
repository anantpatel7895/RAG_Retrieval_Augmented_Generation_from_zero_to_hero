# Phase 1 — Baseline + Eval Harness

**~16 hrs** · Prerequisite: [Phase 0](phase-00.md) · Next: [Phase 2](phase-02.md)

> The highest-leverage phase in the project. Most people build RAG for three months and *then* try to add eval. That is why their systems never improve measurably.

## Phase learning goals

1. Explain what each retrieval metric actually measures — and which decision each one should drive.
2. Build a golden set that can detect a regression, and explain why synthetic questions are contaminated by default.
3. Calibrate an LLM judge against human labels, and name the three biases that make an uncalibrated judge worse than no judge.
4. Compute whether your eval can even *see* the change you're claiming.
5. State why a single aggregate score is actively harmful.

---

## Step 1.1 — The naive baseline

**Learning goal.** Have a permanent control group, and know exactly which naive choice each later phase is beating.

**Concept.** ~200 lines, deliberately dumb: fixed 512-token chunks, one embedding model, top-5 cosine, stuff into the prompt. No cleverness. Every future claim is measured against this.

The reason it must be *this* dumb: if the baseline already includes reranking and hybrid search, you can never answer "what did reranking buy us?"

**Build.** `rag/pipeline/naive.py`. Frozen after this phase — changes go in new modules.

---

## Step 1.2 — Retrieval metrics, from the formula up

**Learning goal.** Derive each metric and say which decision it drives.

### Concept

Retrieval returns a ranked list. The gold set says which items are relevant. Every metric is a different question about that list.

**recall@k** — `|relevant ∩ top-k| ÷ |relevant|`. *Did the answer make it into the candidate set at all?*
This is **the ceiling on everything downstream**. If the gold chunk isn't in the top-k, no reranker, no prompt, no model can recover it. Fix recall first, always.

**precision@k** — `|relevant ∩ top-k| ÷ k`. *How much noise is in the context?*
Matters at the generation stage, not the search stage. Low precision costs tokens and dilutes attention; it does not make the answer unreachable.

**MRR** — `mean(1 ÷ rank_of_first_relevant)`. *How high did the first correct result land?*
Blind to everything after the first hit. Right metric for "one canonical answer" (a lookup), wrong for "synthesize across sources."

**nDCG@k** — position-weighted, graded relevance:
```
DCG@k  = Σ  rel_i ÷ log₂(i + 1)
nDCG@k = DCG@k ÷ IDCG@k        (IDCG = DCG of the perfect ordering)
```
The `log₂(i+1)` discount encodes "position 1 matters far more than position 10." Needs *graded* labels (0–3), not binary, to mean anything. This is your reranking metric — reranking doesn't change *what* is retrieved, only the order, so recall is flat and nDCG is what moves.

### Which metric drives which decision

| Decision | Metric |
|---|---|
| Parser, chunker, embedding model, hybrid search | **recall@20** |
| Reranker, context ordering | **nDCG@10** |
| Context budget / how many chunks to pass | **precision@k** |
| Single-answer lookup quality | **MRR** |

**Build.** `eval/scorers/retrieval.py`, pure functions, no LLM. Runs in seconds — this is your inner loop, and you'll run it hundreds of times.

---

## Step 1.3 — The golden set

**Learning goal.** Build a question set that can detect a regression, and know how it is contaminated.

### Concept

120 hand-built questions, each:

```json
{"qid": "", "question": "", "answer": "",
 "gold_doc_ids": [], "gold_chunk_spans": [],
 "question_type": "factoid|table_lookup|multi_hop|comparison|summarization|unanswerable",
 "difficulty": 1, "must_abstain": false, "notes": ""}
```

**20 must be unanswerable.** A system that cannot say "I don't know" is not production-ready, and without these you have literally no measurement of hallucination.

**Synthetic expansion to ~420, and its contamination problem.** Generate with `claude-opus-5`, hand-verify 20%.

The trap: a question generated *from a chunk* is trivially retrievable *by that chunk* — it shares vocabulary, phrasing, and framing. Your recall looks excellent and means nothing. Two mitigations:
1. Generate from **document-level** context, not chunk text.
2. Explicitly instruct paraphrasing away from source vocabulary, then spot-check that the question doesn't share rare terms with the gold chunk.

**Why slices are mandatory.** A change that lifts factoid 4 pts and drops table_lookup 15 pts nets out as +1 on the aggregate. You'd ship it. Every result is reported per `question_type`.

### Numbers

From [NUMBERS.md §5](../../NUMBERS.md) — at p = 0.8:

| Set | 95% CI | Smallest trustworthy change |
|---|---|---|
| 420 questions | ±3.9 pts | ~8 pts |
| 60-question slice | ±10.1 pts | **~20 pts** |

**A 3-point aggregate "improvement" on 420 questions is noise.** Slice-level moves under ~20 pts on a 60-question slice are noise. This single calculation should change how you read every result for the rest of the project — and it's the argument for **paired** evaluation (same frozen question set, per-question comparison, McNemar), which cuts the required n dramatically because shared question difficulty cancels.

**Build.** `eval/datasets/golden_v1.jsonl` (versioned — never edited in place; new version, new file).

---

## Step 1.4 — LLM-as-judge, calibrated

**Learning goal.** Explain the three judge biases, and prove your judge agrees with you before trusting it.

### Concept

Some things (answer correctness, groundedness) have no string-match scorer. You need a model to grade. An **uncalibrated judge is worse than no judge** — it gives you a number, so you stop thinking, and the number is wrong.

**Three biases to design against:**

| Bias | What happens | Mitigation |
|---|---|---|
| **Verbosity** | Longer answers score higher regardless of correctness | Rubric anchors on factual coverage; test with a deliberately padded answer |
| **Position** | In pairwise comparison, one position wins more | Randomize order, or score absolutely against a rubric instead of pairwise |
| **Self-preference** | A model prefers text generated by itself | Judge ≠ generator. Generate with Sonnet, judge with Opus |

**Calibration is the step everyone skips.** Hand-label 50 items, run the judge, compute **Cohen's κ**:

```
κ = (p_observed − p_chance) ÷ (1 − p_chance)
```

κ < 0.6 means the *rubric* is broken, not the model. Rewrite the rubric with explicit anchors ("score 2 = every factual claim is supported but one relevant qualifier is missing") and re-measure. Do not proceed on an uncalibrated judge.

**Structured output.** Use `output_config.format` so the judge returns a validated object, not prose you regex. A judge that occasionally emits unparseable output silently drops samples and biases your result.

### Numbers

```
Judge on Opus 5: (3000 in × $5 + 500 out × $25) ÷ 1e6 = $0.0275/question
420 questions = $11.55 per full run · 100 runs = $1,155
```

That's the entire argument for a two-tier eval: **retrieval-only in the inner loop (free, seconds), judged eval nightly.**

**Build.** `eval/scorers/judge.py` + `eval/calibration/`.

---

## Step 1.5 — The harness

**Learning goal.** Make "what changed between run 41 and run 42" a lookup, not an investigation.

**Concept.** One command → a `run_id`, and an artifact containing: full serialized config, per-slice metrics, latency percentiles, $/query, and a diff against any previous run.

Two tiers:
- **fast** — retrieval metrics only, no LLM. Seconds. Runs on every change and in CI.
- **full** — adds judged generation metrics. Minutes and dollars. Nightly.

**Build.** `eval/harness.py`, artifacts in `eval/runs/<run_id>/`.

---

## Phase gate

- [ ] Fast eval on 420 questions completes in < 60 s; full eval < 10 min
- [ ] Judge κ ≥ 0.6 against your hand labels — **recorded**, not assumed
- [ ] Baseline recorded in `docs/decisions/0001-baseline.md`; these numbers are sacred
- [ ] Every result reported per slice, with confidence intervals
- [ ] You can state, from memory, the smallest change your eval can detect

## Traps

- Golden sets built only from questions the system already answers.
- Judge and generator being the same model (self-preference).
- No unanswerable questions → zero hallucination measurement.
- Quoting aggregate scores without CIs.
- Editing the golden set in place — version it, or your history becomes uninterpretable.
