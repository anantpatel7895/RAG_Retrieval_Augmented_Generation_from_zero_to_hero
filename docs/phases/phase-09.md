# Phase 9 — Observability & Continuous Evaluation

**~10 hrs** · Prerequisite: [Phase 8](phase-08.md) · Next: [Phase 10](phase-10.md)

## Phase learning goals

1. Root-cause a bad answer from a trace in under 5 minutes.
2. Distinguish retrieval, generation, and parsing failures from telemetry alone.
3. Detect quality drift before users report it.
4. Build a CI gate that blocks a real regression without blocking on noise.

---

## Step 9.1 — Tracing

**Learning goal.** Know what to record so that root-causing is a lookup, not an investigation.

### Concept

One trace per query, spanning `route → rewrite → retrieve → rerank → assemble → generate`. Per span: latency, token counts, cost, and **the intermediate artifacts** — the rewritten query, retrieved chunk ids with scores, post-rerank order, final assembled context.

The artifacts are the point. Latency alone tells you *where* time went; it can't tell you *why the answer was wrong*. Given a bad answer, the diagnostic ladder is:

| Check | If it fails |
|---|---|
| Was the gold chunk in the retrieved set? | **Retrieval** problem → parsing, chunking, or embedding |
| Was it in the retrieved set but not the final context? | **Reranking** problem |
| Was it in the final context but the answer is wrong? | **Generation** problem |
| Was it never in the index at all? | **Ingestion** problem — check the DLQ |

That ladder is why you record intermediate state. Without it, all four look identical from the outside.

**Build.** Langfuse (profile `obs` in [docker-compose.yml](../../docker-compose.yml)) or Phoenix.

---

## Step 9.2 — Online metrics

**Concept.** Offline eval tells you about your golden set. Online metrics tell you about reality — and reality drifts.

Track: latency percentiles per stage, retrieval score distributions, abstention rate, citation-validation failure rate, cost/query, cache hit rate, empty-retrieval rate.

**The two most diagnostic are the ones people don't track:**
- **Retrieval score distribution** — a downward shift means your traffic and your corpus have diverged. It moves *before* user complaints.
- **Abstention rate** — a spike means either the corpus changed or the threshold drifted. Either way, users are being refused.

---

## Step 9.3 — Drift detection

**Concept.** Three things drift independently:
- **Query drift** — users start asking about something new. Detect via query-embedding distribution shift.
- **Corpus drift** — documents change; old chunks become stale.
- **Model drift** — a provider updates a model, or you change one.

Track top-1 retrieval score over time as a single cheap health signal. It captures the first two, and it's one number you can alert on.

---

## Step 9.4 — The feedback loop

**Learning goal.** Keep the eval set alive.

**Concept.** Thumbs up/down + freeform, joined to `trace_id`. **Every negative-feedback query gets triaged into the golden set** with a gold answer. This is the only mechanism that keeps your eval representative of real usage — a golden set frozen in Phase 1 measures Phase 1's idea of the problem.

Budget the triage time explicitly. An untriaged feedback queue is worse than none: it looks like you have a process.

---

## Step 9.5 — CI gates

**Learning goal.** Block regressions without blocking on noise.

### Concept

Two tiers, matching Phase 1:
- **Every PR** — fast retrieval eval, no LLM. Seconds, free.
- **Nightly** — full judged eval.

**The statistics matter here.** From [NUMBERS.md §5](../../NUMBERS.md), a 3-point aggregate change on 420 questions has a ±3.9 pt CI — indistinguishable from noise on *unpaired* comparison. A naive "block if any slice drops 3 pts" gate will fire constantly on noise, everyone will learn to override it, and it will be worthless.

Make it meaningful:
1. **Paired comparison** on a frozen question set (McNemar), not two independent proportions.
2. **Report the CI** alongside every delta.
3. Gate on the **lower bound** of the confidence interval, not the point estimate.

**Build.** GitHub Action + `eval/runs/` leaderboard: config hash, per-slice metrics with CIs, cost, latency, for every run. "What changed between run 41 and run 42" must be one command.

---

## Step 9.6 — Cost attribution

**Concept.** The cost logger from Phase 0 pays off here. Break down $/query by stage. You will find one stage dominates — usually generation context, sometimes the reranker at scale, occasionally the judge if someone left a full eval in the inner loop.

You cannot optimize what you haven't attributed, and retrofitting attribution requires the instrumentation to have been there all along.

---

## Phase gate

- [ ] CI blocks a **deliberately introduced** regression (test it — an untested gate is not a gate)
- [ ] A bad answer root-caused from trace alone in < 5 min, **timed**
- [ ] Cost/query broken down by stage
- [ ] Drift signal defined with an alert threshold
- [ ] Feedback → golden set triage process running

## Traps

- Tracing latency but not intermediate artifacts.
- CI gates that fire on noise until everyone ignores them.
- A feedback queue nobody triages.
- Dashboards nobody looks at — every metric needs a decision it drives.
