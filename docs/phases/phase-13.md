# Phase 13 — Capstone

**~16 hrs** · Prerequisite: [Phase 12](phase-12.md)

## Phase learning goals

1. Present the whole system as a coherent set of decisions, each traced to a number.
2. Design a system 10× larger on a whiteboard, cold.
3. Hand the work to another engineer and have them succeed with it.

---

## Step 13.1 — Ship the system

Deployed service with everything integrated: FastAPI, SSE streaming, auth, tenancy, rate limiting, health checks, graceful degradation per [Phase 11](phase-11.md).

---

## Step 13.2 — The evidence pack

**The full eval report.** Baseline → final, per slice, with confidence intervals, and the delta **attributable to each phase**. This is the artifact that distinguishes engineering from activity: not "we built a RAG system" but "parsing bought 22 points on table lookup, hybrid bought 9 on identifiers, reranking bought 11 nDCG for 240 ms."

**The architecture document.** Every decision traced to a number, including the rejections — the two-plus techniques killed in [Phase 10](phase-10.md), the framework decision from [Phase 11](phase-11.md), pgvector vs Qdrant from [Phase 5](phase-05.md).

**The runbook.** Symptom → diagnosis → action for: p95 latency spike, quality regression, corpus refresh, model deprecation, index corruption, cost spike, cross-tenant incident. Grown from [Appendix A](../../PLAN.md) across every phase.

**Load test report, cost model, red-team report.**

---

## Step 13.3 — The talk

15 minutes: *"how we took recall@20 from X to Y and TTFT from A to B, and what we chose not to build."*

The "chose not to build" half matters as much as the other. Anyone can list techniques they added.

---

## Step 13.4 — Mock system design interview

45 minutes, whiteboard, recorded:

> **Design a RAG system for a global bank — 50M documents, 5k QPS, 12 regions, regulated.**

**The bar:**
- Capacity math **on the fly** — index memory, shard count, GPU count for ingest, cost/day. From [NUMBERS.md](../../NUMBERS.md), derived not recalled.
- Named failure modes with detection and degradation for each.
- At least three **rejected** alternatives with reasons.
- Compliance treated as an architectural driver, not a bolt-on.
- Knowing which numbers you'd need to measure before committing.

**Self-score against the recording.** The failure mode to watch for: describing components instead of making decisions. A design is a set of choices with reasons, not a diagram.

---

## Step 13.5 — Handover test

Give the repo to another engineer. They must deploy it and answer 5 corpus questions **using only your documentation**.

Whatever they get stuck on is the gap in your docs — fix it. This is the honest test of whether the work is real.

---

## Final gate

- [ ] Service deployed and answering
- [ ] Eval report with per-phase attribution and CIs
- [ ] Architecture doc, runbook, load test, cost model, red-team report
- [ ] Talk delivered
- [ ] Mock interview recorded and self-scored
- [ ] Handover test passed
- [ ] [question_answer.md](../../question_answer.md) complete across all 14 phases
- [ ] All five design exercises in `docs/design/`
