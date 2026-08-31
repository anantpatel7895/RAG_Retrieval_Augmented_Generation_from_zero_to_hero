# Phase 8 — Generation, Grounding, Citations & Abstention

**~12 hrs** · Prerequisite: [Phase 7](phase-07.md) · Next: [Phase 9](phase-09.md) · **Design exercise D3 lands here**

## Phase learning goals

1. Explain the difference between *grounded*, *correct*, and *cited* — and why a system can be one without the others.
2. Verify a citation **supports** a claim rather than merely existing.
3. Wire calibrated abstention into the answer path and measure both error directions.
4. Handle conflicting sources by policy rather than by accident.
5. Reconcile streaming with citation validation.

---

## Step 8.1 — Three properties people conflate

**Learning goal.** Diagnose which property failed from a bad answer.

### Concept

| Property | Means | Fails when |
|---|---|---|
| **Grounded** | Every claim traces to retrieved context | Model uses parametric knowledge |
| **Correct** | The claim is true | Retrieved context was wrong or outdated |
| **Cited** | A source marker is attached | Marker points at the wrong chunk |

These are **independent**. A grounded answer can be wrong (the 2023 circular was retrieved instead of 2024). A correct answer can be ungrounded (the model knew it). A cited answer can be both wrong and ungrounded, with a citation that merely *exists*.

**Why this matters for measurement:** groundedness 0.95 with users reporting hallucinations usually means your scorer checks that cited chunks exist, not that they *support* the claim. That is the measurement gap, and it's the most common one.

---

## Step 8.2 — Prompt architecture for RAG

**Concept.** Decisions that measurably move quality:
- **Context block formatting** — clear delimiters, a stable per-chunk id, and inline metadata (`effective_date`, `source`) so the model can reason about recency and provenance.
- **Instruction placement** — test the question before vs after the context. On long contexts, repeating the question after the context often helps (it's the most recent thing).
- **Explicit grounding contract** in the system prompt: answer only from context; if the context is insufficient, say so; cite every claim.
- **Prompt caching** — the system prompt and instructions are stable; put them first with a cache breakpoint, and put the varying context and question after. Verify `cache_read_input_tokens` > 0.

### Numbers

```
System prompt + instructions ≈ 1,500 tokens, stable across every request
Uncached: 1,500 × $2/MTok × 86.4M queries/day = $259k/day  (D2 scale)
Cached at 0.1×:                                 $26k/day
```

Same tokens, one breakpoint, ~$233k/day. And it's free — caching costs nothing when the prefix is genuinely stable. **The failure mode is a volatile prefix**: a timestamp, an unsorted JSON dump, a per-request id placed too early. It silently drops your hit rate to zero and nothing errors.

---

## Step 8.3 — Citations, two ways

**Learning goal.** Compare a prompt-level convention against an API-level guarantee.

### A — Prompt-level

Require `[chunk_id]` markers, then **programmatically verify**:
1. Every cited id exists in the context you sent (catches invented ids)
2. The cited chunk **supports** the claim (needs an entailment check — a second LLM call or an NLI model)

Reject and retry on failure. Check (1) is cheap and catches a lot; check (2) is the one that closes the measurement gap from 8.1.

### B — Native API citations

Pass context as `document` content blocks with `citations: {enabled: true}`. The response comes back split into text blocks, with cited blocks carrying a `citations` array containing `cited_text` and a location (`char_location` for text, `page_location` for PDFs).

The difference is structural: the model isn't *writing* a citation marker it might get wrong, the API is *returning the span it used*. For a corpus where a wrong number is a compliance problem, that's worth a great deal.

Note: incompatible with `output_config.format` — you cannot use native citations and structured output on the same call. That's a real architectural constraint; decide which you need, or split into two calls.

**Build both, measure citation accuracy by hand on 50 samples, and pick with data.**

---

## Step 8.4 — Abstention

**Learning goal.** Wire Phase 6's calibrated threshold into the answer path and measure both directions.

**Concept.** If no chunk exceeds τ, refuse. This is your primary hallucination defence and the reason Phase 1 insisted on 20 unanswerable questions.

**Measure both error directions:**
- False-answer rate on the unanswerable slice (target ≤ 2%)
- False-refusal rate on the answerable slices

Tuning one without the other produces a system that either hallucinates or is uselessly timid. Plot both against τ and choose deliberately.

**Do not use the model's self-reported confidence.** It is not calibrated, and it correlates with fluency rather than correctness. Use the retrieval score distribution — a property of the evidence, not of the prose.

---

## Step 8.5 — Conflicting sources

**Learning goal.** Turn an accident into a policy.

**Concept.** Two retrieved chunks disagree: the 2023 circular says 9%, the 2024 says 11%. Silent behaviour is to pick whichever ranked higher — indefensible in a regulated context.

Policy: **surface the conflict, prefer by `effective_date`, state the basis.** *"As of the 2024 circular, 11% (the 2023 circular specified 9%)."*

This works only because `effective_date` was captured in Phase 2 and carried on every chunk since Phase 0. Build it as an explicit eval slice.

---

## Step 8.6 — Streaming vs validation

**Learning goal.** Resolve a genuine architectural tension.

**Concept.** Users want tokens immediately. Citation validation needs a complete claim. You cannot validate a citation that hasn't finished streaming.

Resolution: **stream the prose, attach citations at block boundaries.** Validate each completed sentence or block as it lands; if validation fails, mark that segment rather than retracting the whole answer. Design the UI contract around this — a citation that appears a beat after its sentence is fine; a retracted paragraph is not.

---

## Design exercise D3

**Multi-tenant RAG API, 200 enterprise tenants, wildly uneven sizes.** → `docs/design/D3-multi-tenancy.md`

Must address: isolation model (collection-per-tenant vs payload filter — cost, blast radius, noisy neighbours), per-tenant quality measurement (each tenant needs their own golden set — who builds it?), cost attribution, cold tenants, and the failure mode where one tenant's 40M-document corpus starves the other 199.

---

## Phase gate

- [ ] Citation accuracy ≥ 0.95, **hand-verified on 50 samples** for *support*, not existence
- [ ] False-answer rate on unanswerable slice ≤ 2%, false-refusal rate recorded
- [ ] Conflict slice handled correctly ≥ 0.9
- [ ] Cache hit rate ≥ 0.8 on repeated-prefix traffic, verified in `usage`
- [ ] D3 written up

## Traps

- Trusting model self-reported confidence.
- Validating that citations exist rather than that they support.
- Abstention tuned in one direction only.
- A volatile prompt prefix silently killing your cache.
