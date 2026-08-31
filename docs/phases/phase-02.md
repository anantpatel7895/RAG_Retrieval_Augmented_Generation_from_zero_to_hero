# Phase 2 — Ingestion & Parsing

**~16 hrs** · Prerequisite: [Phase 1](phase-01.md) · Next: [Phase 3](phase-03.md)

> Where most RAG quality is actually won or lost. Everything downstream operates on what parsing produced — a chunker cannot recover a table that was flattened into word soup.

## Phase learning goals

1. Explain why PDF is a *rendering* format and what that means for extraction.
2. Predict which documents a given parser will silently destroy.
3. Explain why a flattened table breaks retrieval even when the text is technically present.
4. Compute the break-even for a VLM parser against corpus size and re-parse frequency.
5. Design ingestion that is idempotent, resumable, and never silently empty.

---

## Step 2.1 — What a PDF actually is

**Learning goal.** Explain why "extract the text" is not a well-posed problem.

### Concept

A PDF is a **display list**, not a document. It says "draw glyph 'A' at (72, 640) in Helvetica 11pt." It does not say:

- what **reading order** those glyphs form
- what is a heading, a caption, a footnote, or a page number
- which glyphs constitute a **table**, or which cell they belong to
- whether two columns are two columns or one wide paragraph

Everything downstream is **reconstruction**, and every parser reconstructs differently. This is why parser choice moves recall by 20 points.

Three consequences that bite immediately:
- **Multi-column** — naive left-to-right extraction interleaves two columns into nonsense that still looks like prose.
- **Scanned pages** have no text layer. Extraction returns empty. No error. The document silently contributes nothing.
- **Tables** are lines and positioned glyphs. Nothing marks a cell.

### Parser families

| Family | Mechanism | Cost/page | Strong | Weak |
|---|---|---|---|---|
| **Text-layer** (`pymupdf`) | Read glyphs + positions from the PDF | ~0 | Fast, exact for clean digital PDFs | No structure, no tables, dies on scans and columns |
| **Layout model** (`unstructured` hi_res, `docling`) | Object detection over the rendered page → regions, then table-structure models | ~10–100 ms | Real structure, decent tables | Slower, model-dependent, still struggles with nested/spanning cells |
| **VLM** (page image → Claude → markdown) | Multimodal model reads the page like a person | ~$0.005–0.02 | Best on hard layouts, charts, handwriting | 100–1000× cost, latency, non-deterministic |

**Build.** Bake-off on 30 hard pages against a hand-labelled extraction ground truth. Score per family; record cost/page and sec/page.

---

## Step 2.2 — Tables, the single biggest win on this corpus

**Learning goal.** Explain why a flattened table destroys retrieval even when every character survives.

### Concept

Take a capital-adequacy table. Flattened, it becomes:

```
Tier 1 Tier 2 Total 2023 2024 12.4 11.8 3.2 3.9 15.6 15.7
```

Every character is present. The information is gone — the *binding* between row header, column header, and value has been destroyed, and that binding is the answer.

Now embed that. The chunk has no coherent semantic content, so its vector lands nowhere useful. A query like *"what was Tier 1 capital in 2024?"* matches nothing, because the chunk isn't *about* Tier 1 in 2024 — it's about a bag of numbers.

**The fix is dual representation.** Store both:
1. **Row-serialized** — each row becomes a self-contained sentence: *"For fiscal year 2024, Tier 1 capital ratio was 11.8%."* Retrieves precisely; each row is independently meaningful.
2. **Whole table** as one unit, preserving structure — needed for "compare across years" questions where the answer spans rows.

Which one wins depends on question type, which is a preview of Phase 7's routing.

### Numbers — the VLM break-even

```
Corpus:  400 docs × 120 pages = 48,000 pages
pymupdf: ~0 marginal cost
VLM:     48,000 × $0.01 = $480 per full parse
```

Re-parsing quarterly: $1,920/yr. Against an engineer-week of building table heuristics, the VLM is cheap. At D1 scale (10M docs → 1.2B pages) it's $12M and obviously not — you'd route selectively: text-layer by default, VLM only for pages a classifier flags as complex.

**That routing decision is the real lesson.** One parser for all documents is wrong at both ends of the scale.

---

## Step 2.3 — Structure, OCR routing, normalization

**Learning goal.** Ensure no document silently ingests as empty, and no chunk is 30% boilerplate.

**Concept.**
- **Heading hierarchy** — extract it and keep `section_path` on every chunk. Phase 3's structure-aware chunking and Phase 8's citations both depend on it.
- **Reading order & multi-column** — detect column count before extracting.
- **OCR routing** — measure text yield per page (chars ÷ page area). Below threshold → render and route to OCR/VLM. **Measure how many pages in your corpus would have silently ingested as empty.** On a real financial corpus this is usually 2–8%.
- **Boilerplate** — headers, footers, page numbers repeat on every page. Left in, they're 5–15% of every chunk's tokens and they make chunks look similar to each other, which actively hurts retrieval.
- **Normalization** — unicode, ligatures (`ﬁ` → `fi`), hyphenation across line breaks (`obliga-\ntion`), footnote markers glued to words, currency and number formats.

---

## Step 2.4 — Document metadata extraction

**Learning goal.** Treat metadata as a retrieval feature, not decoration.

**Concept.** Extract per document with an LLM: title, `doc_type`, `effective_date`, entities, jurisdiction, version. These become **filter dimensions** in Phase 5 and **conflict resolution** in Phase 8 (2023 vs 2024 circular). Extracting them later means re-processing the corpus.

Use `claude-haiku-4-5` with structured output, batched. Cost: 400 docs × ~2k tokens = 800k tokens ≈ $0.80 with the Batch API. Trivial.

---

## Step 2.5 — Idempotent, resumable ingestion

**Learning goal.** Design ingestion that survives partial failure and never silently drops a document.

**Concept.**
- **Content hashing** → unchanged document is a no-op. Without this every experiment costs a full re-parse.
- **Per-document manifests** with status — a document that failed parsing must be *visible*, not absent.
- **Dead-letter queue** for parse failures, with the error. Silent skips are how corpora develop holes nobody knows about.
- **Deterministic chunk ids** — `hash(doc_id + section_path + chunk_index)`, so re-ingest upserts rather than duplicates.

**Numbers — D1 (exercise after Phase 3).** 10M docs, 5% daily change = 500k docs/day × 150 chunks = 75M chunks/day = **868 chunks/sec sustained**. The manifest checksum diff is what keeps this from being 10M re-embeds.

---

## Phase gate

- [ ] `table_lookup` slice recall@20 improves **≥ 20 pts** over Phase 1 from parsing alone
- [ ] Zero silently-empty documents; DLQ visible and empty or explained
- [ ] Parser bake-off table with cost/page and sec/page recorded
- [ ] Re-running ingest on an unchanged corpus is a no-op (verified)
- [ ] VLM break-even computed for our corpus and at D1 scale

## Traps

- One parser for all document types.
- Discarding layout info (page, bbox) you'll need for Phase 8 citations.
- Parsing configured once and never re-evaluated when the corpus shifts.
- Treating OCR as a fallback you never measure — measure the silent-empty rate.
