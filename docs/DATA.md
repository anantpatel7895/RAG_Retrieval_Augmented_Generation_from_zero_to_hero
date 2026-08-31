# Understanding the Data

Every number here is derived, step by step, from something measurable. Nothing is quoted without its arithmetic.

---

## 1. The five units, and why they get confused

Most confusion about "how big is this corpus" comes from silently switching units. There are five, and they are **not** proportional to each other.

| Unit | What it is | Why it's not the others |
|---|---|---|
| **Documents** | One filing, one circular, one PDF | Sizes vary by 100× |
| **Bytes** | What's on disk | Mostly markup, not content |
| **Characters** | Text after stripping tags | ~12% of bytes for an SEC filing |
| **Tokens** | What the model counts and bills | ~1 token per 4 characters |
| **Chunks** | Unit of retrieval — our choice | 400 tokens each, so tokens ÷ 400 |

Conversions we use throughout:

```
1 token       ≈ 4 characters   ≈ 0.75 words
1 dense page  ≈ 500 tokens
1 chunk       = 400 tokens     (our choice, revisited in Phase 3)
```

**The trap:** 12.9 MB on disk sounds like a big document. It is — but only 11.7% of it is text. Quote bytes when planning disk, tokens when planning cost, chunks when planning the index. Mixing them silently is how capacity estimates end up 6× off.

---


## What is EDGAR?

**EDGAR** stands for *Electronic Data Gathering, Analysis, and Retrieval*.It is the online database system used by the U.S. Securities and Exchange Commission (SEC) to collect and store corporate documents, including annual Form 10-K filings.

## 2. What a 10-K is

A Form 10-K is a comprehensive annual report required by the U.S. Securities and Exchange Commission (SEC) detailing a company's financial performance and business operations. It is distinct from a standard annual report by offering legally binding, dense financial disclosures rather than marketing-heavy summaries. Its fixed structure:

| Item | Contents |
|---|---|
| 1 | Business — what the company does |
| **1A** | **Risk Factors** — long prose, the biggest section |
| 3 | Legal Proceedings |
| **7** | **MD&A** — management's discussion, dense with tables and numbers |
| **8** | **Financial Statements** — the tables, plus footnotes ("Notes") |

For a bank that means capital adequacy ratios, credit exposures, derivative positions, loan loss provisions. Exactly the finance/compliance domain this project targets.

---

## 3. What we actually downloaded — measured

**60 documents: 15 US bank holding companies × 4 fiscal years (FY2022–FY2025).**
Measured by `rag corpus stats`, not estimated:

```
documents                  60
bytes on disk          580.8 MB
total tokens       15,450,946
chunks @ 400 tok       38,627
<table> elements       20,839

token distribution
  min / p50 / max   148,914 / 246,092 / 434,583
  mean              257,515
  top 20% of docs   30% of all tokens
```

Per company, FY2022 → FY2025 (thousands of tokens):

| Company | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| Citigroup | 434 | 431 | 434 | 411 |
| JPMorgan Chase | 368 | 392 | 385 | 378 |
| Goldman Sachs | 347 | 351 | 348 | 335 |
| Bank of America | 315 | 318 | 314 | 314 |
| PNC Financial | 276 | 272 | 272 | 267 |
| Wells Fargo | 237 | 234 | 215 | 265 |
| Morgan Stanley | 259 | 261 | 262 | 262 |
| Capital One | 219 | 236 | 246 | 274 |
| State Street | 256 | 257 | 251 | 238 |
| BNY Mellon | 210 | 219 | 224 | 212 |
| Truist | 192 | 207 | 209 | 212 |
| Citizens Financial | 176 | 182 | 180 | 183 |
| American Express | 164 | 174 | 179 | 179 |
| Charles Schwab | 169 | 181 | 185 | 166 |
| US Bancorp | 224 | 158 | 151 | 148 |

**These are the corrected figures.** The first fetch produced a corpus where Wells
Fargo read 102/97/84/**23**k, BNY Mellon ~63k and US Bancorp ~64k — because for
those three filers we had downloaded the wrong file. §6 is how that was found.

### File naming

```
edgar_jpmorgan_FY2025_10k_008131.htm
  │      │        │      │     └─ accession tail: makes an original and a later
  │      │        │      │        10-K/A amendment distinguishable
  │      │        │      └─ form
  │      │        └─ fiscal year REPORTED ON, not filed (FY2025 is filed Feb 2026)
  │      └─ company slug
  └─ source
```

`doc_id` is the same string, so a chunk's provenance is legible in a trace without
a lookup. Rationale in [ADR 0001](decisions/0001-corpus-and-naming.md).

**Where text_ratio comes from.** For JPMorgan:

```
raw HTML                      12,927,325 characters
after stripping <tags>         1,513,182 characters
ratio = 1,513,182 / 12,927,325 = 0.117 = 11.7%
```

88% of the file is inline XBRL markup and styling — machine-readable financial tagging, not prose. Then:

```
1,513,182 characters ÷ 4 characters/token = 378,295 tokens
```

That's how `est_tokens` in the manifest is produced.

---

## 4. The public dataset — every number derived

[TeraflopAI/SEC-EDGAR](https://huggingface.co/datasets/TeraflopAI/SEC-EDGAR) on HuggingFace. Two figures come straight from its README tables:

```
10-K filings in the dataset:      223,275
10-K tokens in the dataset:    14,518,876,137
```

**Derivation 1 — average size of a 10-K:**

```
14,518,876,137 tokens ÷ 223,275 filings = 65,027 tokens per filing
```

**Derivation 2 — what's in one downloadable shard.** The `10-K/` folder holds 149 parquet files totalling 91.8 GB (counted via the HuggingFace API, not the README):

```
filings per shard = 223,275 filings ÷ 149 shards = 1,498 filings
size per shard    = 91.8 GB ÷ 149 shards         = 0.62 GB = 616 MB
```

So **one 616 MB download gives ~1,498 complete 10-K filings.** That's the number that makes it useful for Phase 5's scale test.

**Derivation 3 — chunks from one shard:**

```
1,498 filings × 65,027 tokens = 97,410,446 tokens
97,410,446 tokens ÷ 400 tokens/chunk = 243,526 chunks
```

A quarter of a million chunks from a single 616 MB download. Our own fetcher, at 7 requests/second, would need days to collect that — and the dataset authors say a full crawl took over 10 days.

---

## 5. Why our corpus is 3.5× the average

```
our mean (60 docs):    229,490 tokens
their mean (223k docs): 65,027 tokens
ratio:      229,490 / 65,027 = 3.5×
```

> **Correction.** Measured on the first 3 filings this ratio looked like **5.7×**.
> With 60 documents it is **3.5×**. The first three were JPMorgan, Bank of America
> and Citigroup — the three largest, i.e. the extreme tail of even our own
> deliberately large-bank sample. A 3-document sample overstated the effect by 63%.
>
> The lesson is not "the estimate was wrong." It is that **a small sample from a
> skewed distribution is biased in the direction you sampled**, and the bias is
> large enough to change decisions. This is the same error, one level down, as the
> one this section is about.

**Both figures are correct. It's a sampling effect, and it's the most important idea on this page.**

Those 223,275 filings include **every US public company** — thousands of tiny ones. A small shell company's 10-K might be 5,000 tokens. JPMorgan's is 378,295. The distribution is enormously skewed, and we deliberately sampled the top of it: large bank holding companies.

An analogy: the average height of all adults is ~1.7 m. The average height of NBA centres is ~2.1 m. Neither number is wrong. Quoting one while sampling the other is the error.

### Why this changes real decisions

Capacity planning multiplies **documents × tokens-per-document**. Pick the wrong figure and everything downstream inherits the error:

| Assumption | 400 documents | Chunks @ 400 tok |
|---|---|---|
| 65,027 tok/doc (all filers) | 26.0M tokens | 65,000 |
| 229,490 tok/doc (large banks, measured) | 91.8M tokens | 229,500 |

**3.5× more chunks, propagating into index memory, embedding time, and cost.** And for contextual retrieval (Phase 3) the error is *squared*, because cost scales as `D²/c` — so 3.5× longer documents cost roughly **12×** more to contextualize.

### The rule this gives you

> For a skewed distribution, **the mean is not the typical value.** Plan with the measured distribution — median plus the tail — not with an average. And always state which slice of the population you sampled.

This is why [rag/cli.py](../rag/cli.py)'s `corpus stats` reports `min / p50 / max` and "top 20% of docs hold X% of all tokens", instead of just a mean.

---

## 6. Data quality — what the profiler found

`rag corpus profile` exists to make **silent under-delivery** visible: pipelines
that return plausible-looking *less* rather than raising. It found three problems.

### 6a. Hollow documents — 12 found, 12 fixed

Some filers satisfy Item 8 by **pointing at a separate exhibit**. US Bancorp's 10-K
said, verbatim:

> *"Information in response to this Item 8 can be found in the 2025 Annual Report on pages 60 to 134..."*

The financial statements were **not in the document we downloaded**. Measured before
the fix:

| Company | Tokens | Tables | Tables per 10k tok |
|---|---|---|---|
| US Bancorp | 66k | 22 | **3.3** |
| BNY Mellon | 63k | 27 | **4.3** |
| Wells Fargo FY2022–24 | 84–102k | 42 | **4.1–5.0** |
| Wells Fargo FY2025 | 23k | 40 | 17.1 *(passed density, failed peer check)* |
| *A real 10-K* | 230k+ | 313 (p50) | **15–18** |

**Root cause: EDGAR's own `primaryDocument` field points at the wrong file.** Every
filing has two HTML documents sharing a stem, and for these filers the API named the
small wrapper:

| Filer | `primaryDocument` | Actual content |
|---|---|---|
| Wells Fargo FY2025 | `wfc-20251231_d2.htm` — 360 KB | `wfc-20251231.htm` — **11.6 MB** |
| Wells Fargo FY2024 | `wfc-20241231.htm` | `wfc-20241231_d2.htm` — **10.9 MB** |
| US Bancorp FY2025 | `usb-20251231.htm` — 973 KB | `usb-20251231_d2.htm` — **5.4 MB** |
| US Bancorp FY2022 | `d410791d10k.htm` — 1.0 MB | `d410791dex13.htm` — **6.7 MB** |
| BNY Mellon FY2025 | `bk-20251231.htm` — 1.3 MB | `bk-20251231_d2.htm` — **7.2 MB** |

Look at the first two rows: **Wells Fargo's convention inverted between FY2024 and
FY2025.** Same company, consecutive years, opposite meaning for `_d2`. No name-based
rule survives that.

**The fix:** fetch each filing's `index.json` and take the largest `.htm`, excluding
`R\d+.htm` XBRL viewer renderings. One rule, no special cases.

An intermediate version *also* required the candidate to share the primary's
filename stem — and it silently failed on US Bancorp FY2022, where the filing agent
named the documents `d410791d10k.htm` and `d410791dex13.htm`. The clever rule kept
the 1 MB wrapper. **Prefer a simple rule plus a verification pass over a complicated
rule you have to trust** — which is exactly what the profiler provides.

**Result:** 0 of 60 hollow. Corpus tokens 13.77M → 15.45M, tables 16,655 → 20,839,
minimum document 23,354 → 148,914 tokens.

### 6b. Three detectors, three blind spots

Finding those took three independent signals, because each one alone was wrong:

| Detector | Caught | Missed | Why |
|---|---|---|---|
| **Text pattern** — Item 8 points elsewhere | US Bancorp, BNY Mellon | Wells Fargo FY2025 | It never uses the phrase |
| | | *False-positived* Capital One | Matched its table-of-contents entry — 501 tables, obviously a real report |
| **Table density** < 8 per 10k tokens | The hollow filers | Wells Fargo FY2025 | 40 tables over 23k tokens reads as *high* density |
| **Peer-relative size** vs same company's other years | Wells Fargo FY2025 (23k vs 97k median) | — | Adjacent years of one filer are the tightest available control |

**The generalizable rules:**
- When a cheap textual signal and a structural one disagree, trust the structural one and treat the disagreement as a bug in your detector.
- Absolute thresholds can't catch what peer-relative comparison can.
- No single signal is sufficient. That is the lesson, not any particular threshold.

### 6c. Near-duplicate overlap between consecutive years — 41%

Measured with hashed 8-word shingles, Jaccard similarity:

| Company | 22→23 | 23→24 | 24→25 |
|---|---|---|---|
| State Street | 49% | 52% | 50% |
| Goldman Sachs | 45% | 50% | 48% |
| American Express | 48% | 48% | 45% |
| JPMorgan Chase | 41% | 41% | 41% |
| **Mean across corpus** | | | **41%** |

> **Correction.** I earlier claimed consecutive 10-Ks are "70–85% identical." Measured, it is **41%** of 8-word shingles. I asserted that number from intuition and it was substantially too high.

41% is still a serious retrieval problem: roughly **half the text is verbatim identical across years.** Query *"what are the credit risk factors"* and four near-identical chunks from four years compete. Which is correct depends on whether the user meant "now" or "in 2023" — and nothing in the text distinguishes them.

This is what makes **Phase 6 dedup/MMR** and **Phase 8 conflict resolution** real rather than hypothetical, and it's why `effective_date` is carried on every chunk.

### 6d. Year coverage — 15/15 companies have ≥2 years

The conflict slice is viable for every company.

---

## 7. One document, end to end

Following JPMorgan's filing through the whole pipeline makes the units concrete:

```
source file on disk                              12,927,325 bytes   (12.9 MB)
  └─ strip HTML tags                    →         1,513,182 characters  (11.7%)
     └─ ÷ 4 chars/token                 →           378,295 tokens
        └─ ÷ 400 tokens/chunk           →               946 chunks
           └─ × 1024 dims × 4 bytes     →         3,874,816 bytes of vectors  (3.9 MB)
```

Worth noticing: the **vectors are 3.9 MB while the source is 12.9 MB.** Embeddings are a *compression* of this document, roughly 3:1 against raw bytes — but against the 1.5 MB of actual text they're a 2.6× *expansion*. Which comparison you quote depends on what you're planning: disk, or index memory.

---

## 7. Which data serves which purpose

| Job | Source | Why |
|---|---|---|
| Working corpus (Phases 1–4, 6–9) | **Our fetcher** | ~200 documents whose provenance we control. Our Feb-2026 filings postdate more model training cutoffs, so they're less contaminated for the golden set |
| Parser bake-off (Phase 2) | **PDFs** — RBI, OCC, annual reports | EDGAR is HTML/SGML. No scanned pages, no multi-column PDFs, no tables drawn as glyphs. The hard parsing problems are not in any SEC dataset |
| Scale test (Phase 5) | **TeraflopAI shards** | Need volume, not mess. One shard ≈ 243k chunks |
| Golden set scaffold (Phase 1) | **FinQA / TAT-QA / ConvFinQA** | Gold Q&A pairs, including multi-turn |

**The principle:** a prepared dataset removes the problem you are trying to learn. Someone already solved parsing for the TeraflopAI set — you inherit their decisions and cannot measure them. That is fatal for Phase 2 and perfect for Phase 5, where you only want a lot of realistic vectors.

---

## 9. Where the corpus stands against the plan

[PLAN.md](../PLAN.md) requires ≥30 documents whose answer lives in a table, and ≥20
needing two documents.

| Requirement | Status |
|---|---|
| ≥30 documents with table-borne answers | **Met** — 60 documents, 20,839 tables, p50 of 344 per document |
| ≥20 multi-document questions | **Met** — 15 companies × 4 years supports cross-year and cross-company comparisons |
| Conflict pairs (same fact, different years) | **Met** — 15/15 companies have ≥2 years, 43% text overlap |
| Every document contains its financial statements | **Met** — 0 hollow after the §6 fix |
| Hard *PDF* parsing (scanned, multi-column, tables as glyphs) | **Not met** — EDGAR is HTML. Needs RBI / OCC / annual-report PDFs |

## 10. Open items

- [ ] **Add PDF sources** — RBI circulars, OCC handbooks, annual reports. Phase 2's difficulty lives here and no SEC dataset provides it. *Blocks Phase 2, not Phase 1.*
- [ ] **Update [NUMBERS.md](../NUMBERS.md)** — replace the 60k tokens/doc planning figure with the measured 257,515 and the real distribution
- [ ] **Measure `chunks/sec`** on the GPU box (Phase 4) — the last estimated number in the capacity chain

**Standing warning.** Every correction in this document came from enlarging the
sample or measuring instead of assuming:

| Claim | First stated | Measured | Off by |
|---|---|---|---|
| Tokens per document | 60,000 | 257,515 | 4.3× |
| Corpus vs EDGAR average | 5.7× (n=3) | 3.5× (n=60) | 63% |
| Consecutive-year overlap | 70–85% | 43% | ~2× |
| Documents containing statements | 60/60 assumed | 48/60 actual | 12 documents |

The 60 documents are still all large US banks in one format. Treat every number here
as conditional on that sample until PDFs and smaller filers are added.
