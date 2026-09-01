# HTML vs PDF: what each format gives you, what it destroys, and what it costs to fix

Foundation document for [Phase 2](phases/phase-02.md).

> **Every number here is reproducible.** The research lives in
> **[`notebooks/02_parsing_html_vs_pdf.ipynb`](../notebooks/02_parsing_html_vs_pdf.ipynb)**
> and the measurement logic in
> [`rag/ingest/format_probe.py`](../rag/ingest/format_probe.py) (tested in
> `tests/test_format_probe.py`). Re-run the notebook to regenerate anything below.
>
> This document was originally written from ad-hoc shell scripts, which made its
> numbers unverifiable and frozen. That is now the standing rule: **research happens
> in a notebook, logic lives in `rag/`, and the document cites the notebook.**

Measurements are on JPMorgan's FY2025 10-K (12.9 MB, 378,295 tokens of text, 679
tables) and on 55 HTML/PDF pairs covering 14 banks × 4 fiscal years.

---

## 1. The fundamental difference

| | HTML | PDF |
|---|---|---|
| What it is | A **markup** language — describes *what things are* | A **display list** — describes *where ink goes* |
| A table is… | `<table><tr><td>` — declared | Lines and positioned glyphs — inferred |
| Reading order | Document order | Not represented at all |
| Headings | `<h1>`, or at least styled elements | A font size you have to guess about |
| Extraction is… | **Reading** | **Reconstruction** |

That last row is the whole story. With HTML you read a structure someone already
declared. With PDF you rebuild a structure that was thrown away at print time.

**Consequence:** parser choice barely matters for HTML and dominates everything for
PDF. This is why our all-HTML corpus makes Phase 2 look trivial, and why that is
dangerous rather than convenient.

---

## 2. HTML: the structure is there, but it is not the structure you want

### 2a. Most of the file is not content

```
raw HTML                12,927,325 chars
text after tag-strip     1,513,182 chars   (11.7%)
inline XBRL tags             8,738
```

**88% of the bytes are markup.** Inline XBRL wraps every reported figure in a
semantically-tagged element:

```html
$<ix:nonFraction unitRef="usd" name="dei:EntityPublicFloat"
                 format="ixt:num-dot-decimal" scale="0">794,433,813,496</ix:nonFraction>
```

That tag is a *gift* — `dei:EntityPublicFloat` tells you exactly what the number
means, with its unit and scale. Almost nobody uses it, and it's arguably the single
most under-exploited signal in financial RAG.

### 2b. HTML tables are laid out for print, not for meaning

```
<td> cells (paired)         22,610
<td .../> self-closing      37,738     <- 62% of all cells are empty spacers
cells containing only a number  10,579
```

Nearly two-thirds of cells exist purely for column spacing. A naive row-reader sees
mostly blanks.

### 2c. The case study: where did the dollar signs go?

Ask *"how many dollar amounts are in this filing?"* and strip tags to text:

| Extraction method | `$` attached to its number | `$` separated |
|---|---|---|
| Naive — every tag → space | 328 (**10%**) | 3,004 |
| Inline-aware — inline tags → `''`, block tags → space | 780 (**23%**) | 2,552 |

The inline-tag fix roughly doubles recovery and still leaves 77% broken. Why? Because:

```
cells containing ONLY '$'                     2,922
'$' cell immediately followed by number cell  2,552   (87% of them)
```

**The currency symbol occupies its own table cell.** And note that
`dollar_cell_then_number` (2,552) equals the inline-aware `split` count (2,552)
**exactly** — every amount that text extraction fails to recover is one whose `$`
sits in a separate cell. The mechanism is proven, not inferred.

> *A correction on this table.* An intermediate draft reported 636/19% for the
> inline-aware row and blamed HTML entities. That was wrong: those figures came from
> measuring the **corrupt** copy of the file described in §8. Re-measured against the
> verified file, the original 780/23% is correct. Fixing a right number using bad
> data is a nastier failure than the original bug — and the reason `rag corpus
> verify` now runs before any measurement.

> **This is not fixable by text processing.** No regex, no whitespace normalization,
> no tag heuristic can reunite `$` with its value, because they are in different
> cells of a table. The only correct fix is to parse the table **as a table** —
> reconstruct the grid, then reassemble each value from its cell group.
>
> This is Phase 2's central claim, and it is now measured rather than asserted.

### 2d. What this does to retrieval

A chunk containing `$ 794,433,813,496` embeds differently from one containing
`$794,433,813,496`, and a query for a dollar figure matches neither well. Worse, when
the row header, column header, and value are all in separate cells and get flattened
into one line, the *binding* between label and number is destroyed — and that binding
**is** the answer.

---

## 3. PDF: nothing is declared

### 3a. What the file actually stores

```
x0=  64.8  y0= 27.6  'RAG'
x0=  82.1  y0= 27.6  '&'
x0=  90.0  y0= 27.6  'Vector'
x0= 499.8  y0= 27.6  'By'
```

Words with coordinates. To know that `'Vector'` at x=90 and `'By'` at x=500 belong to
different columns, you have to work it out. Every notion of row, cell, column,
heading and reading order must be reconstructed from geometry.

### 3b. Even a good layout model gets it wrong

PyMuPDF's table detection on a clean, born-digital PDF (this example is from a
technical handbook, not a filing — it illustrates the mechanism, and 3c has the
on-domain numbers):

```
['P', 'art', '', '', 'What you will learn', '']
```

It split the word "Part" across two cells. That is the *easy* case — a born-digital
PDF with a real text layer.

### 3c. Measured on 55 real financial PDFs

The corpus now holds the **glossy Annual Report to Shareholders (form ARS)** for each
bank — 55 PDFs, 12,529 pages, 12.0M tokens — alongside the 10-K HTML for the same
company and year. Measured, not assumed:

| Pathology | Across 3,037 sampled pages |
|---|---|
| Pages with **no text layer** | 21 |
| Ligatures (`ﬁ ﬂ ﬀ`) | 1,249 (Bank of America alone: 556, 614) |
| Hyphen-newline word breaks | 626 (Citigroup FY2022: 231) |

> **Earlier this section reported measurements from `books/RAG_and_vector_practical_guide.pdf`** —
> a RAG handbook that happened to be in the repo. It was nearly clean (0 scanned
> pages, 0 ligatures, 12 hyphen breaks), and I flagged it as "easy" but not as
> **off-domain**, which was the more serious problem. A technical handbook says
> nothing about how a bank's annual report parses. Those numbers are replaced above.

### 3d. The headline result: PDF table extraction recovers a *fraction* of the tables

Because we hold both formats for the same company-year, we can score PDF extraction
against the HTML as ground truth. `<table>` count in the HTML vs tables PyMuPDF
detects in the PDF:

| Company | HTML tables | PDF detected | Recovered |
|---|---|---|---|
| Truist FY2023 | 144 | 123 | **85%** |
| JPMorgan FY2023 | 670 | 532 | 79% |
| Citigroup FY2025 | 330 | 192 | 58% |
| Schwab FY2022 | 125 | 48 | 38% |
| Bank of America FY2025 | 369 | 64 | 17% |
| Goldman Sachs FY2025 | 493 | 39 | 8% |
| Wells Fargo FY2024 | 408 | 6 | **1.5%** |
| BNY Mellon FY2025 | 359 | 0 | **0%** |
| American Express FY2022 | 116 | 0 | **0%** |

Across all 55 pairs: **median 14.4%, range 0% to 85.4%.** As company means: **1.1%
(BNY Mellon) to 62.3% (Truist).** In absolute terms the HTML declares **19,849**
tables and PDF extraction finds **4,051** of them.

The same parser, on the same document type, in the same industry, varies by more than
an order of magnitude depending only on who published the file.

That variance is the most important number in this document. It means:
- A single parser choice cannot be validated on a sample — it must be measured per publisher
- "We use `docling`, it's good at tables" is not an engineering statement
- **Phase 2's parser routing must be per-document, driven by a measured extraction-quality signal** — not a global config

Where recovery is 0%, the tables are almost certainly *borderless*: aligned with
whitespace rather than drawn rules, so a line-based detector sees nothing at all.
Those documents need a layout model or a VLM, and no amount of tuning PyMuPDF helps.

### 3e. Two failures found in the wild

**A genuinely scanned document.** Citizens Financial's FY2024 ARS: 194 pages, **167
of them (86%) have no text layer**, 116 images in the first 40 pages. Text extraction
returns 13,074 tokens from a full annual report. No error, no warning — the document
would have contributed almost nothing to the index. This is the silent-empty case,
and it is now in our corpus as a test fixture.

**A wrong content assumption — mine.** Morgan Stanley's FY2022 ARS is **12 pages**:
a shareholder letter, not an annual report. Extraction worked perfectly; the document
simply isn't what I assumed form ARS meant. Worth separating from parsing failures,
because the fix is different: detect it in the profiler and exclude it, rather than
reach for a better parser.

---

## 4. The trade-off, side by side

| Axis | HTML (EDGAR) | PDF (annual reports, circulars) |
|---|---|---|
| Table structure | Declared, but print-oriented (63% spacer cells) | Absent — must be inferred |
| Semantic tagging | **XBRL: unit, scale, concept name** | None |
| Reading order | Free | Must be reconstructed |
| Scanned-page risk | None | Real — silently extracts as empty |
| Multi-column risk | None | High — interleaves into plausible nonsense |
| Bytes that are content | ~12% | ~high (but compressed/binary) |
| Parser cost | ~0 | 10 ms → $0.02 per page |
| Where quality is won | Table grid reconstruction | Everything |
| Hardest failure | `$` split from its value | Page ingests as empty, silently |

---

## 5. Problems → solutions → cost

| # | Problem | Format | Solution | Cost |
|---|---|---|---|---|
| 1 | 88% of bytes are markup | HTML | Strip at ingest; never store raw in the index | ~0 |
| 2 | `$` separated from its value | HTML | Parse tables as grids; reassemble from cell groups | Real parser, not regex |
| 3 | 63% spacer cells | HTML | Drop empty cells during grid reconstruction | ~0 |
| 4 | XBRL semantics unused | HTML | Extract `name`, `unitRef`, `scale` into chunk metadata | Low — big payoff |
| 5 | Flattened table loses label↔value binding | both | **Dual representation**: row-serialized sentences *and* whole-table unit | 2× storage |
| 6 | No reading order | PDF | Layout model (`unstructured` hi_res, `docling`) | ~10–100 ms/page |
| 7 | Multi-column interleaving | PDF | Detect column count before extracting | Layout model |
| 8 | Scanned page → silent empty | PDF | Measure text yield per page; route low-yield to OCR/VLM | OCR, or ~$0.01/page VLM |
| 9 | Table drawn as lines + glyphs | PDF | Table-structure model, or VLM on the page image | $0.005–0.02/page |
| 10 | Hyphenation, ligatures, footnote markers | PDF | Normalization pass | ~0 |
| 11 | Boilerplate headers/footers | both | Detect repeats across pages, strip | ~0 |

**Note the shape of this table.** The HTML problems are cheap to fix once
*understood* — the difficulty is knowing they exist. The PDF problems cost real money
per page. That asymmetry should drive the parser routing decision in Phase 2: don't
buy a VLM for a document whose structure is already declared.

---

## 6. Parser decision matrix

| Document | Parser | Why |
|---|---|---|
| EDGAR HTML/iXBRL | HTML parser + grid reconstruction + XBRL extraction | Structure is declared; use it |
| Born-digital PDF, single column | `pymupdf` + normalization | Text layer is reliable |
| Born-digital PDF, tables/multi-column | `unstructured` hi_res or `docling` | Needs layout detection |
| Scanned page (low text yield) | OCR, or VLM on page image | No text layer exists |
| Charts, complex financial tables | VLM (`claude-opus-5` on page image) | Only thing that reads them reliably |

**Route per page, not per corpus.** A 300-page annual report may have 280 easy pages
and 20 that need a VLM. One parser for everything is wrong at both ends of the cost
curve.

---

## 7. The idea that makes Phase 2 measurable

Phase 2 currently budgets for hand-labelling 30 pages of extraction ground truth.
There is a better option available to us specifically.

**Companies publish the same financial statements in both formats.** JPMorgan's
FY2025 figures exist in the 10-K HTML we already hold *and* in a glossy Annual Report
PDF on their investor-relations site.

So: extract a table from the PDF, compare against the same table from the HTML, and
you can **score** each parser against a reference instead of judging it by eye. Same
companies, same years, same numbers — free ground truth for the hardest parsing case.

That converts the parser bake-off from impressionistic to quantitative, and it is the
main argument for adding annual-report PDFs rather than any other PDF source.

---

## 8. Integrity: the check that should have existed from day one

Building the notebook surfaced a corrupt file. One 10-K was **19,075,544 bytes on
disk against 12,927,325 recorded in the manifest** — caught only because a derived
ratio came out at 7.9% instead of 11.7% and the discrepancy was chased.

The manifest had stored a sha256 and byte count for **every** document since the
first fetch. Nothing had ever checked them.

`rag corpus verify` now re-hashes the corpus against the manifest:

```
120 documents in manifest
  verified            120
  missing on disk       0
  size mismatch         0
  checksum mismatch     0
```

**Run it after any fetch and before trusting any measurement.** A file that changes
after the manifest is written makes every downstream number silently wrong while the
manifest continues to look authoritative — and 119 of 120 files being correct is
exactly what makes that dangerous.

---

## 9. On measurement — three wrong attempts at one number

The `$`-separation figure took three tries, and the failures are as instructive as
the result:

| Attempt | Method | Result | What was wrong |
|---|---|---|---|
| 1 | `\$[\d,]+` on flattened text | **0 dollar amounts** in a bank's annual report | Obviously false. Caught only because the answer had to be non-zero |
| 2 | `\$[\d,]{4,}` | "90% split" | The `{4,}` quantifier can't match `$4.4` — a decimal point excluded most amounts, and the regex silently matched a *later* number instead |
| 3 | `\$[\d,]+(?:\.\d+)?` + cell-level check | 10% attached; 2,922 `$`-only cells, 87% adjacent to a number cell | Correct, and confirms the mechanism |

**The lesson:** attempt 2 produced a plausible number, which is why it survived
longer than attempt 1. A measurement that returns something believable gets trusted.
The defence is to predict the answer's rough magnitude *before* measuring, and to
verify a headline number by a second, structurally different method — here,
counting cells rather than matching text.

This is the same failure class as the four corpus bugs in
[ADR 0001](decisions/0001-corpus-and-naming.md): no exception raised, just
plausible-looking wrong.

---

## 10. Corpus status

```
            docs        MB       tokens     pages
HTML          65       585   16,520,035         -
PDF           55       796   11,979,475    12,529
TOTAL        120     1,382   28,499,510    12,529     → 71,248 chunks @ 400 tok
```

**Both formats, same companies, same years.** That satisfies §7: the HTML is ground
truth for scoring PDF extraction, so Phase 2's parser bake-off is quantitative rather
than impressionistic — no hand-labelling required.

## 11. Open items

- [x] ~~Add annual-report PDFs~~ — done via EDGAR form **ARS**, 55 PDFs. They are
      hosted on SEC, so the existing polite fetcher handles them
- [x] ~~Source genuinely hard PDFs~~ — Citizens FY2024 is 86% scanned (§3e)
- [ ] **Build XBRL extraction** (problem 4) — `name`, `unitRef`, `scale` per figure.
      Cheap, and no other corpus format offers it
- [ ] **Implement grid-aware table extraction** (problem 2) — the only fix for the
      `$` problem, and the prerequisite for the dual representation in problem 5
- [x] ~~Extend the profiler to PDFs~~ — `rag/ingest/format_probe.py` provides
      `pdf_probe` and `table_recovery`; results cached to `data/interim/`
- [ ] **Wire recovery into `rag corpus profile`** so extraction quality is tracked on
      every corpus change, not measured once
- [ ] **Regulatory PDFs still missing** — RBI is bot-protected (a plain fetch returns
      a 46 KB CAPTCHA page); OCC's listing page is JS-rendered. Both need a browser
      session or manual download. Lower priority now that ARS supplies the hard case
- [ ] **Exclude Morgan Stanley FY2022 ARS** (12-page shareholder letter, not a report)

**Note on `n_pages`:** the fetcher now extracts PDFs with PyMuPDF at fetch time to
record page count and `empty_pages`, rather than estimating tokens from file size.
That is what surfaced the Citizens scan.
