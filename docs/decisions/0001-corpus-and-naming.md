# ADR 0001 — Corpus source, document resolution, and file naming

**Status:** accepted · **Phase:** 0 (steps 0.6–0.7) · **Date:** 2026-08-31

## Context

Phase 1 builds the golden set. Every later phase measures itself against it, so a
corpus defect baked in now becomes a permanently wrong baseline — worse than no
baseline, because it will be trusted.

## Decision 1 — Raw fetch for the working corpus; prepared dataset only for scale

| Job | Source | Reason |
|---|---|---|
| Working corpus (Ph 1–4, 6–9) | Our own EDGAR fetcher | Provenance we control; FY2025 filings postdate most model cutoffs, so less contaminated for the golden set |
| Scale test (Ph 5, 1M+ vectors) | TeraflopAI/SEC-EDGAR parquet shards | One 616 MB shard ≈ 1,498 filings ≈ 243k chunks. A full crawl takes 10+ days |
| Parser bake-off (Ph 2) | RBI / OCC / annual-report **PDFs** — *still to source* | EDGAR is HTML/SGML: no scanned pages, no multi-column PDFs, no tables drawn as glyphs |
| Golden set scaffold (Ph 1) | FinQA / TAT-QA / ConvFinQA | Gold Q&A pairs incl. multi-turn |

**Rejected:** using the prepared dataset as the working corpus. It has already been
parsed with selectolax and doc2dict — we would inherit someone else's parsing
decisions and be unable to measure them, which deletes Phase 2 entirely. A prepared
dataset removes the problem you are trying to learn.

## Decision 2 — Resolve the real document by size, not by EDGAR's metadata

**EDGAR's `primaryDocument` field cannot be trusted.** Measured on FY2025:

| Filer | `primaryDocument` | Actual content |
|---|---|---|
| Wells Fargo | `wfc-20251231_d2.htm` — 360 KB | `wfc-20251231.htm` — **11.6 MB** |
| US Bancorp | `usb-20251231.htm` — 973 KB | `usb-20251231_d2.htm` — **5.4 MB** |
| BNY Mellon | `bk-20251231.htm` — 1.3 MB | `bk-20251231_d2.htm` — **7.2 MB** |

Note the `_d2` convention is **inverted** between Wells Fargo and the other two, so
any name-based rule breaks on one of them.

**Rule adopted:** fetch each filing's `index.json`; among `.htm` files sharing the
primary's stem (excluding `R\d+.htm` XBRL viewer renderings), take the largest.
Stem-matching excludes genuine exhibits; size is the reliable signal.

**Cost:** one extra request per filing (~60), ~9 s at 7 req/s. Cheap enough to be
the default rather than a flag.

**Rejected:** trusting `primaryDocument` (11 of 60 documents would contain no
financial statements); name heuristics (inverted convention defeats them);
dropping the three filers (removes a common production pattern — one logical
document split across files).

## Decision 3 — Human-readable, collision-proof filenames

**Before:** `edgar-19617-0001628280-26-008131.htm` — company and year both opaque.
**After:** `edgar_jpmorgan_FY2025_10k_008131.htm`

Format: `{source}_{slug}_{FY}_{form}_{accession-tail}`

- **slug** — hand-written per company (`jpmorgan`, `bofa`, `usbancorp`), not derived
  from the SEC name, which is unusable in a filename (`US BANCORP \DE\`,
  `WELLS FARGO & COMPANY/MN`)
- **FY** — fiscal year the filing *reports on* (`reportDate`), not the filing date.
  The FY2025 10-K is filed in Feb 2026; sorting by filing date scrambles the series
- **accession tail** — 6 digits, kept so an original and a later 10-K/A amendment
  for the same company-year cannot silently overwrite each other

`doc_id` is the same string as the filename stem, so a chunk's provenance is legible
in a trace without a lookup.

**Rejected:** `{slug}_{FY}.htm` — reads best but two filings for one company-year
collide silently. Silent overwrite is the same failure class as everything else in
this ADR.

## Decision 4 — CIK verification

CIKs are hand-entered and a wrong one silently fetches a different company, whose
data then carries a filename asserting it is the company you asked for. The fetcher
therefore stores an expected name fragment per CIK and warns when the SEC-resolved
name doesn't contain it.

## Consequences

- Extra `index.json` request per filing; enumeration is slower, correctness is default
- Adding a company means adding a slug — deliberate friction, and it makes the
  corpus roster explicit
- Old filenames are gone; the corpus was re-fetched rather than migrated

## The thread running through all four

Four silent-under-delivery bugs in one session: a scan reporting zero dollar amounts
in a bank's annual report, a size filter that would have rejected every large bank,
an enumerator returning 1 filing when asked for 4, and `primaryDocument` pointing at
a wrapper. **None raised an exception.** Every one returned plausible-looking less.

That is the characteristic failure mode of data pipelines, and it is why the corpus
is profiled before the pipeline is built — see `rag/ingest/profile.py` and
[docs/DATA.md](../DATA.md) §6.
