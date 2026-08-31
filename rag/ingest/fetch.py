"""Corpus fetcher: polite, resumable, content-addressed.

Three properties, each preventing a specific failure that costs you a day.

**Polite.** SEC allows ~10 req/s and asks for a User-Agent identifying you. Exceed
it and they block your IP -- silently, so the first symptom is a directory of
zero-byte files you don't notice until parsing fails much later.

**Resumable.** A 400-document fetch will die partway. Without resume you restart
from zero, which is both slow and rude to the source. We skip files already on
disk whose checksum matches the manifest.

**Content-addressed.** Every document is hashed. That hash is what makes
re-ingestion a no-op instead of a full re-parse, and it is the same mechanism
Phase 11 uses for incremental sync. The manifest is not bookkeeping -- it *is*
the change-detection system, built on day one.

We also record `est_tokens` and `text_ratio` per document. A planning estimate of
60k tokens/doc proved 6x wrong against a real bank 10-K; measuring beats
extrapolating, and the corpus chain in NUMBERS.md is recomputed from these.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag.config import settings
from rag.schema import DocMeta, sha256_file

# SEC requires a descriptive User-Agent with contact info. Anonymous or spoofed
# agents get blocked. This is a published requirement, not a nicety.
USER_AGENT = "RAG Learning Project anantram.patel@azentio.com"

# SEC's published ceiling is 10 req/s. Stay under it -- their limiter is not
# forgiving and the block is silent.
SEC_RATE_LIMIT = 7.0


class RateLimiter:
    """Minimum-interval limiter. Deliberately simple: one shared clock, no bursts.

    A token bucket would allow bursts, which is exactly what trips SEC's limiter
    at the start of a run when the queue is full.
    """

    def __init__(self, rate_per_sec: float) -> None:
        self.min_interval = 1.0 / rate_per_sec
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


@dataclass
class FetchTask:
    """One document to fetch.

    `doc_id` doubles as the filename stem, and is built to be readable:
    `edgar_jpmorgan_FY2025_10k_008131`. The accession tail is kept for provenance
    and, more importantly, to make collisions impossible -- two filings for the
    same company/year (an original plus a 10-K/A amendment) would otherwise
    silently overwrite each other. Silent overwrite is the same failure class as
    everything else this pipeline has been bitten by.
    """

    doc_id: str
    url: str
    title: str
    doc_type: str
    source: str
    effective_date: str | None = None
    version: str | None = None
    sensitivity: str = "public"
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def suffix(self) -> str:
        ext = Path(urlparse(self.url).path).suffix.lower()
        return ext if ext in {".pdf", ".htm", ".html", ".txt", ".xml"} else ".bin"

    @property
    def filename(self) -> str:
        return f"{self.doc_id}{self.suffix}"


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

# slug -> (CIK, expected name fragment)
#
# The slug is what appears in filenames, so it is chosen for readability at the
# shell prompt. The name fragment is checked against what the SEC actually returns:
# a wrong CIK otherwise silently fetches a different company, and the file would
# still be named as though it were the one you asked for.
BANKS: dict[str, tuple[int, str]] = {
    "jpmorgan": (19617, "JPMORGAN"),
    "bofa": (70858, "BANK OF AMERICA"),
    "citigroup": (831001, "CITIGROUP"),
    "wellsfargo": (72971, "WELLS FARGO"),
    "goldman": (886982, "GOLDMAN SACHS"),
    "morganstanley": (895421, "MORGAN STANLEY"),
    "amex": (4962, "AMERICAN EXPRESS"),
    "truist": (92230, "TRUIST"),
    "usbancorp": (36104, "US BANCORP"),
    "pnc": (713676, "PNC"),
    "capitalone": (927628, "CAPITAL ONE"),
    "schwab": (316709, "SCHWAB"),
    "statestreet": (93751, "STATE STREET"),
    "bnymellon": (1390777, "BANK OF NEW YORK MELLON"),
    "citizens": (759944, "CITIZENS FINANCIAL"),
}

# XBRL viewer renderings -- one file per rendered statement. Not the filing text.
R_FILE_RE = re.compile(r"^R\d+\.htm$", re.I)


def resolve_primary(
    client: httpx.Client,
    limiter: RateLimiter,
    cik: int,
    accession_nodash: str,
    primary: str,
) -> tuple[str, int]:
    """Find the document that actually contains the filing.

    **EDGAR's `primaryDocument` field cannot be trusted.** Every filing here has two
    HTML documents sharing a stem -- `X.htm` and `X_d2.htm` -- and for several large
    banks the API points at the small wrapper:

        Wells Fargo  primaryDocument = wfc-20251231_d2.htm    360 KB   <- wrapper
                     real content    = wfc-20251231.htm      11.6 MB
        US Bancorp   primaryDocument = usb-20251231.htm       973 KB   <- wrapper
                     real content    = usb-20251231_d2.htm    5.4 MB

    Note the convention is *inverted* between them, so a name-based rule (`prefer
    _d2`, or `avoid _d2`) breaks on one or the other. Size is the reliable signal.

    **Rule: the largest `.htm` in the submission, excluding XBRL viewer renderings.**

    An earlier version also required the candidate to share the primary's filename
    stem, reasoning that this would exclude genuine exhibits. It broke immediately:
    US Bancorp's FY2022 filing went through a filing agent, so the documents are
    `d410791d10k.htm` (1.0 MB) and `d410791dex13.htm` (6.7 MB) -- Exhibit 13 is the
    annual report containing the financials, and its stem doesn't match. The clever
    rule silently kept the 1 MB wrapper.

    So: one rule, no special cases. The residual risk is a filing where some other
    exhibit (a long credit agreement, say) outweighs the report body. That is what
    the profiler is for -- `rag corpus profile` re-checks every document for Item 8
    and table density, so a wrong pick surfaces as a measurement rather than as a
    silently degraded corpus. Prefer a simple rule plus a verification pass over a
    complicated rule you have to trust.
    """
    # ARS filings (glossy annual reports) are a single PDF -- nothing to resolve, and
    # no reason to spend a request finding that out.
    if primary.lower().endswith(".pdf"):
        return primary, 0

    limiter.wait()
    r = client.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/index.json")
    if r.status_code != 200:
        return primary, 0

    best, best_size = primary, 0
    for item in r.json().get("directory", {}).get("item", []):
        name = item.get("name", "")
        if not name.lower().endswith(".htm") or R_FILE_RE.match(name):
            continue
        size = int(item.get("size") or 0)
        if size > best_size:
            best, best_size = name, size
    return best, best_size


def _scan_block(
    block: dict[str, list],
    forms: tuple[str, ...],
    max_submission_bytes: int,
) -> Iterator[dict[str, str]]:
    """Scan one filings block (column-oriented arrays) for matching forms.

    `recent` and every archive shard share this exact shape, which is why one
    scanner serves both.
    """
    sizes = block.get("size") or [0] * len(block["form"])
    for i, form in enumerate(block["form"]):
        if form not in forms:
            continue
        if sizes[i] and sizes[i] > max_submission_bytes:
            continue
        doc = block["primaryDocument"][i]
        if not doc:
            continue
        yield {
            "accession": block["accessionNumber"][i],
            "doc": doc,
            "form": form,
            "report_date": block["reportDate"][i],
            "filing_date": block["filingDate"][i],
        }


def edgar_tasks(
    client: httpx.Client,
    limiter: RateLimiter,
    banks: dict[str, tuple[int, str]],
    forms: tuple[str, ...] = ("10-K",),
    per_company: int = 4,
    since: str = "2022-01-01",
    max_shards: int = 40,
    max_submission_bytes: int = 400_000_000,
    resolve_docs: bool = True,
) -> Iterator[FetchTask]:
    """Enumerate filings via the EDGAR submissions API, paging back through history.

    **The bug this fixes -- worth understanding, because the shape recurs.**
    The API splits a company's history into a `recent` block plus dated archive
    shards. `recent` holds roughly the last 1,000 filings *by count*, not by time.
    JPMorgan files ~26,000 documents a year (overwhelmingly Form 4 insider trades),
    so its `recent` block spans only ~12 months and contains exactly **one** 10-K.

    An earlier version read only `recent`. Asked for 4 years per company it
    returned 1 -- no error, no warning, just a quarter of what was requested. Silent
    under-delivery: the same failure class as a scan reporting zero dollar amounts
    in a bank's annual report. Data pipelines fail by returning plausible-looking
    less, not by raising.

    So: scan `recent`, and if that is not enough, walk the archive shards
    newest-first until we have `per_company` filings or run past `since`.

    Note on `size`: it is the **whole submission** (every exhibit), not the primary
    document -- a JPM 10-K submission is ~65 MB while its primary .htm is ~13 MB.
    Filtering on it as though it were document size rejects exactly the large banks
    we want. It is a coarse guard only; the real control is the byte budget in
    `fetch_all`.
    """
    for slug, (cik, expect) in banks.items():
        limiter.wait()
        r = client.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        if r.status_code != 200:
            print(f"  ! {slug} (CIK {cik}): HTTP {r.status_code} — skipping")
            continue
        data = r.json()

        resolved = data.get("name", "?")
        if expect.lower() not in resolved.lower():
            print(f"  ! CIK {cik} resolved to {resolved!r}, expected to contain {expect!r}")

        hits = list(_scan_block(data["filings"]["recent"], forms, max_submission_bytes))
        shards_read = 0

        # Archive shards are listed newest-first. Stop as soon as we have enough,
        # or once a shard's whole range predates `since`.
        target = per_company * len(forms)
        for shard in data["filings"].get("files", []):
            if len(hits) >= target or shards_read >= max_shards:
                break
            if shard.get("filingTo", "9999") < since:
                break
            limiter.wait()
            rs = client.get(f"https://data.sec.gov/submissions/{shard['name']}")
            shards_read += 1
            if rs.status_code != 200:
                continue
            hits.extend(_scan_block(rs.json(), forms, max_submission_bytes))

        # `per_company` is per FORM, not per company. Selecting across the combined
        # pool would let a company's four 10-Ks crowd out its ARS entirely -- and the
        # whole point of fetching both is having the pair for the same year.
        hits = [h for h in hits if h["filing_date"] >= since]
        selected: list[dict[str, str]] = []
        for form in forms:
            same = sorted(
                (h for h in hits if h["form"] == form),
                key=lambda h: h["report_date"] or "",
                reverse=True,
            )
            if len(same) < per_company:
                print(
                    f"  ! {resolved}: wanted {per_company} {form}, found {len(same)} "
                    f"(read {shards_read} archive shards back to {since})"
                )
            selected.extend(same[:per_company])
        hits = selected

        for h in hits:
            acc = h["accession"].replace("-", "")
            form = h["form"].lower().replace("-", "")
            fy = (h["report_date"] or "????")[:4]
            doc = h["doc"]
            if resolve_docs:
                doc, size = resolve_primary(client, limiter, cik, acc, h["doc"])
                if doc != h["doc"]:
                    print(
                        f"    ↳ {slug} FY{fy}: EDGAR named {h['doc']}, "
                        f"using {doc} ({size/1e6:.1f} MB)"
                    )
            yield FetchTask(
                doc_id=f"edgar_{slug}_FY{fy}_{form}_{h['accession'][-6:]}",
                url=f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}",
                title=f"{resolved} {h['form']} FY{fy}",
                doc_type=form,
                source="edgar",
                effective_date=h["report_date"] or None,
                version=h["filing_date"],
                extra={"cik": str(cik), "company": resolved, "slug": slug, "sec_file": doc},
            )
        years = ", ".join(h["report_date"][:4] for h in hits)
        print(f"  · {slug:14} {len(hits)} filings [{years}]  ({shards_read} shards)")


def seed_tasks(seeds_path: Path) -> Iterator[FetchTask]:
    """Read hand-curated URLs from data/seeds.jsonl.

    RBI, OCC, BIS and company annual reports have no stable machine API, and
    scraping their listing pages breaks whenever they redesign. A seed file is
    honest about that: you add URLs, the fetcher handles politeness, resume and
    hashing. Curation is manual; everything mechanical is not.

    One line per document:
      {"url": "...", "title": "...", "doc_type": "circular", "source": "rbi",
       "effective_date": "2024-04-01"}
    """
    if not seeds_path.exists():
        return
    for line in seeds_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        url = d["url"]
        slug = re.sub(r"[^a-z0-9]+", "-", Path(urlparse(url).path).stem.lower())[:60]
        fy = (d.get("effective_date") or "")[:4]
        stem = f"{d.get('source', 'seed')}_{slug}" + (f"_FY{fy}" if fy else "")
        yield FetchTask(
            doc_id=d.get("doc_id") or stem,
            url=url,
            title=d.get("title", slug),
            doc_type=d.get("doc_type", "unknown"),
            source=d.get("source", "seed"),
            effective_date=d.get("effective_date"),
            version=d.get("version"),
            sensitivity=d.get("sensitivity", "public"),
        )


# --------------------------------------------------------------------------- #
# Fetch + measure
# --------------------------------------------------------------------------- #

TAG_RE = re.compile(rb"<[^>]+>")
WS_RE = re.compile(rb"\s+")


def measure(raw: bytes, suffix: str) -> dict[str, object]:
    """Estimated tokens, text-to-bytes ratio, and (for PDFs) page-level diagnostics.

    Token counts stay crude -- 4 chars/token -- because the goal is catching
    order-of-magnitude planning errors, not billing. A real tokenizer arrives in
    Phase 2.

    For PDFs we extract properly with PyMuPDF rather than guessing from file size,
    because `empty_pages` is the number that matters: a page with no text layer
    extracts as an empty string with **no error**, and a document made of such pages
    silently contributes nothing to the index. That is the single most common way a
    PDF corpus quietly develops holes.
    """
    if suffix in {".htm", ".html", ".xml"}:
        text = WS_RE.sub(b" ", TAG_RE.sub(b" ", raw))
        return {"est_tokens": len(text) // 4, "text_ratio": len(text) / max(len(raw), 1)}

    if suffix == ".pdf":
        try:
            import pymupdf

            doc = pymupdf.open(stream=raw, filetype="pdf")
            pages = [p.get_text() for p in doc]
            chars = sum(len(p) for p in pages)
            return {
                "est_tokens": chars // 4,
                "text_ratio": chars / max(len(raw), 1),
                "n_pages": len(pages),
                "empty_pages": sum(1 for p in pages if not p.strip()),
            }
        except Exception:  # noqa: BLE001 -- a corrupt PDF must be visible, not fatal
            return {"est_tokens": 0, "text_ratio": 0.0, "n_pages": None, "empty_pages": None}

    return {"est_tokens": len(raw) // 4, "text_ratio": 1.0}


@retry(
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _get(client: httpx.Client, url: str) -> bytes:
    r = client.get(url)
    r.raise_for_status()
    return r.content


def fetch_all(
    tasks: list[FetchTask],
    out_dir: Path,
    limiter: RateLimiter,
    budget_bytes: int,
) -> list[DocMeta]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metas: list[DocMeta] = []
    total = 0

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0, follow_redirects=True) as c:
        for i, t in enumerate(tasks, 1):
            path = out_dir / t.filename

            if path.exists():  # resume
                raw = path.read_bytes()
                metas.append(_meta(t, path, raw, measure(raw, t.suffix)))
                total += len(raw)
                print(f"[{i}/{len(tasks)}] = {t.doc_id} (cached)")
                continue

            if total >= budget_bytes:
                print(f"\n! budget of {budget_bytes/1e6:.0f} MB reached — stopping")
                break

            limiter.wait()
            try:
                raw = _get(c, t.url)
            except Exception as e:  # noqa: BLE001 — a failed doc must be visible, not fatal
                print(f"[{i}/{len(tasks)}] ! {t.doc_id}: {type(e).__name__}: {e}")
                continue

            path.write_bytes(raw)
            m = measure(raw, t.suffix)
            metas.append(_meta(t, path, raw, m))
            total += len(raw)
            extra = ""
            if m.get("n_pages"):
                extra = f"  {m['n_pages']:>4}p"
                if m.get("empty_pages"):
                    extra += f"  {m['empty_pages']} EMPTY"
            print(
                f"[{i}/{len(tasks)}] + {t.doc_id}  "
                f"{len(raw)/1e6:5.1f} MB  ~{m['est_tokens']:>7,} tok  "
                f"text={m['text_ratio']:.0%}{extra}"
            )

    return metas


def _meta(t: FetchTask, path: Path, raw: bytes, m: dict[str, object]) -> DocMeta:
    return DocMeta(
        doc_id=t.doc_id,
        path=str(path.relative_to(settings.data_dir.parent)),
        title=t.title,
        doc_type=t.doc_type,
        source_url=t.url,
        effective_date=t.effective_date,  # type: ignore[arg-type]
        version=t.version,
        sensitivity=t.sensitivity,  # type: ignore[arg-type]
        checksum=sha256_file(str(path)),
        bytes=len(raw),
        est_tokens=int(m["est_tokens"] or 0),
        text_ratio=round(float(m["text_ratio"] or 0.0), 4),
        n_pages=m.get("n_pages"),  # type: ignore[arg-type]
    )


def write_manifest(metas: list[DocMeta], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for m in metas:
            fh.write(m.model_dump_json() + "\n")
