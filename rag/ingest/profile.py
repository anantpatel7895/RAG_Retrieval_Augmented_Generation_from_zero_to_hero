"""Corpus profiler — measure the data before deciding how to process it.

Every metric here exists because it changes a downstream decision. Anything that
would merely be interesting is not collected.

| Measurement            | Decision it drives                                    |
|------------------------|-------------------------------------------------------|
| tables per document    | Whether table extraction is *the* parser requirement  |
| hollow documents       | Whether the corpus actually contains what we think    |
| token distribution     | Chunking strategy; contextual-retrieval affordability |
| near-duplicate overlap | Dedup/MMR (Ph6), conflict resolution (Ph8) difficulty |
| year coverage          | Whether a conflict slice can exist at all             |
| text ratio             | Whether stripping belongs at ingest                   |

The profiler exists mainly to make *silent under-delivery* visible. This project
has already hit three instances: a scan reporting zero dollar amounts in a bank's
annual report, a size filter that would have rejected every large bank, and an
enumerator returning 1 filing when asked for 4. None raised an exception. Data
pipelines fail by returning plausible-looking less.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
ENTITY_RE = re.compile(r"&#?\w+;")

# Item 8 is where the financial statements belong. When a filer satisfies it by
# pointing at a separate exhibit, the document we downloaded is a pointer, not a
# report -- no statements, no footnotes, almost no tables.
INCORP_RE = re.compile(
    r"Item\s*8\.?\s*Financial Statements.{0,300}?"
    r"(can be found|incorporated (herein )?by reference|included in .{0,40}Annual Report)",
    re.I | re.S,
)


@dataclass
class DocProfile:
    doc_id: str
    title: str
    fy: str
    company: str
    tokens: int
    tables: int
    text_ratio: float
    hollow: bool

    @property
    def tables_per_10k_tok(self) -> float:
        return self.tables / max(self.tokens / 10_000, 1e-9)


def plain_text(raw: str) -> str:
    return WS_RE.sub(" ", ENTITY_RE.sub(" ", TAG_RE.sub(" ", raw)))


# A real 10-K carries 10-20 tables per 10k tokens (measured: solid docs sit at
# 15-18). A pointer document carries 3-4. The gap is wide enough to threshold on.
HOLLOW_TABLE_DENSITY = 8.0


def profile_doc(meta: dict) -> DocProfile:
    """Classify a filing using two independent signals, because one lied.

    The text pattern alone (Item 8 pointing elsewhere) produced a false positive on
    Capital One -- 501 tables and 274k tokens, unmistakably a full report -- because
    the regex matched its table-of-contents entry rather than a real incorporation.

    Table *density* is the structural signal and it separates cleanly: pointer
    documents sit at 3-4 tables per 10k tokens, real reports at 15-18. Requiring
    both signals means a wording quirk alone can't misclassify a document.

    Generalizable lesson: when a cheap textual signal and a structural one disagree,
    trust the structural one and treat the disagreement as a bug in your detector.
    """
    raw = Path(meta["path"]).read_text(encoding="utf-8", errors="ignore")
    text = plain_text(raw)
    tables = raw.lower().count("<table")
    tokens = meta.get("est_tokens") or 0
    density = tables / max(tokens / 10_000, 1e-9)
    return DocProfile(
        doc_id=meta["doc_id"],
        title=meta["title"],
        fy=(meta.get("effective_date") or "????")[:4],
        company=meta["title"].split(" 10-K")[0][:26],
        tokens=tokens,
        tables=tables,
        text_ratio=meta.get("text_ratio") or 0.0,
        hollow=density < HOLLOW_TABLE_DENSITY and bool(INCORP_RE.search(text)),
    )


def find_outliers(profiles: list[DocProfile], ratio: float = 0.5) -> list[tuple[DocProfile, str]]:
    """Documents anomalously small compared to their own company's other years.

    The third detector, added because the first two both missed Wells Fargo FY2025:
    23k tokens against its own 102k/97k/84k in prior years. It never says
    "incorporated by reference" (so the text pattern missed it) and its 40 tables
    over 23k tokens reads as *high* density (so the structural check passed).

    Peer-relative comparison catches what absolute thresholds can't: the same filer
    in adjacent years is the tightest possible control. A 10-K that suddenly loses
    three quarters of its content is anomalous regardless of what it says or how
    dense it looks.

    Three detectors, three distinct blind spots. No single signal is sufficient --
    which is the actual lesson, not any one of the thresholds.
    """
    by_co: dict[str, list[DocProfile]] = defaultdict(list)
    for p in profiles:
        by_co[p.company].append(p)

    out: list[tuple[DocProfile, str]] = []
    for co, docs in by_co.items():
        if len(docs) < 2:
            continue
        toks = sorted(d.tokens for d in docs)
        median = toks[len(toks) // 2]
        for d in docs:
            if median and d.tokens < ratio * median:
                out.append((d, f"{d.tokens:,} tok vs {co} median {median:,}"))
    return sorted(out, key=lambda t: t[0].tokens)


def shingles(text: str, k: int = 8, sample: int = 64) -> set[int]:
    """Hashed k-word shingles, sampled ~1/`sample` — a poor man's MinHash.

    Full shingle sets on a 400k-token document are large; sampling by hash keeps
    the estimate unbiased (the same shingle always survives or doesn't, in every
    document) while cutting memory by `sample`x. That consistency is the whole
    trick — random per-document sampling would destroy the comparison.
    """
    words = text.split()
    out: set[int] = set()
    for i in range(len(words) - k):
        h = hash(" ".join(words[i : i + k]))
        if h % sample == 0:
            out.add(h)
    return out


def jaccard(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def profile_corpus(manifest: Path) -> tuple[list[DocProfile], dict[str, list[tuple[str, float]]]]:
    metas = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    profiles = [profile_doc(m) for m in metas]

    # Consecutive-year overlap, per company. This is the number that predicts how
    # hard Phase 6 dedup and Phase 8 conflict resolution will be: if consecutive
    # 10-Ks are 80% identical, a query retrieves four near-identical chunks and
    # something has to decide which year applies.
    by_co: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for m, p in zip(metas, profiles, strict=True):
        by_co[p.company].append((p.fy, m))

    dupes: dict[str, list[tuple[str, float]]] = {}
    for co, items in by_co.items():
        items.sort()
        sh = [
            (fy, shingles(plain_text(Path(m["path"]).read_text(encoding="utf-8", errors="ignore"))))
            for fy, m in items
        ]
        pairs = []
        for (fy1, s1), (fy2, s2) in zip(sh, sh[1:], strict=False):
            pairs.append((f"{fy1}→{fy2}", jaccard(s1, s2)))
        if pairs:
            dupes[co] = pairs
    return profiles, dupes
