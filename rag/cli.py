"""Project CLI.  `rag --help`"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import typer

from rag.config import settings
from rag.ingest.fetch import (
    BANKS,
    SEC_RATE_LIMIT,
    USER_AGENT,
    RateLimiter,
    edgar_tasks,
    fetch_all,
    seed_tasks,
    write_manifest,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)
corpus = typer.Typer(no_args_is_help=True, help="Fetch and inspect the corpus")
app.add_typer(corpus, name="corpus")


@corpus.command("fetch")
def corpus_fetch(
    edgar: bool = typer.Option(True, help="Fetch 10-Ks from SEC EDGAR"),
    seeds: bool = typer.Option(True, help="Fetch hand-curated URLs from data/seeds.jsonl"),
    per_company: int = typer.Option(2, help="Filings per company"),
    companies: int = typer.Option(len(BANKS), help="How many companies"),
    budget_mb: int = typer.Option(1000, help="Stop after this many MB"),
    forms: str = typer.Option(
        "10-K",
        help="Comma-separated EDGAR form types. '10-K' = HTML filing; "
        "'ARS' = the glossy Annual Report to Shareholders, a single PDF — "
        "same facts as the 10-K, so the HTML is ground truth for PDF parsing.",
    ),
) -> None:
    """Download the corpus into data/raw/ and write the manifest.

    Resumable: re-running skips anything already on disk.
    """
    out = settings.raw_dir
    limiter = RateLimiter(SEC_RATE_LIMIT)
    tasks = []

    if edgar:
        typer.echo("Enumerating EDGAR filings…")
        with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0) as c:
            picked = dict(list(BANKS.items())[:companies])
            form_tuple = tuple(f.strip() for f in forms.split(",") if f.strip())
            tasks += list(
                edgar_tasks(c, limiter, picked, forms=form_tuple, per_company=per_company)
            )

    if seeds:
        seed_file = settings.data_dir / "seeds.jsonl"
        s = list(seed_tasks(seed_file))
        typer.echo(f"Seeds: {len(s)} from {seed_file}")
        tasks += s

    if not tasks:
        typer.echo("Nothing to fetch.")
        raise typer.Exit(1)

    typer.echo(f"\n{len(tasks)} documents queued, budget {budget_mb} MB\n")
    metas = fetch_all(tasks, out, limiter, budget_bytes=budget_mb * 1_000_000)

    manifest = out / "manifest.jsonl"
    write_manifest(metas, manifest)
    typer.echo(f"\nmanifest: {manifest} ({len(metas)} docs)")
    corpus_stats()


@corpus.command("stats")
def corpus_stats() -> None:
    """The corpus chain, computed from the manifest rather than estimated.

    This is the number every capacity calculation in NUMBERS.md hangs off. An
    estimate of 60k tokens/doc proved 6x wrong against a real bank 10-K -- and
    since contextual-retrieval cost scales as D^2/c, that error was 36x downstream.
    """
    path = settings.raw_dir / "manifest.jsonl"
    if not path.exists():
        typer.echo(f"no manifest at {path} — run `rag corpus fetch` first")
        raise typer.Exit(1)

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    if not rows:
        typer.echo("manifest is empty")
        raise typer.Exit(1)

    toks = sorted(r.get("est_tokens") or 0 for r in rows)
    total_tok = sum(toks)
    total_b = sum(r.get("bytes") or 0 for r in rows)
    n = len(rows)

    by_type: dict[str, list[int]] = {}
    for r in rows:
        by_type.setdefault(r.get("doc_type", "?"), []).append(r.get("est_tokens") or 0)

    def pct(p: float) -> int:
        return toks[min(int(len(toks) * p), len(toks) - 1)]

    chunk = 400
    typer.echo("\n── corpus chain ───────────────────────────────")
    typer.echo(f"documents        {n:>12,}")
    typer.echo(f"bytes on disk    {total_b/1e6:>11.1f} MB")
    typer.echo(f"total tokens     {total_tok:>12,}")
    typer.echo(f"chunks @ {chunk} tok {total_tok//chunk:>12,}")
    typer.echo("\n── token distribution (skew matters) ──────────")
    typer.echo(f"min / p50 / max  {toks[0]:,} / {pct(0.5):,} / {toks[-1]:,}")
    typer.echo(f"mean             {total_tok//n:,}")
    if total_tok:
        top = sum(toks[-max(1, n // 5) :])
        typer.echo(f"top 20% of docs  {top/total_tok:.0%} of all tokens")
    typer.echo("\n── by doc_type ────────────────────────────────")
    for t, v in sorted(by_type.items(), key=lambda kv: -sum(kv[1])):
        typer.echo(f"{t:<12} {len(v):>4} docs  {sum(v):>12,} tok  (avg {sum(v)//len(v):,})")

    # Contextual retrieval (Phase 3) is O(D^2/c) per document -- the single most
    # counter-intuitive cost in the project. Compute it from the real distribution.
    cached = sum(0.1 * (t / chunk) * t + 1.25 * t for t in toks) / 1e6
    typer.echo(f"\ncontextual-retrieval input (cached, Haiku $1/MTok): ${cached:,.0f}")
    typer.echo("  scales as D^2/c — long documents dominate. See NUMBERS.md.\n")


@corpus.command("verify")
def corpus_verify() -> None:
    """Re-hash every file on disk against the manifest.

    The manifest has recorded a sha256 and byte count for every document since the
    first fetch, and nothing ever checked them. That is a gap, not a detail: a file
    that changes on disk after the manifest is written makes every downstream number
    -- token counts, chunk counts, capacity math -- silently wrong, while the
    manifest continues to look authoritative.

    Run this after any fetch, and before trusting a measurement.
    """
    from rag.schema import sha256_file

    path = settings.raw_dir / "manifest.jsonl"
    if not path.exists():
        typer.echo(f"no manifest at {path}")
        raise typer.Exit(1)

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    missing, size_bad, hash_bad, ok = [], [], [], 0

    for r in rows:
        p = Path(r["path"])
        if not p.exists():
            missing.append(r["doc_id"])
            continue
        actual = p.stat().st_size
        if actual != (r.get("bytes") or -1):
            size_bad.append((r["doc_id"], r.get("bytes"), actual))
            continue
        if r.get("checksum") and sha256_file(str(p)) != r["checksum"]:
            hash_bad.append(r["doc_id"])
            continue
        ok += 1

    typer.echo(f"\n{len(rows)} documents in manifest")
    typer.echo(f"  verified          {ok:>5}")
    typer.echo(f"  missing on disk   {len(missing):>5}")
    typer.echo(f"  size mismatch     {len(size_bad):>5}")
    typer.echo(f"  checksum mismatch {len(hash_bad):>5}")

    for doc_id, expected, actual in size_bad[:20]:
        typer.echo(f"    {doc_id:44} manifest {expected:>12,}  disk {actual:>12,}")
    for doc_id in missing[:10]:
        typer.echo(f"    MISSING  {doc_id}")
    for doc_id in hash_bad[:10]:
        typer.echo(f"    HASH     {doc_id}")

    if missing or size_bad or hash_bad:
        typer.echo("\nRe-fetch the affected files: delete them and run `rag corpus fetch`.")
        raise typer.Exit(1)
    typer.echo("\nall documents match the manifest.\n")


@corpus.command("profile")
def corpus_profile() -> None:
    """Measure the corpus before deciding how to process it (Step 0.7).

    Every metric printed here changes a downstream decision. See
    rag/ingest/profile.py for the metric → decision mapping.
    """
    from rag.ingest.profile import profile_corpus

    path = settings.raw_dir / "manifest.jsonl"
    if not path.exists():
        typer.echo(f"no manifest at {path} — run `rag corpus fetch` first")
        raise typer.Exit(1)

    profiles, dupes = profile_corpus(path)

    hollow = [p for p in profiles if p.hollow]
    solid = [p for p in profiles if not p.hollow]

    typer.echo("\n── content check: does the document contain what we think? ──")
    typer.echo(f"documents            {len(profiles):>6}")
    typer.echo(f"  with statements    {len(solid):>6}")
    typer.echo(f"  hollow (Item 8 →)  {len(hollow):>6}   {len(hollow)/len(profiles):.0%} of corpus")
    if hollow:
        typer.echo("\n  hollow documents — financials live in a separate exhibit:")
        for p in sorted(hollow, key=lambda p: p.company)[:12]:
            typer.echo(f"    {p.company:28} FY{p.fy}  {p.tokens:>7,} tok  {p.tables:>4} tables")

    from rag.ingest.profile import find_outliers

    outliers = find_outliers(profiles)
    if outliers:
        typer.echo("\n  size outliers vs the same company's other years:")
        for p, why in outliers:
            typer.echo(f"    {p.company:28} FY{p.fy}  {why}")

    typer.echo("\n── tables (Phase 2: is table extraction THE requirement?) ──")
    tot_t = sum(p.tables for p in profiles)
    typer.echo(f"total <table> elements  {tot_t:>10,}")
    typer.echo(f"mean per document       {tot_t // len(profiles):>10,}")
    if solid:
        st = sorted(p.tables for p in solid)
        typer.echo(f"solid docs min/p50/max  {st[0]:,} / {st[len(st)//2]:,} / {st[-1]:,}")

    typer.echo("\n── near-duplicate overlap between consecutive years ──")
    typer.echo("   (drives Phase 6 dedup/MMR and Phase 8 conflict resolution)")
    allv = [v for pairs in dupes.values() for _, v in pairs]
    for co, pairs in sorted(dupes.items(), key=lambda kv: -sum(v for _, v in kv[1]))[:8]:
        cells = "  ".join(f"{lbl} {v:.0%}" for lbl, v in pairs)
        typer.echo(f"  {co:28} {cells}")
    if allv:
        typer.echo(f"\n  mean consecutive-year overlap: {sum(allv)/len(allv):.0%}")

    typer.echo("\n── year coverage (a conflict slice needs ≥2 years/company) ──")
    by_co: dict[str, list[str]] = {}
    for p in profiles:
        by_co.setdefault(p.company, []).append(p.fy)
    full = sum(1 for v in by_co.values() if len(v) >= 2)
    typer.echo(f"  companies with ≥2 years: {full}/{len(by_co)}\n")


index_app = typer.Typer(no_args_is_help=True, help="Build and query the vector index")
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build(
    limit: int = typer.Option(0, help="Only the first N documents (0 = all)"),
    recreate: bool = typer.Option(False, help="Drop and rebuild the collection first"),
    batch: int = typer.Option(256, help="Chunks embedded/upserted per call"),
) -> None:
    """Run the Phase 0 pipeline: parse -> chunk -> embed -> upsert."""
    from rag.pipeline.naive import ingest

    typer.echo(f"provider={settings.embed_provider}  model={settings.embed_model}")
    if settings.embed_provider == "stub":
        typer.secho(
            "  stub embedder: vectors are deterministic noise. This proves PLUMBING, "
            "not quality.",
            fg="yellow",
        )

    r = ingest(limit or None, batch=batch, recreate=recreate, progress=typer.echo)

    typer.echo(f"\nrun {r['run_id']}")
    typer.echo(f"  documents        {r['documents']:>8,}   failed {r['failed']}")
    typer.echo(f"  chunks           {r['chunks']:>8,}")
    typer.echo(f"  points written   {r['points_written']:>8,}")
    typer.echo(f"  elapsed          {r['seconds']:>8}s   ({r['chunks_per_sec']} chunks/s)")
    for f in r["failures"][:10]:
        typer.secho(f"    FAILED {f['doc_id']}: {f['error']}", fg="red")


@index_app.command("info")
def index_info() -> None:
    """Collection stats and which payload indexes exist."""
    from rag.index.qdrant_store import QdrantStore

    for k, v in QdrantStore().info().items():
        typer.echo(f"  {k:20} {v}")


@index_app.command("search")
def index_search(
    query: str = typer.Argument(..., help="Natural-language query"),
    limit: int = typer.Option(5, help="Results to return"),
) -> None:
    """Search the index."""
    from rag.pipeline.naive import search

    if settings.embed_provider == "stub":
        typer.secho("  stub embedder: these rankings are meaningless.", fg="yellow")
    for i, hit in enumerate(search(query, limit=limit), 1):
        typer.echo(f"\n{i}. score={hit['score']:.4f}  {hit['doc_id']}  ({hit['doc_type']})")
        typer.echo(f"   {hit['text'][:220]}")


@app.command()
def config() -> None:
    """Print resolved settings."""
    for k, v in settings.model_dump().items():
        if "key" in k and v:
            v = str(v)[:8] + "…"
        typer.echo(f"{k:<20} {v}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
