# Lessons

Transferable principles, each earned from a specific incident in this repo. Not facts
to recall — **judgment to calibrate**. Every entry names the mistake that produced it,
because a principle without its scar tissue doesn't stick.

Distinct from [question_answer.md](../question_answer.md), which is recall practice
for things you'd be *asked*. These are things you'd be *burned by*.

## How this is organised — and why not by phase

**Grouped by the situation that triggers the lesson, not by the phase it came from.**

Lessons here are cross-cutting by nature. L1 came from five separate incidents across
four different steps; filing it under "Phase 0" would hide the fact that it *recurs*,
which is the whole reason it is a lesson and not a bug report. And you will never
retrieve these by phase — you will reach for one when you notice yourself *about to
trust a metadata field*, or *about to quote a number*. That is retrieval by situation.

The phase axis already belongs to `question_answer.md`. For chronology, see the
[origin index](#origin-index) at the end.

| Group | Reach for it when… |
|---|---|
| [A. Trusting your inputs](#a-trusting-your-inputs) | You're about to believe a field, a file, or a source |
| [B. Measurement discipline](#b-measurement-discipline) | You're about to produce or quote a number |
| [C. Sampling & inference](#c-sampling--inference) | You're generalising from a subset |
| [D. Scope & complexity](#d-scope--complexity) | You're deciding whether to build something |
| [E. Process](#e-process) | You're deciding where work lives |

---

## A. Trusting your inputs

### L1. Silent under-delivery is the default failure mode of data pipelines

Data code rarely crashes. It returns plausible-looking **less**.

Five instances in Phase 0 alone, none of which raised an exception:

| Incident | Symptom | Reality |
|---|---|---|
| Dollar-amount scan | `0` amounts | A bank's annual report obviously has thousands |
| Submission-size filter | Quietly skipped filings | Would have rejected *every* large bank |
| Filing enumerator | Returned 1 filing | Asked for 4; `recent` block only spans 12 months for prolific filers |
| `primaryDocument` field | Downloaded fine, HTTP 200 | 12 of 60 documents were cover-page wrappers with no financials |
| Corrupt file on disk | 119/120 files correct | One file 19,075,544 bytes against 12,927,325 recorded |

**Apply:** for every extraction step, ask *"if this silently returned half the data, would anything fail?"* If not, add the check that would.


### L3. When a textual signal and a structural signal disagree, trust the structural one

The hollow-document detector used an Item 8 text pattern. It flagged Capital One — a filing with **501 tables and 274k tokens**, unmistakably a full report. The regex had matched its table of contents.

Table *density* separated the populations cleanly: pointer documents sit at 3–5 tables per 10k tokens, real reports at 15–18.

**Apply:** treat the disagreement as a bug in the cheap signal, not as ambiguity in the data.


### L4. Prefer a simple rule plus a verification pass over a clever rule you must trust

Finding the real document in an EDGAR submission, v1: *"largest `.htm` sharing the primary's filename stem."* The stem check was meant to exclude exhibits.

It silently failed on US Bancorp FY2022, filed through an agent as `d410791d10k.htm` (1.0 MB) and `d410791dex13.htm` (**6.7 MB**). Different stems — the clever rule kept the 1 MB wrapper.

v2: *"largest `.htm`, excluding `R\d+.htm`."* One rule, no special cases, with the profiler re-checking every document afterwards.

**Apply:** a rule's residual risk is acceptable when a cheap verification pass catches its failures. Complexity added to avoid needing verification is usually a bad trade.


### L5. Peer-relative comparison catches what absolute thresholds cannot

Wells Fargo's FY2025 filing defeated both hollow-document detectors: it never says "incorporated by reference", and its 40 tables over 23k tokens read as *high* density.

What caught it: **23,354 tokens against its own prior years' 102k / 97k / 84k.** Adjacent years of the same filer are the tightest control available.

**Apply:** when an absolute threshold has blind spots, compare each record against its own history or its cohort.


### L14. Report *measured* state, never *requested* state

`docker/gpu-compose.yml` advertised `--dtype=float16`. `scripts/embed_server.py` never
implemented it — `SentenceTransformer` does not infer dtype, so it silently loaded
fp32: **2.3 GB of weights instead of 1.2, at roughly half the throughput**, on a GPU
shared with someone else's job.

Nothing errored. The server was healthy, embeddings were correct, and the config file
asserted a setting the code did not have. **Documentation claiming a capability that
doesn't exist is worse than not offering it**, because you stop looking.

What caught it was one field in `/info`:

```python
"model_dtype": str(next(model.parameters()).dtype),   # measured from the loaded model
```

Had it been written the obvious way —

```python
"model_dtype": DTYPE,                                  # echoes the env var back
```

— the endpoint would have reported `float16` with total confidence and the bug would
have survived indefinitely. **The endpoint would have been lying about itself.**

This is the same shape as the checks that already exist in this project, and it is
worth seeing them as one pattern:

| Check | Reads actual state | Instead of trusting |
|---|---|---|
| `verify_contract` | Embeds a probe, measures dim and L2 norm | The config's claimed dimension |
| `rag corpus verify` | Re-hashes files on disk | The manifest's stored checksum |
| `/info` `model_dtype` | `next(model.parameters()).dtype` | The `EMBED_DTYPE` env var |
| `resolve_primary` | File sizes in `index.json` | EDGAR's `primaryDocument` field |

**Apply:** for every configuration knob that matters, expose a readback derived from
the live object, not from the setting that requested it — and compare the two. A
component that reports its inputs is a component that cannot tell you it failed to
apply them.

---

### L8. Verify integrity before measuring, not after

The manifest recorded a sha256 and byte count for every document from the first fetch. **Nothing checked them for weeks.**

One file was corrupt. Measuring against it produced a text ratio of 7.9% instead of 11.7% — and on that basis I "corrected" a figure in `PARSING.md` from 780/23% to 636/19%, inventing a plausible explanation about HTML entities. The original number had been right.

**Fixing a correct number using bad data is worse than the original bug**, because the fix carries a confident rationale.

**Apply:** integrity check first, measurement second. `rag corpus verify` now runs before any measurement, and 119-of-120-correct is exactly what makes corruption dangerous.

---

## B. Measurement discipline

### L2. A plausible number gets trusted. An implausible one gets caught.

The `$`-separation metric took three attempts:

| Attempt | Result | Fate |
|---|---|---|
| 1 | 0 dollar amounts | **Caught instantly** — impossible on its face |
| 2 | "90% of amounts are split" | **Survived into a document** — sounded reasonable |
| 3 | 10%, confirmed by cell counting | Correct |

Attempt 2 was the dangerous one *because it was believable*. Attempt 1's absurdity was its own alarm.

**Apply:** predict the answer's rough magnitude **before** measuring, and verify any headline number by a **structurally different** method — here, counting table cells rather than matching text.


### L7. Never quote a metric without its denominator

Two versions of this bit within one week:

- **Units.** 12.9 MB on disk is 11.7% text, ~378k tokens, ~946 chunks, 3.9 MB of vectors. Bytes for disk planning, tokens for cost, chunks for the index — mixing them silently is how a capacity estimate lands 4× off.
- **Aggregation level.** A chart titled "0% to 79%" plotted *company means* (1%–62%) while the range quoted was *per-filing* (0%–85%). Both true, different populations.

**Apply:** state the unit and the aggregation level in the same breath as the number.


### L10. Never compare estimates produced with different sampling parameters

`pdf_probe` samples pages for table detection. Citigroup FY2025 yields `tables_estimated: 51` at `table_sample=12` and ~190 at `table_sample=25`.

Both are valid estimates of the same quantity. Neither is comparable to the other.

**Apply:** fix sampling parameters once per study and record them alongside every derived number. This is why `pdf_probe` returns `pages_sampled` and `tables_sampled` next to the estimate rather than hiding the extrapolation.

---

## C. Sampling & inference

*Phase 1 will add the statistical-power lessons — how small a change your eval can actually see.*

### L6. A small sample from a skewed distribution is biased in the direction you sampled

| Sample | Corpus vs EDGAR average |
|---|---|
| 3 documents (JPM, BofA, Citi) | 5.7× |
| 60 documents | **3.5×** |

The first three were the *largest* of fifteen large banks — the extreme tail of an already-skewed selection. The 3-sample estimate was 63% high.

**Apply:** for skewed data the mean is not the typical value. Report median and tail, and always state which slice of the population you sampled. Analogy: average adult height ~1.7 m; average NBA centre ~2.1 m. Neither is wrong; quoting one while sampling the other is.

---

## D. Scope & complexity

### L9. Complexity must be earned — and say so out loud when it isn't

Our corpus is 71,248 chunks. Brute-force search over 71k × 1024 dimensions is **~7 ms**. We do not need an ANN index.

We are building one anyway, because Phase 5 depends on understanding `ef_search`, filtered traversal, and the recall/latency curve — and those can only be learned by operating them.

That is a legitimate reason. Pretending the index was *necessary* would not be.

**Apply:** separate "needed for this system" from "needed for this engineer". Both justify work; conflating them produces architecture nobody can defend. If someone proposes a vector database for 50k chunks, the correct answer is a numpy array.


### L13. Never upgrade a stateful service in place to silence a warning

`qdrant-client` 1.19.0 warned on every call that the server was 1.12.4. The fix chosen was to bump the **server** seven minor versions. It panicked on startup:

```
Failed to deserialize segment.json: unknown variant `on_disk`,
                                    expected `mmap` or `in_ram_mmap`
```

v1.12.4 had written `on_disk` into its segment metadata; v1.19.0 renamed that enum and refuses to read the old value. Forward-incompatible storage format, exit code 101.

The low-risk fix was the **opposite direction** — pin the client *down* to 1.12.x and leave the data alone. Changing the stateless component is nearly always cheaper than changing the one holding your data.

We escaped because the volume held 1,719 disposable stub vectors. Had it held 71,248 real embeddings, a cosmetic version bump would have destroyed hours of GPU time.

**Two things worth separating.** Qdrant behaved *correctly*: it crashed loudly rather than starting empty or skipping the unreadable segment. A database that refuses to open state it cannot parse is the right contract — contrast every entry in group A above.

**Apply:** version-bump the stateless side first. For the stateful side, never in place — stand the new version up alongside, backfill, shadow-read, cut over, keep the old until confident. That is Phase 11's D4 exercise, and this was it at 1/1000th scale for the price of five minutes.

---

### L11. A prepared dataset removes the problem you are trying to learn

`TeraflopAI/SEC-EDGAR` offers 8M filings, 43B tokens, already parsed with selectolax and doc2dict.

Using it as the working corpus would have deleted Phase 2 entirely — someone else's parsing decisions, inherited and unmeasurable. But for Phase 5's scale test, where the need is *volume not mess*, it is exactly right: one 616 MB shard ≈ 243k chunks, against 10+ days of polite crawling.

**Apply:** ask what the dataset is *for*. Convenience is correct when the removed difficulty isn't the point, and fatal when it is.

---

## E. Process

### L12. Research belongs in a notebook; documents cite it

`docs/PARSING.md` was first written from ~15 ad-hoc shell scripts. The numbers were unverifiable and frozen — no way to re-run, adjust a threshold, or check the arithmetic.

Retrofitting it as a notebook (with logic moved to `rag/ingest/format_probe.py`) immediately surfaced two wrong figures and one corrupt file.

**Apply:** three layers — measurement logic in `rag/`, research in `notebooks/NN_topic.ipynb`, conclusions in `docs/`. **Every number in a document must be traceable to a notebook cell.** Docstring examples become doctests, so documentation that drifts from the code fails the suite.

---

## Origin index

Chronology, for when you want it. Note how few lessons map to a single step — that is
the argument against organising this file by phase.

| Lesson | Where it came from |
|---|---|
| L1 Silent under-delivery | Steps 0.2, 0.6, 0.7 + the corrupt-file find — five incidents |
| L2 Plausible numbers get trusted | Parsing research (the `$` metric) |
| L3 Structural over textual signals | Step 0.7, hollow-document detector |
| L4 Simple rule + verification | Step 0.6, EDGAR document resolution |
| L5 Peer-relative comparison | Step 0.7, Wells Fargo FY2025 |
| L6 Small samples from skewed data | `docs/DATA.md`, 3 documents → 60 |
| L7 Always quote the denominator | `docs/DATA.md` and `docs/PARSING.md` both |
| L8 Verify integrity before measuring | Notebook 02 retrofit |
| L9 Complexity must be earned | Step 0.5 planning, 71k chunks |
| L10 Don't mix sampling parameters | `pdf_probe` table estimates |
| L11 Prepared datasets remove the problem | Dataset-vs-raw-fetch decision (ADR 0001) |
| L13 Never upgrade stateful in place | Step 0.5, Qdrant 1.12→1.19 storage panic |
| L14 Report measured, not requested, state | Step 0.5, embed server silently ran fp32 |
| L12 Research in notebooks | The `PARSING.md` retrofit |
