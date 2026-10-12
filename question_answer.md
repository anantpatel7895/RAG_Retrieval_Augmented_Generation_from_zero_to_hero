# Question & Answer Log

Running log of questions worth being able to answer **cold** — in an interview, in a design review, or at 3am during an incident.

Every step appends to this file. Reference answers are written concisely; the exercise is to close the file and re-answer in your own words. A question you can only answer by reading it back is not learned yet.

**Legend:** ⭐ = asked in almost every RAG/system-design interview · 🔧 = you will need this during an actual incident

---

# Phase 0 — Environment & foundations

## Step 1 — Interpreter and environment pinning

### Q1.1 🔧 Why can a project "work" but silently produce wrong benchmark numbers because of the Python environment?
Version-manager configs are **hierarchical and inherited**. `pyenv` walks up from the current directory looking for `.python-version`, so a repo without its own file silently adopts a parent's interpreter. Here the repo sat under `~/Anant_Learning/`, which pins 3.11.7, while the intended env was `RAG_VENV` on 3.12.10. Nothing errors. You get a different interpreter, a different resolved dependency set, and — in an ML project — potentially different numerical results between the run you did manually and the run your script did. The fix is an explicit `.python-version` committed at the repo root.

### Q1.2 Why does dependency pinning matter more in a RAG project than in a typical service?
Because your outputs are **measurements**, not just behaviour. An unpinned `pymupdf` bump changes text extraction; an unpinned `sentence-transformers` bump can change tokenization or pooling. Either shifts your eval scores. You then attribute the change to the retrieval work you were doing, and chase a ghost for a day. Reproducible evals require a locked dependency set.

---

## Step 2 — Environment topology (what runs where)

### Q2.1 ⭐ Why serve the embedding model over HTTP instead of importing it in-process?
Three reasons, in order of importance:
1. **It is a remote call in production anyway.** Building against an in-process model and "productionizing later" means every failure path (timeout, retry, backpressure, degradation) is untested until it matters.
2. **Dependency isolation.** CUDA-versioned `torch` wheels never enter the client environment.
3. **Independent scaling.** The embedder scales on GPU throughput; the API scales on request concurrency. Coupling them wastes one or the other.

### Q2.2 ⭐ The embedding service serves two workloads. What are they, and why does that matter architecturally?
| | Ingest | Query |
|---|---|---|
| Shape | Hundreds of thousands of texts, large batches | One text |
| Bound by | GPU throughput | Network RTT + latency SLO |
| On failure | Retry, resume, backpressure | Fail fast, degrade |
| Scale on | Queue depth | p95 latency |
| Tolerance | Minutes | Tens of milliseconds |

Point both at one instance and a nightly reindex saturates the GPU while query p95 goes from 40 ms to seconds. Nothing is "down" — users just see timeouts. In production these are **separate deployments**; the `role` parameter in our provider is the seam that makes that a config change instead of a refactor.

---

## Step 3 — Configuration and schema

### Q3.1 Why must every eval run serialize its full config?
Otherwise "what changed between run 41 and run 42" is guesswork. With hundreds of runs across 14 phases, an unattributable score change is worthless — you cannot tell an improvement from noise from a config drift.

### Q3.2 ⭐ Why is chunk schema a one-way door?
Fields you did not capture cannot be backfilled without re-parsing the corpus. Discovering in Phase 8 that you need `page` and `bbox` for citation highlighting, when you never stored them, means re-running ingestion over every document. Declaring an unused field costs nothing; omitting a needed one costs a full re-parse.

---

## Step 4 — The embedding model contract

### Q4.1 ⭐⭐ Name the two embedding failures that produce no error at all.
1. **Asymmetric prefix bug.** Models like the `e5` family are trained with `query: ` on questions and `passage: ` on documents. Use the same prefix on both sides and you lose roughly 5–15 points of recall. Nothing crashes, nothing logs — the eval just reads worse, and you go hunting in the chunker.
2. **Dimension drift.** Config declares 1024, the served model emits 768. If the collection was created from the same wrong config, everything "works" and the index is meaningless.

Both are silent, which is what makes them expensive. Defence: a version-controlled model registry that owns dim and prefixes, plus a startup contract check that embeds a probe and verifies dimension and L2 norm empirically.

### Q4.2 Why verify the contract by embedding a probe rather than reading a metadata endpoint?
What we actually depend on is **what the vectors are**, not what a service claims to serve. A metadata endpoint can be correct while the loaded weights are wrong, and it may not report dimension at all. Measuring the output tests the property we care about.

### Q4.3 Why does normalization matter for cosine similarity?
Cosine similarity assumes unit vectors. If the model does not normalize and you treat dot product as cosine, vector magnitude leaks into the score — and magnitude correlates with text length. Longer chunks then score higher regardless of relevance.

### Q4.4 Why a model registry in code instead of environment variables?
The registry is a **versioned contract**, and an unregistered model raises instead of defaulting to empty prefixes. Environment variables leave the prefix bug one typo away with nothing to validate it. Adding a model means adding an entry, which forces reading the model card once — that friction is the feature. Structurally it is the same pattern as a schema registry rejecting an incompatible producer.

---

## Step 5 — The vector store (concept)

### Q5.1 ⭐⭐ Why is exact nearest-neighbour search unusable at scale? Show the arithmetic.
```
10M chunks × 1024 dims × 4 bytes = 41 GB just to hold the vectors
10M × 1024                       = 10.2 billion multiply-adds per query
```
Single-threaded that is ~1 second; heavily vectorised across 16 cores, ~100–200 ms — **per query**. Ten concurrent users saturate the machine, and you paid 41 GB of RAM for it.

### Q5.2 ⭐ What does an ANN index trade away, and how is it different from a B-tree?
A B-tree returns *the* answer with a complexity guarantee. An ANN index returns a **probable** answer with a **recall characteristic you tune**. You give up exactness for roughly 1000× speedup. You cannot reason about it the way you reason about a database index — and "index recall" is a distinct thing from "retrieval recall", which is distinct again from "did the user get a good answer".

### Q5.3 ⭐⭐ Explain HNSW. Why does greedy search terminate, and what does that *not* guarantee?
Two ideas stacked:
1. **Navigable small world** — each node links to near neighbours plus a few far ones. Long links cover distance fast, short links refine.
2. **Hierarchy** — each node gets a max layer from an exponentially decaying distribution, so layer 0 holds everything and upper layers are sparse. It is a **skip list in metric space**.

Search: enter at the top-layer entry point; at each layer greedily move to whichever neighbour is closer to the query; stop at a local minimum; drop a layer and repeat; at layer 0 switch to beam search of width `ef_search`.

**Why it terminates:** every greedy step strictly decreases distance to the query, distance is bounded below by zero, and the graph is finite. **What that does not guarantee is correctness** — the guarantee is termination, at a *local* minimum.

**Why the hierarchy matters:** it buys *quality* of that local minimum, not just speed. Coarse long jumps in upper layers land you in the right neighbourhood before layer 0, so the local minimum is very likely the global one.

### Q5.4 🔧 You have `ef_search=64` and recall 0.85; you need 0.95. What do you change, and what can't you touch?
Raise `ef_search` — a wider beam explores more of the graph and escapes more local minima, at the cost of more distance computations and latency. It is the **only** parameter tunable at query time without a rebuild, which makes it the 3am knob. `M` and `ef_construct` are baked into the graph at build time.

### Q5.5 Why is connecting a new node to its M *nearest* neighbours wrong?
It clusters all links in one direction and destroys navigability. HNSW uses a **diversity heuristic**: prefer a candidate if it is closer to the new node than to any already-selected neighbour. Varied link directions are what make greedy routing work at all.

### Q5.6 ⭐⭐ Why is filtered ANN hard? Name the three strategies and how each fails.
| Strategy | Fails when |
|---|---|
| **Post-filter** — search then drop non-matching | The filter is selective: ask for 10, get 2. Worse as it narrows |
| **Pre-filter + brute force** — scan the matching set | The match set is large; 2M matches is a linear scan |
| **Filter during traversal** — skip non-matching nodes | **Breaks the graph**: the path to the answer may run through excluded nodes, so the walk stalls in a disconnected region and recall collapses *silently* |

Real systems estimate the filter's **cardinality** first and switch strategy, plus build extra links so common filtered subgraphs stay connected. **Payload indexes are what make cardinality estimation possible** — without one the engine can neither estimate nor filter efficiently and degrades to a scan. Passes in dev, dies at 500k.

### Q5.7 ⭐ Why is a vector database not just "a database that stores vectors"?
Postgres can store vectors; FAISS can do ANN. What the database adds: **filtered search that preserves recall** (the genuinely hard part), live insert/update/delete without a rebuild, persistence/snapshots/replication/sharding, and payload storage with indexes. If your system had no filters, `faiss` plus a pickle file would genuinely be enough.

### Q5.8 🔧 Why does HNSW have no clean delete?
Removing a node would strand the links pointing at it. Systems tombstone instead, and the graph degrades until rebuilt. This is a **compliance** problem, not just hygiene — a GDPR deletion that leaves the chunk retrievable is a breach.

### Q5.9 ⭐ Why does quantization decide whether you operate a distributed system?
At 100M chunks × 1024 dims: fp32 = **410 GB** (must shard — you now own routing, fan-out, partial failure, rebalancing); binary = **12.8 GB**, plus ~12.8 GB of HNSW links, which fits in RAM on one large node. It also *speeds up* search, because HNSW is memory-latency bound rather than FLOP bound.

### Q5.10 When is Qdrant the wrong choice?
When your corpus is a few million chunks and your metadata already lives in Postgres — **pgvector is very likely correct**, and adding a vector database is unjustified complexity. Qdrant earns its place here because filtering is central (Phases 5 and 12), it has native sparse vectors plus fusion, and strong quantization.

---

## Step 6 — Corpus acquisition

### Q6.1 🔧 Why can't you trust a metadata field that names the "primary" document?
EDGAR's `primaryDocument` pointed at a cover-page wrapper for 12 of our 60 filings. Worse, the naming convention **inverted between consecutive years for the same company**:
```
Wells Fargo FY2025:  API named wfc-20251231_d2.htm (360 KB) → real content wfc-20251231.htm    (11.6 MB)
Wells Fargo FY2024:  API named wfc-20241231.htm            → real content wfc-20241231_d2.htm (10.9 MB)
```
Any name-based rule breaks on one or the other. Size was the reliable signal. See [L3](docs/LESSONS.md) and [L4](docs/LESSONS.md).

### Q6.2 What three properties must a corpus fetcher have, and what does each prevent?
- **Polite** — rate limiting. SEC allows ~10 req/s; exceed it and the block is *silent*, so the first symptom is a directory of zero-byte files.
- **Resumable** — a 400-document fetch will die partway; without resume you restart from zero.
- **Content-addressed** — hashing is what makes re-ingestion a no-op instead of a full re-parse. It's also the same mechanism Phase 11 needs for incremental sync.

### Q6.3 Why does a paginated API return fewer records than you asked for, silently?
EDGAR splits history into a `recent` block plus dated archive shards. `recent` holds roughly the last 1,000 filings **by count, not by time**. JPMorgan files ~26,000 documents a year (mostly Form 4 insider trades), so its `recent` block spans ~12 months and contains exactly **one** 10-K. Asking for 4 years returned 1, with no error.

### Q6.4 Why include an accession tail in the filename when company and year already identify it?
Because an original filing and a later 10-K/A amendment share company and year. Without the tail they collide and **silently overwrite** each other.

---

## Step 7 — Profiling before building

### Q7.1 ⭐⭐ In RAG, how do you tell a data problem from a retrieval problem?
**You can't, from the symptom** — both produce a bad answer. That is the entire argument for profiling before building. Ask *"What were Wells Fargo's total deposits in 2025?"* against a corpus whose Wells Fargo filing is a cover page, and the natural response is to blame retrieval and spend a week tuning chunkers, embedders and rerankers. None of it helps: the answer was never in the index.

The diagnostic ladder (Phase 9) needs recorded intermediate state: was the gold chunk in the retrieved set → in the final context → and still answered wrong? Retrieval, reranking, generation respectively. Never in the index at all → ingestion.

### Q7.2 What makes a profiling metric worth collecting?
It changes a decision. Table density → is table extraction *the* parser requirement. Empty-page ratio → is OCR routing needed or theatre. Token distribution → chunking strategy and whether contextual retrieval is affordable. Near-duplicate overlap → how hard Phase 6 dedup and Phase 8 conflict resolution will be. Anything merely *interesting* is not collected.

### Q7.3 Why did detecting hollow documents need three independent signals?
Each alone was wrong — see [L3](docs/LESSONS.md) and [L5](docs/LESSONS.md). Text pattern false-positived Capital One and missed Wells Fargo FY2025; table density missed WFC FY2025 too (40 tables over 23k tokens reads as *high* density); only peer-relative size caught it. **No single signal was sufficient — that is the lesson, not any threshold.**

### Q7.4 Our consecutive-year 10-Ks overlap 43%. Why does that matter?
Roughly half the text is verbatim identical across years, so a query like *"what are the credit risk factors"* retrieves four near-identical chunks from four years. Deciding which applies requires `effective_date` on every chunk plus explicit recency/conflict policy. It is what makes Phase 6 dedup/MMR and Phase 8 conflict resolution real rather than hypothetical.

---

## Parsing — HTML vs PDF

### Q8.1 ⭐⭐ What is the fundamental difference between HTML and PDF for extraction?
HTML is **markup** — it describes what things *are*, so extraction is *reading*. PDF is a **display list** — it describes where ink *goes*, so extraction is *reconstruction*. A PDF stores words with coordinates and no notion of row, cell, column, heading or reading order. Consequence: parser choice barely matters for HTML and dominates everything for PDF.

### Q8.2 ⭐⭐ Why does flattening a table destroy retrieval even when every character survives?
It destroys the **binding** between row header, column header and value — and that binding *is* the answer. The chunk stops being *about* "Tier 1 capital in 2024" and becomes a bag of numbers, so its vector lands nowhere useful and the query matches nothing.

Measured concretely: **2,922 table cells contain nothing but `$`**, and 87% of those are immediately followed by a cell containing only a number. The currency symbol occupies its own cell. **No text processing can fix this** — no regex, no whitespace rule, no tag heuristic reunites them across a cell boundary. Only grid-aware table parsing can.

### Q8.3 Naive tag-stripping vs inline-aware — how much does it buy?
| Method | `$` still attached to its value |
|---|---|
| Every tag → space | 328 (**10%**) |
| Inline tags → `''`, block tags → space | 780 (**23%**) |

Roughly doubles it, and leaves 77% broken because the separation is structural (Q8.2).

### Q8.4 ⭐ What does inline XBRL give you that nothing else does?
Every reported figure is wrapped with its **concept name, unit and scale**:
```html
<ix:nonFraction unitRef="usd" name="dei:EntityPublicFloat" scale="0">794,433,813,496</ix:nonFraction>
```
`dei:EntityPublicFloat` says exactly what the number *means*. Extracting `name`/`unitRef`/`scale` into chunk metadata is cheap and is arguably the most under-exploited signal in financial RAG.

### Q8.5 ⭐⭐ You measured PDF table extraction against HTML ground truth. What did you find and what does it imply?
Median **14.4%** recovery, per-filing range **0% to 85.4%**, company means **1.1% to 62.3%**. In absolute terms: HTML declares 19,849 tables, PDF extraction finds 4,051.

The same parser, same document type, same industry, varies by more than an order of magnitude **based only on who published the file**. Implications: a parser cannot be validated on a sample; "we use `docling`, it's good at tables" is not an engineering statement; and parser routing must be **per document on a measured signal**, not a global config choice.

### Q8.6 🔧 How does a scanned page fail, and how do you detect it?
It extracts as an **empty string with no error**, so the document silently contributes nothing to the index. Citizens Financial's FY2024 annual report: 194 pages, **167 (86%) with no text layer**, 116 images in the first 40 pages. Detect by measuring text yield per page (chars ÷ page area, or simply blank pages) and routing low-yield pages to OCR or a VLM.

### Q8.7 How do you make a parser bake-off scoreable without hand-labelling pages?
Exploit a document published in **both** formats. Companies file the 10-K as HTML and the annual report as PDF for the same fiscal year, so the HTML's declared structure is free ground truth. Extract from the PDF, compare against the HTML, score. That replaces the 30 hand-labelled pages Phase 2 originally budgeted.

---

## Step 5b — Serving the embedder (ML infra)

### Q5b.1 ⭐⭐ A config file says `--dtype=float16` and the service runs fp32. How would you ever notice?
Only by comparing **reported** state against **intended** state. Our `/info` derives it from the loaded object:
```python
"model_dtype": str(next(model.parameters()).dtype)   # measured
```
Written the obvious way — `"model_dtype": DTYPE` — the endpoint would have echoed the env var back and reported `float16` with total confidence. The service would have been lying about itself.

**The general rule:** every config knob that matters needs a readback derived from the live object, never from the setting that requested it. A component that reports its inputs cannot tell you it failed to apply them. See [L14](docs/LESSONS.md).

### Q5b.2 Why fp16 for embedding inference, and what does it cost?
Halves VRAM (BGE-M3: ~2.3 GB → ~1.2 GB) and roughly doubles throughput on an A10, which has dedicated fp16 tensor cores. Retrieval models are routinely served in fp16 with no measurable quality loss — unlike training, where reduced precision needs loss scaling. On a **shared** GPU the VRAM saving matters as much as the speed.

### Q5b.3 ⭐ On a shared multi-GPU box, why is `--gpus count=1` wrong?
Because `count` lets the scheduler choose, and it picks device 0. Our GPU 0 held another team's vLLM engine using 20.5 GB of 23 GB — TEI would have OOMed or pressured their job. `device_ids: ["1"]` (Docker) or `CUDA_VISIBLE_DEVICES=1` (process) pins it explicitly.

Corollary: on a single-GPU box `count: 1` works perfectly and you'd never learn this. Config that is correct-looking and wrong is the recurring theme of [group A](docs/LESSONS.md#a-trusting-your-inputs).

### Q5b.4 🔧 Why set the GPU at launch rather than in code?
The device is an **operational** decision on a shared machine, and it changes when the box's tenancy changes. Burying it in a source file is how a job silently ends up on someone else's GPU six months later. Related trap: `CUDA_VISIBLE_DEVICES` **remaps** indices, so PyTorch always sees its allotted card as `cuda:0` — logs must print `torch.cuda.get_device_name(0)` or "device 0" gets confused with physical GPU 0.

### Q5b.5 ⭐ Why is `max_seq_length=512` a big deal when the model supports 8192?
Attention is quadratic in sequence length. Our chunks are ~400 tokens, so capping at 512 is a large speedup for zero quality loss **on our data**. The dependency runs the other way too: if Phase 3's chunker starts producing longer chunks, this cap silently truncates them — so the two settings are coupled and must move together.

### Q5b.6 ⭐⭐ Which errors should a retry policy cover?
Only plausibly transient ones: timeouts, read errors, 5xx. **Not** `ConnectError` — a refused connection means nothing is listening, and the host will not start listening within a 20-second backoff.

We learned this by retrying it: one wrong port turned into 5 exponential-backoff attempts, ~30 seconds, and 200 lines of stack trace that buried the actual cause. **Slow failure is worse than fast failure, because it hides the reason.**

### Q5b.7 Why did the embedding server return 501 for `/rerank` instead of implementing it?
Reranking needs a **cross-encoder** — query and document encoded jointly. This server hosts a bi-encoder. Answering `/rerank` with bi-encoder similarity would *look* like a working reranker while delivering none of the benefit, and nothing downstream could detect the substitution. Failing loudly is the correct contract.

### Q5b.8 🔧 A service is healthy on the box, Chrome reaches it, but `curl` gets `ECONNREFUSED`. What's happening?
Almost always a proxy: a managed browser honours the system PAC/proxy config automatically, while `curl` connects directly unless `http_proxy` is set. Note that `ping` succeeding proves nothing here — ICMP uses neither TCP nor proxies.

Distinguish the two failure modes by the error: **refused** means the host is reachable and sent a TCP reset (nothing listening on that port, or a rejecting firewall). **Timeout** means packets vanished (dropping firewall, or no route).

### Q5b.9 Why tunnel over SSH rather than open the port?
Two problems, one fix. The firewall permitted TCP 22 but not 8002, and **TEI has no authentication** — a reachable port would let anyone on that network consume your GPU. `ssh -N -L 8002:localhost:8002 user@host` multiplexes the traffic inside an already-permitted connection and supplies the auth the service lacks.

---

## Cross-cutting

### Q9.1 ⭐ Why must you predict a measurement's magnitude before running it?
Because a **plausible wrong number gets trusted**. A scan reporting *zero* dollar amounts in a bank's annual report was caught instantly; the same metric reporting *"90% split"* survived into a document. See [L2](docs/LESSONS.md).

### Q9.2 🔧 You store checksums in your manifest. What's the point if nothing checks them?
None — and that was the actual situation for weeks. One file was 19,075,544 bytes on disk against 12,927,325 recorded. 119 of 120 were correct, which is precisely what makes it dangerous: one wrong file among a hundred right ones yields a plausible number, not an error. Worse, measuring against it led to "correcting" a figure that had been right. See [L8](docs/LESSONS.md).

### Q9.3 ⭐ When is it legitimate to build something your system doesn't need?
When the goal is understanding rather than throughput — but you must say so. Our 71,248 chunks brute-force in ~7 ms; we do not need ANN. We build it because Phase 5 depends on operating `ef_search`, filtered traversal, and the recall/latency curve, and those can't be learned from reading. Separate "needed for this system" from "needed for this engineer"; both justify work, and conflating them produces architecture nobody can defend. See [L9](docs/LESSONS.md).
