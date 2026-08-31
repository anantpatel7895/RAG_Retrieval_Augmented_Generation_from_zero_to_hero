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

<!-- Step 5 onward appended as we go. -->
