# Vector Search Indexes: A Detailed Guide

## Table of Contents

1. Introduction
2. Why Do We Need Indexes?
3. Exact Search vs Approximate Search
4. Flat Index (Exact Search)
5. HNSW (Hierarchical Navigable Small World)
6. IVF (Inverted File Index / IVFFlat)
7. Product Quantization (PQ)
8. Scalar Quantization (SQ)
9. Comparison Table
10. Choosing the Right Index
11. Common Combinations
12. Performance Metrics
13. Practical Examples
14. Summary

---

# Introduction

Modern AI applications such as semantic search, Retrieval-Augmented Generation (RAG), recommendation systems, image search, and similarity search store data as **vectors (embeddings)**.

Instead of searching text directly, we search vectors.

Example:

```
Text:
"I love machine learning."

Embedding:
[0.123, -0.992, 0.445, ..., 0.381]
```

Every document, image, or audio clip becomes a high-dimensional vector (typically 384–3072 dimensions).

When a user asks a question, we convert the query into another vector and search for the **nearest vectors**.

This operation is called **Nearest Neighbor Search (NNS)**.

---

# Why Do We Need Indexes?

Imagine you have:

* 10,000 vectors
* 1 million vectors
* 100 million vectors
* 1 billion vectors

Searching every vector one by one becomes increasingly expensive.

Without an index:

```
Query

↓

Compare with Vector 1
Compare with Vector 2
Compare with Vector 3
...
Compare with Vector N

↓

Return closest vector
```

Time complexity:

```
O(N)
```

As datasets grow, latency becomes unacceptable.

Indexes reduce the number of comparisons while attempting to return almost the same results.

---

# Exact Search vs Approximate Search

There are two major approaches.

## Exact Search

Returns the mathematically correct nearest neighbors.

Pros

* 100% recall
* Perfect accuracy

Cons

* Slow
* Expensive

---

## Approximate Nearest Neighbor (ANN)

Returns vectors that are *almost always* the nearest neighbors.

Pros

* Very fast
* Scales to billions of vectors

Cons

* May miss a few true neighbors

Typical recall:

```
95–99.9%
```

Most production AI systems use ANN.

---

# Flat Index (Exact Search)

Also called:

* Flat
* Brute Force
* Exhaustive Search
* IndexFlatL2 (FAISS)

---

## How it Works

No index is built.

Every query compares against every vector.

```
Query

↓

Vector 1
Vector 2
Vector 3
...
Vector N

↓

Sort distances

↓

Return Top-K
```

---

## Complexity

Search

```
O(N)
```

Memory

```
Only vectors
```

---

## Advantages

* Perfect accuracy
* No preprocessing
* No tuning
* Easy to implement

---

## Disadvantages

* Slow on large datasets
* Doesn't scale

---

## Best Use Cases

* Small datasets
* Evaluation
* Ground truth generation
* Benchmarking

---

## Example

Dataset

```
10,000 vectors
```

Query

```
Compare all 10,000 vectors
```

Simple and accurate.

---

# HNSW (Hierarchical Navigable Small World)

One of the most popular ANN algorithms.

Used by

* Milvus
* Qdrant
* Weaviate
* pgvector
* OpenSearch
* Elasticsearch

---

## Main Idea

Instead of checking every vector, HNSW creates a graph.

Each vector connects to nearby vectors.

```
A ----- B ----- C
|       |       |
D ----- E ----- F
        |
        G
```

Searching becomes graph navigation.

---

## Multi-Layer Structure

```
Layer 3

      A

Layer 2

   B ----- C

Layer 1

D -- E -- F -- G

Layer 0

Many vectors
```

Higher layers

* fewer nodes
* long-distance navigation

Lower layers

* many nodes
* local refinement

---

## Search Process

Suppose the query is near node G.

```
Start at top layer

↓

Move to closest node

↓

Descend

↓

Repeat

↓

Reach nearest neighbors
```

Instead of checking 10 million vectors, maybe only a few thousand nodes are visited.

---

## Important Parameters

### M

Maximum neighbors per node.

Higher M

* better recall
* more memory

Lower M

* smaller graph
* faster build

Typical

```
16
32
48
```

---

### efConstruction

Controls index quality.

Higher

* slower build
* better graph

Typical

```
100–400
```

---

### efSearch

Controls search accuracy.

Higher

* better recall
* slower search

Typical

```
50–300
```

---

## Advantages

* Extremely fast
* Excellent recall
* No clustering
* Good for dynamic inserts

---

## Disadvantages

* High RAM usage
* Slow index construction

---

## Best Use Cases

* Online semantic search
* Chatbots
* RAG
* Recommendation systems

---

# IVF (Inverted File Index)

Also called

* IVF
* IVFFlat

Instead of graph navigation, IVF partitions the dataset.

---

## Step 1

Cluster vectors.

Suppose we have

```
1 million vectors
```

Cluster into

```
1000 centroids
```

```
Centroid 1

○ ○ ○ ○

Centroid 2

○ ○ ○

Centroid 3

○ ○ ○ ○ ○
```

---

## Step 2

Query arrives.

Find nearest centroid.

```
Query

↓

Closest centroid

↓

Search only inside that cluster
```

Instead of searching

```
1,000,000 vectors
```

Search maybe

```
1,000 vectors
```

Huge speed improvement.

---

# Important Parameters

## nlist

Number of clusters.

Higher

* smaller clusters
* faster search
* more memory

Typical

```
1024
4096
8192
```

---

## nprobe

How many clusters to search.

Example

```
nlist = 1000

nprobe = 10
```

Only 10 clusters searched.

Higher nprobe

* higher recall
* slower

---

## Advantages

* Fast
* Memory efficient
* Great for huge datasets

---

## Disadvantages

* Needs training
* Sensitive to parameter tuning

---

## Best Use Cases

Large datasets

```
10M+
```

---

# Product Quantization (PQ)

PQ compresses vectors.

Instead of storing full floating-point values, it stores compact codes.

---

## Original Vector

```
[0.15,
-0.77,
0.22,
...
0.81]
```

Maybe

```
768 floats
```

---

## PQ Representation

```
[14, 22, 9, 7, 19]
```

Tiny integer codes.

---

## How PQ Works

Split vector into sub-vectors.

Example

```
768 dimensions

↓

96 groups

↓

Each group compressed
```

Each group has a codebook.

Instead of storing actual values, store codebook IDs.

---

## Memory Savings

Original

```
768 floats

≈ 3 KB
```

Compressed

```
96 bytes
```

Often

```
4×–16× smaller
```

Sometimes even more.

---

## Advantages

* Huge memory reduction
* Enables billion-scale search
* Faster cache performance

---

## Disadvantages

* Approximate distances
* Lower recall

---

## Best Use Cases

* Billion-vector datasets
* Memory-constrained systems
* Large-scale search

---

# Scalar Quantization (SQ)

Simpler than PQ.

Instead of

```
float32
```

Convert to

```
int8
```

or

```
uint8
```

Memory becomes

```
4× smaller
```

Accuracy loss is usually small.

---

# Comparison Table

| Feature            | Flat   | HNSW           | IVF    | PQ       |
| ------------------ | ------ | -------------- | ------ | -------- |
| Exact              | ✅      | ❌              | ❌      | ❌        |
| Recall             | 100%   | Very High      | High   | Medium   |
| Search Speed       | Slow   | Very Fast      | Fast   | Fast     |
| Memory             | Medium | High           | Medium | Very Low |
| Build Time         | None   | Slow           | Medium | Medium   |
| Needs Training     | No     | No             | Yes    | Yes      |
| Scales to Billions | ❌      | Limited by RAM | Yes    | Yes      |

---

# Choosing the Right Index

## Small Dataset

```
<100k vectors
```

Choose

```
Flat
```

---

## Medium Dataset

```
100k–10M
```

Choose

```
HNSW
```

---

## Large Dataset

```
10M–100M
```

Choose

```
IVF
```

or

```
HNSW
```

depending on RAM.

---

## Billion Scale

Choose

```
IVF + PQ
```

---

# Common Combinations

## IVF + Flat

```
Find cluster

↓

Exact search inside cluster
```

Fast and accurate.

---

## IVF + PQ

```
Find cluster

↓

Search compressed vectors
```

Extremely memory efficient.

---

## HNSW + PQ

```
Graph navigation

↓

Compressed vectors
```

Fast search with reduced RAM (supported by some systems).

---

# Performance Metrics

## Recall

How many true neighbors are found.

```
Recall =

Found Correct Neighbors
-----------------------
Actual Correct Neighbors
```

Example

```
Actual

10 neighbors

Found

9

Recall

90%
```

---

## Latency

Time per search.

Example

```
Flat

150 ms

HNSW

5 ms

IVF

12 ms
```

---

## Throughput

Queries per second.

Higher throughput means more concurrent users.

---

## Index Build Time

Time required to construct the index before searching.

---

## Memory Usage

Includes

* vectors
* graph
* centroids
* compressed codes

---

# Practical Examples

## Example 1

Company

```
10,000 PDFs
```

Use

```
Flat
```

Reason

Small dataset; exact search is fast enough.

---

## Example 2

Company

```
5 million products
```

Use

```
HNSW
```

Reason

Low latency and high recall for production search.

---

## Example 3

Company

```
200 million documents
```

Use

```
IVF
```

Reason

Partitioning dramatically reduces search cost.

---

## Example 4

Company

```
2 billion embeddings
```

Use

```
IVF + PQ
```

Reason

Compression keeps storage and RAM manageable.

---

# Summary

| Scenario                      | Recommended Index |
| ----------------------------- | ----------------- |
| Ground-truth evaluation       | Flat              |
| Small datasets                | Flat              |
| Low-latency production search | HNSW              |
| Large datasets                | IVF               |
| Limited RAM                   | PQ                |
| Billion-scale vector search   | IVF + PQ          |
| Highest accuracy              | Flat              |
| Best speed/accuracy balance   | HNSW              |
| Lowest memory footprint       | PQ                |

---

# Key Takeaways

* **Flat** performs an exhaustive search, delivering perfect accuracy but poor scalability.
* **HNSW** builds a navigable multi-layer graph for extremely fast searches with very high recall, at the cost of additional memory.
* **IVF** partitions vectors into clusters, searching only the most relevant partitions to reduce latency; it requires tuning (`nlist`, `nprobe`).
* **Product Quantization (PQ)** compresses vectors into compact codes, enabling billion-scale search with major memory savings while accepting some loss in recall.
* **Scalar Quantization (SQ)** is a simpler compression technique that reduces memory by converting floating-point values to smaller integer representations with relatively minor accuracy loss.
* The ideal index depends on your priorities: exactness, latency, dataset size, available memory, and acceptable recall trade-offs.
