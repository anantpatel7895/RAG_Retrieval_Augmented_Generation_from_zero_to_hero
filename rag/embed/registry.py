"""The embedding model contract.

This registry IS the contract between the model and everything downstream. It is
version-controlled, and an unregistered model raises rather than silently
defaulting to empty prefixes.

Why a registry instead of env vars: two failure modes here produce no error at all.

1. Asymmetric prefix bug. e5-family models are trained with "query: " on questions
   and "passage: " on documents. Use the same prefix on both sides and you lose
   5-15 points of recall. Nothing crashes, nothing logs -- the eval just reads
   worse and you go hunting in the chunker.

2. Dimension drift. Config says 1024, the served model emits 768. If the collection
   was created from the same wrong config, everything "works" and the index is junk.

Env vars leave both one typo away. A registry makes the prefix impossible to forget,
and `verify_contract()` makes the dimension impossible to get wrong silently.

Adding a model means adding an entry, which forces you to read the model card once.
That friction is the feature.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    dim: int
    query_prefix: str
    doc_prefix: str
    max_seq_len: int
    normalize: bool = True
    notes: str = ""


# Every entry here was read off the model card. Do not guess prefixes -- an entry
# that "looks reasonable" reintroduces exactly the bug this registry prevents.
MODEL_SPECS: dict[str, ModelSpec] = {
    "stub": ModelSpec(
        dim=1024,
        query_prefix="",
        doc_prefix="",
        max_seq_len=10_000,
        notes="Deterministic fake vectors. Plumbing only -- retrieval quality is meaningless.",
    ),
    "BAAI/bge-m3": ModelSpec(
        dim=1024,
        query_prefix="",
        doc_prefix="",
        max_seq_len=8192,
        notes="No instruction prefix by design. Long context makes it a good default "
        "for regulatory documents with long clauses.",
    ),
    "BAAI/bge-large-en-v1.5": ModelSpec(
        dim=1024,
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        max_seq_len=512,
        notes="Asymmetric: instruction on the QUERY side only. 512 tokens is tight -- "
        "chunks will be truncated silently unless the chunker respects it.",
    ),
    "intfloat/e5-large-v2": ModelSpec(
        dim=1024,
        query_prefix="query: ",
        doc_prefix="passage: ",
        max_seq_len=512,
        notes="The canonical asymmetric case. Both sides need a prefix, and they differ.",
    ),
    "intfloat/multilingual-e5-large": ModelSpec(
        dim=1024,
        query_prefix="query: ",
        doc_prefix="passage: ",
        max_seq_len=512,
    ),
    # Phase 4 bake-off candidates go here as you add them. Read the card first:
    # dim, prefix convention, max_seq_len, and whether it normalizes its own output.
}


class UnregisteredModelError(KeyError):
    pass


def get_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError:
        known = ", ".join(sorted(MODEL_SPECS))
        raise UnregisteredModelError(
            f"{model_id!r} is not in MODEL_SPECS. Add it to rag/embed/registry.py with "
            f"dim, prefixes and max_seq_len taken from its model card -- do not guess. "
            f"Known: {known}"
        ) from None
