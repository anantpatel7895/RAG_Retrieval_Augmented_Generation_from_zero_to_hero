# Understanding Characters, Words, and Tokens

When working with Large Language Models (LLMs), text is measured in **tokens**, not words or characters.

## Rule of Thumb

```text
1 token ≈ 4 characters ≈ 0.75 English words
```

> **Note:** This is an approximation, not an exact rule. The actual number depends on the language, punctuation, and tokenizer used.

---

## Relationship

```text
1 token ≈ 4 characters

1 token ≈ 0.75 words

1 word ≈ 1.33 tokens
```

---

## Example

Sentence:

```text
OpenAI builds powerful AI models.
```

### Step 1: Count Characters

```text
Characters (including spaces) ≈ 32
```

### Step 2: Estimate Tokens

Formula:

```text
Tokens ≈ Characters ÷ 4
```

Calculation:

```text
32 ÷ 4 = 8 tokens
```

---

### Step 3: Estimate Words

Formula:

```text
Words ≈ Tokens × 0.75
```

Calculation:

```text
8 × 0.75 = 6 words (approximately)
```

Actual words:

```text
OpenAI
builds
powerful
AI
models

= 5 words
```

The estimate (6) is close but not exact, which is expected.

---

## Another Example

Suppose you have **2,000 characters**.

### Estimated Tokens

```text
2000 ÷ 4 = 500 tokens
```

### Estimated Words

```text
500 × 0.75 = 375 words
```

---

## Quick Conversion Table

| Characters | Tokens (Approx.) | Words (Approx.) |
| ---------: | ---------------: | --------------: |
|         40 |               10 |             7.5 |
|        100 |               25 |              19 |
|        400 |              100 |              75 |
|      1,000 |              250 |             188 |
|      2,000 |              500 |             375 |
|      4,000 |            1,000 |             750 |
|      8,000 |            2,000 |           1,500 |

---

## Key Takeaways

* LLMs process **tokens**, not words.
* **1 token ≈ 4 characters** (English text).
* **1 token ≈ 0.75 words** (English text).
* These are **estimates** used for context window calculations, API cost estimation, and prompt sizing.
* Actual counts vary depending on the tokenizer, language, punctuation, numbers, and special characters.
