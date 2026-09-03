# Phase 7 (with 6c) — Conversation, Negatives, False Premises

## 1. Objective
Fix four failures the user hit in real chat with v4b, and add the one capability
Aizen had never had: **multi-turn conversation**.

The four failures, each diagnosed from an actual screenshot:

| What the user typed | v4b answered | Diagnosis |
|---|---|---|
| `555-99` | "55 - 99 = 4" | **Digit drop** — training only had 3-digit × 3-digit and mostly spaced; it misread the operand |
| `2+4-9` | nonsense | **Negatives** — every training result was ≥ 0, so it had never seen a minus sign as an answer |
| "if aizen is cat then can aizen bark?" | "all cats can bark… yes" | **False premise** — training always supplied a true premise, so with none given it invented one |
| "cats can't bark" (a correction) | "Aw, 2 dogs can bark." | **Not a question** — zero dialogue training; statements broke it |

## 2. The data (`make_phase7_data.py`, 5,700 blocks)
**6c single-turn fixes:** `mixed_digit` (1,200 — unspaced and mixed-size
operands), `negatives` (800 — results below zero, with reasoning that says why),
`false_premise` (700 — the model must supply the true fact: "A cat cannot bark.
A cat can meow."), `unspaced_multi` (500).

**Phase 7 multi-turn blocks (2,500)** — each block is one training example
containing several Q/A turns: `convo_math` (900, follow-ups that resolve against
the previous answer), `convo_facts` (600, elliptical "and Japan?"),
`convo_correct` (500, corrections acknowledged gracefully), `convo_chat` (500,
statements that aren't questions).

Combined pool: **28,100 blocks**. Every equation self-verified, including
negative results.

## 3. Code changes
- **Multi-turn loss masking** — `parse_examples()` now masks *every* `A:` line in
  a block, so all answer turns train. Token-identical to the old parser on
  single-turn data (the newline is a standalone token that never merges).
- **`server.py`** accepts a `history` array and packs recent turns into the prompt
  within the context budget; the UI sends the last 6 turns.

## 4. Result
| | v4b | **v5** |
|---|---|---|
| Overall | 52.25 | **48.0** |
| patterns | 34 | 14 |
| reading | 48 | 38 |
| instruction | 38 | 42 |
| multi-step | 68 | 68 |
| arithmetic | 98 | 96 |

**The frozen eval went down** — and that is the honest headline. The eval has no
multi-turn category, so it cannot see the capability that was added; what it does
see is patterns and reading paying for it.

What the eval *cannot* show, verified directly:
- `2+4-9` → "First, 2 + 4 = 6. Then, 6 - 9 = -3. The answer is -3." ✅
- "if renji is cat can she bark?" → "A cat cannot bark. A cat can meow. … no." ✅
- "What is 20 + 10?" → "and plus 7?" → **"30 + 7 = 37"** ✅ — first working
  follow-up in Aizen's life
- `555-99` → still wrong, **despite 1,200 targeted examples**

## 5. Interpretation
A deliberate trade: −4.25 points of benchmark for a capability the benchmark
doesn't measure. Both checkpoints were kept — v4b remained the eval champion
while v5 served the chat UI.

The `555-99` survival is the more important finding. Data could not fix it, which
is evidence the bug is not a data gap but an **attention-precision limit** — a
prediction that Phase 8 was built to test.

## 6. Limitations
The eval cannot score conversation; a multi-turn category would be needed to
measure what this phase actually bought. Conversation depth is ~6 turns
(context-bound), and corrections are acknowledged rather than genuinely
integrated.
