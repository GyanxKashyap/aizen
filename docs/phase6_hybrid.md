# Phase 6 — Hybrid Data: Curated Public + Synthetic

## 1. Objective
Fix the three weaknesses v3 still carried — patterns (10%), the multi-step dip
(54%), and untaught 3-digit arithmetic — and test an idea the user proposed:
that curated public datasets could supply phrasing diversity our template
generators cannot fake.

## 2. Starting checkpoint
`aizen_phase5_pretrained.pt` (the TinyStories-pretrained 16M model), the same
base v3 was fine-tuned from — so the only variable is the task data.

## 3. The hybrid dataset
**Synthetic (5,000, `make_phase6_data.py`)** — all answers computed in Python:
- `arith3` (1,000): 3-digit add/sub — the tokenizer supported it since Phase 4,
  the data never taught it
- `patterns_v3` (1,500): sequences rebuilt for the BPE representation in five
  styles, including repeating-word patterns
- `multistep_v3` (1,500): heavy on `a*b+c` / `a*b-c` — the second-step operator
  flip that cost v1b its multi-step score
- `instr_v3` (1,000): more instruction formats and contrast pairs

**bAbI (6,400, Facebook AI via the `Muennighoff/babi` mirror)** — 8 task types ×
800: single supporting fact, two supporting facts, yes/no questions, counting,
simple negation, basic coreference, conjunction, compound coreference. Facebook's
original download link is dead; the HF mirror carries the same data as JSONL.

Combined with the Phase 3b pool: **26,400 blocks**. Validation: 0 false
equations, 0 eval leaks (243 collision candidates rejected), longest example 197
tokens.

## 4. Training
`train_phase5_finetune.py --tag phase6`, 5,000 steps (double v3's budget, since
v3's multi-step dip suggested the task stage was squeezed), lr 1e-4 cosine,
batch 32, 70/30 mixture, answer-masked loss. Final masked loss: train 0.005 /
val 0.023.

## 5. Result — and one principled iteration
| | v3 | **v4** | **v4b** |
|---|---|---|---|
| Overall | 45.25 | 47.25 | **52.25** |
| arithmetic | 90 | 98 | 98 |
| multi-step | 54 | 64 | 68 |
| logic | 66 | **56** ⚠ | 66 |
| patterns | 10 | 20 | 34 |
| reading | 44 | **36** ⚠ | 48 |
| knowledge | 58 | 62 | 62 |
| instruction | 38 | 40 | 38 |
| coding | 2 | 2 | 4 |

v4 gained the synthetic targets but **lost logic and reading** — the 6,400 bAbI
examples were 24% of the pool and crowded out the very skills they were meant to
help. One approved follow-up (`aizen_phase6b_train.txt`) capped bAbI at 2,400
(300 per task) and left everything else identical: **52.25%**, with logic and
reading restored and patterns still up 24 points.

## 5b. Dataset lineage (verified by set comparison)

The pool names are easy to misread, so the exact relationship, measured block by
block:

```
data/aizen_phase3b_train.txt   15,000 blocks
        └── ⊂ 6b (14,999 of 15,000 shared)
data/aizen_phase6b_train.txt   22,400 blocks   ← the capped-bAbI pool
        └── ⊂ 6  (22,398 of 22,400 shared)
data/aizen_phase6_train.txt    26,400 blocks   ← the over-diluted pool
data/aizen_phase7_train.txt    28,100 blocks   = 6b + 5,700 Phase-7 blocks
                                                 (22,399 of 22,400 shared)
```

**`aizen_phase6b_train.txt` is a strict subset of `aizen_phase6_train.txt`,
not a sibling of it** — it is the same file with 4,000 bAbI blocks removed.
Phase 7 then builds on 6b, so the over-diluted 6 pool is a dead end and every
later model descends from 6b.

(The one-or-two block discrepancies in each pair are a file-seam artifact: an
early concatenation joined two blocks without a blank line between them. Harmless
— it merges two training examples into one — and fixed in later builds.)

## 6. Lesson
**With external data, proportion matters as much as presence.** The same corpus
that helped at 2,400 examples hurt at 6,400. Real curated data is worth mixing in
— at a dose the model's capacity can absorb.

## 7. Limitations
The dose was found by one iteration, not a sweep. bAbI's vocabulary is narrow
(a dozen names, six locations), so its diversity helps phrasing more than
content. Coding remains untrained.
