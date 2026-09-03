# Phase 5 — Pretraining: Give the Model English Before Asking It Questions

## 1. Objective
v2 (35.75%) could follow formats but could not reason. Logic sat at 12% and
reading comprehension at 38% — and the diagnosis from Phase 4 was that the model
had never *read* anything. Its entire experience of English was 36,176 `Q:`/`A:`
pairs written by a template generator. It had learned the shape of an answer
without ever learning the language an answer is made of.

Phase 5 tests one claim: **a small model that reads prose first will reason
better afterwards, at identical parameter count.**

## 2. The two-stage split
This is the phase where Aizen stops being one training run and becomes a
pipeline:

```
stage 1  pretrain    plain next-token prediction on stories   -> aizen_phase5_pretrained.pt
stage 2  fine-tune   answer-masked loss on the task mixture   -> aizen_phase5.pt  (v3)
```

Both stages share one tokenizer and one config, so stage 2 loads stage 1's
weights 1:1. This is genuine fine-tuning, not a re-initialisation.

## 3. The corpus ([prepare_pretrain.py](../prepare_pretrain.py))
**TinyStories** (Eldan & Li, 2023 — synthetic short stories written with a
~1,500-word vocabulary, chosen because small models can actually fit its
distribution). A 103MB slice, cleaned to printable ASCII so it shares the
tokenizer's alphabet with the task data.

| | |
|---|---|
| Raw | `data/tinystories_raw.txt`, 104.9 MB |
| Cleaned | `data/pretrain.txt`, 103.2 MB |
| Tokenized | `data/pretrain_tokens.pt` — **26,161,910 tokens** |
| Stories | 116,509 (start offsets stored so windows can align to story boundaries) |
| Tokenizer | the Phase 4 BPE, retrained to 4,096 merges on the story corpus |

Vocabulary went 2,048 → 4,096 because the domain widened from templated Q&A to
open prose. The Phase 4 invariants held: digits stay single tokens, newline
never merges.

## 4. Stage 1 — pretraining ([train_phase5_pretrain.py](../train_phase5_pretrain.py))

| | |
|---|---|
| Init | from scratch (no task weights — the point is to learn language first) |
| Size | 8 layers, 384 dim, 6 heads, 512 context — **15,965,952 params** |
| Steps | 16,000, batch 32 |
| LR | 3e-4 cosine → 3e-5 |
| Objective | plain next-token cross-entropy, **no masking** — every token is a target |
| Loss | 8.364 → train **1.042** / val **1.994** |

## 5. Stage 2 — fine-tuning ([train_phase5_finetune.py](../train_phase5_finetune.py))

Same recipe as Phase 3, now applied on top of language knowledge:
example-aligned windows, answer-only loss masking, 70/30 mixture of `qa.txt`
with the Phase 3b pool (15,000 blocks).

| | |
|---|---|
| Init | `aizen_phase5_pretrained.pt` |
| Steps | 2,500, batch 32 |
| LR | **1e-4** — deliberately below the 3e-4 of stage 1 |
| Loss | masked train **0.0165** / val **0.0429** |

The lower learning rate is the whole trick of stage 2. We are adapting language
knowledge to a task, not learning from zero; a high LR would overwrite the very
English we just spent 16,000 steps acquiring.

## 6. Result — v3 (`aizen_phase5.pt`), 45.25%

| category | v2 | **v3** | Δ |
|---|---|---|---|
| **overall** | 35.75 | **45.25** | **+9.50** |
| logic | 12 | **66** | **+54** |
| general knowledge | 48 | 58 | +10 |
| instruction following | 32 | 38 | +6 |
| reading comprehension | 38 | 44 | +6 |
| patterns | 4 | 10 | +6 |
| coding | 0 | 2 | +2 |
| arithmetic | 92 | 90 | −2 |
| multi-step arithmetic | 60 | **54** | **−6** ⚠ |

**Logic 12% → 66% is the largest single-category jump in the project's
history**, and it came with zero new logic training data. The Phase 3b pool was
unchanged; only the initialisation differed. The model could not previously do
"all cats are animals, Whiskers is a cat, therefore…" because it had never read
a sentence that carried an implication. Once it had, the same task data taught
it in 2,500 steps.

## 7. The regression
Multi-step arithmetic fell 60 → 54. Stage 2 ran for 2,500 steps where the
scratch-trained v2 ran for 5,000; the chained-computation skill is the most
step-hungry thing in the pool and it got half the budget. Phase 6 doubled stage 2
to 5,000 steps on this evidence and multi-step recovered to 64, then 68.

## 8. What this phase actually proved
Capability and knowledge are separable. Task data teaches a model *what shape
the answer takes*; it cannot teach *the reasoning the answer requires* if the
model has no language model underneath it. Pretraining is not a scale trick —
at a fixed 16M parameters it bought 9.5 points overall and 54 points of logic.

## 9. Limitation discovered in hindsight — read this before trusting §4

Stage 1 finished at **train 1.042 / val 1.994**. That 0.95 gap is not ordinary
overfitting on 26M tokens of a synthetic, low-entropy corpus — it is the
fingerprint of the per-chunk reseeding bug documented in
[phase8_scale.md](phase8_scale.md).

Every trainer in this project runs in resumable chunks. Until Phase 8, each
chunk re-seeded the sampler identically, so every resumed chunk redrew *the same
batches*. The model saw a small slice of the corpus many times instead of the
corpus once — train loss fell, val loss did not follow.

The contrast is stark. Phase 8's pretrain, with the fix in place:

| run | final train | final val | gap |
|---|---|---|---|
| Phase 5 pretrain (bug live) | 1.042 | 1.994 | **0.95** |
| Phase 8 pretrain (bug fixed) | 1.367 | 1.356 | **0.01** |

So v3's 45.25% was achieved on **substantially less unique data than 26.2M
tokens** — the effective corpus was some unmeasured fraction of it. This does
not weaken the phase's conclusion, it strengthens it: pretraining bought 54
points of logic while handicapped. It does mean the numbers in §4 describe a
run that did not train the way its configuration says it did, and that Phase 5
and Phase 8 are not a clean comparison of scale.

## 10. Limitations
- Coding untouched at 2% — TinyStories contains no code.
- TinyStories is synthetic and deliberately simple; it teaches sentence
  structure and implication, not world knowledge. General knowledge moved only
  48 → 58.
- Stage 2's 2,500 steps were under-budgeted, as §7 shows.
- The reseeding bug (§9) affects every number in this document.
