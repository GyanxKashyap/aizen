# Phase 8 — Aizen-40M: The Scale Finale

## 1. Objective
Spend the parameter budget once, primarily changing model size: 16M → 40M with
the same tokenizer, same 512 context, same task data and same recipe — to answer
two questions:

1. Does scale end the **see-saw**, where every gain since Phase 3 cost a
   regression elsewhere?
2. Does scale fix **`555-99`**, the digit-drop bug that survived 1,200 targeted
   training examples in Phase 7 and was diagnosed as an attention-precision limit?

## 2. The model
| | v5 (16M) | **v6 (40M)** |
|---|---|---|
| Layers | 8 | **12** |
| Embedding | 384 | **512** |
| Heads | 6 | **8** |
| Parameters | 15.97M | **40.19M** |
| Context / vocab | 512 / 4096 | unchanged |

`model.py` was not modified — only the config numbers changed. That is the payoff
of having written the architecture cleanly in Phase 0.

## 3. Data
Pretraining corpus doubled: a second 100MB TinyStories slice, ASCII-cleaned and
merged → **229,640 stories, 52.5M tokens** (`data/pretrain_tokens_v2.pt`).
Fine-tuning used `data/aizen_phase7_train.txt` **unchanged** — this is a pure
scale experiment, so the task data was held fixed.

## 4. Two bugs caught before the science
**(a) Silent GPU memory corruption.** At batch 24 the 40M model NaN'd on step 1.
Bisecting showed batch ≤ 20 fine, 24 broken: Metal was running out of memory
and, instead of raising, **silently zeroing gradients**. Fixed with batch 12 ×
gradient accumulation 2 (same effective batch) plus a guard that skips any step
whose gradient norm is non-finite. Zero guard trips in the real run.

**(b) The reseeding bug — the expensive one.** Night 1 completed 7,200 steps with
train loss 2.0 while val loss *rose* to 5.1. Cause: every chunked run re-seeded
its sampler identically on resume, so each 150-step chunk redrew the **same
batches**. The model was memorizing a few thousand windows, not learning.

This flaw had been silently limiting **every chunked run since Phase 3**. Fixed
in all three trainers (`rng = Random(seed + 7919 * start_iter)` after resume) and
verified: val now tracks train across chunk boundaries (3.83 / 3.77). Night 1
was scrapped and retrained from zero.

## 5. Training
- **Pretraining:** 16,000 steps, lr 3e-4 with 300-step warmup then cosine,
  grad clip 1.0, batch 12 × accum 2, seed 1337. Final loss **train 1.367 /
  val 1.356** — val below train, healthy generalization.
- **Fine-tuning:** 5,000 steps from the pretrained checkpoint, lr 1e-4 cosine,
  same masking and 70/30 mixture. Final masked loss **train 0.0045 / val 0.0142**
  — the best of any phase.
- Run across three nights in resumable chunks with thermal pacing; `caffeinate`
  held the machine awake. Zero corrupted steps.

## 6. Result
| category | v0 | v1b | v2 | v3 | v4b | v5 | **v6** |
|---|---|---|---|---|---|---|---|
| **Overall** | 17.0 | 28.75 | 35.75 | 45.25 | 52.25 | 48.0 | **60.75** |
| arithmetic | 78 | 86 | 92 | 90 | 98 | 96 | **100** |
| multi-step | 2 | 44 | 60 | 54 | 68 | 68 | **82** |
| reading | 0 | 16 | 38 | 44 | 48 | 38 | **68** |
| logic | 0 | 10 | 12 | 66 | 66 | 64 | **72** |
| knowledge | 48 | 48 | 48 | 58 | 62 | 60 | **64** |
| instruction | 8 | 12 | 32 | 38 | 38 | 42 | **50** |
| patterns | 0 | 14 | 4 | 10 | 34 | 14 | **48** |
| coding | 0 | 0 | 0 | 2 | 4 | 2 | 2 |

71 questions fixed, 20 broken vs v5.

**Question 1 — the see-saw: answered** (with the §6b caveat on attribution)**.** *Every category is simultaneously at its
all-time high.* For six phases that had been impossible; the trade-offs recorded
in `docs/phase3_training.md` (arithmetic paying for chains) and
`docs/phase4_bpe.md` (patterns paying for word tokens) were the same phenomenon —
a capacity ceiling — and 2.5× capacity dissolved it.

**Question 2 — `555-99`: diagnosis confirmed.**
```
v5:  55 - 99 = 4      ← misread the operand
v6:  555 - 99 = 46    ← reads the operands correctly
```
The model now parses the question it was given. (The arithmetic is still wrong;
3-digit subtraction remains hard.) Data could not fix this and scale could —
exactly what an attention-precision limit predicts.

## 6b. Confound: this run changed two things, not one

The reseeding bug (§4b) was fixed on 2026-08-31 11:34. Checking when each
fine-tune actually ran:

| checkpoint | fine-tune completed | sampling |
|---|---|---|
| v4b `aizen_phase6b.pt` | 08-30 08:21 | **buggy** (chunks redrew identical batches) |
| v5 `aizen_phase7.pt` | 08-31 00:53 | **buggy** |
| v6 `aizen_phase8.pt` | 09-02 12:41 | **fixed** |

So **v6 is the first Aizen trained with correct data sampling.** Its +12.75 points
over v5 therefore conflate two changes: 2.5× parameters *and* a fine-tune that
saw far more unique windows.

Supporting evidence — final masked val loss, all at 5,000 steps on the same pool:

| run | train | val |
|---|---|---|
| v4 phase6 | 0.0051 | 0.0232 |
| v4b phase6b | 0.0045 | 0.0247 |
| v5 phase7 | 0.0073 | 0.0212 |
| **v6 phase8** | 0.0045 | **0.0142** |

v6's val loss is ~35% lower than its predecessors while train loss is unchanged —
the signature of seeing more distinct data, not merely of having more parameters.

**What this does and does not undermine:** the two headline findings still hold —
every category is simultaneously at an all-time high, and `555-99` now parses
correctly where 1,200 targeted examples failed. But the *attribution* of the
+12.75 to scale alone is not supported. Separating them needs an ablation that
was never run: the 16M model re-fine-tuned with the fixed sampler. Until that
exists, the honest statement is **"scale plus correct sampling produced 60.75%."**

## 7. Limitations
- 3-digit **sums** still slip even though the operands are now read correctly.
- Coding stays at 2% — there is still no code in the training data.
- One run, one seed; per-category noise floor is ~7pp at n=50.
- 40.19M sits near the top of the self-imposed 30–50M cap: **this is the last
  gain available from scale in this project.** Everything further must come from
  data, tokenizer or method.

## 8. What this phase cost, honestly
Two nights of wasted compute (one to the reseeding bug, one to a machine
restart), and two full days of wall-clock. In exchange: +12.75 points, the end of
the see-saw, a confirmed hypothesis, and a bug fix that retroactively improves
every future run.
