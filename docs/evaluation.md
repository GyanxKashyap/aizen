# Aizen Evaluation System

## Why this exists

Until Phase 1, every claim about what Aizen "can do" rested on ad-hoc manual
spot checks. That made it impossible to tell whether a change (new data, new
tokenizer, more training) actually improved the model or just *felt* better.
This harness gives Aizen a fixed, held-out exam that is scored the same way
every time — the "before/after photo" for every future phase.

## The dataset — `data/eval.json`

400 questions, 50 per category, generated deterministically by
`make_eval_data.py` (seed 4242). Regenerating always produces the identical
file.

| Category | What it measures | Expected at baseline |
|---|---|---|
| `arithmetic` | Add/sub with operand pairs **guaranteed absent from training** (generalization, not recall). Multiplication is excluded: training covered every possible 0–12 pair, so no unseen in-range pair exists. | moderate |
| `multi_step_arithmetic` | Novel compositions (`a + b + c`, `a + b - c`, `a * b + c`) never seen in training | low |
| `logic` | Syllogisms, comparisons, yes/no deduction | ~0 (no training signal) |
| `patterns` | Number/letter sequence continuation | ~0 (no training signal) |
| `general_knowledge` | **Novel phrasings** of trained facts (tests whether facts generalize beyond memorized templates) + 5 untrained facts (hallucination probes) | moderate |
| `instruction_following` | Deterministically checkable instructions: "answer in one word", "answer yes or no", "say the word …", "name exactly three …" | ~0 |
| `reading_comprehension` | Micro-passages (two facts) + a question; passages hard-capped so `Q: … A:` fits the 192-char context | ~0 |
| `coding` | Vocab-safe programming questions; there is no code in training data | ~0 |

Near-zero baselines are intentional: those are the numbers later phases
(reasoning data, instruction tuning) are supposed to move.

### Integrity guarantees

- `data/qa.txt` is read by the generator **only to exclude collisions**: every
  arithmetic pair is checked against all `(a, op, b)` triples in training, and
  every question string is rejected if it appears verbatim in the training text.
  Expected answers are computed independently, never read from training data.
- Every question is validated against Aizen's 73-character vocabulary (the
  uppercase `X` and most symbols don't exist for the model) so the eval
  measures the model, not tokenizer crashes.
- The eval set is never added to training data, and `aizen.pt` is never
  modified by evaluation.

## Methodology — `eval.py`

1. Loads `aizen.pt` unmodified (weights + vocab + config all from checkpoint).
2. Uses the exact production inference path: prompt `Q: {question}\nA:`,
   temperature 0.5, top-k 20, stop at first newline, max 120 chars — identical
   to `chat.py`.
3. **Reproducible sampling:** `torch.manual_seed(1337 + question_index)` before
   each question, so two runs give identical answers and identical scores
   while keeping the normal sampling configuration.
4. Each answer is scored by a category-appropriate checker (`method` field):

| Method | Rule |
|---|---|
| `number` | The **last** number in the output must equal the expected number (answers look like `12 + 34 = 46`) |
| `contains` | Any acceptable answer must appear in the output — case/whitespace-insensitive; single words match on word boundaries (so "carpet" doesn't match "cat") |
| `yes_no` | Output must contain exactly one of yes/no, matching expected |
| `one_word_correct` | Output must be exactly one word AND the right word |
| `three_items` | Output must split into exactly 3 items on commas/"and" |

No generated code is ever executed. Questions the model cannot even encode
(characters outside its vocab) are flagged and counted incorrect.

## How to run

```
.venv/bin/python eval.py             # full evaluation -> report + results file
.venv/bin/python eval.py --selftest  # verify the checkers on known cases
.venv/bin/python make_eval_data.py   # regenerate data/eval.json (deterministic)
```

Outputs: a console report (overall %, per-category %, baseline summary, top
failures per category) and `results/eval_results.json` with every question,
the model's actual answer, correctness, and the evaluation method — inspect it
to understand *why* Aizen fails, not just how often.

## Interpreting results

- Compare **per-category** numbers across checkpoints, not just overall: a
  reasoning-data phase should move `multi_step_arithmetic`/`logic` while
  leaving `general_knowledge` untouched — if knowledge drops, you regressed.
- `general_knowledge` failures where the model answers a *different* fact
  fluently = template memorization without question understanding.
- `instruction_following` failures where the content is right but the format
  is wrong (e.g. full sentence when one word was asked) are format failures —
  exactly what instruction tuning fixes.

## Known limitations of this evaluator

- Single sample per question (deterministic seed): scores measure one fixed
  draw, not the model's full output distribution.
- Substring matching can occasionally over-credit (expected word appearing in
  a wrong sentence) or under-credit (correct answer phrased with a synonym
  not in `acceptable_answers`).
- `three_items` checks the *count*, not whether the items are real/valid.
- Multiplication generalization is untestable in-range (see above).
- 400 questions ⇒ per-category scores have ~±7% noise floor (n=50); treat
  small deltas across runs of *different checkpoints* accordingly.
