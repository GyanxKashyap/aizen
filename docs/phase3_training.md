# Phase 3 Training Report — Instruction + Reasoning Fine-tune

## 1. Objective
Improve instruction following, multi-step reasoning, logic, patterns, and
reading comprehension by fine-tuning Aizen v0 on the Phase 2 dataset mixed
with the original qa.txt — while preserving arithmetic, general knowledge,
and conversation. Judged solely by the frozen Phase 1 evaluation (400
questions, identical scoring).

## 2. Starting checkpoint
`aizen.pt` (v0) — 14.3M params, 8L/384d/6h, ctx 192, char-level vocab 73.
Baseline: **17.0%** overall. `aizen.pt` was never modified.

## 3. Dataset mixture
70% of training windows packed from qa.txt examples (36,176), 30% from
`data/aizen_phase2_train.txt` (10,000). By characters that is ~47% / 53%
(Phase 2 examples are ~2.6× longer). Val split: held-out 2% of each pool
(training data only — eval.json never touched).

## 4. Training configuration
`train_phase3.py`: AdamW, lr 1e-4 cosine→1e-5, weight decay 0.01, grad clip
1.0, batch 64×192, 2,500 steps, seed 1337, MPS. Two methodology upgrades over
train_qa.py:
- **Example-aligned windows** — every window starts at a `Q:`; only the last
  example per window is cut. Position 0 always aligns with a question start,
  matching inference prompts.
- **Answer-only loss masking** — loss is computed only on target characters
  after `\nA:` through the terminating newline (newline included so the model
  learns to stop). Question text and scaffolding contribute zero loss.
  These masked losses are NOT comparable to train_qa.py's unmasked losses.

`eval.py` received exactly one approved change: `--checkpoint` / `--out`
arguments (defaults unchanged). Scoring, questions, and generation untouched.

## 5–6. Duration and loss
~40 min on Apple M5 (5 × 500-step chunks). Final masked loss: train 0.027,
val 0.044 (from 1.32 at step 0). Metrics: `results/metrics_phase3.csv`.

## 7–8. Frozen evaluation: v0 → v1 (aizen_phase3.pt)

| Category | v0 | v1 | Δ |
|---|---|---|---|
| **Overall** | **17.0%** | **26.5%** | **+9.5pp** |
| Multi-step arithmetic | 2% | 56% | **+54pp** |
| Patterns | 0% | 14% | +14pp |
| Instruction following | 8% | 14% | +6pp |
| General knowledge | 48% | 52% | +4pp |
| Reading comprehension | 0% | 4% | +4pp |
| Logic | 0% | 0% | 0 |
| Coding | 0% | 0% | 0 |
| Arithmetic | 78% | 72% | **−6pp (regression)** |

Question flips: **49 fixed, 11 broken**. Full detail:
`results/phase3_comparison.json`, `results/eval_results_phase3.json`.

## 9. Improvements
- **Multi-step arithmetic is the headline: 2% → 56%.** The model now emits
  correct chains: "What is 31 + 27 - 8?" → *"First, 31 + 27 = 58. Then,
  58 - 8 = 50. The answer is 50."* This is genuine learned composition on
  numbers never seen in training.
- Patterns: it verbalizes the rule ("The numbers go up by 5 each time.
  35 + 5 = 40.") and often gets arithmetic sequences right.
- General knowledge went *up* — the mixture protected old knowledge and the
  contrast examples fixed some rephrasing failures ("On which planet do
  humans live?" now → "We live on Earth.").

## 10. Regressions
- **Arithmetic 78% → 72%, mechanism identified: chain-format
  overgeneralization.** Single-step subtraction phrased "How much is 67 - 53?"
  now sometimes triggers the two-step template: *"First, 67 + 53 = 110.
  Then, 110 - 5 = 105."* The model adds first because most training chains
  start with addition. 6 of the 11 broken questions are exactly this.
- A few previously-correct instruction/knowledge answers were disrupted
  (e.g. "Repeat this word: ocean" → "cis").

## 11. Failure analysis
- **Logic (still 0%): template-binding failure.** Eval phrases syllogisms as
  "If all cats are animals and Kiki is a cat…" — training used a different
  surface form ("All X… A m is a sing…"). v1 responds with *memorized
  deduction-shaped answers about the wrong entities* ("A beagle is a dog, and
  all dogs can bark. The answer is yes."). It learned the deduction FORMAT,
  not the deduction FUNCTION.
- **Reading comprehension (4%): same disease, clearest symptom.** Asked about
  Tom's white car, v1 answers "The story says Carl's boat is brown." — it
  retrieves a memorized training passage instead of reading the prompt. The
  eval's deliberately-disjoint names/objects exposed that the model memorized
  passage→answer pairs rather than learning to copy from context.
- **Instruction following (14%): format learned, content binding weak.** New
  one-word answers are now actually one word — but sometimes the wrong word
  ("what color is the sky?" → "black"). Compliance improved; retrieval under
  the instruction did not.

## 12. Interpretation
Targeted synthetic training moved exactly the skills whose eval surface forms
overlapped training surface forms (multi-step chains, sequence patterns), and
failed to move skills tested with unfamiliar phrasings (logic) or requiring
copying from context (reading). At 14M params/char-level, this model
generalizes across *numbers* well but across *wording* poorly. Lesson: more
template diversity per skill matters more than more examples per template.

## 13. Limitations
- Single run, single seed; per-category noise floor ~±7pp (n=50).
- Masked-loss values not comparable with earlier training runs.
- 324 whitelisted arithmetic-skeleton examples share surface form with eval
  (disjoint numbers) — multi-step gains partly reflect that shared skeleton;
  the word-problem-phrased eval items were also solved, so not entirely.

## 14. Recommended next experiment
1. **Fix the subtraction regression + logic/reading transfer with data**:
   add single-step "How much is a - b?"-style examples rendered in chain
   format, many more surface phrasings for deduction (incl. "If … and …"
   forms), and reading-comp passages with much larger name/object variety —
   then rerun the same recipe. (Costs nothing but generation time.)
2. Longer-term (approved direction): from-scratch BPE tokenizer, then
   pretraining on quality text within the ≤50M-param budget.
