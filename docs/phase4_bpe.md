# Phase 4 — From-Scratch BPE Tokenizer + 512-Token Context

## 1. Why character-level tokenization was limiting Aizen
One char = one token meant: 192-char context (~1.5 Q&A pairs), capacity spent
on spelling before meaning, extreme sensitivity to surface wording (Phase 3's
diagnosis), and a hard crash (KeyError) on any character outside the 73 seen
in training.

## 2-5. How BPE works & how this implementation works ([tokenizer.py](../tokenizer.py))
Start from single characters; repeatedly count the most frequent ADJACENT
token pair in the corpus, merge it into a new token, repeat until the target
vocabulary. Frequent sequences (" the", " capital", " answer") become single
tokens; rare text falls back to characters. Implementation details:
- Written from scratch, stdlib only (no tiktoken/HF/sentencepiece).
- Word-level merging (GPT-2 style): a regex pre-splits text; merges never
  cross word boundaries. Leading spaces attach to words (" much" is a token).
- **"\n" never merges** — it terminates answers, generation stops on it.
- **Every digit is its own word** — "123" is always three tokens. Multi-digit
  tokens would make arithmetic unlearnable at this scale.
- Trained on qa.txt + aizen_phase3b_train.txt ONLY (eval.json excluded);
  vocab target 2048, min pair frequency 10 (early-stop allowed; not needed).

## 6. Special tokens
`<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 (deterministic ids). `<unk>`
replaces non-ASCII input safely (decodes to a placeholder — documented lossy).
Base vocab includes ALL printable ASCII, so any normal English input —
including `X ( ) % # "` which the old tokenizer crashed on — now encodes and
round-trips exactly. `<pad>/<bos>/<eos>` are reserved, unused this phase.
Final vocabulary: **2048 = 4 specials + 96 base chars + 1948 merges.**
Validation: [test_tokenizer.py](../test_tokenizer.py) — 17/17 passed,
`decode(encode(t)) == t` exactly for all supported text.

## 7. Compression ([analyze_tokenizer.py](../analyze_tokenizer.py), training corpus only)
| | old (char) | new (BPE) |
|---|---|---|
| corpus tokens | 2,705,595 | 1,234,733 |
| compression | 1.00× | **2.19×** |
| tokens/word | 3.99 | 1.82 |

Example: `Q: What is 7 times 8?` → 21 char-tokens vs 8 BPE tokens
(`'Q' ':' ' What' ' is' ' 7' ' times' ' 8' '?'`).

## 8-9. Context & model changes
Context: 192 chars → **512 BPE tokens ≈ 1,121 chars (5.8× the old window)**.
Architecture otherwise identical (8L/384d/6h, GELU, pre-LN, learned positions,
weight tying, causal mask). New embedding 2048×384 and positions 512×384 →
**15.18M params** (from 14.30M). **Full scratch init** — justified: the vocab
change invalidates embedding/head, token granularity changes every layer's
input statistics, and the positional table can't extend 192→512; partial
transfer would confound the tokenizer experiment.

## 10. Training ([train_phase4.py](../train_phase4.py))
Same recipe as Phase 3: example-aligned windows, answer-only token-level loss
masking, 70/30 qa/new-data window mixture (new-data = Phase2+3b combined,
disclosed deviation — excluding the 3b fixes would conflate tokenizer effect
with a data downgrade). AdamW, **lr 3e-4** cosine→3e-5 (spec suggested 1e-4;
raised because this is scratch training, documented deviation), clip 1.0,
batch 32×512, 3,000 steps, seed 1337. Prompt part and answer part of each
example are tokenized separately so training prompt tokenization is
byte-identical to inference. ~75 min on M5 (with cooling pauses between
chunks after a thermal restart). Final masked loss: train 0.016 / val 0.043.

## 11. Evaluation (frozen 400 questions, identical scoring)
eval.py received tokenizer-plumbing only: encode/decode/stop now come from
the checkpoint (BPE or legacy char, auto-detected); questions, sampling
parameters, and scoring unchanged; legacy path re-verified on v1b.

| category | v0 | v1 | v1b | **v2 (BPE)** |
|---|---|---|---|---|
| **Overall** | 17.0 | 26.5 | 28.8 | **35.8** |
| Arithmetic | 78 | 72 | 86 | **92** |
| Multi-step | 2 | 56 | 44 | **60** |
| Instruction | 8 | 14 | 12 | **32** |
| Reading | 0 | 4 | 16 | **38** |
| Logic | 0 | 0 | 10 | **12** |
| Patterns | 0 | 14 | 14 | **4** ⚠ |
| Knowledge | 48 | 52 | 48 | 48 |
| Coding | 0 | 0 | 0 | 0 |

vs v1b: 53 questions fixed, 25 broken.

## 12. Improvements
Word-level tokens transformed exactly the wording-bound skills Phase 3
diagnosed: **instruction following 12→32** ("Answer in one word: what color is
the sky?" → "blue" — the v0 failure, finally clean), **reading 16→38** (right
attribute AND much better copying, though it still often swaps in a training
name), multi-step best-ever 60, arithmetic best-ever 92. The old
crash-on-unknown-character failure is gone entirely.

## 13. Regressions
**Patterns 14→4.** Number sequences ("19, 22, 25, 28") now tokenize as digit
pieces inside a comma layout the model saw ~500 examples of — too few for the
new representation; it emits arithmetic-shaped noise instead. Genuine
regression, honestly reported: BPE changed which skills are cheap to learn.

## 14. Limitations
3-digit arithmetic still fails (training range is 2-digit); novel long word
problems beyond training length still fail; name-copying in reading is
approximate; coding untrained; single run (±7pp noise/category at n=50).

## 15. Next
Highest-leverage: pretraining on real text (TinyStories-class) now that the
tokenizer can represent it efficiently — plus a patterns-data refresh and
3-digit arithmetic coverage in the next data pass.
