# Aizen — a 40M-parameter LLM built from scratch on a MacBook Air

Aizen is a decoder-only Transformer written line by line in raw PyTorch —
no `transformers`, no `tiktoken`, no pretrained weights. The architecture,
the byte-pair tokenizer, the training loops, the evaluation harness and the
web UI are all in this repository, all readable.

It scores **60.75%** on a frozen 400-question benchmark, up from **17.0%**
at the first honest measurement. Every point of that came from data,
tokenization and method — the parameter count was raised exactly once.

```
  v0  ###########                               17.00%
  v1  ##################                        26.50%
 v1b  ###################                       28.75%
  v2  ########################                  35.75%
  v3  ##############################            45.25%
  v4  ################################          47.25%
 v4b  ###################################       52.25%
  v5  ################################          48.00%
  v6  ########################################  60.75%  <- current
```

---

## Quick start

The trained weights are too large for git (175MB and 166MB, against GitHub's
100MB limit), so they ship as release assets. Grab them into the repo root:

```bash
gh release download v6 --pattern '*.pt'
```

Or download `aizen_phase8.pt` and `aizen_phase8_pretrained.pt` by hand from
the [releases page](../../releases/latest). Then:

```bash
python3 -m venv .venv && .venv/bin/pip install torch flask
.venv/bin/python server.py          # -> http://localhost:8321
```

Everything except the weights is in this repository — if you would rather
train your own instead of downloading mine, skip the release and follow
[docs/phase8_scale.md](docs/phase8_scale.md).

One server, two models, one page:

| mode | checkpoint | what it is |
|---|---|---|
| **Chat** | `aizen_phase8.pt` | the task-tuned assistant — arithmetic, logic, reading, multi-turn |
| **Story** | `aizen_phase8_pretrained.pt` | the same weights *before* fine-tuning — pure TinyStories English |

Terminal instead of browser:

```bash
.venv/bin/python chat_phase4.py
```

Score any checkpoint against the frozen benchmark:

```bash
.venv/bin/python eval.py --checkpoint aizen_phase8.pt --out results/my_run.json
```

---

## What Aizen is

| | |
|---|---|
| Parameters | **40,188,928** (40.19M) |
| Layers / dim / heads | 12 / 512 / 8 |
| Context | 512 tokens |
| Vocabulary | 4,096 BPE merges, learned from scratch |
| Pretraining | TinyStories, 200MB → 52.5M tokens, 16,000 steps |
| Fine-tuning | 5,000 steps on 28,100 task examples, answer-masked loss |
| Trained on | one MacBook Air (M-series), Metal/MPS, over ~6 nights |
| Final losses | pretrain val 1.356 · fine-tune masked val 0.014 |

GPT-2 style: pre-LayerNorm blocks, learned absolute positions, 4× GELU MLP,
causal masking, weight-tied embeddings. `model.py` is 150 lines and every
one of them is commented.

---

## Benchmark

400 questions, 8 categories, 50 each. Written **before** any improvement
work started and never edited since — the two standing rules were *never
touch the questions* and *one training run per approved experiment*.

| | v0 | v1 | v1b | v2 | v3 | v4 | v4b | v5 | **v6** |
|---|---|---|---|---|---|---|---|---|---|
| **overall** | 17.0 | 26.5 | 28.75 | 35.75 | 45.25 | 47.25 | 52.25 | 48.0 | **60.75** |
| arithmetic | 78 | 72 | 86 | 92 | 90 | 98 | 98 | 96 | **100** |
| multi-step arith. | 2 | 56 | 44 | 60 | 54 | 64 | 68 | 68 | **82** |
| logic | 0 | 0 | 10 | 12 | 66 | 56 | 66 | 64 | **72** |
| patterns | 0 | 14 | 14 | 4 | 10 | 20 | 34 | 14 | **48** |
| general knowledge | 48 | 52 | 48 | 48 | 58 | 62 | 62 | 60 | **64** |
| instruction following | 8 | 14 | 12 | 32 | 38 | 40 | 38 | 42 | **50** |
| reading comprehension | 0 | 4 | 16 | 38 | 44 | 36 | 48 | 38 | **68** |
| coding | 0 | 0 | 0 | 0 | 2 | 2 | 4 | 2 | **2** |

v6 is the first generation where **every category is simultaneously at its
all-time high**. Before it, gains traded against each other — the see-saw
that runs through v1–v5 is a capacity ceiling, not a data problem.

Coding is the honest failure: 2%. The model has never read a line of code.

---

## The eight phases

| # | Phase | The idea | Result |
|---|---|---|---|
| 1 | [Evaluation](docs/evaluation.md) | Build the ruler before touching the model | v0 baseline **17.0%** — honest |
| 2 | [Dataset](docs/phase2_dataset.md) | 10,000 examples, every answer computed in Python | 0 false equations, 0 eval leaks |
| 3 | [Fine-tuning](docs/phase3_training.md) | Answer-masked loss, example-aligned windows | 17.0 → **28.75%** |
| 4 | [BPE tokenizer](docs/phase4_bpe.md) | Stop spending capacity on spelling | 2.19× compression → **35.75%** |
| 5 | [Pretraining](docs/phase5_pretraining.md) | Give it English before asking it questions | logic 12 → 66%, overall **45.25%** |
| 6 | [Hybrid data](docs/phase6_hybrid.md) | Mix public bAbI with synthetic — and cap the dose | **52.25%** |
| 7 | [Conversation](docs/phase7_conversation.md) | Multi-turn memory, negatives, false premises | 48.0% — a deliberate trade |
| 8 | [Scale finale](docs/phase8_scale.md) | 14M → 40M, the one time parameters moved | **60.75%** |

---

## Files

**Model & inference**
- `model.py` — the architecture, fully commented
- `tokenizer.py` — byte-pair encoding written from scratch
- `server.py` — Flask app serving both models
- `ui/index.html` — the web interface
- `chat_phase4.py`, `generate.py` — terminal clients

**Evaluation**
- `eval.py` — the frozen scorer (works on any checkpoint, char or BPE)
- `data/eval.json` — the 400 questions
- `results/` — one JSON per generation, plus training CSVs

**Training**
- `train_phase8_pretrain.py` — the 40M pretrainer
- `train_phase5_finetune.py` — the fine-tuner used from phase 5 onward
- `train_phase3.py`, `train_phase4.py`, `train_qa.py` — earlier generations
- `prepare_pretrain.py` — corpus cleaning + tokenization

**Data generation** — `make_qa_data.py`, `make_eval_data.py`,
`make_phase2_data.py`, `make_phase3b_data.py`, `make_phase6_data.py`,
`make_phase7_data.py`. Every answer is computed, never written by hand.

**Documentation** — [AIZEN.md](AIZEN.md) is the complete model card:
sources, licences, token counts, limitations. `docs/` has one report per phase.

---

## Training is resumable

Every trainer runs in chunks so a laptop can close its lid:

```bash
CHUNK_ITERS=250 .venv/bin/python train_phase5_finetune.py --steps 5000 \
  --base aizen_phase8_pretrained.pt --new-data data/aizen_phase7_train.txt --tag phase8
```

Each call trains `CHUNK_ITERS` steps, writes `checkpoints/<tag>_resume.pt`,
and picks up where it stopped. **The sampler is re-seeded per chunk** —
without that, every resumed chunk redraws the identical batches and the run
silently trains on a fraction of its data. That bug was live from phase 3 to
phase 8; finding it is written up in [docs/phase8_scale.md](docs/phase8_scale.md).

---

## Known limits

- **Coding: 2%.** No code in the training corpus. Nothing to be proud of, but honest.
- **3-digit arithmetic** still slips occasionally, even at 100% on the eval's 2-digit set.
- **Phase 8 is confounded.** The scale increase and the sampler fix landed in the
  same run, so "40M caused +12.75 points" is not something this project can claim.
  Separating them needs a 16M-with-fixed-sampler ablation that was never run.
- **It is 40M parameters.** It will confidently state that the capital of
  Australia is Sydney. Treat it as a demonstration of method, not a source of facts.

---

*Built by Gyan. Model, tokenizer, benchmark and interface all from scratch.*

---

## License

MIT — see [LICENSE](LICENSE).
