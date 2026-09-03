"""
Aizen web app - one server, two models, tabbed UI.

  /chat   -> Aizen v6 assistant   (aizen_phase8.pt, task fine-tuned)
  /story  -> Aizen Storyteller    (aizen_phase8_pretrained.pt, pretrain only)
  /meta   -> model card + live benchmark data read from results/

The storyteller is loaded lazily on first use so startup stays fast.

Run:  .venv/bin/python server.py    ->  http://localhost:8321
Env:  AIZEN_CKPT / AIZEN_STORY_CKPT to serve different checkpoints.
"""

import json
import os
from pathlib import Path

import torch
from flask import Flask, Response, jsonify, request, send_file

from model import Aizen

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHAT_CKPT = os.environ.get("AIZEN_CKPT", "aizen_phase8.pt")
STORY_CKPT = os.environ.get("AIZEN_STORY_CKPT", "aizen_phase8_pretrained.pt")


def load(path):
    """Load a checkpoint; auto-detect BPE (phase 4+) vs legacy char tokenizer."""
    ckpt = torch.load(path, map_location=DEVICE)
    if "tokenizer" in ckpt:
        from tokenizer import BPETokenizer
        tok = BPETokenizer.from_state(ckpt["tokenizer"])
        enc, dec, stop, n_vocab = tok.encode, tok.decode, {tok.token_to_id["\n"]}, tok.vocab_size
    else:
        chars = ckpt["chars"]
        stoi = {c: i for i, c in enumerate(chars)}
        itos = dict(enumerate(chars))
        enc = lambda t: [stoi[c] for c in t]
        dec = lambda ids: "".join(itos[i] for i in ids)
        stop, n_vocab = {stoi["\n"]}, len(chars)
    m = Aizen(vocab_size=n_vocab, **ckpt["config"]).to(DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()
    n_params = sum(p.numel() for p in m.parameters())
    return {"model": m, "encode": enc, "decode": dec, "stop": stop,
            "config": ckpt["config"], "params": n_params, "path": path,
            "meta": ckpt.get("meta", {})}


CHAT = load(CHAT_CKPT)
STORY = None  # lazy


def story_model():
    global STORY
    if STORY is None:
        STORY = load(STORY_CKPT)
    return STORY


app = Flask(__name__)


@torch.no_grad()
def generate(bundle, prompt, n, temperature, top_k, stop_at_newline):
    m, enc, dec = bundle["model"], bundle["encode"], bundle["decode"]
    idx = torch.tensor([enc(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(n):
        logits, _ = m(idx[:, -m.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if stop_at_newline and nxt.item() in bundle["stop"]:
            break
        idx = torch.cat((idx, nxt), dim=1)
        yield dec([nxt.item()])


@app.route("/")
def index():
    return send_file("ui/index.html")


@app.route("/chat", methods=["POST"])
def chat():
    b = request.get_json(silent=True) or {}
    q = (b.get("question") or "").strip()
    if not q:
        return {"error": "empty question"}, 400
    prompt = f"Q: {q}\nA:"
    budget = 380 - len(CHAT["encode"](prompt))
    for pq, pa in reversed(b.get("history") or []):
        turn = f"Q: {pq}\nA: {pa}\n"
        cost = len(CHAT["encode"](turn))
        if cost > budget:
            break
        prompt, budget = turn + prompt, budget - cost
    try:
        CHAT["encode"](prompt)
    except KeyError:
        return Response("Sorry, I only understand plain English characters!", mimetype="text/plain")
    return Response(generate(CHAT, prompt, 200, 0.5, 20, True), mimetype="text/plain")


@app.route("/story", methods=["POST"])
def story():
    b = request.get_json(silent=True) or {}
    prompt = (b.get("prompt") or "Once upon a time").strip()
    n = min(int(b.get("tokens", 200)), 400)
    temp = float(b.get("temperature", 0.85))
    return Response(generate(story_model(), prompt, n, temp, 40, False), mimetype="text/plain")


BENCH = [
    ("v0", "Original char-level QA model (14.3M)", "eval_results.json"),
    ("v1", "Phase 3 fine-tune (answer-masked loss)", "eval_results_phase3.json"),
    ("v1b", "Phase 3b targeted-fix dataset", "eval_results_phase3b.json"),
    ("v2", "From-scratch BPE tokenizer + 512 context", "eval_results_phase4.json"),
    ("v3", "TinyStories pretraining + masked-QA fine-tune", "eval_results_phase5.json"),
    ("v4b", "Hybrid data (synthetic + bAbI, dosage tuned)", "eval_results_phase6b.json"),
    ("v5", "Conversation + negatives + false premises", "eval_results_phase7.json"),
    ("v6", "Aizen-40M: scale finale - current", "eval_results_phase8.json"),
]

PHASES = [
    ("Evaluation", "Build the ruler before touching the model.",
     "A frozen set of 400 questions across 8 categories plus eval.py, which scores any checkpoint identically. Two rules: never edit the questions, one training run per approved experiment.",
     "v0 baseline: 17.0% - honest, and every gain after it was real."),
    ("Dataset", "Teach instruction following and reasoning.",
     "10,000 generated examples - 5,000 instruction (exact formats, transforms, classification, extraction) and 5,000 reasoning, every answer computed in Python so the chains are true by construction.",
     "Validated: 0 false equations, 0 eval leaks."),
    ("Fine-tuning", "Answer-masked loss and example-aligned windows.",
     "Loss counts only answer tokens; every training window starts at a question. Mixed 70/30 with the original data to protect old skills.",
     "17.0% -> 26.5%, then a targeted-fix pass -> 28.75%."),
    ("BPE tokenizer", "Stop spending capacity on spelling.",
     "Byte-pair encoding written from scratch - count pairs, merge the most frequent, repeat to 2048 tokens. Digits stay single so arithmetic survives; newline never merges so Q/A structure stays visible.",
     "2.19x compression; 28.75% -> 35.75% with identical parameters."),
    ("Pretraining", "Give the model English before asking it questions.",
     "103MB of TinyStories cleaned to ASCII, a 4096-vocab tokenizer, 26.2M tokens pretrained for 16k steps as a plain language model, then a short masked-QA fine-tune.",
     "Logic 12% -> 66%. Overall 35.75% -> 45.25%."),
    ("Hybrid data", "Mix curated public data with synthetic.",
     "Added bAbI reading/deduction tasks from HuggingFace alongside new synthetic patterns and 3-digit arithmetic. First attempt over-diluted the mix; capping bAbI at 2,400 examples fixed it.",
     "45.25% -> 52.25%. Lesson: proportion matters as much as presence."),
    ("Conversation", "Multi-turn memory, negatives, false premises.",
     "2,500 multi-turn blocks so follow-ups like \"and plus 3?\" resolve against the previous answer, plus negative results and syllogisms with false premises. Every answer turn is masked for loss.",
     "Overall dipped to 48.0% while adding conversation - a deliberate trade."),
    ("Scale finale", "40M parameters, the one variable changed.",
     "12 layers, 512 dim, 8 heads pretrained on 200MB of TinyStories (52.5M tokens, 16k steps) then fine-tuned 5k steps on the full task pool. Caught two bugs first: a per-chunk reseeding flaw that had been shrinking every resumed run, and silent Metal GPU memory corruption above batch ~20.",
     "60.75% - every category at an all-time high, and the see-saw finally stopped."),
]


@app.route("/meta")
def meta():
    versions = []
    for tag, label, fn in BENCH:
        p = Path("results") / fn
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        versions.append({"tag": tag, "label": label,
                         "overall": d["overall_accuracy_pct"],
                         "categories": d["category_accuracy_pct"],
                         "checkpoint": d["checkpoint"]})
    cfg = CHAT["config"]
    return jsonify({
        "checkpoint": CHAT["path"],
        "params": CHAT["params"],
        "config": cfg,
        "story_checkpoint": STORY_CKPT,
        "versions": versions,
        "phases": [{"title": t, "objective": o, "work": w, "result": r} for t, o, w, r in PHASES],
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8321))
    print(f"Aizen is ready -> http://localhost:{port}  (chat: {CHAT_CKPT}, story: {STORY_CKPT})")
    app.run(host="::", port=port, threaded=False)
