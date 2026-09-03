"""
Phase 4: train Aizen v2 FROM SCRATCH with the from-scratch BPE tokenizer
(vocab ~2048) and a 512-TOKEN context window.

Why from scratch (spec section 7 justification): the vocabulary change makes
the old 73-row embedding/head meaningless, AND token granularity changes the
input statistics of every transformer layer (a char model's layer 1 learns
spelling features; a BPE model's layer 1 sees whole words). The positional
table also grows 192 -> 512 and cannot be meaningfully extended. Transferring
partial weights would confound the tokenizer experiment, so nothing is
transferred - full random init, identical architecture otherwise.

Kept from Phase 3's recipe: example-aligned windows, answer-only loss masking
(mask 1 on the tokens after "\nA:" through the terminating newline token),
70/30 qa/new-data window mixture, AdamW + cosine + grad clip 1.0.

Tokenization detail that matters: each example's prompt part ("Q: ...\nA:")
and answer part (" ...\n") are encoded SEPARATELY and concatenated, so the
prompt tokenization during training is byte-identical to how inference
tokenizes prompts. LR is 3e-4 (not the spec's suggested 1e-4) because this is
scratch training, not fine-tuning - documented deviation.

Run:  python3 train_phase4.py [--steps 3000] [--lr 3e-4] [--batch-size 32]
                              [--context-length 512] [--seed 1337]
"""

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Aizen
from tokenizer import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--mix-ratio", type=float, default=0.70)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--vocab-size", type=int, default=2048, help="informational; actual size comes from bpe_tokenizer.json")
parser.add_argument("--new-data", default="data/aizen_phase3b_train.txt")
parser.add_argument("--tag", default="phase4")
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = random.Random(args.seed)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
FINAL_PATH = f"aizen_{args.tag}.pt"
RESUME_PATH = f"checkpoints/{args.tag}_resume.pt"
METRICS_PATH = f"results/metrics_{args.tag}.csv"
EVAL_INTERVAL = 100
CKPT_INTERVAL = 500
BLOCK = args.context_length
GRAD_CLIP = 1.0

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

tok = BPETokenizer.load("bpe_tokenizer.json")
NEWLINE_ID = tok.token_to_id["\n"]
cfg = {"block_size": BLOCK, "n_embd": 384, "n_head": 6, "n_layer": 8}
model = Aizen(vocab_size=tok.vocab_size, **cfg, dropout=0.1).to(DEVICE)

# ------------------------------------------------------- build example pools ---
def parse_examples(text):
    out = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block.startswith("Q: ") or "\nA:" not in block:
            continue
        a_pos = block.index("\nA:") + 3
        prompt_ids = tok.encode(block[:a_pos])      # "Q: ...\nA:" - matches inference
        answer_ids = tok.encode(block[a_pos:] + "\n")
        ids = prompt_ids + answer_ids
        mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
        out.append((ids, mask))
    return out


qa_pool = parse_examples(Path("data/qa.txt").read_text(encoding="utf-8"))
p2_pool = parse_examples(Path(args.new_data).read_text(encoding="utf-8"))
rng.shuffle(qa_pool)
rng.shuffle(p2_pool)
nq, np_ = int(len(qa_pool) * 0.02), int(len(p2_pool) * 0.02)
pools = {"train": (qa_pool[nq:], p2_pool[np_:]), "val": (qa_pool[:nq], p2_pool[:np_])}
print(f"device: {DEVICE} | vocab: {tok.vocab_size} | context: {BLOCK} tokens")
print(f"pools: qa={len(qa_pool)}, new={len(p2_pool)} | avg tokens/example: "
      f"qa {sum(len(e[0]) for e in qa_pool)/len(qa_pool):.1f}, new {sum(len(e[0]) for e in p2_pool)/len(p2_pool):.1f}")


def get_batch(split):
    qa, p2 = pools[split]
    xs, ys, ms = [], [], []
    for _ in range(args.batch_size):
        pool = qa if rng.random() < args.mix_ratio else p2
        ids, mask = [], []
        while len(ids) < BLOCK + 1:
            e_ids, e_mask = pool[rng.randrange(len(pool))]
            ids.extend(e_ids)
            mask.extend(e_mask)
        xs.append(ids[:BLOCK])
        ys.append(ids[1:BLOCK + 1])
        ms.append(mask[1:BLOCK + 1])
    return (torch.tensor(xs, dtype=torch.long, device=DEVICE),
            torch.tensor(ys, dtype=torch.long, device=DEVICE),
            torch.tensor(ms, dtype=torch.bool, device=DEVICE))


def masked_loss(logits, y, m):
    per = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    m = m.view(-1)
    return (per * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def estimate_loss(iters=10):
    model.eval()
    out = {}
    for split in ("train", "val"):
        ls = []
        for _ in range(iters):
            x, y, m = get_batch(split)
            logits, _ = model(x)
            ls.append(masked_loss(logits, y, m).item())
        out[split] = sum(ls) / len(ls)
    model.train()
    return out


@torch.no_grad()
def sample_answer(question):
    model.eval()
    idx = torch.tensor([tok.encode(f"Q: {question}\nA:")], dtype=torch.long, device=DEVICE)
    for _ in range(80):
        logits, _ = model(idx[:, -BLOCK:])
        logits = logits[:, -1, :] / 0.5
        v, _ = torch.topk(logits, 20)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if nxt.item() == NEWLINE_ID:
            break
        idx = torch.cat((idx, nxt), dim=1)
    model.train()
    return tok.decode(idx[0].tolist())


optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr / 10)

start_iter = 0
if os.path.exists(RESUME_PATH):
    ck = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    start_iter = ck["iter"] + 1
    print(f"resumed at iter {ck['iter']}")

CHUNK_ITERS = int(os.environ.get("CHUNK_ITERS", 250))
end_iter = min(start_iter + CHUNK_ITERS, args.steps)

metrics_exists = os.path.exists(METRICS_PATH)
mf = open(METRICS_PATH, "a" if metrics_exists else "w", newline="")
writer = csv.writer(mf)
if not metrics_exists:
    writer.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_sec"])

start = time.time()
est = {"train": float("nan"), "val": float("nan")}
for it in range(start_iter, end_iter + 1):
    if it % EVAL_INTERVAL == 0 or it == args.steps:
        est = estimate_loss()
        print(f"step {it:5d} | lr {scheduler.get_last_lr()[0]:.2e} | masked train {est['train']:.4f} | val {est['val']:.4f} | {time.time()-start:.1f}s", flush=True)
        writer.writerow([it, f"{scheduler.get_last_lr()[0]:.6g}", round(est["train"], 4), round(est["val"], 4), round(time.time() - start, 1)])
        mf.flush()
        if not torch.isfinite(torch.tensor(est["train"])):
            raise SystemExit("FATAL: non-finite loss")

    if it > 0 and it % CKPT_INTERVAL == 0:
        torch.save({"model": model.state_dict(), "config": cfg,
                    "tokenizer": tok.state(), "iter": it},
                   f"checkpoints/aizen_{args.tag}_step_{it:04d}.pt")
        print("-" * 60)
        print(sample_answer("Answer in one word. What color is milk?"))
        print(sample_answer("What is 14 + 27 - 5?"))
        print(sample_answer("Zoe keeps a duck. Raj keeps a goat. What animal does Raj keep?"))
        print("-" * 60, flush=True)

    if it >= args.steps:
        break

    x, y, m = get_batch("train")
    logits, _ = model(x)
    loss = masked_loss(logits, y, m)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    scheduler.step()

mf.close()
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "iter": end_iter}, RESUME_PATH)
print(f"resume checkpoint saved at iter {end_iter}")

if end_iter >= args.steps:
    torch.save({
        "model": model.state_dict(),
        "config": cfg,
        "tokenizer": tok.state(),   # full tokenizer recoverable from checkpoint
        "meta": {
            "phase": args.tag, "init": "from scratch (justified in docstring)",
            "steps": args.steps, "lr": args.lr, "batch_size": args.batch_size,
            "context_tokens": BLOCK, "vocab_size": tok.vocab_size,
            "mix_ratio_windows_qa": args.mix_ratio, "seed": args.seed,
            "new_data": args.new_data, "grad_clip": GRAD_CLIP,
            "loss_masking": "answer-only, token-level",
            "final_masked_train_loss": round(est["train"], 4),
            "final_masked_val_loss": round(est["val"], 4),
        },
    }, FINAL_PATH)
    print(f"training complete. model saved to {FINAL_PATH}")
else:
    print(f"chunk done ({start_iter}->{end_iter}). run again to continue.")
