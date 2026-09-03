"""
Phase 5 stage 1: PRETRAIN Aizen v3 on TinyStories (plain next-token
prediction, no masking) with the v2 BPE tokenizer and 512-token context.

This is the "learn English first" stage every real LLM has and Aizen never
did. Windows are aligned to story starts and packed with consecutive whole
stories; loss is on every position (standard LM pretraining).

Run:  python3 train_phase5_pretrain.py [--steps 20000] [--lr 3e-4]
      (chunked/resumable via CHUNK_ITERS env, like the other trainers)
"""

import argparse
import csv
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import Aizen
from tokenizer import BPETokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--context-length", type=int, default=512)
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = random.Random(args.seed)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BLOCK = args.context_length
RESUME_PATH = "checkpoints/phase5_pretrain_resume.pt"
FINAL_PATH = "aizen_phase5_pretrained.pt"
METRICS_PATH = "results/metrics_phase5_pretrain.csv"
WARMUP = 200

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

tok = BPETokenizer.load("bpe_tokenizer_v2.json")
data = torch.load("data/pretrain_tokens.pt")
ids_all = data["ids"]
starts = data["story_starts"]
n_train_stories = int(len(starts) * 0.99)
print(f"device: {DEVICE} | vocab {tok.vocab_size} | ctx {BLOCK} | "
      f"{len(ids_all):,} tokens, {len(starts):,} stories (1% held out for val)")

cfg = {"block_size": BLOCK, "n_embd": 384, "n_head": 6, "n_layer": 8}
model = Aizen(vocab_size=tok.vocab_size, **cfg, dropout=0.1).to(DEVICE)


def get_batch(split):
    lo, hi = (0, n_train_stories) if split == "train" else (n_train_stories, len(starts))
    xs = []
    for _ in range(args.batch_size):
        s = starts[rng.randrange(lo, hi)].item()
        if s + BLOCK + 1 > len(ids_all):
            s = len(ids_all) - BLOCK - 1
        xs.append(ids_all[s:s + BLOCK + 1].long())
    b = torch.stack(xs).to(DEVICE)
    return b[:, :BLOCK], b[:, 1:]


@torch.no_grad()
def estimate_loss(iters=10):
    model.eval()
    out = {}
    for split in ("train", "val"):
        ls = []
        for _ in range(iters):
            x, y = get_batch(split)
            logits, _ = model(x)
            ls.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item())
        out[split] = sum(ls) / len(ls)
    model.train()
    return out


@torch.no_grad()
def sample_story(prompt="Once upon a time"):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(60):
        logits, _ = model(idx[:, -BLOCK:])
        logits = logits[:, -1, :] / 0.8
        v, _ = torch.topk(logits, 40)
        logits[logits < v[:, [-1]]] = float("-inf")
        idx = torch.cat((idx, torch.multinomial(torch.softmax(logits, dim=-1), 1)), dim=1)
    model.train()
    return tok.decode(idx[0].tolist()).replace("\n", " ")


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

# CRITICAL FIX: re-seed the batch sampler PER CHUNK. Without this, every
# resumed chunk re-draws the identical batch sequence (same seed, fresh
# process) and the model trains on one chunk's worth of data repeatedly.
rng = random.Random(args.seed + 7919 * start_iter)
torch.manual_seed(args.seed + 7919 * start_iter)

CHUNK_ITERS = int(os.environ.get("CHUNK_ITERS", 250))
end_iter = min(start_iter + CHUNK_ITERS, args.steps)

metrics_exists = os.path.exists(METRICS_PATH)
mf = open(METRICS_PATH, "a" if metrics_exists else "w", newline="")
writer = csv.writer(mf)
if not metrics_exists:
    writer.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_sec"])

start = time.time()
for it in range(start_iter, end_iter + 1):
    if it % 250 == 0 or it == args.steps:
        est = estimate_loss()
        print(f"step {it:6d} | lr {scheduler.get_last_lr()[0]:.2e} | train {est['train']:.4f} | val {est['val']:.4f} | {time.time()-start:.1f}s", flush=True)
        writer.writerow([it, f"{scheduler.get_last_lr()[0]:.6g}", round(est["train"], 4), round(est["val"], 4), round(time.time() - start, 1)])
        mf.flush()
        if not torch.isfinite(torch.tensor(est["train"])):
            raise SystemExit("FATAL: non-finite loss")
    if it > 0 and it % 2000 == 0:
        print("sample:", sample_story(), flush=True)

    if it >= args.steps:
        break
    x, y = get_batch("train")
    logits, _ = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
    # linear warmup for the first WARMUP steps of scratch pretraining
    if it < WARMUP:
        for g in optimizer.param_groups:
            g["lr"] = args.lr * (it + 1) / WARMUP
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if it >= WARMUP:
        scheduler.step()

mf.close()
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "iter": end_iter}, RESUME_PATH)
print(f"resume checkpoint saved at iter {end_iter}")

if end_iter >= args.steps:
    torch.save({"model": model.state_dict(), "config": cfg,
                "tokenizer": tok.state(),
                "meta": {"phase": "phase5_pretrain", "steps": args.steps,
                         "lr": args.lr, "batch_size": args.batch_size,
                         "corpus": "TinyStories 103MB slice", "seed": args.seed}},
               FINAL_PATH)
    print(f"pretraining complete. saved {FINAL_PATH}")
else:
    print(f"chunk done ({start_iter}->{end_iter}).")
