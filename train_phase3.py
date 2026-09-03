"""
Phase 3: fine-tune Aizen on a 70/30 mixture of the original qa.txt and the
Phase 2 instruction+reasoning dataset, starting from the aizen.pt checkpoint.

Key differences from train_qa.py (both documented in docs/phase3_training.md):

1. EXAMPLE ALIGNMENT - every training window starts exactly at an example's
   "Q:" and is packed with complete examples from the chosen pool; only the
   final example in a window is cut off. Position 0 = start of a question,
   matching how inference prompts look.

2. ANSWER-FOCUSED LOSS MASKING - each target character carries a 0/1 mask.
   Mask is 1 for the characters the model must produce at inference time:
   everything after "\nA:" through the terminating newline (the newline is
   included so the model learns to STOP after the answer). The question and
   the "Q:"/"A:" scaffolding are mask 0. Loss = mean cross-entropy over
   mask-1 targets only. NOTE: these loss values are NOT comparable to
   train_qa.py's unmasked loss.

3. MIXTURE - each batch row's window is packed from the qa.txt pool with
   probability --mix-ratio (default 0.70) or the Phase 2 pool otherwise.
   Mixture is defined at the WINDOW level; the by-character ratio is lower
   for qa.txt because its examples are shorter (reported in the log).

Never touches aizen.pt. Resumable in chunks (CHUNK_ITERS env, like train_qa).

Run:  python3 train_phase3.py [--steps 2500] [--lr 1e-4] [--batch-size 64]
                              [--mix-ratio 0.7] [--seed 1337]
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

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--mix-ratio", type=float, default=0.70, help="probability a window comes from qa.txt")
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--new-data", default="data/aizen_phase2_train.txt", help="the non-qa training pool")
parser.add_argument("--tag", default="phase3", help="suffix for checkpoints/metrics (e.g. phase3b)")
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = random.Random(args.seed)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_CKPT = "aizen.pt"
FINAL_PATH = f"aizen_{args.tag}.pt"
RESUME_PATH = f"checkpoints/{args.tag}_resume.pt"
METRICS_PATH = f"results/metrics_{args.tag}.csv"
EVAL_INTERVAL = 100
CKPT_INTERVAL = 500
VAL_FRACTION = 0.02
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

Path("checkpoints").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

# ------------------------------------------------- load base model + vocab ---
base = torch.load(BASE_CKPT, map_location=DEVICE)
chars, cfg = base["chars"], base["config"]
stoi = {ch: i for i, ch in enumerate(chars)}
BLOCK = cfg["block_size"]

model = Aizen(vocab_size=len(chars), **cfg).to(DEVICE)
model.load_state_dict(base["model"])

# ------------------------------------------------------- build example pools ---
def parse_examples(text):
    """Split corpus into examples; per-char mask = 1 on ' answer\n' after '\nA:'."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block.startswith("Q: ") or "\nA:" not in block:
            continue
        ex = block + "\n"                       # single newline terminates the answer
        a_pos = ex.index("\nA:") + 3            # first char of " answer...\n"
        ids = [stoi[c] for c in ex]
        mask = [0] * a_pos + [1] * (len(ex) - a_pos)
        out.append((torch.tensor(ids, dtype=torch.long),
                    torch.tensor(mask, dtype=torch.bool)))
    return out


qa_pool = parse_examples(Path("data/qa.txt").read_text(encoding="utf-8"))
p2_pool = parse_examples(Path(args.new_data).read_text(encoding="utf-8"))
rng.shuffle(qa_pool)
rng.shuffle(p2_pool)
n_qa_val, n_p2_val = int(len(qa_pool) * VAL_FRACTION), int(len(p2_pool) * VAL_FRACTION)
pools = {
    "train": (qa_pool[n_qa_val:], p2_pool[n_p2_val:]),
    "val": (qa_pool[:n_qa_val], p2_pool[:n_p2_val]),
}
qa_chars = sum(len(e[0]) for e in qa_pool)
p2_chars = sum(len(e[0]) for e in p2_pool)
char_ratio = args.mix_ratio * (qa_chars / len(qa_pool)) / (
    args.mix_ratio * qa_chars / len(qa_pool) + (1 - args.mix_ratio) * p2_chars / len(p2_pool))
print(f"device: {DEVICE}")
print(f"pools: qa={len(qa_pool)} examples ({qa_chars:,} chars), phase2={len(p2_pool)} examples ({p2_chars:,} chars)")
print(f"mixture: {args.mix_ratio:.0%} of windows from qa.txt (~{char_ratio:.0%} by characters)")


def get_batch(split):
    """Example-aligned packed windows with per-target loss masks."""
    qa, p2 = pools[split]
    xs, ys, ms = [], [], []
    for _ in range(args.batch_size):
        pool = qa if rng.random() < args.mix_ratio else p2
        ids, mask = [], []
        while len(ids) < BLOCK + 1:
            e_ids, e_mask = pool[rng.randrange(len(pool))]
            ids.extend(e_ids.tolist())
            mask.extend(e_mask.tolist())
        ids, mask = ids[:BLOCK + 1], mask[:BLOCK + 1]
        xs.append(ids[:BLOCK])
        ys.append(ids[1:])
        ms.append(mask[1:])                     # mask applies to TARGET chars
    x = torch.tensor(xs, dtype=torch.long, device=DEVICE)
    y = torch.tensor(ys, dtype=torch.long, device=DEVICE)
    m = torch.tensor(ms, dtype=torch.bool, device=DEVICE)
    return x, y, m


def masked_loss(logits, y, m):
    per_tok = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none")
    m = m.view(-1)
    return (per_tok * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def estimate_loss(iters=15):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            x, y, m = get_batch(split)
            logits, _ = model(x)
            losses.append(masked_loss(logits, y, m).item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def sample_answer(question):
    model.eval()
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=DEVICE)
    itos = {i: ch for i, ch in enumerate(chars)}
    for _ in range(110):
        logits, _ = model(idx[:, -BLOCK:])
        logits = logits[:, -1, :] / 0.5
        v, _ = torch.topk(logits, 20)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if itos[nxt.item()] == "\n":
            break
        idx = torch.cat((idx, nxt), dim=1)
    model.train()
    return "".join(itos[i] for i in idx[0].tolist())


# -------------------------------------------------------------- optimizer ---
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr / 10)

start_iter = 0
if os.path.exists(RESUME_PATH):
    ck = torch.load(RESUME_PATH, map_location=DEVICE)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    start_iter = ck["iter"] + 1
    print(f"resumed from {RESUME_PATH} at iter {ck['iter']}")

CHUNK_ITERS = int(os.environ.get("CHUNK_ITERS", 500))
end_iter = min(start_iter + CHUNK_ITERS, args.steps)

# ------------------------------------------------- v0 baseline (read-only) ---
v0 = json.loads(Path("results/eval_results.json").read_text(encoding="utf-8"))
if start_iter == 0:
    print(f"v0 baseline (frozen): {v0['overall_accuracy_pct']}% overall on {v0['checkpoint']}")

# ------------------------------------------------------------------ train ---
metrics_exists = os.path.exists(METRICS_PATH)
mf = open(METRICS_PATH, "a" if metrics_exists else "w", newline="")
writer = csv.writer(mf)
if not metrics_exists:
    writer.writerow(["step", "lr", "train_loss", "val_loss", "elapsed_sec"])

start = time.time()
losses_est = {"train": float("nan"), "val": float("nan")}
for it in range(start_iter, end_iter + 1):
    if it % EVAL_INTERVAL == 0 or it == args.steps:
        losses_est = estimate_loss()
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - start
        print(f"step {it:5d} | lr {lr_now:.2e} | masked train loss {losses_est['train']:.4f} | val {losses_est['val']:.4f} | {elapsed:.1f}s", flush=True)
        writer.writerow([it, f"{lr_now:.6g}", round(losses_est["train"], 4), round(losses_est["val"], 4), round(elapsed, 1)])
        mf.flush()
        if not torch.isfinite(torch.tensor(losses_est["train"])):
            raise SystemExit("FATAL: non-finite training loss - stopping")

    if it > 0 and it % CKPT_INTERVAL == 0:
        torch.save({"model": model.state_dict(), "chars": chars, "config": cfg, "iter": it},
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
        "chars": chars,
        "config": cfg,
        "meta": {
            "phase": args.tag,
            "new_data": args.new_data,
            "base_checkpoint": BASE_CKPT,
            "steps": args.steps,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "mix_ratio_windows_qa": args.mix_ratio,
            "mix_ratio_chars_qa": round(char_ratio, 3),
            "seed": args.seed,
            "grad_clip": GRAD_CLIP,
            "weight_decay": WEIGHT_DECAY,
            "loss_masking": "answer-only (chars after '\\nA:' incl. terminating newline)",
            "final_masked_train_loss": round(losses_est["train"], 4),
            "final_masked_val_loss": round(losses_est["val"], 4),
            "v0_baseline_overall_pct": v0["overall_accuracy_pct"],
            "v0_baseline_by_category": v0["category_accuracy_pct"],
        },
    }, FINAL_PATH)
    print(f"training complete. model saved to {FINAL_PATH}")
else:
    print(f"chunk done ({start_iter}->{end_iter}). run again to continue.")
