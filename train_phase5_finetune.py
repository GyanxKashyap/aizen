"""
Phase 5 stage 2: FINE-TUNE the pretrained Aizen v3 on the task mixture
(qa.txt 70% / Phase2+3b 30% windows), with the Phase 3/4 recipe:
example-aligned windows + answer-only loss masking.

Init: aizen_phase5_pretrained.pt (same vocab/config, weights transfer 1:1 -
this is genuine fine-tuning, unlike Phase 4's justified scratch init).
Lower LR (1e-4) than scratch: we are adapting language knowledge to tasks,
not learning from zero, and we don't want to erase the pretrained English.

Run:  python3 train_phase5_finetune.py [--steps 2500] [--lr 1e-4]
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
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--mix-ratio", type=float, default=0.70)
parser.add_argument("--seed", type=int, default=1337)
parser.add_argument("--new-data", default="data/aizen_phase3b_train.txt")
parser.add_argument("--tag", default="phase5")
parser.add_argument("--base", default="aizen_phase5_pretrained.pt", help="pretrained checkpoint to fine-tune from")
parser.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = random.Random(args.seed)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BASE = args.base
FINAL_PATH = f"aizen_{args.tag}.pt"
RESUME_PATH = f"checkpoints/{args.tag}_finetune_resume.pt"
METRICS_PATH = f"results/metrics_{args.tag}_finetune.csv"

base = torch.load(BASE, map_location=DEVICE)
tok = BPETokenizer.from_state(base["tokenizer"])
cfg = base["config"]
BLOCK = cfg["block_size"]
model = Aizen(vocab_size=tok.vocab_size, **cfg, dropout=0.1).to(DEVICE)
model.load_state_dict(base["model"])
print(f"fine-tuning from {BASE} | vocab {tok.vocab_size} | ctx {BLOCK}")


def parse_examples(text):
    """Line-based masking: EVERY 'A:' line's answer text (+ its newline) is
    masked 1; questions and scaffolding are 0. Multi-turn conversation blocks
    (Q/A/Q/A/...) train every answer turn. Identical token stream to the old
    single-turn parser because '\n' is a standalone token that never merges."""
    out = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block.startswith("Q: ") or "\nA:" not in block:
            continue
        ids, mask = [], []
        for line in block.split("\n"):
            if line.startswith("A:"):
                p = tok.encode("A:")
                a = tok.encode(line[2:] + "\n")
                ids += p + a
                mask += [0] * len(p) + [1] * len(a)
            else:
                l = tok.encode(line + "\n")
                ids += l
                mask += [0] * len(l)
        out.append((ids, mask))
    return out


qa_pool = parse_examples(Path("data/qa.txt").read_text(encoding="utf-8"))
p2_pool = parse_examples(Path(args.new_data).read_text(encoding="utf-8"))
rng.shuffle(qa_pool)
rng.shuffle(p2_pool)
nq, np_ = int(len(qa_pool) * 0.02), int(len(p2_pool) * 0.02)
pools = {"train": (qa_pool[nq:], p2_pool[np_:]), "val": (qa_pool[:nq], p2_pool[:np_])}


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
        xs.append(ids[:BLOCK]); ys.append(ids[1:BLOCK + 1]); ms.append(mask[1:BLOCK + 1])
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
est = {"train": float("nan"), "val": float("nan")}
for it in range(start_iter, end_iter + 1):
    if it % 100 == 0 or it == args.steps:
        est = estimate_loss()
        print(f"step {it:5d} | lr {scheduler.get_last_lr()[0]:.2e} | masked train {est['train']:.4f} | val {est['val']:.4f} | {time.time()-start:.1f}s", flush=True)
        writer.writerow([it, f"{scheduler.get_last_lr()[0]:.6g}", round(est["train"], 4), round(est["val"], 4), round(time.time() - start, 1)])
        mf.flush()
        if not torch.isfinite(torch.tensor(est["train"])):
            raise SystemExit("FATAL: non-finite loss")
    if it >= args.steps:
        break
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.accum):
        x, y, m = get_batch("train")
        logits, _ = model(x)
        loss = masked_loss(logits, y, m)
        (loss / args.accum).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if torch.isfinite(gn):
        optimizer.step()
    else:
        print(f"step {it}: NON-FINITE grad norm - skipping step (MPS memory glitch guard)", flush=True)
    scheduler.step()

mf.close()
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "iter": end_iter}, RESUME_PATH)
print(f"resume checkpoint saved at iter {end_iter}")

if end_iter >= args.steps:
    # If the last chunk began exactly at args.steps the loop body never ran, so
    # `est` is still NaN. Measure once here rather than recording NaN forever.
    if not torch.isfinite(torch.tensor(est["train"])):
        est = estimate_loss()
    torch.save({"model": model.state_dict(), "config": cfg,
                "tokenizer": tok.state(),
                "meta": {"phase": args.tag, "new_data": args.new_data, "init": BASE,
                         "pretrain": base.get("meta", {}),
                         "finetune_steps": args.steps, "lr": args.lr,
                         "batch_size": args.batch_size, "accum": args.accum,
                         "mix_ratio_windows_qa": args.mix_ratio, "seed": args.seed,
                         "final_masked_train_loss": round(est["train"], 4),
                         "final_masked_val_loss": round(est["val"], 4)}},
               FINAL_PATH)
    print(f"fine-tuning complete. saved {FINAL_PATH}")
else:
    print(f"chunk done ({start_iter}->{end_iter}).")
