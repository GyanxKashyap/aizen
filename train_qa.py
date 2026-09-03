"""
Train a bigger Aizen (~14M params) on the Q&A dataset, using the
Mac's GPU via MPS when available.

Run:  python3 make_qa_data.py   (once, to build data/qa.txt)
      python3 train_qa.py       (repeat until it says training complete)

Like train.py, this trains in resumable chunks: each run does CHUNK_ITERS
steps, checkpoints, and exits. Run it again to continue.
"""

import csv
import os
import time
import torch

from model import Aizen

torch.manual_seed(1337)

# ---------------------------------------------------------------- config ---
DATA_PATH = "data/qa.txt"
BLOCK_SIZE = 192
BATCH_SIZE = 64
N_EMBD = 384
N_HEAD = 6
N_LAYER = 8
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 3000
EVAL_INTERVAL = 100
EVAL_ITERS = 20
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

CKPT_PATH = "checkpoint_aizen.pt"
FINAL_PATH = "aizen.pt"
METRICS_PATH = "metrics_aizen.csv"

# ------------------------------------------------------------------ data ---
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.95 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"device: {DEVICE}")
print(f"dataset: {len(text):,} chars, vocab size: {vocab_size}")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([d[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample_answer(model, question):
    """Ask the model a question the same way chat.py will."""
    model.eval()
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(80):
        logits, _ = model(idx[:, -BLOCK_SIZE:])
        logits = logits[:, -1, :] / 0.5
        v, _ = torch.topk(logits, 20)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_id), dim=1)
        if itos[next_id.item()] == "\n":
            break
    model.train()
    return decode(idx[0].tolist())


# ----------------------------------------------------------------- model ---
model = Aizen(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITERS, eta_min=LR / 10)

start_iter = 0
metrics_exists = os.path.exists(METRICS_PATH)
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    start_iter = ckpt["iter"] + 1
    print(f"resumed from checkpoint at iter {ckpt['iter']}")

CHUNK_ITERS = int(os.environ.get("CHUNK_ITERS", 500))
end_iter = min(start_iter + CHUNK_ITERS, MAX_ITERS)

# ---------------------------------------------------------------- train ---
metrics_file = open(METRICS_PATH, "a" if metrics_exists else "w", newline="")
writer = csv.writer(metrics_file)
if not metrics_exists:
    writer.writerow(["iter", "train_loss", "val_loss", "elapsed_sec"])

start = time.time()
for it in range(start_iter, end_iter + 1):
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        losses = estimate_loss(model)
        elapsed = time.time() - start
        print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s", flush=True)
        writer.writerow([it, losses["train"], losses["val"], round(elapsed, 1)])
        metrics_file.flush()

    if it > 0 and it % 500 == 0:
        print("-" * 60)
        print(sample_answer(model, "How are you doing?"))
        print(sample_answer(model, "What is the capital of France?"))
        print(sample_answer(model, "What is 12 + 34?"))
        print("-" * 60, flush=True)

    if it >= MAX_ITERS:
        break

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

metrics_file.close()

torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "iter": end_iter,
}, CKPT_PATH)
print(f"checkpoint saved at iter {end_iter}")

if end_iter >= MAX_ITERS:
    # final weights + vocab + config in one self-contained file
    torch.save({
        "model": model.state_dict(),
        "chars": chars,
        "config": {
            "block_size": BLOCK_SIZE, "n_embd": N_EMBD,
            "n_head": N_HEAD, "n_layer": N_LAYER,
        },
    }, FINAL_PATH)
    print(f"training complete. model saved to {FINAL_PATH}")
else:
    print(f"chunk done ({start_iter}->{end_iter}). run again to continue training.")
