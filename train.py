"""
Train Aizen on tiny-shakespeare, character-level, on CPU.

Run:  python3 train.py
Logs train/val loss to results/metrics.csv and prints sample generations
periodically so you can literally watch the model learn to spell,
then form words, then form (bad) Shakespeare.
"""

import csv
import os
import time
import torch

from model import Aizen

torch.manual_seed(1337)

# ---------------------------------------------------------------- config ---
# Kept deliberately small so a full run finishes in a few minutes on CPU.
DATA_PATH = "data/input.txt"
BLOCK_SIZE = 128       # context length (chars)
BATCH_SIZE = 32
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
LR = 3e-4
MAX_ITERS = 1000
EVAL_INTERVAL = 100
EVAL_ITERS = 30
SAMPLE_INTERVAL = 250
DEVICE = "cpu"

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
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(f"dataset: {len(text):,} chars, vocab size: {vocab_size}")
print(f"train: {len(train_data):,} tokens | val: {len(val_data):,} tokens")


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

# --------------------------------------------------------- resume support ---
# Training happens in short chunks (tool call time limit), so we checkpoint
# model + optimizer + iteration count and resume from where we left off.
CKPT_PATH = "checkpoint.pt"
start_iter = 0
metrics_exists = os.path.exists("results/metrics.csv")
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_iter = ckpt["iter"] + 1
    print(f"resumed from checkpoint at iter {ckpt['iter']}")

CHUNK_ITERS = int(os.environ.get("CHUNK_ITERS", 250))  # how many steps this process
end_iter = min(start_iter + CHUNK_ITERS, MAX_ITERS)

# ---------------------------------------------------------------- train ---
metrics_file = open("results/metrics.csv", "a" if metrics_exists else "w", newline="")
writer = csv.writer(metrics_file)
if not metrics_exists:
    writer.writerow(["iter", "train_loss", "val_loss", "elapsed_sec"])

start = time.time()
for it in range(start_iter, end_iter + 1):
    if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
        losses = estimate_loss(model)
        elapsed = time.time() - start
        print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | {elapsed:.1f}s")
        writer.writerow([it, losses["train"], losses["val"], round(elapsed, 1)])
        metrics_file.flush()

    if it > 0 and it % SAMPLE_INTERVAL == 0:
        context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        sample = decode(model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)[0].tolist())
        print("-" * 60)
        print(f"sample @ step {it}:\n{sample}")
        print("-" * 60)

    if it >= MAX_ITERS:
        break

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

metrics_file.close()

# always checkpoint so the next chunk can resume
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": end_iter}, CKPT_PATH)
print(f"checkpoint saved at iter {end_iter}")

if end_iter >= MAX_ITERS:
    torch.save(model.state_dict(), "tinygpt.pt")
    print("training complete. weights saved to tinygpt.pt, metrics logged to results/metrics.csv")
    print("\nFinal sample:")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()))
else:
    print(f"chunk done ({start_iter}->{end_iter}). run again to continue training.")
