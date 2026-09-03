"""
Chat with the Q&A-trained Aizen.

Run:  python3 chat.py
Then type questions. Ctrl+C or "bye" to quit.

You can also ask a single question from the command line:
    python3 chat.py "What is the capital of France?"
"""

import sys
import torch

from model import Aizen

WEIGHTS_PATH = "aizen.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
chars = ckpt["chars"]
cfg = ckpt["config"]
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

model = Aizen(vocab_size=len(chars), **cfg).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


@torch.no_grad()
def answer(question, temperature=0.5, max_new_tokens=100):
    prompt = f"Q: {question}\nA:"
    try:
        ids = [stoi[c] for c in prompt]
    except KeyError as e:
        return f"(I don't know the character {e} - try plain English!)"
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, 20)
        logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        if itos[next_id.item()] == "\n":  # answer finished
            break
        idx = torch.cat((idx, next_id), dim=1)
    out = "".join(itos[i] for i in idx[0].tolist())
    return out[len(prompt):].strip()


if len(sys.argv) > 1:
    print(answer(" ".join(sys.argv[1:])))
    sys.exit()

print("Aizen Q&A - ask me something! (type 'bye' to quit)")
while True:
    try:
        q = input("\nyou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not q:
        continue
    print("bot:", answer(q))
    if q.lower() in ("bye", "goodbye", "quit", "exit"):
        break
