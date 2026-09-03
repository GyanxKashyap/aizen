"""
Chat with the BPE-tokenized Aizen (Phase 4). Does not touch chat.py.

Run:  python3 chat_phase4.py                       # interactive
      python3 chat_phase4.py "What is 123 + 456?"  # one-shot
"""

import sys
import torch

from model import Aizen
from tokenizer import BPETokenizer

WEIGHTS = "aizen_phase4.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load(WEIGHTS, map_location=DEVICE)
tok = BPETokenizer.from_state(ckpt["tokenizer"])
NEWLINE_ID = tok.token_to_id["\n"]
model = Aizen(vocab_size=tok.vocab_size, **ckpt["config"]).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()


@torch.no_grad()
def answer(question, temperature=0.5, top_k=20, max_new_tokens=200):
    prompt = f"Q: {question}\nA:"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :] / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
        if nxt.item() == NEWLINE_ID:
            break
        idx = torch.cat((idx, nxt), dim=1)
    return tok.decode(idx[0].tolist())[len(prompt):].strip()


if len(sys.argv) > 1:
    print(answer(" ".join(sys.argv[1:])))
    sys.exit()

print("Aizen v2 (BPE) - ask me something! (type 'bye' to quit)")
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
