"""
Load the trained Aizen and generate text from it.

Usage:
    python3 generate.py                      # generate from empty prompt
    python3 generate.py "ROMEO:"             # generate continuing a prompt
    python3 generate.py "ROMEO:" 300 0.7     # prompt, max_new_tokens, temperature
"""

import sys
import torch

from model import Aizen

DATA_PATH = "data/input.txt"
WEIGHTS_PATH = "tinygpt.pt"
BLOCK_SIZE = 128
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DEVICE = "cpu"

# rebuild the same char vocab used during training (order must match!)
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

model = Aizen(vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

prompt = sys.argv[1] if len(sys.argv) > 1 else ""
max_new_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 300
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

if prompt:
    context = torch.tensor([encode(prompt)], dtype=torch.long)
else:
    context = torch.zeros((1, 1), dtype=torch.long)  # single "start" token

out = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=40)
print(decode(out[0].tolist()))
