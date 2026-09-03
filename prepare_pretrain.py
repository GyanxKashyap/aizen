"""
Pre-tokenize the TinyStories pretraining corpus ONCE so the chunked training
runs don't re-encode 103MB of text on every resume.

Writes data/pretrain_tokens.pt:
  {"ids": int16 tensor of all token ids, "story_starts": long tensor of
   offsets where each story begins (windows are aligned to story starts)}

Run:  python3 prepare_pretrain.py
"""

import time
import torch
from pathlib import Path
from tokenizer import BPETokenizer

tok = BPETokenizer.load("bpe_tokenizer_v2.json")
text = Path("data/pretrain.txt").read_text(encoding="utf-8")
stories = [s for s in text.split("\n\n") if s.strip()]
print(f"{len(stories):,} stories, {len(text):,} chars, vocab {tok.vocab_size}")

ids, starts = [], []
t0 = time.time()
for i, s in enumerate(stories):
    starts.append(len(ids))
    ids.extend(tok.encode(s + "\n\n"))
    if (i + 1) % 20000 == 0:
        print(f"  {i + 1:,} stories tokenized ({time.time() - t0:.0f}s)", flush=True)

assert tok.vocab_size < 32768, "int16 storage assumes vocab < 32768"
torch.save({"ids": torch.tensor(ids, dtype=torch.int16),
            "story_starts": torch.tensor(starts, dtype=torch.long)},
           "data/pretrain_tokens.pt")
print(f"done: {len(ids):,} tokens ({len(text) / len(ids):.2f} chars/token), "
      f"saved data/pretrain_tokens.pt ({Path('data/pretrain_tokens.pt').stat().st_size / 1e6:.0f}MB)")
