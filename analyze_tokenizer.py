"""
Old char tokenizer vs new BPE tokenizer - compression analysis
(Phase 4 spec section 5). Uses TRAINING corpus text only, never eval.json.

Run:  python3 analyze_tokenizer.py
"""

import random
from pathlib import Path
from tokenizer import BPETokenizer

tok = BPETokenizer.load("bpe_tokenizer.json")
corpus = (Path("data/qa.txt").read_text(encoding="utf-8") + "\n\n"
          + Path("data/aizen_phase3b_train.txt").read_text(encoding="utf-8"))

n_chars = len(corpus)
old_tokens = n_chars                      # char-level: 1 char = 1 token
bpe_ids = tok.encode(corpus)
new_tokens = len(bpe_ids)
n_words = len(corpus.split())

print("TOKENIZER ANALYSIS (training corpus only)")
print("=" * 45)
print(f"characters:            {n_chars:,}")
print(f"old tokens (char):     {old_tokens:,}")
print(f"new tokens (BPE):      {new_tokens:,}")
print(f"compression ratio:     {old_tokens / new_tokens:.2f}x")
print(f"tokens per character:  old 1.000 | BPE {new_tokens / n_chars:.3f}")
print(f"tokens per word:       old {old_tokens / n_words:.2f} | BPE {new_tokens / n_words:.2f}")
print(f"effective context:     old 192 chars | new 512 tokens = ~{int(512 * n_chars / new_tokens)} chars "
      f"({512 * n_chars / new_tokens / 192:.1f}x the old window)")

print("\nexample tokenizations (from training corpus):")
blocks = [b for b in corpus.split("\n\n") if b.strip()]
for b in random.Random(4).sample(blocks, 3):
    line = b.split("\n")[0]
    toks = [tok.id_to_token[i] for i in tok.encode(line)]
    print(f"  {line}")
    print(f"    chars: {len(line)} | BPE: {len(toks)} -> {toks}")
