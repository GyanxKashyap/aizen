"""
A byte-pair-encoding (BPE) tokenizer written completely from scratch.
No tiktoken, no HuggingFace, no sentencepiece - just Python stdlib.

How BPE works (the whole idea in four sentences):
  1. Start with a vocabulary of single characters, so any text can be encoded.
  2. Count which ADJACENT PAIR of tokens appears most often in the corpus.
  3. Merge that pair into one new token and add it to the vocabulary.
  4. Repeat until the vocabulary reaches the target size. Frequent sequences
     ("the ", "answer", "capital of ") become single tokens; rare text falls
     back to smaller pieces, ultimately single characters.

Implementation choices (documented per Phase 4 spec):
- WORD-LEVEL merging (like GPT-2): text is pre-split into "words" by a regex,
  and merges never cross word boundaries. This keeps training fast and stops
  pathological merges like "e.\nQ".
- The pre-split regex attaches a leading space to each word (so " the" is one
  token, as in GPT-2), keeps "\n" as its own word (it terminates answers -
  generation stops on it, so it must never be merged into anything), and
  splits EVERY DIGIT into its own word so numbers are always sequences of
  single-digit tokens ("123" -> "1","2","3"). Multi-digit tokens would make
  arithmetic nearly unlearnable for a small model.
- Special tokens (fixed, deterministic ids):
      <pad>=0  <unk>=1  <bos>=2  <eos>=3
  <unk> replaces any character not seen in the BPE training corpus - unknown
  input can no longer crash encoding (the old char tokenizer's KeyError).
  <pad>/<bos>/<eos> are reserved for future phases; current training packs
  windows with newline-terminated examples and needs none of them.
- Round trip: decode(encode(t)) == t exactly, for any text whose characters
  appeared in the training corpus. Unknown characters degrade to <unk>
  (decoded as the placeholder U+2370-like "?"-free empty string is ambiguous,
  so we decode <unk> to "\x1a" SUB placeholder - documented lossy).

API:
    tok = BPETokenizer()
    tok.train(text, vocab_size=2048, min_pair_freq=10)
    ids = tok.encode("How much is 12 + 34?")
    text = tok.decode(ids)
    tok.save("bpe_tokenizer.json"); tok = BPETokenizer.load(path)
"""

import json
import re
from collections import Counter

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
UNK_PLACEHOLDER = "\x1a"  # what <unk> decodes to (SUB char) - documented lossy

# newline | space?letter-run | space?single-digit | space?single-symbol | space
_SPLIT = re.compile(r"\n| ?[A-Za-z']+| ?[0-9]| ?[^ \nA-Za-z0-9']| ")


def split_words(text):
    words = _SPLIT.findall(text)
    assert "".join(words) == text, "pre-split regex lost characters"
    return words


class BPETokenizer:
    def __init__(self):
        self.token_to_id = {}   # token string -> id
        self.id_to_token = {}   # id -> token string
        self.merges = {}        # (tok_a, tok_b) -> rank (lower = earlier merge)
        self._cache = {}

    # ------------------------------------------------------------- training
    def train(self, text, vocab_size=2048, min_pair_freq=10, verbose=True):
        # base vocabulary: specials + every corpus character + ALL printable
        # ASCII (so 'X', '(', '%', '"' etc. encode even though the corpus never
        # used them - their embeddings just stay untrained until data uses them)
        import string
        ascii_base = string.ascii_letters + string.digits + string.punctuation + " \n"
        chars = sorted(set(text) | set(ascii_base))
        for i, s in enumerate(SPECIALS):
            self.token_to_id[s] = i
        for ch in chars:
            self.token_to_id[ch] = len(self.token_to_id)

        # corpus as unique words (tuples of 1-char tokens) with frequencies
        word_freq = Counter(split_words(text))
        words = {w: list(w) for w in word_freq}

        merge_list = []
        while len(self.token_to_id) < vocab_size:
            pairs = Counter()
            for w, toks in words.items():
                f = word_freq[w]
                for a, b in zip(toks, toks[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            (a, b), freq = pairs.most_common(1)[0]
            if freq < min_pair_freq:
                break  # remaining merges too rare to be useful
            new_tok = a + b
            self.token_to_id[new_tok] = len(self.token_to_id)
            merge_list.append((a, b))
            for w, toks in words.items():
                if a not in toks:
                    continue
                out, i = [], 0
                while i < len(toks):
                    if i < len(toks) - 1 and toks[i] == a and toks[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(toks[i])
                        i += 1
                words[w] = out
            if verbose and len(merge_list) % 250 == 0:
                print(f"  merge {len(merge_list)}: {a!r}+{b!r} (freq {freq}) -> vocab {len(self.token_to_id)}")

        self.merges = {pair: r for r, pair in enumerate(merge_list)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        if verbose:
            print(f"trained: vocab {len(self.token_to_id)} ({len(merge_list)} merges, {len(chars)} base chars)")

    # ------------------------------------------------------------- encoding
    def _bpe_word(self, word):
        """Apply learned merges to one word, lowest-rank merge first."""
        if word in self._cache:
            return self._cache[word]
        toks = [c if c in self.token_to_id else "<unk>" for c in word]
        while len(toks) > 1:
            ranked = [(self.merges.get((a, b), 1 << 30), i)
                      for i, (a, b) in enumerate(zip(toks, toks[1:]))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            toks = toks[:i] + [toks[i] + toks[i + 1]] + toks[i + 2:]
        self._cache[word] = toks
        return toks

    def encode(self, text):
        ids = []
        for w in split_words(text):
            ids.extend(self.token_to_id.get(t, UNK) for t in self._bpe_word(w))
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            t = self.id_to_token.get(i, "<unk>")
            if t == "<unk>":
                out.append(UNK_PLACEHOLDER)
            elif t in SPECIALS:
                continue
            else:
                out.append(t)
        return "".join(out)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    # ---------------------------------------------------------- persistence
    def state(self):
        return {"specials": SPECIALS,
                "tokens": [self.id_to_token[i] for i in range(len(self.id_to_token))],
                "merges": [[a, b] for (a, b), _ in sorted(self.merges.items(), key=lambda kv: kv[1])]}

    def save(self, path):
        json.dump(self.state(), open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def from_state(cls, st):
        tok = cls()
        tok.token_to_id = {t: i for i, t in enumerate(st["tokens"])}
        tok.id_to_token = dict(enumerate(st["tokens"]))
        tok.merges = {(a, b): r for r, (a, b) in enumerate(st["merges"])}
        return tok

    @classmethod
    def load(cls, path):
        return cls.from_state(json.load(open(path, encoding="utf-8")))


if __name__ == "__main__":
    from pathlib import Path
    corpus = (Path("data/qa.txt").read_text(encoding="utf-8") + "\n\n"
              + Path("data/aizen_phase3b_train.txt").read_text(encoding="utf-8"))
    tok = BPETokenizer()
    tok.train(corpus, vocab_size=2048, min_pair_freq=10)
    tok.save("bpe_tokenizer.json")
    ids = tok.encode("How much is 123 + 456?")
    print("example:", [tok.id_to_token[i] for i in ids])
    print(f"corpus: {len(corpus):,} chars -> {len(tok.encode(corpus)):,} BPE tokens")
