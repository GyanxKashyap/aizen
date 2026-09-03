"""
Tokenizer test suite (Phase 4 spec section 4).
Run:  python3 test_tokenizer.py   -> exits non-zero on any failure.
"""

import sys
from tokenizer import BPETokenizer, UNK, UNK_PLACEHOLDER

tok = BPETokenizer.load("bpe_tokenizer.json")
failures = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        failures.append(name)


def roundtrip(name, text):
    out = tok.decode(tok.encode(text))
    check(name, out == text, "" if out == text else f"{text!r} -> {out!r}")


# 1-2. round trip on normal English
roundtrip("english sentence", "The quick brown fox jumps over the lazy dog.")
roundtrip("training-style QA", "Q: What is the capital of France?\nA: The capital of France is Paris.")
# 3. numbers
roundtrip("numbers", "1 10 100 1000 12345 0 007")
# 4. arithmetic expressions
roundtrip("arithmetic", "What is 123 + 456 - 78 * 9 = ?")
# 5. punctuation (vocab-supported set)
roundtrip("punctuation", "Wait, really? Yes! It's fine: 1+1=2, ok.")
# 6. case
roundtrip("case", "MiXeD CaSe WORDS and lowercase")
# 7. whitespace
roundtrip("whitespace", "a  b   c\nnew line\n\ndouble newline  end ")
# 8. unseen words (novel but made of known chars)
roundtrip("unseen words", "flibbertigibbet zyzzyva qwordly unbelievableness")
# 9. unusual ASCII (never in corpus) round-trips exactly - full printable
# ASCII is in the base vocab, so no <unk> and no crash
roundtrip("unusual ascii chars", "hello (world) [test] {x} 100% #tag @user \"quote\"")
# 10. non-ASCII unicode -> <unk> placeholder, safe, no crash
ids = tok.encode("caf\u00e9 \u00fcber \U0001f600")
check("unicode encodes safely", len(ids) > 0 and UNK in ids)
check("unicode decodes to placeholder", UNK_PLACEHOLDER in tok.decode(ids))

# digits invariant: every digit is its own token
for text in ["123", "45 + 67", "9087"]:
    toks = [tok.id_to_token[i] for i in tok.encode(text)]
    ok = all(sum(c.isdigit() for c in t) <= 1 for t in toks)
    check(f"digits stay single in {text!r}", ok, str(toks))

# newline is never inside a merged token (it is the stop signal)
check("newline standalone", all("\n" not in t or t == "\n" for t in tok.token_to_id))

# specials at fixed deterministic ids
check("special ids", [tok.token_to_id[s] for s in ["<pad>", "<unk>", "<bos>", "<eos>"]] == [0, 1, 2, 3])

# determinism: reload -> identical encoding
tok2 = BPETokenizer.load("bpe_tokenizer.json")
s = "Determinism check 42 + 17, please."
check("save/load determinism", tok.encode(s) == tok2.encode(s))

print(f"\n{len(failures)} failures" if failures else "\nALL TOKENIZER TESTS PASSED")
sys.exit(1 if failures else 0)
