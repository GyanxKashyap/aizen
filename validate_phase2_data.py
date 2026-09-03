"""
Independent validator for the Phase 2 dataset (data/reasoning_train.json +
data/aizen_phase2_train.txt). Recomputes every deterministic answer with its
own logic - it does not trust the generator.

Exits non-zero if ANY check fails.

Run:  python3 validate_phase2_data.py
"""

import json
import re
import sys
from pathlib import Path

failures = []


def fail(msg):
    failures.append(msg)


VOCAB = set("\n !'*+,-.0123456789:=?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz")
MAX_CONTEXT = 192

items = json.loads(Path("data/reasoning_train.json").read_text(encoding="utf-8"))
txt = Path("data/aizen_phase2_train.txt").read_text(encoding="utf-8")
eval_items = json.loads(Path("data/eval.json").read_text(encoding="utf-8"))

INSTRUCTION_CATS = {"fmt_one_word", "fmt_yes_no", "fmt_number_only", "fmt_three_items",
                    "fmt_two_examples", "fmt_three_words", "fmt_no_explanation",
                    "transform_case", "transform_reverse", "transform_count",
                    "transform_sort", "transform_extreme", "classify", "extract", "simple_qa"}
REASONING_CATS = {"two_step", "three_step", "mixed_ops", "comparison", "sorting",
                  "counting", "deduction", "patterns_r", "reading_comp", "multi_hop"}

# ---- counts
if len(items) != 10000:
    fail(f"expected 10000 examples, got {len(items)}")
n_instr = sum(1 for i in items if i["category"] in INSTRUCTION_CATS)
n_reas = sum(1 for i in items if i["category"] in REASONING_CATS)
if n_instr != 5000:
    fail(f"expected 5000 instruction examples, got {n_instr}")
if n_reas != 5000:
    fail(f"expected 5000 reasoning examples, got {n_reas}")
unknown = [i["id"] for i in items if i["category"] not in INSTRUCTION_CATS | REASONING_CATS]
if unknown:
    fail(f"unknown categories on: {unknown[:5]}")

# ---- ids, duplicates, empties
ids = [i["id"] for i in items]
if len(set(ids)) != len(ids):
    fail("duplicate ids found")
qs = [i["question"] for i in items]
if len(set(qs)) != len(qs):
    fail("duplicate questions found")
for i in items:
    if not i["question"].strip():
        fail(f"{i['id']}: empty question")
    if not str(i["answer"]).strip():
        fail(f"{i['id']}: empty answer")

# ---- vocabulary + context length (on the ACTUAL rendered text)
bad_chars = set(txt) - VOCAB
if bad_chars:
    fail(f"rendered txt contains out-of-vocab chars: {bad_chars}")
blocks = [b for b in txt.split("\n\n") if b.strip()]
if len(blocks) != 10000:
    fail(f"rendered txt has {len(blocks)} blocks, expected 10000")
too_long = [b[:40] for b in blocks if len(b) > MAX_CONTEXT]
if too_long:
    fail(f"{len(too_long)} rendered examples exceed {MAX_CONTEXT} chars, e.g. {too_long[0]!r}")

# ---- eval-set leakage
def masked(s):
    return re.sub(r"\d+", "N", s.lower().strip())


def numtuple(s):
    return tuple(int(n) for n in re.findall(r"\d+", s))


eval_exact = {e["question"] for e in eval_items}
whitelist = {masked(e["question"]) for e in eval_items
             if e["category"] in ("arithmetic", "multi_step_arithmetic")}
eval_masked_blocked = {masked(e["question"]) for e in eval_items} - whitelist
eval_tuples = {numtuple(e["question"]) for e in eval_items if numtuple(e["question"])}
for i in items:
    q = i["question"]
    if q in eval_exact:
        fail(f"{i['id']}: EXACT eval collision: {q}")
    if masked(q) in eval_masked_blocked:
        fail(f"{i['id']}: masked eval template collision: {q}")
    if numtuple(q) and numtuple(q) in eval_tuples:
        fail(f"{i['id']}: eval operand-tuple collision: {q}")

# ---- reasoning-chain arithmetic: every equation in every reasoning must be true
EQ = re.compile(r"(\d+)\s*([+\-*])\s*(\d+)\s*=\s*(\d+)")
for i in items:
    for a, op, b, c in EQ.findall(i["reasoning"] + " " + str(i["answer"])):
        a, b, c = int(a), int(b), int(c)
        real = a + b if op == "+" else (a - b if op == "-" else a * b)
        if real != c:
            fail(f"{i['id']}: false equation {a} {op} {b} = {c} (real {real})")

# arithmetic-chain categories: final answer must equal last equation's result
for i in items:
    if i["category"] in ("two_step", "three_step", "mixed_ops"):
        eqs = EQ.findall(i["reasoning"])
        if not eqs:
            fail(f"{i['id']}: no equations in reasoning")
        elif str(i["answer"]) != eqs[-1][3]:
            fail(f"{i['id']}: answer {i['answer']} != last equation result {eqs[-1][3]}")

# ---- category-specific recomputation
for i in items:
    cat, q, ans = i["category"], i["question"], str(i["answer"])
    if cat == "comparison":
        nums = [int(n) for n in re.findall(r"\d+", q)]
        if len(nums) != 2:
            fail(f"{i['id']}: comparison needs 2 numbers")
            continue
        want = max(nums) if ("greater" in q or "bigger" in q) else min(nums)
        if ans != str(want):
            fail(f"{i['id']}: comparison wrong: {q} -> {ans}, want {want}")
    elif cat in ("sorting", "transform_sort"):
        m = re.search(r": (.+?)[.?]?$", q)
        parts = [p.strip() for p in m.group(1).split(",")] if m else []
        if parts and all(re.fullmatch(r"\d+", p) for p in parts):
            nums = [int(p) for p in parts]
            desc = "big to small" in q
            want = ", ".join(map(str, sorted(nums, reverse=desc)))
        elif parts:
            want = ", ".join(sorted(parts))
        else:
            fail(f"{i['id']}: cannot parse sort list: {q}")
            continue
        if ans != want:
            fail(f"{i['id']}: sort wrong: {q} -> {ans}, want {want}")
    elif cat == "transform_reverse":
        m = re.search(r"(?:word|Spell) (\w+)", q)
        if not m or ans != m.group(1)[::-1]:
            fail(f"{i['id']}: reverse wrong: {q} -> {ans}")
    elif cat == "transform_case":
        m = re.search(r"(?:word (\w+) in (upper|lower)case|into (upper|lower)case: (\w+)|this (upper|lower)case: (\w+))", q)
        if not m:
            fail(f"{i['id']}: cannot parse case task: {q}")
            continue
        w = m.group(1) or m.group(4) or m.group(6)
        mode = m.group(2) or m.group(3) or m.group(5)
        want = w.upper() if mode == "upper" else w.lower()
        if ans != want:
            fail(f"{i['id']}: case wrong: {q} -> {ans}, want {want}")
    elif cat == "transform_extreme":
        m = re.search(r": (.+?)\?$", q)
        ws = [w.strip() for w in m.group(1).split(",")] if m else []
        want = max(ws, key=len) if "longest" in q else min(ws, key=len)
        if ans != want:
            fail(f"{i['id']}: extreme wrong: {q} -> {ans}, want {want}")
    elif cat in ("transform_count", "fmt_number_only", "counting"):
        m = re.search(r"letters? (?:are in|does|in) the word (\w+)", q)
        if m and "letter " not in q:
            if ans != str(len(m.group(1))):
                fail(f"{i['id']}: letter count wrong: {q} -> {ans}")
            continue
        m = re.search(r"letter (\w) appear in the word (\w+)", q)
        if m:
            if ans != str(m.group(2).count(m.group(1))):
                fail(f"{i['id']}: letter occurrence wrong: {q} -> {ans}")
            continue
        m = re.search(r"vowels are in the word (\w+)", q)
        if m:
            if ans != str(sum(1 for ch in m.group(1) if ch in "aeiou")):
                fail(f"{i['id']}: vowel count wrong: {q} -> {ans}")
            continue
        m = re.search(r"(?:this list|animals in this list): (.+?)[.?]$", q)
        if m:
            if ans != str(len(m.group(1).split(","))):
                fail(f"{i['id']}: list count wrong: {q} -> {ans}")
            continue
        m = re.search(r"words are here: (.+?)\?", q) or re.search(r"words here: (.+)$", q)
        if m:
            if ans != str(len(m.group(1).strip().rstrip("?").split())):
                fail(f"{i['id']}: word count wrong: {q} -> {ans}")
            continue
        m = re.search(r"How much is (\d+) plus (\d+)", q)
        if m:
            if ans != str(int(m.group(1)) + int(m.group(2))):
                fail(f"{i['id']}: plus wrong: {q} -> {ans}")
            continue
    elif cat == "patterns_r":
        nums = [int(n) for n in re.findall(r"\d+", q)]
        if len(nums) >= 4:
            seq = nums[-4:] if "times the one before" not in q else nums[1:5]
            d1 = seq[1] - seq[0]
            if all(seq[k + 1] - seq[k] == d1 for k in range(3)):
                if ans != str(seq[-1] + d1):
                    fail(f"{i['id']}: arith pattern wrong: {q} -> {ans}")
            elif seq[0] != 0 and seq[1] % seq[0] == 0:
                rr = seq[1] // seq[0]
                if all(seq[k] * rr == seq[k + 1] for k in range(3)):
                    if ans != str(seq[-1] * rr):
                        fail(f"{i['id']}: geo pattern wrong: {q} -> {ans}")
    elif cat == "fmt_yes_no":
        m = re.search(r"Is (\d+) (bigger|smaller) than (\d+)", q)
        if m:
            a, kind, b = int(m.group(1)), m.group(2), int(m.group(3))
            want = "YES" if ((a > b) == (kind == "bigger")) else "NO"
            if ans != want:
                fail(f"{i['id']}: yes/no comparison wrong: {q} -> {ans}")
    elif cat in ("fmt_three_items",):
        if len(ans.split(",")) != 3:
            fail(f"{i['id']}: three_items answer has {len(ans.split(','))} items")
    elif cat == "fmt_two_examples":
        if len(ans.split(",")) != 2:
            fail(f"{i['id']}: two_examples answer has {len(ans.split(','))} items")
    elif cat == "fmt_three_words":
        if len(ans.split()) != 3:
            fail(f"{i['id']}: three_words answer has {len(ans.split())} words")
    elif cat == "fmt_one_word":
        if len(ans.split()) != 1:
            fail(f"{i['id']}: one_word answer has {len(ans.split())} words")

# ---- report
print(f"examples: {len(items)} ({n_instr} instruction / {n_reas} reasoning)")
print(f"rendered blocks: {len(blocks)}, longest: {max(len(b) for b in blocks)} chars (limit {MAX_CONTEXT})")
print(f"vocab: {'OK' if not bad_chars else 'FAIL'}")
if failures:
    print(f"\nVALIDATION FAILED - {len(failures)} problem(s):")
    for f in failures[:30]:
        print("  -", f)
    if len(failures) > 30:
        print(f"  ... and {len(failures) - 30} more")
    sys.exit(1)
print("\nALL CHECKS PASSED")
