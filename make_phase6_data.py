"""
Phase 6 HYBRID dataset generator.

Two sources, one file:

A. SYNTHETIC (Python-computed, guaranteed correct) - targets v3's measured
   weaknesses:
   - arith3       (1000): 3-digit add/sub (tokenizer supports it; data never taught it)
   - patterns_v3  (1500): sequences in many wordings + reasoning lines (10% regression)
   - multistep_v3 (1500): heavy a*b+c / a*b-c (the second-step flip bug) + 3-digit chains
   - instr_v3     (1000): more instruction formats + contrast pairs

B. bAbI (Facebook AI, via the Muennighoff/babi mirror) - real curated
   reading/reasoning data with phrasing diversity we cannot fake:
   tasks 1 (single fact), 2 (two facts), 6 (yes/no), 7 (counting),
   9 (negation), 11 (coreference), 12 (conjunction), 13 (compound coref).
   ~800 per task = ~6,400 examples, converted to our Q/A format with the
   supporting story inline.

Guards (same integrity rules as every phase): eval.json exact/masked-template/
operand-tuple collisions rejected; duplicates rejected; everything encodable
by tokenizer v2; rendered examples <= 400 BPE tokens (fits the 512 window).

Output: data/phase6_extra.txt (+ .json), then combine with the Phase 3b pool
into data/aizen_phase6_train.txt.

Run:  python3 make_phase6_data.py
"""

import json
import random
import re
from pathlib import Path

from tokenizer import BPETokenizer

rng = random.Random(60829)
tok = BPETokenizer.load("bpe_tokenizer_v2.json")

MAX_TOKENS = 400

qa_text = Path("data/qa.txt").read_text(encoding="utf-8")
qa_masked = {re.sub(r"\d+", "N", q.lower()) for q in re.findall(r"^Q: (.+)$", qa_text, flags=re.M)}
eval_items = json.loads(Path("data/eval.json").read_text(encoding="utf-8"))
prev_questions = set()
for block in Path("data/aizen_phase3b_train.txt").read_text(encoding="utf-8").split("\n\n"):
    if block.startswith("Q: "):
        prev_questions.add(block.split("\nA:")[0][3:])


def masked(s):
    return re.sub(r"\d+", "N", s.lower().strip())


def numtuple(s):
    return tuple(int(n) for n in re.findall(r"\d+", s))


eval_exact = {it["question"] for it in eval_items}
WHITELIST = {masked(it["question"]) for it in eval_items
             if it["category"] in ("arithmetic", "multi_step_arithmetic")}
eval_masked_blocked = {masked(it["question"]) for it in eval_items} - WHITELIST
eval_tuples = {numtuple(it["question"]) for it in eval_items if numtuple(it["question"])}

items, used = [], set()
rej = {"eval": 0, "qa": 0, "dup": 0, "long": 0, "nonascii": 0}


def render(q, r, a):
    return f"Q: {q}\nA: {r} The answer is {a}." if r else f"Q: {q}\nA: {a}"


def add(cat, q, r, a):
    q, r, a = " ".join(q.split()), " ".join(r.split()), str(a).strip()
    if not q or not a:
        return False
    text = render(q, r, a)
    if any(ord(c) > 126 for c in text):
        rej["nonascii"] += 1
        return False
    if len(tok.encode(text)) > MAX_TOKENS:
        rej["long"] += 1
        return False
    if q in used or q in prev_questions:
        rej["dup"] += 1
        return False
    m = masked(q)
    if q in eval_exact or m in eval_masked_blocked or \
       (numtuple(q) and m not in WHITELIST and numtuple(q) in eval_tuples) or \
       (m in WHITELIST and numtuple(q) in eval_tuples):
        rej["eval"] += 1
        return False
    if q in qa_text or m in qa_masked:
        rej["qa"] += 1
        return False
    used.add(q)
    items.append({"id": f"{cat}_{sum(1 for i in items if i['category'] == cat) + 1:04d}",
                  "category": cat, "question": q, "reasoning": r, "answer": a})
    return True


def fill(cat, target, fn):
    attempts = 0
    while sum(1 for i in items if i["category"] == cat) < target:
        attempts += 1
        assert attempts < 500000, f"{cat}: exhausted"
        c = fn()
        if c:
            add(cat, *c)
    print(f"  {cat:14s} {target} done")


# ================================================== A. SYNTHETIC
NAMES = ["Raj", "Kim", "Eva", "Zoe", "Omar", "Lily", "Noah", "Ravi", "Nina",
         "Carl", "Aiko", "Yuki", "Gyan", "Mira", "Tara", "Vik", "Asha", "Chen",
         "Devi", "Emil", "Hana", "Iris", "Kofi", "Lena", "Nils", "Pia", "Uma", "Zane"]
OBJECTS = ["mangoes", "marbles", "stamps", "shells", "kites", "beads", "seeds", "notes"]


def cand_arith3():
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    if rng.random() < 0.5:
        res = a + b
        q = rng.choice([f"What is {a} + {b}?", f"How much is {a} plus {b}?",
                        f"Add {a} and {b}.", f"{a}+{b}"])
        r = f"{a} + {b} = {res}."
    else:
        if b > a:
            a, b = b, a
        res = a - b
        q = rng.choice([f"What is {a} - {b}?", f"How much is {a} minus {b}?",
                        f"Take {b} away from {a}.", f"{a}-{b}"])
        r = f"{a} - {b} = {res}."
    return q, r, res


def cand_patterns3():
    style = rng.randrange(5)
    if style == 0:
        start, step = rng.randint(1, 40), rng.randint(2, 12)
        seq = [start + step * k for k in range(4)]
        nxt = seq[-1] + step
        q = rng.choice([f"Find the next number: {', '.join(map(str, seq))}.",
                        f"The sequence is {', '.join(map(str, seq))}. What is next?",
                        f"Look at this pattern: {', '.join(map(str, seq))}. What number follows?",
                        f"{', '.join(map(str, seq))}. Which number comes then?"])
        r = f"The numbers go up by {step} each time. {seq[-1]} + {step} = {nxt}."
        return q, r, nxt
    if style == 1:
        step = rng.randint(2, 12)
        start = rng.randint(4 * step + 5, 120)
        seq = [start - step * k for k in range(4)]
        nxt = seq[-1] - step
        q = rng.choice([f"What follows: {', '.join(map(str, seq))}?",
                        f"Find the next number: {', '.join(map(str, seq))}.",
                        f"The numbers are {', '.join(map(str, seq))}. What comes then?"])
        r = f"The numbers go down by {step} each time. {seq[-1]} - {step} = {nxt}."
        return q, r, nxt
    if style == 2:
        start, mult = rng.randint(1, 12), rng.choice([2, 3])
        seq = [start * (mult ** k) for k in range(4)]
        nxt = seq[-1] * mult
        q = rng.choice([f"Find the next number: {', '.join(map(str, seq))}.",
                        f"Each number is {mult} times the one before: {', '.join(map(str, seq))}. What comes then?"])
        r = f"Each number is {mult} times the one before. {seq[-1]} * {mult} = {nxt}."
        return q, r, nxt
    if style == 3:  # letters
        letters = "abcdefghijklmnopqrstuvwxyz"
        s = rng.randint(0, 21)
        seq = letters[s:s + 4]
        q = rng.choice([f"Find the next letter: {', '.join(seq)}.",
                        f"The letters are {', '.join(seq)}. What letter follows?"])
        r = f"The letters follow the alphabet. After {seq[-1]} comes {letters[s + 4]}."
        return q, r, letters[s + 4]
    # repeating pattern
    a, b = rng.sample(["red", "blue", "sun", "moon", "cat", "dog", "one", "two"], 2)
    reps = rng.choice([2, 3])
    seq = ([a, b] * 3)[:2 * reps + 1]
    nxt = b if seq[-1] == a else a
    q = f"The pattern is {', '.join(seq)}. What comes next?"
    r = f"The pattern repeats {a}, {b}. After {seq[-1]} comes {nxt}."
    return q, r, nxt


def cand_multistep3():
    style = rng.randrange(4)
    if style == 0:  # a*b+c  (the flip bug)
        a, b, c = rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 60)
        s1, res = a * b, a * b + c
        q = rng.choice([f"What is {a} * {b} + {c}?",
                        f"Multiply {a} by {b}, then add {c}.",
                        f"{rng.choice(NAMES)} has {a} bags with {b} {rng.choice(OBJECTS)} each, plus {c} more. How many in all?"])
        r = f"First, {a} * {b} = {s1}. Then, {s1} + {c} = {res}."
        return q, r, res
    if style == 1:  # a*b-c
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        s1 = a * b
        c = rng.randint(1, s1 - 1) if s1 > 1 else 1
        res = s1 - c
        q = rng.choice([f"What is {a} * {b} - {c}?", f"Multiply {a} by {b}, then take away {c}."])
        r = f"First, {a} * {b} = {s1}. Then, {s1} - {c} = {res}."
        return q, r, res
    if style == 2:  # 3-digit two-step
        a, b = rng.randint(100, 600), rng.randint(100, 399)
        c = rng.randint(10, 99)
        s1, res = a + b, a + b - c
        q = f"What is {a} + {b} - {c}?"
        r = f"First, {a} + {b} = {s1}. Then, {s1} - {c} = {res}."
        return q, r, res
    a, b = rng.randint(11, 60), rng.randint(11, 60)  # classic two-step word problem
    c = rng.randint(2, min(a + b - 1, 50))
    s1, res = a + b, a + b - c
    n = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    q = f"{n} has {a} {obj}, gets {b} more, then gives away {c}. How many {obj} does {n} have?"
    r = f"First, {a} + {b} = {s1}. Then, {s1} - {c} = {res}."
    return q, r, res


IFACTS = [("What color is a banana?", "yellow"), ("What do chickens lay?", "eggs"),
          ("What is frozen water called?", "ice"), ("What organ pumps blood?", "heart"),
          ("What season is the coldest?", "winter"), ("What animal has a trunk?", "elephant"),
          ("What is a baby dog called?", "puppy"), ("What do cows give us to drink?", "milk"),
          ("What vehicle runs on rails?", "train"), ("What do we wear on our feet?", "shoes"),
          ("What is the opposite of hot?", "cold"), ("What is the opposite of up?", "down"),
          ("What insect makes webs?", "spider"), ("What place do children go to learn?", "school")]
WORDS6 = ["planet", "candle", "garden", "monkey", "dragon", "castle", "rocket",
          "mirror", "guitar", "helmet", "rabbit", "violin", "anchor", "bridge",
          "engine", "flower", "hammer", "jungle", "magnet", "needle", "parrot",
          "ribbon", "shadow", "temple", "valley", "basket", "donkey", "falcon"]


def cand_instr3():
    style = rng.randrange(4)
    if style == 0:
        body, word = rng.choice(IFACTS)
        tpl = rng.choice(["Give a one word answer: {b}", "Respond with a single word: {b}",
                          "{b} Reply in a single word.", "Just one word: {b}"])
        return tpl.format(b=body), "", word
    if style == 1:
        a, b = rng.randint(1, 999), rng.randint(1, 999)
        if a == b:
            return None
        kind = rng.choice(["bigger", "smaller"])
        ans = "YES" if ((a > b) == (kind == "bigger")) else "NO"
        tpl = rng.choice(["Respond only YES or NO: Is {a} {k} than {b}?",
                          "Is {a} {k} than {b}? Give only YES or NO."])
        return tpl.format(a=a, b=b, k=kind), "", ans
    if style == 2:
        w = rng.choice(WORDS6)
        tpl = rng.choice(["Write only the word {w} and nothing else.",
                          "Reply with just the word {w}.",
                          "Echo this word: {w}"])
        return tpl.format(w=w), "", w
    w = rng.choice(WORDS6)
    tpl = rng.choice(["Respond with a number only: how many letters are in {w}?",
                      "Count the letters in {w}. Give only the number."])
    return tpl.format(w=w), "", len(w)


fill("arith3", 1000, cand_arith3)
fill("patterns_v3", 1500, cand_patterns3)
fill("multistep_v3", 1500, cand_multistep3)
fill("instr_v3", 1000, cand_instr3)

# ================================================== B. bAbI conversion
BABI_TASKS = {1: "babi_onefact", 2: "babi_twofacts", 6: "babi_yesno",
              7: "babi_counting", 9: "babi_negation", 11: "babi_coref",
              12: "babi_conjunction", 13: "babi_compound"}
PER_TASK = 800

by_task = {}
for line in open("data/babi_train.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d["task"] in BABI_TASKS:
        by_task.setdefault(d["task"], []).append(d)

babi_added = 0
for task, cat in BABI_TASKS.items():
    pool = by_task.get(task, [])
    rng.shuffle(pool)
    n = 0
    for d in pool:
        if n >= PER_TASK:
            break
        passage = " ".join(d["passage"].split())
        q = f"{passage} {d['question']}"
        ans = d["answer"].replace(",", ", ")  # multi-answer tasks use commas
        if add(cat, q, "", ans):
            n += 1
            babi_added += 1
    print(f"  {cat:14s} {n} done (task {task})")

# ================================================== write + combine
rng.shuffle(items)
Path("data/phase6_extra.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
Path("data/phase6_extra.txt").write_text(
    "\n\n".join(render(i["question"], i["reasoning"], i["answer"]) for i in items) + "\n", encoding="utf-8")

prev = Path("data/aizen_phase3b_train.txt").read_text(encoding="utf-8").rstrip("\n")
new = Path("data/phase6_extra.txt").read_text(encoding="utf-8").rstrip("\n")
Path("data/aizen_phase6_train.txt").write_text(prev + "\n\n" + new + "\n", encoding="utf-8")

n_total = len([b for b in Path("data/aizen_phase6_train.txt").read_text(encoding="utf-8").split("\n\n") if b.strip()])
print(f"\nphase6 supplement: {len(items)} examples ({babi_added} from bAbI, {len(items)-babi_added} synthetic)")
print(f"combined task pool: {n_total} examples -> data/aizen_phase6_train.txt")
print("rejections:", rej)
