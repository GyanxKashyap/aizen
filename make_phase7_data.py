"""
Phase 6c+7 supplement generator - fixes from Gyan's REAL chat failures, plus
first-ever multi-turn conversation data.

6c fixes (single-turn):
  mixed_digit   (1200): unspaced + mixed-size arithmetic ("555-99", "555+45")
                        - the digit-drop bug
  negatives      (800): results below zero ("2 + 4 - 9 = -3") - Aizen has
                        never seen a negative number
  false_premise  (700): "if aizen is a cat can aizen bark?" -> the model must
                        SUPPLY the true fact ("Cats cannot bark") instead of
                        inventing a false premise
  unspaced_multi (500): "2+4-9" style chains

Phase 7 (multi-turn blocks - each block is ONE training example containing
several Q/A turns; the fine-tuner masks every answer segment):
  convo_math     (900): follow-ups referencing the previous answer
                        ("and plus 3?" -> continues from prior result)
  convo_facts    (600): "and Japan?" style elliptical follow-ups
  convo_correct  (500): user makes a statement/correction -> polite, factual
                        acknowledgment ("cats can't bark" -> "You are right...")
  convo_chat     (500): statements that aren't questions ("i like dogs",
                        "that was wrong", "nice") -> sensible replies

Same guards as always: eval exact/masked/tuple collisions blocked, no dupes,
<=400 BPE tokens per block. Output: data/phase7_extra.txt (+ .json), combined
into data/aizen_phase7_train.txt on top of the phase6b pool.

Run:  python3 make_phase7_data.py
"""

import json
import random
import re
from pathlib import Path

from tokenizer import BPETokenizer

rng = random.Random(70830)
tok = BPETokenizer.load("bpe_tokenizer_v2.json")
MAX_TOKENS = 400

qa_text = Path("data/qa.txt").read_text(encoding="utf-8")
qa_masked = {re.sub(r"\d+", "N", q.lower()) for q in re.findall(r"^Q: (.+)$", qa_text, flags=re.M)}
eval_items = json.loads(Path("data/eval.json").read_text(encoding="utf-8"))
prev_questions = set()
for block in Path("data/aizen_phase6b_train.txt").read_text(encoding="utf-8").split("\n\n"):
    if block.startswith("Q: "):
        prev_questions.add(block.split("\nA:")[0][3:])


def masked(s):
    return re.sub(r"\d+", "N", s.lower().strip())


def numtuple(s):
    return tuple(int(n) for n in re.findall(r"-?\d+", s))


eval_exact = {it["question"] for it in eval_items}
WHITELIST = {masked(it["question"]) for it in eval_items
             if it["category"] in ("arithmetic", "multi_step_arithmetic")}
eval_masked_blocked = {masked(it["question"]) for it in eval_items} - WHITELIST
eval_tuples = {numtuple(it["question"]) for it in eval_items if numtuple(it["question"])}

items, used = [], set()
rej = {"eval": 0, "qa": 0, "dup": 0, "long": 0}


def guard_q(q):
    """Collision guards for one question string."""
    if q in used or q in prev_questions:
        rej["dup"] += 1
        return False
    m = masked(q)
    if q in eval_exact or m in eval_masked_blocked or \
       (numtuple(q) and numtuple(q) in eval_tuples):
        rej["eval"] += 1
        return False
    if q in qa_text or m in qa_masked:
        rej["qa"] += 1
        return False
    return True


def add_single(cat, q, r, a):
    q, r, a = " ".join(q.split()), " ".join(r.split()), str(a).strip()
    text = f"Q: {q}\nA: {r} The answer is {a}." if r else f"Q: {q}\nA: {a}"
    if len(tok.encode(text)) > MAX_TOKENS:
        rej["long"] += 1
        return False
    if not guard_q(q):
        return False
    used.add(q)
    items.append({"id": f"{cat}_{sum(1 for i in items if i['category'] == cat) + 1:04d}",
                  "category": cat, "block": text})
    return True


def add_convo(cat, turns):
    """turns = [(q, a), ...] -> one multi-turn block."""
    lines = []
    for q, a in turns:
        lines.append(f"Q: {' '.join(q.split())}")
        lines.append(f"A: {' '.join(a.split())}")
    text = "\n".join(lines)
    if len(tok.encode(text)) > MAX_TOKENS:
        rej["long"] += 1
        return False
    key = " || ".join(q for q, _ in turns)
    if not guard_q(key):
        return False
    used.add(key)
    items.append({"id": f"{cat}_{sum(1 for i in items if i['category'] == cat) + 1:04d}",
                  "category": cat, "block": text})
    return True


def fill(cat, target, fn):
    attempts = 0
    while sum(1 for i in items if i["category"] == cat) < target:
        attempts += 1
        assert attempts < 500000, f"{cat}: exhausted"
        c = fn()
        if c is None:
            continue
        (add_single if len(c) == 4 else add_convo)(*c)
    print(f"  {cat:14s} {target} done")


# ============================================ 6c: mixed_digit (1200)
def cand_mixed_digit():
    a = rng.choice([rng.randint(100, 999), rng.randint(10, 99)])
    b = rng.choice([rng.randint(100, 999), rng.randint(10, 99), rng.randint(1, 9)])
    op = rng.choice(["+", "-"])
    if op == "-" and b > a:
        a, b = b, a
    res = a + b if op == "+" else a - b
    q = rng.choice([f"{a}{op}{b}", f"{a} {op} {b}", f"What is {a} {op} {b}?",
                    f"How much is {a} {op} {b}?"])
    r = f"{a} {op} {b} = {res}."
    return "mixed_digit", q, r, res


# ============================================ 6c: negatives (800)
def cand_negative():
    if rng.random() < 0.5:  # single-op negative
        a, b = rng.randint(1, 50), rng.randint(1, 99)
        if b <= a:
            return None
        res = a - b
        q = rng.choice([f"What is {a} - {b}?", f"{a}-{b}", f"How much is {a} minus {b}?"])
        r = f"{b} is bigger than {a}, so the result is below zero. {a} - {b} = {res}."
        return "negatives", q, r, res
    a, b = rng.randint(1, 20), rng.randint(1, 20)  # chain ending negative
    c = rng.randint(a + b + 1, a + b + 40)
    s1, res = a + b, a + b - c
    q = rng.choice([f"What is {a} + {b} - {c}?", f"{a}+{b}-{c}"])
    r = f"First, {a} + {b} = {s1}. Then, {s1} - {c} = {res}."
    return "negatives", q, r, res


# ============================================ 6c: false_premise (700)
ABILITIES = [
    # (animal, can-do, cannot-do)
    ("cat", "meow", "bark"), ("dog", "bark", "meow"), ("fish", "swim", "walk"),
    ("bird", "fly", "swim like a fish"), ("cow", "moo", "fly"),
    ("duck", "swim", "bark"), ("frog", "jump", "sing songs"),
    ("horse", "run", "climb trees"), ("hen", "lay eggs", "fly high"),
    ("snake", "crawl", "walk"),
]
PETNAMES7 = ["Renji", "Aizen", "Miko", "Luna", "Toby", "Coco", "Nala", "Ruby",
             "Bobo", "Kira", "Momo", "Ichi"]


def cand_false_premise():
    animal, can, cannot = rng.choice(ABILITIES)
    pn = rng.choice(PETNAMES7)
    who = rng.choice([pn, f"a {animal}"])
    pron = rng.choice(["it", "she", "he"])
    if rng.random() < 0.5:  # ask about the thing it CANNOT do -> no
        q = rng.choice([
            f"If {pn} is a {animal} then can {pn} {cannot.split()[0] if ' ' in cannot else cannot}?",
            f"if {pn.lower()} is {animal} can {pron} {cannot}?",
            f"Can a {animal} {cannot}?",
            f"{pn} is a {animal}. Can {pn} {cannot}?"])
        r = f"A {animal} cannot {cannot}. A {animal} can {can}."
        return "false_premise", q, r, "no"
    q = rng.choice([  # the thing it CAN do -> yes
        f"If {pn} is a {animal} then can {pn} {can}?",
        f"if {pn.lower()} is {animal} can {pron} {can}?",
        f"{pn} is a {animal}. Can {pn} {can}?"])
    r = f"A {animal} can {can}."
    return "false_premise", q, r, "yes"


# ============================================ 6c: unspaced_multi (500)
def cand_unspaced_multi():
    a, b = rng.randint(1, 60), rng.randint(1, 60)
    c = rng.randint(1, 60)
    op2 = rng.choice(["+", "-"])
    s1 = a + b
    res = s1 + c if op2 == "+" else s1 - c
    q = f"{a}+{b}{op2}{c}"
    r = f"First, {a} + {b} = {s1}. Then, {s1} {op2} {c} = {res}."
    return "unspaced_multi", q, r, res


# ============================================ 7: convo_math (900)
def cand_convo_math():
    a, b = rng.randint(2, 60), rng.randint(2, 60)
    s = a + b
    turns = [(rng.choice([f"What is {a} + {b}?", f"{a}+{b}"]), f"{a} + {b} = {s}")]
    cur = s
    for _ in range(rng.choice([1, 2])):
        d = rng.randint(2, 30)
        op = rng.choice(["+", "-"])
        nxt = cur + d if op == "+" else cur - d
        fq = rng.choice([f"and {'plus' if op == '+' else 'minus'} {d}?",
                         f"now {'add' if op == '+' else 'take away'} {d}",
                         f"{'plus' if op == '+' else 'minus'} {d}?"])
        turns.append((fq, f"{cur} {op} {d} = {nxt}"))
        cur = nxt
    return "convo_math", turns


# ============================================ 7: convo_facts (600)
CAPS7 = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
         ("Germany", "Berlin"), ("Spain", "Madrid"), ("Egypt", "Cairo"),
         ("Norway", "Oslo"), ("Greece", "Athens"), ("Canada", "Ottawa"),
         ("Brazil", "Brasilia"), ("China", "Beijing"), ("Russia", "Moscow"),
         ("Turkey", "Ankara"), ("Poland", "Warsaw"), ("Iran", "Tehran"),
         ("Cuba", "Havana"), ("Peru", "Lima"), ("Chile", "Santiago")]
COLORFACTS = [("the sky", "blue"), ("grass", "green"), ("snow", "white"),
              ("a banana", "yellow"), ("blood", "red"), ("coal", "black"),
              ("milk", "white"), ("a carrot", "orange")]


def cand_convo_facts():
    if rng.random() < 0.6:
        picks = rng.sample(CAPS7, rng.choice([2, 3]))
        turns = [(f"What is the capital of {picks[0][0]}?",
                  f"The capital of {picks[0][0]} is {picks[0][1]}.")]
        for c, cap in picks[1:]:
            fq = rng.choice([f"and {c}?", f"what about {c}?", f"and of {c}?"])
            turns.append((fq, f"The capital of {c} is {cap}."))
        return "convo_facts", turns
    picks = rng.sample(COLORFACTS, rng.choice([2, 3]))
    turns = [(f"What color is {picks[0][0]}?", f"{picks[0][0][0].upper()}{picks[0][0][1:]} is {picks[0][1]}.")]
    for thing, col in picks[1:]:
        fq = rng.choice([f"and {thing}?", f"what about {thing}?"])
        turns.append((fq, f"{thing[0].upper()}{thing[1:]} is {col}."))
    return "convo_facts", turns


# ============================================ 7: convo_correct (500)
def cand_convo_correct():
    animal, can, cannot = rng.choice(ABILITIES)
    style = rng.randrange(3)
    if style == 0:  # user corrects a wrong claim
        turns = [(f"can a {animal} {cannot}?", f"No. A {animal} cannot {cannot}. A {animal} can {can}."),
                 (rng.choice([f"{animal}s can't {cannot}", f"right, a {animal} cannot {cannot}"]),
                  rng.choice([f"You are right. A {animal} cannot {cannot}, it can {can}.",
                              f"Yes, that is true. A {animal} can {can}, not {cannot}."]))]
        return "convo_correct", turns
    if style == 1:  # user asserts a true fact
        turns = [(rng.choice([f"a {animal} can {can}", f"{animal}s {can}"]),
                  rng.choice([f"That is right. A {animal} can {can}.",
                              f"Yes! A {animal} can {can}."]))]
        return "convo_correct", turns
    a, b = rng.randint(2, 40), rng.randint(2, 40)  # user corrects math
    s = a + b
    turns = [(f"{a}+{b}", f"{a} + {b} = {s}"),
             (rng.choice(["that is wrong", "are you sure?", "check again"]),
              rng.choice([f"Let me check. {a} + {b} = {s}. I am sure.",
                          f"Checking again: {a} + {b} = {s}. That is my answer."]))]
    return "convo_correct", turns


# ============================================ 7: convo_chat (500)
CHAT_PAIRS = [
    ("i like dogs", ["Nice! Dogs are great friends.", "Dogs are lovely. They can bark and play."]),
    ("i like cats", ["Nice! Cats are calm and soft.", "Cats are lovely. They purr when happy."]),
    ("i am tired", ["You should rest. Sleep helps a lot.", "Take a break and rest well."]),
    ("i am happy", ["That is great to hear!", "Wonderful! I am glad."]),
    ("i am sad", ["I am sorry to hear that. I hope your day gets better.", "That is hard. Better days will come."]),
    ("nice", ["Thanks! Ask me anything else.", "Glad you liked it!"]),
    ("good job", ["Thank you! Happy to help.", "Thanks! Ask me more."]),
    ("that was fast", ["Thanks! I try my best.", "I am small but quick."]),
    ("you are smart", ["Thanks! I am a tiny model, but I try.", "Thank you! I only know what I was taught."]),
    ("you are wrong a lot", ["That is fair. I am a very small model and I make mistakes.", "True. I am tiny, so I get things wrong sometimes."]),
    ("tell me something", ["Bees make honey and a spider has eight legs.", "The sun is a star and the moon orbits Earth."]),
    ("i am bored", ["Ask me a math question or a riddle!", "Want to test me with a question?"]),
    ("ok", ["Okay! Anything else?", "Alright. Ask away."]),
    ("hmm", ["Take your time. Ask me anything.", "I am here when you are ready."]),
    ("i love pizza", ["Pizza is tasty! I only eat data.", "Nice choice. Food sounds fun."]),
    ("i am learning to code", ["That is great! Coding is a superpower.", "Nice! Keep practicing every day."]),
    ("do you like me", ["Of course! You ask good questions.", "Yes! Talking with you is what I do."]),
    ("what else can you do", ["I can do simple math, answer facts, and chat a little.", "Math, facts, and small talk. I am tiny but I try."]),
]


def cand_convo_chat():
    stmt, replies = rng.choice(CHAT_PAIRS)
    turns = [(stmt, rng.choice(replies))]
    if rng.random() < 0.4:  # sometimes follow with a task turn to anchor
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        turns.append((f"ok what is {a} + {b}?", f"{a} + {b} = {a + b}"))
    return "convo_chat", turns


PLAN = [("mixed_digit", 1200, cand_mixed_digit),
        ("negatives", 800, cand_negative),
        ("false_premise", 700, cand_false_premise),
        ("unspaced_multi", 500, cand_unspaced_multi),
        ("convo_math", 900, cand_convo_math),
        ("convo_facts", 600, cand_convo_facts),
        ("convo_correct", 500, cand_convo_correct),
        ("convo_chat", 500, cand_convo_chat)]
for cat, n, fn in PLAN:
    fill(cat, n, fn)

# self-verify every equation (including negative results)
EQ = re.compile(r"(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)")
bad = 0
for i in items:
    for a, op, b, c in EQ.findall(i["block"]):
        real = int(a) + int(b) if op == "+" else int(a) - int(b)
        if real != int(c):
            bad += 1
            print("FALSE EQUATION:", i["block"][:80])
assert bad == 0, f"{bad} false equations"

rng.shuffle(items)
Path("data/phase7_extra.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
Path("data/phase7_extra.txt").write_text("\n\n".join(i["block"] for i in items) + "\n", encoding="utf-8")
prev = Path("data/aizen_phase6b_train.txt").read_text(encoding="utf-8").rstrip("\n")
new = Path("data/phase7_extra.txt").read_text(encoding="utf-8").rstrip("\n")
Path("data/aizen_phase7_train.txt").write_text(prev + "\n\n" + new + "\n", encoding="utf-8")
n = len([b for b in Path("data/aizen_phase7_train.txt").read_text(encoding="utf-8").split("\n\n") if b.strip()])
print(f"\nphase7 supplement: {len(items)} blocks | combined pool: {n} blocks")
print("rejections:", rej)
