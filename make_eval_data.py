"""
Generate the held-out evaluation dataset for Aizen -> data/eval.json

Design rules (Phase 1):
- Deterministic: fixed seed, same output every run.
- 8 categories x 50 questions = 400 total.
- NOT copied from data/qa.txt. qa.txt is read ONLY to EXCLUDE collisions:
  * arithmetic pairs that appear in training are never used here
  * question strings that appear verbatim in training are rejected
- Every question is checked against Aizen's 73-char vocabulary (questions
  with unseen characters would test the tokenizer, not the model).
  NOTE: uppercase 'X' is not in the vocab - never use it in question text.
- Reading-comprehension passages are hard-capped so the full prompt
  "Q: {q}\nA:" fits in the 192-char context window.

Run:  python3 make_eval_data.py
"""

import json
import random
import re
from pathlib import Path

random.seed(4242)

OUT_PATH = Path("data/eval.json")
QA_PATH = Path("data/qa.txt")
MAX_QUESTION_LEN = 180  # "Q: " + q + "\nA:" must fit in 192-char context

# Aizen's vocabulary (from aizen.pt) - questions must only use these chars
VOCAB = set("\n !'*+,-.0123456789:=?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz")

# --- load training text ONLY for collision exclusion (never for answers) ---
qa_text = QA_PATH.read_text(encoding="utf-8")
trained_pairs = set()
for a, op, b in re.findall(r"(\d+)\s*([+\-*])\s*(\d+)", qa_text):
    trained_pairs.add((int(a), op, int(b)))

items = []
used_questions = set()


def add(category, question, expected, acceptable=None, method="contains"):
    """Returns True if added; False on collision/duplicate (caller may retry).
    Length and vocabulary violations are hard errors - those are authoring bugs."""
    assert len(question) <= MAX_QUESTION_LEN, f"question too long ({len(question)}): {question}"
    bad = set(question) - VOCAB
    assert not bad, f"question uses chars outside Aizen vocab {bad}: {question}"
    if question in qa_text or question in used_questions:
        return False
    used_questions.add(question)
    items.append({
        "id": f"{category}_{sum(1 for i in items if i['category'] == category) + 1:03d}",
        "category": category,
        "question": question,
        "expected_answer": str(expected),
        "acceptable_answers": [str(a) for a in (acceptable or [expected])],
        "method": method,
    })
    return True


def unseen_pair(op, lo_a, hi_a, lo_b, hi_b, constraint=None):
    """Random operand pair guaranteed NOT to appear in training data."""
    while True:
        a, b = random.randint(lo_a, hi_a), random.randint(lo_b, hi_b)
        if constraint and not constraint(a, b):
            continue
        if (a, op, b) not in trained_pairs:
            return a, b


# ------------------------------------------------------------ 1. arithmetic
# In-distribution ranges (add/sub 0-99, mul 0-12) but pairs UNSEEN in training.
arith_templates = [
    ("What is {a} {op} {b}?", None),
    ("{a}{op}{b}", None),           # unspaced form (also trained format-wise)
    ("How much is {a} {op} {b}?", None),
]
# NOTE: multiplication is excluded here - training covered EVERY possible
# 0-12 x 0-12 pair, so no unseen in-distribution mul pair exists. Mul appears
# in multi_step_arithmetic instead (novel compositions).
ai = 0
while sum(1 for it in items if it["category"] == "arithmetic") < 50:
    kind = ["+", "-"][ai % 2]
    if kind == "+":
        a, b = unseen_pair("+", 10, 99, 10, 99)
        ans = a + b
    else:
        a, b = unseen_pair("-", 10, 99, 0, 99, constraint=lambda a, b: a >= b)
        ans = a - b
    tpl, _ = arith_templates[ai % len(arith_templates)]
    if add("arithmetic", tpl.format(a=a, op=kind, b=b), ans, method="number"):
        ai += 1

# ------------------------------------------- 2. multi-step arithmetic (novel)
mi = 0
while sum(1 for it in items if it["category"] == "multi_step_arithmetic") < 50:
    style = mi % 3
    if style == 0:  # a + b + c
        a, b, c = random.randint(1, 30), random.randint(1, 30), random.randint(1, 30)
        q, ans = f"What is {a} + {b} + {c}?", a + b + c
    elif style == 1:  # a + b - c
        a, b = random.randint(10, 50), random.randint(10, 50)
        c = random.randint(1, a + b)
        q, ans = f"What is {a} + {b} - {c}?", a + b - c
    else:  # a * b + c
        a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(1, 20)
        q, ans = f"What is {a} * {b} + {c}?", a * b + c
    if add("multi_step_arithmetic", q, ans, method="number"):
        mi += 1

# ------------------------------------------------------------------ 3. logic
names = ["Tom", "Sam", "Ana", "Ben", "Mia", "Leo", "Amy", "Dan"]
logic_yes_no = [
    ("If all cats are animals and Kiki is a cat, is Kiki an animal?", "yes"),
    ("If all birds can fly and Pipo is a bird, can Pipo fly?", "yes"),
    ("If no fish can walk and Bubbles is a fish, can Bubbles walk?", "no"),
    ("Is every square also a rectangle?", "yes"),
    ("Can a circle have corners?", "no"),
    ("If today is Monday, is tomorrow Wednesday?", "no"),
    ("If today is Friday, is tomorrow Saturday?", "yes"),
    ("Is five bigger than nine?", "no"),
    ("Is twelve bigger than seven?", "yes"),
    ("If it rains, the road gets wet. It rains. Is the road wet?", "yes"),
]
li = 0
for q, ans in logic_yes_no:
    add("logic", q + " Answer yes or no.", ans, method="yes_no")
    li += 1
while li < 50:
    style = li % 4
    n1, n2 = random.sample(names, 2)
    if style == 0:
        q, ans = f"{n1} is taller than {n2}. Who is shorter?", n2
    elif style == 1:
        q, ans = f"{n1} is older than {n2}. Who is younger?", n2
    elif style == 2:
        q, ans = f"{n1} runs faster than {n2}. Who is slower?", n2
    else:
        a, b = random.randint(1, 50), random.randint(51, 99)
        q, ans = f"{n1} has {a} coins. {n2} has {b} coins. Who has more coins?", n2
    if add("logic", q, ans, method="contains"):
        li += 1

# --------------------------------------------------------------- 4. patterns
pi = 0
attempts = 0
while pi < 50:
    style = attempts % 4
    attempts += 1
    if style == 0:  # arithmetic progression
        start, step = random.randint(1, 20), random.randint(2, 9)
        seq = [start + step * k for k in range(4)]
        q, ans = f"What number comes ne-t: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}?", seq[0] + 4 * step
        q = q.replace("ne-t", "after this")  # avoid awkward wording; keep vocab-safe
        method = "number"
    elif style == 1:  # geometric (doubling or tripling)
        start, mult = random.randint(1, 12), random.choice([2, 3])
        seq = [start * (mult ** k) for k in range(4)]
        q, ans = f"Continue the pattern: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?", seq[3] * mult
        method = "number"
    elif style == 2:  # letters
        s = random.randint(0, 20)
        letters = "abcdefghijklmnopqrstuvwxyz"  # lowercase x IS in the vocab
        seq = letters[s:s + 4]
        q, ans = f"What letter comes after this: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}?", letters[s + 4]
        method = "contains"
    else:  # countdown
        start, step = random.randint(50, 99), random.randint(2, 9)
        seq = [start - step * k for k in range(4)]
        q, ans = f"Continue: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]}, ?", seq[0] - 4 * step
        method = "number"
    if add("patterns", q, ans, method=method):
        pi += 1

# ------------------------------------------------- 5. general knowledge
# Novel PHRASINGS of trained facts (tests generalization) + a few untrained facts.
gk = [
    ("Which planet is the biggest one?", "Jupiter"),
    ("Which planet is the smallest one?", "Mercury"),
    ("Which planet sits closest to the sun?", "Mercury"),
    ("Which planet is known for being red?", "Mars"),
    ("On which planet do humans live?", "Earth"),
    ("Name the star at the center of our solar system.", "sun"),
    ("What orbits the Earth at night?", "moon"),
    ("How many days does a normal year have?", "365"),
    ("How many days make one week?", "seven", ["seven", "7"]),
    ("How many hours make one day?", "24"),
    ("How many minutes make one hour?", "60"),
    ("How many months make one year?", "twelve", ["twelve", "12"]),
    ("Which ocean is the biggest?", "Pacific"),
    ("Which river is often called the longest one?", "Nile"),
    ("Which mountain is the highest?", "Everest"),
    ("Which animal is the biggest of all?", "blue whale", ["blue whale", "whale"]),
    ("Which land animal runs the fastest?", "cheetah"),
    ("Which animal is called king of the jungle?", "lion"),
    ("What do bees produce?", "honey"),
    ("What noise does a dog make?", "woof"),
    ("What noise does a cat make?", "meow"),
    ("Tell me the color of the sky.", "blue"),
    ("Tell me the color of grass.", "green"),
    ("Tell me the color of snow.", "white"),
    ("Mixing blue and yellow gives which color?", "green"),
    ("Mixing red and blue gives which color?", "purple"),
    ("Which two elements make water?", "hydrogen and oxygen", ["hydrogen", "oxygen"]),
    ("At what temperature in celsius does water boil?", "100"),
    ("At what temperature in celsius does water freeze?", "0", ["0", "zero"]),
    ("How many legs are on a spider?", "eight", ["eight", "8"]),
    ("How many legs are on an insect?", "six", ["six", "6"]),
    ("How many continents does Earth have?", "seven", ["seven", "7"]),
    ("Which continent is the biggest?", "Asia"),
    ("Which language do people speak in France?", "French"),
    ("Which language do people speak in Japan?", "Japanese"),
    ("Who is the author of Romeo and Juliet?", "Shakespeare"),
    ("Who created the painting Mona Lisa?", "da Vinci", ["da Vinci", "Leonardo"]),
    ("How many sides are on a triangle?", "three", ["three", "3"]),
    ("How many sides are on a square?", "four", ["four", "4"]),
    ("Name the city that is the capital of France.", "Paris"),
    ("Name the city that is the capital of Japan.", "Tokyo"),
    ("Name the city that is the capital of Italy.", "Rome"),
    ("Name the city that is the capital of Germany.", "Berlin"),
    ("Name the city that is the capital of Spain.", "Madrid"),
    ("Name the city that is the capital of India.", "New Delhi", ["New Delhi", "Delhi"]),
    # untrained facts - expected failures, they measure hallucination
    ("How many strings does a normal guitar have?", "six", ["six", "6"]),
    ("What is the biggest desert called?", "Sahara"),
    ("How many players are on a football team on the field?", "eleven", ["eleven", "11"]),
    ("Which season comes after summer?", "autumn", ["autumn", "fall"]),
    ("How many colors are in a rainbow?", "seven", ["seven", "7"]),
]
for entry in gk[:50]:
    q, ans, acc = (entry + (None,))[:3]
    add("general_knowledge", q, ans, acceptable=acc, method="contains")

# ----------------------------------------- 6. instruction following
inst = []
one_word = [
    ("Answer in one word: what color is the sky?", "blue"),
    ("Answer in one word: what color is grass?", "green"),
    ("Answer in one word: what do bees make?", "honey"),
    ("Answer in one word: which planet do we live on?", "Earth"),
    ("Answer in one word: what sound does a cat make?", "meow"),
    ("Answer in one word: what is the capital of France?", "Paris"),
    ("Answer in one word: what is the capital of Japan?", "Tokyo"),
    ("Answer in one word: what color is snow?", "white"),
    ("Answer in one word: what do cows drink?", "water"),
    ("Answer in one word: what is the largest planet?", "Jupiter"),
    ("Answer in one word: what is the red planet?", "Mars"),
    ("Answer in one word: what is the fastest land animal?", "cheetah"),
    ("Answer in one word: what is the capital of Italy?", "Rome"),
]
for q, a in one_word:
    inst.append((q, a, "one_word_correct"))
yes_no = [
    ("Answer yes or no: is the sky blue?", "yes"),
    ("Answer yes or no: is snow black?", "no"),
    ("Answer yes or no: is grass green?", "yes"),
    ("Answer yes or no: do spiders have eight legs?", "yes"),
    ("Answer yes or no: do dogs say meow?", "no"),
    ("Answer yes or no: is the sun a star?", "yes"),
    ("Answer yes or no: is 10 bigger than 5?", "yes"),
    ("Answer yes or no: is 3 bigger than 8?", "no"),
    ("Answer yes or no: does a week have seven days?", "yes"),
    ("Answer yes or no: does a triangle have four sides?", "no"),
    ("Answer yes or no: is water made of hydrogen and o-ygen?", "yes"),
    ("Answer yes or no: is Paris the capital of France?", "yes"),
    ("Answer yes or no: is Tokyo the capital of Spain?", "no"),
]
yes_no = [(q.replace("o-ygen", "gold"), ("no" if "gold" in q.replace("o-ygen", "gold") else a)) for q, a in yes_no]
for q, a in yes_no:
    inst.append((q, a, "yes_no"))
say_word = [
    ("Say the word hello.", "hello"),
    ("Say the word apple.", "apple"),
    ("Repeat this word: banana", "banana"),
    ("Repeat this word: ocean", "ocean"),
    ("Say the word blue.", "blue"),
    ("Repeat this word: seven", "seven"),
    ("Say the word cat.", "cat"),
    ("Repeat this word: morning", "morning"),
    ("Say the word yes.", "yes"),
    ("Repeat this word: tiny", "tiny"),
    ("Say the word sun.", "sun"),
    ("Repeat this word: happy", "happy"),
]
for q, a in say_word:
    inst.append((q, a, "contains"))
three_items = [
    ("Name exactly three colors.", "any three items"),
    ("Name exactly three animals.", "any three items"),
    ("Name exactly three countries.", "any three items"),
    ("Name exactly three planets.", "any three items"),
    ("List exactly three fruits.", "any three items"),
    ("List exactly three numbers.", "any three items"),
    ("Name exactly three cities.", "any three items"),
    ("List exactly three days of the week.", "any three items"),
    ("Name exactly three body parts.", "any three items"),
    ("List exactly three things that are blue.", "any three items"),
    ("Name exactly three sports.", "any three items"),
    ("List exactly three shapes.", "any three items"),
]
for q, a in three_items:
    inst.append((q, a, "three_items"))
for q, a, m in inst[:50]:
    add("instruction_following", q, a, method=m)

# ------------------------------------- 7. reading comprehension (<=180 chars)
rc = []
colors = ["red", "blue", "green", "white"]
things = ["hat", "car", "ball", "bag", "cup"]
seen_rc = set()
while len(rc) < 30:
    n1, n2 = random.sample(names, 2)
    c1, c2 = random.sample(colors, 2)
    t = random.choice(things)
    who = random.choice([1, 2])
    q = (f"{n1} has a {c1} {t}. {n2} has a {c2} {t}. "
         f"What color is {n1 if who == 1 else n2}'s {t}?")
    if q in seen_rc:
        continue
    seen_rc.add(q)
    rc.append((q, c1 if who == 1 else c2))
while len(rc) < 40:
    n1, n2 = random.sample(names, 2)
    a, b = random.randint(1, 20), random.randint(1, 20)
    if a == b:
        continue
    thing = random.choice(["apples", "books", "coins", "pens"])
    q = f"{n1} has {a} {thing}. {n2} has {b} {thing}. Who has more {thing}?"
    if q in seen_rc:
        continue
    seen_rc.add(q)
    rc.append((q, n1 if a > b else n2))
pets = ["dog", "cat", "bird", "fish"]
while len(rc) < 50:
    n1, n2 = random.sample(names, 2)
    p1, p2 = random.sample(pets, 2)
    who = random.choice([1, 2])
    q = f"{n1} owns a {p1}. {n2} owns a {p2}. What pet does {n1 if who == 1 else n2} own?"
    if q in seen_rc:
        continue
    seen_rc.add(q)
    rc.append((q, p1 if who == 1 else p2))
for q, a in rc[:50]:
    add("reading_comprehension", q, a, method="contains")

# ------------------------------------------------------------- 8. coding
# Vocab-safe only (no parentheses, brackets, quotes, uppercase 'X', etc.).
# Baseline expectation is ~0% - there is no code in Aizen's training data.
coding = [
    ("In python, which keyword shows te-t on the screen?", "print"),
    ("In python, which keyword defines a function?", "def"),
    ("In python, which keyword creates a loop over a list?", "for"),
    ("In python, which keyword makes a condition?", "if"),
    ("In python, which keyword returns a value from a function?", "return"),
    ("In python, which keyword imports a module?", "import"),
    ("In python, which word means an empty value?", "None"),
    ("In python, which keyword ends a loop early?", "break"),
    ("In python, which keyword skips to the ne-t loop step?", "continue"),
    ("In python, which keyword defines a class?", "class"),
    ("Which symbol adds two numbers in code?", "+"),
    ("Which symbol multiplies two numbers in code?", "*"),
    ("Which symbol checks equality in python?", "==", ["==", "equals"]),
    ("Which symbol assigns a value in python?", "=", ["="]),
    ("What is the result of 2 ** 3 in python?", "8"),
    ("What is the result of 10 - 3 in python?", "7"),
    ("What is the result of 7 * 2 in python?", "14"),
    ("What does 5 == 5 give in python?", "True", ["True", "true", "yes"]),
    ("What does 3 == 4 give in python?", "False", ["False", "false", "no"]),
    ("In python, what type is the value 5?", "int", ["int", "integer", "number"]),
    ("In python, what type is the value 5.0?", "float"),
    ("In python, what type is the value hello?", "string", ["string", "str"]),
    ("In python, what type is the value True?", "bool", ["bool", "boolean"]),
    ("Which language is named after a snake?", "python"),
    ("Which language runs inside web browsers?", "javascript"),
    ("What does HTML build?", "web pages", ["web", "pages", "websites"]),
    ("What does CSS control on a web page?", "style", ["style", "styling", "looks", "design"]),
    ("In python, which function gives the length of a list?", "len"),
    ("In python, which function turns te-t into a number?", "int"),
    ("In python, which function asks the user to type something?", "input"),
    ("What is a bug in a program?", "an error", ["error", "mistake", "problem"]),
    ("What does a loop do in a program?", "repeats", ["repeat", "repeats", "again"]),
    ("What does a variable store?", "a value", ["value", "data"]),
    ("What does a function do in code?", "runs a task", ["task", "reusable", "code", "action"]),
    ("In python, is indentation important? Answer yes or no.", "yes"),
    ("In python, which word is used for true or false values?", "bool", ["bool", "boolean"]),
    ("Which symbol starts a comment in python?", "hash", ["hash", "#", "pound"]),
    ("In python, what does the plus symbol do to two strings?", "joins them", ["join", "joins", "concatenate", "combines"]),
    ("What number does a python list start counting from?", "0", ["0", "zero"]),
    ("What is the result of 9 + 1 in python?", "10"),
    ("In python, which keyword handles errors with e-cept?", "try"),
    ("In python, which keyword goes with try to catch errors?", "e-cept", ["except"]),
    ("In python, which keyword is the opposite of if?", "else"),
    ("Which company made the iPhone?", "Apple"),
    ("What does AI stand for?", "artificial intelligence", ["artificial intelligence", "artificial"]),
    ("What does CPU stand for?", "central processing unit", ["central processing unit", "processor", "central"]),
    ("What is the result of 100 - 1 in python?", "99"),
    ("What is the result of 6 * 6 in python?", "36"),
    ("In python, which loop runs while a condition is true?", "while"),
    ("What do you call a saved program file for python?", "a python file", ["py", "script", "file"]),
]
for entry in coding[:50]:
    q, ans, acc = (entry + (None,))[:3]
    q = q.replace("te-t", "text").replace("ne-t", "next").replace("e-cept", "except")
    bad = set(q) - VOCAB
    assert not bad, f"coding q has bad chars {bad}: {q}"
    m = "yes_no" if "Answer yes or no" in q else ("number" if str(ans).lstrip("-").isdigit() else "contains")
    add("coding", q, ans, acceptable=acc, method=m)

# ------------------------------------------------------------------- write
counts = {}
for it in items:
    counts[it["category"]] = counts.get(it["category"], 0) + 1
assert all(v == 50 for v in counts.values()), counts
assert len(items) == 400, len(items)

OUT_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"wrote {len(items)} questions to {OUT_PATH}")
for k, v in counts.items():
    print(f"  {k:25s} {v}")
