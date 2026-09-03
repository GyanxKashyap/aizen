"""
Phase 3b supplement generator - targeted fixes for the four failure modes
diagnosed in Phase 3 (docs/phase3_training.md section 11):

1. single_step_chain (800): "How much is a - b?" style single-op questions
   answered with ONE equation - kills the chain-format overgeneralization
   that broke subtraction ("First, 67 + 53 = 110...").
2. deduction_v2 (1500): syllogisms in MANY surface forms, including
   "If ... and ...," phrasing, named instances, negatives, every-X-also-Y,
   day-of-week logic, word-number comparisons, and who-is-shorter/older
   comparatives. Entity sets remain disjoint from eval's (no Kiki/Pipo/
   Bubbles/cats-fly, no Tom/Sam/..., no coins).
3. reading_v2 (1500): reading comprehension with a much wider entity space
   (26 extra names, more objects/colors incl. the common ones, 3-person
   passages, varied question forms) so the model must COPY from context
   instead of memorizing passage-answer pairs.
4. instruction_v2 (1200): one-word / YES-NO / number-only / repeat-word over
   a much wider fact+word pool, still excluding every exact fact-instruction
   combo the eval uses.

Same leakage guards as Phase 2: eval exact match blocked, eval digit-masked
templates blocked (arithmetic skeletons whitelisted with disjoint operand
tuples), qa.txt exact + digit-masked-template collisions blocked.
Policy note (disclosed): logic/reading use the same SENTENCE STRUCTURES as
eval with different entities - entity generalization is the skill under test,
exactly as number generalization is for arithmetic.

Output: data/phase3b_extra.txt (rendered) + data/phase3b_extra.json
Run:  python3 make_phase3b_data.py
"""

import json
import random
import re
from pathlib import Path

rng = random.Random(33020828)

VOCAB = set("\n !'*+,-.0123456789:=?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz")
MAX_RENDERED = 190

qa_text = Path("data/qa.txt").read_text(encoding="utf-8")
qa_masked = {re.sub(r"\d+", "N", q.lower()) for q in re.findall(r"^Q: (.+)$", qa_text, flags=re.M)}
eval_items = json.loads(Path("data/eval.json").read_text(encoding="utf-8"))
p2_questions = {i["question"] for i in json.loads(Path("data/reasoning_train.json").read_text(encoding="utf-8"))}


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
rej = {"eval": 0, "qa": 0, "dup": 0, "long": 0}


def render(q, r, a):
    return f"Q: {q}\nA: {r} The answer is {a}." if r else f"Q: {q}\nA: {a}"


def add(cat, q, r, a):
    q, r, a = q.strip(), r.strip(), str(a).strip()
    text = render(q, r, a)
    bad = set(text) - VOCAB
    assert not bad, f"bad chars {bad}: {text!r}"
    if len(text) > MAX_RENDERED:
        rej["long"] += 1
        return False
    if q in used or q in p2_questions:
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


# ------------------------------------------------ 1. single_step_chain (800)
def cand_single():
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    if rng.random() < 0.6:  # subtraction (the regressed op)
        if b > a:
            a, b = b, a
        res = a - b
        q = rng.choice([f"How much is {a} - {b}?",
                        f"How much is {a} minus {b}?",
                        f"Only one step: what is {a} - {b}?"])
        r = f"{a} - {b} = {res}."
    else:
        res = a + b
        q = rng.choice([f"How much is {a} plus {b}? Answer directly.",
                        f"Only one step: what is {a} + {b}?"])
        r = f"{a} + {b} = {res}."
    return q, r, res


# ---------------------------------------------------- 2. deduction_v2 (1500)
NAMES = ["Raj", "Kim", "Eva", "Zoe", "Omar", "Lily", "Noah", "Ravi", "Nina",
         "Carl", "Aiko", "Yuki", "Gyan", "Mira", "Tara", "Vik", "Asha", "Bela",
         "Chen", "Devi", "Emil", "Fumi", "Hana", "Iris", "Jena", "Kofi",
         "Lena", "Mona", "Nils", "Pia", "Sami", "Uma", "Vera", "Wren", "Yara", "Zane"]
PETNAMES = ["Miko", "Pipa", "Bobo", "Luna", "Toby", "Coco", "Nala", "Ruby", "Ollie", "Fifi"]
DY = [  # (grp, sing, members, prop, verb) - all-yes syllogisms
    ("bees", "bee", ["wasp", "honeybee", "bumblebee"], "are insects", "an insect"),
    ("birds", "bird", ["crow", "swan", "dove", "parrot", "heron", "robin"], "have wings", None),
    ("fish", "fish", ["trout", "salmon", "tuna", "carp"], "live in water", None),
    ("dogs", "dog", ["beagle", "poodle", "husky", "pug"], "can bark", None),
    ("cows", "cow", ["farm cow", "brown cow"], "eat grass", None),
    ("plants", "plant", ["fern", "rose", "tulip", "daisy"], "need light", None),
    ("frogs", "frog", ["tree frog", "pond frog"], "can jump", None),
    ("trains", "train", ["metro", "steam train"], "run on rails", None),
    ("spiders", "spider", ["tarantula", "garden spider"], "spin webs", None),
    ("owls", "owl", ["barn owl", "snowy owl"], "hunt at night", None),
]
DN = [  # no-cases
    ("rocks", "rock", ["stone", "pebble", "boulder"], "swim"),
    ("chairs", "chair", ["stool", "bench", "armchair"], "talk"),
    ("plants", "plant", ["fern", "cactus", "rose"], "walk"),
    ("clouds", "cloud", ["storm cloud", "rain cloud"], "sing"),
    ("spoons", "spoon", ["ladle", "teaspoon"], "dance"),
    ("bricks", "brick", ["clay brick", "red brick"], "fly"),
    ("statues", "statue", ["marble statue", "stone statue"], "blink"),
    ("books", "book", ["notebook", "storybook"], "eat"),
]
EVERY = [("rose", "flower", "yes"), ("oak", "tree", "yes"), ("dog", "animal", "yes"),
         ("mango", "fruit", "yes"), ("trout", "fish", "yes"), ("circle", "shape", "yes"),
         ("fruit", "mango", "no"), ("animal", "dog", "no"), ("tree", "oak", "no"),
         ("flower", "rose", "no"), ("shape", "circle", "no"), ("bird", "crow", "no")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WORDNUM = {"two": 2, "three": 3, "four": 4, "six": 6, "seven": 7, "eight": 8,
           "ten": 10, "eleven": 11, "twenty": 20}
COMPARATIVES = [("taller", "shorter"), ("older", "younger"), ("faster", "slower"),
                ("heavier", "lighter"), ("stronger", "weaker")]
OBJECTS = ["marbles", "stamps", "shells", "beads", "seeds", "kites"]


def yn_suffix(q):
    return q + (" Answer yes or no." if rng.random() < 0.5 else "")


def prop_q(prop, member_phrase):
    """Turn 'have wings' into question form 'does ... have wings?'"""
    if prop.startswith("can "):
        return f"can {member_phrase} {prop[4:]}"
    if prop.startswith("are "):
        return f"is {member_phrase} {prop[4:]}"
    return f"does {member_phrase} {prop}"


def cand_deduction2():
    style = rng.randrange(6)
    if style == 0:  # If-form, yes
        grp, sing, members, prop, obj = rng.choice(DY)
        m = rng.choice(members)
        tail = obj if obj else prop.replace("are ", "")
        qq = prop_q(prop if not obj else "are " + obj, f"a {m}")
        q = f"If all {grp} {prop} and a {m} is a {sing}, {qq}?"
        r = f"A {m} is a {sing}, and all {grp} {prop}."
        return yn_suffix(q), r, "yes"
    if style == 1:  # named instance, yes
        grp, sing, members, prop, obj = rng.choice(DY)
        pn = rng.choice(PETNAMES)
        qq = prop_q(prop if not obj else "are " + obj, pn)
        q = f"All {grp} {prop}. {pn} is a {sing}. {qq[0].upper()}{qq[1:]}?"
        r = f"{pn} is a {sing}, and all {grp} {prop}."
        return yn_suffix(q), r, "yes"
    if style == 2:  # negative, no
        grp, sing, members, verb = rng.choice(DN)
        m = rng.choice(members)
        if rng.random() < 0.5:
            q = f"If no {grp} can {verb} and a {m} is a {sing}, can a {m} {verb}?"
        else:
            pn = rng.choice(PETNAMES)
            q = f"No {grp} can {verb}. {pn} is a {sing}. Can {pn} {verb}?"
            r = f"{pn} is a {sing}, and {grp} cannot {verb}."
            return yn_suffix(q), r, "no"
        r = f"A {m} is a {sing}, and {grp} cannot {verb}."
        return yn_suffix(q), r, "no"
    if style == 3:  # every-X-also-Y / days / word numbers
        pick = rng.randrange(3)
        if pick == 0:
            sub, sup, ans = rng.choice(EVERY)
            q = yn_suffix(f"Is every {sub} also a {sup}?")
            r = f"Every {sub} is a kind of {sup}." if ans == "yes" else \
                f"Not every {sub} is a {sup}."
            return q, r, ans
        if pick == 1:
            i = rng.randrange(7)
            j = rng.randrange(7)
            today, guess = DAYS[i], DAYS[j]
            real = DAYS[(i + 1) % 7]
            if (today, guess) in [("Monday", "Wednesday"), ("Friday", "Saturday")]:
                return None  # eval's exact day pairs
            ans = "yes" if guess == real else "no"
            q = yn_suffix(f"If today is {today}, is tomorrow {guess}?")
            r = f"After {today} comes {real}."
            return q, r, ans
        w1, w2 = rng.sample(list(WORDNUM), 2)
        if {w1, w2} in ({"five", "nine"}, {"twelve", "seven"}):
            return None
        ans = "yes" if WORDNUM[w1] > WORDNUM[w2] else "no"
        q = yn_suffix(f"Is {w1} bigger than {w2}?")
        r = f"{w1[0].upper()}{w1[1:]} is {WORDNUM[w1]} and {w2} is {WORDNUM[w2]}."
        return q, r, ans
    if style == 4:  # comparatives -> who is <opposite>?
        n1, n2 = rng.sample(NAMES, 2)
        big, small = rng.choice(COMPARATIVES)
        if rng.random() < 0.5:
            q = f"{n1} is {big} than {n2}. Who is {small}?"
            r = f"{n1} is {big}, so {n2} is {small}."
            return q, r, n2
        q = f"{n1} is {big} than {n2}. Who is {big}?"
        r = f"The story says {n1} is {big} than {n2}."
        return q, r, n1
    # style 5: numeric possession comparison
    n1, n2 = rng.sample(NAMES, 2)
    a, b = rng.randint(1, 60), rng.randint(61, 99)
    obj = rng.choice(OBJECTS)
    if rng.random() < 0.5:
        q = f"{n1} has {a} {obj}. {n2} has {b} {obj}. Who has more {obj}?"
        r = f"{b} is more than {a}."
        return q, r, n2
    q = f"{n1} has {a} {obj}. {n2} has {b} {obj}. Who has fewer {obj}?"
    r = f"{a} is less than {b}."
    return q, r, n1


# ------------------------------------------------------ 3. reading_v2 (1500)
COLORS = ["red", "blue", "green", "white", "yellow", "pink", "purple",
          "orange", "black", "brown", "gray", "gold"]
ROBJ = ["hat", "car", "ball", "bag", "cup", "kite", "sock", "lamp", "boat",
        "coat", "desk", "drum", "ring", "bike", "book"]
RPETS = ["dog", "cat", "bird", "fish", "goat", "hen", "duck", "frog", "pony", "lamb"]
RTHINGS = ["apples", "books", "pens", "mangoes", "marbles", "stamps", "shells", "kites"]


def cand_reading2():
    style = rng.randrange(4)
    ns = rng.sample(NAMES, 3)
    if style == 0:  # 2-person color
        o = rng.choice(ROBJ)
        c1, c2 = rng.sample(COLORS, 2)
        w = rng.choice([0, 1])
        qf = rng.choice(["What color is {n}'s {o}?", "Tell me the color of {n}'s {o}."])
        q = f"{ns[0]} has a {c1} {o}. {ns[1]} has a {c2} {o}. " + qf.format(n=ns[w], o=o)
        r = f"The story says {ns[w]} has a {[c1, c2][w]} {o}."
        return q, r, [c1, c2][w]
    if style == 1:  # 3-person color (harder binding)
        o = rng.choice(ROBJ)
        c1, c2, c3 = rng.sample(COLORS, 3)
        w = rng.randrange(3)
        q = (f"{ns[0]}'s {o} is {c1}. {ns[1]}'s {o} is {c2}. {ns[2]}'s {o} is {c3}. "
             f"What color is {ns[w]}'s {o}?")
        r = f"The story says {ns[w]}'s {o} is {[c1, c2, c3][w]}."
        return q, r, [c1, c2, c3][w]
    if style == 2:  # pets, both directions
        p1, p2 = rng.sample(RPETS, 2)
        if rng.random() < 0.5:
            w = rng.choice([0, 1])
            q = f"{ns[0]} owns a {p1}. {ns[1]} owns a {p2}. What pet does {ns[w]} own?"
            r = f"The story says {ns[w]} owns a {[p1, p2][w]}."
            return q, r, [p1, p2][w]
        w = rng.choice([0, 1])
        q = f"{ns[0]} owns a {p1}. {ns[1]} owns a {p2}. Who owns the {[p1, p2][w]}?"
        r = f"The story says {ns[w]} owns the {[p1, p2][w]}."
        return q, r, ns[w]
    a, b = rng.randint(1, 40), rng.randint(1, 40)  # counts
    if a == b:
        return None
    t = rng.choice(RTHINGS)
    kind = rng.choice(["more", "fewer"])
    ans = ns[0] if ((a > b) == (kind == "more")) else ns[1]
    q = f"{ns[0]} has {a} {t}. {ns[1]} has {b} {t}. Who has {kind} {t}?"
    r = f"{a} and {b} are compared, and {ans} has {kind}."
    return q, r, ans


# -------------------------------------------------- 4. instruction_v2 (1200)
# wider facts, still excluding every exact eval fact-instruction combo
CAPITALS = [("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Scotland", "Edinburgh"),
            ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Switzerland", "Bern"),
            ("Austria", "Vienna"), ("Greece", "Athens"), ("Norway", "Oslo"),
            ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
            ("Poland", "Warsaw"), ("Russia", "Moscow"), ("Ukraine", "Kyiv"),
            ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"),
            ("China", "Beijing"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"),
            ("Pakistan", "Islamabad"), ("Nepal", "Kathmandu"), ("Iran", "Tehran"),
            ("Iraq", "Baghdad"), ("Qatar", "Doha"), ("Australia", "Canberra"),
            ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Chile", "Santiago"),
            ("Peru", "Lima"), ("Cuba", "Havana"), ("Germany", "Berlin"), ("Spain", "Madrid")]
IFACTS = [("What color is a banana?", "yellow"), ("What color is blood?", "red"),
          ("What color is coal?", "black"), ("What color is milk?", "white"),
          ("What color is a carrot?", "orange"), ("What color is a frog?", "green"),
          ("What color is butter?", "yellow"), ("What do chickens lay?", "eggs"),
          ("What do fish swim in?", "water"), ("What is frozen water called?", "ice"),
          ("What is the opposite of hot?", "cold"), ("What is the opposite of day?", "night"),
          ("What is the opposite of up?", "down"), ("What is the opposite of big?", "small"),
          ("What is the opposite of fast?", "slow"), ("What is the opposite of wet?", "dry"),
          ("What organ pumps blood?", "heart"), ("What do we see with?", "eyes"),
          ("What do we hear with?", "ears"), ("What season is the coldest?", "winter"),
          ("What season is the hottest?", "summer"), ("What animal barks?", "dogs"),
          ("What animal purrs?", "cats"), ("What animal has a trunk?", "elephant"),
          ("What insect makes webs?", "spider"), ("What do cows give us to drink?", "milk"),
          ("What is a baby dog called?", "puppy"), ("What is a baby cat called?", "kitten"),
          ("What vehicle flies in the sky?", "plane"), ("What vehicle runs on rails?", "train"),
          ("What do we sleep on at night?", "bed"), ("What do we wear on our feet?", "shoes"),
          ("What room do we cook in?", "kitchen"), ("What place do children go to learn?", "school"),
          ("What star gives Earth light and heat?", "sun"), ("What shape has three corners?", "triangle")]
YN2 = [("Do cows give milk?", "YES"), ("Do fish live in trees?", "NO"),
       ("Do birds lay eggs?", "YES"), ("Can pigs fly?", "NO"),
       ("Is ice cold?", "YES"), ("Is fire cold?", "NO"),
       ("Do cats purr?", "YES"), ("Is milk white?", "YES"),
       ("Is a banana purple?", "NO"), ("Do bees fly?", "YES"),
       ("Is winter cold?", "YES"), ("Do plants need water?", "YES"),
       ("Is rain wet?", "YES"), ("Do elephants have trunks?", "YES"),
       ("Do ducks swim?", "YES"), ("Is honey sweet?", "YES"),
       ("Is a lemon sweet?", "NO"), ("Do trees have roots?", "YES"),
       ("Does a cow have wings?", "NO"), ("Can a chair walk?", "NO")]
RWORDS = ["river", "candle", "garden", "window", "planet", "bottle", "monkey",
          "silver", "dragon", "castle", "island", "forest", "rocket", "pillow",
          "ladder", "mirror", "basket", "camera", "guitar", "helmet", "jacket",
          "lemon", "mango", "pencil", "rabbit", "sailor", "tiger", "violin",
          "wagon", "anchor", "bridge", "cloud", "engine", "flower", "grape",
          "hammer", "jungle", "koala", "magnet", "needle", "parrot", "ribbon",
          "shadow", "temple", "valley", "winter", "carpet", "donkey", "eagle",
          "falcon", "goose", "melon", "olive", "stone", "tulip", "wolf"]


def cand_instr2():
    style = rng.randrange(4)
    if style == 0:  # one word
        if rng.random() < 0.4:
            c, cap = rng.choice(CAPITALS)
            body, word = f"What is the capital of {c}?", cap
        else:
            body, word = rng.choice(IFACTS)
        tpl = rng.choice(["Answer in one word: {b}", "Answer in one word. {b}",
                          "In one word: {b}", "One word only: {b}",
                          "{b} Answer in one word.", "Reply with one word: {b}",
                          "{b} One word answer please."])
        return tpl.format(b=body), "", word
    if style == 1:  # yes/no
        if rng.random() < 0.5:
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            if a == b:
                return None
            kind = rng.choice(["bigger", "smaller"])
            body = f"Is {a} {kind} than {b}?"
            ans = "YES" if ((a > b) == (kind == "bigger")) else "NO"
        else:
            body, ans = rng.choice(YN2)
        tpl = rng.choice(["Answer with YES or NO. {b}", "Answer yes or no: {b}",
                          "{b} Answer yes or no.", "YES or NO: {b}",
                          "Reply YES or NO: {b}"])
        out = ans if "YES or NO" in tpl else ans.lower()
        return tpl.format(b=body), "", out
    if style == 2:  # number only
        w = rng.choice(RWORDS)
        tpl = rng.choice(["Answer with a number only. How many letters are in the word {w}?",
                          "Number only: how many letters does {w} have?",
                          "Give just the number. How many letters in {w}?"])
        return tpl.format(w=w), "", len(w)
    w = rng.choice(RWORDS)  # repeat/say word (eval's 12 say-words not in RWORDS)
    tpl = rng.choice(["Say the word {w}.", "Repeat this word: {w}",
                      "Write only the word {w}.", "Repeat after me: {w}"])
    return tpl.format(w=w), "", w


# ------------------------------------------------------------------- build
PLAN = [("single_step_chain", 800, cand_single),
        ("deduction_v2", 1500, cand_deduction2),
        ("reading_v2", 1500, cand_reading2),
        ("instruction_v2", 1200, cand_instr2)]
for cat, n, fn in PLAN:
    fill(cat, n, fn)
    print(f"  {cat:20s} {n} done")

rng.shuffle(items)
Path("data/phase3b_extra.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
Path("data/phase3b_extra.txt").write_text(
    "\n\n".join(render(i["question"], i["reasoning"], i["answer"]) for i in items) + "\n", encoding="utf-8")
print(f"\nwrote {len(items)} supplement examples")
print("rejections:", rej)
