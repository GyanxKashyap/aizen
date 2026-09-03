"""
Phase 2 dataset generator: instruction-following + reasoning training data.

Outputs:
  data/reasoning_train.json     - 10,000 records {id, category, question, reasoning, answer}
  data/aizen_phase2_train.txt   - rendered char-level training text

Approved format decision (Phase 2 review): reasoning is rendered INLINE on the
answer line - "Q: ...\nA: First, 23 + 17 = 40. Then, 40 - 8 = 32. The answer
is 32." - because every existing inference path (eval.py, chat.py, server.py)
prompts "Q: ...\nA:" and stops at the first newline. A separate "T:" line
would never be generated at eval time. The JSON keeps question/reasoning/
answer as separate fields so any other rendering can be produced later.

Guarantees:
- deterministic (seed 20260827); no duplicates
- every char in Aizen's 73-char vocab (no uppercase X, no ()"; etc.)
- every rendered example <= 190 chars (fits the 192-char context)
- all deterministic answers computed with Python, never guessed
- collision checks vs data/qa.txt and data/eval.json:
    exact question match          -> rejected
    digit-masked template match   -> rejected (except whitelisted arithmetic
                                     skeletons, which are allowed with
                                     guaranteed-disjoint operand tuples)
    operand-tuple match with eval -> rejected
- rejection counts are reported.

Run:  python3 make_phase2_data.py
"""

import json
import random
import re
from pathlib import Path

rng = random.Random(20260827)

OUT_JSON = Path("data/reasoning_train.json")
OUT_TXT = Path("data/aizen_phase2_train.txt")
MAX_RENDERED = 190

VOCAB = set("\n !'*+,-.0123456789:=?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz")

# ------------------------------------------------------------ leakage guards
qa_text = Path("data/qa.txt").read_text(encoding="utf-8")
qa_questions = re.findall(r"^Q: (.+)$", qa_text, flags=re.M)


def masked(s):
    return re.sub(r"\d+", "N", s.lower().strip())


def numtuple(s):
    return tuple(int(n) for n in re.findall(r"\d+", s))


qa_masked = {masked(q) for q in qa_questions}

eval_items = json.loads(Path("data/eval.json").read_text(encoding="utf-8"))
eval_exact = {it["question"] for it in eval_items}
eval_masked_all = {masked(it["question"]) for it in eval_items}
# arithmetic skeletons stay learnable with different numbers (approved policy)
WHITELISTED_MASKED = {masked(it["question"]) for it in eval_items
                      if it["category"] in ("arithmetic", "multi_step_arithmetic")}
eval_masked_blocked = eval_masked_all - WHITELISTED_MASKED
eval_tuples = {numtuple(it["question"]) for it in eval_items if numtuple(it["question"])}

rejections = {"exact_eval": 0, "masked_eval": 0, "tuple_eval": 0,
              "exact_qa": 0, "masked_qa": 0, "duplicate": 0, "too_long": 0}
whitelisted_used = 0

items = []
used_questions = set()


def render(q, reasoning, answer):
    if reasoning:
        return f"Q: {q}\nA: {reasoning} The answer is {answer}."
    return f"Q: {q}\nA: {answer}"


def add(category, q, reasoning, answer):
    """Try to add one example. Returns True if accepted."""
    global whitelisted_used
    q, reasoning, answer = q.strip(), reasoning.strip(), str(answer).strip()
    assert q and answer, f"empty question/answer in {category}"
    text = render(q, reasoning, answer)
    bad = set(text) - VOCAB
    assert not bad, f"chars outside vocab {bad}: {text!r}"
    if len(text) > MAX_RENDERED:
        rejections["too_long"] += 1
        return False
    if q in used_questions:
        rejections["duplicate"] += 1
        return False
    if q in eval_exact:
        rejections["exact_eval"] += 1
        return False
    m = masked(q)
    if m in eval_masked_blocked:
        rejections["masked_eval"] += 1
        return False
    if m in WHITELISTED_MASKED:
        if numtuple(q) in eval_tuples:
            rejections["tuple_eval"] += 1
            return False
        whitelisted_used += 1
    elif numtuple(q) and numtuple(q) in eval_tuples:
        rejections["tuple_eval"] += 1
        return False
    if q in qa_text:
        rejections["exact_qa"] += 1
        return False
    if m in qa_masked:
        rejections["masked_qa"] += 1
        return False
    used_questions.add(q)
    items.append({
        "id": f"{category}_{sum(1 for i in items if i['category'] == category) + 1:04d}",
        "category": category,
        "question": q,
        "reasoning": reasoning,
        "answer": answer,
    })
    return True


def fill(category, target, candidate_fn, max_attempts=400000):
    """Draw candidates until `target` examples are accepted."""
    start = sum(1 for i in items if i["category"] == category)
    attempts = 0
    while sum(1 for i in items if i["category"] == category) - start < target:
        attempts += 1
        assert attempts < max_attempts, f"{category}: exhausted candidates"
        cand = candidate_fn()
        if cand is None:
            continue
        add(category, *cand)


# ============================================================ content pools
NAMES = ["Raj", "Kim", "Eva", "Zoe", "Omar", "Lily", "Noah", "Ravi",
         "Nina", "Carl", "Aiko", "Yuki", "Gyan", "Mira", "Tara", "Vik"]
OBJECTS = ["mangoes", "marbles", "stamps", "shells", "kites", "socks",
           "seeds", "beads", "bricks", "notes"]

# fact pool for instruction formats: (question body, one-word answer, full sentence)
# Deliberately avoids every fact used by eval's one-word/yes-no items.
FACTS = [
    ("What color is a banana?", "yellow", "A banana is yellow."),
    ("What color is blood?", "red", "Blood is red."),
    ("What color is coal?", "black", "Coal is black."),
    ("What color is milk?", "white", "Milk is white."),
    ("What color is a carrot?", "orange", "A carrot is orange."),
    ("What color is a frog?", "green", "A frog is green."),
    ("What color is the night sky?", "black", "The night sky is black."),
    ("What color is butter?", "yellow", "Butter is yellow."),
    ("What do chickens lay?", "eggs", "Chickens lay eggs."),
    ("What do you drink when thirsty?", "water", "You drink water when thirsty."),
    ("What falls from clouds when it rains?", "rain", "Rain falls from clouds."),
    ("What do fish swim in?", "water", "Fish swim in water."),
    ("What do birds build to live in?", "nests", "Birds build nests."),
    ("What do we read that has many pages?", "books", "We read books."),
    ("What do we use to write on paper?", "pens", "We use pens to write."),
    ("What shines in the sky at night with the stars?", "moon", "The moon shines at night."),
    ("What season is the coldest?", "winter", "Winter is the coldest season."),
    ("What season is the hottest?", "summer", "Summer is the hottest season."),
    ("What animal barks?", "dogs", "Dogs bark."),
    ("What animal purrs?", "cats", "Cats purr."),
    ("What animal has a very long neck?", "giraffe", "A giraffe has a very long neck."),
    ("What animal has a trunk?", "elephant", "An elephant has a trunk."),
    ("What animal hops and carries babies in a pouch?", "kangaroo", "A kangaroo hops and has a pouch."),
    ("What animal is known as man's best friend?", "dog", "The dog is man's best friend."),
    ("What insect makes webs?", "spider", "A spider makes webs."),
    ("What fruit keeps the doctor away?", "apple", "An apple a day keeps the doctor away."),
    ("What do cows give us to drink?", "milk", "Cows give us milk."),
    ("What is frozen water called?", "ice", "Frozen water is called ice."),
    ("What is the opposite of hot?", "cold", "The opposite of hot is cold."),
    ("What is the opposite of day?", "night", "The opposite of day is night."),
    ("What is the opposite of up?", "down", "The opposite of up is down."),
    ("What is the opposite of big?", "small", "The opposite of big is small."),
    ("What is the opposite of fast?", "slow", "The opposite of fast is slow."),
    ("What is the opposite of happy?", "sad", "The opposite of happy is sad."),
    ("What is the opposite of open?", "closed", "The opposite of open is closed."),
    ("What is the opposite of wet?", "dry", "The opposite of wet is dry."),
    ("What organ pumps blood?", "heart", "The heart pumps blood."),
    ("What organ do we think with?", "brain", "We think with the brain."),
    ("What do we breathe with?", "lungs", "We breathe with our lungs."),
    ("What do we see with?", "eyes", "We see with our eyes."),
    ("What do we hear with?", "ears", "We hear with our ears."),
    ("What do we smell with?", "nose", "We smell with our nose."),
    ("What covers a bird's body?", "feathers", "Feathers cover a bird's body."),
    ("What covers a fish's body?", "scales", "Scales cover a fish's body."),
    ("What do bears do all winter?", "sleep", "Bears sleep all winter."),
    ("What do caterpillars turn into?", "butterflies", "Caterpillars turn into butterflies."),
    ("What vehicle flies in the sky?", "plane", "A plane flies in the sky."),
    ("What vehicle runs on rails?", "train", "A train runs on rails."),
    ("What vehicle sails on the sea?", "ship", "A ship sails on the sea."),
    ("What has two wheels and pedals?", "bicycle", "A bicycle has two wheels and pedals."),
    ("What do farmers grow in fields?", "crops", "Farmers grow crops in fields."),
    ("What sweet food do bees help make?", "honey", "Bees help make honey."),
    ("What white grains do we cook and eat?", "rice", "We cook rice."),
    ("What hot drink is made from beans?", "coffee", "Coffee is made from beans."),
    ("What do we sleep on at night?", "bed", "We sleep on a bed at night."),
    ("What do we wear on our feet?", "shoes", "We wear shoes on our feet."),
    ("What do we wear on our head in the sun?", "hat", "We wear a hat in the sun."),
    ("What do we use to eat soup?", "spoon", "We use a spoon to eat soup."),
    ("What do we use to cut paper?", "scissors", "We use scissors to cut paper."),
    ("What room do we cook in?", "kitchen", "We cook in the kitchen."),
    ("What place has many books to borrow?", "library", "A library has many books."),
    ("What place do children go to learn?", "school", "Children learn at school."),
    ("What do plants grow from?", "seeds", "Plants grow from seeds."),
    ("What part of the plant is under the ground?", "roots", "The roots are under the ground."),
    ("What gas do we breathe in to live?", "air", "We breathe air to live."),
    ("What star gives Earth light and heat?", "sun", "The sun gives Earth light and heat."),
    ("What shape has no corners at all?", "circle", "A circle has no corners."),
    ("What shape has three corners?", "triangle", "A triangle has three corners."),
    ("What do you call water falling over a cliff?", "waterfall", "It is called a waterfall."),
    ("What is a baby dog called?", "puppy", "A baby dog is a puppy."),
    ("What is a baby cat called?", "kitten", "A baby cat is a kitten."),
    ("What is a baby cow called?", "calf", "A baby cow is a calf."),
    ("What is a baby sheep called?", "lamb", "A baby sheep is a lamb."),
    ("What is a group of wolves called?", "pack", "A group of wolves is a pack."),
    ("What is a home for a king called?", "palace", "A king lives in a palace."),
    ("What is a home for bees called?", "hive", "Bees live in a hive."),
    ("What is a home for birds called?", "nest", "Birds live in a nest."),
]
# one-word capitals (single-word capitals only, none used in eval)
CAPITALS = [
    ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Scotland", "Edinburgh"),
    ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Switzerland", "Bern"),
    ("Austria", "Vienna"), ("Greece", "Athens"), ("Norway", "Oslo"),
    ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Poland", "Warsaw"), ("Russia", "Moscow"), ("Ukraine", "Kyiv"),
    ("Turkey", "Ankara"), ("Egypt", "Cairo"), ("Morocco", "Rabat"),
    ("Nigeria", "Abuja"), ("Kenya", "Nairobi"), ("China", "Beijing"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
    ("Singapore", "Singapore"), ("Philippines", "Manila"), ("Pakistan", "Islamabad"),
    ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Qatar", "Doha"), ("Australia", "Canberra"),
    ("Canada", "Ottawa"), ("Brazil", "Brasilia"), ("Chile", "Santiago"),
    ("Peru", "Lima"), ("Colombia", "Bogota"), ("Venezuela", "Caracas"),
    ("Cuba", "Havana"),
]

WORDS = ["river", "candle", "garden", "window", "planet", "bottle", "monkey",
         "silver", "purple", "orange", "dragon", "castle", "island", "forest",
         "rocket", "pillow", "ladder", "mirror", "basket", "camera", "guitar",
         "helmet", "jacket", "kitten", "lemon", "mango", "napkin", "oven",
         "pencil", "rabbit", "sailor", "tiger", "umbrella", "violin", "wagon",
         "yogurt", "zebra", "anchor", "bridge", "cloud", "danger", "engine",
         "flower", "grape", "hammer", "insect", "jungle", "koala", "lantern",
         "magnet", "needle", "ocean", "parrot", "quilt", "ribbon", "shadow",
         "temple", "under", "valley", "winter", "yellow", "carpet", "donkey",
         "eagle", "falcon", "goose", "heron", "igloo", "jelly", "kite",
         "lizard", "melon", "nut", "olive", "peach", "queen", "rose",
         "stone", "tulip", "urn", "vase", "wolf", "yarn", "zoo", "acorn",
         "berry", "cedar", "daisy", "elm", "fern", "grass", "holly", "ivy"]
SHORT_WORDS = [w for w in WORDS if 3 <= len(w) <= 5] + [
    "cat", "dog", "sun", "map", "pen", "cup", "hat", "bed", "car", "bus",
    "sky", "sea", "leaf", "fish", "bird", "frog", "star", "moon", "rain",
    "snow", "wind", "fire", "tree", "door", "milk", "cake", "rice", "soup",
    "ship", "boat", "road", "hill", "sand", "rock", "ring", "song", "game",
    "ball", "shoe", "coat", "desk", "lamp", "book", "page", "word", "name",
    "time", "year", "hand", "foot", "nose", "hair", "gold", "blue", "pink"]

FRUITS = ["mango", "pear", "plum", "grape", "peach", "melon", "cherry", "banana", "apple", "kiwi", "lemon", "fig"]
ANIMALS = ["cat", "dog", "cow", "goat", "hen", "duck", "frog", "owl", "pig", "horse", "sheep", "deer", "bear", "wolf", "seal"]
COLORS = ["yellow", "pink", "purple", "orange", "black", "brown", "gray", "gold"]
DRINKS = ["water", "milk", "juice", "tea", "coffee", "soda"]
TREES = ["oak", "pine", "elm", "cedar", "palm", "birch"]
JOBS = ["doctor", "farmer", "teacher", "singer", "pilot", "baker", "nurse", "artist"]
VEGGIES = ["carrot", "onion", "bean", "corn", "pea", "potato", "tomato", "radish"]
BIRDS = ["crow", "owl", "duck", "hen", "swan", "dove", "heron", "parrot"]
TOOLS = ["hammer", "saw", "drill", "brush", "rake", "shovel", "wrench", "pliers"]
POOLS3 = {"fruits": FRUITS, "animals": ANIMALS, "colors": COLORS,
          "drinks": DRINKS, "trees": TREES, "jobs": JOBS,
          "vegetables": VEGGIES, "birds": BIRDS, "tools": TOOLS}

PLACES = ["beach", "park", "market", "school", "farm", "lake", "hall", "shop", "garden", "forest"]

# ============================================================ INSTRUCTION DATA

# ---- fmt_one_word (600)
def cand_one_word():
    style = rng.randrange(5)
    if rng.random() < 0.25:
        c, cap = rng.choice(CAPITALS)
        body, word = f"What is the capital of {c}?", cap
    else:
        body, word, _ = rng.choice(FACTS)
    tpl = rng.choice([
        "Answer in one word. {b}",
        "In one word: {b}",
        "One word only: {b}",
        "{b} Answer in one word.",
        "Reply with one word. {b}",
        "{b} Use only one word.",
    ])
    return tpl.format(b=body), "", word


# ---- fmt_yes_no (600) - numeric comparisons + curated facts (none from eval)
YESNO_FACTS = [
    ("Do cows give milk?", "YES"), ("Do fish live in trees?", "NO"),
    ("Do birds lay eggs?", "YES"), ("Can pigs fly?", "NO"),
    ("Is ice cold?", "YES"), ("Is fire cold?", "NO"),
    ("Do cats purr?", "YES"), ("Do rocks eat food?", "NO"),
    ("Is milk white?", "YES"), ("Is a banana purple?", "NO"),
    ("Do bees fly?", "YES"), ("Can a chair walk?", "NO"),
    ("Is winter cold?", "YES"), ("Is the moon made of cheese?", "NO"),
    ("Do plants need water?", "YES"), ("Do dogs bark?", "YES"),
    ("Can a fish ride a bicycle?", "NO"), ("Is rain wet?", "YES"),
    ("Do elephants have trunks?", "YES"), ("Is a circle square?", "NO"),
    ("Do ducks swim?", "YES"), ("Can a spoon sing?", "NO"),
    ("Is honey sweet?", "YES"), ("Is a lemon sweet?", "NO"),
    ("Do trees have roots?", "YES"), ("Does a cow have wings?", "NO"),
]
def cand_yes_no():
    if rng.random() < 0.5:
        a, b = rng.randint(1, 99), rng.randint(1, 99)
        if a == b:
            return None
        kind = rng.choice(["bigger", "smaller"])
        ans = "YES" if ((a > b) == (kind == "bigger")) else "NO"
        body = f"Is {a} {kind} than {b}?"
    else:
        body, ans = rng.choice(YESNO_FACTS)
    tpl = rng.choice([
        "Answer with YES or NO. {b}",
        "{b} Answer with YES or NO.",
        "YES or NO: {b}",
        "Reply YES or NO. {b}",
    ])
    return tpl.format(b=body), "", ans


# ---- fmt_number_only (500)
def cand_number_only():
    style = rng.randrange(3)
    if style == 0:
        w = rng.choice(WORDS)
        body, ans = f"How many letters are in the word {w}?", len(w)
    elif style == 1:
        k = rng.randint(3, 6)
        ws = rng.sample(WORDS, k)
        body, ans = f"How many words are in this list: {', '.join(ws)}?", k
    else:
        a, b = rng.randint(1, 60), rng.randint(1, 60)
        body, ans = f"How much is {a} plus {b}?", a + b
    tpl = rng.choice([
        "Answer with a number only. {b}",
        "{b} Answer with a number only.",
        "Give just the number. {b}",
        "Number only: {b}",
    ])
    return tpl.format(b=body), "", ans


# ---- fmt_three_items (300) / fmt_two_examples (200)
def cand_three_items():
    kind, pool = rng.choice(list(POOLS3.items()))
    picks = rng.sample(pool, 3)
    tpl = rng.choice([
        "Give exactly three {k}.",
        "Write exactly three {k}.",
        "Tell me exactly three {k}.",
        "Give three {k}, no more and no less.",
    ])
    # a varying suffix makes each instruction unique for a different sample
    suffix = rng.choice([f" Start with {picks[0]}.", f" One of them must be {picks[1]}."])
    return tpl.format(k=kind) + suffix, "", ", ".join(picks)


def cand_two_examples():
    kind, pool = rng.choice(list(POOLS3.items()))
    picks = rng.sample(pool, 2)
    tpl = rng.choice([
        "Give exactly two examples of {k}.",
        "Write exactly two {k}.",
        "Name just two {k}.",
    ])
    suffix = rng.choice([f" Start with {picks[0]}.", f" One of them must be {picks[1]}."])
    return tpl.format(k=kind) + suffix, "", ", ".join(picks)


# ---- fmt_three_words (100) - curated deterministic triples
THREE_WORDS = [
    ("ice", "cold hard water"), ("the sun", "big hot star"),
    ("a mouse", "small gray animal"), ("snow", "cold white flakes"),
    ("a desert", "dry hot sand"), ("night", "dark quiet time"),
    ("an oven", "hot metal box"), ("a lake", "calm fresh water"),
    ("a whale", "huge sea animal"), ("a feather", "soft light thing"),
    ("thunder", "loud sky sound"), ("a library", "quiet book place"),
    ("lava", "hot melted rock"), ("a glacier", "slow moving ice"),
    ("honey", "sweet sticky food"), ("a kitten", "small young cat"),
    ("a mountain", "tall rocky land"), ("fog", "thick gray mist"),
    ("a rocket", "fast flying machine"), ("winter", "cold snowy season"),
    ("a drum", "loud round instrument"), ("grass", "green soft plant"),
    ("a candle", "small warm light"), ("the sea", "big salty water"),
    ("a spider", "small web maker"),
]
def cand_three_words():
    thing, triple = rng.choice(THREE_WORDS)
    tpl = rng.choice([
        "Describe {t} in exactly three words.",
        "Use exactly three words to describe {t}.",
        "In exactly three words, describe {t}.",
        "Describe {t}. Use exactly three words.",
    ])
    return tpl.format(t=thing), "", triple


# ---- fmt_no_explanation (300)
def cand_no_explanation():
    if rng.random() < 0.4:
        c, cap = rng.choice(CAPITALS)
        body, ans = f"What is the capital of {c}?", cap
    else:
        body, ans, _ = rng.choice(FACTS)
    tpl = rng.choice([
        "Answer without explanation. {b}",
        "No explanation, just the answer. {b}",
        "{b} Do not explain, just answer.",
    ])
    return tpl.format(b=body), "", ans


# ---- transform_case (300)
def cand_case():
    if rng.random() < 0.5:
        w = rng.choice([x for x in WORDS if "x" not in x])
        tpl = rng.choice([
            "Write the word {w} in uppercase.",
            "Turn this word into uppercase: {w}",
            "Make this uppercase: {w}",
        ])
        return tpl.format(w=w), "", w.upper()
    w = rng.choice([x for x in WORDS if "x" not in x]).upper()
    tpl = rng.choice([
        "Write the word {w} in lowercase.",
        "Turn this word into lowercase: {w}",
        "Make this lowercase: {w}",
    ])
    return tpl.format(w=w), "", w.lower()


# ---- transform_reverse (200)
def cand_reverse():
    w = rng.choice(SHORT_WORDS)
    tpl = rng.choice([
        "Reverse the word {w}.",
        "Write the word {w} backwards.",
        "Spell {w} in reverse.",
    ])
    return tpl.format(w=w), "", w[::-1]


# ---- transform_count (400)
def cand_count():
    if rng.random() < 0.5:
        w = rng.choice(WORDS)
        tpl = rng.choice([
            "Count the letters in the word {w}.",
            "How many letters does the word {w} have?",
        ])
        return tpl.format(w=w), "", len(w)
    k = rng.randint(3, 7)
    ws = rng.sample(WORDS, k)
    sent = " ".join(ws)
    tpl = rng.choice([
        "Count the words here: {s}",
        "How many words are here: {s}?",
    ])
    return tpl.format(s=sent), "", k


# ---- transform_sort (400)
def cand_sort():
    if rng.random() < 0.5:
        k = rng.randint(3, 4)
        nums = rng.sample(range(1, 99), k)
        if rng.random() < 0.5:
            tpl, out = "Sort these numbers from small to big: {s}.", sorted(nums)
        else:
            tpl, out = "Sort these numbers from big to small: {s}.", sorted(nums, reverse=True)
        return tpl.format(s=", ".join(map(str, nums))), "", ", ".join(map(str, out))
    ws = rng.sample(WORDS, 3)
    tpl = rng.choice([
        "Sort these words alphabetically: {s}.",
        "Put these words in alphabetical order: {s}.",
    ])
    return tpl.format(s=", ".join(ws)), "", ", ".join(sorted(ws))


# ---- transform_extreme (200) - longest/shortest word
def cand_extreme():
    ws = rng.sample(WORDS, 3)
    if len({len(w) for w in ws}) < 3:
        return None
    if rng.random() < 0.5:
        tpl, out = "Which word is the longest: {s}?", max(ws, key=len)
    else:
        tpl, out = "Which word is the shortest: {s}?", min(ws, key=len)
    return tpl.format(s=", ".join(ws)), "", out


# ---- classify (400)
POS = ["I love this warm sunny morning", "The party was wonderful",
       "This is the best cake I ever ate", "We had a great day at the beach",
       "My new kite flies so well", "The music made me smile",
       "I am so happy about my gift", "The garden looks lovely today",
       "That was a fantastic game", "Our trip was full of joy",
       "My puppy learned a cute trick", "The soup smells amazing",
       "I passed my test with a top score", "The stars look beautiful tonight",
       "My friends threw me a lovely party", "This song always cheers me up",
       "The bread came out soft and warm", "I found my lost ring at last",
       "Our team won the big match", "The view from the hill is stunning"]
NEG = ["I hate waiting in long lines", "The food was awful",
       "My day was ruined by the rain", "This is the worst movie ever",
       "I lost my favorite pen today", "The room was dirty and cold",
       "That noise gives me a headache", "I am so sad about the news",
       "The trip was a total mess", "My shoes are torn and ugly",
       "The milk went sour this morning", "I missed my bus again today",
       "My kite got stuck in a tree", "The soup tastes terrible",
       "Nobody came to my show", "My phone broke this afternoon",
       "The queue moved painfully slowly", "I failed to fix the leak",
       "The garden is full of weeds now", "My plans fell apart completely"]
CATEG = [("hammer", "tool"), ("saw", "tool"), ("drill", "tool"), ("shovel", "tool"),
         ("rake", "tool"), ("wrench", "tool"), ("mango", "fruit"), ("plum", "fruit"),
         ("grape", "fruit"), ("peach", "fruit"), ("melon", "fruit"), ("cherry", "fruit"),
         ("goat", "animal"), ("owl", "animal"), ("seal", "animal"), ("deer", "animal"),
         ("frog", "animal"), ("horse", "animal"), ("carrot", "vegetable"),
         ("onion", "vegetable"), ("potato", "vegetable"), ("radish", "vegetable"),
         ("corn", "vegetable"), ("tomato", "vegetable")]
CATNAMES = ["tool", "fruit", "animal", "vegetable"]
TRUEFALSE = [
    ("A spider has six legs.", "false"), ("The sun rises in the morning.", "true"),
    ("Fish can breathe under water.", "true"), ("A triangle has five sides.", "false"),
    ("Ice is frozen water.", "true"), ("Cats can fly.", "false"),
    ("A week has ten days.", "false"), ("Bees can make honey.", "true"),
    ("The moon is bigger than the sun.", "false"), ("Plants need light to grow.", "true"),
    ("A square has four equal sides.", "true"), ("Dogs lay eggs.", "false"),
    ("Rain falls up into the sky.", "false"), ("An hour has 60 minutes.", "true"),
    ("Snow is hot.", "false"), ("Birds have feathers.", "true"),
    ("A cow is bigger than a mouse.", "true"), ("Bananas grow under the sea.", "false"),
    ("A candle can melt.", "true"), ("Winter is warmer than summer.", "false"),
    ("The heart pumps blood.", "true"), ("A wall can feel pain.", "false"),
    ("Rice is a food.", "true"), ("A clock tells the time.", "true"),
    ("Stones can grow taller.", "false"), ("Milk comes from cows.", "true"),
    ("A whale lives in the desert.", "false"), ("Books have pages.", "true"),
    ("The night is darker than the day.", "true"), ("Shoes are worn on the hands.", "false"),
]
def cand_classify():
    style = rng.randrange(3)
    if style == 0:
        if rng.random() < 0.5:
            s, ans = rng.choice(POS), "positive"
        else:
            s, ans = rng.choice(NEG), "negative"
        tpl = rng.choice([
            "Is this sentence positive or negative: {s}?",
            "Say if this is positive or negative: {s}",
            "Classify as positive or negative: {s}",
        ])
        return tpl.format(s=s), "", ans
    if style == 1:
        item, cat = rng.choice(CATEG)
        others = rng.sample([c for c in CATNAMES if c != cat], 2)
        opts = [cat] + others
        rng.shuffle(opts)
        return f"What category is a {item}: {opts[0]}, {opts[1]} or {opts[2]}?", "", cat
    s, ans = rng.choice(TRUEFALSE)
    tpl = rng.choice(["True or false: {s}", "Say true or false: {s}", "Is this true or false: {s}"])
    return tpl.format(s=s), "", ans


# ---- extract (300)
def cand_extract():
    style = rng.randrange(4)
    n = rng.choice(NAMES)
    if style == 0:
        act = rng.choice(["planted a tree", "sang a song", "read a book", "baked bread", "drew a map"])
        return f"Give only the name: {n} {act} today.", "", n
    if style == 1:
        num = rng.randint(2, 99)
        obj = rng.choice(OBJECTS)
        return f"Give only the number: {n} bought {num} {obj} at the shop.", "", num
    if style == 2:
        p = rng.choice(PLACES)
        act = rng.choice(["played", "sang", "read", "slept", "danced"])
        return f"Give only the place: {n} {act} at the {p}.", "", p
    food = rng.choice(["rice", "bread", "soup", "salad", "pasta", "curry"])
    when = rng.choice(["at noon", "at night", "this morning", "after school"])
    return f"What did {n} eat? {n} ate {food} {when}.", "", food


# ---- simple_qa (150) - plain contrast versions with sentence answers
def cand_simple_qa():
    body, word, sentence = rng.choice(FACTS)
    q = body if rng.random() < 0.5 else f"Please answer: {body}"
    return q, "", sentence


# ============================================================ REASONING DATA

def two_step_nums():
    a, b = rng.randint(11, 60), rng.randint(11, 60)
    c = rng.randint(2, min(a + b - 1, 40))
    return a, b, c


# ---- two_step (900)
def cand_two_step():
    a, b, c = two_step_nums()
    s1, ans = a + b, a + b - c
    style = rng.randrange(4)
    n = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    if style == 0:
        q = f"{n} has {a} {obj}. {n} gets {b} more and gives away {c}. How many {obj} now?"
    elif style == 1:
        q = f"{n} had {a} {obj}, found {b} more, then lost {c}. How many {obj} are left?"
    elif style == 2:
        q = f"Start with {a}. Add {b}. Take away {c}. What do you get?"
    else:
        q = f"What is {a} + {b} - {c}?"
    r = f"First, {a} + {b} = {s1}. Then, {s1} - {c} = {ans}."
    return q, r, ans


# ---- three_step (600)
def cand_three_step():
    a, b = rng.randint(5, 40), rng.randint(5, 40)
    c = rng.randint(2, min(a + b - 1, 30))
    d = rng.randint(2, 30)
    s1, s2, ans = a + b, a + b - c, a + b - c + d
    style = rng.randrange(3)
    n = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    if style == 0:
        q = f"{n} has {a} {obj}, gets {b}, loses {c}, then finds {d}. How many now?"
    elif style == 1:
        q = f"Start with {a}. Add {b}. Take away {c}. Add {d}. What is left?"
    else:
        q = f"What is {a} + {b} - {c} + {d}?"
    r = f"First, {a} + {b} = {s1}. Then, {s1} - {c} = {s2}. Then, {s2} + {d} = {ans}."
    return q, r, ans


# ---- mixed_ops (600)
def cand_mixed():
    a, b = rng.randint(2, 12), rng.randint(2, 12)
    c = rng.randint(2, 40)
    if rng.random() < 0.5:
        ans = a * b + c
        op, s1 = "+", a * b
        q_forms = [f"What is {a} * {b} + {c}?",
                   f"Multiply {a} by {b}, then add {c}.",
                   f"A box holds {a} rows of {b} seeds. You add {c} more seeds. How many seeds?"]
        r = f"First, {a} * {b} = {s1}. Then, {s1} + {c} = {ans}."
    else:
        s1 = a * b
        if s1 <= c:
            return None
        ans = s1 - c
        q_forms = [f"What is {a} * {b} - {c}?",
                   f"Multiply {a} by {b}, then take away {c}.",
                   f"A tray holds {a} rows of {b} beads. {c} beads fall off. How many are left?"]
        r = f"First, {a} * {b} = {s1}. Then, {s1} - {c} = {ans}."
    return rng.choice(q_forms), r, ans


# ---- comparison (400)
def cand_comparison():
    a, b = rng.randint(1, 99), rng.randint(1, 99)
    if a == b:
        return None
    if rng.random() < 0.5:
        q = rng.choice([f"Which is greater: {a} or {b}?", f"Which number is bigger, {a} or {b}?"])
        ans = max(a, b)
        r = f"{max(a,b)} is more than {min(a,b)}."
    else:
        q = rng.choice([f"Which is smaller: {a} or {b}?", f"Which number is less, {a} or {b}?"])
        ans = min(a, b)
        r = f"{min(a,b)} is less than {max(a,b)}."
    return q, r, ans


# ---- sorting (400)
def cand_sorting():
    k = rng.randint(3, 4)
    nums = rng.sample(range(1, 99), k)
    if rng.random() < 0.5:
        out = sorted(nums)
        q = f"Put these in order from small to big: {', '.join(map(str, nums))}."
        r = f"The smallest is {out[0]} and the biggest is {out[-1]}."
    else:
        out = sorted(nums, reverse=True)
        q = f"Put these in order from big to small: {', '.join(map(str, nums))}."
        r = f"The biggest is {out[0]} and the smallest is {out[-1]}."
    return q, r, ", ".join(map(str, out))


# ---- counting (400)
def cand_counting():
    if rng.random() < 0.5:
        k = rng.randint(3, 6)
        picks = rng.sample(ANIMALS, k)
        q = f"Count the animals in this list: {', '.join(picks)}."
        r = f"The list goes from {picks[0]} to {picks[-1]}."
        return q, r, k
    w = rng.choice(WORDS)
    if rng.random() < 0.5:
        vowels = [ch for ch in w if ch in "aeiou"]
        if not vowels:
            return None
        q = f"How many vowels are in the word {w}?"
        r = f"The vowels in {w} are {', '.join(vowels)}."
        return q, r, len(vowels)
    ch = rng.choice(sorted(set(w)))
    q = f"How many times does the letter {ch} appear in the word {w}?"
    r = f"Looking at each letter of {w}, the letter {ch} shows up {w.count(ch)} time" + ("s." if w.count(ch) > 1 else ".")
    return q, r, w.count(ch)


# ---- deduction (500) - generative syllogisms: group x member x property x names
# (grp_plural, grp_singular, members, property_stmt, property_verb_q)
DEDUCT_YES = [
    ("birds", "bird", ["crow", "swan", "dove", "parrot", "heron", "robin"], "have wings", "have wings"),
    ("birds", "bird", ["crow", "swan", "dove", "parrot", "heron", "robin"], "lay eggs", "lay eggs"),
    ("fish", "fish", ["trout", "salmon", "tuna", "carp", "cod"], "live in water", "live in water"),
    ("fish", "fish", ["trout", "salmon", "tuna", "carp", "cod"], "can swim", "swim"),
    ("dogs", "dog", ["beagle", "poodle", "husky", "pug"], "can bark", "bark"),
    ("cows", "cow", ["dairy cow", "brown cow", "farm cow"], "eat grass", "eat grass"),
    ("insects", "insect", ["ant", "bee", "moth", "beetle", "wasp"], "have six legs", "have six legs"),
    ("plants", "plant", ["fern", "rose", "tulip", "daisy", "vine"], "need light", "need light"),
    ("frogs", "frog", ["tree frog", "green frog", "pond frog"], "can jump", "jump"),
    ("trains", "train", ["metro", "steam train", "night train"], "run on rails", "run on rails"),
]
DEDUCT_NO = [
    ("rocks", "rock", ["stone", "pebble", "boulder"], "cannot swim", "swim"),
    ("chairs", "chair", ["stool", "bench", "armchair"], "cannot talk", "talk"),
    ("plants", "plant", ["fern", "rose", "cactus"], "cannot walk", "walk"),
    ("clouds", "cloud", ["storm cloud", "rain cloud"], "cannot sing", "sing"),
    ("spoons", "spoon", ["ladle", "teaspoon"], "cannot dance", "dance"),
    ("bricks", "brick", ["red brick", "clay brick"], "cannot fly", "fly"),
    ("statues", "statue", ["marble statue", "stone statue"], "cannot blink", "blink"),
    ("books", "book", ["notebook", "storybook"], "cannot eat", "eat"),
]
def cand_deduction():
    yes = rng.random() < 0.55
    grp, sing, members, prop, verb = rng.choice(DEDUCT_YES if yes else DEDUCT_NO)
    m = rng.choice(members)
    kw = "All" if yes else ""
    stmt = f"All {grp} {prop}" if yes else f"{grp.capitalize()} {prop}"
    if rng.random() < 0.5:
        q = f"{stmt}. A {m} is a {sing}. Does a {m} {verb}?" if yes else \
            f"{stmt}. A {m} is a {sing}. Can a {m} {verb}?"
        r = f"A {m} is a {sing}, and {stmt[0].lower()}{stmt[1:]}."
    else:
        n = rng.choice(NAMES)
        q = f"{n} has a {m}. {stmt}. Does {n}'s {m} {verb}?" if yes else \
            f"{n} has a {m}. {stmt}. Can {n}'s {m} {verb}?"
        r = f"A {m} is a {sing}, and {stmt[0].lower()}{stmt[1:]}."
    return q, r, "yes" if yes else "no"


# ---- patterns_r (500)
def cand_patterns():
    style = rng.randrange(4)
    if style == 0:  # arithmetic up
        start, step = rng.randint(1, 30), rng.randint(2, 9)
        seq = [start + step * k for k in range(4)]
        nxt = seq[-1] + step
        q = rng.choice([
            f"Find the ne{'x'}t number: {', '.join(map(str, seq))}.",
            f"The sequence is {', '.join(map(str, seq))}. What is ne{'x'}t?",
        ])
        r = f"The numbers go up by {step} each time. {seq[-1]} + {step} = {nxt}."
        return q, r, nxt
    if style == 1:  # arithmetic down
        step = rng.randint(2, 9)
        start = rng.randint(4 * step + 5, 99)
        seq = [start - step * k for k in range(4)]
        nxt = seq[-1] - step
        q = f"What follows: {', '.join(map(str, seq))}?"
        r = f"The numbers go down by {step} each time. {seq[-1]} - {step} = {nxt}."
        return q, r, nxt
    if style == 2:  # geometric
        start, mult = rng.randint(1, 9), rng.choice([2, 3])
        seq = [start * (mult ** k) for k in range(4)]
        nxt = seq[-1] * mult
        q = f"Each number is {mult} times the one before: {', '.join(map(str, seq))}. What comes then?"
        r = f"{seq[-1]} * {mult} = {nxt}."
        return q, r, nxt
    letters = "abcdefghijklmnopqrstuvwxyz"
    s = rng.randint(0, 21)
    seq = letters[s:s + 4]
    q = f"Find the ne{'x'}t letter: {', '.join(seq)}."
    r = f"The letters follow the alphabet. After {seq[-1]} comes {letters[s+4]}."
    return q, r, letters[s + 4]


# ---- reading_comp (400) - names/objects/colors all disjoint from eval's
RC_ANIMALS = ["goat", "hen", "duck", "frog", "pony", "lamb"]
RC_OBJECTS = ["kite", "sock", "lamp", "ring", "boat", "drum"]
def cand_reading():
    style = rng.randrange(3)
    n1, n2 = rng.sample(NAMES, 2)
    if style == 0:
        a1, a2 = rng.sample(RC_ANIMALS, 2)
        who = rng.choice([0, 1])
        q = f"{n1} keeps a {a1}. {n2} keeps a {a2}. What animal does {[n1, n2][who]} keep?"
        r = f"The story says {[n1, n2][who]} keeps a {[a1, a2][who]}."
        return q, r, [a1, a2][who]
    if style == 1:
        o = rng.choice(RC_OBJECTS)
        c1, c2 = rng.sample(COLORS, 2)
        who = rng.choice([0, 1])
        q = f"{n1}'s {o} is {c1}. {n2}'s {o} is {c2}. What color is {[n1, n2][who]}'s {o}?"
        r = f"The story says {[n1, n2][who]}'s {o} is {[c1, c2][who]}."
        return q, r, [c1, c2][who]
    a, b = rng.randint(1, 40), rng.randint(1, 40)
    if a == b:
        return None
    obj = rng.choice(OBJECTS)
    who = rng.choice(["fewer", "more"])
    if who == "more":
        ans = n1 if a > b else n2
    else:
        ans = n1 if a < b else n2
    q = f"{n1} owns {a} {obj}. {n2} owns {b} {obj}. Who owns {who} {obj}?"
    r = f"{a} and {b} are compared. {ans} owns {who}."
    return q, r, ans


# ---- multi_hop (300)
def cand_multi_hop():
    style = rng.randrange(3)
    n1, n2 = rng.sample(NAMES, 2)
    obj = rng.choice(OBJECTS)
    if style == 0:
        a, d = rng.randint(2, 40), rng.randint(2, 20)
        q = f"{n1} has {a} {obj}. {n2} has {d} more than {n1}. How many {obj} does {n2} have?"
        r = f"{n1} has {a}. So {n2} has {a} + {d} = {a + d}."
        return q, r, a + d
    if style == 1:
        a, d = rng.randint(10, 60), rng.randint(2, 9)
        if d >= a:
            return None
        q = f"{n1} has {a} {obj}. {n2} has {d} fewer than {n1}. How many {obj} does {n2} have?"
        r = f"{n1} has {a}. So {n2} has {a} - {d} = {a - d}."
        return q, r, a - d
    boxes, per = rng.randint(2, 9), rng.randint(2, 9)
    q = f"{n1} has {boxes} bags. Each bag holds {per} {obj}. How many {obj} in all?"
    r = f"{boxes} bags with {per} each means {boxes} * {per} = {boxes * per}."
    return q, r, boxes * per


# ============================================================ build the set
PLAN = [
    # instruction: 5000
    ("fmt_one_word", 600, cand_one_word),
    ("fmt_yes_no", 650, cand_yes_no),
    ("fmt_number_only", 500, cand_number_only),
    ("fmt_three_items", 300, cand_three_items),
    ("fmt_two_examples", 200, cand_two_examples),
    ("fmt_three_words", 100, cand_three_words),
    ("fmt_no_explanation", 300, cand_no_explanation),
    ("transform_case", 300, cand_case),
    ("transform_reverse", 200, cand_reverse),
    ("transform_count", 400, cand_count),
    ("transform_sort", 400, cand_sort),
    ("transform_extreme", 200, cand_extreme),
    ("classify", 350, cand_classify),
    ("extract", 350, cand_extract),
    ("simple_qa", 150, cand_simple_qa),
    # reasoning: 5000
    ("two_step", 900, cand_two_step),
    ("three_step", 600, cand_three_step),
    ("mixed_ops", 600, cand_mixed),
    ("comparison", 400, cand_comparison),
    ("sorting", 400, cand_sorting),
    ("counting", 400, cand_counting),
    ("deduction", 500, cand_deduction),
    ("patterns_r", 500, cand_patterns),
    ("reading_comp", 400, cand_reading),
    ("multi_hop", 300, cand_multi_hop),
]

INSTRUCTION_CATS = {c for c, _, _ in PLAN[:15]}
REASONING_CATS = {c for c, _, _ in PLAN[15:]}

for cat, target, fn in PLAN:
    fill(cat, target, fn)
    print(f"  {cat:22s} {target} done")

rng.shuffle(items)
# re-number ids after shuffle? No - ids must stay stable per category; keep them.

instr = sum(1 for i in items if i["category"] in INSTRUCTION_CATS)
reas = sum(1 for i in items if i["category"] in REASONING_CATS)
assert len(items) == 10000, len(items)
assert instr == 5000 and reas == 5000, (instr, reas)

OUT_JSON.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
OUT_TXT.write_text("\n\n".join(render(i["question"], i["reasoning"], i["answer"]) for i in items) + "\n", encoding="utf-8")

print(f"\nwrote {len(items)} examples ({instr} instruction + {reas} reasoning)")
print(f"txt size: {OUT_TXT.stat().st_size:,} bytes")
print("\nrejections:")
for k, v in rejections.items():
    print(f"  {k:14s} {v}")
print(f"whitelisted arithmetic-skeleton uses (disjoint numbers): {whitelisted_used}")
