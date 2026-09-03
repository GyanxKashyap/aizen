"""
Generate a Q&A + chit-chat training corpus for Aizen.

Produces data/qa.txt full of exchanges in a fixed format:

    Q: <question>
    A: <answer>

The model learns to complete "Q: ...\nA:" with an answer. We include:
  - small talk (greetings, how are you, name, jokes...)
  - facts (country capitals, planets, colors, animals, basic science)
  - arithmetic (addition, subtraction, small multiplication)

Every fact is emitted under MANY phrasings so the model learns the
meaning-ish pattern instead of one exact string.

Run:  python3 make_qa_data.py
"""

import random

random.seed(1337)

OUT_PATH = "data/qa.txt"

pairs = []  # list of (question, answer)


def add(q, a):
    pairs.append((q, a))


# ------------------------------------------------------------- small talk ---
smalltalk = {
    "Hello": ["Hello! How can I help you?", "Hi there!", "Hello! Nice to see you."],
    "Hi": ["Hi! How can I help?", "Hello there!", "Hey! What's up?"],
    "Hey": ["Hey! How are you?", "Hey there!"],
    "Good morning": ["Good morning! Hope you have a great day."],
    "Good night": ["Good night! Sleep well."],
    "How are you?": [
        "I am doing great, thanks for asking!",
        "I'm good! How about you?",
        "Doing well, thank you!",
    ],
    "How are you doing?": ["I'm doing great, thanks for asking!", "Pretty good! And you?"],
    "How is it going?": ["Going great! How about you?"],
    "What is your name?": ["My name is Aizen.", "I am Aizen, a small language model."],
    "Who are you?": ["I am Aizen, a tiny language model trained on a laptop."],
    "What are you?": ["I am a small language model called Aizen."],
    "Who made you?": ["I was trained by my owner on a MacBook."],
    "What can you do?": ["I can chat a little and answer simple questions about facts and math."],
    "Thank you": ["You're welcome!", "No problem!", "Happy to help!"],
    "Thanks": ["You're welcome!", "Anytime!"],
    "Bye": ["Goodbye! Talk to you later.", "Bye! Have a nice day."],
    "Goodbye": ["Goodbye! Take care."],
    "See you later": ["See you! Bye!"],
    "I love you": ["Aw, thanks! I'm just a tiny model, but I appreciate it."],
    "Are you a robot?": ["I am a computer program, a very small language model."],
    "Are you smart?": ["I am quite tiny, so I only know what I was trained on."],
    "Tell me a joke": [
        "Why did the computer go to the doctor? It had a virus!",
        "Why do programmers prefer dark mode? Because light attracts bugs!",
    ],
    "How old are you?": ["I was trained very recently, so I am basically a newborn."],
    "Where do you live?": ["I live inside a file called aizen on a MacBook."],
    "Do you sleep?": ["No, I only run when someone asks me a question."],
    "Are you human?": ["No, I am a small computer program."],
    "What is your favorite color?": ["I like blue. It is a calm color."],
    "Can you help me?": ["Sure! Ask me a simple question and I will try."],
    "What do you like?": ["I like answering questions. It is all I know how to do."],
    "Are you okay?": ["Yes, I am fine. Thanks for checking!"],
    "What is up?": ["Not much! Ready to answer your questions."],
    "Nice to meet you": ["Nice to meet you too!"],
    # casual / typo-style greetings people actually type
    "hii": ["Hi! How can I help?", "Hello! Nice to see you."],
    "hiii": ["Hi there!", "Hey! What's up?"],
    "hlo": ["Hello! How can I help you?"],
    "helo": ["Hello! How can I help you?"],
    "yo": ["Yo! What's up?", "Hey there!"],
    "sup": ["Not much! Ready to answer your questions."],
    "wassup": ["Not much! What about you?"],
    "hey bro": ["Hey! How are you?"],
    "hello bro": ["Hello! How can I help you?"],
    "hi bro": ["Hi! How can I help?"],
    "good evening": ["Good evening! How can I help?"],
    "ok": ["Okay! Anything else you want to ask?"],
    "okay": ["Okay! Anything else you want to ask?"],
    "lol": ["Glad you liked that!"],
    "nice": ["Thanks! Ask me something else."],
}
# repeat small talk so the model sees plenty of it relative to facts
for _ in range(30):
    for q, answers in smalltalk.items():
        add(q, random.choice(answers))

# ------------------------------------------------------------------ facts ---
capitals = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "England": "London", "Ireland": "Dublin",
    "Scotland": "Edinburgh", "Netherlands": "Amsterdam", "Belgium": "Brussels",
    "Switzerland": "Bern", "Austria": "Vienna", "Greece": "Athens",
    "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Poland": "Warsaw", "Russia": "Moscow",
    "Ukraine": "Kyiv", "Turkey": "Ankara", "Egypt": "Cairo",
    "Morocco": "Rabat", "Nigeria": "Abuja", "Kenya": "Nairobi",
    "South Africa": "Pretoria", "India": "New Delhi", "China": "Beijing",
    "Japan": "Tokyo", "South Korea": "Seoul", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Malaysia": "Kuala Lumpur",
    "Singapore": "Singapore", "Philippines": "Manila", "Pakistan": "Islamabad",
    "Bangladesh": "Dhaka", "Nepal": "Kathmandu", "Sri Lanka": "Colombo",
    "Iran": "Tehran", "Iraq": "Baghdad", "Israel": "Jerusalem",
    "Saudi Arabia": "Riyadh", "United Arab Emirates": "Abu Dhabi",
    "Qatar": "Doha", "Australia": "Canberra", "New Zealand": "Wellington",
    "Canada": "Ottawa", "Mexico": "Mexico City", "Brazil": "Brasilia",
    "Argentina": "Buenos Aires", "Chile": "Santiago", "Peru": "Lima",
    "Colombia": "Bogota", "Venezuela": "Caracas", "Cuba": "Havana",
    "the United States": "Washington, D.C.", "the USA": "Washington, D.C.",
}
capital_templates = [
    ("What is the capital of {c}?", "The capital of {c} is {a}."),
    ("Capital of {c}?", "{a}."),
    ("What's the capital of {c}?", "It is {a}."),
    ("Tell me the capital of {c}.", "The capital of {c} is {a}."),
    ("Which city is the capital of {c}?", "{a} is the capital of {c}."),
    ("Do you know the capital of {c}?", "Yes, it is {a}."),
]
for _ in range(20):
    for c, a in capitals.items():
        qt, at = random.choice(capital_templates)
        add(qt.format(c=c), at.format(c=c, a=a))

misc_facts = {
    "How many planets are in the solar system?": "There are eight planets in the solar system.",
    "What is the largest planet?": "Jupiter is the largest planet.",
    "What is the smallest planet?": "Mercury is the smallest planet.",
    "What is the closest planet to the sun?": "Mercury is the closest planet to the sun.",
    "What is the red planet?": "Mars is called the red planet.",
    "Which planet do we live on?": "We live on Earth.",
    "What is the sun?": "The sun is a star at the center of our solar system.",
    "What is the moon?": "The moon is Earth's natural satellite.",
    "How many days are in a year?": "There are 365 days in a year.",
    "How many days are in a week?": "There are seven days in a week.",
    "How many hours are in a day?": "There are 24 hours in a day.",
    "How many minutes are in an hour?": "There are 60 minutes in an hour.",
    "How many months are in a year?": "There are twelve months in a year.",
    "What is the largest ocean?": "The Pacific Ocean is the largest ocean.",
    "What is the longest river?": "The Nile is often called the longest river.",
    "What is the tallest mountain?": "Mount Everest is the tallest mountain.",
    "What is the largest animal?": "The blue whale is the largest animal.",
    "What is the fastest land animal?": "The cheetah is the fastest land animal.",
    "What is the king of the jungle?": "The lion is called the king of the jungle.",
    "What do bees make?": "Bees make honey.",
    "What do cows drink?": "Cows drink water.",
    "What sound does a dog make?": "A dog says woof.",
    "What sound does a cat make?": "A cat says meow.",
    "What color is the sky?": "The sky is blue.",
    "What color is grass?": "Grass is green.",
    "What color is the sun?": "The sun looks yellow.",
    "What color is snow?": "Snow is white.",
    "What color do you get by mixing blue and yellow?": "Blue and yellow make green.",
    "What color do you get by mixing red and blue?": "Red and blue make purple.",
    "What is water made of?": "Water is made of hydrogen and oxygen.",
    "What do plants need to grow?": "Plants need sunlight, water and soil to grow.",
    "What is H2O?": "H2O is water.",
    "At what temperature does water boil?": "Water boils at 100 degrees Celsius.",
    "At what temperature does water freeze?": "Water freezes at 0 degrees Celsius.",
    "How many legs does a spider have?": "A spider has eight legs.",
    "How many legs does an insect have?": "An insect has six legs.",
    "How many continents are there?": "There are seven continents.",
    "What is the largest continent?": "Asia is the largest continent.",
    "What language is spoken in France?": "French is spoken in France.",
    "What language is spoken in Spain?": "Spanish is spoken in Spain.",
    "What language is spoken in Germany?": "German is spoken in Germany.",
    "What language is spoken in Japan?": "Japanese is spoken in Japan.",
    "Who wrote Romeo and Juliet?": "William Shakespeare wrote Romeo and Juliet.",
    "Who painted the Mona Lisa?": "Leonardo da Vinci painted the Mona Lisa.",
    "What is two plus two?": "Two plus two is four.",
    "How many sides does a triangle have?": "A triangle has three sides.",
    "How many sides does a square have?": "A square has four sides.",
    "What shape is a ball?": "A ball is a sphere.",
    "What is the first letter of the alphabet?": "The first letter is A.",
    "What is the last letter of the alphabet?": "The last letter is Z.",
}
for _ in range(25):
    for q, a in misc_facts.items():
        add(q, a)

# ------------------------------------------------------------- arithmetic ---
add_templates = [
    ("What is {x} + {y}?", "{x} + {y} = {r}"),
    ("What is {x} plus {y}?", "{x} plus {y} is {r}."),
    ("How much is {x} + {y}?", "It is {r}."),
    ("{x} + {y} = ?", "{r}"),
    ("Add {x} and {y}.", "{x} + {y} = {r}"),
    ("{x}+{y}", "{x} + {y} = {r}"),
    ("{x}+{y}=?", "{x} + {y} = {r}"),
    ("what is {x}+{y}?", "{x} + {y} = {r}"),
]
sub_templates = [
    ("What is {x} - {y}?", "{x} - {y} = {r}"),
    ("What is {x} minus {y}?", "{x} minus {y} is {r}."),
    ("{x} - {y} = ?", "{r}"),
    ("{x}-{y}", "{x} - {y} = {r}"),
    ("{x}-{y}=?", "{x} - {y} = {r}"),
]
mul_templates = [
    ("What is {x} * {y}?", "{x} * {y} = {r}"),
    ("What is {x} times {y}?", "{x} times {y} is {r}."),
    ("{x} * {y} = ?", "{r}"),
    ("{x}*{y}", "{x} * {y} = {r}"),
    ("{x}*{y}=?", "{x} * {y} = {r}"),
]

for _ in range(9000):
    x, y = random.randint(0, 99), random.randint(0, 99)
    qt, at = random.choice(add_templates)
    add(qt.format(x=x, y=y), at.format(x=x, y=y, r=x + y))

for _ in range(7000):
    x, y = random.randint(0, 99), random.randint(0, 99)
    if y > x:
        x, y = y, x  # keep results non-negative
    qt, at = random.choice(sub_templates)
    add(qt.format(x=x, y=y), at.format(x=x, y=y, r=x - y))

for _ in range(6000):
    x, y = random.randint(0, 12), random.randint(0, 12)
    qt, at = random.choice(mul_templates)
    add(qt.format(x=x, y=y), at.format(x=x, y=y, r=x * y))

# -------------------------------------------------- lowercase augmentation ---
# people often type all-lowercase; teach the model both casings
pairs += [(q.lower(), a) for q, a in random.sample(pairs, int(0.4 * len(pairs)))]

# ------------------------------------------------------------------ write ---
random.shuffle(pairs)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for q, a in pairs:
        f.write(f"Q: {q}\nA: {a}\n\n")

n_chars = sum(len(q) + len(a) + 8 for q, a in pairs)
print(f"wrote {len(pairs):,} Q&A pairs (~{n_chars/1e6:.1f}M chars) to {OUT_PATH}")
