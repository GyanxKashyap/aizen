"""
Evaluation harness for Aizen (Phase 1).

Run:  python3 eval.py             # evaluate aizen.pt on data/eval.json
      python3 eval.py --selftest  # sanity-check the answer checkers only

- Loads aizen.pt untouched (weights, vocab, config all from the checkpoint).
- Uses the EXISTING inference behavior: "Q: {q}\nA:" prompt, temperature 0.5,
  top-k 20, stop at newline - identical to chat.py.
- Reproducible: torch.manual_seed is fixed per question, so two runs of this
  script produce identical outputs and identical scores.
- Writes every individual result to results/eval_results.json.
"""

import argparse
import json
import re
from pathlib import Path

EVAL_PATH = Path("data/eval.json")
RESULTS_PATH = Path("results/eval_results.json")
WEIGHTS_PATH = "aizen.pt"

CATEGORY_ORDER = [
    "arithmetic", "multi_step_arithmetic", "logic", "patterns",
    "general_knowledge", "instruction_following", "reading_comprehension", "coding",
]
CATEGORY_DISPLAY = {
    "arithmetic": "Arithmetic",
    "multi_step_arithmetic": "Multi-step arithmetic",
    "logic": "Logic",
    "patterns": "Patterns",
    "general_knowledge": "General knowledge",
    "instruction_following": "Instruction following",
    "reading_comprehension": "Reading comprehension",
    "coding": "Coding",
}

# ------------------------------------------------------------ answer checkers

def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _words(s):
    return re.findall(r"[a-z']+", s.lower())


def _numbers(s):
    return re.findall(r"-?\d+", s)


def check_number(answer, expected, acceptable):
    """Correct if the LAST number in the model's answer equals the expected
    number (answers look like '12 + 34 = 46', so the last number is the result)."""
    nums = _numbers(answer)
    return bool(nums) and nums[-1] == str(int(expected))


def check_contains(answer, expected, acceptable):
    """Correct if any acceptable answer appears in the output (case/space
    insensitive; multiword answers match as substrings, single words as words)."""
    a = _norm(answer)
    words = set(_words(answer))
    for acc in acceptable:
        accn = _norm(acc)
        if " " in accn or not accn.isalpha():
            if accn in a:
                return True
        elif accn in words:
            return True
    return False


def check_yes_no(answer, expected, acceptable):
    """Output must contain yes or no (not both), matching the expected one."""
    words = set(_words(answer))
    has_yes, has_no = "yes" in words, "no" in words
    if has_yes == has_no:  # neither, or contradictory both
        return False
    return ("yes" if has_yes else "no") == _norm(expected)


def check_one_word_correct(answer, expected, acceptable):
    """Exactly one word in the output, and it must be the expected word."""
    words = _words(answer)
    return len(words) == 1 and words[0] in {_norm(a) for a in acceptable}


def check_three_items(answer, expected, acceptable):
    """Exactly three items, split on commas and 'and'."""
    parts = [p for p in re.split(r",| and ", answer.lower()) if p.strip(" .!?")]
    return len(parts) == 3


CHECKERS = {
    "number": check_number,
    "contains": check_contains,
    "yes_no": check_yes_no,
    "one_word_correct": check_one_word_correct,
    "three_items": check_three_items,
}

# ---------------------------------------------------------------- self-test

def selftest():
    cases = [
        ("number", "12 + 34 = 46", "46", ["46"], True),
        ("number", "The answer is 99.", "46", ["46"], False),
        ("number", "no numbers here", "5", ["5"], False),
        ("contains", "The capital of France is Paris.", "Paris", ["Paris"], True),
        ("contains", "It is New Delhi.", "New Delhi", ["New Delhi", "Delhi"], True),
        ("contains", "I like Rome.", "Paris", ["Paris"], False),
        ("contains", "carpet is nice", "cat", ["cat"], False),  # word-boundary check
        ("yes_no", "Yes, it is.", "yes", ["yes"], True),
        ("yes_no", "No.", "yes", ["yes"], False),
        ("yes_no", "yes and no", "yes", ["yes"], False),
        ("yes_no", "maybe", "yes", ["yes"], False),
        ("one_word_correct", "Paris", "Paris", ["Paris"], True),
        ("one_word_correct", "It is Paris.", "Paris", ["Paris"], False),
        ("one_word_correct", "Rome", "Paris", ["Paris"], False),
        ("three_items", "red, blue, green", "any three items", [], True),
        ("three_items", "red and blue", "any three items", [], False),
        ("three_items", "a, b, c, d", "any three items", [], False),
    ]
    failed = 0
    for method, answer, expected, acceptable, want in cases:
        got = CHECKERS[method](answer, expected, acceptable)
        status = "ok" if got == want else "FAIL"
        if got != want:
            failed += 1
        print(f"  [{status}] {method:18s} {answer!r} -> {got} (want {want})")
    print(f"\nselftest: {len(cases) - failed}/{len(cases)} passed")
    return failed == 0

# ---------------------------------------------------------------- evaluation

def main(weights_path=WEIGHTS_PATH, results_path=RESULTS_PATH):
    import torch
    from model import Aizen

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(weights_path, map_location=device)
    # Tokenizer plumbing only - the checkpoint decides how text becomes ids.
    # Char checkpoints (v0/v1) and BPE checkpoints (phase 4+) both evaluate
    # with IDENTICAL questions, sampling parameters, and scoring.
    if "tokenizer" in ckpt:
        from tokenizer import BPETokenizer
        tokz = BPETokenizer.from_state(ckpt["tokenizer"])
        encode_fn = tokz.encode
        decode_fn = tokz.decode
        stop_ids = {tokz.token_to_id["\n"]}
        vocab_n = tokz.vocab_size
    else:
        chars = ckpt["chars"]
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        def encode_fn(text):
            return [stoi[c] for c in text]  # KeyError on unknown, as before

        def decode_fn(ids):
            return "".join(itos[i] for i in ids)

        stop_ids = {stoi["\n"]}
        vocab_n = len(chars)
    model = Aizen(vocab_size=vocab_n, **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    @torch.no_grad()
    def generate(question, temperature=0.5, top_k=20, max_new_tokens=120):
        prompt = f"Q: {question}\nA:"
        try:
            ids = encode_fn(prompt)
        except KeyError as e:
            return None, f"unencodable character: {e}"
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits, _ = model(idx[:, -model.block_size:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if next_id.item() in stop_ids:
                break
            idx = torch.cat((idx, next_id), dim=1)
        out = decode_fn(idx[0].tolist())
        return out[len(prompt):].strip(), None

    items = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    results = []
    unencodable = 0
    print(f"Evaluating {len(items)} questions on {weights_path} ({device})...\n")
    for i, item in enumerate(items):
        torch.manual_seed(1337 + i)  # deterministic sampling per question
        answer, err = generate(item["question"])
        if err is not None:
            correct, answer = False, f"[{err}]"
            unencodable += 1
        else:
            correct = CHECKERS[item["method"]](
                answer, item["expected_answer"], item["acceptable_answers"])
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "model_answer": answer,
            "correct": correct,
            "evaluation_method": item["method"],
        })
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(items)} done")

    # ---------------------------------------------------------------- scores
    def score(rs):
        c = sum(1 for r in rs if r["correct"])
        return c, len(rs), (100.0 * c / len(rs) if rs else 0.0)

    total_c, total_n, total_pct = score(results)
    cat_scores = {}
    for cat in CATEGORY_ORDER:
        cat_scores[cat] = score([r for r in results if r["category"] == cat])

    print("\nAIZEN EVALUATION")
    print("================\n")
    print(f"Overall: {total_pct:.1f}% ({total_c}/{total_n})\n")
    print("Category:")
    for cat in CATEGORY_ORDER:
        c, n, pct = cat_scores[cat]
        print(f"{CATEGORY_DISPLAY[cat]:24s}{pct:5.1f}%  ({c}/{n})")
    if unencodable:
        print(f"\nUnencodable questions (chars outside vocab): {unencodable} (counted incorrect)")

    # ------------------------------------------------------- baseline summary
    print("\nAIZEN BASELINE\n")
    print("Model:                 Aizen 14.3M (8L / 384d / 6h / ctx 192, char-level)")
    print(f"Checkpoint:            {weights_path}")
    print(f"Overall accuracy:      {total_pct:.1f}%")
    for cat in CATEGORY_ORDER:
        c, n, pct = cat_scores[cat]
        print(f"{CATEGORY_DISPLAY[cat] + ':':23s}{pct:.1f}%")

    # ---------------------------------------------------------- top failures
    print("\nTOP FAILURES")
    print("============")
    for cat in CATEGORY_ORDER:
        fails = [r for r in results if r["category"] == cat and not r["correct"]]
        if not fails:
            continue
        print(f"\n[{CATEGORY_DISPLAY[cat]}] ({len(fails)} failures)")
        for r in fails[:3]:
            print(f"  Question: {r['question']}")
            print(f"  Aizen:    {r['model_answer']}")
            print(f"  Expected: {r['expected_answer']}\n")

    results_path = Path(results_path)
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps({
        "checkpoint": str(weights_path),
        "total": total_n,
        "correct": total_c,
        "overall_accuracy_pct": round(total_pct, 2),
        "category_accuracy_pct": {
            cat: round(cat_scores[cat][2], 2) for cat in CATEGORY_ORDER},
        "unencodable": unencodable,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true", help="test the answer checkers only")
    parser.add_argument("--checkpoint", default=str(WEIGHTS_PATH), help="model checkpoint to evaluate")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="where to write results json")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    main(args.checkpoint, args.out)
