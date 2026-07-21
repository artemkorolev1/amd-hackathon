#!/usr/bin/env python3
"""Build summarization eval sets: train/val/hard_test from all training+validation files.

Pulls ALL summarization questions, deduplicates by prompt text,
and splits 70% train, 15% val, 15% hard test.
Saves to data/eval/summarization_{train,val,hard_test}.json
"""
import json
import os
import random

SEED = 42
random.seed(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILES = [
    "data/eval/training-v1.json",
    "data/eval/training-v2.json",
    "data/eval/training-v3.json",
    "data/eval/validation-v1.json",
    "data/eval/validation-v2.json",
    "data/eval/validation-v3.json",
]


def extract_summarization(filepath: str) -> list[dict]:
    full_path = os.path.join(PROJECT_ROOT, filepath)
    with open(full_path) as f:
        data = json.load(f)
    return [
        {
            "category": "summarization",
            "prompt": q["prompt"],
            "expected_answer": q.get("expected_answer", q.get("answer", "")),
            "source": q.get("source", ""),
            "difficulty": q.get("difficulty", "medium"),
            "task_id": q.get("task_id", ""),
        }
        for q in data
        if q.get("category") == "summarization"
    ]


def main():
    # Collect all summarization questions
    all_questions = []
    for fname in DATA_FILES:
        qs = extract_summarization(fname)
        print(f"  {fname}: {len(qs)} summarization questions")
        all_questions.extend(qs)

    print(f"\nTotal raw: {len(all_questions)}")

    # Dedup by prompt text (exact match)
    seen_prompts = set()
    deduped = []
    for q in all_questions:
        prompt = q["prompt"].strip()
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            deduped.append(q)
    print(f"After dedup: {len(deduped)} unique questions")

    # Shuffle for split
    random.shuffle(deduped)

    # Split 70/15/15
    n = len(deduped)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_set = deduped[:train_end]
    val_set = deduped[train_end:val_end]
    hard_test_set = deduped[val_end:]

    print(f"\nSplit:")
    print(f"  Train:     {len(train_set)} ({100*len(train_set)//n}%)")
    print(f"  Val:       {len(val_set)} ({100*len(val_set)//n}%)")
    print(f"  Hard test: {len(hard_test_set)} ({100*len(hard_test_set)//n}%)")

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "data", "eval")
    for name, data in [
        ("summarization_train.json", train_set),
        ("summarization_val.json", val_set),
        ("summarization_hard_test.json", hard_test_set),
    ]:
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {out_path} ({len(data)} questions)")

    # Stats
    print(f"\nWord count stats (train):")
    word_counts = [len(q["prompt"].split()) for q in train_set]
    print(f"  Min: {min(word_counts)}, Max: {max(word_counts)}, Avg: {sum(word_counts)/len(word_counts):.0f}")
    long = sum(1 for wc in word_counts if wc > 200)
    print(f"  Texts > 200 words: {long}/{len(train_set)} ({100*long//len(train_set)}%)")

    # Print a few samples
    print(f"\nSample questions (first 3):")
    for q in train_set[:3]:
        words = len(q["prompt"].split())
        print(f"  [{words} words] {q['prompt'][:100]}...")
        print(f"  → {q['expected_answer'][:80]}...")


if __name__ == "__main__":
    main()
