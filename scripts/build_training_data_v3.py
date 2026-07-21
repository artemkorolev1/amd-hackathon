#!/usr/bin/env python3
"""
build_training_data_v3.py — Compact 200-question training/validation set.

Constraints:
- Prompt: 50-400 chars (soft), max 500 (hard)
- Answer: 2-150 chars (soft), max 300 (hard)
- 25 per category → 200 total
- Split: 150 train + 50 val
- Only short-form source datasets (no CNN/DM, no long LogiQA, no long SQuAD)
"""

import json, os, random, re, hashlib
from pathlib import Path

random.seed(42)
_HERE = Path(__file__).resolve().parent.parent
OUT_DIR = _HERE / "data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_CAT = 25
TRAIN_PER_CAT = 19
VAL_PER_CAT = 6

PROMPT_MAX = 500
ANSWER_MAX = 300


def task_id(source, idx):
    h = hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:12]
    return f"{source[:20]}-{h}"


def clean_prompt(text):
    text = text.replace("\ufffd", "?").strip()
    return text[:PROMPT_MAX]


def clean_answer(text):
    text = text.replace("\ufffd", "?").strip()
    return text[:ANSWER_MAX]


def save_split(all_questions, name):
    # Sort by category for stable splitting
    all_questions.sort(key=lambda x: (x["category"], x["task_id"]))

    train = []
    val = []
    for cat in sorted(set(q["category"] for q in all_questions)):
        cat_qs = [q for q in all_questions if q["category"] == cat]
        random.shuffle(cat_qs)
        train.extend(cat_qs[:TRAIN_PER_CAT])
        val.extend(cat_qs[TRAIN_PER_CAT:TRAIN_PER_CAT + VAL_PER_CAT])

    train.sort(key=lambda x: (x["category"], x["task_id"]))
    val.sort(key=lambda x: (x["category"], x["task_id"]))

    out_train = OUT_DIR / f"training-{name}.json"
    out_val = OUT_DIR / f"validation-{name}.json"
    with open(out_train, "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(out_val, "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {out_train.relative_to(_HERE)} ({len(train)} Q)")
    print(f"  Wrote {out_val.relative_to(_HERE)} ({len(val)} Q)")
    for subset_name, subset in [("train", train), ("val", val)]:
        print(f"  {subset_name}:")
        cats = {}
        for q in subset:
            cats[q["category"]] = cats.get(q["category"], 0) + 1
        for c in sorted(cats):
            print(f"    {c}: {cats[c]}")
    return train, val


# ─────────────────────────────────────────────────────────────────────
# FACTUAL — nq_open only (short Q&A)
# ─────────────────────────────────────────────────────────────────────
def build_factual():
    from datasets import load_dataset
    questions = []
    try:
        nq = load_dataset("nq_open", split="validation")
        for i, row in enumerate(nq):
            q = row["question"]
            a = row["answer"][0] if row["answer"] else ""
            if not q or not a:
                continue
            if len(q) > PROMPT_MAX or len(a) > ANSWER_MAX:
                continue
            if len(q) < 20:
                continue
            questions.append({
                "category": "factual", "prompt": clean_prompt(q),
                "expected_answer": clean_answer(a), "source": "nq_open",
                "difficulty": "easy" if len(q) < 80 else ("medium" if len(q) < 200 else "hard"),
                "task_id": task_id("nq", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  nq_open: {len([q for q in questions if q['source']=='nq_open'])}")
    except Exception as e:
        print(f"  nq_open FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# MATH — GSM8K (short, final number answer)
# ─────────────────────────────────────────────────────────────────────
def build_math():
    from datasets import load_dataset
    questions = []
    try:
        gsm8k = load_dataset("gsm8k", "main", split="test")
        for i, row in enumerate(gsm8k):
            q = row["question"]
            full_a = row["answer"]
            if len(q) > PROMPT_MAX:
                continue
            if len(q) < 30:
                continue
            final_num = re.search(r"####\s*(-?[\d,.]+)", full_a)
            expected = final_num.group(1) if final_num else full_a.split("####")[-1].strip()
            if len(expected) > 20:
                continue
            prompt = f"Solve: {q}"
            questions.append({
                "category": "math", "prompt": prompt,
                "expected_answer": expected, "source": "gsm8k",
                "difficulty": "easy" if len(q) < 100 else ("medium" if len(q) < 200 else "hard"),
                "task_id": task_id("gsm8k", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  gsm8k: {len([q for q in questions if q['source']=='gsm8k'])}")
    except Exception as e:
        print(f"  gsm8k FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# SENTIMENT — SST-2 only (short sentences)
# ─────────────────────────────────────────────────────────────────────
def build_sentiment():
    from datasets import load_dataset
    questions = []
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}
    try:
        sst2 = load_dataset("glue", "sst2", split="train")
        for i, row in enumerate(sst2):
            text = row["sentence"]
            label = label_map.get(row["label"], "NEUTRAL")
            if len(text) > PROMPT_MAX or len(text) < 10:
                continue
            # Mix: half ask for justification
            if i % 2 == 0:
                prompt = f"Classify the sentiment as POSITIVE or NEGATIVE. Explain briefly.\n\n\"{text}\""
                expected = f"Sentiment: {label}"
            else:
                prompt = f"Classify the sentiment: \"{text}\""
                expected = label
            if len(prompt) > PROMPT_MAX:
                continue
            questions.append({
                "category": "sentiment", "prompt": prompt,
                "expected_answer": expected, "source": "sst2",
                "difficulty": "easy" if len(text) < 60 else ("medium" if len(text) < 150 else "hard"),
                "task_id": task_id("sst2", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  sst2: {len([q for q in questions if q['source']=='sst2'])}")
    except Exception as e:
        print(f"  sst2 FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# SUMMARIZATION — XSum only (short articles, 1-sentence summaries)
# ─────────────────────────────────────────────────────────────────────
def build_summarization():
    from datasets import load_dataset
    questions = []
    try:
        xsum = load_dataset("EdinburghNLP/xsum", split="test")
        for i, row in enumerate(xsum):
            doc = row["document"]
            summary = row["summary"]
            if len(summary) > ANSWER_MAX or len(summary) < 10:
                continue
            if len(doc) > 800:
                doc = doc[:800]
            # QUALITY GUARD: Skip articles that are too short to contain real content
            # XSum has known issues where some articles are only a timestamp/dateline
            if len(doc) < 100:
                continue
            # Skip articles ending with "..." which indicates truncation in source
            if doc.rstrip().endswith("..."):
                continue
            prompt = f"Summarize: {doc}"
            if len(prompt) > PROMPT_MAX:
                continue
            questions.append({
                "category": "summarization", "prompt": prompt,
                "expected_answer": summary, "source": "xsum",
                "difficulty": "hard",
                "task_id": task_id("xsum", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  xsum: {len([q for q in questions if q['source']=='xsum'])}")
    except Exception as e:
        print(f"  xsum FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# NER — tweetner7 + wnut (short text, social media)
# ─────────────────────────────────────────────────────────────────────
def _convert_tner_tags(tokens, tags, label_names):
    entities = []
    current_label = None
    current_tokens = []
    for tok, tag_id in zip(tokens, tags):
        raw_label = label_names[tag_id] if tag_id < len(label_names) else "O"
        if raw_label == "O":
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
                current_label = None
                current_tokens = []
            continue
        if raw_label.startswith("B-"):
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
            current_label = raw_label[2:]
            current_tokens = [tok]
        elif raw_label.startswith("I-"):
            et = raw_label[2:]
            if current_label == et:
                current_tokens.append(tok)
            else:
                if current_tokens:
                    entities.append(f"{current_label}: {' '.join(current_tokens)}")
                current_label = et
                current_tokens = [tok]
        else:
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
            current_label = raw_label
            current_tokens = [tok]
    if current_tokens:
        entities.append(f"{current_label}: {' '.join(current_tokens)}")
    return entities


def build_ner():
    from datasets import load_dataset
    questions = []

    try:
        tweet7 = load_dataset("tner/tweetner7", split="test_2021")
        label_names = tweet7.features["tags"].feature.names
        for i, row in enumerate(tweet7):
            entities = _convert_tner_tags(row["tokens"], row["tags"], label_names)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            if len(text) > PROMPT_MAX:
                continue
            expected = "\n".join(entities)
            if len(expected) > ANSWER_MAX:
                continue
            prompt = f"Extract entities: {text}"
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": expected, "source": "tweetner7",
                "difficulty": "easy" if len(entities) <= 2 else "medium",
                "task_id": task_id("tweet7", i),
            })
            if len(questions) >= TARGET_PER_CAT:
                break
        print(f"  tweetner7: {len([q for q in questions if q['source']=='tweetner7'])}")
    except Exception as e:
        print(f"  tweetner7 FAILED: {e}")

    try:
        wnut = load_dataset("tner/wnut2017", split="test")
        wnut_labels = [
            "O", "B-person", "I-person", "B-location", "I-location",
            "B-corporation", "I-corporation", "B-creative_work", "I-creative_work",
            "B-group", "I-group", "B-product", "I-product",
        ]
        for i, row in enumerate(wnut):
            entities = _convert_tner_tags(row["tokens"], row["tags"], wnut_labels)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            if len(text) > PROMPT_MAX:
                continue
            expected = "\n".join(entities)
            if len(expected) > ANSWER_MAX:
                continue
            prompt = f"Extract entities: {text}"
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": expected, "source": "wnut2017",
                "difficulty": "easy" if len(entities) <= 2 else "medium",
                "task_id": task_id("wnut17", i),
            })
            if len([q for q in questions if q['source']=='wnut2017']) >= TARGET_PER_CAT - len([q for q in questions if q['source']=='tweetner7']):
                break
        print(f"  wnut2017: {len([q for q in questions if q['source']=='wnut2017'])}")
    except Exception as e:
        print(f"  wnut2017 FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# CODE GEN — MBPP only (short function specs)
# ─────────────────────────────────────────────────────────────────────
def build_code_gen():
    from datasets import load_dataset
    questions = []
    try:
        mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
        for i, row in enumerate(mbpp):
            text = row["text"]
            code = row["code"].strip()
            if len(text) > PROMPT_MAX or len(code) > ANSWER_MAX:
                continue
            if len(code) < 10:
                continue
            prompt = f"Write a Python function: {text}"
            questions.append({
                "category": "code_gen", "prompt": prompt,
                "expected_answer": code, "source": "mbpp",
                "difficulty": "easy" if len(text) < 100 else ("medium" if len(text) < 200 else "hard"),
                "task_id": task_id("mbpp", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  mbpp: {len([q for q in questions if q['source']=='mbpp'])}")
    except Exception as e:
        print(f"  mbpp FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# CODE DEBUG — HumanEvalPack (short buggy snippets)
# ─────────────────────────────────────────────────────────────────────
def build_code_debug():
    from datasets import load_dataset
    questions = []
    try:
        hep = load_dataset("bigcode/humanevalpack", "python", split="test")
        for i, row in enumerate(hep):
            buggy = row["buggy_solution"].strip()
            fixed = row["canonical_solution"].strip()
            instruction = row.get("instruction", "") or row.get("docstring", "")
            if len(buggy) > PROMPT_MAX - 80 or len(fixed) > ANSWER_MAX:
                continue
            if not instruction:
                continue
            prompt = f"Fix this Python function:\n{buggy}\n\nTask: {instruction}"
            if len(prompt) > PROMPT_MAX:
                continue
            questions.append({
                "category": "code_debug", "prompt": prompt,
                "expected_answer": fixed, "source": "humanevalpack",
                "difficulty": "hard",
                "task_id": task_id("hep", i),
            })
            if len(questions) >= TARGET_PER_CAT * 3:
                break
        print(f"  humanevalpack: {len([q for q in questions if q['source']=='humanevalpack'])}")
    except Exception as e:
        print(f"  humanevalpack FAILED: {e}")
    return questions


# ─────────────────────────────────────────────────────────────────────
# LOGIC — short LogiQA + Zebra (filtered by length)
# ─────────────────────────────────────────────────────────────────────
def build_logic():
    from datasets import load_dataset
    questions = []

    try:
        logiqa = load_dataset("lucasmccabe/logiqa", split="test")
        for i, row in enumerate(logiqa):
            context = row["context"]
            query = row["query"]
            options = row["options"]
            correct = row["correct_option"]
            if len(context) > 400:
                continue  # skip long passages
            answer_text = options[correct] if options and correct < len(options) else str(correct)
            prompt = f"{context}\n\nQ: {query}"
            answer = f"{correct}. {answer_text}"
            if len(prompt) > PROMPT_MAX or len(answer) > ANSWER_MAX:
                continue
            questions.append({
                "category": "logic", "prompt": clean_prompt(prompt),
                "expected_answer": clean_answer(answer), "source": "logiqa",
                "difficulty": "hard",
                "task_id": task_id("logiqa", i),
            })
            if len(questions) >= TARGET_PER_CAT:
                break
        print(f"  logiqa: {len([q for q in questions if q['source']=='logiqa'])}")
    except Exception as e:
        print(f"  logiqa FAILED: {e}")

    # Fill remaining with short zebra
    try:
        zebra = load_dataset("allenai/zebra_logic_bench", "grid_mode", split="test")
        for i, row in enumerate(zebra):
            puzzle = row["puzzle"]
            if len(puzzle) > PROMPT_MAX:
                continue
            answer = str(row["solution"])
            if len(answer) > ANSWER_MAX:
                continue
            if "___" in answer[:50]:
                continue  # skip unsolved templates
            prompt = f"Solve: {puzzle[:300]}"
            questions.append({
                "category": "logic", "prompt": prompt,
                "expected_answer": answer[:ANSWER_MAX], "source": "zebra",
                "difficulty": "hard",
                "task_id": task_id("zebra", i),
            })
            if len([q for q in questions if 'zebra' in q['source']]) >= TARGET_PER_CAT:
                break
        print(f"  zebra: {len([q for q in questions if 'zebra' in q['source']])}")
    except Exception as e:
        print(f"  zebra FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building compact training/validation set v3 (200 total)")
    print("=" * 60)

    all_questions = []

    print("\n--- Factual (nq_open) ---")
    all_questions.extend(build_factual())

    print("\n--- Math (GSM8K) ---")
    all_questions.extend(build_math())

    print("\n--- Sentiment (SST-2) ---")
    all_questions.extend(build_sentiment())

    print("\n--- Summarization (XSum) ---")
    all_questions.extend(build_summarization())

    print("\n--- NER (tweetner7 + wnut2017) ---")
    all_questions.extend(build_ner())

    print("\n--- Code Gen (MBPP) ---")
    all_questions.extend(build_code_gen())

    print("\n--- Code Debug (HumanEvalPack) ---")
    all_questions.extend(build_code_debug())

    print("\n--- Logic (short LogiQA + solved Zebra) ---")
    all_questions.extend(build_logic())

    print(f"\nTotal collected: {len(all_questions)}")

    cat_counts = {}
    for q in all_questions: cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print("\nPer-category:")
    for c in sorted(cat_counts): print(f"  {c}: {cat_counts[c]}")

    if all(v >= TARGET_PER_CAT for v in cat_counts.values()):
        print(f"\n--- Splitting ({TRAIN_PER_CAT} train + {VAL_PER_CAT} val per cat) ---")
        save_split(all_questions, "v3")
    else:
        print(f"\n⚠ Not enough questions per category — need {TARGET_PER_CAT} each")
        print("Saving what we have as raw pool")
        OUT_DIR.joinpath("training-v3.json").write_text(json.dumps(all_questions, indent=2))

    # Stats
    pl = [len(q["prompt"]) for q in all_questions]
    al = [len(q["expected_answer"]) for q in all_questions]
    import statistics
    print(f"\nLength stats:")
    print(f"  prompt: avg={statistics.mean(pl):.0f} median={statistics.median(pl):.0f} max={max(pl)}")
    print(f"  answer: avg={statistics.mean(al):.0f} median={statistics.median(al):.0f} max={max(al)}")

    print("\nDone.")
