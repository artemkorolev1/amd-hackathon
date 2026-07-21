#!/usr/bin/env python3
"""
build_training_data.py — Download and convert HF datasets into unified
training/validation sets for the AMD ACT II 8-category pipeline.

Outputs:
  data/eval/training-v1.json     — 1,600 questions (200/category)
  data/eval/validation-v1.json   — 400 questions (50/category)

Format: {"category": str, "prompt": str, "expected_answer": str,
         "source": str, "difficulty": str, "task_id": str}
"""

import json, os, random, re, hashlib
from pathlib import Path

random.seed(42)
_HERE = Path(__file__).resolve().parent.parent
OUT_DIR = _HERE / "data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_CAT = 200
VAL_PER_CAT = 50


def task_id(source, idx):
    """Deterministic task_id from source + index."""
    h = hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:12]
    return f"{source[:20]}-{h}"


def difficulty_from_prompt(text):
    """Simple heuristic: longer prompts get harder."""
    if len(text) < 100:
        return "easy"
    elif len(text) < 250:
        return "medium"
    else:
        return "hard"


def save_split(all_questions, name):
    random.shuffle(all_questions)
    train = []
    val = []
    # Track per-category counts
    cat_counts = {}
    for q in all_questions:
        cat = q["category"]
        cat_counts.setdefault(cat, {"train": 0, "val": 0})
        if cat_counts[cat]["val"] < VAL_PER_CAT:
            val.append(q)
            cat_counts[cat]["val"] += 1
        elif cat_counts[cat]["train"] < TRAIN_PER_CAT:
            train.append(q)
            cat_counts[cat]["train"] += 1
        # else: discard excess

    # Sort within each set
    for subset_name, subset in [("train", train), ("val", val)]:
        subset.sort(key=lambda x: (x["category"], x["task_id"]))
        print(f"  {subset_name}: {len(subset)} questions")
        cats = {}
        for q in subset:
            cats[q["category"]] = cats.get(q["category"], 0) + 1
        for c in sorted(cats):
            print(f"    {c}: {cats[c]}")

    # Write both splits
    out_train = OUT_DIR / f"training-{name}.json"
    out_val = OUT_DIR / f"validation-{name}.json"
    with open(out_train, "w") as f:
        json.dump(train, f, indent=2, ensure_ascii=False)
    with open(out_val, "w") as f:
        json.dump(val, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote {out_train.relative_to(_HERE)} ({len(train)} Q)")
    print(f"  Wrote {out_val.relative_to(_HERE)} ({len(val)} Q)")
    return train, val


# ─────────────────────────────────────────────────────────────────────
# T01 — FACTUAL
# ─────────────────────────────────────────────────────────────────────
def build_factual():
    from datasets import load_dataset
    questions = []

    # Natural Questions (open)
    try:
        nq = load_dataset("nq_open", split="validation")
        for i, row in enumerate(nq):
            q_text = row["question"]
            answer = row["answer"][0] if row["answer"] else ""
            if not answer:
                continue
            questions.append({
                "category": "factual",
                "prompt": q_text,
                "expected_answer": answer,
                "source": "nq_open",
                "difficulty": difficulty_from_prompt(q_text),
                "task_id": task_id("nq_open", i),
            })
        print(f"  nq_open: {len([q for q in questions if q['source']=='nq_open'])}")
    except Exception as e:
        print(f"  nq_open FAILED: {e}")

    # MMLU
    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test")
        for i, row in enumerate(mmlu):
            q_text = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            answer_text = choices[answer_idx] if choices and answer_idx < len(choices) else ""
            prompt = f"{q_text}\n\nChoices: {', '.join(choices)}"
            questions.append({
                "category": "factual",
                "prompt": prompt,
                "expected_answer": answer_text,
                "source": "mmlu",
                "difficulty": difficulty_from_prompt(q_text),
                "task_id": task_id("mmlu", i),
            })
        print(f"  mmlu: {len([q for q in questions if q['source']=='mmlu'])}")
    except Exception as e:
        print(f"  mmlu FAILED: {e}")

    # SQuAD v2
    try:
        squad = load_dataset("rajpurkar/squad_v2", split="validation")
        for i, row in enumerate(squad):
            if not row["answers"]["text"]:
                continue  # skip unanswerable
            q_text = row["question"]
            context = row["context"]
            answer = row["answers"]["text"][0]
            prompt = f"Context: {context}\n\nQuestion: {q_text}"
            questions.append({
                "category": "factual",
                "prompt": prompt,
                "expected_answer": answer,
                "source": "squad_v2",
                "difficulty": difficulty_from_prompt(q_text + context),
                "task_id": task_id("squad", i),
            })
        print(f"  squad_v2: {len([q for q in questions if q['source']=='squad_v2'])}")
    except Exception as e:
        print(f"  squad_v2 FAILED: {e}")

    # TruthfulQA
    try:
        tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        for i, row in enumerate(tqa):
            questions.append({
                "category": "factual",
                "prompt": row["question"],
                "expected_answer": row["best_answer"],
                "source": "truthfulqa",
                "difficulty": "hard",  # most TruthfulQA is counterfactual/adversarial
                "task_id": task_id("truthfulqa", i),
            })
        print(f"  truthfulqa: {len([q for q in questions if q['source']=='truthfulqa'])}")
    except Exception as e:
        print(f"  truthfulqa FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T02 — MATH
# ─────────────────────────────────────────────────────────────────────
def build_math():
    from datasets import load_dataset
    questions = []

    # GSM8K
    try:
        gsm8k = load_dataset("gsm8k", "main", split="test")
        for i, row in enumerate(gsm8k):
            q_text = row["question"]
            answer = row["answer"]
            final_num = re.search(r"####\s*(-?[\d,.]+)", answer)
            expected = final_num.group(1) if final_num else answer.split("####")[-1].strip()
            prompt = f"Solve: {q_text}"
            diff = "easy" if len(q_text) < 100 else ("medium" if len(q_text) < 200 else "hard")
            questions.append({
                "category": "math",
                "prompt": prompt,
                "expected_answer": expected,
                "source": "gsm8k",
                "difficulty": diff,
                "task_id": task_id("gsm8k", i),
            })
        print(f"  gsm8k: {len([q for q in questions if q['source']=='gsm8k'])}")
    except Exception as e:
        print(f"  gsm8k FAILED: {e}")

    # SVAMP
    try:
        sv = load_dataset("nguyen-brat/svamp", split="train")
        for i, row in enumerate(sv):
            q_text = row["question"]
            answer = row["answer"][0] if row["answer"] else ""
            prompt = f"Solve: {q_text}"
            diff = "easy" if len(q_text) < 100 else ("medium" if len(q_text) < 200 else "hard")
            questions.append({
                "category": "math",
                "prompt": prompt,
                "expected_answer": str(answer),
                "source": "svamp",
                "difficulty": diff,
                "task_id": task_id("svamp", i),
            })
        print(f"  svamp: {len([q for q in questions if q['source']=='svamp'])}")
    except Exception as e:
        print(f"  svamp FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T03 — SENTIMENT
# ─────────────────────────────────────────────────────────────────────
def build_sentiment():
    from datasets import load_dataset
    questions = []
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    # SST-2
    try:
        sst2 = load_dataset("glue", "sst2", split="train")
        for i, row in enumerate(sst2):
            text = row["sentence"]
            label = label_map.get(row["label"], "NEUTRAL")
            prompt = f"Classify the sentiment: \"{text}\""
            questions.append({
                "category": "sentiment",
                "prompt": prompt,
                "expected_answer": label,
                "source": "sst2",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("sst2", i),
            })
        print(f"  sst2: {len([q for q in questions if q['source']=='sst2'])}")
    except Exception as e:
        print(f"  sst2 FAILED: {e}")

    # IMDB
    try:
        imdb = load_dataset("stanfordnlp/imdb", split="train")
        for i, row in enumerate(imdb):
            text = row["text"]
            label = label_map.get(row["label"], "NEUTRAL")
            prompt = f"Classify the sentiment of this review: \"{text[:800]}\""
            questions.append({
                "category": "sentiment",
                "prompt": prompt,
                "expected_answer": label,
                "source": "imdb",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("imdb", i),
            })
        print(f"  imdb: {len([q for q in questions if q['source']=='imdb'])}")
    except Exception as e:
        print(f"  imdb FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T04 — SUMMARIZATION
# ─────────────────────────────────────────────────────────────────────
def build_summarization():
    from datasets import load_dataset
    questions = []

    # XSum
    try:
        xsum = load_dataset("EdinburghNLP/xsum", split="test")
        for i, row in enumerate(xsum):
            doc = row["document"]
            summary = row["summary"]
            prompt = f"Summarize the following article in 1-2 sentences:\n\n{doc[:1200]}"
            questions.append({
                "category": "summarization",
                "prompt": prompt,
                "expected_answer": summary,
                "source": "xsum",
                "difficulty": difficulty_from_prompt(doc),
                "task_id": task_id("xsum", i),
            })
        print(f"  xsum: {len([q for q in questions if q['source']=='xsum'])}")
    except Exception as e:
        print(f"  xsum FAILED: {e}")

    # CNN/DailyMail
    try:
        cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        for i, row in enumerate(cnn):
            article = row["article"]
            highlights = row["highlights"]
            prompt = f"Summarize the following news article:\n\n{article[:1200]}"
            questions.append({
                "category": "summarization",
                "prompt": prompt,
                "expected_answer": highlights,
                "source": "cnn_dailymail",
                "difficulty": difficulty_from_prompt(article),
                "task_id": task_id("cnndm", i),
            })
        print(f"  cnn_dailymail: {len([q for q in questions if q['source']=='cnn_dailymail'])}")
    except Exception as e:
        print(f"  cnn_dailymail FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T05 — NER
# ─────────────────────────────────────────────────────────────────────
def build_ner():
    from datasets import load_dataset
    questions = []

    # OntoNotes 5 — convert BIO tags to entity list format
    # Label mapping from tner/roberta-large-ontonotes5 model config
    # Key: tag_id -> (B_type, I_type) for tracking spans
    # Only entities where the label part is NOT "O"
    # Full mapping: 0=O, 1=B-CARDINAL, 2=B-DATE, 3=I-DATE, 4=B-PERSON, 5=I-PERSON,
    # 6=B-NORP, 7=B-GPE, 8=I-GPE, 9=B-LAW, 10=I-LAW, 11=B-ORG, 12=I-ORG,
    # 13=B-PERCENT, 14=I-PERCENT, 15=B-ORDINAL, 16=B-MONEY, 17=I-MONEY,
    # 18=B-WORK_OF_ART, 19=I-WORK_OF_ART, 20=B-FAC, 21=B-TIME, 22=I-CARDINAL,
    # 23=B-LOC, 24=B-QUANTITY, 25=I-QUANTITY, 26=I-NORP, 27=I-LOC,
    # 28=B-PRODUCT, 29=I-TIME, 30=B-EVENT, 31=I-EVENT, 32=I-FAC,
    # 33=B-LANGUAGE, 34=I-PRODUCT, 35=I-ORDINAL, 36=I-LANGUAGE
    tag_to_label = {
        1: "CARDINAL", 2: "DATE", 3: "DATE", 4: "PERSON", 5: "PERSON",
        6: "NORP", 7: "GPE", 8: "GPE", 9: "LAW", 10: "LAW",
        11: "ORG", 12: "ORG", 13: "PERCENT", 14: "PERCENT",
        15: "ORDINAL", 16: "MONEY", 17: "MONEY",
        18: "WORK_OF_ART", 19: "WORK_OF_ART",
        20: "FAC", 21: "TIME", 22: "CARDINAL",
        23: "LOC", 24: "QUANTITY", 25: "QUANTITY",
        26: "NORP", 27: "LOC", 28: "PRODUCT", 29: "TIME",
        30: "EVENT", 31: "EVENT", 32: "FAC",
        33: "LANGUAGE", 34: "PRODUCT", 35: "ORDINAL", 36: "LANGUAGE",
    }
    # B-tag IDs (start of a new entity)
    b_tags = {1, 2, 4, 6, 7, 9, 11, 13, 15, 16, 18, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 34, 35}

    try:
        onto = load_dataset("tner/ontonotes5", split="test")
        for i, row in enumerate(onto):
            tokens = row["tokens"]
            tags = row["tags"]
            entities = []
            current_label = None
            current_tokens = []
            for tok, tag_id in zip(tokens, tags):
                if tag_id == 0:
                    if current_tokens:
                        entities.append(f"{current_label}: {' '.join(current_tokens)}")
                        current_label = None
                        current_tokens = []
                    continue
                label = tag_to_label.get(tag_id, f"TYPE{tag_id}")
                if tag_id in b_tags:
                    if current_tokens:
                        entities.append(f"{current_label}: {' '.join(current_tokens)}")
                    current_label = label
                    current_tokens = [tok]
                elif current_label and current_label == label:
                    current_tokens.append(tok)
                else:
                    if current_tokens:
                        entities.append(f"{current_label}: {' '.join(current_tokens)}")
                    current_label = label
                    current_tokens = [tok]
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")

            if not entities:
                continue

            text = " ".join(tokens)
            prompt = f"Extract all named entities from the following text:\n\n{text[:600]}"
            expected = "\n".join(entities)

            questions.append({
                "category": "ner",
                "prompt": prompt,
                "expected_answer": expected,
                "source": "ontonotes5",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("onto5", i),
            })
        print(f"  ontonotes5: {len([q for q in questions if q['source']=='ontonotes5'])}")
    except Exception as e:
        print(f"  ontonotes5 FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T06 — CODE GENERATION
# ─────────────────────────────────────────────────────────────────────
def build_code_gen():
    from datasets import load_dataset
    questions = []

    # HumanEval
    try:
        he = load_dataset("openai_humaneval", split="test")
        for i, row in enumerate(he):
            prompt = row["prompt"]
            solution = row["canonical_solution"]
            questions.append({
                "category": "code_gen",
                "prompt": f"Write a Python function:\n\n{prompt}",
                "expected_answer": solution.strip(),
                "source": "humaneval",
                "difficulty": "hard",
                "task_id": task_id("humaneval", i),
            })
        print(f"  humaneval: {len([q for q in questions if q['source']=='humaneval'])}")
    except Exception as e:
        print(f"  humaneval FAILED: {e}")

    # MBPP
    try:
        mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
        for i, row in enumerate(mbpp):
            prompt = row["text"]
            code = row["code"]
            questions.append({
                "category": "code_gen",
                "prompt": f"Write a Python function:\n\n{prompt}",
                "expected_answer": code.strip(),
                "source": "mbpp",
                "difficulty": difficulty_from_prompt(prompt),
                "task_id": task_id("mbpp", i),
            })
        print(f"  mbpp: {len([q for q in questions if q['source']=='mbpp'])}")
    except Exception as e:
        print(f"  mbpp FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T07 — CODE DEBUG
# ─────────────────────────────────────────────────────────────────────
def build_code_debug():
    from datasets import load_dataset
    questions = []

    # HumanEvalPack (buggy → fixed)
    try:
        hep = load_dataset("bigcode/humanevalpack", "python", split="test")
        for i, row in enumerate(hep):
            buggy = row["buggy_solution"]
            fixed = row["canonical_solution"]
            docstring = row.get("docstring", "")
            instruction = row.get("instruction", "")
            task_desc = instruction or docstring or "Fix the bug in the following code"
            prompt = f"Fix the bug in this Python function:\n\n{buggy.strip()}\n\nTask: {task_desc}"
            questions.append({
                "category": "code_debug",
                "prompt": prompt,
                "expected_answer": fixed.strip(),
                "source": "humanevalpack",
                "difficulty": "hard",
                "task_id": task_id("hep", i),
            })
        print(f"  humanevalpack: {len([q for q in questions if q['source']=='humanevalpack'])}")
    except Exception as e:
        print(f"  humanevalpack FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T08 — LOGIC
# ─────────────────────────────────────────────────────────────────────
def build_logic():
    from datasets import load_dataset
    questions = []

    # LogiQA
    try:
        logiqa = load_dataset("lucasmccabe/logiqa", split="test")
        for i, row in enumerate(logiqa):
            context = row["context"]
            query = row["query"]
            options = row["options"]
            correct = row["correct_option"]
            answer_text = options[correct] if options and correct < len(options) else str(correct)
            prompt = f"{context}\n\nQuestion: {query}\n\nOptions:\n" + "\n".join(f"{j}. {o}" for j, o in enumerate(options))
            questions.append({
                "category": "logic",
                "prompt": prompt,
                "expected_answer": f"{correct}. {answer_text}",
                "source": "logiqa",
                "difficulty": "hard",
                "task_id": task_id("logiqa", i),
            })
        print(f"  logiqa: {len([q for q in questions if q['source']=='logiqa'])}")
    except Exception as e:
        print(f"  logiqa FAILED: {e}")

    # Zebra Logic Bench
    try:
        zebra = load_dataset("allenai/zebra_logic_bench", "grid_mode", split="test")
        for i, row in enumerate(zebra):
            puzzle = row["puzzle"]
            solution = row["solution"]
            prompt = f"Solve the following logic puzzle:\n\n{puzzle}"
            # Format solution as text
            if isinstance(solution, dict):
                expected = json.dumps(solution, ensure_ascii=False)
            else:
                expected = str(solution)
            questions.append({
                "category": "logic",
                "prompt": prompt,
                "expected_answer": expected[:500],
                "source": "zebra_logic_bench",
                "difficulty": "hard",
                "task_id": task_id("zebra", i),
            })
        print(f"  zebra_logic_bench: {len([q for q in questions if q['source']=='zebra_logic_bench'])}")
    except Exception as e:
        print(f"  zebra_logic_bench FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Building training/validation dataset from HF sources")
    print("=" * 60)

    all_questions = []

    print("\n--- T01: Factual ---")
    all_questions.extend(build_factual())

    print("\n--- T02: Math ---")
    all_questions.extend(build_math())

    print("\n--- T03: Sentiment ---")
    all_questions.extend(build_sentiment())

    print("\n--- T04: Summarization ---")
    all_questions.extend(build_summarization())

    print("\n--- T05: NER ---")
    all_questions.extend(build_ner())

    print("\n--- T06: Code Generation ---")
    all_questions.extend(build_code_gen())

    print("\n--- T07: Code Debug ---")
    all_questions.extend(build_code_debug())

    print("\n--- T08: Logic ---")
    all_questions.extend(build_logic())

    print(f"\nTotal raw questions collected: {len(all_questions)}")

    # Per-category stats before splitting
    cat_counts = {}
    for q in all_questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print("\nPer-category totals:")
    for c in sorted(cat_counts):
        print(f"  {c}: {cat_counts[c]}")

    # Save as v1
    print("\n--- Splitting into training/validation (v1) ---")
    save_split(all_questions, "v1")

    print("\nDone.")
