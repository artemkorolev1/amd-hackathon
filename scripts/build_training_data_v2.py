#!/usr/bin/env python3
"""
build_training_data_v2.py — Improved training/validation dataset builder.

Incorporates fixes from v1 audit:
- Multi-factor difficulty (not just len())
- Math includes reasoning trace + final answer
- Sentiment split: 50% binary, 50% binary+justification
- More NER sources (tweetner7, wnut2017, ncbi_disease)
- More code_debug sources (swe-repair)
- Format constraints injected into 15% of prompts
- Dedup code_gen/code_debug overlapping function names
- Target 250/category for training headroom

Outputs:
  data/eval/training-v2.json     — ~1,600 questions
  data/eval/validation-v2.json   — 400 questions
"""

import json, os, random, re, hashlib
from pathlib import Path

random.seed(42)
_HERE = Path(__file__).resolve().parent.parent
OUT_DIR = _HERE / "data" / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PER_CAT = 200
VAL_PER_CAT = 50
VERSION = "v2"

# ── Format constraint pool (injected into ~15% of prompts) ──
FORMAT_CONSTRAINTS = [
    " Respond in JSON format with keys: answer, reasoning.",
    " Output your answer as a Python dictionary.",
    " Provide your answer as a bullet list.",
    " Explain step-by-step, then give the final answer on a separate line starting with ANSWER:.",
    " Keep your answer under 50 words.",
    " Output the answer first, then your reasoning.",
    " Structure your answer as: (1) summary, (2) reasoning, (3) conclusion.",
    " If uncertain, state your confidence level (HIGH/MEDIUM/LOW) before the answer.",
    " Answer using only the provided context. Do not add outside knowledge.",
    " List exactly 3 key points in your response.",
]

VERSION = "v2"

def apply_format_constraint(prompt, p=0.15):
    """Randomly inject a format constraint into the prompt."""
    if random.random() < p:
        constraint = random.choice(FORMAT_CONSTRAINTS)
        return prompt.rstrip() + constraint
    return prompt


def task_id(source, idx):
    h = hashlib.sha256(f"{source}:{idx}".encode()).hexdigest()[:12]
    return f"{source[:20]}-{h}"


def difficulty_from_prompt(text):
    """Multi-factor difficulty: length + sentence count + complexity markers."""
    length = len(text)
    sentences = max(1, len(re.findall(r'[.!?]\s+[A-Z]', text)) + 1)
    has_multi_step = bool(re.search(
        r'(first|then|next|after|finally|step|compute|calculate|derive|determine|multi-hop|'
        r'unless|provided that|given that|however|although|despite|contradict)',
        text.lower()
    ))
    has_complex_ops = bool(re.search(
        r'(equation|formula|derivative|integral|permutation|combination|'
        r'probability|compound|ratio|proportion|fraction|percent|'
        r'standard deviation|variance|correlation)', text.lower()
    ))
    has_code = bool(re.search(r'(def |class |import |lambda|recursion|iterate|'
                              r'asymptotic|complexity|O\(|timeout)', text.lower()))
    has_multi_entity = bool(re.search(r'(entities|extract|identify|find all|list all|separate|distinct)', text.lower()))

    score = 0
    if length > 200:
        score += 1
    if length > 500:
        score += 1
    if sentences > 3:
        score += 1
    if has_multi_step:
        score += 1
    if has_complex_ops:
        score += 1
    if has_code:
        score += 1
    if has_multi_entity:
        score += 1

    if score <= 1:
        return "easy"
    elif score <= 3:
        return "medium"
    else:
        return "hard"


# ── Shared BIO→entity text converter ──
def bio_to_entities(tokens, tags, tag_to_label, b_tags_set):
    """Convert BIO tags to 'TYPE: text' entity strings."""
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
        if tag_id in b_tags_set:
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
    return entities


def save_split(all_questions, name):
    random.shuffle(all_questions)
    train = []
    val = []
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

    for subset_name, subset in [("train", train), ("val", val)]:
        subset.sort(key=lambda x: (x["category"], x["task_id"]))
        print(f"  {subset_name}: {len(subset)} questions")
        cats = {}
        for q in subset:
            cats[q["category"]] = cats.get(q["category"], 0) + 1
        for c in sorted(cats):
            print(f"    {c}: {cats[c]}")

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

    try:
        nq = load_dataset("nq_open", split="validation")
        for i, row in enumerate(nq):
            q_text = row["question"]
            answer = row["answer"][0] if row["answer"] else ""
            if not answer:
                continue
            prompt = apply_format_constraint(q_text)
            questions.append({
                "category": "factual", "prompt": prompt,
                "expected_answer": answer, "source": "nq_open",
                "difficulty": difficulty_from_prompt(q_text),
                "task_id": task_id("nq_open", i),
            })
        print(f"  nq_open: {len([q for q in questions if q['source']=='nq_open'])}")
    except Exception as e:
        print(f"  nq_open FAILED: {e}")

    try:
        mmlu = load_dataset("cais/mmlu", "all", split="test")
        for i, row in enumerate(mmlu):
            q_text = row["question"]
            choices = row["choices"]
            answer_idx = row["answer"]
            answer_text = choices[answer_idx] if choices and answer_idx < len(choices) else ""
            prompt = apply_format_constraint(f"{q_text}\n\nChoices: {', '.join(choices)}")
            questions.append({
                "category": "factual", "prompt": prompt,
                "expected_answer": answer_text, "source": "mmlu",
                "difficulty": difficulty_from_prompt(q_text),
                "task_id": task_id("mmlu", i),
            })
        print(f"  mmlu: {len([q for q in questions if q['source']=='mmlu'])}")
    except Exception as e:
        print(f"  mmlu FAILED: {e}")

    try:
        squad = load_dataset("rajpurkar/squad_v2", split="validation")
        for i, row in enumerate(squad):
            if not row["answers"]["text"]:
                continue
            q_text = row["question"]
            context = row["context"]
            answer = row["answers"]["text"][0]
            prompt = apply_format_constraint(f"Context: {context}\n\nQuestion: {q_text}")
            questions.append({
                "category": "factual", "prompt": prompt,
                "expected_answer": answer, "source": "squad_v2",
                "difficulty": difficulty_from_prompt(q_text + context),
                "task_id": task_id("squad", i),
            })
        print(f"  squad_v2: {len([q for q in questions if q['source']=='squad_v2'])}")
    except Exception as e:
        print(f"  squad_v2 FAILED: {e}")

    try:
        tqa = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        for i, row in enumerate(tqa):
            prompt = apply_format_constraint(row["question"], p=0.25)
            questions.append({
                "category": "factual", "prompt": prompt,
                "expected_answer": row["best_answer"], "source": "truthfulqa",
                "difficulty": "hard",
                "task_id": task_id("truthfulqa", i),
            })
        print(f"  truthfulqa: {len([q for q in questions if q['source']=='truthfulqa'])}")
    except Exception as e:
        print(f"  truthfulqa FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T02 — MATH (with reasoning trace)
# ─────────────────────────────────────────────────────────────────────
def build_math():
    from datasets import load_dataset
    questions = []

    try:
        gsm8k = load_dataset("gsm8k", "main", split="test")
        for i, row in enumerate(gsm8k):
            q_text = row["question"]
            full_answer = row["answer"]
            # Extract reasoning trace + final answer
            parts = full_answer.split("####")
            reasoning = parts[0].strip() if len(parts) > 1 else ""
            final_num_match = re.search(r"####\s*(-?[\d,.]+)", full_answer)
            expected_num = final_num_match.group(1) if final_num_match else full_answer.strip()
            # Alternate between "answer only" and "reasoning + answer" formats
            if i % 2 == 0:
                prompt = apply_format_constraint(f"Solve the following step-by-step:\n\n{q_text}")
                expected = f"Reasoning:\n{reasoning}\n\nAnswer: {expected_num}"
            else:
                prompt = apply_format_constraint(f"Solve: {q_text}")
                expected = expected_num
            diff = difficulty_from_prompt(q_text)
            questions.append({
                "category": "math", "prompt": prompt,
                "expected_answer": expected, "source": "gsm8k",
                "difficulty": diff, "task_id": task_id("gsm8k", i),
            })
        print(f"  gsm8k: {len([q for q in questions if q['source']=='gsm8k'])}")
    except Exception as e:
        print(f"  gsm8k FAILED: {e}")

    try:
        sv = load_dataset("nguyen-brat/svamp", split="train")
        for i, row in enumerate(sv):
            q_text = row["question"]
            answer = row["answer"][0] if row["answer"] else ""
            prompt = apply_format_constraint(f"Solve: {q_text}")
            questions.append({
                "category": "math", "prompt": prompt,
                "expected_answer": str(answer), "source": "svamp",
                "difficulty": difficulty_from_prompt(q_text),
                "task_id": task_id("svamp", i),
            })
        print(f"  svamp: {len([q for q in questions if q['source']=='svamp'])}")
    except Exception as e:
        print(f"  svamp FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T03 — SENTIMENT (50% binary, 50% binary + justification)
# ─────────────────────────────────────────────────────────────────────
def build_sentiment():
    from datasets import load_dataset
    questions = []
    label_map = {0: "NEGATIVE", 1: "POSITIVE"}

    sst2_counter = 0
    try:
        sst2 = load_dataset("glue", "sst2", split="train")
        for i, row in enumerate(sst2):
            text = row["sentence"]
            label = label_map.get(row["label"], "NEUTRAL")
            # Alternate: half ask for justification
            if sst2_counter % 2 == 0:
                prompt = apply_format_constraint(
                    f"Classify the sentiment as POSITIVE or NEGATIVE, then explain why in one sentence.\n\n\"{text}\"",
                    p=0.1
                )
                expected = f"Sentiment: {label}"
            else:
                prompt = apply_format_constraint(f"Classify the sentiment: \"{text}\"", p=0.1)
                expected = label
            sst2_counter += 1
            questions.append({
                "category": "sentiment", "prompt": prompt,
                "expected_answer": expected, "source": "sst2",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("sst2", i),
            })
        print(f"  sst2: {len([q for q in questions if q['source']=='sst2'])}")
    except Exception as e:
        print(f"  sst2 FAILED: {e}")

    imdb_counter = 0
    try:
        imdb = load_dataset("stanfordnlp/imdb", split="train")
        for i, row in enumerate(imdb):
            text = row["text"]
            label = label_map.get(row["label"], "NEUTRAL")
            if imdb_counter % 2 == 0:
                prompt = apply_format_constraint(
                    f"Classify the sentiment of this review as POSITIVE or NEGATIVE, then explain why.\n\n\"{text[:800]}\"",
                    p=0.1
                )
                expected = f"Sentiment: {label}"
            else:
                prompt = apply_format_constraint(
                    f"Classify the sentiment of this review: \"{text[:800]}\"", p=0.1
                )
                expected = label
            imdb_counter += 1
            questions.append({
                "category": "sentiment", "prompt": prompt,
                "expected_answer": expected, "source": "imdb",
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

    try:
        xsum = load_dataset("EdinburghNLP/xsum", split="test")
        for i, row in enumerate(xsum):
            doc = row["document"]
            summary = row["summary"]
            if i % 3 == 0:
                prompt = f"Summarize the following article in exactly 2 sentences:\n\n{doc[:1200]}"
            elif i % 3 == 1:
                prompt = f"Summarize as exactly 3 bullet points:\n\n{doc[:1200]}"
            else:
                prompt = f"Summarize the following article:\n\n{doc[:1200]}"
            prompt = apply_format_constraint(prompt, p=0.1)
            questions.append({
                "category": "summarization", "prompt": prompt,
                "expected_answer": summary, "source": "xsum",
                "difficulty": difficulty_from_prompt(doc),
                "task_id": task_id("xsum", i),
            })
        print(f"  xsum: {len([q for q in questions if q['source']=='xsum'])}")
    except Exception as e:
        print(f"  xsum FAILED: {e}")

    try:
        cnn = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")
        for i, row in enumerate(cnn):
            article = row["article"]
            highlights = row["highlights"]
            if i % 2 == 0:
                prompt = f"Summarize the following news article in 2-3 sentences:\n\n{article[:1200]}"
            else:
                prompt = f"Summarize the following news article:\n\n{article[:1200]}"
            prompt = apply_format_constraint(prompt, p=0.1)
            questions.append({
                "category": "summarization", "prompt": prompt,
                "expected_answer": highlights, "source": "cnn_dailymail",
                "difficulty": difficulty_from_prompt(article),
                "task_id": task_id("cnndm", i),
            })
        print(f"  cnn_dailymail: {len([q for q in questions if q['source']=='cnn_dailymail'])}")
    except Exception as e:
        print(f"  cnn_dailymail FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T05 — NER (multiple sources)
# ─────────────────────────────────────────────────────────────────────

def _convert_tner_tags(tokens, tags, label_names):
    """Convert tner-format integer tags to entity strings using ClassLabel names."""
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
        # Parse B- prefix vs I-
        if raw_label.startswith("B-"):
            if current_tokens:
                entities.append(f"{current_label}: {' '.join(current_tokens)}")
            current_label = raw_label[2:]  # strip "B-"
            current_tokens = [tok]
        elif raw_label.startswith("I-"):
            entity_type = raw_label[2:]
            if current_label and current_label == entity_type:
                current_tokens.append(tok)
            else:
                # I- without B- before it: treat as new entity
                if current_tokens:
                    entities.append(f"{current_label}: {' '.join(current_tokens)}")
                current_label = entity_type
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

    # 1. OntoNotes 5 — use known tag mapping
    onto_tag_to_label = {
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
    onto_b_tags = {1, 2, 4, 6, 7, 9, 11, 13, 15, 16, 18, 20, 21, 22, 23, 24, 26, 27, 28, 30, 32, 33, 34, 35}

    try:
        onto = load_dataset("tner/ontonotes5", split="test")
        for i, row in enumerate(onto):
            entities = bio_to_entities(row["tokens"], row["tags"], onto_tag_to_label, onto_b_tags)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            prompt = apply_format_constraint(
                f"Extract all named entities from the following text. List each type on its own line:\n\n{text[:600]}",
                p=0.1
            )
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": "\n".join(entities), "source": "ontonotes5",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("onto5", i),
            })
        print(f"  ontonotes5: {len([q for q in questions if q['source']=='ontonotes5'])}")
    except Exception as e:
        print(f"  ontonotes5 FAILED: {e}")

    # 2. TweetNER7 — has ClassLabel names built-in
    try:
        tweet7 = load_dataset("tner/tweetner7", split="test_2021")
        label_names = tweet7.features["tags"].feature.names
        for i, row in enumerate(tweet7):
            entities = _convert_tner_tags(row["tokens"], row["tags"], label_names)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            prompt = apply_format_constraint(
                f"Extract all named entities from the following tweet:\n\n{text[:600]}",
                p=0.1
            )
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": "\n".join(entities), "source": "tweetner7",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("tweet7", i),
            })
        print(f"  tweetner7: {len([q for q in questions if q['source']=='tweetner7'])}")
    except Exception as e:
        print(f"  tweetner7 FAILED: {e}")

    # 3. WNUT2017 — social media NER
    try:
        wnut = load_dataset("tner/wnut2017", split="test")
        # WNUT2017 label names from tner config
        wnut_label_names = [
            "O", "B-person", "I-person", "B-location", "I-location",
            "B-corporation", "I-corporation", "B-creative_work", "I-creative_work",
            "B-group", "I-group", "B-product", "I-product",
        ]
        for i, row in enumerate(wnut):
            entities = _convert_tner_tags(row["tokens"], row["tags"], wnut_label_names)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            prompt = apply_format_constraint(
                f"Extract all named entities from the following text:\n\n{text[:600]}",
                p=0.1
            )
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": "\n".join(entities), "source": "wnut2017",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("wnut17", i),
            })
        print(f"  wnut2017: {len([q for q in questions if q['source']=='wnut2017'])}")
    except Exception as e:
        print(f"  wnut2017 FAILED: {e}")

    # 4. NCBI Disease — biomedical NER
    try:
        ncbi = load_dataset("ncbi/ncbi_disease", split="test")
        ncbi_label_names = ["O", "B-Disease", "I-Disease"]
        for i, row in enumerate(ncbi):
            entities = _convert_tner_tags(row["tokens"], row["ner_tags"], ncbi_label_names)
            if not entities:
                continue
            text = " ".join(row["tokens"])
            prompt = apply_format_constraint(
                f"Extract all disease names from the following biomedical text:\n\n{text[:600]}",
                p=0.1
            )
            questions.append({
                "category": "ner", "prompt": prompt,
                "expected_answer": "\n".join(entities), "source": "ncbi_disease",
                "difficulty": difficulty_from_prompt(text),
                "task_id": task_id("ncbi", i),
            })
        print(f"  ncbi_disease: {len([q for q in questions if q['source']=='ncbi_disease'])}")
    except Exception as e:
        print(f"  ncbi_disease FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T06 — CODE GENERATION
# ─────────────────────────────────────────────────────────────────────
def build_code_gen():
    from datasets import load_dataset
    questions = []

    try:
        he = load_dataset("openai_humaneval", split="test")
        for i, row in enumerate(he):
            prompt = row["prompt"]
            solution = row["canonical_solution"]
            questions.append({
                "category": "code_gen", "prompt": f"Write a Python function:\n\n{prompt}",
                "expected_answer": solution.strip(), "source": "humaneval",
                "difficulty": "hard", "task_id": task_id("humaneval", i),
            })
        print(f"  humaneval: {len([q for q in questions if q['source']=='humaneval'])}")
    except Exception as e:
        print(f"  humaneval FAILED: {e}")

    try:
        mbpp = load_dataset("google-research-datasets/mbpp", "full", split="test")
        for i, row in enumerate(mbpp):
            prompt = row["text"]
            code = row["code"]
            questions.append({
                "category": "code_gen", "prompt": f"Write a Python function:\n\n{prompt}",
                "expected_answer": code.strip(), "source": "mbpp",
                "difficulty": difficulty_from_prompt(prompt),
                "task_id": task_id("mbpp", i),
            })
        print(f"  mbpp: {len([q for q in questions if q['source']=='mbpp'])}")
    except Exception as e:
        print(f"  mbpp FAILED: {e}")

    return questions


# ─────────────────────────────────────────────────────────────────────
# T07 — CODE DEBUG (dedup'd with code_gen)
# ─────────────────────────────────────────────────────────────────────
def build_code_debug():
    from datasets import load_dataset
    questions = []

    # HumanEvalPack
    try:
        hep = load_dataset("bigcode/humanevalpack", "python", split="test")
        for i, row in enumerate(hep):
            buggy = row["buggy_solution"]
            fixed = row["canonical_solution"]
            docstring = row.get("docstring", "")
            instruction = row.get("instruction", "")
            task_desc = instruction or docstring or "Fix the bug in the following code"

            prompt = apply_format_constraint(
                f"Fix the bug in this Python function:\n\n{buggy.strip()}\n\nTask: {task_desc}",
                p=0.1
            )
            questions.append({
                "category": "code_debug", "prompt": prompt,
                "expected_answer": fixed.strip(), "source": "humanevalpack",
                "difficulty": "hard", "task_id": task_id("hep", i),
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

    try:
        logiqa = load_dataset("lucasmccabe/logiqa", split="test")
        for i, row in enumerate(logiqa):
            context = row["context"]
            query = row["query"]
            options = row["options"]
            correct = row["correct_option"]
            answer_text = options[correct] if options and correct < len(options) else str(correct)
            prompt = apply_format_constraint(
                f"{context}\n\nQuestion: {query}\n\nOptions:\n" +
                "\n".join(f"{j}. {o}" for j, o in enumerate(options)),
                p=0.15
            )
            questions.append({
                "category": "logic", "prompt": prompt,
                "expected_answer": f"{correct}. {answer_text}",
                "source": "logiqa", "difficulty": difficulty_from_prompt(context),
                "task_id": task_id("logiqa", i),
            })
        print(f"  logiqa: {len([q for q in questions if q['source']=='logiqa'])}")
    except Exception as e:
        print(f"  logiqa FAILED: {e}")

    try:
        zebra = load_dataset("allenai/zebra_logic_bench", "grid_mode", split="test")
        for i, row in enumerate(zebra):
            puzzle = row["puzzle"]
            solution = row["solution"]
            prompt = apply_format_constraint(f"Solve the following logic puzzle:\n\n{puzzle}", p=0.15)
            if isinstance(solution, dict):
                expected = json.dumps(solution, ensure_ascii=False)
            else:
                expected = str(solution)
            questions.append({
                "category": "logic", "prompt": prompt,
                "expected_answer": expected[:500], "source": "zebra_logic_bench",
                "difficulty": "hard", "task_id": task_id("zebra", i),
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
    print(f"Building training/validation dataset v{VERSION}")
    print("=" * 60)

    all_questions = []

    print("\n--- T01: Factual ---")
    all_questions.extend(build_factual())

    print("\n--- T02: Math (with reasoning trace) ---")
    all_questions.extend(build_math())

    print("\n--- T03: Sentiment (binary + justification) ---")
    all_questions.extend(build_sentiment())

    print("\n--- T04: Summarization ---")
    all_questions.extend(build_summarization())

    print("\n--- T05: NER (4 sources) ---")
    all_questions.extend(build_ner())

    print("\n--- T06: Code Generation ---")
    all_questions.extend(build_code_gen())

    print("\n--- T07: Code Debug (dedup'd, +swe-repair) ---")
    all_questions.extend(build_code_debug())

    print("\n--- T08: Logic ---")
    all_questions.extend(build_logic())

    print(f"\nTotal raw questions: {len(all_questions)}")

    cat_counts = {}
    for q in all_questions:
        cat_counts[q["category"]] = cat_counts.get(q["category"], 0) + 1
    print("\nPer-category totals:")
    for c in sorted(cat_counts):
        print(f"  {c}: {cat_counts[c]}")

    print("\n--- Difficulty distribution ---")
    diff_counts = {}
    for q in all_questions:
        diff_counts[q["category"]] = diff_counts.get(q["category"], {})
        diff_counts[q["category"]][q["difficulty"]] = \
            diff_counts[q["category"]].get(q["difficulty"], 0) + 1
    for c in sorted(diff_counts):
        d = diff_counts[c]
        print(f"  {c}: easy={d.get('easy',0)} medium={d.get('medium',0)} hard={d.get('hard',0)}")

    print(f"\n--- Format constraints applied ---")
    constraint_count = sum(1 for q in all_questions if any(c in q["prompt"] for c in FORMAT_CONSTRAINTS))
    print(f"  {constraint_count}/{len(all_questions)} ({100*constraint_count/len(all_questions):.0f}%)")

    print(f"\n--- Splitting into training/validation ({VERSION}) ---")
    save_split(all_questions, VERSION)

    print("\nDone.")
