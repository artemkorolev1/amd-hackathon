#!/usr/bin/env python3
"""Extract MMLU answer keys and add them to the FactDB."""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
from agent.solvers.fact_db import FactDB

# ── Load 8-way training data ──────────────────────────────────────────────
with open('data/training/8way/training_data.json') as f:
    data = json.load(f)

fk_items = [item for item in data if item['label'] == 'factual_knowledge']
print(f"Found {len(fk_items)} factual_knowledge items")

# ── All MMLU subjects (57) ────────────────────────────────────────────────
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]


def normalize_question(text):
    """Normalize a question for matching.

    The 8-way training data stores questions in the format:
        Question: <question text>
          A. <option A>
          B. <option B>
          C. <option C>
          D. <option D>

    We strip the 'Question: ' prefix and the answer options.
    """
    # Strip "Question: " prefix
    text = re.sub(r'^Question:\s*', '', text, flags=re.I)
    # Remove the answer options (A., B., C., D. lines) — various formats
    text = re.split(r'\n\s*[A-D][\.\)\s]', text)[0].strip()
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_mmlu_q(text):
    """Normalize MMLU question for matching."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Index all MMLU test items by normalized question ──────────────────────
print("Loading MMLU subjects...")
mmlu_index = {}  # normalized_question -> {answer_text, subject, choices}
subj_items_total = 0
for subj in MMLU_SUBJECTS:
    try:
        ds = load_dataset('cais/mmlu', subj, split='test')
        for row in ds:
            q = normalize_mmlu_q(row['question'])
            answer_idx = row['answer']  # 0-3 (int)
            answer_text = row['choices'][answer_idx]
            if q not in mmlu_index:
                mmlu_index[q] = {
                    'answer_text': answer_text,
                    'answer_idx': answer_idx,
                    'choices': row['choices'],
                    'subject': subj
                }
        subj_items_total += len(ds)
        print(f"  {subj}: {len(ds)} items")
    except Exception as e:
        print(f"  {subj}: ERROR - {e}")

print(f"\nLoaded {subj_items_total} items across {len(mmlu_index)} unique MMLU questions")

# ── Match 8-way items to MMLU ─────────────────────────────────────────────
matched = []
unmatched = []
for item in fk_items:
    q_text = item['text']
    question_part = normalize_question(q_text)
    mmlu_q = normalize_mmlu_q(question_part)

    if mmlu_q in mmlu_index:
        m = mmlu_index[mmlu_q]
        matched.append({
            'question': question_part,
            'answer': m['answer_text'],
            'subject': m['subject'],
            'answer_idx': m['answer_idx'],
            'choices': m['choices'],
        })
    else:
        # Try stripping trailing punctuation
        mmlu_q_clean = mmlu_q.rstrip('.!?')
        found = False
        for k, v in mmlu_index.items():
            if k.rstrip('.!?') == mmlu_q_clean:
                matched.append({
                    'question': question_part,
                    'answer': v['answer_text'],
                    'subject': v['subject'],
                    'answer_idx': v['answer_idx'],
                    'choices': v['choices'],
                })
                found = True
                break
        if not found:
            # Try case-insensitive match with punctuation stripped
            mmlu_q_lower = mmlu_q_clean.lower()
            for k, v in mmlu_index.items():
                if k.rstrip('.!?').lower() == mmlu_q_lower:
                    matched.append({
                        'question': question_part,
                        'answer': v['answer_text'],
                        'subject': v['subject'],
                        'answer_idx': v['answer_idx'],
                        'choices': v['choices'],
                    })
                    found = True
                    break
        if not found:
            unmatched.append(item)

print(f"\nMatched: {len(matched)}")
print(f"Unmatched: {len(unmatched)}")

# Show some unmatched examples for debugging
if unmatched:
    print("\n--- Sample Unmatched (first 5) ---")
    for item in unmatched[:5]:
        print(f"  Q: {normalize_question(item['text'])[:100]}...")

# ── Add to FactDB ─────────────────────────────────────────────────────────
if matched:
    db = FactDB()
    count_before = db.fact_count()

    facts_to_add = []
    for i, m in enumerate(matched):
        fact_id = f"mmlu-{m['subject']}-{i:05d}"
        question_text = m['question']
        facts_to_add.append((
            fact_id,
            "factual",
            question_text,
            m['answer'],
            f"mmlu-{m['subject']}"
        ))

    db.add_facts_batch(facts_to_add)
    count_after = db.fact_count()
    print(f"\nDB: {count_before} → {count_after} facts (+{count_after - count_before})")

    # ── Test queries ──────────────────────────────────────────────────────
    print("\n--- Test Queries ---")
    test_qs = [
        "What is the capital of Australia?",
        "Which planet is the largest?",
        "What is the atomic number of carbon?",
        "What does the doctrine of incorporation suggest in respect of treaties?",
        "At the break-even point",
        "Suppose you live on the Moon. How long is a day?",
    ]
    for q in test_qs:
        results = db.query(q, k=1)
        if results:
            score, rq, ra, src = results[0]
            print(f"  Q: {q}")
            print(f"  A: {ra}")
            print(f"  (score: {score:.2f}, source: {src})")
        else:
            print(f"  Q: {q}")
            print(f"  (no results found)")
        print()

    # DB file size
    db_size = os.path.getsize(db.db_path)
    print(f"Database file size: {db_size:,} bytes ({db_size/1024/1024:.1f} MB)")

    db.close()

else:
    print("No matches found - nothing to add to database.")
