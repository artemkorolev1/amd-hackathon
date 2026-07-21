#!/usr/bin/env python3
"""
Build and save the routing table from all GEPA eval results.
Run: python3 gepa_plans/build_routing_table.py
Output: gepa_plans/routing_table.json — ready for Pipeline(routing_table=...)
"""

import json, os, sys
from pathlib import Path

sys.path.insert(0, "/home/artem/dev/amd-hackathon")
from agent.routing_table import RoutingTable

BASE = "/home/artem/dev/amd-hackathon"

# ── Best configurations from all evals ───────────────────────────────────

ENTRIES = {
    # ── Sentiment: qwen2.5-1.5b-instruct + explicit_instruction = 100% ──
    "sentiment": {
        "model_key": "qwen2.5-1.5b-instruct",
        "system_prompt": "Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed.",
        "decoding": {"temperature": 0.0, "max_tokens": 64},
        "aggregation": "single",
        "accuracy": 1.0,
        "source": "sentiment_eval_25q",
    },
    # ── NER: deterministic solver beats all LLMs ──
    "ner": {
        "solver": "deterministic",
        "note": "LLM models <44% fuzzy on @entity@ format. Use prototype_ner_v3 instead.",
        "accuracy_fuzzy": 0.88,
        "avg_f1": 0.792,
        "source": "ner_eval_25q_all_models",
    },
    # ── Code Debug: qwen2.5-coder-1.5b + empty/Fix: = 96% ──
    "code_debug": {
        "model_key": "qwen2.5-coder-1.5b-instruct",
        "system_prompt": "",
        "decoding": {"temperature": 0.0, "max_tokens": 256},
        "aggregation": "single",
        "accuracy": 0.96,
        "source": "code_eval_25q_direct",
    },
    # ── Code Gen: qwen2.5-coder-1.5b + "Code:" = 52% ──
    "code_gen": {
        "model_key": "qwen2.5-coder-1.5b-instruct",
        "system_prompt": "Code:",
        "decoding": {"temperature": 0.0, "max_tokens": 256},
        "aggregation": "single",
        "accuracy": 0.52,
        "source": "code_eval_25q_direct",
    },
    # ── Factual: smollm2-1.7b + empty = 32.8% (best on 58q) ──
    "factual": {
        "model_key": "smollm2-1.7b-instruct",
        "system_prompt": "",
        "decoding": {"temperature": 0.0, "max_tokens": 64},
        "aggregation": "single",
        "accuracy": 0.328,
        "source": "factual_58q_eval",
    },
    # ── Math: qwen2.5-math-1.5b + empty = 27.7% (max_tokens=256) ──
    # NOTE: needs max_tokens=512 for chain-of-thought, but current eval shows 256 working
    "math": {
        "model_key": "qwen2.5-math-1.5b-instruct",
        "system_prompt": "",
        "decoding": {"temperature": 0.0, "max_tokens": 512},
        "aggregation": "single",
        "accuracy": 0.277,
        "source": "math_94q_eval",
    },
    # ── Logic: qwen2.5-math-1.5b + step-by-step (~50% on subset) ──
    "logic": {
        "model_key": "qwen2.5-math-1.5b-instruct",
        "system_prompt": "Solve the logic puzzle step by step. Deduce from premises. End with 'Answer: <conclusion>' on its own line.",
        "decoding": {"temperature": 0.0, "max_tokens": 256},
        "aggregation": "single",
        "accuracy": 0.50,
        "source": "logic_handoff_claim_subset",
    },
    # ── Summarization: all models ~36%, qwen2.5-1.5b best ──
    "summarization": {
        "model_key": "qwen2.5-1.5b-instruct",
        "system_prompt": "Summarize the text in at most 2 sentences. Include key names, numbers, and facts.",
        "decoding": {"temperature": 0.0, "max_tokens": 128},
        "aggregation": "single",
        "accuracy": 0.36,
        "source": "summarization_25q_eval",
    },
}

# ── Build routing table ──────────────────────────────────────────────────

table = RoutingTable()

for category, entry in ENTRIES.items():
    # Check if deterministic solver (NER)
    if entry.get("solver") == "deterministic":
        # Store as metadata — Pipeline already has deterministic solvers wired
        table._upsert_entry({
            "category": category,
            "solver": "deterministic",
            "accuracy": entry.get("accuracy_fuzzy", 0.0),
            "note": entry.get("note", ""),
            "source": entry.get("source", ""),
            "updated_at": __import__("time").time(),
        })
        continue

    table._upsert_entry({
        "category": category,
        "cell_name": f"optimized_{category}_{entry['model_key']}",
        "model_key": entry["model_key"],
        "system_prompt": entry["system_prompt"],
        "decoding": entry["decoding"],
        "aggregation": entry.get("aggregation", "single"),
        "accuracy": entry["accuracy"],
        "source": entry.get("source", ""),
        "updated_at": __import__("time").time(),
    })

# Save
out_path = os.path.join(BASE, "gepa_plans/routing_table.json")
table.to_json(out_path)

# Print
print("=" * 70)
print("ROUTING TABLE — All Categories")
print("=" * 70)
print(f"{'Category':<15} {'Solver':<30} {'Accuracy':<10} {'Source'}")
print("-" * 70)
for cat in sorted(table.categories):
    entry = table.get(cat)
    if entry.get("solver") == "deterministic":
        solver = "prototype_ner_v3 (deterministic)"
        acc = f"{entry.get('accuracy', 0):.2%}"
    else:
        solver = f"{entry.get('model_key','?')}"
        acc = f"{entry.get('accuracy', 0):.2%}"
    print(f"{cat:<15} {solver:<30} {acc:<10} {entry.get('source','')[:30]}")

print(f"\nVersion: {table.version}")
print(f"Entries: {len(table.categories)}")
print(f"Saved: {out_path}")
