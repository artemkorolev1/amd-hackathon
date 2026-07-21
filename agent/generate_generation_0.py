#!/usr/bin/env python3
"""
Generate generation_0.json for smollm2-1.7b factual QA prompt variants.
Uses the GEPA genetic operators from gepa_runner.py.

2 elites (best known prompts for smollm2-1.7b)
3 crossover children (from pairs of best prompts)
2 mutants (of best prompts)
1 fresh random variant

For short prompts like "Answer:" and "Fact:" that have minimal text for
mutation/crossover to work on, we use the longer known-good prompts
as mutation/crossover parents to ensure real diversity.
"""

import sys
import os
import json
import random
from pathlib import Path

# Add project root to path so we can import agent.gepa_runner
PROJECT_ROOT = "/home/artem/dev/amd-hackathon"
sys.path.insert(0, PROJECT_ROOT)

from agent.gepa_runner import (
    KNOWN_GOOD_PROMPTS,
    PREFIXES,
    CONSTRAINT_POOL,
    THESAURUS,
    SHORT_STYLES,
    MEDIUM_STYLES,
    TEMP_OPTIONS,
    MAX_TOKENS_DEFAULT,
    MAX_PROMPT_CHARS,
    mutate,
    crossover_prompts,
    generate_random_variant,
    _truncate_prompt,
)

# Seed for reproducibility
random.seed(42)

OUTPUT_PATH = "/home/artem/dev/amd-hackathon/gepa_plans/generation_0.json"


def mutate_with_retry(parent: dict, max_attempts: int = 20) -> dict:
    """Keep mutating until the prompt text actually changes, or until max_attempts."""
    original_text = parent["system_prompt"]
    for attempt in range(max_attempts):
        child = mutate(parent)
        # Clean up the name from what mutate() appended
        if child["system_prompt"] != original_text or child["temperature"] != parent["temperature"]:
            # We got a real change
            return child
    # Fallback: force a change by adding a constraint
    child = dict(parent)
    text = child["system_prompt"]
    if text and not text.endswith("."):
        text += "."
    constraint = random.choice(CONSTRAINT_POOL)
    text = text + " " + constraint
    text = re.sub(r'\s+', ' ', text).strip()
    text = _truncate_prompt(text)
    child["system_prompt"] = text
    child["name"] = parent["name"] + "_forced_mut"
    return child


import re  # needed for the fallback


def main():
    # Set up our parent pool — the 4 known-good prompts
    # with their known performance on smollm2-1.7b
    parent_pool = [
        {
            "name": "Answer:",
            "system_prompt": "Answer:",
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
            "accuracy": 0.421,
        },
        {
            "name": "Fact:",
            "system_prompt": "Fact:",
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
            "accuracy": 0.368,
        },
        {
            "name": "long_prompt",
            "system_prompt": "Answer the question directly. Use exact names, dates, and numbers. Keep under 15 words. No preamble.",
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
            "accuracy": 0.368,
        },
        {
            "name": "Answer directly.",
            "system_prompt": "Answer directly.",
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_DEFAULT,
            "accuracy": 0.316,
        },
    ]

    variants = []

    # --- 2 ELITES (best known prompts for smollm2-1.7b) ---
    # Elite 1: best known — "Answer:" (0.421)
    elite1 = {
        "name": "elite_base",
        "system_prompt": "Answer:",
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_DEFAULT,
    }
    variants.append(elite1)

    # Elite 2: second best — "Fact:" (0.368) — tied with long prompt, but simpler
    elite2 = {
        "name": "elite_fact",
        "system_prompt": "Fact:",
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS_DEFAULT,
    }
    variants.append(elite2)

    print("=== ELITES ===")
    print(f"  {elite1['name']:25s} '{elite1['system_prompt']}'")
    print(f"  {elite2['name']:25s} '{elite2['system_prompt']}'")

    # --- 3 CROSSOVER CHILDREN ---
    # Use the longer prompts for crossover to get meaningful results
    crossover_pairs = [
        # Pair 1: long prompt x "Answer directly."
        (parent_pool[2], parent_pool[3]),
        # Pair 2: "Answer:" x long prompt — will be dominated by long prompt since "Answer:" is just a prefix
        (parent_pool[0], parent_pool[2]),
        # Pair 3: "Fact:" x "Answer directly."
        (parent_pool[1], parent_pool[3]),
    ]

    for idx, (p1, p2) in enumerate(crossover_pairs):
        # Set a stable random seed for deterministic crossover
        rstate = random.getstate()
        random.seed(200 + idx)
        child = crossover_prompts(p1, p2)
        random.setstate(rstate)
        child["name"] = f"xover_{idx + 1}"
        variants.append(child)
        print(f"\n=== CROSSOVER {idx+1} ===")
        print(f"  Parent 1: '{p1['system_prompt']}'")
        print(f"  Parent 2: '{p2['system_prompt']}'")
        print(f"  Child:    '{child['system_prompt']}'")
        print(f"  Temp:     {child['temperature']}")

    # --- 2 MUTANTS of best prompts ---
    # Use the longer prompt for meaningful mutations
    for idx in range(2):
        # Mutant 1: mutate the long prompt (most text to work with)
        # Mutant 2: mutate "Answer directly."
        parent = parent_pool[2] if idx == 0 else parent_pool[3]
        rstate = random.getstate()
        random.seed(300 + idx)
        child = mutate_with_retry(parent)
        random.setstate(rstate)
        # Clean up name — strip the _mut_opX suffix that mutate() adds
        child["name"] = f"mutant_{idx + 1}"
        variants.append(child)
        print(f"\n=== MUTANT {idx+1} ===")
        print(f"  Parent: '{parent['system_prompt']}'")
        print(f"  Child:  '{child['system_prompt']}'")
        print(f"  Temp:   {child['temperature']}")

    # --- 1 FRESH RANDOM VARIANT ---
    rstate = random.getstate()
    random.seed(400)
    fresh = generate_random_variant("random_fresh")
    random.setstate(rstate)
    variants.append(fresh)
    print(f"\n=== RANDOM FRESH ===")
    print(f"  Prompt: '{fresh['system_prompt']}'")
    print(f"  Temp:   {fresh['temperature']}")

    # Ensure we have exactly 8 variants
    assert len(variants) == 8, f"Expected 8 variants, got {len(variants)}"

    # Ensure max_tokens is set
    for v in variants:
        v.setdefault("max_tokens", MAX_TOKENS_DEFAULT)

    # Build output structure
    output = {
        "generation": 0,
        "model": "smollm2-1.7b",
        "previous_best_accuracy": 0.421,
        "previous_best_prompt": "Answer:",
        "mutations_applied": [
            "elite_carryover: Answer: (0.421 accuracy)",
            "elite_carryover: Fact: (0.368 accuracy)",
            "crossover: long_prompt x 'Answer directly.'",
            "crossover: 'Answer:' x long_prompt",
            "crossover: 'Fact:' x 'Answer directly.'",
            "mutation: long_prompt (random GEPA op)",
            "mutation: 'Answer directly.' (random GEPA op)",
            "fresh_random: generate_random_variant()",
        ],
        "variants": variants,
    }

    # Write output
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Generation 0 saved to {OUTPUT_PATH}")
    print(f"Total variants: {len(variants)}")
    print(f"{'='*60}")
    for v in variants:
        print(f"  [{v['name']:20s}] temp={v['temperature']:.1f} | '{v['system_prompt']}'")
    print(f"{'='*60}")

    # Validate the JSON
    with open(out_path) as f:
        data = json.load(f)
    print(f"\nJSON valid: yes, {os.path.getsize(out_path)} bytes")
    print(f"Variant count in file: {len(data['variants'])}")
    for v in data['variants']:
        required = ['name', 'system_prompt', 'temperature', 'max_tokens']
        missing = [k for k in required if k not in v]
        if missing:
            print(f"  WARNING: variant '{v.get('name','?')}' missing keys: {missing}")
        else:
            print(f"  OK: {v['name']:20s} | temp={v['temperature']} | '{v['system_prompt'][:80]}'")


if __name__ == "__main__":
    # re imported for fallback in mutate_with_retry
    import re as re_module
    # Re-run with fixed approach
    main()
