#!/usr/bin/env python3
"""Bundle all eval question sets + context into one deliverable file."""
import json, os
from pathlib import Path

HERE = Path("/home/artem/dev/amd-hackathon")
OUT = HERE / "data" / "eval" / "EVAL_SETS_BUNDLE.md"

sections = []

# ── Context header ──
sections.append("""# AMD ACT II Hackathon — Evaluation Question Sets Bundle

## Hackathon Context

**Project:** Router to Vibehalla — Track 1 (AMD Developer Hackathon ACT II)  
**Hosted by:** lablab.ai  
**Task:** Build an 8-category question classifier/solver pipeline that runs inside a Docker container and outputs answers that are graded via fuzzy-match.

### Submission Container Constraints (HARD)

| Constraint | Value |
|---|---|
| Base image | Python 3.12-slim |
| CPU cores | 2 vCPU |
| RAM | 4 GB |
| GPU | None (CPU-only inference) |
| Total runtime | 600 seconds (10 minutes) |
| Per-question budget | ~28-30 seconds |
| Allowed external API | Fireworks AI (when FIREWORKS_API_KEY + ALLOWED_MODELS provided by grader) |
| Allowed models (FW) | minimax-m3, kimi-k2p7-code (grader-injected at runtime) |
| Grading method | Fuzzy-match against expected answer strings |
| Grading strategies | exact (case-insensitive) -> substring -> token overlap >=70% -> numeric tolerance 5% |
| Questions per run | 20 questions per grading pass |

### 8 Pipeline Categories

1. **sentiment_classification** — Classify text as positive, negative, or neutral (3-class)
2. **factual_knowledge** — Answer world knowledge questions from provided context or general knowledge
3. **logical_reasoning** — Solve constraint puzzles, number sequences, deductive reasoning
4. **math_reasoning** — Multi-step arithmetic word problems, fractions, percentages
5. **code_generation** — Write Python functions from specification
6. **code_debugging** — Fix buggy Python functions
7. **text_summarization** — Summarize news articles / passages
8. **named_entity_recognition** — Extract structured entities from text (bonus category)

### Tested Local Models (for training/validation)

- RTX A4000 8GB GPU for local eval
- GGUF models (Q4_K_M quantization, 1.5B-4B params)
- Qwen2.5-1.5B-Instruct, Phi-4-mini, Nemotron-3-Nano-4B, Gemma-3-1B
- llama-cpp-python for local inference
- Fireworks AI fallback for hard categories via hybrid routing

---

## Included Files

This bundle contains 5 question files totalling 320 unique questions:

| # | File | Questions | Categories | Format | Purpose |
|---|---|---|---|---|---|
| 1 | heldout_40.json | 40 | 7 (no NER) | gold dict | Hackathon team reference examples |
| 2 | build-A-40.json | 40 | 7 (no NER) | gold dict | Custom set A, matched to ref format |
| 3 | build-B-40.json | 40 | 7 (no NER) | gold dict | Custom set B, matched to ref format |
| 4 | training-v3.json | 152 | 8 (includes NER) | plain answer | Compact training set, short prompts |
| 5 | validation-v3.json | 48 | 8 (includes NER) | plain answer | Held-out companion to training-v3 |

### Format Notes

**Heldout format (files 1-3):** Each question is a dict with keys: task_id, category, prompt, gold.
- gold varies by category:
  - sentiment/factual/logic/math: {"answer": str|num, "accept?": [synonyms]}
  - code_gen: {"function": "...", "check_code": "...", "context": "...", "_reference": "..."}
  - code_debug: {"function": "...", "tests": [...], "_reference": "..."}
  - summarization: {"keywords": [...], "min_coverage": 0.5}

**Plain answer format (files 4-5):** Each question is a dict with keys: category, prompt, expected_answer, source, difficulty, task_id.
- expected_answer is a plain string (compatible with fuzzy-match grading)

---

""")

# ── Append each JSON file ──
files = [
    ("heldout_40.json", "input/heldout_40.json"),
    ("build-A-40.json", "data/eval/generated/build-A-40.json"),
    ("build-B-40.json", "data/eval/generated/build-B-40.json"),
    ("training-v3.json", "data/eval/training-v3.json"),
    ("validation-v3.json", "data/eval/validation-v3.json"),
]

for name, relpath in files:
    path = HERE / relpath
    if not path.exists():
        sections.append(f"\n## File: {name}\n\n(FILE NOT FOUND at {relpath})\n\n")
        continue
    with open(path) as f:
        data = json.load(f)
    count = len(data) if isinstance(data, list) else len(data.get("questions", []))
    # Format as compact JSON
    content = json.dumps(data, indent=2, ensure_ascii=False)
    sections.append(f"\n## File: {name} ({count} questions)\n\n```json\n{content}\n```\n\n")

# Write
content = "\n".join(sections)
OUT.write_text(content)
print(f"Written {OUT}")
import statistics
print(f"Total size: {len(content):,} chars / {os.path.getsize(OUT):,} bytes")
