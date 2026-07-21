# Standardized Evaluation Approach

Two eval surfaces, both using the **official grader** (`evaluate.py` fuzzy_match):

### `run_comprehensive_eval.py` — Local GPU eval
- Runs on llama-cpp-python (default: Qwen2.5-1.5B Q4_K_M)
- k=3 consensus with **diverse prompt variants** (normal, step-by-step, terse)
- **Merged prompts** when S2 top-2 scores are within 1.0
- Logs: S2, Router, deterministic solver results, consensus agreement, QC metrics
- Reports: per-category accuracy, decision paths, error analysis

### `eval_fireworks.py` — Fireworks API eval
- Runs on kimi-k2p7-code (or configurable model via FW_MODEL env)
- Single inference (no consensus — Fireworks doesn't need it)
- **Merged prompts** when S2 top-2 scores are within 1.0
- Logs: S2, Router, latency
- Found 95.0% on 60-set vs 71.7% on 1.5B local

### Key design decisions:
1. **Official grader only** — `from evaluate import fuzzy_match` from main repo. No local copy.
2. **Per-question instrumented JSON** — every stage, every score, every sample saved
3. **Classifier logging mandatory** — S2 + Router always recorded
4. **Merged prompts when uncertain** — top-2 S2 score gap < 1.0 triggers build_merged_prompt
5. **Diverse consensus** — 3 prompt variants (different instructions/tiers) per sample, not just temperature

### Running:
```bash
# Local GPU eval (60-set)
MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf python3 run_comprehensive_eval.py

# Local GPU eval (300-set, takes ~6 min)
MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf python3 run_comprehensive_eval.py /home/artem/dev/amd-hackathon-shared/eval_all_300.json

# Fireworks eval
python3 eval_fireworks.py
```
