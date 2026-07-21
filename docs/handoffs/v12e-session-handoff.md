# Handoff: End of Session — v12e + v12g comparison complete

## Current state

**Git:** v12d branch on filtered-build, with uncommitted v12e changes (stash popped).
Untracked additions: `run_v12e.py`, `run_v12g.py`, `grade_v12e.py`, `agent/solvers/lora_model.py`, `agent/complexity.py`, `eval_results/` (all runs saved).

## What we built this session

Three new modules for v12e (Qwen 1.5B + LoRA):
- `agent/solvers/lora_model.py` — transformers+peft inference, category-specific LoRA adapters (8 categories), thread-safe singleton
- `agent/complexity.py` — MiniLM-L6-v2 + LogisticRegression (tested, removed — too high bias toward "complex")
- `run_v12e.py` / `grade_v12e.py` — eval runner + official grader harness (one-answer-per-line with newline flattening)
- `run_v12g.py` — Phi-4-mini via llama-cpp-python with v12g pipeline config, flattened output

## 60 medium-hard question results — final comparison

| Mode | Total | code | factual | logic | math | NER | sentiment | sum. |
|------|-------|------|---------|-------|------|-----|-----------|------|
| 1. All adapters, no few-shot | 41.7% | 93% | 0% | 13% | 0% | 71% | 71% | 0% |
| 2. All adapters + few-shot | 43.3% | 87% | 0% | 25% | 0% | 71% | 86% | 0% |
| 3. No NER adapter + few-shot | 46.7% | 87% | 0% | 38% | 0% | 71% | 86% | 14% |
| 4. **Bare Qwen 1.5B, no adapters** | 50.0% | 100% | 50% | 13% | 0% | 71% | 71% | 0% |
| 5. **v12g (Phi-4-mini)** | **85.0%** | 100% | 100% | 100% | 75% | 86% | 43% | 71% |

## Key learnings

1. **LoRA adapters are hurting, not helping.** Bare Qwen 1.5B (50%) beats every adapter configuration. The adapters overfit on narrow training data.
2. **Phi-4-mini is the best model tested** (85% on this set). Runs at 83 t/s on GPU, ~10 min estimated on Docker CPU. Outperforms Nemotron in accuracy with similar speed.
3. **NAKED_CATEGORIES matter** — bare Qwen's 50% was mostly from NAKED mode (no system prompts for ner/summarization/factual/logic/math). Adding system prompts degraded performance.
4. **Few-shot helped sentiment** (17%→50%) and math (55%→64%) on the balanced 60-set.
5. **Calculator sandbox escape fixed** in `agent/solvers/tools.py` — replaced `eval()` with AST-based SafeVisitor that only allows math ops. Regex blocks `_` and `.`.
6. **CATEGORY_REGISTRY.py PRIORITY fixed** — matched to stage2.py (sentiment=4, ner=3 instead of swapped).
7. **Stage3 complexity** is active (MiniLM removed) — routeing decisions work as in v12d.

## Retraining data

`/home/artem/dev/amd-hackathon-filtered-build/lora_data/retraining_data.jsonl`
17 failures from the 60-set, formatted as `{instruction, output, category}`. Ready for LoRA fine-tuning.

## Models available locally

| Model | Path | Size |
|-------|------|------|
| Qwen2.5-1.5B (transformers) | HuggingFace cache | 2.9 GB |
| Qwen2.5-1.5B (GGUF) | models/qwen2.5-1.5b-instruct-q4_k_m.gguf | 1.1 GB |
| Phi-4-mini (GGUF) | models/phi-4-mini-instruct-q4_k_m.gguf | 2.4 GB |
| Nemotron-3-Nano-4B (GGUF) | models/nemotron-3-nano-4b-Q4_K_M.gguf | 2.6 GB |

## Files modified this session

| File | Change |
|------|--------|
| agent/main.py | v12e pipeline — LoRA inference, concurrent processing, few-shot examples |
| agent/config.py | Parallelization (CONCURRENT_TASKS=4), LORA_CATEGORIES, stage3 complexity |
| agent/solvers/lora_model.py | Transformers+peft Qwen 1.5B + LoRA, thread-safe singleton (new) |
| agent/solvers/tools.py | Calculator sandbox escape fixed (AST SafeVisitor) |
| agent/complexity.py | MiniLM classifier (tested, not in active pipeline) |
| agent/dynamic_prompts.py | NER_ONE_SHOT_EXAMPLE expanded to 5 examples, SENTIMENT_EXAMPLES (5), MATH_EXAMPLES (2) |
| CATEGORY_REGISTRY.py | PRIORITY fixed (sentiment=4, ner=3) |
| requirements.txt | Added peft, accelerate, sentence-transformers, scikit-learn |
| Dockerfile | Updated for transformers+peft, HF model download, adapter copy |

## Adapter directories

`amd_hackathon_adapters_off/` (moved out) — all 8 adapters from training. Currently NOT mounted.
To restore: `mv amd_hackathon_adapters_off amd_hackathon_adapters`

## Next session ideas

1. **Use Phi-4-mini in the Docker** — it scores 85% on this set, best we've seen. Need to build `v12h` image with it.
2. **Retrain LoRA adapters** with the collected 17-failure dataset — potentially fix the overfitting.
3. **Improve sentiment** — Phi-4 scored only 43% on sentiment (worse than Qwen). Need either better few-shot or routing to Fireworks.
4. **Improve math** — Phi-4 got 75% (6/8). The 2 failures were reasoning errors in complex problems. Could route hard math to Fireworks.
5. **Compare against official grader** — run evaluate.py pipeline (not grade_v12e.py) for official scores.

## Eval result files

All saved in `/home/artem/dev/amd-hackathon-filtered-build/eval_results/` with timestamps.
Also saved to `/tmp/eval_mode{1,2,3,4}_*.txt` and `/tmp/eval_v12g_final.txt`.
