# NVIDIA Prompt Task and Complexity Classifier — Evaluation

**Model:** `nvidia/prompt-task-and-complexity-classifier`
**Architecture:** DeBERTa-v3-base + 8 custom heads (MeanPooling + MulticlassHead)
**Size:** 735MB (model.safetensors), 184M params
**Speed:** ~99ms per text on CPU
**License:** NVIDIA Open Model License Agreement

## What it does

Multi-headed classifier with two output families:
1. **Task type** (11 categories): Brainstorming, Chatbot, Classification, Closed QA, Code Generation, Extraction, Open QA, Other, Rewrite, Summarization, Text Generation, Unknown
2. **Complexity dimensions** (6 scores → 0-1 overall):
   - Creativity (weight 0.35)
   - Reasoning (weight 0.25)
   - Constraints (weight 0.15)
   - Domain Knowledge (weight 0.15)
   - Contextual Knowledge (weight 0.05)
   - Number of Few Shots (weight 0.05)

## Accuracy on our hackathon test sets

| Dataset | Accuracy | Notes |
|---------|:--------:|-------|
| Stress Test (20) | **25%** (5/20) | Most tasks classified as "Open QA" or "Closed QA" |
| Mixed Eval (20) | **30%** (6/20) | Same pattern — everything maps to factual_knowledge |

**Root cause:** NVIDIA's 11 task categories don't match our 8 routing categories. The model was trained for general prompt categorization, not for our specific hackathon routing needs. Almost every non-factual prompt gets labeled as "Open QA" or "Closed QA" which maps to our "factual_knowledge" category.

## Comparison with our approaches

| Approach | Stress Test | Speed | Size |
|----------|:----------:|:-----:|:----:|
| Our Hybrid (ensemble + deterministic) | **80%** | 1.5ms | 5.1MB |
| NVIDIA Prompt Classifier | **25%** | 99ms | 735MB |

## Complexity score correlation

The complexity score does correlate positively with human-rated difficulty:
- Easy tasks: avg 0.163
- Hard tasks: avg 0.237
- Trick: 0.393

But the separation is weak (easy vs medium are indistinguishable at 0.163 vs 0.162), and the signal is not strong enough to justify carrying 735MB in the Docker image.

## Verdict

**Not viable for our hackathon Docker submission.** Three reasons:
1. **Wrong categories** — 11 general-purpose task types don't map to our 8 routing categories (25% accuracy)
2. **Too large** — 735MB vs current Docker image total footprint
3. **Too slow on CPU** — 99ms per text adds ~2s of overhead for 19 tasks

## Files

- `benchmark_nvidia.py` — Full accuracy benchmark against stress test + mixed eval
- `benchmark_nvidia_complexity.py` — Complexity dimension extraction + correlation analysis

## References

- Model page: https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
- ONNX quantized version (not usable — multi-head not supported by standard wrapper): https://huggingface.co/botirk/tiny-prompt-task-complexity-classifier
- NeMo Curator docs: https://docs.nvidia.com/nemo/curator/
