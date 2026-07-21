# GEPA Prompt Evolution Results — July 14-15, 2026

## Summary
Ran GEPA (Genetic Pareto Algorithm) prompt evolution on all 8 categories with local GGUF models on GPU. 
Results saved to `gepa_plans/*_gepa_results.json`.

## Per-Category Results

| Category | Model | Best Acc | Best Prompt | Temp | Avg Tokens | Avg Latency |
|----------|-------|:--------:|-------------|:----:|:----------:|:-----------:|
| sentiment | qwen2.5-1.5b | **100%** | "Classify the sentiment as positive or negative. Output only the label." | 0.0 | 1.6 | 20.5ms |
| code_debug | qwen2.5-1.5b | **100%** | "Debug:" | 0.0 | 126.2 | 756ms |
| code_gen | qwen2.5-coder-1.5b | **60%** | "Code: Write clean, working Python code. No preamble." | 0.1 | 111.6 | 1313ms |
| math | qwen2.5-1.5b | **50%** | "Solve the math problem. Output only the answer. Show your work briefly." | 0.0 | 107.2 | 1292ms |
| ner | qwen2.5-coder-1.5b | **36.8%** | (empty prompt) — plateau, format ceiling | 0.0 | 33.1 | 206ms |
| logic | qwen2.5-1.5b | **20%** | "Solve: Consider all possibilities. Logic:." | 0.0 | 118.7 | 682ms |
| summarization | qwen2.5-1.5b | **0%** | Metric mismatch — pipeline gets 75% on 300-set | — | — | — |

## Bottlenecks Identified

### NER (`{@...@}` marker format)
- Local LLMs (1B-1.7B) structurally cannot output `{@...@}` markers — 0% F1
- prototype_ner_v3 (deterministic): 54% F1 on training, 30% on validation
- **Fix**: Route to Fireworks kimi-k2p7-code with corrected format prompt
- fw_router.py updated: format prompt + model routing

### Summarization / Logic (evaluation metric)
- GEPA runner's `fuzzy_match` (token overlap >= 80%) doesn't work for these categories
- Pipeline already achieves 75% (summarization) and 68% (logic) on 300-set
- **Recommendation**: Use existing pipeline, skip GEPA for these

## Recommended Prompt Updates for Pipeline

Replace single-shot prompts in agent/dynamic_prompts.py's "low" complexity tier:

- **sentiment (low)**: "Classify the sentiment as positive or negative. Output only the label."
- **code_debug (low)**: "Debug:"
- **code_gen (low)**: "Code: Write clean, working Python code. No preamble."
- **math (low)**: "Solve the math problem. Output only the answer. Show your work briefly."

## Files Modified This Session
- `agent/solvers/fw_router.py` — NER format prompt + routing model
- `agent/main.py` — prototype_ner_v3 integration, skip local LLM for NER
