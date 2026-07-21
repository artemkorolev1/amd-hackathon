# Research Brief: AMD ACT II Hackathon — Current State

## Competition Conditions

- **Submission container**: Python 3.12-slim, 2 CPU cores, 4 GB RAM, no GPU. Maximum 600 seconds total runtime for the grading harness. Per-question deadline approximately 28 seconds before hard kill.
- **Grading**: Fuzzy-match against expected answer strings. Four strategies: exact (case-insensitive), substring, token overlap >= 70%, numeric tolerance 5%.
- **Allowed API**: Fireworks AI (when the grader supplies FIREWORKS_API_KEY and ALLOWED_MODELS at runtime). The set of allowed model IDs is injected at grading time — not hardcoded in the container.

## Task Categories (8)

1. **sentiment**: Classify text as positive, negative, neutral — including sarcasm, dismissiveness, faint praise, irony where surface words contradict the actual sentiment.
2. **ner**: Extract structured named entities from text. Output format expects category-prefixed lists (e.g. GENE: WNT; DISEASE: medulloblastoma). Entity categories vary per question (persons, orgs, locations, dates, genes, drugs, etc.).
3. **math**: Solve word problems and computation questions. Multi-step reasoning, mixture/alligation, probability, permutations, compound interest, algebra. Multiple-choice with option letter + text.
4. **code_gen**: Write Python functions from docstring specifications. Full function implementation required, including imports and type hints.
5. **code_debug**: Fix buggy Python code. The bug is described in the prompt; the fix should be the corrected code snippet.
6. **factual**: Answer questions from provided context. Often counterfactual or multi-hop reasoning (what-if historical scenarios, scientific extrapolation).
7. **logic**: Solve constraint puzzles, syllogisms, analogies, knights-and-knaves, scheduling problems. Requires deductive reasoning, not lookup.
8. **summarization**: Synthesize multiple sources (2-3) into a balanced summary. Sources may contradict each other. Headline-vs-body discrepancy detection.

## Current Evaluation Datasets

**Files exist** at `/home/artem/dev/amd-hackathon-submit/` and `/home/artem/dev/amd-hackathon-shared/`:
- A 300-question set (eval_all_300.json) — mostly sourced from claude-code-hard-v1
- A 60-question balanced set (eval_60_balanced.json)
- A 40-question hard set (complexity_eval_40.json)
- A 40-question quirky set (complexity_eval_40_quirky.json)
- A 95-question converted dataset (eval_from_datasets_20260712_172443.json) — sample from 6 HuggingFace datasets

**Known data problems (from eval runs within the last 2 hours):**
- The two 40-question sets had overlapping questions that were identified and replaced
- Some logic puzzles had incorrect expected answers (verified by exhaustive search — the answer didn't match the clues)
- Some code_debug expected answers had edge cases where the fix failed (e.g., bill-split rounding produced negative remainders)
- NER expected answers included entity types not listed in the prompt's extraction instruction
- One summarization prompt had a literal [truncated] marker in the text

## LoRA Fine-Tuning Attempt (v12e)

**Base model**: Qwen2.5-1.5B-Instruct
**Training environment**: Google Colab with T4 GPU (16GB VRAM), QLoRA via Unsloth
**Training data**: 16,000 examples across 8 JSONL files (2,000 per category), from SST-2, GSM8K, HumanEval, HumanEvalPack, LogiQA, NCBI Disease, OntoNotes 5.0, XSum
**Data format**: JSONL with {"prompt": "...", "response": "...", "category": "...", "source": "..."}
**LoRA hyperparameters used**: r=16, alpha=16, target_modules = q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj. Model loaded in 4-bit. Learning rate 2e-4 (constant), weight_decay 0.01, warmup_steps 10, batch_size 4, gradient_accumulation 4, max_seq_length 1024. Epochs varied per category (1 or 2).
**Training procedure**: The notebook loaded the base model once, then trained 8 adapters sequentially on the same model instance without reloading between categories.
**Eval result**: 50% on a 60-question eval set. Sentiment improved versus baseline. Factual, math, summarization, and logic dropped versus baseline.

## Models Previously Tested (raw, no pipeline, 300 questions)

Qwen2.5-1.5B-Instruct, Qwen2.5-Coder-1.5B, Nemotron-3-Nano-4B, Phi-4-mini-Q4. Score range on the 300-set was 76-91% raw accuracy. The 4B models scored higher than 1.5B models. Sentiment was the lowest category across all models.

## Research Needed

- Sources of diverse, balanced evaluation data across the 8 categories
- Best practices for creating synthetic evaluation data
- Proper LoRA training procedure for small instruct models (Qwen2.5-1.5B and Qwen2.5-3B specifically)
- Data formatting requirements for Qwen2.5 chat template
- How to avoid quality degradation when fine-tuning with LoRA
