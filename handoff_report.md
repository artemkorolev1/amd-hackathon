# Handoff Report: Finding the Best Agent for Named Entity Recognition (NER)

## Project Context

AMD ACT II Hackathon — 8-way deterministic prompt classifier pipeline at `~/dev/amd-hackathon/`.

**NER category** is one of 8 categories in the pipeline. We have:
- **Deterministic solvers**: `old_solve_ner` (in `agent/solvers/deterministic.py`) and `prototype_ner_v3` (in `agent/solvers/prototype_ner_v3.py`)
- **Local GGUF models** (6): Qwen2.5-1.5B, Qwen2.5-Coder-1.5B, Qwen2.5-Math-1.5B, Gemma-3-1B, SmolLM2-1.7B, Llama-3.2-1B
- **Fireworks API**: Can route NER to kimi-k2p7-code for high accuracy (100% on 6 NER questions in Firefox eval)
- **NER questions**: 19 from `data/eval/training-v3.json` (tweet-style with `{@...@}` markers) + 10 from `input/dev_40.json` + `input/complexity_40.json` (simpler traditional NER)

## Baseline: Deterministic Solvers

### Old `solve_ner` (in `agent/solvers/deterministic.py`)
- Pure regex-based: capitalized multi-word entities, dates, biomedical (diseases, genes, proteins)
- **Does NOT handle** `{@...@}` markers at all
- Outputs comma-separated list (wrong format for training data)
- Result: ~0% on training-v3, decent on simple biomedical/text NER

### `prototype_ner_v3` (in `agent/solvers/prototype_ner_v3.py`) — **Best deterministic option**
- Handles `{@...@}` marker extraction and classification into 7 types (person, group, corporation, location, event, product, creative_work)
- Also extracts unmarked entities: hashtags, @mentions, ALL-CAPS phrases, capitalized names
- Context-based type classification using surrounding words
- **Baseline performance on 19 training-v3 NER questions:**
  - Exact match: 3/19 = **15.8%**
  - Perfect F1 (1.0): 4/19 = **21.1%**
  - F1 >= 0.5: 11/19 = **57.9%**
  - Average F1: **0.544**

#### Failure patterns (prototype v3):
1. **Capitalized entity over-extraction**: `"12th and Arch"` → extracts `"Arch"` as person; `"Sister to Sister"` → extracts `"Sister Meets Shop Talk"` (lowercase proximity confusion)
2. **Type misclassification**: location→person, person→group, entity→related word
3. **Marketing/context**: `"SUPER BOWL"` → classified as event (correct), but `"FridayLivestream"` → missed because `REQUEST` keyword interferes
4. **Spurious entities**: Creates entities from non-entity phrases like `"Education"`, `"Cheif"`, `"Christmas"` by misidentifying capitalized words as entities
5. **Missing `{@...@}` marker duplicates**: The dedup was changed to preserve duplicates but the current code (line 408) dedupes by `(type, text.lower())` which still collapses intentional duplicates

## GGUF Models — Evaluation Results

### Key Finding: **No local LLM (1B-1.7B) can solve the `{@...@}` marker NER format**

Tested all 6 models on 19 training-v3 NER questions with a carefully formatted prompt instructing `type: value` output with `{@...@}` preservation. Results:

| Model | Avg F1 | Exact/19 | Issue |
|-------|:------:|:--------:|-------|
| **prototype_v3 (deterministic)** | **0.544** | **4** | Best — handles markers, good format |
| Qwen2.5-1.5B | 0.107 | 1 | Drops `{@...@}` markers, uses wrong types |
| Llama-3.2-1B | 0.100 | 0 | Hallucinates entities, wrong type format |
| SmolLM2-1.7B | 0.000 | 0 | Prefixes with `type:` → `value:` instead of `type: value` |
| Qwen2.5-Coder-1.5B | ~0 | 0 | Similar marker-dropping |
| Qwen2.5-Math-1.5B | ~0 | 0 | Refuses to answer (RLHF refusal) |
| Gemma-3-1B | ~0 | 0 | Outputs entity types without entity names |

### Root Cause: All 6 models fail at format adherence

1. **`{@...@}` markers stripped**: Models output `Austin McBroom` not `{@Austin McBroom@}`
2. **Line format broken**: Models output comma-separated lists instead of one per line
3. **Wrong entity types**: Models use `person:`, `organization:`, `team:` but expected uses specific 7 types (person, group, corporation, location, event, product, creative_work)
4. **Entity hallucination**: `type: I`, `type: none`, `type: hands`, `type: shakin`
5. **Entity merging**: `person: Austin McBroom, Bryce Hall` (two entities on one line)
6. **Refusal**: Qwen2.5-Math refuses to answer potentially contentious content

The deterministic `prototype_ner_v3` is **5x better** than the best local LLM because it handles the exact `{@...@}` format through explicit regex extraction rather than generation.

### Verdict: Local LLMs are NOT suitable for marker-based NER

The format gap is structural — small LLMs (1B-1.7B) cannot reliably:
- Preserve `{@...@}` markers
- Use the exact required entity type vocabulary  
- Output one entity per line without merging or splitting
- Avoid hallucinated entities

**Recommendation**: Use prototype_ner_v3 for local NER + Fireworks escalation for high accuracy.

## Fireworks API (External — high accuracy but costs tokens)

From previous eval (`eval_fireworks.py`):
- NER on kimi-k2p7-code with dynamic prompts: **100%** (6/6)
- Uses `NER_ONE_SHOT_EXAMPLE` + structured entity format prompts
- Key: the prompt says `type: value` format with `{@...@}` preservation

## Recommended Architecture

Based on all findings, the **best NER agent** depends on the constraint:

| Constraint | Best Approach | Expected F1 |
|-----------|--------------|-------------|
| Zero tokens (Docker, no API) | `prototype_ner_v3` with fixes from `ner_fix_plan.md` | ~0.55-0.60 |
| Has Fireworks API | Route NER to kimi-k2p7-code (always escalate) | ~1.00 |
| Local LLM (1.5B-1.7B) + good prompt | **Not viable** — all models fail format adherence | ~0.10 |
| LoRA fine-tuned Qwen2.5-1.5B | Adapter-trained (Jul 12): **33.3%** (2/6) | ~0.33 |

### Key Fixes Needed for Deterministic Path (from `ner_fix_plan.md`):

1. **HIGH PRIORITY**: Integrate `prototype_ner_v3` as the primary NER solver in `deterministic.py`
2. **HIGH PRIORITY**: Remove `"ner"` from `NAKED_CATEGORIES` in `main.py` so NER enters deterministic solver path
3. **HIGH PRIORITY**: Fix NER system prompts in `dynamic_prompts.py` — use `type: value` per-line format with `{@...@}` preservation
4. **MEDIUM**: Add more known entities to prototype (from training data analysis)
5. **MEDIUM**: Fix hashtag extraction for `# <space>word` pattern
6. **LOW**: Fix dedup to preserve intentional duplicate marker entities

### Prompt Strategy (for local LLM path):

**Conclusion: Prompt engineering alone cannot fix marker-based NER on 1B-1.7B models.**

The format gap persists across all 4 prompt variants:
- **v1_terse**: Entities stripped of markers, comma-separated
- **v2_structured**: Models ignore type vocabulary, use wrong ones
- **v3_fewshot**: Example with markers — models still drop them
- **v4_cot**: Step-by-step instructions produce nonsensical entity extraction

Only a **deterministic solver with explicit `{@...@}` regex extraction** (prototype_ner_v3) or a **capable model via API** (Fireworks kimi-k2p7-code) can handle this task.

## Files to Reference

| File | Purpose |
|------|---------|
| `agent/solvers/prototype_ner_v3.py` | Best deterministic NER solver — handles markers |
| `agent/solvers/deterministic.py` (lines 797-880) | Old solve_ner — needs replacement |
| `ner_fix_plan.md` | Full implementation plan for NER fixes |
| `agent/dynamic_prompts.py` (lines 232-261) | NER system prompts need format fix |
| `agent/main.py` (line 59) | `NAKED_CATEGORIES` blocks NER — needs edit |
| `data/eval/training-v3.json` | 19 tweet-style NER questions (gold standard) |
| `input/dev_40.json` | 5 simpler NER questions |
| `input/complexity_40.json` | 5 simpler NER questions |
| `eval_results/ner_comparison_v2.json` | Model eval results (actual: 19 training-v3 questions) |
| `scripts/eval_ner_models_v2.py` | NER eval script v2 — tested with correct expected answers |
| `scripts/eval_ner_models.py` | NER eval script v1 (broken — dev_40 has no expected_answer) |

## Next Steps

1. **Apply ner_fix_plan.md fixes** — especially integrating prototype_v3 into deterministic.py (the key bottleneck)
2. **Remove `"ner"` from `NAKED_CATEGORIES`** in `main.py` so deterministic NER path actually activates
3. **Fix NER system prompts** in `dynamic_prompts.py` for the Fireworks API path (already works at 100% with good prompts)
4. **Run full pipeline eval** with prototype_v3 integrated to verify NER improvement
5. **Fireworks routing**: NER consistently benefits from Fireworks escalation (100% on capable models). Add to `FIREWORKS_ESCALATION_CATEGORIES` in `config.py`
6. **If LoRA fine-tuning**: NER LoRA adapter scored 33.3% — may need better few-shot examples and format-focused training data

## Suggested Skills

- `hackathon-8way-classifier` — full pipeline context
- `deterministic-solver` — if creating a pure-regex NER solver
- `lora-dataset-preparation` — if fine-tuning Qwen LoRA for NER
