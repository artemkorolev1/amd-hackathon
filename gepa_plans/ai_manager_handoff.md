# Handoff Plan for AI Manager — GEPA Multi-Category Workflow Optimization

## Goal
Complete step-by-step workflow evals for ALL remaining categories, then fine-tune all prompts using GEPA prompt evolution.

## Current State
### ✅ Completed
- **Factual (T01)**: Done. smollm2 + empty prompt = 0.328 on 58q
- **Math (T02)**: Workflow works. Llama-3.2 3-step = 0.564 on 94q. GEPA optimizer running (0.600 best on subset)
- **Logic**: Baseline tested (qwen-coder 3-step = 0.500)
- **Summarization**: Chunk-and-summarize tool built and tested
- **Workflow Engine**: `agent/workflow.py` — WorkflowEngine with ToolRegistry, 3 templates (math_3step, logic_3step, ner_2step)
- **Cell Architecture**: `StepConfig` in cell.py, backward compatible
- **Pipeline integration**: Branch for workflow cells in Pipeline.process()

### ⏳ Remaining
- **Sentiment (T03)**: Not started
- **NER (T05)**: Not started (template exists)
- **Code Gen / Code Debug**: Not tested
- **GEPA fine-tuning**: Only started on math — need to run on all categories
- **Full pipeline integration**: Agent dispatched but not returned

## Action Plan (ordered by priority)

### 1. Run GEPA on remaining math prompts
Continue from where the current GEPA run left off:
- Complete 75 combo grid search on qwen-coder
- Verify best 2 combos on all 94 questions
- Compare with Llama-3.2 baseline (0.564)

### 2. Sentiment (T03)
- Build combined eval set from training-v3, validation-v3, eval_hard_218
- Test 3-4 prompt strategies (empty, "Sentiment:", "Classify:", "Label only")
- Use qwen-coder (best generalist) for single-shot eval
- GEPA-evolve sentiment prompts

### 3. NER (T05)
- Existing `NER_2STEP_WORKFLOW` template in workflow.py (extract → verify)
- Test on combined NER dataset
- Try with spaCy tool integration via ToolRegistry

### 4. Code Gen / Code Debug
- Test qwen-coder on code tasks (it's a code specialist!)
- Build a 3-step workflow: plan → implement → verify
- Use Python execution tool via ToolRegistry

### 5. Summarization GEPA
- Run GEPA on summarization prompts
- Compare single-shot vs chunk-and-summarize

### 6. Routing Table Assembly
- Build final routing table from Pareto-optimal cells per category
- Wire into Pipeline as default routing

## Key Files
- `/home/artem/dev/amd-hackathon/agent/workflow.py` — WorkflowEngine, templates
- `/home/artem/dev/amd-hackathon/agent/cell.py` — Cell, StepConfig
- `/home/artem/dev/amd-hackathon/agent/pipeline.py` — Pipeline.process()
- `/home/artem/dev/amd-hackathon/agent/mutation_agent.py` — GEPA mutation ops
- `/home/artem/dev/amd-hackathon/agent/evaluation_agent.py` — Evaluation with fuzzy_match
- `/home/artem/dev/amd-hackathon/agent/orchestrator.py` — GEPAOrchestrator
- `/home/artem/dev/amd-hackathon/gepa_plans/eval_common.py` — fuzzy_match function
- `/home/artem/dev/amd-hackathon/data/eval/` — All eval datasets

## Models (GGUF, GPU-ready)
- `qwen2.5-coder-1.5b-instruct` — Best generalist (primary)
- `qwen2.5-1.5b-instruct` — Fast generalist (secondary)
- `Llama-3.2-1B-Instruct` — Fastest, best instruction follower
- `Qwen2.5-Math-1.5B-Instruct` — Math specialist (single-shot only)

## GPU
RTX A4000 (8GB). n_gpu_layers=-1 for all models. ~1.5GB per model.
Can load 2-3 models simultaneously if needed (6.5GB free).

## Eval Methodology
- Use fuzzy_match from `gepa_plans/eval_common.py` (4-cascade: exact, substring, numeric ±1%, token overlap 80%)
- Always use temperature=0.0 for deterministic eval
- Report: accuracy, correct/total, avg latency
- Save results as JSON with per-question details

## ToolRegistry
- `agent/workflow.ToolRegistry` supports tool steps
- Built-in: `chunk_text` tool
- Can add: `sympy`, `python_execute`, `spacy_ner`
