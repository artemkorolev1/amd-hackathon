# Evaluation Pipeline Infrastructure Review

## Overview

The project has **9 distinct eval harnesses** across 3 layers (classifier routing, deterministic solver, end-to-end agent). Each was built at a different time, tests different things, and outputs different formats. There is no single command that runs all of them.

---

## 1. Eval Harness Catalog

### 1.1 `classifier_showdown.py` (743 lines) — PRIMARY CLASSIFIER EVAL

| Aspect | Detail |
|--------|--------|
| **What it measures** | Routing accuracy of **9 classifiers** (5 × 8-way category + 4 × 4-way category) + **2 complexity scorers** |
| **Classifiers tested** | `deterministic`, `tfidf_lr`, `minilm_lr`, `ensemble`, `hybrid` (8-way); `deterministic_4way`, `tfidf_4way`, `ensemble_4way`, `hybrid_4way` (4-way); `bitmorphic`, `complexity_score` (complexity) |
| **Data sources** | `eval_stress_test_20.json` (20), `eval_mixed_20.json` (20), `eval_hard_100.json` (100), `classifiers/complexity_training_data.json` (78) — total **218 items** with category labels |
| **Metrics tracked** | Overall accuracy, per-category accuracy, per-difficulty accuracy, confusion matrix, avg/p50/p99 latency, per-prompt results |
| **Output** | JSON + HTML report to `reports/classifier_showdown_<timestamp>.json` and `.html` |
| **Execution** | `python classifier_showdown.py` — single command, no args |
| **Persisted artifacts** | ✅ Full JSON (715KB+ per run, includes per-prompt detail), ✅ HTML visualization |
| **Underscores** | Does not measure answer correctness — only **routing accuracy** (did the classifier pick the right category?) |

### 1.2 `eval_deterministic.py` (465 lines) — DETERMINISTIC SOLVER EVAL

| Aspect | Detail |
|--------|--------|
| **What it measures** | **Answer correctness** of 6 deterministic solvers (arithmetic, logic, sentiment, NER, factual QA, code debugging) on **real HuggingFace datasets** |
| **Datasets tested** | GSM8K, SVAMP, MathQA, LogiQA, SST-2, IMDB, NCBI Disease, SQuAD 2.0, HumanEval |
| **Metrics tracked** | Accuracy per dataset, per-solver (correct/total), average latency ms, failure pattern recording (first 5 failures per dataset), pattern coverage analysis |
| **Output** | Console only + `deterministic_eval_results.json` saved to root dir |
| **Execution** | `python eval_deterministic.py` (supports `--solver`, `--quick`, `--samples N`) |
| **Persisted artifacts** | ✅ `deterministic_eval_results.json` (contains per-solver/dataset accuracy summary) |
| **Underscores** | Tests solver capability boundaries, not routing. Uses custom `compare()` function for fuzzy matching (different from `evaluate.py`'s `fuzzy_match()`) |

### 1.3 `evaluate.py` (347 lines) — END-TO-END AGENT GRADER (Hackathon Judge)

| Aspect | Detail |
|--------|--------|
| **What it measures** | **End-to-end agent accuracy** — runs the full agent pipeline (detect → classify → solve) on task prompts and compares answers against ground truth |
| **Data sources** | `tasks.txt` (19 hackathon-style tasks), `ground_truth.txt` (19 expected answers) |
| **Metrics tracked** | Per-task pass/fail, overall accuracy %, 84.2% gate check |
| **Output** | Console report only (no JSON/HTML saved) |
| **Execution** | `python evaluate.py` or `python evaluate.py --tasks tasks.txt --ground-truth ground_truth.txt` |
| **Persisted artifacts** | ❌ Nothing to disk, only stdout |
| **Unique features** | Fuzzy matching with 4 strategies (exact, substring, numeric ±1%, token overlap). Runs agent via subprocess with TASKS env var. 600s timeout. |
| **Key difference** | This is the **hackathon grader** — it tests what the judge will test. It doesn't measure routing; it measures whether the final answer is correct. |

### 1.4 `eval_v5_ensemble.py` (68 lines) — OLD QUICK ENSEMBLE EVAL

| Aspect | Detail |
|--------|--------|
| **What it measures** | Hybrid classifier + raw ensemble accuracy on stress test (20) + mixed eval (20) |
| **Metrics tracked** | Per-question pass/fail, per-method usage counts, accuracy %, per-item latency |
| **Output** | Console only |
| **Execution** | `python eval_v5_ensemble.py` |
| **Persisted artifacts** | ❌ Nothing |
| **Notes** | Obsolete — superseded by `classifier_showdown.py` |

### 1.5 `benchmark_classifiers.py` (327 lines) — CLASSIFIER COMPARISON BENCHMARK

| Aspect | Detail |
|--------|--------|
| **What it measures** | 3 approaches (hybrid, ML-only TF-IDF+LR, deterministic-only) + MiniLM+LR on stress test (20) + mixed eval (20) |
| **Metrics tracked** | Accuracy %, average latency ms, agreement matrix, error correlation, cross-validation (leave-one-out, 4-fold) |
| **Output** | Console only |
| **Execution** | `python benchmark_classifiers.py` |
| **Persisted artifacts** | ❌ Nothing |
| **Notes** | Research/prototyping script — trains fresh MiniLM+LR on the fly, does cross-validation. Not for regular use. |

### 1.6 `benchmark_nvidia.py` (182 lines) — NVIDIA MODEL BENCHMARK

| Aspect | Detail |
|--------|--------|
| **What it measures** | `nvidia/prompt-task-and-complexity-classifier` accuracy on stress test + mixed eval |
| **Metrics tracked** | Per-question accuracy, NV task label, confidence, latency ms |
| **Output** | Console only |
| **Execution** | `python benchmark_nvidia.py` |
| **Persisted artifacts** | ❌ Nothing |
| **Notes** | Requires torch, transformers, huggingface_hub, safetensors. Downloads 735MB model. Found 25-30% accuracy — not viable. |

### 1.7 `benchmark_nvidia_complexity.py` (139 lines) — NVIDIA COMPLEXITY EVAL

| Aspect | Detail |
|--------|--------|
| **What it measures** | NVIDIA model's 6 complexity dimensions (creativity, reasoning, domain knowledge, constraint, contextual knowledge, few-shots) vs human difficulty ratings |
| **Metrics tracked** | Per-question complexity scores (6 dims + overall), correlation with difficulty (easy/medium/hard/trick) |
| **Output** | Console only |
| **Execution** | `python benchmark_nvidia_complexity.py` |
| **Persisted artifacts** | ❌ Nothing |

### 1.8 `compare_classifiers.py` (83 lines) — ML vs DETERMINISTIC QUICK COMPARE

| Aspect | Detail |
|--------|--------|
| **What it measures** | ML classifier vs deterministic router on 19 hand-crafted tasks |
| **Output** | Console table showing per-task predictions |
| **Execution** | `python compare_classifiers.py` |
| **Persisted artifacts** | ❌ Nothing |
| **Notes** | Quick research script, 19 fixed tasks, no configurable inputs |

### 1.9 `train_and_compare_all.py` (469 lines) — TRAINING + COMPARISON

| Aspect | Detail |
|--------|--------|
| **What it measures** | Trains TF-IDF+LR + MiniLM+LR side by side on full dataset (146k samples from 8 categories), evaluates on all eval sets |
| **Metrics tracked** | Classification report, accuracy, per-category accuracy, training time |
| **Output** | Console + `.pkl` files to `classifiers/` |
| **Execution** | `python train_and_compare_all.py` |
| **Persisted artifacts** | ✅ Trained `classifier.pkl` files to `classifiers/` directory |
| **Notes** | This is the training pipeline, not really an eval harness. It evaluates during training. |

---

## 2. Data Files (Eval Datasets)

| File | Type | Size |
|------|------|------|
| `eval_hard_100.json` | 100 questions, 8 categories, with difficulty + reasoning | ~340KB |
| `eval_mixed_20.json` | 20 questions, 8 categories (from real HF datasets) | ~10KB |
| `eval_stress_test_20.json` | 20 questions, 8 categories, with difficulty + failure_mode | ~16KB |
| `eval_all_300.json` | 300 questions, 8 categories, with difficulty + reasoning | ~359KB |
| `tasks.txt` | 19 plain-text tasks (the actual hackathon eval) | ~859B |
| `ground_truth.txt` | 19 expected answers for tasks.txt | ~531B |
| `classifiers/complexity_training_data.json` | 78 items with complexity labels | (small) |

---

## 3. Metrics Comparison

| Metric | `classifier_showdown` | `eval_deterministic` | `evaluate.py` | `benchmark_classifiers` |
|--------|:--------------------:|:-------------------:|:------------:|:----------------------:|
| Routing accuracy | ✅ | ❌ | ❌ | ✅ |
| Answer correctness | ❌ | ✅ | ✅ | ❌ |
| Per-category accuracy | ✅ | ✅ (per-dataset) | ❌ | ✅ |
| Confusion matrix | ✅ | ❌ | ❌ | ❌ |
| F1 / precision / recall | ❌ | ❌ | ❌ | ❌ |
| Latency (ms) | ✅ (avg/p50/p99) | ✅ (avg) | ❌ | ✅ (avg) |
| Token count / cost | ❌ | ❌ | ❌ | ❌ |
| Per-difficulty accuracy | ✅ | ❌ | ❌ | ❌ |
| Failure recording | ✅ (per-prompt) | ✅ (first 5) | ✅ (reason string) | ❌ |
| Gate check (84.2%) | ❌ | ❌ | ✅ | ❌ |

---

## 4. Persistence vs Ephemeral

### Persisted to disk
- **`classifier_showdown.py`** → `reports/classifier_showdown_<timestamp>.json` + `.html` (always)
- **`eval_deterministic.py`** → `deterministic_eval_results.json` (always, overwrites)
- **`train_and_compare_all.py`** → `classifiers/classifier.pkl` (always, overwrites)

### Console-only (ephemeral, lost on exit)
- `evaluate.py` — **Nothing saved to disk**. The hackathon grader is pure stdout.
- `eval_v5_ensemble.py` — Console only
- `benchmark_classifiers.py` — Console only
- `benchmark_nvidia.py` — Console only
- `benchmark_nvidia_complexity.py` — Console only
- `compare_classifiers.py` — Console only

---

## 5. Can You Run All Evals With One Command?

**No.** Each harness is a separate entry point:

```bash
# Can't do:
python run_all_evals.py  # doesn't exist

# Must do separately:
python classifier_showdown.py
python eval_deterministic.py
python evaluate.py
python benchmark_classifiers.py
# ... etc
```

There is no unified runner, no shared config, no shared output format.

---

## 6. What's Missing for a "Super Eval"

### Critical gaps

1. **No unified runner** — each harness must be called individually with different args
2. **No shared metrics schema** — every harness has its own result dict structure
3. **No token tracking** — none of the harnesses measure token consumption (input/output/cost), which is the #2 scoring criterion (after accuracy) in the hackathon
4. **No F1/precision/recall** — only accuracy and confusion matrices
5. **No cross-harness comparison** — you can't easily see "what did classifier_showdown say vs what did evaluate.py say?"
6. **`evaluate.py` saves nothing** — the grader's results are purely ephemeral stdout
7. **No regression tracking** — no way to compare "did accuracy go up or down vs last run?"
8. **No cost-per-task tracking** — critical for the hackathon scoring (token count determines ranking)

### Suggested fixes

1. **Add a `run_all_evals.sh` or `unified_eval.py`** that chains all harnesses and collects their outputs
2. **Standardize on a result schema** — every harness should emit a JSON summary file with consistent fields: `{timestamp, harness_name, metrics: {accuracy, latency_ms, ...}, config: {...}, dataset: "..."}`
3. **Add token tracking to `evaluate.py`** — instrument the agent to report tokens used per task
4. **Make `evaluate.py` persist results** — save a JSON report alongside stdout output
5. **Add `sklearn.metrics` calls** — compute F1, precision, recall, classification_report in `classifier_showdown.py` (trivial addition)
6. **Add a comparison mode** — `unified_eval.py --compare last` to diff vs previous run

---

## 7. How `evaluate.py` (Grader) Differs From Classifier Eval Harnesses

| Dimension | `evaluate.py` (Grader) | Classifier Harnesses |
|-----------|----------------------|---------------------|
| **What it tests** | End-to-end agent output | Routing classification |
| **Input** | `tasks.txt` + `ground_truth.txt` (plain text) | `eval_*.json` (structured JSON with category labels) |
| **Answer matching** | 4-strategy fuzzy matching (exact, substring, numeric ±1%, token overlap) | Exact string equality (category match only) |
| **Runs agent?** | **Yes** — spawns subprocess with `python -m agent.main` | No — calls classifier functions directly |
| **Output format** | Console report with ✅/❌ per task + summary + gate check | JSON + HTML with accuracy/latency/confusion |
| **Gate logic** | Checks if accuracy ≥ 84.2% (pass/fail for submission) | No gate — just reports numbers |
| **Timeout** | 600s for entire agent run | No timeout (per-call timing) |
| **Token tracking** | ❌ Not tracked | ❌ Not tracked |
| **Persistence** | ❌ None | ✅ JSON + HTML saved |

The grader (`evaluate.py`) is **not really an eval harness in the same sense** — it's a **submission validator** that simulates what the hackathon judge does. The classifier harnesses are **developer tools** for tuning the routing model offline.

---

## 8. Recommended Unified Evaluation Framework

```python
# unified_eval.py — proposed architecture
# Run: python unified_eval.py            (runs all)
# Run: python unified_eval.py --layer classifier   (subset)

class EvalHarness:
    name: str
    datasets: List[EvalDataset]
    metrics: Dict[str, float]  # standardized output
    
def run_all() -> Dict[str, EvalHarness]:
    # 1. classifier_showdown (routing accuracy)
    # 2. eval_deterministic (solver accuracy)
    # 3. evaluate.py logic (end-to-end agent accuracy)
    # 4. Token tracking (NEW)
    # 5. Save unified report with all metrics
    
unified_result = {
    "timestamp": "...",
    "git_sha": "...",
    "results": {
        "classifier_routing": { accuracy, latency, confusion_matrix, f1_score },
        "deterministic_solvers": { accuracy_by_solver, latency },
        "end_to_end": { accuracy, passed_gate, token_count, estimated_cost },
        "summary": { best_routing, best_overall, tokens_saved_vs_llm_only }
    }
}
```

Key design principles:
1. **Single entry point** — `python unified_eval.py` runs everything
2. **Standardized schema** — every harness produces the same shape of result
3. **Token tracking** — instrument the agent to report tokens consumed
4. **Regression detection** — save historical results, highlight regressions
5. **One report to rule them all** — single HTML/JSON with all layers (routing → solver → agent)
