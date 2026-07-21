# Architecture Review & Research Report

**Date**: 2026-07-13
**Scope**: Gap analysis, fine-tuning feasibility, DSPy integration

---

## 1. Missing Workers Analysis (Per Category)

### 1.1 Math — Accuracy: 80.9% (coder-1.5b) | Priority: HIGH

| Aspect | Detail |
|--------|--------|
| **Current solver** | `solve_arithmetic()` in `deterministic.py` (3280 lines of heuristics) → local LLM → Fireworks fallback |
| **Workflow** | `MATH_3STEP_WORKFLOW` (plan → solve → compose) exists in `workflow.py` but not fully wired into the main pipeline |
| **Failure mode** | Word problems (GSM8K: 57.9%), probability, multi-step reasoning. The LLM outputs *"Let's solve this step by step:"* preamble instead of the final answer — format breakdown |
| **Subtasks** | Arithmetic extraction → equation solving → word problem translation → probability/statistics → geometry |
| **Missing workers** | |
| | 🔴 **Math step-maker** — Converts word problems to symbolic equations before solving. Translates natural language to structured math DSL |
| | 🔴 **Math verifier** — Validates step-by-step reasoning and checks intermediate results |
| | 🟡 **Equation DSL extractor** — Better structured approach than current regex soup in `deterministic.py` |
| | 🟢 Math workflow — Exists but not integrated (steps are defined, just not wired into pipeline use) |
| **Verdict** | **Partially built** — strong deterministic for simple arithmetic, workflow template exists, but word problems and multi-step reasoning need better workers |

### 1.2 Summarization — Accuracy: 37.8% (coder-1.5b) | Priority: HIGHEST

| Aspect | Detail |
|--------|--------|
| **Current solver** | `solve_summarization()` in `deterministic.py` → `secondary_summarization.py` (re-classifier) → Fireworks |
| **Failure mode** | XSum: 28%, validation-v3: 33.3%. Model hallucinates dates, wrong entities, misses the core event. 23/37 summarization questions failed. |
| **Subtasks** | Single-source summary → multi-source comparison → extractive → abstractive → length-constrained |
| **Missing workers** | |
| | 🔴 **Extract-then-compress pipeline** — Extract key sentences first, then compress. Zero-shot approach, no model training needed |
| | 🔴 **Length controller** — Enforces word/sentence count constraints via post-processing |
| | 🔴 **Multi-source synthesizer** — For comparison-style prompts (SOURCE 1/SOURCE 2 patterns detected but not handled well) |
| | 🟡 **Attribution checker** — Ensures summaries refer to correct entities from the source |
| | 🟢 Secondary classification — Exists and works well for re-classifying non-summarization prompts |
| **Verdict** | **Most underserved category** — close to zero deterministic support, relies almost entirely on Fireworks. Even Fireworks struggles (Nemotron-70B via Fireworks). This is the #1 target for improvement. |

### 1.3 Factual — Accuracy: 58.6% (coder-1.5b) | Priority: HIGH

| Aspect | Detail |
|--------|--------|
| **Current solver** | `solve_factual_qa()` in `deterministic.py` + `FactDB` (SQLite FTS5) → local LLM → Fireworks |
| **Failure mode** | Natural Questions (26.3%!). 17/24 factual failures on NQ-open. Model hallucinates wrong answers confidently. Counterfactual questions completely fail. |
| **Subtasks** | Simple fact lookup → multi-part QA → counterfactual reasoning → list queries |
| **Missing workers** | |
| | 🔴 **RAG retriever** — Current `FactDB` uses SQLite FTS5 (keyword matching). No embedding-based semantic retrieval. A `sentence-transformers` + FAISS/Chroma backend would dramatically improve retrieval |
| | 🔴 **Counterfactual reasoner** — Current model fails on "what if" questions. Needs a structured reasoning approach |
| | 🟡 **Web search integration** — For live facts not in the internal DB (web_search.py exists but is not wired) |
| | 🟢 FactDB exists — basic SQLite FTS5 with dolly_facts.jsonl, but no semantic search |
| **Verdict** | **Partially built but critically weak** — FactDB provides zero-latency lookup but misses semantic retrieval. Model accuracy is terrible. |

### 1.4 NER — Accuracy: 77.4% (coder-1.5b) / 88% deterministic | Priority: MEDIUM

| Aspect | Detail |
|--------|--------|
| **Current solver** | `solve_ner()` in `deterministic.py` (regex) + `prototype_ner_v2.py`, `prototype_ner_v3.py` → local LLM → Fireworks |
| **Failure mode** | TweetNER7 (83.3%): missing product/event entities. Biomedical NER (q-02c80855): STK11/KEAP1 mutations not recognized |
| **Subtasks** | Entity identification → entity typing → entity linking → relation extraction |
| **Missing workers** | |
| | 🔴 **Domain-specific NER** — Biomedical NER needs domain vocabulary (genes, diseases, drugs). Current prototype solvers only handle generic entities |
| | 🟡 **Entity linker** — Resolves "NVDA" → "NVIDIA Corporation", "Goldman" → "Goldman Sachs". Missing alias resolution |
| | 🟢 Prototype NER v2/v3 — Partially built, extended entity types beyond basic PER/ORG/LOC |
| **Verdict** | **Partially built** — deterministic handles common cases well but misses domain-specific entities. The prototypes show good progress. |

### 1.5 Sentiment — Accuracy: 75.7% (coder-1.5b) / ~83% with hybrid | Priority: LOW

| Aspect | Detail |
|--------|--------|
| **Current solver** | `sentiment_tree.py` (6-layer decision tree) + `sentiment_cascade.py` (2-level coarse→fine) + VADER + `format_normalizer.py` |
| **Failure mode** | Sarcasm detection: 7/9 failures were sarcastic/hedging text misclassified as "positive". SST2 also has 1 error. |
| **Subtasks** | Coarse label (pos/neg/neu/mixed) → fine emotion (joyful, angry, etc.) → sarcasm detection → hedging detection → domain adaptation |
| **Missing workers** | |
| | 🟡 **Sarcasm specialist** — Current sarcasm patterns in `sentiment_tree.py` are regex-based and miss many cases. Could use a small dedicated model |
| | 🟢 Emotion fine-grainer — Already exists in `sentiment_cascade.py` with 19 fine emotions |
| | 🟢 VADER hybrid — Already integrated |
| **Verdict** | **Sufficient** — Best-served category. 6-layer decision tree + cascade + VADER hybrid + format normalizer is robust. Only sarcasm needs improvement. |

### 1.6 Code Debug — Accuracy: 96.9% | Priority: LOW

| Aspect | Detail |
|--------|--------|
| **Current solver** | `solve_code_debugging()` in `deterministic.py` + `code_sandbox.py` (RestrictedPython) + `verify.py` → local LLM → Fireworks |
| **Failure mode** | Only 1 failure: monotonic list detection — format mismatch (different function name/signature) |
| **Missing workers** | None critical. Tests pass reliably. |
| **Verdict** | **Sufficient** — Well-served by deterministic checks + sandbox execution |

### 1.7 Code Gen — Accuracy: 88.9% | Priority: LOW

| Aspect | Detail |
|--------|--------|
| **Current solver** | `code_sandbox.py` (RestrictedPython AST check) + local LLM → Fireworks |
| **Failure mode** | 4 failures, all from MBPP — function name mismatches (expected `compute_Last_Digit`, got `last_digit_factorial`). Signatures differ. |
| **Missing workers** | |
| | 🟡 **Test runner** — Auto-generates + runs unit tests after generation to verify correctness |
| | 🟡 **Function signature matcher** — Post-processor that renames functions to expected signatures |
| **Verdict** | **Sufficient** — Only format/signature issues, not logic errors |

### 1.8 Logic — Accuracy: 95.0% | Priority: LOW

| Aspect | Detail |
|--------|--------|
| **Current solver** | `logic_solver.py` (python-constraint) + `solve_logic()` in `deterministic.py` → local LLM → Fireworks |
| **Failure mode** | 2 failures: word puzzles (digit logic → format issue), LogiQA (syllogism classification format) |
| **Missing workers** | |
| | 🟡 **Logic puzzle DSL extractor** — Currently regex-heavy, could use structured extraction |
| **Verdict** | **Sufficient** — 95% is excellent. Deterministic constraint solver handles most puzzles. |

### Summary of Gap Severity

```
Priority  | Category       | Accuracy | Key Gap
----------|----------------|----------|-----------------------------------
HIGHEST   | summarization  | 37.8%    | No extract-summarize pipeline
HIGH      | factual        | 58.6%    | No semantic RAG, hallucinates facts
HIGH      | math           | 80.9%    | Word-problem → equation translation
MEDIUM    | ner            | 77.4%    | Domain-specific entity types
LOW       | sentiment      | ~83%     | Sarcasm detection only
LOW       | code_gen       | 88.9%    | Signature/format post-processing
LOW       | code_debug     | 96.9%    | Minor format issues
LOW       | logic          | 95.0%    | Minor format issues
```

### Additional Cross-Cutting Missing Workers

| Missing Worker | Why Needed |
|----------------|------------|
| **Task Splitter** (deterministic or LLM) | No current component decomposes complex multi-step tasks into subtasks. A router that detects "this needs math+logic" and dispatches to sub-workers doesn't exist |
| **Quality Judge** | `verify.py` exists but is basic (hedge detection, short/long checks). A learned judge (judge.py, trained classifier) exists in staging but not wired into pipeline |
| **Secondary resolver** | Only `secondary_summarization.py` exists (re-classifies summarization). No secondary solvers for other ambiguous cases (math-vs-logic, factual-vs-sentiment) |
| **Ensemble aggregator** | Consensus voting exists for math/sentiment/ner (`local_vote.py`) but no weighted ensemble across solvers |

---

## 2. Fine-Tuning qwen2.5-1.5b-base with LoRA: Feasibility Report

### 2.1 Model Availability ✅ BOTH formats available

| Format | Path | Size |
|--------|------|------|
| **GGUF** | `/home/artem/models/qwen2.5-1.5b-base-q4_k_m.gguf` | 986 MB |
| **HuggingFace (SafeTensors)** | `/home/artem/models/Qwen2.5-1.5B-base/` | 3.1 GB (`model.safetensors`) |
| **Config** | `config.json`, `tokenizer.json`, `vocab.json`, `merges.txt` | All present |

The HuggingFace format at `/home/artem/models/Qwen2.5-1.5B-base/` is directly usable with `transformers` + `peft`/`unsloth`. No conversion needed.

### 2.2 Training Data ✅ Suitable format, adequate quantity

**Available datasets:**

| Dataset | Total Items | Per Category | Format |
|---------|-------------|-------------|--------|
| `training-v2.json` | 1,514 | 200 each (code_debug: 114) | `{category, prompt, expected_answer, source, difficulty, task_id}` |
| `training-v3.json` | 152 | 19 each (hard subset) | Same format |
| `validation-v2.json` | 400 | 50 each | Same format (can hold out) |
| `sentiment_train.json` | ~1,200 | sentiment only | Available |
| `sentiment_hard_test.json` | 100 | sentiment only | Available |

**Fine-tuning pair format**: Each item has `prompt` and `expected_answer` — directly usable as `(input, target)` for supervised fine-tuning.

**Data adequacy assessment**:
- LoRA fine-tuning with 100-500 examples per category is well-established in the literature (Hu et al., 2021)
- 200 pairs per category is **above the minimum** for effective LoRA on a 1.5B model
- Research shows 100-300 high-quality examples can produce measurable improvement on classification/generation tasks at this model scale

### 2.3 Hardware Feasibility ✅ COMPORTABLE on RTX A4000 8GB

| Resource | Requirement | Available |
|----------|-------------|-----------|
| VRAM (QLoRA 4-bit, 1.5B) | ~2.5-3 GB | 8 GB ✅ |
| VRAM (LoRA FP16, 1.5B) | ~4-5 GB | 8 GB ✅ |
| Training time (200 examples, 3 epochs) | ~5-15 minutes | ✅ |
| GGUF → HF conversion | Not needed — HF format already available | ✅ |

**Recommendation**: Use **Unsloth** for QLoRA with 4-bit NF4 quantization:
- `pip install unsloth`
- Load model in 4-bit: `FastLanguageModel.from_pretrained("Qwen/Qwen2.5-1.5B", load_in_4bit=True)`
- Add LoRA adapters with `r=16`, `lora_alpha=16`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- Training throughput on A4000: ~8,000-15,000 tokens/sec

### 2.4 Adapter Size Recommendations

| LoRA Rank | Parameters | VRAM Overhead | Expected Benefit | Recommendation |
|-----------|-----------|--------------|-----------------|----------------|
| r=8 | ~4M | ~16 MB | Good for simple classification | For sentiment, NER |
| **r=16** | **~8M** | **~32 MB** | **Best balance** | **Default for most categories** |
| r=32 | ~16M | ~64 MB | Slightly better for generation | For code_gen, summarization |
| r=64 | ~32M | ~128 MB | Diminishing returns | Overkill for 1.5B |

### 2.5 Which Category Would Benefit MOST?

Ranked by potential impact:

| Rank | Category | Current Accuracy | Training Pairs | Expected Improvement | Rationale |
|------|----------|-----------------|---------------|-------------------|-----------|
| **1** | **summarization** | 37.8% | 200 | **+20-30%** | Worst accuracy, benefits from instruction-tuned extract-then-compress |
| **2** | **factual** | 58.6% | 200 | **+15-25%** | Knowledge injection via fine-tuning on Q/A pairs |
| **3** | **math** | 80.9% | 200 | **+8-15%** | Reason-format tuning (teach it to output final answer without preamble) |
| 4 | ner | 77.4% | 200 | +5-10% | Domain entity vocabulary injection |
| 5 | sentiment | ~83% | 200+ | +3-5% | Already well-supported by deterministic |
| 6 | code_gen | 88.9% | 200 | +3-5% | Format consistency improvement |
| 7 | logic | 95.0% | 200 | +1-3% | Already good |
| 8 | code_debug | 96.9% | 114 | +1-2% | Already excellent |

**Top candidate: summarization** — 37.8% is unacceptable. Fine-tuning on `(prompt with text, expected summary)` pairs would directly teach the model to extract key information and compress it.

**Second candidate: factual** — With 200 NQ-open + counterfactual pairs, fine-tuning could teach the model to output precise factual answers instead of hallucinating.

**Third candidate: math** — 80.9% → primarily a format problem. Fine-tuning on GSM8K problems with `Answer: <number>` format would fix the "Let's solve this step by step:" preamble issue (which is 11/18 failures).

### 2.6 Fine-Tuning Pathway (Recommended Procedure)

```
1. Install: pip install unsloth transformers datasets peft trl bitsandbytes

2. Load base model:
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       "Qwen/Qwen2.5-1.5B",  # or local: "/home/artem/models/Qwen2.5-1.5B-base/"
       load_in_4bit=True,  # QLoRA
   )

3. Add LoRA:
   model = FastLanguageModel.get_peft_model(
       model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
   )

4. Format training data:
   training-format = {
       "text": f"### Question: {prompt}\n### Answer: {expected_answer}"
   }

5. Train with SFTTrainer:
   from trl import SFTTrainer
   trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
   trainer.train()

6. Export:
   model.save_pretrained("models/adapters/qwen-math-lora-r16")
   tokenizer.save_pretrained("models/adapters/qwen-math-lora-r16")

7. Inference during pipeline:
   Load base model + LoRA adapter, generate, route output through pipeline
```

**Total time to train 4 adapters**: ~30-60 minutes (summarization + factual + math + ner)
**Total disk space for 4 adapters**: ~128 MB (4 × 32 MB)

---

## 3. DSPy Integration Assessment

### 3.1 What is DSPy?

[DSPy](https://dspy.ai) is a framework for algorithmically optimizing LM prompts and weights. Instead of manually crafting prompts, you define a **module** (signature + reasoning pattern) and an **optimizer** that automatically finds good few-shot examples and prompt instructions.

**Key concepts:**
- **`dspy.Signature`** — Declares input/output fields with descriptions
- **`dspy.Module`** — A compiled program (e.g., `ChainOfThought`, `ReAct`, `ProgramOfThought`)
- **`dspy.Optimizer`** — Improves the module using training data (e.g., `BootstrapFewShot`, `MIPROv2`, `COPRO`)

### 3.2 DSPy vs GEPA — Comparison

| Aspect | GEPA (Current) | DSPy (Proposed) |
|--------|---------------|-----------------|
| **Algorithm** | Genetic algorithm — mutate prompt templates, Pareto-optimize accuracy vs latency | Gradient-free optimization — bootstrap few-shot examples, optimize prompt structure |
| **Search space** | System prompt templates + user templates + temperature + max_tokens | Few-shot demonstrations + prompt instructions + module structure |
| **Optimization target** | Prompt text only | Prompt text + few-shot examples + LM call structure |
| **Model support** | llama-cpp-python (local GGUF) | Any LM via `dspy.LM` adapter (OpenAI, local via llama-cpp) |
| **Output** | Best prompt template (system + user) | Best module (prompt + few-shot examples + inference program) |
| **Strengths** | Works fully offline, Pareto multi-objective (acc+latency+tokens) | Optimizes the whole LM interaction, better few-shot management |
| **Weaknesses** | No few-shot optimization, no module-level optimization | Requires training data, less mature Pareto optimization |

**They are COMPLEMENTARY** — DSPy optimizes the LM call itself (few-shot examples, module structure), GEPA optimizes prompt templates. Used together, they would cover different dimensions.

### 3.3 Can DSPy Work with llama-cpp-python? ✅ YES

DSPy has built-in support for local models via `dspy.LM`:

```python
import dspy

# Works with any llama-cpp-python compatible model
lm = dspy.LM('llamacpp', model='/home/artem/models/qwen2.5-1.5b-base-q4_k_m.gguf')
dspy.configure(lm=lm)
```

Or via the OpenAI-compatible API (when llama.cpp server is running):

```python
lm = dspy.OpenAI(
    api_base="http://localhost:8080/v1",
    api_key="",
    model="qwen2.5-1.5b"
)
```

### 3.4 Integration into the Colab Notebook

#### Current GEPA Notebook Structure

The notebook (`colab/GEPA_Evolution_Notebook.ipynb`) has:
1. Setup (install deps, mount Drive, download model)
2. Configuration (eval data, N generations, population size)
3. Core GEPA loop (evaluate_cell, crossover, mutation, Pareto front)
4. Results display + Save to Drive

#### Where DSPy Fits

DSPy would NOT replace GEPA — it would add a second "optimization engine" cell block:

```
┌──────────────────────────┐
│    Cell: Load Model      │  ← shared model loader
└────────┬─────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│ GEPA   │ │ DSPy   │  ← run independently or sequentially
│ Engine │ │ Engine │
└───┬────┘ └───┬────┘
    │          │
    ▼          ▼
┌────────┐ ┌────────┐
│ Best   │ │ Best   │
│ Prompt │ │ Module │
└───┬────┘ └───┬────┘
    │          │
    └────┬─────┘
         ▼
┌──────────────────┐
│  Compare Results │  ← accuracy, tokens, latency
└──────────────────┘
```

#### Specific Integration Steps

1. **Add `pip install dspy` to the setup cell**

2. **Add a new cell block: `DSPy Optimizer (Alternative)`**

```python
# === DSPy Optimization Engine ===
import dspy

# 1. Configure LM
lm = dspy.LM('llamacpp', 
    model=model_path,  # re-use the same GGUF model
    n_gpu_layers=-1,
    n_ctx=2048,
)
dspy.configure(lm=lm)

# 2. Define category signature
class SentimentClassifier(dspy.Signature):
    """Classify text sentiment as positive, negative, neutral, or mixed."""
    text = dspy.InputField(desc="The text to classify")
    label = dspy.OutputField(desc="One word: positive, negative, neutral, mixed")

# 3. Define module
class GePASentiment(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(SentimentClassifier)
    
    def forward(self, text):
        return self.classify(text=text)

# 4. Load training data (same eval_data used by GEPA)
from dspy.datasets import Dataset
trainset = [dspy.Example(text=item['prompt'], 
                          label=item['expected_answer']).with_inputs('text')
            for item in eval_data]

# 5. Run optimizer
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
optimizer = BootstrapFewShotWithRandomSearch(
    metric=lambda gold, pred: fuzzy_match(pred.label, gold.label),
    max_bootstrapped_demos=8,
    max_labeled_demos=16,
)
compiled = optimizer.compile(GePASentiment(), trainset=trainset)

# 6. Evaluate
accuracy = evaluate_module(compiled, eval_data)

# 7. Export results alongside GEPA results
```

3. **Add result comparison** — A cell that compares Pareto fronts from both engines:
   - GEPA: best accuracy / tokens / latency
   - DSPy: best accuracy / token budget
   - Winner per metric

### 3.5 When to Use DSPy vs GEPA

| Situation | Use DSPy | Use GEPA |
|-----------|----------|----------|
| Few-shot examples help | ✅ | ❌ |
| Prompt template needs optimization | ✅ | ✅ (better) |
| Multi-objective (acc+latency) | Limited | ✅ (native) |
| Working offline with local GGUF | ✅ | ✅ (native) |
| Need structured chain-of-thought | ✅ (native modules) | Manual |
| Large evaluation set (1000+) | ✅ | ✅ |
| Very small evaluation set (<50) | Limited | ✅ (genetic works with small) |

### 3.6 DSPy Integration Effort

| Task | Effort | Dependencies |
|------|--------|-------------|
| Add `dspy` to setup cell | 1 line | None |
| Create DSPy LM adapter | 5 lines | llama-cpp-python already installed |
| Define signatures per category | 8 signatures × 5 lines | None |
| BootstrapFewShot optimizer | 10 lines | Training data ready |
| Evaluation loop | 20 lines | Reuse `fuzzy_match` from GEPA |
| Result comparison display | 15 lines | GEPA results structure |
| **Total** | **~60 lines of new code** | All dependencies already met |

---

## 4. Ranked Recommendations

### Tier 1: Build Now (High Impact, Low Effort)

| # | Action | Category | Expected Impact | Effort |
|---|--------|----------|----------------|--------|
| 1 | **Fine-tune summarization LoRA** | summarization | +20-30% accuracy | ~10 min training |
| 2 | **Fine-tune factual LoRA** | factual | +15-25% accuracy | ~10 min training |
| 3 | **Fine-tune math LoRA** (format fix) | math | +8-15% accuracy | ~10 min training |
| 4 | **Extract-then-compress pipeline** | summarization | +10-20% accuracy | 1 day dev |
| 5 | **Add DSPy optimizer to Colab notebook** | all | Alternative engine | ~60 lines (easy) |

### Tier 2: Build Next (Medium Impact, Medium Effort)

| # | Action | Category | Expected Impact | Effort |
|---|--------|----------|----------------|--------|
| 6 | **Semantic RAG for factual** (sentence-transformers + FAISS) | factual | +10-20% | 2 days |
| 7 | **Fine-tune NER LoRA** (biomedical entities) | ner | +5-10% | ~10 min training |
| 8 | **Math word-problem DSL extractor** | math | +5-10% | 2 days |
| 9 | **Task splitter (LLM-based)** | cross-cutting | Enables multi-step | 1 day |
| 10 | **Learnable judge integration** | cross-cutting | Better quality gate | 1 day (model exists in staging) |

### Tier 3: Build Later (Lower Impact / Larger Effort)

| # | Action | Category | Expected Impact | Effort |
|---|--------|----------|----------------|--------|
| 11 | Web search integration | factual | +5-10% | 2 days |
| 12 | Sarcasm specialist model | sentiment | +3-5% | 2 days |
| 13 | Multi-source summarizer | summarization | +5-10% | 3 days |
| 14 | Ensemble voting across all solvers | cross-cutting | +2-5% | 3 days |
| 15 | All 8 categories fine-tuned as LoRA workers | all | +5-20% each | 1 hour total |

### The "Army of Qwens" Vision

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  Category Filter (8-way)     │  ← deterministic + ML
│  + Secondary Resolver        │
└──────────┬──────────────────┘
           │
    ┌──────┴──────┐
    │  Task        │  ← NEW: Task Splitter
    │  Splitter    │     (deterministic or LLM)
    └──────┬──────┘
           │
    ┌──────┴──────┐
    │  Worker      │
    │  Router      │  ← Maps sub-tasks to specialized workers
    └──────┬──────┘
           │
    ┌──────┴──────────────────────────────┐
    │  Worker Pool (all LoRA-adapted)      │
    │                                      │
    │  ┌─────────┐ ┌──────────┐ ┌───────┐ │
    │  │ Qwen    │ │ Qwen     │ │ Qwen  │ │
    │  │ Math    │ │ Factual  │ │ Summar│ │  ← LoRA adapters on base
    │  │ (r=16)  │ │ (r=16)   │ │ (r=16)│ │     model, hot-swappable
    │  └─────────┘ └──────────┘ └───────┘ │
    │  ┌─────────┐ ┌──────────┐ ┌───────┐ │
    │  │ Qwen    │ │ Qwen     │ │ Qwen  │ │
    │  │ NER     │ │ Sentiment│ │ Logic │ │
    │  │ (r=8)   │ │ (r=8)    │ │ (r=16)│ │
    │  └─────────┘ └──────────┘ └───────┘ │
    │  ┌─────────┐ ┌──────────┐            │
    │  │ Qwen    │ │ Qwen     │            │
    │  │ Code    │ │ Debug    │            │
    │  │ (r=16)  │ │ (r=16)   │            │
    │  └─────────┘ └──────────┘            │
    └──────────────────────────────────────┘
           │
           ▼
    ┌──────────────┐
    │  Ensemble /   │
    │  Aggregator   │  ← NEW: weighted voting
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  Quality Gate │  ← verify.py + learned judge
    │  (Judge)      │
    └──────┬───────┘
           │
    ┌──────┴───────┐
    │  Fireworks    │  ← fallback for hard cases
    │  Fallback     │
    └──────────────┘
```

### Immediate Next Steps (This Sprint)

1. `cd /home/artem/dev/amd-hackathon && python3 scripts/finetune/finetune_summarization.py` — Train summarization LoRA adapter
2. `python3 scripts/finetune/finetune_factual.py` — Train factual LoRA adapter
3. `python3 scripts/finetune/finetune_math.py` — Train math LoRA adapter  
4. Wire LoRA adapters into pipeline (category → adapter mapping)
5. Add DSPy optimizer cell to `colab/GEPA_Evolution_Notebook.ipynb`
6. Build extract-then-compress pipeline for summarization

---

## Appendix: Key Files Audit

| File | Role | Status |
|------|------|--------|
| `agent/pipeline.py` | Main routing pipeline | ✅ Production-ready |
| `agent/category_filter.py` | 8-way deterministic classifier | ✅ Production-ready |
| `agent/workflow.py` | Multi-step workflow engine | ✅ Build (math/logic/ner templates exist) |
| `agent/solvers/deterministic.py` | 3280-line deterministic solver hub | ✅ Mature (arithmetic, logic, NER, sentiment, factual, code_debug, summarization) |
| `agent/solvers/sentiment_tree.py` | 6-layer decision tree | ✅ Mature |
| `agent/solvers/sentiment_cascade.py` | 2-level coarse→fine emotions | ✅ Mature |
| `agent/solvers/logic_solver.py` | python-constraint logic solver | ✅ Functional |
| `agent/solvers/code_sandbox.py` | RestrictedPython sandbox | ✅ Functional |
| `agent/solvers/fact_db.py` | SQLite FTS5 factual QA | ✅ Functional (no embeddings) |
| `agent/solvers/fw_router.py` | Fireworks model routing | ✅ Mature (8 categories, caveman prompts) |
| `agent/solvers/verify.py` | Output quality gates | ✅ Functional |
| `agent/judge.py` | Learned judge classifier | 🟡 In staging, not wired |
| `agent/secondary_summarization.py` | Summarization re-classifier | ✅ Functional |
| `colab/GEPA_Evolution_Notebook.ipynb` | GEPA Colab notebook | ✅ Mature (genetic prompt optimization) |
| `data/eval/training-v2.json` | 1,514 training pairs | ✅ Ready for fine-tuning |
| `data/eval/validation-v2.json` | 400 validation pairs | ✅ Ready for fine-tuning |
| `models/Qwen2.5-1.5B-base/` | Base model (HF format) | ✅ Ready for LoRA |
| `models/qwen2.5-1.5b-base-q4_k_m.gguf` | Base model (GGUF) | ✅ Ready for inference |
