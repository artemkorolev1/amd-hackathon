# AMD ACT II Hackathon — Research Recommendations

## Part A: Training & Evaluation Data Strategy
## Part B: LoRA Fine-Tuning Best Practices for Qwen2.5-3B-Instruct

> Generated: 2026-07-12
> Target: Qwen2.5-3B-Instruct (Docker CPU, 2 cores, 4GB RAM)
> Grading: fuzzy_match() — exact, substring, token overlap ≥ 70%, numeric tolerance 5%

---

# PART A: TRAINING & EVALUATION DATA STRATEGY

---

## A1. Best Public Datasets Per Category (for fuzzy_match-compatible eval)

The critical constraint is that `fuzzy_match()` (exact match → substring → token overlap ≥ 70% → numeric tolerance 5%) requires **short, determinate expected answers**. Here are the best datasets for each category, annotated with how to extract fuzzy_match-compatible answers:

### sentiment
| Dataset | HF ID | Best For | Expected Answer Extraction |
|---------|-------|----------|---------------------------|
| **SST-2** | `glue/sst2` | Binary pos/neg basics | `label` field → "positive"/"negative" |
| **TweetEval (sentiment)** | `tweet_eval` (sentiment config) | Sarcasm, irony, nuanced | `label` (0=negative, 1=positive, 2=neutral) |
| **GoEmotions** | `google-research-datasets/go_emotions` | 27 fine-grained emotions | Map to pos/neg/neutral/mixed |
| **IMDB** | `stanfordnlp/imdb` | Longer-form sentiment | `label` → "positive"/"negative" |

**Key insight**: SST-2 only gives binary labels — the hardest sentiment questions require **sarcasm detection** where all surface words are positive. TweetEval and GoEmotions are essential for nuance.

### ner
| Dataset | HF ID | Expected Answer Format |
|---------|-------|----------------------|
| **CoNLL-2003** | `conll2003` | `PER: John; ORG: Acme` (structured) |
| **OntoNotes 5.0** | `tner/ontonotes5` | 18 entity types, structured format |
| **NCBI Disease** | `ncbi/ncbi_disease` | `DISEASE: cancer` (biomedical) |
| **WNUT 2017** | `wnut_2017` | Emerging/rare entities (hard) |

**Format strategy**: Convert BIO tags to `TYPE: entity` semicolon-separated format. This works with token_overlap ≥ 70% matching. Train the model explicitly on this format.

### math
| Dataset | HF ID | Expected Answer Extraction |
|---------|-------|---------------------------|
| **GSM8K** | `openai/gsm8k` | Regex `#### (-?\d+(?:\.\d+)?)` — extract final number |
| **MATH** | `hendrycks/math` | Final answer after `\boxed{}` |
| **SVAMP** | `nguyen-brat/svamp` | `Answer` field — numeric |
| **MathQA** | `allenai/math_qa` | Choice letter (A-E) or extracted number |

**Key insight**: GSM8K answers are always numeric → perfect for fuzzy_match numeric_tolerance_5pct. MATH has competition-level problems (level 4-5) for hard questions.

### code_gen
| Dataset | HF ID | Expected Answer Format |
|---------|-------|----------------------|
| **HumanEval** | `openai/openai_humaneval` | Function signature + body |
| **MBPP** | `google-research-datasets/mbpp` | Full function body |
| **CodeAlpaca-20k** | `sahil2801/CodeAlpaca-20k` | Code snippets |

**Strategy**: Use function signature as expected_answer (e.g., `def max_of_three` works for substring match). For full code, the token overlap strategy works.

### code_debug
| Dataset | HF ID | Expected Answer |
|---------|-------|----------------|
| **HumanEvalPack** | `bigcode/humanevalpack` | Fixed code (buggy → fixed pair) |
| **QuixBugs** | `jkooooonn/QuixBugs` | Bug-introduced → bug-free pairs |
| **CodeXGLUE (defect)** | `microsoft/codexglue` (defect detection) | Binary buggy/clean |

### factual
| Dataset | HF ID | Expected Answer |
|---------|-------|----------------|
| **MMLU** | `cais/mmlu` | Choice letter (A/B/C/D) |
| **TruthfulQA** | `truthful_qa` | Short answer string |
| **TriviaQA** | `trivia_qa` | Short answer string |
| **SQuAD 2.0** | `rajpurkar/squad_v2` | Span from context (short) |

### logic
| Dataset | HF ID | Expected Answer |
|---------|-------|----------------|
| **LogiQA** | `lucasmccabe/logiqa` | Choice letter (A-D) |
| **ZebraLogic** | `allenai/ZebraLogicBench` | Structured assignment string |
| **HellaSwag** | `hellaswag` | Completion index (0-3) |
| **WinoGrande** | `winogrande` | Choice (A or B) |
| **BBH** | `lukaemon/bbh` | Short answer per subtask |

**Key insight**: Multiple-choice answers (A/B/C/D) are excellent for fuzzy_match — exact string match. LogiQA + ZebraLogic provide the core; WinoGrande/HellaSwag provide adversarial difficulty.

### summarization
| Dataset | HF ID | Expected Answer |
|---------|-------|----------------|
| **XSum** | `EdinburghNLP/xsum` | Single sentence (good for fuzzy token overlap) |
| **CNN/DailyMail** | `cnn_dailymail` | Multi-sentence highlight (harder for fuzzy) |
| **SAMSum** | `samsum` | Dialogue summary (short) |

**Strategy**: XSum's single-sentence summaries work best with token_overlap_70pct. For CNN/DM, expect specific key facts/numbers rather than full summaries.

---

## A2. Synthetic Data Generation — Best Practices & Tools

### Self-Instruct (Wang et al., 2022)
**Paper**: https://arxiv.org/abs/2212.10560
**Best for**: Bootstrapping 100-200 new eval questions per category

**Recommendation**: Use GPT-4o or Claude Sonnet to generate 20 seed questions per category → generate 100 variants → filter duplicates → validate expected_answer against fuzzy_match.
- **Key**: Always include the `expected_answer` in the generation prompt, with explicit format instructions.
- **Tool**: No external tool needed — a simple Python script calling the LLM API suffices.

### Evol-Instruct (Xu et al., 2023, WizardLM)
**Paper**: https://arxiv.org/abs/2304.12244
**Best for**: Hardening easy questions → hard versions

**5 operations to implement**:
1. **Add constraints** — "Write code that... but in one line / without loops"
2. **Deepen reasoning** — "Explain X, then apply it to Y, then compare with Z"
3. **Concretize** — Replace vague terms with specific entities
4. **Increase reasoning steps** — Add intermediate sub-questions
5. **Complicate input** — Add distractors, irrelevant data, noise

**Recommendation**: Take existing easy questions (from eval_simple_100 or SST-2/GSM8K easy samples) and run Evol-Instruct to create hard+ variants. Estimated yield: 3-5x per seed question.

### Template-Based Generation (Fast & Deterministic)
**Best for**: Rapid scaling of known answer formats

```python
# Math templates
templates = [
    "What is {a} + {b}?",
    "If x = {a} and y = {b}, what is x * y?",
    "Solve for x: {a}x + {b} = {c}"
]
# NER templates
templates = [
    "Extract named entities: '{sentence}'",
    "Find all PERSON and ORGANIZATION mentions in: '{sentence}'"
]
# Sentiment templates
templates = [
    "Classify the sentiment: '{text}'",
    "Is this positive or negative? '{text}'"
]
```

**Recommendation**: Build a template engine that fills placeholders from entity lists (countries, names, numbers, etc.) and auto-computes expected_answers. This is the fastest way to get 500+ questions.

### Distilabel (Argilla)
**URL**: https://github.com/argilla-io/distilabel
**Best for**: End-to-end synthetic data pipeline with quality control

Distilabel provides:
- Pre-built pipelines for self-instruct, evol-instruct, ultrafeedback
- Integration with 50+ LLM providers
- Quality filtering and deduplication
- Annotation/curation UI

**Recommendation**: Use Distilabel for any large-scale (1000+) synthetic generation. Its `TextGeneration` and `EvolInstruct` pipelines can be adapted to the 8 hackathon categories.

### Argilla
**URL**: https://github.com/argilla-io/argilla
**Best for**: Human-in-the-loop curation of generated data

Use Argilla to:
- Review and correct generated expected_answers
- Flag low-quality generations
- Handle edge cases
- Build a clean curated eval set

---

## A3. Scaling Strategy: 1000+ Balanced Evaluation Questions

### Phased Approach

#### Phase 1: Public Dataset Conversion (400+ questions, 1-2 hours)
Use `convert_datasets_to_eval.py` pattern — already partially done. Extend to all 8 categories:

| Category | Source | Questions | Answer Extraction |
|----------|--------|-----------|-------------------|
| sentiment | SST-2 test split | 50 | `label` → "positive"/"negative" |
| sentiment | TweetEval (sarcasm subset) | 20 | Manual label mapping |
| ner | CoNLL-2003 test split | 50 | BIO → structured format |
| ner | NCBI Disease test split | 20 | DISEASE entity extraction |
| math | GSM8K test split + MATH easy | 60 | `####` number or `\boxed{}` |
| math | MATH hard (level 4-5) | 20 | `\boxed{}` extraction |
| code_gen | HumanEval + MBPP test | 60 | Function signature |
| code_debug | HumanEvalPack | 30 | Fixed code snippet |
| factual | MMLU test (easy subjects) | 50 | Choice letter |
| factual | TruthfulQA | 20 | Short answer |
| logic | LogiQA test | 40 | Choice letter |
| logic | WinoGrande test | 20 | A or B |
| summarization | XSum test | 30 | Single sentence |
| **Total** | | **~470** | |

#### Phase 2: Template-Based Generation (300+ questions, 2-3 hours)
Build parameterized templates for each category:

- **Math**: 50 arithmetic templates × 3 difficulty levels = 150 questions
- **NER**: 20 sentence templates × 5 entity variations = 100
- **Sentiment**: 30 templates × 3 variations = 90
- **Factual**: 50 "What/Who" templates = 50

#### Phase 3: Self-Instruct + Evol-Instruct (300+ questions, using Fireworks/OpenAI)
- Generate 20 hard questions per category (160 total) focused on known weaknesses
- Apply Evol-Instruct to harden 20 easy questions per category (160 total)
- Human review for answer correctness is critical — LLMs often get expected_answers wrong

#### Phase 4: Adversarial Filtering (Ongoing)
- Run current best model on candidate questions
- Keep only questions the model gets wrong
- This naturally creates a challenging eval set that differentiates between approaches

### Target Distribution (1000 questions)

| Category | Easy | Medium | Hard | Hard+ | Total |
|----------|------|--------|------|-------|-------|
| sentiment | 16 | 19 | 13 | 5 | 53 |
| ner | 16 | 19 | 13 | 5 | 53 |
| math | 16 | 19 | 13 | 5 | 53 |
| code_gen | 16 | 19 | 13 | 5 | 53 |
| code_debug | 16 | 19 | 13 | 5 | 53 |
| factual | 16 | 19 | 13 | 5 | 53 |
| logic | 16 | 19 | 13 | 5 | 53 |
| summarization | 16 | 19 | 13 | 5 | 53 |
| **Total** | **128** | **152** | **104** | **40** | **424** |

For 1000+, scale each category to 125 (×2.35) by adding more easy/medium public dataset conversions.

---

## A4. Sentiment Nuance — Sourcing Sarcasm & Dismissive Data

**This is the single weakest area across all models (25-54% accuracy).**

### Data Sources

| Source | Description | Items | How to Convert |
|--------|-------------|-------|----------------|
| **TweetEval (sentiment)** | 3-class (pos/neg/neutral), includes sarcastic tweets | ~12K test | Direct label → expected_answer |
| **iSarcasm Dataset** | 2,500 sarcastic tweets, labeled | 2,500 | Map to "negative" for sarcasm |
| **SARC v2 (SARCASM Corpus)** | 1.3M Reddit comments with sarcasm labels | 1.3M | Needs filtering, large |
| **GoEmotions** | 58K Reddit, 27 emotions → map to pos/neg/neutral/mixed | 58K | Group emotions: joy/amusement→positive, anger/disappointment→negative |
| **MUStARD** | 690 multimodal sarcasm examples (text transcript) | 690 | Extract text → sentiment label |
| **Contrast Set Method** | Generate pairs: flip sentiment with minimal edits | Unlimited | Manual/instruct pipeline |

### Recommended Action

1. **Download TweetEval and GoEmotions** immediately — they're the highest-value sources for nuanced sentiment
2. **Build contrast sets**: Take 20 neutral SST-2 examples, have an LLM generate sarcastic/dismissive variants with flipped expected answers
3. **Create explicit nuance types** in the eval set (sarcasm, dismissive, backhanded, mixed, faint praise)
4. **Hand-craft 10-15 hard sentiment examples** using the template from `complexity_eval_40_quirky.json`

### Example Nuanced Questions to Include

```json
{"prompt": "Classify the sentiment: 'Oh great, another hour-long meeting about meetings.'",
 "expected_answer": "negative", "nuance": "sarcasm"}
{"prompt": "Classify the sentiment: 'I'll take that suggestion under advisement.'",
 "expected_answer": "negative", "nuance": "dismissive"}
{"prompt": "Classify the sentiment: 'You're so articulate for someone of your background.'",
 "expected_answer": "negative", "nuance": "backhanded_compliment"}
{"prompt": "Classify the sentiment: 'Well, that certainly was an experience.'",
 "expected_answer": "negative", "nuance": "deadpan_understatement"}
```

---

## A5. NER Format Training

**Problem**: Models produce freeform text like "The entities are John and Acme" instead of `PERSON: John; ORG: Acme`.

### Solution Strategy

1. **Training data format**: Train the model on `prompt → structured response` pairs where the response is exactly:
   ```
   PERSON: John Smith; ORGANIZATION: Acme Corp; LOCATION: New York
   ```
2. **Critical in-context example**: Always include 1-2 examples in the prompt showing the exact output format before asking. Use few-shot prompting:
   ```
   Extract named entities. Format: TYPE: entity; TYPE: entity
   
   Example:
   Text: "Apple CEO Tim Cook announced the new iPhone."
   Output: PERSON: Tim Cook; ORGANIZATION: Apple
   
   Now extract from: "{text}"
   ```
3. **Post-processing enhancement**: After LLM output, apply a regex to match `TYPE: entity` patterns and reformat if needed. This catches 80%+ of format issues.
4. **NER eval questions**: Expected_answer should always be in the structured format so `fuzzy_match` can use token_overlap_70pct across entity tokens.

---

## A6. Data Quality for Small Model Fine-Tuning

### What makes good training data for 1.5B-3B models

1. **Concise responses** — small models cannot handle long output distributions. Target ≤150 tokens per response.
2. **Consistent format** — every example in a category should follow the same structural template
3. **High signal-to-noise ratio** — no irrelevant preamble, no "Let me think about this" boilerplate
4. **Clean expected answers** — answers must directly match what fuzzy_match expects (short, determinate)
5. **Balanced difficulty** — too many easy examples → model learns patterns; too many hard → model can't learn anything
6. **Negative examples** — include examples where the correct answer is "no" or "neutral" or "None" to prevent positive bias

### Data Quality Checklist

| Criterion | Bad Example | Good Example |
|-----------|-------------|--------------|
| Response length | 200+ tokens explaining step-by-step | 1-10 tokens (answer only) |
| Format consistency | 5 different output formats | One format per category |
| Signal-to-noise | "Let me analyze this carefully..." | Direct answer |
| Answer correctness | LLM-generated, unverified | Extracted from gold dataset |
| Category clarity | Vague "problem solving" label | Exact 8-way category |

---

# PART B: LoRA FINE-TUNING BEST PRACTICES FOR QWEN2.5-3B-INSTRUCT

---

## B1. Root Cause Analysis of 43-50% Crash (from v12e results)

**Baseline without LoRA: 75.3% → With LoRA: 43-50%** — a catastrophic 25-32 point drop.

### Bug #1: Adapter Weight Accumulation (CRITICAL)

**Issue**: The notebook loads the base model once, trains adapter 1 (factual), then adapter 2 (math) on TOP of adapter 1's weights, etc. By adapter 8 (logic), the weights have been modified 8 times.

**Why this destroys quality**: Each subsequent adapter overwrites the previous adapter's specialized weights. The final model is a garbled mixture where no single task performs well. The only reason sentiment improved (50→71%) is that it was the 3rd adapter trained — its positive effect partially survived the subsequent NER and summarization training through the LoRA additive update residual.

**Evidence**: 
- factual: 84% → 0-50% (2nd adapter trained, then overwritten by 6 more)
- math: 65% → 0% (3rd adapter trained, then overwritten by 5 more)
- sentiment: 50% → 71-86% (4th adapter trained, only 4 more adapters after it)

### Bug #2: Data Format Mismatch (CRITICAL)

**Issue**: `formatting_func()` expects `example['messages']` (chat format) but the actual JSONL has `prompt`/`response`/`category`/`source` fields. The call to `tokenizer.apply_chat_template(messages, tokenize=False)` would either:
- Silently produce garbled text if `messages` key doesn't exist (Python KeyError then fallback)
- Or fail and produce meaningless training sequences

**Result**: The model was trained on nonsensical input-output pairs, destroying its instruction-following ability.

### Bug #3: Formulaic Response Patterns

**Issue**: The sentiment training data uses formulaic responses like:
```json
{"response": "Negative. The text conveys a negative or critical tone with language like 'secretions from the parental units'."}
```

This teaches the model to:
- Always output verbose analysis (bad for fuzzy_match which expects short answers)
- Mirror the specific phrasing patterns from the training data
- Ignore the actual question in favor of templated response structures

**Why sentiment improved despite this**: The formulaic responses consistently ended with "positive" or "negative" as the first word, so the model learned to output the right label (just with extra fluff that fuzzy_match could still match via substring).

### Bug #4: lora_r=16 Overfitting on 2000 Examples

**Issue**: For a 1.5B model, lora_r=16 with 4-bit quantization provides ~4.7M trainable parameters. With only 2000 examples per category and 2 epochs, the model quickly overfits to the specific response patterns.

**Evidence**: 
- code_gen and code_debug maintained performance → these had the most natural, diverse responses
- math, factual, logic collapsed → these had the most formulaic, repetitive responses

### Bug #5: Learning Rate Too High

**Issue**: lr=2e-4 is standard for full LoRA on larger models (7B+), but for 1.5B with 4-bit quantization, this is too aggressive. Effective batch size of 16 (4×4 accumulation) means very few update steps per epoch (2000/16 = 125 steps). Large LR + small steps → drastic weight shifts → catastrophic forgetting.

### Bug #6: 4-bit Quantization Amplifying All Issues

**Issue**: 4-bit NF4 quantization preserves ~95% of model quality on forward passes, but during training, the quantization errors compound with the LoRA weight updates. For small models (1.5B), each quantization step loses more relative quality than for large models (70B).

### Summary of Corrective Actions

| Issue | Fix |
|-------|-----|
| Adapter accumulation | Reload base model between categories OR train one adapter on combined data |
| Data format mismatch | Convert JSONL to proper chat template or use direct text format |
| Formulaic responses | Regenerate training data with direct, concise answers |
| lora_r=16 | Lower to r=8, increase alpha to 32 for better generalization |
| LR=2e-4 | Reduce to 5e-5 for 1.5B, 1e-4 for 3B |
| 4-bit quantization | Use 8-bit instead (negligible speed difference, better quality) |
| Sequential training | Train one combined adapter on all categories simultaneously |

---

## B2. Optimal LoRA Hyperparameters for Qwen2.5-3B-Instruct

### Recommended Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **lora_r** | 8 | Lower rank prevents overfitting on small (2K/category) dataset. r=4 for very small datasets (<500). |
| **lora_alpha** | 32 | α/r = 4 gives stronger LoRA signal. Higher alpha = stronger adaptation, good for task-specific tuning. |
| **lora_dropout** | 0.1 | Dropout prevents overfitting. Set to 0 only if data is >10K per category. |
| **target_modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj | Qwen2.5 architecture uses all linear layers in attention + FFN. Keeping all 7 is standard. |
| **bias** | 'none' | Standard for LoRA. Training bias adds ~7M params for minimal gain. |
| **learning_rate** | 1e-4 | Conservative for 3B model. Use cosine scheduler with 50-step warmup. |
| **lr_scheduler_type** | cosine | Better than constant — allows settling into optimal region. |
| **warmup_steps** | 50 | 5-10% of total steps. Critical for stable LoRA convergence. |
| **batch_size** | 4 | Limited by T4 GPU memory. |
| **gradient_accumulation** | 4 | Effective batch size = 16. Good for stable gradients. |
| **max_seq_length** | 1024 | Covers all eval questions. Use 2048 only if summarization articles exceed 1024 tokens. |
| **weight_decay** | 0.01 | Prevents LoRA weight drift. |
| **optimizer** | adamw_8bit (bitsandbytes) | Memory-efficient, same quality as full AdamW. |
| **quantization** | **8-bit** (load_in_8bit=True) | 8-bit preserves more quality than 4-bit. Only use 4-bit if GPU memory is tight (<12GB). |
| **epochs** | 2-3 per category | Start with 2, eval after each epoch. Stop early if eval loss increases. |
| **gradient_checkpointing** | 'unsloth' or True | Reduces memory by ~30%. Needed for 3B model on T4. |

### Why These Values

**lora_r=8, lora_alpha=32**: The Hu et al. (2021) paper recommends α/r = 2-4. For task-specific fine-tuning of a 3B model on 16K total examples, r=8 provides sufficient rank to learn category-specific patterns without overfitting. Going to r=16 doubles trainable params (9.4M vs 4.7M) which is unnecessary for the relatively small dataset.

**8-bit > 4-bit quantization**: Dettmers et al. (2023, QLoRA paper) shows 4-bit NF4 works for 7B+ models, but for 3B models the quantization error is a larger fraction of total model capacity. 8-bit uses ~2GB more VRAM but preserves base model quality much better. Test: If GPU has <12GB VRAM, use 4-bit; otherwise use 8-bit.

### Learning Rate Schedule

```
LR schedule config:
  lr_scheduler_type: cosine
  warmup_ratio: 0.05   # 5% of steps for warmup
  learning_rate: 1e-4
```

Alternative (if you want constant LR for simpler tuning):
```
lr_scheduler_type: constant_with_warmup  # Keeps LR after warmup
learning_rate: 5e-5  # Lower for constant schedule
```

---

## B3. Data Formatting — Correct Chat Template for Qwen2.5

### Root Problem

The existing notebook uses `tokenizer.apply_chat_template(messages, tokenize=False)` but the JSONL data has `prompt`/`response` fields, not `messages`. 

### Correct Format for HuggingFace SFTTrainer

**Option A: Convert to ChatML format (Recommended)**

Each JSONL entry should be:
```json
{
  "messages": [
    {"role": "user", "content": "Classify the sentiment: 'This movie was terrible.'"},
    {"role": "assistant", "content": "negative"}
  ],
  "category": "sentiment",
  "source": "sst2"
}
```

This matches Qwen2.5's chat template and works natively with `apply_chat_template()`.

**Option B: Direct Text Format (Faster, less flexible)**

If the notebook just needs `{prompt}\n{response}` without template:
```python
def formatting_func(example):
    text = f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"
    return {'text': text}
```

Then set `dataset_text_field='text'` in SFTTrainer.

### Teaching Concise Answers (THE KEY INSIGHT)

The LoRA should teach the model to output **only the answer**, not analysis. The training data must reflect this:

| ❌ Bad Training Data | ✅ Good Training Data |
|----------------------|----------------------|
| `"Negative. The text conveys a negative or critical tone..."` | `"negative"` |
| `"Let me solve this step by step: 48/2 = 24, 48+24 = 72. Answer: 72"` | `"72"` |
| `"The entities are John (PERSON) and Acme (ORG)."` | `"PERSON: John; ORG: Acme"` |

**Recommendation**: Regenerate ALL training data with concise, direct answers. Use the `expected_answer` format from the eval sets as the template.

### Per-Category Answer Format Template

| Category | Training Response Format | Example |
|----------|-------------------------|---------|
| sentiment | `positive` or `negative` or `neutral` or `mixed` | `negative` |
| ner | `TYPE: entity; TYPE: entity` | `PERSON: John; ORG: Acme` |
| math | `{number}` | `72` |
| code_gen | `{full function code}` | Full Python function |
| code_debug | `{fixed code}` | Corrected line |
| factual | `{short answer}` | `Mars` |
| logic | `{letter}` or `{short answer}` | `C` |
| summarization | `{1 sentence summary}` | Single line |

---

## B4. Catastrophic Forgetting Prevention

### Strategy #1: Single Combined Adapter (RECOMMENDED)

**Instead of training 8 adapters sequentially**, train ONE LoRA adapter on all 16K examples combined.

**Why**: 
- No sequential overwriting
- One forward pass through data trains category-specific patterns simultaneously
- Can control category proportions precisely
- Easier to eval and deploy

**How**:
```python
# Combine all categories
all_data = []
for cat in CATEGORIES:
    data = load_jsonl(f'lora_data/{cat}.jsonl')
    for d in data:
        all_data.append(d)

# Randomize category order
random.shuffle(all_data)

# Train one adapter on all data
dataset = Dataset.from_list(all_data).map(formatting_func)
trainer = SFTTrainer(..., train_dataset=dataset)
trainer.train()
```

### Strategy #2: Independent Adapters with Reload (Alternative)

If you want separate adapters per category:
```python
for category, epochs in CATEGORIES:
    # RELOAD base model for each adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, load_in_8bit=True, ...
    )
    model = FastLanguageModel.get_peft_model(model, r=8, ...)
    train_adapter(category, epochs)
    # Save adapter, then delete model to free memory
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
```

This is slower (each reload takes ~30s) but produces truly independent adapters.

### Category Mixing Proportions

For the combined adapter approach, balance the categories but avoid equal weighting:

| Category | Current Items | Target Proportion | Rationale |
|----------|--------------|-------------------|-----------|
| sentiment | 2000 | 15% | Weakest baseline, needs most help |
| ner | 2000 | 12% | Middle performance |
| math | 2000 | 15% | Hard math needs more data |
| code_gen | 2000 | 10% | Already strong |
| code_debug | 2000 | 10% | Already strong |
| factual | 2000 | 13% | Moderate |
| logic | 2000 | 15% | Hard logic needs more |
| summarization | 2000 | 10% | Already moderate |

**Key insight**: Oversample the categories where the model is weakest (sentiment, math, logic) by using more epochs or synthetic augmentation.

### Negative Examples / Hard Negatives

Include examples where the correct answer is counterintuitive:
- Sentiment: "I love how you ignore everything I say" → negative
- Factual: "What year was the War of 1812?" → 1812-1815 (trick question about duration vs start)
- Math: Add distracting numbers that look like the answer but aren't

**Recommendation**: Add 5-10% hard negative examples per category.

---

## B5. Evaluation of LoRA Quality

### The 60-set Eval is NOT Sufficient

The eval_60_balanced.json has only 60 questions — 7-8 per category. This is far too small for reliable measurement:
- **Confidence interval**: ±12% at 60 questions (binomial proportion CI)
- **Inter-category comparison**: Impossible with 7-8 items per category
- **95% CI for a 60% score**: [46.5%, 72.4%] — a 26-point range!

### Recommended Eval Strategy

| Eval Set | Questions | Purpose | Priority |
|----------|-----------|---------|----------|
| eval_60_balanced.json | 60 | Quick sanity check during training | ✅ Use for iteration |
| eval_all_300.json | 300 | Post-training evaluation | ✅ Must use |
| eval_hard_218.json | 218 | Hard-only evaluation | ✅ Use for capability gaps |
| **eval_from_datasets_*.json** | 95 | Held-out dataset eval | ✅ Must use (clean dataset Qs) |
| **New: 100 held-out questions** | 100 | Never-seen-before questions | **Must create** |

**Minimum viable evaluation**: **400 questions** (300 from all_300 + 95 from datasets + reserve from simple/medium).

### Metrics to Track

| Metric | What It Measures | How |
|--------|-----------------|-----|
| **Overall accuracy** | Gross capability | `correct / total` |
| **Per-category accuracy** | Category-specific | `correct_per_cat / total_per_cat` |
| **Δ vs baseline** | Did LoRA help or hurt? | `accuracy_lora - accuracy_baseline` |
| **Easy-drop rate** | Did LoRA forget easy things? | easy accuracy before vs after |
| **fuzzy_match strategy distribution** | Is output format improving? | exact vs substring vs token_overlap vs numeric |
| **Hallucination rate** | Is model making things up? | Manual check on factual subset |

### Proper Evaluation Protocol

```bash
# 1. Baseline (no LoRA)
python eval_pipeline.py --mode model --model Qwen/Qwen2.5-3B-Instruct --qs eval_all_300.json --label baseline

# 2. After LoRA
python eval_pipeline.py --mode model --model Qwen/Qwen2.5-3B-Instruct --qs eval_all_300.json --adapter path/to/adapter --label lora_v1

# 3. Compare
python eval_pipeline.py --mode compare
```

---

## B6. Known Qwen2.5-3B LoRA Pitfalls

### 1. Tokenizer Special Tokens
Qwen2.5 uses special tokens that MUST be preserved during training:
- `<|im_start|>` and `<|im_end|>` — chat format markers
- `<|endoftext|>` — end of sequence

**Fix**: Always use `tokenizer.apply_chat_template()` to ensure correct formatting. Never manually construct prompt strings.

### 2. Architecture Quirks
- **Qwen2.5-3B** has 36 layers (same as 7B but with smaller hidden dim). All linear layers are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **No extra embedding layers** — standard RoPE + SwiGLU.
- **RMSNorm in each block** — DO NOT add LoRA to norm layers; it destabilizes training.

### 3. BFloat16 vs Float16
Qwen2.5 was pre-trained in bfloat16. If your GPU doesn't support bf16 (T4 doesn't), use `fp16=True` instead. The quality difference is negligible for LoRA.

### 4. Trust Remote Code
```python
model = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,  # REQUIRED for Qwen2.5
)
```

### 5. Max Sequence Length
Qwen2.5 supports up to 32K context. Setting `max_seq_length=1024` is fine for training, but if the summarization data has long articles, either truncate or increase to 2048.

### 6. Padding Side
`tokenizer.padding_side = 'right'` (default for Qwen2.5). Keep this; 'left' padding causes training instability.

### 7. No Flash Attention on T4
T4 GPUs don't support Flash Attention 2. Unsloth handles this automatically with xformers fallback, but be aware of slower training.

---

## ACTION PLAN SUMMARY

### Immediate (Today)

1. **Fix training data**: Regenerate all JSONL entries with:
   - Correct `messages` format (Option A)
   - Concise, direct answers matching eval expected_answer format
   - No formulaic explanations

2. **Fix training notebook**:
   - Use 8-bit quantization instead of 4-bit
   - Set `lora_r=8`, `lora_alpha=32`, `lora_dropout=0.1`
   - Set `learning_rate=1e-4`, `lr_scheduler_type='cosine'`
   - Train ONE combined adapter (not 8 sequential)
   - Fix data format for tokenizer compatibility

3. **Fix eval protocol**:
   - Use eval_all_300.json (300 questions) as primary eval set
   - Hold out 95 dataset-converted questions as held-out evaluation
   - Measure Δ vs baseline, not absolute accuracy

### This Week

4. **Expand eval data**:
   - Convert public datasets → 400+ eval questions
   - Build template generator → 300+ questions
   - Download TweetEval + GoEmotions for nuanced sentiment

5. **Add hard negative examples**:
   - 5-10% per category
   - Focus on sentiment (sarcasm detection) and math (distractors)

6. **Train and eval combined LoRA**:
   - Train on all 16K examples combined
   - Evaluate vs baseline. Target: no regression on any category, +5-15% on sentiment
   - If regression occurs, reduce LR, increase dropout, or add weight decay

### Next Week

7. **Iterate on weak categories**:
   - If math still weak: add MATH competition problems to training data (not just GSM8K)
   - If logic still weak: add more ZebraLogic and multi-step reasoning examples
   - If sentiment still weak: add TweetEval + contrast set data

8. **Consider dual-model approach**:
   - Base Qwen2.5-3B for code/ner/summarization (already strong)
   - LoRA-tuned Qwen2.5-3B for sentiment/math/logic (weak areas)
   - Route per category based on classifier output

---

## REFERENCES

1. Hu et al. (2021) — LoRA: Low-Rank Adaptation of Large Language Models. https://arxiv.org/abs/2106.09685
2. Dettmers et al. (2023) — QLoRA: Efficient Finetuning of Quantized LLMs. https://arxiv.org/abs/2305.14314
3. Wang et al. (2022) — Self-Instruct: Aligning Language Models with Self-Generated Instructions. https://arxiv.org/abs/2212.10560
4. Xu et al. (2023) — WizardLM: Empowering Large Language Models to Follow Complex Instructions (Evol-Instruct). https://arxiv.org/abs/2304.12244
5. Zellers et al. (2019) — Adversarial Filtering (HellaSwag). https://arxiv.org/abs/1905.07830
6. Gardner et al. (2020) — Contrast Sets (Evaluation). https://arxiv.org/abs/2004.02709
7. Qwen2.5 Technical Report. https://arxiv.org/abs/2412.15115
8. Unsloth Documentation. https://github.com/unslothai/unsloth
9. Distilabel Documentation. https://distilabel.argilla.io
10. Argilla Documentation. https://docs.argilla.io
