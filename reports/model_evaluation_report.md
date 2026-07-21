# Multi-Model Evaluation Report (July 12, 2026)

## Models Evaluated (6 total, 1B-class, GGUF Q4)
- qwen2.5-1.5b-instruct-q4_k_m.gguf
- qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
- Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf (worst general model, only use for sentiment)
- gemma-3-1b-it-Q4_K_M.gguf
- smollm2-1.7b-instruct-q4_k_m.gguf
- Llama-3.2-1B-Instruct-Q4_K_M.gguf

## Test Sets
1. dev_40.json — 40 balanced (5/category), text-answer gold
2. heldout_40.json — 40 math-heavy (14/40), text-answer gold  
3. complexity_40.json — 40 balanced (5/category) with difficulty labels
4. eval_comprehensive_300.json — 300 questions (218 hard), official comprehensive eval

## Grader: Official 4-cascade fuzzy match (exact → substring → numeric 1% tolerance → token overlap)

## Best Models per Category (from 300-set)

| Category | 1st | 2nd | 3rd |
|---|---|---|---|
| code_debug | qwen2.5-1.5b / coder / gemma / Llama (100%) | — | — |
| code_gen | gemma / Llama (100%) | qwen2.5-1.5b / coder (95%) | smollm2 (90%) |
| factual | qwen2.5-1.5b (81%) | coder (70%) | Llama (69%) |
| logic | qwen2.5-1.5b (68%) | coder / smollm2 (63%) | Llama (61%) |
| math | qwen2.5-1.5b (63%) | gemma (59%) | coder / smollm2 (57%) |
| ner | coder (100%) | qwen2.5-1.5b / Llama (83%) | gemma (67%) |
| sentiment | Qwen2.5-Math (92%) | gemma (75%) | smollm2 (67%) |
| summarization | qwen2.5-1.5b (75%) | gemma / coder (58%) | smollm2 / Llama (50%) |

## Recommended Ensemble: 3 models

### Primary: qwen2.5-1.5b-instruct
Wins: factual (81%), logic (68%), math (63%), summarization (75%)
Overall best on 300-set: 77.7% (233/300)

### Secondary: qwen2.5-coder-1.5b  
Wins: NER (100%), co-leads code (100%), strong on logic/factual

### Tertiary: gemma-3-1b-it
Fills: sentiment (75%), co-leads code_gen (100%), strong on math (59%)

### Why not 4 models
4th model (smollm2 or Llama) doesn't add any new category win — diminishing returns. Three models cover all 9 categories.

### Why not Qwen2.5-Math
37% on 300-set. Only useful for sentiment (92%) but terrible at everything else.

## Model Paths (all under /home/artem/models/)
```
qwen2.5-1.5b-instruct-q4_k_m.gguf      1.1 GB
qwen2.5-coder-1.5b-instruct-q4_k_m.gguf 1.1 GB
gemma-3-1b-it-Q4_K_M.gguf               0.8 GB
```

## Run All Three
```bash
cd /home/artem/dev/amd-hackathon
python3 multi_runner.py --eval input/cx_300.json \
  --models /home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf,/home/artem/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf,/home/artem/models/gemma-3-1b-it-Q4_K_M.gguf \
  --gpu
```

## Key Insight
The 1B-class models cannot pass the 84.2% gate on the 300-set (best: 77.7% by qwen2.5-1.5b). The v12d baseline achieved 93.7% using Nemotron-3-Nano-4B + Fireworks routing — the extra parameter count and API escalation for hard questions is what makes the difference.
