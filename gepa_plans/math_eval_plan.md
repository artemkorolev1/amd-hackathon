# Math (T02) Eval Plan — Ready to Dispatch

## Goal
Find best (model, prompt) pair for math word problems across 4 models × 6 prompts on 94 questions.

## Models
- qwen2.5-math-1.5b: /home/artem/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf (math specialist)
- smollm2-1.7b: /home/artem/models/smollm2-1.7b-instruct-q4_k_m.gguf
- qwen2.5-1.5b-instruct: /home/artem/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
- llama-3.2-1b: /home/artem/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf

## 6 Prompt Strategies
0: "" (empty)
1: "Math:"
2: "Answer only with a number."
3: "Let's think step by step."
4: "Calculate:"
5: "Answer: "

## Data
/home/artem/dev/amd-hackathon/data/eval/math_combined_80.json (94 unique math questions)

## Eval methodology
- GPU (n_gpu_layers=-1), load one model at a time
- Temperature=0.0, max_tokens=64
- fuzzy_match cascade (same as factual)
- Chat format with system+user messages
- Save to /home/artem/dev/amd-hackathon/gepa_plans/math_94q_results.json
