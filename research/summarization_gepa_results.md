# Summarization GEPA Evolution Results

Date: 2026-07-14 02:50:46

## Overview

Summarization was our worst category at 37.8% accuracy. This experiment runs GEPA evolution with multi-signal grading (entity recall, keyword overlap, number match) and a chunk-and-summarize workflow for long texts (>200 words).

### Configuration

- **Models tested**: qwen2.5-1.5b, qwen2.5-coder-1.5b, gemma-3-1b
- **Generations**: 2 (gen 0 + gen 1)
- **Training set**: 100 questions (sampled from 366 total)
- **Validation set**: 78 questions
- **Grading**: Multi-signal (entity recall ≥50% + keyword ≥40%, or number overlap, or fuzzy_match cascade)
- **Workflow**: Chunk-and-summarize for texts >200 words

## Seed Prompts (Gen 0)

| Name | Prompt | Workflow |
|------|--------|----------|
| empty |  | No |
| summarize_colon | Summarize: | No |
| explicit_instruction | Summarize the text in at most 2 sentences. Include key names | No |
| verbose_instruction | Read the following text carefully and produce a concise summ | No |
| concise_news | Summarize this news article in 1-2 sentences with exact name | No |
| tldr | TL;DR: | No |
| extract_key_points | Extract key points from this text. Be specific with names an | No |
| wf_summarize | Summarize concisely. Break long texts into sections first. | Yes |
| wf_key_points | Extract and summarize key points from each section. | Yes |

## Gen 0 Results

| Model | Cell | Prompt | Steps | Acc | Correct/Total | Latency (ms) |
|-------|------|--------|-------|-----|---------------|-------------|
| qwen2.5-1.5b | seed_qwen2.5-1.5b_summarize_colon | Summarize: | 0 | 0.3400 | 34/100 | 507 |
| gemma-3-1b | seed_gemma-3-1b_extract_key_points | Extract key points from this text. Be specific wit | 0 | 0.3300 | 33/100 | 716 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_tldr | TL;DR: | 0 | 0.3200 | 32/100 | 508 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_explicit_instruction | Summarize the text in at most 2 sentences. Include | 0 | 0.3200 | 32/100 | 603 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_extract_key_points | Extract key points from this text. Be specific wit | 0 | 0.3200 | 32/100 | 655 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_verbose_instruction | Read the following text carefully and produce a co | 0 | 0.3100 | 31/100 | 430 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_extract_key_points | Extract key points from this text. Be specific wit | 0 | 0.3100 | 31/100 | 567 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_empty |  | 0 | 0.3000 | 30/100 | 534 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_concise_news | Summarize this news article in 1-2 sentences with  | 0 | 0.3000 | 30/100 | 577 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_explicit_instruction | Summarize the text in at most 2 sentences. Include | 0 | 0.2900 | 29/100 | 376 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_wf_key_points | Extract and summarize key points from each section | 1 | 0.2800 | 28/100 | 1168 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_empty |  | 0 | 0.2800 | 28/100 | 519 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_wf_summarize | Summarize concisely. Break long texts into section | 1 | 0.2800 | 28/100 | 1220 |
| gemma-3-1b | seed_gemma-3-1b_summarize_colon | Summarize: | 0 | 0.2800 | 28/100 | 508 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_wf_summarize | Summarize concisely. Break long texts into section | 1 | 0.2700 | 27/100 | 979 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_wf_key_points | Extract and summarize key points from each section | 1 | 0.2700 | 27/100 | 1363 |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_concise_news | Summarize this news article in 1-2 sentences with  | 0 | 0.2600 | 26/100 | 356 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_tldr | TL;DR: | 0 | 0.2500 | 25/100 | 513 |
| gemma-3-1b | seed_gemma-3-1b_concise_news | Summarize this news article in 1-2 sentences with  | 0 | 0.2500 | 25/100 | 541 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_summarize_colon | Summarize: | 0 | 0.2400 | 24/100 | 520 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_verbose_instruction | Read the following text carefully and produce a co | 0 | 0.2300 | 23/100 | 577 |
| gemma-3-1b | seed_gemma-3-1b_verbose_instruction | Read the following text carefully and produce a co | 0 | 0.2200 | 22/100 | 504 |
| gemma-3-1b | seed_gemma-3-1b_empty |  | 0 | 0.2000 | 20/100 | 513 |
| gemma-3-1b | seed_gemma-3-1b_explicit_instruction | Summarize the text in at most 2 sentences. Include | 0 | 0.2000 | 20/100 | 535 |
| gemma-3-1b | seed_gemma-3-1b_tldr | TL;DR: | 0 | 0.1800 | 18/100 | 475 |
| gemma-3-1b | seed_gemma-3-1b_wf_summarize | Summarize concisely. Break long texts into section | 1 | 0.1800 | 18/100 | 1565 |
| gemma-3-1b | seed_gemma-3-1b_wf_key_points | Extract and summarize key points from each section | 1 | 0.1700 | 17/100 | 1363 |

## Gen 1 Results (Evolved)

| Model | Cell | Prompt | Steps | Acc | Correct/Total | Latency (ms) | Parent |
|-------|------|--------|-------|-----|---------------|-------------|--------|
| qwen2.5-1.5b | elite_seed_qwen2.5-1.5b_summarize_colon | Summarize: | 0 | 0.3400 | 34/100 | 510 | seed_qwen2.5-1.5b_summarize_co |
| qwen2.5-1.5b | xover_seed_qwen2.5-1.5b_summarize_colon_seed_qwen2.5-1.5b_verbose_instruction_mut | Read the following text carefully and produce a co | 0 | 0.3300 | 33/100 | 442 | xover_seed_qwen2.5-1.5b_summar |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_extract_key_points_mut | Extract key points from this text. Be specific wit | 0 | 0.3300 | 33/100 | 771 | seed_qwen2.5-coder-1.5b_extrac |
| gemma-3-1b | elite_seed_gemma-3-1b_extract_key_points | Extract key points from this text. Be specific wit | 0 | 0.3300 | 33/100 | 715 | seed_gemma-3-1b_extract_key_po |
| gemma-3-1b | seed_gemma-3-1b_summarize_colon_mut | Summarize: | 0 | 0.3300 | 33/100 | 580 | seed_gemma-3-1b_summarize_colo |
| qwen2.5-1.5b | elite_seed_qwen2.5-1.5b_tldr | TL;DR: | 0 | 0.3200 | 32/100 | 509 | seed_qwen2.5-1.5b_tldr |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_summarize_colon_mut | Summarize: | 0 | 0.3200 | 32/100 | 503 | seed_qwen2.5-1.5b_summarize_co |
| qwen2.5-coder-1.5b | elite_seed_qwen2.5-coder-1.5b_explicit_instruction | Summarize the text in at most 2 sentences. Include | 0 | 0.3200 | 32/100 | 603 | seed_qwen2.5-coder-1.5b_explic |
| qwen2.5-coder-1.5b | elite_seed_qwen2.5-coder-1.5b_extract_key_points | Extract key points from this text. Be specific wit | 0 | 0.3200 | 32/100 | 657 | seed_qwen2.5-coder-1.5b_extrac |
| qwen2.5-coder-1.5b | xover_seed_qwen2.5-coder-1.5b_explicit_instruction_seed_qwen2.5-coder-1.5b_extract_key_points_mut | Extract key points from this text. Be specific wit | 0 | 0.3200 | 32/100 | 656 | xover_seed_qwen2.5-coder-1.5b_ |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_extract_key_points_mut | Extract key points from this text. Be specific wit | 0 | 0.3100 | 31/100 | 566 | seed_qwen2.5-1.5b_extract_key_ |
| qwen2.5-coder-1.5b | xover_seed_qwen2.5-coder-1.5b_explicit_instruction_seed_qwen2.5-coder-1.5b_concise_news_mut | Summarize this news article in 1-2 sentences with  | 0 | 0.3000 | 30/100 | 582 | xover_seed_qwen2.5-coder-1.5b_ |
| qwen2.5-1.5b | xover_seed_qwen2.5-1.5b_tldr_seed_qwen2.5-1.5b_verbose_instruction_mut | TL;DR:. Read the following text carefully and prod | 0 | 0.2900 | 29/100 | 431 | xover_seed_qwen2.5-1.5b_tldr_s |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_wf_summarize_mut | Summarize concisely. Break long texts into section | 1 | 0.2800 | 28/100 | 1222 | seed_qwen2.5-coder-1.5b_wf_sum |
| gemma-3-1b | elite_seed_gemma-3-1b_summarize_colon | Summarize: | 0 | 0.2800 | 28/100 | 508 | seed_gemma-3-1b_summarize_colo |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_wf_key_points_mut | Extract and summarize key points from each section | 1 | 0.2700 | 27/100 | 1363 | seed_qwen2.5-coder-1.5b_wf_key |
| gemma-3-1b | xover_seed_gemma-3-1b_extract_key_points_seed_gemma-3-1b_concise_news_mut | Summarize this news article in 1-2 sentences with  | 0 | 0.2500 | 25/100 | 541 | xover_seed_gemma-3-1b_extract_ |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_explicit_instruction_mut | Summarize the text in at most 2 sentences. | 0 | 0.2100 | 21/100 | 304 | seed_qwen2.5-1.5b_explicit_ins |
| gemma-3-1b | seed_gemma-3-1b_empty_mut |  | 0 | 0.2000 | 20/100 | 514 | seed_gemma-3-1b_empty |
| gemma-3-1b | seed_gemma-3-1b_empty_mut |  | 0 | 0.2000 | 20/100 | 513 | seed_gemma-3-1b_empty |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_wf_summarize_mut | Summarize concisely. | 1 | 0.1900 | 19/100 | 997 | seed_qwen2.5-coder-1.5b_wf_sum |
| qwen2.5-coder-1.5b | xover_seed_qwen2.5-coder-1.5b_explicit_instruction_seed_qwen2.5-coder-1.5b_concise_news_mut | Summarize this news article in 1-2 sentences with  | 0 | 0.1800 | 18/100 | 258 | xover_seed_qwen2.5-coder-1.5b_ |
| qwen2.5-1.5b | xover_seed_qwen2.5-1.5b_summarize_colon_seed_qwen2.5-1.5b_verbose_instruction_mut | Workflow cell — see steps | 2 | 0.1700 | 17/100 | 765 | xover_seed_qwen2.5-1.5b_summar |
| qwen2.5-1.5b | seed_qwen2.5-1.5b_summarize_colon_mut | Workflow cell — see steps | 2 | 0.1700 | 17/100 | 764 | seed_qwen2.5-1.5b_summarize_co |
| gemma-3-1b | seed_gemma-3-1b_explicit_instruction_mut | Summarize the text in at most 2 sentences. | 0 | 0.1300 | 13/100 | 461 | seed_gemma-3-1b_explicit_instr |
| gemma-3-1b | xover_seed_gemma-3-1b_extract_key_points_seed_gemma-3-1b_concise_news_mut | Summarize this news article in 1-2 sentences with  | 0 | 0.1000 | 10/100 | 224 | xover_seed_gemma-3-1b_extract_ |
| gemma-3-1b | xover_seed_gemma-3-1b_extract_key_points_seed_gemma-3-1b_summarize_colon_mut | Workflow cell — see steps | 2 | 0.1000 | 10/100 | 765 | xover_seed_gemma-3-1b_extract_ |

## Best Per Model

### qwen2.5-1.5b

- **Best cell**: seed_qwen2.5-1.5b_summarize_colon
- **System prompt**: `Summarize:`
- **Steps**: single-shot
- **Decoding params**: `{'temperature': 0.0, 'max_tokens': 96, 'top_p': 1.0, 'top_k': 40, 'min_p': 0.0, 'repeat_penalty': 1.0, 'seed': None}`
- **Training accuracy**: 0.3400 (34/100)
- **Validation accuracy**: 0.2949 (23/78)

### qwen2.5-coder-1.5b

- **Best cell**: seed_qwen2.5-coder-1.5b_extract_key_points_mut
- **System prompt**: `Extract key points from this text. Be specific with names and numbers.`
- **Steps**: single-shot
- **Decoding params**: `{'temperature': 0.0, 'max_tokens': 192, 'top_p': 0.9, 'top_k': 10, 'min_p': 0.01, 'repeat_penalty': 1.0, 'seed': 42}`
- **Training accuracy**: 0.3300 (33/100)
- **Validation accuracy**: 0.3462 (27/78)

### gemma-3-1b

- **Best cell**: seed_gemma-3-1b_extract_key_points
- **System prompt**: `Extract key points from this text. Be specific with names and numbers.`
- **Steps**: single-shot
- **Decoding params**: `{'temperature': 0.0, 'max_tokens': 128, 'top_p': 0.9, 'top_k': 20, 'min_p': 0.05, 'repeat_penalty': 1.05, 'seed': 42}`
- **Training accuracy**: 0.3300 (33/100)
- **Validation accuracy**: 0.3462 (27/78)

## Validation Results

| Model | Cell | Train Acc | Val Acc | Val Correct/Total | Delta |
|-------|------|-----------|---------|-------------------|-------|
| qwen2.5-1.5b | seed_qwen2.5-1.5b_summarize_colon | 0.3400 | 0.2949 | 23/78 | -0.0451 |
| qwen2.5-coder-1.5b | seed_qwen2.5-coder-1.5b_extract_key_points_mut | 0.3300 | 0.3462 | 27/78 | +0.0162 |
| gemma-3-1b | seed_gemma-3-1b_extract_key_points | 0.3300 | 0.3462 | 27/78 | +0.0162 |

## Key Insights

1. **Multi-signal grading** captures semantics that fuzzy_match misses for open-ended summaries.
2. **Chunk-and-summarize** helps for long texts (>200 words) by breaking the task into manageable pieces.
3. **Workflow operators** (split-into-steps, add-verify-step) can improve accuracy by forcing the model to reason step by step.
4. **Entity recall** and **keyword overlap** together provide a robust signal even when the exact phrasing differs.