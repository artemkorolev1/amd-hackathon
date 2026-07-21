# Sentiment GEPA v2 Results

Date: 2026-07-13 15:25

## Config

- Models: ['gemma-3-1b', 'qwen2.5-1.5b', 'qwen2.5-coder-1.5b']
- Generations: 2
- Questions per eval: 100
- Hybrid mode: True
- Temperature: 0.0

## Results per Generation

### Generation 0

| Model | Prompt | Params | Train Acc | Val Acc |
|-------|--------|--------|-----------|--------|
| gemma-3-1b | Classify the sentiment. Output EXACTLY one word: p | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 76.8% | 66.0% |
| qwen2.5-1.5b | Is the sentiment positive or negative? Reply with  | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 78.8% | 83.0% |
| qwen2.5-coder-1.5b | What is the sentiment? Reply with one word. | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 76.8% | 65.0% |

### Generation 1

| Model | Prompt | Params | Train Acc | Val Acc |
|-------|--------|--------|-----------|--------|
| gemma-3-1b | Classify the sentiment. Output EXACTLY one word: p | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 76.8% | 66.0% |
| qwen2.5-1.5b | Is the sentiment positive or negative? Reply with  | {'top_p': 0.80570169649468, 'top_k': 30, 'min_p': 0.06256462124948563} | 79.8% | 83.0% |
| qwen2.5-coder-1.5b | What is the sentiment? Reply with one word. | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 76.8% | 66.0% |

### Generation 2

| Model | Prompt | Params | Train Acc | Val Acc |
|-------|--------|--------|-----------|--------|
| gemma-3-1b | Classify the sentiment. Output EXACTLY one word: p | {'top_p': 0.9, 'top_k': 20, 'min_p': 0.05} | 76.8% | 66.0% |
| qwen2.5-1.5b | Is the sentiment positive or negative? Reply with  | {'top_p': 0.80570169649468, 'top_k': 30, 'min_p': 0.06256462124948563} | 79.8% | 82.0% |
| qwen2.5-coder-1.5b | Classify the sentiment. | {'top_p': 0.872025991947291, 'top_k': 25, 'min_p': 0.03268152311860239} | 79.8% | 66.0% |

## Best Overall

- **Model**: qwen2.5-1.5b
- **Prompt**: `Is the sentiment positive or negative? Reply with one word.`
- **Params**: `{'top_p': 0.9, 'top_k': 20, 'min_p': 0.05}`
- **Train accuracy**: 78.8%
- **Val accuracy**: 83.0%

## Per-Model Best

### gemma-3-1b

- **Prompt**: `Classify the sentiment. Output EXACTLY one word: positive, negative, neutral, or mixed.`
- **Params**: `{'top_p': 0.9, 'top_k': 20, 'min_p': 0.05}`
- **Train**: 76.8%
- **Val**: 66.0%

### qwen2.5-1.5b

- **Prompt**: `Is the sentiment positive or negative? Reply with one word.`
- **Params**: `{'top_p': 0.9, 'top_k': 20, 'min_p': 0.05}`
- **Train**: 78.8%
- **Val**: 83.0%

### qwen2.5-coder-1.5b

- **Prompt**: `What is the sentiment? Reply with one word.`
- **Params**: `{'top_p': 0.9, 'top_k': 20, 'min_p': 0.05}`
- **Train**: 76.8%
- **Val**: 66.0%

