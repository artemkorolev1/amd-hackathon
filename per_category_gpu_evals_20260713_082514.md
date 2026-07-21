# Comprehensive Per-Category GPU Eval Report

- **Model**: qwen2.5-1.5b-instruct
- **Model path**: models/qwen2.5-1.5b-instruct-q4_k_m.gguf
- **Date**: 2026-07-13 08:25:14
- **GPU**: RTX A4000 8GB, N_GPU_LAYERS=-1

## Overall Summary

| Metric | Value |
|---|---|
| Total Questions | 50 |
| Correct | 46 |
| **Accuracy** | **92.00%** |
| 84.2% Gate | ✅ PASS |
| Avg Time/Question | 4015.4 ms |
| Max Time/Question | 9622.7 ms |
| Total Time | 200.8 s |

## Per-Category Breakdown

| Category | Total | Correct | Accuracy | Avg Time (ms) |
|---|---|---|---|---|
| ✅ code_debug | 13 | 13 | 100.0% | 2130 |
| ✅ code_gen | 12 | 11 | 91.7% | 3654 |
| ✅ factual | 13 | 11 | 84.6% | 2784 |
| ✅ logic | 12 | 11 | 91.7% | 7753 |

## Per-Source Breakdown

| Source | Total | Correct | Accuracy |
|---|---|---|---|
| claude-code-hard-v1 | 50 | 46 | 92.0% |

## Per-Difficulty Breakdown

| Difficulty | Total | Correct | Accuracy |
|---|---|---|---|
| hard | 50 | 46 | 92.0% |

## Failures (4)

| # | Task ID | Category | Expected | Got (snippet) | Reason |
|---|---|---|---|---|---|
| 1 | q-015486f5 | code_gen | from typing import Tuple
from functools import lru_cache


def regex_match(text: | ```python
import re

def regex_match(text: str, pattern: str) -> bool:
    retur | expected: from typing import Tuple
from functools import lru |
| 2 | q-003d8024 | factual | Scholars would likely have access to definitive textual evidence to resolve the  | If the Library of Alexandria had survived intact through all its historical thre | expected: Scholars would likely have access to definitive te |
| 3 | q-0489df6b | factual | NASA's Apollo program would almost certainly have continued beyond Apollo 17, wi | If the Soviet Union had successfully landed a cosmonaut on the Moon in August 19 | expected: NASA's Apollo program would almost certainly have  |
| 4 | q-01c82564 | logic | 625 | Let's solve this step by step:

1. **The hundreds digit is 3 times the tens digi | expected: 625, got: Let's solve this step by step:

1. **The |

### code_gen Failures (1)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-015486f5 | from typing import Tuple
from functools import lru_cache


d | ```python
import re

def regex_match(text: str, pattern: str | expected: from typing import Tuple
from functools  |

### factual Failures (2)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-003d8024 | Scholars would likely have access to definitive textual evid | If the Library of Alexandria had survived intact through all | expected: Scholars would likely have access to def |
| q-0489df6b | NASA's Apollo program would almost certainly have continued  | If the Soviet Union had successfully landed a cosmonaut on t | expected: NASA's Apollo program would almost certa |

### logic Failures (1)

| Task ID | Expected | Got | Reason |
|---|---|---|---|
| q-01c82564 | 625 | Let's solve this step by step:

1. **The hundreds digit is 3 | expected: 625, got: Let's solve this step by step: |
