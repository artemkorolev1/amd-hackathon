# Deterministic Debug Tool Validation Report
**Date:** 2026-07-15
**Total code_debug examples:** 233

## Per-Tool Summary

| Tool | Examples | Detection Rate | Notes |
|------|----------|---------------|-------|
| pyflakes | 233 | 100.0% | Detected errors in buggy code |
| pyflakes (FP) | 233 | 233 (100.0%) | False positives on EXPECTED (fixed) code |
| bandit | 233 | 100.0% | Found issues in buggy code |
| parso | 233 | 0.0% | Parsed with error recovery |
| libcst | 233 | 0.0% | Parsed as valid Python |

## Detailed Breakdown by Dataset

### data/eval/training-v2.json (114 items)
| Tool | Count | Rate |
|------|-------|------|
| pyflakes | 114/114 | 100% |
| bandit | 114/114 | 100% |
| parso | 0/114 | 0% |
| libcst | 0/114 | 0% |

### data/eval/training-v3.json (19 items)
| Tool | Count | Rate |
|------|-------|------|
| pyflakes | 19/19 | 100% |
| bandit | 19/19 | 100% |
| parso | 0/19 | 0% |
| libcst | 0/19 | 0% |

### data/eval/validation-v1.json (50 items)
| Tool | Count | Rate |
|------|-------|------|
| pyflakes | 50/50 | 100% |
| bandit | 50/50 | 100% |
| parso | 0/50 | 0% |
| libcst | 0/50 | 0% |

### data/eval/validation-v2.json (50 items)
| Tool | Count | Rate |
|------|-------|------|
| pyflakes | 50/50 | 100% |
| bandit | 50/50 | 100% |
| parso | 0/50 | 0% |
| libcst | 0/50 | 0% |

## Examples of pyflakes Detections

### Prompt: Fix the bug in this Python function:

lis = list()
    for i...
```
  CRASH:No module named 'pyflakes'
```

### Prompt: Fix the bug in this Python function:

temp_a, temp_b = a, b
...
```
  CRASH:No module named 'pyflakes'
```

### Prompt: Fix the bug in this Python function:

sum_value = 0
    prod...
```
  CRASH:No module named 'pyflakes'
```

### Prompt: Fix the bug in this Python function:

value_map = {
        ...
```
  CRASH:No module named 'pyflakes'
```

### Prompt: Fix the bug in this Python function:

for i in range(1, len(...
```
  CRASH:No module named 'pyflakes'
```

## Examples pyflakes MISSED (bug not detected)