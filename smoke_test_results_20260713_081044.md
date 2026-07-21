# GPU Smoke Test Report

- **Date:** 2026-07-13T08:10:33.835766
- **GPU:** NVIDIA RTX A4000 Laptop GPU (8 GB VRAM)
- **Prompt:** "What is 2+2?"
- **Temperature:** 0.1
- **Max tokens:** 100

## Summary

| Model | Role | Load OK | Infer OK | Load Time | VRAM Used | Infer Time | Tokens | Tok/s |
|-------|------|---------|----------|-----------|-----------|------------|--------|-------|
| qwen2.5-1.5b-instruct | primary | ✅ | ✅ | 0.85s | 1438 MiB | 231.3 ms | 8 | 34.6 |
| qwen2.5-coder-1.5b-instruct | code specialist | ✅ | ✅ | 1.25s | 1292 MiB | 61.7 ms | 8 | 129.7 |
| gemma-3-1b-it | fast instruction follower | ✅ | ✅ | 1.12s | 1332 MiB | 75.1 ms | 8 | 106.5 |

**Overall status:** ✅ ALL TESTS PASSED

---

## qwen2.5-1.5b-instruct (primary)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 0.85s
- **VRAM before:** 88 MiB
- **VRAM after load:** 1526 MiB
- **VRAM used by model:** 1438 MiB
- **GPU util before:** 46%
- **GPU util after load:** 86%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 equals 4.`
- **Inference time:** 231.3 ms
- **Tokens output:** 8
- **Tokens/second:** 34.6
- **VRAM after inference:** 1564 MiB
- **GPU util after inference:** 86%

## qwen2.5-coder-1.5b-instruct (code specialist)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 1.25s
- **VRAM before:** 260 MiB
- **VRAM after load:** 1552 MiB
- **VRAM used by model:** 1292 MiB
- **GPU util before:** 9%
- **GPU util after load:** 50%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 equals 4.`
- **Inference time:** 61.7 ms
- **Tokens output:** 8
- **Tokens/second:** 129.7
- **VRAM after inference:** 1564 MiB
- **GPU util after inference:** 50%

## gemma-3-1b-it (fast instruction follower)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 1.12s
- **VRAM before:** 260 MiB
- **VRAM after load:** 1592 MiB
- **VRAM used by model:** 1332 MiB
- **GPU util before:** 9%
- **GPU util after load:** 91%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 = 4`
- **Inference time:** 75.1 ms
- **Tokens output:** 8
- **Tokens/second:** 106.5
- **VRAM after inference:** 1596 MiB
- **GPU util after inference:** 91%

---
*Report generated at 2026-07-13T08:10:44.115722*
