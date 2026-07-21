# GPU Smoke Test Report

- **Date:** 2026-07-13T08:10:58.851128
- **GPU:** NVIDIA RTX A4000 Laptop GPU (8 GB VRAM)
- **Prompt:** "What is 2+2?"
- **Temperature:** 0.1
- **Max tokens:** 100

## Summary

| Model | Role | Load OK | Infer OK | Load Time | VRAM Used | Infer Time | Tokens | Tok/s |
|-------|------|---------|----------|-----------|-----------|------------|--------|-------|
| qwen2.5-1.5b-instruct | primary | ✅ | ✅ | 0.83s | 1438 MiB | 215.7 ms | 8 | 37.1 |
| qwen2.5-coder-1.5b-instruct | code specialist | ✅ | ✅ | 0.52s | 1292 MiB | 60.7 ms | 8 | 131.8 |
| gemma-3-1b-it | fast instruction follower | ✅ | ✅ | 0.59s | 1332 MiB | 102.3 ms | 8 | 78.2 |

**Overall status:** ✅ ALL TESTS PASSED

---

## qwen2.5-1.5b-instruct (primary)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 0.83s
- **VRAM before:** 88 MiB
- **VRAM after load:** 1526 MiB
- **VRAM used by model:** 1438 MiB
- **GPU util before:** 38%
- **GPU util after load:** 53%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 equals 4.`
- **Inference time:** 215.7 ms
- **Tokens output:** 8
- **Tokens/second:** 37.1
- **VRAM after inference:** 1564 MiB
- **GPU util after inference:** 53%

## qwen2.5-coder-1.5b-instruct (code specialist)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 0.52s
- **VRAM before:** 260 MiB
- **VRAM after load:** 1552 MiB
- **VRAM used by model:** 1292 MiB
- **GPU util before:** 9%
- **GPU util after load:** 86%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 equals 4.`
- **Inference time:** 60.7 ms
- **Tokens output:** 8
- **Tokens/second:** 131.8
- **VRAM after inference:** 1564 MiB
- **GPU util after inference:** 86%

## gemma-3-1b-it (fast instruction follower)

### Load

- **Status:** ✅ Loaded successfully
- **Load time:** 0.59s
- **VRAM before:** 260 MiB
- **VRAM after load:** 1592 MiB
- **VRAM used by model:** 1332 MiB
- **GPU util before:** 9%
- **GPU util after load:** 10%

### Inference

- **Status:** ✅ Completed successfully
- **Prompt:** "What is 2+2?"
- **Answer:** `2 + 2 = 4`
- **Inference time:** 102.3 ms
- **Tokens output:** 8
- **Tokens/second:** 78.2
- **VRAM after inference:** 1596 MiB
- **GPU util after inference:** 12%

---
*Report generated at 2026-07-13T08:11:07.938995*
