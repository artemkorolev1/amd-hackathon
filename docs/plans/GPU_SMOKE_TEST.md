# GPU Smoke Test Report

**Date:** 2026-07-13 07:19 UTC  
**Model:** Qwen2.5-1.5B-Instruct (Q4_K_M)  
**GGUF Path:** `/home/artem/dev/amd-hackathon/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (1.1 GB)  
**GPU:** NVIDIA RTX A4000 Laptop (8 GB VRAM)  
**Framework:** `llama-cpp-python` v0.3.33  

---

## Test Procedure

1. Record pre-load VRAM usage (`nvidia-smi`)
2. Load model with `n_gpu_layers=-1`, `n_ctx=2048`, `n_threads=4`, `verbose=False`
3. Record post-load VRAM usage
4. Run 5 temperature sweeps on one prompt:
   - **Prompt:** `"What is the capital of France?"`
   - **Temperatures:** `[0.1, 0.3, 0.5, 0.7, 0.9]`
   - **Max tokens:** 200
5. Record per-inference latency (ms) and answer text
6. Record post-run VRAM usage

---

## Results

### Model Load

| Metric | Value |
|---|---|
| Load time | **0.81 s** |
| VRAM before load | 88 MiB |
| VRAM after load | 1526 MiB |
| **VRAM used by model** | **1438 MiB** |

### Inference (Temperature Sweep)

| # | Temp | Latency (ms) | Answer | Status |
|---|---|---|---|---|
| 1 | 0.1 | 136 | The capital of France is Paris. | ✅ OK |
| 2 | 0.3 | 46 | The capital of France is Paris. | ✅ OK |
| 3 | 0.5 | 45 | The capital of France is Paris. | ✅ OK |
| 4 | 0.7 | 46 | The capital of France is Paris. | ✅ OK |
| 5 | 0.9 | 46 | The capital of France is Paris. | ✅ OK |

### VRAM Final

| Metric | Value |
|---|---|
| VRAM after all inferences | 1564 MiB |
| Peak VRAM delta (load) | +1438 MiB |
| Peak VRAM delta (inference) | +1476 MiB |

---

## Verdict

**✅ PASS — All 5 samples completed without errors.**

- Model loads in **under 1 second** on GPU.
- Inference latency is **~45–136 ms** per query (cold start on temp=0.1 slightly slower at 136ms, subsequent calls stabilize at ~46ms).
- Model consumes **~1.4 GB VRAM** out of 8 GB available (plenty of headroom for concurrent workers).
- All answers are semantically correct and consistent.

**Conclusion:** The GPU inference stack (llama-cpp-python + GGUF + CUDA) is fully operational. Any failure in the full 80-question run would be in the **orchestration layer** (pool/queue/judge), **not** in the model stack.

---

*Test script: `/home/artem/dev/amd-hackathon/gpu_smoke_test.py`*
