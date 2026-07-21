#!/usr/bin/env python3
"""Comprehensive GPU smoke test for all 3 GGUF models.

Tests:
  1. Qwen2.5-1.5B-Instruct (Q4_K_M) – primary
  2. Qwen2.5-Coder-1.5B-Instruct (Q4_K_M) – code specialist
  3. Gemma-3-1B-IT (Q4_K_M) – fast instruction follower

For each model:
  - Load with n_gpu_layers=-1 (full GPU offload)
  - Record VRAM before/after load
  - Run inference: "What is 2+2?"
  - Record latency and output
  - Record VRAM after inference
"""

import subprocess, json, time, sys, os
from datetime import datetime


def get_gpu_mem():
    """Return (used_mib, total_mib) on GPU 0."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    parts = r.stdout.strip().split(", ")
    return int(parts[0]), int(parts[1])


def get_gpu_util():
    """Return GPU util %."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    return int(r.stdout.strip())


MODELS = {
    "qwen2.5-1.5b-instruct": {
        "path": "/home/artem/dev/amd-hackathon/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "role": "primary",
    },
    "qwen2.5-coder-1.5b-instruct": {
        "path": "/home/artem/dev/amd-hackathon/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        "role": "code specialist",
    },
    "gemma-3-1b-it": {
        "path": "/home/artem/dev/amd-hackathon/models/gemma-3-1b-it-Q4_K_M.gguf",
        "role": "fast instruction follower",
    },
}

PROMPT = "What is 2+2?"
MAX_TOKENS = 100
TEMPERATURE = 0.1


def test_model(name, info):
    """Test a single model. Returns dict of results."""
    from llama_cpp import Llama

    print(f"\n{'='*70}", flush=True)
    print(f"  MODEL: {name}", flush=True)
    print(f"  Role:  {info['role']}", flush=True)
    print(f"  File:  {info['path']}", flush=True)
    print(f"{'='*70}", flush=True)

    # Pre-load VRAM
    mem_used_before, mem_total = get_gpu_mem()
    util_before = get_gpu_util()
    print(f"\n  GPU state BEFORE load:", flush=True)
    print(f"    VRAM used:  {mem_used_before} MiB / {mem_total} MiB", flush=True)
    print(f"    GPU util:   {util_before}%", flush=True)

    # Load model
    print(f"\n  Loading model (n_gpu_layers=-1, full GPU offload)...", flush=True)
    t0 = time.time()
    try:
        llm = Llama(
            model_path=info["path"],
            n_gpu_layers=-1,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.2f}s", flush=True)
    except Exception as e:
        load_time = time.time() - t0
        print(f"  ** MODEL LOAD FAILED ** after {load_time:.2f}s: {e}", flush=True)
        return {
            "model": name,
            "role": info["role"],
            "load_success": False,
            "load_time_s": round(load_time, 2),
            "error": str(e),
        }

    # Post-load VRAM
    mem_used_after_load, _ = get_gpu_mem()
    util_after_load = get_gpu_util()
    vram_delta_load = mem_used_after_load - mem_used_before
    print(f"\n  GPU state AFTER load:", flush=True)
    print(f"    VRAM used:  {mem_used_after_load} MiB (delta: +{vram_delta_load} MiB)", flush=True)
    print(f"    GPU util:   {util_after_load}%", flush=True)

    # Run inference
    print(f"\n  Running inference...", flush=True)
    print(f"    Prompt: \"{PROMPT}\"", flush=True)
    t_start = time.time()
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": PROMPT}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        inference_time = time.time() - t_start
        answer = response["choices"][0]["message"]["content"].strip()
        tokens_used = response["usage"]["completion_tokens"]
        tokens_per_sec = tokens_used / inference_time if inference_time > 0 else 0

        # Truncate answer for display
        answer_display = answer[:200].replace("\n", "\\n")
        if len(answer) > 200:
            answer_display += "..."

        print(f"    Inference time: {inference_time*1000:.0f} ms", flush=True)
        print(f"    Tokens output:  {tokens_used} ({tokens_per_sec:.1f} tok/s)", flush=True)
        print(f"    Answer:         {answer_display}", flush=True)
        inference_ok = True
        inference_error = None
    except Exception as e:
        inference_time = time.time() - t_start
        answer = None
        tokens_used = None
        tokens_per_sec = None
        print(f"\n    ** INFERENCE FAILED **: {e}", flush=True)
        inference_ok = False
        inference_error = str(e)

    # Post-run VRAM
    mem_used_after_run, _ = get_gpu_mem()
    util_after_run = get_gpu_util()
    vram_delta_run = mem_used_after_run - mem_used_after_load
    print(f"\n  GPU state AFTER inference:", flush=True)
    print(f"    VRAM used:  {mem_used_after_run} MiB (delta from load: {vram_delta_run:+d} MiB)", flush=True)
    print(f"    GPU util:   {util_after_run}%", flush=True)

    # Unload model
    del llm

    return {
        "model": name,
        "role": info["role"],
        "load_success": True,
        "load_time_s": round(load_time, 2),
        "vram_before_mib": mem_used_before,
        "vram_after_load_mib": mem_used_after_load,
        "vram_after_inference_mib": mem_used_after_run,
        "vram_used_by_model_mib": vram_delta_load,
        "inference_success": inference_ok,
        "inference_time_ms": round(inference_time * 1000, 1) if inference_time else None,
        "tokens_output": tokens_used,
        "tokens_per_second": round(tokens_per_sec, 1) if tokens_per_sec else None,
        "answer": answer,
        "error": inference_error,
        "gpu_util_before_pct": util_before,
        "gpu_util_after_load_pct": util_after_load,
        "gpu_util_after_inference_pct": util_after_run,
        "memory_total_mib": mem_total,
    }


def generate_markdown_report(all_results, timestamp):
    """Generate a markdown report from all results."""
    lines = []

    lines.append(f"# GPU Smoke Test Report")
    lines.append(f"")
    lines.append(f"- **Date:** {timestamp}")
    lines.append(f"- **GPU:** NVIDIA RTX A4000 Laptop GPU (8 GB VRAM)")
    lines.append(f"- **Prompt:** \"{PROMPT}\"")
    lines.append(f"- **Temperature:** {TEMPERATURE}")
    lines.append(f"- **Max tokens:** {MAX_TOKENS}")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Model | Role | Load OK | Infer OK | Load Time | VRAM Used | Infer Time | Tokens | Tok/s |")
    lines.append(f"|-------|------|---------|----------|-----------|-----------|------------|--------|-------|")

    overall_ok = True
    for r in all_results:
        load_ok = "✅" if r.get("load_success") else "❌"
        infer_ok = "✅" if r.get("inference_success") else "❌"
        load_time = f"{r['load_time_s']}s" if r.get("load_time_s") else "N/A"
        vram = f"{r.get('vram_used_by_model_mib', 'N/A')} MiB"
        infer_time = f"{r.get('inference_time_ms', 'N/A')} ms"
        tokens = r.get("tokens_output", "N/A")
        tok_s = r.get("tokens_per_second", "N/A")
        lines.append(f"| {r['model']} | {r['role']} | {load_ok} | {infer_ok} | {load_time} | {vram} | {infer_time} | {tokens} | {tok_s} |")
        if not r.get("load_success") or not r.get("inference_success"):
            overall_ok = False

    lines.append(f"")
    lines.append(f"**Overall status:** {'✅ ALL TESTS PASSED' if overall_ok else '❌ SOME TESTS FAILED'}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    for r in all_results:
        lines.append(f"## {r['model']} ({r['role']})")
        lines.append(f"")
        lines.append(f"### Load")
        lines.append(f"")
        if r.get("load_success"):
            lines.append(f"- **Status:** ✅ Loaded successfully")
            lines.append(f"- **Load time:** {r['load_time_s']}s")
            lines.append(f"- **VRAM before:** {r['vram_before_mib']} MiB")
            lines.append(f"- **VRAM after load:** {r['vram_after_load_mib']} MiB")
            lines.append(f"- **VRAM used by model:** {r['vram_used_by_model_mib']} MiB")
            lines.append(f"- **GPU util before:** {r.get('gpu_util_before_pct', 'N/A')}%")
            lines.append(f"- **GPU util after load:** {r.get('gpu_util_after_load_pct', 'N/A')}%")
        else:
            lines.append(f"- **Status:** ❌ Failed to load")
            lines.append(f"- **Error:** `{r.get('error', 'Unknown')}`")

        lines.append(f"")
        lines.append(f"### Inference")
        lines.append(f"")
        if r.get("inference_success"):
            lines.append(f"- **Status:** ✅ Completed successfully")
            lines.append(f"- **Prompt:** \"{PROMPT}\"")
            lines.append(f"- **Answer:** `{r['answer']}`")
            lines.append(f"- **Inference time:** {r['inference_time_ms']} ms")
            lines.append(f"- **Tokens output:** {r['tokens_output']}")
            lines.append(f"- **Tokens/second:** {r['tokens_per_second']}")
            lines.append(f"- **VRAM after inference:** {r.get('vram_after_inference_mib', 'N/A')} MiB")
            lines.append(f"- **GPU util after inference:** {r.get('gpu_util_after_inference_pct', 'N/A')}%")
        else:
            lines.append(f"- **Status:** ❌ Failed")
            if r.get("error"):
                lines.append(f"- **Error:** `{r['error']}`")
            else:
                lines.append(f"- *(load failed, no inference attempted)*")

        lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Report generated at {datetime.now().isoformat()}*")
    lines.append(f"")

    return "\n".join(lines)


def main():
    timestamp = datetime.now().isoformat()
    print(f"=" * 70, flush=True)
    print(f"  GPU SMOKE TEST — ALL 3 MODELS", flush=True)
    print(f"  {timestamp}", flush=True)
    print(f"  GPU: NVIDIA RTX A4000 Laptop GPU", flush=True)
    print(f"=" * 70, flush=True)

    # Get initial GPU state
    mem_used, mem_total = get_gpu_mem()
    util = get_gpu_util()
    print(f"\nInitial GPU state:", flush=True)
    print(f"  VRAM: {mem_used} / {mem_total} MiB")
    print(f"  Util: {util}%", flush=True)

    all_results = []
    for name, info in MODELS.items():
        print(f"\n{'#'*70}", flush=True)
        print(f"  STARTING: {name}", flush=True)
        print(f"{'#'*70}", flush=True)

        result = test_model(name, info)
        all_results.append(result)

        # Give GPU a moment to settle between models
        print(f"\n  Cooling down between models...", flush=True)
        time.sleep(2)

    # Generate report
    timestamp_str = timestamp.replace(":", "-").replace(".", "-")
    report_filename = f"/home/artem/dev/amd-hackathon/smoke_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report = generate_markdown_report(all_results, timestamp)
    with open(report_filename, "w") as f:
        f.write(report)

    # Also write structured JSON
    json_result = {"timestamp": timestamp, "gpu": "NVIDIA RTX A4000 Laptop GPU", "vram_total_mib": mem_total, "results": all_results}
    json_path = "/tmp/gpu_smoke_all_results.json"
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\n{'='*70}", flush=True)
    print(f"  REPORT SAVED: {report_filename}", flush=True)
    print(f"  JSON SAVED:   {json_path}", flush=True)
    print(f"{'='*70}", flush=True)

    return all_results


if __name__ == "__main__":
    results = main()
    total = len(results)
    passed = sum(1 for r in results if r.get("load_success") and r.get("inference_success"))
    print(f"\n{'='*70}", flush=True)
    print(f"  FINAL RESULT: {passed}/{total} models passed", flush=True)
    print(f"{'='*70}", flush=True)
    sys.exit(0 if passed == total else 1)
